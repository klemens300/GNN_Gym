"""Migration barriers of all jumps out of the current vacancy site.

    E_a(A -> B) = B_sym(A, B) + 1/2 dE(A -> B)

This is what the KMC calls once per step. Two things keep a step cheap.

Shared initial state
    All candidate jumps leave the same configuration. It is encoded once and
    broadcast against the candidate final states. The encoder couples no
    samples, so this is an exact rewrite of encoding it once per candidate.

Halo instead of the whole cell
    Both models read out a ball around the jumping pair and the encoder's
    reach R is measured, so a site feature inside the read-out ball depends
    only on species within R hops of it. The step is evaluated on a fixed
    halo of
        hops = max(1 + 2 R_dE, 1 + pool_hops + R_Bsym)
    around the vacancy (the jumping atom is one hop away). Bonds leaving the
    halo point at one padding site; the error this introduces travels inward
    at most R hops from the boundary and never reaches the read-out ball.
    The cost per step is then independent of the supercell size. If the halo
    would cover the whole cell, the full cell is evaluated instead.
    `tests/test_predictor.py` checks halo against full cell.

Read-out masks and halo are the same sets of lattice offsets for every
vacancy position, so they are built once and translated with modular
arithmetic at every step.
"""
import time

import numpy as np
import torch

from ..lattice import hop_ball
from .checkpoint import load_model
from .graph import LatticeGraph

VACANCY = -1    # species value of the vacancy in simulation arrays


class JumpPredictor:
    """The two jump models bound to one simulation supercell.

    cell       gnn_gym.lattice.Supercell of the simulation
    elements   element symbols of the simulation's species indices; they are
               mapped onto the model's element order by symbol
    halo       evaluate a fixed halo around the vacancy (False: full cell)
    """

    def __init__(self, de_path, bsym_path, cell, elements, device="auto",
                 barrier_floor_eV=1e-3, halo=True, verbose=True):
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.de_net, self.de_cfg, de_ck = load_model(de_path, device, bind=False)
        self.bs_net, self.bs_cfg, bs_ck = load_model(bsym_path, device,
                                                     bind=False)
        if self.de_cfg.target != "de" or self.bs_cfg.target != "bsym":
            raise ValueError("expected a site-energy and a barrier checkpoint")
        if self.de_cfg.lattice != cell.lattice.name:
            raise ValueError(f"model lattice {self.de_cfg.lattice} != "
                             f"simulation lattice {cell.lattice.name}")
        self.de_reach = int(de_ck["reach"])
        self.bs_reach = int(bs_ck["reach"])
        try:
            self.species_map = np.array(
                [self.de_cfg.elements.index(e) for e in elements],
                dtype=np.int64)
        except ValueError as exc:
            raise ValueError(f"elements {list(elements)} are not a subset of "
                             f"the model's {self.de_cfg.elements}") from exc
        self.vac_index = self.de_cfg.vac_index
        self.barrier_floor_eV = barrier_floor_eV
        self.cell = cell
        self.n_clamped = 0
        self.n_calls = 0
        self.t_predict = 0.0
        self._build(halo, verbose)

    # -- geometry, built once ------------------------------------------------
    def _build(self, halo, verbose):
        cell, lat = self.cell, self.cell.lattice
        shells = list(self.de_cfg.shells)
        adj = cell.adjacency_lists(shells)
        v0 = 0
        self.slot_of = {}
        self.de_off, self.bs_off = {}, {}
        for slot, j in enumerate(cell.neighbours(lat.jump_shell)[v0]):
            d = tuple(cell.min_image(cell.coords[j] - cell.coords[v0]))
            ball_de = hop_ball(adj, [v0, int(j)], self.de_reach)
            ball_bs = hop_ball(adj, [v0, int(j)], self.bs_cfg.bsym_pool_hops)
            self.de_off[slot] = cell.min_image(cell.coords[ball_de] - cell.coords[v0])
            self.bs_off[slot] = cell.min_image(cell.coords[ball_bs] - cell.coords[v0])
            self.slot_of[d] = slot

        self.use_halo = bool(halo) and self._build_halo(adj, v0, shells)
        if self.use_halo:
            graph, n_eval = self.halo_graph, self.n_sub
        else:
            graph = LatticeGraph.for_supercell(cell, shells,
                                               self.bs_cfg.bsym_line_graph,
                                               device=self.device)
            n_eval = cell.n_sites
        self.de_net.set_graph(graph)
        self.bs_net.set_graph(graph)
        if verbose:
            mode = (f"halo of {self.halo_hops} hops" if self.use_halo
                    else "full cell")
            print(f"[predictor] {cell.size}^3 {lat.name} cell, {cell.n_sites} "
                  f"sites; dE reach {self.de_reach} hops "
                  f"({len(self.de_off[0])} sites), B_sym reach "
                  f"{self.bs_reach} hops, read-out "
                  f"{self.bs_cfg.bsym_pool_hops} hops ({len(self.bs_off[0])} "
                  f"sites); {n_eval} sites per configuration ({mode})",
                  flush=True)

    def _build_halo(self, adj, v0, shells):
        """Sub-lattice around the vacancy on which a step is evaluated.

        One halo serves all candidates, so the initial configuration is
        shared; per-candidate halos would be smaller but would each need
        their own encoding of the initial state.
        """
        cell, lat = self.cell, self.cell.lattice
        hops = max(1 + 2 * self.de_reach,
                   1 + self.bs_cfg.bsym_pool_hops + self.bs_reach)
        sites = np.flatnonzero(hop_ball(adj, [v0], hops))
        if len(sites) >= cell.n_sites:
            return False
        off = cell.min_image(cell.coords[sites] - cell.coords[v0])
        off = off[np.lexsort((off[:, 2], off[:, 1], off[:, 0]))]
        n_sub = len(off) + 1                       # last index: padding site
        pad = n_sub - 1
        local_of = {tuple(o): k for k, o in enumerate(off)}

        # Neighbour tables in local indices. The lookup goes through the
        # minimum image, so a neighbour across the periodic boundary is found
        # when the halo wraps around a small cell; anything outside the halo
        # points at the padding site, which points at itself.
        tables = []
        for s in shells:
            vec = lat.shell_vectors(s)
            tab = np.full((n_sub, len(vec)), pad, dtype=np.int64)
            for k, o in enumerate(off):
                for t, d in enumerate(vec):
                    tab[k, t] = local_of.get(tuple(cell.min_image(o + d)), pad)
            tables.append(tab)

        line_vectors = None
        if self.bs_cfg.bsym_line_graph or self.de_cfg.de_line_graph:
            # Bond vectors come from the shell vectors, so bonds to the
            # padding site still fall into a real angle class.
            line_vectors = np.tile(lat.shell_vectors(lat.jump_shell), (n_sub, 1))
        self.halo_graph = LatticeGraph.from_tables(
            tables, line_vectors, lat.angle_dots(), device=self.device)
        self.n_sub, self.pad, self.halo_hops, self.halo_off = n_sub, pad, hops, off
        self.v_local = local_of[(0, 0, 0)]

        z = lat.coordination(lat.jump_shell)
        self.j_local = {}
        de_loc = np.zeros((z, n_sub), dtype=bool)
        bs_loc = np.zeros((z, n_sub), dtype=bool)
        for d, slot in self.slot_of.items():
            self.j_local[slot] = local_of[d]
            for src, out in ((self.de_off[slot], de_loc),
                             (self.bs_off[slot], bs_loc)):
                for o in src:
                    k = local_of.get(tuple(o))
                    if k is None:
                        raise RuntimeError("read-out ball is not contained in "
                                           "the halo")
                    out[slot, k] = True
        self.de_mask_local = torch.from_numpy(de_loc).to(self.device)
        self.bs_mask_local = torch.from_numpy(bs_loc).to(self.device)
        return True

    # -- evaluation -------------------------------------------------------------
    def model_species(self, species, vacancy):
        """Simulation species (VACANCY = -1) -> model species indices."""
        sp = np.asarray(species, dtype=np.int64)
        z = np.where(sp == VACANCY, self.vac_index,
                     self.species_map[np.clip(sp, 0, None)])
        z[vacancy] = self.vac_index
        return z

    @torch.no_grad()
    def barriers(self, species, vacancy, targets):
        """E_a in eV of the jumps vacancy -> each site in `targets`."""
        t0 = time.perf_counter()
        vac = int(vacancy)
        js = np.asarray(targets, dtype=np.int64)
        z0 = self.model_species(species, vac)
        if self.use_halo:
            zi, zf, de_mask, bs_mask = self._halo_inputs(vac, js, z0)
        else:
            zi, zf, de_mask, bs_mask = self._cell_inputs(vac, js, z0)

        eps = self.de_net.site_energies(torch.cat([zi, zf]))
        dE = ((eps[1:] - eps[:1]) * de_mask).sum(-1)
        xi = self.bs_net.encoder(zi)
        xf = self.bs_net.encoder(zf)
        bsym = self.bs_net.readout(xi, xf, bs_mask)
        ea = (bsym + 0.5 * dE).cpu().numpy().astype(np.float64)

        self.n_calls += 1
        low = ea < self.barrier_floor_eV
        if low.any():
            self.n_clamped += int(low.sum())
            ea[low] = self.barrier_floor_eV
        self.t_predict += time.perf_counter() - t0
        return ea

    def _slots(self, vac, js):
        c = self.cell.coords[vac]
        return np.array([self.slot_of[tuple(self.cell.min_image(
            self.cell.coords[j] - c))] for j in js])

    def _halo_inputs(self, vac, js, z0):
        c = self.cell.coords[vac]
        glob = self.cell.index(self.halo_off + c)
        z_sub = np.empty(self.n_sub, dtype=np.int64)
        z_sub[:-1] = z0[glob]
        z_sub[self.pad] = self.vac_index            # never read out
        slots = self._slots(vac, js)
        j_loc = np.array([self.j_local[s] for s in slots])
        z_fin = np.repeat(z_sub[None], len(js), axis=0)
        z_fin[np.arange(len(js)), self.v_local] = z_sub[j_loc]
        z_fin[np.arange(len(js)), j_loc] = self.vac_index
        return (torch.from_numpy(z_sub[None]).to(self.device),
                torch.from_numpy(z_fin).to(self.device),
                self.de_mask_local[slots], self.bs_mask_local[slots])

    def _cell_inputs(self, vac, js, z0):
        n = self.cell.n_sites
        z_fin = np.repeat(z0[None], len(js), axis=0)
        z_fin[np.arange(len(js)), vac] = z0[js]
        z_fin[np.arange(len(js)), js] = self.vac_index
        c = self.cell.coords[vac]
        de = np.zeros((len(js), n), dtype=bool)
        bs = np.zeros((len(js), n), dtype=bool)
        for k, slot in enumerate(self._slots(vac, js)):
            de[k, self.cell.index(self.de_off[slot] + c)] = True
            bs[k, self.cell.index(self.bs_off[slot] + c)] = True
        return (torch.from_numpy(z0[None]).to(self.device),
                torch.from_numpy(z_fin).to(self.device),
                torch.from_numpy(de).to(self.device),
                torch.from_numpy(bs).to(self.device))

    def stats(self):
        return dict(calls=self.n_calls, clamped=self.n_clamped,
                    seconds=self.t_predict)

    def reset_timers(self):
        self.n_calls = self.n_clamped = 0
        self.t_predict = 0.0
