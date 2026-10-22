"""Canonical swap Metropolis on the site-energy model.

The energy of a configuration is the sum of the model's site energies. A
move exchanges the atoms on two sites p and q of different species and is
accepted with probability min[1, exp(-dE / k_B T)]. The per-species gauge of
the site energies cancels in every swap because the composition is fixed.

ENERGY OF A SWAP

    A swap changes the environment only of the sites within `reach` hops of
    p or q (their read-out balls). Every other site keeps its energy, so

        dE = sum over both read-out balls of [eps_i(after) - eps_i(before)],

    counting the overlap of the two balls once. This is exact, and it avoids
    differencing two cell totals in float32.

HALO AND LAYER LEVELS

    The energies of a read-out ball are exact when the network is evaluated
    on the 2*reach-hop ball (the halo) around the swapped site, with every
    neighbour outside the halo pointing at one padding site: the error the
    padding introduces travels inward one hop per layer and stops short of
    the read-out ball. Layer l only has to be evaluated where layer l+1 reads
    it: the embedding on the whole halo, layer l on the (reach + L - l)-hop
    ball, the last layer on the read-out ball. Each move is evaluated on two
    such halos, one around p and one around q, with the swap applied in both.
    The neighbour mean of a halo row is taken over its table row (padding
    entries included), which is the row-normalised neighbour sum of the
    full-cell graph.

SEVERAL DECISIONS PER CALL

    B proposals on the current configuration are evaluated together, with a
    B x B conflict matrix: two proposals conflict if any pair of their four
    sites lies within 2*reach hops, so that accepting one changes the dE of
    the other. The proposals are decided in order, each by a Metropolis test
    with its exact dE at the current configuration. Where the walk through
    the batch stops is set by `walk`:

    until_conflict (default; the walk of the published chains)
        Decide up to the first proposal that conflicts with a move already
        accepted in the call; that one and the rest of the batch are
        discarded. Several moves can be accepted per call. A proposal that
        falls into the neighbourhood of a move accepted earlier in the call
        is replaced by a fresh proposal only at the next call, so for the
        rest of the call that neighbourhood is proposed less often than in
        the single-swap chain. The walk is then not exactly the single-swap
        chain: where neighbouring swaps interact, its stationary
        distribution deviates from the Boltzmann distribution (on a ring
        model with nearest-neighbour interactions comparable to k_B T, by a
        few per cent of the energy).

    until_accept
        Stop right after the first accepted move and discard the rest of
        the batch. Every decided proposal is then a fresh uniform proposal
        tested at the current configuration, so the decided steps are
        exactly the single-swap Metropolis chain for any batch size. It
        needs more calls per sweep, the more so the higher the acceptance.

    With batch = 1 the two walks are the same chain.

    On the Mo-Nb-Ta-W site-energy model (12^3 cell, batch 8, chains started
    from the same configuration) the two walks agree within the statistical
    error: at 700 K in Ta-W and Mo-Nb-Ta-W; at 460 K in Mo-Nb-Ta-W (E within
    0.02 meV/atom, alpha_1(Mo-Ta) within 0.001, two chains each); and at the
    heat-capacity maximum of Ta-W, 440 K, where a chain switches between the
    ordered and the disordered state over thousands of sweeps, E within
    0.6 +- 0.3 meV/atom (until_conflict lower; five chains each, about 1 K
    of the transition per 0.13 meV/atom). until_accept needs 14 % (Ta-W,
    440 K) to 27 % (Mo-Nb-Ta-W, 460 K) more calls per sweep.

    A call is one host-to-device copy of the proposals, one replay of a
    captured CUDA graph and one device-to-host copy of B + B^2 numbers. The
    configuration and the site energies stay on the device; the host keeps a
    copy of the configuration for proposing and for the observables. Once per
    sweep the site energies are recomputed on the whole cell (`refresh`),
    which also measures how far the halo values deviate from it.

The geometry assumes that every site is equivalent to every other by a
lattice translation (a Bravais lattice, as bcc), so one set of halo offsets
serves all sites; this is checked when the geometry is built.
"""
import numpy as np
import torch

from ..lattice import hop_ball
from ..model import LatticeGraph


def capture(fn, device):
    """CUDA graph of fn() (no arguments, static tensors), or (None, None)
    when capture is not possible; the caller then runs fn eagerly."""
    if not str(device).startswith("cuda"):
        return None, None
    try:
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                fn()
        torch.cuda.current_stream().wait_stream(s)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            out = fn()
        return g, out
    except Exception as exc:                                     # noqa: BLE001
        print(f"[sampler] CUDA graph capture failed ({exc!r}); eager kernel")
        return None, None


class SwapGeometry:
    """Halo, read-out ball and layer levels around one swapped site.

    All sets are stored as integer lattice offsets from the swapped site, so
    they are translated to any site with modular arithmetic.

        off         (n_h, 3) halo offsets, lexicographic; local index = row
        pad         local index of the padding site (= n_h)
        lut         (box,)*3 offset -> local index, -1 outside the halo
        centre      local index of the swapped site itself
        read_off    (n_read, 3) offsets of the read-out ball, site order
        read_local  their local indices
        levels      per layer: rows of the previous level it reads (own) and
                    its neighbour tables into the previous level, per shell
    """

    def __init__(self, cell, shells, reach, n_layers):
        self.cell = cell
        box = cell.box
        adj = cell.adjacency_lists(shells)
        self.hops = 2 * int(reach)
        origin = 0
        sites = np.flatnonzero(hop_ball(adj, [origin], self.hops))
        if len(sites) >= cell.n_sites:
            raise ValueError(f"a halo of {self.hops} hops covers the whole "
                             f"{cell.size}^3 cell; use a larger cell")
        off = cell.min_image(cell.coords[sites] - cell.coords[origin])
        off = off[np.lexsort((off[:, 2], off[:, 1], off[:, 0]))]
        self._check_translation_invariance(adj, off)
        n_h = len(off)
        self.off, self.n_h, self.n_sub, self.pad = off, n_h, n_h + 1, n_h
        lut = -np.ones((box,) * 3, dtype=np.int64)
        lut[tuple((off % box).T)] = np.arange(n_h)
        self.lut = lut
        self.centre = int(lut[0, 0, 0])

        read = np.flatnonzero(hop_ball(adj, [origin], int(reach)))
        self.read_off = cell.min_image(cell.coords[read] - cell.coords[origin])
        self.read_local = lut[tuple((self.read_off % box).T)]
        if (self.read_local < 0).any():
            raise RuntimeError("read-out ball is not inside the halo")
        self._build_levels(adj, shells, int(reach), int(n_layers))

    def _check_translation_invariance(self, adj, off):
        cell = self.cell
        ref = {tuple(o) for o in off.tolist()}
        for b in range(1, len(cell.lattice.basis)):
            s = cell.index(np.asarray(cell.lattice.basis[b]))
            ball = np.flatnonzero(hop_ball(adj, [s], self.hops))
            o = cell.min_image(cell.coords[ball] - cell.coords[s])
            if {tuple(x) for x in o.tolist()} != ref:
                raise ValueError("the swap sampler needs a lattice in which "
                                 "every site is equivalent by translation")

    def _build_levels(self, adj, shells, reach, n_layers):
        cell, box, lut, pad = self.cell, self.cell.box, self.lut, self.pad

        def local_of(sites):
            d = cell.min_image(cell.coords[sites] - cell.coords[0]) % box
            return lut[d[:, 0], d[:, 1], d[:, 2]]

        levels = [np.arange(self.n_sub)]                 # embedding: whole halo
        for lvl in range(1, n_layers + 1):
            if lvl == n_layers:
                loc = self.read_local                   # read-out order
            else:
                hops = reach + (n_layers - lvl)
                loc = local_of(np.flatnonzero(hop_ball(adj, [0], hops)))
                loc = np.sort(loc[loc >= 0])
            levels.append(np.append(loc, pad))
        self.level_sizes = [len(v) for v in levels]

        own, tabs = [], []
        for lvl in range(1, n_layers + 1):
            prev = levels[lvl - 1]
            pos = -np.ones(self.n_sub, dtype=np.int64)
            pos[prev] = np.arange(len(prev))             # pad -> last row
            cur = levels[lvl][:-1]
            own_l = pos[levels[lvl]]
            if (own_l < 0).any():
                raise RuntimeError("level not contained in the previous level")
            own.append(own_l)
            per_shell = []
            for s in shells:
                vec = cell.lattice.shell_vectors(s)
                nb = (self.off[cur][:, None, :] + vec[None]) % box
                k = lut[nb[..., 0], nb[..., 1], nb[..., 2]]
                k = np.where(k < 0, pad, k)
                t = pos[k]
                if (t < 0).any():
                    raise RuntimeError("neighbour outside the previous level")
                t = np.vstack([t, np.full((1, len(vec)), len(prev) - 1)])
                per_shell.append(t)
            tabs.append(per_shell)
        self.level_own, self.level_tabs = own, tabs


class BatchedSwapChain:
    """Proposals and the in-order decision walk of a batch.

    A subclass supplies `call(p, q)` -> (dE (B,), conflict (B, B) bool,
    payload) for B proposals on the current configuration `self.z`, and
    `apply(taken, p, q, payload)`, which performs the accepted moves. The
    walk itself is independent of how dE is evaluated.
    """

    WALKS = ("until_conflict", "until_accept")

    def __init__(self, z, batch, seed, walk="until_conflict"):
        if walk not in self.WALKS:
            raise ValueError(f"walk must be one of {self.WALKS}, not {walk!r}")
        self.z = np.array(z, dtype=np.int64)
        self.n = len(self.z)
        self.B = int(batch)
        self.rng = np.random.default_rng(seed)
        self.walk = walk
        self.one_move_per_call = walk == "until_accept"

    def propose(self, B):
        """B swaps of unlike species: p uniform, q uniform among the sites
        whose species differs from that of p (a symmetric proposal)."""
        z = self.z
        p = self.rng.integers(0, self.n, B)
        q = self.rng.integers(0, self.n, B)
        bad = z[p] == z[q]
        while bad.any():
            q[bad] = self.rng.integers(0, self.n, int(bad.sum()))
            bad = z[p] == z[q]
        return p, q

    def sweep(self, kT):
        """One sweep = n decided proposals. Returns (acceptance, calls)."""
        attempted = accepted = calls = 0
        B = self.B
        while attempted < self.n:
            p, q = self.propose(B)
            u = self.rng.random(B)
            dE, conf, payload = self.call(p, q)
            calls += 1
            acc = u < np.exp(-np.maximum(dE, 0.0) / kT)
            taken = []
            for r in range(B):
                if taken and (self.one_move_per_call or conf[r, taken].any()):
                    break                       # discard the rest
                attempted += 1
                if acc[r]:
                    taken.append(r)
                if attempted >= self.n:
                    break
            if taken:
                self.apply(np.asarray(taken), p, q, payload)
            accepted += len(taken)
        return accepted / attempted, calls

    def call(self, p, q):
        raise NotImplementedError

    def apply(self, taken, p, q, payload):
        raise NotImplementedError


class SwapSampler(BatchedSwapChain):
    """The swap chain on a site-energy model (see the module docstring).

    net      SiteEnergyModel (gnn_gym.model), without a line graph
    cell     gnn_gym.lattice.Supercell of the simulation
    shells   graph shells of the model, model order
    reach    measured reach of the model in hops
    z        start configuration in model species codes
    walk     until_conflict or until_accept (module docstring)
    """

    def __init__(self, net, cell, shells, reach, z, device, batch=8, seed=0,
                 graph=True, walk="until_conflict"):
        super().__init__(z, batch, seed, walk)
        if net.encoder.use_line:
            raise ValueError("the swap sampler needs a site-energy model "
                             "without a line graph")
        self.net, self.dev = net, device
        self.cell, self.box = cell, cell.box
        self.full_graph = LatticeGraph.for_supercell(cell, shells, False,
                                                     device=device)
        net.set_graph(self.full_graph)
        geo = SwapGeometry(cell, shells, reach, net.encoder.n_layers)
        self.geometry = geo
        dev = device
        index_of = -np.ones((self.box,) * 3, dtype=np.int64)
        index_of[tuple(cell.coords.T)] = np.arange(cell.n_sites)
        self.ijk = torch.from_numpy(np.array(cell.coords)).to(dev)
        self.index_of = torch.from_numpy(index_of).to(dev)
        self.off = torch.from_numpy(geo.off).to(dev)
        self.read_off = torch.from_numpy(geo.read_off).to(dev)
        self.lut = torch.from_numpy(geo.lut).to(dev)
        read_lut = np.zeros((self.box,) * 3, dtype=bool)
        ro = geo.read_off % self.box
        read_lut[ro[:, 0], ro[:, 1], ro[:, 2]] = True
        self.read_lut = torch.from_numpy(read_lut).to(dev)
        self.conflict_lut = torch.from_numpy(geo.lut >= 0).to(dev)
        self.level_own = [torch.from_numpy(o).to(dev) for o in geo.level_own]
        self.level_tabs = [[torch.from_numpy(t).to(dev) for t in per_shell]
                           for per_shell in geo.level_tabs]
        self.n_sub, self.pad, self.centre = geo.n_sub, geo.pad, geo.centre
        self.n_h = geo.n_h
        self.pq = torch.zeros(2 * self.B, dtype=torch.long, device=dev)

        self.z_dev = torch.from_numpy(self.z.copy()).to(dev)
        self.eps = torch.from_numpy(self.site_energies(self.z[None, :])[0]).to(dev)
        self.graph, self._out = (capture(self._kernel, dev) if graph
                                 else (None, None))

    # -- full cell ------------------------------------------------------------
    @torch.no_grad()
    def site_energies(self, z_batch):
        """(batch, n_sites) site energies of whole configurations (eV)."""
        zt = torch.from_numpy(np.asarray(z_batch, dtype=np.int64)).to(self.dev)
        return self.net.site_energies(zt).double().cpu().numpy()

    def eps_cpu(self):
        return self.eps.cpu().numpy()

    def refresh(self):
        """Recompute the site energies on the whole cell, return the largest
        deviation of the running values from them (eV), resynchronise."""
        exact = self.site_energies(self.z[None, :])[0]
        dev = float(np.abs(exact - self.eps_cpu()).max())
        self.eps.copy_(torch.from_numpy(exact))
        return dev

    # -- the device part of one call -------------------------------------------
    def _halo_energies(self, z_sub):
        """(2B, n_read + 1) site energies in read-out order, padding last."""
        enc = self.net.encoder
        x = enc.embedding(z_sub)
        for k in range(enc.n_layers):
            xs = x[:, self.level_own[k]]
            parts = [xs] + [x[:, t].mean(2) for t in self.level_tabs[k]]
            x = enc.atom_norm[k](xs + enc.atom_update[k](torch.cat(parts, -1)))
        return self.net.site(x).squeeze(-1)

    @torch.no_grad()
    def _kernel(self):
        B = self.B
        p, q = self.pq[:B], self.pq[B:]
        cen = self.pq                                         # [p, q]
        par = torch.cat([q, p])                               # [q, p]
        c = self.ijk[cen]                                     # (2B, 3)
        pos = (self.off[None] + c[:, None]) % self.box        # (2B, n_h, 3)
        glob = self.index_of[pos[..., 0], pos[..., 1], pos[..., 2]]
        z_sub = torch.empty((2 * B, self.n_sub), dtype=torch.long,
                            device=self.dev)
        z_sub[:, :self.n_h] = self.z_dev[glob]
        z_sub[:, self.pad] = 0                     # never read out
        z_sub[:, self.centre] = self.z_dev[par]    # the swap, seen from here
        d = (self.ijk[par] - c) % self.box
        k = self.lut[d[:, 0], d[:, 1], d[:, 2]]
        k = torch.where(k >= 0, k, torch.full_like(k, self.pad))
        z_sub.scatter_(1, k[:, None], self.z_dev[cen][:, None])
        vals = self._halo_energies(z_sub)[:, :-1]
        rpos = (self.read_off[None] + c[:, None]) % self.box
        read_glob = self.index_of[rpos[..., 0], rpos[..., 1], rpos[..., 2]]
        diff = vals.double() - self.eps[read_glob]
        # Where the read-out balls of one proposal overlap, both carry the
        # same value; count the overlap once.
        dq = (self.ijk[q] - self.ijk[p]) % self.box
        opos = (self.read_off[None] + dq[:, None]) % self.box
        ov = self.read_lut[opos[..., 0], opos[..., 1], opos[..., 2]]
        dE = diff[:B].sum(1) + (diff[B:] * (~ov).double()).sum(1)
        # conflict matrix: any of the four site pairs within the halo
        s = self.ijk[torch.stack([p, q], 1)]                  # (B, 2, 3)
        dd = (s[:, None, :, None, :] - s[None, :, None, :, :]) % self.box
        conf = self.conflict_lut[dd[..., 0], dd[..., 1], dd[..., 2]]
        conf = conf.any(-1).any(-1)                            # (B, B)
        out = torch.cat([dE, conf.reshape(-1).double()])
        return out, vals, read_glob

    def call(self, p, q):
        self.pq.copy_(torch.from_numpy(np.concatenate([p, q])))
        if self.graph is not None:
            self.graph.replay()
            out, vals, read_glob = self._out
        else:
            out, vals, read_glob = self._kernel()
        o = out.cpu().numpy()
        B = self.B
        return o[:B], o[B:].reshape(B, B) > 0.5, (vals, read_glob)

    def apply(self, taken, p, q, payload):
        vals, read_glob = payload
        B = self.B
        pa, qa = p[taken], q[taken]
        zp, zq = self.z[pa].copy(), self.z[qa].copy()
        self.z[pa], self.z[qa] = zq, zp
        k = len(taken)
        packed = torch.from_numpy(
            np.concatenate([pa, qa, zq, zp, taken, B + taken])).to(self.dev)
        self.z_dev[packed[:2 * k]] = packed[2 * k:4 * k]
        rows = packed[4 * k:]
        self.eps[read_glob[rows].reshape(-1)] = vals[rows].reshape(-1).double()
