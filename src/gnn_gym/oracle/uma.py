"""The uMLIP reference calculator: UMA through fairchem, wrapped for ASE.

Every reference energy of the paper comes from this class: relaxations of
the defect-free cell (cell shape and positions), ionic relaxations of the
jump endpoints at fixed cell, and climbing-image NEB between them. All
optimisations use FIRE. The predictor is loaded once and shared by all
calculator objects; ASE requires a separate calculator object per NEB image,
but sharing the network underneath them is exact.

Convergence is taken from the return value of `Optimizer.run`, which reports
it consistently across ASE versions.
"""
import time
import warnings

import numpy as np

# FrechetCellFilter warns about the accuracy of the matrix logarithm on every
# cell step at the 1e-13 level; harmless.
warnings.filterwarnings("ignore", message="logm result may be inaccurate",
                        category=RuntimeWarning)


class Uma:
    def __init__(self, model, task, device="auto", log=print):
        import torch
        from fairchem.core import pretrained_mlip
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        t0 = time.time()
        self.predictor = pretrained_mlip.get_predict_unit(model, device=device)
        self.model, self.task = model, task
        log(f"[uma] {model} ({task}) on {device}, loaded in "
            f"{time.time() - t0:.1f} s")

    @classmethod
    def from_recipe(cls, recipe, **kwargs):
        return cls(recipe["uma"]["model"], recipe["uma"]["task"], **kwargs)

    def calc(self):
        from fairchem.core import FAIRChemCalculator
        return FAIRChemCalculator(self.predictor, task_name=self.task)

    def energy(self, atoms):
        """Single-point energy (eV)."""
        a = atoms.copy()
        a.calc = self.calc()
        E = float(a.get_potential_energy())
        a.calc = None
        return E

    def relax(self, atoms, fmax, steps=2000):
        """Ionic relaxation at fixed cell -> (atoms, E, converged, steps)."""
        from ase.optimize import FIRE
        a = atoms.copy()
        a.calc = self.calc()
        opt = FIRE(a, logfile=None)
        conv = bool(opt.run(fmax=fmax, steps=steps))
        E = float(a.get_potential_energy())
        n = opt.get_number_of_steps()
        a.calc = None
        return a, E, conv, n

    def relax_cell(self, atoms, fmax, steps=2000):
        """Cell shape and ions -> (atoms, E, converged, steps)."""
        from ase.filters import FrechetCellFilter
        from ase.optimize import FIRE
        a = atoms.copy()
        a.calc = self.calc()
        opt = FIRE(FrechetCellFilter(a), logfile=None)
        conv = bool(opt.run(fmax=fmax, steps=steps))
        E = float(a.get_potential_energy())
        n = opt.get_number_of_steps()
        a.calc = None
        return a, E, conv, n

    def neb(self, initial, final, n_images, fmax, steps=500, k=0.1,
            climb=True, method="improvedtangent"):
        """NEB on a linear interpolation -> (images, energies, conv, steps).

        n_images counts the endpoints. The tangent method is always given
        explicitly, independent of the ASE default.
        """
        from ase.mep import NEB
        from ase.optimize import FIRE
        images = [initial.copy()]
        for i in range(1, n_images - 1):
            img = initial.copy()
            f = i / (n_images - 1)
            img.positions = (1 - f) * initial.positions + f * final.positions
            images.append(img)
        images.append(final.copy())
        for img in images:
            img.calc = self.calc()
        band = NEB(images, k=k, climb=climb, method=method)
        opt = FIRE(band, logfile=None)
        conv = bool(opt.run(fmax=fmax, steps=steps))
        energies = [float(img.get_potential_energy()) for img in images]
        nsteps = opt.get_number_of_steps()
        out = [img.copy() for img in images]
        for img in images:
            img.calc = None
        return out, energies, conv, nsteps


def neb_labels(energies, E_initial, E_final):
    """Labels of one jump from the NEB energies and the relaxed endpoints."""
    E_sad = max(energies)
    fwd = E_sad - energies[0]
    bwd = E_sad - energies[-1]
    dE = E_final - E_initial
    return dict(dE=dE, E_saddle=E_sad, E_fwd=fwd, E_bwd=bwd,
                B_sym=0.5 * (fwd + bwd), saddle_image=int(np.argmax(energies)),
                db_residual=fwd - bwd - dE)
