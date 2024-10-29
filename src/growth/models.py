import abc
from dataclasses import dataclass

import basix
import dolfinx
import ufl


class GrowthTensor(abc.ABC):
    @abc.abstractmethod
    def apply_stimulus(self, u: dolfinx.fem.Function): ...

    @property
    @abc.abstractmethod
    def tensor(self): ...


@dataclass
class ConstantGrowthTensor:
    mesh: dolfinx.mesh.Mesh
    f0: dolfinx.fem.Function | ufl.Coefficient
    s0: dolfinx.fem.Function | ufl.Coefficient
    n0: dolfinx.fem.Function | ufl.Coefficient
    theta_f: dolfinx.fem.Constant | float = 1.0
    theta_s: dolfinx.fem.Constant | float = 1.0
    theta_n: dolfinx.fem.Constant | float = 1.0

    def __post_init__(self):
        if not isinstance(self.theta_f, dolfinx.fem.Constant):
            self.theta_f = dolfinx.fem.Constant(self.mesh, self.theta_f)
        if not isinstance(self.theta_s, dolfinx.fem.Constant):
            self.theta_s = dolfinx.fem.Constant(self.mesh, self.theta_s)
        if not isinstance(self.theta_n, dolfinx.fem.Constant):
            self.theta_n = dolfinx.fem.Constant(self.mesh, self.theta_n)

    @property
    def tensor(self):
        return (
            self.theta_f * ufl.outer(self.f0, self.f0)
            + self.theta_s * ufl.outer(self.s0, self.s0)
            + self.theta_n * ufl.outer(self.n0, self.n0)
        )

    def apply_stimulus(self, u):
        # This growth tensor does not depend on any external stimulus
        ...


class StrainBasedGrowthTensor(GrowthTensor, abc.ABC):
    mesh: dolfinx.mesh.Mesh
    f0: dolfinx.fem.Function | ufl.Coefficient
    s0: dolfinx.fem.Function | ufl.Coefficient
    n0: dolfinx.fem.Function | ufl.Coefficient
    theta_f: ufl.Coefficient
    theta_s: ufl.Coefficient
    theta_n: ufl.Coefficient

    def __post_init__(self):
        self.u = dolfinx.fem.Function(dolfinx.fem.functionspace(self.mesh, ("P", 2, (3,))))
        self.u_old = dolfinx.fem.Function(dolfinx.fem.functionspace(self.mesh, ("P", 2, (3,))))
        self._error = dolfinx.fem.form(
            (self.u - self.u_old) ** 2
            * ufl.dx(domain=self.mesh, metadata={"quadrature_degree": 4}),
        )
        self.initialize()

    @abc.abstractmethod
    def initialize(self): ...

    @property
    def tensor(self):
        return (
            self.theta_f * ufl.outer(self.f0, self.f0)
            + self.theta_s * ufl.outer(self.s0, self.s0)
            + self.theta_n * ufl.outer(self.n0, self.n0)
        )

    def apply_stimulus(self, u: dolfinx.fem.Function) -> None:
        self.u_old.interpolate(self.u)
        self.u.interpolate(u)

    def change(self):
        return dolfinx.fem.assemble_scalar(self._error)


@dataclass
class SimpleStrain(StrainBasedGrowthTensor):
    mesh: dolfinx.mesh.Mesh
    f0: dolfinx.fem.Function | ufl.Coefficient
    s0: dolfinx.fem.Function | ufl.Coefficient
    n0: dolfinx.fem.Function | ufl.Coefficient
    set_point: dolfinx.fem.Constant | float = 0.0
    d: dolfinx.fem.Constant | float = 0.1

    def initialize(self):
        F = ufl.grad(self.u) + ufl.Identity(3)
        C = F.T * F
        E = 0.5 * (C - ufl.Identity(3))

        Eff = ufl.inner(E * self.f0, self.f0)
        Ess = ufl.inner(E * self.s0, self.s0)
        Enn = ufl.inner(E * self.n0, self.n0)

        self.theta_f = 1.0 + self.d * ufl.sqrt(1 + 2 * Eff) - self.set_point
        self.theta_s = 1.0 + self.d * ufl.sqrt(1 + 2 * Ess) - self.set_point
        self.theta_n = 1.0 + self.d * ufl.sqrt(1 + 2 * Enn) - self.set_point


def k_growth(F_g_cum, slope, F_50):
    return 1 / (1 + ufl.exp(slope * (F_g_cum - F_50)))


@dataclass
class KOM(StrainBasedGrowthTensor):
    mesh: dolfinx.mesh.Mesh
    f0: dolfinx.fem.Function | ufl.Coefficient
    s0: dolfinx.fem.Function | ufl.Coefficient
    n0: dolfinx.fem.Function | ufl.Coefficient
    f_ff_max: float = 0.3
    f_f: float = 150
    s_l50: float = 0.06
    F_ff50: float = 1.35
    f_l_slope: float = 40
    f_cc_max: float = 0.1
    c_f: float = 75
    s_t50: float = 0.07
    F_cc50: float = 1.28
    c_th_slope: float = 60
    dt: float = 0.1

    def initialize(self):
        F = ufl.grad(self.u) + ufl.Identity(3)
        C = F.T * F
        E = 0.5 * (C - ufl.Identity(3))

        element = basix.ufl.quadrature_element(
            scheme="default",
            degree=4,
            value_shape=(3, 3),
            cell=basix.CellType[self.mesh.ufl_cell().cellname()],
        )

        self.W = dolfinx.fem.functionspace(self.mesh, element)
        self.Fg_tot = dolfinx.fem.Function(self.W)
        self.Fg_tot.interpolate(
            dolfinx.fem.Expression(
                ufl.grad(ufl.SpatialCoordinate(self.mesh)),
                self.W.element.interpolation_points(),
            ),
        )

        Fg_ff = ufl.inner(self.Fg_tot * self.f0, self.f0)
        Fg_ss = ufl.inner(self.Fg_tot * self.s0, self.s0)

        Eff = ufl.inner(E * self.f0, self.f0)
        Efs = ufl.inner(E * self.f0, self.s0)
        Esf = ufl.inner(E * self.s0, self.f0)
        Ess = ufl.inner(E * self.s0, self.s0)

        alg_max_princ_strain = (Eff + Ess) / 2 + ufl.sqrt(((Eff - Ess) / 2) ** 2 + (Efs * Esf))

        self.theta_f = ufl.conditional(
            ufl.ge(Eff, 0),
            k_growth(Fg_ff, self.f_l_slope, self.F_ff50)
            * self.f_ff_max
            * self.dt
            / (1 + ufl.exp(-self.f_f * (Eff - self.s_l50)))
            + 1,
            -self.f_ff_max * self.dt / (1 + ufl.exp(self.f_f * (Eff + self.s_l50))) + 1,
        )

        self.theta_n = self.theta_s = ufl.conditional(
            ufl.ge(alg_max_princ_strain, 0),
            ufl.sqrt(
                k_growth(Fg_ss, self.c_th_slope, self.F_cc50)
                * self.f_cc_max
                * self.dt
                / (1 + ufl.exp(-self.c_f * (alg_max_princ_strain - self.s_t50)))
                + 1,
            ),
            ufl.sqrt(
                -self.f_cc_max
                * self.dt
                / (1 + ufl.exp(self.c_f * (alg_max_princ_strain + self.s_t50)))
                + 1,
            ),
        )
        self.Fg_inc = (
            self.theta_f * ufl.outer(self.f0, self.f0)
            + self.theta_s * ufl.outer(self.s0, self.s0)
            + self.theta_n * ufl.outer(self.n0, self.n0)
        )

    @property
    def tensor(self):
        return self.Fg_tot

    def apply_stimulus(self, u: dolfinx.fem.Function) -> None:
        super().apply_stimulus(u)
        self.Fg_tot.interpolate(
            dolfinx.fem.Expression(
                self.Fg_tot * self.Fg_inc,
                self.W.element.interpolation_points(),
            ),
        )
