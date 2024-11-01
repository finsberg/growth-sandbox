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
        volume = dolfinx.fem.assemble_scalar(
            dolfinx.fem.form(dolfinx.fem.Constant(self.mesh, 1.0) * ufl.dx(domain=self.mesh)),
        )
        self._error = dolfinx.fem.form(
            ((self.u - self.u_old) ** 2 / volume)
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
    dt: float = 0.005

    def amps(self, E):
        """Algebraic maximum principal strain"""
        Ess = ufl.inner(E * self.s0, self.s0)
        Enn = ufl.inner(E * self.n0, self.n0)
        Esn = ufl.inner(E * self.s0, self.n0)
        Ens = ufl.inner(E * self.n0, self.s0)
        return (Enn + Ess) / 2 + ufl.sqrt(((Enn - Ess) / 2) ** 2 + (Esn * Ens))

    def initialize(self):
        self.t = dolfinx.fem.Constant(self.mesh, 0.0)

        element = basix.ufl.quadrature_element(
            scheme="default",
            degree=4,
            value_shape=(),
            cell=basix.CellType[self.mesh.ufl_cell().cellname()],
        )

        self.W = dolfinx.fem.functionspace(self.mesh, element)
        self.Eff_set = dolfinx.fem.Function(self.W)
        self.Ecross_set = dolfinx.fem.Function(self.W)

        # Define cumulative growth tensors
        self.Fg_ff_cum = dolfinx.fem.Function(self.W)
        self.Fg_ff_cum.x.array[:] = 1.0
        self.Fg_cc_cum = dolfinx.fem.Function(self.W)
        self.Fg_cc_cum.x.array[:] = 1.0

        self.Fg_ff_inc = dolfinx.fem.Function(self.W)
        self.Fg_ff_inc.x.array[:] = 1.0
        self.Fg_cc_inc = dolfinx.fem.Function(self.W)
        self.Fg_cc_inc.x.array[:] = 1.0
        self.update()

    def update(self):
        F = ufl.grad(self.u) + ufl.Identity(3)
        Fe = F * ufl.inv(self.tensor)
        Ce = Fe.T * Fe
        E = 0.5 * (Ce - ufl.Identity(3))

        Eff = ufl.inner(E * self.f0, self.f0)
        Ecross_max = self.amps(E)

        sl = Eff - self.Eff_set
        st = Ecross_max - self.Ecross_set

        kff = 1 / (1 + ufl.exp(self.f_l_slope * (self.Fg_ff_cum - self.F_ff50)))
        kss = 1 / (1 + ufl.exp(self.c_th_slope * (self.Fg_cc_cum - self.F_cc50)))

        # Define incremental growth
        Fg_ff_inc = ufl.conditional(
            ufl.ge(sl, 0),
            kff * self.f_ff_max * self.dt / (1 + ufl.exp(-self.f_f * (sl - self.s_l50))) + 1,
            -self.f_ff_max * self.dt / (1 + ufl.exp(self.f_f * (Eff + self.s_l50))) + 1,
        )

        Fg_cc_inc = ufl.conditional(
            ufl.ge(st, 0),
            ufl.sqrt(
                kss * self.f_cc_max * self.dt / (1 + ufl.exp(-self.c_f * (st - self.s_t50))) + 1,
            ),
            ufl.sqrt(
                -self.f_cc_max * self.dt / (1 + ufl.exp(self.c_f * (st + self.s_t50))) + 1,
            ),
        )
        self.Fg_ff_inc.interpolate(
            dolfinx.fem.Expression(Fg_ff_inc, self.W.element.interpolation_points()),
        )
        self.Fg_cc_inc.interpolate(
            dolfinx.fem.Expression(Fg_cc_inc, self.W.element.interpolation_points()),
        )

    def specify_setpoint(self, u):
        F = ufl.grad(u) + ufl.Identity(3)
        Fe = F * ufl.inv(self.tensor)
        Ce = Fe.T * Fe
        E = 0.5 * (Ce - ufl.Identity(3))

        Eff = ufl.inner(E * self.f0, self.f0)
        self.Eff_set.interpolate(
            dolfinx.fem.Expression(Eff, self.W.element.interpolation_points()),
        )
        Ecross_max = self.amps(E)
        self.Ecross_set.interpolate(
            dolfinx.fem.Expression(Ecross_max, self.W.element.interpolation_points()),
        )

    @property
    def tensor(self):
        return self.Fg_ff_cum * ufl.outer(self.f0, self.f0) + self.Fg_cc_cum * (
            ufl.Identity(3) - ufl.outer(self.f0, self.f0)
        )
        # return (
        #     self.Fg_ff_cum * ufl.outer(self.f0, self.f0)
        #     + self.Fg_cc_cum * ufl.outer(self.s0, self.s0)
        #     + self.Fg_cc_cum * ufl.outer(self.n0, self.n0)
        # )

    def apply_stimulus(self, u: dolfinx.fem.Function) -> None:
        super().apply_stimulus(u)
        self.update()
        self.t.value += self.dt

        self.Fg_ff_cum.interpolate(
            dolfinx.fem.Expression(
                self.Fg_ff_inc * self.Fg_ff_cum,
                self.W.element.interpolation_points(),
            ),
        )
        self.Fg_cc_cum.interpolate(
            dolfinx.fem.Expression(
                self.Fg_cc_inc * self.Fg_cc_cum,
                self.W.element.interpolation_points(),
            ),
        )
