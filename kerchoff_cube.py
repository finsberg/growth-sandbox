import shutil
from pathlib import Path

from mpi4py import MPI

import dolfinx
import matplotlib.pyplot as plt
import numpy as np
import ufl
from dolfinx import log
from dolfinx.fem.petsc import NonlinearProblem
from dolfinx.nls.petsc import NewtonSolver

# global constants
f_ff_max = 0.3
f_f = 150
s_l50 = 0.06
F_ff50 = 1.35
f_l_slope = 40
f_cc_max = 0.1
c_f = 75
s_t50 = 0.07
F_cc50 = 1.28
c_th_slope = 60


def incr_growth_tensor(s_l, s_t, dt, Fg_ff_cum, Fg_cc_cum):
    kff = 1 / (1 + ufl.exp(f_l_slope * (Fg_ff_cum - F_ff50)))
    kss = 1 / (1 + ufl.exp(c_th_slope * (Fg_cc_cum - F_cc50)))

    F_g_i_ff = ufl.conditional(
        ufl.ge(s_l, 0),
        kff * f_ff_max * dt / (1 + ufl.exp(-f_f * (s_l - s_l50))) + 1,
        -f_ff_max * dt / (1 + ufl.exp(f_f * (s_l + s_l50))) + 1,
    )

    F_g_i_cc = ufl.conditional(
        ufl.ge(s_t, 0),
        ufl.sqrt(
            kss * f_cc_max * dt / (1 + ufl.exp(-c_f * (s_t - s_t50))) + 1,
        ),
        ufl.sqrt(
            -f_cc_max * dt / (1 + ufl.exp(c_f * (s_t + s_t50))) + 1,
        ),
    )

    return F_g_i_ff, F_g_i_cc


def grow_unit_cube(lmbda, T, N, E_f_set=0, E_c_set=0):
    log.set_log_level(log.LogLevel.INFO)

    comm = MPI.COMM_WORLD
    mesh = dolfinx.mesh.create_unit_cube(
        comm=comm,
        cell_type=dolfinx.mesh.CellType.tetrahedron,
        nx=3,
        ny=3,
        nz=3,
    )

    V = dolfinx.fem.functionspace(mesh, ("P", 2, (3,)))
    u = dolfinx.fem.Function(V)
    v = ufl.TestFunction(V)

    mu = dolfinx.fem.Constant(mesh, 10.0)
    kappa = dolfinx.fem.Constant(mesh, 1e4)

    f0 = ufl.as_vector([1, 0, 0])
    s0 = ufl.as_vector([0, 1, 0])

    F = ufl.variable(ufl.grad(u) + ufl.Identity(3))
    C = F.T * F

    I1 = ufl.tr(C)
    J = ufl.det(F)

    psi = (mu / 2.0) * (I1 - 3) + kappa * (J - 1) ** 2 / 2.0

    P = ufl.diff(psi, F)

    L = 1.0

    fdim = mesh.topology.dim - 1
    x0_facets = dolfinx.mesh.locate_entities_boundary(mesh, fdim, lambda x: np.isclose(x[0], 0))
    x1_facets = dolfinx.mesh.locate_entities_boundary(mesh, fdim, lambda x: np.isclose(x[0], L))
    y0_facets = dolfinx.mesh.locate_entities_boundary(mesh, fdim, lambda x: np.isclose(x[1], 0))
    z0_facets = dolfinx.mesh.locate_entities_boundary(mesh, fdim, lambda x: np.isclose(x[2], 0))

    # Concatenate and sort the arrays based on facet indices.
    # Left facets marked with 1, right facets with two
    marked_facets = np.hstack([x0_facets, x1_facets, y0_facets, z0_facets])

    marked_values = np.hstack(
        [
            np.full_like(x0_facets, 1),
            np.full_like(x1_facets, 2),
            np.full_like(y0_facets, 3),
            np.full_like(z0_facets, 4),
        ],
    )
    sorted_facets = np.argsort(marked_facets)

    facet_tag = dolfinx.mesh.meshtags(
        mesh,
        fdim,
        marked_facets[sorted_facets],
        marked_values[sorted_facets],
    )

    V0, _ = V.sub(0).collapse()
    u_right = dolfinx.fem.Function(V0)
    zero = dolfinx.fem.Function(V0)

    x0_dofs = dolfinx.fem.locate_dofs_topological((V.sub(0), V0), facet_tag.dim, facet_tag.find(1))
    x1_dofs = dolfinx.fem.locate_dofs_topological((V.sub(0), V0), facet_tag.dim, facet_tag.find(2))
    y0_dofs = dolfinx.fem.locate_dofs_topological((V.sub(1), V0), facet_tag.dim, facet_tag.find(3))
    z0_dofs = dolfinx.fem.locate_dofs_topological((V.sub(2), V0), facet_tag.dim, facet_tag.find(4))

    bcs = [
        dolfinx.fem.dirichletbc(zero, x0_dofs, V.sub(0)),
        dolfinx.fem.dirichletbc(u_right, x1_dofs, V.sub(0)),
        dolfinx.fem.dirichletbc(zero, y0_dofs, V.sub(1)),
        dolfinx.fem.dirichletbc(zero, z0_dofs, V.sub(2)),
    ]

    metadata = {"quadrature_degree": 4}
    dx = ufl.Measure("dx", domain=mesh, metadata=metadata)
    R = ufl.inner(ufl.grad(v), P) * dx

    problem = NonlinearProblem(R, u, bcs)
    solver = NewtonSolver(mesh.comm, problem)

    # Set Newton solver options
    solver.atol = 1e-8
    solver.rtol = 1e-8
    solver.convergence_criterion = "incremental"

    # Solve intial problem of stretching the cube just to get the initial displacement
    disp_file = "disp.bp"
    shutil.rmtree(disp_file, ignore_errors=True)
    u.name = "displacement"
    vtx_u = dolfinx.io.VTXWriter(comm, disp_file, [u], engine="BP5")
    vtx_u.write(0.0)

    disp_bc = 0.1
    for bc_disp in np.linspace(0, disp_bc, 5):
        u_right.x.array[:] = bc_disp
        solver.solve(u)
    vtx_u.write(1.0)

    # time measured in days, N steps
    time = np.linspace(0, T, N + 1)
    dt_growth = T / N

    # Could use quadrature spaces here (but doesn't seem to make any big difference)
    # element = basix.ufl.quadrature_element(
    #     scheme="default",
    #     degree=4,
    #     value_shape=(),
    #     cell=basix.CellType[mesh.ufl_cell().cellname()],
    # )
    # W = dolfinx.fem.functionspace(mesh, element)
    W = dolfinx.fem.functionspace(mesh, ("DG", 0))

    Fg_ff = dolfinx.fem.Function(W, name="Fg_ff_cum")
    Fg_ff.x.array[:] = 1.0
    Fg_cc = dolfinx.fem.Function(W, name="Fg_cc_cum")
    Fg_cc.x.array[:] = 1.0
    Fg_ff_inc = dolfinx.fem.Function(W, name="Fg_ff_inc")
    Fg_ff_inc.x.array[:] = 1.0
    Fg_cc_inc = dolfinx.fem.Function(W, name="Fg_cc_inc")
    Fg_cc_inc.x.array[:] = 1.0
    sl_func = dolfinx.fem.Function(W, name="sl")
    sl_func.x.array[:] = 0.0
    st_func = dolfinx.fem.Function(W, name="st")
    st_func.x.array[:] = 0.0

    strain_file = Path("growth.bp")
    shutil.rmtree(strain_file, ignore_errors=True)
    vtx = dolfinx.io.VTXWriter(
        comm,
        strain_file,
        [Fg_ff, Fg_cc, Fg_ff_inc, Fg_cc_inc, st_func, sl_func],
        engine="BP5",
    )

    F_g_f_tot = np.ones_like(time)
    F_g_c_tot = np.ones_like(time)
    dx = ufl.Measure("dx", domain=mesh)
    vol = dolfinx.fem.assemble_scalar(dolfinx.fem.form(dolfinx.fem.Constant(mesh, 1.0) * dx))
    F_g_f_tot_form = dolfinx.fem.form((Fg_ff / vol) * dx)
    F_g_c_tot_form = dolfinx.fem.form((Fg_cc / vol) * dx)

    F_new = F * ufl.inv(ufl.diag(ufl.as_vector([Fg_ff, Fg_cc, Fg_cc])))
    E_new = 0.5 * (F_new.T * F_new - ufl.Identity(3))
    sl = ufl.inner(E_new * f0, f0) - E_f_set
    st = ufl.inner(E_new * s0, s0) - E_c_set

    # Growth loop
    for i in range(N):
        print("Step ", i)

        F_g_i_ff, F_g_i_cc = incr_growth_tensor(sl, st, dt_growth, Fg_ff, Fg_cc)
        Fg_ff_expr = Fg_ff * F_g_i_ff
        Fg_cc_expr = Fg_cc * F_g_i_cc

        for func, expr in zip(
            [Fg_ff, Fg_cc, Fg_ff_inc, Fg_cc_inc, sl_func, st_func],
            [Fg_ff_expr, Fg_cc_expr, F_g_i_ff, F_g_i_cc, sl, st],
        ):
            func.interpolate(
                dolfinx.fem.Expression(expr, W.element.interpolation_points()),
            )
        vtx.write(i)

        F_g_f_tot[i + 1] = dolfinx.fem.assemble_scalar(F_g_f_tot_form)
        F_g_c_tot[i + 1] = dolfinx.fem.assemble_scalar(F_g_c_tot_form)

        lmbda = 1 + disp_bc
        fig, ax = plt.subplots()
        ax.plot(time[:i], F_g_f_tot[:i], label="Fiber")
        ax.plot(time[:i], F_g_c_tot[:i], label="Cross-fiber")
        ax.plot(time, np.ones_like(time) * lmbda, ":")
        ax.plot(time, np.ones_like(time) * 1 / np.sqrt(lmbda), ":")

        ax.set_title(rf"Uniaxial stretch, $\lambda$ = {lmbda}")
        ax.set_xlabel("Time [days]")
        ax.set_ylabel("Cumulative growth tensor components")
        ax.legend()
        fig.savefig("cumulative_growth.png")
        plt.close(fig)


if __name__ == "__main__":
    grow_unit_cube(lmbda=1.1, T=300, N=5000, E_f_set=0, E_c_set=0)
