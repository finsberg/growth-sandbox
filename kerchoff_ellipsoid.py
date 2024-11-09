import shutil
from pathlib import Path

from mpi4py import MPI

import adios4dolfinx
import cardiac_geometries
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


def grow(F, f0, s0, mesh, T, dt_growth, E_f_set=0.0, E_c_set=0.0):
    time = np.arange(0, T + dt_growth, dt_growth)

    comm = mesh.comm
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

    strain_file = Path("growth_lv.bp")
    shutil.rmtree(strain_file, ignore_errors=True)
    vtx = dolfinx.io.VTXWriter(
        comm,
        strain_file,
        [Fg_ff, Fg_cc, Fg_ff_inc, Fg_cc_inc, st_func, sl_func],
        engine="BP5",
    )

    F_g_f_tot = np.ones_like(time)
    F_g_c_tot = np.ones_like(time)
    dx = ufl.Measure("dx", domain=mesh, metadata={"quadrature_degree": 4})

    vol = dolfinx.fem.assemble_scalar(dolfinx.fem.form(dolfinx.fem.Constant(mesh, 1.0) * dx))
    F_g_f_tot_form = dolfinx.fem.form((Fg_ff / vol) * dx)
    F_g_c_tot_form = dolfinx.fem.form((Fg_cc / vol) * dx)

    Fg = Fg_ff * ufl.outer(f0, f0) + Fg_cc * (ufl.Identity(3) - ufl.outer(f0, f0))
    F_new = F * ufl.inv(Fg)
    E_new = 0.5 * (F_new.T * F_new - ufl.Identity(3))
    sl = ufl.inner(E_new * f0, f0) - E_f_set
    st = ufl.inner(E_new * s0, s0) - E_c_set

    # Growth loop
    for i, ti in enumerate(time[1:], start=1):
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

        F_g_f_tot[i] = dolfinx.fem.assemble_scalar(F_g_f_tot_form)
        F_g_c_tot[i] = dolfinx.fem.assemble_scalar(F_g_c_tot_form)

        fig, ax = plt.subplots()
        ax.plot(time[:i], F_g_f_tot[:i], label="Fiber")
        ax.plot(time[:i], F_g_c_tot[:i], label="Cross-fiber")
        ax.set_xlabel("Time [days]")
        ax.set_ylabel("Cumulative growth tensor components")
        ax.legend()
        fig.savefig("cumulative_growth_lv.png")
        plt.close(fig)

    return Fg_ff, Fg_cc


def main(T, dt_growth, E_f_set=0, E_c_set=0):
    log.set_log_level(log.LogLevel.INFO)

    comm = MPI.COMM_WORLD
    geodir = Path("lv")
    if not geodir.exists():
        comm.barrier()
        cardiac_geometries.mesh.lv_ellipsoid(
            outdir=geodir,
            create_fibers=True,
            fiber_space="DG_0",
            psize_ref=7.0,
            fiber_angle_endo=60,
            fiber_angle_epi=-60,
        )

    # If the folder already exist, then we just load the geometry

    geo = cardiac_geometries.geometry.Geometry.from_folder(
        comm=MPI.COMM_WORLD,
        folder=geodir,
    )

    V = dolfinx.fem.functionspace(geo.mesh, ("P", 2, (3,)))
    u = dolfinx.fem.Function(V)
    v = ufl.TestFunction(V)

    mu = dolfinx.fem.Constant(geo.mesh, 10.0)
    kappa = dolfinx.fem.Constant(geo.mesh, 1e4)

    W = dolfinx.fem.functionspace(geo.mesh, ("DG", 0))

    Fg_ff = dolfinx.fem.Function(W, name="Fg_ff_cum")
    Fg_ff.x.array[:] = 1.0
    Fg_cc = dolfinx.fem.Function(W, name="Fg_cc_cum")
    Fg_cc.x.array[:] = 1.0

    Fg = Fg_ff * ufl.outer(geo.f0, geo.f0) + Fg_cc * (ufl.Identity(3) - ufl.outer(geo.f0, geo.f0))
    F = ufl.variable(ufl.grad(u) + ufl.Identity(3))
    Fe = F * ufl.inv(Fg)

    Ce = Fe.T * Fe
    I1e = ufl.tr(Ce)
    Je = ufl.det(Fe)

    psi = (mu / 2.0) * (I1e - 3) + kappa * (Je - 1) ** 2 / 2.0

    P = ufl.diff(psi, F)

    N = ufl.FacetNormal(geo.mesh)
    traction = dolfinx.fem.Constant(geo.mesh, 0.0)
    spring = dolfinx.fem.Constant(geo.mesh, 1.0)
    ds = ufl.Measure("ds", domain=geo.mesh, subdomain_data=geo.ffun)

    n = traction * ufl.det(F) * ufl.inv(F).T * N
    R_neumann = ufl.inner(v, n) * ds(geo.markers["ENDO"][0])

    robin_value = ufl.inner(spring * u, N)
    R_robin = ufl.inner(robin_value * v, N) * ds(geo.markers["EPI"][0]) + ufl.inner(
        robin_value * v,
        N,
    ) * ds(geo.markers["BASE"][0])

    metadata = {"quadrature_degree": 4}
    dx = ufl.Measure("dx", domain=geo.mesh, metadata=metadata)
    R = ufl.inner(ufl.grad(v), P) * dx + R_neumann + R_robin

    problem = NonlinearProblem(R, u, [])
    solver = NewtonSolver(geo.mesh.comm, problem)

    # Set Newton solver options
    solver.atol = 1e-8
    solver.rtol = 1e-8
    solver.convergence_criterion = "incremental"

    # Solve intial problem of stretching the cube just to get the initial displacement
    disp_file = "disp_lv.bp"
    shutil.rmtree(disp_file, ignore_errors=True)
    u.name = "displacement"
    vtx_u = dolfinx.io.VTXWriter(comm, disp_file, [u], engine="BP5")
    vtx_u.write(0.0)

    for lvp in np.linspace(0, 1.0, 3):
        traction.value = lvp
        solver.solve(u)
    vtx_u.write(1.0)

    rerun = False

    if Path("Fg.bp").exists() and not rerun:
        new_Fg_cc = dolfinx.fem.Function(W, name="Fg_cc")
        new_Fg_ff = dolfinx.fem.Function(W, name="Fg_ff")
        adios4dolfinx.read_function("Fg.bp", new_Fg_ff, time=0.0, name="Fg_ff")
        adios4dolfinx.read_function("Fg.bp", new_Fg_cc, time=0.0, name="Fg_cc")
    else:
        new_Fg_ff, new_Fg_cc = grow(
            F=Fe,
            f0=geo.f0,
            s0=geo.s0,
            mesh=geo.mesh,
            T=T,
            dt_growth=dt_growth,
            E_f_set=E_f_set,
            E_c_set=E_c_set,
        )
        adios4dolfinx.write_function("Fg.bp", new_Fg_ff, time=0.0, name="Fg_ff")
        adios4dolfinx.write_function("Fg.bp", new_Fg_cc, time=0.0, name="Fg_cc")

    num_steps = 500
    dalpha = 1 / num_steps
    for j in range(num_steps):
        print(j)
        Fg_ff.x.array[:] = 1 * (1 - j * dalpha) + j * dalpha * new_Fg_ff.x.array
        Fg_cc.x.array[:] = 1 * (1 - j * dalpha) + j * dalpha * new_Fg_cc.x.array
        solver.solve(u)
        vtx_u.write(j + 2)


if __name__ == "__main__":
    main(T=200.0, dt_growth=0.01)
