import shutil
from pathlib import Path

from mpi4py import MPI

import basix
import cardiac_geometries
import dolfinx
import numpy as np
import scifem
import ufl
from dolfinx import log
from dolfinx.fem.petsc import NonlinearProblem
from dolfinx.nls.petsc import NewtonSolver

from growth import KOM


def main():
    log.set_log_level(log.LogLevel.INFO)

    comm = MPI.COMM_WORLD
    geodir = Path("lv")
    if not geodir.exists():
        comm.barrier()
        cardiac_geometries.mesh.lv_ellipsoid(
            outdir=geodir,
            create_fibers=True,
            fiber_space="Quadrature_4",
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

    f0 = geo.f0
    s0 = geo.s0
    n0 = geo.n0
    mesh = geo.mesh

    F = ufl.variable(ufl.grad(u) + ufl.Identity(3))
    Fg = KOM(geo.mesh, f0, s0, n0)
    Fe = F * ufl.inv(Fg.tensor)

    C = F.T * F
    # Cg = Fg.tensor.T * Fg.tensor
    Ce = Fe.T * Fe

    E = 0.5 * (C - ufl.Identity(3))
    # Eg = 0.5 * (Cg - ufl.Identity(3))
    # Ee = 0.5 * (Ce - ufl.Identity(3))

    I1e = ufl.tr(Ce)
    Je = ufl.det(Fe)
    J = ufl.det(F)
    Jg = ufl.det(Fg.tensor)

    psi = (mu / 2.0) * (I1e - 3) + kappa * (Je - 1) ** 2 / 2.0

    P = ufl.diff(psi, F)

    N = ufl.FacetNormal(mesh)
    traction = dolfinx.fem.Constant(mesh, 0.0)
    spring = dolfinx.fem.Constant(mesh, 1.0)
    ds = ufl.Measure("ds", domain=mesh, subdomain_data=geo.ffun)

    n = traction * ufl.det(F) * ufl.inv(F).T * N
    R_neumann = ufl.inner(v, n) * ds(geo.markers["ENDO"][0])

    robin_value = ufl.inner(spring * u, N)
    R_robin = ufl.inner(robin_value * v, N) * ds(geo.markers["EPI"][0]) + ufl.inner(
        robin_value * v,
        N,
    ) * ds(geo.markers["BASE"][0])

    metadata = {"quadrature_degree": 4}
    dx = ufl.Measure("dx", domain=mesh, metadata=metadata)
    R = ufl.inner(ufl.grad(v), P) * dx + R_neumann + R_robin

    problem = NonlinearProblem(R, u, [])
    solver = NewtonSolver(mesh.comm, problem)

    # Set Newton solver options
    solver.atol = 1e-8
    solver.rtol = 1e-8
    solver.convergence_criterion = "incremental"

    disp_file = "results_disp_lv_KOM.bp"
    strain_file_init = Path("results_strain_lv_KOM_init.xdmf")
    strain_file_end = Path("results_strain_lv_KOM_end.xdmf")
    shutil.rmtree(disp_file, ignore_errors=True)
    strain_file_init.unlink(missing_ok=True)
    strain_file_init.with_suffix(".h5").unlink(missing_ok=True)
    strain_file_end.unlink(missing_ok=True)
    strain_file_end.with_suffix(".h5").unlink(missing_ok=True)

    shutil.rmtree(disp_file, ignore_errors=True)

    element = basix.ufl.quadrature_element(
        scheme="default",
        degree=4,
        value_shape=(),
        cell=basix.CellType[mesh.ufl_cell().cellname()],
    )
    W = dolfinx.fem.functionspace(mesh, element)
    int_points = W.element.interpolation_points()
    Exx = dolfinx.fem.Function(W, name="Exx")
    Eyy = dolfinx.fem.Function(W, name="Eyy")
    Fg_ff_inc = dolfinx.fem.Function(W, name="Fg_ff_inc")
    Fg_cc_inc = dolfinx.fem.Function(W, name="Fg_cc_inc")
    Fg_ff_cum = dolfinx.fem.Function(W, name="Fg_ff_cum")
    Fg_cc_cum = dolfinx.fem.Function(W, name="Fg_cc_cum")
    vol = dolfinx.fem.assemble_scalar(dolfinx.fem.form(dolfinx.fem.Constant(mesh, 1.0) * ufl.dx))

    vtx = dolfinx.io.VTXWriter(MPI.COMM_WORLD, disp_file, [u], engine="BP4")

    vtx.write(0.0)
    for lvp in np.linspace(0, 1.0, 3):
        traction.value = lvp
        solver.solve(u)

    # Define set point
    Fg.specify_setpoint(u)

    for lvp in np.linspace(1.0, 2.0, 3):
        traction.value = lvp
        solver.solve(u)

    print("J = ", dolfinx.fem.assemble_scalar(dolfinx.fem.form(J * dx)) / vol)
    print("Jg = ", dolfinx.fem.assemble_scalar(dolfinx.fem.form(Jg * dx)) / vol)
    print("Je = ", dolfinx.fem.assemble_scalar(dolfinx.fem.form(Je * dx)) / vol)

    Exx.interpolate(dolfinx.fem.Expression(ufl.inner(E * f0, f0), int_points))
    Eyy.interpolate(dolfinx.fem.Expression(ufl.inner(E * s0, s0), int_points))
    for f, name in [
        (Fg_ff_inc, "Fg_ff_inc"),
        (Fg_cc_inc, "Fg_cc_inc"),
        (Fg_ff_cum, "Fg_ff_cum"),
        (Fg_cc_cum, "Fg_cc_cum"),
    ]:
        f.interpolate(dolfinx.fem.Expression(getattr(Fg, name), int_points))

    vtx.write(1.0)

    i = 2.0
    M = 20
    scifem.xdmf.create_pointcloud(
        strain_file_init,
        [Exx, Eyy, Fg_ff_inc, Fg_cc_inc, Fg_ff_cum, Fg_cc_cum],
    )
    for i in range(2, M + 2):
        for i, ti in enumerate(np.arange(0, 1.0, Fg.dt), start=2):
            Fg.apply_stimulus(u)
            # print("Solve for ti = ", ti)
            solver.solve(u)
            print("J = ", dolfinx.fem.assemble_scalar(dolfinx.fem.form(J * dx)) / vol)
            print("Jg = ", dolfinx.fem.assemble_scalar(dolfinx.fem.form(Jg * dx)) / vol)
            print("Je = ", dolfinx.fem.assemble_scalar(dolfinx.fem.form(Je * dx)) / vol)
            print(
                "Fg_ff_cum = ",
                dolfinx.fem.assemble_scalar(dolfinx.fem.form(Fg.Fg_ff_cum * ufl.dx)) / vol,
            )
            print(
                "Fg_ff_inc = ",
                dolfinx.fem.assemble_scalar(dolfinx.fem.form(Fg.Fg_ff_inc * ufl.dx)) / vol,
            )
        vtx.write(i)
        change = Fg.change()
        print("Change = ", change)
        # if change < 1e-5:
        #     break

        for f, name in [
            (Fg_ff_inc, "Fg_ff_inc"),
            (Fg_cc_inc, "Fg_cc_inc"),
            (Fg_ff_cum, "Fg_ff_cum"),
            (Fg_cc_cum, "Fg_cc_cum"),
        ]:
            f.interpolate(dolfinx.fem.Expression(getattr(Fg, name), int_points))

        scifem.xdmf.create_pointcloud(
            strain_file_init,
            [Exx, Eyy, Fg_ff_inc, Fg_cc_inc, Fg_ff_cum, Fg_cc_cum],
        )


if __name__ == "__main__":
    main()
