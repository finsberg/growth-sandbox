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

from growth import SimpleStrain


def main():
    log.set_log_level(log.LogLevel.INFO)

    comm = MPI.COMM_WORLD
    geodir = Path("slab")
    if not geodir.exists():
        comm.barrier()
        cardiac_geometries.mesh.slab(
            comm=comm,
            lx=1.0,
            ly=1.0,
            lz=1.0,
            dx=0.5,
            outdir=geodir,
            create_fibers=True,
            fiber_space="Quadrature_4",
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
    Fg = SimpleStrain(geo.mesh, f0, s0, n0)
    Fe = F * ufl.inv(Fg.tensor)

    C = Fe.T * Fe
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

    facet_tag = geo.ffun
    markers = geo.markers

    V0, _ = V.sub(0).collapse()
    u_right = dolfinx.fem.Function(V0)
    zero = dolfinx.fem.Function(V0)

    x0_dofs = dolfinx.fem.locate_dofs_topological(
        (V.sub(0), V0),
        facet_tag.dim,
        facet_tag.find(markers["X0"][0]),
    )
    x1_dofs = dolfinx.fem.locate_dofs_topological(
        (V.sub(0), V0),
        facet_tag.dim,
        facet_tag.find(markers["X1"][0]),
    )
    y0_dofs = dolfinx.fem.locate_dofs_topological(
        (V.sub(1), V0),
        facet_tag.dim,
        facet_tag.find(markers["Y0"][0]),
    )
    z0_dofs = dolfinx.fem.locate_dofs_topological(
        (V.sub(2), V0),
        facet_tag.dim,
        facet_tag.find(markers["Z0"][0]),
    )

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

    disp_file = "results_disp_slab_simple_strain.bp"
    strain_file_init = Path("results_strain_slab_simple_strain_init.xdmf")
    strain_file_end = Path("results_strain_slab_simple_strain_end.xdmf")
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
    Exx = dolfinx.fem.Function(W, name="Exx")
    Eyy = dolfinx.fem.Function(W, name="Eyy")

    vtx = dolfinx.io.VTXWriter(MPI.COMM_WORLD, disp_file, [u], engine="BP4")

    vtx.write(0.0)
    for bc_disp in np.linspace(0, 0.1, 5):
        u_right.x.array[:] = bc_disp
        solver.solve(u)

    print("J = ", dolfinx.fem.assemble_scalar(dolfinx.fem.form(J * dx)))
    print("Jg = ", dolfinx.fem.assemble_scalar(dolfinx.fem.form(Jg * dx)))
    print("Je = ", dolfinx.fem.assemble_scalar(dolfinx.fem.form(Je * dx)))
    Exx.interpolate(
        dolfinx.fem.Expression(
            ufl.inner(E * f0, f0),
            Exx.function_space.element.interpolation_points(),
        ),
    )
    Eyy.interpolate(
        dolfinx.fem.Expression(
            ufl.inner(E * s0, s0),
            Eyy.function_space.element.interpolation_points(),
        ),
    )

    vtx.write(1.0)
    scifem.xdmf.create_pointcloud(strain_file_init, [Exx, Eyy])

    N = 400
    for i in range(2, N + 2):
        Fg.apply_stimulus(u)
        solver.solve(u)
        print("J = ", dolfinx.fem.assemble_scalar(dolfinx.fem.form(J * dx)))
        print("Jg = ", dolfinx.fem.assemble_scalar(dolfinx.fem.form(Jg * dx)))
        print("Je = ", dolfinx.fem.assemble_scalar(dolfinx.fem.form(Je * dx)))
        vtx.write(i)
        change = Fg.change()
        print("Change = ", change)
        if change < 1e-12:
            break

    Exx.interpolate(
        dolfinx.fem.Expression(
            ufl.inner(E * f0, f0),
            Exx.function_space.element.interpolation_points(),
        ),
    )
    Eyy.interpolate(
        dolfinx.fem.Expression(
            ufl.inner(E * s0, s0),
            Eyy.function_space.element.interpolation_points(),
        ),
    )

    scifem.xdmf.create_pointcloud(strain_file_end, [Exx, Eyy])


if __name__ == "__main__":
    main()
