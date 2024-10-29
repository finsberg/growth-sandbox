import shutil

from mpi4py import MPI

import dolfinx
import numpy as np
import ufl
from dolfinx import log
from dolfinx.fem.petsc import NonlinearProblem
from dolfinx.nls.petsc import NewtonSolver

from growth import ConstantGrowthTensor


def main():
    log.set_log_level(log.LogLevel.INFO)

    comm = MPI.COMM_WORLD
    mesh = dolfinx.mesh.create_unit_cube(
        comm=comm,
        cell_type=dolfinx.mesh.CellType.tetrahedron,
        nx=3,
        ny=3,
        nz=3,
    )

    with dolfinx.io.XDMFFile(comm, "mesh.xdmf", "w") as file:
        file.write_mesh(mesh)

    V = dolfinx.fem.functionspace(mesh, ("P", 2, (3,)))
    u = dolfinx.fem.Function(V)
    v = ufl.TestFunction(V)

    mu = dolfinx.fem.Constant(mesh, 10.0)
    kappa = dolfinx.fem.Constant(mesh, 1e4)

    f0 = ufl.as_vector([1, 0, 0])
    s0 = ufl.as_vector([0, 1, 0])
    n0 = ufl.as_vector([0, 0, 1])

    F = ufl.variable(ufl.grad(u) + ufl.Identity(3))
    Fg = ConstantGrowthTensor(mesh, f0, s0, n0)
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

    disp_file = "results_disp_unit_cube_constant.bp"
    strain_file = "results_strain_unit_cube_constant.bp"
    shutil.rmtree(disp_file, ignore_errors=True)
    shutil.rmtree(strain_file, ignore_errors=True)

    W = dolfinx.fem.functionspace(mesh, ("DG", 1))
    Exx = dolfinx.fem.Function(W, name="Exx")
    Eyy = dolfinx.fem.Function(W, name="Eyy")

    vtx_E = dolfinx.io.VTXWriter(MPI.COMM_WORLD, strain_file, [Exx, Eyy], engine="BP4")
    vtx = dolfinx.io.VTXWriter(MPI.COMM_WORLD, disp_file, [u], engine="BP4")

    vtx.write(0.0)
    vtx_E.write(0.0)
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
    vtx_E.write(1.0)

    # Just set the value of theta_f to 1.5
    Fg.theta_f.value = 1.5

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

    vtx.write(2.0)
    vtx_E.write(2.0)


if __name__ == "__main__":
    main()
