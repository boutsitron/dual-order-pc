import time

import firedrake as fd
from firedrake import PETSc


def solve_problem(solver_options, n_iter, print_iterations=False):
    mesh = fd.UnitSquareMesh(20, 20)
    mh = fd.MeshHierarchy(mesh, 5)
    mesh = mh[-1]  # Finest mesh
    V = fd.FunctionSpace(mesh, "CG", 1)
    PETSc.Sys.Print(f"DOFs: {V.dim()}")
    u = fd.Function(V)
    v = fd.TestFunction(V)
    a = fd.inner(fd.grad(u), fd.grad(v)) * fd.dx + fd.Constant(0.1) * u * u * v * fd.dx
    bcs = [fd.DirichletBC(V, fd.Constant(0), "on_boundary")]
    L = fd.Constant(1) * v * fd.dx
    F = a - L

    problem = fd.NonlinearVariationalProblem(F, u, bcs=bcs)
    solver = fd.NonlinearVariationalSolver(problem, solver_parameters=solver_options)

    start = time.time()
    for i in range(n_iter):
        if print_iterations:
            PETSc.Sys.Print(f"Iteration {i}")
        u.assign(0)
        solver.solve()
        if print_iterations:
            PETSc.Sys.Print(" ")
    end = time.time()
    PETSc.Sys.Print(f"Time taken: {end - start} seconds")


# Classic solver options (direct solve)
solver_options_classic = {
    "ksp_type": "preonly",
    "pc_type": "lu",
    "pc_factor_mat_solver_type": "mumps",
    "mat_mumps_icntl_14": 200,
    "mat_mumps_icntl_24": 1,
    "snes_monitor": None,
}

# AssembledPC solver options
solver_options_assembled = {
    "mat_type": "matfree",
    "ksp_type": "preonly",
    "pc_type": "python",
    "pc_python_type": "firedrake.AssembledPC",
    "assembled_pc_type": "lu",
    "assembled_pc_factor_mat_solver_type": "mumps",
    "assembled_mat_mumps_icntl_14": 200,
    "assembled_mat_mumps_icntl_24": 1,
    "snes_monitor": None,
}

# DualOrderPC solver options
solver_options_dual_order_pc = {
    "snes_rtol": 1e-6,
    "snes_atol": 1e-7,
    "ksp_type": "preonly",
    "mat_type": "matfree",  # Changed from aij
    "pc_type": "python",  # Changed from lu
    "pc_python_type": "dual_order_pc.DualOrderPC",
    "mat_mumps_icntl_14": 200,
    "mat_mumps_icntl_24": 1,
    "snes_type": "newtonls",
    "snes_max_it": 50,
    "snes_linesearch_type": "bt",
    "snes_linesearch_max_it": 10,
    "snes_linesearch_minlambda": 1e-4,
    "snes_monitor": None,
}

n_iter = 10

# Solve with classic options
PETSc.Sys.Print("\nSolving with classic options:")
solve_problem(solver_options_classic, n_iter, print_iterations=True)

# Solve with AssembledPC options
PETSc.Sys.Print("\nSolving with AssembledPC:")
solve_problem(solver_options_assembled, n_iter, print_iterations=True)

# Solve with DualOrderPC options
PETSc.Sys.Print("\nSolving with DualOrderPC:")
solve_problem(solver_options_dual_order_pc, n_iter, print_iterations=True)
