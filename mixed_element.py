from mpi4py import MPI
from petsc4py import PETSc
import numpy as np

import ufl
import dolfinx
from dolfinx.fem import Constant, Function, functionspace, assemble_scalar, dirichletbc, form, locate_dofs_geometrical, locate_dofs_topological, set_bc
from dolfinx import default_scalar_type
from dolfinx.fem.petsc import assemble_matrix, assemble_vector, apply_lifting, create_vector, set_bc
from dolfinx import mesh
from basix.ufl import element, mixed_element
from ufl import (FacetNormal, Identity, TestFunction, TrialFunction,
                 div, dot, ds, dx, inner, lhs, nabla_grad, rhs, sym)
from dolfinx.fem.petsc import NonlinearProblem
from dolfinx.nls.petsc import NewtonSolver
from dolfinx.io import XDMFFile
from dolfinx.mesh import GhostMode


########################## Importing Mesh ################################

# Read the mesh
with XDMFFile(MPI.COMM_WORLD, "domain.xdmf", "r") as xdmf:
    domain = xdmf.read_mesh(ghost_mode=GhostMode.shared_facet, name="mesh")
    domain.topology.create_connectivity(domain.topology.dim - 1, domain.topology.dim)
    ft = xdmf.read_meshtags(domain, "Facet markers")

print("--------------------------------------")
print(f"Mesh dimension: {domain.topology.dim}")
print(f"Facet tags: {np.unique(ft.values)}")
print("--------------------------------------")


########################## Constants ################################

t = 0                                       # intiial time
final_t = 8                                 # Final time
dt = 1 / 1600                               # Time step size
num_steps = int(final_t / dt)               # number of steps
D_gc = 0.01  # Drift diffusion coeff
u_slip = Constant(domain, np.array((0.0, 1e-2), dtype=np.float64))
p_ref = 1e5
M_hydrogen = 2.02e-3  # kg/mol
T = 500
R = 8.31
rho_l = 2729.3 - 0.73 * T
mg_l = Constant(domain, 1e-5)
mu_l = Constant(domain, 4E-4 * np.exp(4170 / T))
g = Constant(domain,[0.0, -9.81])  # Example for 2D gravity vector
dt = Constant(domain, 10.0)


########################## Defining function spaces ################################

# Define the elements
V_ele = element("CG", domain.basix_cell(), 2)  # Continuous Lagrange, degree 2
Q_ele = element("CG", domain.basix_cell(), 1)  # Continuous Lagrange, degree 1

# Create function spaces from the elements
V_space = functionspace(domain, V_ele)
Q_space = functionspace(domain, Q_ele)

# Create a mixed function space
W = functionspace(domain, mixed_element([V_ele, Q_ele, Q_ele]))


########################## Defining functions and test/trial functions ################################

upphi = Function(W)
upphi_n = Function(W)

vu, vp, vphi = ufl.TestFunctions(W)

u_l, p, phi_g = ufl.split(upphi)
u_l_n, p_n, phi_g_n = ufl.split(upphi_n)

phi_l = 1 - phi_g
phi_l_n = 1 - phi_g_n

fdim = domain.topology.dim - 1


########################## Boundary Conditions ################################

# Inlet velocity: constant vector (e.g., flow in +y direction)
# Target the 0-th component (velocity space)
V_sub = W.sub(0)

# Locate the facets corresponding to the inlet (tag = 2)
inlet_facets = ft.indices[ft.values == 2]
inlet_dofs = locate_dofs_topological(V_sub, domain.topology.dim - 1, inlet_facets)

# Create the function for the boundary condition (e.g., u_inlet = 1.0 on the inlet)
u_inlet = Function(domain,1.0)

# Apply the boundary condition on the inlet DOFs
bcu_inlet = dirichletbc(u_inlet, inlet_dofs, V_sub)

print("--------------------------------------")
print("No slip BC succesful")
print("--------------------------------------")

# Wall BC (no-slip)
u_noslip = np.array((0,) * mesh.geometry.dim, dtype=default_scalar_type)
wall_dofs = locate_dofs_topological(V_space, fdim, 4)
bcu_walls = dirichletbc(u_noslip, wall_dofs, V_space)

bcu = [bcu_inflow, bcu_walls]

# Outlet pressure BC
outlet_dofs = locate_dofs_topological(Q_space, fdim, 3)
bcp_outlet = dirichletbc(PETSc.ScalarType(0), outlet_dofs, Q_space)
bcp = [bcp_outlet]


########################## Velocity definitions ################################

u_drift = -D_gc * ufl.as_vector(ufl.grad(phi_g)) / (phi_g + 1e-16)
u_g = u_l
rho_g = (p_ref + p) * M_hydrogen / (R * T)


########################## Weak form equations ################################

continuity = (
    rho_l * (phi_l - phi_l_n) + rho_g * (phi_g - phi_g_n)
) / dt * vp * ufl.dx + ufl.inner(
    ufl.div(rho_l * phi_l * u_l + rho_g * phi_g * u_g), vp
) * ufl.dx

gas_phase_transport = (
    (rho_g * (phi_g - phi_g_n)) / dt * vphi * ufl.dx
    + ufl.div(rho_g * phi_g * u_l) * vphi * ufl.dx
    + ufl.inner(D_gc * rho_g * ufl.grad(phi_g), ufl.grad(vphi)) * ufl.dx
    + mg_l * vphi * ufl.dx
)

momentum = (
    phi_l * rho_l * ufl.inner((u_l - u_l_n) / dt, vu) * ufl.dx
    + phi_l * mu_l * ufl.inner(ufl.grad(u_l), ufl.grad(vu)) * ufl.dx
    + phi_l * rho_l * ufl.inner(ufl.grad(u_l), ufl.grad(vu)) * ufl.dx  # Fixed
    + p * ufl.div(vu) * ufl.dx
    - phi_l * rho_l * ufl.inner(g, vu) * ufl.dx
)

F = momentum + continuity + gas_phase_transport


########################## Initialization ################################

problem = NonlinearProblem(F == 0, upphi, bcs=[bcu, bcp])
solver = NewtonSolver(MPI.COMM_WORLD, problem)
solver.convergence_criterion = "incremental"
solver.rtol = 1e-6
solver.report = True

# PETSc solver options
ksp = solver.krylov_solver
opts = PETSc.Options()
option_prefix = ksp.getOptionsPrefix()
opts[f"{option_prefix}ksp_type"] = "gmres"
opts[f"{option_prefix}ksp_rtol"] = 1.0e-8
opts[f"{option_prefix}pc_type"] = "hypre"
opts[f"{option_prefix}pc_hypre_type"] = "boomeramg"
opts[f"{option_prefix}pc_hypre_boomeramg_max_iter"] = 1
opts[f"{option_prefix}pc_hypre_boomeramg_cycle_type"] = "v"
ksp.setFromOptions()

# Save directory
save_dir = "results/"

# Open XDMF files
file_velocity = XDMFFile(MPI.COMM_WORLD, save_dir + "velocity.xdmf", "w")
file_pressure = XDMFFile(MPI.COMM_WORLD, save_dir + "pressure.xdmf", "w")
file_phi = XDMFFile(MPI.COMM_WORLD, save_dir + "gas_fraction.xdmf", "w")

# Write mesh once (choose one file)
file_velocity.write_mesh(domain)
file_pressure.write_mesh(domain)
file_phi.write_mesh(domain)


########################## Solver loop ################################

# Time loop
t = 0
while t < final_t:
    dolfinx.log.set_log_level(dolfinx.log.LogLevel.INFO)
    n, converged = solver.solve(upphi)
    assert converged
    print(f"Time = {t:.2f}, iterations = {n}")

    u, p, phi_g = upphi.split()
    file_velocity.write_function(u, t)
    file_pressure.write_function(p, t)
    file_phi.write_function(phi_g, t)

    upphi_n.assign(upphi)  # update previous timestep
    t += float(dt)
