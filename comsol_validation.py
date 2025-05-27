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
from dolfinx.io import XDMFFile, gmshio
from dolfinx.mesh import GhostMode, locate_entities_boundary


########################## Creating Mesh ################################

inlet_id = 5
top_id = 4
domain_id = 10
domain, _, facet_tags = gmshio.read_from_msh("out_gmsh/c_shape.msh", MPI.COMM_WORLD, gdim=2)

########################## Constants ################################

t = 0                                       # Initial time
final_t = 5                             # Final time
dt = Constant(domain, 0.5)
D_gc = 1e-5  # Drift diffusion coeff
u_slip = Constant(domain, np.array((0.0, 1e-3), dtype=np.float64))
p_ref = 1e5
M_hydrogen = 2.02e-3 # kg/mol
T = 500
R = 8.31
rho_l = 2729.3 - 0.73*T
mg_l = Constant(domain, 0.00002)
mu_l = Constant(domain, 4E-4 * np.exp(4170 / T))
g = Constant(domain,[0.0, -9.81])  # 2D gravity vector

########################## Defining function spaces ################################

# Define the elements
V_ele = element("CG", domain.topology.cell_name(), 2, shape=(domain.geometry.dim, ))
Q_ele = element("CG", domain.basix_cell(), 1)  # Continuous Lagrange, degree 1

# Create a mixed function space
W = functionspace(domain, mixed_element([V_ele, Q_ele, Q_ele, Q_ele]))

# Definign sub spaces
V_sub, _ = W.sub(0).collapse()
Q1_sub, _ = W.sub(1).collapse()
Q2_sub, _ = W.sub(2).collapse()
Q3_sub, _ = W.sub(3).collapse()

u_ = Function(V_sub)
u_.name = "u"
p_ = Function(Q1_sub)
p_.name = "p"
phi_g_ = Function(Q2_sub)
phi_g_.name = "phi_g"
c_ = Function(Q3_sub)
c_.name = "concentration"

########################## Defining functions and test/trial functions ################################

upphic = Function(W)
upphic_n = Function(W)

vu, vp, vphi, vc = ufl.TestFunctions(W)

u_l, p, phi_g, c = ufl.split(upphic)
u_l_n, p_n, phi_g_n, c_n = ufl.split(upphic_n)

phi_l = 1 - phi_g
phi_l_n = 1 - phi_g_n

########################## Boundary Conditions ################################

n = ufl.FacetNormal(domain)

bcs = []
# Inlet velocity conditions
inlet_u = Function(V_sub)
inlet_u.interpolate(
    lambda x: np.array([np.zeros(x.shape[1]), 0.5 * np.ones(x.shape[1])])
)
gas_inlet = Function(Q2_sub)
gas_inlet.x.array[:] = 0.01

for tag in [5]:  # inlet boundaries
    inlet_facets = facet_tags.find(tag)
    inlet_dofs = locate_dofs_topological((W.sub(0), V_sub), 1, inlet_facets)
    inlet_dofs_phi = locate_dofs_topological((W.sub(2), Q2_sub), 1, inlet_facets)
    bc_phi = dirichletbc(gas_inlet, inlet_dofs_phi, W.sub(2))
    inlet_bc = dirichletbc(inlet_u, inlet_dofs, W.sub(0))
    bcs.append(inlet_bc)
    bcs.append(bc_phi)

# # Gas fraction BC at walls (zero gas at solid boundaries)
# gas_wall = Function(Q2_sub)
# gas_wall.x.array[:] = 0.0
# for tag in [1, 2, 3]:
#     wall_facets = facet_tags.find(tag)
#     wall_dofs_phi = locate_dofs_topological((W.sub(2), Q2_sub), 1, wall_facets)
#     wall_bc_phi = dirichletbc(gas_wall, wall_dofs_phi, W.sub(2))
#     # bcs.append(wall_bc_phi)

# No-slip boundary condition for velocity
u_noslip = Function(V_sub)
u_noslip.x.array[:] = 0.0
for tag in [1, 2, 3]:  # wall boundaries
    wall_facets = facet_tags.find(tag)
    wall_dofs = locate_dofs_topological((W.sub(0), V_sub), 1, wall_facets)
    wall_bc = dirichletbc(u_noslip, wall_dofs, W.sub(0))
    bcs.append(wall_bc)

# Slip boundary condition for velocity
Vy, _ = W.sub(0).sub(1).collapse()
u_slip = Function(Vy)
u_slip.x.array[:] = 0.0
for tag in [4]:  # wall boundaries
    slip_facets = facet_tags.find(tag)
    slip_dofs = locate_dofs_topological((W.sub(0).sub(1), Vy), 1, slip_facets)
    slip_bc = dirichletbc(u_slip, slip_dofs, W.sub(0).sub(1))
    bcs.append(slip_bc)

# P_slip = Function(Q1_sub)
# P_slip.x.array[:] = 1e5
# for tag in [3,4,5]:  # wall boundaries
#     slip_facets = facet_tags.find(tag)
#     slip_dofs = locate_dofs_topological((W.sub(1), Q1_sub), 1, slip_facets)
#     P_slip_bc = dirichletbc(P_slip, slip_dofs, W.sub(1))
#     bcs.append(P_slip_bc)

P_outlet = Function(Q1_sub)
P_outlet.x.array[:] = 0
for tag in [4]:  # wall boundaries
    slip_facets = facet_tags.find(tag)
    slip_dofs = locate_dofs_topological((W.sub(1), Q1_sub), 1, slip_facets)
    P_out_bc = dirichletbc(P_outlet, slip_dofs, W.sub(1))
    bcs.append(P_out_bc)


print("--------------------------------------")

########################## derivation definitions ################################
from ufl import sym, Identity

mu_eff = mu_l + 0  # Effective viscosity (laminar + turbulent)
I = Identity(u_l.ufl_shape[0])

# Deviatoric stress tensor: tau = mu_eff * (grad(u) + grad(u)^T - 2/3 div(u) * I)
tau = mu_eff * (sym(ufl.grad(u_l)) - (1/3) * div(u_l) * I)

u_drift = -D_gc * ufl.grad(phi_g) / (phi_g + 1e-16)
u_g = u_l + u_drift
rho_g = (p_ref + p) * M_hydrogen / (R * T)
# rho_g = 0.01
D = 1e-7
H = 1
k = 1

# a = (4*3.14)**(1/3) * (3*phi_g)**(2/3)
c_star = (p+p_ref)/H
mg_l = 1e-7*(c_star-c)*M_hydrogen

########################## Weak form equations ################################

F = Constant(domain,(0.0, 0.0))
dx = ufl.Measure("dx", domain=domain)

continuity = (
    rho_l * (phi_l - phi_l_n) + rho_g * (phi_g - phi_g_n)
)/ dt * vp * dx + ufl.inner(
    ufl.div(rho_l * phi_l * u_l + rho_g * phi_g * u_g), vp
) * dx


gas_phase_transport = (
    (rho_g * (phi_g - phi_g_n)) / dt * vphi * dx
    + ufl.div(rho_g * phi_g * u_g) * vphi * dx  # Use u_g instead of u_l + u_slip
    + ufl.inner(D_gc * rho_g * ufl.grad(phi_g), ufl.grad(vphi)) * dx
    + mg_l * vphi * dx
)

momentum = (
    phi_l * rho_l * ufl.inner((u_l - u_l_n) / dt, vu) * dx
    + phi_l * mu_l * ufl.inner(ufl.grad(u_l), ufl.grad(vu)) * dx
    + phi_l * rho_l * ufl.dot(ufl.dot(ufl.grad(u_l), u_l), vu) * dx
    + p * ufl.div(vu) * dx
    - phi_l * rho_l * ufl.inner(g, vu) * dx
    - ufl.inner(F, vu) * dx
)

species = (
    (c - c_n)/dt *vc*dx
    + D * ufl.inner(ufl.grad(c), ufl.grad(vc)) * dx
    + ufl.inner(ufl.dot(ufl.grad(c), u_l), vc) * dx
    - mg_l/M_hydrogen*vc*dx
)


F = momentum + continuity + gas_phase_transport + species
J = ufl.derivative(F,upphic)
########################## Initialization ################################

problem = NonlinearProblem(F, upphic, bcs, J)
solver = NewtonSolver(MPI.COMM_WORLD, problem)
solver.convergence_criterion = "residual"
solver.rtol = 1e-8
solver.report = True

# PETSc solver options
ksp = solver.krylov_solver
opts = PETSc.Options()
option_prefix = ksp.getOptionsPrefix()
opts[f"{option_prefix}ksp_type"] = "preonly"
opts[f"{option_prefix}pc_type"] = "lu"
ksp.getPC().setFactorSolverType("mumps")
ksp.setFromOptions()
dolfinx.log.set_log_level(dolfinx.log.LogLevel.INFO)
ksp.setErrorIfNotConverged(True)

import os
from dolfinx.io import VTXWriter

# Save directory
save_dir = "results5/"
os.makedirs(save_dir, exist_ok=True)

# Initialize VTX writers 
vtx_u = VTXWriter(MPI.COMM_WORLD, f"{save_dir}u.bp", [u_], "BP5")
vtx_p = VTXWriter(MPI.COMM_WORLD, f"{save_dir}p.bp", [p_],"BP5")
vtx_phi = VTXWriter(MPI.COMM_WORLD, f"{save_dir}phi_g.bp", [phi_g_],"BP5")
vtx_c = VTXWriter(MPI.COMM_WORLD, f"{save_dir}c.bp", [c_],"BP5")

vtx_u.write(t)
vtx_p.write(t)
vtx_phi.write(t)
vtx_c.write(t)

# Time loop
while t < float(final_t):
    n, converged = solver.solve(upphic)
    print(f"Converged: {converged} in {n} Newton iterations")
    
    u, p, phi_g, c = upphic.sub(0).collapse(), upphic.sub(1).collapse(), upphic.sub(2).collapse(), upphic.sub(3).collapse()

    u_.x.array[:] = u.x.array[:]
    p_.x.array[:] = p.x.array[:]
    phi_g_.x.array[:] = phi_g.x.array[:]
    phi_g.x.array[:] = np.clip(phi_g.x.array, 0.0, 1.0)

    c_.x.array[:] = c.x.array[:]

    vtx_u.write(t)
    vtx_p.write(t)
    vtx_phi.write(t)
    vtx_c.write(t)

    print(f"Time = {t:.2f}, iterations = {n}")

    upphic_n.x.array[:] = upphic.x.array[:]

    t += float(dt)

vtx_u.close()
vtx_p.close()
vtx_phi.close()
vtx_c.close()
