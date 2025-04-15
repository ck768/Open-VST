from mpi4py import MPI
import numpy as np
import gmsh
import dolfinx

## making the mesh

gmsh.initialize()

# Mesh definition
width, height = 0.5, 1
nx, ny = 100, 100
# domain = mesh.create_rectangle(MPI.COMM_WORLD, [[0, 0], [width, height]], [nx, ny], mesh.CellType.triangle)

gdim = 2
mesh_comm = MPI.COMM_WORLD
model_rank = 0
if mesh_comm.rank == model_rank:
    domain = gmsh.model.occ.addRectangle(0, 0, 0, width, height, tag=1)
    gmsh.model.occ.synchronize()

fluid_marker = 1
if mesh_comm.rank == model_rank:
    volumes = gmsh.model.getEntities(dim=gdim)
    assert (len(volumes) == 1)
    gmsh.model.addPhysicalGroup(volumes[0][0], [volumes[0][1]], fluid_marker)
    gmsh.model.setPhysicalName(volumes[0][0], fluid_marker, "Fluid")

inlet_marker, outlet_marker, wall_marker = 2, 3, 4
# Domain size
inlet_width = 0.025  # Half-width of inlet (e.g., 5% of width)

inlet, outlet, walls = [], [], []

if mesh_comm.rank == model_rank:
    # Get all 1D boundary entities from 2D geometry
    boundaries = gmsh.model.getBoundary(volumes, oriented=False)
    for boundary in boundaries:
        dim, tag = boundary
        com = gmsh.model.occ.getCenterOfMass(dim, tag)

        # Check for INLET (centered at bottom middle)
        if np.isclose(com[1], 0.0):
            if (width / 2 - inlet_width) < com[0] < (width / 2 + inlet_width):
                inlet.append(tag)
            else:
                walls.append(tag)

        # Check for OUTLET (top boundary)
        elif np.isclose(com[1], height):
            outlet.append(tag)

        # All other sides are WALLS
        else:
            walls.append(tag)

    # Tag physical groups
    gmsh.model.addPhysicalGroup(1, inlet, inlet_marker)
    gmsh.model.setPhysicalName(1, inlet_marker, "Inlet")

    gmsh.model.addPhysicalGroup(1, outlet, outlet_marker)
    gmsh.model.setPhysicalName(1, outlet_marker, "Outlet")

    gmsh.model.addPhysicalGroup(1, walls, wall_marker)
    gmsh.model.setPhysicalName(1, wall_marker, "Walls")

gmsh.option.setNumber("Mesh.CharacteristicLengthMin", 0.05)
gmsh.option.setNumber("Mesh.CharacteristicLengthMax", 0.05)
gmsh.model.mesh.generate(gdim)


domain, cell_markers, ft = dolfinx.io.gmshio.model_to_mesh(gmsh.model, mesh_comm, model_rank, gdim=gdim)
ft.name = "Facet markers"

from dolfinx.io import XDMFFile, gmshio

with XDMFFile(mesh_comm, "domain.xdmf", "w") as xdmf:
    xdmf.write_mesh(domain)
    xdmf.write_meshtags(ft, domain.geometry)


