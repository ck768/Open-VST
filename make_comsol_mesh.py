
import numpy as np
import matplotlib.pyplot as plt
import gmsh
import dolfinx
from mpi4py import MPI
from dolfinx.fem import (Function, FunctionSpace, assemble_scalar, form,
                        dirichletbc, locate_dofs_topological)
from dolfinx.fem.petsc import LinearProblem
from dolfinx.io import XDMFFile, gmshio
from dolfinx.mesh import locate_entities_boundary
from dolfinx.io import XDMFFile, gmshio

try:
    import gmsh  # type: ignore
except ImportError:
    print("This demo requires gmsh to be installed")
    exit(0)
def create_mesh(comm: MPI.Comm, model: gmsh.model, name: str, filename: str, mode: str):
    """Create a DOLFINx mesh from a Gmsh model and output to file."""
    try:
        # Get mesh from model
        mesh, cell_tags, facet_tags = gmshio.model_to_mesh(model, comm, rank=0)
        
        # Set names
        mesh.name = name
        cell_tags.name = f"{name}_cells"
        facet_tags.name = f"{name}_facets"
        
        # Write to file
        with XDMFFile(mesh.comm, filename, mode) as file:
            file.write_mesh(mesh)
            file.write_meshtags(
                cell_tags, 
                mesh.geometry, 
                geometry_xpath=f"/Xdmf/Domain/Grid[@Name='{mesh.name}']/Geometry"
            )
            file.write_meshtags(
                facet_tags, 
                mesh.geometry, 
                geometry_xpath=f"/Xdmf/Domain/Grid[@Name='{mesh.name}']/Geometry"
            )
        
        print(f"Mesh created and saved to {filename}")
        return mesh, cell_tags, facet_tags
        
    except Exception as e:
        print(f"Error in create_mesh: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
import gmsh
import numpy as np
from mpi4py import MPI
import dolfinx
from dolfinx.io import XDMFFile, gmshio
from dolfinx.fem import (Function, dirichletbc, form, locate_dofs_topological)
from dolfinx.fem.petsc import LinearProblem
import ufl
import os
from dataclasses import dataclass
from contextlib import contextmanager

@dataclass
class CShapeConfig:
    """Configuration for C-shaped domain generation"""
    width_lower: float = 1.0
    width_upper: float = 1.2
    height: float = 0.8
    inlet_width: float = 0.2
    mesh_size: float = 0.01
    boundaries: dict = None
    
    def __post_init__(self):
        # Define default boundary tags and names
        self.boundaries = {
            1: "bottom",
            2: "right", 
            3: "left",
            4: "top",
            5: "inlet",
            10: "domain"
        }

class CShapeGenerator:
    """Automated generator for C-shaped domains"""
    def __init__(self, config=None):
        self.config = config if config else CShapeConfig()
        self.model = gmsh.model()
        self._initialize()
        
    def _initialize(self):
        """Initialize the GMSH model"""
        self.model.add("CShape")
        self.model.setCurrent("CShape")
        self.points = {}
        self.lines = {}
        
    def generate_points(self):
        """Generate all geometric points"""
        c = self.config
        
        # Create 8 points (bottom left to top left, counter-clockwise)
        self.points[1] = self.model.geo.addPoint(0, 0, 0, c.mesh_size)
        self.points[2] = self.model.geo.addPoint(c.inlet_width/2, 0, 0, c.mesh_size)
        self.points[3] = self.model.geo.addPoint(c.width_lower/2, 0, 0, c.mesh_size)

        self.points[4] = self.model.geo.addPoint(c.width_upper/2, c.height, 0, c.mesh_size)
        self.points[5] = self.model.geo.addPoint(-c.width_upper/2, c.height, 0, c.mesh_size)
        self.points[6] = self.model.geo.addPoint(-c.width_lower/2, 0, 0, c.mesh_size)

        self.points[7] = self.model.geo.addPoint(-c.inlet_width/2, 0, 0, c.mesh_size)

    def generate_lines(self):
        """Connect points with lines"""
        p = self.points
        
        self.lines[1] = self.model.geo.addLine(p[1], p[2])  # Bottom
        self.lines[2] = self.model.geo.addLine(p[2], p[3])  # Right bottom
        self.lines[3] = self.model.geo.addLine(p[3], p[4])  # Cutout bottom
        self.lines[4] = self.model.geo.addLine(p[4], p[5])  # Cutout left
        self.lines[5] = self.model.geo.addLine(p[5], p[6])  # Cutout top
        self.lines[6] = self.model.geo.addLine(p[6], p[7])  # Right top
        self.lines[7] = self.model.geo.addLine(p[7], p[1])  # Top

    def define_boundaries(self):
        """Create physical groups for boundaries and domain"""
        #Synchronize the geometry
        self.model.geo.synchronize()
        
        # Create curve loop and surface
        curve_loop = self.model.geo.addCurveLoop(list(self.lines.values()))
        surface = self.model.geo.addPlaneSurface([curve_loop])
        self.model.geo.synchronize()
        
        # Clear any existing physical groups
        for dim in [1, 2]:
            for tag in self.model.getPhysicalGroups(dim):
                self.model.removePhysicalGroups([(dim, tag[1])])
        
        # Create new physical groups
        self.model.addPhysicalGroup(1, [self.lines[2], self.lines[6]], tag=1)  # Bottom
        self.model.addPhysicalGroup(1, [self.lines[3]], tag=2)  # Right
        self.model.addPhysicalGroup(1, [self.lines[5]], tag=3)
        self.model.addPhysicalGroup(1, [self.lines[4]], tag=4)  # Top
        self.model.addPhysicalGroup(1, [self.lines[1], self.lines[7]], tag=5)  # inlet

            # 1: "bottom",
            # 2: "right", 
            # 3: "left",
            # 4: "top",
            # 5: "inlet",
            # 10: "domain"

        # Get surface tag after synchronization
        surfaces = self.model.getEntities(2)
        surface_tag = surfaces[0][1] if surfaces else 1
        self.model.addPhysicalGroup(2, [surface_tag], tag=10)
        
        # Set names
        for tag, name in self.config.boundaries.items():
            dim = 2 if tag == 10 else 1
            self.model.setPhysicalName(dim, tag, name)
            
    def generate_mesh(self):
        """Finalize and generate the mesh"""
        self.generate_points()
        self.generate_lines()
        self.define_boundaries()
        self.model.mesh.generate(2)
        return self.model

@contextmanager
def gmsh_session():
    """Context manager for GMSH session"""
    try:
        gmsh.initialize()
        gmsh.option.setNumber("General.Terminal", 1)
        yield
    finally:
        gmsh.finalize()

def create_mesh(comm: MPI.Comm, model: gmsh.model, name: str, filename: str, mode: str):
    """Convert GMSH model to DOLFINx mesh and save to file"""
    try:
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        mesh, cell_tags, facet_tags = gmshio.model_to_mesh(model, comm, rank=0)
        
        mesh.name = name
        cell_tags.name = f"{name}_cells"
        facet_tags.name = f"{name}_facets"
        
        with XDMFFile(mesh.comm, filename, mode) as file:
            mesh.topology.create_connectivity(2, 2)
            mesh.topology.create_connectivity(1, 2)
            file.write_mesh(mesh)
            file.write_meshtags(cell_tags, mesh.geometry)
            file.write_meshtags(facet_tags, mesh.geometry)
            
        print(f"Mesh saved to {filename}")
        return mesh, cell_tags, facet_tags
        
    except Exception as e:
        print(f"Mesh creation failed: {e}")
        raise

def solve_heat_transfer(mesh, facet_tags, k=1.0, T_hot=1.0, T_cold=0.0, filename="heat_solution.xdmf"):
    """Solve steady-state heat transfer with boundary conditions"""
    V = dolfinx.fem.functionspace(mesh, ("Lagrange", 1))
    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    
    a = k * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = dolfinx.fem.form(dolfinx.fem.Constant(mesh, 0.0) * v * ufl.dx)
    
    # Boundary conditions
    bcs = []
    hot_bc = dirichletbc(dolfinx.fem.Constant(mesh, T_hot), 
                        locate_dofs_topological(V, 1, facet_tags.find(5)), V)
    bcs.append(hot_bc)
    
    for tag in [1, 2, 3, 4,]:  # Cold boundaries
        cold_bc = dirichletbc(dolfinx.fem.Constant(mesh, T_cold),
                            locate_dofs_topological(V, 1, facet_tags.find(tag)), V)
        bcs.append(cold_bc)
    
    # Solve and save
    uh = LinearProblem(a, L, bcs=bcs).solve()
    with XDMFFile(mesh.comm, filename, "w") as file:
        file.write_mesh(mesh)
        file.write_function(uh)
    
    return uh

def main():
    with gmsh_session():
        # Configuration
        config = CShapeConfig(
            width_lower=1.0,
            width_upper = 1.2,
            height=0.8,
            inlet_width=0.05,
            mesh_size=0.01
        
        )
        
        # Generate mesh
        generator = CShapeGenerator(config)
        model = generator.generate_mesh()
        
        solve = True
        if solve:
            # Create and solve
            mesh, _, facet_tags = create_mesh(MPI.COMM_SELF, model, "c_shape", "out_gmsh/c_shape.xdmf", "w")
            gmsh.write("out_gmsh/c_shape.msh")
            
            if mesh and facet_tags:
                solve_heat_transfer(
                    mesh, facet_tags,
                    k=1.0, T_hot=1.0, T_cold=0.0,
                    filename="out_gmsh/heat_solution.xdmf"
                )
            
            print("""
            Simulation complete. Visualize with:
            - Mesh: out_gmsh/c_shape.xdmf
            - Solution: out_gmsh/heat_solution.xdmf
            """)

if __name__ == "__main__":
    main()
