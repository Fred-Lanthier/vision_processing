import numpy as np
import pyvista as pv

def create_thick_shell(input_mesh_path, output_mesh_path, thickness_mm=2.0):
    """
    Converts a single surface into a watertight thick shell.
    
    The idea is simple:
    - Offset the surface outward by half the thickness → outer wall
    - Offset the surface inward by half the thickness → inner wall  
    - Connect the open edges (rim) with a strip of triangles
    
    Result: a closed solid with the specified wall thickness.
    """
    print(f"📦 Creating {thickness_mm}mm thick shell from: {input_mesh_path}")
    
    # Convert mm to meters (your point cloud is in meters)
    offset = (thickness_mm / 2.0) * 0.001  # 1mm = 0.001m each direction
    
    # =============================================
    # STEP 1: Load mesh and compute normals
    # =============================================
    mesh = pv.read(input_mesh_path)
    mesh = mesh.compute_normals(auto_orient_normals=True, consistent_normals=True)
    
    vertices = np.array(mesh.points)
    normals = np.array(mesh.point_data['Normals'])
    
    # PyVista stores faces as [n, v0, v1, v2, n, v0, v1, v2, ...]
    # We need to extract just the vertex indices
    faces_flat = np.array(mesh.faces)
    faces = faces_flat.reshape(-1, 4)[:, 1:4]  # Remove the '3' prefix, keep indices
    
    n_verts = len(vertices)
    n_faces = len(faces)
    
    print(f"   Original mesh: {n_verts} vertices, {n_faces} faces")
    
    # =============================================
    # STEP 2: Create outer and inner surfaces
    # =============================================
    # Outer surface: push vertices along normal direction
    # new_position = old_position + normal × offset
    vertices_outer = vertices + normals * offset
    
    # Inner surface: push vertices opposite to normal
    # new_position = old_position - normal × offset
    vertices_inner = vertices - normals * offset
    
    # For the inner surface, we flip the face winding order
    # Original face [A, B, C] → Flipped face [A, C, B]
    # This makes the normals point inward (into the shell cavity)
    faces_inner = faces[:, [0, 2, 1]]  # Swap columns 1 and 2
    
    # Inner surface indices are shifted by n_verts (they come after outer vertices)
    faces_inner_shifted = faces_inner + n_verts
    
    # =============================================
    # STEP 3: Find boundary edges (the open rim)
    # =============================================
    # An edge is on the boundary if it belongs to only ONE triangle.
    # Interior edges are shared by exactly TWO triangles.
    
    def find_boundary_edges(faces):
        """
        Returns boundary edges as pairs of vertex indices,
        ordered to form a continuous loop around the rim.
        """
        edge_count = {}
        
        # Count how many faces each edge belongs to
        for face in faces:
            for i in range(3):
                v1, v2 = face[i], face[(i + 1) % 3]
                edge = tuple(sorted([v1, v2]))  # Normalize edge direction
                edge_count[edge] = edge_count.get(edge, 0) + 1
        
        # Boundary edges appear exactly once
        boundary_edges = [e for e, count in edge_count.items() if count == 1]
        
        if len(boundary_edges) == 0:
            return []
        
        # Now order the edges into a continuous loop
        # Start from any edge, then find the next edge that shares a vertex
        ordered_vertices = list(boundary_edges[0])
        used = {0}
        
        while len(used) < len(boundary_edges):
            current_end = ordered_vertices[-1]
            found = False
            
            for i, edge in enumerate(boundary_edges):
                if i in used:
                    continue
                if edge[0] == current_end:
                    ordered_vertices.append(edge[1])
                    used.add(i)
                    found = True
                    break
                elif edge[1] == current_end:
                    ordered_vertices.append(edge[0])
                    used.add(i)
                    found = True
                    break
            
            if not found:
                print("   ⚠️ Warning: boundary is not a single loop (multiple holes?)")
                break
        
        # Remove duplicate end vertex if loop closes
        if len(ordered_vertices) > 1 and ordered_vertices[0] == ordered_vertices[-1]:
            ordered_vertices = ordered_vertices[:-1]
        
        return ordered_vertices
    
    boundary_loop = find_boundary_edges(faces)
    print(f"   Boundary rim: {len(boundary_loop)} vertices")
    
    if len(boundary_loop) == 0:
        print("   ❌ No boundary found - mesh is already closed!")
        return
    
    # =============================================
    # STEP 4: Create rim triangles
    # =============================================
    # We connect each edge on the outer boundary to the corresponding
    # edge on the inner boundary using two triangles (forming a quad).
    #
    #   outer[i] -------- outer[i+1]
    #       |  \            |
    #       |    \   T2     |
    #       |      \        |
    #       |  T1    \      |
    #       |          \    |
    #   inner[i] -------- inner[i+1]
    #
    # T1 = [outer[i], inner[i], inner[i+1]]
    # T2 = [outer[i], inner[i+1], outer[i+1]]
    
    rim_faces = []
    n_boundary = len(boundary_loop)
    
    for i in range(n_boundary):
        # Current and next vertex on the boundary (wrapping around)
        curr = boundary_loop[i]
        next_v = boundary_loop[(i + 1) % n_boundary]
        
        # Outer surface uses original indices
        outer_curr = curr
        outer_next = next_v
        
        # Inner surface indices are shifted by n_verts
        inner_curr = curr + n_verts
        inner_next = next_v + n_verts
        
        # Two triangles to form the quad
        rim_faces.append([outer_curr, inner_curr, inner_next])
        rim_faces.append([outer_curr, inner_next, outer_next])
    
    rim_faces = np.array(rim_faces)
    print(f"   Rim strip: {len(rim_faces)} triangles")
    
    # =============================================
    # STEP 5: Combine everything into one mesh
    # =============================================
    # Stack all vertices: [outer_vertices, inner_vertices]
    all_vertices = np.vstack([vertices_outer, vertices_inner])
    
    # Combine all faces: outer surface + inner surface + rim
    all_faces = np.vstack([faces, faces_inner_shifted, rim_faces])
    
    print(f"   Final mesh: {len(all_vertices)} vertices, {len(all_faces)} faces")
    
    # Convert to PyVista format (prepend '3' to each face)
    faces_pv = np.hstack([[3, *f] for f in all_faces])
    
    thick_mesh = pv.PolyData(all_vertices, faces_pv)
    
    # =============================================
    # STEP 6: Clean up and verify
    # =============================================
    # Remove any duplicate vertices/faces that might cause issues
    thick_mesh = thick_mesh.clean()
    
    # Recompute normals for nice visualization
    thick_mesh = thick_mesh.compute_normals(auto_orient_normals=True)
    
    # Check if watertight
    n_open_edges = thick_mesh.n_open_edges
    if n_open_edges == 0:
        print("   ✅ Mesh is watertight!")
    else:
        print(f"   ⚠️ Mesh has {n_open_edges} open edges (not fully watertight)")
    
    # Save
    thick_mesh.save(output_mesh_path)
    print(f"   💾 Saved to: {output_mesh_path}")
    
    # =============================================
    # Visualization
    # =============================================
    p = pv.Plotter(shape=(1, 2))
    
    # Left: solid view
    p.subplot(0, 0)
    p.add_mesh(thick_mesh, color="lightblue", pbr=True, metallic=0.3)
    p.add_title(f"Thick Shell ({thickness_mm}mm)")
    
    # Right: cross-section to see the wall thickness
    # We slice through the middle of the bowl
    center = thick_mesh.center
    p.subplot(0, 1)
    p.add_mesh(thick_mesh.clip(normal='y', origin=center), color="lightblue", pbr=True)
    p.add_title("Cross-Section View")
    
    p.link_views()
    p.show()
    
    return thick_mesh


# =============================================
# MAIN
# =============================================
if __name__ == "__main__":
    import os
    
    input_file = "mon_bol_parfait.obj"
    output_file = "mon_bol_thick_shell.obj"
    
    if os.path.exists(input_file):
        create_thick_shell(input_file, output_file, thickness_mm=2.0)
    else:
        print(f"❌ File not found: {input_file}")
        print("   Make sure to run Create_surface.py and Repair_holes.py first.")