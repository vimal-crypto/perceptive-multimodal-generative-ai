import open3d as o3d
import numpy as np

def visualize_pointcloud(ply_path: str):
    pcd = o3d.io.read_point_cloud(ply_path)
    o3d.visualization.draw_geometries([pcd])

def visualize_mesh(obj_path: str):
    mesh = o3d.io.read_triangle_mesh(obj_path)
    mesh.compute_vertex_normals()
    o3d.visualization.draw_geometries([mesh])

def save_render(obj_path: str, output_image: str = 'render.png'):
    mesh = o3d.io.read_triangle_mesh(obj_path)
    mesh.compute_vertex_normals()
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False)
    vis.add_geometry(mesh)
    vis.update_geometry(mesh)
    vis.poll_events()
    vis.update_renderer()
    vis.capture_screen_image(output_image)
    vis.destroy_window()
    print(f'Render saved to {output_image}')

if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        visualize_mesh(sys.argv[1])
