import open3d as o3d
import os


# ply_file = "superpoint_debug/vccs_superpoints_original_batch0.ply"
# ply_file = "superpoint_debug/improved_vccs_batch0.ply"
ply_file = "superpoint_debug/vccs_comparison_batch0.ply"


if os.path.exists(ply_file):
    print(f"Loading {ply_file}...")
    pcd = o3d.io.read_point_cloud(ply_file)
    if not pcd.has_points():
        print("No points in the point cloud.")
    else:
        print(f"Loaded {len(pcd.points)} points.")
        if pcd.has_colors():
            print("point cloud has colors.")
            o3d.visualization.draw_geometries([pcd], window_name="Point Cloud with Colors")
        else:
            print("point cloud does not have colors.")
            o3d.visualization.draw_geometries([pcd], window_name="Point Cloud without Colors")
else:
    print(f"File {ply_file} does not exist.")
    exit(1)