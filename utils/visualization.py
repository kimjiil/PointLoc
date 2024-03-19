import open3d as o3d
import glob
import numpy as np
from scipy.spatial.transform import Rotation as R
def get_6dof(path):
    path = "E:/DeepLearning/localization/자료/(2022) PointLoc/vReLoc/full/seq-01/frame-000000.pose.txt"
    pose = np.loadtxt(path, delimiter=',')

    # 이동 벡터 추출
    translation_vector = pose[:3, 3]

    # 회전 행렬 추출
    rotation_matrix = pose[:3, :3]
    r = R.from_matrix(rotation_matrix)
    rot_deg = r.as_euler('xyz', degrees=True)
    print()

    return translation_vector, rotation_matrix


if __name__ == '__main__':
    get_6dof(3)
    path = "E:/DeepLearning/localization/자료/(2022) PointLoc/vReLoc/full/seq-02/*.bin"
    def load_point_cloud(file_path):
        return o3d.io.read_point_cloud(file_path)

        # Initialize visualizer with key callbacks

    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window()

    # Load point clouds
    file_paths = glob.glob(path)
    file_paths.sort()

    # point_clouds = [load_point_cloud(fp) for fp in file_paths]
    point_clouds = []
    for fp in file_paths:
        data = np.fromfile(fp, dtype=np.float32).reshape((4, -1)).T
        data[:, 3] = 1
        pose_path = fp.replace(".bin", ".pose.txt")
        mat = np.loadtxt(pose_path, delimiter=',')

        data = data @ mat.T
        data = data[:, :3]
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(data)
        point_clouds.append(pcd)

    # point_clouds = [np.fromfile(fp, dtype=np.float32).reshape((4, -1)).T[:, :3] for fp in file_paths]

    current_index = 0  # Start from the first point cloud

    # Add the first point cloud to visualizer
    vis.add_geometry(point_clouds[current_index])

    # Get view control and capture initial viewpoint
    view_ctl = vis.get_view_control()
    viewpoint_params = view_ctl.convert_to_pinhole_camera_parameters()


    def update_visualization(vis, point_cloud, view_ctl, viewpoint_params):
        vis.clear_geometries()  # Clear existing geometries
        vis.add_geometry(point_cloud)  # Add new geometry
        view_ctl.convert_from_pinhole_camera_parameters(viewpoint_params)


    def next_callback(vis):
        global current_index, viewpoint_params
        if current_index < len(point_clouds) - 1:
            # Capture current viewpoint before moving to next
            viewpoint_params = view_ctl.convert_to_pinhole_camera_parameters()
            current_index += 1
            update_visualization(vis, point_clouds[current_index], view_ctl, viewpoint_params)


    def previous_callback(vis):
        global current_index, viewpoint_params
        if current_index > 0:
            # Capture current viewpoint before moving to previous
            viewpoint_params = view_ctl.convert_to_pinhole_camera_parameters()
            current_index -= 1
            update_visualization(vis, point_clouds[current_index], view_ctl, viewpoint_params)


    def quit_callback(vis):
        vis.close()  # Close the visualizer


    # Register key callbacks
    vis.register_key_callback(ord('.'), next_callback)
    vis.register_key_callback(ord(','), previous_callback)
    vis.register_key_callback(ord('Q'), quit_callback)

    # Run the visualizer
    vis.run()
    vis.destroy_window()