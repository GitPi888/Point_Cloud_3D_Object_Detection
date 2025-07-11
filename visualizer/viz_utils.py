""" Visualization code for point clouds and 3D bounding boxes with Open3D.

Modified by Charles R. Qi 
Date: September 2017
Updated to use Open3D instead of mayavi

Ref: https://github.com/hengck23/didi-udacity-2017/blob/master/baseline-04/kitti_data/draw.py
"""

import numpy as np
import matplotlib.pyplot as plt

import open3d as o3d


def normalize(vec):
    """normalizes an Nd list of vectors or a single vector
    to unit length.
    The vector is **not** changed in place.
    For zero-length vectors, the result will be np.nan.
    :param numpy.array vec: an Nd array with the final dimension
        being vectors
        ::
            numpy.array([ x, y, z ])
        Or an NxM array::
            numpy.array([
                [x1, y1, z1],
                [x2, y2, z2]
            ]).
    :rtype: A numpy.array the normalized value
    """
    # calculate the length
    # this is a duplicate of length(vec) because we
    # always want an array, even a 0-d array.
    return (vec.T / np.sqrt(np.sum(vec ** 2, axis=-1))).T


def rotation_matrix_numpy0(axis, theta, dtype=None):
    # dtype = dtype or axis.dtype
    # make sure the vector is normalized
    if not np.isclose(np.linalg.norm(axis), 1.0):
        axis = normalize(axis)

    thetaOver2 = theta * 0.5
    sinThetaOver2 = np.sin(thetaOver2)

    return np.array(
        [
            sinThetaOver2 * axis[0],
            sinThetaOver2 * axis[1],
            sinThetaOver2 * axis[2],
            np.cos(thetaOver2),
        ]
    )


def rotation_matrix_numpy(axis, theta):

    axis = axis / np.sqrt(np.dot(axis, axis))
    a = np.cos(theta / 2.0)
    b, c, d = -axis * np.sin(theta / 2.0)

    return np.array(
        [
            [a * a + b * b - c * c - d * d, 2 * (b * c - a * d), 2 * (b * d + a * c)],
            [2 * (b * c + a * d), a * a + c * c - b * b - d * d, 2 * (c * d - a * b)],
            [2 * (b * d - a * c), 2 * (c * d + a * b), a * a + d * d - b * b - c * c],
        ]
    )


def rotx(t):
    """ 3D Rotation about the x-axis. """
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def roty(t):
    """ Rotation about the y-axis. """
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def rotz(t):
    """ Rotation about the z-axis. """
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def draw_lidar_simple(pc, color=None):
    """ Draw lidar points. simplest set up. """
    # Create point cloud
    pcd = o3d.geometry.PointCloud()
    # Only use first 3 columns (x, y, z) for point coordinates
    pcd.points = o3d.utility.Vector3dVector(pc[:, :3])
    
    if color is None:
        # Use z-coordinate for color
        colors = np.zeros((len(pc), 3))
        colors[:, 0] = pc[:, 2]  # Red channel based on z
        colors[:, 1] = pc[:, 2]  # Green channel based on z  
        colors[:, 2] = pc[:, 2]  # Blue channel based on z
        pcd.colors = o3d.utility.Vector3dVector(colors)
    else:
        # Use provided color
        if len(color.shape) == 1:
            colors = np.zeros((len(pc), 3))
            colors[:, 0] = color
            colors[:, 1] = color
            colors[:, 2] = color
        else:
            colors = color
        pcd.colors = o3d.utility.Vector3dVector(colors)
    
    # Create coordinate frame
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=2.0)
    
    # Create sphere at origin
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.2)
    sphere.paint_uniform_color([1, 1, 1])  # White
    
    # Visualize
    o3d.visualization.draw_geometries([pcd, coordinate_frame, sphere])
    
    return pcd


def draw_lidar(
    pc,
    color=None,
    fig=None,
    bgcolor=(0, 0, 0),
    pts_scale=0.3,
    pts_mode="sphere",
    pts_color=None,
    color_by_intensity=False,
    pc_label=False,
):
    """ Draw lidar points
    Args:
        pc: numpy array (n,3) of XYZ
        color: numpy array (n) of intensity or whatever
        fig: open3d geometry list, if None create new one otherwise will use it
    Returns:
        geometries: list of open3d geometries
    """
    print("====================", pc.shape)
    
    geometries = []
    
    # Create point cloud
    pcd = o3d.geometry.PointCloud()
    # Only use first 3 columns (x, y, z) for point coordinates
    pcd.points = o3d.utility.Vector3dVector(pc[:, :3])
    
    # Tô màu theo chiều cao z bằng colormap matplotlib
    if color is None:
        z = pc[:, 2]
        z_min, z_max = z.min(), z.max()
        norm = (z - z_min) / (z_max - z_min + 1e-8)
        cmap = plt.get_cmap('jet')
        colors = cmap(norm)[:, :3]  # Bỏ alpha
    else:
        if len(color.shape) == 1:
            colors = np.zeros((len(pc), 3))
            colors[:, 0] = color
            colors[:, 1] = color
            colors[:, 2] = color
        else:
            colors = color
    pcd.colors = o3d.utility.Vector3dVector(colors)
    
    geometries.append(pcd)

    # Create coordinate frame
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=2.0)
    geometries.append(coordinate_frame)
    
    # Create sphere at origin
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.2)
    sphere.paint_uniform_color([1, 1, 1])  # White
    geometries.append(sphere)

    # Draw FOV lines
    fov = np.array([[20.0, 20.0, 0.0], [20.0, -20.0, 0.0]], dtype=np.float64)
    
    for i in range(2):
        line = o3d.geometry.LineSet()
        points = o3d.utility.Vector3dVector([[0, 0, 0], fov[i]])
        lines = o3d.utility.Vector2iVector([[0, 1]])
        line.points = points
        line.lines = lines
        line.paint_uniform_color([1, 1, 1])  # White
        geometries.append(line)

    # Draw square region
    TOP_Y_MIN = -20
    TOP_Y_MAX = 20
    TOP_X_MIN = 0
    TOP_X_MAX = 40

    x1, x2 = TOP_X_MIN, TOP_X_MAX
    y1, y2 = TOP_Y_MIN, TOP_Y_MAX
    
    # Create square boundary lines
    square_points = [
        [x1, y1, 0], [x1, y2, 0],  # Left vertical
        [x2, y1, 0], [x2, y2, 0],  # Right vertical  
        [x1, y1, 0], [x2, y1, 0],  # Bottom horizontal
        [x1, y2, 0], [x2, y2, 0]   # Top horizontal
    ]
    
    for i in range(0, len(square_points), 2):
        line = o3d.geometry.LineSet()
        points = o3d.utility.Vector3dVector([square_points[i], square_points[i+1]])
        lines = o3d.utility.Vector2iVector([[0, 1]])
        line.points = points
        line.lines = lines
        line.paint_uniform_color([0.5, 0.5, 0.5])  # Gray
        geometries.append(line)

    return geometries


def draw_gt_boxes3d(
    gt_boxes3d,
    color=(1, 1, 1),
    color_list=None,
    label=""
):
    """ Draw 3D bounding boxes
    Args:
        gt_boxes3d: numpy array (n,8,3) for XYZs of the box corners
        fig: open3d geometry list
        color: RGB value tuple in range (0,1), box line color
        line_width: box line width (not used in open3d)
        draw_text: boolean, if true, write box indices beside boxes (not implemented in open3d)
        text_scale: three number tuple (not used in open3d)
        color_list: a list of RGB tuple, if not None, overwrite color.
        label: text label (not implemented in open3d)
    Returns:
        geometries: list of open3d geometries
    """
    geometries = []
    num = len(gt_boxes3d)
    
    for n in range(num):
        b = gt_boxes3d[n]
        if color_list is not None:
            color = color_list[n]
        
        # Create lines for the 3D box
        # Define the 12 edges of the box
        edges = [
            # Bottom face
            (0, 1), (1, 2), (2, 3), (3, 0),
            # Top face  
            (4, 5), (5, 6), (6, 7), (7, 4),
            # Vertical edges
            (0, 4), (1, 5), (2, 6), (3, 7)
        ]
        
        # Create line set for this box
        line_set = o3d.geometry.LineSet()
        points = o3d.utility.Vector3dVector(b)
        lines = o3d.utility.Vector2iVector(edges)
        line_set.points = points
        line_set.lines = lines
        line_set.paint_uniform_color(color)
        
        geometries.append(line_set)
    
    return geometries


def xyzwhl2eight(xyzwhl):
    """ Draw 3d bounding box in image
        qs: (8,3) array of vertices for the 3d box in following order:
            7 -------- 6
           /|         /|
          4 -------- 5 .
          | |        | |
          . 3 -------- 2
          |/         |/
          0 -------- 1
    """
    x, y, z, w, h, l = xyzwhl[:6]
    box8 = np.array(
        [
            [
                x + w / 2,
                x + w / 2,
                x - w / 2,
                x - w / 2,
                x + w / 2,
                x + w / 2,
                x - w / 2,
                x - w / 2,
            ],
            [
                y - h / 2,
                y + h / 2,
                y + h / 2,
                y - h / 2,
                y - h / 2,
                y + h / 2,
                y + h / 2,
                y - h / 2,
            ],
            [
                z - l / 2,
                z - l / 2,
                z - l / 2,
                z - l / 2,
                z + l / 2,
                z + l / 2,
                z + l / 2,
                z + l / 2,
            ],
        ]
    )
    return box8.T


def draw_xyzwhl(
    gt_boxes3d,
    fig,
    color=(1, 1, 1),
    line_width=1,
    draw_text=True,
    text_scale=(1, 1, 1),
    color_list=None,
    rot=False,
):
    """ Draw 3D bounding boxes
    Args:
        gt_boxes3d: numpy array (n,6) for XYZWHL format
        fig: open3d geometry list
        color: RGB value tuple in range (0,1), box line color
        line_width: box line width (not used in open3d)
        draw_text: boolean, if true, write box indices beside boxes (not implemented in open3d)
        text_scale: three number tuple (not used in open3d)
        color_list: a list of RGB tuple, if not None, overwrite color.
        rot: boolean, if true apply rotation
    Returns:
        geometries: list of open3d geometries
    """
    geometries = []
    num = len(gt_boxes3d)
    
    for n in range(num):
        print(gt_boxes3d[n])
        box6 = gt_boxes3d[n]
        b = xyzwhl2eight(box6)
        
        if rot:
            b = b.dot(rotz(box6[7]))
            vec = np.array([-1, 1, 0])
            b = b.dot(rotation_matrix_numpy(vec, box6[6]))

        print(b.shape, b)
        if color_list is not None:
            color = color_list[n]
        
        # Create lines for the 3D box
        # Define the 12 edges of the box
        edges = [
            # Bottom face
            (0, 1), (1, 2), (2, 3), (3, 0),
            # Top face  
            (4, 5), (5, 6), (6, 7), (7, 4),
            # Vertical edges
            (0, 4), (1, 5), (2, 6), (3, 7)
        ]
        
        # Create line set for this box
        line_set = o3d.geometry.LineSet()
        points = o3d.utility.Vector3dVector(b)
        lines = o3d.utility.Vector2iVector(edges)
        line_set.points = points
        line_set.lines = lines
        line_set.paint_uniform_color(color)
        
        geometries.append(line_set)
    
    return geometries


if __name__ == "__main__":
    try:
        pc = np.loadtxt(r"E:\WorkSpace\Python\Point_cloud\data\kitti_sample_scan.txt")
        geometries = draw_lidar(pc)
        o3d.visualization.draw_geometries(geometries)
    except FileNotFoundError:
        print("Sample data file not found.")
    except Exception as e:
        print(f"Error: {e}")