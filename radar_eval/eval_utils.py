import warnings
import torch

import numpy as np
from pathlib import Path

import conversions as tgm

device = torch.device(
    "cuda") if torch.cuda.is_available() else torch.device("cpu")


class RadarEvalOdom():
    """Evaluate odometry results for ground-truth and predicted odometry. 
    Includes utility functions to convert odometry predictions to trajectories.
    If this class is instantiated with a gt file, it aligns the predictions with gt. They can be of different sizes.
    Usage example:
        vo_eval = RadarEvalOdom()
        vo_eval.eval(gt_pose_txt_dir, result_pose_txt_dir)
    """

    def __init__(self, f_gt, dataset, timestamp_list=None):
        """Initialize with ground truth file.

        Args:
            f_gt (Path): Path to ground truth trajectory file.
            eul (boolean): If the gt format is in Euler xyz.
        """
        self.dataset = dataset
        if self.dataset == 'hand':
            self.gt = load_xyz_from_txt(f_gt)  # [N,3]
        elif self.dataset == 'robotcar':
            self.gt = load_odometry_from_txt(f_gt)  # [N,4,4]
        elif self.dataset == 'radiate':
            self.gt = load_odometry_from_folder(f_gt, timestamp_list)  # [N,4,4]
        else:
            raise RuntimeError('Unknown dataset name is provded!')

        # self.gt = align_to_origin(self.gt)

    # def eval_ref_poses(self, all_poses, all_inv_poses, k):
    #     """Evaluate ATE pose error from the predicted poses. Each prediction in inputs is the relative pose for a sequence of [src_p, tgt, src_n].
    #     We form a trajectory from a chained sequence with k skip frames, e.g. [k, 2*k, 3*k, ..., N]. We shift the sequence by {i:i<k} to evaluate
    #     the full prediction.

    #     Args:
    #         all_poses (list): Predicted relative pose values for each src-to-tgt pair. List of torch.Tensor objects size [seq_length, B, 6]. rtvec=[rx, ry, rz, tx, ty, tz]
    #         all_inv_poses (list): Predicted relative pose values for each tgt-to-src pair. List of torch.Tensor objects.
    #         k (int): Skip frames. rtvec=[rx, ry, rz, tx, ty, tz]

    #     Returns:
    #         torch.Tensor: Mean and std of the calculated ATE for forward and backward pose predictions.
    #     """

    #     # pred = torch.zeros(self.gt.size(), dtype=self.gt.dtype, device = self.gt.device)
    #     all_poses_t = torch.cat(all_poses, 1)  # [seq_length, N, 6]
    #     all_inv_poses_t = torch.cat(all_inv_poses, 1)  # [seq_length, N, 6]

    #     N = all_poses_t.shape[1]  # len(all_poses) # number of sequences

    #     ate_bs = []
    #     ate_fs = []
    #     f_pred = None
    #     f_pred_xyz = None
    #     # i=0
    #     for i in range(k):
    #         idx = torch.arange(i, N, k)

    #         # Previous src
    #         # (src2tgt + inv(tgt2src))/2 [n,6]
    #         b_pose = (all_poses_t[0, idx] - all_inv_poses_t[0, idx])/2
    #         # b_pose = all_poses_t[0, idx] #src2tgt [n,6]
    #         b_pose = -b_pose  # tgt2src [n,6]
    #         b_pose = tgm.rtvec_to_pose(b_pose)  # tgt2src [n,4,4]
    #         # b_pose = tgm.inv_rigid_tform(b_pose) # tgt2src [n,4,4]
    #         # b_inv_pose = tgm.rtvec_to_pose(all_inv_poses_t[0, idx]) # inv(src2tgt) [n,4,4]
    #         b_pose = rel2abs_traj(b_pose)  # [n,4,4]
    #         # b_pose = b_pose.cumsum(dim=0) # [n,6]
    #         # b_pose = b_pose.cumsum(dim=0) # [n,6]
    #         gt_idx = idx+i  # torch.arange(k+i, N+k, k)
    #         gt_seq_i = self.gt[gt_idx]
    #         ate_b, b_pred_xyz, b_pred = self.calculate_ate_from_hom(
    #             b_pose, gt_seq_i)
    #         ate_bs.append(ate_b)

    #         # Next src
    #         # (src2tgt + inv(tgt2src))/2 [n,6]
    #         f_pose = (all_poses_t[1, idx] - all_inv_poses_t[1, idx])/2
    #         # f_pose = all_poses_t[1, idx] #src2tgt [n,6]
    #         # f_pose = -f_pose # tgt2src [n,6]
    #         # f_pose = f_pose.cumsum(dim=0) # [n,6]
    #         f_pose = tgm.rtvec_to_pose(f_pose)  # src2tgt [n,4,4]
    #         f_pose = rel2abs_traj(f_pose)  # [n,4,4]
    #         gt_idx = idx+2*i  # torch.arange(2*k, N+2*k, k)
    #         gt_seq_i = self.gt[gt_idx, :]
    #         ate_f, f_pred_xyz, f_pred = self.calculate_ate_from_hom(
    #             f_pose, gt_seq_i)
    #         ate_fs.append(ate_f)

    #     ate_bs = torch.cat(ate_bs)
    #     ate_fs = torch.cat(ate_fs)

    #     return ate_bs.mean(), ate_bs.std(), ate_fs.mean(), ate_fs.std(), f_pred_xyz, f_pred

    def eval_ref_poses(self, all_poses, all_inv_poses, k, estimate_scale: bool = False):
        """Evaluate ATE pose error from the predicted poses. Each prediction in inputs is the relative pose for a sequence of [src_p, tgt, src_n].
        We form a trajectory from a chained sequence with k skip frames, e.g. [k, 2*k, 3*k, ..., N]. We shift the sequence by {i:i<k} to evaluate 
        the full prediction.

        Args:
            all_poses (list): Predicted relative pose values for each src-to-tgt pair. List of torch.Tensor objects size [seq_length, B, 6]. rtvec=[rx, ry, rz, tx, ty, tz]
            all_inv_poses (list): Predicted relative pose values for each tgt-to-src pair. List of torch.Tensor objects.
            k (int): Skip frames. rtvec=[rx, ry, rz, tx, ty, tz]

        Returns:
            torch.Tensor: Mean and std of the calculated ATE for forward and backward pose predictions.
        """

        # pred = torch.zeros(self.gt.size(), dtype=self.gt.dtype, device = self.gt.device)
        all_poses_t = torch.cat(all_poses, 1)  # [seq_length, N, 6]
        all_inv_poses_t = torch.cat(all_inv_poses, 1)  # [seq_length, N, 6]

        N = all_poses_t.shape[1]  # len(all_poses) # number of sequences

        # f_pred = None
        # f_pred_xyz = None
        # i = k-1
        i = 0
        # for i in range(k):

        idx = torch.arange(i, N, k)

        # Next src
        f_pose = (all_poses_t[1, idx] - all_inv_poses_t[1, idx])/2
        f_pose = tgm.rtvec_to_pose(f_pose)  # src2tgt [n,4,4]
        f_pose = rel2abs_traj(f_pose)  # [n,4,4]
        ate_f, f_pred_xyz, f_pred = self.calculate_ate_from_hom(
            f_pose, self.gt, estimate_scale)

        return ate_f, f_pred_xyz, f_pred

    def calculate_ate_from_hom(self, pred, gt=None, estimate_scale: bool = False):
        """Calculate Absolute Trajectory Error between predicted and ground truth trajectories, using Umeyama alignment.
        Both prediction and gt is absolute trajectories.

        Args:
            pred (torch.Tensor): Predicted trajectory in the form of homogenous transformation matrix. Shape: [N,4,4]
            gt (torch.Tensor, optional): Absolute ground truth tracjectory in the form of homogenous transformation matrix Shape: [N,4,4]. Defaults to None.

        Returns:
            torch.Tensor: Root mean squared error (RMSE) of ATE. Scalar tensor.
            torch.Tensor: Aligned pedicted trajectory.
        """
        if gt == None:
            gt = self.gt

        # rmse_ate, aligned_pred_xyz, aligned_pred = calculate_ate(pred, gt)
        rmse_ate, aligned_pred_xyz, aligned_pred = calculate_ate(
            pred, gt, estimate_scale)

        return rmse_ate, aligned_pred_xyz, aligned_pred
        # return aligned_pred_xyz, aligned_pred


def calculate_ate(pred, gt, estimate_scale: bool = False):
    """Calculate Absolute Trajectory Error between predicted and ground truth trajectories, using Umeyama alignment.
    Both prediction and gt is absolute trajectories.

    Args:
        pred (torch.Tensor): Predicted trajectory in the form of homogenous transformation matrix. Shape: [N,3]
        gt (torch.Tensor, optional): Absolute ground truth tracjectory in the form of homogenous transformation matrix Shape: [N,3.

    Returns:
        torch.Tensor: Root mean squared error (RMSE) of ATE. Scalar tensor.
        torch.Tensor: Aligned pedicted trajectory.
    """

    gt_xyz = gt[:, :3, 3]
    pred_xyz = pred[:, :3, 3]

    # if pred_xyz.shape != gt_xyz.shape:
    #     idx = torch.randperm(gt_xyz.shape[0], device=gt_xyz.device)
    #     idx = idx[:pred_xyz.shape[0]]
    #     idx, _ = torch.sort(idx, dim=0)
    #     gt_xyz = gt_xyz[idx, :]

    # None for batching, batch=1
    gt_xyz = gt_xyz[None]
    pred_xyz = pred_xyz[None]
    R, T, s = corresponding_points_alignment(
        pred_xyz, gt_xyz, estimate_scale=estimate_scale)

    # apply the estimated similarity transform to Xt_init
    aligned_pred_xyz = _apply_similarity_transform(pred_xyz, R, T, s)

    aligned_pred = pred.clone()
    aligned_pred[:, :3, 3] = aligned_pred_xyz

    # compute the root mean squared error
    # TODO: Not yet compatible for sets with different sizes
    rmse_ate = ((aligned_pred_xyz - gt_xyz) ** 2).mean(1).sqrt()

    return rmse_ate, aligned_pred_xyz, aligned_pred
    # return aligned_pred_xyz, aligned_pred


def load_KITTI_poses_from_txt(file_name):
    """Load poses from txt (KITTI format)
    Each line in the file should follow the following structure
        (1) pose(3x4 matrix in terms of 12 numbers)        

    Args:
        file_name (str): txt file path

    Returns:
        torch.Tensor: Trajectory in the form of homogenous transformation matrix. Shape [N,4,4]
    """
    global device

    poses = np.genfromtxt(file_name, delimiter=',')
    poses16 = np.zeros((poses.shape[0], 16))
    poses16[:, :12] = poses
    poses16[:, -1] = 1.0
    poses_mat = poses16.reshape((-1, 4, 4))
    poses_mat = torch.Tensor(poses_mat).to(device)
    return poses_mat


def load_xyz_from_txt(file_name):
    """Load xyz poses from txt. Each line is: x,y,x       

    Args:
        file_name (str): txt file path

    Returns:
        torch.Tensor: Trajectory in the form of homogenous transformation matrix. Shape [N,4,4]
    """
    global device

    poses = np.genfromtxt(file_name, delimiter=',')
    poses = torch.Tensor(poses).to(device)
    poses = tgm.rtvec_to_pose(poses)  # [n,4,4]
    return poses


def load_odometry_from_txt(file_name):
    """Load xyz poses from odometry data. Columns [2,3,4,5,6,7] are [x,y,z,rx,ry,rz].      

    Args:
        file_name (str): txt file path

    Returns:
        torch.Tensor: Trajectory in the form of homogenous transformation matrix. Shape [N,4,4]
    """
    global device

    gt = np.genfromtxt(file_name, delimiter=',',
                       usecols=np.arange(2, 8), skip_header=True)
    gt[:, np.arange(6)] = gt[:, [3, 4, 5, 0, 1, 2]]  # [rx,ry,rz,x,y,x] format
    gt_t = torch.Tensor(gt).to(device)  # [n,6]
    gt_t = tgm.rtvec_to_pose(gt_t)  # [n,4,4]
    gt_t = rel2abs_traj(gt_t)  # [n,4,4]

    return gt_t


# def load_odometry_from_folder(path, timestamp_list):
#     # let path be "../data/radiate/tiny_foggy/ 
#     # there is "../data/radiate/tiny_foggy/GPS_IMU_Twist/" where contains xxxx.load_xyz_from_txt
#     # there is ""../data/radiate/tiny_foggy/GPS_IMU_Twist.txt"" where contains 
#     #Frame: 000001 Time: 1574859835.749949000
#     #Frame: 000002 Time: 1574859835.749949000
#     #Frame: 000003 Time: 1574859835.799949000
#     # the timestamp_list contains [157xxxxx, ...], for each of them find the nearest one in GPS_IMU_Twist.txt and parse the frame
#     # and then enumerate all the correspoding frames to get traj
#     gps_path = Path(path)
#     twists = sorted(list(gps_path.glob('*.txt')))
#     traj = np.zeros((len(twists), 6))
#     for i, path in enumerate(twists):
#         # Read first GPS line: Latitude, Longitude, Altitude in degrees
#         twist = np.genfromtxt(path, delimiter=',',
#                               dtype=np.float32, max_rows=1)
#         xyz = twist
#         traj[i, 3:] = xyz

#     gt_t = torch.Tensor(traj).to(device)  # [n,6]
#     gt_t = tgm.rtvec_to_pose(gt_t)  # [n,4,4]
#     # gt_t = rel2abs_traj(gt_t)  # [n,4,4]
#     return gt_t



def load_odometry_from_folder(path, timestamp_list):
    # Let path be "../data/radiate/tiny_foggy/" 
    # There is "../data/radiate/tiny_foggy/GPS_IMU_Twist/" which contains the GPS/IMU data
    # There is "../data/radiate/tiny_foggy/GPS_IMU_Twist.txt" which contains:
    # Frame: 000001 Time: 1574859835.749949000
    # Frame: 000002 Time: 1574859835.749949000
    # Frame: 000003 Time: 1574859835.799949000
    # The timestamp_list contains [157xxxxx, ...]
    # For each timestamp, find the nearest one in GPS_IMU_Twist.txt and parse the frame
    # Enumerate all the corresponding frames to get traj
    
    gps_path = Path(path) / "GPS_IMU_Twist"
    twist_file = Path(path) / "GPS_IMU_Twist.txt"
    
    # Parse timestamps from the twist file
    frame_times = []
    with open(twist_file, 'r') as file:
        for line in file:
            parts = line.strip().split()
            time = float(parts[3])
            frame_times.append(time)
    
    # Normalize the times by subtracting the first value
    frame_times = np.array(frame_times)
    frame_times -= frame_times[0]
    
    timestamp_list = np.array(timestamp_list)
    timestamp_list -= timestamp_list[0]
    
    # Find nearest frames for each normalized timestamp
    traj_frames = []
    for timestamp in timestamp_list:
        nearest_index = np.argmin(np.abs(frame_times - timestamp))
        traj_frames.append(nearest_index)
    
    # Load the corresponding twist data
    rtvecs = []
    origin = None
    for i, frame_index in enumerate(traj_frames):
        frame = frame_index + 1  # Assuming frames are 1-indexed
        twist_path = gps_path / f"{frame:06d}.txt"

        position, orientation_angles = read_data_from_file(twist_path)
        
        if origin is None:
            origin = position
        
        # Adjust position relative to the origin
        relative_position = position - origin
        
        # Convert orientation angles to Rodrigues vector
        rtvecs.append(torch.tensor([*orientation_angles, *relative_position]))
    rtvecs = torch.stack(rtvecs, dim=0).to(device)
    print(rtvecs.dtype)
    gt_t = tgm.rtvec_to_pose(rtvecs)  # [n,4,4]
    gt_t = gt_t.float()
    # gt_t = rel2abs_traj(gt_t)  # [n,4,4]
    return gt_t

def read_data_from_file(filepath):
    with open(filepath, 'r') as file:
        lines = file.readlines()

    # 解析数据
    latitude = float(lines[0].split(',')[0])
    longitude = float(lines[0].split(',')[1])
    altitude = float(lines[0].split(',')[2])

    orientation = [float(x) for x in lines[4].split(',')]
    roll, pitch, yaw = quaternion_to_euler(orientation)

    # position = np.array([latitude, longitude, altitude])
    
    er = 6378137.0  
    scale = np.cos(latitude * np.pi / 180.0)
    tx = scale * longitude * np.pi * er / 180.0
    ty = scale * er * np.log(np.tan((90.0 + latitude) * np.pi / 360.0))
    tz = altitude
    position = np.array([tx, ty, tz])
    
    orientation_angles = np.array([roll, pitch, yaw])
    return position, orientation_angles

def quaternion_to_euler(quat):
    """Convert quaternion to Euler angles."""
    w, x, y, z = quat
    ysqr = y * y
    
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + ysqr)
    roll = np.arctan2(t0, t1)
    
    t2 = +2.0 * (w * y - z * x)
    t2 = np.clip(t2, -1.0, +1.0)
    pitch = np.arcsin(t2)
    
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (ysqr + z * z)
    yaw = np.arctan2(t3, t4)
    
    return roll, pitch, yaw

def getTraj(all_poses, all_inv_poses, k):
    """Convert the predicted poses to absolute trajectory. Each prediction in inputs is the relative pose for a sequence of [src_p, tgt, src_n].
    We form a trajectory from a chained sequence with k skip frames, e.g. [k, 2*k, 3*k, ..., N]. We shift the sequence by {i:i<k} to evaluate the full prediction.

    Args:
        all_poses (list): Predicted relative pose values for each src-to-tgt pair. List of torch.Tensor objects size [seq_length, B, 6]. rtvec=[rx, ry, rz, tx, ty, tz]
        all_inv_poses (list): Predicted relative pose values for each tgt-to-src pair. List of torch.Tensor objects.
        k (int): Skip frames. rtvec=[rx, ry, rz, tx, ty, tz]

    Returns:
        torch.Tensor: Backward and forward predicted trajectories
    """

    # pred = torch.zeros(self.gt.size(), dtype=self.gt.dtype, device = self.gt.device)
    all_poses_t = torch.cat(all_poses, 1)  # [seq_length, N, 6]
    all_inv_poses_t = torch.cat(all_inv_poses, 1)  # [seq_length, N, 6]

    N = all_poses_t.shape[1]  # len(all_poses) # number of sequences

    i = 0
    # for i in range(k):
    idx = torch.arange(i, N, k)

    # Previous src
    # (src2tgt + inv(tgt2src))/2 [n,6]
    b_pose = (all_poses_t[0, idx] - all_inv_poses_t[0, idx])/2
    # b_pose = all_poses_t[0, idx] #src2tgt [n,6]
    b_pose = -b_pose  # tgt2src [n,6]
    b_pose = tgm.rtvec_to_pose(b_pose)  # tgt2src [n,4,4]
    # b_pose = tgm.inv_rigid_tform(b_pose) # tgt2src [n,4,4]
    # b_inv_pose = tgm.rtvec_to_pose(all_inv_poses_t[0, idx]) # inv(src2tgt) [n,4,4]
    b_pose = rel2abs_traj(b_pose)  # [n,4,4]
    # b_pose = b_pose.cumsum(dim=0) # [n,6]

    # Next src
    # f_pose = tgm.rtvec_to_pose(all_poses_t[1, idx]) # tgt2src [n,4,4]
    # f_inv_pose = tgm.rtvec_to_pose(all_inv_poses_t[1, idx]) # inv(tgt2src) [n,4,4]
    # f_inv_pose = tgm.inv_rigid_tform(f_inv_pose) # inv(inv(tgt2src)) [n,4,4]
    # f_pose = (f_pose + f_inv_pose)/2 # (tgt2src + inv(inv(tgt2src)))/2 [n,4,4]
    # f_pose = rel2abs_traj(f_pose)
    # (src2tgt + inv(tgt2src))/2 [n,6]
    f_pose = (all_poses_t[1, idx] - all_inv_poses_t[1, idx])/2
    # f_pose = all_poses_t[1, idx] #src2tgt [n,6]
    # f_pose = -f_pose # tgt2src [n,6]
    # f_pose = f_pose.cumsum(dim=0) # [n,6]
    f_pose = tgm.rtvec_to_pose(f_pose)  # src2tgt [n,4,4]
    f_pose = rel2abs_traj(f_pose)  # [n,4,4]
    # end for

    f_xyz = f_pose[:, :3, 3]
    b_xyz = b_pose[:, :3, 3]
    return b_xyz.squeeze(), f_xyz.squeeze()


def align_to_origin(pose):
    """Aligns a given trajectory to the origin.

    Args:
        pose (torch.Tensor): Absolute trajectory in the form of homogenous transformation matrix. Shape: [N,4,4]

    Returns:
        torch.Tensor: Origin-aligned absolute trajectory.
    """

    aligned_pose = pose.clone()
    inv_pose0 = tgm.inv_rigid_tform(pose[0:1])
    aligned_pose = torch.matmul(aligned_pose, inv_pose0)
    return aligned_pose


def rel2abs_traj(rel_pose):
    """Convert a given relative pose sequences to absolute pose sequences.

    Args:
        rel_pose (torch.Tensor): Relative pose sequence in the form of homogenous transformation matrix. Shape: [N,4,4]

    Returns:
        torch.Tensor: Absolute pose sequence in the form of homogenous transformation matrix. Shape: [N,4,4]
    """

    global_pose = torch.eye(4, dtype=rel_pose.dtype, device=rel_pose.device)
    abs_pose = torch.zeros_like(rel_pose)
    for i in range(rel_pose.shape[0]):
        global_pose = global_pose @ rel_pose[i]
        abs_pose[i] = global_pose

    return abs_pose


# threshold for checking that point crosscorelation
# is full rank in corresponding_points_alignment
AMBIGUOUS_ROT_SINGULAR_THR = 1e-15


def corresponding_points_alignment(
    X,
    Y,
    weights=None,
    estimate_scale: bool = False,
    allow_reflection: bool = False,
    eps: float = 1e-9,
):
    """
    Finds a similarity transformation (rotation `R`, translation `T`
    and optionally scale `s`)  between two given sets of corresponding
    `d`-dimensional points `X` and `Y` such that:

    `s[i] X[i] R[i] + T[i] = Y[i]`,

    for all batch indexes `i` in the least squares sense.

    The algorithm is also known as Umeyama [1]. Code is based on PyTorch3d.

    Args:
        **X**: Batch of `d`-dimensional points of shape `(minibatch, num_point, d)`
            or a `Pointclouds` object.
        **Y**: Batch of `d`-dimensional points of shape `(minibatch, num_point, d)`
            or a `Pointclouds` object.
        **weights**: Batch of non-negative weights of
            shape `(minibatch, num_point)` or list of `minibatch` 1-dimensional
            tensors that may have different shapes; in that case, the length of
            i-th tensor should be equal to the number of points in X_i and Y_i.
            Passing `None` means uniform weights.
        **estimate_scale**: If `True`, also estimates a scaling component `s`
            of the transformation. Otherwise assumes an identity
            scale and returns a tensor of ones.
        **allow_reflection**: If `True`, allows the algorithm to return `R`
            which is orthonormal but has determinant==-1.
        **eps**: A scalar for clamping to avoid dividing by zero. Active for the
            code that estimates the output scale `s`.

    Returns:
        3-element named tuple `SimilarityTransform` containing
        - **R**: Batch of orthonormal matrices of shape `(minibatch, d, d)`.
        - **T**: Batch of translations of shape `(minibatch, d)`.
        - **s**: batch of scaling factors of shape `(minibatch, )`.

    References:
        [1] Shinji Umeyama: Least-Suqares Estimation of
        Transformation Parameters Between Two Point Patterns
    """

    Xt = X
    Yt = Y
    num_points = X.shape[1] * torch.ones(  # type: ignore
        X.shape[0], device=X.device, dtype=torch.int64
    )
    num_points_Y = Y.shape[1] * torch.ones(  # type: ignore
        Y.shape[0], device=Y.device, dtype=torch.int64
    )

    if (Xt.shape != Yt.shape) or (num_points != num_points_Y).any():
        raise ValueError(
            "Point sets X and Y have to have the same \
            number of batches, points and dimensions."
        )

    b, n, dim = Xt.shape

    # compute the centroids of the point sets
    # oputil.wmean(Xt, weight=weights, eps=eps)
    Xmu = Xt.mean(dim=1, keepdim=True)
    # oputil.wmean(Yt, weight=weights, eps=eps)
    Ymu = Yt.mean(dim=1, keepdim=True)

    # mean-center the point sets
    Xc = Xt - Xmu
    Yc = Yt - Ymu

    total_weight = torch.clamp(num_points, 1)
    # special handling for heterogeneous point clouds and/or input weights
    if weights is not None:
        Xc *= weights[:, :, None]
        Yc *= weights[:, :, None]
        total_weight = torch.clamp(weights.sum(1), eps)

    if (num_points < (dim + 1)).any():
        warnings.warn(
            "The size of one of the point clouds is <= dim+1. "
            + "corresponding_points_alignment cannot return a unique rotation."
        )

    # compute the covariance XYcov between the point sets Xc, Yc
    XYcov = torch.bmm(Xc.transpose(2, 1), Yc)
    XYcov = XYcov / total_weight[:, None, None]

    # decompose the covariance matrix XYcov
    U, S, V = torch.svd(XYcov)

    # catch ambiguous rotation by checking the magnitude of singular values
    if (S.abs() <= AMBIGUOUS_ROT_SINGULAR_THR).any() and not (
        num_points < (dim + 1)
    ).any():
        warnings.warn(
            "Excessively low rank of "
            + "cross-correlation between aligned point clouds. "
            + "corresponding_points_alignment cannot return a unique rotation."
        )

    # identity matrix used for fixing reflections
    E = torch.eye(dim, dtype=XYcov.dtype, device=XYcov.device)[
        None].repeat(b, 1, 1)

    if not allow_reflection:
        # reflection test:
        #   checks whether the estimated rotation has det==1,
        #   if not, finds the nearest rotation s.t. det==1 by
        #   flipping the sign of the last singular vector U
        R_test = torch.bmm(U, V.transpose(2, 1))
        E[:, -1, -1] = torch.det(R_test)

    # find the rotation matrix by composing U and V again
    R = torch.bmm(torch.bmm(U, E), V.transpose(2, 1))

    if estimate_scale:
        # estimate the scaling component of the transformation
        trace_ES = (torch.diagonal(E, dim1=1, dim2=2) * S).sum(1)
        Xcov = (Xc * Xc).sum((1, 2)) / total_weight

        # the scaling component
        s = trace_ES / torch.clamp(Xcov, eps)

        # translation component
        T = Ymu[:, 0, :] - s[:, None] * torch.bmm(Xmu, R)[:, 0, :]
    else:
        # translation component
        T = Ymu[:, 0, :] - torch.bmm(Xmu, R)[:, 0, :]

        # unit scaling since we do not estimate scale
        s = T.new_ones(b)

    return R, T, s


def _apply_similarity_transform(
    X: torch.Tensor, R: torch.Tensor, T: torch.Tensor, s: torch.Tensor
) -> torch.Tensor:
    """
    Applies a similarity transformation parametrized with a batch of orthonormal
    matrices `R` of shape `(minibatch, d, d)`, a batch of translations `T`
    of shape `(minibatch, d)` and a batch of scaling factors `s`
    of shape `(minibatch,)` to a given `d`-dimensional cloud `X`
    of shape `(minibatch, num_points, d)`
    Code is based on PyTorch3d.
    """
    X = s[:, None, None] * torch.bmm(X, R) + T[:, None, :]
    return X


def _apply_similarity_transform_hom(
    X: torch.Tensor, R: torch.Tensor, T: torch.Tensor, s: torch.Tensor
) -> torch.Tensor:
    """
    Applies a similarity transformation parametrized with a batch of orthonormal
    matrices `R` of shape `(minibatch, d, d)`, a batch of translations `T`
    of shape `(minibatch, d)` and a batch of scaling factors `s`
    of shape `(minibatch,)` to a given `d`-dimensional poses `X`
    of shape `(minibatch, num_points, d, d)`
    Code is based on PyTorch3d.
    """
    X_tformed = X.clone()

    tform = torch.eye(4, dtype=X.dtype, device=X.device)
    tform = tform[None, :, :]
    tform = tform.repeat(X_tformed.shape[0], 1, 1)
    tform[:, :3, :3] = s[:, None, None] * R
    tform[:, :3, 3] = T
    for i in range(X_tformed.shape[0]):
        # [N,4,4] @ [4,4] = [N,4,4]
        X_tformed[i] = torch.matmul(X_tformed[i], tform[i])
    return X_tformed  # [B,N,4,4]
