import torch.utils.data as data
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Tuple
from PIL import Image
import random
import lidar
import utils_warp as utils
from colour_demosaicing import demosaicing_CFA_Bayer_bilinear as demosaic
from .robotcar_camera.camera_model import CameraModel
import torch
from torchvision import transforms
import scipy.ndimage
from utils.utils import euler_to_rotation_matrix
import torchvision
import yaml
from utils.calibration import Calibration
import conversions as tgm
from collections import defaultdict
import utils.custom_transforms_cam as CustomT
from scipy import sparse
import torch.nn.functional as F


def find_nonzero_max_index(radar_tgt_img, col_index):
    """
    Finds the index of the maximum non-zero element in the specified column of a tensor.
    
    Args:
    - radar_tgt_img (torch.Tensor): The input tensor, assumed to be of shape [h, w].
    - col_index (int): The column index for which to find the maximum non-zero index.
    
    Returns:
    - max_nonzero_idx (int or None): The index of the maximum non-zero element in the column, or None if no non-zero elements are found.
    """
    # Extract the specified column (shape: [h])
    column = radar_tgt_img[:, col_index]
    
    # Find the indices where the values are non-zero
    nonzero_indices = torch.nonzero(column, as_tuple=False).squeeze(1)
    
    if nonzero_indices.numel() == 0:
        # No non-zero elements found
        return None
    
    # Find the maximum non-zero index
    max_nonzero_idx = nonzero_indices.max().item()
    
    return max_nonzero_idx


def find_nonzero_minval(radar_pov):

    radar_pov_squeezed = radar_pov.squeeze(0)  # shape becomes [h, w]
    radar_pov_squeezed_nonzero = radar_pov_squeezed.clone()
    radar_pov_squeezed_nonzero[radar_pov_squeezed_nonzero == 0] = float('inf')
    
    # Find the minimum non-zero value in each column (along height dimension)
    min_vals, _ = radar_pov_squeezed_nonzero.min(dim=0, keepdim=True)  # shape becomes [1, w]
    return min_vals.min()
  


class SequenceFolder(data.Dataset):
    """LIDAR dataset loader.
    
    root: dataset root
    dataset: type
    lidar/cam trans rots external parameter
    
    """

    def __init__(self, 
                 root, 
                 dataset, 
                 seed=None, 
                 mode='train', 
                 sequence_length=3,
                 skip_frames=1,
                 load_mutlimodal=True, 
                 cam_preprocessed=True, 
                 nsamples=0,
                 cam_transform=None, 
                 transform=None, 
                 sequence=None,
                 sample_num=-1,
                 load_lidar=False,
                 get_vo=True,
                 get_lo=False,
                 get_radar_processed=True, 
                 config_file='config/config.yaml', 
                 cart_resolution=200./512, 
                 cart_pixels=512,
                 depth_gt_dir='depth_ac',
                 img_aug=False,
                 img_norm=True,
                 radar_slices=True,
                 radar_pov_vertical_aug=False):
        
        '''
            in __getitem__ funciton, default not to load_lidar data
        '''
        np.random.seed(seed)
        random.seed(seed)
        self.root = Path(root)
        self.dataset = dataset
        self.preprocessed = cam_preprocessed
        # self.res = res
        self.multimodal_sampler = SampleFinder([-1, 1])
        self.load_mutlimodal= load_mutlimodal
        self.load_lidar = load_lidar
        self.get_lo = get_lo
        self.get_vo = get_vo
        self.img_aug = img_aug
        self.img_norm = img_norm
        self.radar_slices = radar_slices
        self.radar_pov_vertical_aug = radar_pov_vertical_aug

        if sequence is not None:
            self.scenes = [self.root/sequence]
        else:
            scene_list_path = self.root/(mode + '.txt')
            self.scenes = [self.root/folder.strip()
                           for folder in open(scene_list_path) if not folder.strip().startswith("#")]

        self.isCartesian = True
        if self.isCartesian:
            self.radar_folder = 'radar_cart' if dataset == 'robotcar' else 'Navtech_Cartesian'
        else:
            self.radar_folder = 'radar' if dataset == 'robotcar' else 'Navtech_Polar'
        # self.max_range = max_range# if dataset == 'radiate' else 50.0

        
        self.radar_pov_dir = 'radar_pov'
        self.radar_pov_slices = 'radar_pov_slices'

        # load camera
        if dataset == 'radiate':
            if self.preprocessed:
                self.stereo_left_folder = 'stereo_undistorted/left'
                self.stereo_right_folder = 'stereo_undistorted/right'
            else:
                self.stereo_left_folder = 'zed_left'
                self.stereo_right_folder = 'zed_right'
        else:
            raise NotImplementedError(
                'The chosen dataset is not implemented yet! Given: {}'.format(dataset))

             
        self.transform = transform
        self.cam_transform = cam_transform
        self.nsamples = nsamples
        self.mode = mode
        self.k = skip_frames
        self.depth_gt_dir = depth_gt_dir
        self.get_radar_processed = get_radar_processed
        if dataset == 'radiate':
            self.lidar_folder = 'velo_lidar'
            self.lidar_ext = '*.csv'
            self.lidar_timestamps = 'velo_lidar.txt'
            self.stereo_timestamps = 'zed_left.txt'
            self.radar_timestamps = 'Navtech_Polar.txt'
        else:
            raise NotImplementedError(
                'The chosen dataset is not implemented yet! Given: {}'.format(dataset))
        self.ground_thr = -1.8 if dataset == 'radiate' else 1.0
        
        self.crawl_folders(sequence_length)
        if sample_num != -1:
            self.samples = self.samples[:sample_num]
            
            
            
        # radar setting
        
        
        with open(config_file, 'r') as file:
            self.config = yaml.full_load(file)
        with open(self.config['calib_file'], 'r') as file:
            self.calib = yaml.full_load(file)
        self.config.update(self.calib)
        
        self.calib = Calibration(self.config)
        self.RadarToLeftT = self.calib.RadarT - self.calib.LeftT
        self.RadarToRightT = self.calib.RadarT - self.calib.RightT

        self.RadarToLeftR = self.calib.RadarR - self.calib.LeftR
        self.RadarToRightR = self.calib.RadarR - self.calib.RightR
        self.RadarToLeft = self.calib.transform(
            self.RadarToLeftR, self.RadarToLeftT)
        
        self.left_cam_mat =  self.calib.left_cam_mat
        self.cart_resolution = cart_resolution
        self.cart_pixels = cart_pixels
        self.padding_mode = "zeros"
        # self.height = 1.8

        ranges_x = (torch.arange(self.cart_pixels)-self.cart_pixels//2)
        ranges_y = (torch.arange(self.cart_pixels)-self.cart_pixels//2)

        ranges_x = ranges_x*self.cart_resolution
        ranges_y = ranges_y*self.cart_resolution

        x, y = torch.meshgrid(ranges_x, ranges_y)
        x = torch.flatten(x)
        y = torch.flatten(y)

        # [3,N] Augment with zero z column
        # radar coord the z is 0
        xy = torch.vstack((x, y, torch.zeros_like(x)))
        xy = torch.transpose(xy, 0, 1)  # [N,3]
        self.xy_hom = tgm.convert_points_to_homogeneous(xy)  # [N,4]
        self.RADAR_PTS_NUM = 1500
        


    def crawl_folders(self, sequence_length):
        # k skip frames
        sequence_set = []
        demi_length = (sequence_length-1)//2
        self.shifts = list(range(-demi_length * self.k,
                                 demi_length * self.k + 1, self.k))
        self.shifts.pop(demi_length)
        for scene in self.scenes:
            # print(scene)
            # intrinsics = np.genfromtxt(scene/'cam.txt').astype(np.float32).reshape((3, 3))
            intrinsics = utils.get_intrinsics_matrix(
                self.dataset, preprocessed=self.preprocessed)

            lidar_imgs = sorted(list((scene/self.lidar_folder).glob(self.lidar_ext)))
            radar_imgs = sorted(list((scene/self.radar_folder).glob('*.png')))
            
            
            # depth setting
            depth_gt_path = scene/self.depth_gt_dir
            depth_paths = sorted(list(depth_gt_path.glob('*.tiff')))
            depth_timestamps = [float(filename.name.split('_depth')[0]) for filename in depth_paths]

            if len(radar_imgs) < sequence_length:
                continue
            if len(radar_imgs) < sequence_length:
                continue

            if self.load_mutlimodal:
                left_imgs = sorted(
                    list((scene/self.stereo_left_folder).glob('*.png')))

                if len(left_imgs) < sequence_length or len(lidar_imgs) < sequence_length:
                    continue

                f_lt = scene/self.lidar_timestamps
                f_mt = scene/self.stereo_timestamps
                f_rt = scene/self.radar_timestamps

                if self.dataset == 'radiate':
                    rts = [float(folder.strip().split(':')[-1].strip())
                           for folder in open(f_rt)]
                    lts = [float(folder.strip().split(':')[-1].strip())
                           for folder in open(f_lt)]
                    mts = [float(folder.strip().split(':')[-1].strip())
                           for folder in open(f_mt)]
                    
                elif self.dataset == 'robotcar':
                    # Robotcar timestamps are in microsecs.
                    # Read them in secs.
                    rts = [float(folder.strip().split()[0].strip())/1e6
                           for folder in open(f_rt)]
                    mts = [float(folder.strip().split()[0].strip())/1e6
                           for folder in open(f_mt)]
                    lts = [float(folder.strip().split()[0].strip())/1e6
                           for folder in open(f_lt)]
                else:
                    raise NotImplementedError(
                        'Currently, RADIATE and RobotCar datasets supported for VO')
                # Some scenes contain timestamps more than images. Drop the extra timestamps.
                mts = mts[:len(left_imgs)]

                radar_idxs = list(
                    range(demi_length * self.k, len(radar_imgs)-demi_length * self.k))
                
                cam_matches_all, lidar_matches_all = self.multimodal_sampler.find_cam_lidar_samples(radar_idxs,lts=lts,mts=mts,rts=rts)

            for cnt, i in enumerate(range(demi_length * self.k, len(radar_imgs)-demi_length * self.k)):
                sample = {'tgt': radar_imgs[i], 'ref_imgs': [], 'tgt_timestamp': rts[i]}
                for j in self.shifts:
                    sample['ref_imgs'].append(radar_imgs[i+j])

                if self.load_mutlimodal:

                    cam_matches = cam_matches_all[cnt]
                    lidar_matches = lidar_matches_all[cnt]
                    sample['tgt_depth_gt']  = None
                    
                    if cam_matches and lidar_matches:
                        # Add all the monocular frames between the matched source and target frames.
                        sample['intrinsics'] = intrinsics
                        sample['vo_tgt_img'] = left_imgs[cam_matches[0]]
                        
                        
                        if depth_gt_path.exists():
                            cam_timestamp = mts[cam_matches[0]]
                            depth_timestamp = min(depth_timestamps, key=lambda x: abs(cam_timestamp - x))
                            depth_file = depth_gt_path/(str(depth_timestamp) + "_depth.tiff")
                            sample['tgt_depth_gt'] = depth_file
                        
                        
                        sample['vo_ref_imgs'] = []
                        sample['lo_ref_imgs'] = []
                        sample['scene'] = scene

                        sample['vo_ref_imgs'].extend([
                            left_imgs[ref] for ref in cam_matches[1:]])
                        # for j in self.shifts:
                        #     sample['intrinsics'].append(intrinsics)
                        
                        sample['lo_tgt_img'] = lidar_imgs[lidar_matches[0]]
                        sample['lo_ref_imgs'].extend([
                        lidar_imgs[ref] for ref in lidar_matches[1:]])
                    else:
                        continue

                sequence_set.append(sample)
        if self.mode.startswith('train'):
            random.shuffle(sequence_set)
        self.samples = sequence_set

        # Subsample dataset
        if self.nsamples > 0 and self.nsamples < len(self.samples):
            skip = len(self.samples)//self.nsamples
            self.samples = self.samples[0:skip*self.nsamples:skip]





    def load_camera_img_as_float(self, path):
        img = Image.open(path)
        # img = img.resize((640, 384))
        if self.dataset == 'robotcar':
            img = demosaic(img, 'gbrg')
            img = self.cam_model_left.undistort(img)
        img = np.array(img).astype(np.uint8)
        # img = img.astype(np.float32) / 255.
        return img

    def load_undistorted_mono_img_as_float(self, path):
        img = Image.open(path)
        return img
    
    
    
    def load_cart_as_float(self, path):
        raw_data = Image.open(path)
        raw_data = np.array(raw_data)
        # raw_data = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        cart_img = raw_data.astype(np.float32)[np.newaxis, :, :] / 255.
        # cart_img[cart_img < 0.3] = 0

        # Calculate new dimensions based on max_range and resolution
        new_width = self.cart_pixels
        new_height = self.cart_pixels

        # Rescale the image using scipy.ndimage.zoom
        height_scale = cart_img.shape[1] / new_height
        width_scale = cart_img.shape[2] / new_width
        zoom_factors = (1, 1/height_scale, 1/width_scale)
        resized_img = scipy.ndimage.zoom(cart_img, zoom_factors, order=0)
        resized_img[resized_img < 0.3] = 0

        return resized_img

    def project_radar(self, radar, radar_extrinsics, cam_intrinsic):
        """
        Method to project the radar into the camera and ensure only the front of the radar is projected

        :type radar: torch.tensor
        :param radar: radar point cloud with shape Nx5 (x,y,z,intensity,ring)

        :type radar_extrinsics: torch.tensor
        :param radar_extrinsics: 4x4 matrix with radar extrinsic parameters (Rotation and translations)

        :type cam_intrinsic: torch.tensor
        :param cam_intrinsic: 3x3 matrix with camera intrinsic parameters in the form
            [[fx 0 cx],
            [0 fx cy],
            [0 0 1]]

        :rtype: torch.tensor
        :return: returns the projected radar into the respective camera with the same size as the camera
        
        coordinates for camera and the point clouds processed
        #           z
        #          /
        #         /
        #        0---------> x
        #        |  
        #        |   
        #        | 
        #       y 
        """
        
        

        # Camera intrinsics
        fx = cam_intrinsic[0, 0]
        fy = cam_intrinsic[1, 1]
        cx = cam_intrinsic[0, 2]
        cy = cam_intrinsic[1, 2]

        # Initialize radar image with zeros using torch
        im_radar = torch.zeros((self.config['left_cam_calib']['res'][1],
                                self.config['left_cam_calib']['res'][0]), dtype=torch.float32)

        # Radar points and transformation (radar_points is Nx3)
        radar_points = radar[:, :3].T  # Transpose to match the matrix multiplication shape
        R = radar_extrinsics[:3, :3]  # Rotation matrix
        radar_points = torch.matmul(R, radar_points).T  # Apply rotation
        radar_points += radar_extrinsics[:3, 3]  # Apply translation

        projected_points = []
        for i in range(radar.shape[0]):
            if radar_points[i, 2] > 0 and radar_points[i, 2] < self.config['radar_proj']['max_dist']:
                xx = int(((radar_points[i, 0] * fx) / radar_points[i, 2]) + cx)
                yy = int(((radar_points[i, 1] * fy) / radar_points[i, 2]) + cy)

                if (xx > 0 and xx < self.config['left_cam_calib']['res'][0] and
                        yy > 0 and yy < self.config['left_cam_calib']['res'][1]):
                    dist = torch.sqrt(radar_points[i, 0]**2 +
                                    radar_points[i, 1]**2 +
                                    radar_points[i, 2]**2)
                    
                    # Append the projected point and distance
                    projected_points.append((yy, xx, dist))

        # Process the projected points to handle duplicates
        points_dict = defaultdict(list)

        for yy, xx, dist in projected_points:
            points_dict[(yy, xx)].append(dist)

        # Update the image with minimum distance for each pixel
        for (yy, xx), dists in points_dict.items():
            im_radar[yy, xx] = torch.min(torch.tensor(dists))  # Use torch.min on dists

        # Replace any infinite values with 0 (assuming inf represents missing depth)
        im_radar[im_radar == float('inf')] = 0

        # Get radar points (projecting radar points in front of the camera)
        radar_pts = radar_points[radar_points[:, 2] > 0]

        # Handle radar points padding or subsampling based on the count
        valid_radar_pts_cnt = radar_pts.shape[0]
        if valid_radar_pts_cnt <= self.RADAR_PTS_NUM:
            padding_radar_pts = torch.zeros((self.RADAR_PTS_NUM, 3), dtype=radar_pts.dtype)
            padding_radar_pts[:valid_radar_pts_cnt, :] = radar_pts
        else:
            # Random subsampling of radar points using torch.randperm
            random_idx = torch.randperm(valid_radar_pts_cnt)[:self.RADAR_PTS_NUM]
            padding_radar_pts = radar_pts[random_idx, :]

        return im_radar, padding_radar_pts, valid_radar_pts_cnt    
        
    def get_pov_pts(self, radar_bev):
        radar_bev_flat = radar_bev.squeeze().T.flatten()
        non_zero_indices = torch.nonzero(radar_bev_flat, as_tuple=True)
        radar_points = self.xy_hom[non_zero_indices]
        
        radar_depth, front_pts, valid_radar_pts_cnt = self.project_radar(radar_points, 
                                                    torch.tensor(self.RadarToLeft, device=radar_points.device, dtype=radar_points.dtype), 
                                                    torch.tensor(self.left_cam_mat, device=radar_points.device, dtype=radar_points.dtype))

        # get only the 1500
        return radar_depth, front_pts, valid_radar_pts_cnt

    @staticmethod
    def radar_vertical_aug(radar_pov):

        radar_pov_squeezed = radar_pov.squeeze(0)  # shape becomes [h, w]
        radar_pov_squeezed_nonzero = radar_pov_squeezed.clone()
        radar_pov_squeezed_nonzero[radar_pov_squeezed_nonzero == 0] = float('inf')
        
        # Find the minimum non-zero value in each column (along height dimension)
        min_vals, _ = radar_pov_squeezed_nonzero.min(dim=0, keepdim=True)  # shape becomes [1, w]
        min_vals[min_vals == float('inf')] = 0
        radar_pov_augmented = min_vals.repeat(radar_pov_squeezed.size(0), 1)  # shape becomes [h, w]
        radar_pov_augmented = radar_pov_augmented.unsqueeze(0)  # shape becomes [1, h, w]
        
        return radar_pov_augmented
        
    @staticmethod
    def save_depth_as_png(depth_tensor, file_path):
        """
        Saves a depth tensor of shape [1, h, w] as a PNG image.

        Args:
        - depth_tensor (torch.Tensor): The depth tensor of shape [1, h, w].
        - file_path (str): The file path to save the PNG image.

        Returns:
        - None
        """
        # Ensure the tensor is on the CPU and remove the batch dimension (shape: [h, w])
        depth_image = depth_tensor.squeeze(0).cpu().numpy()  # Convert to NumPy array
 
        depth_min = depth_image.min()
        depth_max = depth_image.max()
        
        if depth_max - depth_min > 0:
            depth_image = (depth_image - depth_min) / (depth_max - depth_min)  # Scale to [0, 1]
        depth_image = (depth_image * 255).astype(np.uint8)  # Scale to [0, 255] and convert to uint8
        
        image = Image.fromarray(depth_image)
        image.save(file_path)

    def __getitem__(self, index):
        sample = self.samples[index]

        radar_tgt_img = self.load_cart_as_float(sample['tgt'])
        radar_ref_imgs = [self.load_cart_as_float(ref_img)
                    for ref_img in sample['ref_imgs']]

        if self.transform is not None:
            raise NotImplementedError
  
        if self.get_vo:
            if self.preprocessed or self.dataset == 'radiate':
                cam_tgt_img = self.load_undistorted_mono_img_as_float(
                    sample['vo_tgt_img'])
                cam_ref_imgs = [self.load_undistorted_mono_img_as_float(
                    ref_img) for ref_img in sample['vo_ref_imgs']]
            else:
               raise NotImplementedError
                

        else:
            cam_tgt_img = []
            cam_ref_imgs = []
            
        
        tgt_timestamp = sample['tgt_timestamp']   
        intrinsics = sample['intrinsics'] 
        radar_tgt_img = torch.tensor(radar_tgt_img)
        radar_ref_imgs = torch.stack([torch.tensor(img) for img in radar_ref_imgs])
        
        # from the radar_tgt_img to the pov and pts
        if self.get_radar_processed:
            # TODO add var radar_slices
            radar_pov_dir = Path(self.radar_pov_dir if not self.radar_slices else self.radar_pov_slices)
            suffix = '.tiff' if not self.radar_slices else '.npz'
            tgt_pov_path = sample['tgt'].parents[0].parent / radar_pov_dir / (sample['tgt'].stem + suffix)
            ref_pov_paths = [ref_path.parents[0].parent / radar_pov_dir / (ref_path.stem + suffix) for ref_path in sample['ref_imgs'] ]
            
            if self.radar_slices:
                radar_tgt_pov = torch.from_numpy( (sparse.load_npz(tgt_pov_path) ).toarray().reshape(16, 376, 672))
                radar_ref_povs = [torch.from_numpy( (sparse.load_npz(ref_pov_path)).toarray().reshape(16, 376, 672)) \
                    for ref_pov_path in ref_pov_paths]
            else:
                transform = torchvision.transforms.Compose([
                    torchvision.transforms.ToTensor(),
                ])
                radar_tgt_pov = transform(Image.open(tgt_pov_path))
                radar_ref_povs = [transform(Image.open(ref_pov_path)) for ref_pov_path in ref_pov_paths]
                
            radar_ref_povs = torch.stack(radar_ref_povs)
            radar_povs = torch.cat([radar_tgt_pov, radar_ref_povs.view(radar_ref_povs.size(0)*radar_ref_povs.size(1), radar_ref_povs.size(2), radar_ref_povs.size(3))], dim=0) 
            radar_povs = F.interpolate(radar_povs.unsqueeze(0), (cam_tgt_img.size[1], cam_tgt_img.size[0]), mode='nearest').squeeze()
            
            if self.radar_pov_vertical_aug:
                for i in range(radar_povs.size(0)):
                    radar_povs[i] = self.radar_vertical_aug(radar_povs[i])
              
        else:
            raise NotImplementedError

        
        imgs, radar_povs, intrinsics = self.pil_to_tensor([cam_tgt_img]+ cam_ref_imgs, radar_povs, [np.copy(sample['intrinsics']) for i in range(3)])
        cam_tgt_img = imgs[0]
        cam_ref_imgs = torch.stack(imgs[1:], dim=0)
        
        radar_channel_num = 1 if not self.radar_slices else 16
        radar_tgt_pov = radar_povs[:radar_channel_num, ...]
        radar_ref_povs = radar_povs[radar_channel_num:, ...].view(2, radar_channel_num, radar_tgt_pov.size(-2), radar_tgt_pov.size(-1))
        intrinsics = torch.tensor(intrinsics[0])
        tgt_timestamp = torch.tensor(tgt_timestamp, dtype=float)
        
        
        if sample['tgt_depth_gt'] is not None:
            depth_file = sample['tgt_depth_gt']
            depth_transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
                ])
            tgt_depth_gt = depth_transform(Image.open(depth_file))
            # tgt_depth_gt = torch.tensor(sample['tgt_depth_gt'])
        else:
            tgt_depth_gt = []
            
        return_data = {'radar_tgt_img':radar_tgt_img, 'radar_ref_imgs':radar_ref_imgs,
                       'radar_tgt_pov':radar_tgt_pov, 'radar_ref_povs':radar_ref_povs,
                       'cam_tgt_img':cam_tgt_img, 'cam_ref_imgs':cam_ref_imgs,
                       'tgt_timestamp':tgt_timestamp,
                       'intrinsics':intrinsics,
                       'tgt_depth_gt': tgt_depth_gt
                       }
        
        return return_data


    def pil_to_tensor(self, imgs, radar_povs, intrinsics):
        # transform = transforms.ToTensor()
        # TODO verify especcial the raddom scale crop and the to tensor seq
        compose_list = []
        if self.img_aug and self.mode.startswith('train'):
            compose_list.append(CustomT.ColorJitter(brightness=0.1, contrast=0.1,
                                    saturation=0.1, hue=0.1))
            compose_list.append(CustomT.RandomScaleCrop())
        compose_list.append(CustomT.ToTensor())
        if self.img_norm:
            compose_list.append(CustomT.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]))
        transform = CustomT.Compose(compose_list)
        return transform(imgs, radar_povs, intrinsics)


    def __len__(self):
        return len(self.samples)




class SampleFinder:
    def __init__(self, shifts: List[int]):
        self.shifts = shifts
        

    def find_neighbor_samples(self, t: int, mts: List[List[float]], last_search_idx: int) -> int:
        """Finds the nearest monocular timestamp for the given lidar timestamp

        Args:
            t (int): Timestamp of the target frame
            mts (List[List[float]]): List of monocular timestamps

        Returns:
            int: Index of the matched monocular frame
        """

        del_t = 0.050  # the match must be within 50ms of t
        # First check if t is outside monocular frames but still within thr close
        # Check if t comes before monocular frames
        if t < mts[last_search_idx]:
            return last_search_idx if mts[last_search_idx]-t < del_t else -1
        # Check if t comes after monocular frames
        if t > mts[-1]:
            return len(mts)-1 if t-mts[-1] < del_t else -1
        # Otherwise search within monocular frames
        for i in range(last_search_idx, len(mts)-1):
            if t > mts[i] and t < mts[i+1]:
                idx = i if (t-mts[i]) < (mts[i+1]-t) else i+1
                idx = idx if abs(mts[idx]-t) < del_t else -1
                return idx
        return -1


    def find_cam_lidar_samples(self, 
                               t_idxs: List[int], 
                               lts: List[float], 
                               mts: List[float],
                               rts: List[float]) -> Tuple[List[List[int]], List[List[int]]]:
        """Returns indexes of monocular frames in the form of
        [[tgt, [src-1,...,tgt], [tgt,...,src+1]],
            [tgt, [src-1,...,tgt], [tgt,...,src+1]],...]

        radar's fps is lowest, so make it base and search for another 2 modal data
        Args:
            t_idxs (List[int]): Indices of the target radar frames
            lts (List[float]): List of lidar timestamps
            mts (List[float]): List of monocular timestamps
            rts (List[float]): List of radar timestamps

        Returns:
            Tuple[List[List[int]], List[List[int]]]: Indexes of the matched monocular and lidar frames.
        """
        mono_t_matches = []
        lidar_t_matches = []
        mono_last_search_idx = 0
        lidar_last_search_idx = 0
        
        last_search_failed_flg = False
        
        

        for t_idx in t_idxs:
            
            if last_search_failed_flg:
                mono_t_matches.append([])
                lidar_t_matches.append([])
                last_search_failed_flg = False
                
            mono_idxs = []
            lidar_idxs = []

            mono_idx = self.find_neighbor_samples(rts[t_idx], mts, mono_last_search_idx)
            if mono_idx < 0:
                last_search_failed_flg = True
                continue

            lidar_idx = self.find_neighbor_samples(rts[t_idx], lts, lidar_last_search_idx)
            if lidar_idx < 0:
                last_search_failed_flg = True
                continue

            mono_idxs.append(mono_idx)
            lidar_idxs.append(lidar_idx)

            for s in self.shifts:
                if t_idx + s >= len(rts):
                    last_search_failed_flg = True
                    break

                mono_idx = self.find_neighbor_samples(rts[t_idx + s], mts, mono_last_search_idx)
                if mono_idx < 0:
                    last_search_failed_flg = True
                    break

                lidar_idx = self.find_neighbor_samples(rts[t_idx + s], lts, lidar_last_search_idx)
                if lidar_idx < 0:
                    last_search_failed_flg = True
                    break

                mono_idxs.append(mono_idx)
                lidar_idxs.append(lidar_idx)
                
            if last_search_failed_flg:
                continue

            # Check if any of the indices are not found or if the matched indices are not unique
            if len(set(mono_idxs)) != len(self.shifts) + 1 or len(set(lidar_idxs)) != len(self.shifts) + 1:
                mono_t_matches.append([])
                lidar_t_matches.append([])
                continue

            mono_last_search_idx = mono_idxs[1]
            lidar_last_search_idx = lidar_idxs[1]
            mono_t_matches.append(mono_idxs)
            lidar_t_matches.append(lidar_idxs)
            
            
        if last_search_failed_flg:
            mono_t_matches.append([])
            lidar_t_matches.append([])

        return mono_t_matches, lidar_t_matches
    
    
    
if __name__ == '__main__':
    # 1. check the timestamp of the camera and the radar
    # 2. check the pov radar img and points if possible
    # dataset = SequenceFolder()
    

    
    dataset = SequenceFolder('../data/radiate', dataset='radiate', 
                                   lidar_trans=lidar_trans, lidar_rots=lidar_rots,
                                #    res=1., max_range=100.,
                                   seed=3407, mode='train',
                                   sample_num=-1)
    
    
    print(len(dataset))
    for i in range(len(dataset)):
        data = dataset[i]
    
    # pass
