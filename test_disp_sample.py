from PIL import Image
import numpy as np
from pathlib import Path
import argparse
from tqdm import tqdm
import time

import models
# import utils
from utils.utils import *
# import custom_transforms_mono as T

import torch
import torch.backends.cudnn as cudnn
import torchvision as tv
import torchvision.transforms as T
# from datasets.sequence_folders_disp import ImageFolder
from datasets.sequence_folders_clr import SequenceFolder
import torch.nn.functional as F
import pandas as pd


parser = argparse.ArgumentParser(description='Script for testing depth predictions with the corresponding ground truth',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('data', metavar='DIR', help='path to dataset')
parser.add_argument('--dataset', type=str, choices=[
                    'hand', 'robotcar', 'radiate'], default='radiate', help='the dataset to train')
parser.add_argument('--with-preprocessed', type=int, default=1,
                    help='use the preprocessed undistorted images')
parser.add_argument('--with-testfile', type=int, default=1,
                    help='use the test.txt file containing test sequences')
parser.add_argument("--nsamples", default=2000, type=int,
                    help="Number of samples to subsample from each scene.")
parser.add_argument('--with-timing', type=int, default=0,
                    help='use the timing benchmark to evaluate the runtime speed')
parser.add_argument("--pretrained-disp", required=True,
                    type=str, help="pretrained DispNet path")
parser.add_argument('-j', '--workers', default=32, type=int,
                    metavar='N', help='number of data loading workers')
parser.add_argument('-b', '--batch-size', default=4,
                    type=int, metavar='N', help='mini-batch size')
parser.add_argument('--seed', default=3407, type=int,
                    help='seed for random functions, and network initialization')
parser.add_argument("--img-height", default=192, type=int, help="Image height")
parser.add_argument("--img-width", default=320, type=int, help="Image width")
parser.add_argument('--img-norm', action='store_true',
                    help='weather to perform the norm to img input')
parser.add_argument('--radar-pov-vertical-aug', action='store_true',
                    help='weather to perform the vertical aug to radar input')
parser.add_argument('--radar-channels', default=1, type=int,
                    help='radar channels num')
# parser.add_argument("--min-depth", default=1e-3)
# parser.add_argument("--max-depth", default=80)
# parser.add_argument("--dataset-dir", default='.', type=str, help="Dataset directory")
# parser.add_argument("--dataset-list", default=None, type=str, help="Dataset list file")
# parser.add_argument("--output-dir", default=None, required=True, type=str, help="Output directory for saving predictions in a big 3D numpy file")
parser.add_argument('--results-dir', default='results', metavar='PATH',
                    help='directory where to save predicted depth maps and stats')
parser.add_argument('--resnet-layers',  type=int, default=18,
                    choices=[18, 50], help='number of ResNet layers for depth estimation')

device = torch.device(
    "cuda") if torch.cuda.is_available() else torch.device("cpu")

def compute_errors(gt, pred):
    """Computation of error metrics between predicted and ground truth depths
    """
    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25     ).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()

    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())

    rmse_log = (np.log(gt) - np.log(pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())

    abs_rel = np.mean(np.abs(gt - pred) / gt)

    sq_rel = np.mean(((gt - pred) ** 2) / gt)

    return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3


MIN_DEPTH = 1e-3
MAX_DEPTH = 80

@torch.no_grad()
def main():
    args = parser.parse_args()

    results_dir = Path(args.results_dir)  # /args.sequence
    results_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    cudnn.deterministic = True
    cudnn.benchmark = True


    if args.with_testfile:
        root = Path(args.data)
        scene_list_path = root/'test_dep.txt'
        scenes = [root/folder.strip()/'stereo_undistorted/left'
                  for folder in open(scene_list_path) if not folder.strip().startswith("#")]
        mono_frames_files  = [root/folder.strip()/'zed_left.txt'
                       for folder in open(scene_list_path) if not folder.strip().startswith("#")]
        scene_names = [folder.strip()
                       for folder in open(scene_list_path) if not folder.strip().startswith("#")]
    else:
        raise NotImplementedError

    print("=> creating model")
    disp_net = models.DispResNet(
        args.resnet_layers, False).to(device)

    print("=> using pre-trained weights for DispResNet")

    weights = torch.load(args.pretrained_disp)
    disp_net.load_state_dict(weights['state_dict'], strict=False)

    disp_net.eval()

    columns = ["scene_name", "abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3", "samples"]
    results_df = pd.DataFrame(columns=columns)
    save_interval = 10
    for scene_name, scene, mono_frames_file in zip(scene_names, scenes, mono_frames_files):
        print("=> Processing:", scene_name)

        results_depth_dir = results_dir/scene_name/'depth'
        results_depth_dir.mkdir(parents=True, exist_ok=True)

    
        print("img norm : {}".format(args.img_norm))
        test_set = SequenceFolder(
            args.data,
            dataset='radiate', 
            seed=3407, mode='test',
            sequence=scene_name,
            img_aug=False,
            img_norm=args.img_norm,
            radar_pov_vertical_aug=args.radar_pov_vertical_aug,
            radar_slices=args.radar_channels > 1,
            depth_gt_dir='depth_ac'
        )
        # ImageFolder(
        #     path=scene, depth_gt_path=scene_depth_gt, mono_frames_file = mono_frames_file, transform=valid_transform, nsamples=args.nsamples)
        nframes = len(test_set)
        print('{} samples found in {} '.format(
            nframes, scene_name))
        test_loader = torch.utils.data.DataLoader(dataset=test_set,
                                                  batch_size=args.batch_size,
                                                  shuffle=False,
                                                  num_workers=args.workers,
                                                  pin_memory=True)

        avg_time = 0
        errors = []
        ratios = []
        for i, inputs in tqdm(enumerate(test_loader)):
            # tgt_img = tgt_img.to(device)
            # tgt_depth_gt = tgt_depth_gt.to(device)
            inputs = {key: (value.to("cuda") if not isinstance(value, list) else value) for key, value in inputs.items()}
            if args.with_timing:
                # compute speed
                torch.cuda.synchronize()
                t_start = time.time()


            tgt_depth_gt = inputs['tgt_depth_gt']
            tgt_img = inputs['cam_tgt_img']
    

            tgt_depth = [disp_to_depth(disp) for disp in disp_net(tgt_img)]
        
            # tgt_depth = [disp_to_depth(disp) for disp in disp_net(tgt_img)]

            if args.with_timing:
                torch.cuda.synchronize()
                elapsed_time = time.time() - t_start
                avg_time += elapsed_time


            # 0. reisze the pred to gt
            # 1. mask the gt
            # 2. scale_factor and median_scaling
            # 3. clamp the pred by min and max depth
            for j, (img, depth, depth_gt) in enumerate(zip(inputs['cam_tgt_img'], tgt_depth[0], tgt_depth_gt)):
                if (i * args.batch_size + j) % save_interval == 0:
                    out_file_path = results_depth_dir / f'{i * args.batch_size + j:06d}.png'
                    
                    # Convert the depth prediction to a color image
                    colour_depth = tensor2array(depth, max_value=None, colormap='inferno')
                    colour_depth = colour_depth[:3, :, :]
                    colour_depth = colour_depth.transpose(1, 2, 0) * 255
                    colour_depth = colour_depth.astype(np.uint8)
                    depth_img = Image.fromarray(colour_depth)

                    # Resize and normalize the ground truth depth for visualization
                    depth_gt_resized = F.interpolate(depth_gt.unsqueeze(0), (depth.size(-2), depth.size(-1)), mode='nearest').squeeze()
                    colour_depth_gt = tensor2array(depth_gt_resized, max_value=None, colormap='inferno')
                    colour_depth_gt = colour_depth_gt[:3, :, :]
                    colour_depth_gt = colour_depth_gt.transpose(1, 2, 0) * 255
                    colour_depth_gt = colour_depth_gt.astype(np.uint8)

                    # Convert the original image to uint8 and create an Image object
                    img_uint8 = img.mul(255).byte().cpu().numpy().transpose(1, 2, 0)
                    img_pil = Image.fromarray(img_uint8)

                    # Overlay the ground truth depth on the camera image
                    img_overlay = np.copy(img_uint8)
                    # non_zero_mask = np.any(depth_gt_resized.squeeze().cpu().numpy() > 0, axis=0)
                    # img_overlay[non_zero_mask] = colour_depth_gt[non_zero_mask]
                    img_overlay[np.nonzero(depth_gt_resized.cpu().numpy())]  =  colour_depth_gt[np.nonzero(depth_gt_resized.cpu().numpy())]
                    overlay_img = Image.fromarray(img_overlay.astype(np.uint8))

                    # Concatenate the overlay with the predicted depth for comparison
                    combined_img = Image.new('RGB', (img_pil.width * 2, img_pil.height))
                    combined_img.paste(overlay_img, (0, 0))
                    combined_img.paste(depth_img, (img_pil.width, 0))

                    # Save the combined image
                    combined_img.save(out_file_path)
                
                
                depth = F.interpolate(depth.unsqueeze(0), (depth_gt.size(-2), depth_gt.size(-1)), mode='nearest').squeeze(0)
                mask = depth_gt > 0
                depth = depth[mask]
                depth_gt = depth_gt[mask]
                
                ratio = depth_gt.mean() / depth.mean()
                ratios.append(ratio)
                # print(ratio)
                # ratios.append(ratio)
                depth *= ratio
                depth = torch.clamp(depth, MIN_DEPTH, MAX_DEPTH)
                errors.append(compute_errors(depth_gt.cpu().numpy(), depth.cpu().numpy()))

                
                
                # colour_depth = utils.tensor2array(
                #     depth, max_value=None, colormap='inferno')
                # colour_depth = colour_depth.transpose(1, 2, 0)*255
                # colour_depth = colour_depth.astype(np.uint8)
                # im = Image.fromarray(colour_depth)
                # # im.save(results_depth_dir /
                # #         '{0:03d}.png'.format(i*args.batch_size+j))
                # im.save(results_depth_dir / f_name)
                
                

        mean_errors = np.array(errors).mean(0)
        print(scene_name)
        # print("samples: {}".format(len(mean_errors)))
        print("\n  " + ("{:>8} | " * 7).format("abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3"))
        print(("&{: 8.3f}  " * 7).format(*mean_errors.tolist()) + "\\\\")
        print("\n-> Done!")
        
        scene_results = [scene_name] + mean_errors.tolist() + [nframes]
        results_df.loc[len(results_df)] = scene_results  # 将结果添加到 DataFrame
        
        if args.with_timing:
            avg_time /= nframes
            print('Avg Time: ', avg_time, ' seconds.')
            print('Avg Speed: ', 1.0 / avg_time, ' fps')

    results_df.to_csv('{}_results.csv'.format(args.pretrained_disp), index=False)


def save_depth(depth:torch.Tensor, path:str):
    colour_depth = tensor2array(depth, max_value=None, colormap='inferno')
    colour_depth = colour_depth.transpose(1, 2, 0) * 255
    colour_depth = colour_depth.astype(np.uint8)
    im = Image.fromarray(colour_depth)
    im.save(path)
    return

def disp_to_depth(disp):
    # depth_scale = 10.0
    # id_disp = torch.rand(disp.shape).to(device)*1e-12
    # disp = disp + id_disp
    depth = 1./disp
    # depth = depth/depth_scale
    depth = depth.clamp(min=1e-6)
    return depth


if __name__ == '__main__':
    with torch.cuda.amp.autocast(enabled=False):
        main()
