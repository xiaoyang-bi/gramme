from radar_eval.eval_odometry import EvalOdom
from radar_eval.eval_utils import RadarEvalOdom, getTraj
from datasets.sequence_folders import SequenceFolder
import custom_transforms
import models
import utils

import argparse
import time
from pathlib import Path

import torch
from tqdm import tqdm


parser = argparse.ArgumentParser(description='Script for visualizing depth map and masks',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('data', metavar='DIR', help='path to dataset')
parser.add_argument('--sequence-length', type=int, metavar='N',
                    help='sequence length for training', default=3)
parser.add_argument('--skip-frames', type=int, metavar='N',
                    help='gap between frames', default=5)
parser.add_argument('-j', '--workers', default=4, type=int,
                    metavar='N', help='number of data loading workers')
parser.add_argument('-b', '--batch-size', default=4,
                    type=int, metavar='N', help='mini-batch size')
parser.add_argument('--seed', default=0, type=int,
                    help='seed for random functions, and network initialization')
parser.add_argument('--results-dir', default='results', metavar='PATH',
                    help='directory where to save predicted trajectories and stats')
parser.add_argument('--resnet-layers',  type=int, default=18,
                    choices=[18, 50], help='number of ResNet layers for depth estimation')
parser.add_argument('--dataset', type=str, choices=[
                    'hand', 'robotcar', 'radiate'], default='robotcar', help='the dataset to train')
parser.add_argument("--sequence", default='',
                    type=str, help="sequence to test")
parser.add_argument('--with-testfile', type=int, default=0,
                    help='use the test.txt file containing test sequences')
parser.add_argument('--cam-mode', type=str, choices=[
                    'mono', 'stereo'], default='stereo', help='the dataset to train')
# parser.add_argument('--pretrained-disp', dest='pretrained_disp', default=None, metavar='PATH', help='path to pre-trained dispnet model')
parser.add_argument('--pretrained-pose', required=True, dest='pretrained_pose',
                    metavar='PATH', help='path to pre-trained Pose net model')
parser.add_argument('--with-gt', action='store_true',
                    help='Evaluate with ground-truth')
# parser.add_argument('--gt-file', metavar='DIR', help='path to ground truth validation file')
parser.add_argument('--radar-format', type=str,
                    choices=['cartesian', 'polar'], default='polar', help='Range-angle format')
parser.add_argument('--range-res', type=float,
                    help='Range resolution of FMCW radar in meters', metavar='W', default=0.0977)
parser.add_argument('--angle-res', type=float,
                    help='Angular azimuth resolution of FMCW radar in radians', metavar='W', default=1.0)
parser.add_argument('--cart-res', type=float,
                    help='Cartesian resolution of FMCW radar in meters/pixel', metavar='W', default=0.25)
parser.add_argument('--cart-pixels', type=int,
                    help='Cartesian size in pixels (used for both height and width)', metavar='W', default=501)

device = torch.device(
    "cuda") if torch.cuda.is_available() else torch.device("cpu")


@torch.no_grad()
def main():
    global device
    args = parser.parse_args()

    # results_dir = Path(args.results_dir)/args.sequence
    # results_dir.mkdir(parents=True)

    print("=> fetching scenes in '{}'".format(args.data))
    ds_transform = custom_transforms.Compose(
        [custom_transforms.ArrayToTensor()])

    ro_params = {
        'cart_resolution': args.cart_res,
        'cart_pixels': args.cart_pixels,
        'rangeResolutionsInMeter': args.range_res,
        'angleResolutionInRad': args.angle_res,
        'radar_format': args.radar_format
    }

    # create model
    print("=> creating model")
    weights_pose = torch.load(args.pretrained_pose)
    pose_net = models.PoseResNet(
        args.dataset, args.resnet_layers, False).to(device)
    pose_net.load_state_dict(weights_pose['state_dict'], strict=False)
    pose_net = torch.nn.DataParallel(pose_net)
    pose_net.eval()

    if args.with_testfile:
        root = Path(args.data)
        scene_list_path = root/'test.txt'
        sequences = [folder.strip()
                     for folder in open(scene_list_path) if not folder.strip().startswith("#")]
    elif args.sequence:
        sequences = [args.sequence]
    else:
        raise argparse.ArgumentError(
            'Neither a test file nor a sequence name given!')

    for sequence in sequences:
        results_dir = Path(args.results_dir)/sequence
        results_dir.mkdir(parents=True)

        print("=> fetching scenes in '{}'".format(Path(args.data)/sequence))
        val_set = SequenceFolder(
            args.data,
            transform=ds_transform,
            seed=args.seed,
            train=False,
            sequence_length=args.sequence_length,
            skip_frames=args.skip_frames,
            dataset=args.dataset,
            ro_params=ro_params,
            sequence=sequence,
            radar_format=args.radar_format
        )

        val_loader = torch.utils.data.DataLoader(
            val_set, batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True)

        nframes = len(val_set)
        print('{} samples found in {} valid scenes'.format(
            len(val_set), len(val_set.scenes)))

        all_poses = []
        all_inv_poses = []

        t_del = 0
        for i, (tgt_img, ref_imgs) in tqdm(enumerate(val_loader)):
            #(tgt_img, ref_imgs) = val_it.next()

            tgt_img = tgt_img.to(device)
            ref_imgs = [img.to(device) for img in ref_imgs]

            torch.cuda.synchronize()
            inf_t0 = time.time()
            poses, poses_inv = compute_pose_with_inv(
                pose_net, tgt_img, ref_imgs)
            torch.cuda.synchronize()
            t_del += time.time() - inf_t0

            all_poses.append(poses)
            all_inv_poses.append(poses_inv)

        # Total time for forward and backward poses
        print(
            'Average time for inference: pair of frames/{:.2f}sec'.format(1./(t_del/(nframes*2))))

        if args.with_gt:
            print("=> converting odometry predictions to trajectory")
            gt_file = Path(args.data, sequence, 'gt', 'radar_odometry.csv')
            ro_eval = RadarEvalOdom(gt_file, args.dataset)
            ate_f, f_pred_xyz, f_pred = ro_eval.eval_ref_poses(all_poses, all_inv_poses,
                                                               args.skip_frames)
            # save_traj_plots_with_gt(results_dir, f_pred_xyz, ro_eval.gt)
            print("=> evaluating the trajectory")
            isPartialSequence = 'partial' in args.sequence
            odom_eval = EvalOdom(isPartial=isPartialSequence)
            odom_eval.eval(f_pred.cpu().numpy(),
                           ro_eval.gt.cpu().numpy(), results_dir)
        else:
            b_pred_xyz, f_pred_xyz = getTraj(
                all_poses, all_inv_poses, args.skip_frames)
            utils.save_traj_plots(results_dir, f_pred_xyz, b_pred_xyz)


def compute_pose_with_inv(pose_net, tgt_img, ref_imgs):
    poses = []
    poses_inv = []
    for ref_img in ref_imgs:
        poses.append(pose_net(tgt_img, ref_img))
        poses_inv.append(pose_net(ref_img, tgt_img))

    return torch.stack(poses), torch.stack(poses_inv)


if __name__ == '__main__':
    main()
