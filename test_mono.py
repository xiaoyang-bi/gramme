from radar_eval.eval_utils import getTraj
from datasets.sequence_folders_mono import SequenceFolder
import custom_transforms_mono as T
import utils
from inverse_warp_vo import MonoWarper
import models

import argparse
import time
from pathlib import Path

import torch
from tqdm import tqdm


parser = argparse.ArgumentParser(description='Script for visualizing depth map and masks',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('data', metavar='DIR', help='path to dataset')
# parser.add_argument('--pretrained-disp', required=True, dest='pretrained_disp',
#                     default=None, metavar='PATH', help='path to pre-trained dispnet model')
parser.add_argument('--pretrained-pose', required=True, dest='pretrained_pose',
                    metavar='PATH', help='path to pre-trained Pose net model')
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
parser.add_argument('--img-height', type=int,
                    help='resized image height', metavar='W', default=192)
parser.add_argument('--img-width', type=int,
                    help='resized image width', metavar='W', default=320)
parser.add_argument('--dataset', type=str, choices=[
                    'hand', 'robotcar', 'radiate'], default='hand', help='the dataset to train')
parser.add_argument("--sequence", default='2019-01-10-14-36-48-radar-oxford-10k-partial',
                    type=str, help="sequence to test")
parser.add_argument('--with-gt', action='store_true',
                    help='Evaluate with ground-truth')

device = torch.device(
    "cuda") if torch.cuda.is_available() else torch.device("cpu")


@torch.no_grad()
def main():
    global device
    args = parser.parse_args()

    results_dir = Path(args.results_dir)/args.sequence
    results_dir.mkdir(parents=True)

    imagenet_mean = utils.imagenet_mean
    imagenet_std = utils.imagenet_std
    img_size = (args.img_height, args.img_width)

    valid_transform = T.Compose([
        T.ToPILImage(),
        T.CropBottom(),
        T.Resize(img_size),
        T.ToTensor(),
        T.Normalize(imagenet_mean, imagenet_std)
    ])

    print("=> fetching scenes in '{}'".format(args.data))
    val_set = SequenceFolder(
        args.data,
        transform=valid_transform,
        seed=args.seed,
        train=False,
        sequence_length=args.sequence_length
    )

    nframes = len(val_set)
    print('{} samples found in {} valid scenes'.format(
        len(val_set), len(val_set.scenes)))

    val_loader = torch.utils.data.DataLoader(
        val_set, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True
    )

    # create model
    # print("=> creating loss object")
    # mono_warper = MonoWarper(
    #     max_scales=args.num_scales,
    #     dataset=args.dataset,
    #     # batch_size=args.batch_size,
    #     padding_mode=args.padding_mode)

    # create model
    print("=> creating model")
    # disp_net = models.DispResNet(
    #     args.resnet_layers, False).to(device)
    pose_net = models.PoseResNetMono(18, False).to(device)

    # load parameters
    # print("=> using pre-trained weights for DispResNet")
    # weights = torch.load(args.pretrained_disp)
    # disp_net.load_state_dict(weights['state_dict'], strict=False)

    print("=> using pre-trained weights for PoseResNet")
    weights = torch.load(args.pretrained_pose)
    pose_net.load_state_dict(weights['state_dict'], strict=False)

    # disp_net = torch.nn.DataParallel(disp_net)
    pose_net = torch.nn.DataParallel(pose_net)

    pose_net.eval()
    # disp_net.eval()

    all_poses = []
    all_inv_poses = []

    t_del = 0
    for i, (tgt_img, ref_imgs, intrinsics) in tqdm(enumerate(val_loader)):
        tgt_img = tgt_img.to(device)
        ref_imgs = [img.to(device) for img in ref_imgs]
        intrinsics = intrinsics.to(device)

        torch.cuda.synchronize()
        inf_t0 = time.time()
        poses, poses_inv = compute_pose_with_inv(pose_net, tgt_img, ref_imgs)
        torch.cuda.synchronize()
        t_del += time.time() - inf_t0

        all_poses.append(torch.stack(poses))
        all_inv_poses.append(torch.stack(poses_inv))

    # Total time for forward and backward poses
    print(
        'Average time for inference: pair of frames/{:.2f}sec'.format(1./(t_del/(nframes*2))))

    if args.with_gt:
        print('Mono evaluation with GT is not supported yet!')
        # ro_eval = RadarEvalOdom(args.gt_file, args.dataset)

        # ate_bs_mean, ate_bs_std, ate_fs_mean, ate_fs_std, f_pred_xyz, f_pred = ro_eval.eval_ref_poses(
        #     all_poses, all_inv_poses, args.skip_frames)

        # if log_outputs:
        #     # Plot and log aligned trajectory
        #     fig = utils.traj2Fig_withgt(
        #         f_pred_xyz.squeeze(), ro_eval.gt[:, :3, 3].squeeze())
        #     # fig2= utils.traj2Fig(f_pred[:,:3,3])
        #     val_writer.add_figure('val/fig/traj_aligned_pred', fig, epoch)
        #     # output_writers[0].add_figure('val/fig/traj_pred_full_aligned', fig2, epoch)

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

    return poses, poses_inv


if __name__ == '__main__':

    main()