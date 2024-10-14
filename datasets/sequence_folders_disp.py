import torch.utils.data as data
from pathlib import Path

from PIL import Image


class ImageFolder(data.Dataset):
    def __init__(self, path, depth_gt_path, mono_frames_file, transform=None, nsamples=0):
        self.path = path
        self.image_paths = sorted(list(self.path.glob('*.png')))
        if nsamples > 0 and nsamples < len(self.image_paths):
            skip = len(self.image_paths)//nsamples
            # idx = list(range(0, skip*nsamples, skip))
            self.image_paths = self.image_paths[0:skip*nsamples:skip]
        self.transform = transform
        self.depth_gt_path = depth_gt_path
        self.depth_paths = sorted(list(self.depth_gt_path.glob('*.tiff')))
        self.depth_timestamps = [float(filename.name.split('_depth')[0]) for filename in self.depth_paths]
        self.mono_frames = {}
        with open(mono_frames_file, 'r') as file:
            for line in file:
                parts = line.split()
                frame = parts[1]  # Frame number
                time = parts[3]   # Time value
                self.mono_frames[frame] = float(time)

    def __getitem__(self, index):
        x = Image.open(self.image_paths[index])
        if self.transform:
            x = self.transform(x)
        
        image_idx = self.image_paths[index].name
        mono_timestamp = self.mono_frames[image_idx[:-4]]
        depth_timestamp = min(self.depth_timestamps, key=lambda x: abs(mono_timestamp - x))
        depth_file = self.depth_gt_path/(str(depth_timestamp) + "_depth.tiff")
        depth_gt = self.transform(Image.open(depth_file))
        return x, depth_gt, self.image_paths[index].name

    def __len__(self):
        return len(self.image_paths)
  