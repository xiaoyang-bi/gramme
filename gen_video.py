import sys
from pathlib import Path
import cv2
import numpy as np

def read_timestamps(timestamp_file):
    timestamps = {}
    with open(timestamp_file, 'r') as file:
        for line in file:
            parts = line.strip().split()
            if parts[0] == 'Frame:':
                frame_id = parts[1]
                timestamp = float(parts[3])
                timestamps[frame_id] = timestamp
    return timestamps

def create_video(img_dir, depth_dir, timestamps, output_file='output_video.mp4'):
    # Get the list of depth files and sort them
    depth_files = sorted(depth_dir.glob('*.png'))
    if not depth_files:
        print("No depth images found.")
        return

    # Read the first image to get the dimensions
    sample_img_path = img_dir / depth_files[1].name
    sample_img = cv2.imread(str(sample_img_path))
    height, width, _ = sample_img.shape
    frame_size = (width * 2, height)  # Width doubled for concatenation
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_file, fourcc, 20.0, frame_size)
    
    for depth_file in depth_files:
        frame_id = int(depth_file.stem) + 1 # Get the frame ID from the filename without extension
        # timestamp = timestamps.get(frame_id)
        
        # if timestamp is None:
        #     print(f"No timestamp found for frame {frame_id}")
        #     continue

        img_file = img_dir / f"{frame_id:06d}.png"
        if not img_file.exists():
            print(f"Image file {img_file} not found.")
            continue

        img = cv2.imread(str(img_file))
        depth_img = cv2.imread(str(depth_file))
        
        combined_img = np.concatenate((img, depth_img), axis=1)
        video_writer.write(combined_img)
        # print(f"Added frame {frame_id} with timestamp {timestamp} to the video.")
    
    video_writer.release()
    print(f"Video saved as {output_file}")


if __name__ == '__main__':
    depth_dir = Path(sys.argv[1])
    img_root_dir = Path(sys.argv[2])
    
    img_dir = img_root_dir / 'stereo_undistorted' / 'left'
    img_timestampes_file = img_root_dir / 'zed_left.txt'
    
    # Read timestamps
    timestamps = read_timestamps(img_timestampes_file)
    
    # Create video from images and depth maps
    output_name = depth_dir.parent.parent.name + '_' + depth_dir.parent.name + '.mp4'
    print(output_name)
    create_video(img_dir, depth_dir, timestamps, output_file=output_name)
