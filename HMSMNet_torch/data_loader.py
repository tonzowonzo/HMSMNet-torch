import os
import random
import numpy as np
import torch
import cv2
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import scipy.signal as sig


kx = np.array([[-1, 0, 1]])
ky = np.array([[-1], [0], [1]])


class MyDataset(Dataset):
    def __init__(self, root_dir, crop_size, min_disp, max_disp):
        self.root_dir = root_dir
        self.crop_size = crop_size
        self.min_disp = min_disp
        self.max_disp = max_disp
        self.left_paths = sorted([os.path.join(root_dir, 'left', f) for f in os.listdir(os.path.join(root_dir, 'left'))])
        self.right_paths = sorted([os.path.join(root_dir, 'right', f) for f in os.listdir(os.path.join(root_dir, 'right'))])

    def __len__(self):
        return len(self.left_paths)

    def __getitem__(self, idx):
        left_path = self.left_paths[idx]
        right_path = self.right_paths[idx]

        # Load left and right images
        left_img = Image.open(left_path)
        right_img = Image.open(right_path)

        # Normalize the input images
        left_img = np.array(left_img).astype(np.float32) / 500.0 - 1.0
        right_img = np.array(right_img).astype(np.float32) / 500.0 - 1.0

        # Compute the gradients of the left image
        bdx, bdy = sig.convolve2d(left_img[:, :], kx, 'same'), sig.convolve2d(left_img[:, :], ky, 'same')
        left_img = left_img.astype('float32') / 500.0 - 1.0
        dx = bdx.astype('float32') / 500.0
        dy = bdy.astype('float32') / 500.0
    
        # Compute the ground truth disparity map
        disp = Image.open(left_path.replace('left', 'disparity'))

        # Randomly flip the input images horizontally
        if random.random() > 0.5:
            left_img = np.fliplr(left_img)
            right_img = np.fliplr(right_img)
            disp = np.fliplr(disp)

        # Convert numpy arrays to PyTorch tensors
        left_tensor = torch.from_numpy(np.expand_dims(left_img, axis=0).copy())
        right_tensor = torch.from_numpy(np.expand_dims(right_img, axis=0).copy())
        gx_tensor = torch.from_numpy(np.expand_dims(np.transpose(dx, (1, 0)), axis=0))
        gy_tensor = torch.from_numpy(np.expand_dims(np.transpose(dy, (1, 0)), axis=0))
        disp_tensor = torch.from_numpy(np.expand_dims(disp, axis=0).copy())

        return (left_tensor, right_tensor, gx_tensor, gy_tensor), disp_tensor


if __name__ == "__main__":
    train_dataset = MyDataset(root_dir='/mnt/d/whu_stereo/experimental_data/with_ground_truth/train', crop_size=512, min_disp=-128.0, max_disp=64.0)
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
