import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image

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

        # Randomly crop the input images
        w, h = left_img.size
        x = random.randint(0, w - self.crop_size)
        y = random.randint(0, h - self.crop_size)
        left_img = left_img.crop((x, y, x+self.crop_size, y+self.crop_size))
        right_img = right_img.crop((x, y, x+self.crop_size, y+self.crop_size))

        # Normalize the input images
        left_img = np.array(left_img).astype(np.float32) / 255.0
        right_img = np.array(right_img).astype(np.float32) / 255.0

        # Compute the gradients of the left image
        gy, gx = np.gradient(left_img)

        # Compute the ground truth disparity map
        disp = np.load(left_path.replace('left', 'disp'))

        # Randomly flip the input images horizontally
        if random.random() > 0.5:
            left_img = np.fliplr(left_img)
            right_img = np.fliplr(right_img)
            disp = np.fliplr(disp)

        # Convert numpy arrays to PyTorch tensors
        left_tensor = torch.from_numpy(np.transpose(left_img, (2, 0, 1)))
        right_tensor = torch.from_numpy(np.transpose(right_img, (2, 0, 1)))
        gx_tensor = torch.from_numpy(np.expand_dims(np.transpose(gx, (1, 0)), axis=0))
        gy_tensor = torch.from_numpy(np.expand_dims(np.transpose(gy, (1, 0)), axis=0))
        disp_tensor = torch.from_numpy(np.expand_dims(disp, axis=0))

        return (left_tensor, right_tensor, gx_tensor, gy_tensor), disp_tensor


if __name__ == "__main__":
    train_dataset = MyDataset(root_dir='/mnt/d/whu_stereo/experimental_data/with_ground_truth/train', crop_size=512, min_disp=-128.0, max_disp=64.0)
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)

    print(next(train_loader))
