import os
import torch
import numpy as np

class LuxorDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, first_idx, after_last_index):
        self.root_dir = root_dir
        self.first_idx = first_idx
        self.after_last_index = after_last_index

        self.xs = []
        for idx in range(self.first_idx, self.after_last_index):
            x_filename = '0' * (3 - len(str(idx))) + str(idx) + ".in"
            xs = np.loadtxt(os.path.join(self.root_dir, x_filename))
            # split a line into 6 letters
            xs = np.split(xs, 6, axis=1)
            self.xs.extend(xs)

        self.ys = []
        with open(os.path.join(self.root_dir, "sample.out")) as f:
            for line_idx, line in enumerate(f, 1):
                if line_idx < first_idx:
                    continue
                if line_idx == after_last_index:
                    break
                for i in range(6): # for each letter in line
                    self.ys.append(ord(line[i]) - ord('a'))

    def __len__(self):
        return 6 * (self.after_last_index - self.first_idx)

    def __getitem__(self, i):
        x = torch.FloatTensor(self.xs[i])
        y = []
        if i < len(self.ys):
            y = self.ys[i]
        return {'x': x, 'y': y}
