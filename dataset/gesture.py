from torch.utils.data import Dataset, DataLoader
from blob import load_np_array
from scipy.io import savemat
import numpy as np
import torch
from tqdm import tqdm
import os


class GestureDataset:

    def __init__(self, root_dir, modality):
        self.holdouts = {}
        for holdout in sorted(os.listdir(root_dir)):
            if holdout.startswith(modality):
                holdout_abs_path = os.path.join(root_dir, holdout)
                if os.path.isdir(holdout_abs_path):
                    tokens = holdout.split('_')
                    if tokens[1] == tokens[2]:
                        holdout = []
                        for test_sj in tqdm(sorted(os.listdir(holdout_abs_path))):
                            train_test_split = {'X_train': [], 'y_train': [], 'X_test': [], 'y_test': []}
                            test_sj_abs_path = os.path.join(holdout_abs_path, test_sj, 'feature', 'iter_800')
                            for train_sj in sorted(os.listdir(test_sj_abs_path)):
                                train_sj_abs_path = os.path.join(test_sj_abs_path, train_sj)
                                for cl in sorted(os.listdir(train_sj_abs_path), key=lambda x: '{:02d}'.format(int(x))):
                                    label = int(cl) - 1
                                    cl_abs_path = os.path.join(train_sj_abs_path, cl)
                                    for repeat in sorted(os.listdir(cl_abs_path)):
                                        sample_file = os.path.join(cl_abs_path, repeat, '000001.fc6')
                                        if train_sj == test_sj:
                                            train_test_split['X_test'].append(sample_file)
                                            train_test_split['y_test'].append(label)
                                        else:
                                            train_test_split['X_train'].append(sample_file)
                                            train_test_split['y_train'].append(label)
                            holdout.append(self.load_data(train_test_split))
                        self.holdouts.update({tokens[1]: holdout})
        print(len(self.holdouts))

    def load_data(self, train_test_split):
        train_test_split = train_test_split.copy()
        train_test_split['X_train'] = np.array(list(map(lambda x: load_np_array(x), train_test_split['X_train'])), dtype=np.float)
        train_test_split['X_test'] = np.array(list(map(lambda x: load_np_array(x), train_test_split['X_test'])), dtype=np.float)
        train_test_split['y_train'] = np.array(train_test_split['y_train']).astype(np.long)
        train_test_split['y_test'] = np.array(train_test_split['y_test']).astype(np.long)
        return train_test_split


if __name__ == '__main__':
    dataset = GestureDataset(root_dir='/home/inspiros/Documents/mica/datasets/MultiviewGestureCompact',
                             modality='RGB')
    savemat('gesture.mat', dataset.holdouts)
