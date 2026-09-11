import numpy as np
import torch
import h5py
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import random
import warnings

warnings.filterwarnings("ignore")


class TrainDataset_All_withNeighborIds(torch.utils.data.Dataset):
    def __init__(self, X, Y, Miss_list, Neighbor_list, Idxs):
        self.X = X
        self.Y = Y
        self.Miss_list = Miss_list
        self.Neighbor_list = Neighbor_list
        self.Idxs = Idxs
        self.view_size = len(X)

    def __getitem__(self, index):
        cur_idx = self.Idxs[index]
        cur_xs_list = [self.X[i][index] for i in range(self.view_size)]
        cur_ys_list = [self.Y[i][index] for i in range(self.view_size)]
        cur_miss_list = [self.Miss_list[i][index] for i in range(self.view_size)]
        cur_neighbor_list = [self.Neighbor_list[i][index] for i in range(self.view_size)]

        return cur_xs_list, cur_ys_list, cur_miss_list, cur_neighbor_list, cur_idx

    def __len__(self):
        return self.X[0].shape[0]


class TrainDataset_All(torch.utils.data.Dataset):
    def __init__(self, X, Y, Miss_list, Idxs):
        self.X = X
        self.Y = Y
        self.Miss_list = Miss_list
        self.Idxs = Idxs
        self.view_size = len(X)

    def __getitem__(self, index):
        return [self.X[i][index] for i in range(self.view_size)], \
               [self.Y[i][index] for i in range(self.view_size)], \
               [self.Miss_list[i][index] for i in range(self.view_size)], \
               self.Idxs[index]

    def __len__(self):
        return self.X[0].shape[0]


def get_mask_unbalance(view_num, data_len, missing_rate):
    if view_num == 2:
        miss_list = [0.0, 1.0]
    elif view_num == 3:
        miss_list = [0.0, 0.5, 1.0]
    elif view_num == 4:
        miss_list = [0.0, 0.25, 0.75, 1.0]
    elif view_num == 5:
        miss_list = [0.0, 0.25, 0.5, 0.75, 1.0]
    elif view_num == 6:
        miss_list = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

    miss_mat = np.ones((data_len, view_num))

    num_incomplete = int(data_len * missing_rate)

    if num_incomplete == 0:
        return torch.tensor(miss_mat, dtype=torch.int)

    incomplete_sample_indices = np.random.choice(data_len, size=num_incomplete, replace=False)

    max_allowed_present = np.round(num_incomplete * (1 - np.array(miss_list))).astype(int)
    pool = np.concatenate([np.full(quota, i) for i, quota in enumerate(max_allowed_present)])
    np.random.shuffle(pool)
    kept_views_for_samples = pool[:num_incomplete]

    for idx, sample_idx in enumerate(incomplete_sample_indices):
        kept_view = kept_views_for_samples[idx]
        for v in range(view_num):
            miss_prob = miss_list[v]

            if v == kept_view:
                miss_mat[sample_idx, v] = 1
            else:
                if np.random.random() < miss_prob:
                    miss_mat[sample_idx, v] = 0
                else:
                    miss_mat[sample_idx, v] = 1

    for idx, sample_idx in enumerate(incomplete_sample_indices):
        if miss_mat[sample_idx].sum() == 0:
            kept_view = kept_views_for_samples[idx]
            miss_mat[sample_idx, kept_view] = 1

    miss_mat = torch.tensor(miss_mat, dtype=torch.int)
    return miss_mat


def load_data(data_name, missrate):
    path = 'D:/MultiView Dataset/'
    data = h5py.File(path + data_name + ".mat")
    X, Y = [], []
    Label = np.array(data['Y']).T
    Label = Label.reshape(Label.shape[0])

    Label = Label.astype(int)
    if Label.min() > 0:
        Label = Label - Label.min()

    mm = MinMaxScaler()

    for i in range(data['X'].shape[1]):
        diff_view = data[data['X'][0, i]]
        diff_view = np.array(diff_view, dtype=np.float32).T
        std_view = mm.fit_transform(diff_view)
        X.append(std_view)
        Y.append(Label)

    input_dims = []
    for i in range(len(X)):
        input_dims.append(X[i].shape[1])

    unique = np.unique(Y[0])
    cluster_num = np.size(unique, axis=0)
    data_num = len(Y[0])
    view_num = len(X)
    view_dims = input_dims

    index = [i for i in range(data_num)]
    np.random.shuffle(index)
    for v in range(view_num):
        X[v] = X[v][index]
        Y[v] = Y[v][index]


    Miss_mat = get_mask_unbalance(view_num, data_num, missrate)
    Miss_vecs = [row for row in Miss_mat.T]

    return X, Y, Miss_vecs, cluster_num, data_num, view_num, view_dims




