import os
import random
import sys
from pathlib import Path
import numpy as np
import scipy.io as sio
from scipy import sparse
from sklearn.model_selection import train_test_split
from pcdc_imvc.utils import util
import h5py

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = PROJECT_ROOT / 'data'

# ------修改部分-----------------------------------------------------------------------------------------------------
def _augment_view(x, drop_prob=0.0, noise_std=0.0):
    """Apply light feature-level augmentation for tabular features."""
    x_aug = x.copy()
    if drop_prob > 0:
        mask = np.random.binomial(1, 1 - drop_prob, size=x_aug.shape).astype(x_aug.dtype)
        x_aug = x_aug * mask
    if noise_std > 0:
        noise = np.random.normal(loc=0.0, scale=noise_std, size=x_aug.shape).astype(x_aug.dtype)
        x_aug = x_aug + noise
    return x_aug


def maybe_augment_views(X_list, config):
    """Optionally augment all views based on config['augmentation'] settings."""
    if not isinstance(config, dict):
        return X_list
    aug_cfg = config.get('augmentation', {}) or {}
    if not aug_cfg.get('enable', False):
        return X_list

    drop_prob = float(aug_cfg.get('feature_dropout', 0.0))
    noise_std = float(aug_cfg.get('gaussian_std', 0.0))

    augmented = []
    for x in X_list:
        augmented.append(_augment_view(x, drop_prob=drop_prob, noise_std=noise_std))
    return augmented


def _apply_input_norm_if_needed(x, config):
    """Apply optional per-view feature normalization/standardization."""
    if not isinstance(config, dict):
        return x

    norm_cfg = config.get('input_norm', {}) or {}
    if not bool(norm_cfg.get('enable', False)):
        return x

    method = str(norm_cfg.get('method', 'zscore')).lower()
    eps = float(norm_cfg.get('eps', 1e-12))
    clip_value = norm_cfg.get('clip', None)

    if method == 'zscore':
        return util.standardize_per_feature(x, eps=eps, clip_value=clip_value)
    if method == 'minmax':
        return util.normalize_per_feature(x, eps=eps)
    return x
# ------修改部分-----------------------------------------------------------------------------------------------------


def load_data(config):
    data_name = config['dataset']
    main_dir = DATA_DIR
    X_list = []
    Y_list = []
    if data_name in ['Scene_15']:
        mat = sio.loadmat(main_dir / 'Scene_15.mat')
        X = mat['X'][0]
        X_list.append(X[0].astype('float32'))
        X_list.append(X[1].astype('float32'))
        Y_list.append(np.squeeze(mat['Y']))
        Y_list.append(np.squeeze(mat['Y']))

    elif data_name in ['LandUse_21']:
        mat = sio.loadmat(main_dir / 'LandUse_21.mat')
        train_x = []
        train_x.append(sparse.csr_matrix(mat['X'][0, 0]).A)  # 20
        train_x.append(sparse.csr_matrix(mat['X'][0, 1]).A)  # 59
        train_x.append(sparse.csr_matrix(mat['X'][0, 2]).A)  # 40
        index = random.sample(range(train_x[0].shape[0]), 2100)  # 30000
        for view in [1, 2]:
            x = train_x[view][index]
            y = np.squeeze(mat['Y']).astype('int')[index]
            X_list.append(x)
            Y_list.append(y)

    elif data_name in ['NoisyMNIST']:
        data = sio.loadmat(main_dir / 'NoisyMNIST.mat')
        train = DataSet_NoisyMNIST(data['X1'], data['X2'], data['trainLabel'])
        tune = DataSet_NoisyMNIST(data['XV1'], data['XV2'], data['tuneLabel'])
        test = DataSet_NoisyMNIST(data['XTe1'], data['XTe2'], data['testLabel'])
        X_list.append(np.concatenate([tune.images1, test.images1], axis=0))
        X_list.append(np.concatenate([tune.images2, test.images2], axis=0))
        Y_list.append(np.concatenate([np.squeeze(tune.labels[:, 0]), np.squeeze(test.labels[:, 0])]))
        Y_list.append(np.concatenate([np.squeeze(tune.labels[:, 0]), np.squeeze(test.labels[:, 0])]))

    elif data_name in ['Caltech101-20']:
        mat = sio.loadmat(main_dir / (data_name + '.mat'))
        X = mat['X'][0]
        for view in [3, 4]:
            x = X[view]
            x = util.normalize(x).astype('float32')
            y = np.squeeze(mat['Y']).astype('int')
            X_list.append(x)
            Y_list.append(y)

    elif data_name in ['Hdigit']:
        mat = sio.loadmat(main_dir / 'Hdigit.mat')
        X = mat['data']
        X_list.append(X[0][0].astype('float32').T)
        X_list.append(X[0][1].astype('float32').T)
        Y_list.append(np.squeeze(mat['truelabel'][0, 0]))

    elif data_name in ['2V_BDGP']:
        mat = sio.loadmat(main_dir / '2V_BDGP.mat')
        X1 = mat['X1'].astype('float32')
        X2 = mat['X2'].astype('float32')
        Y = np.squeeze(mat['Y']).astype('int')
        X_list.append(X1)
        X_list.append(X2)
        Y_list.append(Y)
        Y_list.append(Y)

    elif data_name in ['MNIST-USPS']:
        # 加载 MNIST-USPS 数据集
        mat = sio.loadmat(main_dir / 'MNIST-USPS.mat')
        X1 = mat['X1']  # (5000, 28, 28, 1)
        X2 = mat['X2']  # (5000, 28, 28, 1)
        Y = mat['Y'].reshape(-1)  # (5000,)

        # 展平为 (5000, 784)
        X1 = X1.reshape((X1.shape[0], -1)).astype('float32')
        X2 = X2.reshape((X2.shape[0], -1)).astype('float32')

        X_list.append(X1)
        X_list.append(X2)
        Y_list = [Y for _ in range(2)]
        return X_list, Y_list

    # ------修改部分---------
    X_list = maybe_augment_views(X_list, config)
    # ------修改部分---------
    return X_list, Y_list

def load_multiview_data(config):
    data_name = config['dataset']
    main_dir = DATA_DIR
    X_list = []
    Y_list = []

    if data_name in ['Mfeat']:
        mat = sio.loadmat(main_dir / (data_name + '.mat'))
        X = mat['data']
        truelabel = mat['truelabel']
        for view in range(5):  #
            x = X[0, view]
            x = x.T
            x = util.normalize(x).astype('float32')  #
            y = truelabel[0, view].flatten()
            y = y.astype('int')
            X_list.append(x)
            Y_list.append(y)
        return X_list, Y_list

    elif data_name in ['handwritten']:
        mat = sio.loadmat(main_dir / 'handwritten.mat')
        X = mat['X']  # 1x6 cell
        Y = mat['Y'].reshape(-1)  # (2000,)
        for i in range(6):
            x = X[0, i].astype('float32')
            X_list.append(x)
        Y_list = [Y for _ in range(6)]  # 每个视图都用同一标签
        return X_list, Y_list

    elif data_name in ['Scene_15']:
        mat = sio.loadmat(main_dir / 'Scene_15.mat')
        X = mat['X']  # 假设 X 是 1x3 cell，每个 cell 是 (样本数, 特征数)
        Y = mat['Y'].reshape(-1)
        for i in range(3):
            x = X[0, i].astype('float32')
            X_list.append(x)
        Y_list = [Y for _ in range(3)]
        return X_list, Y_list

    elif data_name in ['100leaves']:
        mat = sio.loadmat(main_dir / '100leaves.mat')
        X = mat['X']  # 1x3 cell
        Y = mat['Y'].reshape(-1)
        for i in range(3):
            x = X[0, i].astype('float32')
            X_list.append(x)
        Y_list = [Y for _ in range(3)]
        return X_list, Y_list

    elif data_name in ['cub']:
        mat = sio.loadmat(main_dir / 'cub.mat')
        X = mat['X']  # 1x2 cell
        Y = mat['gt'].reshape(-1)
        for i in range(2):
            x = X[0, i].T.astype('float32')
            X_list.append(x)
        Y_list = [Y for _ in range(2)]
        return X_list, Y_list

    elif data_name in ['MSRC_v1']:
        mat = sio.loadmat(main_dir / 'MSRC_v1.mat')
        X = mat['X']  # 1x5 cell
        Y = mat['Y'].reshape(-1)
        for i in range(5):
            x = X[0, i].astype('float32')
            X_list.append(x)
        Y_list = [Y for _ in range(5)]
        return X_list, Y_list

    elif data_name in ['NUS']:
        mat = sio.loadmat(main_dir / 'NUS.mat')
        X = mat['X']  # 1x5 cell，每个cell为(样本数, 特征数)
        Y = mat['Y'].reshape(-1)
        for i in range(6):
            x = X[0, i].astype('float32')
            X_list.append(x)
        Y_list = [Y for _ in range(6)]
        return X_list, Y_list

    elif data_name in ['reuters_1200']:
        mat = h5py.File(main_dir / 'reuters_1200.mat', 'r')
        X = mat['X']
        X_list = []
        for i in range(X.shape[0]):
            ref = X[i, 0]
            arr = np.array(mat[ref]).T  # 转置为(1200, 2000)
            X_list.append(arr.astype('float32'))
        Y = np.array(mat['Y']).reshape(-1)
        Y_list = [Y for _ in range(5)]
        return X_list, Y_list

    elif data_name in ['ORL']:
        mat = sio.loadmat(main_dir / 'ORL.mat')
        X = mat['X']  # 1x4 cell
        Y = mat['Y'].reshape(-1)  # (400,)
        for i in range(4):
            x = X[0, i].astype('float32')
            X_list.append(x)
        Y_list = [Y for _ in range(4)]
        return X_list, Y_list

    elif data_name in ['COIL20']:  
        mat = sio.loadmat(main_dir / 'COIL20.mat')
        X = mat['X']  # 1x3 cell
        Y = mat['Y'].reshape(-1)  # (1440,)
        for i in range(3):
            x = X[0, i].astype('float32')
            X_list.append(x)
        Y_list = [Y for _ in range(3)]
        return X_list, Y_list

    elif data_name in ['ALOI_100']:
        mat = sio.loadmat(main_dir / 'ALOI_100.mat')
        X = mat['fea']  # 1x4 cell
        Y = mat['gt'].reshape(-1)  # (10800,)
        for i in range(4):
            x = X[0, i].astype('float32')
            x = _apply_input_norm_if_needed(x, config).astype('float32')
            X_list.append(x)
        Y_list = [Y for _ in range(4)]
        return X_list, Y_list

    elif data_name in ['Caltech101-7', 'caltech101-7', 'caltech101_7']:
        mat = sio.loadmat(main_dir / 'Caltech101-7.mat')
        X = mat['X']  # 1x6 cell
        Y = mat['Y'].reshape(-1)
        for i in range(6):
            x = X[0, i].astype('float32')
            X_list.append(x)
        Y_list = [Y for _ in range(6)]
        return X_list, Y_list



class DataSet_NoisyMNIST(object):

    def __init__(self, images1, images2, labels, fake_data=False, one_hot=False,
                 dtype=np.float32):
        """Construct a DataSet.
        one_hot arg is used only if fake_data is true.  `dtype` can be either
        `uint8` to leave the input as `[0, 255]`, or `float32` to rescale into
        `[0, 1]`.
        """
        if dtype not in (np.uint8, np.float32):
            raise TypeError('Invalid image dtype %r, expected uint8 or float32' % dtype)


        if fake_data:
            self._num_examples = 10000
            self.one_hot = one_hot
        else:
            assert images1.shape[0] == labels.shape[0], (
                    'images1.shape: %s labels.shape: %s' % (images1.shape,
                                                            labels.shape))
            assert images2.shape[0] == labels.shape[0], (
                    'images2.shape: %s labels.shape: %s' % (images2.shape,
                                                            labels.shape))
            self._num_examples = images1.shape[0]
            # 归一化到[0,1]，兼容uint8和float32
            from pcdc_imvc.utils import util
            images1 = util.normalize(images1).astype(np.float32)
            images2 = util.normalize(images2).astype(np.float32)

        self._images1 = images1
        self._images2 = images2
        self._labels = labels
        self._epochs_completed = 0
        self._index_in_epoch = 0

    @property
    def images1(self):
        return self._images1

    @property
    def images2(self):
        return self._images2

    @property
    def labels(self):
        return self._labels

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def next_batch(self, batch_size, fake_data=False):
        """Return the next `batch_size` examples from this data set."""
        if fake_data:
            fake_image = [1] * 784
            if self.one_hot:
                fake_label = [1] + [0] * 9
            else:
                fake_label = 0
            return [fake_image for _ in range(batch_size)], [fake_image for _ in range(batch_size)], [fake_label for _
                                                                                                      in range(
                    batch_size)]

        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1
            # Shuffle the data
            perm = np.arange(self._num_examples)
            np.random.shuffle(perm)
            self._images1 = self._images1[perm]
            self._images2 = self._images2[perm]
            self._labels = self._labels[perm]
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._num_examples

        end = self._index_in_epoch
        return self._images1[start:end], self._images2[start:end], self._labels[start:end]


def load_NoisyMNIST():
    data = sio.loadmat('./data/NoisyMNIST.mat')

    train = DataSet_NoisyMNIST(data['X1'], data['X2'], data['trainLabel'])

    tune = DataSet_NoisyMNIST(data['XV1'], data['XV2'], data['tuneLabel'])

    test = DataSet_NoisyMNIST(data['XTe1'], data['XTe2'], data['testLabel'])

    return train, tune, test