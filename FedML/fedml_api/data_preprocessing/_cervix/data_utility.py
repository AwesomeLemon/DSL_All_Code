# Alex: based on "exp2_path"
import numpy as np
import h5py
import importlib
import stringcase

from fedml_api.data_preprocessing.my_transforms.compose import Compose


def init_transform(transforms, transforms_args):
    transform_instances = []
    for module_name in transforms:
        transform = importlib.import_module(f"fedml_api.data_preprocessing.my_transforms.{stringcase.snakecase(module_name)}")
        if module_name in transforms_args:
            transform_arg = transforms_args[module_name]
        else:
            transform_arg = []

        instance = getattr(transform, module_name)(*transform_arg)
        transform_instances.append(instance)

    compose = Compose(transform_instances)
    return compose



def count_samples(h5_filepath, path='train'):
    count = 0
    with h5py.File(h5_filepath, 'r') as h5_file:
        count = len(h5_file[path].keys())

    return count


def build_pairs(dataset, sample_rate=1.0, im_size=None):
    assert im_size is None
    keys = list(dataset['images'].keys())
    im_arr = []
    label_arr = []

    if 1 > sample_rate > 0.:
        sample_size = max(1, int(len(keys) * sample_rate))
        sample_idx = np.random.choice(len(keys), sample_size, replace=False)
        keys = [keys[i] for i in sample_idx]

    for key in keys:
        im = dataset[f"images/{key}"][()]
        label = dataset[f"labels/{key}"][()]

        assert len(im.shape) == 2
        assert len(label.shape) == 2
        # print(f"Before: label.max(): {label.max()} | label.min(): {label.min()}")
        label = label[np.newaxis, ...].astype("uint8")
        # print(f"After: label.max(): {label.max()} | label.min(): {label.min()}")

        # if im_size is not None:
        #     im, label = resize_keep_spacing(im, label, im_size)  # padded center crop
        im = im[np.newaxis, ...]

        im_arr.append(im)
        label_arr.append(label)

        # print(f"im.shape: {im.shape} | label.shape: {label.shape}")
        # print(f"im.max(): {im.max()}")

    return im_arr, label_arr, keys