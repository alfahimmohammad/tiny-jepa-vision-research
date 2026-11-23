"""Reproducible, non-overlapping CIFAR-10 splits.

From the official TRAIN split (50,000), using one permutation drawn with SPLIT_SEED:
  pretrain     10,000  uniform random (labels are NOT used to choose them)
  probe_train   2,000  class-stratified, 200 / class, taken from the remaining images
  val           1,000  class-stratified, 100 / class, taken after probe_train
From the official TEST split (10,000), using a permutation drawn with TEST_SEED:
  test          2,000  class-stratified, 200 / class

Stratifying the probe/val/test subsets keeps the small labelled sets balanced. Labels only decide
which images go into those three subsets; the pretraining subset is chosen without labels and
pretraining never reads them.
"""
import hashlib
import json
import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import datasets, transforms

SPLIT_SEED = 0
TEST_SEED = 1
SIZES = {"pretrain": 10_000, "probe_train": 2_000, "val": 1_000, "test": 2_000}
N_CLASSES = 10


def load_cifar(root):
    tr = datasets.CIFAR10(root, train=True, download=True)
    te = datasets.CIFAR10(root, train=False, download=True)
    return (tr.data, np.asarray(tr.targets, dtype=np.int64),
            te.data, np.asarray(te.targets, dtype=np.int64))


def _stratified_take(order, labels, per_class, exclude=()):
    taken, counts, excl = [], np.zeros(N_CLASSES, dtype=int), set(exclude)
    for i in order:
        i = int(i)
        if i in excl or counts[labels[i]] >= per_class:
            continue
        taken.append(i); counts[labels[i]] += 1
        if counts.min() >= per_class:
            break
    assert counts.min() == per_class, "not enough images for stratified split"
    return taken


def make_splits(train_labels, test_labels, split_seed=SPLIT_SEED, test_seed=TEST_SEED):
    perm = np.random.default_rng(split_seed).permutation(len(train_labels))
    pretrain = [int(i) for i in perm[:SIZES["pretrain"]]]
    rest = perm[SIZES["pretrain"]:]
    probe = _stratified_take(rest, train_labels, SIZES["probe_train"] // N_CLASSES)
    val = _stratified_take(rest, train_labels, SIZES["val"] // N_CLASSES, exclude=probe)
    tperm = np.random.default_rng(test_seed).permutation(len(test_labels))
    test = _stratified_take(tperm, test_labels, SIZES["test"] // N_CLASSES)
    splits = {"pretrain": sorted(pretrain), "probe_train": sorted(probe),
              "val": sorted(val), "test": sorted(test)}
    return {"split_seed": split_seed, "test_seed": test_seed, "sizes": SIZES,
            "source": {"pretrain": "train", "probe_train": "train", "val": "train", "test": "test"},
            "fingerprint": fingerprint(splits), "indices": splits}


def fingerprint(indices):
    blob = json.dumps({k: indices[k] for k in sorted(indices)}).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


def load_or_create_splits(path, train_labels, test_labels):
    fresh = make_splits(train_labels, test_labels)
    if os.path.exists(path):
        with open(path) as f:
            saved = json.load(f)
        if saved["fingerprint"] != fresh["fingerprint"]:
            raise RuntimeError(f"{path} does not match regenerated splits")
        return saved
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(fresh, f)
    return fresh


# Upstream pretraining augmentation, unchanged (ijepa.py:123-124).
def pretrain_transform(mean, std):
    return transforms.Compose([transforms.RandomResizedCrop(32, scale=(0.3, 1.0)),
                               transforms.ToTensor(), transforms.Normalize(mean, std)])


class PretrainSet(Dataset):
    """Unlabelled pretraining images. Returns only the image, so labels cannot reach the encoder."""

    def __init__(self, images_uint8, indices, transform):
        self.images, self.indices, self.transform = images_uint8, list(indices), transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        return self.transform(Image.fromarray(self.images[self.indices[i]]))


def to_unit_tensor(images_uint8, indices):
    """uint8 NHWC -> float NCHW in [0, 1] (no normalisation; corruptions are applied in this space)."""
    x = torch.from_numpy(images_uint8[np.asarray(indices)]).permute(0, 3, 1, 2).float() / 255.0
    return x.contiguous()


def normalize(x01, mean, std):
    m = torch.tensor(mean, dtype=x01.dtype, device=x01.device).view(1, 3, 1, 1)
    s = torch.tensor(std, dtype=x01.dtype, device=x01.device).view(1, 3, 1, 1)
    return (x01 - m) / s
