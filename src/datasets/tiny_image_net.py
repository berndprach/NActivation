"""
Tiny ImageNet 200 dataset.
Code adapted from
"1-Lipschitz Layers Compared: Memory, Speed, and Certifiable Robustness", 2023.
"""

import os

from torchvision.datasets import ImageFolder

DATASET_URL = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
DATASET_FILENAME = "tiny-imagenet-200"
DATASET_ZIP_FILENAME = "tiny-imagenet-200.zip"
VAL_ANNOTATION_FILENAME = "val_annotations.txt"


class TinyImageNet(ImageFolder):
    def __init__(self,
                 root: str,
                 *args,
                 train: bool = True,
                 download: bool = True,
                 **kwargs
                 ):

        if download:
            download_and_prepare_tiny_image_net(root)

        subfolder = "train" if train else "val"
        image_folder = os.path.join(root, DATASET_FILENAME, subfolder)
        super().__init__(image_folder, *args, **kwargs)


def download_and_prepare_tiny_image_net(root):
    zip_filename = os.path.join(root, DATASET_ZIP_FILENAME)
    if os.path.exists(zip_filename):
        return

    print("Downloading Tiny ImageNet 200 dataset...")
    old_cwd = os.getcwd()
    os.chdir(root)
    os.system(f"wget -nc {DATASET_URL}")

    print("Unzipping Tiny ImageNet 200 dataset...")
    os.system(f"unzip -n {DATASET_ZIP_FILENAME}")

    print("Moving validation images to sub-folders...")
    val_dir = os.path.join(DATASET_FILENAME, "val")
    val_annotation_filename = os.path.join(val_dir, VAL_ANNOTATION_FILENAME)
    with open(val_annotation_filename) as f:
        for line in f:
            fields = line.split()
            img_filename = fields[0]
            label = fields[1]
            label_dir = os.path.join(val_dir, label)
            os.makedirs(label_dir, exist_ok=True)
            # move image to sub-folder
            os.rename(
                os.path.join(val_dir, "images", img_filename),
                os.path.join(label_dir, img_filename),
            )

    # Remove empty image folder:
    os.rmdir(os.path.join(val_dir, "images"))

    print("Done.")
    os.chdir(old_cwd)
