import os
import numpy as np
import pandas as pd
import cv2

from tqdm import tqdm
from transformers import ViTFeatureExtractor
from torchvision import transforms
from torchvision.datasets import CIFAR100, CIFAR10, ImageFolder
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import CenterCrop, Normalize, Resize, ToTensor
from torchvision.transforms.functional import InterpolationMode
from utils.helper import generate_path
import configs


class ImageDataset(Dataset):
    """
    PyTorch dataset for loading images with their labels.

    Args:
        data (list): List of images.
        labels (list): List of labels.
        transform (callable, optional): A function/transform that takes in an image and returns a transformed version.
        multilabel (bool, optional): If True, treats labels as multilabel.
    """
    def __init__(self, data, labels, transform=None, multilabel=False):
        super(ImageDataset, self).__init__()
        self.transform_fn = transform
        self.data_list = data
        self.label_list = labels
        self.is_multilabel = multilabel

    def __len__(self):
        """Returns the length of the dataset."""
        return len(self.data_list)

    def __getitem__(self, index):
        """Returns the image and its corresponding label for the given index."""
        image = self.data_list[index]
        label = self.label_list[index]
        if self.transform_fn is not None:
            image = self.transform_fn(image)
        if not self.is_multilabel:
            label = int(label)
        return image, label


class ViTDataset(Dataset):
    """
    PyTorch dataset for loading images and their corresponding labels, processed by the ViT model.

    Args:
        dataset (Dataset): The dataset object containing the images and labels.
        convert_image (bool, optional): If True, reads images from the given file paths and converts them to RGB.
    """
    def __init__(self, dataset, convert_image=False):
        super(ViTDataset, self).__init__()
        self.feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224-in21k')
        self.data = dataset.data
        self.labels = dataset.targets
        self.convert_image = convert_image

    def __len__(self):
        """Returns the length of the dataset."""
        return len(self.data)

    def __getitem__(self, index):
        """
        Returns the image and its corresponding label for the given index.

        If `convert_image` is True, reads the image from the file path and converts it to RGB before processing it with the ViT model.
        """
        if self.convert_image:
            image_path = self.data[index]
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = np.moveaxis(image, source=-1, destination=0)
            features = self.feature_extractor(image, return_tensors="pt")
        else:
            features = self.feature_extractor(self.data[index], return_tensors="pt")
        return features['pixel_values'][0], self.labels[index]


def get_label_indices(image_label):
    """
    Given an image label string, returns a list of integer indices corresponding to the labels.

    Args:
        image_label (str): A string containing one or more integer labels separated by spaces.

    Returns:
        A list of integer indices corresponding to the labels in the input string.
    """
    label_indices = [int(x) for x in image_label.split(" ")]
    return label_indices


def get_first_label_index(image_label):
    """
    Given an image label string, returns the integer index of the first label.

    Args:
        image_label (str): A string containing one or more integer labels separated by spaces.

    Returns:
        The integer index of the first label in the input string.
    """
    label_indices = get_label_indices(image_label)
    return label_indices[0]


def one_hot_encode(image_label, n_classes):
    """
    Given an image label string and the total number of classes, returns a one-hot encoded array
    of the labels for the given image.

    Args:
        image_label (str): A string containing one or more integer labels separated by spaces.
        n_classes (int): The total number of classes in the dataset.

    Returns:
        A one-hot encoded numpy array of the labels for the given image.
    """
    label_one_hot = np.zeros(n_classes)
    label_indices = get_label_indices(image_label)
    label_one_hot[label_indices] = 1
    return label_one_hot


def load_image_dataset(image_name, data_dir='../datasets', preprocess=None, multilabel=False):
    """
    Loads the specified image dataset from the given data directory with the specified preprocessing.

    Args:
        image_name (str): The name of the dataset to be loaded. Can be 'cifar10', 'cifar100', 'imagenet', or 'coco'.
        data_dir (str, optional): The path to the directory containing the dataset. Defaults to '../datasets'.
        preprocess (torchvision.transforms, optional): The preprocessing to be applied to the images. Defaults to None.
        multilabel (bool, optional): If True, loads the labels as multilabel. Defaults to False.

    Returns:
        The loaded dataset.
    """
    print('Loading image dataset:', image_name)

    if image_name.startswith('cifar'):
        if image_name == 'cifar100':
            dataset = CIFAR100(data_dir, transform=preprocess, download=True, train=False)
        elif image_name == 'cifar10':
            dataset = CIFAR10(data_dir, transform=preprocess, download=True, train=False)
        if preprocess is None:
            dataset = ViTDataset(dataset)

    elif image_name.startswith("imagenet"):
        if preprocess is None:
            preprocess = transforms.Compose([
                Resize([224], interpolation=InterpolationMode.BICUBIC),
                CenterCrop(224),
                ToTensor(),
                Normalize(configs.means[image_name], configs.stds[image_name]),
            ])
        dataset = ImageFolder(os.path.join(data_dir, f'{image_name}/val/'), preprocess)

    elif image_name == "coco":
        images, labels = load_image_data(image_name, 1000, False, "./", multilabel=multilabel)
        dataset = ViTDataset(ImageDataset(images, labels, multilabel=True), convert_image=True)

    return dataset


def load_image_data(image_name, num_images, using_filtered_images, src_lang, tgt_lang, preprocess=None,
                    multilabel=False):
    """
    Load and return image data and their corresponding labels for a given image dataset.

    Parameters:
    image_name (str): Name of the image dataset.
    num_images (int): Number of images to be returned.
    using_filtered_images (bool): If True, the images returned will be filtered by language.
    src_lang (str): Source language to filter images by.
    tgt_lang (str): Target language to filter images by.
    preprocess (torchvision.transforms.Compose): Preprocessing steps for the images.
    multilabel (bool): If True, labels will be one-hot encoded.

    Returns:
    images (numpy.ndarray): The image data.
    labels (numpy.ndarray): The labels for the image data.
    """
    num_classes = configs.num_classes[image_name]
    print("Load image data", image_name, num_images)

    if image_name == 'coco':
        captions_path = f"../datasets/coco/captions/en/{configs.caption_names['en']}"
        df = pd.read_csv(f"{captions_path}")
        image_ids = df["image_id"].values
        if multilabel:
            labels = df["labels"].apply(lambda x: one_hot_encode(x, configs.num_classes[image_name])).values
        else:
            labels = df["labels"].apply(get_first_label_index).values

        indices = np.arange(len(image_ids))
        np.random.seed(42)
        np.random.shuffle(indices)
        image_ids = image_ids[indices]
        final_labels = labels[indices]

        image_path = f"../datasets/coco/images/{configs.image_folders['en']}"
        image_filenames = [f"{image_path}/{configs.image_prefixes['en']}{str(image_ids[i]).zfill(12)}.jpg" for i in
                           range(len(image_ids))]
        images = image_filenames
        return images[:num_images], final_labels[:num_images]

    image_dataset = load_image_dataset(image_name, preprocess=preprocess)
    if using_filtered_images:
        fpath = generate_path('img_shared_index',
                              {'image_data': image_name, 'src_lang': src_lang, 'tgt_lang': tgt_lang})
        dct = np.load(fpath, allow_pickle=True).item()
        images = list()
        for idx in list(dct.values()):
            images += [image_dataset[idx[i]][0].numpy() for i in range(min(num_images, len(idx)))]
        return np.stack(images, axis=0)

    if image_name in ['cifar100', 'cifar10']:
        labels = np.asarray(image_dataset.targets)
    elif image_name.startswith('imagenet'):
        label_path = generate_path('img_label', {'image_data': image_name})
        if os.path.isfile(label_path):
            labels = np.load(label_path, allow_pickle=True)
        else:
            dataloader = DataLoader(image_dataset, batch_size=128, shuffle=False, drop_last=True, num_workers=4)
            labels = list()
            for batch in tqdm(dataloader):
                labels.append(batch[1].numpy())
            labels = np.stack(labels, axis=0)
            np.save(label_path, np.asarray(labels))
        labels = labels.flatten()
    # final images
    images = list()
    for c in range(num_classes):
        indices = np.argwhere(labels == c)

        if len(indices) == 0:
            continue
        images += [image_dataset[indices[i][0]][0] for i in range(num_images)]
    images = np.stack(images, axis=0)
    return images
