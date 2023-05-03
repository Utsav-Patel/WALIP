import torch
import torchvision
import cv2
from PIL import Image
import albumentations as A
import numpy as np
from transformers import ViTFeatureExtractor
import pandas as pd
np.random.seed(0)

def load_data(params):
    train_df, valid_df = prepare_dataframe(params.captions_path)
    train_loader = build_loaders(train_df, params)
    valid_loader = build_loaders(valid_df, params)
    return train_loader, valid_loader

def prepare_dataframe(captions_path):
    df = pd.read_csv(captions_path)

    unique_values = list(set(df["image_id"].values))
    im_ids = np.arange(0, len(unique_values))
    random_size = int(0.2 * len(unique_values))

    rn_ids = np.random.choice(im_ids, size=random_size, replace=False)
    tr_ids = [image_id for image_id in im_ids if image_id not in rn_ids]
    tr_imgs = [unique_values[i] for i in tr_ids]
    vl_imgs = [unique_values[i] for i in rn_ids]

    tr_df = df[df["image_id"].isin(tr_imgs)]
    tr_df = tr_df.reset_index(drop=True)
    vl_df = df[df["image_id"].isin(vl_imgs)]
    vl_df = vl_df.reset_index(drop=True)

    return tr_df, vl_df

def build_loaders(df, params):
    image_ids = df["image_id"].values
    name2path = lambda path, prefix, im_id: path + prefix + str(im_id).zfill(12) + ".jpg"
    files = [name2path(params.image_path, params.image_path, image_ids[i]) for i in range(len(image_ids))]
    dataset = CLIPDataset_ViT(files, df["caption"].values)
    
    batch_size = params.batch_size
    num_workers = params.num_workers
    dataloader = torch.utils.data.DataLoader(dataset, num_workers=num_workers, batch_size=batch_size, shuffle=True)
    return dataloader

class CLIPDataset_ViT(torch.utils.data.Dataset):
    def __init__(self, files, labels):
        self.labels = list(labels)
        self.files = files
        self.extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224-in21k')

    def __getitem__(self, idx):
        im = cv2.imread(self.files[idx])
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        im2np = np.moveaxis(im, source=-1, destination=0)
        transformed_img = self.extractor(im2np, return_tensors="pt") # transforms already

        data = {
            "image": transformed_img['pixel_values'][0],
            "caption": self.labels[idx],
        }

        return data

    def __len__(self):
        return len(self.labels)


class CLIPDataset_resnet(torch.utils.data.Dataset):
    def __init__(self, files, labels):
        self.labels = list(labels)
        self.files = files
        imagenet_mean = [0.485, 0.456, 0.406]
        imagenet_std = [0.229, 0.224, 0.225]
        self.transforms = torchvision.transforms.Compose([
            torchvision.transforms.Resize(256),
            torchvision.transforms.CenterCrop(224),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
        ])

    def __getitem__(self, idx):
        image = Image.open(self.files[idx])
        image = image.convert('RGB')
        if self.transforms is not None:
            image = self.transforms(image)

        data = {
            "image": image['pixel_values'][0],
            "caption": self.labels[idx],
        }

        return data

    def __len__(self):
        return len(self.labels)

    