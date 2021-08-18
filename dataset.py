import cv2
import pdb
import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
import torchvision.transforms as tf 
import numpy as np

def process_lines(lines):
    image_names, labels = [], []
    for line in lines:
        line = line.split()
        image_names.append(line[0])
        label = torch.Tensor([1 if ele == '1' else 0 for ele in line[1:]])
        labels.append(label)
    return image_names, labels

class CelebADataset(Dataset):
    def __init__(self, image_folder_path, annotation_path, device=torch.device('cpu'), transforms=None):
        
        assert isinstance(device, torch.device)
        self.image_folder_path = image_folder_path
        self.annotation_path = annotation_path
        self.transforms = transforms
        self.device = device



        with open(annotation_path, 'r') as f:
            lines = f.readlines()
            self.attributes = lines[1].split()
            self.image_names, self.labels = process_lines(lines[2:])

    def __getitem__(self, idx):
        image_name = self.image_names[idx]
        label = self.labels[idx]
        image = cv2.imread(self.image_folder_path+'/'+image_name)
        if self.transforms is not None:
            image = self.transforms(image)
        
        image = image.to(self.device)
        label = label.to(self.device)
        return image, label
    
    def __len__(self) -> int:
        return len(self.image_names)


if __name__ == "__main__":
    annotation_path = './data/list_attr_celeba.txt'
    image_folder_path = './data/img_align_celeba'
    transforms = tf.Compose([tf.ToTensor()])
    dataset = CelebADataset(image_folder_path, annotation_path, transforms=transforms)
    loader = iter(DataLoader(dataset, 32, True))
    a, b = next(loader)
    pdb.set_trace()


    