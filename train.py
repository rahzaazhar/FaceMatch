import pdb
import torch
import argparse
import numpy as np
import torch.nn as nn
import sklearn.metrics as skmetrics
import torchvision
from torch.utils.data import Subset, dataloader
import torchvision.transforms as tf
from torch.utils.data import DataLoader
import torch.optim as optim
from sklearn.model_selection import train_test_split
from model import AttributeModel
from dataset import CelebADataset
from utils import AverageMeter

def get_train_val_dataset(dataset, val_split=0.25):
    train_idx, val_idx = train_test_split(list(range(len(dataset))), test_size=val_split)
    datasets = {}
    train_dataset = Subset(dataset, train_idx)
    val_dataset = Subset(dataset, val_idx)
    return train_dataset, val_dataset

def tensor_to_numpy(tensor):
    if tensor.is_cuda:
        tensor = tensor.cpu().detach().numpy()
    else:
        tensor = tensor.detach().numpy()
    
    return tensor


def train(args, attributemodel, dataset):
    
    #logging
    loss_avg = AverageMeter()
    train_dataset, val_dataset = get_train_val_dataset(dataset, val_split=0.2)
    optimizer = optim.Adam(attributemodel.model.parameters(), lr=args.lr)
    Loss_function = nn.BCELoss()
    trainloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    valloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    
    for epoch in range(args.epochs):
        loss_avg.reset()
        for idx, (image_batch, label_batch) in enumerate(trainloader):
            preds = attributemodel(image_batch)
            loss = Loss_function(preds, label_batch)

            loss_avg.update(loss.item())
            if idx%args.print_iter == 0:
                print(f'Epoch:{epoch} iteration:{idx} Loss:{loss_avg.avg}') 
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        val_metrics = validate(attributemodel, valloader)
        print(f'End of Epoch:{epoch}, Loss:{loss_avg.avg}')
        print('Val Loss:{loss} Recall:{recall} Precision:{precision}'.format(**val_metrics))


def validate(attributemodel, valloader):
    loss_avg = AverageMeter()
    precision_avg = AverageMeter()
    recall_avg = AverageMeter()
    Loss_function = nn.BCELoss()
    with torch.no_grad():
        for idx, (image_batch, label_batch) in enumerate(valloader):
            preds = attributemodel(image_batch)
            loss = Loss_function(preds, label_batch)
            loss_avg.update(loss.item())

            preds = 1*(preds>0.5)
            preds = tensor_to_numpy(preds)
            label_batch = tensor_to_numpy(label_batch)
            prec = skmetrics.precision_score(y_true=label_batch, y_pred=preds, average='samples')
            recall = skmetrics.recall_score(y_true=label_batch, y_pred=preds, average='samples')
            precision_avg.update(prec)
            recall_avg.update(recall)
    
    return {'loss':loss_avg.avg, 'recall':recall_avg.avg, 'precision':precision_avg.avg}
            
            


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_folder_path', type=str, default='./data/img_align_celeba')
    parser.add_argument('--annotation_path', type=str, default='./data/list_attr_celeba.txt')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--backbone_name', type=str, default='resnet50')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--print_iter', dest='print_iter', type=int, default=10)
    parser.add_argument('--quick_run', dest='quick_run', action='store_true')
    parser.add_argument('--val_iter', dest='val_iter', type=int, default=10)
    args = parser.parse_args()
    return args



if __name__ == "__main__":
    args = parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    transforms = tf.Compose([tf.ToTensor()])

    dataset = CelebADataset(image_folder_path=args.image_folder_path, 
                            annotation_path=args.annotation_path, 
                            device=device, 
    
                            transforms=transforms)
    num_classes = len(dataset.attributes)

    if args.quick_run:
        dataset = Subset(dataset, list(range(1000)))
        args.epochs = 1
        print(f'Running quick test, dataset length:{len(dataset)}')

    
    
    attributemodel = AttributeModel(backbone=args.backbone_name, no_attributes=num_classes, device=device, pretrain=True)
    best_model = train(args, attributemodel, dataset)

    
