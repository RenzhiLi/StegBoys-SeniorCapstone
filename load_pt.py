import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import time
import os
import copy

class DataLoader_withpath(datasets.ImageFolder):
    def __getitem__(self, index):
        """
        Args:
            index (int): Index
    
        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        path, target = self.imgs[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target, path

data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}
if __name__ == '__main__':
    data_dir = './densenet_test_pics'
    image_datasets = DataLoader_withpath(data_dir, data_transforms['val'])
    dataloaders = torch.utils.data.DataLoader(image_datasets, batch_size=4,
                                                shuffle=True, num_workers=4)
    dataset_sizes = len(image_datasets)
    # class_names = image_datasets['train'].classes

    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Get a batch of training data
    # inputs, classes = next(iter(dataloaders['train']))

    # Make a grid from batch
    # out = torchvision.utils.make_grid(inputs)

    # imshow(out, title=[class_names[x] for x in classes])

    model_ft = models.densenet161(pretrained=True)
    # model_ft = CNN()
    # num_ftrs = len(model_ft.features)
    num_ftrs = model_ft.classifier.in_features
    model_ft.fc = nn.Linear(num_ftrs, 2)

    # model_ft = model_ft.to(device)
    model_ft.load_state_dict(torch.load('./modelv1.pt', map_location='cpu'))
    model_ft.eval() # run if you only want to use it for inference

    # predict

    # Iterate over data.
    running_corrects = 0
    result_dict = {}
    for inputs, labels, path in dataloaders:
        outputs = model_ft(inputs)
        _, preds = torch.max(outputs, 1)
        predlist = list(preds.numpy())
        for i in range(len(predlist)):
            file_name = path[i].split('\\')[-1]
            result_dict[file_name] = 'good' if (predlist[i] == 1) else 'bad'
        running_corrects += torch.sum(preds == labels.data)

    print("here")
    print(result_dict)
    print("HERE")
