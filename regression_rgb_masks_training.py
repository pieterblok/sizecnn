# thanks to: https://github.com/cfotache/pytorch_imageclassifier
# thanks to: https://towardsdatascience.com/how-to-train-an-image-classifier-in-pytorch-and-use-it-to-perform-basic-inference-on-single-images-99465a1e9bf5
# thanks to: Deep Learning with PyTorch course from opencv (week 7)

import torch
from torch import nn
from torch import optim

from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms, models
from torchvision.transforms import functional as F

import matplotlib.pyplot as plt

import os
import numpy as np

from PIL import Image

# this class is altered from class MonkeySpecies10Dataset from the opencv course (week 7)
class regression_dataset(Dataset):
    """
    This custom dataset class take root directory and train flag, 
    and return dataset training dataset id train flag is true 
    else is return validation dataset.
    """
    
    def __init__(self, data_root, train=True, image_shape=None, transform=None):
        
        """
        init method of the class.
        
         Parameters:
         
         data_root (string): path of root directory.
         
         train (boolean): True for training dataset and False for test dataset.
         
         image_shape (int or tuple or list): [optional] int or tuple or list. Defaut is None. 
                                             It is not None image will resize to the given shape.
                                 
         transform (method): method that will take PIL image and transforms it.
         
        """
        
        # set image_resize attribute
        if image_shape is not None:
            if isinstance(image_shape, int):
                self.image_shape = (image_shape, image_shape)
            
            elif isinstance(image_shape, tuple) or isinstance(image_shape, list):
                assert len(image_shape) == 1 or len(image_shape) == 2, 'Invalid image_shape tuple size'
                if len(image_shape) == 1:
                    self.image_shape = (image_shape[0], image_shape[0])
                else:
                    self.image_shape = image_shape
            else:
                raise NotImplementedError 
        else:
            self.image_shape = image_shape
            
        # set transform attribute
        self.transform = transform
        
        # initialize the data dictionary
        self.data_dict = {
            'image_path': [],
            'label': []
        }
        
        # training data path, this will be used as data root if train = True
        if train:
            img_dir = os.path.join(data_root, 'training')
            
        # validation data path, this will be used as data root if train = False
        else:
            img_dir = os.path.join(data_root, 'validation')
            
        for img in os.listdir(img_dir):
            if img.endswith(".jpg") or img.endswith(".png"):
                img_path = os.path.join(img_dir, img)
                self.data_dict['image_path'].append(img_path)

                basename, file_extension = os.path.splitext(img)
                labelname = basename + ".txt"

                file1 = open(os.path.join(img_dir, labelname),'r')  
                labelstr = file1.read()
                # label = (float(labelstr.split(",")[0]), float(labelstr.split(",")[1]), float(labelstr.split(",")[2]), float(labelstr.split(",")[3]))
                label = float(labelstr)
                self.data_dict['label'].append(label)
                    
    def __len__(self):
        """
        return length of the dataset
        """
        return len(self.data_dict['label'])
    
    
    def __getitem__(self, idx):
        """
        For given index, return images with resize and preprocessing.
        """
        
        image = Image.open(self.data_dict['image_path'][idx]).convert("RGB")
        
        if self.image_shape is not None:
            image = F.resize(image, self.image_shape)
            
        if self.transform is not None:
            image = self.transform(image)
            
        target = self.data_dict['label'][idx]
        
        return image, target            


# data root directory
data_root = '/home/pieterdeeplearn/harvestcnn/datasets/20200713_size_experiment_realsense/rgb_masks'

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize((0.485,0.456,0.406), (0.229,0.224,0.225))
    ])

train_dataset =  regression_dataset(data_root, train=True, image_shape=256, transform=preprocess)
test_dataset =  regression_dataset(data_root, train=False, image_shape=256, transform=preprocess)

print('Length of the train dataset: {}'.format(len(train_dataset)))
print('Length of the test dataset: {}'.format(len(test_dataset)))

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=9, shuffle=True, num_workers=2)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=9, shuffle=True, num_workers=2)

# plt.rcParams["figure.figsize"] = (9, 9)
# plt.figure
# for images, labels in test_loader:
#     for i in range(len(labels)):
#         plt.subplot(3, 3, i+1)
#         img = F.to_pil_image(images[i])
#         plt.imshow(img)
#         plt.gca().set_title('Target: {0}'.format(labels[i]))
#     plt.show()
#     break

# use cuda:
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet50(pretrained=True)

# Freeze parameters so we don't backprop through them
for param in model.parameters():
    param.requires_grad = False
    
model.fc = nn.Sequential(nn.Linear(2048, 512),
                                 nn.ReLU(),
                                 nn.Dropout(0.2),
                                 nn.Linear(512, 1))

print(model)

# test_performance: 18.5 mm
# criterion = nn.L1Loss()
# optimizer = optim.SGD(model.fc.parameters(), lr=0.001, momentum=0.9)

# test_performance: 15.3 mm
criterion = nn.L1Loss()
optimizer = optim.Adam(model.fc.parameters(), lr=0.005)

model.to(device)

epochs = 20
steps = 0
running_loss = 0
print_every = 10
train_losses, test_losses = [], []

for epoch in range(epochs):
    for inputs, labels in train_loader:
        steps += 1
        inputs, labels = inputs.to(device), labels.to(device)
        labels = labels.view(labels.shape[0], 1)

        optimizer.zero_grad()
        logps = model.forward(inputs)
        loss = criterion(logps, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        
        if steps % print_every == 0:
            test_loss = 0
            accuracy = 0
            model.eval()
            with torch.no_grad():
                for inputs, labels in test_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    labels = labels.view(labels.shape[0], 1)

                    logps = model.forward(inputs)
                    batch_loss = criterion(logps, labels)
                    test_loss += batch_loss.item()

            train_losses.append(running_loss/len(train_loader))
            test_losses.append(test_loss/len(test_loader))                    
            print("Epoch:" + str(epoch))
            print("Train loss:" + str(running_loss/len(train_loader)))
            print("Test loss:" + str(test_loss/len(test_loader)))
            running_loss = 0
            model.train()


PATH = './diameter_regression_rgb_masks.pth'

# only save the weights:
# torch.save(model.state_dict(), PATH)

# save the entire model (for first checking):
torch.save(model, PATH)