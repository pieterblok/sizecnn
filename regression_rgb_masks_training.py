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
import time

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

                # train only the diameter
                d = np.float32(labelstr)
                self.data_dict['label'].append(d)

                # train the x, y, z and the diameter
                # x = np.float32(labelstr.split(",")[0])
                # y = np.float32(labelstr.split(",")[1])
                # z = np.float32(labelstr.split(",")[2])
                # d = np.float32(labelstr.split(",")[3])
                # self.data_dict['label'].append(np.asarray((x, y, z, d)))

                    
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

# use cuda:
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet50(pretrained=True)

# Freeze parameters so we don't backprop through them
# for param in model.parameters():
#     param.requires_grad = False
    
model.fc = nn.Sequential(nn.Linear(2048, 512),
                                 nn.ReLU(),
                                 nn.Dropout(0.2),
                                 nn.Linear(512, 1))

print(model)

# error on train-set: 8.6 mm (avg), 22.6 mm (max)
# error on test-set: 7.8 mm (avg), 22.2 mm (max)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

model.to(device)

# training loop extracted from: https://github.com/spmallick/learnopencv/blob/master/Image-Classification-in-PyTorch/image_classification_using_transfer_learning_in_pytorch.ipynb
epochs = 50
start = time.time()
history = []
save_path = './weights/D_regression_rgb_masks'

for epoch in range(epochs):
    epoch_start = time.time()
    print("Epoch: {}/{}".format(epoch+1, epochs))
    
    # Set to training mode
    model.train()
    
    # Loss within the epoch
    train_loss = 0.0  
    valid_loss = 0.0
    
    for i, (inputs, labels) in enumerate(train_loader):

        inputs = inputs.to(device)
        labels = labels.to(device)

        if labels.ndim == 1:
            labels = labels.view(labels.shape[0], 1)
        
        # Clean existing gradients
        optimizer.zero_grad()
        
        # Forward pass - compute outputs on input data using the model
        outputs = model(inputs)
        
        # Compute loss
        loss = criterion(outputs, labels)
        
        # Backpropagate the gradients
        loss.backward()
        
        # Update the parameters
        optimizer.step()
        
        # Compute the total loss for the batch and add it to train_loss
        train_loss += loss.item() * inputs.size(0)
        
        # print("Batch number: {:03d}, Training: Loss: {:.4f}".format(i, loss.item()))

        
    # Validation - No gradient tracking needed
    with torch.no_grad():

        # Set to evaluation mode
        model.eval()

        # Validation loop
        for j, (inputs, labels) in enumerate(test_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            if labels.ndim == 1:
                labels = labels.view(labels.shape[0], 1)

            # Forward pass - compute outputs on input data using the model
            outputs = model(inputs)

            # Compute loss
            loss = criterion(outputs, labels)

            # Compute the total loss for the batch and add it to valid_loss
            valid_loss += loss.item() * inputs.size(0)

            # print("Validation Batch number: {:03d}, Validation: Loss: {:.4f}".format(j, loss.item()))
        
    # Find average training loss
    avg_train_loss = train_loss/len(train_dataset) 

    # Find average training loss 
    avg_valid_loss = valid_loss/len(test_dataset)

    history.append([avg_train_loss, avg_valid_loss])
            
    epoch_end = time.time()

    print("Training loss: {:.4f}, Validation loss : {:.4f}, Time: {:.4f}s".format(avg_train_loss, avg_valid_loss, epoch_end-epoch_start))
    
    # Save the model for every epoch:
    torch.save(model, save_path+'/epoch_'+str(epoch+1).zfill(3)+'.pt')