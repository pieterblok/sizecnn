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

# specifically needed to work with float32 tiff files
import skimage.transform

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
            if img.endswith(".jpg") or img.endswith(".png") or img.endswith(".tiff") or img.endswith(".npy"):
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
        
        with open(self.data_dict['image_path'][idx], 'rb') as f:
            image = np.load(f)
        
        if self.image_shape is not None:
            # alternative for transforms.Resize
            image = skimage.transform.resize(image, self.image_shape, preserve_range=True)
            
  
        if self.transform is not None:
            # alternative for transforms.CenterCrop  
            width, height = image.shape[:2]
            startx = int((width - 224)/2)
            starty = int((height - 224)/2)
            image = image[startx:startx+224, starty:starty+224, :]

            # alternative for transforms.ToTensor() which includes normalization between 0 and 1
            
            # from all masks these are the extremes:
            # min_x = float(-173.15782)
            # max_x = float(266.65762)
            # min_y = float(-266.70273)
            # max_y = float(210.61345)
            # min_z = float(0)
            # max_z = float(751.0)

            min_x = float(-267)
            max_x = float(267)
            min_y = float(-267)
            max_y = float(267)
            min_z = float(0)
            max_z = float(751)
            
            image[:,:,0] = (image[:,:,0] - min_x) / (max_x - min_x)
            image[:,:,1] = (image[:,:,1] - min_y) / (max_y - min_y)
            image[:,:,2] = (image[:,:,2] - min_z) / (max_z - min_z)

            image = np.transpose(image,(2,0,1))
            image = torch.from_numpy(image)

            # transforms.Normalize
            image = self.transform(image)

            
        target = self.data_dict['label'][idx]
        
        return image, target           


# data root directory
data_root = '/home/pieterdeeplearn/harvestcnn/datasets/20200713_size_experiment_realsense/xyz_masks'

preprocess = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

test_dataset =  regression_dataset(data_root, train=False, image_shape=256, transform=preprocess)
print('Length of the test dataset: {}'.format(len(test_dataset)))
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=100, shuffle=True, num_workers=2)

# use cuda:
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.load('./weights/D_regression_xyz_masks/epoch_046.pt')
model.to(device)
model.eval()

for inputs, labels in test_loader:
    inputs, labels = inputs.to(device), labels.to(device)

    if labels.ndim == 1:
        labels = labels.view(labels.shape[0], 1)

    output = model(inputs)

np.set_printoptions(formatter={'float_kind':'{:f}'.format})

if output.shape[1] == 1:
    real_d = labels.data.cpu().numpy()
    real_d = real_d.astype(np.float32)
    predicted_d = output.data.cpu().numpy()
    diff_d = np.subtract(real_d, predicted_d)

    overview_d = np.concatenate((real_d, predicted_d, diff_d), axis=1)

    print("Overview D:")
    print(overview_d)
    print()
    print("Average error of D when testing on {0:.0f} broccoli's: {1:.1f} mm".format(len(test_dataset), np.average(np.abs(diff_d))))
    print("Biggest error of D when testing on {0:.0f} broccoli's: {1:.1f} mm".format(len(test_dataset), np.max(np.abs(diff_d))))
    
    plt.hist(diff_d, 30)
    plt.show()


elif output.shape[1] == 4:
    real_x = labels[:,0].data.cpu().numpy()
    predicted_x = output[:,0].data.cpu().numpy()
    diff_x = np.subtract(real_x, predicted_x)
    overview_x = np.concatenate((np.expand_dims(real_x, axis=1), np.expand_dims(predicted_x, axis=1), np.expand_dims(diff_x, axis=1)), axis=1)
    print("Overview X:")
    print(overview_x)
    print()

    real_y = labels[:,1].data.cpu().numpy()
    predicted_y = output[:,1].data.cpu().numpy()
    diff_y = np.subtract(real_y, predicted_y)
    overview_y = np.concatenate((np.expand_dims(real_y, axis=1), np.expand_dims(predicted_y, axis=1), np.expand_dims(diff_y, axis=1)), axis=1)
    print("Overview Y:")
    print(overview_y)
    print()

    real_z = labels[:,2].data.cpu().numpy()
    predicted_z = output[:,2].data.cpu().numpy()
    diff_z = np.subtract(real_z, predicted_z)
    overview_z = np.concatenate((np.expand_dims(real_z, axis=1), np.expand_dims(predicted_z, axis=1), np.expand_dims(diff_z, axis=1)), axis=1)
    print("Overview Z:")
    print(overview_z)
    print()

    real_d = labels[:,3].data.cpu().numpy()
    predicted_d = output[:,3].data.cpu().numpy()
    diff_d = np.subtract(real_d, predicted_d)
    overview_d = np.concatenate((np.expand_dims(real_d, axis=1), np.expand_dims(predicted_d, axis=1), np.expand_dims(diff_d, axis=1)), axis=1)
    print("Overview D:")
    print(overview_d)
    print()

    print("Average error of X when testing on {0:.0f} broccoli's: {1:.1f} mm".format(len(test_dataset), np.average(np.abs(diff_x))))
    print("Average error of Y when testing on {0:.0f} broccoli's: {1:.1f} mm".format(len(test_dataset), np.average(np.abs(diff_y))))
    print("Average error of Z when testing on {0:.0f} broccoli's: {1:.1f} mm".format(len(test_dataset), np.average(np.abs(diff_z))))
    print("Average error of D when testing on {0:.0f} broccoli's: {1:.1f} mm".format(len(test_dataset), np.average(np.abs(diff_d))))