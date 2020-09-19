# thanks to: https://github.com/cfotache/pytorch_imageclassifier
# thanks to: https://towardsdatascience.com/how-to-train-an-image-classifier-in-pytorch-and-use-it-to-perform-basic-inference-on-single-images-99465a1e9bf5
# thanks to: Deep Learning with PyTorch course from opencv (week 7)
# thanks to: https://discuss.pytorch.org/t/how-to-classify-single-image-using-loaded-net/1411

import torch
from torch import nn
from torch import optim

from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms, models
from torchvision.transforms import functional as F
from torch.autograd import Variable

import matplotlib.pyplot as plt

import os
import numpy as np

from PIL import Image

# specifically needed to work with float32 tiff files
import skimage.transform
import csv

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
            img_dir = os.path.join(data_root, 'test')
            
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
            # min_x: -174.15604
            # max_x: 266.94846
            # min_y: -230.35011
            # max_y: 214.26303
            # min_z: 0.0
            # max_z: 747

            min_x = float(-267)
            max_x = float(267)
            min_y = float(-231)
            max_y = float(231)
            min_z = float(0)
            max_z = float(750)
            
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
data_root = '/home/pieterdeeplearn/harvestcnn/datasets/20201231_size_experiment_realsense/xyz_masks'

preprocess = transforms.Normalize((0.5, 0.5, 0.5, 0.5), (0.5, 0.5, 0.5, 0.5))

test_dataset =  regression_dataset(data_root, train=False, image_shape=224, transform=preprocess)
print('Length of the test dataset: {}'.format(len(test_dataset)))

# use cuda:
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.load('./weights/Dreal_regression_Resnext101_32x8d_xyz_masks_400x400pixels_4channel/epoch_063.pt')
model.to(device)
model.eval()

diameters = []
diffs = []

writedir1 = "/home/pieterdeeplearn/harvestcnn/results/dl_regression_4channel"

with open(os.path.join(writedir1, 'broccoli_diameter_regression.csv'), 'w', newline='') as csvfile:
    csvwriter = csv.writer(csvfile, delimiter=',',quotechar='|', quoting=csv.QUOTE_MINIMAL)
    csvwriter.writerow(['plant_id', 'real-world diameter (mm)', 'diameter regression (mm)', 'difference in diameter (mm)'])


for i in range(len(test_dataset)):
    image, gt = regression_dataset.__getitem__(test_dataset, i)
    image_path = test_dataset.data_dict['image_path'][i]
    plant_id = int(image_path.split("/")[-1].split("_")[2].split("plant")[-1])
    image = Variable(image, requires_grad=True)
    image = image.unsqueeze(0)
    image = image.to(device)

    output = model(image)
    prediction = output.data.cpu().numpy()
    pred = round(float(prediction.ravel()), 1)
    diff = round(float(np.subtract(gt, prediction)), 1)
    diameters.append([gt, pred, diff])
    diffs.append(diff)

    with open(os.path.join(writedir1, 'broccoli_diameter_regression.csv'), 'a', newline='') as csvfile:
        csvwriter = csv.writer(csvfile, delimiter=',',quotechar='|', quoting=csv.QUOTE_MINIMAL)
        csvwriter.writerow([plant_id, round(gt, 1), pred, diff])


np.set_printoptions(formatter={'float_kind':'{:f}'.format})

print("Diameters overview:")
for i in range(len(diameters)):
    print(diameters[i])
print()
print("Average error of D when testing on {0:.0f} broccoli's: {1:.1f} mm".format(len(test_dataset), np.average(np.abs(diffs))))
print("Biggest error of D when testing on {0:.0f} broccoli's: {1:.1f} mm".format(len(test_dataset), np.max(np.abs(diffs))))

error_below5 = np.where(np.logical_and(np.asarray(diffs)>=-5, np.asarray(diffs)<=5))
error_below10 = np.where(np.logical_and(np.asarray(diffs)>=-10, np.asarray(diffs)<=10))

perc_below5 = (len(error_below5[0]) / len(test_dataset)) * 100
perc_below10 = (len(error_below10[0]) / len(test_dataset)) * 100

print()
print("Percentage of estimates that was within 5 mm from the ground truth {0:.2f} %".format(perc_below5))
print("Percentage of estimates that was within 10 mm from the ground truth {0:.2f} %".format(perc_below10))

plt.hist(diffs)
plt.title("Difference between ground truth and regression network")
plt.xlabel("Difference (mm)")
plt.ylabel("Occurence")
plt.show()