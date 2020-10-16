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
from pyexcel_ods import get_data

# specifically needed to work with float32 tiff files
import tifffile
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
            img_dir = os.path.join(data_root, 'train')
            
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
        
        image = tifffile.imread(self.data_dict['image_path'][idx])
        
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
            # min_x: -280.19092
            # max_x: 321.93063
            # min_y: -256.23294
            # max_y: 282.15326
            # min_z: 0.0
            # max_z: 1090.2516

            min_x = float(-285)
            max_x = float(330)
            min_y = float(-260)
            max_y = float(285)
            min_z = float(0)
            max_z = float(1100)
            
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
data_root = '/home/pieterdeeplearn/harvestcnn/datasets/train_val_test_files/xyz_masks'

preprocess = transforms.Normalize((0.5, 0.5, 0.5, 0.5), (0.5, 0.5, 0.5, 0.5))

test_dataset =  regression_dataset(data_root, train=False, image_shape=224, transform=preprocess)
print('Length of the test dataset: {}'.format(len(test_dataset)))

# use cuda:
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.load('./weights/Dreal_regression_Resnext101_32x8d_unfiltered_xyz_masks_600x600pixels_4channel/epoch_096.pt')
model.to(device)
model.eval()

diameters = []
diffs = []
vprs = []
gtsizes = []

writedir1 = "./results/dl_regression_4channel_unfiltered"

with open(os.path.join(writedir1, 'broccoli_diameter_regression.csv'), 'w', newline='') as csvfile:
    csvwriter = csv.writer(csvfile, delimiter=',',quotechar='|', quoting=csv.QUOTE_MINIMAL)
    csvwriter.writerow(['image_id', 'real-world diameter (mm)', 'diameter regression (mm)', 'difference in diameter (mm)', 'visible pixel ratio'])

vprfile = '/home/pieterdeeplearn/harvestcnn/datasets/train_val_test_files/visible_pixel_ratios.ods'
vpr = get_data(vprfile)

for i in range(len(test_dataset)):
    image, gt = regression_dataset.__getitem__(test_dataset, i)
    image_path = test_dataset.data_dict['image_path'][i]

    image_name, file_extension = os.path.splitext(image_path)
    basename = os.path.basename(image_name)
    xyzname = basename + ".tiff"
    rgbname = basename.replace("xyz", "rgb")
    rgbname = rgbname + ".png"

    for k in range(1, len(vpr['visible_pixel_ratios'])):
        vpr_data = vpr['visible_pixel_ratios'][k]
        if len(vpr_data) > 0 :
            rgbname_read = vpr_data[0]
            
            if rgbname == rgbname_read:
                vpr_rgb = vpr_data[2]
                vprs.append(vpr_rgb)

    image = Variable(image, requires_grad=True)
    image = image.unsqueeze(0)
    image = image.to(device)

    output = model(image)
    prediction = output.data.cpu().numpy()
    pred = round(float(prediction.ravel()), 1)
    diff = round(float(np.subtract(gt, prediction)), 1)
    diameters.append([gt, pred, diff])
    gtsizes.append(gt)
    diffs.append(diff)

    with open(os.path.join(writedir1, 'broccoli_diameter_regression.csv'), 'a', newline='') as csvfile:
        csvwriter = csv.writer(csvfile, delimiter=',',quotechar='|', quoting=csv.QUOTE_MINIMAL)
        csvwriter.writerow([xyzname, round(gt, 1), pred, diff, vpr_rgb])


if diffs:
    np.set_printoptions(formatter={'float_kind':'{:f}'.format})

    print("Average error when sizing {0:.0f} broccoli heads: {1:.1f} mm".format(len(test_dataset), np.average(np.abs(diffs))))
    print("Biggest error when sizing {0:.0f} broccoli heads: {1:.1f} mm".format(len(test_dataset), np.max(np.abs(diffs))))

    def ceil_to_25(x, base=25):
        if x >= 0:
            rounded = base * np.ceil(x/base)
        else:
            rounded = base * np.floor(x/base)
        return rounded

    min_bin = ceil_to_25(-np.max(np.abs(diffs)))
    max_bin = ceil_to_25(np.max(np.abs(diffs)))
    bin_range = np.abs(max_bin - min_bin)
    
    digit_size = 12
    text_size = 15
    
    
    ## Plot the diameter error in a histogram
    bins = list(np.arange(min_bin, max_bin + (bin_range/10), bin_range/10))
    counts, bins, patches = plt.hist(diffs, bins)
    try:
        plt.xticks(range(int(min_bin), int(max_bin) + int(bin_range/10), int(bin_range/10)), fontsize=digit_size)
        plt.yticks(range(0, int(np.max(counts)+10), int(np.max(counts)/10)), fontsize=digit_size)
    except:
        plt.xticks(fontsize=digit_size)
        plt.yticks(fontsize=digit_size)
    plt.grid(axis='y', alpha=0.75)
    plt.title("Diameter error from the ground truth", fontsize=text_size)
    plt.xlabel("Diameter error (mm)", fontsize=text_size)
    plt.ylabel("Frequency", fontsize=text_size)

    bin_centers = 0.5 * np.diff(bins) + bins[:-1]
    for count, x in zip(counts, bin_centers):
        if count < 10 :
            plt.annotate('n={:.0f}'.format(count), (x-6, count+2))
        elif count < 100:
            plt.annotate('n={:.0f}'.format(count), (x-7, count+2))
        else:
            plt.annotate('n={:.0f}'.format(count), (x-8, count+2))

    plt.show()
    

    ## Plot the diameter error as a function of the occlusion rate (scatter plot)
    occlusion_perc =  [(1-ele)*100 for ele in vprs]
    diffs_abs =  [abs(ele) for ele in diffs]
    plt.plot(occlusion_perc, diffs_abs, 'o', color='blue', alpha=0.75)
    plt.xticks(range(0, 110, 10), fontsize=digit_size)
    try:
        plt.yticks(range(0, int(max_bin), int(max_bin/10)), fontsize=digit_size)
    except:
        plt.yticks(fontsize=digit_size)
    plt.title("Diameter error as a function of the occlusion rate", fontsize=text_size)
    plt.xlabel("Occlusion rate (%)", fontsize=text_size)
    plt.ylabel("Absolute error on diameter (mm)", fontsize=text_size)
    plt.show()
    

    ## Plot the diameter error as a function of the broccoli size (scatter plot)
    plt.plot(gtsizes, diffs_abs, 'o', color='blue', alpha=0.75)
    plt.xticks(range(50,275,25), fontsize=digit_size)
    try:
        plt.yticks(range(0, int(max_bin), int(max_bin/10)), fontsize=digit_size)
    except:
        plt.yticks(fontsize=digit_size)
    plt.title("Diameter error as a function of the broccoli size", fontsize=text_size)
    plt.xlabel("Ground truth size of the broccoli head (mm)", fontsize=text_size)
    plt.ylabel("Absolute error on diameter (mm)", fontsize=text_size)
    plt.show()