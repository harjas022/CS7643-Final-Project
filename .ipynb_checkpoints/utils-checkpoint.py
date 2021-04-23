import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import cv2
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torch
import ast

# https://stackoverflow.com/questions/42755214/how-to-keep-numpy-array-when-saving-pandas-dataframe-to-csv is where I got the converter code from
def from_np_array(array_string):
    array_string = ','.join(array_string.replace('[ ', '[').split())
    return np.array(ast.literal_eval(array_string))

def print_image(img_num, annotations_path="./annotations.csv", images_path="./images/", figsize=(10,10)):
    '''
    print a waldo image with waldo highlighted
    img_num (string): the number labeling of the image
    path (string): path to the annotation dataset
    '''
    annotations= pd.read_csv(annotations_path)
    img = Image.open(images_path+img_num+'.jpg')
#     img_ann = annotations[annotations['filename'] == img_num+'.jpg']
    img_anno = annotations[annotations['filename'].str.split('/', expand=True).iloc[:,-1] == img_num + '.jpg']
    rectangles = []
    for i in img_anno.index:
        
        xmin= img_anno.loc[i]['xmin']
        ymin= img_anno.loc[i]['ymin']
        xmax= img_anno.loc[i]['xmax']
        ymax= img_anno.loc[i]['ymax']
        # print("Xmin: ", xmin)
        width= xmax - xmin ## Change these to float types
        height= ymax - ymin ## Changed these to float types
        rectangles.append([xmin, ymin, xmax, ymax, width, height])
        
    
    # Create figure and axes
    fig, ax = plt.subplots(figsize= figsize)

    # Display the image
    ax.imshow(img)
    
    for r in rectangles:
        ax.add_patch(patches.Rectangle((r[0], r[1]), 
                                       r[4], 
                                       r[5], 
                                       linewidth=3, 
                                       edgecolor='r', facecolor='none'))
    plt.show()


def save_annotated_images(img_num, annotations_path="./annotations.csv", images_path="./images/", figsize=(10,10)):
    '''
    print a waldo image with waldo highlighted
    img_num (string): the number labeling of the image
    path (string): path to the annotation dataset
    '''
    annotations= pd.read_csv(annotations_path)
    img = Image.open(images_path+img_num+'.jpg')
    img_anno = annotations[annotations['filename'].str.split('/', expand=True).iloc[:,-1] == img_num + '.jpg']
    rectangles = []
    for i in img_anno.index:
        xmin= img_anno.loc[i]['xmin']
        ymin= img_anno.loc[i]['ymin']
        xmax= img_anno.loc[i]['xmax']
        ymax= img_anno.loc[i]['ymax']
        width= xmax-xmin
        height= ymax-ymin
        rectangles.append([xmin, ymin, xmax, ymax, width, height])
        
    
    # Create figure and axes
    fig, ax = plt.subplots(figsize= figsize)

    for r in rectangles:
        ax.add_patch(patches.Rectangle((r[0], r[1]), 
                                       r[4], 
                                       r[5], 
                                       linewidth=3, 
                                       edgecolor='r', facecolor='none'))
    plt.savefig('./images_annotated/' + img_num + '.jpg')
    

# https://towardsdatascience.com/bounding-box-prediction-from-scratch-using-pytorch-a8525da51ddc
def read_image(path):
    return cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)

# https://towardsdatascience.com/bounding-box-prediction-from-scratch-using-pytorch-a8525da51ddc
def create_mask(bb, x):
    """Creates a mask for the bounding box of same shape as image"""
    rows,cols,*_ = x.shape
    Y = np.zeros((rows, cols))
    bb = bb.astype(np.int)
    Y[bb[0]:bb[2], bb[1]:bb[3]] = 1.
    return Y

# https://towardsdatascience.com/bounding-box-prediction-from-scratch-using-pytorch-a8525da51ddc
def mask_to_bb(Y):
    """Convert mask Y to a bounding box, assumes 0 as background nonzero object"""
    cols, rows = np.nonzero(Y)
    if len(cols)==0: 
        return np.zeros(4, dtype=np.float32)
    top_row = np.max(rows)
    left_col = np.max(cols)
    bottom_row = np.min(rows)
    right_col = np.min(cols)
    return np.array([left_col, top_row, right_col, bottom_row], dtype=np.float32)

# https://towardsdatascience.com/bounding-box-prediction-from-scratch-using-pytorch-a8525da51ddc
def create_bb_array(x):
    """Generates bounding box array from a train_df row"""
    return np.array([x[5],x[4],x[7],x[6]])

# inspired by https://towardsdatascience.com/bounding-box-prediction-from-scratch-using-pytorch-a8525da51ddc
def resize_image_bb(read_path,write_path,bb,sz):
    """
    Resize an image and its bounding box and write image to new path
    read_path: directory path to an image
    write_path: the new path that the resized
    """
    
    new_h= sz
    new_w= sz
    
    im = read_image(read_path)
    im_height, im_width, _ = im.shape
    im_resized = cv2.resize(im, (new_h, new_w))
    im_resized_height, im_resized_width, _ = im_resized.shape
    
    ## calculate y and x positions for new bb
    bb_ymin= bb[0]
    bb_xmin= bb[1]
    bb_ymax= bb[2]
    bb_xmax= bb[3]

    x_scale= new_w / im_width
    y_scale= new_h / im_height
    
    new_bb_xmin= int(np.round(bb_xmin * x_scale))
    new_bb_ymin= int(np.round(bb_ymin * y_scale))
    new_bb_xmax= int(np.round(bb_xmax * x_scale))
    new_bb_ymax= int(np.round(bb_ymax * y_scale))
    
    new_bb= [new_bb_xmin, new_bb_ymin, new_bb_xmax, new_bb_ymax]
    new_bb = np.array(new_bb) ## NEw Code
    
    new_path = str(write_path/read_path.split("/")[-1])
    cv2.imwrite(new_path, cv2.cvtColor(im_resized, cv2.COLOR_RGB2BGR))
    return new_path, new_bb

def normalize(img):
    '''normalize images with mean centered at 0'''
    img= (img - img.mean()) / img.std()
    return img

class WaldoDataset(Dataset):
    def __init__(self, paths, bb, y, transforms=False):
        self.paths= paths.values
        self.bb= bb.values
        self.y= y.values
        
    def __len__(self):
        return len(self.paths)
    
    def __getitem__(self, idx):
        path= self.paths[idx]
        y_bb= self.bb[idx]
        x= read_image(path)
        x= normalize(x)
        return x, y_bb