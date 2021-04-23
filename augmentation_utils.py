import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import cv2
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from matplotlib import transforms
import scipy
import glob
import re
import csv
import math


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

    # For some reason if you don't show the images you can't save them properly
    ax.imshow(img)

    for r in rectangles:
        ax.add_patch(patches.Rectangle((r[0], r[1]), 
                                       r[4], 
                                       r[5], 
                                       linewidth=3, 
                                       edgecolor='r', facecolor='none'))
    plt.savefig('./images_annotated/' + img_num + '.jpg')


def save_rotated_annotations(read_annotations_path='./annotations_map_resized.csv', save_annotations_path='./annotations_rotated.csv'):
    '''
    Takes in the resized annotations csv, calculates the xmin, ymin, xmax, and ymax values after each rotation, and writes this to the save_annotations_path csv.
    read_annotations_path (string): path to the input csv file. Typical use is the resized annotations, but it should work with any annotations file
    save_annotations_path (string): path to the annotations csv to write to
    '''
    annotations = pd.read_csv(read_annotations_path)

    with open(save_annotations_path, mode='w') as csv_file:
        fieldnames = ['filename','width','height','class','xmin','ymin','xmax','ymax']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

        for i in range(len(annotations)):
            for rotation in ([0, 90, 180, 270]):
                # not sure why the 360 - rotation is necessary
                radians = np.deg2rad(360 - rotation)
                file_name = re.findall('\d+', annotations.iloc[i,0])
                origin_x = annotations.iloc[i,1]/2
                origin_y = annotations.iloc[i,2]/2
                writer.writerow({
                    'filename': 'images_resized_rotated/' + file_name[0] + '_' + str(rotation) + '.jpg',
                    'width': annotations.iloc[i,1],
                    'height': annotations.iloc[i,2],
                    'class': annotations.iloc[i,3],
                    'xmin': round(origin_x + math.cos(radians) * (annotations.iloc[i,4] - origin_x) - math.sin(radians) * (annotations.iloc[i,5] - origin_y)),
                    'ymin': round(origin_y + math.sin(radians) * (annotations.iloc[i,4] - origin_x) + math.cos(radians) * (annotations.iloc[i,5] - origin_y)),
                    'xmax': round(origin_x + math.cos(radians) * (annotations.iloc[i,6] - origin_x) - math.sin(radians) * (annotations.iloc[i,7] - origin_y)),
                    'ymax': round(origin_y + math.sin(radians) * (annotations.iloc[i,6] - origin_x) + math.cos(radians) * (annotations.iloc[i,7] - origin_y))
                })


def save_flipped_annotations(read_annotations_path='./annotations_rotated.csv', save_annotations_path='./annotations_rotated_flipped.csv'):
    '''
    print a waldo image with waldo highlighted
    img_num (string): the number labeling of the image
    path (string): path to the annotation dataset
    '''
    annotations = pd.read_csv(read_annotations_path)

    with open(save_annotations_path, mode='w') as csv_file:
        fieldnames = ['filename','width','height','class','xmin','ymin','xmax','ymax']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

        for i in range(len(annotations)):
            file_name = re.findall('[^\/]+\d', annotations.iloc[i,0])
            writer.writerow({
                'filename': file_name[0] + '_flipped.jpg',
                'width': annotations.iloc[i,1],
                'height': annotations.iloc[i,2],
                'class': annotations.iloc[i,3],
                'xmin': annotations.iloc[i,1] - annotations.iloc[i,4],
                'ymin': annotations.iloc[i,5],
                'xmax': annotations.iloc[i,1] - annotations.iloc[i,6],
                'ymax': annotations.iloc[i,7]
            })



def save_rotated_images(images_path='./images_resized/*.jpg', save_folder='images_resized_rotated'):
    '''
    Takes images as input and saves rotated images
    images_path (string): glob path to the images to be flipped. Usually should be resized images, but can handle anything
    save_folder (string): path to foleder for saving the images
    '''
    for image in glob.iglob(images_path):
        for rotation in ([0, 90, 180, 270]):
            img = Image.open(image)
            img = img.rotate(rotation)
            file_name = re.findall('\d+', image)
            img.save('./' + save_folder + '/' + file_name[0] + '_' + str(rotation) + '.jpg')


def save_flipped_images(images_path='./images_resized_rotated/*.jpg', save_folder='images_resized_flipped'):
    '''
    Takes annotated images as input and saves left right flipped images.
    Since we are flipping the rotations as well, we only need a left right flip. The top bottom flip is covered in this.
    images_path (string): glob path to the images to be flipped. Images should be of the name <img_num>_<rotation_angle>.jpg.
    save_folder (string): path to foleder for saving the images
    '''
    for image in glob.iglob(images_path):
        img = Image.open(image)
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
        file_name = re.findall('[^\/]+\d', image)
        img.save('./' + save_folder + '/' + file_name[0] + '_flipped.jpg')
    