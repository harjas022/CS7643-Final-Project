import sys
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from datetime import datetime
from IPython.core.display import display, HTML
import cv2
from PIL import Image
from pathlib import Path
import torchvision
import torch
import torch.nn as nn
import torch.nn.functional as F
import ast

from sklearn.model_selection import train_test_split

from utils import *
from models import CNN

def train(model, optimizer, train_dl, valid_dl, criterion, epochs=20, plot= False, return_loss= False, verbose=False):
    idx = 0
    model= model.float()
    validation_loss_list= []
    training_loss_list= []
    for i in range(epochs):
        model.train()
        sum_loss = 0
        for x, y_bb in train_dl:
            size_of_batch= x.shape[0]
            y_bb= torch.tensor(y_bb)
            if torch.cuda.is_available():
                x= x.cuda().float()
                y_bb= y_bb.cuda().float()
            else:
                x= x.float()
            out_bb = model(x)
            loss= criterion(out_bb, y_bb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            idx += 1
            sum_loss += loss.detach().item()
        train_loss = sum_loss/size_of_batch
        if verbose== True:
            print(" ")
            print("--------------------------------------------------------")
            print("Training Loss for Epoch {0}: {1}".format(i,train_loss))
        valid_loss= validate(model= model, valid_dl= valid_dl, epoch= i, criterion= criterion, verbose= verbose)
        training_loss_list.append(train_loss)
        validation_loss_list.append(valid_loss)

    if plot== True:
        #plot loss curves
        plt.plot(training_loss_list)
        plt.title('Training Loss')
        plt.show()

        plt.plot(validation_loss_list)
        plt.title('Validation Loss')
        plt.show()

    if return_loss== True:
        return training_loss_list, validation_loss_list
    


def validate(model, valid_dl, epoch, criterion, verbose= False):
    idx= 0
    model.eval()
    sum_loss= 0
    for x, y_bb in valid_dl:
        size_of_batch= x.shape[0]
        y_bb= torch.tensor(y_bb)
        if torch.cuda.is_available():
            x= x.cuda().float()
            y_bb= y_bb.cuda().float()
        else:
            x= x.float()
        out_bb= model(x)
        loss= criterion(out_bb, y_bb)
        idx += 1
        sum_loss += loss.detach().item()
    validation_loss= sum_loss/size_of_batch
    if verbose== True:
        print("Validation Loss for Epoch {0}: {1}".format(epoch, validation_loss))
    return validation_loss


## Define Hyperparameters -- Currently setting values that we can modify
def hp_grid_search(model_type, 
                   lr_list, 
                   momentum_list, 
                   reg_list, 
                   batch_size_list, 
                   train_ds,
                   valid_ds,
                   optimizer,
                   epochs,
                   loss_type_list= ["l1"],
                   save_all_plots= "No",
                   save_final_plot= "No",
                   final_plot_prefix= None,
                   return_all_loss= False):
    
    '''
    model (numeric): initialized model to test
    lr_list (list of numeric): list of learning rates
    momentum_list (list of numeric): list of momentums
    reg_list (list of numeric): list of regularization penaltys
    batch_size_list (list of numeric): list of sizes of the batches
    train_ds: training dataset after using WaldoDataset
    valid_ds: validation dataset after using WaldoDataset
    loss_type_list (list of str): list of losses if you want to try more than one
    save_all_plots (str): Do you want to save every plot? default to "No" 
    save_final_plot (str): if you just want to save the final plot.  default to "No".  Final plot will automaticall save if "save_all_plots"==Yes
    final_plot_prefix (str): provide a prefix for the final plot name
    '''

    i=0
    all_loss_train= []
    all_loss_valid= []
    
    for lr in lr_list:
        for r in reg_list:
            for m in momentum_list:
                for b in batch_size_list:
                    for loss_type in loss_type_list:

                        print('HP ITERATION: ', i)
                        i+=1
                        print('learning_rate: ', lr)
                        print('regularization: ', r)
                        print('momentum: ', m)
                        print('batch_size: ', b)
                        print('loss type: ', loss_type)
                        
                        param_str= "{0}_{1}_{2}_{3}_{4}".format(model_type, str(lr), str(r), str(m), str(b), loss_type)
                        print(param_str)

                        if loss_type == "l1":
                            criterion = nn.L1Loss()

                        if loss_type == "l2":
                            criterion = nn.MSELoss()

                        if model_type== "SimpleCNN":
                            model= CNN()
            
                        train_dl= DataLoader(train_ds, batch_size=b, shuffle=True)
                        valid_dl= DataLoader(valid_ds, batch_size=b)
                        
                        optimizer = torch.optim.SGD(model.parameters(), lr, momentum=m, weight_decay=r)
                        
                        train_loss, valid_loss= train(model= model,optimizer= optimizer,train_dl= train_dl, valid_dl= valid_dl, epochs= epochs, criterion= criterion, return_loss= True,plot= True,verbose= True)
                        
                        all_loss_train.append(train_loss)
                        all_loss_valid.append(valid_loss)

                        plt.plot(valid_loss)
                        plt.title('Validation Loss')
                        plt.xlabel('Epoch')
                        plt.ylabel('Perplexity')
                        if save_all_plots== "Yes":
                            print('./figures/V_{0}.png'.format(param_str))
                            plt.savefig('./figures/V_{0}.png'.format(param_str))
                        plt.show()

                        plt.plot(train_loss)
                        plt.title('Training Loss')
                        plt.xlabel('Epoch')
                        plt.ylabel('Loss')
                        if save_all_plots== "Yes":
                            plt.savefig('./figures/T_{0}.png'.format(param_str))
                        plt.show()
    
    for pt in all_loss_train:
        plt.plot(pt)
    plt.title('All Plots Training')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    if save_final_plot=="Yes":
        plt.savefig('./figures/{0} All Training Loss.png'.format(final_plot_prefix))
    plt.show()
    
    for pv in all_loss_valid:
        plt.plot(pv)
    plt.title('All Plots Validation')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    if save_final_plot== "Yes":
        plt.savefig('./figures/{0}All Validation Loss.png'.format(final_plot_prefix))
    plt.show()
        
    if return_all_loss== True:
        return all_loss_train, all_loss_valid