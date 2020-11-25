#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 14 12:12:49 2020

@author: peterg
"""
from load_data import load_data
import pandas as pd
import os
import torch
import numpy as np
import scipy.stats as ss
from scipy.stats import genextreme as gev
import matplotlib.pyplot as plt
import nibabel as nib
from nilearn import plotting

def plot_gev(abs_only = True, critical = 0.05, show_outliers = True, plot = True):
    path = "./../../../../mnt/projects/GPR_LISA"
    X, Y, index, shape, labels = load_data(threshold = 200)
    index2 = ~labels.duplicated(keep = 'last').values  # Remove duplicates
    
    X = X[:, index2, :]
    y = Y[index, :]
    y1 = (y - y.mean(0))/y.std(0)
    y = y1[:, index2]
    
    
    filenames_mean = sorted(os.listdir(path + "/Data/pred_mean"))
    filenames_var  = sorted(os.listdir(path + "/Data/pred_var"))
    
    mean = torch.cat([torch.load(path + "/Data/pred_mean/" + name).unsqueeze(0) for name in filenames_mean]).T
    var  = torch.cat([torch.load(path + "/Data/pred_var/" + name).unsqueeze(0) for name in filenames_var]).T
 
    perc1 = index.sum() // 100
    Z = (y-mean)/torch.sqrt(var)
    if not abs_only:
        Z_top1_pos  = torch.topk(Z, perc1, 0, largest = True).values
        Z_top1_neg  = torch.topk(Z, perc1, 0, largest = False).values
        Z_top1_abs  = torch.topk(torch.abs(Z),perc1, 0, largest = True).values
       
        Z_scores = [Z_top1_pos, -1*Z_top1_neg, Z_top1_abs]
        fig = plt.figure(figsize = (30,10))
    
        for i in range(3):
            fig.add_subplot(1,3,i+1)
            trim_mean = ss.trim_mean(Z_scores[i], 0.05)
            fit = gev.fit(trim_mean)
            x = np.linspace(0,10, 1000)
            plt.plot(x, gev.pdf(x, *fit))
            plt.hist(trim_mean, density = True, bins = 40)
        plt.show()
    else:
        Z_top1_abs  = torch.topk(torch.abs(Z),perc1, 0, largest = True).values

        trim_mean = ss.trim_mean(Z_top1_abs, 0.05)
        fit = gev.fit(trim_mean)
        x = np.linspace(0,10, 1000)
        plt.hist(trim_mean, density = True, bins = 30)
        plt.plot(x, gev.pdf(x, *fit),)
        if show_outliers:
            plt.fill_between(x[gev.sf(x,*fit)<0.05], gev.pdf(x[gev.sf(x,*fit)<critical], *fit),
                             color = "red", alpha = 0.6, zorder = 3)
            
       
        plt.show()
        
      
    return trim_mean, fit, mean, var, y, shape, index, labels
def plot_tensor(tensor, index, shape, vmin = -3, vmax = 3, threshold = 0):
    path = "./../../../../mnt/projects/GPR_LISA/ASL/J878_ASL_CBF_Tmean_mni.nii"
    obj = nib.load(path)
    Z = torch.zeros(index.size())
    Z[index] = tensor
    Z = Z.reshape(shape)
    img = nib.Nifti1Image(Z.numpy(), obj.affine)
    plotting.plot_img(img, colorbar = True, threshold=threshold, vmin = vmin, vmax = vmax)
    
def get_outliers(tm, fit, fdr = 0.05):
    p_vals = gev.sf(tm, *fit)
    q_vals = (np.argsort(p_vals) + 1)/len(p_vals) * fdr
    
    return p_vals < q_vals, p_vals, q_vals
