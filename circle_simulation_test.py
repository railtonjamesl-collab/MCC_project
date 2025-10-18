# -*- coding: utf-8 -*-
"""
Created on Sat Oct 18 12:44:38 2025

@author: railt
"""

import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt
from numpy.linalg import inv

def compute_boundary(mask):
    """
    

    Parameters
    ----------
    mask : 2d array of boolean value
        array of 2d values of siginificant region of interest

    Returns
    -------
    boundary points, s1 = inside boundary, s0 = outside boundary

    """
    bdry = np.zeros_like(mask, dtype = bool)

    s0=[] #outside
    s1=[] #inside

    up = (mask[1:,:] & ~mask[:-1,:])
    bdry[1:,:] |= up
    coord = np.argwhere(up)
    s0.append(coord )
    s1.append(coord + [1,0])

    down = (~mask[1:,:] & mask[:-1,:]) 
    bdry[:-1,:] |= down
    coord = np.argwhere(down)
    s0.append(coord + [1,0])
    s1.append(coord)

    left = (mask[:,1:] & ~mask[:,:-1])
    bdry[:,1:] |= left 
    coord = np.argwhere(left)
    s0.append(coord)
    s1.append(coord + [0,1])

    right = (~mask[:,1:] & mask[:,:-1])
    bdry[:,:-1] |= (~mask[:,1:] & mask[:,:-1])
    coord=np.argwhere(right)
    s0.append(coord + [0,1])
    s1.append(coord)

    s0 = np.concatenate(s0)
    s1 = np.concatenate(s1)
    return(bdry, s1, s0)

def generate_circle_100grid(radius, magnitude):
    center = np.array([50,50])
    x = np.meshgrid(np.arange(100),np.arange(100))
    circle_mask =  (x[0] - center[0])**2 + (x[1] - center[0])**2 <= radius **2
    
    #generating a circle
    circle = np.zeros((100,100))
    signif_point = np.where(circle_mask)
    circle[signif_point] = magnitude
    return(circle, signif_point)

circle, signif_point = generate_circle_100grid(25,3)

instances = []
for i in range(100):
    epsilon = np.random.rand(100,100) * 2
    instances.append(ndimage.gaussian_filter((circle + epsilon), 3))
for i in range(5):
    plt.figure()
    plt.imshow(instances[i])
    plt.colorbar()
    
