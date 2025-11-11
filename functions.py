# -*- coding: utf-8 -*-
"""
Created on Tue Nov 11 17:01:20 2025

@author: railt
"""

import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt
from numpy.linalg import inv

def rademacher(n):
    """
    generate n sample of 1 or -1 with probability 1/2 for each outcome
    """
    outcome = np.random.choice([-1,1], size = n, replace = True)
    return(outcome)

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
    """
    
    """
    
    center = np.array([50,50])
    x = np.meshgrid(np.arange(100),np.arange(100))
    circle_mask =  (x[0] - center[0])**2 + (x[1] - center[1])**2 <= radius **2
    
    #generating a circle
    circle = np.zeros((100,100))
    signif_point = np.where(circle_mask)
    circle[signif_point] = magnitude
    return(circle, signif_point)
def generate_cross():
    """
    generate a + symbol
    """
    boi = np.zeros((100,100))
    boi[40:61,50] = 6
    boi[50,40:61] = 6
    return(boi)

def generate_gradient(minsig,maxsig):
    """
    generate a gradient signal     
    """

    gradient = np.linspace(minsig,maxsig,100)
    gradient = np.tile(gradient, (100,1))
    return(gradient)

def boundary_dilation(bdry):
    
    #initialise some values
    bdry_dil = np.zeros_like(bdry, dtype = bool)
    #upward dilation
    bdry_dil[:-1,:] |= bdry[1:,:]

    #downward dilation
    bdry_dil[1:,:] |= bdry[:-1,:]
    
    #left
    bdry_dil[:,:-1] |= bdry[:,1:]
    
    #right 
    bdry_dil[:,1:] |= bdry[:,:-1]

    #include original boundary
    bdry_dil |= bdry
    
    return(bdry_dil)

def hausdorff(mask1,mask2):
    
    #initialise some distance
    subset1_yet = False
    subset2_yet = False
    epsilon1 = 0
    epsilon2 = 0
    bdry1 = compute_boundary(mask1)[0]
    bdry2 = compute_boundary(mask2)[0]
    
    #initialise dilation
    bdry1_dil = bdry1
    bdry2_dil = bdry2
    
    while not subset1_yet:
        #check subset
        if np.all(bdry1_dil >= bdry2):
            subset1_yet = True 
        else:
            bdry1_dil = boundary_dilation(bdry1_dil)
            epsilon1 += 1
    
    while not subset2_yet:
        #check subset
        if np.all(bdry2_dil >= bdry1):
            subset2_yet = True
        else:
            bdry2_dil = boundary_dilation(bdry2_dil)
            epsilon2 += 1
    
    haus = max(epsilon1, epsilon2)
    return(haus)
