# -*- coding: utf-8 -*-
"""
Created on Fri Oct 17 10:58:15 2025

@author: railt
"""

import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt
from numpy.linalg import inv

ep = np.random.randn(100,100) #+ 10
ep = ndimage.gaussian_filter(ep, 2)

mask = ep >= 0.1

plt.figure()
plt.imshow(ep)
plt.colorbar()

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

plt.figure()
plt.imshow(bdry)
plt.colorbar()





