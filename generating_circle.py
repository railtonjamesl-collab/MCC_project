# -*- coding: utf-8 -*-
"""
Created on Sat Oct 18 12:01:47 2025

@author: railt
"""

#Creating circle

import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage

#generating a mask centered at 50,50
center = np.array([50,50])
x = np.meshgrid(np.arange(100),np.arange(100))
circle_mask =  (x[0] - center[0])**2 + (x[1] - center[0])**2 <= 25 **2

#generating a circle
circle = np.zeros((100,100))
signif_point = np.where(circle_mask)
circle[signif_point] = 3
circle = ndimage.gaussian_filter(circle,2)
plt.figure()
plt.imshow(circle)
plt.colorbar()

