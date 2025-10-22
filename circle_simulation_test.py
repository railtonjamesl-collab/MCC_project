# -*- coding: utf-8 -*-
"""
Created on Sat Oct 18 12:44:38 2025

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
for i in range(1000):
    epsilon = np.random.randn(100,100)
    instances.append(ndimage.gaussian_filter((circle + epsilon), 5/(2*np.sqrt(2*np.log(2))))) #3fwhm
    
    
#using the fact that Linear regression consist only of intercept's coefficient is just
#the mean value we have that can compute beta easily

coefficient = np.mean(instances, axis = 0)

c = 2

mask = coefficient > 2

bdry, s1, s0 = compute_boundary(mask)

plt.figure()
plt.imshow(coefficient)
plt.colorbar()

plt.figure()
plt.imshow(mask)
plt.colorbar()

plt.figure()
plt.imshow(bdry)
plt.colorbar()

b_s0 = coefficient[s0[:,0], s0[:,1]]
b_s1 = coefficient[s1[:,0], s1[:,1]]

diff = b_s1 - b_s0

m1 = (b_s1 - c)/diff
m2 = (c - b_s0)/diff

epsilons  = (instances - coefficient)
eps_std = epsilons.std(axis=0)
epsilons = (epsilons)/eps_std

epsilon_star = m1 * epsilons[:,s1[:,0], s1[:,1]] + m2 * epsilons[:,s0[:,0], s0[:,1]]

#computing G*
number_subject, number_point = np.shape(epsilon_star)
repeat = 1000
G = []
for i in range(repeat):
    rad = rademacher(number_subject)
    rad = np.tile(rad, (number_point,1)).T
    boot_residual = rad * epsilon_star
    std = boot_residual.std(axis=0)
    G_values = np.sum(boot_residual, axis=0)/(std * np.sqrt(number_subject))
    sup = np.max(G_values)
    G.append(sup)
    
plt.figure()
plt.hist(G)
k = np.quantile(G, 0.99)

vw = 1

upper_coeff = coefficient - k * eps_std * vw
lower_coeff = coefficient + k * eps_std * vw

upper_mask = upper_coeff >= c
lower_mask = lower_coeff >= c

upper_region = np.argwhere(upper_mask)
lower_region = np.argwhere(lower_mask) 

fig, ax = plt.subplots()

ax.contourf(lower_mask, levels = [0.75,1], colors = 'blue', alpha = 0.25)
ax.contour(lower_mask, levels=[0.75], colors= 'blue')

ax.contourf(upper_mask, levels = [0.75,1], colors = 'red', alpha=0.25)
ax.contour(upper_mask, levels=[0.75], colors = 'red')

ax.scatter(s0[1][0],s0[1][1], s = 10, color = 'green')
plt.show()

#interpolating boundary

upper_s0 = upper_coeff[s0[:,0],s0[:,1]]
upper_s1 = upper_coeff[s1[:,0],s1[:,1]]
lower_s0 = lower_coeff[s0[:,0],s0[:,1]]
lower_s1 = lower_coeff[s1[:,0],s1[:,1]]

upper_boundary = m1 * (upper_s1) + m2 * (upper_s0) # if s* in upper the violocate
lower_boundary = m1 * (lower_s1) + m2 * (lower_s0) #if s* not in lower then violated

upper_test = upper_boundary >= c
lower_test = lower_boundary < c




