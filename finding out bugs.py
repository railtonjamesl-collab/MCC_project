# -*- coding: utf-8 -*-
"""
Created on Fri Oct 24 15:42:22 2025

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
    generate 100x100 circle 
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

#genearte a gradient
x= generate_gradient(0,3)
x = ndimage.gaussian_filter(x, 3/(2*np.sqrt(2*np.log(2))))
#generating a circular signal and smoothing it
circle = generate_circle_100grid(30, 3)[0]
circle = ndimage.gaussian_filter(circle, 3/(2*np.sqrt(2*np.log(2))))
#circle = x
c =2
#finding and interpolate true boundary
true_mask = circle > c
true_bdry, true_s1, true_s0 = compute_boundary(true_mask)

true_b_s0 = circle[true_s0[:,0], true_s0[:,1]]
true_b_s1 = circle[true_s1[:,0], true_s1[:,1]]



true_diff = true_b_s1 - true_b_s0

true_m1 = (true_b_s1 - c)/true_diff
true_m2 = (c - true_b_s0)/true_diff



#generating instances  of smoothed circle + smoothed noise
instances = []
#eppy = []
for i in range(3000):
    epsilon = ndimage.gaussian_filter(np.random.randn(100,100), 3/(2*np.sqrt(2*np.log(2))))
    epsilon /= epsilon.std() #renormalise epsilon
    #eppy.append(epsilon)
    instances.append(circle + epsilon) #3fwhm
    
    
#using the fact that Linear regression consist only of intercept. 
#The Intercept's coefficient is just the mean value we have that can compute beta easily
coefficient = np.mean(instances, axis = 0)



#mask for the significant regression coefficient values
mask = coefficient > c
bdry, s1, s0 = compute_boundary(mask)

#register coefficients values at the inside boundary s0 and its neighbour outside boundary s1
b_s0 = coefficient[s0[:,0], s0[:,1]]
b_s1 = coefficient[s1[:,0], s1[:,1]]

diff = np.abs(b_s1 - b_s0)

#weight based on distances the regression coeffcient has from threshold (assume linear distance based on c)
m1 = (b_s1 - c)/diff
m2 = (c - b_s0)/diff



epsilons  = (instances - coefficient)
eps_std = epsilons.std(axis=0, ddof=1)
epsilons = (epsilons)/eps_std

epsilon_star = m1 * epsilons[:,s1[:,0], s1[:,1]] + m2 * epsilons[:,s0[:,0], s0[:,1]]
     
#computing G*
number_subject, number_point = np.shape(epsilon_star)
repeat = 1000

#computing G the original way
G = []
for i in range(repeat):
    rad = rademacher(number_subject)
    rad = np.tile(rad, (number_point,1)).T
    boot_residual = rad * epsilon_star
    std = boot_residual.std(axis=0)
    G_values = np.sum(boot_residual, axis=0)/(std * np.sqrt(number_subject))
    sup = np.max(np.abs(G_values))
    G.append(sup)
"""
#computing G the way alex code it (less efficient)
G2 = []

for i in range(repeat):
    rad = rademacher(number_subject)[:, None, None]
    boot_residual = rad * epsilons
    boot_std = boot_residual.std(axis=0, ddof=1)
    t_field = boot_residual.sum(axis=0) / (boot_std * np.sqrt(number_subject))
    t_star = m1 * t_field[s1[:,0], s1[:,1]] + m2 * t_field[s0[:,0], s0[:,1]]
    G2.append(np.max(np.abs(t_star)))
"""
k = np.quantile(G, 0.95)
#k2 = np.quantile(G2, 0.95)

#weight
vw = 1/np.sqrt(number_subject)

#compute upper_lower region
upper_coeff = coefficient - k * eps_std *vw
lower_coeff = coefficient + k * eps_std *vw

upper_mask = upper_coeff >= c
lower_mask = lower_coeff >= c

upper_region = np.argwhere(upper_mask)
lower_region = np.argwhere(lower_mask) 



#value of upper and lower region at boundaries
upper_s0 = upper_coeff[true_s0[:,0],true_s0[:,1]]
upper_s1 = upper_coeff[true_s1[:,0],true_s1[:,1]]
lower_s0 = lower_coeff[true_s0[:,0],true_s0[:,1]]
lower_s1 = lower_coeff[true_s1[:,0],true_s1[:,1]]

upper_boundary = true_m1 * (upper_s1) + true_m2 * (upper_s0) 
lower_boundary = true_m1 * (lower_s1) + true_m2 * (lower_s0) 

#test if the condition failed
upper_test = upper_boundary >= c # if s* in upper then violate
lower_test = lower_boundary < c #if s* not in lower then violated

test = not (np.any(upper_test) or np.any(lower_test))
    
#contourplot
fig, ax = plt.subplots()

ax.contourf(upper_mask, levels= [0.5,1], colors = 'red', alpha=0.25)
ax.contour(upper_mask, levels = [0.5,1], colors = 'red')

ax.contourf(lower_mask, levels = [0.5,1], colors='blue', alpha=0.25)
ax.contour(lower_mask, levels = [0.5,1], colors = 'blue')

ax.scatter(true_s0[:,1],true_s0[:,0], s=10)
plt.show()
print(test)


