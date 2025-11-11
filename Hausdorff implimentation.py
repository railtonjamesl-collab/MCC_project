# -*- coding: utf-8 -*-
"""
Created on Tue Nov 11 12:11:44 2025

@author: railt
"""
import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt
from numpy.linalg import inv
from functions import generate_circle_100grid, hausdorff, boundary_dilation, compute_boundary, rademacher
#initialise some parameter!
thr = 2
boot_rep = 100

#generating a circular signal and smoothing it
circle = generate_circle_100grid(30, 3)[0]
circle = ndimage.gaussian_filter(circle, 3/(2*np.sqrt(2*np.log(2))))

instances = []
#eppy = []
for i in range(200):
    epsilon = ndimage.gaussian_filter(np.random.randn(100,100), 3/(2*np.sqrt(2*np.log(2))))
    epsilon /= epsilon.std() #renormalise epsilon
    #eppy.append(epsilon)
    instances.append(circle + epsilon) #3fwhm
    
    
#The Intercept's coefficient is just the mean value we have that can compute beta easily
instances = np.stack(instances)
coefficient = np.mean(instances, axis = 0)
mask = coefficient > thr
#bootstrapping
epsilons = instances - coefficient
number_subject = np.shape(epsilons)[0]
gen_haus = []
gen_mask = [] # check how similar circles are

#wild t bootstrap
for i in range(boot_rep):
    rad = rademacher(number_subject)[:,None,None]
    boot_res = rad * epsilons
    boot_instances = boot_res + coefficient                 #generate new instances
    boot_coefficient = np.mean(boot_instances, axis = 0)    #compute regression coeff
    boot_mask = boot_coefficient > thr                      #compute the mask
    boot_haus = hausdorff(mask, boot_mask)                  #hausdorff
    gen_mask.append(boot_mask)
    gen_haus.append(boot_haus)
    
#for trouble shoot fix checking how similar circles are
for i in range(10):
    plt.figure()
    plt.imshow(gen_mask[i])

#
plt.hist(gen_haus)


    





