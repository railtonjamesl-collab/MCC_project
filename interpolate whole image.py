# -*- coding: utf-8 -*-
"""
Created on Wed Nov 19 23:14:15 2025

@author: railt
"""

import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt
from numpy.linalg import inv
from functions import (generate_circle_100grid, hausdorff, boundary_dilation, 
                       compute_boundary, rademacher, interpolate1, generate_gradient)
from scipy.spatial.distance import directed_hausdorff
from scipy.ndimage import zoom

def refine_bdry(coef, zoom_factor, thr):
    coef_fine = zoom(coef, zoom_factor, order=1)
    mask_fine = coef_fine > thr
    bdry_fine, *_ = compute_boundary(mask_fine)
    bdry_fine = np.argwhere(bdry_fine)
    bdry_refined = bdry_fine/zoom_factor
    return(bdry_refined)



rep = 1
success = 0
thr = 2
boot_rep = 1
alpha = 0.1
zoom_factor = 10
for j in range(rep):
    #initialise some parameter!

    #generating a circular signal and smoothing it
    #circle = generate_circle_100grid(30, 3)[0]
    circle = generate_gradient(0,3)
    circle = ndimage.gaussian_filter(circle, 3/(2*np.sqrt(2*np.log(2))))

    instances = []
    #eppy = []
    for i in range(30):
        epsilon = ndimage.gaussian_filter(np.random.randn(100,100), 3/(2*np.sqrt(2*np.log(2))))
        epsilon /= epsilon.std() #renormalise epsilon
        #eppy.append(epsilon)
        instances.append(circle + epsilon) #3fwhm
        
        
    #The Intercept's coefficient is just the mean value we have that can compute beta easily
    instances = np.stack(instances)
    coefficient = np.mean(instances, axis = 0)
    mask = coefficient > thr
    bdry_list = compute_boundary(mask)
    
    
    
    yolo = refine_bdry(coefficient, zoom_factor, thr)
    #bootstrapping
    epsilons = instances - coefficient
    number_subject = np.shape(epsilons)[0]
    gen_haus = []
    gen_mask = [] # check how similar circles are
    n = np.shape(instances)[0]
    #wild t bootstrap
    for i in range(boot_rep):
        rad = rademacher(number_subject)[:,None,None]
        boot_res = rad * epsilons
        boot_std = boot_res.std(axis=0, ddof=0)
        boot_res = boot_res/(boot_std )
        boot_instances = boot_res + circle           #generate new instances
        boot_coefficient = np.mean(boot_instances, axis = 0) 
        boot_mask = boot_coefficient > thr
        boot_intp = refine_bdry(boot_coefficient, zoom_factor, thr)
        boot_haus1 = directed_hausdorff(yolo, boot_intp)[0]
        boot_haus2 = directed_hausdorff(boot_intp, yolo)[0]
        boot_haus = max(boot_haus1, boot_haus2)
        gen_haus.append(boot_haus)
        """
        boot_mask = boot_coefficient > thr                      #compute the mask
        boot_haus = hausdorff(mask, boot_mask)                  #hausdorff
        gen_mask.append(boot_mask)
        gen_haus.append(boot_haus)
        """
    
    
    
    
    plt.figure()
    plt.hist(gen_haus)
    plt.show()
    
    k = np.quantile(gen_haus, 1-alpha)
    print('k values',k)
    
    #scale from 1 pixel to 0.1 pixel distance
    zoom_factor = 10
    
    
    true_bdry = refine_bdry(circle, zoom_factor, thr) 
    est_bdry = refine_bdry(coefficient, zoom_factor, thr)
    
    h_true_est_refined = directed_hausdorff(true_bdry, est_bdry)[0] #max_a inf_b d(a,b)

    
    print('max distance from true set to coefficient set', h_true_est_refined)

    contains_true = (h_true_est_refined <= k)
    
    print("condition satisfy ->", contains_true)
    if contains_true:
        success += 1
        
boi, *__ = compute_boundary(circle>thr)
coef_bdry, *__ = compute_boundary(mask)
jesus = directed_hausdorff(true_bdry, est_bdry)

r1, c1 = true_bdry[[jesus[1]]][0]
r2, c2 = est_bdry[[jesus[2]]][0]

plt.figure()
plt.imshow(boot_mask)
plt.title('mask for boot strapping')
plt.figure()
plt.imshow(coef_bdry)
plt.title('coefficient')
plt.scatter(c1,r1)
plt.scatter(c2,r2)
plt.figure()
plt.imshow(boi)
plt.title('real boundary')
plt.scatter(c1,r1)
plt.scatter(c2,r2)
boot_bdry, *__ = compute_boundary(boot_mask)
plt.figure()
plt.imshow(boot_bdry)
plt.title('bootstrap')


