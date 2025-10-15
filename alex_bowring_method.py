# -*- coding: utf-8 -*-
"""
Created on Fri Oct 10 11:34:22 2025

@author: railt
"""
import numpy as np
from scipy import ndimage

def rademacher(n):
    """
    generate n sample of 1 or -1 with probability 1/2 for each outcome
    """
    outcome = np.random.choice([-1,1], size = n, replace = True)
    return(outcome)

def boundary_erosion(contrast_values, threshold_value, dim):
    
    """
    Identify boundary point by finding a point satisfying mu(s)>=c and such that 
    exist neighbour point with mu(s)<c. this one only recognize point attached
    to the surface as neighbour. I use the binary erosion idea from
    morphology here. Not sure how to avoid donut shape yet. Here the 
    erosion remove outer points, and so boundary is just the intersection between
    significatn points and the complement of eroded points
    """
    
    #create binary mask then take the A\(A erosion B) as boundary points
    np.asarray(contrast_values)
    significant_points = contrast_values > threshold_value
    structure_matrix = ndimage.generate_binary_structure(dim,1)
    erosion = ndimage.binary_erosion(significant_points,
                                     structure = structure_matrix)
    boundary = significant_points & ~erosion
    
    return(boundary, significant_points)

def boundary_linear_interpolation(contrast_values, residual_values, 
                                  threshold_value, dim):
    
    """
    take in residual_values as 4d object (n, x, y, z) where n is the the
    label number for the subject and x,y,z is coordinate
    
    """
    
    #initialise some valeus
    c = threshold_value
    neighbourhood = []
    #find coordinate for the boundary
    boundary,significant_points = boundary_erosion(contrast_values, 
                                                   c,  dim)
    boundary_point = np.argwhere(boundary)
    
    #create a neighbour point to examine (0,1),(1,0),(1,1) etc. using erosion 
    #structure
    structure_matrix = ndimage.generate_binary_structure(dim,1)
    shape = np.shape(significant_points)
    for x,y,z in np.argwhere(structure_matrix):
        if not (x==1 and y==1 and z ==1):
            neighbourhood.append((x-1,y-1,z-1))
    

    #compute and store all pair which generate s* points and save m1, m2 for each pair
    samples = []
    for point in boundary_point:
        x,y,z = point
        
        for neighbour in neighbourhood:
            i,j,k = neighbour
            nx, ny, nz = x + i, y + j, z + k
            
            if (0 <= nx < shape[0] and
                0 <= ny < shape[1] and
                0 <= nz < shape[2] and
                not significant_points[nx,ny,nz]):
                
                s1 = contrast_values[x,y,z]
                s0 = contrast_values[nx,ny,nz]
                
                diff = s1 - s0
                
                m1 = (s1 - c)/diff
                m2 = (c-s0)/diff
                
                samples.append((x,y,z),(nx,ny,nz), m1, m2)
        
    #here m is just number of recorded s* and n = number of subject
    m = len(samples)
    n = np.shape(residual_values)[0]

    
    #compute standard deviation for each voxel then normalised the residual
    #assume degree of freedom is 1 here, not sure number of feature yet
    residual_std = residual_values.std(axis=0)
    residual_standard = residual_values/residual_std
    
    #initialise collection of reisdual at estimated boundary
    residual_boundary = np.zeros((n,m))
    
    #compute residual at s star
    for i, (s1,s0,m1,m2 ) in enumerate(samples):
        x, y, z = s1
        nx, ny, nz = s0
        
        residual_1 = residual_standard[:, x, y, z]
        residual_0 = residual_standard[:, nx, ny, nz]
        
        residual_boundary[:,i] = residual_0 * m1 + residual_1 * m2
        
    return(residual_boundary)
        
def wildt_bootstrap(contrast_values, residual_values,
                                  threshold_value, dim, repetition, alpha):
    """
    take in 4d residual_boundary
    """
    #compute the residual values at s*
    residual_boundary = boundary_linear_interpolation(contrast_values, 
                                                      residual_values, 
                                                      threshold_value, dim)
    
    #initialise repetition number for boot strapping
    m = repetition
    
    #number of subject
    sample_size, points = np.shape(residual_boundary)
    
    #G values repetition row 
    sup_G = np.zeros(m)
    
    
    for i in range(m):
        rademacher_int = rademacher(sample_size)
        #clone the realisation for each point
        rademacher_int = np.tile(rademacher_int[:,None], points)
        #compute g for each point
        G = 1/np.sqrt(sample_size) * np.sum(residual_boundary * rademacher_int, axis=0)
        
        sup_G[i] = np.max(G)
    
    k = np.quantile(sup_G, 1-alpha)
    
    return(k)

def lower_high_confidence_region(k,)

    
        
        
        
        
        
        
        
        
        
        
        
     
    
    
    
    
    
    
    return()
        
        
        
        
    
 





