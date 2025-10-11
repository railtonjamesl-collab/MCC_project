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

def boundary_erosion(effect_values, threshold_value, dim):
    
    """
    Identify boundary point by finding a point satisfying mu(s)>=c and such that 
    exist neighbour point with mu(s)<c. this one only recognize point attached
    to the surface as neighbour. I use the binary erosion idea from
    morphology here. Not sure how to avoid donut shape yet. Here the 
    erosion remove outer points, and so boundary is just the intersection between
    significatn points and the complement of eroded points
    """
    
    np.asarray(effect_values)
    significant_points = effect_values > threshold_value
    structure_matrix = ndimage.generate_binary_structure(dim,1)
    erosion = ndimage.binary_erosion(significant_points,
                                     structure = structure_matrix)
    boundary = significant_points & ~erosion
    
    return(boundary, significant_points)

def boundary_linear_interpolation(effect_values, residual_values, threshold_value, dim):
    
    """
    take in 3d numpy array
    
    """
    c = threshold_value
    residual_boundaries = np.array()
    boundary,significant_points = boundary_erosion(effect_values, 
                                                   c,  dim)
    
    boundary_point = np.argwhere(boundary)
    #it just happened that the binary structure form a cube so here i'll
    #use it to compute neighbour points for surface connected points
    
    neighbourhood = np.array()
    structure_matrix = ndimage.generate_binary_structure(dim,1)
    shape = np.shape(significant_points)
    for x,y,z in np.argwhere(structure_matrix):
        if not (x==1 and y==1 and z ==1):
            neighbourhood = np.append(neighbourhood, (x-1,y-1,z-1))
    
    
    for point in boundary_point:
        x,y,z = point
        
        for neighbour in neighbourhood:
            i,j,k = neighbour
            nx, ny, nz = x + i, y + j, z + k
            
            if (0 <= nx < shape[0] and
                0 <= ny < shape[1] and
                0 <= nz < shape[2] and
                ~significant_points[nx,ny,nz]):
                
                s1 = effect_values[x,y,z]
                s0 = effect_values[nx,ny,nz]
                
                diff = s1 - s0
                
                m1 = (s1 - c)/diff
                m2 = (c-s0)/diff
                
                residual_est = (m1 * residual_values[nx,ny,nz] + 
                            m2 * residual_values[x,y,z])
                residual_boundaries = np.append(residual_boundaries, residual_est)
            return(residual_boundaries)
                
            
            
def wildt_bootstrap(residual_boundaries, repetition):
    """
    wild t bootstrap for each location
    """
    epsilon_values = np.array(residual_boundaries)
    sample_size = len(residual_boundaries)
    m = repetition
    G_est = np.zeros(m)
    
    for i in range(m):
        rademacher_values = rademacher(sample_size)
        boot_residual = epsilon_values * rademacher_values
        sigma_est = np.std(boot_residual)
        residual_est = boot_residual/sigma_est
        G_est[i] = 1/np.sqrt(sample_size) * sum(residual_est)
    
    return(G_est)
 





