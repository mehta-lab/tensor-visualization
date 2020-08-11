# Parameters
initial_num_points = 8000
final_points = 5000
alpha = 1.0
max_time = 80.0
min_time = 5.0
num_iterations = 200
boundary = 30
c_drag = 30.0

import numpy as np
import matplotlib.pyplot as plt

import tifffile as tf
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import cm
import cv2
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.integrate import odeint
import time
from numpy import linalg as LA
import random
from itertools import cycle
from scipy import interpolate
import cv2
import multiprocessing as mp

def nanRobustBlur(I, dim):
    V=I.copy()
    V[I!=I]=0
    VV=cv2.blur(V,dim)   
    W=0*I.copy()+1
    W[I!=I]=0
    WW=cv2.blur(W,dim)    
    Z=VV/WW
    return Z 

anisotropy = cv2.imread('2d_data/img_retardance3D_t000_p000_z044.tif', -1).astype('float32')
orientation = cv2.imread('2d_data/img_azimuth_t000_p000_z044.tif', -1).astype('float32')
anisotropy = anisotropy[100:200, 100:200]
orientation = orientation[100:200, 100:200]
orientation = orientation / 18000*np.pi
anisotropy = anisotropy / 10000

def return_smooth(orientation, anisotropy):
    U, V =  anisotropy*np.cos(2 * orientation), anisotropy*np.sin(2 * orientation)
    USmooth = nanRobustBlur(U, (5, 5)) # plot smoothed vector field
    VSmooth = nanRobustBlur(V, (5, 5)) # plot smoothed vector field
    azimuthSmooth = (0.5*np.arctan2(VSmooth,USmooth)) % np.pi
    RSmooth = np.sqrt(USmooth**2+VSmooth**2)
    
    return RSmooth, azimuthSmooth

anisotropy, orientation = return_smooth(orientation, anisotropy)

def return_D(position):
    scale_value = anisotropy[position[0]][position[1]]
    theta = orientation[position[0]][position[1]]
    
    if abs(scale_value) > 1:
        scale_matrix = np.matrix([[scale_value, 0], [0, 1]])
    else:
        scale_matrix = np.matrix([[1, 0], [0, scale_value]])
    
    angle_matrix = np.matrix([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    angle_matrix_2 = np.matrix([[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]])
    
    D_matrix = np.matmul(np.matmul(angle_matrix, scale_matrix), angle_matrix_2)
    
    return D_matrix

memory_inverse = {}

D1 = np.zeros_like(anisotropy, dtype=np.float32)
D2 = np.zeros_like(anisotropy, dtype=np.float32)
D3 = np.zeros_like(anisotropy, dtype=np.float32)
D4 = np.zeros_like(anisotropy, dtype=np.float32)

x = np.arange(0, anisotropy.shape[0], 1)
y = np.arange(0, anisotropy.shape[1], 1)

for i in x:
    for j in y:
        scale_matrix = return_D([i, j])
        memory_inverse[(i, j)] = np.linalg.inv(scale_matrix)

        D1[i, j] = scale_matrix[0, 0]
        D2[i, j] = scale_matrix[0, 1]
        D3[i, j] = scale_matrix[1, 0]
        D4[i, j] = scale_matrix[1, 1]

D1_interp = interpolate.interp2d(x, y, D1.T, kind='cubic')
D2_interp = interpolate.interp2d(x, y, D2.T, kind='cubic')
D3_interp = interpolate.interp2d(x, y, D3.T, kind='cubic')
D4_interp = interpolate.interp2d(x, y, D4.T, kind='cubic')

def return_D_interp(position):
    D1_value = D1_interp(position[0], position[1])[0]
    D2_value = D2_interp(position[0], position[1])[0]
    D3_value = D3_interp(position[0], position[1])[0]
    D4_value = D4_interp(position[0], position[1])[0]
    
    return np.matrix([[D1_value, D2_value], [D3_value, D4_value]])

def return_D_inverse(pos_a, pos_b):
    pos_a = ([int(x) for x in pos_a])
    pos_b = ([int(x) for x in pos_b])
    pos_a = (pos_a[0], pos_a[1])
    pos_b = (pos_b[0], pos_b[1])

    if pos_a not in memory_inverse.keys() and pos_b not in memory_inverse.keys():
        D_inverse_ab = np.identity(2)
    elif pos_a not in memory_inverse.keys():
        D_inverse_ab = memory_inverse[pos_b]
    elif pos_b not in memory_inverse.keys():
        D_inverse_ab = memory_inverse[pos_a]
    else:
        D_inverse_ab = (memory_inverse[pos_a] + memory_inverse[pos_b])/2.0

    return D_inverse_ab

gamma = 0.5
def force_function_list(r):
    
    mask_1 = np.logical_and(r < 1, r > 0)
    mask_2 = np.logical_and(r >= 1, r <= 1 + gamma)
    mask_3 = r > 1+gamma
    
    return mask_1*(r-1) + mask_2*(((r-1)*(1+gamma-r)**2)/gamma**2) + mask_3*(0)

from numba import jit, prange

@jit(nopython=True)
def invert_matrix(matrices):
    inverted_list = np.empty_like(matrices)
    
    for i in prange(matrices.shape[0]):
        inverted_list[i] = np.linalg.inv(matrices[i])

    return inverted_list

def return_matrix(position):
    D1_value = D1_interp(position[0], position[1])[0]
    D2_value = D2_interp(position[0], position[1])[0]
    D3_value = D3_interp(position[0], position[1])[0]
    D4_value = D4_interp(position[0], position[1])[0]

    matrix = np.array([[D1_value, D2_value], [D3_value, D4_value]])
    
    return matrix

def return_total_force(pos_a, pos_b_positions, approx=True):
    total_force = np.zeros((2, 1))
    matrices = np.zeros((len(pos_b_positions), 2, 2))
    identity = np.identity(2, dtype=float)
    
    if approx:
        D_inverse_list = []
        for pos_b in pos_b_positions:
                D_inverse_list.append(return_D_inverse(pos_a, pos_b))

        D_inverse_list = np.array(D_inverse_list)
    else:            
        positions = np.add(pos_a, pos_b_positions)/2.0
        
        for k in range(len(positions)):
            matrices[k] = return_matrix(positions[k])
        
        D_inverse_list = invert_matrix(matrices)

    pos_a_np = np.asarray(pos_a, dtype='float32')

    y_ab_stack = pos_a_np - pos_b_positions
    y_ab_stack = y_ab_stack.reshape(len(pos_b_positions), 2, 1)

    x_ab_list = np.transpose(np.einsum('ijk,ikl->ijl', D_inverse_list, y_ab_stack)[:,:,0])/(2*alpha)
    x_ab_dist_list = LA.norm(x_ab_list, axis=0) + 1e-5

    mult = np.transpose(x_ab_list).reshape(x_ab_list.shape[1], x_ab_list.shape[0], 1)
    force_list = -force_function_list(x_ab_dist_list)*np.transpose(np.einsum('ijk,ikl->ijl', D_inverse_list, mult)[:,:,0])/(2*alpha*x_ab_dist_list)
    total_force += np.sum(force_list, axis = 1).reshape((2, 1))   
            
    return total_force

random_rows = np.random.choice(anisotropy.shape[0], initial_num_points)
random_columns = np.random.choice(anisotropy.shape[1], initial_num_points)
random_points = list(zip(random_rows, random_columns))

det_D_list = [np.linalg.det(return_D_interp(pos)) for pos in random_points]
max_det = max(det_D_list)
det_D_list = [x/max_det for x in det_D_list]
points_rejected = np.random.choice(len(random_points), initial_num_points - final_points, det_D_list)
final_points = [random_points[i] for i in range(len(random_points)) if i not in points_rejected]

eig_D = [max(np.linalg.eig(return_D_interp(pos))[0]) for pos in random_points]
max_eig = max(eig_D)

edge_length = 2*alpha*max_eig*(1+gamma)

def bin_coords(pos):
    x_coord = np.floor(pos[0]/edge_length)
    y_coord = np.floor(pos[1]/edge_length)
    
    return (x_coord, y_coord)

curr_points = final_points

def form_dict(points):
    dict_points = {}

    for point in points:
        bin_coord = bin_coords(point)

        if bin_coord in dict_points.keys():
            dict_points[bin_coord].append(point)
        else:
            dict_points[bin_coord] = [point]
    
    return dict_points

def total_force_on_point(index, list_points, map_points, approx=True):
    p_a = list_points[index]
    bin_coord = bin_coords(p_a)
    bin_x = bin_coord[0]
    bin_y = bin_coord[1]

    points_cons = []

    points_cons += map_points[bin_coord]

    if bin_x - 1 >= 0:
        if (bin_x - 1, bin_y) in map_points.keys():
            points_cons += map_points[(bin_x - 1, bin_y)]

    if bin_y - 1 >= 0:
        if (bin_x, bin_y - 1) in map_points.keys():
            points_cons += map_points[(bin_x, bin_y - 1)]

    if bin_x + 1 < anisotropy.shape[0]:
        if (bin_x + 1, bin_y) in map_points.keys():
            points_cons += map_points[(bin_x + 1, bin_y)]

    if bin_y + 1 < anisotropy.shape[1]:
        if (bin_x, bin_y + 1) in map_points.keys():
            points_cons += map_points[(bin_x, bin_y + 1)]
    
    total_force = return_total_force(p_a, points_cons, approx)

    return total_force, len(points_cons)

def diff_equation(y, t, c_drag, total_force):
    p0, p1, r0, r1 = y
    dydt = [r0, r1, total_force[0] - c_drag*r0, total_force[1] - c_drag*r1]
    return dydt

def solve_particle_path(index, initial_pos, map_points, time_integrate, approx=True):
    y0 = [initial_pos[index][0], initial_pos[index][1], 0.0, 0.0]  
    t = np.linspace(0, time_integrate, 10) 
    tf, tot_points = total_force_on_point(index, initial_pos, map_points, approx)
    sol = odeint(diff_equation, y0, t, args=(c_drag, tf))
    
    return sol, tot_points

map_points = form_dict(curr_points)

slope = (max_time - min_time)/(1 - num_iterations)
constant = (num_iterations*max_time - min_time)/(num_iterations - 1)
start_algo = time.clock()
pool = mp.Pool(mp.cpu_count())

for k in range(num_iterations):
    start_iteration = time.clock()
    final_positions = []
    num_particles = 0
    total_dist = 0
    new_map_points = {}
    avg_total_points = 0
    time_integrate = slope*(k+1) + constant
    
    if k < 8:
        processes = [pool.apply_async(solve_particle_path, args=(i, curr_points, map_points, time_integrate, True)) for i in range(len(curr_points))]
    else:
        processes = [pool.apply_async(solve_particle_path, args=(i, curr_points, map_points, time_integrate, False)) for i in range(len(curr_points))] 

    results = [process.get() for process in processes]

    for i in range(len(curr_points)):   
        sol = results[i][0]
        tot_points = results[i][1]
        
        if sol[-1, 0] >= -boundary and sol[-1, 0] <= anisotropy.shape[0] + boundary and \
           sol[-1, 1] >= -boundary and sol[-1, 1] <= anisotropy.shape[1] + boundary:
            final_positions.append((sol[-1, 0], sol[-1, 1]))
            num_particles += 1
            avg_total_points += tot_points
            total_dist += LA.norm(np.array([sol[-1, 0], sol[-1, 1]]) - np.array([sol[0, 0], sol[0, 1]]))     

            new_bin_coord = bin_coords((sol[-1, 0], sol[-1, 1]))

            if new_bin_coord in new_map_points.keys():
                new_map_points[new_bin_coord].append((sol[-1, 0], sol[-1, 1]))
            else:
                new_map_points[new_bin_coord] = [(sol[-1, 0], sol[-1, 1])]
    
    if k%4 == 0: 
        print("Distances moved by particle:", total_dist/num_particles)
        print("Number of particles:", num_particles)
        print("Number of particles for force:", avg_total_points/num_particles)
        end_iteration = time.clock()
        print("Time taken for iteration: ", end_iteration - start_iteration)
        print("")
    
    curr_points = final_positions
    map_points = new_map_points
    
    if total_dist/num_particles < 0.3:
        break

end_algo = time.clock()

print("Total time taken: ", end_algo - start_algo)

final_positions = np.array(final_positions)
final_positions = np.around(final_positions, decimals=0)

np.save('fp_glpyh_packing', final_positions)
