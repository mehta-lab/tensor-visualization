import pyvista as pv
from pyvistaqt import BackgroundPlotter
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import glob
import cv2
import numpy as np
from scipy.ndimage import uniform_filter

from skimage.restoration import (denoise_tv_chambolle, denoise_bilateral,
                                 denoise_wavelet, estimate_sigma)
from orientation_to_rgb import orientation_3D_to_rgb
import time
from numpy import linalg as LA

dataset = 'mouse'

# Generate colormap
spaces_th = 139
spaces_az = 157
th,ph = np.meshgrid(np.linspace(0,np.pi,spaces_th),np.linspace(0,np.pi,spaces_az))

if dataset == 'mouse':
    orientation_image = np.transpose(np.array([((ph + np.pi/2)%(2*np.pi))/2/np.pi, th, np.ones_like(th)]),(1,2,0))
else:
    orientation_image = np.transpose(np.array([ph/2/np.pi, th, np.ones_like(th)]),(1,2,0))
face_color = orientation_3D_to_rgb(orientation_image, interp_belt = 100/180*np.pi, sat_factor = 1)

mapping = np.linspace(0, spaces_th*spaces_az - 1, spaces_th*spaces_az)
newcolors = np.zeros((spaces_th*spaces_az, 4))

for i in range(spaces_az):
    for j in range(spaces_th):
        color = face_color[i][j]
        newcolor = np.ones((4,))
        newcolor[:3] = color
        newcolors[mapping == (i*spaces_az + j)] = newcolor

# Make the colormap from the listed colors
my_colormap = ListedColormap(newcolors)

line_az = np.linspace(0,np.pi,spaces_az)
line_th = np.linspace(0,np.pi,spaces_th)
iwidth_az = line_az[1] - line_az[0]
iwidth_th = line_th[1] - line_th[0]

def visualization_3d(ret_files, azimuth_files, theta_files, linelength=20, denoise_weight=5, filter_size=(12, 12, 10), anisotropy_scale=0.2, z_stack=96,
                     spacing_xy=12, spacing_z=10, radius_scale=1.0, height_scale=10, colormap=my_colormap, neg_retardance=True, denoise=True):
    N_z = z_stack
    ret = []
    azimuth = []
    theta = []
    for i in range(N_z):
        ret.append(cv2.imread(ret_files[i], -1).astype('float32'))
        azimuth.append(cv2.imread(azimuth_files[i], -1).astype('float32'))
        theta.append(cv2.imread(theta_files[i], -1).astype('float32'))

    ret = np.transpose(np.array(ret),(1,2,0))/10000
    azimuth = np.transpose(np.array(azimuth),(1,2,0))/18000*np.pi
    theta = np.transpose(np.array(theta),(1,2,0))/18000*np.pi
    
    anisotropy= anisotropy_scale*np.abs(ret)

    f_1c = anisotropy*linelength*(np.sin(theta)**2)*np.cos(2*azimuth)
    f_1s = anisotropy*linelength*(np.sin(theta)**2)*np.sin(2*azimuth)
    f_2c = anisotropy*linelength*np.sin(2*theta)*np.cos(azimuth)
    f_2s = anisotropy*linelength*np.sin(2*theta)*np.sin(azimuth)

    if denoise:
        f_1c_denoised = denoise_tv_chambolle(f_1c, weight=denoise_weight)
        f_1s_denoised = denoise_tv_chambolle(f_1s, weight=denoise_weight)
        f_2c_denoised = denoise_tv_chambolle(f_2c, weight=denoise_weight)
        f_2s_denoised = denoise_tv_chambolle(f_2s, weight=denoise_weight)
    else:
        f_1c_denoised = f_1c
        f_1s_denoised = f_1s
        f_2c_denoised = f_2c
        f_2s_denoised = f_2s

    f_1c_smooth = uniform_filter(f_1c_denoised, filter_size)
    f_1s_smooth = uniform_filter(f_1s_denoised, filter_size)
    f_2c_smooth = uniform_filter(f_2c_denoised, filter_size)
    f_2s_smooth = uniform_filter(f_2s_denoised, filter_size)
    
    reg_ret_ap = 0.00001
    
    if not neg_retardance:
        azimuth_n = (np.arctan2(f_1s_smooth, f_1c_smooth)/2)%np.pi
        del_f_sin_square_n = f_1s_smooth*np.sin(2*azimuth_n) + f_1c_smooth*np.cos(2*azimuth_n)
        del_f_sin2theta_n = f_2s_smooth*np.sin(azimuth_n) + f_2c_smooth*np.cos(azimuth_n)
        theta_n = np.arctan2(2*del_f_sin_square_n, del_f_sin2theta_n)
        retardance_ap_n = del_f_sin_square_n * np.sin(theta_n)**2 / (np.sin(theta_n)**4 + reg_ret_ap)
    else:
        azimuth_n = (np.arctan2(-f_1s_smooth, -f_1c_smooth)/2)%np.pi
        del_f_sin_square_n = f_1s_smooth*np.sin(2*azimuth_n) + f_1c_smooth*np.cos(2*azimuth_n)
        del_f_sin2theta_n = f_2s_smooth*np.sin(azimuth_n) + f_2c_smooth*np.cos(azimuth_n)
        theta_n = np.arctan2(-2*del_f_sin_square_n, -del_f_sin2theta_n)
        retardance_ap_n = del_f_sin_square_n * np.sin(theta_n)**2 / (np.sin(theta_n)**4 + reg_ret_ap)
    
    USmooth = retardance_ap_n*np.cos(azimuth_n)*np.sin(theta_n)
    VSmooth = retardance_ap_n*np.sin(azimuth_n)*np.sin(theta_n)
    WSmooth = retardance_ap_n*np.cos(theta_n)

    nY, nX, nZ = USmooth.shape
    y, x, z = np.mgrid[0:nY,0:nX,0:nZ]

    # Plot sparsely sampled vector lines
    Plotting_X = x[::-spacing_xy, ::spacing_xy, ::spacing_z]
    Plotting_Y = y[::-spacing_xy, ::spacing_xy, ::spacing_z]
    Plotting_Z = z[::-spacing_xy, ::spacing_xy, ::spacing_z]
    Plotting_U = USmooth[::-spacing_xy, ::spacing_xy, ::spacing_z]
    Plotting_V = VSmooth[::-spacing_xy, ::spacing_xy, ::spacing_z]
    Plotting_W = WSmooth[::-spacing_xy, ::spacing_xy, ::spacing_z]
    Plotting_theta = theta_n[::-spacing_xy, ::spacing_xy, ::spacing_z]
    Plotting_azimuth = azimuth_n[::-spacing_xy, ::spacing_xy, ::spacing_z]

    azimuth_list = np.empty((Plotting_X.size, 1))
    azimuth_list[:, 0] = Plotting_azimuth.ravel('F')

    theta_list =  np.empty((Plotting_X.size, 1))
    theta_list[:, 0] = Plotting_theta.ravel('F')

    points = np.empty((Plotting_X.size, 3))
    points[:, 0] = Plotting_X.ravel('F')
    points[:, 1] = Plotting_Y.ravel('F')
    points[:, 2] = Plotting_Z.ravel('F')

    directions = np.empty((Plotting_X.size, 3))
    directions[:, 0] = Plotting_U.ravel('F')
    directions[:, 1] = Plotting_V.ravel('F')
    directions[:, 2] = Plotting_W.ravel('F')
    
    pos_x_list = np.ceil((azimuth_list - line_az[0]) / iwidth_az)
    pos_y_list = np.ceil((theta_list - line_th[0]) / iwidth_th)
    scalars = pos_x_list*spaces_az + pos_y_list
    
    point_cloud = pv.PolyData(points)
    point_cloud['vectors'] = directions
    point_cloud['values'] = scalars
    arrows = point_cloud.glyph(orient='vectors', scale=True, factor=5, geom=pv.Cylinder(radius=radius_scale, height=height_scale, resolution=200))

    filename = "viz_3d_denoised.mp4"
    pv.set_plot_theme("document")
    plotter = pv.Plotter()
    plotter.open_movie(filename)
    plotter.add_mesh(arrows, scalars='values', cmap=my_colormap)

    cpos = [(41914.98405034885, 45261.74263508663, 33163.87155659542),
            (8751.112493753433, 12097.871078491211, 0.0),
            (0.0, 0.0, 1.0)]
    #cpos = [(598.635899228783, 618.3063588568882, 496.33323602172896),
    #        (139.62683308124542, 159.2972927093506, 37.324169874191284),
    #        (0.0, 0.0, 1.0)]
    #plotter.camera_position = cpos
    plotter.show(auto_close=False)
    #plotter.write_frame()

    for i in range(200):
    #         cpos = [(598.635899228783, 618.3063588568882, 496.33323602172896 + i*5),
    #                 (139.62683308124542, 159.2972927093506, 37.324169874191284),
    #                 (0.0, 0.0, 1.0)]
        cpos = [(41914.98405034885, 45261.74263508663, 33163.87155659542 + i*100),
                (8751.112493753433, 12097.871078491211, 0.0),
                (0.0, 0.0, 1.0)]
        plotter.camera_position = cpos
        plotter.write_frame()  # Write this frame

    for j in range(200):
    #         cpos = [(598.635899228783, 618.3063588568882 - j*5, 496.33323602172896 + i*5),
    #                 (139.62683308124542, 159.2972927093506, 37.324169874191284),
    #                 (0.0, 0.0, 1.0)]
        cpos = [(41914.98405034885, 45261.74263508663 - j*100, 33163.87155659542 + i*100),
                (8751.112493753433, 12097.871078491211, 0.0),
                (0.0, 0.0, 1.0)]
        plotter.camera_position = cpos
        plotter.write_frame()  # Write this frame

    for k in range(100):
        cpos = [(41914.98405034885, 45261.74263508663 - j*100, 33163.87155659542 + i*100),
                (8751.112493753433, 12097.871078491211, 0.0),
                (0.0 + k*0.005, 0.0 + k*0.005, 1.0 + k*0.005)]
        plotter.camera_position = cpos
        plotter.write_frame()  # Write this frame

    plotter.close()

if dataset == 'kaza':
    ret_path = '/mnt/comp_micro/Projects/visualization/dataset/3D_orientation_data/20200223_63x_3D_Old_Kazansky_Target/retardance3D/*'
    azimuth_path = '/mnt/comp_micro/Projects/visualization/dataset/3D_orientation_data/20200223_63x_3D_Old_Kazansky_Target/azimuth/*'
    theta_path = '/mnt/comp_micro/Projects/visualization/dataset/3D_orientation_data/20200223_63x_3D_Old_Kazansky_Target/theta/*'

    ret_files = sorted(glob.glob(ret_path))
    azimuth_files = sorted(glob.glob(azimuth_path))
    theta_files = sorted(glob.glob(theta_path))
    
if dataset == 'mouse':
    ret_path = '/mnt/comp_micro/Projects/visualization/dataset/3D_orientation_data/20200302_20x_2D_whole_Mouse_brain/img_retardance2D_t000_p000_z000.tif'
    azimuth_path = '/mnt/comp_micro/Projects/visualization/dataset/3D_orientation_data/20200302_20x_2D_whole_Mouse_brain/img_azimuth_t000_p000_z000.tif'
    theta_path = '/mnt/comp_micro/Projects/visualization/dataset/3D_orientation_data/20200302_20x_2D_whole_Mouse_brain/img_theta_t000_p000_z000.tif'

    ret_files = [ret_path]
    azimuth_files = [azimuth_path]
    theta_files = [theta_path]

visualization_3d(ret_files, azimuth_files, theta_files, linelength=20, denoise_weight=5, filter_size=(30, 30, 10), anisotropy_scale=0.3, z_stack=1,
                 spacing_xy=100, spacing_z=1, radius_scale=0.4, height_scale=3.0, colormap=my_colormap, neg_retardance=False, denoise=False)
