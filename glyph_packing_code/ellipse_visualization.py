# Parameters
alpha = 1.5
dataset = 'u2'
pos_folder = 'results/'
positions_file = 'fp_u2_alpha_1.0_numpoints_50000_drag_30.0.npy'
import cv2
from matplotlib.patches import Ellipse
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx
import numpy as np
from skimage.transform import rescale

anisotropy = cv2.imread('2d_data/' + dataset + '/retardance.tif', -1).astype('float32')
orientation = cv2.imread('2d_data/' + dataset + '/azimuth.tif', -1).astype('float32')

if dataset == 'kaza':
    orientation = orientation / 18000*np.pi
    anisotropy = anisotropy / 10000

if dataset == 'u2':
    anisotropy = rescale(anisotropy, 0.5, anti_aliasing=True)
    orientation = rescale(orientation, 0.5, anti_aliasing=True)
    anisotropy = anisotropy / 65535*10
    orientation = orientation / 18000*np.pi
#     anisotropy = np.ones_like(anisotropy) + 1
    USmooth, VSmooth = anisotropy*np.cos(orientation), anisotropy*np.sin(orientation)
    VSmooth = VSmooth*-1
    orientation = np.arctan2(VSmooth, USmooth)

def nanRobustBlur(I, dim):
    V=I.copy()
    V[I!=I]=0
    VV=cv2.blur(V,dim)   
    W=0*I.copy()+1
    W[I!=I]=0
    WW=cv2.blur(W,dim)    
    Z=VV/WW
    return Z

def return_smooth(orientation, anisotropy):
    U, V =  anisotropy*np.cos(2 * orientation), anisotropy*np.sin(2 * orientation)
    USmooth = nanRobustBlur(U, (5, 5)) # plot smoothed vector field
    VSmooth = nanRobustBlur(V, (5, 5)) # plot smoothed vector field
    azimuthSmooth = (0.5*np.arctan2(VSmooth,USmooth)) % np.pi
    RSmooth = np.sqrt(USmooth**2+VSmooth**2)
    
    return RSmooth, azimuthSmooth

un_avg_anisotropy = anisotropy
anisotropy, orientation = return_smooth(orientation, anisotropy)

final_positions = np.load(pos_folder + positions_file)
final_positions = final_positions.astype(int)

within_boundary_pos = []

for i in range(len(final_positions)):
    if final_positions[i][0] >= 0 and final_positions[i][0] < anisotropy.shape[0] and final_positions[i][1] >= 0 and final_positions[i][1] < anisotropy.shape[1]:
        within_boundary_pos.append(final_positions[i])

major_axis_len = []
minor_axis_len = []
orientation_list = []
scale_ellipse_major = alpha*1.3
scale_ellipse_minor = alpha*0.7

for p in within_boundary_pos:
    orientation_list.append(orientation[p[0], p[1]])
    anisotropy_value = anisotropy[p[0], p[1]]

    if abs(anisotropy_value) < 1.1:
        major_axis_len.append(1*0.01)
        minor_axis_len.append(abs(anisotropy_value)*0.01)
    else:
        minor_axis_len.append(1*scale_ellipse_minor)
        major_axis_len.append(abs(anisotropy_value)*scale_ellipse_major)

jet = cm = plt.get_cmap('hsv_r')

cNorm  = colors.Normalize(vmin=min(orientation_list), vmax=max(orientation_list))
scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=jet)

ells = [Ellipse(xy=within_boundary_pos[i][::-1],
                width=major_axis_len[i], height=minor_axis_len[i],
                angle=orientation_list[i]*(180/np.pi))
        for i in range(len(orientation_list))]

cmapImage = 'gray'
cmapOrient = 'hsv'

fig, ax = plt.subplots(subplot_kw={'aspect': 'equal'})
fig.set_figheight(30)
fig.set_figwidth(30)
if dataset == 'u2':
    plt.imshow(un_avg_anisotropy, cmap=cmapImage)
else:
    plt.imshow(un_avg_anisotropy, cmap=cmapImage, origin='lower')
k = 0
for e in ells:
    ax.add_artist(e)
    e.set_facecolor(scalarMap.to_rgba(orientation_list[k]))
    e.set_edgecolor('black')
    e.set_linewidth(0.5)
    k += 1

fig.savefig(pos_folder + 'ellipse_' + positions_file + '.png', bbox_inches='tight')
plt.close(fig)
