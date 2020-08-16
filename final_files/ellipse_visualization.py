# Parameters
alpha = 2.0
dataset = 'u2'
positions_file = 'fp_u2_alpha_3.0_numpoints_40000_drag_30.0.npy'

import cv2
from matplotlib.patches import Ellipse
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx
import numpy as np
from skimage.transform import rescale

anisotropy = cv2.imread('2d_data/' + dataset + '/retardance.tif', -1).astype('float32')
orientation = cv2.imread('2d_data/' + dataset + '/azimuth.tif', -1).astype('float32')
anisotropy = rescale(anisotropy, 0.5, anti_aliasing=True)
orientation = rescale(orientation, 0.5, anti_aliasing=True)
anisotropy = anisotropy / 65535*10
orientation = orientation / 18000*np.pi

if dataset == 'u2':
    USmooth, VSmooth = anisotropy*np.cos(orientation), anisotropy*np.sin(orientation)
    VSmooth = VSmooth*-1
    orientation = np.arctan2(VSmooth, USmooth)

final_positions = np.load(positions_file)
final_positions = final_positions.astype(int)

within_boundary_pos = []

for i in range(len(final_positions)):
    if final_positions[i][0] >= 0 and final_positions[i][0] < anisotropy.shape[0] and final_positions[i][1] >= 0 and final_positions[i][1] < anisotropy.shape[1]:
        within_boundary_pos.append(final_positions[i])

major_axis_len = []
minor_axis_len = []
orientation_list = []
scale_ellipse_major = alpha*1.2
scale_ellipse_minor = alpha*0.8

for p in within_boundary_pos:
    orientation_list.append(orientation[p[0], p[1]])
    anisotropy_value = anisotropy[p[0], p[1]]

    if abs(anisotropy_value) < 1:
        major_axis_len.append(1*0.2)
        minor_axis_len.append(abs(anisotropy_value)*0.2)
    else:
        minor_axis_len.append(1*scale_ellipse_minor)
        major_axis_len.append(abs(anisotropy_value)*scale_ellipse_major)

jet = cm = plt.get_cmap('hsv')

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
    plt.imshow(anisotropy, cmap=cmapImage)
else:
    plt.imshow(anisotropy, cmap=cmapImage, origin='lower')
k = 0
for e in ells:
    ax.add_artist(e)
    e.set_facecolor(scalarMap.to_rgba(orientation_list[k]))
    e.set_edgecolor('black')
    e.set_linewidth(0.5)
    k += 1

fig.savefig('ellipse_' + positions_file + '.png', bbox_inches='tight')
plt.close(fig)
