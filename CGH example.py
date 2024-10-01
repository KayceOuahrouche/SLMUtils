import numpy as np 
import matplotlib.pyplot as plt
import cv2
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
from GS_MRAF import PhasePattern
plt.style.use(['default'])



nx = ny = 1024

PatternGenerator = PhasePattern(nx, ny)

input_amplitude = PatternGenerator.gaussian_beam(400)
target_intensity = cv2.imread('SLM Pictures/A_letter_256.tif',cv2.IMREAD_GRAYSCALE)
target_intensity = PatternGenerator.pad(target_intensity)
target_intensity = PatternGenerator.pad(target_intensity)


size = 300
pitch = 50
x0 = ny//2   
y0 = nx//2

PatternGenerator.setInputAmplitude(input_amplitude)
PatternGenerator.setTargetIntensity(target_intensity)
PatternGenerator.setMRAF_masks(size,shape = 'square')

reconstructed_image,phase_mask,w = PatternGenerator.PhaseRetrieve(n = 100,m = 0.4)




computation_extent = np.array([-nx//2,nx//2,-ny//2,ny//2])

fig,ax = plt.subplots(1,3,dpi = 300)


#ax[0].set_title('Target Intensity')
im1 = ax[0].imshow(target_intensity,cmap = 'Blues',extent = computation_extent)

ax1_divider = make_axes_locatable(ax[0])
cax1 = ax1_divider.append_axes("top", size="7%", pad="2%")
cbar = fig.colorbar(im1,cax = cax1,orientation = 'horizontal')
plt.title(r'$I_{target}$')
cax1.xaxis.set_ticks_position("top")




#ax[1].set_title('Reconstructed Phase')
im2 = ax[1].imshow(phase_mask,cmap = 'gray',extent = computation_extent)
ax2_divider = make_axes_locatable(ax[1])
cax2 = ax2_divider.append_axes("top", size="7%", pad="2%")
cbar2 = fig.colorbar(im2,cax =cax2,orientation = 'horizontal')
plt.title(r'$\phi_{SLM}$')
cbar2.set_ticks([0,255])
cax2.xaxis.set_ticks_position("top")



#ax[2].set_title('Reconstructed Intensity')
im3 = ax[2].imshow(abs(reconstructed_image)**2,cmap = 'Blues',extent =computation_extent)
ax1_divider = make_axes_locatable(ax[2])
cax3 = ax1_divider.append_axes("top", size="7%", pad="2%")
cbar3 = fig.colorbar(im3,cax = cax3,orientation = 'horizontal')
plt.title(r'$I_{reconstructed}$')
cbar3.set_ticks([0,1])
cax3.xaxis.set_ticks_position("top")
