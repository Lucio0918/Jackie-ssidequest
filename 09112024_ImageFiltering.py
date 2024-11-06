import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage as ndi
from skimage.filters._gaussian import gaussian
from skimage.filters.thresholding import threshold_multiotsu
from skimage.io import imread
from skimage import morphology
from skimage import measure
import cv2 as cv

# load same image to try filtering
img = cv.imread('cameraman.tif')[:,:,0]

fig, ax = plt.subplots(1,2)
plt.axes(ax[0])
cols = ax[0].imshow(img,'gray')
plt.colorbar(cols)
plt.axes(ax[1])
plt.hist(img.ravel(),bins=np.arange(256))
plt.yscale('log')
plt.ylabel('# of occurences')
plt.xlabel('Pixel intensities')

# threshold_multiotsu permits multi-class thresholding
thresholds = threshold_multiotsu(img, classes=2, nbins=256)
plt.plot([thresholds[0], thresholds[0]], [.1, 10000], 'r')

C, R = np.meshgrid(np.arange(256), np.arange(256))
plt.axes(ax[0])
# isocontouring draws lines to separate neighboring grid points that have intensities above/below the provided isolevel
# Choosing an isolevel of 127.5 (halfway between 0 and 255) will draw the isocontour halfway between fore and background pixels
cntr = plt.contour(C,R, img, levels=[thresholds[0] + .5], colors='red')



# try filtering
# define width of box filter
sigma = 1
boxsz = np.round(2*sigma).astype(int)
# Filter must have odd number of elements
if (boxsz+1)//2 == boxsz//2:
    boxsz += 1

print(boxsz)
box = np.ones([boxsz, boxsz])
box /= np.sum(box)

# ndi.convolve works for 2d or 3d images
img_box = ndi.convolve(img, box, mode='constant', cval=0)

fig, ax = plt.subplots()
ax.imshow(img_box, 'gray')
cntr = plt.contour(C,R, img_box, levels=[thresholds[0] + .5], colors='red')

# try gaussian
img_gauss = gaussian(img, sigma=sigma, mode='constant', cval=0) * 255
print(np.amin(img_gauss))
print(np.amax(img_gauss))

fig, ax = plt.subplots()

# for i in range(1, 11):
#     img_gauss_loop = gaussian(img,sigma=i,mode='constant',cval=0) * 255
#     plt.cla()
#     ax.imshow(img_gauss_loop, 'gray')
#     cntr = plt.contour(C,R, img_gauss_loop, levels=[thresholds[0] + .5], colors='red')
#     plt.title(f'Sigma = {i}')
#     plt.pause(1)

# display isocontour from filtered image on top of original image
col = ax.imshow(img, 'gray')
cntr = plt.contour(C,R,img_gauss, levels=[thresholds[0]+.5], colors='red')

# create binary segmentation for morphological filtering
img_seg = img <= thresholds[0]
fig, ax = plt.subplots()
ax.imshow(img_seg, 'gray')

se = np.ones((3,3))
img_seg_ero = morphology.binary_erosion(img_seg, se)
fig, ax = plt.subplots(num='erosion 3x3')
ax.imshow(img_seg_ero, 'gray')

se = np.ones((5,5))
img_seg_ero = morphology.binary_erosion(img_seg, se)
fig, ax = plt.subplots(num='erosion 5x5')
ax.imshow(img_seg_ero, 'gray')

se = np.ones((7,7))
img_seg_ero = morphology.binary_erosion(img_seg, se)
fig, ax = plt.subplots(num='erosion 7x7')
ax.imshow(img_seg_ero, 'gray')

img_seg_open = morphology.binary_dilation(img_seg_ero, se)
fig, ax = plt.subplots(num='opening 7x7')
ax.imshow(img_seg_open, 'gray')

img_seg_open_dil = morphology.binary_dilation(img_seg_open, se)
img_seg_open_close = morphology.binary_erosion(img_seg_open_dil, se)
fig, ax = plt.subplots(num='opening and closing 7x7')
ax.imshow(img_seg_open_close, 'gray')

# display final morphological filtering contour on original image
fig, ax = plt.subplots()
ax.imshow(img, 'gray')
cntr = plt.contour(C,R,img_seg_open_close, levels=[.5], colors='red')


# plt.close('all')

# display image with original simple threshold segmentation
fig, ax = plt.subplots()
ax.imshow(img, 'gray')
plt.contour(C,R, img_seg, levels=[0.5], colors='red')

# perform connected component labelling
conn_comp_labels = measure.label(img_seg, background=0)
# uncomment to visualize individual labelled components
# fig, ax = plt.subplots()
# ax.imshow(conn_comp_labels*10, 'gray')
# plt.show()

# props is a list of RegionProperty objects, 1 for each found connected component
props = measure.regionprops(conn_comp_labels)

# use area of each connected component to find the largest one
mxi = 0
for i in range(1,len(props)):
    if props[i].area > props[mxi].area:
        mxi = i

img_seg_cc = props[mxi].label == conn_comp_labels

# visualize the result
plt.contour(C,R,img_seg_cc, levels=[0.5], colors='yellow')
plt.show()
# will continue on 9/16