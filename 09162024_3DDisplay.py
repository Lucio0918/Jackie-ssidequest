import matplotlib.pyplot as plt
import numpy as np
import json

# Load a CT image to demo 3D display
f = open('CT.json', encoding='utf-8-sig')
d = json.load(f)
f.close()

img = np.array(d['data'], dtype=np.int16)
voxsz = np.array(d['voxsz'], dtype=np.float64)

print(np.shape(img)) # [512, 512, 107]
print(voxsz)# [1.12, 1.12, 3]

# Choose a 2D slice to visualize, start with Axial slice 80, custom level+contrast
slc = 80
level = 0
contrast = 1000
# CT uses Hounsfield Units typically limitted to range [-1024, 3071]
# this level+contrast maps white to intensities >=500, black to <=-500, gray to 0
fig, ax = plt.subplots()
ax.imshow(img[:, :, slc].T, 'gray', interpolation='bilinear',
          vmin=level-contrast/2, vmax=level+contrast/2)
ax.set_aspect(voxsz[1]/voxsz[0])
plt.xlabel('x')
plt.ylabel('y')
plt.title(f'Slice z = {slc}')

# repeat with sagittal slice direction
slc = 255
level = 0
contrast = 1000
fig, ax = plt.subplots()
ax.imshow(img[slc, :, :].T, 'gray', interpolation='bilinear',
          vmin=level-contrast/2, vmax=level+contrast/2)
ax.set_aspect(voxsz[2]/voxsz[1])
ax.set_ylim(bottom=0, top=np.shape(img)[2]-1)
plt.xlabel('y')
plt.ylabel('z')
plt.title(f'Slice x = {slc}')

# coronal direction
slc = 255
level = 0
contrast = 1000
# CT uses Hounsfield Units [-1000, 3071]
fig, ax = plt.subplots()
ax.imshow(img[:, slc, :].T, 'gray', interpolation='bilinear',
          vmin=level-contrast/2, vmax=level+contrast/2)
ax.set_aspect(voxsz[2]/voxsz[0])
ax.set_ylim(bottom=0, top=np.shape(img)[2]-1)
plt.xlabel('x')
plt.ylabel('z')
plt.title(f'Slice y = {slc}')

# Create a class to interactively change the displayed slice
class Display3D:
    def __init__(self, fig, ax, img, voxsz, slc, contrast, level):
        self.img = img
        self.voxsz = voxsz
        self.fig = fig
        self.ax = ax
        self.slc = slc # this variable will store which slice is currently displayed
        self.contrast = contrast
        self.level = level
        plt.connect('key_press_event', self.on_key_press)
        plt.ion()

    def on_key_press(self, event):
        # if key is up arrow or 'a', and the current slice isn't the last one, move slice number up by 1
        if event.key == 'up' or event.key == 'a':
            if self.slc < np.shape(self.img)[1]-1:
                self.slc += 1
        # if key is down arrow or 'z', and the current slice isn't the first one, move slice number down by 1
        elif event.key == 'down' or event.key == 'z':
            if self.slc > 0:
                self.slc -= 1

        # repaint the figure wih the new slice
        plt.cla()
        ax.imshow(self.img[:,self.slc,:].T,'gray',interpolation='bilinear',
                  vmin=self.level - self.contrast / 2,vmax=self.level + self.contrast / 2)
        ax.set_aspect(self.voxsz[2] / self.voxsz[0])
        ax.set_ylim(bottom=0,top=np.shape(self.img)[2] - 1)
        plt.xlabel('x')
        plt.ylabel('z')
        plt.title(f'Slice y = {self.slc}')

#uncomment to try this class
# d3d = Display3D(fig, ax, img, voxsz, slc, contrast, level)
# plt.show()
# while (1):
#     fig.canvas.draw_idle()
#     fig.canvas.start_event_loop(.3)


# Try the volumeViewer class I am providing
from volumeViewer import *
vv = volumeViewer()
vv.setImage(img, voxsz, contrast=contrast, level=level)

msk = img > 1500
vv.addMask(msk, color=[0,1,0])


# controls:
# main window:
#   up,down,'a','z' page through slices
#   'g','v' adjust contrast
#   'd','c' adjust level
#   escape or q to close the figure
#   double left-click centers all three views on a point
#   double right-click resets the view
#   Pyplot's built in zoom/pan functions
# 3D window:
#   hold right click to zoom
#   hold left click to rotate
#   hold middle-mouse button to pan
#   press 'u' to pick a point
vv.display()