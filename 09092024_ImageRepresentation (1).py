import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import matplotlib
matplotlib.use('Qt5Agg')

#Create synthetic image and display it
img = np.zeros((10, 15)) # 10 row by 15 column
for i in range(10):
    for j in range(15):
        img[i,j] = i*j

fig, ax = plt.subplots()
plt.imshow(img, 'gray')
plt.colorbar()
plt.xlabel('columns')
plt.ylabel('rows')

# image loading and display
# Put cameraman.tif (Brightspace) in same directory as this script
img2 = cv.imread('cameraman.tif')# grayscale image with rgb channels, let's convert to grayscale
img2 = img2[:,0:250,0]
fig, ax = plt.subplots()
ax.imshow(img2, 'gray')

# image manipulation and output
img2_anon = np.copy(img2)
img2_anon[40:86, 90:135] = 127
fig, ax = plt.subplots()
ax.imshow(img2_anon, 'gray')

cv.imwrite('cameraman_anon.tif', img2_anon)

# inspecting image intensity distributions with histograms
fig, ax = plt.subplots(1,2)
plt.axes(ax[0])
cols = ax[0].imshow(img2, 'gray')
plt.colorbar(cols)
plt.axes(ax[1])
plt.hist(img2.ravel(), bins=np.arange(256))
plt.yscale('log')
plt.ylabel('# of occurences')
plt.xlabel('Pixel intensities')


# Segmentation via Otsu thresholding
# Analyzes histogram to find threshold separating two classes that maximizes inter-class variance
t, img2_thrsh = cv.threshold(img2, 0, 255, cv.THRESH_BINARY_INV+cv.THRESH_OTSU)
plt.plot([t,t], [.1, 10000], 'r')
print(t)
# display result as binary mask img2_thrsh. Each pixel is assigned either 0 or 255
fig2, ax2 = plt.subplots()
ax2.imshow(img2_thrsh, 'gray')
plt.close(fig2)

#display result as isocontour using meshgrid to define pixel coordinates
X,Y = np.meshgrid([0,1], [2,3,4])
print(X)
print(Y)
# meshgrid provides all combinations of input X and Y grid coordinates

# use meshgrid to obtain all combinations of column and row coordinates
C, R = np.meshgrid(np.arange(250), np.arange(256))
plt.close('all')
plt.figure(fig)
plt.axes(ax[0])
# isocontouring draws lines to separate neighboring grid points that have intensities above/below the provided isolevel
# Choosing an isolevel of 127.5 (halfway between 0 and 255) will draw the isocontour halfway between fore and background pixels
cntr = plt.contour(C,R, img2_thrsh, levels=[127.5], colors='red')
# can also directly isocontour the image with the Otsu threshold
cntr = plt.contour(C,R, img2, levels=[t + .5], colors='yellow')


#create an GUI to manually select a threshold
class imageThresholdSelector:
    def __init__(self, fig, ax, img):
        self.img = img
        self.fig = fig
        self.ax = ax
        self.t = 0 # will store selected threshold
        # Pre compute min and max intensities and row/column coordinates for contouring
        self.mn = np.amin(self.img)
        self.mx = np.amax(self.img)
        self.C, self.R = np.meshgrid(np.arange(np.shape(img)[1]), np.arange(np.shape(img)[0]))
        plt.connect('button_press_event', self.on_mouse_click)
        plt.ion()

    def on_mouse_click(self, event):
        # if click is in histogram
        if event.inaxes == self.ax[1]:
            self.t = event.xdata
            # if selected values falls within valid intensity range
            if self.t >=self.mn and self.t<self.mx:
                # redraw the contour
                plt.axes(self.ax[0])
                plt.cla()
                self.ax[0].imshow(self.img, 'gray')
                plt.contour(self.C, self.R, self.img, [self.t], colors='red')

                # redraw the histogram with newly selected threshold
                plt.axes(self.ax[1])
                plt.cla()
                plt.hist(self.img.ravel(), bins=np.arange(256))
                plt.yscale('log')
                plt.plot([self.t, self.t], [.1,10000], 'r')

# create an instance of the class
selector = imageThresholdSelector(fig, ax, img2)
plt.show()

while (1):
    fig.canvas.draw_idle()
    fig.canvas.start_event_loop(.3)
