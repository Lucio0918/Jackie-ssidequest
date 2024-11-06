import json
from volumeViewer import *
import numpy as np
from interactiveGraphCut import *
from graphCut import *

#load a CT image to play with
f = open('CT.json', encoding='utf-8-sig')
d = json.load(f)
f.close()

img = np.array(d['data'], dtype=np.int16)
voxsz = np.array(d['voxsz'], dtype=np.float64)

# downsample in the x and y directions so that graph cut runs faster
img = img[::2,::2,:]
voxsz[0]*=2
voxsz[1]*=2

# start the GUI
# Step 1: Choose some foreground seeds with left-clicks
# Step 2: Choose some background seeds with left-clicks
# Step 3: Generate likelihood functions from histograms
# Step 4: Run graph cuts
# Step 5: inspect results and repeat steps 1-4 as needed
igc = interactiveGraphcut(img, voxsz, sigma=100, lmbda=0.001, alpha=.99)

# Re-running using previously defined seeds
igc2 = interactiveGraphcut(img, voxsz, sigma=100, lmbda=0.001, alpha=.99,
                           seed_fore = np.copy(igc.seed_fore),
                           seed_back = np.copy(igc.seed_back))

# Running it programmatically rather than interactively
# Need to instantiate with parameters, update N links, update T links, update seeds, then run
gc = graphCut(sigma=100, alpha=0.99, lmbda=0.001)
gc.updateNLinks(img)
seed_fore = igc2.seed_fore
seed_back = igc2.seed_back

# using histograms to estimate likelihood functions
mn = np.amin(img)
mx = np.amax(img)
bins = np.linspace(mn,mx,65)
histfore, x = np.histogram(img[seed_fore>0].ravel(), bins=bins)
histback, x = np.histogram(img[seed_back>0].ravel(), bins=bins)
imgbin = np.floor(63.999*(img-mn)/(mx-mn)).astype(np.int32) # intensities mapped to bins
histfore = histfore / np.sum(histfore)
histback = histback / np.sum(histback)
tot = histfore + histback
tot[tot==0] = 1
pdffore = histfore/tot
pdfback = histback/tot
pdffore[pdffore < 0.001] = 0.001
pdfback[pdfback < 0.001] = 0.001

gc.updateTLinks(np.log(pdffore[imgbin]), np.log(pdfback[imgbin]))

# seeds should be labelled 1 for foreground, 2 for background, and 0 for non-seed nodes
cls = np.copy(seed_fore)
cls[cls == 0] = 2 * seed_back[cls == 0]

gc.updateSeeds(cls)

seg = gc.run()

# View the result
vv = volumeViewer()
vv.setImage(img, voxsz)
vv.addMask(seg, color=[0,1,1], opacity=0.5)
vv.display()

