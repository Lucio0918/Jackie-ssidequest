import json
from volumeViewer import *
import numpy as np
from interactiveGraphCut import *
from graphCut import *

f=open('CT.json', encoding='utf-8-sig')
d=json.load(f)
f.close()

img=np.array(d['data'],dtype=np.int16)
