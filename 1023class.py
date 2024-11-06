import numpy as np
import matplotlib.pyplot as plt

fs= 13200
f1= 440 #puretone A
f2= 660 #pureetone E

T=5 # create a 5 seconds

t=np.linspace(0,T, T*fs, endpoint=False)
y1= .5 * np.cos(2*np.pi * f1 * t)
y2= np.cos(2*np.pi * f2 * t + np.pi /4)

y=f1+ f2 #double pure tone

fig,ax=plt.subplots(1,3,sharey=True,sharex=True)

plt.axes(ax[0])
plt.plot(t,y1)

plt.axes(ax[1])
plt.plot(t,y2)

plt.axes(ax[2])
plt.plot(t,y)
plt.axis([0,3/220,-2,2])





import sksound


plt.show()


