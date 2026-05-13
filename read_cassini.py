import math
import os
from datetime import datetime

import pdr
import wget
from matplotlib import pyplot as plt
plt.plot([0,1],[2,3])
plt.show() # for unknown reasons the plot at the end only runs for me if this is here
import cv2
from skimage import feature
import scipy
import time
from utils import *
import numpy as np
from get_cassini_frames import get_what_to_render








imsize = 0.0302  # 0.0302  # 0.0292#0.0302 # <--- trial and error
instrument = "CASSINI_ISS_WAC"
# if "NARROW" in data_time[2]["INSTRUMENT_NAME"]:
#     imsize /= 2003.44 / 200.77  # 0.00302  # 0.00285
#     instrument = "CASSINI_ISS_NAC"


start_et = spice.str2et("1997-10-15T08:43:00")
end_et   = spice.str2et("2017-09-15T11:55:46")

# start off with 1000 evenly spaced, will add in more later for detail
et_values = np.linspace(start_et, end_et, 1000)



earth_txyz = []
for et_value in et_values:
    try:
        spicey_time = et_value
        position, _ = spice.spkpos("Earth", spicey_time, "J2000", "NONE", ORIGIN_BODY)

        earth_txyz.append([et_value,*position])
    except:
        earth_txyz.append([et_value,0,0,0])
        pass
earth_txyz = np.array(earth_txyz)
np.savetxt("earth_positions.csv",earth_txyz,delimiter=",",fmt="%.16f",header="t,x,y,z",comments="")

saturn_txyz = []
for et_value in et_values:
    try:
        spicey_time = et_value
        position, _ = spice.spkpos("Saturn", spicey_time, "J2000", "NONE", ORIGIN_BODY)

        saturn_txyz.append([et_value,*position])
    except:
        saturn_txyz.append([et_value,0,0,0])
        pass
saturn_txyz = np.array(saturn_txyz)
np.savetxt("saturn_positions.csv",saturn_txyz,delimiter=",",fmt="%.16f",header="t,x,y,z",comments="")

for k in range(3):
    txyz = []
    for et_value in et_values:
        try:
            spicey_time = et_value
            orientation = spice.pxform("J2000", instrument, spicey_time)
            position, _ = spice.spkpos("Cassini", spicey_time, "J2000", "NONE", ORIGIN_BODY)

            txyz.append([et_value,*position,*np.reshape(orientation,(9,))])
        except:
            txyz.append([et_value,0,0,0,0,0,0,0,0,0,0,0,0])
            pass
    txyz = np.array(txyz)

    t_to_use = []
    for i in range(len(txyz)-1):
        t_to_use.append(txyz[i,0])
        try:
            t_between = (txyz[i,0]+txyz[i+1,0])/2
            orientation = spice.pxform("J2000", instrument, t_between)
            position, _ = spice.spkpos("Cassini", t_between, "J2000", "NONE", ORIGIN_BODY)
            from_last = (position-txyz[i,1:4])
            to_next = (txyz[i+1,1:4]-position)
            difference_factor = np.linalg.norm(to_next-from_last)/np.linalg.norm(from_last)

            if difference_factor>0.01:
                # if there's enough of a bend, add an inbetween point
                t_to_use.append(t_between)
        except:
            pass
    t_to_use.append(txyz[-1,0])
    print(len(t_to_use))
    if len(t_to_use)==len(et_values):
        break
    et_values = np.array(t_to_use)

np.savetxt("cassini_positions.csv",txyz,delimiter=",",fmt="%.16f",header="t,x,y,z,m00,m01,m02,m10,m11,m12,m20,m21,m22",comments="")


ellipses_list = []
for et_value in et_values:
    ellipses = get_what_to_render(et_value)
    ellipses_list.append(ellipses)
np.savetxt("ellipses_list.csv",ellipses_list,delimiter=",",fmt="%.16f",comments="")




plt.plot(txyz[:,0],txyz[:,4])
plt.plot(txyz[:,0],txyz[:,5])
plt.plot(txyz[:,0],txyz[:,6])
plt.show()
