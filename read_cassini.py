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

import spiceypy as spice
import numpy as np

def axis_rotmat(axis,angle=None):
    if angle is None:
        if np.linalg.norm(axis)==0:
            return np.eye(3)
        angle = np.linalg.norm(axis)
        axis = axis/angle
    if angle==0:
        return np.eye(3)
    axis = normalize(axis)
    sa = np.sin(angle)
    ca = np.cos(angle)
    omc = (1-np.cos(angle))
    return np.array([
        [ca+axis[0]**2*omc,axis[0]*axis[1]*omc-axis[2]*sa,axis[0]*axis[2]*omc+axis[1]*sa],
        [axis[1]*axis[0]*omc+axis[2]*sa,ca+axis[1]**2*omc,axis[1]*axis[2]*omc-axis[0]*sa],
        [axis[2]*axis[0]*omc-axis[1]*sa,axis[2]*axis[1]*omc+axis[0]*sa,ca+axis[2]**2*omc]
    ])


ORIGIN_BODY = "Sun"



# 2000-275T08:43:11.695   to 2000-305T15:08:02.460
# 2000-10-01T08:43:11.696 to 2000-10-31T15:08:02
# can be run without:
# spice.furnsh('spice_furnish/00275_00305ca_ISS.bc')
# spice.furnsh('spice_furnish/00275_00305ca_ISS.lbl')
# spice.furnsh('12244_12249rb.bc')
# spice.furnsh('14233_14238ra.bc')

# reconstructed cassini ephemeris, for titan in front of saturn
spice.furnsh("../cassini/spice_furnish/12124_12129ra.bc")
# for same
# spice.furnsh('12097_12170py_as_flown.bc')
# for same
# spice.furnsh('12122_12152ca_ISS.bc')
# predicted orientation
# spice.ckbrief('12097_12170pf_live.bc')
# spice.furnsh('12097_12170pf_live.bc')

for file_name in os.listdir("../cassini/spice_furnish/"):
    if file_name.endswith("ra.bc"):
        spice.furnsh("../cassini/spice_furnish/" + file_name)
for file_name in os.listdir("../cassini/spice_furnish/"):
    if file_name.endswith("rb.bc"):
        spice.furnsh("../cassini/spice_furnish/" + file_name)
for file_name in os.listdir("../cassini/spice_furnish/"):
    if file_name.endswith("rc.bc"):
        spice.furnsh("../cassini/spice_furnish/" + file_name)
    if file_name.endswith(".bsp") and ("R_SC" in file_name or "P_SE" in file_name):
        spice.furnsh("../cassini/spice_furnish/" + file_name)
for file_name in os.listdir("../../sediment/"):
    if file_name.endswith(".bsp") and ("SCEPH_L" in file_name):
        spice.furnsh("../../sediment/" + file_name)
    # if file_name.endswith(".bsp"):
    #     spice.furnsh('../../sediment/'+file_name)

# # Cassini ephemeris from launch to saturn
# spice.furnsh('spice_furnish/971103_SCEPH_LP0_SP0.bsp')
# # Cassini ephemeris from 2001 to 2004
# spice.furnsh('spice_furnish/041014R_SCPSE_01066_04199.bsp')

# everything
# spice.furnsh('spice_furnish/171215R_SCPSEops_97288_17258.bsp')
# specifically for titan in front of saturn
spice.furnsh("../cassini/spice_furnish/180628RU_SCPSE_12116_12136.bsp")
# Planet Ephemeris from 1994 to 2016
# spice.furnsh('spice_furnish/040506AP_PE_94328_16357.bsp')

# spice.furnsh('spice_furnish/00001_00092rc.bc')
# #spice.furnsh('spice_furnish/00001_00092rc.lbl')
# spice.furnsh('spice_furnish/00183_00275rc.bc')
# #spice.furnsh('spice_furnish/00183_00275rc.lbl')
# spice.furnsh('spice_furnish/00275_01001rc.bc')
# spice.furnsh('spice_furnish/00275_01001rc.bc.lbl')

# can be run without:
# spice.furnsh('spice_furnish/cas_caps_00275_00306_v2.bc')

# SPK files
# Irregular Satellites of Saturn

# spice.furnsh('spice_furnish/140809BP_IRRE_00256_25017.bsp')
# spice.furnsh('spice_furnish/140809BP_IRRE_00256_25017.bsp.lbl')
# spice.furnsh('spice_furnish/040506AP_PE_94328_16357.bsp.lbl')

# # combined, has Cassini relative to Sun, Saturn Barycenter
# spice.furnsh('spice_furnish/010420R_SCPSE_EP1_JP83.bsp')
# spice.furnsh('spice_furnish/010420R_SCPSE_EP1_JP83.bsp.lbl')

# Cassini from Jupiter to Saturn
# spice.furnsh('spice_furnish/000331_SK_JM229_SP0.bsp')
# spice.furnsh('spice_furnish/000331_SK_JM229_SP0.bsp.lbl')

# print(spice.ktotal())
# print(rtast)
# spice.furnsh('spice_furnish/00306_00335ca_ISS.bc')
# spice.furnsh('spice_furnish/00306_00335ca_ISS.lbl')
# mentions that:
# CASSINI_ISS_WAC  = -82360
# CASSINI_SC_COORD = -82000
# can be run without:
# spice.furnsh('spice_furnish/cas_v37.tf')
# defines CASSINI_SC_COORD relative to J2000
spice.furnsh("../cassini/spice_furnish/cas_v43.tf")

# leapseconds kernal
spice.furnsh("../cassini/spice_furnish/naif0007.tls")

# SCLK kernel
# spice.furnsh('spice_furnish/cas00172.tsc')
spice.furnsh("../cassini/cas00155.tsc")




imsize = 0.0302  # 0.0302  # 0.0292#0.0302 # <--- trial and error
instrument = "CASSINI_ISS_WAC"
# if "NARROW" in data_time[2]["INSTRUMENT_NAME"]:
#     imsize /= 2003.44 / 200.77  # 0.00302  # 0.00285
#     instrument = "CASSINI_ISS_NAC"


start_et = spice.str2et("1997-10-15T08:43:00")
end_et   = spice.str2et("2017-09-15T11:55:46")

# Create 1000 evenly spaced ET values
et_values = np.linspace(start_et, end_et, 1000)
print(et_values)
txyz = []
for et_value in et_values:

    try:
        spicey_time = et_value#spice.str2et(str("2007-06-05T12:34:56"))
        # print(spicey_time)
        orientation = spice.pxform("J2000", instrument, spicey_time)
        position, _ = spice.spkpos("Cassini", spicey_time, "J2000", "NONE", ORIGIN_BODY)

        txyz.append([et_value,*position])
        print("position",et_value,position)
        # print("orientation",orientation)
    except:
        txyz.append([et_value,0,0,0])
        pass
print("darsdarsdrsd")
txyz = np.array(txyz)
np.savetxt("cassini_positions.csv",txyz,delimiter=",",fmt="%.16f",header="t,x,y,z",comments="")
# np.savetxt("cassini_positions.csv",txyz,delimiter=",")
plt.plot(txyz[:,0],txyz[:,1])
plt.plot(txyz[:,0],txyz[:,2])
plt.plot(txyz[:,0],txyz[:,3])
plt.show()
