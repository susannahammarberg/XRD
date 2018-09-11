# -*- coding: utf-8 -*-
"""
Created on Tue Sep 11 11:54:10 2018

@author: Sanna
"""

# load data via ptypy
# this script also does XRD analysis of the InGaP Bragg peak of Bragg ptycho scan 458_---

import ptypy
from ptypy.core import Ptycho
from ptypy import utils as u
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation




import sys   #to collect system path ( to collect function from another directory)
sys.path.insert(0, 'C:/Users/Sanna/Documents/python_utilities') #can I collect all functions in this folder?
from movie_maker import movie_maker
import h5py


sample = 'JWX33_NW2'; scans = range(192, 200+1)+range(205, 222+1)



# gather motorpositions from first rotation in scan list for plotting
scan_name_int = scans[0]
scan_name_string = '%d' %scan_name_int 
metadata_directory = p.scans.scan01.data.datapath + sample + '.h5'
metadata = h5py.File( metadata_directory ,'r')
motorpositions_directory = '/entry%s' %scan_name_string  

motorposition_gonphi = np.array(metadata.get(motorpositions_directory + '/measurement/gonphi'))
# calculate mean value of dy
motorpositiony = np.array(metadata.get(motorpositions_directory + '/measurement/samy'))
dy = (motorpositiony[-1] - motorpositiony[0])*1./len(motorpositiony)
# calculate mean value of dx
# instead of samx, you find the motorposition in flysca ns from 'adlink_buff' # obs a row of zeros after values in adlinkAI_buff
motorpositionx_AdLink = np.mean( np.array( metadata.get(motorpositions_directory + '/measurement/AdLinkAI_buff')), axis=0)
motorpositionx_AdLink = np.trim_zeros(motorpositionx_AdLink)
dx = (motorpositionx_AdLink[-1] - motorpositionx_AdLink[0])*1./ len(motorpositionx_AdLink)

#TODO find read out these (Nx Ny) from P somewhere-
nbr_rows = len(motorpositiony) -(np.max(p.scans.scan01.data.vertical_shift) - np.min(p.scans.scan01.data.vertical_shift))                        
nbr_cols = len(motorpositionx_AdLink)-(np.max(p.scans.scan01.data.horizontal_shift) - np.min(p.scans.scan01.data.horizontal_shift))


extent_motorpos = [ 0, dx*nbr_cols,0, dy*nbr_rows]
# load and look at the probe and object
#probe = P.probe.storages.values()[0].data[0]#remember, last index [0] is just for probe  
#obj = P.obj.storages.values()[0].data
# save masked diffraction patterns as 'data'
data = P.diff.storages.values()[0].data*(P.mask.storages.values()[0].data[0])#        (storage_data[:,scan_COM,:,:])

# shape paramter to make code readable
shape = p.scans.scan01.data.shape
nbr_rot = len(scans)

# plot the sum of all used diffraction images
plt.figure()
plt.imshow(np.log10(sum(sum(data))),cmap='jet', interpolation='none')

#movie_maker(np.log10(abs(probe)))

def bright_field(data,x,y):
    index = 0
    photons = np.zeros((y,x)) 
    for row in range(0,y):
        for col in range(0,x):
            photons[row,col] = sum(sum(data[index])) #/ max_intensity
            index += 1            
    return photons

# do BF for all rotations
brightfield = np.zeros((len(scans), nbr_rows, nbr_cols))
for jj in range(0,len(scans)):
    brightfield[jj] = bright_field(data[:,jj,:,:],nbr_cols,nbr_rows)
    #Normalize each image ( so that i can plot a better 3D image)
    #brightfield[jj] = brightfield[jj] / brightfield[jj].max()


def plot_BF2d():
    interval=1 #plotting interval
    #plot every something 2d bright fields
    for ii in range(0,len(scans),interval):
        plt.figure()
        plt.imshow(brightfield[ii], cmap='jet', interpolation='none', extent=extent_motorpos) 
        plt.title('Bright field sorted in gonphi %d'%ii)  
        plt.xlabel('x [$\mu m$]') 
        plt.ylabel('y [$\mu m$]')
        plt.savefig("BF/gonphi%d"%ii)   
    # plot average bright field image (average over rotation)
    plt.figure()
    plt.imshow(np.mean(brightfield, axis=0), cmap='jet', interpolation='none',extent=extent_motorpos)
    plt.title('Average image from bright field') 
    plt.xlabel('x [$\mu m$]') 
    plt.ylabel('y [$\mu m$]')
    plt.savefig("BF/Average_brightfield")     
    
    plt.figure()
    plt.imshow(sum(brightfield), cmap='jet', interpolation='none',extent=extent_motorpos)
    plt.title('Bright field summed over all positions') 
    plt.xlabel('x [$\mu m$]') 
    plt.ylabel('y [$\mu m$]')
plot_BF2d()
