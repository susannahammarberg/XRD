"""
 loads data via ptypy and perform xrd analysis from NanoMAX experiment

Prosedure:
* load the data into a ptypy storage by preparing a paramter three p (so i dont need to write my own script to put the data in a ptypy storage)
* load the data from the ptycho object P = Ptycho(p,level=2)
* do bright field analysis
* load the q-vectors from the geometry object and calculate absq
* make a quantitaive plot of a 3d peak with the right axes and the terminology from Berenguer
* test1 where you move the data (intensity points in a 3d array) from the natural basis to the orthogonal basis and plot
* Make a q-space meshgrid: first make a meshgrid from the three scattering vectors in the natural basis
  and move that to the orthogonal basis with the ptypy function transformed_grid
* Do XRD analysis:
* For each position move the 3d scattering data from the natural to the orthogonal basis
* Send the orthogonal basis data along with the orthogonal basis meshgrid to the COM analysis function
* find the COM of the Bragg peak in the orthogonal system
* TODO make some tests where you plot a 3d peak and print out the results from the COM
* TODO check that x y z notation in COM analysis and whn putting it into XRD_x-..etc
* put the 3 coordinate values into 3 separate 2d matrices (dim same as scanning grid)
* plot COM_qx,COM_qy,COM_qz, calculate XRD_absq and 2 rotations. from XRD_absq,calculate strain and plot. mask and plot etc.

Want to know the change of Q in reciprocal space, that is, the orthogoanl space reciprocal to real space.

# XRD analysis: use ptypy coordinate shifting systems to shift the data 

"""
import ptypy
from ptypy.core import Ptycho
from ptypy import utils as u
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from ptypy.experiment.nanomax3d import NanomaxBraggJune2017 # after update need to update spec ptyScan class

# TODO
# if InP
#       choose InP roi and change name

p = u.Param()
p.run = 'XRD_JWX33'   # 'XRD_InP'

sample = 'JWX33_NW2'; scans = range(192, 200+1)+range(205, 222+1)
#sample = 'JWX29A_NW1'; scans = [458,459,460,461,462,463,464,465,466,467,468,469,470,471,518,473,474,475,476,477,478,479,480,481,482,483,484,485,486,519,488, 496,497,498, 499, 500, 501, 502, 503, 504, 505, 506,507, 508, 509, 510, 511, 512, 513, 514, 515]

p.data_type = "single"   #or "double"
# for verbose output
p.verbose_level = 5

# use special plot layout for 3d data  (but the io.home part tells ptypy where to save recons and dumps)
p.io = u.Param()
p.io.home = './'
p.io.autoplot = u.Param()
p.io.autoplot.layout = 'bragg3d'
p.io.autoplot.dump = True
p.io.autoplot.interval=1
 
p.scans = u.Param()
p.scans.scan01 = u.Param()
p.scans.scan01.name = 'Bragg3dModel'
#p.scans.scan01.illumination = illumination
p.scans.scan01.data= u.Param()
p.scans.scan01.data.name = 'NanomaxBraggJune2017'
#base= 'D:/exp20170628_Wallentin_nanomax/exp20170628_Wallentin/'
p.scans.scan01.data.datapath = 'F:/exp20170628_Wallentin_nanomax/exp20170628_Wallentin/%s/' % sample
p.scans.scan01.data.datafile = '%s.h5' % sample
p.scans.scan01.data.detfilepattern = 'scan_%04d_merlin_%04d.hdf5'
# not sure if this loads properly
p.scans.scan01.data.maskfile = 'C:/Users/Sanna/Documents/Beamtime/NanoMAX062017/merlin_mask.h5'
p.scans.scan01.data.scans = scans
#p.scans.scan01.data.center = (155,191)#None     #can also uses =None
p.scans.scan01.data.theta_bragg = 12.0
#p.scans.scan01.data.shape = 256# 150#150#256#150#512#150#60#290#128 --THIS
#p.scans.scan01.data.auto_center= False # dont do it..
# ptypy says: Setting center for ROI from None to [ 75.60081158  86.26238307].   but that must be in the images that iI cut out from the detector


#p.scans.scan01.data.center = (342,245) # (200,270) #(512-170,245)     #(512-170,245) for 192_   #Seems like its y than x
#calculates the center based on the first pic said Alex.  186.75310731  265.64597192] thats not wr

#p.scans.scan01.data.load_parallel = 'all'
p.scans.scan01.data.psize = 55e-6
p.scans.scan01.data.energy = 9.49
p.scans.scan01.data.distance =  0.7   #or 1?

# This shifts the entire scan (projection) in real space, in units of steps 
##############################################################################
#S segmented NWs
##############################################################################
# the smallers values here (minus values) will include the 0:th scanning position
##TODO test with no shifting and compare result: [0]*51 #
#p.scans.scan01.data.vertical_shift =  [-1,-1,0,0,0,  0,0,2,1,0,  1,1,1,0,-1,  -1,-1,-1,-1,0,  -1,-1,0,0,1,  1,-1,0,1,0,   2,0,0,1,1,  1,0,0,1,1,  1,2,2,2,4,  3,3,3,3,3,   3]
#p.scans.scan01.data.horizontal_shift =  [3,2,0,1,2,  3,4,3,4,5,  5,6,6,5,6,  5,4,7,8,8,  8,8,10,11,12,  11,12,12,11,12,  12,11,12,13,13,  14,15,14,14,14,  13,15,16,15,14,  17,19,18,18,17,   17]
            # InGaP = [116:266,234:384]
            # InP = [116:266,80:230]
#p.scans.scan01.data.detector_roi_indices = [116,266,80,230]  #[116,266,80,230]  #    # this one should not be needed since u have shape and center...

##############################################################################
# homogenius NWs
##############################################################################
#p.scans.scan01.data.vertical_shift = [1] * len(scans) 
#p.scans.scan01.data.horizontal_shift = [1] * len(scans) 
            # referens ska vara 0, alla andra -
            #open this file
# when it says horizontal i have used vertical because i am an i
# I will just have fo change the names everyw here             
#p.scans.scan01.data.horizontal_shift = list(np.load('C:\Users\Sanna\Documents\Beamtime\NanoMAX062017\Analysis_ptypy\ptycho_192_222\\horizontal_shift_vector.npy'))
#vertical_shift_vector
#p.scans.scan01.data.vertical_shift = list(np.load('C:\Users\Sanna\Documents\Beamtime\NanoMAX062017\Analysis_ptypy\ptycho_192_222\\vertical_shift_vector.npy'))

# TODO is this what I should use, or the list in the prev. row?            
#p.scans.scan01.data.vertical_shift = [ 0, 0, 1, 2, 2, 2, 3, 3, 3, 1, 0, 0, 0, 0, 0, -2, -2, -2, -3, -3, -2, -2,  -3, -2, -3, -3,  -4]
#p.scans.scan01.data.horizontal_shift = [ -8, -8, -8, -8, -8, -3, -2, -4, 0, -5, -3, -6, -3, -6, -3, -5, -5, -7, -4,  -7, -3, -6, -2, -7, -2, -6, -3 ] 

#p.scans.scan01.data.detector_roi_indices = [275,425,150,300]  # this one should not be needed since u have shape and center...

p.scans.scan01.illumination = u.Param()
p.scans.scan01.illumination.aperture = u.Param() 
p.scans.scan01.illumination.aperture.form = 'circ'
p.scans.scan01.illumination.aperture.size = 100e-9 
p.scans.scan01.sample = u.Param()
p.scans.scan01.sample.fill = 1e-3

p.engines = u.Param()
p.engines.engine00 = u.Param()
p.engines.engine00.name = 'DM'    #Not 'DM_3dBragg' ? 
p.engines.engine00.numiter = 200
p.engines.engine00.probe_update_start = 100000
p.engines.engine00.probe_support = None
#p.engines.engine00.sample_support = None

# p.engines.engine00.sample_support = u.Param()
# p.engines.engine00.sample_support.coefficient = 0.0 
# p.engines.engine00.sample_support.type = 'rod'
# p.engines.engine00.sample_support.size = 200e-9
# p.engines.engine00.sample_support.shrinkwrap = u.Param()
# p.engines.engine00.sample_support.shrinkwrap.cutoff = .3
# p.engines.engine00.sample_support.shrinkwrap.smooth = None
# p.engines.engine00.sample_support.shrinkwrap.start = 15
# p.engines.engine00.sample_support.shrinkwrap.plot = True

# prepare and run
P = Ptycho(p,level=2)

# ---------------------------------------------------------
# Load data, some metadata, gemetry object, and do some initial plotting
#-----------------------------------------------------------

import sys   #to collect system path ( to collect function from another directory)
sys.path.insert(0, 'C:/Users/Sanna/Documents/python_utilities') #can I collect all functions in this folder?
from movie_maker import movie_maker
import h5py

# gather motorpositions from first rotation in scan list for plotting
scan_name_int = scans[2]
scan_name_string = '%d' %scan_name_int 
metadata_directory = p.scans.scan01.data.datapath + sample + '.h5'
metadata = h5py.File( metadata_directory ,'r')
motorpositions_directory = '/entry%s' %scan_name_string  

# gonphi is the rotation of the rotatation stage on which all motors samx/y/z and samsx/y/z and sample rests on.
# when gonphi is orthoganoal to the incoming beam, it is equal to 180. Therfore, 
# get the list of scans numbers and theta (read from gonphi)
def sort_scans_after_theta():
    gonphi_list = []
    # read in all gonphi postins
    for i in range(0,len(scans)):
        scan_name_int = scans[i]
        scan_name_string = '%d' %scan_name_int 
        metadata_directory = p.scans.scan01.data.datapath + sample + '.h5'
        metadata = h5py.File( metadata_directory ,'r')
        motorpositions_directory = '/entry%s' %scan_name_string     
        motorposition_gonphi = np.array(metadata.get(motorpositions_directory + '/measurement/gonphi'))
        gonphi_list.append(motorposition_gonphi[0])

    # order the scan list after gonphi
    # first put them together
    theta_array = 180 - np.array(gonphi_list)
    zipped = zip(theta_array,scans)
    # then sort after the first col
    zipped.sort()
    # Now, because gonphi is gonphi = 180 - theta, and the data is sorted with higher gonphi,
    # for this list to correspond to how the data is sorted in diffdata, this list should be reversed
    zipped = np.flipud(zipped)
    return zipped
scans_sorted_theta = sort_scans_after_theta()  
      
# JW: Define your coordinate system! What are x,y,z, gonphi, and what are their directions
# calculate mean value of dy
# positive samy movement moves sample positive in y (which means the beam moves Down on the sample)
motorpositiony = np.array(metadata.get(motorpositions_directory + '/measurement/samy'))
dy = (motorpositiony[-1] - motorpositiony[0])*1./len(motorpositiony)

# calculate mean value of dx
# instead of samx, you find the motorposition in flysca ns from 'adlink_buff' # obs a row of zeros after values in adlinkAI_buff
motorpositionx_AdLink = np.mean( np.array( metadata.get(motorpositions_directory + '/measurement/AdLinkAI_buff')), axis=0)
motorpositionx_AdLink = np.trim_zeros(motorpositionx_AdLink)
dx = (motorpositionx_AdLink[-1] - motorpositionx_AdLink[0])*1./ len(motorpositionx_AdLink)

# calc number of rows and cols that is used from measuremtn after realigning postions
nbr_rows = len(motorpositiony) #-(np.max(p.scans.scan01.data.vertical_shift)# - np.min(p.scans.scan01.data.vertical_shift))                        
nbr_cols = len(motorpositionx_AdLink)#-(np.max(p.scans.scan01.data.horizontal_shift)# - np.min(p.scans.scan01.data.horizontal_shift))


extent_motorpos = [ 0, dx*nbr_cols,0, dy*nbr_rows]
# load and look at the probe and object
#probe = P.probe.storages.values()[0].data[0]#remember, last index [0] is just for probe  
#obj = P.obj.storages.values()[0].data
# save masked diffraction patterns
# in diff data the rotations are sorted according to gonphi, starting with the LOWEST GONPHI which is the reversed of theta. Hence, diff_data[0] is is theta=12.16 and scan=222 
diff_data = P.diff.storages.values()[0].data*P.mask.storages.values()[0].data[0]#        (storage_data[:,scan_COM,:,:])

# load the geometry instance to acces the geometry parameters
# one geometry connected to each POD but it this case it is the same for each pod (since its all from the same measurement, same setup. Everything is the same, psixe energy, distance etc)
g = P.pods.values()[0].geometry
# g. psize real space psize: ( Rocking curve step (in degrees) and pixel sizes (in meters))
# g.resolution: "3D sample pixel size (in meters)." "doc: Refers to the conjugate (natural) coordinate system as (r3, r1, r2)"

# plot the sum of all used diffraction images
plt.figure()
plt.imshow(np.log10(sum(sum(diff_data))),cmap='jet', interpolation='none')
plt.title('Summed intensity InP NW (log)')
#plt.savefig("summed_intensity") 
#movie_maker(np.log10(abs(probe)))
# plot the sum of all position for one rotations
plot_rotation = 13

plt.figure()
plt.imshow(sum(((diff_data[0:,plot_rotation]))),cmap='jet', interpolation='none')
plt.title('Summed intensity for all rotations for scan %d InP NW (log)'%scans_sorted_theta[plot_rotation][1])

# ---------------------------------------------------------
# Do bright field analysis
#-----------------------------------------------------------

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
    brightfield[jj] = bright_field(diff_data[:,jj,:,:],nbr_cols,nbr_rows)
    #Normalize each image ( so that i can plot a better 3D image)
    #brightfield[jj] = brightfield[jj] / brightfield[jj].max()


def plot_bright_field():
    interval=10 #plotting interval
    #plot every something 2d bright fields
    for ii in range(0,len(scans),interval):
        plt.figure()
        plt.imshow(brightfield[ii], cmap='gray', interpolation='none', extent=extent_motorpos) 
        plt.title('Bright field sorted in gonphi %d'%scans_sorted_theta[ii][1])  
        plt.xlabel('x [$\mu m$]') 
        plt.ylabel('y [$\mu m$]')
        #plt.savefig('C:\Users\Sanna\Documents\Beamtime\NanoMAX062017\Analysis_ptypy\ptycho_192_222\BF_Merlin_hor_ver_aligned\\scan%d'%((scans[ii])), bbox_inches='tight')
        #plt.savefig("BF/scan%d"%scans_gonphi[ii])   
    # plot average bright field image (average over rotation)
    plt.figure()
    plt.imshow(np.mean(brightfield, axis=0), cmap='jet', interpolation='none',extent=extent_motorpos)
    plt.title('Average image from bright field') 
    plt.xlabel('x [$\mu m$]') 
    plt.ylabel('y [$\mu m$]')
    #plt.savefig("BF/Average_brightfield")     
    
    plt.figure()
    plt.imshow(sum(brightfield), cmap='jet', interpolation='none',extent=extent_motorpos)
    plt.title('Bright field summed over all positions') 
    plt.xlabel('x [$\mu m$]') 
    plt.ylabel('y [$\mu m$]')
plot_bright_field()

# help function. Makes fast-check-plotting easier. 
def imshow(data):
    plt.figure()
    plt.imshow(data, cmap='jet')
    plt.colorbar()
    
# define q1 q2 q3 + q_Abs from the geometry function 
def def_q_vectors():
    global q3, q1, q2, q_abs    
    #  units of reciprocal meters [m-1]
    q_abs = 4 * np.pi / g.lam * g.sintheta
    q3 = np.linspace(-g.dq3*g.shape[0]/2.+q_abs, g.dq3*g.shape[0]/2.+q_abs, g.shape[0])    
    q1 = np.linspace(-g.dq1*g.shape[-1]/2., g.dq1*g.shape[-1]/2., g.shape[-1])
    q2 = np.copy(q1)
def_q_vectors()

# --------------------------------------------------------------
# Make a meshgrid of q3 q1 q2 and transform it to qx qz qy.
# Also define the vectors qx qz qy
#----------------------------------------------------------------

# in the transformation is should be input and output: (qx, qz, qy), or (q3, q1, q2).
# make q-vectors into a tuple to transform to the orthogonal system; Large Q means meshgrid, small means vector
Q1,Q3,Q2 = np.meshgrid(q1, q3, q2) # NOTE when you make a mesh grid the first two axes are interchanged!!!! why???

tup = Q3, Q1, Q2   
Qx, Qz, Qy = g.transformed_grid(tup, input_space='reciprocal', input_system='natural')
# NOTE Q2-Qy should not have changed but the other ones should. note 2, Q3 and Qx should be much larger (never negative).
qx = np.linspace(Qx.min(),Qx.max(),g.shape[0])
qz = np.linspace(Qz.min(),Qz.max(),g.shape[-1])
qy = np.linspace(Qy.min(),Qy.max(),g.shape[-1])

#-----------------------------------------------------------------------------    
# Find max scattering position and plot that positions 3d bragg peak in 2d cuts 
# using Berenguer terminology
# Plot the 'naive' Bragg peak (not skewed coordinates) in a single position in 
# 3dim.
# test trying to skew the system (the diff data) from the measurement coordinate system q3q1q2 (in reciprocal space)
# to the orthogonal reciprocal space qxqzqy with the help of the ptypy class coordinate_shift in geometry_class.py 
# (qxqzqy is what ptypy calls it, not Berenguer which has no name for it)
#------------------------------------------------------------------------------

# check which positions has the most intensity, for a nice 3d Bragg peak plot
pos_vect_naive = np.sum(np.sum(np.sum(diff_data, axis =1), axis =1), axis =1)
max_pos_naive = np.argmax(pos_vect_naive)

# find where the peak is in the detector plane
q1max = np.argmax(np.sum(sum(diff_data[max_pos_naive]),axis=0))
q2max = np.argmax(np.sum(sum(diff_data[max_pos_naive]),axis=1))
    
# plot the 'naive' 3d peak of the most diffracting position, and centered on the peak for q1 and q2 
factor = 1E-10  #if you want to plot in m or Angstroms, user 1 or 1E-10
plt.figure()
plt.suptitle('Naive plot of single position Bragg peak (Berenguer terminology) in natural ( measurement) coord system q3q1q2')
plt.subplot(221)
plt.imshow(diff_data[max_pos_naive][g.shape[0]/2,:,:], cmap='jet', interpolation='none', extent=[q1[0]*factor, q1[-1]*factor, q2[0]*factor, q2[-1]*factor])
plt.xlabel('$q_1$ $ (\AA ^{-1}$)')   #l(' [$\mu m$]')#
plt.ylabel('$q_2$ $ (\AA ^{-1}$)') ; plt.colorbar()
plt.subplot(222) # OBS FIRST AXIS IS Y
plt.imshow(diff_data[max_pos_naive][:,q2max,:], cmap='jet', interpolation='none', extent=[q1[0]*factor, q1[-1]*factor, q3[0]*factor, q3[-1]*factor])
plt.xlabel('$q_1$ $ (\AA ^{-1}$)')   #l(' [$\mu m$]')#
plt.ylabel('$q_3$ $ (\AA ^{-1}$)'); plt.colorbar()
plt.subplot(223)
plt.imshow(diff_data[max_pos_naive][:,:,q1max], cmap='jet', interpolation='none', extent=[q2[0]*factor, q2[-1]*factor, q3[0]*factor, q3[-1]*factor])
plt.xlabel('$q_2$ $ (\AA ^{-1}$)')   #l(' [$\mu m$]')#
plt.ylabel('$q_3$ $ (\AA ^{-1}$)'); plt.colorbar()

#TODO make a copy of the data instead of putting into diff storage. but it must be a ptypy storage as input in coordinate_shift.
# here I put the masked data in the data. for all frames. So this I need to keep before xrd analysis. (or make a new storage with masked data) 
P.diff.storages.values()[0].data = P.diff.storages.values()[0].data * P.mask.storages.values()[0].data

test_shift_coord = ptypy.core.geometry_bragg.Geo_Bragg.coordinate_shift(g, P.diff.storages.values()[0], input_space='reciprocal',
                         input_system='natural', keep_dims=True,
                         layer=max_pos_naive)
# check if the data look good and is masked
plt.figure()
plt.imshow(np.log10(sum(test_shift_coord.data[0])))
plt.title('Check if there are hot pixels in here');  plt.colorbar()

factor = 1E-10  #if you want to plot in m or Angstroms, user 1 or 1E-10
plt.figure()
plt.suptitle('Single position Bragg peak in orthogonal system (Berenguer terminology) qxqzqy')
plt.subplot(221)
plt.imshow(test_shift_coord.data[0][g.shape[0]/2,:,:], cmap='jet', interpolation='none', extent=[qz[0]*factor, qz[-1]*factor, qy[0]*factor, qy[-1]*factor])
plt.xlabel('$q_z$ $ (\AA ^{-1}$)')   
plt.ylabel('$q_y$ $ (\AA ^{-1}$)');  plt.colorbar()
plt.subplot(222)
plt.imshow(test_shift_coord.data[0][:,q2max,:], cmap='jet', interpolation='none', extent=[qz[0]*factor, qz[-1]*factor, qx[0]*factor, qx[-1]*factor])
plt.xlabel('$q_z$ $ (\AA ^{-1}$)')   
plt.ylabel('$q_x$ $ (\AA ^{-1}$)'); plt.colorbar()
plt.subplot(223)
plt.imshow(test_shift_coord.data[0][:,:,q1max], cmap='jet', interpolation='none', extent=[qy[0]*factor, qy[-1]*factor, qx[0]*factor, qx[-1]*factor])
plt.xlabel('$q_y$ $ (\AA ^{-1}$)') 
plt.ylabel('$q_x$ $ (\AA ^{-1}$)'); plt.colorbar()


###############################################################################
# XRD analysis
###############################################################################

# TODO: remove single photon count, if the COM is calculated for very small values like pixels with 1 photon counts, 
#then the result will be missleading. Set a threshold that keeps the resulting pixel on a mean value, like if sum(sum(sum(diffPattern)))< threshold. sum(sum(sum()))==bkg_value
# input is 4d matrix with [nbr_diffpatterns][nbr_rotations][nbr_pixels_x][nbr_pixels_y]
def COM_voxels_reciproc(data, vect_Qx, vect_Qz, vect_Qy ):

    # meshgrids for center of mass calculations in reciprocal space
    #TODO order does not matter?
    #TODO
    #TODO
    COM_x = np.sum(data* vect_Qx)/np.sum(data)
    COM_z = np.sum(data* vect_Qz)/np.sum(data)
    COM_y = np.sum(data* vect_Qy)/np.sum(data)

    print 'coordinates in reciprocal space:'
    print COM_x, COM_z, COM_y
    return COM_x, COM_z, COM_y

# loop through all scanning postitions and move the 3D Bragg peak from the 
# natural to the orthogonal coordinate system (to be able to calculate COM)
# Calculate COM for every peak - this gives the XRD matrices
def XRD_analysis():
    print 'hej'
    position_idx = 0
    XRD_x = np.zeros((nbr_rows,nbr_cols))
    XRD_z = np.zeros((nbr_rows,nbr_cols))
    XRD_y = np.zeros((nbr_rows,nbr_cols))

    for row in range(0,nbr_rows):
        for col in range(0,nbr_cols):
            
            # if keep_dims is False, shouldnt the axis qz change? (q1 -->qz)
            data_orth_coord = ptypy.core.geometry_bragg.Geo_Bragg.coordinate_shift(g, P.diff.storages.values()[0], input_space='reciprocal',
                         input_system='natural', keep_dims=False,
                         layer=position_idx)         # layer is the first col in P.diff.storages.values()[0]
      
            # do the 3d COM analysis to find the orthogonal reciprocal space coordinates of each Bragg peak
            # DODO FEl på x y z på det som kommer ut
            COM_x, COM_z, COM_y = COM_voxels_reciproc(data_orth_coord.data[0], Qx, Qz, Qy)
            print 'COM_x'
            # insert coordinate in reciprocal space maps 
            XRD_x[row,col] = COM_x
            XRD_z[row,col] = COM_z
            XRD_y[row,col] = COM_y
            
            position_idx +=1
            
            #plot every other 3d peak and print out the postion of the COM analysis
            if (position_idx%50==0):
                #plt.figure()
                #plt.imshow(sum(P.diff.storages.values()[0][position_idx]), cmap = 'jet', interpolation='none', extent=[ q1[0], q1[-1], q2[0], q2[-1] ])
                #plt.title('before shift')

                plt.figure()
                # TODO check coordinates of extent
                plt.imshow(data_orth_coord.data[0][:,0,:], cmap='jet', extent=[ qy[0], qy[-1], qz[0], qz[-1] ])
                plt.scatter(COM_y, COM_z, s=500, c='red', marker='o')
                plt.title('summed in x. COM found at red dot')

    return XRD_x, XRD_z, XRD_y, data_orth_coord

XRD_x, XRD_z, XRD_y, data_orth_coord = XRD_analysis()

#test plot for the coordinate system: (only works for the last position, the other peaks are not saved)
def test_coordShift():            
    plt.figure()
    plt.imshow(np.log10(np.sum(diff_data[-1],axis=2)), cmap='jet', interpolation='none', extent=[ q1[0], q1[-1], q3[0], q3[-1] ])
    plt.title('natural')
    plt.xlabel('$q_1$ $ (\AA ^{-1}$)')   #l(' [$\mu m$]')#
    plt.ylabel('$q_3$ $ (\AA ^{-1}$)')
    plt.colorbar()
    plt.figure()
    plt.imshow(np.log10(np.sum(data_orth_coord.data[0],axis=2)), cmap='jet', interpolation='none')
    plt.title('cartesian')
    plt.xlabel('$q_x$ $ (\AA ^{-1}$)')
    plt.ylabel('$q_z$ $ (\AA ^{-1}$)')
    plt.colorbar()        
#test_coordShift()

def plot_XRD_xyz():
    # plot reciprocal space map x y z 
    plt.figure()
    plt.subplot(411)
    plt.imshow(XRD_x, cmap='jet',interpolation='none',extent=extent_motorpos)
    plt.title('Reciprocal space map, $q_x$ $ (\AA ^{-1}$) ')
    plt.ylabel('y [$\mu m$]')
    plt.colorbar()
    plt.subplot(412)
    plt.imshow(XRD_y, cmap='jet',interpolation='none',extent=extent_motorpos) 
    plt.title('Reciprocal space map, $q_y$ $ (\AA ^{-1}$) ')
    plt.ylabel('y [$\mu m$]')
    plt.colorbar()
    plt.subplot(413)
    plt.imshow(XRD_z, cmap='jet',interpolation='none',extent=extent_motorpos)
    plt.title('Reciprocal space map, $q_z$ $(\AA ^{-1}$) ')
    plt.ylabel('y [$\mu m$]')
    plt.colorbar()
    plt.subplot(414)
    plt.title('Bright field (sum of all rotations)')
    plt.imshow(sum(brightfield), cmap='jet', interpolation='none',extent=extent_motorpos)
    plt.xlabel('x [$\mu m$]') 
    plt.ylabel('y [$\mu m$]')
    plt.colorbar()
plot_XRD_xyz()

# calc abs q and the angles
XRD_absq =  np.sqrt(XRD_x**2 + XRD_y**2 + XRD_z**2)
XRD_alpha = XRD_y / XRD_z
XRD_beta = -XRD_x / XRD_z

def plot_XRD_polar():    
    # cut the images in x-range:start from the first pixel: 
    # remove the 1-pixel. (cheating) Did not remove it the extent, because then the 0 will not be shown in the X-scale and that will look weird
    start_cutXat = 0 
    # whant to cut to the right so that the scale ends with an even number
    #x-pixel nbr 67 is at 2.0194197798363955
    cutXat = nbr_cols+1 # 67
    # replace the x-scales end-postion in extent_motorposition. 
    extent_motorpos_cut = np.copy(extent_motorpos)
    ###extent_motorpos_cut[1] = 2.0194197798363955 segmentedNW
    
    # plot abs q to select pixels that are 'background', not on wire, and set these pixels to NaN (make them white)
    
    plt.figure()
    #plt.suptitle(
    plt.subplot(411)
    plt.title('Summed up intensity (bright field)') #sum of all rotations
    plt.imshow(sum(brightfield[:,:,0:]), cmap='jet', interpolation='none',extent=extent_motorpos)
    plt.ylabel('y [$\mu m$]')
    #po = plt.colorbar(ticks=(10,20,30,40))#,fraction=0.046, pad=0.04) 
    plt.colorbar()
    # create a mask from the BF matrix, for the RSM analysis
    XRD_mask = np.copy(sum(brightfield))
    XRD_mask[XRD_mask < 81000 ] = np.nan
    XRD_mask[XRD_mask > 0] = 1       #make binary, all values not none to 1

    # if you want no mask use:
    XRD_mask = np.ones((XRD_mask.shape))
    
    plt.subplot(412)   
    #calculate lattice constant a from |q|:                             
    a_lattice_exp = np.pi*2./ XRD_absq *np.sqrt(3)
    #imshow(a_lattice_exp)
    #plt.title('Lattice conastant a [$\AA$]')
    mean_strain = np.nanmean(XRD_mask[:,start_cutXat:cutXat]*a_lattice_exp[:,start_cutXat:cutXat])
    #TODO try with reference strain equal to the center of the largest segment (for InP) # tody try with reference from the other NWs
    #mean_strain = a_lattice_exp[:,start_cutXat:cutXat].max() 
    
    plt.imshow(100*XRD_mask[:,start_cutXat:cutXat]*(a_lattice_exp[:,start_cutXat:cutXat]-mean_strain)/mean_strain, cmap='jet',interpolation='none',extent=extent_motorpos_cut) # not correct!'
    #plt.title('Relative length of Q-vector |Q|-$Q_{mean}$ $(10^{-3}/\AA$)')
    plt.title('Strain $\epsilon$ (%)')
    plt.ylabel('y [$\mu m$]');plt.colorbar()   

    plt.subplot(413)
    plt.imshow(XRD_mask[:,start_cutXat:cutXat]*1E3*XRD_alpha[:,start_cutXat:cutXat], cmap='jet',interpolation='none',extent=extent_motorpos_cut) # not correct!
    # cut in extent_motorposition. x-pixel nbr 67 is at 2.0194197798363955
    plt.title('Rotation around $q_x$ ($mrad$)')
    plt.ylabel('y [$\mu m$]')
    po = plt.colorbar()
    #po = plt.colorbar(ticks=(0,1,2,3,4))
    #po.set_label('Bending around $q_x$ $\degree$')
   
    plt.subplot(414)
    plt.imshow(XRD_mask[:,start_cutXat:cutXat]*1E3*XRD_beta[:,start_cutXat:cutXat], cmap='jet',interpolation='none',extent=extent_motorpos_cut) # not correct!
    plt.title('Rotation around $q_y$ ($mrad$) ')
    plt.ylabel('y [$\mu m$]')
    plt.xlabel('x [$\mu m$]') 
    po = plt.colorbar()
    #po = plt.colorbar(ticks=(5, 10, 15 ))
    #po.set_label('Bending around $q_y$ $\degree$')
    
plot_XRD_polar()


plt.figure()
plt.imshow(2*np.pi/XRD_absq, cmap='jet',interpolation='none',extent=extent_motorpos) 
plt.ylabel('y [$\mu m$]')
po = plt.colorbar()
po.set_label('$d_{hkl}$')


def XRD_lineplot():
    plot = 10
    start = 0
    plt.figure()
    plt.suptitle('XRD lineplots')
    plt.subplot(411)
    for i in range(start,XRD_absq.shape[0]):
        plt.plot(XRD_absq[i,:] ) # choose a row in the middle of the wire    
    plt.legend()
    plt.subplot(412)
    for i in range(start,XRD_absq.shape[0]):
        plt.plot(XRD_alpha[i,:] ) # choose a row in the middle of the wire  
    plt.subplot(413)
    for i in range(start,XRD_absq.shape[0]):
        plt.plot(XRD_beta[i,:] ) # choose a row in the middle of the wire  
    plt.subplot(414)
    #plt.title('BF intensity')
    plt.plot(sum(brightfield)[plot,:] )

#XRD_lineplot()    
    
    
def save_np_array(nparray):
    np.save('filename', nparray)
    

def rocking_curve_plot():
    #458: theta= 13.1    #515 theta = 12.1
    # Is the first point 458? they are sorted according to gonphi, right? In that case it is right.
    # find highest intensity point
    alla = np.sum(np.sum(np.sum(diff_data,axis=1),axis=1),axis=1)
    index_max = np.argmax(alla)
    #theta = np.linspace(12.1,13.1,51    # dthetea =0.02   
    plt.figure(); plt.plot((np.sum(np.sum(diff_data[index_max],axis=1),axis=1)))
    plt.yscale('log')
    plt.title('Rocking curve at highest-intensity point')
    
    plt.ylabel('Photon counts');plt.xlabel('Rotation $\Theta$ ($\degree$)')
    plt.grid(True)
rocking_curve_plot()
 
    
###############################################################################
# Testing a few Mayavi plotting functions
###############################################################################    
#def plot_3d_isosurface():
from mayavi import mlab   #if you can do this instde function it is ood because it changes to QT fram    

# check which positions has the most intensity, for a nice 3d Bragg peak plot
#TODO this is for the 'naive' system (diff_data is)
pos_vect = np.sum(np.sum(np.sum(diff_data, axis =1), axis =1), axis =1)
max_pos = np.argmax(pos_vect)

plt.figure()
plt.text(5, np.max(pos_vect), 'Max at: ' + str(max_pos), fontdict=None, withdash=False)
plt.plot(pos_vect)
plt.title('Summed intensity as a function of position')
#plt.savefig('SumI_vs_position')

plt.figure()
plt.imshow(diff_data[max_pos,5], cmap='jet')
plt.title('InP Bragg peak projection')


# plot this position:
pos = max_pos +0
# change the coordinate system of this data
data_orth_coord = ptypy.core.geometry_bragg.Geo_Bragg.coordinate_shift(g, P.diff.storages.values()[0], input_space='reciprocal',
                         input_system='natural', keep_dims=True,
                         layer=pos) 
# check to see it is fine

plt.figure()
plt.imshow(data_orth_coord.data[0,0])
plt.title('InP Bragg peak projection with fringes, orth coord')

plot_data = diff_data[pos] #data_orth_coord.data[0]     #data[0]0


def slice_plot():
    mlab.pipeline.image_plane_widget(mlab.pipeline.scalar_field(plot_data),
                                plane_orientation='x_axes',
                                slice_index=10,
                            )
    mlab.pipeline.image_plane_widget(mlab.pipeline.scalar_field(plot_data),
                                plane_orientation='y_axes',
                                slice_index=10,
                            )
    mlab.outline()

def plot3dvolume(): #  this looks very good, but almost never works 
    x, y, z = np.ogrid[-10:10:20j, -10:10:20j, -10:10:20j]
    s = np.sin(x*y*z)/(x*y*z)
    mlab.pipeline.volume(mlab.pipeline.scalar_field(s))
    # pipeline.scalar_filed makes data on a regular grid
    #mlab.pipeline.volume(data[max_pos], vmin=0, vmax=0.8)

# TODO axes must be wrong here. check
#def contour3d():   #iso surface
mlab.figure()
xmin=qx[0]*1E0; xmax = qx[-1]*1E0; ymin=qy[0]*1E0; ymax=qy[-1]*1E0; zmin=qz[0]*1E0; zmax=qz[-1]*1E0
obj = mlab.contour3d( plot_data, contours=10, opacity=0.5, transparent=False, extent=[ qz[0], qz[-1],qy[0], qy[-1] , qx[0], qx[-1] ])  #  , vmin=0, vmax=0.8)
mlab.axes(ranges=[xmin, xmax, ymin, ymax, zmin, zmax])
mlab.xlabel('$Q_z$ [$\AA^{-1}$]'); mlab.ylabel('$Q_y$ [$\AA^{-1}$]'); mlab.zlabel('$Q_z$ [$\AA^{-1}$]')
#C:\Users\Sanna\Documents\Beamtime\NanoMAX062017\Analysis_ptypy\scan461_\bragg_peak_stacking\InP\
#mlab.savefig('pos_'+ str(pos) +'.jpg')


# Another way to maka an iso surface. Can also be combined with cut planes
position = 0 
def iso_pipeline_plot():
    src = mlab.pipeline.scalar_field(plot_data)  # this creates a regular space data
    mlab.pipeline.iso_surface(src, contours=[diff_data[position].min()+0.1*diff_data[position].ptp(), ], opacity=0.5)
    mlab.show()    
iso_pipeline_plot()

 
###############################################################################
# MOVIE makers
###############################################################################

# calls a movie maker function in another script
movie_maker(abs((diff_data[:,13])),'all_poitions_S192') 

# alternative movie maker. 
def movie_maker2(data, name, rotation_nbr, nbr_plots):
    if nbr_plots == 1:
        #figure for animation
        fig = plt.figure()
        # Initialize vector for animation data
        ims = []  
        index = 0
        for ii in range(0,len(data)):

                
                im = plt.imshow(np.log10(data[index][rotation_nbr]), animated=True, cmap = 'jet', interpolation = 'none')#, origin='lower')
    #        #plt.clim(0,4) to change range of colorbar
    #        im = plt.title('Col %d'%i)    # does not work for movies
    #        txt = plt.text(0.2,0.8,i)   # (x,y,string)
                ims.append([im])    #ims.append([im, txt])
        ani = animation.ArtistAnimation(fig, ims, interval=2000, blit=True,repeat_delay=0)  
        #txt = plt.text(0.1,0.8,'row: ' + str(y) + ' col: ' + str(x) )  # (x,y,string)
        ims.append([im])
        #ims.append([[im],txt])
        plt.axis('off')
        plt.show()
        # save animation:
        ani.save(name +'.mp4', writer="mencoder") 
    elif nbr_plots == 2:           
        fig, (ax1, ax2)   = plt.subplots(2,1)  
        ims = [] 
        # for x and y index of 1 col
        #index = 0
        for index in range(5,5+21):
            #for x in range(20,50):
    #            im = plt.imshow(np.log10(data[index]), animated=True, cmap = 'jet', interpolation = 'none')#, origin='lower')
                #plt..subplot(21)
                im = ax1.imshow(np.log10(data[index][rotation_nbr]), animated=True, cmap = 'jet', interpolation = 'none')#, origin='lower')
                #plt.subplot(22)
                im2 = ax2.imshow(np.log10(data[index][rotation_nbr]), animated=True, cmap = 'jet', interpolation = 'none')#, origin='lower')
                #plt.subplot(313)
                #im3 = plt.imshow(np.log10(data[index])[2], animated=True, cmap = 'jet', interpolation = 'none')#, origin='lower')
                index += 1
                #plt.clim(0,4) to change range of colorbar
                #im = plt.title('Angle %d'%i)    # does not work for movies
     #           txt = plt.text(0.1,0.8,'row: ' + str(y) + ' col: ' + str(x) )  # (x,y,string)
                #ims.append([[im, im2],txt])
                ims.append([im, im2])           
                
        ani = animation.ArtistAnimation(fig, ims, interval=500, blit=True,repeat_delay=0)  
        plt.axis('on')
        plt.show()
        # save animation:
        #ani.save(name +'.mp4', writer="mencoder")    
movie_maker2(diff_data,'all_diff_data_S192',rotation_nbr=13, nbr_plots=1)
# TODO read out scan number from scans_sorted_theta

plt.figure()
plt.imshow(np.log10(diff_data[147][13]), cmap = 'jet', interpolation = 'none')#, origin='lower')