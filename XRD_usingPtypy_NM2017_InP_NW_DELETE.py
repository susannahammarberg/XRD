# load data via ptypy
# this script also does XRD analysis of the InGaP Bragg peak of Bragg ptycho scan 458_---

import ptypy
from ptypy.core import Ptycho
from ptypy import utils as u
import numpy as np

p = u.Param()
p.run = 'XRD_InP'

sample = 'JWX33_NW2'; scans = range(192, 200+1)+range(205, 222+1) # list((192,207,222)) # range(192, 200+1)+range(205, 222+1)  #list((192,203,206))   list((192,207,210))

p.data_type = "single"   #or "double"
# for verbose output
p.verbose_level = 5

# use special plot layout for 3d data  (but the io.home part tells ptypy where to save recons and dumps)
p.io = u.Param()
p.io.home = './'
p.io.autoplot = u.Param()
p.io.autoplot.layout = 'bragg3d'
p.io.autoplot.dump = True

 
p.scans = u.Param()
p.scans.scan01 = u.Param()
p.scans.scan01.name = 'Bragg3dModel'
#p.scans.scan01.illumination = illumination
p.scans.scan01.data= u.Param()
p.scans.scan01.data.name = 'NanomaxBraggJune2017'
#base= 'D:/exp20170628_Wallentin_nanomax/exp20170628_Wallentin/'
p.scans.scan01.data.datapath = 'D:/exp20170628_Wallentin_nanomax/exp20170628_Wallentin/%s/' % sample
p.scans.scan01.data.datafile = '%s.h5' % sample
p.scans.scan01.data.detfilepattern = 'scan_%04d_merlin_%04d.hdf5'
# not sure if this loads properly
p.scans.scan01.data.maskfile = 'C:/Users/Sanna/Documents/Beamtime/NanoMAX062017/merlin_mask.h5' 
p.scans.scan01.data.scans = scans

p.scans.scan01.data.theta_bragg = 12.0
p.scans.scan01.data.shape = 150#512#150#60#290#128
#added now
p.scans.scan01.data.psize = 55e-6
p.scans.scan01.data.energy = 9.49
p.scans.scan01.data.distance = 1
# this is arbitrary, usually it calc this automatically. OBS they mean the center on the raw detector image
p.scans.scan01.data.center = None # x y 
#p.scans.scan01.data.center = (225,350)  # (512-170,245)  #(75,75) #(200,270) #(512-170,245)     #(512-170,245) for 192_   #Seems like its y than x
#p.scans.scan01.data.load_parallel = 'all'

#sannas new parameters
# OBS, must iclude the reference (0-shift position)
        #vertical_shift = [0,1,1] # [0,0,-5]   ## [0,0,-5]    #in the order of the scans (not the angle)
        #pos removes from the beginning nad -from the end (scans for rotations)
        #horizontal_shift = [-5,5,5] #[11,12,11] #[0,0,0]#  #[0,0,5]
p.scans.scan01.data.vertical_shift = [1] * 27 # np.ones((27,1))# [0,1,1]
p.scans.scan01.data.horizontal_shift = [1] * 27 #[-5,5,5] 
            # InGaP = [116:266,234:384]
            # InP = [116:266,80:230]    This is for another ptycho set. probably we moved the detector
p.scans.scan01.data.detector_roi_indices = [275,425,150,300]  # this one should not be needed since u have shape and center...

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
P = Ptycho(p,level=2) #level 2 for XRD analysis

#TODO find read out these (Nx Ny) from P somewhere-
nbr_rows = 17 -(np.max(p.scans.scan01.data.vertical_shift) - np.min(p.scans.scan01.data.vertical_shift))                        
nbr_cols = 21-(np.max(p.scans.scan01.data.horizontal_shift) - np.min(p.scans.scan01.data.horizontal_shift))

import numpy as np
import matplotlib.pyplot as plt
import sys   #to collect system path ( to collect function from another directory)
sys.path.insert(0, 'C:/Users/Sanna/Documents/python_utilities') #can I collect all functions in this folder?
from movie_maker import movie_maker
import h5py

# gather motorpositions for plotting
scan_name_int = 192 
scan_name_string = '%d' %scan_name_int 
metadata_directory = p.scans.scan01.data.datapath + sample + '.h5'
metadata = h5py.File( metadata_directory ,'r')
motorpositions_directory = '/entry%s' %scan_name_string  

#dataset_motorposition_gonphi = metadata.get(motorpositions_directory + '/measurement/gonphi')      
motorpositiony = np.array(metadata.get(motorpositions_directory + '/measurement/samy'))
dy = (motorpositiony[-1] - motorpositiony[0])*1./len(motorpositiony)
# instead of samx, you find the motorposition in flysca ns from 'adlink_buff' # obs a row of zeros after values in adlinkAI_buff
#todo cut awy zeros in a good way
motorpositionx_AdLink = np.mean( np.array( metadata.get(motorpositions_directory + '/measurement/AdLinkAI_buff')), axis=0)[0:21] 
dx = (motorpositionx_AdLink[-1] - motorpositionx_AdLink[0])*1./ len(motorpositionx_AdLink)
extent_motorpos = [ 0, dx*nbr_cols,0, dy*nbr_rows]
# load and look at the probe and object
#probe = P.probe.storages.values()[0].data[0]#remember, last index [0] is just for probe  
#obj = P.obj.storages.values()[0].data
# save masked diffraction patterns as 'a'
a = P.diff.storages.values()[0].data*(P.mask.storages.values()[0].data[0])#        (storage_data[:,scan_COM,:,:])

plt.figure()
plt.imshow(np.log10(sum(sum(a))),cmap='jet', interpolation='none')

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
brightfield = np.zeros((a.shape[1], nbr_rows, nbr_cols))
for jj in range(0,a.shape[1]):
    brightfield[jj] = bright_field(a[:,jj,:,:],nbr_cols,nbr_rows)
    #Normalize each image ( so that i can plot a better 3D image)
    #brightfield[jj] = brightfield[jj] / brightfield[jj].max()


def make_movie():
 #   movie_maker(brightfield)
    movie_maker(abs((a[:,0]))) #movie over summation over all position, loop over rotations. to see where there is signal
#make_movie()
def plot_BF2d():
    #plot every something 2d bright fields
    something = 10
    for ii in range(0,a.shape[1],something):
        plt.figure()
        plt.imshow(brightfield[ii], cmap='jet', interpolation='none', extent=extent_motorpos) 
        plt.title('Single image from bright field %d'%ii)  
        plt.xlabel('x [$\mu m$]') 
        plt.ylabel('y [$\mu m$]')
        #plt.savefig("single_brightfield_%d"%jj)   
    # plot average bright field image (average over rotation)
    plt.figure()
    plt.imshow(np.mean(brightfield, axis=0), cmap='jet', interpolation='none',extent=extent_motorpos)
    plt.title('Average image from bright field') 
    plt.xlabel('x [$\mu m$]') 
    plt.ylabel('y [$\mu m$]')
    #plt.savefig("Average_brightfield")     
    
    plt.figure()
    plt.imshow(sum(brightfield), cmap='jet', interpolation='none',extent=extent_motorpos)
    plt.title('Bright field summed over all positions') 
    plt.xlabel('x [$\mu m$]') 
    plt.ylabel('y [$\mu m$]')
plot_BF2d()

def bright_field_voxels(data,x,y):
    index = 0
    photons = np.zeros((y,x)) 
    for row in range(0,y):
        for col in range(0,x):
            #instead of saving data (0,0), save up all diffpatterns for that position, that is, every 
            photons[row,col] = sum(sum(sum(data[index,:])))#/ max_intensity
            index += 1
            
    return photons
brightfield_voxel = bright_field_voxels(a,nbr_cols,nbr_rows)

nbr_rot = len(scans)


def COM2d(data,nbr_cols,nbr_rows):
    COM_hor = np.zeros((nbr_rot,nbr_rows,nbr_cols))
    COM_ver = np.zeros((nbr_rot,nbr_rows,nbr_cols))
    COM_mag = np.zeros((nbr_rot,nbr_rows,nbr_cols))
    COM_ang = np.zeros((nbr_rot,nbr_rows,nbr_cols))

    # define a vector with length of the length of roi on the detector
    roix = np.linspace(1, data.shape[1], data.shape[1])
    ## define a vector with length of the height of roi on the detector
    roiy = np.linspace(1,data.shape[2],data.shape[2])     #shape 1 or 2 ?
    # meshgrids for center of mass calculations
    X, Y = np.meshgrid(roix,roiy)
    
    COM_hor = np.zeros((nbr_rows,nbr_cols))
    COM_ver = np.zeros((nbr_rows,nbr_cols))
    COM_mag = np.zeros((nbr_rows,nbr_cols))
    COM_ang = np.zeros((nbr_rows,nbr_cols))
    index = 0
    for row in range(0,nbr_rows):
        for col in range(0,nbr_cols):
            COM_hor[row,col] = sum(sum(data[index]*X))/sum(sum(data[index]))
            COM_ver[row,col] = sum(sum(data[index]*Y))/sum(sum(data[index]))
            if row == 0 and col == 0:
                bkg_hor = 152.4#65.1#152#COM_hor[row,col] #152.4#
                bkg_ver = 101.8#64.6#101#COM_ver[row,col]  #101.8#
                        # DPC in polar coordinates. r then phi:
            COM_mag[row, col] = np.sqrt( (COM_hor[row,col]-bkg_hor)**2 + (COM_ver[row,col]-bkg_ver)**2) 
            COM_ang[row, col] = np.arctan( (COM_hor[row,col]) / (COM_ver[row,col]))
    
            index += 1
    return COM_hor, COM_ver, COM_mag, COM_ang

def do_plot_COM2d():
    for jj in range(0,nbr_rot):
        l,m,n,o = COM2d(a[:,jj,:,:],nbr_cols,nbr_rows)
        COM_hor[jj,:,:]=l
        COM_ver[jj,:,:]=m
        COM_mag[jj,:,:]=n
        COM_ang[jj,:,:]=o
        plt.figure()
        plt.imshow(o)
        plt.colorbar()
#do_plot_COM2d()

# TODO: remove single photon count, if the COM is calculated for very small values like pixels with 1 photon counts, then the result will be missleading. Set a threshold that keeps the resulting pixel on a mean value, like if sum(sum(sum(diffPattern)))< threshold. sum(sum(sum()))==bkg_value
# its not so good its a bit wired
# input here is 4d matrix with [nbr_diffpatterns][nbr_rotations][nbr_pixels_x][nbr_pixels_y]
def COM_voxels(data,nbr_cols,nbr_rows):
    # define a vector with length of the length of roi on the detector
    roix = np.linspace(1, data.shape[2], data.shape[2])
    ## define a vector with length of the height of roi on the detector
    roiy = np.linspace(1,data.shape[3],data.shape[3])
    roiz = np.linspace(1,nbr_rot,nbr_rot)    
    # meshgrids for center of mass calculations
    Z, X, Y = np.meshgrid(roix,roiz,roiy)
    
    COM_hor = np.zeros((nbr_rows,nbr_cols))
    COM_ver = np.zeros((nbr_rows,nbr_cols))
    COM_rot = np.zeros((nbr_rows,nbr_cols))
    COM_mag = np.zeros((nbr_rows,nbr_cols))
    COM_ang = np.zeros((nbr_rows,nbr_cols))
    index = 0
    for row in range(0,nbr_rows):
        for col in range(0,nbr_cols):
            threshold = 3000   #dont know how to set this threshold. but should be when the data it is summung is just some single photon ocunts on each image
            if sum(sum(sum(data[index]))) > threshold:
                COM_hor[row,col] = sum(sum(sum(data[index]*X)))/sum(sum(sum(data[index])))
                COM_ver[row,col] = sum(sum(sum(data[index,:]*Y)))/sum(sum(sum(data[index])))
                COM_rot[row,col] = sum(sum(sum(data[index,:]*Z)))/sum(sum(sum(data[index])))
            else:
                COM_hor[row,col] = 13.616534672254996      # == np.mean(COM_hor) without the if-sats
                COM_ver[row,col] = 64.117565383940558
                COM_rot[row,col] = 61.397211314625821             

            if row == 0 and col == 0:
                bkg_hor = 152.4#65.1#152#COM_hor[row,col] #152.4#
                bkg_ver = 101.8#64.6#101#COM_ver[row,col]  #101.8#
                bkg_rot = 0
            # DPC in polar coordinates. r then phi: . although does not make much sence
            COM_mag[row, col] = np.sqrt( (COM_hor[row,col]-bkg_hor)**2 + (COM_ver[row,col]-bkg_ver)**2 + (COM_rot[row,col]-bkg_rot)**2) 
            COM_ang[row, col] = np.arctan( (COM_hor[row,col]) / (COM_ver[row,col]))
    
            index += 1
    return COM_hor, COM_ver, COM_rot, COM_mag, COM_ang

COM_hor,COM_ver,COM_rot,COM_mag,not_corr_ang = COM_voxels(a,nbr_cols,nbr_rows)

def plot_COM():
    plt.figure()
    plt.suptitle('Center of masss analysis')
    plt.subplot(411)
    plt.imshow(COM_hor, cmap='jet',interpolation='none',extent=extent_motorpos) 
    plt.title('COM hor')
    plt.ylabel('y [$\mu m$]')
    plt.colorbar()
    plt.subplot(412)
    plt.imshow(COM_ver, cmap='jet',interpolation='none',extent=extent_motorpos)
    plt.title('COM ver')
    plt.ylabel('y [$\mu m$]')
    plt.colorbar()
    plt.subplot(413)
    plt.imshow(COM_rot, cmap='jet',interpolation='none',extent=extent_motorpos) 
    plt.title('COM rot')
    plt.ylabel('y [$\mu m$]')
    plt.colorbar()
    plt.subplot(414)
    plt.title('Bright field (sum of all rotations)')
    plt.imshow(sum(brightfield), cmap='jet', interpolation='none',extent=extent_motorpos)
    plt.xlabel('x [$\mu m$]') 
    plt.ylabel('y [$\mu m$]')
    plt.colorbar()
    #plt.savefig("C:\Users\Sanna\Documents\Beamtime\NanoMAX062017\Analysis_ptypy\scan461_\COM_3d") 
    
plot_COM()

def imshow(data):
    plt.figure()
    plt.imshow(data, cmap='jet')
    plt.colorbar()
    
    
def numpy2vtk(data,filename,dx=1.0,dy=1.0,dz=1.0,x0=0.0,y0=0.0,z0=0.0):
   # http://www.vtk.org/pdf/file-formats.pdf
   f=open(filename,'w')
   nx,ny,nz=data.shape
   f.write("# vtk DataFile Version 2.0\n")
   f.write("Test data\n")
   f.write("ASCII\n")
   f.write("DATASET STRUCTURED_POINTS\n")
   f.write("DIMENSIONS %u %u %u\n"%(nz,ny,nx))
   f.write("SPACING %f %f %f\n"%(dx,dy,dz))
   f.write("ORIGIN %f %f %f\n"%(x0,y0,z0))
   f.write("POINT_DATA %u\n"%len(data.flat))
   f.write("SCALARS volume_scalars float 1\n")
   f.write("LOOKUP_TABLE default\n")
   for i in data.flat:
     f.write("%f "%i)
   f.close()
   return ()
# save vtk file     
#KO = np.random.randint(1,12,size=(13,8,8))
#vtk_out = numpy2vtk(KO,'KOtest.vtk')
#vtk_out = numpy2vtk(np.log10(a[623,:,140:180,170:240]),'single_braggpeak_log.vtk')


def plot3ddata(data):
    plt.figure()
    plt.suptitle('Single position Bragg peak')
    plt.subplot(221)
    #plt.title('-axis')
    plt.imshow((abs((data[data.shape[0]/2,:,:]))), cmap='jet', interpolation='none') 
    plt.colorbar()
    plt.subplot(222)
    #plt.title('-axis')
    plt.imshow(abs(data[:,data.shape[1]/2,:]), cmap='jet', interpolation='none') 
    plt.colorbar()
    plt.subplot(223)
    #plt.title('-axis')
    plt.imshow((abs(data[:,:,data.shape[2]/2])), cmap='jet', interpolation='none') 
    plt.colorbar()
    
#plot3ddata(a[623,:,140:180,170:240])

#TODO not generall, not updated for InP
def plot3d_singleBraggpeak(data):
    
    # plots the 'naive' Bragg peak (not skewed coordinates) in a single position in 3dim
    dq1 = 0.00027036386641665936    #1/angstrom    dq1=dq2   dthete=0.02 (checked gonphi) (see calculate smapling conditions script)
    # lattice constant Ga(0.51In(0.49)P
    #lattice_constant_a = 5.653E-10 
    #d = lattice_constant_a / np.sqrt(3)
    #q_abs = 2*np.pi / d
    
    energy = 9.49
    wavelength =  1.23984E-9 / energy     #1.2781875567010311e-10
    theta = 12 # degrees or 11.1
    
    # AB calculations dq3= np.deg2rad(self.psize[0]) * 4 * np.pi / self.lam * self.sintheta 
    q_abs = 4 * np.pi / wavelength* np.sin(theta*np.pi/180)     *1E-10
    dq3 = np.deg2rad(0.02) * q_abs
    print dq3
    
    global q3, q1, q2
    q3 = np.linspace(-dq3*a.shape[1]/2+q_abs, dq3*a.shape[1]/2 + q_abs, len(scans))
    
    q1 = np.linspace(-dq1*p.scans.scan01.data.shape/2, dq1*p.scans.scan01.data.shape/2, p.scans.scan01.data.shape)
    q2 = np.copy(q1)
    
    plt.figure()
    plt.suptitle('Single position Bragg peak')
    plt.subplot(221)
    plt.imshow((abs((data[data.shape[0]/2,:,:]))), cmap='jet', interpolation='none', extent=[ -dq1*p.scans.scan01.data.shape/2, dq1*p.scans.scan01.data.shape/2, -dq1*p.scans.scan01.data.shape/2, dq1*p.scans.scan01.data.shape/2]) 
    plt.xlabel('$q_1$ $ (\AA ^{-1}$)')   #l(' [$\mu m$]')#
    plt.ylabel('$q_2$ $ (\AA ^{-1}$)') 
    plt.colorbar()
    # OBS FIRST AXIS IS Y
    plt.subplot(222)
    #plt.title('-axis')
    plt.imshow(abs(data[:,data.shape[1]/2,:]), cmap='jet', interpolation='none', extent=[  -dq1*p.scans.scan01.data.shape/2, dq1*p.scans.scan01.data.shape/2, -dq3*a.shape[1]/2+q_abs, dq3*a.shape[1]/2+q_abs]) 
    plt.xlabel('$q_1$ $ (\AA ^{-1}$)')   #l(' [$\mu m$]')#
    plt.ylabel('$q_3$ $ (\AA ^{-1}$)') 
    plt.colorbar()
    plt.subplot(223)
    #plt.title('-axis')
    plt.imshow((abs(data[:,:,data.shape[2]/2])), cmap='jet', interpolation='none', extent=[ -dq1*p.scans.scan01.data.shape/2, dq1*p.scans.scan01.data.shape/2,-dq3*a.shape[1]/2+q_abs, dq3*a.shape[1]/2+q_abs]) 
    plt.xlabel('$q_2$ $ (\AA ^{-1}$)')   #l(' [$\mu m$]')#
    plt.ylabel('$q_3$ $ (\AA ^{-1}$)') 
    plt.colorbar()
    
    
plot3d_singleBraggpeak(a[0])

 
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def movie_maker2(data, name):
    #figure for animation
    #fig = plt.figure()
    # Initialize vector for animation data
    #ims = []  
#    for i in range(0,5):#len(data)):
#        im = plt.imshow(np.log10(data[i]), animated=True, cmap = 'jet', interpolation = 'none')#, origin='lower')
#        #plt.clim(0,4) to change range of colorbar
#        #im = plt.title('Angle %d'%i)    # does not work for movies
#        txt = plt.text(0.2,0.8,i)   # (x,y,string)
#        ims.append([im, txt])

    fig, (ax1, ax2)   = plt.subplots(2,1)  
    ims = [] 
    # for x and y index of 1 col
    #index = 0
    for index in range(353,367):
        #for x in range(20,50):
#            im = plt.imshow(np.log10(data[index]), animated=True, cmap = 'jet', interpolation = 'none')#, origin='lower')
            #plt..subplot(21)
            im = ax1.imshow(np.log10(data[index+82][0]), animated=True, cmap = 'jet', interpolation = 'none')#, origin='lower')
            #plt.subplot(22)
            im2 = ax2.imshow(np.log10(data[index+82+82][0]), animated=True, cmap = 'jet', interpolation = 'none')#, origin='lower')
            #plt.subplot(313)
            #im3 = plt.imshow(np.log10(data[index])[2], animated=True, cmap = 'jet', interpolation = 'none')#, origin='lower')
            index += 1
            #plt.clim(0,4) to change range of colorbar
            #im = plt.title('Angle %d'%i)    # does not work for movies
 #           txt = plt.text(0.1,0.8,'row: ' + str(y) + ' col: ' + str(x) )  # (x,y,string)
            #ims.append([[im, im2],txt])
            ims.append([im, im2])
                
    ani = animation.ArtistAnimation(fig, ims, interval=500, blit=True,repeat_delay=0)  
    plt.axis('off')
    plt.show()
    # save animation:
    ani.save(name +'.mp4', writer="mencoder")    
#movie_maker2(a[:,22:25],'rot22__InP')

   
# test trying to skew the system (the diff data) from the measurement coordinate system (in reciprocal space) to the orthogonal reciprocal space
# with the help of the ptypy class coordinate_shift in geometry_class.py 

# need to create this object first with the relevant parameters
g = ptypy.core.geometry_bragg.Geo_Bragg(
    psize=[ 0.02   ,  0.000055,  0.000055], 
    shape=[ len(scans), 150, 150] ,
    energy=9.49, #9.7
    distance=1, 
    theta_bragg=12)

# make a copy of P first (not to change in P) (P ptycho instance)
#TODO dont know if this works or if i am just making another reference to P (like if you try to copy a np array without np.copy(array))
copy_P = P     

copy_P.diff.storages.values()[0].data = copy_P.diff.storages.values()[0].data * copy_P.mask.storages.values()[0].data

# make a test transformation (further down i do it for all postions before XRD calc)
# Choose postion (position on 2d scanning grid):
position = 0
hhhh = ptypy.core.geometry_bragg.Geo_Bragg.coordinate_shift(g, copy_P.diff.storages.values()[0], input_space='reciprocal',
                         input_system='natural', keep_dims=True,
                         layer=position)


#plot3d_singleBraggpeak(np.log10(hhhh.data[0]))   #np.log10

# also transform the reciprocal vectors
tup = q1, q2, q3
llll = ptypy.core.geometry_bragg.Geo_Bragg.transformed_grid(g, tup, input_space='reciprocal',input_system='natural')

#compare natural and unscewed coordinate systems:
plt.figure()
plt.imshow(np.log10(np.sum(a[position],axis=2)), cmap='jet', interpolation='none')#, extent=[ q1[0], q1[-1], q3[0], q3[-1] ])
plt.title('natural')
plt.xlabel('$q_1$ $ (\AA ^{-1}$)')   
plt.ylabel('$q_3$ $ (\AA ^{-1}$)')
plt.colorbar()
plt.figure()
plt.imshow(np.log10(np.sum(hhhh.data[0],axis=2)), cmap='jet', interpolation='none')#, extent=[llll[0][0],llll[0][-1],llll[2][0],llll[2][-1] ])
plt.title('cartesian')
plt.xlabel('$q_x$ $ (\AA ^{-1}$)')   #q1~~qx
plt.ylabel('$q_z$ $ (\AA ^{-1}$)')     #q3~qz
plt.colorbar()
# and 3d cuts
plot3d_singleBraggpeak(np.log10(a[position]))
plot3d_singleBraggpeak(np.log10(hhhh.data[0]))

# compare sum of all diffraction images from the data in the 'natural' coordinates and the 
# data that should be tranformed to the 'cartesian' coordinate system
def plot_compare2():
    plt.figure()
    plt.suptitle('Single position Bragg peak summed images in 2 coordinate systems')
    plt.subplot(121)
    #plt.title('-axis')
    plt.imshow(sum(a[0]), cmap='jet', interpolation='none') 
    plt.colorbar()
    plt.subplot(122)
    #plt.title('-axis')
    plt.imshow(sum(abs(hhhh.data[0])), cmap='jet', interpolation='none') 
    plt.colorbar()
plot_compare2()
    

# its not so good its a bit wired
# input here is 4d matrix with [nbr_diffpatterns][nbr_rotations][nbr_pixels_x][nbr_pixels_y]
def COM_voxels_reciproc():
    # define a vector with length of the length of roi on the detector
    #roix = np.linspace(1, data.shape[2], data.shape[2])
    ## define a vector with length of the height of roi on the detector
    #roiy = np.linspace(1,data.shape[3],data.shape[3])
    #roiz = np.linspace(1,nbr_rot,nbr_rot)    
    # meshgrids for center of mass calculations
    #Z, X, Y = np.meshgrid(roix,roiz,roiy)
    
    # meshgrids for center of mass calculations in reciprocal space
    Qx,Qz,Qy = np.meshgrid(q1,q3,q2)
    
    
#    COM_hor = np.zeros((nbr_rows,nbr_cols))
#    COM_ver = np.zeros((nbr_rows,nbr_cols))
#    COM_rot = np.zeros((nbr_rows,nbr_cols))
#    COM_mag = np.zeros((nbr_rows,nbr_cols))
#    COM_ang = np.zeros((nbr_rows,nbr_cols))
#    index = 0
#    for row in range(0,nbr_rows):
#        for col in range(0,nbr_cols):
#            threshold = 3000   #dont know how to set this threshold. but should be when the data it is summung is just some single photon ocunts on each image
#            if sum(sum(sum(data[index]))) > threshold:
    
    COM_x = sum(sum(sum(hhhh.data[0]* Qx)))/sum(sum(sum(hhhh.data[0])))
    COM_y = sum(sum(sum(hhhh.data[0]* Qy)))/sum(sum(sum(hhhh.data[0])))
    COM_z = sum(sum(sum(hhhh.data[0]* Qz)))/sum(sum(sum(hhhh.data[0])))
#            else:
#                COM_hor[row,col] = 13.616534672254996      # == np.mean(COM_hor) without the if-sats
#                COM_ver[row,col] = 64.117565383940558
#                COM_rot[row,col] = 61.397211314625821             
#                print 'peeeeep'
#
#            if row == 0 and col == 0:
#                bkg_hor = 152.4#65.1#152#COM_hor[row,col] #152.4#
#                bkg_ver = 101.8#64.6#101#COM_ver[row,col]  #101.8#
#                bkg_rot = 0
#            # DPC in polar coordinates. r then phi: . although does not make much sence
#            COM_mag[row, col] = np.sqrt( (COM_hor[row,col]-bkg_hor)**2 + (COM_ver[row,col]-bkg_ver)**2 + (COM_rot[row,col]-bkg_rot)**2) 
#            COM_ang[row, col] = np.arctan( (COM_hor[row,col]) / (COM_ver[row,col]))
#    
            #index += 1
    print 'coordinates in reciprocal space:'
    print COM_x, COM_y, COM_z
    return COM_x, COM_y, COM_z

COM_x, COM_y, COM_z = COM_voxels_reciproc()

#def XRD_analysis():
position_obj = 0
XRD_x = np.zeros((nbr_rows,nbr_cols))
XRD_z = np.zeros((nbr_rows,nbr_cols))
XRD_y = np.zeros((nbr_rows,nbr_cols))

for row in range(0,nbr_rows):
    for col in range(0,nbr_cols):
        
        # shift the 3d data (from one position on the wire) from the natural coordinate system to an orthogonal one 
        #make hhhh global if make this into a function
        # hhhh here must be called hhhh because that is what the function COM_voxels_reciprc() uses!!!!!
        hhhh = ptypy.core.geometry_bragg.Geo_Bragg.coordinate_shift(g, copy_P.diff.storages.values()[0], input_space='reciprocal',
                     input_system='natural', keep_dims=True,
                     layer=position_obj)         # layer is the first col in copy_P.diff.storages.values()[0]
        position_obj +=1
        # do the 3d COM analysis to find the orthogonal reciprocal space coordinates
        COM_x, COM_y, COM_z = COM_voxels_reciproc()
        
        # insert coordinate in reciprocal space maps 
        XRD_x[row,col] = COM_x
        XRD_z[row,col] = COM_z
        XRD_y[row,col] = COM_y
#test plot for the coordinate system:   (remove these two plots)
plt.figure()
plt.imshow(np.log10(np.sum(a[position_obj-1],axis=2)), cmap='jet', interpolation='none', extent=[ q1[0], q1[-1], q3[0], q3[-1] ])
plt.title('natural')
plt.xlabel('$q_1$ $ (\AA ^{-1}$)')   #l(' [$\mu m$]')#
plt.ylabel('$q_3$ $ (\AA ^{-1}$)')
plt.colorbar()
plt.figure()
plt.imshow(np.log10(np.sum(hhhh.data[0],axis=2)), cmap='jet', interpolation='none', extent=[llll[0][0],llll[0][-1],llll[2][0],llll[2][-1] ])
plt.title('cartesian')
plt.xlabel('$q_x$ $ (\AA ^{-1}$)')   #q1~~qx
plt.ylabel('$q_z$ $ (\AA ^{-1}$)')     #q3~qz
plt.colorbar()        


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
     # cut the images in x-range: 
    # cut in extent_motorposition. x-pixel nbr 67 is at 2.0194197798363955
    extent_motorpos[1] = 2.0194197798363955
    # remove the 1-pixel. (cheating) Did not remove it the extent, because then the 0 will not be shown in the X-scale and that will look weird
    start_cutXat = 0 
    cutXat = len(XRD_absq[0])         #67
    # plot abs q to select pixels that are 'background', not on wire, and set these pixels to NaN (make them white)
    
    plt.figure()
    #plt.suptitle(
    plt.subplot(411)
    plt.title('Summed up intensity (bright field)') #sum of all rotations
    plt.imshow(sum(brightfield[:,:,0:cutXat]), cmap='jet', interpolation='none',extent=extent_motorpos)
    plt.ylabel('y [$\mu m$]')
    #po = plt.colorbar(ticks=(10,20,30,40))#,fraction=0.046, pad=0.04) 
    plt.colorbar()
    # create a mask from the BF matrix, for the RSM analysis
    XRD_mask = np.copy(sum(brightfield))
    XRD_mask[XRD_mask < 40000 ] = np.nan
    XRD_mask[XRD_mask > 0] = 1       #make binary, all values not none to 1

    # if you want no mask use:
    XRD_mask = np.ones((XRD_mask.shape))
    
    plt.subplot(412)   
    #calculate lattice constant a from |q|:
    a_lattice_exp = np.pi*2./ XRD_absq *np.sqrt(3)
    mean_strain = np.nanmean(XRD_mask[:,start_cutXat:cutXat]*a_lattice_exp[:,start_cutXat:cutXat])
    #try with reference strain equal to the center of the largest segment (for InP) # tody try with reference from the other NWs
    #mean_strain = a_lattice_exp[:,start_cutXat:cutXat].max() 
    
    plt.imshow(100*XRD_mask[:,start_cutXat:cutXat]*(a_lattice_exp[:,start_cutXat:cutXat]-mean_strain)/mean_strain, cmap='jet',interpolation='none',extent=extent_motorpos) # not correct!'
    #plt.title('Relative length of Q-vector |Q|-$Q_{mean}$ $(10^{-3}/\AA$)')
    plt.title('Strain $\epsilon$ (%)')
    plt.ylabel('y [$\mu m$]');plt.colorbar()   

    plt.subplot(413)
    plt.imshow(XRD_mask[:,start_cutXat:cutXat]*1E3*XRD_alpha[:,start_cutXat:cutXat], cmap='jet',interpolation='none',extent=extent_motorpos) # not correct!
    # cut in extent_motorposition. x-pixel nbr 67 is at 2.0194197798363955
    plt.title('Rotation around $q_x$ ($mrad$)')
    plt.ylabel('y [$\mu m$]')
    po = plt.colorbar(ticks=(0,1,2,3,4))
    #po.set_label('Bending around $q_x$ $\degree$')
   
    plt.subplot(414)
    plt.imshow(XRD_mask[:,start_cutXat:cutXat]*1E3*XRD_beta[:,start_cutXat:cutXat], cmap='jet',interpolation='none',extent=extent_motorpos) # not correct!
    plt.title('Rotation around $q_y$ ($mrad$) ')
    plt.ylabel('y [$\mu m$]')
    plt.xlabel('x [$\mu m$]') 
    po = plt.colorbar()
    #po = plt.colorbar(ticks=(5, 10, 15 ))
    #po.set_label('Bending around $q_y$ $\degree$')
plot_XRD_polar()


plt.figure()
plt.imshow(2*np.pi/XRD_absq, cmap='jet',interpolation='none',extent=extent_motorpos) # not correct!'
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

XRD_lineplot()    
    
    
def save_np_array(nparray):
    np.save('filename', nparray)
    

def rocking_curve_plot():
    #458: theta= 13.1    #515 theta = 12.1
    # Is the first point 458? they are sorted according to gonphi, right? In that case it is right.
    # find highest intensity point
    alla = np.sum(np.sum(np.sum(a,axis=1),axis=1),axis=1)
    index_max = np.argmax(alla)
    theta = np.linspace(12.1,13.1,51)    # dthetea =0.02   
    plt.figure(); plt.plot(theta,(np.sum(np.sum(a[index_max],axis=1),axis=1)))
    plt.yscale('log')
    plt.title('Rocking curve at highest-intensity poaint (nbr 536)');plt.ylabel('Photon counts');plt.xlabel('Rotation $\Theta$ ($\degree$)')
    plt.grid(True)
#rocking_curve_plot()


instruct = np.zeros((11,82))
index = 0 
for qqq in range(0,11):
    for qqq2 in range(0,82):
        instruct[qqq,qqq2]=index
        index +=1


