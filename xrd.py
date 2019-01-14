# -*- coding: utf-8 -*-

"""
XRD as a separate function
"""



print 'test3'
# input P
def XRD_fun(P, scans, shape):
	

  import ptypy
  from ptypy.core import Ptycho
  from ptypy import utils as u
  import numpy as np
  import matplotlib.pyplot as plt
	
  diff_data= P.diff.storages.values()[0].data * P.mask.storages.values()[0].data[0][0]

  diff_data[:,:,76,74] = 0 
  ## must define these
  nbr_rows = 8
  nbr_cols = 12

  print 'starting xrd analysis'
  plt.figure(1)
  plt.imshow(sum(sum(diff_data)
	
  def bright_field(data,x,y):
      import numpy as np
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



  # define q1 q2 q3 and make them global.
  #TODO  Read in from P

  def def_q_vectors():
    import numpy as np
    print 'qvectors'
    global dq1, dq2, dq3, q_abs
    dq1=0.00027036386641665936    #1/angstrom    dq1=dq2   dthete=0.02 (checked gonphi) (see calculate smapling conditions script)
    # lattice constant Ga(0.51In(0.49)P
    #lattice_constant_a = 5.653E-10 
    #d = lattice_constant_a / np.sqrt(3)
    #q_abs = 2*np.pi / d
    
    energy = 9.49#8.800#9.5   #10.72   #keV    
    wavelength =  1.23984E-9 / energy     #1.2781875567010311e-10
    theta = 11 # degrees
    
    # AB calculations dq3= np.deg2rad(self.psize[0]) * 4 * np.pi / self.lam * self.sintheta 
    q_abs = 4 * np.pi / wavelength* np.sin(theta*np.pi/180)     *1E-10
    
    dq3= np.deg2rad(0.02) * q_abs
    global q3, q1, q2
    
    q3 = np.linspace(-dq3*len(scans)/2.+q_abs, dq3*len(scans)/2+q_abs, len(scans))    
    q1 = np.linspace(-dq1*shape/2, dq1*shape/2, shape)
    q2 = np.copy(q1)
  def_q_vectors()



  def COM_voxels_reciproc(data, vect1, vect2, vect3):
      import numpy as np
      # meshgrids for center of mass calculations in reciprocal space
      #TODO correct order?
      #TODO
      #TODO
      #TODO
      Qx,Qz,Qy = np.meshgrid(vect1,vect3,vect2)
      
      COM_x = sum(sum(sum(data* Qx)))/sum(sum(sum(data)))
      COM_y = sum(sum(sum(data* Qy)))/sum(sum(sum(data)))
      COM_z = sum(sum(sum(data* Qz)))/sum(sum(sum(data)))

      print 'coordinates in reciprocal space:'
      print COM_x, COM_y, COM_z
      return COM_x, COM_y, COM_z

  
  # one geometry connected to each POD but it this case it is the same for each pod.
  g = P.pods.values()[0].geometry

  # here I put the masked data in the data. 
  #3 Here I do that for all frames  
  # So this I need to keep
  P.diff.storages.values()[0].data = P.diff.storages.values()[0].data * P.mask.storages.values()[0].data
                 
  # also transform the reciprocal vectors   
  tup = q1, q2, q3
  q1_orth, q2_orth, q3_orth = ptypy.core.geometry_bragg.Geo_Bragg.transformed_grid(g, tup, input_space='reciprocal',input_system='natural')


  # loop through all scanning postitions and move the 3D Bragg peak from the 
  # natural to the orthogonal coordinate system (to be able to calculate COM)
  # Calculate COM for every peak - this gives the XRD matrices
  def XRD_analysis():
      import numpy as np
      from ptypy.core import Ptycho
      position_idx = 0
      XRD_x = np.zeros((nbr_rows,nbr_cols))
      XRD_z = np.zeros((nbr_rows,nbr_cols))
      XRD_y = np.zeros((nbr_rows,nbr_cols))
      print 'xrd_analysis function'
      
      for row in range(0,nbr_rows):
	  for col in range(0,nbr_cols):
	      
	      
	      data_orth_coord = ptypy.core.geometry_bragg.Geo_Bragg.coordinate_shift(g, P.diff.storages.values()[0], input_space='reciprocal',
			  input_system='natural', keep_dims=True,
			  layer=position_idx)         # layer is the first col in P.diff.storages.values()[0]
	      
	      # do the 3d COM analysis to find the orthogonal reciprocal space coordinates of each Bragg peak
	      #TODO what units comes out of this function?
	      COM_x, COM_y, COM_z = COM_voxels_reciproc(data_orth_coord.data[0], q1_orth, q2_orth, q3_orth)
	      print 'COM_x'
	      # insert coordinate in reciprocal space maps 
	      XRD_x[row,col] = COM_x
	      XRD_z[row,col] = COM_z
	      XRD_y[row,col] = COM_y
	      
	      position_idx += 1
	      
	      # plot every other 3d peak and print out the postion of the COM analysis
	      #if (position_idx%10=0):
      return XRD_x, XRD_z, XRD_y, data_orth_coord

  XRD_x, XRD_z, XRD_y, data_orth_coord = XRD_analysis()

  #test plot for the coordinate system: (only works for the last position, the other peaks are not saved)
  def test_coordShift():
	      
      plt.figure()
      plt.imshow(np.log10(np.sum(diff_data[-1],axis=2)), cmap='jet', interpolation='none')#, extent=[ q1[0], q1[-1], q3[0], q3[-1] ])
      plt.title('natural')
      plt.xlabel('$q_1$ $ (\AA ^{-1}$)')   #l(' [$\mu m$]')#
      plt.ylabel('$q_3$ $ (\AA ^{-1}$)')
      plt.colorbar()
      plt.figure()
      plt.imshow(np.log10(np.sum(data_orth_coord.data[0],axis=2)), cmap='jet', interpolation='none')#, extent=[[ q1_orth[0], q1_orth[-1], q3_orth[0], q3_orth[-1] ] ])
      plt.title('cartesian')
      plt.xlabel('$q_x$ $ (\AA ^{-1}$)')   #q1~~qx
      plt.ylabel('$q_z$ $ (\AA ^{-1}$)')     #q3~qz
      plt.colorbar()        
  #test_coordShift()
  #plot3d_singleBraggpeak(np.log10(hhhh.data[0])) 
      

  def plot_XRD_xyz():
      # plot reciprocal space map x y z 
      plt.figure()
      plt.subplot(411)
      plt.imshow(XRD_x, cmap='jet',interpolation='none')#,extent=extent_motorpos)
      plt.title('Reciprocal space map, $q_x$ $ (\AA ^{-1}$) ')
      plt.ylabel('y [$\mu m$]')
      plt.colorbar()
      plt.subplot(412)
      plt.imshow(XRD_y, cmap='jet',interpolation='none')#,extent=extent_motorpos) 
      plt.title('Reciprocal space map, $q_y$ $ (\AA ^{-1}$) ')
      plt.ylabel('y [$\mu m$]')
      plt.colorbar()
      plt.subplot(413)
      plt.imshow(XRD_z, cmap='jet',interpolation='none')#,extent=extent_motorpos)
      plt.title('Reciprocal space map, $q_z$ $(\AA ^{-1}$) ')
      plt.ylabel('y [$\mu m$]')
      plt.colorbar()
      plt.subplot(414)
      plt.title('Bright field (sum of all rotations)')
      plt.imshow(sum(brightfield), cmap='jet', interpolation='none')#,extent=extent_motorpos)
      plt.xlabel('x [$\mu m$]') 
      plt.ylabel('y [$\mu m$]')
      plt.colorbar()
  plot_XRD_xyz()

  # calc abs q and the angles
  XRD_absq =  np.sqrt(XRD_x**2 + XRD_y**2 + XRD_z**2)
  XRD_alpha = XRD_y / XRD_z
  XRD_beta = -XRD_x / XRD_z

  def plot_XRD_polar():    
      import numpy as np
      import matplotlib.pyplot as plt
      # cut the images in x-range:start from the first pixel: 
      # remove the 1-pixel. (cheating) Did not remove it the extent, because then the 0 will not be shown in the X-scale and that will look weird
      start_cutXat = 0 
      # whant to cut to the right so that the scale ends with an even number
      #x-pixel nbr 67 is at 2.0194197798363955
      cutXat = nbr_cols+1 # 67
      # replace the x-scales end-postion in extent_motorposition. 
      #extent_motorpos_cut = np.copy(extent_motorpos)
      ###extent_motorpos_cut[1] = 2.0194197798363955 segmentedNW
      
      # plot abs q to select pixels that are 'background', not on wire, and set these pixels to NaN (make them white)
      
      plt.figure()
      #plt.suptitle(
      plt.subplot(411)
      plt.title('Summed up intensity (bright field)') #sum of all rotations
      plt.imshow(sum(brightfield[:,:,0:]), cmap='jet', interpolation='none')#,extent=extent_motorpos)
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
      
      plt.imshow(100*XRD_mask[:,start_cutXat:cutXat]*(a_lattice_exp[:,start_cutXat:cutXat]-mean_strain)/mean_strain, cmap='jet',interpolation='none')#,extent=extent_motorpos_cut) # not correct!'
      #plt.title('Relative length of Q-vector |Q|-$Q_{mean}$ $(10^{-3}/\AA$)')
      plt.title('Strain $\epsilon$ (%)')
      plt.ylabel('y [$\mu m$]');plt.colorbar()   

      plt.subplot(413)
      plt.imshow(XRD_mask[:,start_cutXat:cutXat]*1E3*XRD_alpha[:,start_cutXat:cutXat], cmap='jet',interpolation='none')#,extent=extent_motorpos_cut) # not correct!
      # cut in extent_motorposition. x-pixel nbr 67 is at 2.0194197798363955
      plt.title('Rotation around $q_x$ ($mrad$)')
      plt.ylabel('y [$\mu m$]')
      po = plt.colorbar()
      #po = plt.colorbar(ticks=(0,1,2,3,4))
      #po.set_label('Bending around $q_x$ $\degree$')
    
      plt.subplot(414)
      plt.imshow(XRD_mask[:,start_cutXat:cutXat]*1E3*XRD_beta[:,start_cutXat:cutXat], cmap='jet',interpolation='none')#,extent=extent_motorpos_cut) # not correct!
      plt.title('Rotation around $q_y$ ($mrad$) ')
      plt.ylabel('y [$\mu m$]')
      plt.xlabel('x [$\mu m$]') 
      po = plt.colorbar()
      plt.show()
      #po = plt.colorbar(ticks=(5, 10, 15 ))
      #po.set_label('Bending around $q_y$ $\degree$')
      
  plot_XRD_polar()
