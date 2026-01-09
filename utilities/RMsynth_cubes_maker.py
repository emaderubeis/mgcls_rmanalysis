#################################################################
# The aim of this script is to produce Stokes Q and U cubes for #
# RM synthesis starting from the images produced with WSClean.  #
#																#
# The code is expected to produce a cube for both Q and U Stokes# 
# that can be corrected for the primary beam, and a list of RMS #
# noise for each channel that constitute the Q and U cubes.     #
# 																#
# Author: De Rubeis Emanuele                                    #
#################################################################

import numpy as np
from numpy import nan
import astropy.units as u
from astropy.io import fits
from astropy.utils import data
from spectral_cube import SpectralCube
from astropy.table import QTable
import regions
import statistics
import math
import os


## To modify for your specific case

# Number of channels (once removed the flagged ones, so check the results of the calibration) and dimesion of the cube images

nchan = 56

'''
# Number of flagged channels
nchan_flag = 8
'''

# paths definition

path_home = '.'
path_q = './fits/'
path_u = './fits/'

# image names (these are the names that comes as WSClean output)
imagename_q = 'A2255_A_pol_target_rmsynth'
imagename_u = 'A2255_A_pol_target_rmsynth'

# check the size of the images

hdu = fits.open(path_q+imagename_q+'-0000-I-image.fits')
data = hdu[0].data[:,:,:,:]
nx = data.shape[2]
ny = data.shape[3]
hdu.close()

cube_q = np.zeros((nchan,ny,nx))
cube_u = np.zeros((nchan,ny,nx))

# Name of the file of the region within which evaluate the RMS

region_name = 'taper10_RMS_2.reg'


# name of outputs
name_cube_u = path_home+'A2255_Aconf_2asec_cubeU.fits'
#name_cube_u_sub= path_home+'PSZ096_U_64_cube_taper16_sub_TESTARTICOLO_PBCORRECTED.fits'
name_cube_q = path_home+'A2255_Aconf_2asec_cubeQ.fits'
#name_cube_q_sub= path_home+'PSZ096_Q_64_cube_taper16_sub_TESTARTICOLO_PBCORRECTED.fits'
name_freq_list = path_home+'freq_list.txt'
#name_rms_q_list = path_home+'lista_RMS_Q_taper16_TESTARTICOLO_PBCORRECTED.txt'
#name_rms_u_list = path_home+'lista_RMS_U_taper16_TESTARTICOLO_PBCORRECTED.txt'
#name_rms_qu_list = path_home+'lista_RMS_QU_64chans.txt'

# producing the Q and U cubes and writing a list of frequencies corresponding to the ones that characterize the images of the cube
with open(name_freq_list,'w') as lfr:
	for i in range(0,nchan):
		hdu_q = fits.open(path_q+imagename_q+str("-{:04d}".format(i))+'-Q-image.fits')
		hdu_u = fits.open(path_u+imagename_u+str("-{:04d}".format(i))+'-U-image.fits')
		data_q = hdu_q[0].data[:,:]
		data_u = hdu_u[0].data[:,:]
		header_q = hdu_q[0].header
		header_u = hdu_u[0].header
		frequ = hdu_q[0].header['CRVAL3']
		lfr.write(str(frequ)+'\n')
		cube_q[i,:,:] = data_q
		cube_u[i,:,:] = data_u


# writing the cube
hdu_cube_q = fits.PrimaryHDU(cube_q,header_q)
hdu_cube_q.writeto(name_cube_q, overwrite=True)
print("Q cube written!")

hdu_cube_u = fits.PrimaryHDU(cube_u,header_u)
hdu_cube_u.writeto(name_cube_u, overwrite=True)
print("U cube written!")

# Primary Beam correction: divide our Q/U cube for the cube produced with widebandpbcor in CASA
# if we need to correct for the primary beam please set 'pbcorrection=1'

pbcorrection = 0

if pbcorrection == 1:
	
	print("PB correction selected")
	cube_pbcor_q = fits.open(name_cube_q)
	img_cube_q = np.array(cube_pbcor_q[0].data)

	cube_pbcor_u = fits.open(name_cube_u)
	img_cube_u = np.array(cube_pbcor_u[0].data)

	cube_pbcor = fits.open(path_home+'PSZ096_totale_pbcor_taper10.pbcor.workdirectory/PSZ096_totale_pbcor_taper10.pb.cube.fits')
	img_cube_pbcor = np.array(cube_pbcor[0].data)


	cube_q_corrected = img_cube_q/img_cube_pbcor
	cube_u_corrected = img_cube_u/img_cube_pbcor

	cube_q_corrected_2 = np.zeros((nchan,ny,nx))
	cube_u_corrected_2 = np.zeros((nchan,ny,nx))

	cube_q_corrected_2[:,:,:] = cube_q_corrected[0,:,:,:]
	cube_u_corrected_2[:,:,:] = cube_u_corrected[0,:,:,:]

	# writing the cubes corrected for the primary beam
	hdu_cube_q_corrected = fits.PrimaryHDU(cube_q_corrected_2,header_q)
	hdu_cube_q_corrected.writeto(name_cube_q, overwrite=True)

	hdu_cube_u_corrected = fits.PrimaryHDU(cube_u_corrected_2,header_u)
	hdu_cube_u_corrected.writeto(name_cube_u, overwrite=True)
else :
	print("No PB correction selected")



'''
# here we remove the empty images, so those corresponding to the flagged channels
# you have to change this for your own cube in case you have flagged channels

cubo_q = np.zeros((nchan-nchan_flag,ny,nx))
cubo_u = np.zeros((nchan-nchan_flag,ny,nx))


for i in range(0,nchan):
	hdu_2_q = fits.open(name_cube_q)
	hdu_2_u = fits.open(name_cube_u)
	dati_q = hdu_2_q[0].data
	dati_u = hdu_2_u[0].data
	cubo_2_q = dati_q
	cubo_2_u = dati_u

	if i<32:
		cubo_q[i,:,:] = cubo_2_q[i,:,:]
		cubo_u[i,:,:] = cubo_2_u[i,:,:]

	elif i>39:
		cubo_q[i-8,:,:] = cubo_2_q[i,:,:]
		cubo_u[i-8,:,:] = cubo_2_u[i,:,:]
	else:
		print('no')



# writing the "empty-channels-subtracted" Q and U cubes
hdu_cube_q_corrected = fits.PrimaryHDU(cubo_q,header_q)
hdu_cube_q_corrected.writeto(name_cube_q, overwrite=True)
print("Empty channels removed from the Q cube!")

hdu_cube_u_corrected = fits.PrimaryHDU(cubo_u,header_u)
hdu_cube_u_corrected.writeto(name_cube_u, overwrite=True)
print("Empty channels removed from the U cube!")
'''


'''
# Here we select a region for which a "sub-cube" is obtained and write the RMS for each image of the cube 
# for both Q and U into a text file

region_lis = regions.Regions.read(path_home+region_name)

cube_sub_u = SpectralCube.read(name_cube_u)
cube_sub_q = SpectralCube.read(name_cube_q)

sub_cube_u = cube_sub_u.subcube_from_regions(region_lis)
sub_cube_q = cube_sub_q.subcube_from_regions(region_lis)

hdu_sub_cube_u = fits.PrimaryHDU(sub_cube_u,header_u)
hdu_sub_cube_u.writeto(name_cube_u_sub,overwrite=True)
hdu_sub_cube_q = fits.PrimaryHDU(sub_cube_q,header_q)
hdu_sub_cube_q.writeto(name_cube_q_sub,overwrite=True)



hdu_q_rms = fits.open(name_cube_q_sub)
hdu_u_rms = fits.open(name_cube_u_sub)
datiepz_q = hdu_q_rms[0].data[:,:]
datiepz_u = hdu_u_rms[0].data[:,:]
cube_rms_q = datiepz_q
cube_rms_u = datiepz_u


cubo_rms_q = np.zeros((56,40,40))
cubo_rms_u = np.zeros((56,40,40))

region_area = 1./1286.	# this is the inverse of the region area



# Writing the lists in Q and U of the rms for each image of the cube. This could be used, for example, for evaluating an average RMS for the whole cube in both Q and U
print("Noise list in Q in "+str(name_rms_q_list)+"")
with open(name_rms_q_list,'w') as rmsqlisttxt:
	for i in range (0,nchan):
		cubo_rms_q[i,:,:] = np.array(cube_rms_q[i,:,:])
		cubo_rms_q_list = list(cube_rms_q[i,:,:].flatten())
		new_cubo_rms_q_list = [item for item in cubo_rms_q_list if not (math.isnan(item)) == True]
		rms_sq_q = sum(item*item for item in new_cubo_rms_q_list)
		noise_rms_q = np.sqrt(region_area*(rms_sq_q))
		rmsqlisttxt.write(str(noise_rms_q)+'\n')

print("Noise list in U in "+str(name_rms_u_list)+"")
with open(name_rms_u_list,'w') as rmsulisttxt:
	for i in range (0,nchan):
		cubo_rms_u[i,:,:] = np.array(cube_rms_u[i,:,:])
		cubo_rms_u_list = list(cube_rms_u[i,:,:].flatten())
		new_cubo_rms_u_list = [item for item in cubo_rms_u_list if not (math.isnan(item)) == True]
		rms_sq_u = sum(item*item for item in new_cubo_rms_u_list)
		noise_rms_u = np.sqrt(region_area*(rms_sq_u))
		rmsulisttxt.write(str(noise_rms_u)+'\n')



list_tot = np.zeros((56))

rms_u = open(path_home+name_rms_u_list)
rms_q = open(path_home+name_rms_q_list)

list_q = []
list_u = []


for line in rms_u:
	stripped_line = line.strip()
	list_u.append(stripped_line)

for line in rms_q:
        stripped_line = line.strip()
        list_q.append(stripped_line)


lista_q = np.array(list_q)

lista_u = np.array(list_u)

for i in range(0,56):
	list_tot[i] = (float(lista_u[i])+float(lista_q[i]))/2.


lista_QU = list_tot.tolist()

count = 0

with open(path_home+name_rms_qu_list,'w') as lqu:
	for item in lista_QU:
		lqu.write(str(item)+ '\n')
		count = count+1

media_QU = statistics.mean(lista_QU)
stdev_QU = statistics.stdev(lista_QU)

print('Average RMS between Q and U % s ' %(media_QU))
print('Standard deviation between RMS of Q and U % s ' %(stdev_QU))
'''