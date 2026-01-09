#################################################################
# The aim of this script is to use Stokes Q and U cubes from the MGCLS (Knowles et al. 2022)   
# to do the polarization analysis through the RM-synthesis (using RMtool), correct for the Galactic RM,
# and obtain a final RM grid. It is thought to work specifically with MGCLS cubes, so many
# parameters are fine-tuned for that. Please consider this if you want to use this script, or part of it, for
# your own purposes.
# To be executed within the directory with target Q and U cubes
# For more, see https://github.com/emaderubeis/Tools-for-Rotation-Measure-Synthesis-Technique
# Author: De Rubeis Emanuele                                    
#################################################################

import numpy as np
from numpy import nan
import astropy.units as u
from astropy.io import fits
from astropy.utils import data
from astropy.table import QTable, Table
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord, Angle
from spectral_cube import SpectralCube
import regions
import statistics
import math
import os
import argparse
import smplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import matplotlib.cm as cm


def header_checker(input_fits):
	for item in input_fits:
		hdu = fits.open(item)

		data = hdu[0].data
		head = hdu[0].header

		data2 = np.zeros((np.shape(data)[1], np.shape(data)[2], np.shape(data)[3]))
		data2[:,:,:] = data[0,:,:,:]

		head['NAXIS'] = 3
		head['CTYPE3'] = 'FREQ'

		head.remove('NAXIS4')
		#head.remove('CRTYPE3')
		head.remove('CTYPE4')
		head.remove('CDELT4')
		head.remove('CRPIX4')
		head.remove('CROTA4')
		head.remove('CRVAL4')
		for i in range(1,13):
			head.remove('FREQ'+str("{:04d}".format(i))+'')
			head.remove('FREL'+str("{:04d}".format(i))+'')
			head.remove('FREH'+str("{:04d}".format(i))+'')
		head.remove('DO3D')
		hdu2 = fits.PrimaryHDU(data2, head)
		hdu2.writeto(item, overwrite = True)
		hdu.close()
		print("Header fixed for " + str(item) + "")


def qu_noise_calculator(nchan, region_name, input_fits, target):

	noise_rms_q = np.zeros((nchan))
	noise_rms_u = np.zeros((nchan))

	region_lis = regions.Regions.read(region_name)

	cube_sub_u = SpectralCube.read(input_fits[1])
	cube_sub_q = SpectralCube.read(input_fits[0])

	sub_cube_u = cube_sub_u.subcube_from_regions(region_lis)
	sub_cube_q = cube_sub_q.subcube_from_regions(region_lis)

	sub_cube_u_array = np.array(sub_cube_u._data)
	sub_cube_q_array = np.array(sub_cube_q._data)

	print(sub_cube_u_array)

	print(sub_cube_u_array.ndim)
	print(sub_cube_u_array.shape)

	print("Area in pixels: " + str(float(np.shape(sub_cube_u_array[0,:,:].flatten())[0])) + "")

	region_area = 1. / float(np.shape(sub_cube_u_array[0,:,:].flatten())[0])

	print("Noise list in Q in "+target+"_noiselistQ.dat")
	with open('./' +str(target)+ '/' +str(target)+ '_noiselistQ.dat','w') as rmsqlisttxt:
		for i in range(0,nchan):
			cube_rms_q_list = list(sub_cube_q_array[i,:,:].flatten())
			new_cube_rms_q_list = [item for item in cube_rms_q_list if not (math.isnan(item)) == True]
			rms_sq_q = sum(item*item for item in new_cube_rms_q_list)
			noise_rms_q[i] = np.sqrt(region_area * rms_sq_q)
			rmsqlisttxt.write(str(noise_rms_q)+'\n')

	print("Noise list in U in "+target+"_noiselistU.dat")
	with open('./' +str(target)+ '/' +str(target)+ '_noiselistU.dat','w') as rmsulisttxt:
		for i in range(0,nchan):
			cube_rms_u_list = list(sub_cube_u_array[i,:,:].flatten())
			new_cube_rms_u_list = [item for item in cube_rms_u_list if not (math.isnan(item)) == True]
			rms_sq_u = sum(item*item for item in new_cube_rms_u_list)
			noise_rms_u[i] = np.sqrt(region_area * rms_sq_u)
			rmsulisttxt.write(str(noise_rms_u)+'\n')


	list_tot = np.zeros((nchan))

	with open('./' +str(target)+ '/' +str(target)+ '_noiselistQUavg.dat','w') as lqu:
		for i in range(0,nchan):
			list_tot[i] = (float(noise_rms_q[i])+float(noise_rms_u[i]))/2.
			lqu.write(str(list_tot[i]) + '\n')


	media_QU = statistics.mean(list_tot)
	stdev_QU = statistics.stdev(list_tot)

	print('Average RMS between Q and U % s ' %(media_QU))
	print('Standard deviation between RMS of Q and U % s ' %(stdev_QU))




def pol_maps_maker(target, name_i, RMSF_FWHM):
	
	name_rm_cluster = '' + str(target) +'_RMobs_clean_masked.fits' #... name of RM image *NOT* corrected for the Milky Way contribution
	name_err_rm_cluster = '' + str(target) +'_RMobs_err_clean_masked.fits' # name of error RM image
	name_p = '' + str(target) +'_P_clean_6sig_masked.fits' #... name of polarization image
	name_pola = '' + str(target) +'_polangle_clean_6sig.fits' #... name of polarization angle image
	name_polf = '' + str(target) +'_polfrac_clean_6sig.fits' #... name of polarization fraction image

	#open input images

	hdu_tot = fits.open('' + str(target) + '_FDF_clean_tot.fits')
	tot = np.array(hdu_tot[0].data) # [phi,y,x]
	head = hdu_tot[0].header

	hdu_q = fits.open('' + str(target) + '_FDF_clean_real.fits')
	cube_q = np.array(hdu_q[0].data)

	hdu_u = fits.open('' + str(target) + '_FDF_clean_im.fits')
	cube_u = np.array(hdu_u[0].data)

	hdu_i = fits.open(name_i)
	img_i = np.array(hdu_i[0].data[0,0,:,:]) # [Stokes=1, Frequency=1, y, x]
	head_i = hdu_i[0].header

	#build the Faraday depth axis

	nphi = head['NAXIS3']
	dphi = head[ 'CDELT3']
	phi_axis = np.linspace(-int(nphi/2)*dphi,int(nphi/2)*dphi,nphi)

	#check how many pixels are in one image
	nx=head['NAXIS1'] 
	ny=head['NAXIS2'] 

	#check the observing wavelegth squared (remember shift theorem)
	lambda2_0=head['LAMSQ0']


	# Masking in Stokes I  !!!!! To check the rms of Stokes I image !!!
	rms_i = 1.5e-05   # Jy/beam
	img_i_masked = np.copy(img_i)
	img_i_masked[img_i_masked<3.*rms_i] = np.nan
	mask_i = img_i_masked/img_i_masked

	#initialize output images
	img_p = np.zeros([1,1,ny,nx])
	img_rm_cluster = np.zeros([1,1,ny,nx])
	img_err_rm_cluster = np.zeros([1,1,ny,nx])
	img_pola = np.zeros([1,1,ny,nx])

	# Noise arrays -- We evaluate the noise in the first and last 150 rad/m2 range, meaning [-351,-201] and [+201,+351] rad/m2
	# We may think to enlarge the phi axis to increase the statistics
	noise_q = np.zeros((50))
	noise_q_2 = np.zeros((50))
	noise_u = np.zeros((50))
	noise_u_2 = np.zeros((50))

	factor = 1./100.     # to be improved in a smarter way

	#average_noise_QU = 0.

	for yy in range (0,ny):
		for xx in range (0, nx):
			#compute the f, q, u and rm values at the peak position
			f = np.max(tot[:,yy,xx])
			q = cube_q[np.argmax(tot[:,yy,xx]),yy,xx]
			u = cube_u[np.argmax(tot[:,yy,xx]),yy,xx]
			rm = phi_axis[np.argmax(tot[:,yy,xx])]
			#for i in range (500,535):
			#	noise[i-500] = np.array(tot[i,yy,xx])
			#	noise_list = noise.tolist()
			noise_q[0:50] = np.array(cube_q[185:235,yy,xx])
			noise_q_2[0:50] = np.array(cube_q[0:50,yy,xx])
			q_list = noise_q.tolist()
			q_list_2 = noise_q_2.tolist()
			sum_sq_q = sum(i*i for i in q_list)
			sum_sq_q_2 = sum(i*i for i in q_list_2)
			noise_q_rms = np.sqrt((factor)*(sum_sq_q+sum_sq_q_2))
			#noise_q_mean = statistics.mean(q_list)
			noise_u[0:50] = np.array(cube_u[185:235,yy,xx])
			noise_u_2[0:50] = np.array(cube_u[0:50,yy,xx])
			u_list = noise_u.tolist()
			u_list_2 = noise_u_2.tolist()
			sum_sq_u = sum(i*i for i in u_list)
			sum_sq_u_2 = sum(i*i for i in u_list_2)
			noise_u_rms = np.sqrt((factor)*(sum_sq_u+sum_sq_u_2))
			#noise_u_mean = statistics.mean(u_list)
			noise_rms = 0.5*(noise_q_rms+noise_u_rms)
			#print(noise_rms)
			#noise_img[0,yy,xx] = noise_rms
			#average_noise_QU += noise_rms
			#select only pixels detected in polarization above a certain threshold
			if f>=6.*noise_rms:
				#correct for the ricean bias and write p
				img_p[0,0,yy,xx] = np.sqrt(f*f-2.3*noise_rms*noise_rms)
				#cluster's RM
				img_rm_cluster[0,0,yy,xx] = rm   #!! to be added the correction for the Galactic RM !!
				#error on RM
				img_err_rm_cluster[0,0,yy,xx] = (RMSF_FWHM/2)/(img_p[0,0,yy,xx]/noise_rms)
				#polarization angle (de-rotated wrt the observed one, to obtain the intrinsic one)
				img_pola[0,0,yy,xx] = ((0.5*np.arctan2(u,q))-rm*lambda2_0)*(180./np.pi)
			else:
				img_p[0,0,yy,xx]=np.nan
				img_rm_cluster[0,0,yy,xx]=np.nan
				img_err_rm_cluster[0,0,yy,xx]=np.nan
				img_pola[0,0,yy,xx]=np.nan

	#compute polarization fraction map
	img_polf=img_p/img_i_masked
	img_polf[img_polf<0] = np.nan
	img_polf[img_polf>1] = np.nan

	#Write the results in a fits file. We first modify the header to set the right units for each image
	hdu_p = fits.PrimaryHDU(img_p * mask_i,head_i)
	hdu_p.writeto(name_p, overwrite=True) 

	head_rm=head_i
	head_rm['BUNIT']='rad/m/m'
	hdu_rm = fits.PrimaryHDU(img_rm_cluster * mask_i,head_rm)
	hdu_rm.writeto(name_rm_cluster, overwrite=True) 

	head_err_rm=head_i
	head_err_rm['BUNIT']='rad/m/m'
	hdu_err_rm = fits.PrimaryHDU(img_err_rm_cluster * mask_i,head_err_rm)
	hdu_err_rm.writeto(name_err_rm_cluster, overwrite=True) 

	head_pola=head_i
	head_pola['BUNIT']='deg'
	hdu_pola = fits.PrimaryHDU(img_pola * mask_i,head_pola)
	hdu_pola.writeto(name_pola, overwrite=True) 

	head_polf=head_i
	head_polf['BUNIT']=''
	hdu_polf = fits.PrimaryHDU(img_polf * mask_i,head_polf)
	hdu_polf.writeto(name_polf, overwrite=True)




def extract_rm_for_sources(catalog_file, mask_file, rm_map_file):
    """
    Extract mean RM values for each source in a SoFIA catalog using mask regions.
    """
    
    # Load the SoFIA catalog
    print("Loading SoFIA catalog...")
    col_names = ['name', 'id', 'x', 'y', 'z', 'x_min', 'x_max', 'y_min', 'y_max', 'z_min', 'z_max', 
                 'n_pix', 'f_min', 'f_max', 'f_sum', 'rel', 'flag', 'fill', 'mean', 'std', 'skew', 
                 'kurt', 'rms', 'w20', 'w50', 'wm50', 'ell_maj', 'ell_min', 'ell_pa', 'ell3s_maj', 
                 'ell3s_min', 'ell3s_pa', 'kin_pa', 'err_x', 'err_y', 'err_z', 'err_f_sum', 'ra', 
                 'dec', 'freq', 'x_peak', 'y_peak', 'z_peak', 'ra_peak', 'dec_peak', 'freq_peak']
    catalog = Table.read(catalog_file, format='ascii', names=col_names)
    print(f"Found {len(catalog)} sources")
    
    # Load the mask
    print("Loading mask...")
    with fits.open(mask_file) as hdul:
        mask = hdul[0].data
        mask_header = hdul[0].header
    
    # Handle different dimensional structures
    if mask.ndim == 4:
        mask = mask[0, 0, :, :]  # Remove degenerate axes
    elif mask.ndim == 3:
        mask = mask[0, :, :]
    
    # Load the RM map
    print("Loading RM map...")
    with fits.open(rm_map_file) as hdul:
        rm_map = hdul[0].data
        rm_header = hdul[0].header
        rm_wcs = WCS(rm_header)
    
    # Handle different dimensional structures for RM map
    if rm_map.ndim == 4:
        rm_map = rm_map[0, 0, :, :]
        # Adjust WCS to 2D
        rm_wcs = rm_wcs.celestial
    elif rm_map.ndim == 3:
        rm_map = rm_map[0, :, :]
        rm_wcs = rm_wcs.celestial
    
    # Initialize arrays to store RM values
    rm_mean = np.full(len(catalog), np.nan)
    rm_std = np.full(len(catalog), np.nan)
    rm_median = np.full(len(catalog), np.nan)
    n_pixels = np.zeros(len(catalog), dtype=int)
    
    # Get unique source IDs from the mask
    unique_sources = np.unique(mask[mask > 0])
    print(f"Processing {len(unique_sources)} masked sources...")
    
    # Loop through each source
    for source_id in unique_sources:
        # Find pixels belonging to this source
        source_mask = (mask == source_id)
        
        # Extract RM values for this source
        rm_values = rm_map[source_mask]
        
        # Remove NaN values
        rm_values = rm_values[~np.isnan(rm_values)]
        
        if len(rm_values) > 0:
            # Find the corresponding row in catalog
            catalog_idx = np.where(catalog['id'] == source_id)[0]
            
            if len(catalog_idx) > 0:
                idx = catalog_idx[0]
                rm_mean[idx] = np.mean(rm_values)
                rm_std[idx] = np.std(rm_values)
                rm_median[idx] = np.median(rm_values)
                n_pixels[idx] = len(rm_values)
    
    # Add RM columns to catalog
    catalog['RM_mean'] = rm_mean
    catalog['RM_std'] = rm_std
    catalog['RM_median'] = rm_median
    catalog['RM_npix'] = n_pixels
    
    # Report statistics
    n_with_rm = np.sum(~np.isnan(rm_median))
    print(f"\nSuccessfully extracted RM values for {n_with_rm}/{len(catalog)} sources")
    print(f"RM range: {np.nanmin(rm_median):.2f} to {np.nanmax(rm_median):.2f} rad/m2")

    valid_sources = ~np.isnan(catalog['RM_median'])
    catalog = catalog[valid_sources]
    print(f"Retained {len(catalog)} sources with valid (not NaN) RM values")
    
    # Select only the desired columns for output
    output_columns = ['name', 'id', 'ra', 'dec', 'RM_mean', 'RM_std', 'RM_median', 'RM_npix']
    catalog_output = catalog[output_columns]


    return catalog_output


def gal_rm_correction(catalog, mask_file, rm_map_file, exclrad, outrad, nsources, output_file=None):
    """
    Apply Galactic RM correction using the annulus method to all pixels of each source.
    """
    
    # Load mask and RM map
    with fits.open(mask_file) as hdul:
        mask = hdul[0].data
        mask_header = hdul[0].header
    
    if mask.ndim == 4:
        mask = mask[0, 0, :, :]
    elif mask.ndim == 3:
        mask = mask[0, :, :]
    
    with fits.open(rm_map_file) as hdul:
        rm_map = hdul[0].data
        rm_header = hdul[0].header
    
    if rm_map.ndim == 4:
        rm_map = rm_map[0, 0, :, :]
    elif rm_map.ndim == 3:
        rm_map = rm_map[0, :, :]
    
    # Create SkyCoord objects for all sources
    ra = catalog['ra']
    dec = catalog['dec']
    
    # Check if columns already have units, if not add degrees
    if not hasattr(ra, 'unit') or ra.unit is None:
        ra = ra * u.deg
    if not hasattr(dec, 'unit') or dec.unit is None:
        dec = dec * u.deg
    
    coords = SkyCoord(ra=ra, dec=dec)
    
    # Initialize corrected RM map
    rm_map_corrected = rm_map.copy()
    
    # Initialize arrays to store galactic RM and corrected RM values
    rm_galactic = np.full(len(catalog), np.nan)
    rm_corrected = np.full(len(catalog), np.nan)
    
    print(f"Applying Galactic RM correction to {len(catalog)} sources...")
    
    # Loop through each source
    for i, source in enumerate(catalog):
        source_id = source['id']
        
        # Find all pixels belonging to this source
        source_pixels = (mask == source_id)
        
        # Skip if no pixels for this source
        if not np.any(source_pixels):
            continue
        
        # Calculate angular distances from this source to all others
        separations = coords[i].separation(coords).deg
        
        # Find sources in the annulus (exclrad to outrad degrees)
        annulus_mask = (separations > exclrad) & (separations < outrad)
        annulus_indices = np.where(annulus_mask)[0]
        
        if len(annulus_indices) == 0:
            continue
        
        # Get RM values of sources in annulus from catalog
        annulus_rm_values = catalog['RM_median'][annulus_indices]
        
        # Remove NaN values
        valid_rm = annulus_rm_values[~np.isnan(annulus_rm_values)]
        
        if len(valid_rm) == 0:
            continue
        
        # Get distances for sources with valid RM
        valid_indices = annulus_indices[~np.isnan(annulus_rm_values)]
        distances_in_annulus = separations[valid_indices]
        
        # Sort by distance and select the first nsources
        sorted_order = np.argsort(distances_in_annulus)
        selected_indices = sorted_order[:nsources]  # Take up to nsources
        selected_rm = valid_rm[selected_indices]
        
        if len(selected_rm) > 0:
            # Calculate median RM of reference sources
            median_gal_rm = np.median(selected_rm)
            
            # Store the galactic RM value
            rm_galactic[i] = median_gal_rm
            
            # Apply correction to all pixels of this source
            rm_map_corrected[source_pixels] = rm_map[source_pixels] - median_gal_rm
            
            # Calculate and store the median corrected RM for this source
            corrected_values = rm_map_corrected[source_pixels]
            corrected_values = corrected_values[~np.isnan(corrected_values)]
            if len(corrected_values) > 0:
                rm_corrected[i] = np.median(corrected_values)
        
        if (i + 1) % 100 == 0:
            print(f"Processed {i + 1}/{len(catalog)} sources")
    
    # Add the new columns to the catalog
    catalog['RM_galactic'] = rm_galactic
    catalog['RM_corrected'] = rm_corrected
    
    print(f"\nGalactic RM correction completed")
    
    # Save output if requested
    if output_file:
        hdu = fits.PrimaryHDU(rm_map_corrected, header=rm_header)
        hdu.writeto(output_file, overwrite=True)
        print(f"Saved corrected RM map to {output_file}")
        
        # Also save the updated catalog with the new columns
        catalog_output_file = output_file.replace('_galrm_corrected.fits', '_galrm_corrected_cat.txt')
        catalog.write(catalog_output_file, format='ascii', overwrite=True)
        print(f"Saved updated catalog to {catalog_output_file}")
    
    return catalog, rm_map_corrected



def plot_rmgrid(catalog, output_file):
    """
    Plot spatial distribution of sources with circle size proportional to RM_corrected
    and color scale based on RM_corrected values.
    
    Parameters:
    -----------
    catalog : astropy.table.Table
        Catalog with source positions (ra, dec) and RM_corrected values
    output_file : str
        Path to save the output plot (PNG)
    """
    
    # Extract the data
    ra = catalog['ra']
    dec = catalog['dec']
    rm_corrected = catalog['RM_corrected']
    
    #ra = Angle(ra, unit=u.deg).to_string(unit=u.hour, sep=':', precision=2)
    #dec = Angle(dec, unit=u.deg).to_string(unit=u.degree, sep=':', precision=2)
    
    '''
    # Remove any NaN values for plotting
    valid_idx = ~np.isnan(rm_corrected)
    ra = ra[valid_idx]
    dec = dec[valid_idx]
    rm_corrected = rm_corrected[valid_idx]
    '''
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Normalize RM values for color scale
    norm = Normalize(vmin=np.nanmin(rm_corrected), vmax=np.nanmax(rm_corrected))
    cmap = cm.get_cmap('coolwarm')
    
    # Scale RM values to circle sizes (area proportional to RM_corrected)
    # Normalize RM to reasonable circle sizes
    rm_min = np.nanmin(rm_corrected)
    rm_max = np.nanmax(rm_corrected)
    
    # Map RM values to circle areas (scale to 10-200 in terms of matplotlib size)
    if rm_max != rm_min:
        sizes = 10 + (rm_corrected - rm_min) / (rm_max - rm_min) * 190
    else:
        sizes = np.full_like(rm_corrected, 100.0)
    
    # Plot scatter with color and size
    scatter = ax.scatter(ra, dec, s=sizes, c=rm_corrected, cmap='coolwarm', 
                         norm=norm, alpha=0.6, edgecolors='black', linewidth=0.5)
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax, label='RM_corrected (rad/m2)')
    
    # Labels and title
    ax.set_xlabel('RA (deg)', fontsize=12)
    ax.set_ylabel('DEC (deg)', fontsize=12)
    ax.set_title('Galactic corrected RM grid', fontsize=14)
    ax.grid(True, alpha=0.3)
    
    # Save figure
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()




def radial_profile_annulus(fits_file, ra, dec, radius, num_rings):
    """
    Generate radial profile using annulus rings from a FITS image.
    
    Parameters:
    - fits_file: path to FITS image
    - ra, dec: center coordinates in degrees
    - radius: outer radius in degrees
    - num_rings: number of annulus rings
    """
    # Load FITS image
    hdul = fits.open(fits_file)
    image_data = hdul[0].data
    header = hdul[0].header
    hdul.close()
    
    # Get pixel scale (assumes square pixels)
    pixel_scale = abs(header['CDELT1'])  # degrees per pixel
    
    # Convert coordinates to pixel positions
    center_coord = SkyCoord(ra=ra*u.deg, dec=dec*u.deg)
    radius_pix = radius / pixel_scale
    
    # Assume simple pixel coordinate system (center of image)
    cx, cy = image_data.shape[1] / 2, image_data.shape[0] / 2
    
    # Create annulus rings and compute averages
    ring_radii = np.linspace(0, radius_pix, num_rings + 1)
    averages = []
    
    for i in range(num_rings):
        inner_r = ring_radii[i]
        outer_r = ring_radii[i + 1]
        
        # Create annulus mask
        y, x = np.ogrid[:image_data.shape[0], :image_data.shape[1]]
        dist = np.sqrt((x - cx)**2 + (y - cy)**2)
        mask = (dist >= inner_r) & (dist < outer_r)
        
        # Calculate average value in ring
        avg = np.nanmean(image_data[mask])

        # Calculate median
        #avg = np.nanmedian(image_data[mask])
        averages.append(avg)

    
    # Convert radii back to degrees for plotting
    ring_centers = (ring_radii[:-1] + ring_radii[1:]) / 2 * pixel_scale
    
    # Plot radial profile
    plt.figure(figsize=(10, 6))
    plt.plot(ring_centers, averages, 'o-')
    plt.xlabel('Radius (degrees)')
    plt.ylabel('Average Value')
    plt.title('Radial Profile')
    plt.grid(True)
    plt.savefig(fits_file.replace('.fits', '_avgrmprofile.png'), dpi=300, bbox_inches='tight')
    #plt.show()



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='RM-synthesis for MGCLS cubes')
    parser.add_argument('--inputs', help='Input Stokes Q and U cubes in this precise order', nargs=2, type=str)
    parser.add_argument('--noise', help='Region file used to evaluate the noise in the cube channels', type=str)
    parser.add_argument('--target', help='Name of the target source, should be the name of the folder containing the cubes. Used to name all the generated files.', type=str, default='targetcluster')
    parser.add_argument('--rmsynth', help='Do the RM-synthesis 3D using RMtool', default=False)
    parser.add_argument('--rmclean', help='Do the RM-clean 3D using RMtool', default=False)
    parser.add_argument('--exclrad', help='Exclusion radius in deg units for annulus method', default=0.4)
    parser.add_argument('--outrad', help='Outside radius in deg units for annulus method', default=0.8)
    parser.add_argument('--nsources', help='Number of sources to retain for annulus method', default=40)
	parser.add_argument('--ra', help='RA of the target in degrees', type=float)
	parser.add_argument('--dec', help='Dec of the target in degrees', type=float)
	parser.add_argument('--rfh', help='R500 of the cluster in degrees', type=float)
    args = parser.parse_args()

    ## To modify for your specific case

	# !!!!!!!!!!!
	#
	## !! TO BE DONE !!
	# .) load and correct also Stokes I image for masking and cataloguing
	
	name_i='Abell_4038_aFix_pol_I_15arcsec_5pln_cor.fits.gz'
	
	# .) implement support for Python SoFIA package
	#
	# !!!!!!!!!!!

    # Number of channels (once removed the flagged ones, so check the results of the calibration) and dimesion of the cube images
    # Here forced to be 12 because of the data channelization from the MGCLS
    nchan = 12

    # Definition of all the variables associated to the input arguments
    region_name = args.noise    # region file within which evaluate the RMS
    catalog_file = '' + str(args.target) + '_sofia_output_cat.txt'     # SoFIA ASCII catalog file
    mask_file = '' + str(args.target) + '_sofia_output_mask.fits'       # Mask FITS file from SoFIA
    rm_map_file = '' + str(target) +'_RMobs_clean_masked.fits'  # Observed RM map (possibly to be removed, given that it is created in a function in this script)
    output_file = '' + str(target) +'_galrm_corrected.fits'   # Output name of the catalog and galaxy-corrected RM map
    exclrad = args.exclrad      # Exclusion radius in degrees for the annulus method (see Anderson et al. 2024)
    outrad = args.outrad        # Outside radius in degrees for the annulus method (see Anderson et al. 2024)
    nsources = args.nsources    # Number of sources to retain for annulus method (see Anderson et al. 2024)

    # Check and correct the header of the MGCLS Q and U cubes
    header_checker(args.inputs)


    # Write here the file with the list of frequencies - for these cubes these are already known

    # !!! TO BE FIXED !!!!
    frequ = np.array([908.e06, 952.e06, 996.e06, 1044.e06, 1093.e06, 1145.e06, 1318.e06, 1382.e06, 1448.e06, 1482.e06, 1594.e06, 1656.e06])
    with open(''+args.target+'_freqlist.dat', 'w') as lfr:
		for item in frequ:
			lfr.write(str(item)+'\n')


    # Here we select a region for which a "sub-cube" is obtained and write the RMS for each image of the cube 
    # for both Q and U into a text file
    qu_noise_calculator(nchan, region_name, args.inputs, args.target)


    # Execution of rmsynth3D
    if (args.rmsynth == True):
        print("RM-synthesis 3D execution:")
        print("~/.local/bin/rmsynth3d " + str(args.inputs[0]) + " " + str(args.inputs[1]) + " " + args.traget +"_freq_list.dat -n ./" + str(args.target) + "_noiselistQUavg.dat -o " + str(args.target) + "_ -l 350 -d 3 -v")
        os.system("~/.local/bin/rmsynth3d " + str(args.inputs[0]) + " " + str(args.inputs[1]) + " " + args.traget +"_freq_list.dat -n ./" + str(args.target) + "_noiselistQUavg.dat -o " + str(args.target) + "_ -l 350 -d 3 -v")
        #os.system("mkdir -p rmsynthesis && mv *FDF*.fits rmsynthesis/ && mv *RMSF*.fits rmsynthesis/")
        #print("Products moved to ./rmsynthesis/")
    else:
        print("Command to execute the RM-synthesis 3D on your own")
        print("~/.local/bin/rmsynth3d " + str(args.inputs[0]) + " " + str(args.inputs[1]) + " " + args.traget +"_freq_list.dat -n ./" + str(args.target) + "_noiselistQUavg.dat -o " + str(args.target) + "_ -l 350 -d 3 -v")


    # Execution of the rmclean3D
    if (args.rmclean == True):
        print("~/.local/bin/rmclean3d " + str(args.target) + "_FDF_tot_dirty.fits " +str(args.target) + "_RMSF_tot.fits -c " + str(media_QU) + " -n 1000 -v -o " + str(args.target) + "_")
        os.system("~/.local/bin/rmclean3d " + str(args.target) + "_FDF_tot_dirty.fits " +str(args.target) + "_RMSF_tot.fits -c " + str(media_QU) + " -n 1000 -v -o " + str(args.target) + "_")
	else:
		print("Command to execute the rmclean3d on your own")
		print("~/.local/bin/rmclean3d " + str(args.target) + "_FDF_tot_dirty.fits " +str(args.target) + "_RMSF_tot.fits -c " + str(media_QU) + " -n 1000 -v -o " + str(args.target) + "_")

	# After doing RM-synthesis (and eventually rmclean), is now time to extract the RM information from the (cleaned)FDF

	#useful numbers as input to the script

	#sigma_p = 4.33e-6 # Jy/beam,read from the RMsynth_parameters.py script
	RMSF_FWHM = 45.44 # in rad/m2,read from the RMSF_FWHM.fits image or from the RMsynth_parameters.py script (theoretical value)

	pol_maps_maker(args.target, name_i, RMSF_FWHM)

	## Catalog using SoFIA !! TO BE UPDATED WITH LOCAL INSTALLATION OF SOFIA AND CORRECT PARAMFILE !!
	print("TO BE UPDATED WITH LOCAL INSTALLATION OF SOFIA AND CORRECT PARAMFILE")
	os.system("singularity exec docker://sofiapipeline/sofia2:latest sofia sofia_mgcls.par")

	## Galactic RM correction


	# Run the extraction
    catalog_with_rm = extract_rm_for_sources(
        catalog_file, 
        mask_file, 
        rm_map_file
    )

	# Apply Galactic RM correction to all source pixels
    catalog_with_rm, rm_corrected_map = gal_rm_correction(
        catalog_with_rm,
        mask_file,
        rm_map_file,
        exclrad,
        outrad,
        nsources,
        output_file
    )

	# Plot the corrected RM grid following Loi et al. (2025)
    plot_rmgrid(
        catalog_with_rm,
        output_file = output_file.replace('.fits', '_rmgrid.png')
    )

	radial_profile_annulus(
		output_file, 
		args.ra,
		args.dec, 
		args.rfh, 
		5
	)