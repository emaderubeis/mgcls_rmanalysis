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
import astropy.units as u
from astropy.io import fits
from astropy.utils import data
from astropy.table import QTable, Table
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord, Angle
from spectral_cube import SpectralCube
import regions
from regions import CircleAnnulusPixelRegion, CirclePixelRegion, PixCoord
import statistics
import math
import os, os.path
import argparse
import smplotlib
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.colors import Normalize
from cmcrameri import cm
from scipy.spatial import Voronoi
from matplotlib.patches import Polygon


def findrms(mIn, maskSup=1e-7):
    """
    find the rms of an array, from Cyril Tasse/kMS
    """
    m = mIn[np.abs(mIn) > maskSup]
    rmsold = np.std(m)
    diff = 1e-1
    cut = 3.
    med = np.median(m)
    for i in range(10):
        ind = np.where(np.abs(m - med) < rmsold * cut)[0]
        rms = np.std(m[ind])
        if np.abs((rms - rmsold) / rmsold) < diff: break
        rmsold = rms
    return rms


def header_checker(input_fits):
    for item in input_fits:
        hdu = fits.open(item)

        data = hdu[0].data
        head = hdu[0].header
        if (data.ndim == 3):
            print("" + str(item) + " header is already correct, move forward")
            continue

        if os.path.exists('' + str(item) + '_cor_header.fits'):
            print("" + str(item) + " header is already correct, move forward")
            continue

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
        hdu2.writeto('' + str(item) + '_cor_header.fits', overwrite = True)
        hdu.close()
        print("Header fixed for " + str(item) + "")


def header_checker_stokesi(input):
    hdu = fits.open(input)
    data = hdu[0].data
    head = hdu[0].header
    data2 = np.zeros((np.shape(data)[2], np.shape(data)[3]))
    data2[:,:] = data[0,0,:,:]
    
    head['NAXIS'] = 3
    head['CTYPE3'] = 'FREQ'
    head.remove('NAXIS4')
    head.remove('CTYPE4')
    head.remove('CDELT4')
    head.remove('CRPIX4')
    head.remove('CROTA4')
    head.remove('CRVAL4')
    head.remove('DO3D')
    hdu2 = fits.PrimaryHDU(data2, head)
    hdu2.writeto('' + str(input) + '_cor_header.fits', overwrite = True)
    hdu.close()
    print("Header fixed for " + str(input) + "")


def plot_stokesi(target, input, ra, dec, radius):

    print("Plotting Stokes I map")
    hdul = fits.open(input)
    data = hdul[0].data
    header = hdul[0].header
    header.remove('CRPIX3')
    header.remove('CTYPE3')
    header.remove('CRVAL3')
    header.remove('CDELT3')
    hdul.close()

    if data.ndim == 4:
        data = data[0, 0, :, :]
    elif data.ndim == 3:
        data = data[0, :, :]

    wcs = WCS(header).dropaxis(2)
    
    fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(projection=wcs))
    im = ax.imshow(data, cmap='cmc.batlowK', vmax=0.02*np.max(data), origin='lower')
    cbar = plt.colorbar(im, ax=ax, label=r'Surface brightness [$\rm{Jy~beam^{-1}}$]')

    circle_center = SkyCoord(ra, dec, unit=u.deg)
    circle = plt.Circle((ra, dec), radius, transform=ax.get_transform('world'), 
                           fill=False, edgecolor='red', linewidth=2, label=f'R500')
    ax.add_patch(circle)
    
    ax.set_xlabel('RA (J2000)', fontsize='large')
    ax.set_ylabel('Dec (J2000)', fontsize='large')
    ax.tick_params(labelsize='large')
    
    plt.tight_layout()
    plt.savefig('' + str(target) + '_stokesI_15asec.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_polint(target, input, ra, dec, radius):
    
    print("Plotting unmasked polarized intensity map")
    
    hdul = fits.open(input)
    data = hdul[0].data
    header = hdul[0].header
    header.remove('CRPIX3')
    header.remove('CTYPE3')
    header.remove('CRVAL3')
    header.remove('CDELT3')
    header.remove('CROTA3')
    header.remove('CRPIX4')
    header.remove('CTYPE4')
    header.remove('CRVAL4')
    header.remove('CDELT4')
    header.remove('CROTA4')
    #header.remove('CUNIT4')
    header.remove('NAXIS3')
    header.remove('NAXIS4')
    header['NAXIS'] = 2
    header.remove('DO3D')
    #header['WCSAXES'] = 2
    hdul.close()

    if data.ndim == 4:
        data = data[0, 0, :, :]
    elif data.ndim == 3:
        data = data[0, :, :]

    wcs = WCS(header)
    
    fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(projection=wcs))
    im = ax.imshow(data, cmap='cmc.batlowK', vmax=0.02*np.max(data), origin='lower')
    cbar = plt.colorbar(im, ax=ax, label=r'Polarized surface brightness [$\rm{Jy~beam^{-1}}$]')

    circle_center = SkyCoord(ra, dec, unit=u.deg)
    circle = plt.Circle((ra, dec), radius, transform=ax.get_transform('world'), 
                           fill=False, edgecolor='red', linewidth=2, label=f'R500')
    ax.add_patch(circle)
    
    ax.set_xlabel('RA (J2000)', fontsize='large')
    ax.set_ylabel('Dec (J2000)', fontsize='large')
    ax.tick_params(labelsize='large')
    
    plt.tight_layout()
    plt.savefig(str(target) + '_unmasked_polint_15asec.png', dpi=300, bbox_inches='tight')
    plt.close()



def plot_polfrac(target, input, ra, dec, radius):
    
    print("Plotting fractional polarization map")
    
    hdul = fits.open(input)
    data = hdul[0].data
    header = hdul[0].header
    header.remove('CRPIX3')
    header.remove('CTYPE3')
    header.remove('CRVAL3')
    header.remove('CDELT3')
    header.remove('CROTA3')
    header.remove('CRPIX4')
    header.remove('CTYPE4')
    header.remove('CRVAL4')
    header.remove('CDELT4')
    header.remove('CROTA4')
    #header.remove('CUNIT4')
    header.remove('NAXIS3')
    header.remove('NAXIS4')
    header['NAXIS'] = 2
    header.remove('DO3D')
    #header['WCSAXES'] = 2
    hdul.close()

    if data.ndim == 4:
        data = data[0, 0, :, :]
    elif data.ndim == 3:
        data = data[0, :, :]

    wcs = WCS(header)
    
    fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(projection=wcs))
    im = ax.imshow(data, cmap='rainbow', origin='lower')
    cbar = plt.colorbar(im, ax=ax, label=r'Fractional polarization')

    circle_center = SkyCoord(ra, dec, unit=u.deg)
    circle = plt.Circle((ra, dec), radius, transform=ax.get_transform('world'), 
                           fill=False, edgecolor='black', linewidth=2, label=f'R500')
    ax.add_patch(circle)
    
    ax.set_xlabel('RA (J2000)', fontsize='large')
    ax.set_ylabel('Dec (J2000)', fontsize='large')
    ax.tick_params(labelsize='large')
    
    plt.tight_layout()
    plt.savefig(str(target) + '_masked_polfrac_15asec.png', dpi=300, bbox_inches='tight')
    plt.close()


def qu_noise_calculator(nchan, region_name, input_fits, target):

    noise_rms_q = np.zeros((nchan))
    noise_rms_u = np.zeros((nchan))

    region_lis = regions.Regions.read(region_name)

    cube_sub_u = SpectralCube.read('' + str(input_fits[1]) + '_cor_header.fits')
    cube_sub_q = SpectralCube.read('' + str(input_fits[0]) + '_cor_header.fits')

    sub_cube_u = cube_sub_u.subcube_from_regions(region_lis)
    sub_cube_q = cube_sub_q.subcube_from_regions(region_lis)

    sub_cube_u_array = np.array(sub_cube_u._data)
    sub_cube_q_array = np.array(sub_cube_q._data)

    #print(sub_cube_u_array)
    #print(sub_cube_u_array.ndim)
    #print(sub_cube_u_array.shape)

    print("Area in pixels: " + str(float(np.shape(sub_cube_u_array[0,:,:].flatten())[0])) + "")

    region_area = 1. / float(np.shape(sub_cube_u_array[0,:,:].flatten())[0])

    print("Noise list in Q in "+target+"_noiselistQ.dat")
    with open('./' +str(target)+ '_noiselistQ.dat','w') as rmsqlisttxt:
        for i in range(0,nchan):
            cube_rms_q_list = list(sub_cube_q_array[i,:,:].flatten())
            new_cube_rms_q_list = [item for item in cube_rms_q_list if not (math.isnan(item)) == True]
            rms_sq_q = sum(item*item for item in new_cube_rms_q_list)
            noise_rms_q[i] = np.sqrt(region_area * rms_sq_q)
            rmsqlisttxt.write(str(noise_rms_q)+'\n')

    print("Noise list in U in "+target+"_noiselistU.dat")
    with open('./' +str(target)+ '_noiselistU.dat','w') as rmsulisttxt:
        for i in range(0,nchan):
            cube_rms_u_list = list(sub_cube_u_array[i,:,:].flatten())
            new_cube_rms_u_list = [item for item in cube_rms_u_list if not (math.isnan(item)) == True]
            rms_sq_u = sum(item*item for item in new_cube_rms_u_list)
            noise_rms_u[i] = np.sqrt(region_area * rms_sq_u)
            rmsulisttxt.write(str(noise_rms_u)+'\n')


    list_tot = np.zeros((nchan))

    with open('./' +str(target)+ '_noiselistQUavg.dat','w') as lqu:
        for i in range(0,nchan):
            list_tot[i] = (float(noise_rms_q[i])+float(noise_rms_u[i]))/2.
            lqu.write(str(list_tot[i]) + '\n')

    media_QU = statistics.mean(list_tot)
    stdev_QU = statistics.stdev(list_tot)

    print('Average RMS between Q and U % s ' %(media_QU))
    print('Standard deviation between RMS of Q and U % s ' %(stdev_QU))

    return media_QU




def pol_maps_maker(target, name_i, RMSF_FWHM):
    
    print("Making polarization maps - Ricean bias correction")
    name_rm_cluster = '' + str(target) +'_RMobs_clean_masked.fits' #... name of RM image *NOT* corrected for the Milky Way contribution
    name_err_rm_cluster = '' + str(target) +'_RMobs_err_clean_masked.fits' # name of error RM image
    name_p = '' + str(target) +'_P_clean_6sig_masked.fits' #... name of polarization image
    name_pp = '' + str(target) +'_P_clean_6sig_unmasked.fits'
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


    # Masking in Stokes I
    rms_i = findrms(img_i)
    print("RMS noise in Stokes I map is: " + str(rms_i) + " Jy/beam")
    img_i_masked = np.copy(img_i)
    img_i_masked[img_i_masked<5.*rms_i] = np.nan
    mask_i = img_i_masked/img_i_masked

    #initialize output images
    img_p = np.zeros([1,1,ny,nx])
    img_pp = np.zeros([1,1,ny,nx])
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
            img_pp[0,0,yy,xx] = f
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
                img_rm_cluster[0,0,yy,xx] = rm
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

    mask_p = img_p / img_p

    mask_tot = mask_i * mask_p
    
    head_mask = head_i
    head_mask['BITPIX'] = 16
    hdu_mask = fits.PrimaryHDU(mask_tot, head_mask)
    hdu_mask.writeto('' + str(target) +'_maskip.fits', overwrite=True)

    #Write the results in a fits file. We first modify the header to set the right units for each image
    hdu_p = fits.PrimaryHDU(img_p * mask_i,head_i)
    hdu_p.writeto(name_p, overwrite=True) 

    hdu_pp = fits.PrimaryHDU(img_pp,head_i)
    hdu_pp.writeto(name_pp, overwrite=True)

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

    return rms_i, mask_tot[0,0,:,:]




def extract_rm_for_sources(catalog_file, mask_file, rm_map_file):
    """
    Extract mean RM values for each source in a SoFIA catalog using mask regions.
    """
    
    # Load the SoFIA catalog
    print("Loading SoFIA catalog...")
    col_names = ['name', 'id', 'x', 'y', 'z', 'x_min', 'x_max', 'y_min', 'y_max', 'z_min', 'z_max', 
                 'n_pix', 'f_min', 'f_max', 'f_sum', 'rel', 'flag', 'rms', 'w20', 'w50', 'wm50', 'ell_maj', 'ell_min', 'ell_pa', 'ell3s_maj', 
                 'ell3s_min', 'ell3s_pa', 'kin_pa', 'err_x', 'err_y', 'err_z', 'err_f_sum', 'snr', 'snr_max', 'ra', 
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
            # !!! Check RM VALUES, SEE THE GALRM_CORRECTION.PY FILE !!!
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



def plot_rmgrid(catalog, stokesi, rms_i, mask_tot, output_file):
    ''' 
    # Extract the data
    ra = catalog['ra']
    dec = catalog['dec']
    rm_corrected = catalog['RM_corrected']
    
    #ra = Angle(ra, unit=u.deg).to_string(unit=u.hour, sep=':', precision=2)
    #dec = Angle(dec, unit=u.deg).to_string(unit=u.degree, sep=':', precision=2)
    
    
    # Remove any NaN values for plotting
    valid_idx = ~np.isnan(rm_corrected)
    ra = ra[valid_idx]
    dec = dec[valid_idx]
    rm_corrected = rm_corrected[valid_idx]
    
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Normalize RM values for color scale
    norm = Normalize(vmin=np.nanmin(rm_corrected), vmax=np.nanmax(rm_corrected))
    cmap = matplotlib.cm.get_cmap('coolwarm')
    
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
    cbar = plt.colorbar(scatter, ax=ax, label=r'RRM ($\rm{rad~m^{-2}}$)')
    
    # Labels and title
    ax.set_xlabel('RA (deg)', fontsize=12)
    ax.invert_xaxis()
    ax.set_ylabel('DEC (deg)', fontsize=12)
    ax.set_title('RRM grid', fontsize=14)
    ax.grid(True, alpha=0.3)
    
    # Save figure
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    '''

    '''

    # !!! To update with FITS file writing and equal intervals in RM (+/- same RM, to have zero in the "greyish" area) for the RMgrid !!!
    
    points = np.column_stack([ra,dec])
    vor = Voronoi(points)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 12))
    
    # Normalize RM values for color scale
    norm = Normalize(vmin=-np.nanmax(rm_corrected), vmax=np.nanmax(rm_corrected))
    cmap = matplotlib.cm.get_cmap('coolwarm')
    
    for point_idx in range(len(points)):
        region = vor.regions[vor.point_region[point_idx]]
        
        # Skip empty regions or regions with infinite vertices
        if len(region) == 0 or -1 in region:
            continue
        
        # Get vertices of the region
        vertices = vor.vertices[region]
        
        # Create polygon colored by RM value
        polygon = Polygon(vertices, closed=True, alpha=0.7, 
                         facecolor=cmap(norm(rm_corrected[point_idx])), 
                         edgecolor='black', linewidth=0.5)
        ax.add_patch(polygon)
    
    # Plot source points
    scatter = ax.scatter(ra, dec, c=rm_corrected, cmap='coolwarm', 
                        norm=norm, s=50, edgecolors='black', linewidth=1, zorder=5)
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax, label=r'RRM ($\rm{rad~m^{-2}}$)')
    
    # Set axis limits
    ax.set_xlim(np.min(ra), np.max(ra))
    ax.set_ylim(np.min(dec), np.max(dec))
    
    # Labels and title
    ax.set_xlabel('RA (deg)', fontsize=12)
    ax.invert_xaxis()
    ax.set_ylabel('DEC (deg)', fontsize=12)
    ax.set_title('RRM grid', fontsize=14)
    ax.grid(True, alpha=0.3)
    
    # Save figure
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    '''
  
    ra = catalog['ra']
    dec = catalog['dec']
    rm_corrected = catalog['RM_corrected']

    # 2. Setup WCS and Data
    with fits.open(stokesi) as hdui:
        wcs = WCS(hdui[0].header).celestial 
        contour_data = np.squeeze(hdui[0].data)

    # Convert RA/Dec to Pixel coordinates for accurate Voronoi
    coords_pix = wcs.all_world2pix(np.column_stack([ra, dec]), 0)
    vor = Voronoi(coords_pix)
    
    fig, ax = plt.subplots(figsize=(12, 10), subplot_kw=dict(projection=wcs))
    vmax = np.nanmax(np.abs(rm_corrected))
    norm = Normalize(vmin=-vmax, vmax=vmax)
    cmap = matplotlib.cm.get_cmap('coolwarm')
    
    for point_idx in range(len(coords_pix)):
        region_idx = vor.point_region[point_idx]
        region = vor.regions[region_idx]
        if not region or -1 in region:
            continue
        vertices = vor.vertices[region]
        # Since we are already in pixel space and the ax is projected via WCS, 
        # we don't need a transform here IF we plot in pixel units.
        polygon = Polygon(vertices, closed=True, alpha=0.5, 
                         facecolor=cmap(norm(rm_corrected[point_idx])), 
                         edgecolor='grey', linewidth=0.5)
        ax.add_patch(polygon)
    
    contour_levels = [5.* rms_i, 25.* rms_i, 125. * rms_i, 625.* rms_i]
    ax.contour(contour_data, levels=contour_levels, colors='black', 
               linewidths=1.0, alpha=0.8)
    
    scatter = ax.scatter(ra, dec, c=rm_corrected, cmap='coolwarm', norm=norm, 
                        s=20, edgecolors='grey', linewidth=0.5, 
                        transform=ax.get_transform('world'), zorder=10)
    
    ax.coords[0].set_axislabel('Right Ascension')
    ax.coords[1].set_axislabel('Declination')
    
    # Set limits based on data spread (in pixels)
    pad = 20
    ax.set_xlim(np.min(coords_pix[:,0]) - pad, np.max(coords_pix[:,0]) + pad)
    ax.set_ylim(np.min(coords_pix[:,1]) - pad, np.max(coords_pix[:,1]) + pad)
    
    plt.colorbar(scatter, label=r'RRM (rad m$^{-2}$)', fraction=0.046, pad=0.04)
    plt.title('RRM Grid')
    
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

    if image_data.ndim == 3:
        image_data = image_data[0,:,:]
    
    # Get pixel scale (assumes square pixels)
    pixel_scale = abs(header['CDELT1'])  # degrees per pixel
    print("pixel_scale")
    print(pixel_scale)

    radius_pix = radius / pixel_scale
    
    # Convert coordinates to pixel positions
    center_coord = SkyCoord(ra=ra*u.deg, dec=dec*u.deg)
    wcs = WCS(header)
    wcs_2d = wcs.celestial
    cx, cy = wcs_2d.world_to_pixel(center_coord)
    print("cx, cy")
    print(cx, cy)
    
    # Create annulus rings and compute averages
    ring_radii = np.linspace(0., radius_pix, num_rings + 1)
    print("ring_radii")
    print(ring_radii * pixel_scale * 60.)
    averages = []
    stddeviations = []
    
    for i in range(num_rings):
        inner_r = ring_radii[i]
        outer_r = ring_radii[i + 1]

        # Create annulus region

        if i == 0:
            annulus = CirclePixelRegion(
                center=PixCoord(cx, cy),
                radius=outer_r
                )
        else:        
            annulus = CircleAnnulusPixelRegion(
                center=PixCoord(cx,cy),
                inner_radius=inner_r,
                outer_radius=outer_r
                )

        mask = annulus.to_mask().to_image(image_data.shape)
        
        # Calculate average value in ring
        avg = np.nanmean(abs(image_data[mask.astype(bool)]))
        std = np.nanstd(abs(image_data[mask.astype(bool)]))

        # Calculate median
        #avg = np.nanmedian(abs(image_data[mask.astype(bool)]))

        averages.append(avg)
        stddeviations.append(std)

    
    # Convert radii back to degrees for plotting
    ring_centers = (ring_radii[:-1] + ring_radii[1:]) / 2 * pixel_scale
    
    # Plot radial profile
    plt.figure(figsize=(10, 6))
    plt.errorbar(ring_centers, averages, yerr=stddeviations, fmt='o-')
    plt.xlabel('Radius (degrees)')
    plt.ylabel(r'<|RRM|> ($\rm{rad~m^{-2}}$)')
    plt.title('Radial <|RRM|> Profile')
    plt.grid(True)
    plt.savefig(fits_file.replace('.fits', '_avgrmprofile.png'), dpi=300, bbox_inches='tight')
    #plt.show()



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='RM-synthesis for MGCLS cubes')
    parser.add_argument('--inputs', help='Input Stokes Q and U cubes (15") in this precise order', nargs=2, type=str)
    parser.add_argument('--stokesi', help='Stokes I (15") PB corrected map (advanced products -> 5pln)', type=str)
    parser.add_argument('--noise', help='Region file used to evaluate the noise in the cube channels', type=str)
    parser.add_argument('--target', help='Name of the target source, should be the name of the folder containing the cubes. Used to name all the generated files.', type=str, default='targetcluster')
    parser.add_argument('--rmsynth', help='Do the RM-synthesis 3D using RMtool', action='store_true')
    parser.add_argument('--rmclean', help='Do the RM-clean 3D using RMtool', action='store_true')
    parser.add_argument('--exclrad', help='Exclusion radius in deg units for annulus method', default=0.4)
    parser.add_argument('--outrad', help='Outside radius in deg units for annulus method', default=0.8)
    parser.add_argument('--nsources', help='Number of sources to retain for annulus method', default=40)
    parser.add_argument('--ra', help='RA of the target in degrees', type=float)
    parser.add_argument('--dec', help='Dec of the target in degrees', type=float)
    parser.add_argument('--rfh', help='R500 of the cluster in degrees', type=float)
    parser.add_argument('--sofia', help='SoFIA executable (complete path) for cataloging the sources. If not provided, the offical SoFIA Docker image will be used', type=str, default='/hs/fs08/data/group-brueggen/e.derubeis/softwares/SoFiA-2-v2.6.0/sofia')
    parser.add_argument('--param', help='SoFIA paramfile', type=str, default='/hs/fs08/data/group-brueggen/e.derubeis/meerkat/mgcls_rmanalysis/utilities/sofia_mgcls.par')
    args = parser.parse_args()

    ## To modify for your specific case

    print("############################")
    print("")
    print("MGCLS RM analysis")
    print("Check https://github.com/emaderubeis/mgcls_rmanalysis/tree/main for more")
    print("")
    print("############################")
   
    name_i=args.stokesi

    # Number of channels (once removed the flagged ones, so check the results of the calibration) and dimesion of the cube images
    # Here forced to be 12 because of the data channelization from the MGCLS
    nchan = 12

    # Definition of all the variables associated to the input arguments
    region_name = args.noise    # region file within which evaluate the RMS
    catalog_file = '' + str(args.target) + '_sofia_output_cat.txt'     # SoFIA ASCII catalog file
    mask_file = '' + str(args.target) + '_sofia_output_mask.fits'       # Mask FITS file from SoFIA
    rm_map_file = '' + str(args.target) +'_RMobs_clean_masked.fits'  # Observed RM map (possibly to be removed, given that it is created in a function in this script)
    output_file = '' + str(args.target) +'_galrm_corrected.fits'   # Output name of the catalog and galaxy-corrected RM map
    exclrad = args.exclrad      # Exclusion radius in degrees for the annulus method (see Anderson et al. 2024)
    outrad = args.outrad        # Outside radius in degrees for the annulus method (see Anderson et al. 2024)
    nsources = args.nsources    # Number of sources to retain for annulus method (see Anderson et al. 2024)

    '''
    # Check and correct the header of the MGCLS Q and U cubes
    header_checker(args.inputs)

    # Write here the file with the list of frequencies - for these cubes these are already known
    frequ = np.array([908.e06, 952.e06, 996.e06, 1044.e06, 1093.e06, 1145.e06, 1318.e06, 1382.e06, 1448.e06, 1482.e06, 1594.e06, 1656.e06])
    with open(''+args.target+'_freqlist.dat', 'w') as lfr:
        for item in frequ:
            lfr.write(str(item)+'\n')


    # Here we select a region for which a "sub-cube" is obtained and write the RMS for each image of the cube 
    # for both Q and U into a text file
    if os.path.exists('./' +str(args.target)+ '_noiselistQUavg.dat'):
        print("Noise on Q and U cubes has already been calculated")
    else:
        print("Calculating noise on Q and U cubes")
        media_QU = qu_noise_calculator(nchan, region_name, args.inputs, args.target)


    # Execution of rmsynth3D
    if (args.rmsynth == True):
        print("RM-synthesis 3D execution:")
        print("~/.local/bin/rmsynth3d " + str(args.inputs[0]) + "_cor_header.fits " + str(args.inputs[1]) + "_cor_header.fits " + args.target +"_freqlist.dat -n ./" + str(args.target) + "_noiselistQUavg.dat -o " + str(args.target) + "_ -l 350 -d 3 -v")
        os.system("~/.local/bin/rmsynth3d " + str(args.inputs[0]) + "_cor_header.fits " + str(args.inputs[1]) + "_cor_header.fits " + args.target +"_freqlist.dat -n ./" + str(args.target) + "_noiselistQUavg.dat -o " + str(args.target) + "_ -l 350 -d 3 -v")
    else:
        print("Command to execute the RM-synthesis 3D on your own")
        print("~/.local/bin/rmsynth3d " + str(args.inputs[0]) + "_cor_header.fits " + str(args.inputs[1]) + "_cor_header.fits " + args.target +"_freqlist.dat -n ./" + str(args.target) + "_noiselistQUavg.dat -o " + str(args.target) + "_ -l 350 -d 3 -v")

    
    # Execution of the rmclean3D
    if (args.rmclean == True):
        print("~/.local/bin/rmclean3d " + str(args.target) + "_FDF_tot_dirty.fits " +str(args.target) + "_RMSF_tot.fits -c 3.464e-5 -n 1000 -v -o " + str(args.target) + "_")
        os.system("~/.local/bin/rmclean3d " + str(args.target) + "_FDF_tot_dirty.fits " +str(args.target) + "_RMSF_tot.fits -c 3.464e-5 -n 1000 -v -o " + str(args.target) + "_")
    else:
        print("Command to execute the rmclean3d on your own")
        print("~/.local/bin/rmclean3d " + str(args.target) + "_FDF_tot_dirty.fits " +str(args.target) + "_RMSF_tot.fits -c 3.464e-5 -n 1000 -v -o " + str(args.target) + "_")

    '''
    print("Stokes I map header correction")
    header_checker_stokesi(name_i)
    
    
    #sigma_p = 4.33e-6 # Jy/beam,read from the RMsynth_parameters.py script
    RMSF_FWHM = 45.44 # in rad/m2,read from the RMSF_FWHM.fits image or from the RMsynth_parameters.py script (theoretical value)

    rms_i, mask_tot = pol_maps_maker(args.target, name_i, RMSF_FWHM)
    
    print("Plotting maps")
    plot_stokesi(args.target, "" + str(name_i) + "_cor_header.fits", args.ra, args.dec, args.rfh)
    plot_polint(args.target, '' + str(args.target) +'_P_clean_6sig_unmasked.fits', args.ra, args.dec, args.rfh)
    plot_polfrac(args.target, '' + str(args.target) +'_polfrac_clean_6sig.fits', args.ra, args.dec, args.rfh)

    ## Catalog using SoFIA
    print("SoFIA Catalog creation")
    
    if os.path.exists(args.sofia):
        print("Local SoFIA installation found: " + str(args.sofia) + "")
        os.system("" + str(args.sofia) + " " + str(args.param)+ " input.data=" + str(name_i) + "_cor_header.fits output.filename=" + str(args.target) + "_sofia_output")
    else:
        print("Local SoFIA installation not found/provided. Using the Singularity image")
        os.system("singularity exec docker://sofiapipeline/sofia2:latest sofia " + str(args.param) + " input.data=" + str(name_i) + " output.filename=" + str(args.target) + "_sofia_output")

    # input.mask  you can give the Stokes I and Q/U combined mask
  
    
    ## Galactic RM correction
    print("Extract RM for the catalog sources")
    catalog_with_rm = extract_rm_for_sources(
        catalog_file, 
        mask_file, 
        rm_map_file
    )

    # Apply Galactic RM correction to all source pixels
    print("Apply Galactic RM correction to all source pixels")
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
        "" + str(name_i) + "_cor_header.fits",
        rms_i,
        mask_tot,
        output_file = output_file.replace('.fits', '_rmgrid.png')
    )

    radial_profile_annulus(
        output_file, 
        args.ra,
        args.dec, 
        args.rfh, 
        5
    )