import numpy as np
from astropy.io import fits
from ReadWCS import Read_WCS
from astropy.wcs import utils
# version: Jun 19, 2023

class FoV_Estimator:
    @staticmethod
    def FVE(FITS_obj, EXTINDEX=0, EXTNAME=None, CalFoVC=False, CalPIXSCAL=False, CalFoV=False, \
        VERBOSE_LEVEL=2):
        
        hdr, w = None, None
        FoVC, PIXSCAL, FoV = None, None, None

        # * estimate the sky coordinate of FoV center from FITS header
        if CalFoVC:
            if EXTNAME is not None:
                hdr = fits.getheader(FITS_obj, extname=EXTNAME)
            else: hdr = fits.getheader(FITS_obj, ext=EXTINDEX)
            w = Read_WCS.RW(hdr, VERBOSE_LEVEL=VERBOSE_LEVEL)
            
            x_FoVC = 0.5 + int(hdr['NAXIS1'])/2.0
            y_FoVC = 0.5 + int(hdr['NAXIS2'])/2.0
            FoVC = w.all_pix2world(np.array([[x_FoVC, y_FoVC]]), 1)[0]

            if VERBOSE_LEVEL in [1, 2]:
                _message = 'Image Center at sky coordinate (%.6f, %.6f)' %(FoVC[0], FoVC[1])
                print('MeLOn CheckPoint: %s' %_message)
            
        # * estimate the pixel scale
        if CalPIXSCAL:
            if hdr is None:
                if EXTNAME is not None:
                    hdr = fits.getheader(FITS_obj, extname=EXTNAME)
                else: hdr = fits.getheader(FITS_obj, ext=EXTINDEX)
                w = Read_WCS.RW(hdr, VERBOSE_LEVEL=VERBOSE_LEVEL)
            
            pixscal_x, pixscal_y = utils.proj_plane_pixel_scales(w) 
            if w.wcs.cunit[0] != 'deg' or w.wcs.cunit[1] != 'deg':
                _error_message = 'Please use unit of deg for WCS!'
                raise Exception('MeLOn ERROR: %s' %_error_message)
            
            disc = 100. * np.abs(pixscal_x - pixscal_y)/pixscal_x
            if disc > 1.:
                _error_message = 'pixel scale along two axes have too large discrepancy [%s]!' \
                    %('{:.2%}'.format(disc))
                raise Exception('MeLOn ERROR: %s' %_error_message)
            
            PIXSCAL = (pixscal_x + pixscal_y) * 1800
            if VERBOSE_LEVEL in [1, 2]:
                _message = 'measured Pixel Scale [%.6f arcsec/pix]!' %PIXSCAL
                print('MeLOn CheckPoint: %s' %_message)

        # * calculate the Field of View (Angular-Distance-Span on X & Y axis)
        if CalFoV:
            if not CalPIXSCAL: 
                _error_message = 'Please set CalPIXSCAL=True for calculating FoV!'
                raise Exception('MeLOn ERROR: %s' %_error_message)
            
            FoV = int(hdr['NAXIS1'])*pixscal_x, int(hdr['NAXIS2'])*pixscal_y
            if VERBOSE_LEVEL in [1, 2]:
                _message = 'measured FoV [%.6f x %.6f deg]!' %(FoV[0], FoV[1])
                print('MeLOn CheckPoint: %s' %_message)

        return FoVC, PIXSCAL, FoV
