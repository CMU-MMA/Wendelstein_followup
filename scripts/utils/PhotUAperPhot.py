import warnings
import numpy as np
from photutils.utils import calc_total_error
from astropy.stats import sigma_clipped_stats
from SkyLevelEstimator import SkyLevel_Estimator
from photutils import CircularAnnulus, CircularAperture, aperture_photometry
# version: Jan 21, 2023

class PhotU_AperPhot:
    @staticmethod
    def PUAP(PixA_obj, XY, HALF_APER, effective_gain=None, donut_inner_radius=None, return_magnitude=True, VERBOSE_LEVEL=2):
        
        """
        # MeLOn Notes 
        # *** Remarks on the software cross-usage systematic error ***
        # Both SExtractor and Photutils can perform Aperture Photometry, while it is not convinent to use SEx for Force-Photometry!
        # Therefore it is more economical to empoly Photutils when target coordinates are given, typically, when we want to count 
        # aperture flux on difference images for known candidates. It is thus necessary to keep in mind the systematic error.
        #
        # i. When we use a collection of stars to statistically determine the flux zeropoint of the image. They only have
        #    some tiny divergence, about 0.002 - 0.005 mag, which suggests even we derive OptAper zeropoint of Image J via SEx, 
        #    it is still ok, in most cases, to use Photutils on the corresponding difference image. 
        # ii. However, If you perform photometry on some individual source by SEx & Photutils respectively, 
        #     the diverenge can large as 0.02 mag. This means if you choose using Photutils on differences, 
        #     please always do in that way. Do NOT mix up the two softwares arbitrarily.
        #
        # * Remarks on Aperture Photometry 
        #    It is well-known that PSF varys across the whole FOV, especially for wide field surveys. 
        #    Nevertheless, Aperture Photometry is still proper for the point sources with spatial-varying PSF. 
        #    For example, when Gaussian Optimal Aperture is applied, there is ~ 89% (estimated from Gaussian Profile) 
        #    flux enclosed in the aperture. The PSF spatial variation probably has inappreciable impact on the 
        #    proportion of leak light. That is why Aperture Photometry is quite popular even for wide field surveys in practice.
        #    ZP_APER is estimated with averaging such effect due to PSF spatial variation.
        #
        """

        # * Convert XY (default, in Fortran framework) into Numpy XY_C
        XY_C = XY - 1.0
        apertures = CircularAperture(XY_C, r=HALF_APER)
        
        if VERBOSE_LEVEL in [2]:
            print('MeLOn CheckPoint: Convert input Fortran (default) Coordinate into C (Numpy) Coordinate!')
            print('MeLOn REMINDER: photutils uses Radius, while SEx aperture is Diameter!')

        # NOTE effective_gain=None means DO NOT calculate noise.
        # NOTE donut_inner_radius is the inner radius of background donut.
        #      None means DO NOT subtract a background measured from a donut.
        #      (for example, set donut_inner_radius = 3*fwhm)

        CalcNoise, AnnBkg = False, False
        if effective_gain is not None: CalcNoise = True
        if donut_inner_radius is not None: AnnBkg = True

        data = PixA_obj.T
        if not CalcNoise:
            AstPhot = aperture_photometry(data, apertures)

        if CalcNoise:
            # Remarks on Noise Calculation 
            # a. Effective_gain has unit electrons/ADU, which can be a 2D exposure map (useful for mosaic image)
            # b. Simple error calculation with taking Possion Noise & Background Noise (sky ...) 
            #     into account ----- sig_tot = sqrt(sig_bkg^2 + I/gain)
            bkg_error = SkyLevel_Estimator.SLE(PixA_obj=PixA_obj)[1]
            error = calc_total_error(data, bkg_error, effective_gain)             
            AstPhot = aperture_photometry(data, apertures, error=error) 

        if not AnnBkg:
            AstPhot['FLUX_APER']  = AstPhot['aperture_sum'] 
            if return_magnitude: AstPhot['MAG_APER']  = -2.5 * np.log10(np.abs(AstPhot['FLUX_APER']))
            if CalcNoise: 
                AstPhot['FLUXERR_APER'] = AstPhot['aperture_sum_err']
                if return_magnitude: AstPhot['MAGERR_APER']  = 1.0857 * np.abs(AstPhot['FLUXERR_APER'] / AstPhot['FLUX_APER'])

        if donut_inner_radius is not None:
            r_in = donut_inner_radius
            r_out = r_in + 24.0  # SEx Rectangle 24 thickness 
            annulus_apertures = CircularAnnulus(XY_C, r_in=r_in, r_out=r_out)
            annulus_masks = annulus_apertures.to_mask(method='center')

            bkg_median = []
            for mask in annulus_masks:
                annulus_data = mask.multiply(data, fill_value=np.nan)
                annulus_data_1d = annulus_data[mask.data > 0]
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    median_sigclip = sigma_clipped_stats(annulus_data_1d, sigma=3.0, maxiters=5)[1]
                bkg_median.append(median_sigclip)
            bkg_median = np.array(bkg_median)

            AstPhot['annulus_median'] = bkg_median
            AstPhot['aper_bkg'] = bkg_median * apertures.area
            AstPhot['aper_sum_bkgsub'] = AstPhot['aperture_sum'] - AstPhot['aper_bkg']
            
            AstPhot['FLUX_APER']  = AstPhot['aper_sum_bkgsub'] 
            if return_magnitude: AstPhot['MAG_APER']  = -2.5 * np.log10(np.abs(AstPhot['FLUX_APER']))
            if CalcNoise: 
                AstPhot['FLUXERR_APER']  = AstPhot['aperture_sum_err']
                if return_magnitude: AstPhot['MAGERR_APER']  = 1.0857 * np.abs(AstPhot['FLUXERR_APER'] / AstPhot['FLUX_APER'])

        AstPhot['Xcen'] = AstPhot['xcenter']
        AstPhot['Ycen'] = AstPhot['ycenter']

        for col in AstPhot.colnames:
            AstPhot[col].info.format = '%.8g'  # for consistent table output
        
        return AstPhot
