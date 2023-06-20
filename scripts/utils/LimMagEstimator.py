import math
import warnings
import numpy as np
from PYSEx import PY_SEx
from scipy.stats import iqr
from astropy.io import fits
from PhotUAperPhot import PhotU_AperPhot
from SkyLevelEstimator import SkyLevel_Estimator
from SamplingBackground import Sampling_Background
# version: Feb 4, 2023

class LimMag_Estimator:

    def __init__(self, FITS_obj, APER=5.0, ZP_APER=30.0, VERBOSE_LEVEL=2):
        
        self.FITS_obj = FITS_obj
        self.APER = APER
        self.ZP_APER = ZP_APER
        self.VERBOSE_LEVEL = VERBOSE_LEVEL

    def BySkySig(self, NSIG=[3,5]):
        
        # NOTE fact: faint source at limiting magnitude is background noise domainated.
        # NOTE assumption: independent Gaussian background noise.
        
        PixA_obj = fits.getdata(self.FITS_obj, ext=0).T
        skysig = SkyLevel_Estimator.SLE(PixA_obj=PixA_obj)[1]
        AREA = np.pi*(self.APER/2.0)**2
        FLUX_LIM = np.array(NSIG) * skysig * np.sqrt(AREA)
        MAG_LIM = self.ZP_APER - 2.5 * np.log10(FLUX_LIM)

        return MAG_LIM

    def create_detect_mask(self, BACK_TYPE='AUTO', BACK_VALUE=0.0, BACK_SIZE=64, BACK_FILTERSIZE=3, \
        DETECT_THRESH=1.0, DETECT_MINAREA=5, DETECT_MAXAREA=0):

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # NOTE: GAIN, SATURATE, ANALYSIS_THRESH, DEBLEND_MINCONT, BACKPHOTO_TYPE do not affect the detection mask.
            DETECT_MASK = PY_SEx.PS(FITS_obj=self.FITS_obj, SExParam=['X_IMAGE', 'Y_IMAGE'], GAIN_KEY='PHGAIN', \
                SATUR_KEY='PHSATUR', BACK_TYPE=BACK_TYPE, BACK_VALUE=BACK_VALUE, BACK_SIZE=BACK_SIZE, \
                BACK_FILTERSIZE=BACK_FILTERSIZE, DETECT_THRESH=DETECT_THRESH, ANALYSIS_THRESH=1.5, \
                DETECT_MINAREA=DETECT_MINAREA, DETECT_MAXAREA=DETECT_MAXAREA, DEBLEND_MINCONT=0.005, \
                BACKPHOTO_TYPE='GLOBAL', CHECKIMAGE_TYPE='OBJECTS', MDIR=None, \
                VERBOSE_LEVEL=self.VERBOSE_LEVEL)[1][0].astype(bool)

        return DETECT_MASK

    def ByRandomForcePhot(self, NSIG=[3,5], DETECT_MASK=None, NUM_SAMP=1024, RandomSeed=None):
        
        # NOTE A FACT: faint source at limiting magnitude is background noise domainated.
        # NOTE WARNING: when DETECT_MASK is not provided, please make sure that 
        #               most apertures are background-only pixels (e.g., diff).
        
        APERHW = math.ceil(self.APER/2.0-0.5)
        if DETECT_MASK is not None:
            RC_pit = Sampling_Background.SB(DETECT_MASK=DETECT_MASK, HW=APERHW, \
                NUM_SAMP=NUM_SAMP, RandomSeed=RandomSeed)
            if RandomSeed is not None: np.random.seed(RandomSeed)
            XY_phot = RC_pit + np.random.uniform(0.5, 1.5, RC_pit.shape)
        else:
            phdr = fits.getheader(self.FITS_obj, ext=0)
            N0, N1 = int(phdr['NAXIS1']), int(phdr['NAXIS2'])
            if RandomSeed is not None: np.random.seed(RandomSeed)
            X_phot = np.random.uniform(APERHW+0.5, N0-APERHW+0.5, NUM_SAMP)
            if RandomSeed is not None: np.random.seed(RandomSeed)
            Y_phot = np.random.uniform(APERHW+0.5, N1-APERHW+0.5, NUM_SAMP)
            XY_phot = np.array([X_phot, Y_phot]).T

        PixA_obj = fits.getdata(self.FITS_obj, ext=0).T
        AstPU = PhotU_AperPhot.PUAP(PixA_obj=PixA_obj, XY=XY_phot, HALF_APER=self.APER/2.0, \
            effective_gain=None, donut_inner_radius=2*self.APER, return_magnitude=False, \
            VERBOSE_LEVEL=self.VERBOSE_LEVEL)
        
        FSIG = iqr(np.array(AstPU['FLUX_APER']), nan_policy='omit') / 1.349
        FLUX_LIM = np.array(NSIG) * FSIG
        MAG_LIM = -2.5 * np.log10(FLUX_LIM) + self.ZP_APER

        return MAG_LIM
