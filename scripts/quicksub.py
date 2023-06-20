import os
import re
import sys
import glob
import numpy as np
import os.path as pa
from astropy.io import fits
from tempfile import mkdtemp
import scipy.ndimage as ndimage

GDIR = pa.dirname(pa.dirname(pa.abspath(__file__)))
sys.path.insert(1, GDIR + '/scripts/utils')
#from CustomizedPacket import Customized_Packet
from sfft.CustomizedPacket import Customized_Packet
from astropy.table import Table, Column, vstack
from SymmetricMatch import Sky_Symmetric_Match
from ZeroPointCalculator import ZeroPoint_Calculator
from SExSkySubtract import SEx_SkySubtract
from FoVEstimator import FoV_Estimator
from LimMagEstimator import LimMag_Estimator
from PYSWarp import PY_SWarp
from PYSEx import PY_SEx

def MAIN(FITS_obj):

    tmp1, tmp2 = re.split('S230615az', pa.basename(FITS_obj))[1][:-12], re.split('_', pa.basename(FITS_obj))[1]
    tempFile = GDIR + '/ligo_wst/data/S230615az/Wendelstein/decals_ligodesi_S230615az_%s_%s_%s.fits' %(tmp1[:6], tmp1[6:], tmp2)
    if pa.exists(tempFile):
        
        # * resampling
        FITS_ref = tempFile
        FITS_resamp = FITS_obj[:-5] + '.resampled.fits'
        PY_SWarp.PS(FITS_obj, FITS_ref, FITS_resamp=FITS_resamp, GAIN_KEY='GAIN', SATUR_KEY='ESATUR', \
            OVERSAMPLING=1, RESAMPLING_TYPE='LANCZOS3', SUBTRACT_BACK='N', FILL_VALUE=np.nan, \
            VERBOSE_TYPE='NORMAL', VERBOSE_LEVEL=2)

        # * make subtraction mask
        SExParam = ['X_IMAGE', 'Y_IMAGE', 'XWIN_IMAGE', 'YWIN_IMAGE', 'SNR_WIN', 'FLAGS', \
                    'FLUX_AUTO', 'FLUXERR_AUTO', 'MAG_AUTO', 'MAGERR_AUTO', 'FLUX_RADIUS', \
                    'FWHM_IMAGE', 'FLUX_MAX', 'A_IMAGE', 'B_IMAGE', 'ELLIPTICITY']

        AstSEx_ref = PY_SEx.PS(FITS_obj=FITS_ref, SExParam=SExParam, GAIN_KEY='GAIN', \
            SATUR_KEY='SATURATE', BACK_TYPE='MANUAL', BACK_VALUE=0.0, \
            BACK_SIZE=64, BACK_FILTERSIZE=3, DETECT_THRESH=1.5, \
            ANALYSIS_THRESH=1.5, DETECT_MINAREA=5, DETECT_MAXAREA=0, \
            DEBLEND_MINCONT=0.005, BACKPHOTO_TYPE='LOCAL', \
            CHECKIMAGE_TYPE='NONE', AddRD=True, ONLY_FLAGS=None, XBoundary=10., \
            YBoundary=10., MDIR=None, VERBOSE_LEVEL=1)[0]

        XY_ref = np.array([AstSEx_ref['X_IMAGE'], AstSEx_ref['Y_IMAGE']]).T
        PSRES = PY_SEx.PS(FITS_obj=FITS_resamp, SExParam=SExParam, GAIN_KEY='GAIN', \
            SATUR_KEY='ESATUR', BACK_TYPE='MANUAL', BACK_VALUE=0.0, \
            BACK_SIZE=64, BACK_FILTERSIZE=3, DETECT_THRESH=1.5, \
            ANALYSIS_THRESH=1.5, DETECT_MINAREA=5, DETECT_MAXAREA=0, \
            DEBLEND_MINCONT=0.005, BACKPHOTO_TYPE='LOCAL', \
            CHECKIMAGE_TYPE='SEGMENTATION', AddRD=True, ONLY_FLAGS=None, XBoundary=10., \
            YBoundary=10., XY_Quest=XY_ref, Match_xytol=2.0, Preserve_NoMatch=False, \
            MDIR=None, VERBOSE_LEVEL=1)
        AstSEx_resamp, PixA_SEG_resamp = PSRES[0], PSRES[1][0]

        DETMASK = np.in1d(PixA_SEG_resamp.flatten(), np.array(AstSEx_resamp['SEGLABEL'])).reshape(PixA_SEG_resamp.shape)
        NOISEMASK = PixA_SEG_resamp == 0

        struct21 = ndimage.generate_binary_structure(2, 1)
        DETMASK_DIL = ndimage.binary_dilation(DETMASK, structure=struct21, iterations=10)
        SFFTMASK = np.logical_or(DETMASK, np.logical_and(DETMASK_DIL, NOISEMASK))

        PixA_ref = fits.getdata(FITS_ref, ext=0).T
        PixA_resamp = fits.getdata(FITS_resamp, ext=0).T
        NaNmask = np.logical_or(np.isnan(PixA_ref), np.isnan(PixA_resamp))
        SFFTMASK[NaNmask] = False

        FITS_SFFTMASK = FITS_resamp[:-5] + '.sfftmask.fits'
        with fits.open(FITS_ref) as hdl:
            hdl[0].data[:, :] = SFFTMASK.astype(int).T
            hdl.writeto(FITS_SFFTMASK, overwrite=True)
        
        # * run subtraction
        FITS_REF = FITS_ref
        FITS_SCI = FITS_resamp
        SFFTMASK = fits.getdata(FITS_SFFTMASK, ext=0).T.astype(bool)
        TDIR = mkdtemp(suffix=None, prefix=None, dir=None)

        FITS_mREF = TDIR + '/%s.masked.fits' %(pa.basename(FITS_REF)[:-5])
        with fits.open(FITS_REF) as hdl:
            PixA_base = hdl[0].data.T
            PixA_base[~SFFTMASK] = 0.0
            hdl[0].data[:, :] = PixA_base.T
            hdl.writeto(FITS_mREF, overwrite=True)

        FITS_mSCI = TDIR + '/%s.masked.fits' %(pa.basename(FITS_SCI)[:-5])
        with fits.open(FITS_SCI) as hdl:
            PixA_base = hdl[0].data.T
            PixA_base[~SFFTMASK] = 0.0
            hdl[0].data[:, :] = PixA_base.T
            hdl.writeto(FITS_mSCI, overwrite=True)

        GKerHW = 12
        FITS_DIFF = FITS_SCI[:-5] + '.sfftdiff.fits'
        Customized_Packet.CP(FITS_REF=FITS_REF, FITS_SCI=FITS_SCI, FITS_mREF=FITS_mREF, \
            FITS_mSCI=FITS_mSCI, FITS_DIFF=FITS_DIFF, ForceConv='REF', GKerHW=GKerHW, \
            KerPolyOrder=1, BGPolyOrder=0, ConstPhotRatio=True, \
            BACKEND_4SUBTRACT='Cupy', CUDA_DEVICE_4SUBTRACT='0', \
            NUM_CPU_THREADS_4SUBTRACT=8, VERBOSE_LEVEL=2)   

        """
        np.random.seed(10086)
        N0, N1 = PixA_ref.shape
        XY_REGULARIZE = np.array([
            np.random.uniform(10., N0-10., 512),
            np.random.uniform(10., N1-10., 512)
        ]).T
        
        # a new sfft function, not public yet.
        Customized_Packet.CP(FITS_REF=FITS_REF, FITS_SCI=FITS_SCI, FITS_mREF=FITS_mREF, \
            FITS_mSCI=FITS_mSCI, FITS_DIFF=FITS_DIFF, ForceConv='REF', GKerHW=GKerHW, \
            KerSpType='Polynomial', KerSpDegree=1, KerIntKnotX=[], KerIntKnotY=[], \
            SEPARATE_SCALING=True, ScaSpType='Polynomial', ScaSpDegree=0, ScaIntKnotX=[], ScaIntKnotY=[], \
            BkgSpType='Polynomial', BkgSpDegree=0, BkgIntKnotX=[], BkgIntKnotY=[], \
            REGULARIZE_KERNEL=True, IGNORE_LAPLACIAN_KERCENT=True, XY_REGULARIZE=XY_REGULARIZE, 
            WEIGHT_REGULARIZE=None, LAMBDA_REGULARIZE=1e-5, BACKEND_4SUBTRACT='Cupy', \
            CUDA_DEVICE_4SUBTRACT='0', MAX_THREADS_PER_BLOCK=8, MINIMIZE_GPU_MEMORY_USAGE=False, \
            NUM_CPU_THREADS_4SUBTRACT=8, VERBOSE_LEVEL=2)
        """
        os.system('rm -rf %s' %TDIR)

        return None

for FITS_obj in glob.glob(GDIR + '/procdata/s.tvvmepdnybo-*/s.tvvmepdnybo*.skysub.fits'):
    MAIN(FITS_obj)
