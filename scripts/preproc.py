import re
import os
import sys
import glob
import numpy as np
import os.path as pa
from astropy.io import fits
from astropy.table import Table, Column, vstack

GDIR = pa.dirname(pa.dirname(pa.abspath(__file__)))
sys.path.insert(1, GDIR + '/scripts/utils')
from ZeroPointCalculator import ZeroPoint_Calculator
from SymmetricMatch import Sky_Symmetric_Match
from LimMagEstimator import LimMag_Estimator
from SExSkySubtract import SEx_SkySubtract
from FoVEstimator import FoV_Estimator
from PYSEx import PY_SEx

AstBrick = Table.read(GDIR + '/lsdr10/tractor-combined.fits')

def MAIN(FITS_iobj):

    objname = pa.basename(FITS_iobj)[:-5]
    outdir = GDIR + '/procdata/%s' %objname
    try: 
        os.makedirs(outdir)
    except:
        if pa.exists(outdir):
            pass

    FITS_obj = outdir + '/%s' %(pa.basename(FITS_iobj))
    if not pa.exists(FITS_obj):
        os.system('ln -s %s %s' %(FITS_iobj, FITS_obj))

    with fits.open(FITS_obj, mode='update') as hdl:
        hdl[0].header['GAIN'] = (1.0, 'MeLOn')
        hdl[0].header['SATURATE'] = (50000.0, 'MeLOn')
        if 'FILTER' not in hdl[0].header:
            hdl[0].header['FILTER'] = re.split('_', pa.basename(FITS_obj))[1]
        hdl.flush()

    # * sky subtraction
    FITS_skysub = FITS_obj[:-5] + '.skysub.fits'
    SEx_SkySubtract.SSS(
        FITS_obj=FITS_obj,
        FITS_skysub=FITS_skysub,
        SATUR_KEY="SATURATE",
        ESATUR_KEY="ESATUR",
        BACK_SIZE=64,
        BACK_FILTERSIZE=3,
        DETECT_THRESH=1.5,
        DETECT_MINAREA=5,
        DETECT_MAXAREA=0,
    )

    # * phot to get optimal apeture (AUTO)
    SExParam = ['X_IMAGE', 'Y_IMAGE', 'XWIN_IMAGE', 'YWIN_IMAGE', 'SNR_WIN', 'FLAGS', \
                'FLUX_AUTO', 'FLUXERR_AUTO', 'MAG_AUTO', 'MAGERR_AUTO', 'FLUX_RADIUS', \
                'FWHM_IMAGE', 'FLUX_MAX', 'A_IMAGE', 'B_IMAGE', 'ELLIPTICITY']

    AstSEx_AUTO = PY_SEx.PS(FITS_obj=FITS_skysub, SExParam=SExParam, GAIN_KEY='GAIN', \
        SATUR_KEY='ESATUR', BACK_TYPE='MANUAL', BACK_VALUE=0.0, \
        BACK_SIZE=64, BACK_FILTERSIZE=3, DETECT_THRESH=1.5, \
        ANALYSIS_THRESH=1.5, DETECT_MINAREA=5, DETECT_MAXAREA=0, \
        DEBLEND_MINCONT=0.005, BACKPHOTO_TYPE='LOCAL', \
        CHECKIMAGE_TYPE='NONE', AddRD=True, ONLY_FLAGS=None, XBoundary=10., \
        YBoundary=10., MDIR=None, VERBOSE_LEVEL=1)[0]

    # * cross match to legacy survey
    Match_rdtol = 1.
    RD_A = np.array([AstSEx_AUTO['X_WORLD'], AstSEx_AUTO['Y_WORLD']]).T
    RD_B = np.array([AstBrick['ra'], AstBrick['dec']]).T
    Symm = Sky_Symmetric_Match.SSM(RD_A=RD_A, RD_B=RD_B, tol=Match_rdtol, return_distance=False)
    AstSEx_M = AstSEx_AUTO[Symm[:, 0]]
    AstBrick_M = AstBrick[Symm[:, 1]]

    FWHM = np.median(AstSEx_M['FWHM_IMAGE'][AstBrick_M['type'] == 'PSF'])
    OPTAPER = round(1.3462*FWHM, 2)

    # * rephot with optimal aperture
    SExParam = ['X_IMAGE', 'Y_IMAGE', 'XWIN_IMAGE', 'YWIN_IMAGE', 'SNR_WIN', 'FLAGS', \
                'FLUX_APER', 'FLUXERR_APER', 'MAG_APER', 'MAGERR_APER', 'FLUX_RADIUS', \
                'FWHM_IMAGE', 'FLUX_MAX', 'A_IMAGE', 'B_IMAGE', 'ELLIPTICITY']

    AstSEx_APER = PY_SEx.PS(FITS_obj=FITS_skysub, SExParam=SExParam, GAIN_KEY='GAIN', \
        SATUR_KEY='ESATUR', BACK_TYPE='MANUAL', BACK_VALUE=0.0, \
        BACK_SIZE=64, BACK_FILTERSIZE=3, DETECT_THRESH=1.5, \
        ANALYSIS_THRESH=1.5, DETECT_MINAREA=5, DETECT_MAXAREA=0, \
        DEBLEND_MINCONT=0.005, BACKPHOTO_TYPE='LOCAL', PHOT_APERTURES=OPTAPER, \
        CHECKIMAGE_TYPE='NONE', AddRD=True, ONLY_FLAGS=None, XBoundary=10., \
        YBoundary=10., MDIR=None, VERBOSE_LEVEL=1)[0]

    # * cross match to legacy survey again
    Match_rdtol = 1.
    RD_A = np.array([AstSEx_APER['X_WORLD'], AstSEx_APER['Y_WORLD']]).T
    RD_B = np.array([AstBrick['ra'], AstBrick['dec']]).T
    Symm = Sky_Symmetric_Match.SSM(RD_A=RD_A, RD_B=RD_B, tol=Match_rdtol, return_distance=False)
    AstSEx_M = AstSEx_APER[Symm[:, 0]]
    AstBrick_M = AstBrick[Symm[:, 1]]

    PSF_MASK = AstBrick_M['type'] == 'PSF'
    AstSEx_MPSF = AstSEx_M[PSF_MASK]
    AstBrick_MPSF = AstBrick_M[PSF_MASK]

    # * calculate ZP for PSF stars
    FILTER = fits.getheader(FITS_obj, ext=0)['FILTER']
    MAG_MOBJ = np.array(AstSEx_MPSF['MAG_APER'])
    MAGERR_MOBJ = np.array(AstSEx_MPSF['MAGERR_APER'])
    MAG_MREF = np.array(AstBrick_MPSF['%smag' %FILTER])
    MAGERR_MREF = np.array(AstBrick_MPSF['e_%smag' %FILTER])

    ZP, ZPERR = ZeroPoint_Calculator.ZPC(MAG_MOBJ=MAG_MOBJ, MAGERR_MOBJ=MAGERR_MOBJ, \
        MAG_MREF=MAG_MREF, MAGERR_MREF=MAGERR_MREF, MINFRAC_FIT=0.5)
    print('MeLOn CheckPoint: Magnitude ZeroPoint [%.2f +/- %.4f]!' %(ZP, ZPERR))

    # * simple estimate of object image 
    LME = LimMag_Estimator(
        FITS_obj=FITS_skysub, APER=OPTAPER, ZP_APER=ZP
    )
    LM3, LM5 = LME.BySkySig(NSIG=[3, 5])
    
    row = [objname, np.sum(PSF_MASK), FWHM, OPTAPER, ZP, ZPERR, LM3, LM5]
    return row
    
rowdata = []
for FITS_iobj in glob.glob(GDIR + '/ligo_wst/data/S230615az/*/s.tvvmepdnybo*.fits'):
    try: 
        row = MAIN(FITS_iobj)
        rowdata.append(row)
    except: 
        outdir = GDIR + '/procdata/%s' %(pa.basename(FITS_iobj)[:-5])
        os.system('rm -rf %s' %outdir)
        print('MeLOn ERROR: SOMETHING WRONG AND SKIP [%s]!' %FITS_iobj)

AstSUM = Table(rows=rowdata, names=['FILENAME_SCI', 'NMATCH_LSDR10PSF', 'FWHM_IMAGE', 'OPTIMAL_APERTURE', 'ZP_SCI', 'eZP_SCI', 'LIMMAG3_SCI', 'LIMMAG5_SCI'])
AstSUM.write(GDIR + '/procdata/wst_growth_depth.csv', format='ascii.csv', overwrite=True)
