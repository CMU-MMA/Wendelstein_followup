import os
import sys
import glob
import numpy as np
import os.path as pa
from astropy.table import Table, Column, vstack

GDIR = pa.dirname(pa.dirname(pa.abspath(__file__)))
sys.path.insert(1, GDIR + '/scripts/utils')
from FoVEstimator import FoV_Estimator

LSLINK = "https://portal.nersc.gov/project/cosmo/data/legacysurvey"
DOWNLINK_LS10Ma = LSLINK + "/dr10/south/survey-bricks-dr10-south.fits.gz"

FITS_LSMa = GDIR + '/lsdr10/survey-bricks-dr10-south.fits'
if not pa.exists(FITS_LSMa):
    try: 
        os.makedirs(GDIR + '/lsdr10')
    except OSError:
        if pa.exists(GDIR + '/lsdr10'):
            pass
        
    print('downloading Legacy Survey DR10 Summary Catalog... \n # %s' %FITS_LSMa)
    os.system(
        'wget --no-verbose --no-clobber --output-document %s %s' \
        %(FITS_LSMa, DOWNLINK_LS10Ma)
    )

AstLSMa = Table.read(FITS_LSMa)

def MAIN(FITS_obj, radius=0.2):

    try:
        FoVC, PIXSCAL, FoV = FoV_Estimator.FVE(FITS_obj=FITS_obj, \
            CalFoVC=True, CalPIXSCAL=True, CalFoV=True, VERBOSE_LEVEL=1)
        RA_obj, DEC_obj = FoVC
    except: 
        print('MeLOn ERROR: FAIL TO MEASURE ITS FOV [%s]' %FITS_obj)
        return None

    # Find bricks within radius
    covering_mask = (
        (AstLSMa["ra"] - RA_obj) ** 2
        + (AstLSMa["dec"] - DEC_obj) ** 2
    ) ** 0.5 < radius

    for line in AstLSMa[covering_mask]:
        brickname = line['brickname']
        bricklink = 'https://portal.nersc.gov/project/cosmo/data/legacysurvey/dr10/south/tractor/%s/tractor-%s.fits' %(brickname[:3], brickname)
        FITS_brick = GDIR + '/lsdr10/tractor-%s.fits' %brickname
        if not pa.exists(FITS_brick):
            os.system('wget --no-verbose --no-clobber --output-document=%s %s' %(FITS_brick, bricklink))

    return None

# download ls bricks
for FITS_obj in glob.glob(GDIR + '/ligo_wst/data/S230615az/*/s.tvvmepdnybo*.fits'):
    MAIN(FITS_obj)

# combine the bricks
AstBrick = []
for FITS_brick in glob.glob(GDIR + '/lsdr10/tractor-*.fits'):
    Astb = Table.read(FITS_brick)
    
    # ** add columns for griz
    _FLUX, _IVAR = Astb["flux_g"], Astb["flux_ivar_g"]
    _FLUX[_FLUX <= 0] = np.nan
    _IVAR[_IVAR <= 0] = np.nan
    gMAG = -2.5 * np.log10(_FLUX) + 22.5
    gMAGERR = 1.0857 * (np.sqrt(1 / _IVAR) / _FLUX)
    Astb.add_column(Column(gMAG, name="gmag"))
    Astb.add_column(Column(gMAGERR, name="e_gmag"))

    _FLUX, _IVAR = Astb["flux_r"], Astb["flux_ivar_r"]
    _FLUX[_FLUX <= 0] = np.nan
    _IVAR[_IVAR <= 0] = np.nan
    rMAG = -2.5 * np.log10(_FLUX) + 22.5
    rMAGERR = 1.0857 * (np.sqrt(1 / _IVAR) / _FLUX)
    Astb.add_column(Column(rMAG, name="rmag"))
    Astb.add_column(Column(rMAGERR, name="e_rmag"))
    
    _FLUX, _IVAR = Astb["flux_i"], Astb["flux_ivar_i"]
    _FLUX[_FLUX <= 0] = np.nan
    _IVAR[_IVAR <= 0] = np.nan
    iMAG = -2.5 * np.log10(_FLUX) + 22.5
    iMAGERR = 1.0857 * (np.sqrt(1 / _IVAR) / _FLUX)
    Astb.add_column(Column(iMAG, name="imag"))
    Astb.add_column(Column(iMAGERR, name="e_imag"))

    _FLUX, _IVAR = Astb["flux_z"], Astb["flux_ivar_z"]
    _FLUX[_FLUX <= 0] = np.nan
    _IVAR[_IVAR <= 0] = np.nan
    zMAG = -2.5 * np.log10(_FLUX) + 22.5
    zMAGERR = 1.0857 * (np.sqrt(1 / _IVAR) / _FLUX)
    Astb.add_column(Column(zMAG, name="zmag"))
    Astb.add_column(Column(zMAGERR, name="e_zmag"))
    UID = ["%s_%s" % (x, y) for x, y in zip(Astb["brickname"], Astb["objid"])]
    Astb.add_column(Column(UID, name="UID"))
    
    AstBrick.append(Astb)
AstBrick = vstack(AstBrick)
AstBrick.write(GDIR + '/lsdr10/tractor-combined.fits', overwrite=True)
