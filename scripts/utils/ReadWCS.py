import warnings
from astropy.wcs import WCS, FITSFixedWarning
# version: Mar 17, 2023

class Read_WCS:
    @staticmethod
    def RW(hdr, VERBOSE_LEVEL=2):

        """
        # Remarks on the WCS in FITS header and astropy.WCS
        # Case A:
        #    FITS header: 
        #        CTYPE1 = 'RA---TAN'   CTYPE2 = 'DEC--TAN'
        #        CD1_1 = ...           CD1_2 = ...
        #        CD2_1 = ...           CD2_2 = ... 
        #        NOTE: NO DISTORTION TERMS!
        #    
        #    astropy.WCS:        
        #        # print(WCS(phr).to_header())
        #        CTYPE1 = 'RA---TAN'   CTYPE2 = 'DEC--TAN'
        #        PC1_1 = ...           PC1_2 = ...
        #        PC2_1 = ...           PC2_2 = ... 
        #        NOTE: if CD1_2 = CD2_1 = 0.0, then PC1_2 and PC2_1 will be absent.
        #
        # Case B (Distorted tangential with the TPV FITS convention, added officially to the FITS registry in Aug. 2012):
        #    FITS header: 
        #        CTYPE1 = 'RA---TPV'   CTYPE2 = 'DEC--TPV'
        #        CD1_1 = ...           CD1_2 = ...
        #        CD2_1 = ...           CD2_2 = ... 
        #        PV1_0 = ...           PV1_1 = ...
        #        ...
        #    
        #    astropy.WCS:        
        #        # print(WCS(phr).to_header())
        #        CTYPE1 = 'RA---TPV'   CTYPE2 = 'DEC--TPV'
        #        PC1_1 = ...           PC1_2 = ...
        #        PC2_1 = ...           PC2_2 = ... 
        #        PV1_0 = ...           PV1_1 = ...
        #        ...
        #        NOTE: if CD1_2 = CD2_1 = 0.0, then PC1_2 and PC2_1 will be absent.
        #
        # Case B' (Distorted tangential with Greisen & Calabretta's 2000 draft, obsoleted):
        #    FITS header:
        #        CTYPE1 = 'RA---TAN'   CTYPE2 = 'DEC--TAN'
        #        CD1_1 = ...           CD1_2 = ...
        #        CD2_1 = ...           CD2_2 = ... 
        #        PV1_0 = ...           PV1_1 = ...
        #        ...
        #    
        #    WARNING: To avoid Astropy.WCS incompatibility, one should make corrections 
        #        as follows before reading by Astropy.WCS
        #        phr['CTYPE1'] = 'RA---TPV'
        #        phr['CTYPE1'] = 'DEC--TPV'
        #        w = WCS(phr)
        #      
        #    astropy.WCS:        
        #        # print(w.to_header())
        #        CTYPE1 = 'RA---TPV'   CTYPE2 = 'DEC--TPV'
        #        PC1_1 = ...           PC1_2 = ...
        #        PC2_1 = ...           PC2_2 = ... 
        #        PV1_0 = ...           PV1_1 = ...
        #        ...
        #        NOTE: if CD1_2 = CD2_1 = 0.0, then PC1_2 and PC2_1 will be absent.
        #
        # Case C:
        #    FITS header: 
        #        CTYPE1 = 'RA---TAN-SIP'   CTYPE2 = 'DEC--TAN-SIP'
        #        CD1_1 = ...           CD1_2 = ...
        #        CD2_1 = ...           CD2_2 = ... 
        #        A_ORDER = 2           
        #        A_0_0 = ...           A_0_1 = ...
        #        ...
        #        B_ORDER = 2           
        #        B_0_0 = ...           B_0_1 = ...
        #        ...
        #        AP_ORDER = 2           
        #        AP_0_0 = ...          AP_0_1 = ...
        #        ...
        #        BP_ORDER = 2           
        #        BP_0_0 = ...          BP_0_1 = ...
        #        ...
        #    
        #    astropy.WCS:        
        #        # print(WCS(phr).to_header(relax=True))
        #        CTYPE1 = 'RA---TAN-SIP'   CTYPE2 = 'DEC--TAN-SIP'
        #        PC1_1 = ...           PC1_2 = ...
        #        PC2_1 = ...           PC2_2 = ... 
        #        A_ORDER = 2           
        #        A_0_0 = ...           A_0_1 = ...
        #        ...
        #        B_ORDER = 2           
        #        B_0_0 = ...           B_0_1 = ...
        #        ...
        #        AP_ORDER = 2           
        #        AP_0_0 = ...          AP_0_1 = ...
        #        ...
        #        BP_ORDER = 2           
        #        BP_0_0 = ...          BP_0_1 = ...
        #        ...
        #        NOTE: if CD1_2 = CD2_1 = 0.0, then PC1_2 and PC2_1 will be absent.
        #
        # P.S. One may use PC1_1 to replace CD1_1 in FITS header for above cases, not recommended though.
        #
        """

        with warnings.catch_warnings():
            if VERBOSE_LEVEL in [0, 1]: behavior = 'ignore'
            if VERBOSE_LEVEL in [2]: behavior = 'default'
            warnings.filterwarnings(behavior, category=FITSFixedWarning)

            if hdr['CTYPE1'] == 'RA---TAN' and 'PV1_0' in hdr:
                _hdr = hdr.copy()
                _hdr['CTYPE1'] = 'RA---TPV'
                _hdr['CTYPE2'] = 'DEC--TPV'
            else: _hdr = hdr
            w = WCS(_hdr)
        
        return w
