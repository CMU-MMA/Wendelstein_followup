import numpy as np
from scipy.spatial import cKDTree
from astropy.convolution import convolve
# version: Sep 28, 2022

class Sampling_Background:
    @staticmethod
    def SB(DETECT_MASK, HW=35, NUM_SAMP=256, RandomSeed=None):

        # * MeLOn Notes
        #   @ Extract pixel-boxes on background as samples
        #     METHOD: Consider a pixel-box centred at pixel (r, c), It can be a sample of background 
        #     iff the box is empty, where pixels are all undetected. This criteria is equivalent to 
        #     performing a convolution on OBJmask by a all-unity kernel (box-size), then all  
        #     available box-centers are just the zero-value pixels on the convolved image.
        #     P.S. DETECT_MASK can be produced by SEx with CHECKIMAGE_TYPE='OBJECTS'. 
        
        NKer = np.ones((2*HW+1, 2*HW+1))
        PixA_convd = convolve(DETECT_MASK.astype(int), NKer, boundary='extend', normalize_kernel=False)

        # give all available box-centers (considering boundaries)
        CMASK = PixA_convd == 0
        CMASK[:HW, :] = False
        CMASK[-HW:, :] = False
        CMASK[:, :HW] = False
        CMASK[:, -HW:] = False
        
        # randomly select box-centers
        if RandomSeed is not None: np.random.seed(RandomSeed)
        RIDX = np.random.choice(np.arange(np.sum(CMASK)), int(1.5*NUM_SAMP))   # redundant
        R, C = np.where(CMASK)
        RC_pit = np.array([R[RIDX], C[RIDX]]).T
        
        # eliminate the overlapping cases
        tol = np.sqrt(2*HW**2)
        Tree = cKDTree(RC_pit+0.5)
        IDX = Tree.query(RC_pit+0.5, k=2, distance_upper_bound=tol)[1][:, 1]
        Avmask = IDX == RC_pit.shape[0]
        RC_pit = RC_pit[Avmask][:NUM_SAMP]
        
        return RC_pit
