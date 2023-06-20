import numpy as np
from scipy.optimize import minimize
# version: Mar 20, 2023

class ZeroPoint_Calculator:
    @staticmethod
    def ZPC(MAG_MOBJ, MAGERR_MOBJ, MAG_MREF, MAGERR_MREF, MINFRAC_FIT=0.5):

        """
        # A simple calculator based minimization of chi-square with iteration of rejections
        # Quick NOTE: MAG_MOBJ + zp ~ MAG_MREF
        #
        """
        
        mag_res = MAG_MREF - MAG_MOBJ
        mag_res_err2 = MAGERR_MREF ** 2 + MAGERR_MOBJ ** 2
        MIN_USED = int(MINFRAC_FIT * len(mag_res))
        
        zp, clipping_steps = 25.0, []
        idxrec = np.arange(len(mag_res))
        while len(mag_res) >= 3:
            
            # ** chi-square minimization procedure
            fchi2 = lambda zp: np.sum((zp - mag_res) ** 2 / mag_res_err2)
            minchi2 = minimize(fchi2, zp, method='Nelder-Mead')
            
            zp = minchi2.x[0]               # zp is the optimal value which minimizes chi2            
            dof = len(mag_res) - 2          # degree of freedom, N_samples - N_fitting_var - 1.
            rchi2  = minchi2.fun / dof      # derive reduced chi-square 

            # ** estimate zero-point error
            zp_err = np.sqrt(np.average((zp - mag_res)**2, weights=1/mag_res_err2) + np.mean(mag_res_err2))
            clipping_steps.append([zp, zp_err, rchi2, idxrec])

            # ** simple outlier rejection
            for t in range(max([1, int(len(mag_res)/50)])):
                rej_idx = np.argmax(np.absolute(mag_res - zp))
                mag_res = np.delete(mag_res, rej_idx)
                mag_res_err2 = np.delete(mag_res_err2, rej_idx)
                idxrec = np.delete(idxrec, rej_idx)
        
        # * Select best-fit zeropoint based on minimum reduced-chi2
        best_idx = np.nanargmin([step[2] for step in clipping_steps])

        # * Modify the best-fit index (increase the number of sources for fitting until MIN_USED)
        if len(clipping_steps[best_idx][3]) < MIN_USED:
            while len(clipping_steps[best_idx][3]) < MIN_USED and best_idx > 0: best_idx -= 1
        
        zp, zp_err = clipping_steps[best_idx][:2]
        zp, zp_err = round(zp, 6), round(zp_err, 6)

        return zp, zp_err
