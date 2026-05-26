import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import pandas as pd
import os
from pathlib import Path
from astropy.table import Table
from astropy.wcs import WCS
from PIL import Image
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.table import Table, Column
import funcs

#funcs.match_manga_to_radio()

path_data_final = "../manga_rmorph_data/manga_with_vlass_lotss_first.fits"
data_final = fits.open(path_data_final)[1].data




def add_spectral_slopes(final_table):
    
    # 1. Define frequencies (in MHz)
    nu_lotss = 144.0
    nu_first = 1400.0
    nu_vlass = 3000.0

    # 2. Pre-calculate the log denominators for slopes
    log_den_144_1400 = np.log10(nu_first / nu_lotss)
    log_den_1400_3000 = np.log10(nu_vlass / nu_first)
    log_den_144_3000 = np.log10(nu_vlass / nu_lotss)
    
    # Pre-calculate natural log denominators for error propagation
    ln_den_144_1400 = np.log(nu_first / nu_lotss)
    ln_den_1400_3000 = np.log(nu_vlass / nu_first)
    ln_den_144_3000 = np.log(nu_vlass / nu_lotss)

    # 3. Column name definitions
    f_lotss, e_lotss = "lotss_Total_flux", "lotss_E_Total_flux"
    f_first, e_first = "first_Fint", "first_Rms"
    f_vlass, e_vlass = "vlass_Total_flux", "vlass_E_Total_flux"
    
    # Create masks to check if sources are matched and fluxes are strictly positive (to avoid log(<=0) errors)
    # Assuming there's an 'is_first_match' column; if not, we check if the FIRST flux is not NaN
    is_first = final_table.get('is_first_match', final_table[f_first].notna())
    
    valid_lotss = final_table['is_lotss_match'] & (final_table[f_lotss] > 0)
    valid_first = is_first & (final_table[f_first] > 0)
    valid_vlass = final_table['is_vlass_match'] & (final_table[f_vlass] > 0)

    # 4. Helper function to compute vectorized alpha and error
    def calc_alpha_and_err(f1_col, f2_col, e1_col, e2_col, log_den, ln_den, mask):
        # Initialize arrays with NaNs
        alpha = np.full(len(final_table), np.nan)
        err = np.full(len(final_table), np.nan)
        
        # Only compute where the mask is True
        f1 = final_table.loc[mask, f1_col].values
        f2 = final_table.loc[mask, f2_col].values
        e1 = final_table.loc[mask, e1_col].values
        e2 = final_table.loc[mask, e2_col].values
        
        # Calculate spectral index
        alpha[mask] = (np.log10(f2) - np.log10(f1)) / log_den
        
        # Calculate error analytically
        err[mask] = (1.0 / np.abs(ln_den)) * np.sqrt((e1 / f1)**2 + (e2 / f2)**2)
        
        return alpha, err

    # 5. Apply the helper function for all three pairs
    
    # --- 144 to 1400 MHz ---
    mask_144_1400 = valid_lotss & valid_first
    a, e = calc_alpha_and_err(f_lotss, f_first, e_lotss, e_first, log_den_144_1400, ln_den_144_1400, mask_144_1400)
    final_table['alpha_144_1400'] = a
    final_table['err_alpha_144_1400'] = e

    # --- 1400 to 3000 MHz ---
    mask_1400_3000 = valid_first & valid_vlass
    a, e = calc_alpha_and_err(f_first, f_vlass, e_first, e_vlass, log_den_1400_3000, ln_den_1400_3000, mask_1400_3000)
    final_table['alpha_1400_3000'] = a
    final_table['err_alpha_1400_3000'] = e

    # --- 144 to 3000 MHz ---
    mask_144_3000 = valid_lotss & valid_vlass
    a, e = calc_alpha_and_err(f_lotss, f_vlass, e_lotss, e_vlass, log_den_144_3000, ln_den_144_3000, mask_144_3000)
    final_table['alpha_144_3000'] = a
    final_table['err_alpha_144_3000'] = e

    return final_table
