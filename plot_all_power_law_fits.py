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
from astropy.table import Table, Column, hstack, MaskedColumn
import funcs
from matplotlib.lines import Line2D
from matplotlib.colors import ListedColormap


#data = funcs.match_manga_to_radio()

path_data_final = "../manga_rmorph_data/manga_with_vlass_lotss_first_with_power_law.fits"
data_final = fits.open(path_data_final)[1].data

def plot_all_power_law_fits(final_table):
    """
    Generates and saves log-log plots for all sources with valid power law fits.
    Saves them to ../manga_rmorph_data/power_law_fits_plots/
    """
    # 1. Define output directory and create it if it doesn't exist
    out_dir = "../manga_rmorph_data/power_law_fits_plots/"
    os.makedirs(out_dir, exist_ok=True)

    # 2. Filter for rows that actually have a computed fit
    # We can just check where the slope is not NaN
    valid_fits_mask = ~np.isnan(final_table['power_law_slope'])
    fits_table = final_table[valid_fits_mask]

    print(f"Found {len(fits_table)} valid fits to plot.")

    # 3. Define Frequencies (GHz)
    freqs = np.array([0.144, 1.4, 3.0])
    # Smooth frequency array for plotting the fit line smoothly
    freqs_fit = np.geomspace(0.1, 4.0, 100)

    # 4. Loop through each valid source and plot
    for row in fits_table:
        # Extract MaNGA ID and format the filename (replace hyphens with underscores)
        # We use str().strip() to cleanly handle Astropy string columns
        raw_mangaid = str(row['mangaid']).strip()
        safe_mangaid = raw_mangaid.replace('-', '_')
        filename = os.path.join(out_dir, f"{safe_mangaid}.png")

        # Extract Fluxes
        F = np.array([
            row['lotss_Total_flux'],
            row['first_Fint'],
            row['vlass_Total_flux']
        ])
        
        # Extract Errors (Note: matching 'first_Rms' from your ColDefs)
        F_err = np.array([
            row['lotss_E_Total_flux'],
            row['first_Rms'],
            row['vlass_E_Total_flux']
        ])

        # Extract Fit Parameters
        C = np.exp(row['power_law_intercept'])
        alpha = row['power_law_slope']
        chi2 = row['power_law_chi2']

        # Compute the fitted line F = C * nu^alpha
        F_fit = C * (freqs_fit ** alpha)

        # --- Plotting ---
        plt.figure(figsize=(8, 6))

        # Plot data points with error bars
        plt.errorbar(freqs, F, yerr=F_err, fmt='o', color='black', 
                     capsize=5, markersize=8, label='Observed Fluxes')

        # Plot the fit line
        plt.plot(freqs_fit, F_fit, color='red', linestyle='--', 
                 label=f'Fit: $\\alpha$ = {alpha:.2f}')

        # Formatting axes
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel('Frequency (GHz)', fontsize=12)
        plt.ylabel('Flux (mJy)', fontsize=12) 
        
        # Ensure x-ticks show up as standard numbers instead of 10^0
        plt.xticks(freqs, labels=[str(f) for f in freqs])
        
        # Annotations (Title and text box for Chi^2)
        plt.title(f'MaNGA ID: {raw_mangaid}', fontsize=14, fontweight='bold')
        
        # Place a text box with the Chi-squared value
        # transform=plt.gca().transAxes uses relative axes coordinates (0 to 1)
        plt.text(0.05, 0.05, f'$\chi^2$ = {chi2:.2f}', 
                 transform=plt.gca().transAxes, fontsize=12,
                 bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'))

        plt.legend(loc='upper right')
        plt.grid(True, which="both", ls="--", alpha=0.5)

        # Save and close
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()


plot_all_power_law_fits(data_final)        