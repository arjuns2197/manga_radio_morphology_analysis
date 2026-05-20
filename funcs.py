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
from astropy.cosmology import FlatLambdaCDM
import matplotlib.patheffects as pe
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse
import matplotlib.patheffects as pe
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import ListedColormap
from matplotlib.colors import SymLogNorm
import statistics
import time

# Paths
map_dir = Path("../manga_rmorph_data/manga_maps")
optical_dir = Path("../manga_rmorph_data/Optical")
radio_dir = Path("../manga_rmorph_data/Radio")
output_dir = Path("../manga_rmorph_data/plots_3m")
output_dir.mkdir(exist_ok=True, parents=True)
drpall_path = "../manga_rmorph_data/drpall-v3_1_1.fits"


def process_radio_catalog(catalog_table, prefix, c_manga, max_sep, n_manga, ra_col='RA', dec_col='DEC', coord_unit=u.deg):
    """
    Helper function to prefix columns, match coordinates, and return an aligned masked table.
    Converts RA and Dec to decimal degrees.
    """
    # Prefix columns to avoid name collisions
    for col in catalog_table.colnames:
        catalog_table.rename_column(col, f"{prefix}_{col}")

    # Create SkyCoord using the provided units (e.g., hourangle for FIRST)
    c_radio = SkyCoord(
        catalog_table[f'{prefix}_{ra_col}'],
        catalog_table[f'{prefix}_{dec_col}'],
        unit=coord_unit
    )

    # ==========================================
    # NEW: Force the columns to be decimal degrees 
    # ==========================================
    catalog_table[f'{prefix}_{ra_col}'] = c_radio.ra.deg
    catalog_table[f'{prefix}_{dec_col}'] = c_radio.dec.deg

    # Find the closest match
    idx_radio, d2d, _ = c_manga.match_to_catalog_sky(c_radio)
    valid_match = d2d < max_sep

    # Create empty masked table
    aligned_table = Table(masked=True)

    for col in catalog_table.colnames:
        dummy_data = np.zeros(n_manga, dtype=catalog_table[col].dtype)
        aligned_table[col] = MaskedColumn(dummy_data, mask=np.ones(n_manga, dtype=bool))

    # Fill in valid matches
    aligned_table[valid_match] = catalog_table[idx_radio[valid_match]]

    # Unmask the filled rows
    for col in aligned_table.colnames:
        aligned_table[col].mask[valid_match] = False

    aligned_table[f"is_{prefix}_match"] = valid_match

    return aligned_table



def add_power_law_fits(final_table):
    """
    Fits a power law to LoTSS, FIRST, and VLASS fluxes, computes chi-squared,
    and adds the results as new columns to the final_table directly.
    """
    # 1. Initialize new columns with NaNs for all rows
    final_table['power_law_intercept'] = np.nan
    final_table['power_law_slope'] = np.nan
    final_table['power_law_chi2'] = np.nan

    # 2. Safely check for positive fluxes (avoiding Astropy masked array warnings)
    lotss_f = np.asarray(final_table['lotss_Total_flux'], dtype=float)
    first_f = np.asarray(final_table['first_Fint'], dtype=float)
    vlass_f = np.asarray(final_table['vlass_Total_flux'], dtype=float)

    # Use np.errstate to suppress warnings if there are NaNs in unmatched rows
    with np.errstate(invalid='ignore'):
        valid_flux_mask = (lotss_f > 0) & (first_f > 0) & (vlass_f > 0)

    # Update mask: must have matches AND positive flux values
    match_mask = (final_table['is_vlass_match'] & 
                  final_table['is_first_match'] & 
                  final_table['is_lotss_match'] & 
                  valid_flux_mask)
    
    matches = final_table[match_mask]
    
    # If no valid matches exist, return the table with the NaN columns
    if len(matches) == 0:
        return final_table

    # 3. Extract Fluxes and Errors as PURE NumPy arrays 
    # (This strips Astropy's MaskedArray wrapper which caused the broadcasting crash)
    F = np.column_stack((
        np.asarray(matches['lotss_Total_flux'], dtype=float),
        np.asarray(matches['first_Fint'], dtype=float),
        np.asarray(matches['vlass_Total_flux'], dtype=float)
    ))
    
    F_err = np.column_stack((
        np.asarray(matches['lotss_E_Total_flux'], dtype=float),
        np.asarray(matches['first_Rms'], dtype=float),
        np.asarray(matches['vlass_E_Total_flux'], dtype=float)
    ))

    # --- FIT THE POWER LAW ---
    Y = np.log(F) 
    sigma2_log = (F_err / F)**2 
    W = 1.0 / sigma2_log 
    
    freqs = np.array([0.144, 1.4, 3.0])
    A = np.column_stack((np.ones(3), np.log(freqs)))

    At_W_A = np.matmul(A.T, W[..., None] * A)
    At_W_Y = np.matmul(W * Y, A)[..., None]
    
    thetas = np.linalg.solve(At_W_A, At_W_Y).squeeze(-1)

    # Add fit parameters to the matched rows
    final_table['power_law_intercept'][match_mask] = thetas[:, 0]
    final_table['power_law_slope'][match_mask] = thetas[:, 1]

    # --- COMPUTE CHI-SQUARED ---
    C = np.exp(thetas[:, 0]) 
    alpha = thetas[:, 1]
    
    F_fit = C[:, None] * (freqs[None, :] ** alpha[:, None])
    chi2 = np.sum(((F - F_fit) / F_err)**2, axis=1)

    # Add chi2 to the matched rows
    final_table['power_law_chi2'][match_mask] = chi2

    return final_table



def add_all_central_ionization_types(final_table):
    """
    Calculates and adds SII, NII, and the combined dominant central 
    ionisation types for the central 5x5 pixels in a single pass.
    """
    sii_results = []
    nii_results = []
    combined_results = []

    for plateifu in final_table['plateifu']:
        plate, ifu = plateifu.split("-")
        
        # 1. Get both classifications
        sii_type = categorise_central_ionization_type_SII(plate, ifu)
        nii_type = categorise_central_ionization_type_NII(plate, ifu)
        
        sii_results.append(sii_type)
        nii_results.append(nii_type)
 

        # 2. Determine combined classification based on your logic
        combined_type = 'Little Ionisation' # Default fallback
        
        if nii_type == 'SF':
            if sii_type in ['SF', 'Optical AGN', 'LI(N)ER']:
                combined_type = 'SF'
                
        elif nii_type == 'Optical AGN':
            if sii_type == 'SF':
                combined_type = 'SF'
            elif sii_type == 'Optical AGN':
                combined_type = 'Optical AGN'
            elif sii_type == 'LI(N)ER':
                combined_type = 'LI(N)ER'
                
        combined_results.append(combined_type)

    # 3. Assign all lists to the table columns at once (fixes the == bug and is much faster)
    final_table['central_ionisation_type_SII'] = sii_results
    final_table['central_ionisation_type_NII'] = nii_results
    final_table['central_ionisation_type'] = combined_results

    return final_table



def match_manga_to_radio():
    max_sep = 3.0 * u.arcsec

    # =====================================================
    # Load MaNGA
    # =====================================================
    path_manga = "../manga_rmorph_data/manga_radio_vac_1.0.fits"
    manga = Table.read(path_manga)
    n_manga = len(manga)

    # Pass units cleanly here as well just to be safe
    c_manga = SkyCoord(manga['ra'], manga['dec'], unit=u.deg)

    # =====================================================
    # Process VLASS
    # =====================================================
    print("Processing VLASS...")
    path_vlass = "../manga_rmorph_data/CIRADA_VLASS1QLv3.1_table1_components.csv"
    vlass = Table.from_pandas(pd.read_csv(path_vlass))
    vlass_aligned = process_radio_catalog(vlass, "vlass", c_manga, max_sep, n_manga, ra_col='RA', dec_col='DEC')

    # =====================================================
    # Process LoTSS
    # =====================================================
    print("Processing LoTSS...")
    path_lotss = "../manga_rmorph_data/LoTSS_DR3_v1.0.srl.fits"
    lotss = Table.read(path_lotss)
    lotss_aligned = process_radio_catalog(lotss, "lotss", c_manga, max_sep, n_manga, ra_col='RA', dec_col='DEC')

    # =====================================================
    # Process FIRST
    # =====================================================
    print("Processing FIRST...")
    path_first = "../manga_rmorph_data/first_catalog.fit"
    first = Table.read(path_first)
    
    # Use (u.hourangle, u.deg) for standard VizieR FIRST catalogs
    first_aligned = process_radio_catalog(
        first, 
        "first",
        c_manga, 
        max_sep, 
        n_manga, 
        ra_col='RAJ2000', 
        dec_col='DEJ2000',
        coord_unit=(u.hourangle, u.deg) # <--- Added this parameter
    )

    # =====================================================
    # Combine catalogs
    # =====================================================
    print("Stacking catalogs...")
    final_table = hstack([manga, vlass_aligned, lotss_aligned, first_aligned])

    print("Rows:", len(final_table))
    print("Columns:", len(final_table.colnames))

    # =====================================================
    # Add Power Law Fits
    # =====================================================
    
    final_table = add_power_law_fits(final_table)

    # =====================================================
    # Add Dominant Central ionisation Type
    # =====================================================

    final_table = add_all_central_ionization_types(final_table)

    # =====================================================
    # Save output
    # =====================================================
    output_path = "../manga_rmorph_data/manga_with_vlass_lotss_first.fits"
    final_table.write(output_path, overwrite=True)

    print("Saved catalog:", output_path)

    return final_table


def get_target_info(data_final):

    manga_targets = []

    for i in range(len(data_final)):

        if (data_final['is_vlass_match'][i]) or (data_final['is_lotss_match'][i]) or (data_final['radio_class'][i] != 0):

            plate, ifu = data_final['plateifu'][i].split("-")
            manga_targets.append({"mangaid" : data_final['mangaid'][i],"plate" : plate, "ifu" : ifu,"fov" : 1,
            "scale_bar_arcsec": 15.0, "z" : data_final['nsa_redshift'][i], "ra_manga" : data_final['ra'][i],
            "dec_manga" : data_final['dec'][i], "log_mass": data_final['log_mass'][i], "log_ssfr" : data_final['log_ssfr'][i],
            "log_sfr_sf" : data_final['log_sfr_sf'][i], "is_detected_agn" : data_final['is_detected_agn'][i],

            "is_vlass_match" : data_final['is_vlass_match'][i],
            "vlass_RA" : data_final['vlass_RA'][i], "vlass_DEC" : data_final['vlass_DEC'][i], "vlass_Maj" : data_final['vlass_Maj'][i],
            "vlass_Min" : data_final['vlass_Min'][i], "vlass_PA" : data_final['vlass_PA'][i],
            "vlass_Fint" : data_final['vlass_Total_flux'][i], "vlass_freq" : 3e9,

            "is_first_match" : data_final['is_first_match'][i],
            "first_RA" : data_final['first_RAJ2000'][i], "first_DEC" : data_final['first_DEJ2000'][i],
            "first_Maj" : data_final['first_MajAxis'][i], "first_Min" : data_final['first_MinAxis'][i], 
            "first_PA" : data_final['first_PA'][i], "first_Fint" : data_final['first_Fint'][i],
            "first_freq" : 1.4e9,
        
            "is_lotss_match" : data_final['is_lotss_match'][i],
            "lotss_RA" : data_final['lotss_RA'][i], "lotss_DEC" : data_final['lotss_DEC'][i],
            "lotss_Maj" : data_final['lotss_Maj'][i],
            "lotss_Min" : data_final['lotss_Min'][i], "lotss_PA" : data_final['lotss_PA'][i],
            "lotss_Fint" : data_final['lotss_Total_flux'][i], "lotss_freq" : 144e6,

            "power_law_intercept" : data_final['power_law_intercept'][i],
            "power_law_slope" : data_final['power_law_slope'][i],
            "power_law_chi2" : data_final['power_law_chi2'][i]})

    
    return manga_targets


def get_spectral_slope(target_info):
    # 1. Define the preferred order of survey comparisons
    preferred_pairs = [
        ('lotss', 'vlass'),
        ('lotss', 'first'),
        ('first', 'vlass')
    ]
    
    # 2. Loop through the pairs and return the first one that matches both
    for s1, s2 in preferred_pairs:
        # Using .get() is safer in case a key is missing from the dictionary
        if target_info.get(f'is_{s1}_match') and target_info.get(f'is_{s2}_match'):
            
            fint_ratio = target_info[f'{s1}_Fint'] / target_info[f'{s2}_Fint']
            freq_ratio = target_info[f'{s1}_freq'] / target_info[f'{s2}_freq']
            
            return np.log10(fint_ratio) / np.log10(freq_ratio)
            
    # 3. If no pairs match, return None
    return None


def find_map_file(directory,plate, ifu):
    """Find maps file matching plate-ifu"""

    directory = directory/plate/ifu
    pattern = f"manga-{plate}-{ifu}*"
    matches = list(directory.glob(pattern))
    
    for f in directory.iterdir():
        print(f.name)

    print(directory)

    return matches[0] if matches else None


def find_cutout_file(directory, mangaid, survey_pattern, plate = None, ifu = None):
    """Find cutout file matching manga ID and survey pattern."""
    mangaid_clean = mangaid.replace("-", "_")
    pattern = f"{mangaid_clean}_{survey_pattern}*"
    matches = list(directory.glob(pattern))


    return matches[0] if matches else None


def load_fits_data_map(filepath, property_hdu_ext = None, property_channel = None):
    """Load FITS file and return data and WCS."""

    try:
        with fits.open(filepath) as hdul:

            wcs_map = WCS(hdul[3].header)
            map_shape = hdul[3].shape
            property_map = hdul[property_hdu_ext].data

            if property_channel != None:
                property_map = property_map[property_channel,:,:]

            return property_map,wcs_map,map_shape
            
        
    except:
        return None, None, None
    


def load_fits_data(filepath):
    """Load FITS file and return data and WCS."""
    try:
        with fits.open(filepath) as hdul:
            data = hdul[0].data
            header = hdul[0].header

            # Handle 3D/4D data
            while data.ndim > 2:
                data = data[0]
            
            # Check for valid data
            if data is None or data.size == 0:
                return None, None
            
            valid_data = data[np.isfinite(data)]
            if len(valid_data) == 0 or np.all(valid_data == 0):
                return None, None
            
            wcs = WCS(header, naxis=2)
            return data, wcs
    except:
        return None, None
    

def reproject_to_reference(data, source_wcs, target_wcs, output_shape):
    """Reproject data to target WCS."""
    try:
        from reproject import reproject_interp
        reprojected, _ = reproject_interp((data, source_wcs), target_wcs, shape_out=output_shape)
        return reprojected
    except ImportError:
        print("Warning: reproject library not available")
        return None
    

def load_legacy_jpg_and_wcs(jpg_file, fits_file):
    """Load SDSS Legacy JPEG and WCS from FITS."""
    try:
        img = Image.open(jpg_file)
        rgb_data = np.array(img).astype(float) / 255.0
        rgb_data = np.flipud(rgb_data)  # JPEG origin is 'upper', FITS is 'lower'
        
        with fits.open(fits_file) as hdul:
            wcs = WCS(hdul[0].header, naxis=2)
        
        return rgb_data, wcs
    except:
        return None, None
    

def add_source_ellipse(target_info,survey_config,survey,map_wcs,pixscale,ax):

    coord = SkyCoord(ra=target_info[survey + '_RA'], dec=target_info[survey + '_DEC'], unit=u.deg)
    x, y = map_wcs.world_to_pixel(coord)

    center = (float(x), float(y)) # Ensure they are scalars
    width_vlass = target_info[survey + '_Maj'] / pixscale
    height_vlass = target_info[survey + '_Min'] / pixscale

    # 2. Fix typo and adjust PA for Matplotlib's coordinate system

    if survey == 'vlass':

        angle = -(90 + target_info[survey + '_PA']) 
    
    else:
        angle = 90 + target_info[survey + '_PA']

    if survey == 'lotss':
        survey = 'lofar'

    # 3. Create the patch (changed lw to an int)
    ellipse = Ellipse(center, width_vlass, height_vlass, angle=angle, 
                            edgecolor=survey_config[survey]['color'],
                            fc='None', lw=5)

    ax.add_patch(ellipse)


def prepare_and_plot_map(target_info, fig, subplot_number, property_hdu_ext=None, property_channel=None,
                         is_gline_sigma=None, draw_scalebar=False):

    ### METADATA ###
    plate = target_info['plate']
    ifu = target_info['ifu']
    fov = target_info['fov']
    mangaid = target_info['mangaid']
    z = target_info['z']

    survey_order = ["lofar", "first", "vlass"]

    survey_config = {
        "lofar": {"color": "#FE4203", "label": "LoTSS 144 MHz", "linewidth": 2, "alpha": 0.8},    
        "first": {"color": "#A671F7", "label": "FIRST 1.4 GHz", "linewidth": 2, "alpha": 0.8},
        "vlass": {"color": "#FFCC00", "label": "VLASS 3 GHz", "linewidth": 2, "alpha": 0.8},
    }

    sigmas_multiples = np.array([3,5,10,20,40])
    sigmas_multiples_lofar = np.array([5,10,20,40])

    sigma_lofar = 83e-6
    levels_lofar = sigma_lofar*sigmas_multiples_lofar
    sigma_first = 0.15e-3
    levels_first = sigma_first*sigmas_multiples
    sigma_vlass = 70e-6
    levels_vlass = sigma_vlass*sigmas_multiples

    if property_hdu_ext != 37:
        cmap = 'winter'
    else:
        cmap = 'berlin'

    contour_outline_color = "black"  

    scale_bar_arcsec = 10.0  
    scale_bar_margin_x = 0.05  
    scale_bar_margin_y = 0.89 
    scale_bar_linewidth = 4  
    scale_bar_fontsize = 15  
    scale_bar_text_offset = 0.02  

    ### PLOT MAP###
    map_file_dir = find_map_file(map_dir, plate, ifu)
    
    # ------------------------------------------------------------------
    # DATA LOADING AND S/N CUTS
    # ------------------------------------------------------------------
    with np.errstate(invalid='ignore', divide='ignore'):
        map_data, map_wcs, map_shape = load_fits_data_map(map_file_dir, property_hdu_ext, property_channel)

        if map_data is None:
            return None
        
        ivar, _, __ = load_fits_data_map(map_file_dir, property_hdu_ext+1, property_channel)
        bitmask, _, __ = load_fits_data_map(map_file_dir, property_hdu_ext+2, property_channel)
        
        # We use np.abs() here so that negative velocities aren't masked out due to a "negative" SNR
        snr = np.abs(map_data) * np.sqrt(ivar)
        
        map_data = np.where((bitmask == 0) & (snr > 2), map_data, np.nan)

    if map_data is None:
        print("Skipping; Galaxy MAP not found")
        return None

    if property_hdu_ext == 40: 
        emline_instsigma_hdu_ext = 43
        instsigma_map, _, __ = load_fits_data_map(map_file_dir,emline_instsigma_hdu_ext,property_channel)

        if instsigma_map is None:
            print("Skipping; Galaxy MAP not found")
            return None
        
        variance_diff = map_data**2 - instsigma_map**2
        map_data = np.sqrt(np.clip(variance_diff, 0, None))

    if map_wcs is None:
        print(f"  Skipping: Galaxy (WCS) not found")
        return None  

    map_height, map_width = map_shape
    
    if fov is not None:
        center_x, center_y = map_width / 2.0, map_height / 2.0
        try:
            pixscale = np.abs(map_wcs.pixel_scale_matrix[0, 0]) * 3600.0
        except:
            pixscale = np.abs(map_wcs.wcs.cdelt[0]) * 3600.0
        
        fov_pixels = int(fov * 60.0 / pixscale)
        half_size = fov_pixels / 2.0
        x_min = max(0, int(center_x - half_size))
        x_max = min(map_width, int(center_x + half_size))
        y_min = max(0, int(center_y - half_size))
        y_max = min(map_height, int(center_y + half_size))
        
        map_data = map_data[y_min:y_max, x_min:x_max]
        map_wcs.wcs.crpix[0] -= x_min
        map_wcs.wcs.crpix[1] -= y_min
    
    map_height, map_width = map_data.shape 
    ax = fig.add_subplot(subplot_number, projection=map_wcs)

    finite_map = map_data[np.isfinite(map_data)]
    nonzero_vals = finite_map[finite_map != 0]
    
    if len(nonzero_vals) == 0:
        print("Skipping; Map empty after masking.")
        ax.axis('off')
        return None

    ratio = np.nanmax(finite_map) / np.nanmin(nonzero_vals)
    
    # ==========================================
    # Map Imshow & Color Limits
    # ==========================================
    if property_hdu_ext != 37:
        if ratio > 10:
            log_data = np.log10(map_data)
            im = ax.imshow(log_data, cmap=cmap, origin='lower')
            log_scale = True
            
            valid_log_data = log_data[np.isfinite(log_data)]
            
            if len(valid_log_data) > 0:
                vmin = np.percentile(valid_log_data, 3)  
                vmax = np.percentile(valid_log_data, 99) 
                
                if property_hdu_ext == 30: 
                    im.set_clim(vmin, vmax)
                elif property_hdu_ext == 40: 
                    im.set_clim(max(vmin, 1.0), vmax)
        else:
            im = ax.imshow(map_data, cmap=cmap, origin='lower')
            log_scale = False
    else:   
        if len(finite_map) > 0:
            v_limit = np.percentile(np.abs(finite_map), 98)
            v_limit = max(v_limit, 50) 
        else:
            v_limit = 350 
            
        lin_thresh = min(150, v_limit * 0.5)
        norm = SymLogNorm(linthresh=lin_thresh, vmin=-v_limit, vmax=v_limit, base=10)
        
        im = ax.imshow(map_data, cmap=cmap, origin='lower', norm=norm)
        log_scale = False
        ax.v_limit = v_limit 

    # ==========================================
    # Horizontal Inset Colorbar
    # ==========================================
    cbaxes = ax.inset_axes([0.12, 0.08, 0.76, 0.04]) 
    
    cbar = fig.colorbar(im, cax=cbaxes, orientation='horizontal')
    cbaxes.xaxis.set_ticks_position('top')
    cbaxes.xaxis.set_label_position('bottom')
    
    # ==========================================
    # Set Labels
    # ==========================================
    if property_hdu_ext == 30:
        if property_channel == 23:
            cbar_label = r"$\log_{\rm 10} \left(H\alpha \right)$ Flux" if log_scale else r"$H\alpha$ Flux"
        if property_channel == 16:
            cbar_label = r"$\log_{\rm 10} \left(\rm O[III] \right)$ Flux" if log_scale else r"$\rm O[III]$ Flux"
    elif property_hdu_ext == 37:
        cbar_label = "LoS O[III] Velocity (km/s)"
    elif property_hdu_ext == 40:
        cbar_label = r"$\log \sigma_{\rm [OIII]}$ (km/s)" if log_scale else r"$\sigma_{\rm [OIII]}$ (km/s)"
    else:
        cbar_label = ""
        
    cbar.set_label(cbar_label, fontsize=20, color='black', weight='bold', labelpad=2)

    # ==========================================
    # Custom Ticks Formatting
    # ==========================================
    if property_hdu_ext == 37: 
        # Velocity maps
        v_limit = getattr(ax, 'v_limit', 350)
        cbar.ax.set_xlim(-v_limit, v_limit) # Force limits
        cbar.set_ticks([-v_limit, 0, v_limit])
        cbar.set_ticklabels([f"-{int(v_limit)}", "0", f"{int(v_limit)}"])
        
    else:
        # Flux and Sigma maps
        vmin, vmax = im.get_clim()
        vcenter = (vmin + vmax) / 2.0
        
        cbar.ax.set_xlim(vmin, vmax) # Prevent edge clipping
        cbar.set_ticks([vmin, vcenter, vmax]) # Force exactly 3 ticks
        
        def fmt(val):
            rounded = round(val, 1)
            return f"{int(rounded)}" if rounded == int(rounded) else f"{rounded:.1f}"
            
        cbar.set_ticklabels([fmt(vmin), fmt(vcenter), fmt(vmax)])

    cbar.ax.tick_params(labelsize=20, colors='black', length=3, pad=1)
    
    cb_outline = [pe.withStroke(linewidth=1, foreground="black")]
    cbar.ax.xaxis.label.set_path_effects(cb_outline)
    for tick_label in cbar.ax.get_xticklabels():
        tick_label.set_path_effects(cb_outline)

    # ==========================================
    # Contours
    # ==========================================
    for survey_name in survey_order:
        if survey_name not in survey_config:
            continue
        survey_cfg = survey_config[survey_name]
        radio_file = find_cutout_file(radio_dir, mangaid, survey_name)
        if radio_file is None:
            continue
        
        radio_data, radio_wcs = load_fits_data(radio_file)
        if radio_data is None:
            continue
        
        reprojected = reproject_to_reference(radio_data, radio_wcs, map_wcs, map_shape)
        if reprojected is None:
            continue
        
        reprojected_peak = np.nanmax(reprojected)
        if reprojected_peak <= 0:
            continue
        
        if survey_name == "lofar": levels = levels_lofar
        elif survey_name == "first": levels = levels_first
        elif survey_name == "vlass": levels = levels_vlass

        outline_color = target_info.get("contour_outline_color", contour_outline_color)
        
        ax.contour(reprojected, levels=levels, colors=outline_color,
                  linewidths=survey_cfg["linewidth"] + 0.5, alpha=0.8,
                  transform=ax.get_transform('pixel'))
        
        ax.contour(reprojected, levels=levels, colors=survey_cfg["color"],
                  linewidths=survey_cfg["linewidth"], alpha=survey_cfg["alpha"],
                  transform=ax.get_transform('pixel'))

    if target_info['is_lotss_match'] == True:
        add_source_ellipse(target_info,survey_config,'lotss', map_wcs, pixscale, ax)
    if target_info['is_first_match'] == True:
        add_source_ellipse(target_info,survey_config,'first', map_wcs, pixscale, ax)
    if target_info['is_vlass_match'] == True:
        add_source_ellipse(target_info,survey_config,'vlass', map_wcs, pixscale, ax)
    
    # ==========================================
    # Scale Bar (Bottom Left)
    # ==========================================
    if draw_scalebar and z is not None and z > 0:
        sb_arcsec = target_info.get("scale_bar_arcsec", scale_bar_arcsec)
        sb_margin_x = target_info.get("scale_bar_margin_x", scale_bar_margin_x)
        sb_margin_y = target_info.get("scale_bar_margin_y", scale_bar_margin_y)
        sb_linewidth = target_info.get("scale_bar_linewidth", scale_bar_linewidth)
        sb_fontsize = target_info.get("scale_bar_fontsize", scale_bar_fontsize)
        sb_text_offset = target_info.get("scale_bar_text_offset", scale_bar_text_offset)

        cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
        d_a = cosmo.angular_diameter_distance(z)
        scale_kpc = (sb_arcsec / 3600.0 * np.pi / 180.0) * d_a.value * 1000.0
        
        try:
            pixscale = np.abs(map_wcs.pixel_scale_matrix[0, 0]) * 3600.0
        except:
            pixscale = np.abs(map_wcs.wcs.cdelt[0]) * 3600.0
        
        scale_pixels = sb_arcsec / pixscale
        scale_rel = scale_pixels / map_width
        
        bar_x_start = sb_margin_x
        bar_x_end = bar_x_start + scale_rel
        bar_y = sb_margin_y
        
        outline = [pe.withStroke(linewidth=1.5, foreground="white")]

        ax.plot([bar_x_start, bar_x_end], [bar_y, bar_y], 'k-',
               linewidth=sb_linewidth, transform=ax.transAxes, path_effects=outline)
        
        bar_x_center = (bar_x_start + bar_x_end) / 2
        ax.text(bar_x_center, bar_y + sb_text_offset,
               f"{sb_arcsec:.0f}\"",
               transform=ax.transAxes, fontsize=sb_fontsize,
               color='k', ha='center', va='bottom', weight='bold', path_effects=outline)
        ax.text(bar_x_center, bar_y - sb_text_offset,
               f"{scale_kpc:.1f} kpc",
               transform=ax.transAxes, fontsize=sb_fontsize,
               color='k', ha='center', va='top', weight='bold', path_effects=outline)
    
    ax.coords[0].set_ticks_visible(False)
    ax.coords[0].set_ticklabel_visible(False)
    ax.coords[1].set_ticks_visible(False)
    ax.coords[1].set_ticklabel_visible(False)
    ax.coords[0].set_axislabel('')
    ax.coords[1].set_axislabel('')
    ax.coords.grid(color='none', draw_grid=False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    
    return im


def SII_BPT_diagnostic(map_log_SII_Ha, map_log_OIII_Hb, plate, ifu):

    ew_hdu_ext = 33
    Ha_channel = 23


    map_file_dir = find_map_file(map_dir, plate, ifu)

    # Load EW map. We SKIP applying the bitmask here because EW masks often flag 
    # the continuum inappropriately. The flux S/N cuts above already handle the bad spaxels!
    map_Ha_ew, map_wcs, map_shape = load_fits_data_map(map_file_dir, ew_hdu_ext, Ha_channel)
    
    # Ensure inputs are numpy arrays
    x = np.asarray(map_log_SII_Ha)
    y = np.asarray(map_log_OIII_Hb)
    ew = np.asarray(map_Ha_ew)
    
    # Only evaluate spaxels that survived the Flux S/N cuts
    valid = ~np.isnan(x) & ~np.isnan(y) & ~np.isnan(ew)
    
    # Suppress divide-by-zero warnings for values where x >= 0.32
    with np.errstate(divide='ignore', invalid='ignore'):
        kewley_limit = (0.72 / (x - 0.32)) + 1.30
        
    # Using <= 3 so if it magically lands perfectly on 3.0, it doesn't get dropped
    cond_lier = valid & (ew <= 3)
    cond_sf = valid & (ew > 3) & (x < 0.32) & (y < kewley_limit)
    cond_agn = valid & (ew > 3) & (~cond_sf) & (y > (1.89 * x + 0.76))
    cond_true_liner = valid & (ew > 3) & (~cond_sf) & (y <= (1.89 * x + 0.76))
    
    result = np.full(x.shape, np.nan, dtype=object)
    
    result[cond_sf] = "SF"
    result[cond_agn] = "AGN"
    result[cond_lier | cond_true_liner] = "LI(N)ER"
    
    return result


def NII_BPT_diagnostic_kewley(map_log_NII_Ha, map_log_OIII_Hb):
    """
    Classifies spaxels as Star-Forming (SF) or Active Galactic Nucleus (AGN) 
    using the strict Kewley et al. (2001) maximum starburst line.
    """
    
    # Ensure inputs are numpy arrays
    x = np.asarray(map_log_NII_Ha)
    y = np.asarray(map_log_OIII_Hb)
    
    # Only evaluate spaxels that have valid flux ratios
    valid = ~np.isnan(x) & ~np.isnan(y)
    
    # Suppress divide-by-zero warnings for values where x >= 0.47 (the asymptote)
    with np.errstate(divide='ignore', invalid='ignore'):
        kewley_limit = (0.61 / (x - 0.47)) + 1.19
        
    # Kewley et al. (2001) Conditions:
    # SF: Must be below the asymptote (x < 0.47) AND below the Kewley curve
    cond_sf = valid & (x < 0.47) & (y < kewley_limit)
    
    # AGN: Anything that is valid but NOT classified as SF falls into the AGN region
    # (This inherently includes all spaxels where x >= 0.47)
    cond_agn = valid & (~cond_sf)
    
    # Initialize result array with NaNs
    result = np.full(x.shape, np.nan, dtype=object)
    
    # Apply classifications
    result[cond_sf] = "SF"
    result[cond_agn] = "AGN"
    
    return result


def prepare_and_plot_resolved_bpt_map(target_info, fig, subplot_number, draw_scalebar=False):
    
    property_hdu_ext = 30 # EMLINE GFLUX

    Ha_channel = 23
    Hb_channel = 14
    OIII_channel = 16
    SII_6718_channel = 25
    SII_6732_channel = 26

    ### METADATA ###
    plate = target_info['plate']
    ifu = target_info['ifu']
    fov = target_info['fov']
    mangaid = target_info['mangaid']
    z = target_info['z']

    survey_order = ["lofar", "first", "vlass"]

    survey_config = {
        "lofar": {"color": "#FE4203", "label": "LoTSS 144 MHz", "linewidth": 2, "alpha": 0.8},    
        "first": {"color": "#A671F7", "label": "FIRST 1.4 GHz", "linewidth": 2, "alpha": 0.8},
        "vlass": {"color": "#FFCC00", "label": "VLASS 3 GHz", "linewidth": 2, "alpha": 0.8},
    }

    sigmas_multiples = np.array([3,5,10,20,40])
    sigmas_multiples_lofar = np.array([5,10,20,40])

    sigma_lofar = 83e-6
    levels_lofar = sigma_lofar*sigmas_multiples_lofar
    sigma_first = 0.15e-3
    levels_first = sigma_first*sigmas_multiples
    sigma_vlass = 70e-6
    levels_vlass = sigma_vlass*sigmas_multiples

    contour_outline_color = "black"  

    scale_bar_arcsec = 10.0  
    scale_bar_margin_x = 0.05  
    scale_bar_margin_y = 0.89  
    scale_bar_linewidth = 4  
    scale_bar_fontsize = 15  
    scale_bar_text_offset = 0.02  

    ### PLOT MAP###
    map_file_dir = find_map_file(map_dir, plate, ifu)


    # ------------------------------------------------------------------
    # DATA LOADING AND S/N CUTS
    # ------------------------------------------------------------------
    # Suppress warnings for sqrt of negative ivars in bad spaxels
    with np.errstate(invalid='ignore', divide='ignore'):
        
        # Strong lines: Keep S/N > 2
        map_Ha, map_wcs, map_shape = load_fits_data_map(map_file_dir, property_hdu_ext, Ha_channel)

        if map_Ha is None:
            return None
        
        ivar_Ha, _, __ = load_fits_data_map(map_file_dir, property_hdu_ext+1, Ha_channel)
        bitmask_Ha, _, __ = load_fits_data_map(map_file_dir, property_hdu_ext+2, Ha_channel)
        snr_Ha = np.abs(map_Ha) * np.sqrt(ivar_Ha)
        map_Ha = np.where((bitmask_Ha == 0) & (snr_Ha > 2), map_Ha, np.nan)

        map_OIII, _, __ = load_fits_data_map(map_file_dir, property_hdu_ext, OIII_channel)
        ivar_OIII, _, __ = load_fits_data_map(map_file_dir, property_hdu_ext+1, OIII_channel)
        bitmask_OIII, _, __ = load_fits_data_map(map_file_dir, property_hdu_ext+2, OIII_channel)
        snr_OIII = np.abs(map_OIII) * np.sqrt(ivar_OIII)
        map_OIII = np.where((bitmask_OIII == 0) & (snr_OIII > 2), map_OIII, np.nan)

        # Weak lines: Relax S/N to > 1.5 to preserve the core
        map_Hb, _, __ = load_fits_data_map(map_file_dir, property_hdu_ext, Hb_channel)
        ivar_Hb, _, __ = load_fits_data_map(map_file_dir, property_hdu_ext+1, Hb_channel)
        bitmask_Hb, _, __ = load_fits_data_map(map_file_dir, property_hdu_ext+2, Hb_channel)
        snr_Hb = np.abs(map_Hb) * np.sqrt(ivar_Hb)
        map_Hb = np.where((bitmask_Hb == 0) & (snr_Hb > 2), map_Hb, np.nan)

        # [SII] Doublet: Combine fluxes AND variances first, then apply S/N cut!
        map_SII_6718, _, __ = load_fits_data_map(map_file_dir, property_hdu_ext, SII_6718_channel)
        ivar_SII_6718, _, __ = load_fits_data_map(map_file_dir, property_hdu_ext+1, SII_6718_channel)
        bitmask_SII_6718, _, __ = load_fits_data_map(map_file_dir, property_hdu_ext+2, SII_6718_channel)
        
        map_SII_6732, _, __ = load_fits_data_map(map_file_dir, property_hdu_ext, SII_6732_channel)
        ivar_SII_6732, _, __ = load_fits_data_map(map_file_dir, property_hdu_ext+1, SII_6732_channel)
        bitmask_SII_6732, _, __ = load_fits_data_map(map_file_dir, property_hdu_ext+2, SII_6732_channel)
        
        map_SII_sum = map_SII_6732 + map_SII_6718
        # Variance adds together: var_sum = var1 + var2 --> 1/ivar_sum = 1/ivar1 + 1/ivar2
        ivar_SII_sum = 1.0 / (1.0/ivar_SII_6732 + 1.0/ivar_SII_6718)
        snr_SII_sum = np.abs(map_SII_sum) * np.sqrt(ivar_SII_sum)
        
        valid_SII = (bitmask_SII_6718 == 0) & (bitmask_SII_6732 == 0) & (snr_SII_sum > 2)
        map_SII = np.where(valid_SII, map_SII_sum, np.nan)

    # ------------------------------------------------------------------
    # MAP CALCULATIONS
    # ------------------------------------------------------------------

    map_log_SII_Ha = np.log10(map_SII/map_Ha)
    map_log_OIII_Hb = np.log10(map_OIII/map_Hb)

    bpt_map = SII_BPT_diagnostic(map_log_SII_Ha, map_log_OIII_Hb, target_info)

    if bpt_map is None:
        print("Skipping; Galaxy MAP not found")
        return None

    if map_wcs is None:
        print(f"  Skipping: Galaxy (WCS) not found")
        return None  

    map_height, map_width = map_shape
    
    # 1. Apply FOV Crop to the BPT Map first
    if fov is not None:
        center_x, center_y = map_width / 2.0, map_height / 2.0
        try:
            pixscale = np.abs(map_wcs.pixel_scale_matrix[0, 0]) * 3600.0
        except:
            pixscale = np.abs(map_wcs.wcs.cdelt[0]) * 3600.0
        
        fov_pixels = int(fov * 60.0 / pixscale)
        half_size = fov_pixels / 2.0
        x_min = max(0, int(center_x - half_size))
        x_max = min(map_width, int(center_x + half_size))
        y_min = max(0, int(center_y - half_size))
        y_max = min(map_height, int(center_y + half_size))
        
        bpt_map = bpt_map[y_min:y_max, x_min:x_max]
        map_wcs.wcs.crpix[0] -= x_min
        map_wcs.wcs.crpix[1] -= y_min

    # 2. Convert to numeric array safely AFTER cropping
    categories = ['SF', 'AGN', 'LI(N)ER'] 
    mapping = {cat: i for i, cat in enumerate(categories)}
    
    numeric_data = np.full(bpt_map.shape, np.nan, dtype=float)
    for category, num_val in mapping.items():
        numeric_data[bpt_map == category] = num_val

    # 3. Setup Plotting
    colors = ['#1f77b4', '#d62728', '#2ca02c']  
    cmap = ListedColormap(colors)

    ax = fig.add_subplot(subplot_number, projection=map_wcs)
    im = ax.imshow(numeric_data, cmap=cmap, vmin=-0.5, vmax=len(categories)-0.5)

    # 4. Add Legend 
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='SF', 
               markerfacecolor=colors[0], markersize=15),
        Line2D([0], [0], marker='o', color='w', label='AGN', 
               markerfacecolor=colors[1], markersize=15),
        Line2D([0], [0], marker='o', color='w', label='LI(N)ER', 
               markerfacecolor=colors[2], markersize=15)
    ]
    
    ax.legend(handles=legend_elements, loc='upper right', framealpha=0, fontsize=15)

    # ==========================================
    # Contours
    # ==========================================
    for survey_name in survey_order:
        if survey_name not in survey_config:
            continue
        survey_cfg = survey_config[survey_name]
        radio_file = find_cutout_file(radio_dir, mangaid, survey_name)
        if radio_file is None:
            continue
        
        radio_data, radio_wcs = load_fits_data(radio_file)
        if radio_data is None:
            continue
        
        reprojected = reproject_to_reference(radio_data, radio_wcs, map_wcs, bpt_map.shape)
        if reprojected is None:
            continue
        
        reprojected_peak = np.nanmax(reprojected)
        if reprojected_peak <= 0:
            continue
        
        if survey_name == "lofar": levels = levels_lofar
        elif survey_name == "first": levels = levels_first
        elif survey_name == "vlass": levels = levels_vlass

        outline_color = target_info.get("contour_outline_color", contour_outline_color)
        
        ax.contour(reprojected, levels=levels, colors=outline_color,
                  linewidths=survey_cfg["linewidth"] + 0.5, alpha=0.8,
                  transform=ax.get_transform('pixel'))
        
        ax.contour(reprojected, levels=levels, colors=survey_cfg["color"],
                  linewidths=survey_cfg["linewidth"], alpha=survey_cfg["alpha"],
                  transform=ax.get_transform('pixel'))

    if target_info['is_lotss_match'] == True:
        add_source_ellipse(target_info,survey_config,'lotss', map_wcs, pixscale, ax)
    if target_info['is_first_match'] == True:
        add_source_ellipse(target_info,survey_config,'first', map_wcs, pixscale, ax)
    if target_info['is_vlass_match'] == True:
        add_source_ellipse(target_info,survey_config,'vlass', map_wcs, pixscale, ax)
    
    # ==========================================
    # Scale Bar (Bottom Left)
    # ==========================================
    if draw_scalebar and z is not None and z > 0:
        sb_arcsec = target_info.get("scale_bar_arcsec", scale_bar_arcsec)
        sb_margin_x = target_info.get("scale_bar_margin_x", scale_bar_margin_x)
        sb_margin_y = target_info.get("scale_bar_margin_y", scale_bar_margin_y)
        sb_linewidth = target_info.get("scale_bar_linewidth", scale_bar_linewidth)
        sb_fontsize = target_info.get("scale_bar_fontsize", scale_bar_fontsize)
        sb_text_offset = target_info.get("scale_bar_text_offset", scale_bar_text_offset)

        cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
        d_a = cosmo.angular_diameter_distance(z)
        scale_kpc = (sb_arcsec / 3600.0 * np.pi / 180.0) * d_a.value * 1000.0
        
        try:
            pixscale = np.abs(map_wcs.pixel_scale_matrix[0, 0]) * 3600.0
        except:
            pixscale = np.abs(map_wcs.wcs.cdelt[0]) * 3600.0
        
        map_h, map_w = bpt_map.shape
        scale_pixels = sb_arcsec / pixscale
        scale_rel = scale_pixels / map_w
        
        bar_x_start = sb_margin_x
        bar_x_end = bar_x_start + scale_rel
        bar_y = sb_margin_y
        
        outline = [pe.withStroke(linewidth=1.5, foreground="white")]

        ax.plot([bar_x_start, bar_x_end], [bar_y, bar_y], 'k-',
               linewidth=sb_linewidth, transform=ax.transAxes, path_effects=outline)
        
        bar_x_center = (bar_x_start + bar_x_end) / 2
        ax.text(bar_x_center, bar_y + sb_text_offset,
               f"{sb_arcsec:.0f}\"",
               transform=ax.transAxes, fontsize=sb_fontsize,
               color='k', ha='center', va='bottom', weight='bold', path_effects=outline)
        ax.text(bar_x_center, bar_y - sb_text_offset,
               f"{scale_kpc:.1f} kpc",
               transform=ax.transAxes, fontsize=sb_fontsize,
               color='k', ha='center', va='top', weight='bold', path_effects=outline)
    
    ax.coords[0].set_ticks_visible(False)
    ax.coords[0].set_ticklabel_visible(False)
    ax.coords[1].set_ticks_visible(False)
    ax.coords[1].set_ticklabel_visible(False)
    ax.coords[0].set_axislabel('')
    ax.coords[1].set_axislabel('')
    ax.coords.grid(color='none', draw_grid=False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    
    return im


def prepare_and_plot_cutout(target_info, fov, fig, subplot_number):


    ### METADATA ###

    # Target Info
    plate = target_info['plate']
    ifu = target_info['ifu']
    fov = fov
    mangaid = target_info['mangaid']
    z = target_info['z']

    survey_order = ["lofar", "first", "vlass"]

    survey_config = {
        "lofar": {"color": "#FE4203", "label": "LoTSS 144 MHz", "linewidth": 2, "alpha": 0.8},    
        "first": {"color": "#A671F7", "label": "FIRST 1.4 GHz", "linewidth": 2, "alpha": 0.8},
        "vlass": {"color": "#FFCC00", "label": "VLASS 3 GHz", "linewidth": 2, "alpha": 0.8},
    }

    # Contour levels
    sigmas_multiples = np.array([3,5,10,20,40])
    sigmas_multiples_lofar = np.array([5,10,20,40])

    sigma_lofar = 83e-6
    levels_lofar = sigma_lofar*sigmas_multiples_lofar

    sigma_first = 0.15e-3
    levels_first = sigma_first*sigmas_multiples

    sigma_vlass = 70e-6
    levels_vlass = sigma_vlass*sigmas_multiples

    # Contour outline parameters
    contour_outline_color = "black"  # Default color for contour outline (background line)

    # Scale bar parameters
    scale_bar_arcsec = 10.0  # Scale bar length in arcseconds
    scale_bar_margin_x = 0.30  # Margin from right edge in relative coordinates (0-1)
    scale_bar_margin_y = 0.2  # INCREASED from 0.1 so text doesn't get cut off
    scale_bar_linewidth = 4  # Line width for scale bar
    scale_bar_fontsize = 15  # Font size for scale bar labels
    scale_bar_text_offset = 0.02  # Offset for text labels from bar in relative coordinates (0-1)

    mangaid_clean = mangaid.replace("-", "_")
    
    # Find files
    legacy_jpg_file = find_cutout_file(optical_dir, mangaid, "sdsslegacy_jpeg")
    wcs_file = find_cutout_file(optical_dir, mangaid, "sdsslegacy_fits")
    
    if legacy_jpg_file is None:
        print(f"  Skipping: SDSS Legacy JPEG not found")
        return False
    if wcs_file is None:
        print(f"  Skipping: SDSS Legacy FITS (WCS) not found")
        return False
    
    # Load optical data
    rgb_data, optical_wcs = load_legacy_jpg_and_wcs(legacy_jpg_file, wcs_file)
    if rgb_data is None or optical_wcs is None:
        print(f"  Skipping: corrupted SDSS Legacy files (JPEG or FITS)")
        return False
    
    img_height, img_width = rgb_data.shape[:2]
    
    # Crop to FOV if specified
    if fov is not None:
        center_x, center_y = img_width / 2.0, img_height / 2.0
        
        # Get pixel scale
        try:
            pixscale = np.abs(optical_wcs.pixel_scale_matrix[0, 0]) * 3600.0
        except:
            pixscale = np.abs(optical_wcs.wcs.cdelt[0]) * 3600.0
        
        # Calculate crop size
        fov_pixels = int(fov * 60.0 / pixscale)
        half_size = fov_pixels / 2.0
        x_min = max(0, int(center_x - half_size))
        x_max = min(img_width, int(center_x + half_size))
        y_min = max(0, int(center_y - half_size))
        y_max = min(img_height, int(center_y + half_size))
        
        # Crop
        rgb_data = rgb_data[y_min:y_max, x_min:x_max]
        optical_wcs.wcs.crpix[0] -= x_min
        optical_wcs.wcs.crpix[1] -= y_min
    
    img_height, img_width = rgb_data.shape[:2]

    ax = fig.add_subplot(subplot_number, projection = optical_wcs)
    ax.imshow(rgb_data, origin='lower')
    
    # Overlay radio contours (in specified order: LoTSS -> FIRST -> VLASS)
    contour_surveys = []
    
    for survey_name in survey_order:
        if survey_name not in survey_config:
            continue
        survey_cfg = survey_config[survey_name]
        radio_file = find_cutout_file(radio_dir, mangaid, survey_name)
        if radio_file is None:
            print(f"  {survey_cfg['label']}: file not found")
            continue
        
        radio_data, radio_wcs = load_fits_data(radio_file)
        if radio_data is None:
            print(f"  {survey_cfg['label']}: no data (corrupted or outside coverage)")
            continue
        
        # Reproject to optical WCS
        reprojected = reproject_to_reference(radio_data, radio_wcs, optical_wcs, rgb_data.shape[:2])
        if reprojected is None:
            continue
        
        # Calculate contour levels
        reprojected_peak = np.nanmax(reprojected)
        if reprojected_peak <= 0:
            continue
        
        if survey_name == "lofar":
            levels = levels_lofar
        
        elif survey_name == "first":
            levels = levels_first
        
        elif survey_name == "vlass":
            levels = levels_vlass
        
        else:
            print(survey_name)

       # Get outline color from target_info or use default
        outline_color = target_info.get("contour_outline_color", contour_outline_color)
        
        # Draw outline first (for contrast)
        ax.contour(reprojected, levels=levels, colors=outline_color,
                  linewidths=survey_cfg["linewidth"] + 0.5, alpha=0.8,
                  transform=ax.get_transform('pixel'))
        
        # Draw colored contour on top
        ax.contour(reprojected, levels=levels, colors=survey_cfg["color"],
                  linewidths=survey_cfg["linewidth"], alpha=survey_cfg["alpha"],
                  transform=ax.get_transform('pixel'))
        
        contour_surveys.append(survey_name)

    
    # Add scale bar
    # Get scale bar parameters from target_info or use defaults
    sb_arcsec = target_info.get("scale_bar_arcsec", scale_bar_arcsec)
    sb_margin_x = target_info.get("scale_bar_margin_x", scale_bar_margin_x)
    sb_margin_y = target_info.get("scale_bar_margin_y", scale_bar_margin_y)
    sb_linewidth = target_info.get("scale_bar_linewidth", scale_bar_linewidth)
    sb_fontsize = target_info.get("scale_bar_fontsize", scale_bar_fontsize)
    sb_text_offset = target_info.get("scale_bar_text_offset", scale_bar_text_offset)

    if z is not None and z > 0:
        cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
        d_a = cosmo.angular_diameter_distance(z)
        scale_kpc = (sb_arcsec / 3600.0 * np.pi / 180.0) * d_a.value * 1000.0
        
        try:
            pixscale = np.abs(optical_wcs.pixel_scale_matrix[0, 0]) * 3600.0
        except:
            pixscale = np.abs(optical_wcs.wcs.cdelt[0]) * 3600.0
        
        scale_pixels = sb_arcsec / pixscale
        scale_rel = scale_pixels / img_width
        
        # Position in bottom right corner using relative coordinates (0-1)
        bar_x_end = 1.0 - sb_margin_x
        bar_x_start = bar_x_end - scale_rel
        bar_y = sb_margin_y
        
        # Path effects for high contrast
        outline = [pe.withStroke(linewidth=3, foreground="black")]

        # Draw scale bar using relative coordinates
        ax.plot([bar_x_start, bar_x_end], [bar_y, bar_y], 'w-',
               linewidth=sb_linewidth, transform=ax.transAxes, path_effects=outline)
        
        # Add labels using relative coordinates
        bar_x_center = (bar_x_start + bar_x_end) / 2
        ax.text(bar_x_center, bar_y + sb_text_offset,
               f"{sb_arcsec:.0f}\"",
               transform=ax.transAxes, fontsize=sb_fontsize,
               color='white', ha='center', va='bottom', weight='bold', path_effects=outline)
        ax.text(bar_x_center, bar_y - sb_text_offset,
               f"{scale_kpc:.1f} kpc",
               transform=ax.transAxes, fontsize=sb_fontsize,
               color='white', ha='center', va='top', weight='bold', path_effects=outline)
    
    # Hide all axes elements
    ax.coords[0].set_ticks_visible(False)
    ax.coords[0].set_ticklabel_visible(False)
    ax.coords[1].set_ticks_visible(False)
    ax.coords[1].set_ticklabel_visible(False)
    ax.coords[0].set_axislabel('')
    ax.coords[1].set_axislabel('')
    ax.coords.grid(color='none', draw_grid=False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    
    # REMOVED subplots_adjust here so it doesn't break the grid layout
    return True



def plot_map_cutout_rcontour(target_info, property_name, property_hdu_ext, property_channel):

    plate = target_info['plate']
    ifu = target_info['ifu']
    fov = target_info['fov']
    mangaid = target_info['mangaid']
    z = target_info['z']
    log_mass = target_info['log_mass']
    log_sfr = target_info['log_sfr_sf']
    ra = target_info['ra_manga']
    dec = target_info['dec_manga']
    is_detected_agn = target_info['is_detected_agn']
    spectral_slope = get_spectral_slope(target_info)
    slope_str = f"{spectral_slope:.2f}\n" if spectral_slope is not None else "N/A\n"

    fig_width = 8.0  
    fig_height = 8.0  
    dpi = 400  


    if property_name == "Ha":
        property_label = r"H$\alpha$"
    
    if property_name == "O3":
        property_label = "O[III]"


    fig = plt.figure(figsize=(fig_width, fig_height), dpi=dpi)

    plt.suptitle(property_label + " Map, Optical Image and Radio Contours", fontsize = 24, y = 1.1)

    # wspace and hspace set to 0.0 to stitch the panels seamlessly
    fig.subplots_adjust(left=0.0, right=1, bottom=0.0, top=1, wspace=0.0, hspace=0.0)

    # ==========================================
    # panel 1: Info, Legend, and Colorbar
    # ==========================================
    ax_info = fig.add_subplot(221)
    ax_info.axis('off') 

    survey_config = {
        "lofar": {"color": "#FE4203", "label": "LoTSS 144 MHz", "linewidth": 2, "alpha": 0.8},    
        "first": {"color": "#A671F7", "label": "FIRST 1.4 GHz", "linewidth": 2, "alpha": 0.8},
        "vlass": {"color": "#FFCC00", "label": "VLASS 3 GHz", "linewidth": 2, "alpha": 0.8},
    }

    # 1. Add Text Info (Top Half, Left-Aligned)
    info_text = (
        rf"Spectral Slope $\alpha$: {slope_str}"
        f"Is Detected AGN: {is_detected_agn}\n"
        f"MaNGA ID: {mangaid}\n"
        f"Plate: {plate}\n"
        f"IFU: {ifu}\n"
        f"RA: {ra:.5f}\n"
        f"DEC: {dec:.5f}\n"
        f"z: {z:.4f}\n"
        r"$\log_{\rm 10} \left( M_{\star} / M_{\odot}\right)$: " + f"{log_mass:.2f}\n"
        r"$\log_{\rm 10} \left( \mathrm{SFR} / M_{\odot}~yr^{-1} \right)$: " + f"{log_sfr:.2f}"
        
    )
    
    # Text anchored at the top left (x=0.1)
    ax_info.text(0.0, 1, info_text, transform=ax_info.transAxes,
                 fontsize=12, verticalalignment='top', horizontalalignment='left')

    # 2. Add Custom Legend (Bottom, Left-Aligned)
    legend_elements = [
        Line2D([0], [0], color=config['color'], lw=config['linewidth'], 
               label=config['label'], alpha=config['alpha'])
        for key, config in survey_config.items()
    ]

    # Legend aligned to the bottom left (y shifted down to 0.05)
    legend = ax_info.legend(handles=legend_elements, loc='lower left', 
                            bbox_to_anchor=(0, 0.0), fontsize=18, 
                            frameon=False)

    # ==========================================
    # panel 2: manga_map
    # ==========================================
    mappable = prepare_and_plot_map(target_info, fig, 222, property_hdu_ext, property_channel)

    # ==========================================
    # 3. Add Colorbar to Panel 1 (Vertical, Right Side)
    # ==========================================
    if mappable is not None:
        # Shifted x from 0.85 to 0.80 to increase right margin
        # Adjusted width slightly to 0.04 so it isn't too thick
        cax = ax_info.inset_axes([0.93, 0.15, 0.04, 0.70])
        
        # Draw the vertical colorbar
        cb = fig.colorbar(mappable, cax=cax, orientation='vertical')
        
        # MOVE ticks and label to the left so they don't get cut off by the panel boundary
        cb.ax.yaxis.set_ticks_position('left')
        cb.ax.yaxis.set_label_position('left')
        
        # Force ticks, ticklabels, and the label to be black
        cb.ax.tick_params(color='black', labelcolor='black', labelsize=12)
        
        # Set the label in black
        cb.set_label(property_label + r" Flux ($10^{-17}$ erg s$^{-1}$ cm$^{-2}$ spaxel$^{-1}$)", 
                     color='black', fontsize=12)
        
        # Add a black border around the colorbar
        cb.outline.set_edgecolor('black')

    # ==========================================
    # panel 3: cutout fov 1 arcmin
    # ==========================================
    prepare_and_plot_cutout(target_info, 1, fig, 223)

    # ==========================================
    # panel 4: cutout fov 3 arcmin
    # ==========================================
    prepare_and_plot_cutout(target_info, 3, fig, 224)

    mangaid_str = str(mangaid).replace("-", "_")

    plt.savefig("../manga_rmorph_data/"+ property_name + "map_cutout_rcontour/" + mangaid_str + ".png", 
                bbox_inches='tight')
    
    
    plt.close(fig)




def plot_Ha_O3_O3kinematics_cutout_rcontour(target_info):

    Ha_channel = 23
    O3_channel = 16

    emline_gflux_hdu_ext = 30
    emline_gvel_hdu_ext = 37
    emline_gsigma_hdu_ext = 40

    plate = target_info['plate']
    ifu = target_info['ifu']
    fov = target_info['fov']
    mangaid = target_info['mangaid']
    z = target_info['z']
    log_mass = target_info['log_mass']
    log_sfr = target_info['log_sfr_sf']
    ra = target_info['ra_manga']
    dec = target_info['dec_manga']
    is_detected_agn = target_info['is_detected_agn']
    spectral_slope = get_spectral_slope(target_info)
    slope_str = f"{spectral_slope:.2f}\n" if spectral_slope is not None else "N/A\n"

    # ==========================================
    # FIXED CANVAS DIMENSIONS
    # ==========================================
    # A 3-row, 2-column grid of square images needs to be taller than it is wide.
    fig_width = 9.0  
    fig_height = 10.0  
    dpi = 400  

    fig = plt.figure(figsize=(fig_width, fig_height), dpi=dpi)

    plt.suptitle(r"H$\alpha$, O[III], O[III] Kinematics Maps, Optical Image and Radio Contours", fontsize=18, y=0.96)

    # ==========================================
    # PERFECTED GRID LAYOUT
    # ==========================================
    # These exact fractions ensure the bounding box for the plots matches the 
    # 3:2 height-to-width ratio of your image panels, eliminating the white space.
    fig.subplots_adjust(left=0.30, right=0.95, bottom=0.08, top=0.92, wspace=0.0, hspace=0.0)

    # ==========================================
    # Info Text & Legend (Placed outside panels)
    # ==========================================
    survey_config = {
        "lofar": {"color": "#FE4203", "label": "LoTSS 144 MHz", "linewidth": 2, "alpha": 0.8},    
        "first": {"color": "#A671F7", "label": "FIRST 1.4 GHz", "linewidth": 2, "alpha": 0.8},
        "vlass": {"color": "#FFCC00", "label": "VLASS 3 GHz", "linewidth": 2, "alpha": 0.8},
    }

    info_text = (
        rf"Spectral Slope $\alpha$: {slope_str}"
        f"Is Detected AGN: {is_detected_agn}\n"
        f"MaNGA ID: {mangaid}\n"
        f"Plate: {plate}\n"
        f"IFU: {ifu}\n"
        f"RA: {ra:.5f}\n"
        f"DEC: {dec:.5f}\n"
        f"z: {z:.4f}\n"
        r"$\log_{\rm 10} \left( M_{\star} / M_{\odot}\right)$: " + f"{log_mass:.2f}\n"
        r"$\log_{\rm 10} \left( \mathrm{SFR} / M_{\odot}~yr^{-1} \right)$: " + f"{log_sfr:.2f}"
    )
    
    # Left column for text
    fig.text(0.02, 0.5, info_text, fontsize=14, verticalalignment='center', horizontalalignment='left')

    legend_elements = [
        Line2D([0], [0], color=config['color'], lw=config['linewidth'], 
               label=config['label'], alpha=config['alpha'])
        for key, config in survey_config.items()
    ]
    
    # Legend centered underneath the image grid (X=0.625 is the exact center of the 0.30 to 0.95 grid span)
    fig.legend(handles=legend_elements, loc='lower center', 
               bbox_to_anchor=(0.625, 0.01), fontsize=13, frameon=False, ncol=3)

    # ==========================================
    # Panel 1 (321): cutout fov 1 arcmin
    # ==========================================
    prepare_and_plot_cutout(target_info, 1, fig, 321)

    # ==========================================
    # Panel 2 (322): cutout fov 3 arcmin
    # ==========================================
    prepare_and_plot_cutout(target_info, 3, fig, 322)

    # ==========================================
    # Panel 3 (323): H-alpha Flux Map
    # ==========================================
    prepare_and_plot_map(target_info, fig, 323, emline_gflux_hdu_ext, Ha_channel)

    # ==========================================
    # Panel 4 (324): O[III] Flux Map
    # ==========================================
    prepare_and_plot_map(target_info, fig, 324, emline_gflux_hdu_ext, O3_channel)

    # ==========================================
    # Panel 5 (325): O[III] Velocity Map
    # ==========================================
    prepare_and_plot_map(target_info, fig, 325, emline_gvel_hdu_ext, O3_channel, draw_scalebar= True)

    # ==========================================
    # Panel 6 (326): O[III] Velocity Dispersion Map
    # ==========================================
    prepare_and_plot_map(target_info, fig, 326, emline_gsigma_hdu_ext, O3_channel)

    mangaid_str = str(mangaid).replace("-", "_")

    plt.savefig("../manga_rmorph_data/Ha_O3_O3kinematics_maps_cutout_rcontour/" + mangaid_str + ".png", 
                bbox_inches='tight')
    
    plt.close(fig)



def plot_cutout_BPT_Ha_O3_O3kinematics_rcontour(target_info):

    Ha_channel = 23
    O3_channel = 16

    emline_gflux_hdu_ext = 30
    emline_gvel_hdu_ext = 37
    emline_gsigma_hdu_ext = 40

    plate = target_info['plate']
    ifu = target_info['ifu']
    fov = target_info['fov']
    mangaid = target_info['mangaid']
    z = target_info['z']
    log_mass = target_info['log_mass']
    log_sfr = target_info['log_sfr_sf']
    ra = target_info['ra_manga']
    dec = target_info['dec_manga']
    is_detected_agn = target_info['is_detected_agn']
    spectral_slope = get_spectral_slope(target_info)
    slope_str = f"{spectral_slope:.2f}\n" if spectral_slope is not None else "N/A\n"

    power_law_slope = target_info['power_law_slope']
    power_law_intercept = target_info['power_law_intercept']
    power_law_ch2 = target_info['power_law_chi2']
    
    has_all_three_matches = target_info['is_lotss_match'] & target_info['is_vlass_match'] & target_info['is_first_match']

    # ==========================================
    # CANVAS SETUP
    # ==========================================
    fig_width = 8.0  
    fig_height = 12.0  
    dpi = 400  

    fig = plt.figure(figsize=(fig_width, fig_height), dpi=dpi)
    fig.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=1.0, wspace=0.0, hspace=0.0)

    # ==========================================
    # Panel 1 (321): 3 arcmin cutout + Info Overlay
    # ==========================================
    n_axes_before = len(fig.axes)
    res1 = prepare_and_plot_cutout(target_info, 3, fig, 321)
    ax1 = fig.axes[-1] if len(fig.axes) > n_axes_before else fig.add_subplot(321)
    
    if not res1:
        ax1.axis('off')
        text_color_ax1 = 'black' # Change to black so it's visible on empty white panel
    else:
        ax1.margins(0)             
        text_color_ax1 = 'white'
        if hasattr(ax1, 'coords'):
            for coord in ax1.coords:
                coord.set_ticks_visible(False)
                coord.set_ticklabel_visible(False)
        else:
            ax1.set_xticks([])
            ax1.set_yticks([])
            ax1.tick_params(axis='x', bottom=False, top=False, labelbottom=False)
            ax1.tick_params(axis='y', left=False, right=False, labelleft=False)

    info_overlay_text_ax1 = (
        f"z: {z:.4f}\n"
        r"$\mathbf{log \left( M_{\star}\right)}$: " + f"{log_mass:.2f}\n"
        r"$\mathbf{log \left( SFR \right)}$: " + f"{log_sfr:.2f}"
    )

    ax1.text(0.05, 0.95, info_overlay_text_ax1, color=text_color_ax1, fontsize=20, 
             verticalalignment='top', horizontalalignment='left', 
             transform=ax1.transAxes, fontweight='bold')

    # ==========================================
    # Panel 2 (322): Resolved BPT Map + Stats + MaNGA ID
    # ==========================================
    n_axes_before = len(fig.axes)
    res2 = prepare_and_plot_resolved_bpt_map(target_info, fig, 322, draw_scalebar=False)
    ax2 = fig.axes[-1] if len(fig.axes) > n_axes_before else fig.add_subplot(322)
    
    if not res2:
        ax2.axis('off')

    lines = []
    if is_detected_agn:
        lines.append(r"$\mathbf{Is\ Detected\ AGN}$")

    if has_all_three_matches:
        lines.append(rf"$\boldsymbol{{\alpha}}$: $\mathbf{{{power_law_slope:.2f}}}$")
    elif spectral_slope is not None:
        lines.append(rf"$\boldsymbol{{\alpha}}$: $\mathbf{{{spectral_slope:.2f}}}$")

    if has_all_three_matches:
        lines.append(rf"$\boldsymbol{{\chi^2}}$: $\mathbf{{{power_law_ch2:.2f}}}$")

    info_overlay_text_ax2 = "\n".join(lines)

    ax2.text(0.05, 0.95, info_overlay_text_ax2, color='black', fontsize=20, 
             verticalalignment='top', horizontalalignment='left', 
             transform=ax2.transAxes, fontweight='bold')

    ax2.text(0.98, 0.02, f"MaNGA ID: {mangaid}", color='black', fontsize=20, 
             verticalalignment='bottom', horizontalalignment='right', 
             transform=ax2.transAxes, fontweight='bold')

    # ==========================================
    # PANELS 3-6: Maps
    # ==========================================
    # Panel 3
    n_axes_before = len(fig.axes)
    res3 = prepare_and_plot_map(target_info, fig, 323, emline_gflux_hdu_ext, Ha_channel)
    ax3 = fig.axes[-1] if len(fig.axes) > n_axes_before else fig.add_subplot(323)
    if not res3: ax3.axis('off')

    # Panel 4
    n_axes_before = len(fig.axes)
    res4 = prepare_and_plot_map(target_info, fig, 324, emline_gflux_hdu_ext, O3_channel)
    ax4 = fig.axes[-1] if len(fig.axes) > n_axes_before else fig.add_subplot(324)
    if not res4: ax4.axis('off')

    # Panel 5
    n_axes_before = len(fig.axes)
    res5 = prepare_and_plot_map(target_info, fig, 325, emline_gvel_hdu_ext, O3_channel, draw_scalebar=True)
    ax5 = fig.axes[-1] if len(fig.axes) > n_axes_before else fig.add_subplot(325)
    if not res5: ax5.axis('off')

    # Panel 6
    n_axes_before = len(fig.axes)
    res6 = prepare_and_plot_map(target_info, fig, 326, emline_gsigma_hdu_ext, O3_channel)
    ax6 = fig.axes[-1] if len(fig.axes) > n_axes_before else fig.add_subplot(326)
    if not res6: ax6.axis('off')

    # ==========================================
    # THE ALIGNMENT FIX (PERFECT LEGO SNAP)
    # ==========================================
    fig.canvas.draw()
    
    p3 = ax3.get_position()
    col_w = p3.width
    row_h = p3.height
    
    sq_h = col_w * (fig_width / fig_height)
    
    x_left = 0.5 - col_w
    x_right = 0.5
    y_start = 0.0 
    
    # Bottom Row
    ax5.set_position([x_left, y_start, col_w, row_h])
    ax6.set_position([x_right, y_start, col_w, row_h])
    
    # Middle Row
    ax3.set_position([x_left, y_start + row_h, col_w, row_h])
    ax4.set_position([x_right, y_start + row_h, col_w, row_h])
    
    # Top Row
    ax1.set_position([x_left, y_start + 2*row_h, col_w, sq_h])
    ax2.set_position([x_right, y_start + 2*row_h, col_w, sq_h])

    # ==========================================
    # PANEL-SPECIFIC MATCH LEGENDS 
    # ==========================================
    has_lofar = target_info.get('is_lotss_match', False)  
    has_first = target_info.get('is_first_match', False)
    has_vlass = target_info.get('is_vlass_match', False)

    # Panel 3 (ax3): LoTSS (Only plot if map exists)
    if has_lofar and res3:
        color = "#FE4203"
        line = [Line2D([0], [0], color=color, lw=2, label="LoTSS 144 MHz", alpha=0.8)]
        leg = ax3.legend(handles=line, loc='upper left', fontsize=20, frameon=False, borderaxespad=0.05, borderpad=0.05)
        for text in leg.get_texts():
            text.set_color(color)
            text.set_weight("bold")

    # Panel 4 (ax4): FIRST (Only plot if map exists)
    if has_first and res4:
        color = "#A671F7"
        line = [Line2D([0], [0], color=color, lw=2, label="FIRST 1.4 GHz", alpha=0.8)]
        leg = ax4.legend(handles=line, loc='upper left', fontsize=20, frameon=False, borderaxespad=0.05, borderpad=0.05)
        for text in leg.get_texts():
            text.set_color(color)
            text.set_weight("bold")

    # Panel 6 (ax6): VLASS (Only plot if map exists)
    if has_vlass and res6:
        color = "#FFCC00"
        line = [Line2D([0], [0], color=color, lw=2, label="VLASS 3 GHz", alpha=0.8)]
        leg = ax6.legend(handles=line, loc='upper left', fontsize=20, frameon=False, borderaxespad=0.05, borderpad=0.05)
        for text in leg.get_texts():
            text.set_color(color)
            text.set_weight("bold")

    # ==========================================
    # SAVE
    # ==========================================
    mangaid_str = str(mangaid).replace("-", "_")
    
    plt.savefig("../manga_rmorph_data/cutout_BPT_Ha_O3_O3kinematics_rcontour/" + mangaid_str + ".png", 
                facecolor='white', bbox_inches='tight', pad_inches=0)
    

    plt.close(fig)



def categorise_central_ionization_type_SII(plate, ifu):

    property_hdu_ext = 30 # EMLINE GFLUX

    Ha_channel = 23
    Hb_channel = 14
    OIII_channel = 16

    NII_6585_channel = 24
    SII_6718_channel = 25
    SII_6732_channel = 26

    ### METADATA ###

    fov = 1 #arcmin

    map_file_dir = find_map_file(map_dir, plate, ifu)

    # ------------------------------------------------------------------
    # DATA LOADING AND S/N CUTS
    # ------------------------------------------------------------------
    # Suppress warnings for sqrt of negative ivars in bad spaxels
    with np.errstate(invalid='ignore', divide='ignore'):
        
        # Strong lines: Keep S/N > 2
        map_Ha, map_wcs, map_shape = load_fits_data_map(map_file_dir, property_hdu_ext, Ha_channel)

        if map_Ha is None:
            return None
        
        ivar_Ha, _, __ = load_fits_data_map(map_file_dir, property_hdu_ext+1, Ha_channel)
        bitmask_Ha, _, __ = load_fits_data_map(map_file_dir, property_hdu_ext+2, Ha_channel)
        snr_Ha = np.abs(map_Ha) * np.sqrt(ivar_Ha)
        map_Ha = np.where((bitmask_Ha == 0) & (snr_Ha > 2), map_Ha, np.nan)

        map_OIII, _, __ = load_fits_data_map(map_file_dir, property_hdu_ext, OIII_channel)
        ivar_OIII, _, __ = load_fits_data_map(map_file_dir, property_hdu_ext+1, OIII_channel)
        bitmask_OIII, _, __ = load_fits_data_map(map_file_dir, property_hdu_ext+2, OIII_channel)
        snr_OIII = np.abs(map_OIII) * np.sqrt(ivar_OIII)
        map_OIII = np.where((bitmask_OIII == 0) & (snr_OIII > 2), map_OIII, np.nan)

        map_NII, _, __ = load_fits_data_map(map_file_dir, property_hdu_ext, NII_6585_channel)
        ivar_NII, _, __ = load_fits_data_map(map_file_dir, property_hdu_ext+1, NII_6585_channel)
        bitmask_NII, _, __ = load_fits_data_map(map_file_dir, property_hdu_ext+2, NII_6585_channel)
        snr_NII = np.abs(map_NII) * np.sqrt(ivar_NII)
        map_NII = np.where((bitmask_NII == 0) & (snr_NII > 2), map_NII, np.nan)

        # Weak lines: Relax S/N to > 1.5 to preserve the core
        map_Hb, _, __ = load_fits_data_map(map_file_dir, property_hdu_ext, Hb_channel)
        ivar_Hb, _, __ = load_fits_data_map(map_file_dir, property_hdu_ext+1, Hb_channel)
        bitmask_Hb, _, __ = load_fits_data_map(map_file_dir, property_hdu_ext+2, Hb_channel)
        snr_Hb = np.abs(map_Hb) * np.sqrt(ivar_Hb)
        map_Hb = np.where((bitmask_Hb == 0) & (snr_Hb > 2), map_Hb, np.nan)

        # [SII] Doublet: Combine fluxes AND variances first, then apply S/N cut!
        map_SII_6718, _, __ = load_fits_data_map(map_file_dir, property_hdu_ext, SII_6718_channel)
        ivar_SII_6718, _, __ = load_fits_data_map(map_file_dir, property_hdu_ext+1, SII_6718_channel)
        bitmask_SII_6718, _, __ = load_fits_data_map(map_file_dir, property_hdu_ext+2, SII_6718_channel)
        
        map_SII_6732, _, __ = load_fits_data_map(map_file_dir, property_hdu_ext, SII_6732_channel)
        ivar_SII_6732, _, __ = load_fits_data_map(map_file_dir, property_hdu_ext+1, SII_6732_channel)
        bitmask_SII_6732, _, __ = load_fits_data_map(map_file_dir, property_hdu_ext+2, SII_6732_channel)
        
        map_SII_sum = map_SII_6732 + map_SII_6718
        # Variance adds together: var_sum = var1 + var2 --> 1/ivar_sum = 1/ivar1 + 1/ivar2
        ivar_SII_sum = 1.0 / (1.0/ivar_SII_6732 + 1.0/ivar_SII_6718)
        snr_SII_sum = np.abs(map_SII_sum) * np.sqrt(ivar_SII_sum)
        
        valid_SII = (bitmask_SII_6718 == 0) & (bitmask_SII_6732 == 0) & (snr_SII_sum > 2)
        map_SII = np.where(valid_SII, map_SII_sum, np.nan)

    # ------------------------------------------------------------------
    # MAP CALCULATIONS
    # ------------------------------------------------------------------

    map_log_SII_Ha = np.log10(map_SII/map_Ha)
    map_log_OIII_Hb = np.log10(map_OIII/map_Hb)

    bpt_map = SII_BPT_diagnostic(map_log_SII_Ha, map_log_OIII_Hb, plate, ifu)


    if bpt_map is None:
        print("Skipping; Galaxy MAP not found")
        return None

    if map_wcs is None:
        print(f"  Skipping: Galaxy (WCS) not found")
        return None  

    map_height, map_width = map_shape
    
    # 1. Apply FOV Crop to the BPT Map first
    if fov is not None:
        center_x, center_y = map_width / 2.0, map_height / 2.0
        try:
            pixscale = np.abs(map_wcs.pixel_scale_matrix[0, 0]) * 3600.0
        except:
            pixscale = np.abs(map_wcs.wcs.cdelt[0]) * 3600.0
        
        fov_pixels = int(fov * 60.0 / pixscale)
        half_size = fov_pixels / 2.0
        x_min = max(0, int(center_x - half_size))
        x_max = min(map_width, int(center_x + half_size))
        y_min = max(0, int(center_y - half_size))
        y_max = min(map_height, int(center_y + half_size))
        
        bpt_map = bpt_map[y_min:y_max, x_min:x_max]
        map_wcs.wcs.crpix[0] -= x_min
        map_wcs.wcs.crpix[1] -= y_min

    # 2. Convert to numeric array safely AFTER cropping
    categories_SII = ['SF', 'AGN', 'LI(N)ER'] 
    mapping_SII = {cat: i for i, cat in enumerate(categories_SII)}
    
    numeric_data_SII = np.full(bpt_map.shape, np.nan, dtype=float)
    for category, num_val in mapping_SII.items():
        numeric_data_SII[bpt_map == category] = num_val


    H, W = numeric_data_SII.shape
    center_y, center_x = H // 2, W // 2
    central_5x5_array = numeric_data_SII[center_y-2 : center_y+3, center_x-2 : center_x+3]

    print(central_5x5_array)

    # 1. Flatten the array so we can easily count the elements
    flat_array = central_5x5_array.flatten()

    # 2. Convert 'None' to 'np.nan' to ensure they are counted together
    # and force the array to float so np.isnan() works safely.
    flat_array = np.where(flat_array == None, np.nan, flat_array)
    flat_array = np.asarray(flat_array, dtype=float)

    # 3. Get unique values and their frequencies. 
    # (In numpy >= 1.24, equal_nan=True is the default, grouping all NaNs together)
    values, counts = np.unique(flat_array, return_counts=True)

    # 4. Find the index of the highest count, and get the corresponding value
    dominant_index = np.argmax(counts)
    dominant_value = values[dominant_index]

    # 5. Return the corresponding classification
    if dominant_value == 0:
        return "SF"
    elif dominant_value == 1:
        return "Optical AGN"
    elif dominant_value == 2:
        return "LI(N)ER"
    elif np.isnan(dominant_value):
        # This will trigger if np.nan or None is the dominant type
        return "Little Ionization" 
    else:
        # Catch-all for unexpected numbers
        return "Unknown"
    



def categorise_central_ionization_type_NII(plate, ifu):

    property_hdu_ext = 30 # EMLINE GFLUX

    Ha_channel = 23
    Hb_channel = 14
    OIII_channel = 16

    NII_6585_channel = 24
    SII_6718_channel = 25
    SII_6732_channel = 26

    ### METADATA ###

    fov = 1 #arcmin

    map_file_dir = find_map_file(map_dir, plate, ifu)

    # ------------------------------------------------------------------
    # DATA LOADING AND S/N CUTS
    # ------------------------------------------------------------------
    # Suppress warnings for sqrt of negative ivars in bad spaxels
    with np.errstate(invalid='ignore', divide='ignore'):
        
        # Strong lines: Keep S/N > 2
        map_Ha, map_wcs, map_shape = load_fits_data_map(map_file_dir, property_hdu_ext, Ha_channel)

        if map_Ha is None:
            return None
        
        ivar_Ha, _, __ = load_fits_data_map(map_file_dir, property_hdu_ext+1, Ha_channel)
        bitmask_Ha, _, __ = load_fits_data_map(map_file_dir, property_hdu_ext+2, Ha_channel)
        snr_Ha = np.abs(map_Ha) * np.sqrt(ivar_Ha)
        map_Ha = np.where((bitmask_Ha == 0) & (snr_Ha > 2), map_Ha, np.nan)

        map_OIII, _, __ = load_fits_data_map(map_file_dir, property_hdu_ext, OIII_channel)
        ivar_OIII, _, __ = load_fits_data_map(map_file_dir, property_hdu_ext+1, OIII_channel)
        bitmask_OIII, _, __ = load_fits_data_map(map_file_dir, property_hdu_ext+2, OIII_channel)
        snr_OIII = np.abs(map_OIII) * np.sqrt(ivar_OIII)
        map_OIII = np.where((bitmask_OIII == 0) & (snr_OIII > 2), map_OIII, np.nan)

        map_NII, _, __ = load_fits_data_map(map_file_dir, property_hdu_ext, NII_6585_channel)
        ivar_NII, _, __ = load_fits_data_map(map_file_dir, property_hdu_ext+1, NII_6585_channel)
        bitmask_NII, _, __ = load_fits_data_map(map_file_dir, property_hdu_ext+2, NII_6585_channel)
        snr_NII = np.abs(map_NII) * np.sqrt(ivar_NII)
        map_NII = np.where((bitmask_NII == 0) & (snr_NII > 2), map_NII, np.nan)

        # Weak lines: Relax S/N to > 1.5 to preserve the core
        map_Hb, _, __ = load_fits_data_map(map_file_dir, property_hdu_ext, Hb_channel)
        ivar_Hb, _, __ = load_fits_data_map(map_file_dir, property_hdu_ext+1, Hb_channel)
        bitmask_Hb, _, __ = load_fits_data_map(map_file_dir, property_hdu_ext+2, Hb_channel)
        snr_Hb = np.abs(map_Hb) * np.sqrt(ivar_Hb)
        map_Hb = np.where((bitmask_Hb == 0) & (snr_Hb > 2), map_Hb, np.nan)

        # [SII] Doublet: Combine fluxes AND variances first, then apply S/N cut!
        map_SII_6718, _, __ = load_fits_data_map(map_file_dir, property_hdu_ext, SII_6718_channel)
        ivar_SII_6718, _, __ = load_fits_data_map(map_file_dir, property_hdu_ext+1, SII_6718_channel)
        bitmask_SII_6718, _, __ = load_fits_data_map(map_file_dir, property_hdu_ext+2, SII_6718_channel)
        
        map_SII_6732, _, __ = load_fits_data_map(map_file_dir, property_hdu_ext, SII_6732_channel)
        ivar_SII_6732, _, __ = load_fits_data_map(map_file_dir, property_hdu_ext+1, SII_6732_channel)
        bitmask_SII_6732, _, __ = load_fits_data_map(map_file_dir, property_hdu_ext+2, SII_6732_channel)
        
        map_SII_sum = map_SII_6732 + map_SII_6718
        # Variance adds together: var_sum = var1 + var2 --> 1/ivar_sum = 1/ivar1 + 1/ivar2
        ivar_SII_sum = 1.0 / (1.0/ivar_SII_6732 + 1.0/ivar_SII_6718)
        snr_SII_sum = np.abs(map_SII_sum) * np.sqrt(ivar_SII_sum)
        
        valid_SII = (bitmask_SII_6718 == 0) & (bitmask_SII_6732 == 0) & (snr_SII_sum > 2)
        map_SII = np.where(valid_SII, map_SII_sum, np.nan)

    # ------------------------------------------------------------------
    # MAP CALCULATIONS
    # ------------------------------------------------------------------

    map_log_NII_Ha = np.log10(map_NII/map_Ha)
    map_log_OIII_Hb = np.log10(map_OIII/map_Hb)

    bpt_map = NII_BPT_diagnostic_kewley(map_log_NII_Ha, map_log_OIII_Hb)


    if bpt_map is None:
        print("Skipping; Galaxy MAP not found")
        return None

    if map_wcs is None:
        print(f"  Skipping: Galaxy (WCS) not found")
        return None  

    map_height, map_width = map_shape
    
    # 1. Apply FOV Crop to the BPT Map first
    if fov is not None:
        center_x, center_y = map_width / 2.0, map_height / 2.0
        try:
            pixscale = np.abs(map_wcs.pixel_scale_matrix[0, 0]) * 3600.0
        except:
            pixscale = np.abs(map_wcs.wcs.cdelt[0]) * 3600.0
        
        fov_pixels = int(fov * 60.0 / pixscale)
        half_size = fov_pixels / 2.0
        x_min = max(0, int(center_x - half_size))
        x_max = min(map_width, int(center_x + half_size))
        y_min = max(0, int(center_y - half_size))
        y_max = min(map_height, int(center_y + half_size))
        
        bpt_map = bpt_map[y_min:y_max, x_min:x_max]
        map_wcs.wcs.crpix[0] -= x_min
        map_wcs.wcs.crpix[1] -= y_min

    # 2. Convert to numeric array safely AFTER cropping
    categories_NII = ['SF', 'AGN'] 
    mapping_NII = {cat: i for i, cat in enumerate(categories_NII)}
    
    numeric_data_NII = np.full(bpt_map.shape, np.nan, dtype=float)
    for category, num_val in mapping_NII.items():
        numeric_data_NII[bpt_map == category] = num_val


    H, W = numeric_data_NII.shape
    center_y, center_x = H // 2, W // 2
    central_5x5_array = numeric_data_NII[center_y-2 : center_y+3, center_x-2 : center_x+3]

    # 1. Flatten the array so we can easily count the elements
    flat_array = central_5x5_array.flatten()

    # 2. Convert 'None' to 'np.nan' to ensure they are counted together
    # and force the array to float so np.isnan() works safely.
    flat_array = np.where(flat_array == None, np.nan, flat_array)
    flat_array = np.asarray(flat_array, dtype=float)

    # 3. Get unique values and their frequencies. 
    # (In numpy >= 1.24, equal_nan=True is the default, grouping all NaNs together)
    values, counts = np.unique(flat_array, return_counts=True)

    # 4. Find the index of the highest count, and get the corresponding value
    dominant_index = np.argmax(counts)
    dominant_value = values[dominant_index]

    # 5. Return the corresponding classification
    if dominant_value == 0:
        return "SF"
    elif dominant_value == 1:
        return "Optical AGN"

    elif np.isnan(dominant_value):
        # This will trigger if np.nan or None is the dominant type
        return "Little Ionization" 
    
    else:
        # Catch-all for unexpected numbers
        return "Unknown"