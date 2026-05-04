# Plot SDSS optical images with radio contours from VLASS, LOFAR, FIRST
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.wcs import WCS
from astropy.cosmology import FlatLambdaCDM
from astropy.table import Table
from pathlib import Path
from PIL import Image
import warnings
warnings.filterwarnings('ignore')

# Paths
optical_dir = Path("../manga_rmorph_data/Optical")
radio_dir = Path("../manga_rmorph_data/Radio")
output_dir = Path("../manga_rmorph_data/plots")
output_dir.mkdir(exist_ok=True, parents=True)
drpall_path = "../manga_rmorph_data/drpall-v3_1_1.fits"

# Cosmology
cosmo = FlatLambdaCDM(H0=70, Om0=0.3)

# Survey configuration
survey_config = {
    "lofar": {"color": "#FE4203", "label": "LoTSS 144 MHz", "linewidth": 0.5, "alpha": 0.8},
    # "lofar": {"color": "#FF8860", "label": "LoTSS 144 MHz", "linewidth": 0.5, "alpha": 0.8},    
    "first": {"color": "#A671F7", "label": "FIRST 1.4 GHz", "linewidth": 0.5, "alpha": 0.8},
    "vlass": {"color": "#FFCC00", "label": "VLASS 3 GHz", "linewidth": 0.5, "alpha": 0.8},
}

# Survey drawing order (LoTSS -> FIRST -> VLASS)
survey_order = ["lofar", "first", "vlass"]

# Contour levels
sigmas_multiples = np.array([3,5,10,20,40])

sigma_lofar = 83e-6
levels_lofar = sigma_lofar*sigmas_multiples

sigma_first = 0.15e-3
levels_first = sigma_first*sigmas_multiples

sigma_vlass = 70e-6
levels_vlass = sigma_vlass*sigmas_multiples

# Each entry can have:
#   - mangaid: MaNGA ID (required)
#   - fov: field of view in arcminutes (optional)
#   - scale_bar_arcsec: scale bar length in arcseconds (optional)
#   - scale_bar_margin_x: margin from right edge in relative coordinates (0-1) (optional)
#   - scale_bar_margin_y: margin from bottom edge in relative coordinates (0-1) (optional)
#   - scale_bar_linewidth: line width for scale bar (optional)
#   - scale_bar_fontsize: font size for scale bar labels (optional)
#   - scale_bar_text_offset: offset for text labels from bar in pixels (optional)
#   - contour_outline_color: color for contour outline/background line (optional, default: "black")

path_data_final = "../manga_rmorph_data/manga_with_vlass_lotss.fits"
data_final = fits.open(path_data_final)[1].data

manga_targets = []

for i in range(len(data_final)):

    if (data_final['is_vlass_match'][i]) or (data_final['is_lotss_match'][i]) or (data_final['radio_class'][i] != 0):

        manga_targets.append({"mangaid" : data_final['mangaid'][i], "fov" : 1.0, "scale_bar_arcsec": 15.0})

# Plotting parameters
fig_width = 3.0  # Figure width in inches
fig_height = 3.0  # Figure height in inches
dpi = 300  # DPI for output PDF

# Label parameters
label_x = 0.02  # X position for survey labels (left margin)
label_y_start = 0.98  # Starting Y position for survey labels (top)
label_spacing = 0.06  # Vertical spacing between labels
label_fontsize = 12  # Font size for labels
mangaid_fontsize = 14  # Font size for MaNGA ID label

# Scale bar parameters
scale_bar_arcsec = 10.0  # Scale bar length in arcseconds
scale_bar_margin_x = 0.05  # Margin from right edge in relative coordinates (0-1)
scale_bar_margin_y = 0.1  # Margin from bottom edge in relative coordinates (0-1)
scale_bar_linewidth = 1  # Line width for scale bar
scale_bar_fontsize = 10  # Font size for scale bar labels
scale_bar_text_offset = 0.02  # Offset for text labels from bar in relative coordinates (0-1)

# Contour outline parameters
contour_outline_color = "black"  # Default color for contour outline (background line)

def load_drpall(filepath):
    """Load drpall file."""
    try:
        return Table.read(filepath, 'MANGA')
    except:
        return fits.getdata(filepath)

def get_redshift(drp, mangaid):
    """Get redshift for manga ID."""
    matches = drp[drp['mangaid'] == mangaid]
    if len(matches) == 0:
        return None
    row = matches[0] if len(matches) == 1 else matches[np.argmax(matches['redsn2'])]
    
    for z_col in ['nsa_z', 'NSA_Z', 'z', 'Z']:
        if z_col in (row.colnames if hasattr(row, 'colnames') else row.dtype.names):
            return row[z_col]
    return None

def find_cutout_file(directory, mangaid, survey_pattern):
    """Find cutout file matching manga ID and survey pattern."""
    mangaid_clean = mangaid.replace("-", "_")
    pattern = f"{mangaid_clean}_{survey_pattern}*"
    matches = list(directory.glob(pattern))
    return matches[0] if matches else None

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

def calculate_contour_levels(data, fractions):
    """Calculate contour levels from data."""
    peak = np.nanmax(data)
    if peak <= 0:
        return []
    return [peak * frac for frac in fractions]

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
    

def plot_manga_target(target_info):
    """Plot SDSS Legacy JPEG with radio contours."""
    mangaid = target_info["mangaid"]
    fov = target_info.get("fov", None)
    
    print(f"\nProcessing {mangaid} (FOV={fov:.1f} arcmin)")
    
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
    
    # Create figure
    fig = plt.figure(figsize=(fig_width, fig_height), dpi=dpi)
    ax = fig.add_subplot(111, projection=optical_wcs)
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
        
        if survey_name == "first":
            levels = levels_first
        
        if survey_name == "vlass":
            levels = levels_vlass
        
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
    
    # Add MaNGA ID in top right corner (white text with black outline)
    # Draw black shadow/outline first (8 directions around text)
    for dx, dy in [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]:
        ax.text(0.98 + dx*0.001, 0.98 + dy*0.001, mangaid, transform=ax.transAxes,
               fontsize=mangaid_fontsize, color='black', ha='right', va='top',
               weight='bold', alpha=0.5)
    
    # Draw white text on top
    ax.text(0.98, 0.98, mangaid, transform=ax.transAxes,
           fontsize=mangaid_fontsize, color='white', ha='right', va='top',
           weight='bold')
    
    # Add survey labels (one under another in column)
    if contour_surveys:
        for i, survey_name in enumerate(contour_surveys):
            survey_cfg = survey_config[survey_name]
            label_y = label_y_start - i * label_spacing
            # Draw black shadow/outline first (8 directions around text)
            for dx, dy in [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]:
                ax.text(label_x + dx*0.001, label_y + dy*0.001, survey_cfg["label"],
                       transform=ax.transAxes, fontsize=label_fontsize,
                       color='black', ha='left', va='top', weight='bold', alpha=0.5)
            
            # Draw colored text on top
            ax.text(label_x, label_y, survey_cfg["label"],
                   transform=ax.transAxes, fontsize=label_fontsize,
                   color=survey_cfg["color"], ha='left', va='top', weight='bold')
    
    # Add scale bar
    # Get scale bar parameters from target_info or use defaults
    sb_arcsec = target_info.get("scale_bar_arcsec", scale_bar_arcsec)
    sb_margin_x = target_info.get("scale_bar_margin_x", scale_bar_margin_x)
    sb_margin_y = target_info.get("scale_bar_margin_y", scale_bar_margin_y)
    sb_linewidth = target_info.get("scale_bar_linewidth", scale_bar_linewidth)
    sb_fontsize = target_info.get("scale_bar_fontsize", scale_bar_fontsize)
    sb_text_offset = target_info.get("scale_bar_text_offset", scale_bar_text_offset)
    
    drp = load_drpall(drpall_path)
    z = get_redshift(drp, mangaid)
    
    if z is not None and z > 0:
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
        
        # Draw scale bar using relative coordinates
        ax.plot([bar_x_start, bar_x_end], [bar_y, bar_y], 'w-',
               linewidth=sb_linewidth, transform=ax.transAxes)
        
        # Add labels using relative coordinates
        bar_x_center = (bar_x_start + bar_x_end) / 2
        ax.text(bar_x_center, bar_y + sb_text_offset,
               f"{sb_arcsec:.0f}\"",
               transform=ax.transAxes, fontsize=sb_fontsize,
               color='white', ha='center', va='bottom', weight='bold')
        ax.text(bar_x_center, bar_y - sb_text_offset,
               f"{scale_kpc:.1f} kpc",
               transform=ax.transAxes, fontsize=sb_fontsize,
               color='white', ha='center', va='top', weight='bold')
    
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
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    
    # Save figure
    output_file = output_dir / f"{mangaid_clean}_sdss_legacy.pdf"
    
    plt.savefig(output_file, dpi=dpi, bbox_inches='tight', pad_inches=0)
    plt.close()
    
    print(f"  Saved: {output_file.name}")
    return True

# Main execution
print("=" * 60)
print("Plotting SDSS images with radio contours")
print("=" * 60)

success_count = 0
fail_count = 0


for target in manga_targets:

    try:
        if plot_manga_target(target):
            success_count += 1
        else:
            fail_count += 1
    except Exception as e:
        print(f"  ERROR: {e}")
        fail_count += 1


print("\n" + "=" * 60)
print(f"Completed: {success_count} successful, {fail_count} failed")
print("=" * 60)


