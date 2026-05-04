# Generate wget commands for downloading Radio Cutout Images for MaNGA Sample
# Generates: download_radio_commands.sh

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import pandas as pd
import os
from pathlib import Path
from astropy.table import Table

path_data_final = "../manga_rmorph_data/manga_with_vlass_lotss_first.fits"
data_final = fits.open(path_data_final)[1].data

manga_targets = []

for i in range(len(data_final)):

    if (data_final['is_vlass_match'][i]) or (data_final['is_lotss_match'][i]) or (data_final['radio_class'][i] != 0):

        manga_targets.append({"mangaid" : data_final['mangaid'][i], "fov" : 5.0})
        

drpall_path = "../manga_rmorph_data/drpall-v3_1_1.fits"

output_dir = Path("../manga_rmorph_data") / "Radio"
output_dir.mkdir(exist_ok=True, parents=True)
script_dir = Path(__file__).parent

surveys = {
    "lofar": {
        "pixscale": 6.0,
        "base_url": "https://lofar-surveys.org/dr3-cutout.fits"
    },
    "vlass": {
        "pixscale": 1.0,
        "hips_url": "https://vlass-dl.nrao.edu/vlass/HiPS/MedianStack/Quicklook/",
        "hips_service": "https://alasky.u-strasbg.fr/hips-image-services/hips2fits"
    },
    "first": {
        "pixscale": 1.8,
        "base_url": "https://sundog.stsci.edu/cgi-bin/firstcutout"
    }
}

try:
    drp = Table.read(drpall_path, 'MANGA')
except:
    drp = fits.getdata(drpall_path)

def get_coords(mangaid):
    matches = drp[drp['mangaid'] == mangaid]
    if len(matches) == 0:
        return None, None
    row = matches[np.argmax(matches['redsn2'])] if len(matches) > 1 else matches[0]
    ra_col = next((c for c in ['objra', 'OBJRA', 'ra', 'RA', 'ifura', 'IFURA'] if c in (row.colnames if hasattr(row, 'colnames') else row.dtype.names)), None)
    dec_col = next((c for c in ['objdec', 'OBJDEC', 'dec', 'DEC', 'ifudec', 'IFUDEC'] if c in (row.colnames if hasattr(row, 'colnames') else row.dtype.names)), None)
    return (row[ra_col], row[dec_col]) if ra_col and dec_col else (None, None)

commands = []

for target in manga_targets:
    mangaid, fov = target["mangaid"], target.get("fov", 5.0)
    ra, dec = get_coords(mangaid)
    if ra is None:
        continue

    mangaid_clean = mangaid.replace("-", "_")

    for survey_name, config in surveys.items():
        pixscale = config["pixscale"]
        size_pixels = int(fov * 60.0 / pixscale)
        filename = f"{mangaid_clean}_{survey_name}_fov{fov:.1f}arcmin_pixscale{pixscale:.2f}_npix{size_pixels}.fits"

        if survey_name == "lofar":
            url = f"{config['base_url']}?pos={ra:.6f}+{dec:+.6f}&size={fov:.1f}"

        elif survey_name == "vlass":
            fov_deg = size_pixels * pixscale / 3600.0
            url = f"{config['hips_service']}?ra={ra:.6f}&dec={dec:.6f}&hips={config['hips_url']}&width={size_pixels}&height={size_pixels}&format=fits&fov={fov_deg:.6f}"

        elif survey_name == "first":
            url = f"{config['base_url']}?RA={ra/15.0:.6f}&Dec={dec:.6f}&ImageSize={fov:.1f}&ImageType=FITS"
        
        commands.append(f"wget -O '{output_dir / filename}' '{url}'")

script_path = script_dir / "download_radio_commands.sh"
with open(script_path, 'w') as f:
    f.write("#!/bin/bash\n# Generated wget commands for radio cutouts\n\n")
    f.write("\n".join(commands) + "\n")
os.chmod(script_path, 0o755)

print(f"Generated {len(commands)} commands -> {script_path.name}")