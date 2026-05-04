# Generate wget commands for downloading Optical Cutout Images for MaNGA Sample
# Generates: download_optical_commands.sh

import os
from pathlib import Path
from astropy.table import Table
from astropy.io import fits
import numpy as np

path_data_final = "../manga_rmorph_data/manga_with_vlass_lotss_first.fits"
data_final = fits.open(path_data_final)[1].data

manga_targets = []

for i in range(len(data_final)):

    if (data_final['is_vlass_match'][i]) or (data_final['is_lotss_match'][i]) or (data_final['radio_class'][i] != 0):

        manga_targets.append({"mangaid" : data_final['mangaid'][i], "fov" : 3.0})


drpall_path = "../manga_rmorph_data/drpall-v3_1_1.fits"

output_dir = Path("../manga_rmorph_data") / "Optical"
output_dir.mkdir(exist_ok=True, parents=True)
script_dir = Path(__file__).parent

surveys = {
    "sdss": {
        "pixscale": 0.396,
        "base_url": "https://skyserver.sdss.org/dr16/SkyServerWS/ImgCutout/getjpeg",
        "extension": "jpg",
        "params": "TaskName=Skyserver.Explore.Image&opt=G"
    },
    "sdsslegacy_jpeg": {
        "pixscale": 0.396,
        "base_url": "https://www.legacysurvey.org/viewer/cutout.jpg",
        "extension": "jpg",
        "params": "layer=sdss"
    },
    "sdsslegacy_fits": {
        "pixscale": 0.396,
        "base_url": "https://www.legacysurvey.org/viewer/cutout.fits",
        "extension": "fits",
        "params": "layer=sdss"
    },
    "legacy_jpeg": {
        "pixscale": 0.262,
        "base_url": "https://www.legacysurvey.org/viewer/jpeg-cutout",
        "extension": "jpg",
        "params": "layer=ls-dr9&bands=grz"
    },
    "legacy_fits": {
        "pixscale": 0.262,
        "base_url": "https://www.legacysurvey.org/viewer/fits-cutout",
        "extension": "fits",
        "params": "layer=ls-dr9&bands=grz"
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
    mangaid, fov = target["mangaid"], target.get("fov", 3.0)

    if "ra" in target and "dec" in target:
        ra, dec = target["ra"], target["dec"]
    else:
        ra, dec = get_coords(mangaid)
        if ra is None:
            continue
    
    mangaid_clean = mangaid.replace("-", "_")

    for survey_name, config in surveys.items():
        pixscale = config["pixscale"]
        size_pixels = int(fov * 60.0 / pixscale)
        filename = f"{mangaid_clean}_{survey_name}_fov{fov:.1f}arcmin_pixscale{pixscale:.3f}_npix{size_pixels}.{config['extension']}"

        if survey_name == "sdss":
            url = f"{config['base_url']}?{config['params']}&ra={ra:.6f}&dec={dec:.6f}&scale={pixscale:.3f}&width={size_pixels}&height={size_pixels}"
        else:
            url = f"{config['base_url']}?ra={ra:.6f}&dec={dec:.6f}&{config['params']}&pixscale={pixscale:.3f}&width={size_pixels}&height={size_pixels}"
        
        commands.append(f"wget -O '{output_dir / filename}' '{url}'")

script_path = script_dir / "download_optical_commands.sh"
with open(script_path, 'w') as f:
    f.write("#!/bin/bash\n# Generated wget commands for optical cutouts\n\n")
    f.write("\n".join(commands) + "\n")
os.chmod(script_path, 0o755)

print(f"Generated {len(commands)} commands -> {script_path.name}")