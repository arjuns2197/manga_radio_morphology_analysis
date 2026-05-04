import funcs
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.wcs import WCS
from astropy.cosmology import FlatLambdaCDM
from astropy.table import Table
from pathlib import Path
from PIL import Image
import warnings
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

warnings.filterwarnings('ignore')

path_data_final = "../manga_rmorph_data/manga_with_vlass_lotss_first.fits"
data_final = fits.open(path_data_final)[1].data

# Check manga maps data model
property_hdu_ext = 30  # EMLINE GFLUX
property_array_index = 16 # H\alpha

manga_targets = funcs.get_target_info(data_final)

print(len(manga_targets))

for i in range(len(manga_targets)):
    funcs.plot_map_cutout_rcontour(manga_targets[i],"O3",property_hdu_ext, property_array_index)