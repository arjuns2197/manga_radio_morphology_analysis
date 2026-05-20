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


manga_targets = funcs.get_target_info(data_final)

print(len(manga_targets))
n = 0

for i in range(len(manga_targets)):

    funcs.plot_cutout_BPT_Ha_O3_O3kinematics_rcontour(manga_targets[i])
    


print("Total cutouts plotted: ", n)


        