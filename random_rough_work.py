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
import statistics

data = funcs.match_manga_to_radio()

# path_data_final = "../manga_rmorph_data/manga_with_vlass_lotss_first_with_power_law.fits"
# data_final = fits.open(path_data_final)[1].data

# manga_targets = funcs.get_target_info(data_final)

# for i in range(len(manga_targets)):
    
#     if manga_targets[i]['mangaid'] == "1-617323":
#         n = i


# print(manga_targets[n]['mangaid'])
# print(funcs.categorise_central_ionization_type_NII(manga_targets[n]))

# print(manga_targets[n]['mangaid'])