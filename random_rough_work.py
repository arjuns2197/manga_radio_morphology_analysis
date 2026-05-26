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

# data = funcs.match_manga_to_radio()

path_data_final = "../manga_rmorph_data/manga_with_vlass_lotss_first.fits"
data_final = fits.open(path_data_final)[1].data

print(data_final.columns)
n = 0
type1 = "Little Ionization"
type2 = "Little Ionization"

for i in range(len(data_final)):

    if data_final['central_ionisation_type_NII'][i] == type1:

        if data_final['central_ionisation_type_SII'][i] == type2:

            n = n+1

print(n)