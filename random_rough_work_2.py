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

funcs.match_manga_to_radio()

path_data_final = "../manga_rmorph_data/manga_with_vlass_lotss_first.fits"
data_final = fits.open(path_data_final)[1].data

