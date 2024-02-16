import rasterio.merge as rio_merge
from pathlib import Path 
import os

from fastcore.script import *

@call_parse
def merge_matching(
    datapath:Path, # Path that contains possible duplicate mosaics
):
    """Merge mosaics that are from the same timestep but possibly contain different areas
    and rename them to yyyymmdd.tif -format. If a timestep only contains one mosaic, just rename it
    """
    mosaics = [f for f in os.listdir(datapath) if f.endswith('.tif')]

    timesteps = {m.split('_')[2].split('T')[0] for m in mosaics}

    for t in timesteps:
        tmos = [m for m in mosaics if t in m]
        if len(tmos) == 1:
            # Only one mosaic present, just rename it
            os.rename(datapath/tmos[0], datapath/f'{t}.tif')
        else:
            # Multiple mosaics from same day -> Merge them with method='max'
            rio_merge.merge([datapath/t for t in tmos], dst_path=datapath/f'{t}.tif', method='max')
            for t in tmos:
                os.remove(datapath/t)