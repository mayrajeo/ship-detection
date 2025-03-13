from geo2ml.data.tiling import Tiler
from geo2ml.data.cv import *
from pathlib import Path
import os
import rasterio as rio
import rasterio.mask as riomask
from shutil import rmtree
import geopandas as gpd
from itertools import product
import shapely
from fastcore.script import *
from tqdm import tqdm

def make_fold_grid(poly, num_edge, crs) -> gpd.GeoDataFrame:
    xmin, ymin, xmax, ymax = poly.bounds
    dx = (xmax-xmin)//num_edge
    dy = (ymax-ymin)//num_edge
    polys = []
    folds = []
    for x, y in product(range(0, num_edge), range(0, num_edge)):
        xm = xmin+x*dx
        ym = ymin+y*dy
        polys.append(shapely.geometry.box(*(xm, ym, xm+dx, ym+dy)))
        folds.append(f'fold_{(x-y)%num_edge + 1}')
    return gpd.GeoDataFrame(data={'fold': folds, 'geometry': polys}, crs=crs)

@call_parse
def make_revision_data(
    dataset_path:Path, # Path to the folder containing the annotated ships
    mosaic_path:Path, # Path to the folder containing the S2-mosaics
    outpath:Path, # Where to save the dataset and resulting files
    tilesize:int=320, # The size of tiles for the dataset
    overlap:int=0, # The overlap for tiles
    n_folds:int=5 # How many folds to create
):
    "Convert `gpkg` annotations to yolo-format data split into `n_folds` spatial folds"
    cats = ['boat']

    overlap = (overlap, overlap)
    tempdir = Path('temp_data')
    os.makedirs(tempdir, exist_ok=True)
    train_datasets = ['34VEM', '34WFT', '35VLG']
    datasets = [dataset_path/f'{d}.gpkg' for d in train_datasets]

    # Tile data
    for ds in datasets:
        print(f'Processing {ds.stem}')
        os.makedirs(tempdir/ds.stem, exist_ok=True)
        mosaics = [mosaic_path/ds.stem/f for f in os.listdir(mosaic_path/ds.stem) if f.endswith('tif')]
        for m in (mosaics):
            print(f'Processing {m.stem}')
            with rio.open(m) as src:
                bounds = src.bounds
                crs = src.crs

            outdir = tempdir/ds.stem/m.stem
            os.makedirs(outdir, exist_ok=True)
            tiler = Tiler(outdir, gridsize_x=tilesize,
                            gridsize_y=tilesize, overlap=overlap)
            tiler.tile_raster(m)
            tiler.tile_vector(str(ds), gpkg_layer=m.stem)
            im_dir = outdir/'images'
            vector_dir = outdir/'vectors'

            fold_grid = make_fold_grid(shapely.geometry.box(*bounds), n_folds, crs)

            # Move data into correct places
            for row in fold_grid.itertuples():
                fold_dir = outpath/f'{row.fold}'
                if not os.path.exists(fold_dir): 
                    os.makedirs(fold_dir)
                    os.makedirs(fold_dir/'images')
                    os.makedirs(fold_dir/'vectors')
                    os.makedirs(fold_dir/'labels')

                # Mask the raster so that only areas within fold are present
                for im in tqdm(os.listdir(im_dir)):
                    with rio.open(im_dir/im) as src:
                        im_bounds = shapely.geometry.box(*src.bounds)
                        if not im_bounds.intersects(row.geometry):
                            # Not within folds, no need to do anything
                            continue
                        clip_poly = im_bounds.intersection(row.geometry)
                        out_im, out_tfm = riomask.mask(src, [clip_poly], crop=False)
                        if out_im.max() == out_im.min(): 
                            # Empty image, not useful
                            continue

                        out_prof = src.profile

                    out_prof.update({'driver': 'GTiff',
                                     'height': out_im.shape[1],
                                     'width': out_im.shape[2],
                                     'transform': out_tfm})
                    outfn = f'{ds.stem}_{m.stem}_{im}'
                    with rio.open(fold_dir/'images'/outfn, 'w', **out_prof) as dest:
                        dest.write(out_im)
                    if os.path.exists(vector_dir/im.replace('tif', 'geojson')):
                        gdf = gpd.read_file(vector_dir/im.replace('tif', 'geojson'))
                        gdf = gdf.clip(clip_poly)
                        gdf.to_file(fold_dir/'vectors'/outfn.replace('tif', 'geojson'))
            rmtree(outdir)
    folds = [f for f in os.listdir(outpath) if os.path.isdir(outpath/f)]
    for fold in folds:
        shp_to_yolo(outpath/fold/'images', outpath/fold/'vectors', outpath/fold, 
                    'id', cats, ann_format='box', min_bbox_area=0)

        with open(outpath/f'{fold}.yaml', 'w') as dest:
            dest.write(
                "# Train/val/test sets as 1) dir: path/to/imgs, 2) file: path/to/imgs.txt, or 3: list: [path/to_imgs1, path_to_imgs2, ..]\n"
            )
            train_folds = [f for f in folds if f != fold]
            dest.write(f"path: {outpath} # dataset root dir \n")
            dest.write(f"train:  [{', '.join(train_folds)}] # train images (relative to 'path')\n")
            dest.write(f"val: [{fold}] # val images (relative to 'path')\n")
            dest.write(f"test: # test images (relative to 'path')\n")
            dest.write("\n# Classes\n")
            dest.write("names:\n")
            for n, c in enumerate(cats):
                dest.write(f"  {n}: {c}\n")