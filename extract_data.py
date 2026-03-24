import h5py
import os
from dotenv import load_dotenv
import numpy as np
import rasterio
from rasterio.transform import from_origin
from pyproj import Transformer
from shapely.geometry import box
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt

if __name__ == "__main__":

    load_dotenv()

    new_root = os.getenv("SO2SAT_DIR") #'path to so2sat-lzc42 v4'
    dataset = "training" # training, testing, validation

    for dataset in ["training", "testing", "validation"]:
        img_fname = f'{dataset}.h5'
        aux_fname = f'{dataset}_geo.h5'
        PROJECT_CRS = "EPSG:4326"

        h5_file = os.path.join(new_root,img_fname)

        with h5py.File(h5_file, 'r') as fid1:
            sen1_shape = fid1['sen1'].shape
            print("sen1_data shape:", sen1_shape)

        with h5py.File(h5_file, 'r') as fid1:
            sen1_data = np.array(fid1['sen1'])  # (N, patchSize, patchSize, 8)
            sen2_data = np.array(fid1['sen2'])
            label_data = np.array(fid1['label'])

        h5_file_aux = os.path.join(new_root,aux_fname)

        with h5py.File(h5_file_aux, 'r') as fid1:
            tfw_new = np.array(fid1['tfw'])
            epsg_new = np.array(fid1['epsg'])

        _, labels = np.where(label_data == 1)
        labels += 1
        lcz_classes, lcz_counts = np.unique(labels, return_counts=True)

        bbox_lst = []
        start_idx = 0

        for sample_idx in range(start_idx, label_data.shape[0]):
            
            sample_sen1 = np.transpose(sen1_data[sample_idx - start_idx], (2, 0, 1))
            sample_sen2 = np.transpose(sen2_data[sample_idx - start_idx], (2, 0, 1))

            sample_label = label_data[sample_idx - start_idx]

            origin_x = tfw_new[sample_idx - start_idx, 4]
            pixel_width = abs(tfw_new[sample_idx - start_idx, 0])
            origin_y = tfw_new[sample_idx - start_idx, 5]
            pixel_height = abs(tfw_new[sample_idx - start_idx, 3])
            transform = from_origin(origin_x, origin_y, pixel_width, pixel_height)
            epsg_code = int(epsg_new[sample_idx - start_idx, 0])

            sen1_outfile = os.path.join(new_root, f'{dataset}/sentinel1/sen1_patch_{sample_idx:06d}.tif')    
            with rasterio.open(
                sen1_outfile, 'w',
                driver='GTiff',
                height=sample_sen1.shape[1],
                width=sample_sen1.shape[2],
                count=sample_sen1.shape[0],
                dtype=sample_sen1.dtype,
                crs=f'EPSG:{epsg_code}',
                transform=transform
            ) as dst:
                dst.write(sample_sen1)

            sen2_outfile = os.path.join(new_root, f'{dataset}/sentinel2/sen2_patch_{sample_idx:06d}.tif')    
            with rasterio.open(
                sen2_outfile, 'w',
                driver='GTiff',
                height=sample_sen2.shape[1],
                width=sample_sen2.shape[2],
                count=sample_sen2.shape[0],
                dtype=sample_sen2.dtype,
                crs=f'EPSG:{epsg_code}',
                transform=transform
            ) as dst:
                dst.write(sample_sen2)

            height = sample_sen1.shape[1]
            width = sample_sen1.shape[2]
            xmin = origin_x
            xmax = origin_x + width * pixel_width
            ymax = origin_y
            ymin = origin_y + height * pixel_height  # pixel_height usually negative
            # Reproject bbox
            transformer = Transformer.from_crs(
                f'EPSG:{epsg_code}',
                PROJECT_CRS,
                always_xy=True
            )
            x1, y1 = transformer.transform(xmin, ymin)
            x2, y2 = transformer.transform(xmax, ymax)
            patch_bbox = [x1, y1, x2, y2]
            print(f"Saving sample {sample_idx}: {labels[sample_idx]} - {patch_bbox}")
            bbox_lst.append(patch_bbox)
        
        
        polygon_lst = [box(*bbox) for bbox in bbox_lst]
        bbox_gdf = gpd.GeoDataFrame(geometry=polygon_lst, crs=PROJECT_CRS)
        bbox_gdf["patch_id"] = bbox_gdf.index
        bbox_gdf['dataset'] = dataset
        bbox_gdf["LCZ_class"] = labels.tolist()
        bbox_gdf.to_file(os.path.join(new_root, f'{dataset}/labels.gpkg'), index=False, driver='GPKG')