from pathlib import Path
import os
import rioxarray as rxr
import pandas as pd
import geopandas as gpd
from pyproj import Transformer
from shapely.geometry import box

if __name__ == "__main__":

    from dotenv import load_dotenv

    PROJECT_CRS = "EPSG:4326"
    so2sat_dir = Path(os.getenv("SO2SAT_DIR"))

    dataset_lst = ["training", "validation", "testing"]
    bbox_info_lst = []

    for dataset in dataset_lst:
        print(f"Processing {dataset} set...")
        imagery = so2sat_dir / dataset / "sentinel2"
        patches_lst = list(imagery.glob("*.tif"))
        patches_lst.sort()
        print(f"Found {len(patches_lst)} patches in {dataset} set...")
        bbox_lst = []

        for file in patches_lst:
            print(f"Processing {file} patch in {dataset} set...")
            patch_da = rxr.open_rasterio(file)
            patch_crs = patch_da.rio.crs
            patch_utm_bbox = patch_da.rio.bounds()
            transformer = Transformer.from_crs(patch_crs, PROJECT_CRS)
            coord1 = transformer.transform(patch_utm_bbox[0], patch_utm_bbox[1])
            coord2 = transformer.transform(patch_utm_bbox[2], patch_utm_bbox[3])
            patch_bbox = [coord1[1], coord1[0], coord2[1], coord2[0]]
            bbox_lst.append(patch_bbox)

        polygon_lst = [box(*bbox) for bbox in bbox_lst]
        bbox_gdf = gpd.GeoDataFrame(geometry=polygon_lst, crs=PROJECT_CRS)
        bbox_gdf["patch_id"] = bbox_gdf.index
        bbox_gdf['dataset'] = dataset
        bbox_gdf.to_file(so2sat_dir / f'{dataset}_reference_rxr.gpkg', index=False, driver='GPKG')
        bbox_info_lst.append(bbox_gdf)

    bbox_info_gdf = pd.concat(bbox_info_lst)
    bbox_info_gdf.to_file(so2sat_dir / 'patches_reference_rxr.geojson', index=False, driver='GeoJSON')
    bbox_info_gdf.to_file(so2sat_dir / 'patches_reference_rxr.gpkg', index=False, driver='GPKG')