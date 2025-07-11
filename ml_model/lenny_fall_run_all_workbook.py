# Setup dotenv
from dotenv import load_dotenv
import os
import sys

load_dotenv()

from pymongo import MongoClient
import numpy as np
import pandas as pd
import rasterio
import matplotlib.pyplot as plt
import time
from datetime import datetime
from pathlib import Path
import json
from pyproj import Proj
import uuid
from tqdm import tqdm
import gc
import model_data

IS_CPU_MODE = os.getenv("IS_CPU_MODE").lower() == "true"
IS_IN_PRODUCTION_MODE = os.getenv("IS_PRODUCTION_MODE").lower() == "true"
VISUALIZE_PREDICTIONS = os.getenv("VISUALIZE_PREDICTIONS").lower() == "true"
print("IS_CPU_MODE: ", IS_CPU_MODE)
print("IS_IN_PRODUCTION_MODE: ", IS_IN_PRODUCTION_MODE)

print("\nCalling model training module...")
if IS_CPU_MODE:
    import cpu_model_training as model_training
else:
    import model_training
andrew_model = model_training.andrew_model

print("Finished calling model training module!\n")

if IS_IN_PRODUCTION_MODE:
    mongo_client = MongoClient(os.getenv("MONGO_CONNECTION_URI"))
    mongo_prod_db = mongo_client["prod"]
    mongo_lakes_collection = mongo_prod_db.lakes
    mongo_spatial_predictions_collection = mongo_prod_db.spatial_predictions

    all_lakes = list(mongo_lakes_collection.find({}))

session_uuid = str(uuid.uuid4())
print("Current session id: ", session_uuid)


input_tif_folder = os.getenv("INPUT_TIF_FOLDER") # Specify the folder inside of the tar
paths = os.listdir(input_tif_folder)
print("Number of files to run: ", len(paths))

png_out_folder = os.path.join("all_png_out", f"png_out_{session_uuid}")
if not os.path.exists(png_out_folder):
    os.makedirs(png_out_folder)
    
tif_out_folder = os.path.join("all_tif_out", f"tif_out_{session_uuid}")
if not os.path.exists(tif_out_folder):
    os.makedirs(tif_out_folder)

session_statues_path = "session_statuses/"
if not os.path.exists(session_statues_path):
    os.makedirs(session_statues_path)

error_paths = []


def add_suffix_to_filename_at_tif_path(filename : str, suffix : str):
    parts = filename.split(".")
    newfilename =  parts[0] + f"_{suffix}." + ".".join(parts[1:])

    to_tif_folder_path = os.path.join(tif_out_folder, os.path.basename(newfilename))

    return to_tif_folder_path


def modify_tif(input_tif : str, SA_SQ_KM_FROM_SHAPEFILE_constant : float, pct_dev_constant: float, pct_ag_constant : float) -> str:
    with rasterio.open(input_tif) as src:
        raster_data = src.read()
        profile = src.profile  # Get the profile of the existing raster
        tags = src.tags()
    
    # print(tags)
    satellite = tags["satellite"]
    # print("satellite: ", satellite)

    # create new bands
    SA_SQ_KM_band = np.full_like(raster_data[0], SA_SQ_KM_FROM_SHAPEFILE_constant, dtype=raster_data.dtype)
    # Max_depth_band = np.full_like(raster_data[0], Max_depth_constant, dtype=raster_data.dtype)
    pct_dev_band = np.full_like(raster_data[0], pct_dev_constant, dtype=raster_data.dtype)
    pct_ag_band = np.full_like(raster_data[0], pct_ag_constant, dtype=raster_data.dtype)

    # update profile to reflect additional bands
    profile.update(count=12)  # (update count to include original bands + 4 new bands)

    # output GeoTIFF file
    modified_tif = add_suffix_to_filename_at_tif_path(input_tif, "modified")
    if satellite.startswith("sentinel"):
        bands_to_fill = 0
    elif satellite.startswith("landsat"):
        bands_to_fill = 9 - 5 # Landsat has 5, not 9 bands, so fill 4 bands
    else:
        raise Exception(f'Satellite "{satellite}" predictions not implemented yet.')
    
    with rasterio.open(modified_tif, 'w', **profile) as dst:
        # write original bands
        for i in range(1, raster_data.shape[0] + 1):
            dst.write(raster_data[i-1], indexes=i)

        bands_to_fill = 9 - 5 
        for i in range(raster_data.shape[0] + 1, raster_data.shape[0] + 1 + bands_to_fill):
            # print(f"Writing null band... at ({i})")
            null_band = np.full_like(raster_data[0], model_data.NAN_SUBSTITUTE_CONSANT, dtype=raster_data.dtype)
            dst.write(null_band, indexes=i)

        # # write additional bands
        dst.write(SA_SQ_KM_band, indexes=raster_data.shape[0] + bands_to_fill + 1)
        dst.write(pct_dev_band, indexes=raster_data.shape[0] + bands_to_fill + 2)
        dst.write(pct_ag_band, indexes=raster_data.shape[0] + bands_to_fill + 3)

        dst.transform = src.transform
        dst.crs = src.crs

    # print(f"Created {modified_tif} with the four extra bands data from constants")
    return modified_tif


def predict(input_tif : str, lakeid: int, tags, display = True):
    modified_tif = add_suffix_to_filename_at_tif_path(input_tif, "modified")
    with rasterio.open(modified_tif) as src:
        raster_data = src.read()
        profile = src.profile

    n_bands, n_rows, n_cols = raster_data.shape
    n_samples = n_rows * n_cols
    raster_data_2d = raster_data.transpose(1, 2, 0).reshape((n_samples,n_bands))

    non_finite_val_mask = ~np.isfinite(raster_data[0]) # if first band at that pixel is nan, inf, or -inf, usually rest are too (helps remove "garbage" val from output later)
    
    raster_data_2d[~np.isfinite(raster_data_2d)] = model_data.NAN_SUBSTITUTE_CONSANT # Replace with NAN_SUB_CONSTANT or mean of general, but this pixels output  will later be removed anyway

    # perform the prediction
    predictions = andrew_model.predict(raster_data_2d)

    if not IS_IN_PRODUCTION_MODE:
        df = pd.DataFrame(raster_data_2d, columns=model_training.X_test.columns)
        df["lagoslakeid"] = lakeid
        df["pred"] = predictions
        df = df.drop_duplicates()
        output_tif_csv = add_suffix_to_filename_at_tif_path(input_tif, "predicted") + '.csv'
        df.to_csv(output_tif_csv)
        print("csv saved to " +  os.path.join(os.getcwd(), output_tif_csv))

    # print(predictions)

    # reshape the predictions back to the original raster shape
    predictions_raster = predictions.reshape(n_rows, n_cols)

    predictions_raster[non_finite_val_mask] = np.nan # if the input value was originally nan, -inf, or, ignore its (normal-seeming) output and make it nan

    if not IS_IN_PRODUCTION_MODE:
        print("Min predictions: ", np.nanmin(predictions_raster))
        print("Max predictions: ", np.nanmax(predictions_raster))
        print("Avg predictions: ", np.nanmean(predictions_raster))
        print("STD predictions: ", np.nanstd(predictions_raster))

    # save the prediction result as a new raster file
    output_tif = add_suffix_to_filename_at_tif_path(input_tif, "predicted")

    profile.update(count=1) # only 1 output band, chl_a concentration!
    with rasterio.open(output_tif, 'w', **profile) as dst:
        dst.write(predictions_raster, 1) # write to band 1
        dst.update_tags(**tags)

    # plot the result
    if display:
        min_cbar_value = 0
        max_cbar_value = 60
        plt.imshow(predictions_raster, cmap='viridis', vmin=min_cbar_value, vmax=max_cbar_value)
        plt.colorbar()
        plt.title(f"Predicted values for lake{lakeid} on {tags["date"]}")
        plt.show()

    return output_tif, predictions_raster


def save_png(input_tif, out_folder, predictions_raster, date, scale, display=True):
    # Masking of NaNs already happens in predict function, so no need to mask here
    min_value = 0
    max_value = 60
    increment = 5

    fig = plt.figure(figsize=(10, 8))
    plt.imshow(predictions_raster, cmap='viridis', interpolation='none', vmin=min_value, vmax=max_value)
    plt.axis('off')
    stem = Path(input_tif).stem

    # values = np.arange(min_value, max_value + increment, increment)
    # cbar = plt.colorbar()
    # cbar.set_label(f'Predicted chlorophyll-A in ug/L \n ({date}, scale: {scale})')
    # cbar.set_ticks(values)
    # cbar.set_ticklabels([str(val) for val in values])

    # png filename
    output_png = stem + ".png"
    output_png_path = os.path.join(out_folder, output_png)
    # save the png
    plt.savefig(output_png_path, format='png', bbox_inches='tight', pad_inches=0, transparent=True)
    if display:
        plt.show()
    else:
        plt.close(fig)
    return output_png_path


def upload_spatial_map(lakeid : int, raster_image_path: str, display_image_path : str, datestr : str, corners : list, scale : int):
    # Step 1: Find Lakeid

    filtered_list = list(filter(lambda lake: lake["lagoslakeid"] == lakeid, all_lakes))
    if len(filtered_list) == 0:
        raise Exception(f'No lake was found with lagoslakeid of "{lakeid}"')

    lake_db_id = filtered_list[0]["_id"]

    dateiso = datetime.strptime(datestr, '%Y-%m-%d').isoformat()
    # Step 2

    body = {
        "raster_image" :  os.path.basename(raster_image_path),
        "display_image" : os.path.basename(display_image_path),
        "date" : dateiso, # utc
        "corner1latitude": corners[0][0],
        "corner1longitude": corners[0][1],
        "corner2latitude": corners[1][0],
        "corner2longitude": corners[1][1],
        "scale" : scale,
        "session_uuid" : session_uuid,
        "lake" : lake_db_id,
        "lagoslakeid" : lakeid
    }
    mongo_spatial_predictions_collection.insert_one(body)

for path_tif in tqdm(paths):
    path_tif = os.path.join(input_tif_folder, path_tif)
    try:
        # print(f"Opening {path_tif  }")
        with rasterio.open(path_tif) as raster:
            tags = raster.tags()
            id = int(tags["id"])
            date = tags["date"] # date does NOT do anything here, just for title
            scale = tags["scale"] # scale does NOT do anything here, just for title

            top_left = raster.transform * (0, 0)
            bottom_right = raster.transform * (raster.width, raster.height)
            crs = raster.crs

        p = Proj(crs)
        # Output is in the format: (lat, long)
        corner1 = list(p(top_left[0], top_left[1], inverse=True)[::-1])
        corner2 = list(p(bottom_right[0], bottom_right[1], inverse=True)[::-1])
        corners = [corner1, corner2]
        # print("id: ", id, " date: ", date, " scale: ", scale, " corners: ", corners)

        # Get constants
        SA_SQ_KM_constant, pct_dev_constant, pct_ag_constant = model_data.get_constants(id)
        # print(f"Constants based on id({id}): ", SA_SQ_KM_constant, pct_dev_constant, pct_ag_constant)

        modified_path_tif = modify_tif(path_tif, SA_SQ_KM_constant, pct_dev_constant, pct_ag_constant)

        output_tif, predictions_loop = predict(path_tif, id, tags, display = VISUALIZE_PREDICTIONS)

        if VISUALIZE_PREDICTIONS:
            print("Output tif: ", os.path.join(os.getcwd(),output_tif))

        output_path_png = save_png(path_tif, png_out_folder, predictions_loop, date, scale, display = False)
        
        if IS_IN_PRODUCTION_MODE:
            upload_spatial_map(id, output_tif, output_path_png, date, corners, scale)

        with open(os.path.join(session_statues_path, f"successes_{session_uuid}.status.txt"), "a") as file_obj:
            file_obj.write(path_tif +"\n")

        gc.collect() # Clear memory after prediction
    except Exception as e:
        print("Error: ", e)
        error_paths.append(path_tif)

print(f"Successfully finished {len(paths)} uploads with {len(error_paths)} errors")
print("Session ID: ", session_uuid)

with open(os.path.join(session_statues_path, f"error_paths_{session_uuid}.json"), "w") as file:
    file.write(json.dumps(error_paths))
