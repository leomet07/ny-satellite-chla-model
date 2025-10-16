# Setup dotenv
from dotenv import load_dotenv
import os
import pandas as pd
import time
import numpy as np
from matplotlib import pyplot as plt

load_dotenv()

NAN_SUBSTITUTE_CONSANT = -99999
RANDOM_STATE = 621

# Creating the proper dataframe (csv) from other datasets
training_df_path = os.getenv("INSITU_CHLA_TRAINING_DATA_PATH")  # training data
lagosid_path = os.getenv(
    "CCRI_LAKES_WITH_LAGOSID_PATH"
)  # needed to get lagoslakeid for training data entries
lulc_path = os.getenv("LAGOS_LAKE_INFO_PATH")  # General lake info (for constants)
lake_area_csv_path = os.getenv(
    "LAKE_AREA_CSV_PATH"
)  # Maps lagoslakeid to area in square kilometers (from GIS shapefile)


def prepare_data(df_path, lagosid_path, lulc_path, lake_area_csv_path):
    # read csvs
    df = pd.read_csv(df_path)
    lagosid = pd.read_csv(lagosid_path)

    # select relevant columns from lagosid
    lagosid = lagosid[["lagoslakei"]]
    df = pd.concat([lagosid, df], axis=1)

    df = df[df["chl_a"] < 2000]

    # create chl-a mathematical inputs
    df["NDCI"] = (df["703"] - df["665"]) / (df["703"] + df["665"])
    df["NDVI"] = (df["864"] - df["665"]) / (df["864"] + df["665"])
    df["3BDA"] = ((df["493"] - df["665"]) / (df["493"] + df["665"])) - df["560"]

    # create rough vol based on depth & SA
    df["lake_vol"] = df["SA"] * df["Max.depth"]

    # load lagos-ne iws data, merged with dataframe
    iws_lulc = pd.read_csv(lulc_path)

    # summarize percent developed land
    iws_lulc["iws_nlcd2011_pct_dev"] = (
        iws_lulc["iws_nlcd2011_pct_21"]
        + iws_lulc["iws_nlcd2011_pct_22"]
        + iws_lulc["iws_nlcd2011_pct_23"]
        + iws_lulc["iws_nlcd2011_pct_24"]
    )

    # summarize percent agricultural land
    iws_lulc["iws_nlcd2011_pct_ag"] = (
        iws_lulc["iws_nlcd2011_pct_81"] + iws_lulc["iws_nlcd2011_pct_82"]
    )

    # create new dataframe with the two variables
    iws_human = iws_lulc[["lagoslakeid", "iws_nlcd2011_pct_dev", "iws_nlcd2011_pct_ag"]]

    iws_human = iws_human.rename(
        columns={
            "lagoslakeid": "lagoslakei",
            "iws_nlcd2011_pct_dev": "pct_dev",
            "iws_nlcd2011_pct_ag": "pct_ag",
        }
    )
    # At this point, iws_human contains 50k rows (ALL lagos lakes, including the 4400 NYS lakes)
    # print("one of 4k lakes after: ", iws_human.loc[lambda x: x['lagoslakei'] == 57171])
    # print("# of rows in iws_human: ", len(iws_human))

    # left join df with iws_human, which cuts # of lakes in df to 360 (14k entries of in situ readings)
    df = df.merge(iws_human, on="lagoslakei")

    sa_sq_km_df = pd.read_csv(lake_area_csv_path)
    df = df.merge(sa_sq_km_df, on="lagoslakei")
    return df, iws_human, sa_sq_km_df  # Array of pandas dataframes


def prepared_cleaned_data(unclean_data):  # Returns CUDF df
    # unclean_data = unclean_data[(unclean_data['satellite'] == "LC08") | (unclean_data['satellite'] == "LC09")] # if this line uncommented, only Landsat8/9 satellites
    # unclean_data = unclean_data[(unclean_data['satellite'] == "1") | (unclean_data['satellite'] == "2")] # if this line uncommented, only Sentinel2A/2B satellites

    # Filter to everything that is less than 200 ug/L
    unclean_data = unclean_data[
        unclean_data["chl_a"] < 200
    ]  # most values are 0-100, remove the crazy 4,000 outlier
    unclean_data = unclean_data.fillna(NAN_SUBSTITUTE_CONSANT)
    return unclean_data  # Now it is clean


def reduce_to_training_columns(all_data_cleaned):
    input_cols = [
        "443",
        "493",
        "560",
        "665",
        "703",
        "740",
        "780",
        "834",
        "864",
        "SA_SQ_KM",
        "pct_dev",
        "pct_ag",
    ]
    all_data_cleaned = all_data_cleaned[["chl_a"] + input_cols]

    return all_data_cleaned


# define constants for new bands
def get_constants(lakeid):
    lagos_lookup_table_filtered = lagos_lookup_table[
        lagos_lookup_table["lagoslakei"] == lakeid
    ]
    sa_sq_km_lookup_table_filtered = sa_sq_km_lookup_table[
        sa_sq_km_lookup_table["lagoslakei"] == lakeid
    ]

    SA_SQ_KM = sa_sq_km_lookup_table_filtered["SA_SQ_KM"].iloc[0]
    pct_dev = lagos_lookup_table_filtered["pct_dev"].iloc[
        0
    ]  # Lagos look up table should have this
    pct_ag = lagos_lookup_table_filtered["pct_ag"].iloc[
        0
    ]  # Lagos look up table should have this

    return SA_SQ_KM, pct_dev, pct_ag


all_data_uncleaned, lagos_lookup_table, sa_sq_km_lookup_table = prepare_data(
    training_df_path, lagosid_path, lulc_path, lake_area_csv_path
)  # Returns insitu points merged with lagoslookup table AND lagoslookup table for all non-insitu lakes as well
all_data_cleaned = prepared_cleaned_data(all_data_uncleaned)

print(
    "Satellites (Sentinel 2A/B: 1/2, Landsat 8/9: LC08/LC09): ",
    np.unique(all_data_cleaned["satellite"]),
)
# print("# of insitu lakes: ", len(np.unique(all_data_cleaned["lagoslakei"] )))

all_data_cleaned.to_csv("all_data_cleaned.csv")

# print("mean date diff: ", all_data_cleaned["date_diff"].mean())
# print("std date diff: ", all_data_cleaned["date_diff"].std())
# all_data_cleaned_three_days = all_data_cleaned[all_data_cleaned["date_diff"] <= 6]
# print("mean three date diff: ", all_data_cleaned_three_days["date_diff"].mean())
# print("std three  date diff: ", all_data_cleaned_three_days["date_diff"].std())
# print("Number of rows in three date: ", len(all_data_cleaned_three_days))
# # all_data_cleaned = all_data_cleaned_three_days

training_data = reduce_to_training_columns(all_data_cleaned)
print(training_data)


def get_number_of_trainig_rows_for_max_date_diff(max_date_diff: int):
    return len(all_data_cleaned[all_data_cleaned["date_diff"] <= max_date_diff])


if __name__ == "__main__":
    lagos_lookup_table.to_csv("generated_lagoslookuptable.csv")

    print("Max diff: ", all_data_cleaned["date_diff"].max())
    plt.rcParams.update(
        {"font.size": 18, "font.family": "sans-serif", "font.sans-serif": "Arial"}
    )

    plt.figure("# of Training Pts vs Max Date Diff", (7, 7))
    xs = list(range(1, 7 + 1))
    number_of_training_pts = list(map(get_number_of_trainig_rows_for_max_date_diff, xs))
    print(number_of_training_pts)
    plt.plot(xs, number_of_training_pts)
    plt.xlabel("Max Date Diff")
    plt.ylabel("# of training pts")
    plt.tight_layout()
    plt.savefig(
        os.path.join("charts_etr", "number_of_training_pts_versus_max_date_diff.jpg"),
        bbox_inches="tight",
        dpi=400,
    )
    plt.show()
