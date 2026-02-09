import ee
import os
import pandas as pd
import io
import requests
import rasterio
from rasterio.transform import from_bounds
import multiprocessing
import sys
import datetime
from pprint import pprint
import matplotlib.pyplot as plt

## GLOBAL CONSTANTS FOR THIS PROJECT
CLOUD_FILTER = 50


def see_if_all_image_bands_valid(band_values):
    for band in band_values:
        if band_values[band] != None:
            return True
    # if it made it all the way here, all values in this dict are None
    return False


"""
Set up GEE account and get the name of your GEE project.
"""


def open_gee_project(project: str):
    try:
        ee.Initialize(opt_url="https://earthengine-highvolume.googleapis.com")
    except Exception as e:
        ee.Authenticate()
        ee.Initialize(
            project=project, opt_url="https://earthengine-highvolume.googleapis.com"
        )


"""
Import assets from gee depending on which lake you want an image of
"""


def import_assets(lakeid: int, projectName: str) -> ee.FeatureCollection:
    LakeShp = ee.FeatureCollection(
        f"projects/{projectName}/assets/LAGOS_NY_4ha_Polygons_v2"
    )
    # print(f"size of dataset", LakeShp.size().getInfo())
    # print(lakeid)
    LakeShp = ee.FeatureCollection(LakeShp.filter(ee.Filter.eq("lagoslakei", lakeid)))
    # print(f"size of our lake shapefile", LakeShp.size().getInfo())
    # NewLakeShp = LakeShp.map(mapLakeFeature)
    return LakeShp


def mapLakeFeature(feature: ee.Feature):
    # print("Feature: ", feature)
    return feature.transform("EPSG:4326")


""""
MAIN Atmospheric Correction
Page, B.P., Olmanson, L.G. and Mishra, D.R., 2019. A harmonized image processing workflow using Sentinel-2/MSI and Landsat-8/OLI for mapping water clarity in optically variable lake systems. Remote Sensing of Environment, 231, p.111284.
https://github.com/Nateme16/geo-aquawatch-water-quality/blob/main/Atmospheric%20corrections/main_L8L9.ipynb
"""


def MAIN_S2A(img):
    JRC = ee.Image("JRC/GSW1_4/GlobalSurfaceWater")
    mask = JRC.select("occurrence").gt(0)
    pi = ee.Image(3.141592)
    # msi bands
    bands = ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"]

    # rescale
    rescale = img.select(bands).divide(10000)

    # tile footprint
    footprint = rescale.geometry()

    # dem
    DEM = ee.Image("USGS/SRTMGL1_003").clip(footprint)

    # ozone
    DU = ee.Image(300)
    # ee.Image(ozone.filterDate(startDate,endDate).filterBounds(footprint).mean())

    # Julian Day
    imgDate = ee.Date(img.get("system:time_start"))
    FOY = ee.Date.fromYMD(imgDate.get("year"), 1, 1)
    JD = imgDate.difference(FOY, "day").int().add(1)

    # earth-sun distance
    myCos = ((ee.Image(0.0172).multiply(ee.Image(JD).subtract(ee.Image(2)))).cos()).pow(
        2
    )
    cosd = myCos.multiply(pi.divide(ee.Image(180))).cos()
    d = ee.Image(1).subtract(ee.Image(0.01673)).multiply(cosd).clip(footprint)

    # sun azimuth
    SunAz = ee.Image.constant(img.get("MEAN_SOLAR_AZIMUTH_ANGLE")).clip(footprint)

    # sun zenith
    SunZe = ee.Image.constant(img.get("MEAN_SOLAR_ZENITH_ANGLE")).clip(footprint)
    cosdSunZe = SunZe.multiply(pi.divide(ee.Image(180))).cos()  # in degrees
    sindSunZe = SunZe.multiply(pi.divide(ee.Image(180))).sin()  # in degrees

    # sat zenith
    SatZe = ee.Image.constant(img.get("MEAN_INCIDENCE_ZENITH_ANGLE_B5")).clip(footprint)
    cosdSatZe = (SatZe).multiply(pi.divide(ee.Image(180))).cos()
    sindSatZe = (SatZe).multiply(pi.divide(ee.Image(180))).sin()

    # sat azimuth
    SatAz = ee.Image.constant(img.get("MEAN_INCIDENCE_AZIMUTH_ANGLE_B5")).clip(
        footprint
    )

    # relative azimuth
    RelAz = SatAz.subtract(SunAz)
    cosdRelAz = RelAz.multiply(pi.divide(ee.Image(180))).cos()

    # Pressure
    P = (
        ee.Image(101325)
        .multiply(
            ee.Image(1).subtract(ee.Image(0.0000225577).multiply(DEM)).pow(5.25588)
        )
        .multiply(0.01)
    )
    Po = ee.Image(1013.25)

    # esun
    ESUN = (
        ee.Image(
            ee.Array(
                [
                    ee.Image(img.get("SOLAR_IRRADIANCE_B1")),
                    ee.Image(img.get("SOLAR_IRRADIANCE_B2")),
                    ee.Image(img.get("SOLAR_IRRADIANCE_B3")),
                    ee.Image(img.get("SOLAR_IRRADIANCE_B4")),
                    ee.Image(img.get("SOLAR_IRRADIANCE_B5")),
                    ee.Image(img.get("SOLAR_IRRADIANCE_B6")),
                    ee.Image(img.get("SOLAR_IRRADIANCE_B7")),
                    ee.Image(img.get("SOLAR_IRRADIANCE_B8")),
                    ee.Image(img.get("SOLAR_IRRADIANCE_B8A")),
                    ee.Image(img.get("SOLAR_IRRADIANCE_B11")),
                    ee.Image(img.get("SOLAR_IRRADIANCE_B12")),
                ]
            )
        )
        .toArray()
        .toArray(1)
    )

    ESUN = ESUN.multiply(ee.Image(1))

    ESUNImg = ESUN.arrayProject([0]).arrayFlatten([bands])

    # create empty array for the images
    imgArr = rescale.select(bands).toArray().toArray(1)

    # pTOA to Ltoa
    Ltoa = imgArr.multiply(ESUN).multiply(cosdSunZe).divide(pi.multiply(d.pow(2)))

    # band centers
    bandCenter = (
        ee.Image(444)
        .divide(1000)
        .addBands(ee.Image(496).divide(1000))
        .addBands(ee.Image(560).divide(1000))
        .addBands(ee.Image(664).divide(1000))
        .addBands(ee.Image(704).divide(1000))
        .addBands(ee.Image(740).divide(1000))
        .addBands(ee.Image(782).divide(1000))
        .addBands(ee.Image(835).divide(1000))
        .addBands(ee.Image(865).divide(1000))
        .addBands(ee.Image(1613).divide(1000))
        .addBands(ee.Image(2202).divide(1000))
        .toArray()
        .toArray(1)
    )

    # ozone coefficients
    koz = (
        ee.Image(0.0040)
        .addBands(ee.Image(0.0244))
        .addBands(ee.Image(0.1052))
        .addBands(ee.Image(0.0516))
        .addBands(ee.Image(0.0208))
        .addBands(ee.Image(0.0112))
        .addBands(ee.Image(0.0079))
        .addBands(ee.Image(0.0021))
        .addBands(ee.Image(0.0019))
        .addBands(ee.Image(0))
        .addBands(ee.Image(0))
        .toArray()
        .toArray(1)
    )

    # Calculate ozone optical thickness
    Toz = koz.multiply(DU).divide(ee.Image(1000))

    # Calculate TOA radiance in the absense of ozone
    Lt = Ltoa.multiply(
        ((Toz))
        .multiply((ee.Image(1).divide(cosdSunZe)).add(ee.Image(1).divide(cosdSatZe)))
        .exp()
    )

    # Rayleigh optical thickness
    Tr = (
        (P.divide(Po))
        .multiply(ee.Image(0.008569).multiply(bandCenter.pow(-4)))
        .multiply(
            (
                ee.Image(1)
                .add(ee.Image(0.0113).multiply(bandCenter.pow(-2)))
                .add(ee.Image(0.00013).multiply(bandCenter.pow(-4)))
            )
        )
    )

    # Specular reflection (s- and p- polarization states)
    theta_V = ee.Image(0.0000000001)
    sin_theta_j = sindSunZe.divide(ee.Image(1.333))

    theta_j = sin_theta_j.asin().multiply(ee.Image(180).divide(pi))

    theta_SZ = SunZe

    R_theta_SZ_s = (
        (
            (theta_SZ.multiply(pi.divide(ee.Image(180)))).subtract(
                theta_j.multiply(pi.divide(ee.Image(180)))
            )
        )
        .sin()
        .pow(2)
    ).divide(
        (
            (
                (theta_SZ.multiply(pi.divide(ee.Image(180)))).add(
                    theta_j.multiply(pi.divide(ee.Image(180)))
                )
            )
            .sin()
            .pow(2)
        )
    )

    R_theta_V_s = ee.Image(0.0000000001)

    R_theta_SZ_p = (
        ((theta_SZ.multiply(pi.divide(180))).subtract(theta_j.multiply(pi.divide(180))))
        .tan()
        .pow(2)
    ).divide(
        (
            ((theta_SZ.multiply(pi.divide(180))).add(theta_j.multiply(pi.divide(180))))
            .tan()
            .pow(2)
        )
    )

    R_theta_V_p = ee.Image(0.0000000001)

    R_theta_SZ = ee.Image(0.5).multiply(R_theta_SZ_s.add(R_theta_SZ_p))

    R_theta_V = ee.Image(0.5).multiply(R_theta_V_s.add(R_theta_V_p))

    # Sun-sensor geometry
    theta_neg = ((cosdSunZe.multiply(ee.Image(-1))).multiply(cosdSatZe)).subtract(
        (sindSunZe).multiply(sindSatZe).multiply(cosdRelAz)
    )

    theta_neg_inv = theta_neg.acos().multiply(ee.Image(180).divide(pi))

    theta_pos = (cosdSunZe.multiply(cosdSatZe)).subtract(
        sindSunZe.multiply(sindSatZe).multiply(cosdRelAz)
    )

    theta_pos_inv = theta_pos.acos().multiply(ee.Image(180).divide(pi))

    cosd_tni = theta_neg_inv.multiply(pi.divide(180)).cos()  # in degrees

    cosd_tpi = theta_pos_inv.multiply(pi.divide(180)).cos()  # in degrees

    Pr_neg = ee.Image(0.75).multiply((ee.Image(1).add(cosd_tni.pow(2))))

    Pr_pos = ee.Image(0.75).multiply((ee.Image(1).add(cosd_tpi.pow(2))))

    # Rayleigh scattering phase function
    Pr = Pr_neg.add((R_theta_SZ.add(R_theta_V)).multiply(Pr_pos))

    # rayleigh radiance contribution
    denom = ee.Image(4).multiply(pi).multiply(cosdSatZe)
    Lr = (ESUN.multiply(Tr)).multiply(Pr.divide(denom))

    # rayleigh corrected radiance
    Lrc = Lt.subtract(Lr)
    LrcImg = Lrc.arrayProject([0]).arrayFlatten([bands])
    prcImg = Lrc.multiply(pi).multiply(d.pow(2)).divide(ESUN.multiply(cosdSunZe))
    prcImg = prcImg.arrayProject([0]).arrayFlatten([bands])

    # Aerosol Correction

    # Bands in nm
    bands_nm = (
        ee.Image(444)
        .addBands(ee.Image(496))
        .addBands(ee.Image(560))
        .addBands(ee.Image(664))
        .addBands(ee.Image(703))
        .addBands(ee.Image(740))
        .addBands(ee.Image(782))
        .addBands(ee.Image(835))
        .addBands(ee.Image(865))
        .addBands(ee.Image(0))
        .addBands(ee.Image(0))
        .toArray()
        .toArray(1)
    )

    # Lam in SWIR bands
    Lam_10 = LrcImg.select("B11")
    Lam_11 = LrcImg.select("B12")

    # Calculate aerosol type
    eps = (
        (((Lam_11).divide(ESUNImg.select("B12"))).log()).subtract(
            ((Lam_10).divide(ESUNImg.select("B11"))).log()
        )
    ).divide(ee.Image(2190).subtract(ee.Image(1610)))

    # Calculate multiple scattering of aerosols for each band
    Lam = (
        (Lam_11)
        .multiply(((ESUN).divide(ESUNImg.select("B12"))))
        .multiply(
            (eps.multiply(ee.Image(-1)))
            .multiply((bands_nm.divide(ee.Image(2190))))
            .exp()
        )
    )

    # diffuse transmittance
    trans = (
        Tr.multiply(ee.Image(-1))
        .divide(ee.Image(2))
        .multiply(ee.Image(1).divide(cosdSatZe))
        .exp()
    )

    # Compute water-leaving radiance
    Lw = Lrc.subtract(Lam).divide(trans)

    # water-leaving reflectance
    pw = Lw.multiply(pi).multiply(d.pow(2)).divide(ESUN.multiply(cosdSunZe))

    # remote sensing reflectance
    Rrs_S2A = pw.divide(pi).arrayProject([0]).arrayFlatten([bands]).slice(0, 9)

    # set negatives to null
    Rrs_S2A = Rrs_S2A.updateMask(Rrs_S2A.select("B1").gt(0)).multiply(mask)

    return Rrs_S2A.set("system:time_start", img.get("system:time_start"))


"""
Same thing for but Sentinel 2B
"""


def MAIN_S2B(img):
    JRC = ee.Image("JRC/GSW1_4/GlobalSurfaceWater")
    mask = JRC.select("occurrence").gt(0)
    pi = ee.Image(3.141592)
    # msi bands
    bands = ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"]

    # rescale
    rescale = img.select(bands).divide(10000)

    # tile footprint
    footprint = rescale.geometry()

    # dem
    DEM = ee.Image("USGS/SRTMGL1_003").clip(footprint)

    # ozone
    DU = ee.Image(300)

    # Julian Day
    imgDate = ee.Date(img.get("system:time_start"))
    FOY = ee.Date.fromYMD(imgDate.get("year"), 1, 1)
    JD = imgDate.difference(FOY, "day").int().add(1)

    # earth-sun distance
    myCos = ((ee.Image(0.0172).multiply(ee.Image(JD).subtract(ee.Image(2)))).cos()).pow(
        2
    )
    cosd = myCos.multiply(pi.divide(ee.Image(180))).cos()
    d = ee.Image(1).subtract(ee.Image(0.01673)).multiply(cosd).clip(footprint)

    # sun azimuth
    SunAz = ee.Image.constant(img.get("MEAN_SOLAR_AZIMUTH_ANGLE")).clip(footprint)

    # sun zenith
    SunZe = ee.Image.constant(img.get("MEAN_SOLAR_ZENITH_ANGLE")).clip(footprint)
    cosdSunZe = SunZe.multiply(pi.divide(ee.Image(180))).cos()  # in degrees
    sindSunZe = SunZe.multiply(pi.divide(ee.Image(180))).sin()  # in degrees

    # sat zenith
    SatZe = ee.Image.constant(img.get("MEAN_INCIDENCE_ZENITH_ANGLE_B5")).clip(footprint)
    cosdSatZe = (SatZe).multiply(pi.divide(ee.Image(180))).cos()
    sindSatZe = (SatZe).multiply(pi.divide(ee.Image(180))).sin()

    # sat azimuth
    SatAz = ee.Image.constant(img.get("MEAN_INCIDENCE_AZIMUTH_ANGLE_B5")).clip(
        footprint
    )

    # relative azimuth
    RelAz = SatAz.subtract(SunAz)
    cosdRelAz = RelAz.multiply(pi.divide(ee.Image(180))).cos()

    # Pressure
    P = (
        ee.Image(101325)
        .multiply(
            ee.Image(1).subtract(ee.Image(0.0000225577).multiply(DEM)).pow(5.25588)
        )
        .multiply(0.01)
    )
    Po = ee.Image(1013.25)

    # esun
    ESUN = (
        ee.Image(
            ee.Array(
                [
                    ee.Image(img.get("SOLAR_IRRADIANCE_B1")),
                    ee.Image(img.get("SOLAR_IRRADIANCE_B2")),
                    ee.Image(img.get("SOLAR_IRRADIANCE_B3")),
                    ee.Image(img.get("SOLAR_IRRADIANCE_B4")),
                    ee.Image(img.get("SOLAR_IRRADIANCE_B5")),
                    ee.Image(img.get("SOLAR_IRRADIANCE_B6")),
                    ee.Image(img.get("SOLAR_IRRADIANCE_B7")),
                    ee.Image(img.get("SOLAR_IRRADIANCE_B8")),
                    ee.Image(img.get("SOLAR_IRRADIANCE_B8A")),
                    ee.Image(img.get("SOLAR_IRRADIANCE_B11")),
                    ee.Image(img.get("SOLAR_IRRADIANCE_B12")),
                ]
            )
        )
        .toArray()
        .toArray(1)
    )

    ESUN = ESUN.multiply(ee.Image(1))

    ESUNImg = ESUN.arrayProject([0]).arrayFlatten([bands])

    # create empty array for the images
    imgArr = rescale.select(bands).toArray().toArray(1)

    # pTOA to Ltoa
    Ltoa = imgArr.multiply(ESUN).multiply(cosdSunZe).divide(pi.multiply(d.pow(2)))

    # band centers
    bandCenter = (
        ee.Image(442)
        .divide(1000)
        .addBands(ee.Image(492).divide(1000))
        .addBands(ee.Image(559).divide(1000))
        .addBands(ee.Image(665).divide(1000))
        .addBands(ee.Image(703).divide(1000))
        .addBands(ee.Image(739).divide(1000))
        .addBands(ee.Image(779).divide(1000))
        .addBands(ee.Image(833).divide(1000))
        .addBands(ee.Image(864).divide(1000))
        .addBands(ee.Image(1610).divide(1000))
        .addBands(ee.Image(2185).divide(1000))
        .toArray()
        .toArray(1)
    )

    # ozone coefficients
    koz = (
        ee.Image(0.0037)
        .addBands(ee.Image(0.0223))
        .addBands(ee.Image(0.1027))
        .addBands(ee.Image(0.0505))
        .addBands(ee.Image(0.0212))
        .addBands(ee.Image(0.0112))
        .addBands(ee.Image(0.0085))
        .addBands(ee.Image(0.0022))
        .addBands(ee.Image(0.0021))
        .addBands(ee.Image(0))
        .addBands(ee.Image(0))
        .toArray()
        .toArray(1)
    )

    # Calculate ozone optical thickness
    Toz = koz.multiply(DU).divide(ee.Image(1000))

    # Calculate TOA radiance in the absense of ozone
    Lt = Ltoa.multiply(
        ((Toz))
        .multiply((ee.Image(1).divide(cosdSunZe)).add(ee.Image(1).divide(cosdSatZe)))
        .exp()
    )

    # Rayleigh optical thickness
    Tr = (
        (P.divide(Po))
        .multiply(ee.Image(0.008569).multiply(bandCenter.pow(-4)))
        .multiply(
            (
                ee.Image(1)
                .add(ee.Image(0.0113).multiply(bandCenter.pow(-2)))
                .add(ee.Image(0.00013).multiply(bandCenter.pow(-4)))
            )
        )
    )

    # Specular reflection (s- and p- polarization states)
    theta_V = ee.Image(0.0000000001)
    sin_theta_j = sindSunZe.divide(ee.Image(1.333))

    theta_j = sin_theta_j.asin().multiply(ee.Image(180).divide(pi))

    theta_SZ = SunZe

    R_theta_SZ_s = (
        (
            (theta_SZ.multiply(pi.divide(ee.Image(180)))).subtract(
                theta_j.multiply(pi.divide(ee.Image(180)))
            )
        )
        .sin()
        .pow(2)
    ).divide(
        (
            (
                (theta_SZ.multiply(pi.divide(ee.Image(180)))).add(
                    theta_j.multiply(pi.divide(ee.Image(180)))
                )
            )
            .sin()
            .pow(2)
        )
    )

    R_theta_V_s = ee.Image(0.0000000001)

    R_theta_SZ_p = (
        ((theta_SZ.multiply(pi.divide(180))).subtract(theta_j.multiply(pi.divide(180))))
        .tan()
        .pow(2)
    ).divide(
        (
            ((theta_SZ.multiply(pi.divide(180))).add(theta_j.multiply(pi.divide(180))))
            .tan()
            .pow(2)
        )
    )

    R_theta_V_p = ee.Image(0.0000000001)

    R_theta_SZ = ee.Image(0.5).multiply(R_theta_SZ_s.add(R_theta_SZ_p))

    R_theta_V = ee.Image(0.5).multiply(R_theta_V_s.add(R_theta_V_p))

    # Sun-sensor geometry
    theta_neg = ((cosdSunZe.multiply(ee.Image(-1))).multiply(cosdSatZe)).subtract(
        (sindSunZe).multiply(sindSatZe).multiply(cosdRelAz)
    )

    theta_neg_inv = theta_neg.acos().multiply(ee.Image(180).divide(pi))

    theta_pos = (cosdSunZe.multiply(cosdSatZe)).subtract(
        sindSunZe.multiply(sindSatZe).multiply(cosdRelAz)
    )

    theta_pos_inv = theta_pos.acos().multiply(ee.Image(180).divide(pi))

    cosd_tni = theta_neg_inv.multiply(pi.divide(180)).cos()  # in degrees

    cosd_tpi = theta_pos_inv.multiply(pi.divide(180)).cos()  # in degrees

    Pr_neg = ee.Image(0.75).multiply((ee.Image(1).add(cosd_tni.pow(2))))

    Pr_pos = ee.Image(0.75).multiply((ee.Image(1).add(cosd_tpi.pow(2))))

    # Rayleigh scattering phase function
    Pr = Pr_neg.add((R_theta_SZ.add(R_theta_V)).multiply(Pr_pos))

    # rayleigh radiance contribution
    denom = ee.Image(4).multiply(pi).multiply(cosdSatZe)
    Lr = (ESUN.multiply(Tr)).multiply(Pr.divide(denom))

    # rayleigh corrected radiance
    Lrc = Lt.subtract(Lr)
    LrcImg = Lrc.arrayProject([0]).arrayFlatten([bands])
    prcImg = Lrc.multiply(pi).multiply(d.pow(2)).divide(ESUN.multiply(cosdSunZe))
    prcImg = prcImg.arrayProject([0]).arrayFlatten([bands])

    # Aerosol Correction

    # Bands in nm
    bands_nm = (
        ee.Image(442)
        .addBands(ee.Image(492))
        .addBands(ee.Image(559))
        .addBands(ee.Image(665))
        .addBands(ee.Image(703))
        .addBands(ee.Image(739))
        .addBands(ee.Image(779))
        .addBands(ee.Image(833))
        .addBands(ee.Image(864))
        .addBands(ee.Image(0))
        .addBands(ee.Image(0))
        .toArray()
        .toArray(1)
    )

    # Lam in SWIR bands
    Lam_10 = LrcImg.select("B11")  # = 0
    Lam_11 = LrcImg.select("B12")  # = 0

    # Calculate aerosol type
    eps = (
        (((Lam_11).divide(ESUNImg.select("B12"))).log()).subtract(
            ((Lam_10).divide(ESUNImg.select("B11"))).log()
        )
    ).divide(ee.Image(2190).subtract(ee.Image(1610)))

    # Calculate multiple scattering of aerosols for each band
    Lam = (
        (Lam_11)
        .multiply(((ESUN).divide(ESUNImg.select("B12"))))
        .multiply(
            (eps.multiply(ee.Image(-1)))
            .multiply((bands_nm.divide(ee.Image(2190))))
            .exp()
        )
    )

    # diffuse transmittance
    trans = (
        Tr.multiply(ee.Image(-1))
        .divide(ee.Image(2))
        .multiply(ee.Image(1).divide(cosdSatZe))
        .exp()
    )

    # Compute water-leaving radiance
    Lw = Lrc.subtract(Lam).divide(trans)

    # water-leaving reflectance
    pw = Lw.multiply(pi).multiply(d.pow(2)).divide(ESUN.multiply(cosdSunZe))

    # remote sensing reflectance
    Rrs_S2B = pw.divide(pi).arrayProject([0]).arrayFlatten([bands]).slice(0, 9)

    # set negatives to null
    Rrs_S2B = Rrs_S2B.updateMask(Rrs_S2B.select("B1").gt(0)).multiply(mask)

    return Rrs_S2B.set("system:time_start", img.get("system:time_start"))


""" 
Creating mask functions
These functions mask clouds based on the QA_PIXEL band (maskL8sr), select pixels 
that are >= 75% water (jrcMask), and a 30m buffer around roads to mask bridges (roadMask)
"""


# import S2 collection & join s2cloudless, add SCL band from SR
def get_s2_sr_cld_col(start_date, end_date, LakeShp) -> ee.ImageCollection:
    AOI = LakeShp
    # Import and filter s2cloudless.
    s2_cloudless_col = (
        ee.ImageCollection(
            "COPERNICUS/S2_CLOUD_PROBABILITY"
        )  # https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S2_CLOUD_PROBABILITY
        .filterBounds(AOI)
        .filterDate(start_date, end_date)
    )
    # print("s2_cloudless_col size: ", s2_cloudless_col.size().getInfo())

    # Import and filter s2 raw images.
    s2_raw_col = (
        ee.ImageCollection(
            "COPERNICUS/S2_HARMONIZED"
        )  # https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S2_HARMONIZED
        .filterBounds(AOI)
        .filterDate(start_date, end_date)
        .filter(ee.Filter.lte("CLOUDY_PIXEL_PERCENTAGE", CLOUD_FILTER))
    )
    # print("s2_raw_col size: ", s2_raw_col.size().getInfo())

    # Join the filtered s2cloudless collection to the S2 collection by the 'system:index' property.
    s2_sr_cld_col_eval = ee.ImageCollection(
        ee.Join.saveFirst("s2cloudless").apply(
            **{
                "primary": s2_raw_col,
                "secondary": s2_cloudless_col,
                "condition": ee.Filter.equals(
                    **{"leftField": "system:index", "rightField": "system:index"}
                ),
            }
        )
    )

    # add SCL (Scene Classification Map) band from SR
    # Before 2019, no stuff from USA in here
    s2_sr_col = (
        ee.ImageCollection(
            "COPERNICUS/S2_SR_HARMONIZED"
        )  # https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S2_SR_HARMONIZED
        .filterBounds(AOI)
        .filterDate(start_date, end_date)
        .filter(ee.Filter.lte("CLOUDY_PIXEL_PERCENTAGE", CLOUD_FILTER))
        # .select("SCL") # only take SCL band just to be safe
    )

    # print("s2_sr_col size: ", s2_sr_col.size().getInfo())

    s2_sr_cld_col_eval = ee.ImageCollection.combine(s2_sr_cld_col_eval, s2_sr_col)
    # print("s2_sr_cld_col_eval size: ", s2_sr_cld_col_eval.size().getInfo())

    s2_sr_cld_col_eval = s2_sr_cld_col_eval.select(
        "B1",
        "B2",
        "B3",
        "B4",
        "B5",
        "B6",
        "B7",
        "B8",
        "B8A",
        "B9",
        "B10",
        "B11",
        "B12",
        "QA10",
        "QA20",
        "QA60",
        "SCL",  # only band from S2_SR_HARMONIZED
    )

    return s2_sr_cld_col_eval


# add cloud bands
def add_cloud_bands(img):
    # Get s2cloudless image, subset the probability band.
    cld_prb = ee.Image(img.get("s2cloudless")).select("probability")

    # Condition s2cloudless by the probability threshold value.
    is_cloud = cld_prb.gt(CLOUD_FILTER).rename("clouds")

    # Add the cloud probability layer and cloud mask as image bands.
    return img.addBands(ee.Image([cld_prb, is_cloud]))


def add_shadow_bands(img):
    CLD_PRJ_DIST = 1
    NIR_DRK_THRESH = 0.15
    # Identify water pixels from the SCL band.
    not_water = img.select("SCL").neq(6)

    # Identify dark NIR pixels that are not water (potential cloud shadow pixels).
    SR_BAND_SCALE = 1e4
    dark_pixels = (
        img.select("B8")
        .lt(NIR_DRK_THRESH * SR_BAND_SCALE)
        .multiply(not_water)
        .rename("dark_pixels")
    )

    # Determine the direction to project cloud shadow from clouds (assumes UTM projection).
    shadow_azimuth = ee.Number(90).subtract(
        ee.Number(img.get("MEAN_SOLAR_AZIMUTH_ANGLE"))
    )

    # Project shadows from clouds for the distance specified by the CLD_PRJ_DIST input.
    cld_proj = (
        img.select("clouds")
        .directionalDistanceTransform(shadow_azimuth, CLD_PRJ_DIST * 10)
        .reproject(**{"crs": img.select(0).projection(), "scale": 100})
        .select("distance")
        .mask()
        .rename("cloud_transform")
    )

    # Identify the intersection of dark pixels with cloud shadow projection.
    shadows = cld_proj.multiply(dark_pixels).rename("shadows")

    # Add dark pixels, cloud projection, and identified shadows as image bands.
    return img.addBands(ee.Image([dark_pixels, cld_proj, shadows]))


def add_cld_shdw_mask(img):
    BUFFER = 50
    # Add cloud component bands.
    img_cloud = add_cloud_bands(img)

    # Add cloud shadow component bands.
    img_cloud_shadow = add_shadow_bands(img_cloud)

    # Combine cloud and shadow mask, set cloud and shadow as value 1, else 0.
    is_cld_shdw = (
        img_cloud_shadow.select("clouds").add(img_cloud_shadow.select("shadows")).gt(0)
    )

    # Remove small cloud-shadow patches and dilate remaining pixels by BUFFER input.
    # 20 m scale is for speed, and assumes clouds don't require 10 m precision.
    is_cld_shdw = (
        is_cld_shdw.focalMin(2)
        .focalMax(BUFFER * 2 / 20)
        .reproject(**{"crs": img.select([0]).projection(), "scale": 20})
        .rename("cloudmask")
    )

    # Add the final cloud-shadow mask to the image.
    return img_cloud_shadow.addBands(is_cld_shdw)


def apply_cld_shdw_mask(img):
    # Subset the cloudmask band and invert it so clouds/shadow are 0, else 1.
    not_cld_shdw = img.select("cloudmask").Not()
    # Subset reflectance bands and update their masks, return the result.
    return img.select("B.*").updateMask(not_cld_shdw)


def get_masked_coll(LakeShp, start_date, end_date):
    img = (
        get_s2_sr_cld_col(start_date, end_date, LakeShp)
        .map(add_cld_shdw_mask)
        .map(apply_cld_shdw_mask)
    )
    return img


# jrc water occurrence mask
def jrcMask(image):
    jrc = ee.Image("JRC/GSW1_4/GlobalSurfaceWater")
    # select only water occurence
    occurrence = jrc.select("occurrence")
    # selectonly water occurences of greater than 75%
    water_mask = occurrence.mask(occurrence.gt(50))
    return image.updateMask(water_mask)


# Creating 30m road buffer mask
def roadMask(image):
    roads = ee.FeatureCollection("TIGER/2016/Roads")

    # 30m road buffer
    def bufferPoly30(feature):
        return feature.buffer(30)

    Buffer = roads.map(bufferPoly30)

    # Convert 'areasqkm' property from string to number.
    def func_uem(feature):
        num = ee.Number.parse(ee.String(feature.get("linearid")))
        return feature.set("linearid", num)

    roadBuffer = Buffer.map(func_uem)
    roadRaster = roadBuffer.reduceToImage(["linearid"], ee.Reducer.first())
    # create an image with a constant value of one to apply roadmask to
    blank = ee.Image.constant(1)
    inverseMask = blank.updateMask(roadRaster)
    # get reverse mask to have everything but roads kept
    mask = inverseMask.mask().Not()
    return image.updateMask(mask)


"""
Import Collections
We call this in the get raster image function.
"""


def import_collections(masked_coll, filter_range, LakeShp) -> ee.Image:
    # Import Collections w/ Sentinel-2
    # MSI = ee.ImageCollection('COPERNICUS/S2_HARMONIZED')
    ozone = ee.ImageCollection("TOMS/MERGED")

    JRC = ee.Image("JRC/GSW1_4/GlobalSurfaceWater")
    # Process
    mask = JRC.select("occurrence").gt(0)
    # dump_m = masked_coll.filter(filter_range).filterBounds(LakeShp).first()

    # dump_m = dump_m.clip(LakeShp)

    # url_m = dump_m.getDownloadURL(
    #     {
    #         "format": "GEO_TIFF",
    #         "scale": 30,  #  increasing this makes predictions more blocky but reduces request size (smaller means more resolution tho!)
    #         "region": LakeShp.geometry(),
    #         "filePerBand": False,
    #         "crs": "EPSG:4326",
    #     }
    # )
    # print("URL of masked_coll_first in debug: ", url_m)
    # print("Cloudy pixel percentage of dump_m: ", dump_m.get("CLOUDY_PIXEL_PERCENTAGE").getInfo())

    FC_S2A = (
        masked_coll.filter(filter_range)
        .filterBounds(LakeShp)
        .filterMetadata("SPACECRAFT_NAME", "equals", "Sentinel-2A")
        .filterMetadata("CLOUDY_PIXEL_PERCENTAGE", "less_than", 30)
        .map(jrcMask)
        .map(roadMask)
        .sort("system:time_start")
    )

    FC_S2B = (
        masked_coll.filter(filter_range)
        .filterBounds(LakeShp)
        .filterMetadata("SPACECRAFT_NAME", "equals", "Sentinel-2B")
        .filterMetadata("CLOUDY_PIXEL_PERCENTAGE", "less_than", 30)
        .map(jrcMask)
        .map(roadMask)
        .sort("system:time_start")
    )

    # filter S2A by the filtered buffer and apply atm corr
    Rrs_S2A = FC_S2A.map(MAIN_S2A).sort("system:time_start")

    Rrs_S2A = Rrs_S2A.select(["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A"])

    # #filter S2B by the filtered buffer and apply atm corr
    Rrs_S2B = FC_S2B.map(MAIN_S2B).sort("system:time_start")

    Rrs_S2B = Rrs_S2B.select(["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A"])

    # print("Number of S2A images:", Rrs_S2A.size().getInfo())
    # print("Number of S2B images:", Rrs_S2B.size().getInfo())

    Rrs_S2_merged = Rrs_S2A.merge(Rrs_S2B)

    return Rrs_S2_merged


"""
Get Raster Image
ex. start_date = '2020-07-01'
    end_date = '2020-08-01'
"""


def get_image_and_date_from_image_collection(coll, index, shp):
    image = ee.Image(coll.toList(coll.size()).get(index))
    image_index = image.get("system:index").getInfo()
    date = ee.Date(image.get("system:time_start")).format("YYYY-MM-dd").getInfo()
    image = image.clip(shp)
    image = image.toFloat()
    return image, image_index, date


def get_raster(start_date, end_date, LakeShp, scale) -> ee.Image:
    date_range = ee.Filter.date(start_date, end_date)
    filter_range = ee.Filter.Or(date_range)
    masked_coll = get_masked_coll(LakeShp, start_date=start_date, end_date=end_date)
    # print("Masked Coll size: ", masked_coll.size().getInfo())  # if this is zero, nothing found
    merged_s2_coll = import_collections(masked_coll, filter_range, LakeShp)

    merged_s2_coll_len = merged_s2_coll.size().getInfo()

    if merged_s2_coll_len == 0:
        raise Exception("NO IMAGES FOUND")

    for i in range(0, merged_s2_coll_len):
        image, image_index, date = get_image_and_date_from_image_collection(
            merged_s2_coll, i, LakeShp
        )

        min_value = image.reduceRegion(
            reducer=ee.Reducer.min(),
            geometry=LakeShp.geometry(),  # or your specific geometry
            scale=scale,
            maxPixels=1e9,
            crs="EPSG:4326",
        ).getInfo()

        if see_if_all_image_bands_valid(min_value):
            return image, image_index, date
    # if it made it here, all have blank images (due to NASA JPL aggressive cloud alterer/filter)
    raise Exception("IMAGE IS ALL BLANK :(((")


def inspect_raster(image):
    num_bands = len(image.getInfo()["bands"])
    if num_bands != 9:
        print("This image has less than 9 bands.")
    else:
        print(
            "Image has 9 bands, good to go ahead and add 4 more constant bands and run model."
        )
        info = image.getInfo()
        pprint(info)
        bands = info["bands"]
        width = -1
        height = -1
        for band in bands:
            dimensions = band["dimensions"]
            print(
                f"Band: {band['id']}, Width: {dimensions[0]} pixels, Height: {dimensions[1]} pixels"
            )
            width = dimensions[0]
            height = dimensions[1]


def get_width(image) -> int:
    info = image.getInfo()
    bands = info["bands"]
    width = int(bands[-1]["dimensions"][0])
    return width


def get_height(image) -> int:
    info = image.getInfo()
    bands = info["bands"]
    height = int(bands[-1]["dimensions"][1])
    return height


def visualize(tif_path: str):
    # Open the GeoTIFF file
    with rasterio.open(tif_path) as src:
        # Read the number of bands and the dimensions
        num_bands = src.count
        height = src.height
        width = src.width
        tags = src.tags()
        title = f"Date: {tags["date"]}, ID: {tags["id"]}, Scale: {tags["scale"]}\n"

        print(f"Number of bands: {num_bands}")
        # print(f"Dimensions: {width} x {height}")

        # Read the entire image into a numpy array (bands, height, width)
        img = src.read()
        # Display each band separately
        fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(15, 10))

        for i, ax in enumerate(axes.flatten()):
            if i < num_bands:
                ax.imshow(img[i, :, :], cmap="gray")  # Display each band separately
                ax.set_title(f"Band {i+1}")
                ax.axis("off")
                # print(img[i, :, :])
        plt.tight_layout()
        plt.suptitle(title, fontsize=24)  # Super title for all the subplots!
        plt.show()


def export_raster_main(
    out_dir: str,
    out_filename: str,
    project: str,
    lakeid: int,
    start_date: str,
    end_date: str,
    scale: int,
    shouldVisualize: bool = False,
):

    # print("LakeID: ", lakeid)
    LakeShp = import_assets(lakeid, project)  # get shape of lake
    # print("Lakeshp fetched")

    # get raster of lake, inspect to make sure you have 9 bands
    image, image_index, date = get_raster(
        start_date=start_date, end_date=end_date, LakeShp=LakeShp, scale=scale
    )

    # print("Getting download url...")
    # get download URL
    url = image.getDownloadURL(
        {
            "format": "GEO_TIFF",
            "scale": scale,  #  increasing this makes predictions more blocky but reduces request size (smaller means more resolution tho!)
            "region": LakeShp.geometry(),
            "filePerBand": False,
            "crs": "EPSG:4326",
        }
    )

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # export!
    out_filepath = os.path.join(out_dir, out_filename)
    # download image, and then view metadata with rasterio
    # print("Downloading raster...")

    response = requests.get(url)
    with open(out_filepath, "wb") as f:
        f.write(response.content)

    new_metadata = {
        "date": date,
        "id": lakeid,
        "scale": scale,
        "image_index": image_index[2:],  # remove preface for collection
    }
    if image_index[:2] == "1_":  # first collection in merge, sentinel2a
        new_metadata["satellite"] = "sentinel2a"
    elif image_index[:2] == "2_":  # second collection in merge, sentinel2b
        new_metadata["satellite"] = "sentinel2b"
    else:
        raise Exception("Can't determine the specific satellite from image index")
    # print(new_metadata)

    with rasterio.open(out_filepath, "r+") as dst:
        dst.update_tags(**new_metadata)

    # print(f"Image saved to {out_filepath}")

    if shouldVisualize:
        print("Saved image metadata: ", new_metadata)
        visualize(out_filepath)


if __name__ == "__main__":
    if len(sys.argv) != 8:
        print(
            "python export_raster.py <out_dir> <project> <lakeid> <start_date> <end_date> <scale> <out_filename>"
        )
        sys.exit(1)

    out_dir = sys.argv[1]
    project = sys.argv[2]
    lakeid = int(sys.argv[3])
    start_date = sys.argv[4]  # STR, in format YYYY-MM-DD
    end_date = sys.argv[5]  # STR, in format YYYY-MM-DD
    scale = int(sys.argv[6])
    out_filename = sys.argv[7]

    open_gee_project(project=project)

    export_raster_main(
        out_dir=out_dir,
        out_filename=out_filename,
        project=project,
        lakeid=lakeid,
        start_date=start_date,
        end_date=end_date,
        scale=scale,
        shouldVisualize=True,
    )
