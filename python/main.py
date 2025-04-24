import pandas as pd
import GEDI, ATL08, gee_fun
from osgeo import gdal, ogr
import rasterio, ee, folium
import numpy as np
import matplotlib.pyplot as plt
import math
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive

if __name__ == '__main__':
    # gauth = GoogleAuth()
    # gauth.LocalWebserverAuth()  # client_secrets.json need to be in the same directory as the script
    # drive = GoogleDrive(gauth)
    # Data download:
    # GEDI L2A, slope, aspect (GEE)
    # ATL08 (Python)

    # root_dir = "D:\\project_icesat-2\\data\\"
    # gedi_dir = root_dir + "GEDI\\in\\raw_data\\jsonoutput\\"
    # las_dir = root_dir + "3DEP\\Tipp\\LAS"
    # atl08_dir = root_dir + "ATL08\\TippCo\\raw_data\\"
    area = "tipp"
    root_dir = "C:\\Users\\tian133\\OneDrive - purdue.edu\\Projects\\GEDI_ICESAT2\\data\\"
    root_dir = "E:\\OneDrive - purdue.edu\\Projects\\GEDI_ICESAT2\\data\\"
    gedi_dir = root_dir + "GEDI\\" + area + "\\"
    atl08_dir = root_dir + "ATL08\\" + area + "\\"
    if area == 'tipp':
        reference_dem_dir = root_dir + "others\\" + area + "\\dem_3dep.tif"
    elif area == 'mend':
        reference_dem_dir = "D:\\Project_ICESat-2\\data\\3DEP\\CA\\1-meter\\CA_1m3DEP.tif"
    srtm_dem_dir = root_dir + "others\\" + area + "srtm.tif"

    gedi_keys = ['lon', 'lat', 'elev_lowest', 'degrade', 'quality', 'sensitivity']
    atl08_keys = ['lon', 'lat', 'h_te_bestfit', 'al_track_slope', 'orbit_orient']

    in_epsg = "EPSG:4326"
    if area == "tipp":
        out_epsg = "EPSG:2968"
        out_epsg = "EPSG:2793" # Tipp
        dist_thresh = 100
    elif area == "mend":
        out_epsg = "EPSG:2226"
        out_epsg = "EPSG:2767"# Mend
        dist_thresh = 60
    regress_method = "random_forest_alongslopeatl08"
    feature_list = ['GEDI original height', 'degrade flag', 'quality flag', 'sensitivity',
                    'slope1', 'slope2']

    regress_method = "random_forest_noslope"
    feature_list = ['GEDI original height', 'degrade flag', 'quality flag', 'sensitivity']
    #
    # regress_method = "random forest (slopeatl08)"
    # feature_list = ['GEDI original height', 'degrade flag', 'quality flag', 'sensitivity',
    #                 'slope1', 'slope2', 'slope3', 'slope4']

    # GEDI.process(gedi_dir, gedi_keys, atl08_dir, atl08_keys, in_epsg, out_epsg, regress_method, dist_thresh,
    #              feature_list)
    # GEDI.DEM_diff(gedi_dir+"regression model\\"+regress_method+"\\",
    #               reference_dem_dir, dist_thresh, 30, "diff_RF30_3DEP.tif")
    # GEDI.DEM_diff(gedi_dir + "\\regression model\\" + regress_method + "\\",
    #               reference_dem_dir, dist_thresh, 90, "diff_RF90_3DEP.tif")
    # GEDI.DEM_diff(root_dir + "others\\" + area + "\\srtm.tif",
    #               reference_dem_dir, dist_thresh, 90, "diff_SRTM90_3DEP.tif")
    GEDI.evaluate(gedi_dir + "regression_model\\"+regress_method+"\\", "diff_DEM_3DEP_" + area + ".xlsx")

    print('done')
