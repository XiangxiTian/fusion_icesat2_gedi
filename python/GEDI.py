import json
import os
import pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import reproject, Resampling, calculate_default_transform
from rasterio.enums import Resampling
import time, math
import scipy.stats as stats
import termtables as tt
import termtables as tt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
# from pyGEDI import *
import general, regression, gee_fun, GEDI_dataset


# def GEDI_L2A_download(bbox, version, out_dir, username="txx1220", password="Wsnd1220"):
#     """
#     bbox = [ul_lat,ul_lon,lr_lat,lr_lon]
#     """
#     session = sessionNASA(username, password)
#     product_2A = 'GEDI02_A'
#     gediDownload(out_dir, product_2A, version, bbox, session)


# def json_merge(indir, key, outjson):
#     """merge multiple json file into one json file
#     indir: directory of the multiple json file
#     key: key search in file name
#     outjson: output location
#     """
#     gedijsonfiles = [indir + "/" + f for f in os.listdir(indir) if f.endswith('.json') and key in f]
#     data = []
#     for g in gedijsonfiles:
#         print(g)
#         # for f in glob.glob(g):
#         with open(g, 'r') as infile:
#             sourcedata = json.load(infile)
#             features = sourcedata['features']
#             data.extend(features)

#     outpath = indir + '/' + outjson
#     print(outpath)
#     # data=json.dumps(data)
#     with open(outpath, 'w') as outfile:
#         json.dump({"features": data}, outfile)


# def json_load(indir, injson, keys):
#     with open(indir + injson, 'r') as f:
#         source = json.load(f)

#     features = source['features']
#     data = []
#     for feature in features:
#         tmp = []
#         for key in keys:
#             tmp.append(feature['properties'][key])
#         data.append(tmp)
#     return np.asarray(data, dtype=float)


# def qua_filter(indir, injson, outjson_good, outjson_bad, sensitivity):
#     """
#     Filter data according to the sensitivity feature
#     :param indir: directory of the input json file
#     :param injson: file name of the input json file
#     :param outjson_good: remained good quality data
#     :param outjson_bad: filtered bad quality data
#     :param sensitivity: sensitivity level
#     """
#     good_data = []
#     bad_data = []
#     with open(indir + '/' + injson, 'r') as f:
#         source = json.load(f)
#     features = source['features']
#     print('Before filtering:' + str(len(features)))
#     for feature in features:
#         if feature['properties']['quality_flag'] != 0 and feature['properties']['sensitivity'] >= sensitivity:
#             good_data.append(feature)
#         else:
#             bad_data.append(feature)
#     with open(indir + '/' + outjson_good, 'w') as f:
#         json.dump({"features": good_data}, f)
#     print('Good data left:' + str(len(good_data)))
#     print('Bad data filtered:' + str(len(bad_data)))
#     with open(indir + '/' + outjson_bad, 'w') as f:
#         json.dump({"features": bad_data}, f)
#     return good_data, bad_data


# def z_interp(XY, las_file, kdtree, k):
#     if len(XY.shape) == 1:
#         n = 1
#     else:
#         n = XY.shape[0]
#     num_returns = las_file.num_returns
#     return_num = las_file.return_num
#     z = lidarPC.scaled_dimension(las_file, 'z')
#     z = z[num_returns == return_num]
#     intensity = las_file.intensity
#     intensity = intensity[num_returns == return_num]
#     # geoDF.csr='EPSG:2968'
#     dist, idx = kdtree.query(XY, k)
#     dist = dist.reshape((n, k))
#     idx = idx.reshape((n, k))
#     wList = 1 / dist
#     zList = z[idx]
#     z_interp = np.sum(wList * zList, axis=1) / np.sum(wList, axis=1)
#     z_nn = zList[:, 0]
#     # output = np.asarray([z_interp, z_nn])
#     return z_interp, z_nn


# def get_Z_3dep(XY_gedi, las_dir, k):
#     X = XY_gedi[:, 0]
#     Y = XY_gedi[:, 1]
#     las_kdtree_dir = las_dir + '/kdtree'
#     file_list_kdtree = [o for o in os.listdir(las_kdtree_dir) if o.endswith('.pickle')]
#     file_list_las = [o for o in os.listdir(las_dir) if o.endswith('.las')]
#     boundaries = np.loadtxt(las_dir + '/boundaries.txt', delimiter=',')
#     las_idx = -1 * np.ones(np.max(XY_gedi.shape))
#     z = np.zeros(XY_gedi.shape)
#     for i, bounds in enumerate(boundaries):
#         start = time.time()
#         if i % 10 == 0:
#             print('Checking in file {}/{}'.format(i, len(file_list_las)))
#         # tmp = np.bitwise_and(bounds[0] <= X, X <= bounds[2], las_idx == -1)
#         # tmp = np.bitwise_and(tmp, bounds[1] <= Y, Y <= bounds[3])
#         tmp = np.array(bounds[0] <= X) & np.array(X <= bounds[2]) & np.array(las_idx == -1) \
#               & np.array(bounds[1] <= Y) & np.array(Y <= bounds[3])
#         boolArray = np.where(tmp)[0]
#         if boolArray.size == 0:
#             continue
#         else:
#             las_idx[boolArray] = i
#             with open(las_kdtree_dir + '/' + file_list_kdtree[i], 'rb') as file:
#                 tree = pickle.load(file)
#             las_file = laspy.read(las_dir + '/' + file_list_las[i])
#             z[boolArray, 0], z[boolArray, 1] = z_interp(XY_gedi[boolArray, :], las_file, tree, k)
#             end = time.time()
#             print('timer: ', end - start)
#     return z


# def load_GEDI_pars(indir, gedi_Z, gedi_lonlat, gedi_XY, atl08_XY, atl08_Z, distance_threshold, k):
#     n_points = gedi_XY.shape[0]
#     if os.path.exists(indir + "sentinel-2.csv"):
#         df = pd.read_csv(indir + "sentinel-2.csv")
#         NDVI = df['NDVI'].to_numpy()
#         EVI = df['EVI'].to_numpy()
#     else:
#         gedi_tmp = np.concatenate(
#             (np.arange(0, n_points, 1).reshape((n_points, 1)), gedi_lonlat, gedi_Z.reshape((n_points, 1))),
#             axis=-1)
#         gedi_df = pd.DataFrame(gedi_tmp, columns=['id', 'longitude', 'latitude', 'elev_lowestmode'])
#         interval = 20000
#         out_df = pd.DataFrame()
#         for i in range(math.ceil(gedi_df.shape[0] / interval)):
#             if i == math.ceil(gedi_df.shape[0] / interval) - 1:
#                 tmp = gedi_df[i * interval:]
#             else:
#                 tmp = gedi_df[i * interval:(i + 1) * interval]
#             tmp_out = gee_fun.extract_from_ImageCollection(tmp, 'sentinel-2', ["2019-09-20", "2019-09-25"])
#             out_df = out_df.append(tmp_out, ignore_index=True)
#         out_df.to_csv(indir + "sentinel-2.csv")
#         NDVI = out_df['NDVI'].to_numpy()
#         EVI = out_df['EVI'].to_numpy()


#     if os.path.exists(indir + "slope.csv"):
#         df = pd.read_csv(indir + "slope.csv")
#         slope = df['slope'].to_numpy()
#     else:
#         gedi_tmp = np.concatenate(
#             (np.arange(0, n_points, 1).reshape((n_points, 1)), gedi_lonlat, gedi_Z.reshape((n_points, 1))),
#             axis=-1)
#         gedi_df = pd.DataFrame(gedi_tmp, columns=['id', 'longitude', 'latitude', 'elev_lowestmode'])
#         interval = 20000
#         out_df = pd.DataFrame()
#         for i in range(math.ceil(gedi_df.shape[0] / interval)):
#             if i == math.ceil(gedi_df.shape[0] / interval) - 1:
#                 tmp = gedi_df[i * interval:]
#             else:
#                 tmp = gedi_df[i * interval:(i + 1) * interval]
#             tmp_out = gee_fun.extract_from_ImageCollection(tmp, 'slope', ["2019-09-20", "2019-09-25"])
#             out_df = out_df.append(tmp_out, ignore_index=True)
#         out_df.to_csv(indir + "slope.csv")
#         slope = out_df['slope'].to_numpy()


#     if os.path.exists(indir + "aspect.csv"):
#         df = pd.read_csv(indir + "aspect.csv")
#         aspect = df['aspect'].to_numpy()
#     else:
#         gedi_tmp = np.concatenate(
#             (np.arange(0, n_points, 1).reshape((n_points, 1)), gedi_lonlat, gedi_Z.reshape((n_points, 1))),
#             axis=-1)
#         gedi_df = pd.DataFrame(gedi_tmp, columns=['id', 'longitude', 'latitude', 'elev_lowestmode'])
#         interval = 20000
#         out_df = pd.DataFrame()
#         start = time.time()
#         for i in range(math.ceil(gedi_df.shape[0] / interval)):
#             if i == math.ceil(gedi_df.shape[0] / interval) - 1:
#                 tmp = gedi_df[i * interval:]
#             else:
#                 tmp = gedi_df[i * interval:(i + 1) * interval]
#             tmp_out = gee_fun.extract_from_ImageCollection(tmp, 'aspect', ["2019-09-20", "2019-09-25"])
#             out_df = out_df.append(tmp_out, ignore_index=True)
#         out_df.to_csv(indir + "aspect.csv")
#         print("extraction time:", (time.time() - start) / 60)
#         aspect = out_df['aspect'].to_numpy()


#     if os.path.exists(indir + "elevation.csv"):
#         df = pd.read_csv(indir + "elevation.csv")
#         elevation = df['elevation'].to_numpy()
#     else:
#         gedi_tmp = np.concatenate(
#             (np.arange(0, n_points, 1).reshape((n_points, 1)), gedi_lonlat, gedi_Z.reshape((n_points, 1))),
#             axis=-1)
#         gedi_df = pd.DataFrame(gedi_tmp, columns=['id', 'longitude', 'latitude', 'elev_lowestmode'])
#         interval = 20000
#         out_df = pd.DataFrame()
#         for i in range(math.ceil(gedi_df.shape[0] / interval)):
#             if i == math.ceil(gedi_df.shape[0] / interval) - 1:
#                 tmp = gedi_df[i * interval:]
#             else:
#                 tmp = gedi_df[i * interval:(i + 1) * interval]
#             tmp_out = gee_fun.extract_from_ImageCollection(tmp, 'elevation', ["2019-09-20", "2019-09-25"])
#             out_df = out_df.append(tmp_out, ignore_index=True)
#         out_df.to_csv(indir + "elevation.csv")
#         elevation = out_df['elevation'].to_numpy()


#     if os.path.exists(indir + "land_cover.csv"):
#         df = pd.read_csv(indir + "land_cover.csv")
#         lc = df['LC_Type1'].to_numpy()
#     else:
#         gedi_tmp = np.concatenate(
#             (np.arange(0, n_points, 1).reshape((n_points, 1)), gedi_lonlat, gedi_Z.reshape((n_points, 1))),
#             axis=-1)
#         gedi_df = pd.DataFrame(gedi_tmp, columns=['id', 'longitude', 'latitude', 'elev_lowestmode'])
#         interval = 20000
#         out_df = pd.DataFrame()
#         start = time.time()
#         for i in range(math.ceil(gedi_df.shape[0] / interval)):
#             if i == math.ceil(gedi_df.shape[0] / interval) - 1:
#                 tmp = gedi_df[i * interval:]
#             else:
#                 tmp = gedi_df[i * interval:(i + 1) * interval]
#             tmp_out = gee_fun.extract_from_ImageCollection(tmp, 'LC_Type1', ["2019-09-20", "2019-09-25"])
#             out_df = out_df.append(tmp_out, ignore_index=True)
#         out_df.to_csv(indir + "land_cover.csv")
#         end = time.time()
#         print("Time for extraction of land cover: ", end - start)
#         lc = out_df['LC_Type1'].to_numpy()


#     if os.path.exists(indir + "interpolated_elip_height_from_ATL08_" + str(distance_threshold) + ".csv"):
#         gedi_z_atl08_tmp = np.loadtxt(
#             indir + "interpolated_elip_height_from_ATL08_" + str(distance_threshold) + ".csv", delimiter=',')
#         gedi_z_atl08_interp = gedi_z_atl08_tmp[:, 1]
#         gedi_valid_indices = gedi_z_atl08_tmp[:, 0].astype(np.int)
#         # gedi_atl08_dist, gedi_atl08_idx = general.nn(gedi_XY, atl08_XY, k)
#         # gedi_valid_indices1 = np.where(gedi_atl08_dist[:, 0] <= distance_threshold)[0]
#         # gedi_valid_indices = np.intersect1d(gedi_valid_indices0, gedi_valid_indices1)
#         # gedi_z_atl08_interp = gedi_z_atl08_interp[gedi_valid_indices]
#     else:
#         gedi_atl08_dist, gedi_atl08_idx = general.nn(gedi_XY, atl08_XY, k)
#         gedi_valid_indices0 = np.where(gedi_atl08_dist[:, 0] <= distance_threshold)[0]


#         print("finding the natural neighbors")
#         gedi_z_atl08_interp = general.natural_neighbor_points(atl08_XY, atl08_Z, gedi_XY[gedi_valid_indices0, :])
#         gedi_valid_indices1 = np.where(~np.isnan(gedi_z_atl08_interp))[0]
#         gedi_valid_indices = gedi_valid_indices0[gedi_valid_indices1]
#         gedi_z_atl08_interp = gedi_z_atl08_interp[gedi_valid_indices1]
#         print("done finding the natural neighbors")
#         # gedi_atl08_dist = gedi_atl08_dist[gedi_valid_indices, :]
#         # gedi_atl08_idx = gedi_atl08_idx[gedi_valid_indices]
#         #
#         # gedi_z_atl08 = atl08_Z[gedi_atl08_idx]
#         # gedi_z_atl08_interp = general.simple_idw(gedi_atl08_dist, gedi_z_atl08)
#         # gedi_z_atl08_tmp = np.stack((gedi_valid_indices, gedi_z_atl08_interp), axis=-1)


#         # gedi_z_atl08_interp = general.natural_neighbor_points(atl08_XY, atl08_Z, gedi_XY)
#         # gedi_valid_indices1 = np.where(~np.isnan(gedi_z_atl08_interp))[0]
#         # gedi_valid_indices = np.intersect1d(gedi_valid_indices0, gedi_valid_indices1)
#         gedi_z_atl08_tmp = np.stack((gedi_valid_indices, gedi_z_atl08_interp), axis=-1)
#         np.savetxt(indir + "interpolated_elip_height_from_ATL08_" + str(distance_threshold) + ".csv",
#                    gedi_z_atl08_tmp, delimiter=',')


#     NDVI = NDVI.reshape((NDVI.shape[0], 1))
#     EVI = EVI.reshape((EVI.shape[0], 1))
#     slope = slope.reshape((slope.shape[0], 1))
#     aspect = aspect.reshape((aspect.shape[0], 1))
#     land_cover = lc.reshape((lc.shape[0], 1))
#     ortho_height_3dep1m = elevation.reshape((elevation.shape[0], 1))
#     return NDVI, EVI, slope, aspect, land_cover, ortho_height_3dep1m, gedi_z_atl08_interp, gedi_valid_indices


# def load_atl08_pars(indir, atl08_data, in_epsg, out_epsg):
#     if os.path.exists(indir + 'geoid_height.csv'):
#         atl08_geoid_height = np.loadtxt(indir + 'geoid_height.csv', delimiter=',')
#     else:
#         egm_path = "C:\\Users\\tian133\\OneDrive - purdue.edu\\Projects\\GEDI_ICESAT2\\data\\egm\\geoids\\egm2008-5.pgm"
#         atl08_geoid_height = general.get_geoid_height(atl08_data[:, 0], atl08_data[:, 1], egm_path)
#         np.savetxt(indir + 'geoid_height.csv', atl08_geoid_height, delimiter=',')
#     atl08_XY = general.point_project(atl08_data[:, 0], atl08_data[:, 1], in_epsg, out_epsg)


#     if os.path.exists(indir + 'ortho_height_from_3dep1m.csv'):
#         ortho_height_3dep1m = np.loadtxt(indir + 'ortho_height_from_3dep1m.csv', delimiter=',')
#     else:
#         ortho_height_3dep1m = general.get_data_ee(atl08_data[:, 0:2], 'elevation')
#         np.savetxt(indir + 'ortho_height_from_3dep1m.csv', ortho_height_3dep1m, delimiter=',')
#     atl08_EH_3dep_1m = ortho_height_3dep1m + atl08_geoid_height
#     return atl08_XY, atl08_EH_3dep_1m


# def GEDI_analysis(indir, gedi_tmp, pars_list, analysis, cap_analysis, footprint_visual, cap_visual0, cap_visual1,
#                   classify, minZ, maxZ):
#     if analysis:
#         general.data_analysis("GEDI", gedi_tmp, pars_list, plot=True,
#                               save_path=indir + cap_analysis + ".png")
#     if footprint_visual:
#         gedi_ellipsoidH = gedi_tmp[:, 4]
#         gedi_EH_3dep_1m = gedi_tmp[:, 5]
#         gedi_height_error = (gedi_ellipsoidH - gedi_EH_3dep_1m)
#         if classify:
#             gedi_height_classified = np.copy(gedi_ellipsoidH)
#             gedi_height_classified[gedi_height_classified >= maxZ] = math.ceil(maxZ) + 1
#             gedi_height_classified[gedi_height_classified <= minZ] = math.ceil(minZ) - 1
#             tmp = np.concatenate(
#                 (gedi_tmp[:, 0:2], gedi_height_classified.reshape((gedi_height_classified.shape[0], 1))),
#                 axis=1)

#             gedi_height_error_classified = np.copy(gedi_height_error)
#             gedi_height_error_classified[gedi_height_error_classified >= 20] = 21
#             gedi_height_error_classified[gedi_height_error_classified <= -20] = -21
#             tmp1 = np.concatenate((gedi_tmp[:, 0:2],
#                                    gedi_height_error_classified.reshape((gedi_height_error_classified.shape[0], 1))),
#                                   axis=1)
#         else:
#             tmp = gedi_tmp[:, [0, 1, 4]]
#             tmp1 = np.concatenate((gedi_tmp[:, 0:2], gedi_height_error.reshape((gedi_height_error.shape[0], 1))),
#                                   axis=1)
#         general.footprint_visualization(tmp, ['lon', 'lat', 'height'], 2,
#                                         caption=cap_visual0,
#                                         save_path=indir + cap_visual0 + ".html")
#         general.footprint_visualization(tmp1, ["lon", "lat", "height error (m)"], 2,
#                                         caption=cap_visual1,
#                                         save_path=indir + cap_visual1 + ".html")


# def atl08_analysis(indir, atl08_tmp, pars_list, analysis, cap_analysis, footprint_visual, cap_visual0, cap_visual1):
#     if analysis:
#         general.data_analysis("ATL08", atl08_tmp, pars_list, plot=True,
#                               save_path=indir + cap_analysis + ".png")
#     if footprint_visual:
#         general.footprint_visualization(atl08_tmp[:, [0, 1, 4]], ['lon', 'lat', 'height'], 2,
#                                         caption=cap_visual0,
#                                         save_path=indir + cap_visual0 + ".html")
#         height_error = (atl08_tmp[:, 4] - atl08_tmp[:, 5])
#         tmp = np.concatenate((atl08_tmp[:, :2], height_error.reshape((atl08_tmp.shape[0], 1))), axis=1)
#         general.footprint_visualization(tmp, ["lon", "lat", "height error (m)"], 2,
#                                         caption=cap_visual1,
#                                         save_path=indir + cap_visual1 + ".html")


# def reproj_match(infile, match, outfile):
#     """Reproject a file to match the shape and projection of existing raster.


#     Parameters
#     ----------
#     infile : (string) path to input file to reproject
#     match : (string) path to raster with desired shape and projection
#     outfile : (string) path to output file tif
#     """
#     # open input
#     with rasterio.open(infile) as src:
#         src_transform = src.transform
#         # open input to match
#         with rasterio.open(match) as match:
#             dst_crs = match.crs
#             # calculate the output transform matrix
#             dst_transform, dst_width, dst_height = calculate_default_transform(
#                 src.crs,  # input CRS
#                 dst_crs,  # output CRS
#                 match.width,  # input width
#                 match.height,  # input height
#                 *match.bounds,  # unpacks input outer boundaries (left, bottom, right, top)
#             )


#         # set properties for output
#         dst_kwargs = src.meta.copy()
#         dst_kwargs.update({"crs": dst_crs,
#                            "transform": dst_transform,
#                            "width": dst_width,
#                            "height": dst_height,
#                            "nodata": 0})
#         print("Coregistered to shape:", dst_height, dst_width, '\n Affine', dst_transform)
#         # open output
#         with rasterio.open(outfile, "w", **dst_kwargs) as dst:
#             # iterate through bands and write using reproject function
#             for i in range(1, src.count + 1):
#                 reproject(
#                     source=rasterio.band(src, i),
#                     destination=rasterio.band(dst, i),
#                     src_transform=src.transform,
#                     src_crs=src.crs,
#                     dst_transform=dst_transform,
#                     dst_crs=dst_crs,
#                     resampling=Resampling.nearest)


# def dem_match(target_fpath, reference_fpath, outfile):
#     CR = COREG(reference_fpath, target_fpath, path_out=outfile)
#     CR.calculate_spatial_shifts()
#     CR.correct_shifts()
#     return None



# def DEM_diff(dem_dir, reference_dem_dir, save_path):
#     with rasterio.open(dem_dir) as src:
#         dem = src.read(1)
#         dst_crs = src.crs
#         dst_width = src.width
#         dst_height = src.height

#     if "srtm" in dem_dir:
#         dem_path = dem_dir
#         if os.path.exists(reference_dem_dir.replace('.tif', '_reg2_srtm.tif')):
#             with rasterio.open(reference_dem_dir.replace('.tif', '_reg2_srtm.tif')) as src:
#                 reference_dem = src.read(1)
#         else:
#             general.reproj_match(infile=reference_dem_dir,
#                          match=dem_path,
#                          outfile=reference_dem_dir.replace('.tif', '_reg2_srtm.tif'))
#             with rasterio.open(reference_dem_dir.replace('.tif', '_reg2_srtm.tif')) as src:
#                 reference_dem = src.read(1)
#     else:
#         dem_path = dem_dir
#         if os.path.exists(dem_path.replace('.tif', 'refDEM_coreg.tif')):
#             with rasterio.open(dem_path.replace('.tif', 'refDEM_coreg.tif')) as src:
#                 reference_dem = src.read(1)
#         else:
#             if not os.path.exists(reference_dem_dir.replace('.tif', '_proj.tif')):
#                 general.raster_project(reference_dem_dir, dst_crs.data['init'], reference_dem_dir.replace('.tif', '_proj.tif'))
#             general.reproj_match(infile=reference_dem_dir.replace('.tif', '_proj.tif'),
#                          match=dem_path,
#                          outfile=dem_path.replace('.tif', 'refDEM_coreg.tif'))
#             with rasterio.open(dem_path.replace('.tif', 'refDEM_coreg.tif')) as src:
#                 # print("original 3DEP DEM crs: ", src.crs)
#                 reference_dem = src.read(1)
#                 # print("original 3DEP DEM shape: ", reference_dem.shape)

#     if 'tipp' in dem_path:
#         dem_off = dem - 0.3048 * reference_dem
#     else:
#         dem_off = dem - reference_dem
#     dem_off[dem_off < -9999] = np.nan
#     dem_off = dem_off.reshape(dem.shape)
#     with rasterio.open(save_path, 'w',
#                        height=dst_height, width=dst_width, count=1, dtype=dem_off.dtype,
#                        crs=dst_crs) as dst:
#         dst.write(dem_off, 1)
#     return dem_off


# def evaluate(indir, srtm_dir, out_fname):
#     t_lst = []
#     headers = ['', 'mean', 'median', 'rmse', 'STD', 'skewness', 'kurtosis']
#     res_srtm_path = srtm_dir + "diff_SRTM90_3DEP.tif"
#     res_NED_path = indir + "diff_NED30_3DEP.tif"
#     res_rf90_path = indir + "diff_RF90_3DEP.tif"
#     res_rf30_path = indir + "diff_RF30_3DEP.tif"
#     print("Evaluating SRTM 90:")
#     stats_srtm = general.raster_statistics(res_srtm_path)
#     # print("Evaluating NED 30:")
#     # stats_NED = general.raster_statistics(res_NED_path)
#     print("Evaluating RF 90:")
#     stats_rf90 = general.raster_statistics(res_rf90_path)
#     print("Evaluating RF 30:")
#     stats_rf30 = general.raster_statistics(res_rf30_path)
#     t_lst.append(['SRTM 90m DEM', "{:.3f}".format(stats_srtm[0]), "{:.3f}".format(stats_srtm[1]), "{:.3f}".format(stats_srtm[2])
#         , "{:.3f}".format(stats_srtm[3]), "{:.3f}".format(stats_srtm[4]), "{:.3f}".format(stats_srtm[5])])
#     t_lst.append(['30m DEM', "{:.3f}".format(stats_rf30[0]), "{:.3f}".format(stats_rf30[1]), "{:.3f}".format(stats_rf30[2])
#         , "{:.3f}".format(stats_rf30[3]), "{:.3f}".format(stats_rf30[4]), "{:.3f}".format(stats_rf30[5])])
#     t_lst.append(['90m DEM', "{:.3f}".format(stats_rf90[0]), "{:.3f}".format(stats_rf90[1]), "{:.3f}".format(stats_rf90[2])
#         , "{:.3f}".format(stats_rf90[3]), "{:.3f}".format(stats_rf90[4]), "{:.3f}".format(stats_rf90[5])])
#     string = tt.to_string(
#         t_lst,
#         header=headers,
#         style=tt.styles.ascii_thin_double,
#         padding=(0,1),
#         alignment='c'*len(headers)
#     )
#     print(string)
#     stats = np.stack((stats_rf30, stats_rf90, stats_srtm), axis=0)
#     df = pd.DataFrame(stats, columns=["mean", "median", "rmse", "std", "skew", "kurtosis"])
#     df = df.rename(index={0: 'rf30', 1: 'rf90', 2: 'srtm90'})
#     df.to_excel(indir + out_fname)

# def atl08_dataset(area, atl08_dir, atl08_keys, in_epsg, out_epsg, egm_path, path_3dep, clip, boundary_fpath, slope_tiff_path):
#     pd.set_option('mode.chained_assignment', None)
#     # ================================================ATL08 dataset creation and filtering =====================================================================
#     atl08 = GEDI_dataset.MyDataset(area_name=area, 
#                                   data_type='atl08',
#                                   area_bbox=None,
#                                   directory=atl08_dir,
#                                   in_epsg=in_epsg)
#     atl08.set_keys(atl08_keys)
#     atl08.load_data()
#     if clip:
#         # atl08.mask_aoi(boundary_fpath, atl08.in_epsg)
#         print('Before clipping: ', atl08.lonlat.shape[0])
#         atl08.clip(boundary_fpath)
#         print('After clipping: ', atl08.lonlat.shape[0])

#     atl08.get_XY(out_epsg=out_epsg, path=atl08.root_directory+'atl08_XY.csv')
#     atl08.get_elev3dep(elev3dep_path=path_3dep, save_path=atl08.root_directory+'atl08_fp_3dep_elev.csv')
#     atl08.elev3dep = np.array(atl08.elev3dep)
#     atl08.elev3dep[(atl08.elev3dep)<-9999] = np.nan
#     atl08.get_geoidHeight(egm_path=egm_path, geoid_path=atl08.root_directory+'atl08_geoidH.csv')
#     atl08.get_orthoHeight()
#     # atl08.get_slope(slope_tiff_path, save_path=atl08.root_directory+'atl08_slope.csv')
#     # atl08.get_slope(slope_tiff_path, save_path=atl08.root_directory+'atl08_slope_3dep.csv')

#     atl08_filtered = atl08.filtering([85])
#     tmp_df = pd.concat((atl08_filtered.raw_dataframe,pd.DataFrame(atl08_filtered.XY, columns=['X','Y']).set_index(atl08_filtered.raw_dataframe.index)),axis=1)
#     np.savetxt(atl08_filtered.root_directory+atl08_filtered.data_type+'_masked.csv', tmp_df, delimiter=',', header=','.join(tmp_df.columns.to_list()), comments='')
#     variogram_atl08 = None
#     # if not os.path.exists(atl08.root_directory+'variogram.pkl'):
#     #     if atl08_filtered.XY.shape[0] > 3000:
#     #         random_sample_idx = np.random.choice(atl08_filtered.XY.shape[0], 3000, replace=False)
#     #         sample_xy = atl08_filtered.XY[random_sample_idx, :]
#     #         sample_h = atl08_filtered.ortho_height[random_sample_idx]
#     #         variogram_atl08 = general.build_variogram(sample_xy, sample_h)
#     #     else:
#     #         variogram_atl08 = general.build_variogram(atl08_filtered.XY, atl08_filtered.ortho_height)
#     #     with open(atl08.root_directory+'variogram.pkl', 'wb') as handle:
#     #         pickle.dump(variogram_atl08, handle)
#     # else:
#     #     with open(atl08.root_directory+'variogram.pkl', 'rb') as handle:
#     #         variogram_atl08 = pickle.load(handle)
#     return atl08_filtered, variogram_atl08

# def gedi_dataset(area, gedi_dir, gedi_keys, in_epsg, out_epsg, egm_path, path_3dep, clip, boundary_fpath, slope_tiff_path, atl08_filtered, t_dist_cross_over):
#     pd.set_option('mode.chained_assignment', None)
#     # ================================================GEDI dataset creation and filtering =====================================================================
#     gedi = GEDI_dataset.MyDataset(area_name=area, 
#                                   data_type='gedi',
#                                  area_bbox=None,
#                                  directory=gedi_dir,
#                                  in_epsg=in_epsg)
#     gedi.set_keys(gedi_keys)
#     gedi.load_data()
#     if clip:
#         print('Before clipping: ', gedi.lonlat.shape[0])
#         gedi.clip(boundary_fpath)
#         print('After clipping: ', gedi.lonlat.shape[0])
#     gedi.get_XY(out_epsg=out_epsg, path=gedi.root_directory+'GEDI_XY.csv')
#     # gedi.get_elevSRTM(save_path=gedi.root_directory+'gedi_elevSRTM.csv')
#     gedi.get_elev3dep(elev3dep_path=path_3dep, save_path=gedi.root_directory+'gedi_3dep.csv')
#     gedi.elev3dep = np.array(gedi.elev3dep)
#     gedi.elev3dep[(gedi.elev3dep)<-9999] = np.nan
#     gedi.get_geoidHeight(egm_path=egm_path, geoid_path=gedi.root_directory+'gedi_geoidH.csv')
#     gedi.get_orthoHeight()

#     # gedi.get_slope(slope_tiff_path, save_path=gedi.root_directory+'gedi_slope_3dep.csv')

#     gedi_feature = gedi.filtering([0.7, 1.3, atl08_filtered])
#     gedi_feature = gedi.filtering([0.7, 1.3, atl08_filtered])
#     np.savetxt(gedi.root_directory+gedi.data_type+'_raw.csv', gedi.raw_dataframe, delimiter=',', header=','.join(gedi.raw_dataframe.columns.to_list()), comments='')
#     tmp_df = pd.concat((gedi_feature.raw_dataframe,pd.DataFrame(gedi_feature.XY, columns=['X','Y']).set_index(gedi_feature.raw_dataframe.index)),axis=1)
#     np.savetxt(gedi_feature.root_directory+gedi_feature.data_type+'_feature.csv', tmp_df, delimiter=',', header=','.join(tmp_df.columns.to_list()), comments='')
#     gedi_filtered = gedi.filtering([0.95, 1.1, atl08_filtered])
#     tmp_df = pd.concat((gedi_filtered.raw_dataframe,pd.DataFrame(gedi_filtered.XY, columns=['X','Y']).set_index(gedi_filtered.raw_dataframe.index)),axis=1)
#     np.savetxt(gedi_filtered.root_directory+gedi_filtered.data_type+'_label.csv', tmp_df, delimiter=',', header=','.join(tmp_df.columns.to_list()), comments='')

#     gedi_feature.find_cross_over(atl08_filtered, t_dist_cross_over, plot=False)

#     return gedi_feature, gedi_filtered, gedi

def process2(area, gedi_dir, gedi_keys, atl08_dir, atl08_keys, in_epsg, out_epsg, regress_method,
            num_gedi_neighbor, feature_list, egm_path,  path_3dep, path_srtm,
            cell_size, 
            clip, boundary_fpath):
    pd.set_option('mode.chained_assignment', None)
    # ================================================ATL08 dataset creation and filtering =====================================================================
    atl08_raw = GEDI_dataset.MyDataset(area_name=area, 
                                  data_type='atl08',
                                  area_bbox=None,
                                  directory=atl08_dir,
                                  in_epsg=in_epsg)
    atl08_raw.set_keys(atl08_keys)
    atl08_raw.load_data()
    if clip:
        atl08_raw.mask_aoi(boundary_fpath, atl08_raw.in_epsg)
    atl08_raw.get_XY(out_epsg=out_epsg, path=atl08_raw.root_directory+'atl08_XY.csv')
    atl08_raw.get_elev3dep(elev3dep_path=path_3dep, save_path=atl08_raw.root_directory+'atl08_fp_3dep_elev.csv')
    atl08_raw.elev3dep = np.array(atl08_raw.elev3dep)
    atl08_raw.elev3dep[(atl08_raw.elev3dep)<-9999] = np.nan
    atl08_raw.get_geoidHeight(egm_path=egm_path, geoid_path=atl08_raw.root_directory+'atl08_geoidH.csv')
    atl08_raw.get_orthoHeight()
    atl08 = atl08_raw.filtering([95])
    tmp_df = pd.concat((atl08.raw_dataframe,pd.DataFrame(atl08.XY, columns=['X','Y']).set_index(atl08.raw_dataframe.index)),axis=1)
    np.savetxt(atl08.root_directory+atl08.data_type+'_processed.csv', tmp_df, delimiter=',', header=','.join(tmp_df.columns.to_list()), comments='')
    # ================================================GEDI dataset creation and filtering =====================================================================
    gedi_raw = GEDI_dataset.MyDataset(area_name=area, 
                                  data_type='gedi',
                                 area_bbox=None,
                                 directory=gedi_dir,
                                 in_epsg=in_epsg)
    gedi_raw.set_keys(gedi_keys)
    gedi_raw.load_data()
    if clip:
        gedi_raw.mask_aoi(boundary_fpath, gedi_raw.in_epsg)
    gedi_raw.get_XY(out_epsg=out_epsg, path=gedi_raw.root_directory+'GEDI_XY.csv')
    gedi_raw.get_elevSRTM(save_path=gedi_raw.root_directory+'gedi_elevSRTM.csv')
    gedi_raw.get_elev3dep(elev3dep_path=path_3dep, save_path=gedi_raw.root_directory+'gedi_3dep.csv')
    gedi_raw.elev3dep = np.array(gedi_raw.elev3dep)
    gedi_raw.elev3dep[(gedi_raw.elev3dep)<-9999] = np.nan
    gedi_raw.get_geoidHeight(egm_path=egm_path, geoid_path=gedi_raw.root_directory+'gedi_geoidH.csv')
    gedi_raw.get_orthoHeight()

    gedi = gedi_raw.filtering([0.5, 1.5, atl08])
    np.savetxt(gedi.root_directory+gedi.data_type+'_raw.csv', gedi.raw_dataframe, delimiter=',', header=','.join(gedi.raw_dataframe.columns.to_list()), comments='')
    tmp_df = pd.concat((gedi.raw_dataframe,pd.DataFrame(gedi.XY, columns=['X','Y']).set_index(gedi.raw_dataframe.index)),axis=1)
    np.savetxt(gedi.root_directory+gedi.data_type+'_feature.csv', tmp_df, delimiter=',', header=','.join(tmp_df.columns.to_list()), comments='')
    # tmp_df = pd.concat((gedi.raw_dataframe,pd.DataFrame(gedi.XY, columns=['X','Y']).set_index(gedi.raw_dataframe.index)),axis=1)
    # np.savetxt(gedi.root_directory+gedi.data_type+'_label.csv', tmp_df, delimiter=',', header=','.join(tmp_df.columns.to_list()), comments='')
    del egm_path, gedi_dir, atl08_dir, gedi_keys, atl08_keys, in_epsg, out_epsg
    # ================================================ raster creation =====================================================================
    my_raster = GEDI_dataset.MyRaster(
        epsg=gedi.out_epsg, 
        XY=np.concatenate((gedi.XY, atl08.XY), axis=0), 
        resolution=cell_size, 
        nodata_value=np.nan)
    results_directory = gedi.root_directory.replace('GEDI', 'results') + str(my_raster.resolution) + 'm/'
    if not os.path.exists(results_directory):
            os.makedirs(results_directory)
    my_raster.create_empty_raster(results_directory+'empty_raster.tif')
    my_raster.create_atl08_fp_raster(atl08.XY, atl08.ortho_height, results_directory+'atl08_fp_raster.tif')
    if 'mend' in area:
        my_raster.atl08_fp_raster = my_raster.clip_raster(my_raster.atl08_fp_raster_path, boundary_fpath, my_raster.atl08_fp_raster_path.replace('.tif', '_clip.tif'))
        my_raster.atl08_fp_raster_path = my_raster.atl08_fp_raster_path.replace('.tif', '_clip.tif')
    my_raster.create_GEDI_fp_raster(gedi.XY, gedi.ortho_height, results_directory+'gedi_fp_raster.tif')
    if 'mend' in area:
        my_raster.gedi_fp_raster = my_raster.clip_raster(my_raster.gedi_fp_raster_path, boundary_fpath, my_raster.gedi_fp_raster_path.replace('.tif', '_clip.tif'))
        my_raster.gedi_fp_raster_path = my_raster.gedi_fp_raster_path.replace('.tif', '_clip.tif')
    my_raster.create_reference_raster(path_3dep)
    my_raster.reference_raster = np.flipud(my_raster.reference_raster)
    # my_raster.reference_raster = np.pad(my_raster.reference_raster, ((1, 0), (0, 0)), 'constant', constant_values=np.nan)
    my_raster.gedi_fp_mask = ~np.isnan(my_raster.gedi_fp_raster)
    my_raster.get_srtm_raster(path_srtm)
    my_raster.srtm = np.flipud(my_raster.srtm)
    # my_raster.srtm = np.pad(my_raster.srtm, ((1, 0), (0, 0)), 'constant', constant_values=0)
    if np.isnan(my_raster.gedi_fp_mask).all():
        raise ValueError("No GEDI footprint in the area")
    my_raster.atl08_fp_mask = ~np.isnan(my_raster.atl08_fp_raster)
    if np.isnan(my_raster.atl08_fp_mask).all():
        raise ValueError("No ATL08 footprint in the area")
    
    # nd = 1
    # my_raster.I1 = my_raster.local_moran(my_raster.srtm_coregistered_path, nd, my_raster.srtm_coregistered_path.replace('.tif', '_moran_'+str(nd)+'.tif'))


    # print('GEDI (#. {}) error vs reference dem. median:{:.4f}, mean:{:.4f}, std:{:.4f}, kurt:{:.4f}'.format(
    #     len(np.where(my_raster.gedi_fp_mask)[0]),
    #     np.nanmedian(my_raster.gedi_fp_raster[my_raster.gedi_fp_mask]-my_raster.reference_raster[my_raster.gedi_fp_mask]),
    #     np.nanmean(my_raster.gedi_fp_raster[my_raster.gedi_fp_mask]-my_raster.reference_raster[my_raster.gedi_fp_mask]),
    #     np.nanstd(my_raster.gedi_fp_raster[my_raster.gedi_fp_mask]-my_raster.reference_raster[my_raster.gedi_fp_mask]),
    #     stats.kurtosis(my_raster.gedi_fp_raster[my_raster.gedi_fp_mask]-my_raster.reference_raster[my_raster.gedi_fp_mask], fisher=True)))

    # ================================================ regression model initiate and features selection =====================================================================
    myRegressor = GEDI_dataset.Regressor(regression_method=regress_method,
                                         results_directory=results_directory + regress_method + '/',
                                         feature_list=feature_list)

    # if not os.path.exists(myRegressor.results_directory+'features_all_cells.pkl'):
    if True:
        raster_XY = np.stack((my_raster.x_coords.flatten(), my_raster.y_coords.flatten()), axis=1)
        # raster_atl08_XY = np.stack((my_raster.x_coords[my_raster.atl08_fp_mask].flatten(), my_raster.y_coords[my_raster.atl08_fp_mask].flatten()), axis=1)
        # raster_gedi_XY = np.stack((my_raster.x_coords[my_raster.gedi_fp_mask].flatten(), my_raster.y_coords[my_raster.gedi_fp_mask].flatten()), axis=1)
        # raster_gedi_XY = gedi.XY

        num_neighbor = num_gedi_neighbor
        # dist_atl08_0, idx_atl08_0 = general.nn(raster_XY, raster_atl08_XY, k=num_neighbor+1)
        # dist_atl08 = np.copy(dist_atl08_0[:, 0:-1])
        # dist_atl08[np.where(dist_atl08_0[:, 0]<0.0000001)[0], :] = dist_atl08_0[np.where(dist_atl08_0[:, 0]<0.0000001)[0], 1:]
        # idx_atl08 = np.copy(idx_atl08_0[:, 0:-1])
        # idx_atl08[np.where(dist_atl08_0[:, 0]<0.0000001)[0], :] = idx_atl08_0[np.where(dist_atl08_0[:, 0]<0.0000001)[0], 1:]
        # del dist_atl08_0, idx_atl08_0, raster_atl08_XY
        # z_atl08 = my_raster.atl08_fp_raster[my_raster.atl08_fp_mask].flatten()[idx_atl08]
        # z_atl08_mean = np.nanmean(z_atl08, axis=1)
        # z_atl08_std = np.nanstd(z_atl08, axis=1)
        # z_atl08_median = np.nanmedian(z_atl08, axis=1)
        # del idx_atl08

        training_XY = atl08.XY
        training_labels = atl08.ortho_height
        training_reference = atl08.elev3dep
        training_predictions = np.zeros(training_labels.shape[0])
        if num_neighbor != 0:
            # training_XY = np.concatenate((atl08.XY, gedi.XY), axis=0)        
            # training_labels = np.concatenate((atl08.ortho_height, gedi.ortho_height), axis=0)
            # training_predictions = np.zeros(len(np.where(training_labels)[0]))
            # training_reference = np.concatenate((atl08.elev3dep, gedi.elev3dep), axis=0)

            

            # training_idx = np.random.choice(raster_XY.shape[0], math.ceil(raster_XY.shape[0]/5)).astype(int)
            # print('Experimenting with {} self sampled points'.format(len(training_idx)))
            # training_XY = raster_XY[training_idx, :]
            # training_labels = my_raster.reference_raster.flatten()[training_idx]
            # training_predictions = np.zeros(training_labels.shape[0])
            # training_reference = my_raster.reference_raster.flatten()[training_idx]
            # df = pd. DataFrame(
            #     training_XY,
            #     columns=['x','y']
            # )
            # df.to_csv(myRegressor.results_directory+'xy_training_selfsamples.csv', index=False)

            # dist_gedi_lft, z_gedi_lft, sensitivity_gedi_lft, xy_gedi_lft = general.nn_lr_track(raster_XY, gedi, num_neighbor, 1, -9999)
            # dist_gedi_rgt, z_gedi_rgt, sensitivity_gedi_rgt, xy_gedi_rgt = general.nn_lr_track(raster_XY, gedi, num_neighbor, -1, -9999)
            # # dist_gedi_lft, z_gedi_lft, sensitivity_gedi_lft, xy_gedi_lft = general.nn_lr_track(raster_XY, training_XY, num_neighbor, 1, -9999)
            # # dist_gedi_rgt, z_gedi_rgt, sensitivity_gedi_rgt, xy_gedi_rgt = general.nn_lr_track(raster_XY, training_XY, num_neighbor, -1, -9999)
            # dist_gedi = np.concatenate((dist_gedi_lft, dist_gedi_rgt), axis=1)
            # dist_gedi = np.where(dist_gedi != 0, dist_gedi, np.nan)
            # z_gedi = np.concatenate((z_gedi_lft, z_gedi_rgt), axis=1)
            # sensitivity_gedi = np.concatenate((sensitivity_gedi_lft, sensitivity_gedi_rgt), axis=1)
            # xy_gedi = np.concatenate((xy_gedi_lft, xy_gedi_rgt), axis=1)
            # z_gedi_mean = np.nanmean(z_gedi, axis=1)
            # z_gedi_std = np.nanstd(z_gedi, axis=1)
            # z_gedi_median = np.nanmedian(z_gedi, axis=1)

            # dist_atl08_lft, z_atl08_lft, xy_atl08_lft = general.nn_lr_track(raster_XY, atl08, num_neighbor, 1, -9999)
            # dist_atl08_rgt, z_atl08_rgt, xy_atl08_rgt = general.nn_lr_track(raster_XY, atl08, num_neighbor, -1, -9999)
            # # dist_atl08_lft, z_atl08_lft, xy_atl08_lft = general.nn_lr_track(raster_XY, training_XY, num_neighbor, 1, -9999)
            # # dist_atl08_rgt, z_atl08_rgt, xy_atl08_rgt = general.nn_lr_track(raster_XY, training_XY, num_neighbor, -1, -9999)
            # dist_atl08 = np.concatenate((dist_atl08_lft, dist_atl08_rgt), axis=1)
            # dist_atl08 = np.where(dist_atl08 != 0, dist_atl08, np.nan)
            # z_atl08 = np.concatenate((z_atl08_lft, z_atl08_rgt), axis=1)
            # xy_atl08 = np.concatenate((xy_atl08_lft, xy_atl08_rgt), axis=1)
            # z_atl08_mean = np.nanmean(z_atl08, axis=1)
            # z_atl08_std = np.nanstd(z_atl08, axis=1)
            # z_atl08_median = np.nanmedian(z_atl08, axis=1)

            if "svr" in regress_method:
                dist_gedi_0, idx_gedi_0 = general.nn(raster_XY, gedi.XY, k=num_neighbor+1)
                # dist_gedi_0, idx_gedi_0 = general.nn(raster_XY, training_XY, k=num_neighbor+1)
                idx_gedi = np.copy(idx_gedi_0[:, 0:-1])
                idx_gedi[np.where(dist_gedi_0[:, 0]<0.0000001)[0], :] = idx_gedi_0[np.where(dist_gedi_0[:, 0]<0.0000001)[0], 1:]
                del idx_gedi_0
                dist_gedi = np.copy(dist_gedi_0[:, 0:-1])
                dist_gedi[np.where(dist_gedi_0[:, 0]<0.0000001)[0], :] = dist_gedi_0[np.where(dist_gedi_0[:, 0]<0.0000001)[0], 1:]
                del dist_gedi_0
                z_gedi = gedi.ortho_height[idx_gedi]
                # z_gedi = training_labels[idx_gedi]
                # sensitivity_gedi = gedi.raw_dataframe['sensitivity'].to_numpy()[idx_gedi]
                # sensitivity_gedi = np.zeros(z_gedi.shape)
                z_gedi_mean = np.nanmean(z_gedi, axis=1)
                z_gedi_std = np.nanstd(z_gedi, axis=1)
                z_gedi_median = np.nanmedian(z_gedi, axis=1)
                del idx_gedi
            else:
                # num_neighbor = 2*num_neighbor
                dist_gedi_0, idx_gedi_0 = general.nn(raster_XY, gedi.XY, k=num_neighbor+1)
                idx_gedi = np.copy(idx_gedi_0[:, 0:-1])
                idx_gedi[np.where(dist_gedi_0[:, 0]<0.0000001)[0], :] = idx_gedi_0[np.where(dist_gedi_0[:, 0]<0.0000001)[0], 1:]
                del idx_gedi_0
                dist_gedi = np.copy(dist_gedi_0[:, 0:-1])
                dist_gedi[np.where(dist_gedi_0[:, 0]<0.0000001)[0], :] = dist_gedi_0[np.where(dist_gedi_0[:, 0]<0.0000001)[0], 1:]
                del dist_gedi_0
                z_gedi = gedi.ortho_height[idx_gedi]

                dist_atl08_0, idx_atl08_0 = general.nn(raster_XY, atl08.XY, k=2)
                idx_atl08 = np.copy(idx_atl08_0[:, 0:-1])
                idx_atl08[np.where(dist_atl08_0[:, 0]<0.0000001)[0], :] = idx_atl08_0[np.where(dist_atl08_0[:, 0]<0.0000001)[0], 1:]
                del idx_atl08_0
                dist_atl08 = np.copy(dist_atl08_0[:, 0:-1])
                dist_atl08[np.where(dist_atl08_0[:, 0]<0.0000001)[0], :] = dist_atl08_0[np.where(dist_atl08_0[:, 0]<0.0000001)[0], 1:]
                del dist_atl08_0
                z_atl08 = atl08.ortho_height[idx_atl08]

                dist_gedi = np.concatenate((dist_atl08, dist_gedi), axis=1)
                z_gedi = np.concatenate((z_atl08, z_gedi), axis=1)

                z_gedi_mean = np.nanmean(z_gedi, axis=1)
                z_gedi_std = np.nanstd(z_gedi, axis=1)
                z_gedi_median = np.nanmedian(z_gedi, axis=1)
                del idx_gedi

            #========================================================= training data preparation
            if "svr" in regress_method:  
                dist_gedi_0, idx_gedi_0 = general.nn(training_XY, gedi.XY, k=num_neighbor+1)
                # dist_gedi_0, idx_gedi_0 = general.nn(training_XY, training_XY, k=num_neighbor+1)

                training_idx_gedi = np.copy(idx_gedi_0[:, 0:-1])
                training_idx_gedi[np.where(dist_gedi_0[:, 0]<0.0000001)[0], :] = idx_gedi_0[np.where(dist_gedi_0[:, 0]<0.0000001)[0], 1:]
                del idx_gedi_0

                training_dist_gedi = np.copy(dist_gedi_0[:, 0:-1])
                training_dist_gedi[np.where(dist_gedi_0[:, 0]<0.0000001)[0], :] = dist_gedi_0[np.where(dist_gedi_0[:, 0]<0.0000001)[0], 1:]
                del dist_gedi_0

                training_z_gedi = gedi.ortho_height[training_idx_gedi]

                training_z_gedi_mean = np.nanmean(training_z_gedi, axis=1)
                training_z_gedi_std = np.nanstd(training_z_gedi, axis=1)
                training_z_gedi_median = np.median(training_z_gedi, axis=1)
                del training_idx_gedi
            else:
                dist_gedi_0, idx_gedi_0 = general.nn(training_XY, gedi.XY, k=num_neighbor+1)
                training_idx_gedi = np.copy(idx_gedi_0[:, 0:-1])
                training_idx_gedi[np.where(dist_gedi_0[:, 0]<0.0000001)[0], :] = idx_gedi_0[np.where(dist_gedi_0[:, 0]<0.0000001)[0], 1:]
                del idx_gedi_0
                training_dist_gedi = np.copy(dist_gedi_0[:, 0:-1])
                training_dist_gedi[np.where(dist_gedi_0[:, 0]<0.0000001)[0], :] = dist_gedi_0[np.where(dist_gedi_0[:, 0]<0.0000001)[0], 1:]
                del dist_gedi_0
                training_z_gedi = gedi.ortho_height[training_idx_gedi]

                dist_atl08_0, idx_atl08_0 = general.nn(training_XY, atl08.XY, k=2)
                training_idx_atl08 = np.copy(idx_atl08_0[:, 0:-1])
                training_idx_atl08[np.where(dist_atl08_0[:, 0]<0.0000001)[0], :] = idx_atl08_0[np.where(dist_atl08_0[:, 0]<0.0000001)[0], 1:]
                del idx_atl08_0
                training_dist_atl08 = np.copy(dist_atl08_0[:, 0:-1])
                training_dist_atl08[np.where(dist_atl08_0[:, 0]<0.0000001)[0], :] = dist_atl08_0[np.where(dist_atl08_0[:, 0]<0.0000001)[0], 1:]
                del dist_atl08_0
                training_z_atl08 = atl08.ortho_height[training_idx_atl08]

                training_dist_gedi = np.concatenate((training_dist_atl08, training_dist_gedi), axis=1)
                training_z_gedi = np.concatenate((training_z_atl08, training_z_gedi), axis=1)
                training_z_gedi_mean = np.nanmean(training_z_gedi, axis=1)
                training_z_gedi_std = np.nanstd(training_z_gedi, axis=1)
                training_z_gedi_median = np.nanmedian(training_z_gedi, axis=1)

        srtm_neighbors = my_raster.extract_neighbors_raster(my_raster.srtm, 1, False)
        srtm_neighbors = srtm_neighbors.reshape(-1, srtm_neighbors.shape[-1])

        _, idx_srtm = general.nn(training_XY, raster_XY, k=9)
        training_srtm_neighbors = my_raster.srtm.flatten()[idx_srtm]
        del idx_srtm
        if num_neighbor != 0:
            features_all_cells = np.concatenate((
                raster_XY,
                srtm_neighbors,
                np.stack((z_gedi_median, z_gedi_mean, z_gedi_std), axis=1),
                dist_gedi, z_gedi,
                ), axis=1)
            
            myRegressor.feature_list = [
                'X', 'Y',
                'srtm',
                'srtm_upper', 'srtm_bottom', 'srtm_left', 'srtm_right',
                'srtm_upper_left', 'srtm_upper_right', 'srtm_bottom_left', 'srtm_bottom_right'
                ] \
                + ['z_gedi_median', 'z_gedi_mean', 'z_gedi_std'] \
                + ['dist_gedi_'+str(i) for i in range(dist_gedi.shape[1])] + ['z_gedi_'+str(i) for i in range(z_gedi.shape[1])] 
            del raster_XY, dist_gedi, z_gedi, z_gedi_mean, z_gedi_std, z_gedi_median, srtm_neighbors
            training_features = np.concatenate((
                training_XY,
                training_srtm_neighbors,
                np.stack((training_z_gedi_median, training_z_gedi_mean, training_z_gedi_std), axis=1),
                training_dist_gedi, training_z_gedi,
                ), axis=1)
            del training_srtm_neighbors, training_z_gedi, training_z_gedi_mean, training_z_gedi_std, training_z_gedi_median, training_dist_gedi
            # del training_z_atl08, training_z_atl08_mean, training_z_atl08_std, training_z_atl08_median, training_dist_atl08

        else:
            features_all_cells = np.concatenate((
                raster_XY,

                srtm_neighbors
                ), axis=1)
            del raster_XY, srtm_neighbors
            myRegressor.feature_list = [
                'X', 'Y',
                                        'srtm',
                                        'srtm_upper', 'srtm_bottom', 'srtm_left', 'srtm_right',
                                        'srtm_upper_left', 'srtm_upper_right', 'srtm_bottom_left', 'srtm_bottom_right'
                                        ]
            
            training_features = np.concatenate((
                atl08.XY,
                training_srtm_neighbors
            ), axis=1)
            del training_srtm_neighbors

        df = pd.DataFrame(features_all_cells, columns=myRegressor.feature_list)
        df.to_pickle(myRegressor.results_directory+'features_all_cells.pkl')
        df_training = pd.DataFrame(training_features, columns=myRegressor.feature_list)
        df_training.to_pickle(myRegressor.results_directory+'features_training.pkl')
        df_training_temp = df_training.copy()
        # df_training_temp['labels'] = np.concatenate((atl08.ortho_height, gedi.ortho_height), axis=0)
        df_training_temp['labels'] = atl08.ortho_height
        df_training_temp.to_csv(myRegressor.results_directory+'features_training.csv', index=False)
        del df_training_temp

    myRegressor.feature_list = df.columns.tolist()
    features_all_cells = df.values
    del df_training, df
    
    if features_all_cells.shape[0] == my_raster.x_coords.flatten().shape[0]:
        index = np.arange(features_all_cells.shape[0])
        index = np.reshape(index, my_raster.x_coords.shape)
    else:
        raise ValueError('features_all_cells.shape[0] != raster_XY.shape[0]')


    # mask = my_raster.atl08_fp_mask
    # if 'test' in myRegressor.regression_method:
    #     labels_raster = my_raster.reference_raster.copy()
    #     labels_raster[my_raster.atl08_fp_mask==False]=np.nan
    # else:
    #     labels_raster = my_raster.atl08_fp_raster.copy()
    # mask = ~np.isnan(labels_raster)
    # footprints_XY = np.zeros((len(np.where(mask)[0]), 2))
    # footprints_labels = np.zeros(len(np.where(mask)[0]))
    # footprints_predictions = np.zeros(len(np.where(mask)[0]))

    # training_XY = np.concatenate((atl08.XY, gedi.XY), axis=0)
    # # training_XY = atl08.XY
    # training_labels = np.concatenate((atl08.ortho_height, gedi.ortho_height), axis=0)
    # # training_labels = atl08.ortho_height
    # training_predictions = np.zeros(len(np.where(training_labels)[0]))
    # training_reference = np.concatenate((atl08.elev3dep, gedi.elev3dep), axis=0)
    # # training_reference = atl08.elev3dep
    print('Labels (atl08) (#. {}) error vs reference dem. median:{:.4f}, mean:{:.4f}, std:{:.4f}, kurt:{:.4f}'.format(
                        atl08.elev3dep.shape[0],
                        np.nanmedian(atl08.ortho_height-atl08.elev3dep),
                        np.nanmean(atl08.ortho_height-atl08.elev3dep),
                        np.nanstd(atl08.ortho_height-atl08.elev3dep),
                        stats.kurtosis(atl08.ortho_height-atl08.elev3dep, fisher=True)))
    print('Labels (gedi) (#. {}) error vs reference dem. median:{:.4f}, mean:{:.4f}, std:{:.4f}, kurt:{:.4f}'.format(
                        gedi.elev3dep.shape[0],
                        np.nanmedian(gedi.ortho_height-gedi.elev3dep),
                        np.nanmean(gedi.ortho_height-gedi.elev3dep),
                        np.nanstd(gedi.ortho_height-gedi.elev3dep),
                        stats.kurtosis(gedi.ortho_height-gedi.elev3dep, fisher=True)))
    print('Labels (#. {}) error vs reference dem. median:{:.4f}, mean:{:.4f}, std:{:.4f}, kurt:{:.4f}'.format(
                        training_labels.shape[0],
                        np.nanmedian(training_labels-training_reference),
                        np.nanmean(training_labels-training_reference),
                        np.nanstd(training_labels-training_reference),
                        stats.kurtosis(training_labels-training_reference, fisher=True)))
    print('Features gedi (#. {}) error vs reference dem. median:{:.4f}, mean:{:.4f}, std:{:.4f}, kurt:{:.4f}'.format(
                        gedi.elev3dep.shape[0],
                        np.nanmedian(gedi.ortho_height-gedi.elev3dep),
                        np.nanmean(gedi.ortho_height-gedi.elev3dep),
                        np.nanstd(gedi.ortho_height-gedi.elev3dep),
                        stats.kurtosis(gedi.ortho_height-gedi.elev3dep, fisher=True)))

    # my_raster.srtm = None
    # my_raster.atl08_fp_raster = None
    # my_raster.atl08_fp_mask = None
    # my_raster.gedi_fp_raster = None
    # my_raster.gedi_fp_mask = None

    # ========================================================= subsample with batches for memory efficiency =========================================================
    if 'krr' in myRegressor.regression_method or 'svr' in myRegressor.regression_method or 'spline' in myRegressor.regression_method:
    # if 'krr' in myRegressor.regression_method:
        print('subsampling into batches...')
        if 'svr' in myRegressor.regression_method:
            if my_raster.resolution == 30:
                batch_size = 500
            elif my_raster.resolution == 90:
                batch_size = 150
            else:
                batch_size = 1600
        elif 'krr' in myRegressor.regression_method:
            batch_size = 300
        else:
            batch_size = 500
        num_batches_per_row = math.ceil(my_raster.shape[1]/batch_size)
        num_batches_per_col = math.ceil(my_raster.shape[0]/batch_size)
        num_batches = num_batches_per_row * num_batches_per_col
        my_raster.predicted_raster = np.copy(my_raster.empty_raster.copy())
        my_raster.predicted_raster_error = np.copy(my_raster.empty_raster.copy())
        my_raster.predicted_raster_path = myRegressor.results_directory+'prediction_'+str(my_raster.resolution)+'m.tif'
        regressor_info = []
        id = 0
        for i in range(num_batches_per_col):
            for j in range(num_batches_per_row):
                print('================ batch {}/{}, batch size: {}======================='.format(i*num_batches_per_row+j+1, num_batches, batch_size))
                row_index_start = i*batch_size
                row_index_end = min((i+1)*batch_size, my_raster.shape[0])
                col_index_start = j*batch_size
                col_index_end = min((j+1)*batch_size, my_raster.shape[1])
                mask_batch = np.full(my_raster.empty_raster.shape, False, dtype=bool)
                mask_batch[row_index_start:row_index_end, col_index_start:col_index_end] = True
                # mask_batch_with_label = mask_batch & mask
                # if len(np.where(mask_batch_with_label)[0]) < 7:
                # if not mask_batch_with_label.any():
                minx_batch = my_raster.x_coords[row_index_start:row_index_end, col_index_start:col_index_end].min()
                maxx_batch = my_raster.x_coords[row_index_start:row_index_end, col_index_start:col_index_end].max()
                miny_batch = my_raster.y_coords[row_index_start:row_index_end, col_index_start:col_index_end].min()
                maxy_batch = my_raster.y_coords[row_index_start:row_index_end, col_index_start:col_index_end].max()
                boolArray = (training_XY[:, 0] >= minx_batch) & (training_XY[:, 0] <= maxx_batch) & (training_XY[:, 1] >= miny_batch) & (training_XY[:, 1] <= maxy_batch)
                training_idx_within_batch = np.where(boolArray)[0]
                if len(training_idx_within_batch) < 7:
                    print("no enough label within this batch. skip...")
                    regressor_info.append([np.nan]*11)
                    continue
                else:
                    # myRegressor.set_features(features_all_cells[index[mask_batch_with_label].flatten(), :])
                    # myRegressor.set_labels(labels_raster[mask_batch_with_label].flatten())
                    myRegressor.set_features(training_features[training_idx_within_batch, :])
                    myRegressor.set_labels(training_labels[training_idx_within_batch])
                    
                    myRegressor.reference_height = training_reference[training_idx_within_batch]
                    
                    # ================================================ regression =====================================================================
                    print("Performing regression...")
                    myRegressor.regress([700])
                                        
                    myRegressor.error2reference = myRegressor.prediction.flatten() - myRegressor.reference_height
                    myRegressor.error2label = myRegressor.prediction.flatten() - myRegressor.labels
                    # footprints_XY[id:id+myRegressor.error2reference.shape[0], :] = features_all_cells[index[mask_batch_with_label].flatten(), :2]
                    # footprints_labels[id:id+myRegressor.error2reference.shape[0]] = labels_raster[mask_batch_with_label].flatten()
                    # training_predictions[id:id+myRegressor.labels.shape[0]] = myRegressor.prediction.flatten()
                    
                    training_predictions[training_idx_within_batch] = myRegressor.prediction.flatten()
                    id += myRegressor.labels.shape[0]
                    print('Predictions of labels (#. {}) error vs reference dem. median:{:.4f}, mean:{:.4f}, std:{:.4f}, kurt:{:.4f}'.format(
                        myRegressor.error2reference.shape[0],
                        np.nanmedian(myRegressor.error2reference),
                        np.nanmean(myRegressor.error2reference),
                        np.nanstd(myRegressor.error2reference),
                        stats.kurtosis(myRegressor.error2reference[~np.isnan(myRegressor.error2reference)], fisher=True)))
                    
                    if 'svr' or 'krr' in myRegressor.regression_method:
                        prediction_for_batch_cells = myRegressor.regressor.predict(myRegressor.scaler.transform(features_all_cells[index[mask_batch]]))
                        prediction_for_batch_cells = np.reshape(myRegressor.scaler_label.inverse_transform(prediction_for_batch_cells.reshape(-1, 1)),
                                                                (len(np.unique(np.where(mask_batch)[0])), len(np.unique(np.where(mask_batch)[1]))))
                    else:
                        prediction_for_batch_cells = np.reshape(myRegressor.regressor.predict(features_all_cells[index[mask_batch]]),
                                                                (len(np.unique(np.where(mask_batch)[0])), len(np.unique(np.where(mask_batch)[1]))))
                    my_raster.predicted_raster[mask_batch] = prediction_for_batch_cells.flatten()
                    my_raster.predicted_raster_error[mask_batch] = prediction_for_batch_cells.flatten()- my_raster.reference_raster[mask_batch]
                    if 'svr' in myRegressor.regression_method:
                        regressor_info.append([i, j, 
                                            myRegressor.regressor.C, myRegressor.regressor.gamma, 
                                            myRegressor.error2reference.shape[0], 
                                            np.nanmedian(myRegressor.error2reference), np.nanmean(myRegressor.error2reference), np.nanstd(myRegressor.error2reference),
                                            np.nanmedian(my_raster.predicted_raster_error[mask_batch]), np.nanmean(my_raster.predicted_raster_error[mask_batch]), np.nanstd(my_raster.predicted_raster_error[mask_batch])])
                    elif 'krr' in myRegressor.regression_method:
                        regressor_info.append([i, j, 
                                            myRegressor.regressor.alpha, myRegressor.regressor.gamma, 
                                            myRegressor.error2reference.shape[0], 
                                            np.nanmedian(myRegressor.error2reference), np.nanmean(myRegressor.error2reference), np.nanstd(myRegressor.error2reference),
                                            np.nanmedian(my_raster.predicted_raster_error[mask_batch]), np.nanmean(my_raster.predicted_raster_error[mask_batch]), np.nanstd(my_raster.predicted_raster_error[mask_batch])])
                    # print('Predicted {}m DEM error vs reference dem. median:{:.4f}, mean:{:.4f}, std:{:.4f}, kurt:{:.4f}'.format(
                    #     str(my_raster.resolution),
                    #     np.nanmedian(my_raster.predicted_raster_error[mask_batch]),
                    #     np.nanmean(my_raster.predicted_raster_error[mask_batch]),
                    #     np.nanstd(my_raster.predicted_raster_error[mask_batch]),
                    #     stats.kurtosis(my_raster.predicted_raster_error[mask_batch].flatten(), fisher=True)))
                    # plt.figure()
                    # plt.subplot(1,2,1)
                    # plt.imshow(np.flipud(my_raster.predicted_raster), cmap='gray')
                    # plt.subplot(1,2,2)
                    # plt.imshow(np.flipud(my_raster.predicted_raster_error), cmap='gray', vmin=my_raster.predicted_raster_error[mask_batch].min(), vmax=my_raster.predicted_raster_error[mask_batch].max())

                    # plt.show(block=True)
                    # plt.close('all')

        r2 = r2_score(training_labels[training_predictions!=0], training_predictions[training_predictions!=0])
        regression.plot_regression_results_R(
            training_labels[training_predictions!=0],
            training_predictions[training_predictions!=0],
            "SVR all dataset",
            (r"$R^2={:.2f}$").format(
                r2
            ),
            0,
        )
        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
        plt.savefig(myRegressor.results_directory+'relationship_all.png')
        if 'svr' in myRegressor.regression_method:
            df = pd.DataFrame(regressor_info, columns=['i', 'j', 'C', 'gamma', 'n_label', 'label_error_median', 'label_error_mean', 'label_error_std', 'median_pred_all', 'mean_pred_all', 'std_pred_all'])
            df.to_csv(myRegressor.results_directory+'regressor_info_all.csv')
            del df
        elif 'krr' in myRegressor.regression_method:
            df = pd.DataFrame(regressor_info, columns=['i', 'j', 'alpha', 'gamma', 'n_label', 'label_error_median', 'label_error_mean', 'label_error_std', 'median_pred_all', 'mean_pred_all', 'std_pred_all'])
            df.to_csv(myRegressor.results_directory+'regressor_info_all.csv')
            del df
    
    else:
        # ================================================ entire dataset (without batch) =====================================================================

        myRegressor.set_features(training_features)
        myRegressor.set_labels(training_labels)
        myRegressor.reference_height = training_reference

        print("Performing regression...")
        myRegressor.regress([100])

        myRegressor.error2reference = myRegressor.prediction - myRegressor.reference_height
        myRegressor.error2label = myRegressor.prediction - myRegressor.labels
        print('Predictions of labels (#. {}) error vs reference dem. median:{:.4f}, mean:{:.4f}, std:{:.4f}, kurt:{:.4f}'.format(
            myRegressor.error2reference.shape[0],
            np.nanmedian(myRegressor.error2reference),
            np.nanmean(myRegressor.error2reference),
            np.nanstd(myRegressor.error2reference),
            stats.kurtosis(myRegressor.error2reference[~np.isnan(myRegressor.error2reference)], fisher=True)))

        prediction_for_all_cells = np.reshape(myRegressor.regressor.predict(features_all_cells), my_raster.shape)
        my_raster.predicted_raster = prediction_for_all_cells
        my_raster.predicted_raster_path = myRegressor.results_directory+'prediction_for_all_cells_'+str(my_raster.resolution)+'m.tif'

    # ================================================ save results and visualization =====================================================================
    footprints = pd.DataFrame({
        'X': training_XY[:, 0],
        'Y': training_XY[:, 1],
        'label': training_labels,
        'prediction': training_predictions,
        'error2label': training_predictions-training_labels,
    })
    footprints.to_csv(myRegressor.results_directory+'prediction_vs_label_'+str(my_raster.resolution)+'m_footprints.csv', index=False)
    my_raster.predicted_raster_path = my_raster.predicted_raster_path.replace('.tif', '_'+str(num_gedi_neighbor)+'.tif')
    my_raster.save_raster(my_raster.empty_raster_path, my_raster.predicted_raster, my_raster.predicted_raster_path)
    df = pd.read_pickle(myRegressor.results_directory+'features_all_cells.pkl')
    df['prediction'] = my_raster.predicted_raster.flatten()
    df['X'] = my_raster.x_coords.flatten()
    df['Y'] = my_raster.y_coords.flatten()
    df.to_csv(myRegressor.results_directory+'features_all_cells.csv', index=False)
    del df

    if 'mend' in area:
        my_raster.predicted_raster = my_raster.clip_raster(my_raster.predicted_raster_path, boundary_fpath, my_raster.predicted_raster_path.replace('.tif', '_clip.tif'))
        my_raster.predicted_raster_path = my_raster.predicted_raster_path.replace('.tif', '_clip.tif')
    
    my_raster.predicted_raster_error = my_raster.predicted_raster - my_raster.reference_raster
    my_raster.predicted_raster_error_path = myRegressor.results_directory+'prediction_'+str(my_raster.resolution)+'m_error_'+str(num_gedi_neighbor)+'.tif'
    my_raster.save_raster(my_raster.empty_raster_path, my_raster.predicted_raster_error, my_raster.predicted_raster_error_path)
    # print('Predicted {}m DEM (raw out) error vs reference dem. median:{:.4f}, mean:{:.4f}, std:{:.4f}, kurt:{:.4f}'.format(
    #     str(my_raster.resolution),
    #     np.nanmedian(my_raster.predicted_raster_error),
    #     np.nanmean(my_raster.predicted_raster_error),
    #     np.nanstd(my_raster.predicted_raster_error),
    #     stats.kurtosis(my_raster.predicted_raster_error.flatten(), fisher=True)))

    my_raster.predicted_raster[my_raster.atl08_fp_mask] = my_raster.atl08_fp_raster[my_raster.atl08_fp_mask]
    my_raster.predicted_raster_path = my_raster.predicted_raster_path.replace('.tif', '_atl08_filled.tif')
    my_raster.save_raster(my_raster.empty_raster_path, my_raster.predicted_raster, my_raster.predicted_raster_path)
    my_raster.predicted_raster_error = my_raster.predicted_raster - my_raster.reference_raster
    my_raster.predicted_raster_error_path = myRegressor.results_directory+'prediction_'+str(my_raster.resolution)+'m_error_'+str(num_gedi_neighbor)+'_atl08_filled.tif'
    my_raster.save_raster(my_raster.empty_raster_path, my_raster.predicted_raster_error, my_raster.predicted_raster_error_path)
    my_raster.predicted_raster_error_path = myRegressor.results_directory+'prediction_'+str(my_raster.resolution)+'m_error_'+str(num_gedi_neighbor)+'_atl08_filled_no_boundary.tif'
    predicted_raster_error = np.full(my_raster.predicted_raster_error.shape, np.nan)
    if 'svr' in myRegressor.regression_method:
        predicted_raster_error[10:-10, 10:-10] = my_raster.predicted_raster_error[10:-10, 10:-10]
        predicted_raster_error[abs(predicted_raster_error) > 20] = 0
    else:
        predicted_raster_error[10:-10, 10:-10] = my_raster.predicted_raster_error[10:-10, 10:-10]
    my_raster.save_raster(my_raster.empty_raster_path, predicted_raster_error, my_raster.predicted_raster_error_path)
    
    # print('Predicted {}m DEM (atl08 filled) error vs reference dem. median:{:.4f}, mean:{:.4f}, std:{:.4f}, kurt:{:.4f}'.format(
    #     str(my_raster.resolution),
    #     np.nanmedian(my_raster.predicted_raster_error),
    #     np.nanmean(my_raster.predicted_raster_error),
    #     np.nanstd(my_raster.predicted_raster_error),
    #     stats.kurtosis(my_raster.predicted_raster_error.flatten(), fisher=True)))
    
    plt.close('all')
    plt.figure(figsize=(23,4))
    ax1 = plt.subplot(141)
    im1 = ax1.imshow(np.flipud(my_raster.predicted_raster), cmap='gray')
    ax1.set_title('Predicted DEM (m)')
    plt.colorbar(im1, ax=ax1) 
    ax2 = plt.subplot(142, sharex=ax1, sharey=ax1)
    im2 = ax2.imshow(np.flipud(predicted_raster_error), cmap='gray')
    ax2.set_title('Predicted DEM - reference DEM  (m)')
    im2.set_clim(-25, 30)
    plt.colorbar(im2, ax=ax2)
    ax3 = plt.subplot(143)
    ax3.hist(predicted_raster_error[~np.isnan(predicted_raster_error)], bins=100, color='blue', density=True)
    ax3.set_xlim(-25, 30)
    ax3.set_title('Predicted DEM - reference DEM histogram')
    ax4 = plt.subplot(144)
    ax4.set_xticks([])
    ax4.set_yticks([])
    ax4.set_frame_on(False)
    # plot text
    ax4.text(0.5, 0.5, 'Predicted DEM - reference DEM\nmedian: {:.4f}m, \nmean: {:.4f}m, \nstd: {:.4f}m'.format(
        np.nanmedian(predicted_raster_error[predicted_raster_error>-100]),
        np.nanmean(predicted_raster_error[predicted_raster_error>-100]),
        np.nanstd(predicted_raster_error[predicted_raster_error>-100])
    ), fontsize=10, ha='center', va='center')

    plt.savefig(myRegressor.results_directory+'predicted_dem&error.png')
    print('Figure saved to {}'.format(myRegressor.results_directory+'predicted_dem&error.png'))
    
    if 'tipp' in area:
        print('Predicted {}m DEM error vs reference dem. median:{:.4f}, mean:{:.4f}, std:{:.4f}'.format(
            str(my_raster.resolution),
            np.nanmedian(predicted_raster_error),
            np.nanmean(predicted_raster_error),
            np.nanstd(predicted_raster_error)
            ))
    elif 'mend' in area:
        print('Predicted {}m DEM error vs reference dem. median:{:.4f}, mean:{:.4f}, std:{:.4f}'.format(
            str(my_raster.resolution),
            np.nanmedian(my_raster.predicted_raster_error[my_raster.predicted_raster_error>-100]),
            np.nanmean(my_raster.predicted_raster_error[my_raster.predicted_raster_error>-100]),
            np.nanstd(my_raster.predicted_raster_error[my_raster.predicted_raster_error>-100]),
            ))
    plt.show(block=False)
    print('Done regression')


def dems_hist_plot(root_dir, regression_methods, resolution_lst, bins):
    total_num = len(regression_methods) * len(resolution_lst)
    fig, axs = plt.subplots(len(resolution_lst), len(regression_methods), figsize=(20, 10))
    # fig2, axs2 = plt.subplots(len(resolution_lst), len(regression_methods), figsize=(20, 10))
    y_max = np.zeros(len(resolution_lst))
    for i, resolution in enumerate(resolution_lst):
        for j, regression_method in enumerate(regression_methods):
            img_path = root_dir + regression_method + '/' + str(resolution) + 'm/prediction_for_all_cells_' + str(resolution) + 'm_error.tif'
            img = rasterio.open(img_path).read(1)
            img = img[15:-15, 15:-15]
            if 'tipp' in root_dir:
                img = img.flatten()[abs(img.flatten())<=35]
            if 'mend' in root_dir:
                if resolution == 30:
                    img = img[:, :3800]
                elif resolution == 90:
                    img = img[:, :1263]
                img = img.flatten()[abs(img.flatten())<=150]
            img = img.flatten()
            # axs2[i, j].imshow(np.flipud(img), cmap='gray')
            # axs2[i, j].set_title(regression_method + ' ' + str(resolution) + 'm')
            counts, bins, patches = axs[i, j].hist(img, bins=bins, weights = np.ones(len(img)) / len(img))
            print(np.nanmedian(img), np.nanmean(img), np.std(img))
            if counts.max() > y_max[i]:
                y_max[i] = counts.max()
            mean = np.nanmean(img)
            std = np.nanstd(img)
            axs[i, j].axvline(mean, color='r', linestyle='--', label='Mean', linewidth=1)
            # axs[i, j].axvline(mean - std, color='gray', linestyle='--', label='1 Std')
            # axs[i, j].axvline(mean + std, color='gray', linestyle='--')
            axs[i, j].legend()

            if 'random_forest' in regression_method:
                regression_method = 'RFSI'
            elif 'krr' in regression_method:
                if 'custom' in regression_method:
                    regression_method = 'KRR (custom)'
                else:
                    regression_method = 'KRR'
            elif 'svr' in regression_method:
                if 'custom' in regression_method:
                    regression_method = 'SVR (custom)'
                else:
                    regression_method = 'SVR'
                
            
            axs[i, j].set_title(regression_method + ' ' + str(resolution) + 'm')
            # for k, patch in enumerate(patches):
            #     height = patch.get_height()
            #     axs[i, j].text(patch.get_x() + patch.get_width() / 2,
            #                     height + 2,
            #                     f"{counts[k]:.0f}",
            #                     ha='center', va='bottom')

    fig.suptitle("Histograms of elevation difference of DEMs")
    for i, ax_row in enumerate(axs):
        for ax in ax_row:
            ax.set(xlabel="Elevation Difference (m)", ylabel="Percentage")
            # ax.set_xlim([-150, 150])
            # ax.set_ylim([0, y_max[i]*1.1])
            ax.set_ylim([0, 0.55])
    fig.tight_layout()

    plt.show()
    print('done')


def dems_hist_plot_ablation(root_dir, regression_method, ablation_lst, resolution_lst, bins):
    total_num = len(ablation_lst) * len(resolution_lst)
    if len(resolution_lst) != 1:
        fig, axs = plt.subplots(len(resolution_lst), len(ablation_lst), figsize=(20, 10))
    else:
        fig, axs = plt.subplots(len(resolution_lst), len(ablation_lst), figsize=(20, 5))
        res_lst = []
    # fig2, axs2 = plt.subplots(len(resolution_lst), len(regression_methods), figsize=(20, 10))
    y_max = np.zeros(len(resolution_lst))
    for i, resolution in enumerate(resolution_lst):
        for j, ablation in enumerate(ablation_lst):
            img_path = root_dir + regression_method + '/' + str(resolution) + 'm/prediction_for_all_cells_' + str(resolution) + 'm_error_'+str(ablation)+'.tif'
            img = rasterio.open(img_path).read(1)
            img = img[15:-15, 15:-15]
            if 'tipp' in root_dir:
                img = img.flatten()[abs(img.flatten())<=35]
            if 'mend' in root_dir:
                if resolution == 30:
                    img = img[:, :3800]
                elif resolution == 90:
                    img = img[:, :1263]
                img = img.flatten()[abs(img.flatten())<=150]
            img = img.flatten()
            # axs2[i, j].imshow(np.flipud(img), cmap='gray')
            # axs2[i, j].set_title(regression_method + ' ' + str(resolution) + 'm')
            if len(resolution_lst) != 1:
                counts, bins, patches = axs[i, j].hist(img, bins=bins, weights = np.ones(len(img)) / len(img))
            else:
                counts, bins, patches = axs[j].hist(img, bins=bins, weights = np.ones(len(img)) / len(img))
                res_lst.append([np.nanmedian(img), np.nanmean(img), np.std(img)])

            print(np.nanmedian(img), np.nanmean(img), np.std(img))
            if counts.max() > y_max[i]:
                y_max[i] = counts.max()
            mean = np.nanmean(img)
            std = np.nanstd(img)
            if len(resolution_lst) != 1:
                axs[i, j].axvline(mean, color='r', linestyle='--', label='Mean', linewidth=1)
            else:
                axs[j].axvline(mean, color='r', linestyle='--', label='Mean', linewidth=1)
            # axs[i, j].axvline(mean - std, color='gray', linestyle='--', label='1 Std')
            # axs[i, j].axvline(mean + std, color='gray', linestyle='--')
            if len(resolution_lst) != 1:
                axs[i, j].legend()
            else:
                axs[j].legend()

            if 'random_forest' in regression_method:
                regression_method = 'RFSI'
            elif 'krr' in regression_method:
                if 'custom' in regression_method:
                    regression_method = 'KRR (custom)'
                else:
                    regression_method = 'KRR'
            elif 'svr' in regression_method:
                if 'custom' in regression_method:
                    regression_method = 'SVR (custom)'
                else:
                    regression_method = 'SVR'
                
            if len(resolution_lst) != 1:
                axs[i, j].set_title(str(ablation) + ' at ' + str(resolution) + 'm')
            else:
                axs[j].set_title(str(ablation) + ' at ' + str(resolution) + 'm')
            # for k, patch in enumerate(patches):
            #     height = patch.get_height()
            #     axs[i, j].text(patch.get_x() + patch.get_width() / 2,
            #                     height + 2,
            #                     f"{counts[k]:.0f}",
            #                     ha='center', va='bottom')

    fig.suptitle("Histograms of elevation difference of DEMs")
    if len(resolution_lst) != 1:
        for i, ax_row in enumerate(axs):
            for ax in ax_row:
                ax.set(xlabel="Elevation Difference (m)", ylabel="Percentage")
                # ax.set_xlim([-150, 150])
                ax.set_ylim([0, y_max[i]*1.1])
                # ax.set_ylim([0, 0.55])
    else:
        df = pd.DataFrame(res_lst, columns=['median', 'mean', 'std'], index=ablation_lst)
        df.to_csv(root_dir + regression_method +'/'+str(resolution_lst[0])+'m/ablation_effect_'+str(resolution_lst[0])+'m.csv')
        for ax in axs:
            ax.set(xlabel="Elevation Difference (m)", ylabel="Percentage")
            # ax.set_xlim([-150, 150])
            ax.set_ylim([0, y_max[0]*1.1])
            # ax.set_ylim([0, 0.55])
    fig.tight_layout()

    plt.show()
    print('done')