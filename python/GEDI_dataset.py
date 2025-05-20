import os
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
import general, regression
import sklearn.cluster as clustering
import math
from copy import deepcopy
# from pyGEDI import *

class MyDataset:
    def __init__(self, area_name, data_type, area_bbox, directory, in_epsg):
        """
        bbox = [ul_lat,ul_lon,lr_lat,lr_lon]
        """
        self.data_type = data_type
        self.area_name = area_name
        self.area_bbox = area_bbox
        self.root_directory = directory
        if not os.path.exists(self.root_directory):
            os.makedirs(self.root_directory)
        self.data_directory = directory + "rawdata/"
        if not os.path.exists(self.data_directory):
            os.makedirs(self.data_directory)
        self.in_epsg = in_epsg

        self.raw_dataframe = None
        self.keys = None
        self.lonlat = None
        self.XY = None
        self.out_epsg = None
        self.path_XY = None
        self.geoid_height = None
        self.egm_path = None
        self.geoid_path = None
        self.ortho_height = None
        self.slope_tiff_directory = None
        self.slope_srtm_directory = None
        self.srtm_slope = None
        self.aspect_tiff_directory = None
        self.aspect_srtm_directory = None
        self.srtm_aspect = None
        self.elev3dep_tiff_directory = None
        self.elev3dep_directory = None
        self.elev3dep = None
        self.srtm_tiff_directory = None
        self.elev_srtm_directory = None
        self.srtm_elev = None
        self.canopy_height = None
        self.slope = None
        
    def download_data(self, version):
        username="txx1220"
        password="Wsnd1220"
        session = sessionNASA(username, password)
        product_2A = 'GEDI02_A'

        bbox = self.area_bbox
        out_dir = self.data_directory
        gediDownload(out_dir, product_2A, version, bbox, session)

    def set_keys(self, gedi_keys):
        self.keys = gedi_keys

    def load_data(self):
        gedi_df, keys = general.csv_load(self.data_directory, self.data_type, self.keys)
        gedi_df = gedi_df.dropna()
        column_list = gedi_df.columns
        self.keys = keys
        self.raw_dataframe = gedi_df
        key_lon = [column_list[i] for i in range(len(column_list)) if 'lon' in column_list[i]][0]
        key_lat = [column_list[i] for i in range(len(column_list)) if 'lat' in column_list[i]][0]
        self.lonlat = gedi_df[[key_lon, key_lat]].values
        # self.lat = gedi_df['latitude'].values
        if 'gedi' in self.data_type:
            self.canopy_height = self.raw_dataframe['elev_highestreturn'].to_numpy() - self.raw_dataframe['elev_lowestmode'].to_numpy()

        return self.raw_dataframe

    def get_size(self, items):
        return np.shape(items)

    def project_point(self, out_epsg, save_path):
        self.out_epsg = out_epsg
        self.path_XY = save_path
        self.XY = general.point_project(self.lonlat[:, 0], self.lonlat[:, 1], self.in_epsg, self.out_epsg)
        df = pd.DataFrame(self.XY, columns=['X', 'Y'])
        df.to_csv(save_path)
        return self.XY
    
    def get_XY(self, out_epsg, path):
        if os.path.exists(path):
            self.out_epsg = out_epsg
            self.XY = pd.read_csv(path).values[:,1:3]
        else:
            self.XY = self.project_point(out_epsg, path)
        self.path_XY = path
        return self.XY
    
    def get_geoidHeight(self, egm_path, geoid_path):
        if os.path.exists(geoid_path):
            self.geoid_height = pd.read_csv(geoid_path).values[:,1]
        else:
            self.geoid_height = general.get_geoid_height(self.lonlat[:, 0], self.lonlat[:, 1], egm_path)
            df = pd.DataFrame(self.geoid_height, columns=['gedi_geoidH'])
            df.to_csv(geoid_path)
        self.geoid_path = geoid_path
        self.egm_path = egm_path
        return self.geoid_height
    
    def get_orthoHeight(self):
        if self.data_type=='gedi':
            self.ortho_height = self.raw_dataframe['elev_lowestmode'].to_numpy() - self.geoid_height
        elif self.data_type=='atl08':
            self.ortho_height = self.raw_dataframe['h_te_bestfit'].to_numpy() - self.geoid_height
        return self.ortho_height
    
    def get_slopeSRTM(self, save_path):
        slope_srtm_path = self.root_directory.replace('/GEDI/', '/others/') + 'slope.tif'
        self.slope_tiff_directory = slope_srtm_path
        self.slope_srtm_directory = save_path
        if os.path.exists(save_path):
            self.srtm_slope = pd.read_csv(save_path)['slope'].values
        else:
            self.srtm_slope = general.extract_value_from_raster(self.lonlat, 'slope', self.slope_tiff_directory,
                                                               self.slope_srtm_directory)
        return self.srtm_slope
    
    def get_slope(self, slope_path, save_path):
        self.slope_tiff_directory = slope_path
        self.slope_directory = save_path
        if os.path.exists(save_path):
            self.slope = pd.read_csv(save_path)['slope'].values
        else:
            self.slope = np.array(general.extract_value_from_raster(self.XY, 'slope', self.slope_tiff_directory,
                                                               self.slope_directory))
        return self.slope

    def get_elevSRTM(self, save_path):
        elev_srtm_path = self.root_directory.replace('/GEDI/', '/others/') + 'srtm90.tif'
        self.srtm_tiff_directory = elev_srtm_path
        self.elev_srtm_directory = save_path
        if os.path.exists(save_path):
            self.srtm_elev = pd.read_csv(save_path).values[:, 3]
        else:
            self.srtm_elev = general.extract_value_from_raster(self.lonlat, 'orthometric_height', self.srtm_tiff_directory,
                                                               self.elev_srtm_directory)
        return self.srtm_elev
    
    def get_aspectSRTM(self, save_path):
        aspect_srtm_path = self.root_directory.replace('/GEDI/', '/others/') + 'aspect.tif'
        self.aspect_tiff_directory = aspect_srtm_path
        self.aspect_srtm_directory = save_path
        if os.path.exists(save_path):
            self.srtm_aspect = pd.read_csv(save_path)['aspect'].values
        else:
            self.srtm_aspect = general.extract_value_from_raster(self.lonlat, 'aspect', self.aspect_tiff_directory,
                                                               self.aspect_srtm_directory)

        return self.srtm_aspect
    
    def get_elev3dep(self, elev3dep_path, save_path):
        self.elev3dep_tiff_directory = elev3dep_path
        self.elev3dep_directory = save_path
        if os.path.exists(save_path):
            self.elev3dep = pd.read_csv(save_path)['elevation'].values
            if 'tipp' in self.area_name:
                self.elev3dep = self.elev3dep * 0.3048
            
        else:
            if 'mend' in self.root_directory:
                XY = general.point_project(self.lonlat[:, 0], self.lonlat[:, 1], self.in_epsg, 26910)
            else:
                XY = general.point_project(self.lonlat[:, 0], self.lonlat[:, 1], self.in_epsg, self.out_epsg)
            self.elev3dep = general.extract_value_from_raster(XY, 'elevation', self.elev3dep_tiff_directory,
                                                                self.elev3dep_directory)
            if 'tipp' in self.area_name:
                self.elev3dep = np.array(self.elev3dep) * 0.3048

        return self.elev3dep
    
    def mask_aoi(self, aoi_json_path, target_crs):
        df = gpd.read_file(open(aoi_json_path, 'r'))
        aoi_original_crs = df.crs.srs
        if aoi_original_crs != target_crs:
            df = df.to_crs(target_crs)
        bounds = df.bounds
        pts = self.lonlat
        idx = np.where((pts[:, 0] >= bounds['minx'][0]) & (pts[:, 0] <= bounds['maxx'][0]) & (pts[:, 1] >= bounds['miny'][0]) & (pts[:, 1] <= bounds['maxy'][0]))[0]
        self.lonlat = self.lonlat[idx, :]
        self.raw_dataframe = self.raw_dataframe.iloc[idx, :]
        # np.savetxt(self.root_directory+self.data_type+'_masked.csv', self.raw_dataframe, delimiter=',', header=','.join(self.raw_dataframe.columns.to_list()), comments='')


    def clip(self, boundary_file):
        self.boundary_file = boundary_file
        if os.path.exists(self.root_directory + 'GEDI_idx_within_boundary.csv'):
            # print('loading...')
            in_area_idx = np.loadtxt(self.root_directory + 'GEDI_idx_within_boundary.csv', delimiter=',')
            in_area_idx = np.array(in_area_idx, dtype=int)
        else:
            print("clipping...")
            in_area_idx = general.boundary_mask(self.boundary_file, self.lonlat)
            in_area_idx = np.where(in_area_idx == True)[0]
            np.savetxt(self.root_directory + 'GEDI_idx_within_boundary.csv', in_area_idx, delimiter=',')
        self.in_area_idx = in_area_idx
        self.apply_index(in_area_idx)

    def apply_index(self, index):
        if type(self.raw_dataframe) != type(None):
            self.raw_dataframe = self.raw_dataframe.iloc[index, :]
        if type(self.lonlat) != type(None):
            self.lonlat = self.lonlat[index, :]
        if type(self.XY) != type(None):
            self.XY = self.XY[index, :]
        if type(self.geoid_height) != type(None):
            self.geoid_height = self.geoid_height[index]
        if type(self.slope) != type(None):
            self.slope = self.slope[index]
        if type(self.ortho_height) != type(None):
            self.ortho_height = self.ortho_height[index]
        if type(self.srtm_slope) != type(None):
            self.srtm_slope = self.srtm_slope[index]
        if type(self.srtm_elev) != type(None):
            self.srtm_elev = np.array(self.srtm_elev)[index]
        if type(self.srtm_aspect) != type(None):
            self.srtm_aspect = self.srtm_aspect[index]
        if type(self.elev3dep) != type(None):
            self.elev3dep = self.elev3dep[index]
        if type(self.canopy_height) != type(None):
            self.canopy_height = self.canopy_height[index]
        if hasattr(self, 'features_idx') and type(self.features_idx) != type(None):
            self.features_idx = self.features_idx[index]

    def filtering(self, thresholds):
        if 'gedi' in self.data_type:
            print(
                'Before filtering: {} (#. {}) error vs reference dem. median:{:.4f}, mean:{:.4f}, std:{:.4f}'.format(
                self.data_type,
                self.ortho_height.shape[0],
                np.nanmedian(self.ortho_height-self.elev3dep),
                np.nanmean(self.ortho_height-self.elev3dep),
                np.nanstd(self.ortho_height-self.elev3dep))
                )
            output = deepcopy(self)
            atl08 = thresholds[2]
            dist, idx = general.nn_(self.XY, atl08.XY, 'number', 6)
            h_atl08_neighbors = atl08.ortho_height[idx]
            h_atl08_neighbors_min = np.nanmin(h_atl08_neighbors, axis=1)
            h_atl08_neighbors_max = np.nanmax(h_atl08_neighbors, axis=1)
            idx = np.where(
                (self.raw_dataframe['sensitivity'] > thresholds[0]) & (self.raw_dataframe['sensitivity'] < thresholds[1])
                # & (self.raw_dataframe['quality_flag']==1) & (self.raw_dataframe['degrade_flag']>0)
                & (self.ortho_height >= h_atl08_neighbors_min) & (self.ortho_height <= h_atl08_neighbors_max)
                )[0]
            
            output.apply_index(idx)
            print(
                'After filtering: {} (#. {}) error vs reference dem. median:{:.4f}, mean:{:.4f}, std:{:.4f}'.format(
                output.data_type,
                output.ortho_height.shape[0],
                np.nanmedian(output.ortho_height-output.elev3dep),
                np.nanmean(output.ortho_height-output.elev3dep),
                np.nanstd(output.ortho_height-output.elev3dep))
                )
            return output

        elif 'atl08' in self.data_type:
            print(
                'Before filtering: {} (#. {}) error vs reference dem. median:{:.4f}, mean:{:.4f}, std:{:.4f}'.format(
                self.data_type,
                self.ortho_height.shape[0],
                np.nanmedian(self.ortho_height-self.elev3dep),
                np.nanmean(self.ortho_height-self.elev3dep),
                np.nanstd(self.ortho_height-self.elev3dep))
                )
                
            idx = np.where(
                (self.raw_dataframe['h_te_uncert'] <= 1e10)
                )[0]
            self.apply_index(idx)
            idx = np.where(
                (self.raw_dataframe['h_te_uncert'] <= np.percentile(self.raw_dataframe['h_te_uncert'], thresholds[0]))
                )[0]
            output = deepcopy(self)
            output.apply_index(idx)
            print(
                'After filtering: {} (#. {}) error vs reference dem. median:{:.4f}, mean:{:.4f}, std:{:.4f}'.format(
                output.data_type,
                output.ortho_height.shape[0],
                np.nanmedian(output.ortho_height-output.elev3dep),
                np.nanmean(output.ortho_height-output.elev3dep),
                np.nanstd(output.ortho_height-output.elev3dep))
                )
            return output
        else:
            print('Not viable data type for filtering.')
            pass

    def find_cross_over(self, other_dataset, distance_threshold, plot=False):
        """
        Find cross over points between two datasets
        """
        dist_lst, idx_lst = general.nn(self.XY, other_dataset.XY, 'number', 10)
        idx_lst[dist_lst > distance_threshold] = -999
        dist_lst[dist_lst > distance_threshold] = np.nan
        idx_lst = idx_lst[:, ~np.all(np.isnan(dist_lst), axis=0)]
        dist_lst = dist_lst[:, ~np.all(np.isnan(dist_lst), axis=0)]
    
        print('Number of points have cross overs: {}'.format(np.sum(~np.all(np.isnan(dist_lst), axis=1))))
        cross_over_flag = np.zeros(self.XY.shape[0], dtype=int)
        cross_over_flag[~np.all(np.isnan(dist_lst), axis=1)] = 1

        cross_over_dx = np.copy(dist_lst)
        cross_over_dy = np.copy(dist_lst)
        for i, idx in enumerate(idx_lst):
            if ~np.all(np.isnan(dist_lst[i, :])):
                cross_over_dx[i, ~np.isnan(dist_lst[i, :])] = self.XY[i, 0] - other_dataset.XY[idx[~np.isnan(dist_lst[i, :])], 0]
                cross_over_dy[i, ~np.isnan(dist_lst[i, :])] = self.XY[i, 1] - other_dataset.XY[idx[~np.isnan(dist_lst[i, :])], 1]

        # dh = self.ortho_height[cross_over_flag == 1] - other_dataset.ortho_height[idx_lst[cross_over_flag == 1, 0]]
        # invalid_idx = np.where((dh < np.percentile(dh, 5)) | (dh > np.percentile(dh, 95)))[0]
        # print('Number of points have invalid cross overs: {}'.format(invalid_idx.shape[0]))
        # print(dh.min(), dh.max(), np.percentile(dh, 5), np.percentile(dh, 95))
        # cross_over_flag[cross_over_flag == 1][invalid_idx] = 0
        # dist_lst[cross_over_flag == 1][invalid_idx] = np.nan
        # idx_lst[cross_over_flag == 1][invalid_idx, :] = -999
        # cross_over_dx[cross_over_flag == 1][invalid_idx] = np.nan
        # cross_over_dy[cross_over_flag == 1][invalid_idx] = np.nan
        
        self.cross_over_flag = cross_over_flag
        self.cross_over_dist_lst = dist_lst
        self.cross_over_idx_lst = idx_lst
        self.cross_over_dx = cross_over_dx
        self.cross_over_dy = cross_over_dy

        if plot:
            plt.figure()
            plt.scatter(self.XY[:, 0], self.XY[:, 1], s=5, c='blue', label='GEDI')
            # plt.scatter(other_dataset.XY[:, 0], other_dataset.XY[:, 1], s=5, c='black', label='ATL08')
            plt.scatter(self.XY[~np.all(np.isnan(dist_lst), axis=1), 0], self.XY[~np.all(np.isnan(dist_lst), axis=1), 1], s=10, c='green', label='GEDI cross over')
            plt.scatter(other_dataset.XY[idx_lst[idx_lst != -999], 0], other_dataset.XY[idx_lst[idx_lst != -999], 1], s=10, c='red', label='ATL08 cross over')
            plt.legend()
            ax = plt.gca()
            ax.set_aspect('equal', adjustable='box')
            plt.show(block=False)
        
    def calc_semivariance(self, neighbor_dataset, variogram, k):
        dist, idx = general.nn(self.XY, neighbor_dataset.XY, 'number', k)
        semivariance_gedi_neighbors = variogram.fitted_model(dist.flatten()).reshape(dist.shape)
        h_gedi_neighbors = neighbor_dataset.ortho_height[idx]

        self.semivariance_neighbors = semivariance_gedi_neighbors
        self.h_neighbors = h_gedi_neighbors
        self.idx_neighbors = idx
        self.dist_neighbors = dist

    def get_neighbors(self, sources, k, distance_threshold):
        dist, idx = general.nn(self.XY, sources, 'number', k)
        valid_idx = np.where(dist[:, 0] <= distance_threshold)[0]
        return dist, idx, valid_idx
    
    def clustering(self, features, args):
        if args['clustering_method'] == 'kmeans':
            self.labels_cluster = clustering.MiniBatchKMeans(args['n_clusters'], n_init='auto', batch_size=1000).fit_predict(features)
        elif args['clustering_method'] == 'dbscan':
            self.labels_cluster = clustering.DBSCAN(eps=args['eps'], min_samples=args['min_samples'], n_jobs=-1).fit(features)
        else:
            raise ValueError("clustering method not supported")
        return self.labels_cluster
    
    
class Regressor:
    def __init__(self, results_directory, regression_method, feature_list):
        self.regression_method = regression_method
        self.results_directory = results_directory
        if not os.path.exists(self.results_directory):
            os.makedirs(self.results_directory)
        self.feature_list = feature_list
        self.features = None
        self.labels = None
        self.reference_height = None
        self.kernel_transormer = None
        if not os.path.exists(self.results_directory):
            os.makedirs(self.results_directory)

    def calc_features(self, gedi_features, gedi_labels, atl08):
        variogram_gedi_atl08 = general.build_variogram(np.concatenate((gedi_labels.XY, atl08.XY),axis=0), 
                                                   np.concatenate((gedi_labels.ortho_height, atl08.ortho_height),axis=0))
        variogram_atl08 = general.build_variogram(atl08.XY, atl08.ortho_height)
        dist, idx = general.nn(gedi_labels.XY, atl08.XY, 'radius', variogram_atl08.parameters[0])

    def set_features(self, features):
        self.features = features
        return self.features
    
    def set_labels(self, labels):
        self.labels = labels
        return self.labels
    
    def regress(self, args):
        if not os.path.exists(self.results_directory):
            os.makedirs(self.results_directory)

        if "random_forest" in self.regression_method:
            regressor, scaler, prediction, train_indices, test_indices = \
                regression.regressor_RF(self.features, self.labels, self.feature_list,
                                        save_path=self.results_directory + '/variable_importance.png', num=args[0])
            self.regressor = regressor
            self.scaler = scaler
            self.train_indices = train_indices
            self.test_indices = test_indices
            self.prediction = prediction
        elif "svr" in self.regression_method:
            if 'custom' in self.regression_method:
                regressor, scaler, scaler_label, prediction, train_indices, test_indices = \
                    regression.regressor_svm(self.features, self.labels, kernel_func='custom', save_path=self.results_directory)
            else:
                regressor, scaler, scaler_label, prediction, train_indices, test_indices = \
                    regression.regressor_svm(self.features, self.labels, kernel_func='rbf', save_path=self.results_directory)
            self.scaler_label = scaler_label
            self.regressor = regressor
            self.scaler = scaler
            self.train_indices = train_indices
            self.test_indices = test_indices
            self.prediction = prediction
            # self.kernel_transormer = rbf_transformer
        elif "krr" in self.regression_method:
            if 'custom' in self.regression_method:
                regressor, scaler, scaler_label, prediction, train_indices, test_indices = \
                    regression.regressor_krr(self.features, self.labels, kernel_func='custom', save_path=self.results_directory)
            else:
                regressor, scaler, scaler_label, prediction, train_indices, test_indices = \
                    regression.regressor_krr(self.features, self.labels, kernel_func='rbf', save_path=self.results_directory)

            self.scaler_label = scaler_label
            self.regressor = regressor
            self.scaler = scaler
            self.train_indices = train_indices
            self.test_indices = test_indices
            self.prediction = prediction
            # self.kernel_transormer = rbf_transformer
        elif "gwr" in self.regression_method:
            XY = self.features[:, :2]
            features = self.features[:, 2:]
            regressor = \
                regression.regressor_gwr(XY, features, self.labels, save_path=self.results_directory)
            # self.model_residuals = model_residuals
            self.regressor = regressor
            # self.scaler = scaler
            # self.prediction = prediction
        elif 'spline' in self.regression_method:
            regressor, scaler, scaler_label, prediction, train_indices, test_indices = \
                regression.regressor_spline(self.features, self.labels, save_path=self.results_directory)
            self.scaler_label = scaler_label
            self.regressor = regressor
            self.scaler = scaler
            self.train_indices = train_indices
            self.test_indices = test_indices
            self.prediction = prediction
        elif 'linear' in self.regression_method:
            regressor, scaler, scaler_label, prediction, train_indices, test_indices = \
                regression.regressor_linear(self.features, self.labels, save_path=self.results_directory)
            self.scaler_label = scaler_label
            self.regressor = regressor
            self.scaler = scaler
            self.train_indices = train_indices
            self.test_indices = test_indices
            self.prediction = prediction
        elif 'poly' in self.regression_method:
            regressor, scaler, scaler_label, prediction, train_indices, test_indices = \
                regression.regressor_polyn(self.features, self.labels, degree=args[0], save_path=self.results_directory)
            self.scaler_label = scaler_label
            self.regressor = regressor
            self.scaler = scaler
            self.train_indices = train_indices
            self.test_indices = test_indices
            self.prediction = prediction
        elif 'mlp' in self.regression_method:
            regressor, scaler, scaler_label, prediction, train_indices, test_indices = \
                regression.regressor_MLP(self.features, self.labels, save_path=self.results_directory)
            self.scaler_label = scaler_label
            self.regressor = regressor
            self.scaler = scaler
            self.train_indices = train_indices
            self.test_indices = test_indices
            self.prediction = prediction
        else:
            pass

    
class MyRaster:
    def __init__(self, epsg, XY, resolution, nodata_value):
        # self.raster_path = raster_path
        self.epsg = epsg
        # self.bbox = bbox
        self.resolution = resolution
        self.XY = XY
        self.x_coords = None
        self.y_coords = None
        self.shape = None
        self.nodata_value = nodata_value

        self.empty_raster = None
        self.atl08_fp_raster = None
        self.gedi_fp_raster = None
        self.predicted_raster = None
        self.reference_raster = None
        self.gedi_fp_mask = None
        self.atl08_fp_mask = None

    def create_empty_raster(self, save_path):
        self.x_coords, self.y_coords, self.empty_raster = general.empty_raster(self.resolution, self.XY, self.nodata_value, save_path, self.epsg)
        self.empty_raster_path = save_path
        self.shape = self.x_coords.shape

    def create_atl08_fp_raster(self, atl08_fp, atl08_z, save_path):
        # if not os.path.exists(save_path):
        self.atl08_fp_raster = general.points2raster(atl08_fp, atl08_z, self.empty_raster_path, self.nodata_value, save_path)
        src = rasterio.open(save_path, 'r')
        self.atl08_fp_raster = src.read(1)
        self.atl08_fp_raster_path = save_path

    def create_GEDI_fp_raster(self, GEDI_fp, GEDI_z, save_path):
        # if not os.path.exists(save_path):
        self.gedi_fp_raster = general.points2raster(GEDI_fp, GEDI_z, self.empty_raster_path, self.nodata_value, save_path)
        src = rasterio.open(save_path, 'r')
        self.gedi_fp_raster = src.read(1)
        self.gedi_fp_raster_path = save_path

    def create_reference_raster(self, reference_data_path):
        # if not os.path.exists(reference_data_path.replace('.tif', '_coregistered_'+str(self.resolution)+'m.tif')):
        if not os.path.exists(reference_data_path.replace('.tif', '_proj.tif')):
            general.raster_project(reference_data_path, self.epsg, reference_data_path.replace('.tif', '_proj.tif'))
        general.reproj_match(reference_data_path.replace('.tif', '_proj.tif'),
                                self.empty_raster_path,
                                reference_data_path.replace('.tif', '_coregistered_'+str(self.resolution)+'m.tif'))
        
        src = rasterio.open(reference_data_path.replace('.tif', '_coregistered_'+str(self.resolution)+'m.tif'), 'r')
        gedi_3dep = src.read(1)
        gedi_3dep[gedi_3dep==src.nodata]=self.nodata_value
        if 'tipp' in self.empty_raster_path:
            gedi_3dep = gedi_3dep * 0.3048
        self.reference_raster = gedi_3dep
        self.reference_coregistered_path = reference_data_path.replace('.tif', '_coregistered_'+str(self.resolution)+'m.tif')

    def get_srtm_raster(self, srtm_path):
        if not os.path.exists(srtm_path.replace('.tif', '_proj.tif')):
            general.raster_project(srtm_path, self.epsg, srtm_path.replace('.tif', '_proj.tif'))
        general.reproj_match(srtm_path.replace('.tif', '_proj.tif'),
                             self.empty_raster_path,
                             srtm_path.replace('.tif', '_coregistered_'+str(self.resolution)+'m.tif'))
        
        src = rasterio.open(srtm_path.replace('.tif', '_coregistered_'+str(self.resolution)+'m.tif'), 'r')
        srtm = src.read(1).astype(np.float32)
        # srtm[srtm==src.nodata]=self.nodata_value

        self.srtm = srtm
        self.srtm_coregistered_path = srtm_path.replace('.tif', '_coregistered_'+str(self.resolution)+'m.tif')

    def clip_raster(self, raster_path, boundary_path, save_path):
        out = general.clip_raster(raster_path, boundary_path, save_path)
        return out
        
    def save_raster(self, example_raster_path, raster, save_path):
        with rasterio.open(example_raster_path, 'r') as src:
            src_crs = src.crs
            src_transform = src.transform
            src_width = src.width
            src_height = src.height

        with rasterio.open(
            save_path, 
            "w",
            driver = "GTiff",
            crs = src_crs,
            transform = src_transform,
            dtype = rasterio.float32,
            count = 1,
            width = src_width,
            height = src_height,
            nodata=self.nodata_value) as dst:
            dst.write(raster, indexes = 1)
        print('Saved raster to {}'.format(save_path))

    def extract_neighbors_raster(self, input_img, neighbor_dist, exclude_self):
        input_img_padding = np.pad(input_img, ((neighbor_dist, neighbor_dist), (neighbor_dist, neighbor_dist)), 'constant', constant_values=0.0)
        window_size = (neighbor_dist*2+1, neighbor_dist*2+1)
        windows = general.rolling_window(input_img_padding, window_size)
        cols, rows = np.meshgrid(np.arange(input_img.shape[1]), np.arange(input_img.shape[0]))
        input_img_neighbors = windows[rows, cols]
        input_img_neighbors = input_img_neighbors.reshape((input_img.shape[0], input_img.shape[1], -1))
        if exclude_self:
            input_img_neighbors = np.delete(input_img_neighbors, int((input_img_neighbors.shape[2]-1)/2), axis=2)
            return input_img_neighbors
        else:
            return input_img_neighbors
        
    def calc_dist_grid(self, window_size):
        x_size, y_size = window_size
        x_arr, y_arr = np.mgrid[0:x_size, 0:y_size]
        cell = (math.floor(window_size[0]/2), math.floor(window_size[0]/2))
        dists = np.sqrt((x_arr - cell[0])**2 + (y_arr - cell[1])**2)
        return dists


    def local_moran(self, input_raster_path, neighbor_dist, save_path):
        # s2=sum((xj-mu)**2)/(n-1)
        # ci=(xi-mu)/s2
        # Ii=ci*sum(wij*(xj-mu))

        with rasterio.open(input_raster_path, 'r') as src:
            input_img = src.read(1)
        input_img_mean = np.mean(input_img)
        input_img_neighbors = self.extract_neighbors_raster(input_img, neighbor_dist, True)
        input_img_neighbors = input_img_neighbors.reshape(-1, input_img_neighbors.shape[-1])

        s2 = np.sum((input_img_neighbors - input_img_mean)**2, axis=1) / (input_img_neighbors.shape[1] - 1)
        c = (input_img.flatten() - input_img_mean) / s2
        dists = self.calc_dist_grid((neighbor_dist*2+1, neighbor_dist*2+1)).flatten()
        dists = np.delete(dists, int((dists.shape[0]-1)/2), axis=0)
        wi = (1/dists) / np.sum(1/dists)
        w = np.repeat(wi.reshape(1,-1), input_img_neighbors.shape[0], axis=0)
        del wi, dists, s2
        I = c * np.sum(w * (input_img_neighbors - input_img_mean), axis=1)

        self.moran_local = np.flipud(I.reshape(input_img.shape))
        self.save_raster(self.empty_raster_path, self.moran_local, save_path)
        self.moran_local_path = save_path

        return self.moran_local


