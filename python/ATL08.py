import h5py
import os
from zipfile import ZipFile
from pyproj import Proj, transform
from osgeo import ogr, osr
import numpy as np
import csv
import pandas as pd
import icepyx as ipx
import scipy.spatial as spatial
from itertools import repeat
from multiprocessing import Pool
import multiprocessing.pool as mpp
import tqdm


# import plotly.graph_objects as go
# from plotly.offline import iplot
# import pptk
# from mpl_toolkits.basemap import Basemap


def download_hdf5(product, bbox, dates_range, root_path):
    region_tipp = ipx.Query(product, bbox, dates_range, start_time='00:00:00', end_time='23:59:59')
    region_tipp.earthdata_login(uid='txx1220', email='txx1220@gmail.com')
    # region_tipp.earthdata_login()
    region_tipp.download_granules(root_path)


def hdf2csv(atl08_dir, epsg):
    """
    convert the original hdf5 atl08 file into csv file with only key features
    :param atl08_dir: directory of the input hdf5 file
    :param epsg: target reference system
    """
    zip_lst = [os.path.abspath(os.path.join(atl08_dir, x)) for x in os.listdir(atl08_dir) if x.endswith('.zip')]
    subfolder_lst = next(os.walk(atl08_dir))[2]
    if len(zip_lst) != 0 and len(subfolder_lst) == 0:
        for zip_f in zip_lst:
            with ZipFile(zip_f, 'r') as zip_ref:
                zip_ref.extractall(atl08_dir)

    OA_BEAMS = ['gt1r', 'gt1l', 'gt2r', 'gt2l', 'gt3r', 'gt3l']
    fnames = [x for x in os.listdir(atl08_dir) if x.endswith('.h5')]
    fpaths = [os.path.abspath(os.path.join(atl08_dir, x)) for x in
              os.listdir(atl08_dir) if x.endswith('.h5')]
    for i, fpath in enumerate(fpaths):
        fname = fnames[i]
        print('processing file NO. {}/{}.'.format(i + 1, len(fpaths)))
        print(fname)

        with h5py.File(fpath, 'r') as f:
            track_id = list(f.get('orbit_info/rgt'))
            # orbit_orient = list(f.get('orbit_info/sc_orient'))
            # series = []
            beam_id = []
            seg_id_beg, seg_id_end = [], []
            lon, lat = [], []
            X, Y = [], []
            h_te_bestfit, h_te_interp, h_te_median, h_te_mean, h_te_uncert, terrian_n_ph = [], [], [], [], [], []
            h_canopy, h_canopy_uncert, canopy_cover, along_track_slope = [], [], [], []
            atlas_pa = []
            beam_azimuth, beam_coelev = [], []
            day_night, snow = [], []
            for beam in OA_BEAMS:
                if (f.get(beam + '/land_segments/segment_id_beg')) is None:
                    continue
                else:
                    seg_id_beg = seg_id_beg + (list(f.get(beam + '/land_segments/segment_id_beg')))
                    seg_id_end = seg_id_end + (list(f.get(beam + '/land_segments/segment_id_end')))
                    lon = lon + (list(f.get(beam + '/land_segments/longitude')))
                    lat = lat + (list(f.get(beam + '/land_segments/latitude')))
                    h_te_bestfit = h_te_bestfit + (list(f.get(beam + '/land_segments/terrain/h_te_best_fit')))
                    h_te_interp = h_te_interp + (list(f.get(beam + '/land_segments/terrain/h_te_interp')))
                    h_te_median = h_te_median + (list(f.get(beam + '/land_segments/terrain/h_te_median')))
                    h_te_mean = h_te_mean + (list(f.get(beam + '/land_segments/terrain/h_te_mean')))
                    h_te_uncert = h_te_uncert + (list(f.get(beam + '/land_segments/terrain/h_te_uncertainty')))
                    along_track_slope = along_track_slope + (list(f.get(beam + '/land_segments/terrain/terrain_slope')))
                    h_canopy = h_canopy + list(f.get(beam + '/land_segments/canopy/h_canopy'))
                    h_canopy_uncert = h_canopy_uncert + list(f.get(beam + '/land_segments/canopy/h_canopy_uncertainty'))
                    canopy_cover = canopy_cover + list(f.get(beam + '/land_segments/canopy/segment_cover'))
                    terrian_n_ph = terrian_n_ph + (list(f.get(beam + '/land_segments/terrain/n_te_photons')))
                    atlas_pa = atlas_pa + (list(f.get(beam + '/land_segments/atlas_pa')))
                    beam_azimuth = beam_azimuth + (list(f.get(beam + '/land_segments/beam_azimuth')))
                    beam_coelev = beam_coelev + (list(f.get(beam + '/land_segments/beam_coelev')))
                    day_night = day_night + (list(f.get(beam + '/land_segments/night_flag')))
                    snow = snow + (list(f.get(beam + '/land_segments/segment_snowcover')))

                    beam_id = beam_id + ([beam] * len(seg_id_beg))
            for j, longitude in enumerate(lon):
                # print(type(longitude), type(lat[j]))
                tmp = projection(float(longitude), float(lat[j]), 4326, epsg)
                X.append(tmp[0])
                Y.append(tmp[1])
            if (X[0] - X[1]) * (Y[0] - Y[1]) > 0:
                orbit_orient = [0]
            elif (X[0] - X[1]) * (Y[0] - Y[1]) < 0:
                orbit_orient = [1]
            else:
                orbit_orient = [-1]
            temp_df = list(
                zip([fname] * len(seg_id_beg), track_id * len(seg_id_beg), orbit_orient * len(seg_id_beg),
                    beam_id, seg_id_beg, seg_id_end,
                    lon, lat, X, Y,
                    h_te_bestfit, h_te_interp, h_te_median, h_te_mean, h_te_uncert, along_track_slope, terrian_n_ph,
                    h_canopy, h_canopy_uncert, canopy_cover,
                    atlas_pa, beam_azimuth, beam_coelev, day_night, snow))
            df = pd.DataFrame(temp_df, columns=[
                'fname', 'track_id', 'orbit_orient', 'beam', 'seg_id_beg', 'seg_id_end',
                'lon', 'lat', 'X', 'Y',
                'h_te_bestfit', 'h_te_interp', 'h_te_median', "h_te_mean", 'h_te_uncert', 'al_track_slope',
                'terrian_n_ph', 'h_canopy', 'h_canopy_uncertainty', 'canopy_cover',
                'atlas_pa', 'beam_azimuth', 'beam_coelev', 'day_night', 'snow'])
            df.to_csv(fpath.replace('.h5', '.csv'))


def csv_merge(rawdatapath):
    fpath = [os.path.abspath(os.path.join(rawdatapath, x)) for x in os.listdir(rawdatapath) if x.endswith('.csv')]
    fname = [x for x in os.listdir(rawdatapath) if x.endswith('.csv')]
    # print(fpath,fname)
    frames = []
    l = 0
    for path in fpath:
        l += 1
        print('reading file NO. {}/{}.'.format(l, len(fpath)))
        df = pd.read_csv(path)
        date = []
        for index, row in df.iterrows():
            fname = row['fname']
            date.append(fname[16:24])
        df['date'] = date
        frames.append(df)
    tmp = pd.concat(frames)

    tmp.to_csv(rawdatapath + '\\ATL08_total' + '.csv')
    print('Saving done!!')


def csv_load(indir, in_csv, keys=None):
    if keys is None:
        keys = ['lon', 'lat', 'elev_lowest', 'degrade', 'quality', 'sensitivity',
                'ndvi', 'evi', 'slope', 'aspect', 'landcover', '3DEP']

    with open(indir + in_csv, 'rb') as f:
        data = np.loadtxt(f, delimiter=',', skiprows=1, usecols=keys)
    return data


def uncertainty_filtering(atl08_data):
    uncertainty = atl08_data[:, -1]


def projection(lon, lat, in_epsg, out_epsg):
    point = ogr.Geometry(ogr.wkbPoint)
    point.AddPoint(lat, lon)

    inRef = osr.SpatialReference()
    inRef.ImportFromEPSG(in_epsg)
    outRef = osr.SpatialReference()
    outRef.ImportFromEPSG(out_epsg)

    coordT = osr.CoordinateTransformation(inRef, outRef)
    point.Transform(coordT)
    X = point.GetX()
    Y = point.GetY()
    return X, Y


def get_slope(target, sources_upper, sources_lower):
    flag = 0
    if (sources_upper.shape[0] < 3) and (sources_lower.shape[0] >= 3):
        sources_upper = np.copy(sources_lower)
        flag = 1
    elif (sources_upper.shape[0] >= 3) and (sources_lower.shape[0] < 3):
        sources_lower = np.copy(sources_upper)
        flag = 1
    dist_m_upper = spatial.distance.cdist(np.array([target]), sources_upper[:, :2], 'euclidean')
    dist_m_lower = spatial.distance.cdist(np.array([target]), sources_lower[:, :2], 'euclidean')
    sorted_idx_upper = np.argsort(dist_m_upper.flatten())
    idx_upper = sorted_idx_upper[:3]
    sorted_idx_lower = np.argsort(dist_m_lower.flatten())
    idx_lower = sorted_idx_lower[:3]

    # dist_upper12 = spatial.distance.euclidean(sources_upper[idx_upper[1], 0:2], sources_upper[idx_upper[2], 0:2])
    # if sources_upper[idx_upper[1], 1] < sources_upper[idx_upper[2], 1]:
    #     slope_upper = (sources_upper[idx_upper[2], 2] - sources_upper[idx_upper[1], 2]) / dist_upper12
    # else:
    #     slope_upper = (sources_upper[idx_upper[1], 2] - sources_upper[idx_upper[2], 2]) / dist_upper12
    #
    # dist_lower12 = spatial.distance.euclidean(sources_lower[idx_lower[1], 0:2], sources_lower[idx_lower[2], 0:2])
    # if sources_lower[idx_lower[1], 1] < sources_lower[idx_lower[2], 1]:
    #     slope_lower = (sources_lower[idx_lower[2], 2] - sources_lower[idx_lower[1], 2]) / dist_lower12
    # else:
    #     slope_lower = (sources_lower[idx_lower[1], 2] - sources_lower[idx_lower[2], 2]) / dist_lower12

    slope_upper = sources_upper[idx_upper[0], 3]
    slope_lower = sources_lower[idx_lower[0], 3]

    weight_lower = 1 / dist_m_lower.flatten()[idx_lower[0]]
    weight_upper = 1 / dist_m_upper.flatten()[idx_upper[0]]
    weight_total = weight_upper + weight_lower
    slope_along_track = (weight_lower / weight_total) * slope_lower + (weight_upper / weight_total) * slope_upper

    if flag == 1:
        slope_across_track = 0
    else:
        dist_upper_lower = spatial.distance.euclidean(sources_upper[idx_upper[0], 0:2],
                                                      sources_lower[idx_lower[0], 0:2])
        if sources_upper[idx_upper[0], 1] >= sources_lower[idx_lower[0], 1]:
            slope_across_track = (sources_upper[idx_upper[0], 2] - sources_lower[idx_lower[0], 2]) / dist_upper_lower
        else:
            slope_across_track = (sources_lower[idx_lower[0], 2] - sources_upper[idx_upper[0], 2]) / dist_upper_lower
    return slope_along_track, slope_across_track


def calc_slope(target, source_pts):
    source_orient0 = source_pts[source_pts[:, 4] == 0, :]
    source_orient1 = source_pts[source_pts[:, 4] == 1, :]
    X, Y = target[0], target[1]
    slopes = np.zeros(4)

    sources_4 = source_orient0[(source_orient0[:, 0] >= X) & (source_orient0[:, 1] >= Y), :]
    sources_2 = source_orient0[(source_orient0[:, 0] < X) & (source_orient0[:, 1] < Y), :]
    sources_1 = source_orient1[(source_orient1[:, 0] >= X) & (source_orient1[:, 1] >= Y), :]
    sources_3 = source_orient1[(source_orient1[:, 0] < X) & (source_orient1[:, 1] < Y), :]

    if (sources_1.shape[0] < 3) and (sources_3.shape[0] < 3):
        slopes[0] = 0
        slopes[2] = 0
    else:
        flag = 0
        if (sources_1.shape[0] < 3) and (sources_3.shape[0] >= 3):
            sources_1 = np.copy(sources_3)
            flag = 1
        elif (sources_1.shape[0] >= 3) and (sources_3.shape[0] < 3):
            sources_3 = np.copy(sources_1)
            flag = 1
        dist_m_upper = spatial.distance.cdist(np.array([target]), sources_1[:, :2], 'euclidean')
        dist_m_lower = spatial.distance.cdist(np.array([target]), sources_3[:, :2], 'euclidean')
        sorted_idx_upper = np.argsort(dist_m_upper.flatten())
        idx_upper = sorted_idx_upper[:3]
        sorted_idx_lower = np.argsort(dist_m_lower.flatten())
        idx_lower = sorted_idx_lower[:3]

        slope_upper = sources_1[idx_upper[0], 3]
        slope_lower = sources_3[idx_lower[0], 3]

        weight_lower = 1 / dist_m_lower.flatten()[idx_lower[0]]
        weight_upper = 1 / dist_m_upper.flatten()[idx_upper[0]]
        weight_total = weight_upper + weight_lower
        slopes[0] = (weight_lower / weight_total) * slope_lower + (weight_upper / weight_total) * slope_upper

        if flag == 1:
            slopes[2] = 0
        else:
            dist_upper_lower = spatial.distance.euclidean(sources_1[idx_upper[0], 0:2],
                                                          sources_3[idx_lower[0], 0:2])
            if sources_1[idx_upper[0], 1] >= sources_3[idx_lower[0], 1]:
                slopes[2] = (sources_1[idx_upper[0], 2] - sources_3[
                    idx_lower[0], 2]) / dist_upper_lower
            else:
                slopes[2] = (sources_3[idx_lower[0], 2] - sources_1[
                    idx_upper[0], 2]) / dist_upper_lower

    if (sources_2.shape[0] < 3) or (sources_4.shape[0] < 3):
        slopes[1] = 0
        slopes[3] = 0
    else:
        flag = 0
        if (sources_2.shape[0] < 3) and (sources_4.shape[0] >= 3):
            sources_2 = np.copy(sources_4)
            flag = 1
        elif (sources_2.shape[0] >= 3) and (sources_4.shape[0] < 3):
            sources_4 = np.copy(sources_2)
            flag = 1
        dist_m_upper = spatial.distance.cdist(np.array([target]), sources_2[:, :2], 'euclidean')
        dist_m_lower = spatial.distance.cdist(np.array([target]), sources_4[:, :2], 'euclidean')
        sorted_idx_upper = np.argsort(dist_m_upper.flatten())
        idx_upper = sorted_idx_upper[:3]
        sorted_idx_lower = np.argsort(dist_m_lower.flatten())
        idx_lower = sorted_idx_lower[:3]

        slope_upper = sources_2[idx_upper[0], 3]
        slope_lower = sources_4[idx_lower[0], 3]

        weight_lower = 1 / dist_m_lower.flatten()[idx_lower[0]]
        weight_upper = 1 / dist_m_upper.flatten()[idx_upper[0]]
        weight_total = weight_upper + weight_lower
        slopes[1] = (weight_lower / weight_total) * slope_lower + (weight_upper / weight_total) * slope_upper

        if flag == 1:
            slopes[3] = 0
        else:
            dist_upper_lower = spatial.distance.euclidean(sources_2[idx_upper[0], 0:2],
                                                          sources_4[idx_lower[0], 0:2])
            if sources_2[idx_upper[0], 1] >= sources_4[idx_lower[0], 1]:
                slopes[3] = (sources_2[idx_upper[0], 2] - sources_4[
                    idx_lower[0], 2]) / dist_upper_lower
            else:
                slopes[3] = (sources_4[idx_lower[0], 2] - sources_2[
                    idx_upper[0], 2]) / dist_upper_lower
    return slopes


def calc_slope_star(args):
    return calc_slope(*args)


def istarmap(self, func, iterable, chunksize=1):
    """starmap-version of imap
    """
    self._check_running()
    if chunksize < 1:
        raise ValueError(
            "Chunksize must be 1+, not {0:n}".format(
                chunksize))

    task_batches = mpp.Pool._get_tasks(func, iterable, chunksize)
    result = mpp.IMapIterator(self)
    self._taskqueue.put(
        (
            self._guarded_task_generation(result._job,
                                          mpp.starmapstar,
                                          task_batches),
            result._set_length
        ))
    return (item for chunk in result for item in chunk)


mpp.Pool.istarmap = istarmap

def get_slopes(targets, source_pts):
    print("calc slopes from ATL08")
    input = zip(targets, repeat(source_pts))
    slopes = []
    with Pool(processes=8) as pool:
        # slopes = pool.starmap(calc_slope, tqdm.tqdm(input, total=len(targets)))
        for slope in tqdm.tqdm(pool.istarmap(calc_slope, input, chunksize=5), total=len(targets)):
            slopes.append(slope)

    # num_targets = targets.shape[0]
    # slope_along_track = np.zeros((num_targets, 2))
    # slope_across_track = np.zeros((num_targets, 2))
    #
    # source_orient0 = source_pts[source_pts[:, 4] == 0, :]
    # source_orient1 = source_pts[source_pts[:, 4] == 1, :]
    #
    # for i, target in enumerate(targets):
    #     if i % 10000 == 0:
    #         print("======== GEDI {}%, {} / {} =========".format(100 * i / num_targets, i, num_targets))
    #     X, Y = target[0], target[1]
    #     sources_4 = source_orient0[(source_orient0[:, 0] >= X) & (source_orient0[:, 1] >= Y), :]
    #     sources_2 = source_orient0[(source_orient0[:, 0] < X) & (source_orient0[:, 1] < Y), :]
    #     sources_1 = source_orient1[(source_orient1[:, 0] >= X) & (source_orient1[:, 1] >= Y), :]
    #     sources_3 = source_orient1[(source_orient1[:, 0] < X) & (source_orient1[:, 1] < Y), :]
    #     if (sources_1.shape[0] < 3) and (sources_3.shape[0] < 3):
    #         continue
    #     else:
    #         slope_along_track[i, 0], slope_across_track[i, 0] = get_slope(target, sources_1, sources_3)
    #
    #     if (sources_2.shape[0] < 3) or (sources_4.shape[0] < 3):
    #         continue
    #     else:
    #         slope_along_track[i, 1], slope_across_track[i, 1] = get_slope(target, sources_2, sources_4)
    # slopes = np.concatenate((slope_along_track, slope_across_track), axis=1)

    return slopes



root_dir = "E:\\OneDrive - purdue.edu\\Projects\\GEDI_ICESAT2\\data\\"
atl08_dir = root_dir + "ATL08\\sanborn\\rawdata\\"
product = 'ATL08'
bbox = [-88.05457749878411, 37.816498862018186, -84.76966538940911, 41.78174686595749]
bbox_piute = [-112.33474710844403, 38.155618239910304, -111.84860208891278, 38.50035641539945]
bbox_sanborn = [-98.33026891855741, 43.85264904206784, -97.85099035410428, 44.19628206013791]

dates_range = ['2018-01-02', '2022-12-14']
download_hdf5(product, bbox_sanborn, dates_range, atl08_dir)
hdf2csv(atl08_dir, 2842)
csv_merge(atl08_dir)

# def visualization(rawdatapath, csv_name):
#     df = pd.read_csv(rawdatapath + csv_name)
#
#     P = np.stack((np.array(df['lat']), np.array(df['lon']), np.array(df['h_te_bestfit'])), axis=-1)
#     v = pptk.viewer(P)
#     v.set(point_size=0.01)
#
#     fig=plt.figure()
#     ax=Axes3D(fig)
#     ax.scatter(df['lat'],df['lon'],df['h_te_bestfit'],c='r')
#     plt.show()
#
#     oa_plots = []
#     oa_plots.append(go.Scatter3d(x=df['lat'], y=df['lon'], z=df['h_te_bestfit'],
#                                         marker=dict(
#                                             size=2,
#                                             color='black',
#                                             # colorscale='Viridis',   # choose a colorscale
#                                             opacity=0.8
#                                         )))
#     oa_plots.append(go.Scatter3d(x=df['lat'], y=df['lon'], z=df['h_te_bestfit']+df['h_canopy'],
#                                         marker=dict(
#                                             size=2,
#                                             color='green',
#                                             # colorscale='Viridis',   # choose a colorscale
#                                             opacity=0.8
#                                         )))
#     layout = go.Layout(
#         width=2400,
#         height=1200,
#         scene=dict(aspectmode='manual', aspectratio=dict(x=1, y=1, z=0.5),
#                    xaxis=dict(title='Latitude'), yaxis=dict(title='Longitude'), zaxis=dict(title='Elevation (m)'))
#     )
#
#     print('Plotting...')
#
#     fig = go.Figure(data=oa_plots, layout=layout)
#
#     iplot(fig)
#
#     height = np.stack((dist_ph_along, h_ph, lon_ph, lat_ph), axis=-1)
#     segment = np.stack((seg_id, seg_ph_cnt, ph_indx_beg, seg_len), axis=-1)
#
#     # with open('gt2lATL03_height.csv','w') as f:
#     #     writer=csv.writer(f,delimiter=',')
#     #     writer.writerows(height)
#     # with open('gt2lATL03_segment5.csv','w') as f:
#     #     writer=csv.writer(f,delimiter=',')
#     #     writer.writerows(segment)
#
#     # # flist=['class','h_ph','seg_id','seg_cnt','ph_indx']
#     # class_ph=np.stack((classed_pc_indx,classed_pc_flag,ph_seg_id),axis=-1)
#     # with open('gt2lATL03_class_ph5.csv','w') as f:
#     #     writer=csv.writer(f,delimiter=',')
#     #     writer.writerows(class_ph)
#     # lat=np.min(height[:][2])
#     # lon=np.min(height[:][3])
#     # coors=height[:,2:4]
#     # print(coors)
#     # coors5=height5[:,2:4]
#     # m=folium.Map(location=[lat,lon],zoom_start=6)
#     # FastMarkerCluster(data=coors).add_to(m)
#     # folium.LayerControl().add_to(m)
#     # for c in coors:
#     #     # folium.Marker(location=[c[0],c[1]],fill_color='#43d9de', radius=2).add_to(m)
#     #     folium.CircleMarker(radius=2,location=[c[0],c[1]],color='red').add_to(m)
#     # print('done!')
#     # for c in coors5:
#     #     # folium.Marker(location=[c[0],c[1]],fill_color='#3186cc', radius=2).add_to(m)
#     #     folium.Circle(radius=2,location=[c[0],c[1]],color='blue').add_to(m)
#     # print('done!')
#     # m.save('map1.html')
#
#     # fig = plt.figure(figsize=(8, 8))
#     # m = Basemap(projection='lcc', resolution='h',
#     #             lat_0=lat, lon_0=lon,
#     #             width=1E6, height=1.2E6)
#     # m.shadedrelief()
#     # m.drawcoastlines(color='gray')
#     # m.drawcountries(color='gray')
#     # m.drawstates(color='gray')
#     # m.scatter(coors[:,1], coors[:,0], latlon=True,
#     #           cmap='Reds', alpha=0)
