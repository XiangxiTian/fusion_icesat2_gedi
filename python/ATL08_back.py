import h5py
import os
from zipfile import ZipFile
from pyproj import Proj, transform
from osgeo import ogr, osr
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import csv
# import folium
# from folium.plugins import FastMarkerCluster
import pandas as pd
# import plotly.graph_objects as go
# from plotly.offline import iplot
# import pptk
# from mpl_toolkits.basemap import Basemap


def hdf2csv(atl08_dir, epsg):
    """
    convert the original hdf5 atl08 file into csv file with only key features
    :param atl08_dir: directory of the input hdf5 file
    :param epsg: target reference system
    """
    zip_lst = [os.path.abspath(os.path.join(atl08_dir, x)) for x in os.listdir(atl08_dir) if x.endswith('.zip')]
    subfolder_lst = next(os.walk(atl08_dir))[1]
    if len(zip_lst) != 0 and len(subfolder_lst) == 0:
        for zip_f in zip_lst:
            with ZipFile(zip_f, 'r') as zip_ref:
                zip_ref.extractall(atl08_dir)

    OA_BEAMS = ['gt1r', 'gt1l', 'gt2r', 'gt2l', 'gt3r', 'gt3l']
    for i, folder in enumerate(subfolder_lst):
        print('processing file NO. {}/{}.'.format(i, len(subfolder_lst)))
        fpath = [os.path.abspath(os.path.join(atl08_dir, folder, x)) for x in
                 os.listdir(os.path.join(atl08_dir, folder)) if x.endswith('.h5')]
        fname = [x for x in os.listdir(os.path.join(atl08_dir, folder)) if x.endswith('.h5')]

        with h5py.File(fpath[0], 'r') as f:
            track_id = list(f.get('orbit_info/rgt'))
            # series = []
            beam_id = []
            seg_id_beg, seg_id_end = [], []
            lon, lat = [], []
            X, Y = [], []
            h_te_bestfit, h_te_interp, h_te_median, h_te_mean, h_te_uncert, terrian_n_ph = [], [], [], [], [], []
            h_canopy, h_canopy_uncert, canopy_cover = [], [], []
            atlas_pa = []
            beam_azimuth, beam_coelev = [], []
            day_night, snow = [], []
            # print('Keys: %s' % f.keys())
            # group_key = list(f.keys())
            # for key in group_key:
            # print('%s Under %s : %s' % ('ATL08',key, list(f[key])))
            for beam in OA_BEAMS:
                seg_id_beg = seg_id_beg + (list(f.get(beam + '/land_segments/segment_id_beg')))
                seg_id_end = seg_id_end + (list(f.get(beam + '/land_segments/segment_id_end')))
                lon = lon + (list(f.get(beam + '/land_segments/longitude')))
                lat = lat + (list(f.get(beam + '/land_segments/latitude')))
                h_te_bestfit = h_te_bestfit + (list(f.get(beam + '/land_segments/terrain/h_te_best_fit')))
                h_te_interp = h_te_interp + (list(f.get(beam + '/land_segments/terrain/h_te_interp')))
                h_te_median = h_te_median + (list(f.get(beam + '/land_segments/terrain/h_te_median')))
                h_te_mean = h_te_mean + (list(f.get(beam + '/land_segments/terrain/h_te_mean')))
                h_te_uncert = h_te_uncert + (list(f.get(beam + '/land_segments/terrain/h_te_uncertainty')))
                h_canopy = h_canopy + list(f.get(beam + '/land_segments/canopy/h_canopy'))
                h_canopy_uncert = h_canopy_uncert + list(f.get(beam + '/land_segments/canopy/h_canopy_uncertainty'))
                canopy_cover = canopy_cover + list(f.get(beam + '/land_segments/canopy/landsat_perc'))
                terrian_n_ph = terrian_n_ph + (list(f.get(beam + '/land_segments/terrain/n_te_photons')))
                atlas_pa = atlas_pa + (list(f.get(beam + '/land_segments/atlas_pa')))
                beam_azimuth = beam_azimuth + (list(f.get(beam + '/land_segments/beam_azimuth')))
                beam_coelev = beam_coelev + (list(f.get(beam + '/land_segments/beam_coelev')))
                day_night = day_night + (list(f.get(beam + '/land_segments/night_flag')))
                snow = snow + (list(f.get(beam + '/land_segments/segment_snowcover')))
                beam_id = beam_id + ([beam] * len(seg_id_beg))
            for j, longitude in enumerate(lon):
                tmp = projection(longitude, lat[i], epsg)
                X.append(tmp[0])
                Y.append(tmp[1])
            temp_df = list(
                zip(fname * len(seg_id_beg), track_id * len(seg_id_beg), beam_id, seg_id_beg, seg_id_end,
                    lon, lat, X, Y,
                    h_te_bestfit, h_te_interp, h_te_median, h_te_mean, h_te_uncert, terrian_n_ph,
                    h_canopy, h_canopy_uncert, canopy_cover,
                    atlas_pa, beam_azimuth, beam_coelev, day_night, snow))
            df = pd.DataFrame(temp_df, columns=[
                'fname', 'track_id', 'beam', 'seg_id_beg', 'seg_id_end',
                'lon', 'lat', 'X', 'Y',
                'h_te_bestfit', 'h_te_interp', 'h_te_median', "h_te_mean", 'h_te_uncert', 'terrian_n_ph',
                'h_canopy', 'h_canopy_uncertainty', 'canopy_cover',
                'atlas_pa', 'beam_azimuth', 'beam_coelev', 'day_night', 'snow'])
            temp_fname = fname[0].replace('.h5', '')
            df.to_csv(rawdatapath + '\\' + temp_fname + '.csv')


def projection(lon, lat, epsg):
    point = ogr.Geometry(ogr.wkbPoint)
    point.AddPoint(lon, lat)

    inRef = osr.SpatialReference()
    inRef.ImportFromEPSG(4326)
    outRef = osr.SpatialReference()
    outRef.ImportFromEPSG(epsg)

    coordT = osr.CoordinateTransformation(inRef, outRef)
    point.Transform(coordT)
    X = point.GetX()
    Y = point.GetY()
    return X, Y



rawdatapath='F:\\project_icesat-2\\data\\ATL08\\colorado\\raw_data'
# rawdatafname=[os.path.abspath(os.path.join(rawdatapath,x)) for x in os.listdir(rawdatapath) if x.endswith('.zip')]
# print(rawdatafname)
# for p in rawdatafname:
#     with ZipFile(p,'r') as zip_ref:
#         zip_ref.extractall(rawdatapath)
# ######################################################## ATL08 hdf5 into csv###############################################################
# unzipfolder=next(os.walk(rawdatapath))[1]
# # print(unzipfolder)
# # print(unzipfolder[0])
# l=0
# OA_BEAMS = ['gt1r', 'gt1l', 'gt2r', 'gt2l', 'gt3r', 'gt3l']
# for folder in unzipfolder:
#     l+=1
#     print('processing file NO. {}/{}.'.format(l,len(unzipfolder)))
#     fpath=[os.path.abspath(os.path.join(rawdatapath,folder,x)) for x in os.listdir(os.path.join(rawdatapath,folder)) if x.endswith('.h5')]
#     fname=[x for x in os.listdir(os.path.join(rawdatapath,folder)) if x.endswith('.h5')]
#     # print(fpath,fname)
#     with h5py.File(fpath[0],'r') as f:
#         series=[]
#         seg_id_beg=[]
#         seg_id_end=[]
#         lon=[]
#         lat=[]
#         h_te_bestfit=[]
#         h_te_interp=[]
#         h_te_median=[]
#         h_te_mean=[]
#         h_te_uncert=[]
#         h_canopy=[]
#         h_canopy_uncert=[]
#         beam_id=[]
#         canopy_cover=[]
#         terrian_n_ph=[]
#         atlas_pa=[]
#         beam_azimuth=[]
#         beam_coelev=[]
#         day_night=[]
#         snow=[]
#         track_id=list(f.get('orbit_info/rgt'))
#         # print('Keys: %s' % f.keys())
#         group_key=list(f.keys())
#         # for key in group_key:
#             # print('%s Under %s : %s' % ('ATL08',key, list(f[key])))
#         for beam in OA_BEAMS:
#             seg_id_beg=seg_id_beg+(list(f.get(beam + '/land_segments/segment_id_beg')))
#             # print(list(f.get(beam + '/land_segments/longitude')))
#             seg_id_end=seg_id_end+(list(f.get(beam + '/land_segments/segment_id_end')))
#             lon=lon+(list(f.get(beam + '/land_segments/longitude')))
#             lat=lat+(list(f.get(beam + '/land_segments/latitude')))
#             h_te_bestfit=h_te_bestfit+(list(f.get(beam + '/land_segments/terrain/h_te_best_fit')))
#             h_te_interp=h_te_interp+(list(f.get(beam + '/land_segments/terrain/h_te_interp')))
#             h_te_median=h_te_median+(list(f.get(beam + '/land_segments/terrain/h_te_median')))
#             h_te_mean=h_te_mean+(list(f.get(beam + '/land_segments/terrain/h_te_mean')))
#             h_te_uncert=h_te_uncert+(list(f.get(beam + '/land_segments/terrain/h_te_uncertainty')))
#             h_canopy=h_canopy+list(f.get(beam + '/land_segments/canopy/h_canopy'))
#             h_canopy_uncert=h_canopy_uncert+list(f.get(beam + '/land_segments/canopy/h_canopy_uncertainty'))
#             canopy_cover=canopy_cover+list(f.get(beam + '/land_segments/canopy/landsat_perc'))
#             terrian_n_ph=terrian_n_ph+(list(f.get(beam + '/land_segments/terrain/n_te_photons')))
#             atlas_pa=atlas_pa+(list(f.get(beam + '/land_segments/atlas_pa')))
#             beam_azimuth=beam_azimuth+(list(f.get(beam + '/land_segments/beam_azimuth')))
#             beam_coelev=beam_coelev+(list(f.get(beam + '/land_segments/beam_coelev')))
#             day_night=day_night+(list(f.get(beam + '/land_segments/night_flag')))
#             snow=snow+(list(f.get(beam + '/land_segments/segment_snowcover')))
#             beam_id=beam_id+([beam]*len(seg_id_beg))
#             # series.append({'seg_id_beg':seg_id_beg, 'seg_id_end':seg_id_end, 'lon':lon, 'lat':lat, 'h_te_bestfit':h_te_bestfit,'h_te_uncert':h_te_uncert, 'beam':[beam]*len(seg_id_beg),'track_id':track_id*len(seg_id_beg),'fname':fname*len(seg_id_beg)})
#         temp_df=list(zip(seg_id_beg,seg_id_end,lon,lat,h_te_bestfit,h_te_interp,h_te_median,h_te_mean,h_te_uncert,h_canopy,h_canopy_uncert,canopy_cover,terrian_n_ph,beam_id,atlas_pa,beam_azimuth,beam_coelev,day_night,snow,track_id*len(seg_id_beg),fname*len(seg_id_beg)))
#         df=pd.DataFrame(temp_df,columns=['seg_id_beg', 'seg_id_end', 'lon', 'lat', 'h_te_bestfit','h_te_interp','h_te_median',"h_te_mean",'h_te_uncert','h_canopy','h_canopy_uncertainty',
#         'canopy_cover','terrian_n_ph', 'beam','atlas_pa','beam_azimuth','beam_coelev','day_night','snow','track_id','fname'])
#         temp_fname=fname[0].replace('.h5','')
#         df.to_csv(rawdatapath + '\\'+temp_fname + '.csv')
# ################################################################## ATL03
    # if folder=='167400536':
    #     fpath=[os.path.abspath(os.path.join(rawdatapath,folder,x)) for x in os.listdir(os.path.join(rawdatapath,folder)) if x.endswith('.h5')]
    #     fname=[x for x in os.listdir(os.path.join(rawdatapath,folder)) if x.endswith('.h5')]
    #     print(fpath,fname)
    #     with h5py.File(fpath[0],'r') as f:
    #         print('Keys: %s' % f.keys())
    #         group_key=list(f.keys())
    #         for key in group_key:
    #             print('%s Under %s : %s' % ('ATL08',key, list(f[key])))
    #         classed_pc_indx=np.array(f.get('gt2l/signal_photons/classed_pc_indx'))
    #         classed_pc_flag=np.array(f.get('gt2l/signal_photons/classed_pc_flag'))
    #         ph_seg_id=np.array(f.get('gt2l/signal_photons/ph_segment_id'))
    #         print(classed_pc_indx[0])
    # if folder=='167565976':
    #     fpath=[os.path.abspath(os.path.join(rawdatapath,folder,x)) for x in os.listdir(os.path.join(rawdatapath,folder)) if x.endswith('.h5')]
    #     fname=[x for x in os.listdir(os.path.join(rawdatapath,folder)) if x.endswith('.h5')]
    #     print(fpath,fname)
    #     with h5py.File(fpath[0],'r') as f:
    #         print('Keys: %s' % f.keys())
    #         group_key=list(f.keys())
    #         for key in group_key:
    #             print('%s Under %s : %s' % ('ATL03',key, list(f[key])))
    #         dist_ph_along=np.array(f.get('gt2l/heights/dist_ph_along'))
    #         lat_ph=np.array(f.get('gt2l/heights/lat_ph'))
    #         lon_ph=np.array(f.get('gt2l/heights/lon_ph'))
    #         h_ph=np.array(f.get('gt2l/heights/h_ph'))
    #         seg_id=np.array(f.get('gt2l/geolocation/segment_id'))
    #         seg_ph_cnt=np.array(f.get('gt2l/geolocation/segment_ph_cnt'))
    #         ph_indx_beg=np.array(f.get('gt2l/geolocation/ph_index_beg'))
    #         seg_len=np.array(f.get('gt2l/geolocation/segment_length'))
    #     height=np.stack((dist_ph_along,h_ph,lat_ph,lon_ph),axis=-1)
    #     segment=np.stack((seg_id,seg_ph_cnt,ph_indx_beg,seg_len),axis=-1)

    # if folder=='172449345':
    #     fpath=[os.path.abspath(os.path.join(rawdatapath,folder,x)) for x in os.listdir(os.path.join(rawdatapath,folder)) if x.endswith('.h5')]
    #     fname=[x for x in os.listdir(os.path.join(rawdatapath,folder)) if x.endswith('.h5')]
    #     print(fpath,fname)
    #     with h5py.File(fpath[0],'r') as f:
    #         print('Keys: %s' % f.keys())
    #         group_key=list(f.keys())
    #         for key in group_key:
    #             print('%s Under %s : %s' % ('ATL08',key, list(f[key])))
    #         classed_pc_indx=np.array(f.get('gt2l/signal_photons/classed_pc_indx'))
    #         classed_pc_flag=np.array(f.get('gt2l/signal_photons/classed_pc_flag'))
    #         ph_seg_id=np.array(f.get('gt2l/signal_photons/ph_segment_id'))
    #         print(classed_pc_indx[0])
    # if folder=='172454382':
        # fpath=[os.path.abspath(os.path.join(rawdatapath,folder,x)) for x in os.listdir(os.path.join(rawdatapath,folder)) if x.endswith('.h5')]
        # fname=[x for x in os.listdir(os.path.join(rawdatapath,folder)) if x.endswith('.h5')]
        # print(fpath,fname)
        # with h5py.File(fpath[0],'r') as f:
        #     print('Keys: %s' % f.keys())
        #     group_key=list(f.keys())
        #     for key in group_key:
        #         print('%s Under %s : %s' % ('ATL03',key, list(f[key])))
        #     dist_ph_along=np.array(f.get('gt2l/heights/dist_ph_along'))
        #     lat_ph=np.array(f.get('gt2l/heights/lat_ph'))
        #     lon_ph=np.array(f.get('gt2l/heights/lon_ph'))
        #     h_ph=np.array(f.get('gt2l/heights/h_ph'))
        #     seg_id=np.array(f.get('gt2l/geolocation/segment_id'))
        #     seg_ph_cnt=np.array(f.get('gt2l/geolocation/segment_ph_cnt'))
        #     ph_indx_beg=np.array(f.get('gt2l/geolocation/ph_index_beg'))
        #     seg_len=np.array(f.get('gt2l/geolocation/segment_length'))
        # height5=np.stack((dist_ph_along,h_ph,lat_ph,lon_ph),axis=-1)
        # segment5=np.stack((seg_id,seg_ph_cnt,ph_indx_beg,seg_len),axis=-1)

# ###################################################### ATL08 visualization ############################################
fpath=[os.path.abspath(os.path.join(rawdatapath,x)) for x in os.listdir(rawdatapath) if x.endswith('.csv')]
fname=[x for x in os.listdir(rawdatapath) if x.endswith('.csv')]
# print(fpath,fname)
frames=[]
l=0
for path in fpath:
    l+=1
    print('reading file NO. {}/{}.'.format(l,len(unzipfolder)))
    df=pd.read_csv(path)
    date=[]
    for index,row in df.iterrows():
        fname=row['fname']
        date.append(fname[6:14])
    df['date']=date
    frames.append(df)
tmp=pd.concat(frames)

# TippCo
# data=tmp.loc[(tmp['lon'] >= -87.097) & (tmp['lon'] <= -86.694) & (tmp['lat'] >= 40.214) & (tmp['lat'] <= 40.566)]
# data=tmp.loc[(tmp['lon'] >= -124.042) & (tmp['lon'] <= -122.751) & (tmp['lat'] >= 38.725) & (tmp['lat'] <= 40.007)]
data=tmp.loc[(tmp['lon'] >= -108.22171) & (tmp['lon'] <= -108.00313) & (tmp['lat'] >= 39.00714) & (tmp['lat'] <= 39.06442)]
data.to_csv(rawdatapath + '\\ATL08_total' + '.csv')
print('Saving done!!')

df=pd.read_csv(rawdatapath + '\\ATL08_total' + '.csv')

# P=np.stack((np.array(df['lat']),np.array(df['lon']),np.array(df['h_te_bestfit'])),axis=-1)
# v = pptk.viewer(P) 
# v.set(point_size=0.01)

# fig=plt.figure()
# ax=Axes3D(fig)
# ax.scatter(df['lat'],df['lon'],df['h_te_bestfit'],c='r')
# plt.show()

# oa_plots=[]
# oa_plots.append(go.Scatter3d(x=df['lat'], y=df['lon'], z=df['h_te_bestfit'],
#                                     marker=dict(
#                                         size=2,
#                                         color='black',
#                                         # colorscale='Viridis',   # choose a colorscale
#                                         opacity=0.8
#                                     )))
# oa_plots.append(go.Scatter3d(x=df['lat'], y=df['lon'], z=df['h_te_bestfit']+df['h_canopy'],
#                                     marker=dict(
#                                         size=2,
#                                         color='green',
#                                         # colorscale='Viridis',   # choose a colorscale
#                                         opacity=0.8
#                                     )))
# layout = go.Layout(
#     width=2400,
#     height=1200,
#     scene = dict(aspectmode = 'manual', aspectratio =dict(x=1, y=1, z=0.5),
#                  xaxis=dict(title='Latitude'), yaxis=dict(title='Longitude'), zaxis=dict(title='Elevation (m)'))
# )

# print('Plotting...')

# fig = go.Figure(data=oa_plots, layout=layout)

# iplot(fig)


# height=np.stack((dist_ph_along,h_ph,lon_ph,lat_ph),axis=-1)
# segment=np.stack((seg_id,seg_ph_cnt,ph_indx_beg,seg_len),axis=-1)

# with open('gt2lATL03_height.csv','w') as f:
#     writer=csv.writer(f,delimiter=',')
#     writer.writerows(height)
# with open('gt2lATL03_segment5.csv','w') as f:
#     writer=csv.writer(f,delimiter=',')
#     writer.writerows(segment)

# # flist=['class','h_ph','seg_id','seg_cnt','ph_indx']
# class_ph=np.stack((classed_pc_indx,classed_pc_flag,ph_seg_id),axis=-1)
# with open('gt2lATL03_class_ph5.csv','w') as f:
#     writer=csv.writer(f,delimiter=',')
#     writer.writerows(class_ph)
# lat=np.min(height[:][2])
# lon=np.min(height[:][3])
# coors=height[:,2:4]
# print(coors)
# coors5=height5[:,2:4]
# m=folium.Map(location=[lat,lon],zoom_start=6)
# FastMarkerCluster(data=coors).add_to(m)
# folium.LayerControl().add_to(m)
# for c in coors:
#     # folium.Marker(location=[c[0],c[1]],fill_color='#43d9de', radius=2).add_to(m)
#     folium.CircleMarker(radius=2,location=[c[0],c[1]],color='red').add_to(m)
# print('done!')
# for c in coors5:
#     # folium.Marker(location=[c[0],c[1]],fill_color='#3186cc', radius=2).add_to(m)
#     folium.Circle(radius=2,location=[c[0],c[1]],color='blue').add_to(m)
# print('done!')
# m.save('map1.html')

# fig = plt.figure(figsize=(8, 8))
# m = Basemap(projection='lcc', resolution='h', 
#             lat_0=lat, lon_0=lon,
#             width=1E6, height=1.2E6)
# m.shadedrelief()
# m.drawcoastlines(color='gray')
# m.drawcountries(color='gray')
# m.drawstates(color='gray')
# m.scatter(coors[:,1], coors[:,0], latlon=True,
#           cmap='Reds', alpha=0)