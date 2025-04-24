import ee
import numpy as np
import pandas as pd
import time


def getNDVI(image):
    # Normalized difference vegetation index (NDVI)
    ndvi = image.normalizedDifference(['B8', 'B4']).rename("NDVI")
    image = image.addBands(ndvi)
    return image


def getEVI(image):
    # Compute the EVI using an expression.
    EVI = image.expression(
        '2.5 * ((NIR - RED) / (NIR + 6 * RED - 7.5 * BLUE + 1))', {
            'NIR': image.select('B8').divide(10000),
            'RED': image.select('B4').divide(10000),
            'BLUE': image.select('B2').divide(10000)
        }).rename("EVI")
    image = image.addBands(EVI)
    return image


def addDate(image):
    img_date = ee.Date(image.date())
    img_date = ee.Number.parse(img_date.format('YYYYMMdd'))
    return image.addBands(ee.Image(img_date).rename('date').toInt())


def df2features(df):
    ee.Initialize()
    features = []
    for index, row in df.iterrows():
        poi_geometry = ee.Geometry.Point([row['longitude'], row['latitude']])
        poi_properties = dict(row)
        poi_feature = ee.Feature(poi_geometry, poi_properties)
        features.append(poi_feature)
    ee_fc = ee.FeatureCollection(features)
    # ee_fc.getInfo()
    return ee_fc


def rasterExtraction(image, scale, feature_collection, band_list):
    def func(img=image):
        feature = img.sampleRegions(
            collection=feature_collection,  # feature collection here
            scale=scale  # Cell size of raster
        )
        return feature

    results = image.filterBounds(feature_collection).select(band_list).map(addDate). \
        map(func).flatten()
    return results


def extract_from_ImageCollection(df, data_name, dates):
    eeFC = df2features(df)
    column_df = list(df.columns)
    ee.Initialize()
    if "sentinel" in data_name:
        ImageCollection = ee.ImageCollection('COPERNICUS/S2') \
            .filterDate(dates[0], dates[1]) \
            .map(getNDVI).map(getEVI).map(addDate)
        res = rasterExtraction(ImageCollection, 10, eeFC, ['NDVI', 'EVI'])
        print("Showing an example of retrieved data:")
        print(res.limit(1).getInfo())
        # url_csv = res.getDownloadURL('csv')
        # print(url_csv)
        # task = ee.batch.Export.table.toDrive(**{
        #     'collection': res,
        #     'description': 'vectorsToDriveExample',
        #     'fileFormat': 'csv'
        # })
        # task.start()
        # while task.active():
        #     print('Polling for task (id: {}).'.format(task.id))
        #     time.sleep(60)
        column_df.extend(['NDVI', 'EVI', 'date'])
        nested_list = res.reduceColumns(ee.Reducer.toList(len(column_df)), column_df).values().get(0)
        data = nested_list.getInfo()
        out_df = pd.DataFrame(data, columns=column_df)
        out_df = out_df.groupby(['id']).mean()
        return out_df
    elif "LC_Type1" in data_name:
        ImageCollection = ee.ImageCollection('MODIS/006/MCD12Q1')
        res = rasterExtraction(ImageCollection, 1000, eeFC, data_name)
        # print("Showing an example of retrieved data:")
        # print(res.limit(1).getInfo())
        column_df.extend(['LC_Type1'])
        nested_list = res.reduceColumns(ee.Reducer.toList(len(column_df)), column_df).values().get(0)
        data = nested_list.getInfo()
        out_df = pd.DataFrame(data, columns=column_df)
        out_df = out_df.groupby(['id']).mean()
        return out_df
    elif "elevation" in data_name:
        ImageCollection = ee.ImageCollection('USGS/3DEP/1m')
        res = rasterExtraction(ImageCollection, 1000, eeFC, data_name)
        print("Showing an example of retrieved data:")
        print(res.limit(1).getInfo())
        column_df.extend(["elevation"])
        nested_list = res.reduceColumns(ee.Reducer.toList(len(column_df)), column_df).values().get(0)
        data = nested_list.getInfo()
        out_df = pd.DataFrame(data, columns=column_df)
        out_df = out_df.groupby(['id']).mean()
        return out_df
    elif 'slope' in data_name:
        srtm = ee.Image('CGIAR/SRTM90_V4')
        data = ee.Terrain.slope(srtm)
        res = data.select('slope').sampleRegions(
            collection=eeFC,  # feature collection here
            scale=1000  # Cell size of raster
        )
        print("Showing an example of retrieved data:")
        print(res.first().getInfo())
        column_df.extend(["slope"])
        nested_list = res.reduceColumns(ee.Reducer.toList(len(column_df)), column_df).values().get(0)
        data = nested_list.getInfo()
        out_df = pd.DataFrame(data, columns=column_df)
        out_df = out_df.groupby(['id']).mean()
        return out_df
    elif 'aspect' in data_name:
        srtm = ee.Image('CGIAR/SRTM90_V4')
        data = ee.Terrain.aspect(srtm)
        res = data.select('aspect').sampleRegions(
            collection=eeFC,  # feature collection here
            scale=1000  # Cell size of raster
        )
        print("Showing an example of retrieved data:")
        print(res.first().getInfo())
        column_df.extend(["aspect"])
        nested_list = res.reduceColumns(ee.Reducer.toList(len(column_df)), column_df).values().get(0)
        data = nested_list.getInfo()
        out_df = pd.DataFrame(data, columns=column_df)
        out_df = out_df.groupby(['id']).mean()
        return out_df
    else:
        return None


def get_data_ee(lon_lat_gedi, data_type):
    ee.Initialize()
    if data_type == 'slope':
        srtm = ee.Image('CGIAR/SRTM90_V4')
        data = ee.Terrain.slope(srtm)
    elif data_type == 'LC_Type1':
        data = ee.ImageCollection('MODIS/006/MCD12Q1').first()
    elif data_type == 'elevation':
        data = ee.ImageCollection('USGS/3DEP/1m')
    else:
        return None
    output = np.zeros((lon_lat_gedi.shape[0], 1))
    for i, p in enumerate(lon_lat_gedi):
        if i % 500 == 0:
            print('Getting data from ee for point {}/{}'.format(i, lon_lat_gedi.shape[0]))
        u_lon = p[0]
        u_lat = p[1]
        u_poi = ee.Geometry.Point(u_lon, u_lat)
        scale = 1000
        if data_type == 'elevation':
            output[i, 0] = data.mean().sample(u_poi, scale).first().get(data_type).getInfo()
        else:
            output[i, 0] = data.sample(u_poi, scale).first().get(data_type).getInfo()
    return output


def getGEDI_file_name(start_time, end_time, minx, maxx, miny, maxy, save_path):
    ee.Initialize()
    bbox = ee.Geometry.Rectangle([minx, miny, maxx, maxy])
    gedi_index = ee.FeatureCollection('LARSE/GEDI/GEDI02_A_002_INDEX').filter(
        "time_start > '" + start_time + "' && time_end < '" + end_time + "'") \
        .filterBounds(bbox)
    column_df = ['time_start', 'time_end', 'table_id']

    gedi_index_list = gedi_index.select(column_df).getInfo()['features']
    gedi_index_list1 = [row['properties'] for row in gedi_index_list]
    out_df = pd.DataFrame(gedi_index_list1, columns=column_df)
    out_df.to_csv(save_path)
    return out_df


# # def get_GEDI_L2A(df_filenames, save_path):
#
# def quality_mask(img):
#     return img.updateMask(img.select('quality_flag').eq(1)).updateMask(img.select('degrade_flag').eq(0))
#
# # features_list = ['degrade_flag', 'quality_flag', 'sensitivity',
#                  # 'elev_lowestmode', 'lon_highestreturn', 'lat_highestreturn', '']
# ee.Initialize()
# bbox = ee.Geometry.Rectangle([minx, miny, maxx, maxy])
#
# img_collection = ee.ImageCollection('LARSE/GEDI/GEDI02_A_002_MONTHLY')
# img_collection = img_collection.filterBounds(bbox)


