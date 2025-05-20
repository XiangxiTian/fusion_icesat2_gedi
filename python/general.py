
from matplotlib.path import Path
import pygeodesy
import tqdm
import os, json, math
from pygeodesy.ellipsoidalKarney import LatLon
import numpy as np
import pyproj
from scipy import spatial
from scipy.spatial import Delaunay
from scipy.spatial import distance
import scipy.stats as stats
import pandas as pd
import plotly.express as px
import plotly.io as pio
# from pykrige.ok import OrdinaryKriging
import rasterio
from rasterio.transform import Affine
from scipy.interpolate import griddata
from itertools import repeat
from multiprocessing import Pool
import multiprocessing.pool as mpp
# from metpy.interpolate import interpolate_to_grid, natural_neighbor_to_points
# import pynninterp
from rasterio.warp import reproject, Resampling, calculate_default_transform
from rasterio.enums import Resampling
import rasterio
import geopandas as gpd
from rasterio.mask import mask
import geopandas
from shapely.geometry import Point
from scipy.spatial.distance import cdist
# import skgstat as skg


def boundary_mask(boundary_fpath, targets):
    with open(boundary_fpath) as f:
        boundary_vert = json.load(f)['features'][0]['geometry']['coordinates'][0]

    path_p = Path(np.array(boundary_vert))
    inside_points_idx = path_p.contains_points(targets)
    points_within_p = targets[inside_points_idx, :]
    return inside_points_idx


def extract_value_from_raster(XY, feature_name, path_to_tif, save_path):
    src = rasterio.open(path_to_tif)
    tmp = [x[0] for x in src.sample(XY)]
    df = pd.DataFrame(np.concatenate((XY, np.reshape(tmp, (len(tmp), 1))), axis=1),
                      columns=['lon', 'lat', feature_name])
    df.to_csv(save_path)
    print('Successfully extract {} and saved at {}.'.format(feature_name, save_path))
    return tmp

def extract_neighbors_from_raster(XY, feature_name, k, path_to_tif, save_path):
    src = rasterio.open(path_to_tif)
    band1 = src.read(1)

    height = band1.shape[0]
    width = band1.shape[1]
    cols, rows = np.meshgrid(np.arange(width), np.arange(height))
    xs, ys = rasterio.transform.xy(src.transform, rows, cols)
    raster_XY = np.concatenate((np.array(xs).reshape((-1,1)), np.array(ys).reshape((-1,1))), axis=1)
    raster_value = band1.flatten()

    _, idx = nn(XY, raster_XY, 'number', k)
    tmp = raster_value[idx]
    tmp_columns = [feature_name+str(i+1) for i in range(tmp.shape[1])]

    df = pd.DataFrame(np.concatenate((XY, tmp.reshape((-1, k))), axis=1),
                      columns=['X', 'Y']+tmp_columns)
    df.to_csv(save_path)
    print('Successfully extract {} and saved at {}.'.format(feature_name, save_path))
    return tmp


def csv_load(indir, product, keys=None):
    # print(indir)
    if keys is None:
        keys = ['lon', 'lat', 'elev_lowest', 'degrade', 'quality', 'sensitivity',
                'ndvi', 'evi', 'slope', 'aspect', 'landcover', '3DEP']
    if 'gedi' in product:
        fpaths = [os.path.abspath(os.path.join(indir, x)) for x in os.listdir(indir) if ('GEDI_L2A_' in x) and
                  (x.endswith('.csv'))]
    else:
        fpaths = [os.path.abspath(os.path.join(indir, x)) for x in os.listdir(indir) if ('ATL08_total' in x) and
                  (x.endswith('.csv'))]
    print(fpaths)
    frames = []
    for i, path in enumerate(fpaths):
        df = pd.read_csv(path)
        frames.append(df)
    df_total = pd.concat(frames)
    column_list = df_total.columns
    keys_column_idx = []
    for key in keys:
        keys_column_idx.append([i for i in range(len(column_list)) if key in column_list[i]])
    keys_column_idx = np.array(keys_column_idx).flatten().tolist()
    # print(df_total.shape)
    df_out = df_total.iloc[:, keys_column_idx]
    # print(df_out.shape)
    df_out = df_out.drop_duplicates()
    # print(df_out.shape)
    return df_out, keys


def get_geoid_height(lons, lats, egm_path):
    ginterpolator = pygeodesy.GeoidKarney(egm_path)
    geoid_height = []
    for i, lon in enumerate(lons):
        lat = lats[i]
        pos = LatLon(lat, lon)
        geoid_height.append(ginterpolator(pos))
    geoid_height = np.array(geoid_height)
    return geoid_height.flatten()


def point_project(lon, lat, in_epsg, out_epsg):
    projection = pyproj.Transformer.from_crs(in_epsg, out_epsg, always_xy=True)
    XY = projection.transform(lon, lat)
    XY = np.array(XY).T
    return XY


def nn_(targets, sources, type, criteria):
    tree = spatial.KDTree(sources)
    if 'number' in type:
        dist, idx = tree.query(targets, criteria)
        return dist, idx
    elif 'radius' in type:
        idx = tree.query_ball_point(targets, criteria)
        # dist = [] 
        # for i, pts in enumerate(targets):
        #     dist.append([distance.euclidean(pts, sources[j]) for j in idx[i]])
        return idx
    
def nn(targets, sources, k):
    tree = spatial.KDTree(sources)
    dist, idx = tree.query(targets, k)
    return dist, idx


def nn_del(target, sourceXY):
    source_ = np.append(sourceXY, target.reshape((1, -1)), axis=0)
    vertex_id = source_.shape[0] - 1
    tri = Delaunay(source_)
    index_pointers, indices = tri.vertex_neighbor_vertices
    result_id = indices[index_pointers[vertex_id]:index_pointers[vertex_id + 1]]
    return result_id


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


def nn_delaunay(targets, source):
    sourceXY = source[:, 0:2]
    input = zip(targets, repeat(sourceXY))
    results = []
    n_workers = 10
    chunk_size = 10
    i = 0
    with Pool(processes=n_workers) as pool:
        for ID in tqdm.tqdm(pool.istarmap(nn_del, input, chunksize=chunk_size), total=len(targets)):
            results.append({'GEDI ID': i, 'ATL08 ID': ID.tolist()})
            i += 1

    return results

def get_neighbor_profile(target_dataset1, sourcesXY, sourcesZ, distance_list, save_path):
    if os.path.exists(save_path):
        h_neighbors1 = np.loadtxt(save_path, delimiter=',')
    else:
        from scipy.spatial import cKDTree
        from tqdm import tqdm

        tree = cKDTree(sourcesXY)
        h_neighbors1 = np.zeros((len(target_dataset1.XY), len(distance_list)-1))
        for i in tqdm(range(len((distance_list[1:])))):
            d = distance_list[i+1]
            h_neighbors_idx1 = tree.query_ball_point(target_dataset1.XY, d)
            if i != 0:
                # h_neighbors_idx_inner_circle = tree.query_ball_point(target_dataset1.XY, distance_list[i]
                # h_neighbors_idx1 = np.setdiff1d(h_neighbors_idx1, h_neighbors_idx_inner_circle)
                for j in tqdm(range(len(h_neighbors_idx1)), leave=False):
                    idx = h_neighbors_idx1[j]
                    # idx_inner = h_neighbors_idx_inner_circle[j]
                    # idx = np.setdiff1d(idx, idx_inner)
                    if np.isnan(idx).all():
                        continue
                    else:
                        idx = np.array(idx)
                        hs = sourcesZ[idx[~np.isnan(idx)]]
                        h_neighbors1[j, i] = np.nanmean(hs)
        np.savetxt(save_path, h_neighbors1, delimiter=',')
    return h_neighbors1

def natural_neighbor_points(points_XY, points_Z, points_to_be_interpolated):
    results = natural_neighbor_to_points(points_XY, points_Z, points_to_be_interpolated)
    return results


def voronoi_neighbors(points_XY, points_to_be_interpolated):
    results = []
    pindex = points_XY.shape[0]
    for idx, p in enumerate(points_to_be_interpolated):
        triang = Delaunay(np.append(points_XY, p, axis=0))
        neighbors = triang.vertex_neighbor_vertices[1][
                    triang.vertex_neighbor_vertices[0][pindex]:triang.vertex_neighbor_vertices[0][pindex + 1]]
        results.append(neighbors)
    return results


def simple_idw(dist, z):
    # dist = distance_matrix(x, y, xi, yi)
    # In IDW, weights are 1 / distance
    weights = 1.0 / dist
    # Make weights sum to one
    tmp = weights.sum(axis=1)
    tmp = tmp.reshape((tmp.shape[0], 1))
    weights = weights / tmp
    # Multiply the weights for each interpolated point by all observed Z-values
    zi = np.sum(weights * z, axis=1)
    return zi


def export_kde_raster(Z, XX, YY, min_x, max_x, min_y, max_y, filename):
    '''Export and save a kernel density raster.'''
    # Get resolution
    xres = (max_x - min_x) / len(XX)
    yres = (max_y - min_y) / len(YY)
    # Set transform
    transform = Affine.translation(min_x - xres / 2, min_y - yres / 2) * Affine.scale(xres, yres)

    # Export array as raster
    with rasterio.open(
            filename,
            mode="w",
            driver="GTiff",
            height=Z.shape[0],
            width=Z.shape[1],
            count=1,
            dtype=Z.dtype,
            transform=transform,
    ) as new_dataset:
        new_dataset.write(Z, 1)


# def rasterize(cell_size, footprintsXY, footprintZ, save_path, out_epsg):
#     min_x = np.min(footprintsXY[:, 0])
#     max_x = np.max(footprintsXY[:, 0])
#     min_y = np.min(footprintsXY[:, 1])
#     max_y = np.max(footprintsXY[:, 1])
#     x = np.arange(min_x, max_x, cell_size)
#     y = np.arange(min_y, max_y, cell_size)

#     x_coords, y_coords = np.meshgrid(x, y)

#     grid_z2 = pynninterp.NaturalNeighbour(footprintsXY[:, 0], footprintsXY[:, 1], footprintZ, x_coords, y_coords)

#     xres = (x[-1] - x[0]) / len(x)
#     yres = (y[-1] - y[0]) / len(y)

#     transform = Affine.translation(x[0] - xres / 2, y[0] - yres / 2) * Affine.scale(xres, yres)
#     crs = rasterio.crs.CRS({"init": out_epsg})
   
#     with rasterio.open(save_path.replace('.png', '.tif'), 'w',
#                        height=grid_z2.shape[0], width=grid_z2.shape[1], count=1, dtype=grid_z2.dtype,
#                        transform=transform, crs=crs) as dst:
#         dst.write(grid_z2.reshape(x_coords.shape), 1)
    
#     return x_coords.flatten(), y_coords.flatten(), grid_z2


def empty_raster(cell_size, footprintsXY, fill_value, save_path, out_epsg):
    min_x = np.min(footprintsXY[:, 0])
    max_x = np.max(footprintsXY[:, 0])
    min_y = np.min(footprintsXY[:, 1])
    max_y = np.max(footprintsXY[:, 1])
    x = np.arange(min_x, max_x, cell_size)
    y = np.arange(min_y, max_y, cell_size)
    if x.max() < max_x:
        x = np.append(x, x.max()+cell_size)
    if y.max() < max_y:
        y = np.append(y, y.max()+cell_size)

    x_coords, y_coords = np.meshgrid(x, y)
    z_values = np.ones(x_coords.shape) * fill_value

    xres = (x[-1] - x[0]) / len(x)
    yres = (y[-1] - y[0]) / len(y)

    transform = Affine.translation(x[0] - xres / 2, y[0] - yres / 2) * Affine.scale(xres, yres)
    crs = rasterio.crs.CRS({"init": out_epsg})
   
    with rasterio.open(save_path, 'w',
                       height=x_coords.shape[0], width=x_coords.shape[1], count=1, dtype=z_values.dtype,
                       transform=transform, crs=crs) as dst:
        dst.write(z_values.reshape(x_coords.shape), 1)
    
    return x_coords, y_coords, z_values


def footprint_visualization(data, column_list, feature_index, caption, save_path):
    data_DF = pd.DataFrame(data, columns=column_list)
    # px.set_mapbox_access_token(open(".mapbox_token").read())
    name = column_list[feature_index]
    pio.renderers.default = "browser"
    fig = px.scatter_mapbox(data_DF,
                            lat="lat",
                            lon="lon",
                            hover_name=column_list[feature_index],
                            # hover_data='height',
                            color=column_list[feature_index],
                            color_continuous_scale=px.colors.cyclical.IceFire,
                            zoom=10)
    fig.update_layout(
        mapbox_style="white-bg",
        mapbox_layers=[
            {
                "below": 'traces',
                "sourcetype": "raster",
                "sourceattribution": "United States Geological Survey",
                "source": [
                    "https://basemap.nationalmap.gov/arcgis/rest/services/USGSImageryOnly/MapServer/tile/{z}/{y}/{x}"
                ]
            }
        ],
    )
    fig.update_layout(
        title=caption,
        autosize=True,
        margin={"r": 80, "t": 80, "l": 80, "b": 80})
    fig.update_layout(
        title_x=0.5
    )
    fig.write_html(save_path)
    fig.show()


def data_analysis(data_name, data, feature_list, plot, save_path):
    lons, lats, X, Y, height, reference_height = [], [], [], [], [], []
    for index, feature in enumerate(feature_list):
        if 'lon' in feature:
            lons = data[:, index]
        elif 'lat' in feature:
            lats = data[:, index]
        elif 'X' in feature:
            X = data[:, index]
        elif 'Y' in feature:
            Y = data[:, index]
        elif 'reference' in feature:
            reference_height = data[:, index]
        else:
            height = data[:, index]
    dist, idx = nn(np.stack((X, Y), axis=-1), np.stack((X, Y), axis=-1), 2)
    dist = dist[:, 1]
    error = height - reference_height
    error = error[reference_height > 0]
    reference_height = reference_height[reference_height > 0]
    print("{} data footprint distance \nmin: {}, max: {}, mean: {}, median: {}, std: {}.".
          format(data_name, np.min(dist), np.max(dist), np.mean(dist), np.median(dist), np.std(dist)))
    print("{} data footprint height \nmin: {}, max: {}, mean: {}, median: {}, std: {}.".
          format(data_name, np.min(height), np.max(height), np.mean(height), np.median(height), np.std(height)))
    print("{} data footprint reference height \nmin: {}, max: {}, mean: {}, median: {}, std: {}.".
          format(data_name, np.min(reference_height), np.max(reference_height),
                 np.mean(reference_height), np.median(reference_height), np.std(reference_height)))
    print("{} data footprint height error \nmin: {}, max: {}, mean: {}, median: {}, std: {}.".
          format(data_name, np.min(error), np.max(error), np.mean(error), np.median(error), np.std(error)))

    if plot:
        plt.figure(figsize=(11, 4))
        plt.style.use('seaborn-paper')
        ax1 = plt.subplot(141)
        ax1.hist(dist, bins=50, label=["footprint distance (m)"])
        mu = dist.mean()
        med = np.median(dist)
        sigma = dist.std()
        textstr = '\n'.join((
            r'$\mu=%.2f $m' % (mu,),
            r'$\sigma=%.2f $m' % (sigma,),
            r'$\mathrm{median}=%.2f $m' % (med,)))
        props = dict(boxstyle='round', facecolor='white', alpha=0.5)
        ax1.text(0.5, 0.9, textstr, transform=ax1.transAxes, fontsize=7,
                 verticalalignment='top', bbox=props)
        ax1.set_xlabel("footprint distance (m)")
        ax1.set_ylabel("frequency")
        ax1.legend()

        ax2 = plt.subplot(142)
        ax2.hist(height, bins=50, label=["raw height (m)"])
        mu = height.mean()
        med = np.median(height)
        sigma = height.std()
        textstr = '\n'.join((
            r'$\mu=%.2f $m' % (mu,),
            r'$\sigma=%.2f $m' % (sigma,),
            r'$\mathrm{median}=%.2f $m' % (med,)))
        props = dict(boxstyle='round', facecolor='white', alpha=0.5)
        ax2.text(0.5, 0.9, textstr, transform=ax2.transAxes, fontsize=7,
                 verticalalignment='top', bbox=props)
        ax2.set_xlabel("height (m)")
        ax2.legend()

        ax3 = plt.subplot(143)
        ax3.hist(error, bins=50, label=["height error (m)"])
        mu = error.mean()
        med = np.median(error)
        sigma = error.std()
        textstr = '\n'.join((
            r'$\mu=%.2f $m' % (mu,),
            r'$\sigma=%.2f $m' % (sigma,),
            r'$\mathrm{median}=%.2f $m' % (med,)))
        props = dict(boxstyle='round', facecolor='white', alpha=0.5)
        ax3.text(0.5, 0.9, textstr, transform=ax3.transAxes, fontsize=7,
                 verticalalignment='top', bbox=props)
        ax3.set_xlabel("height error (m)")
        ax3.legend()

        ax4 = plt.subplot(144)
        ax4.hist(reference_height, bins=50, label=["reference height (m)"])
        mu = reference_height.mean()
        med = np.median(reference_height)
        sigma = reference_height.std()
        textstr = '\n'.join((
            r'$\mu=%.2f $m' % (mu,),
            r'$\sigma=%.2f $m' % (sigma,),
            r'$\mathrm{median}=%.2f $m' % (med,)))
        props = dict(boxstyle='round', facecolor='white', alpha=0.5)
        ax4.text(0.5, 0.9, textstr, transform=ax4.transAxes, fontsize=7,
                 verticalalignment='top', bbox=props)
        ax4.set_xlabel("reference height (m)")
        ax4.legend()
        # plt.title(data_name)
        plt.tight_layout()
        plt.savefig(save_path)
        plt.show(block=True)

    # visual = (gvts.EsriImagery * gv.Points(data_DF).options(color=column_list[colorFeature_index],
    #                                                         cmap='plasma', size=3,
    #                                                         tools=['hover'],
    #                                                         clim=(np.min(data[:, colorFeature_index]),
    #                                                               np.max(data[:, colorFeature_index])),
    #                                                         colorbar=True,
    #                                                         clabel=column_list[colorFeature_index],
    #                                                         title=caption,
    #                                                         fontsize={'xticks': 10, 'yticks': 10, 'xlabel': 16,
    #                                                                 'clabel': 12,
    #                                                                 'cticks': 10, 'title': 16,
    #                                                                 'ylabel': 16})).options(height=500,
    #                                                                                         width=900)
    #
    # renderer = gv.renderer('bokeh')
    # renderer.save(visual, caption)


def filter(atl08_ellipsoidH, reference, clip_percent):
    atl08_EH_error = atl08_ellipsoidH - reference
    boolArray = (atl08_EH_error <= np.quantile(atl08_EH_error, clip_percent[1])) & \
                (atl08_EH_error >= np.quantile(atl08_EH_error, clip_percent[0]))
    filter_idx = np.where(boolArray==True)[0]
    return filter_idx


def raster_statistics(indir):
    # img = Image.open(indir)
    with rasterio.open(indir, 'r') as src:
        img_array = src.read(1)
    img_array = img_array.flatten()
    img_array = img_array[~np.isnan(img_array)]
    img_array = img_array[img_array > -200]
    img_array = img_array[img_array < 200]
    # print("Shape of tiff: ", img_array.shape)
    mean = img_array.mean()
    median = np.median(img_array)
    std = img_array.std()
    rmse = np.sqrt(np.nanmean(img_array**2))
    skewness = stats.skew(img_array)
    kurtosis = stats.kurtosis(img_array, fisher=True)
    output = np.array([mean, median, rmse, std, skewness, kurtosis]).flatten()
    # print("mean-----median-----std------skewness-----kurtosis")
    # print(output)
    return output


# def raster_coregister(ref, target, no_data, save_path):
#     with gw.config.update(ref_image=ref):
#         with gw.open(target, resampling="bilinear", nodata=-no_data) as src:
#             src.gw.to_raster(
#                 save_path,
#                 overwrite=True,
#             )
#     return src.data[0]


def raster_project(infile, out_crs, save_path):
    with rasterio.open(infile) as src:
        transform, width, height = calculate_default_transform(
            src.crs, out_crs, src.width, src.height, *src.bounds)
        kwargs = src.meta.copy()
        kwargs.update({
            'crs': out_crs,
            'transform': transform,
            'width': width,
            'height': height
        })

        with rasterio.open(save_path, 'w', **kwargs) as dst:
            for i in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, i),
                    destination=rasterio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=out_crs,
                    resampling=Resampling.nearest)


def reproj_match(infile, match, outfile):
    """Reproject a file to match the shape and projection of existing raster.


    Parameters
    ----------
    infile : (string) path to input file to reproject
    match : (string) path to raster with desired shape and projection
    outfile : (string) path to output file tif
    """
    # open input
    with rasterio.open(infile) as src:
        src_transform = src.transform
    
        # open input to match
        with rasterio.open(match) as match:
            dst_crs = match.crs
            
            # calculate the output transform matrix
            dst_transform, dst_width, dst_height = calculate_default_transform(
                src.crs,     # input CRS
                dst_crs,     # output CRS
                match.width,   # input width
                match.height,  # input height 
                *match.bounds,  # unpacks input outer boundaries (left, bottom, right, top)
            )

        # set properties for output
        dst_kwargs = src.meta.copy()
        dst_kwargs.update({"crs": dst_crs,
                           "transform": dst_transform,
                           "width": dst_width,
                           "height": dst_height})
        # print("Coregistered to shape:", dst_height,dst_width,'\n Affine',dst_transform)
        # open output
        with rasterio.open(outfile, "w", **dst_kwargs) as dst:
            # iterate through bands and write using reproject function
            for i in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, i),
                    destination=rasterio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=dst_transform,
                    dst_crs=dst_crs,
                    # dtype = np.float32,
                    resampling=Resampling.nearest)


# suppose you have a raster file and a shapefile, and they have different projections
# you want to clip the raster file to the shapefile
def clip_raster(in_raster, boundary_shp, save_path):
    Vector=gpd.read_file(boundary_shp)

    with rasterio.open(in_raster) as src:
        in_img = src.read(1)
        Vector=Vector.to_crs(src.crs)
        # print(Vector.crs)
        out_image, out_transform=mask(src,Vector.geometry,nodata=np.nan)
        out_meta=src.meta.copy() # copy the metadata of the source DEM
        
    out_meta.update({
        "driver":"Gtiff",
        "height":out_image.shape[1], # height starts with shape[1]
        "width":out_image.shape[2], # width starts with shape[2]
        "transform":out_transform
    })
                
    with rasterio.open(save_path,'w',**out_meta) as dst:
        dst.write(out_image)
    return out_image[0]

def save_raster(grid_x, grid_y, grid_z, crs, save_path):
    res_x = math.ceil(abs(grid_x[0, 1] - grid_x[0, 0]))
    res_y = math.ceil(abs(grid_y[1, 0] - grid_y[0, 0]))
    transform = Affine.translation(grid_x.min() - res_x / 2, grid_y.max() - res_y / 2) * Affine.scale(res_x, -res_y)
    with rasterio.open(save_path, 'w',
                       height=grid_z.shape[0], width=grid_z.shape[1], count=1, dtype=grid_z.dtype, 
                       crs=crs, transform=transform) as dst:
        dst.write(grid_z, 1)


def points2raster(XY, Z, example_raster, nodata, save_path):
    raster = rasterio.open(example_raster)
    s = [Point(XY[i, 0], XY[i, 1]) for i in range(XY.shape[0])]
    s = geopandas.GeoSeries(s)

    geom_value = ((geom,value) for geom, value in zip(s, Z))

    # Rasterize vector using the shape and transform of the raster
    rasterized = rasterio.features.rasterize(geom_value,
                                    out_shape = raster.shape,
                                    transform = raster.transform,
                                    all_touched = True,
                                    fill = nodata,   # background value
                                    merge_alg = rasterio.enums.MergeAlg.replace,
                                    dtype = np.float32)
    
    with rasterio.open(
            save_path, "w",
            driver = "GTiff",
            crs = raster.crs,
            transform = raster.transform,
            dtype = rasterio.float32,
            count = 1,
            width = raster.width,
            height = raster.height,
            nodata=nodata) as dst:
        dst.write(rasterized, indexes = 1)
    
    return rasterized


def plot_hist(data, num_bin, title):
    # Plot the histogram
    fig, ax = plt.subplots()
    n, bins, patches = ax.hist(data, bins=num_bin)

    # Calculate and display the mean, median, and standard deviation
    mean = np.nanmean(data)
    median = np.nanmedian(data)
    std_dev = np.nanstd(data)
    textstr = '\n'.join((
        r'$\mathrm{median}=%.4f$' % (median, ),
        r'$\mathrm{mean}=%.4f$' % (mean, ),
        r'$\mathrm{std\ dev}=%.4f$' % (std_dev, )))

    # Add a text box to the plot
    props = dict(boxstyle='round', facecolor='white', alpha=0.5)
    ax.text(0.65, 0.95, textstr, transform=ax.transAxes, fontsize=12,
            verticalalignment='top', bbox=props)
    ax.set_title(title)

    # Show the plot
    plt.show()


def rolling_window(arr: np.ndarray, window_size: tuple = (3, 3)) -> np.ndarray:
    """
    Gets a view with a window of a specific size for each element in arr.

    Parameters
    ----------
    arr : np.ndarray
        NumPy 2D array.
    window_size : tuple
        Tuple with the number of rows and columns for the window. Both values
        have to be positive (i.e. greater than zero) and they cannot exceed
        arr dimensions.

    Returns
    -------
    NumPy 4D array

    Notes
    -----
    This function has been slightly adapted from the one presented on:
    https://rigtorp.se/2011/01/01/rolling-statistics-numpy.html.

    It is advised to read the notes on the numpy.lib.stride_tricks.as_strided
    function, which can be found on:
    https://docs.scipy.org/doc/numpy-1.17.0/reference/generated/numpy.lib.stride_tricks.as_strided.html
    """
    # validate window size
    err1 = 'window size must be postive'
    err2 = 'window size exceeds input array dimensions'
    assert window_size[0] > 0 and window_size[1] > 0, err1
    assert window_size[0] <= arr.shape[0] and window_size[1] <= arr.shape[1], err2

    # calculate output array's shape
    y_size = (arr.shape[0] - window_size[0]) + 1
    x_size = (arr.shape[1] - window_size[1]) + 1
    shape = (y_size, x_size) + window_size

    # define strides
    strides = arr.strides * 2

    return np.lib.stride_tricks.as_strided(arr, shape, strides, writeable=False)


def get_gram(x1, x2, my_kernel, gamma):
    return np.array([[my_kernel(_x1, _x2, gamma) for _x2 in x2] for _x1 in x1])
def my_kernel(X, Y, gamma=None):
    """
    Compute the rbf (gaussian) kernel between X and Y::
        K(x, y) = exp(-gamma ||x-y||^2)
    for each pair of rows x in X and y in Y.
    Parameters
    ----------
    X : ndarray of shape (n_samples_X, n_features), coordinates of X locates in the first 2 columns
    Y : ndarray of shape (n_samples_Y, n_features), coordinates of X locates in the first 2 columns
        If `None`, uses `Y=X`.
    gamma : float, default=None
        If None, defaults to 1.0 / n_features.
    Returns
    -------
    kernel_matrix : ndarray of shape (n_samples_X, n_samples_Y)
    """
    if gamma is None:
        # print('gamma is None')
        try:
            gamma = 1.0 / X.shape[1]
        except IndexError:
            gamma = 1.0 / len(X)
    # global gamma
    try:
        dist_matrix = cdist(X[:, 0:2], Y[:, 0:2], 'euclidean')
        feature_matrix = cdist(X[:, 2:], Y[:, 2:], 'sqeuclidean')
    except IndexError:
        dist_matrix = cdist(X[0:2].reshape(1,-1), Y[0:2].reshape(1,-1), 'euclidean')
        feature_matrix = cdist(X[2:].reshape(1,-1), Y[2:].reshape(1,-1), 'sqeuclidean')
    K = -gamma * (feature_matrix + dist_matrix)
    K = np.exp(K, K)  # exponentiate K in-place
    return K
    # return np.exp(-gamma * (np.linalg.norm(x1[2]-x2[2]) + np.linalg.norm(x1[0:2]-x2[0:2])))

def calculate_compass_bearing(p1, p2):
    dx = p2[:,0][:, np.newaxis] - p1[:, 0]
    dy = p2[:,1][:, np.newaxis] - p1[:, 1]

    compass_bearing = np.degrees(np.arctan2(dy.T, dx.T))
    lr_indicator = np.zeros(compass_bearing.shape)
    lr_indicator[np.where(compass_bearing >= 0)] = -1
    lr_indicator[np.where(compass_bearing < 0)] = 1

    if (len(np.where(lr_indicator==-1)[0]) + len(np.where(lr_indicator==1)[0])) != compass_bearing.size:
        raise Exception("Please check the lft rgt track indicator")
    return compass_bearing, lr_indicator

def nn_lr_track(targetsXY, sources, num_neighbor, lr_indicator, nodata):
    _, lr_source = calculate_compass_bearing(targetsXY, sources.XY)
    # xy_source_0 = sources.XY
    dist_mat_source = distance.cdist(targetsXY, sources.XY, 'euclidean')
    idx_source_0 = np.argsort(dist_mat_source, axis=1)
    dist_mat_source_sorted = np.take_along_axis(dist_mat_source, idx_source_0, axis=1)
    lr_source_sorted = np.take_along_axis(lr_source, idx_source_0, axis=1)

    dist_source_0 = np.where(lr_source_sorted == lr_indicator, dist_mat_source_sorted, nodata)
    idx_source_0 = np.where(lr_source_sorted == lr_indicator, idx_source_0, nodata)
    rows, cols = np.where(dist_source_0 != nodata)
    rows_unique, indexs, cnts = np.unique(rows, return_index=True, return_counts=True)
    dist_source_ = np.ones((dist_mat_source.shape[0], num_neighbor)) * nodata
    idx_source_ = np.ones((dist_mat_source.shape[0], num_neighbor), dtype=int) * nodata
    z_source_ = np.ones((dist_mat_source.shape[0], num_neighbor)) * nodata
    xy_source_ = np.ones((dist_mat_source.shape[0], num_neighbor*2)) * nodata
    if sources.data_type == 'gedi':
        sensitivity_source_ = np.ones((dist_mat_source.shape[0], num_neighbor)) * nodata
    for i, row in enumerate(rows_unique):
        if cnts[i] >= num_neighbor:
            dist_source_[row, 0:num_neighbor] = dist_source_0[row, cols[indexs[i]:indexs[i]+num_neighbor]]
            idx_source_[row, 0:num_neighbor] = idx_source_0[row, cols[indexs[i]:indexs[i]+num_neighbor]]
        else:
            dist_source_[row, 0:cnts[i]] = dist_source_0[row, cols[indexs[i]:indexs[i]+cnts[i]]]
            dist_source_[row, cnts[i]:] = dist_source_0[row, cols[indexs[i]+cnts[i]-1]]
            idx_source_[row, 0:cnts[i]] = idx_source_0[row, cols[indexs[i]:indexs[i]+cnts[i]]]
            idx_source_[row, cnts[i]:] = idx_source_0[row, cols[indexs[i]+cnts[i]-1]]
        z_source_[row, :] = sources.ortho_height[idx_source_[row, :]]
        xy_source_[row, :] = sources.XY[idx_source_[row, :]].flatten()
        if sources.data_type == 'gedi':
            sensitivity_source_[row, :] = sources.raw_dataframe['sensitivity'].to_numpy()[idx_source_[row, :]]
    
    dist_source_ = np.where(dist_source_!=nodata, dist_source_, np.nan)
    z_source_ = np.where(z_source_!=nodata, z_source_, np.nan)
    xy_source_ = np.where(xy_source_!=nodata, xy_source_, np.nan)
    if sources.data_type == 'gedi':
        sensitivity_source_ = np.where(sensitivity_source_!=nodata, sensitivity_source_, np.nan)
        return dist_source_, z_source_, sensitivity_source_, xy_source_
    else:
        return dist_source_, z_source_, xy_source_
    
def build_variogram(xy, z):
    model_lst = ['spherical', 'exponential', 'gaussian']
    best_model = None
    best_score = 1e10
    best_v = None
    for m in model_lst:
        v = skg.Variogram(xy, z, model=m, normalize=False, fit_method='lm', maxlag=0.7, n_lags=100)
        if v.rmse < best_score:
            best_score = v.rmse
            best_model = m
            best_v = v
    print("The best model is {}, with rmse: {}".format(best_model, best_score))
    return best_v

def calc_variogram(v, distances):
    return v.fitted_model(distances)


def variogram_sampling(v, num_sample):
    start_value_linear = v.bins.min()
    stop_value_linear = v.bins.max()
    start_log = np.log10(start_value_linear)
    stop_log = np.log10(stop_value_linear)
    log_space = np.logspace(start_log, stop_log, num_sample, base=10)
    return log_space


def print_evaluation_results(values, references, title):
    print(title +': (#. {}), median:{:.4f}, mean:{:.4f}, std:{:.4f}'.format(
        len(values),
        np.nanmedian(values.flatten()-references),
        np.nanmean(values.flatten()-references),
        np.nanstd(values.flatten()-references),
        ))
    

def uniform_sampling(data, num_bins, num_samples):
    cnts, edges = np.histogram(data, bins=num_bins)
    probs = 1 - cnts / cnts.sum()
    probs = probs / np.sum(probs)
    chosen_bins = np.random.choice(np.arange(num_bins), size=num_samples, p=probs)
    chosen_bins_values, chosen_bins_cnts = np.unique(chosen_bins, return_counts=True)
    samples_idx = []
    for i, bin_value in enumerate(chosen_bins_values):
        bin_cnts = chosen_bins_cnts[i]
        if bin_cnts == 0:
            continue
        else:
            b1, b2 = edges[i: i+2]
            data_in_range_idx = np.where((data >= b1) & (data < b2))[0]
            if len(data_in_range_idx) != 0:
                samples_idx += np.random.choice(data_in_range_idx, size=bin_cnts, replace=True).tolist()
    return samples_idx