import matplotlib.pyplot as plt
from matplotlib import image
from PIL import Image
import pygeodesy
import os
from pygeodesy.ellipsoidalKarney import LatLon
import numpy as np
import pyproj
from scipy import spatial
from scipy.spatial import Delaunay
import scipy.stats as stats
import pandas as pd
import plotly.express as px
import plotly.io as pio
from sklearn.gaussian_process import GaussianProcessRegressor
from pykrige.ok import OrdinaryKriging
import rasterio
from rasterio.transform import Affine
from scipy.interpolate import griddata
from metpy.interpolate import interpolate_to_grid, natural_neighbor_to_points
import pynninterp


# from scipy.spatial import Delaunay


def extract_value_from_raster(XY, feature_name, path_to_tif, save_path):
    src = rasterio.open(path_to_tif)
    tmp = [x[0] for x in src.sample(XY)]
    df = pd.DataFrame(np.concatenate((XY, np.reshape(tmp, (len(tmp), 1))), axis=1),
                      columns=['lon', 'lat', feature_name])
    df.to_csv(save_path)
    print('Successfully extract {} and saved at {}.'.format(feature_name, save_path))
    return tmp


def csv_load(indir, product, keys=None):
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
    keys_column_idx = np.array(keys_column_idx).flatten()
    return df_total.iloc[:, keys_column_idx], keys


def get_geoid_height(lons, lats, egm_path="D:\\Project_ICESat-2\\data\\egm\\geoids\\egm2008-5.pgm"):
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


def nn(XY_gedi, XY_atl08, k):
    tree_atl08 = spatial.KDTree(XY_atl08)
    dist, idx = tree_atl08.query(XY_gedi, k)
    return dist, idx


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


def rasterize(cell_size, footprintsXY, footprintZ, save_path, out_epsg):
    min_x = np.min(footprintsXY[:, 0])
    max_x = np.max(footprintsXY[:, 0])
    min_y = np.min(footprintsXY[:, 1])
    max_y = np.max(footprintsXY[:, 1])
    x = np.arange(min_x, max_x, cell_size)
    y = np.arange(min_y, max_y, cell_size)

    x_coords, y_coords = np.meshgrid(x, y)
    # grid_z0 = griddata(footprintsXY, footprintZ, (x_coords, y_coords), method='linear')
    # grid_z1 = griddata(footprintsXY, footprintZ, (x_coords, y_coords), method='nearest')
    # grid_z2 = griddata(footprintsXY, footprintZ, (x_coords, y_coords), method='cubic')

    grid_z2 = pynninterp.NaturalNeighbour(footprintsXY[:, 0], footprintsXY[:, 1], footprintZ, x_coords, y_coords)

    # gx, gy, grid_z2 = interpolate_to_grid(footprintsXY[:, 0], footprintsXY[:, 1], footprintZ,
    #                                   interp_type='natural_neighbor', hres=90)
    #
    # posXY = np.stack((x_coords.flatten(), y_coords.flatten()), axis=-1)
    # dist, idx = nn(posXY, footprintsXY, 6)
    # z = footprintZ[idx]
    # z_interp = simple_idw(dist, z)
    # grid_z3 = z_interp.reshape(x_coords.shape)

    xres = (x[-1] - x[0]) / len(x)
    yres = (y[-1] - y[0]) / len(y)

    transform = Affine.translation(x[0] - xres / 2, y[0] - yres / 2) * Affine.scale(xres, yres)
    crs = rasterio.crs.CRS({"init": out_epsg})
    # with rasterio.open(save_path.replace('.png', ' linear.tif'), 'w',
    #                    height=grid_z0.shape[0], width=grid_z0.shape[1], count=1, dtype=grid_z0.dtype,
    #                    transform=transform, crs=crs) as dst:
    #     dst.write(grid_z0.reshape(x_coords.shape), 1)
    # with rasterio.open(save_path.replace('.png', ' nearest.tif'), 'w',
    #                    height=grid_z0.shape[0], width=grid_z0.shape[1], count=1, dtype=grid_z0.dtype,
    #                    transform=transform, crs=crs) as dst:
    #     dst.write(grid_z1.reshape(x_coords.shape), 1)
    with rasterio.open(save_path.replace('.png', '.tif'), 'w',
                       height=grid_z2.shape[0], width=grid_z2.shape[1], count=1, dtype=grid_z2.dtype,
                       transform=transform, crs=crs) as dst:
        dst.write(grid_z2.reshape(x_coords.shape), 1)
    # with rasterio.open(save_path.replace('.png', ' idw.tif'), 'w',
    #                    height=grid_z0.shape[0], width=grid_z0.shape[1], count=1, dtype=grid_z0.dtype,
    #                    transform=transform, crs=crs) as dst:
    #     dst.write(grid_z3, 1)

    # plt.subplot(141)
    # ext = (min_x, max_x, min_y, max_y)
    # size = 3
    # alpha = 0.1
    # plt.imshow(grid_z0, extent=ext, origin='lower', cmap='gray')
    # plt.scatter(footprintsXY[:, 0], footprintsXY[:, 1], s=size, alpha=alpha, c=footprintZ, cmap='coolwarm')
    # plt.title('Linear')
    # plt.subplot(142)
    # plt.imshow(grid_z1, extent=ext, origin='lower',cmap='gray')
    # plt.scatter(footprintsXY[:, 0], footprintsXY[:, 1], s=size, alpha=alpha, c=footprintZ, cmap='coolwarm')
    # plt.title('Nearest')
    # plt.subplot(143)
    # plt.imshow(grid_z2, extent=ext, origin='lower',cmap='gray')
    # plt.scatter(footprintsXY[:, 0], footprintsXY[:, 1], s=size, alpha=alpha, c=footprintZ, cmap='coolwarm')
    # plt.title('Cubic')
    # plt.subplot(144)
    # plt.imshow(grid_z3, extent=ext, origin='lower',cmap='gray')
    # plt.scatter(footprintsXY[:, 0], footprintsXY[:, 1], s=size, alpha=alpha, c=footprintZ, cmap='coolwarm')
    # plt.title('IDW')
    # plt.gcf().set_size_inches(31, 7)
    # plt.tight_layout()
    # plt.savefig(save_path)
    # plt.show(block=True)
    # ext = (min_x, max_x, min_y, max_y)
    # fig, ax = plt.subplots()
    # ax.imshow(grid_z3, extent=ext, origin='lower', cmap='gray')
    # scatter = ax.scatter(footprintsXY[:, 0], footprintsXY[:, 1], s=3, alpha=0.6, c=footprintZ, cmap='Spectral_r')
    # handles, labels = scatter.legend_elements(prop="colors", alpha=0.6)
    # legend = ax.legend(handles, labels, loc="upper right", title="Height")
    # plt.title('IDW')
    # plt.gcf().set_size_inches(13, 13)
    # plt.tight_layout()
    # plt.savefig(save_path.replace(".png", "_footprint.png"))
    # plt.show(block=False)
    #
    # ext = (min_x, max_x, min_y, max_y)
    # fig, ax = plt.subplots()
    # ax.imshow(grid_z3, extent=ext, origin='lower', cmap='gray')
    # plt.title('IDW')
    # plt.gcf().set_size_inches(13, 13)
    # plt.tight_layout()
    # plt.savefig(save_path)
    # plt.show(block=True)

    return x_coords.flatten(), y_coords.flatten(), grid_z2.flatten()


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
    return boolArray


def raster_statistics(indir):
    # img = Image.open(indir)
    with rasterio.open(indir, 'r') as src:
        img_array = src.read(1)
    img_array = img_array.flatten()
    img_array = img_array[~np.isnan(img_array)]
    img_array = img_array[img_array > -10000]
    print("Shape of tiff: ", img_array.shape)
    mean = img_array.mean()
    median = np.median(img_array)
    std = img_array.std()
    skewness = stats.skew(img_array)
    kurtosis = stats.kurtosis(img_array, fisher=True)
    output = np.array(["{:.2f}".format(mean), "{:.2f}".format(median), "{:.2f}".format(std),
                       "{:.2f}".format(skewness), "{:.2f}".format(kurtosis)], ).flatten()
    print("mean-----median-----std------skewness-----kurtosis")
    print(output)
    return output



