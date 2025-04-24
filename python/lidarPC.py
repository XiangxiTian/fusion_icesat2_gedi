import os, laspy
import numpy as np
import pandas as pd
import os, gdal, glob, json, pickle
from osgeo import ogr, osr
from scipy import spatial


def las2kdtree(directory):
    file_list = [o for o in os.listdir(directory) if o.endswith('.las')]
    boundaries = np.zeros((len(file_list), 4))
    for i, f in enumerate(file_list):
        fname = directory + '\\' + f
        inFile = laspy.read(fname)
        tmpmax = inFile.header.max
        tmpmin = inFile.header.min
        boundaries[i, 0:2] = tmpmin[0:2]
        boundaries[i, 2:4] = tmpmax[0:2]

        num_returns = inFile.num_returns
        return_num = inFile.return_num
        x = scaled_dimension(inFile, 'x')
        x = x[num_returns == return_num]
        y = scaled_dimension(inFile, 'y')
        y = y[num_returns == return_num]
        z = scaled_dimension(inFile, 'z')
        z = z[num_returns == return_num]
        intensity = inFile.intensity
        intensity = intensity[num_returns == return_num]
        ds = np.vstack([x, y]).transpose()
        tree = spatial.KDTree(ds)

        with open(directory + '\\kdtree\\' + f.replace('.las', '_kdtree.pickle'), 'wb') as file:
            pickle.dump(tree, file)
        # np.savetxt(directory + '\\intensity\\' + f.replace('.las', '_intensity.txt'), intensity, delimiter=',')
        # np.savetxt(directory + '\\z\\' + f.replace('.las', '_z.txt'), z, delimiter=',')
    np.savetxt(directory + '\\boundaries.txt', boundaries, delimiter=',')


def scaled_dimension(las_file, indicator):
    """
    scale the lidar point cloud
    :param las_file: las file
    :param indicator:
    :return:
    """
    if indicator == "x":
        x_dimension = las_file.X
        scale = las_file.header.scale[0]
        offset = las_file.header.offset[0]
        output = x_dimension * scale + offset
    elif indicator == 'y':
        y_dimension = las_file.Y
        scale = las_file.header.scale[1]
        offset = las_file.header.offset[1]
        output = y_dimension * scale + offset
    else:
        z_dimension = las_file.Z
        scale = las_file.header.scale[2]
        offset = las_file.header.offset[2]
        output = z_dimension * scale + offset
    return output


def nn_las(n, m, indir, k):
    """
    find the nearest neighbor point in lidar point cloud
    :param n: x for target point
    :param m: y for target point
    :param indir: directory of las files
    :return:
    """
    output = []
    file_list = [o for o in os.listdir(indir) if o.endswith('.las')]
    dist = 10000000000
    for i, f in enumerate(file_list):
        # if i % 150 == 0:
            # print('Checking in file {}... ({}/{})'.format(f, i, len(file_list)))
        fname = indir + '\\' + f
        # print('reading las file...')
        inFile = laspy.read(fname)
        tmpmax = inFile.header.max
        tmpmin = inFile.header.min
        # print('finished reading las file...')
        if tmpmin[0] < n < tmpmax[0] and tmpmin[1] < m < tmpmax[1]:
            num_returns = inFile.num_returns
            return_num = inFile.return_num
            x = scaled_dimension(inFile, 'x')
            x = x[num_returns == return_num]
            y = scaled_dimension(inFile, 'y')
            y = y[num_returns == return_num]
            z = scaled_dimension(inFile, 'z')
            z = z[num_returns == return_num]
            intensity = inFile.intensity
            intensity = intensity[num_returns == return_num]
            # geoDF.csr='EPSG:2968'
            ds = np.vstack([x, y]).transpose()
            tree = spatial.KDTree(ds)
            tmp = tree.query(np.asarray([n, m]), k)
            wList = np.asarray([1 / tmp[0][0], 1 / tmp[0][1], 1 / tmp[0][2], 1 / tmp[0][3]])
            zList = np.asarray([z[tmp[1][0]], z[tmp[1][1]], z[tmp[1][2]], z[tmp[1][3]]])
            z_interp = np.sum(wList * zList) / np.sum(wList)
            tmp = tree.query(np.asarray([n, m]))
            z_nn = z[tmp[1]]
            output = np.asarray([z_interp, z_nn, tmp[0], f])
            # print('nn for query point is found in {} file {}.'.format(i, f))
            break
        # else:
        #     if i % 150 == 0:
        #         print('Moving to next las file.')
        #     if i == len(file_list):
        #         print('Query point out of ROI.')
    return output


def h_ref3DEP(gedidir, lasdir, outcsv):
    """
    calculate the height difference between gedi and lidar point cloud
    :param gedidir: directory of the gedi shapefile
    :param lasdir: directory of the las files
    :param outcsv: output csv file name
    """
    file_list = [o for o in os.listdir(lasdir) if o.endswith('.las')]
    maxs = [100, 100]
    mins = [999999999999, 999999999999]
    
    for f in file_list:
        fname = lasdir + '\\' + f
        inFile = laspy.file.File(fname, mode='r')
        tmpmax = inFile.header.max
        tmpmin = inFile.header.min
        if tmpmax[0] > maxs[0]:
            maxs[0] = tmpmax[0]
        if tmpmax[1] > maxs[1]:
            maxs[1] = tmpmax[1]
        if tmpmin[0] < mins[0]:
            mins[0] = tmpmin[0]
        if tmpmin[1] < mins[1]:
            mins[1] = tmpmin[1]
    print(mins, maxs)
    # Clip = gp.GeoDataFrame([1], geometry=[maxs[1],mins[0],mins[1],maxs[0]], crs='EPSG:2968')
    driver = ogr.GetDriverByName('ESRI Shapefile')
    inDS = driver.Open(gedidir)
    inLayer = inDS.GetLayer(0)
    X, Y, Z_GEDI, Z_3DEP_interp, Z_3DEP_nn = [], [], [], [], []
    dist, las_src = [], []
    MSL = []
    l = 0
    for feature in inLayer:
        l += 1
        print('Looking nn for point {}/{}.'.format(l, len(inLayer)))
        geom = feature.GetGeometryRef()
        # print(geom.GetX(),geom.GetY())
        X.append(geom.GetX())
        Y.append(geom.GetY())
        Z_GEDI.append(feature.GetField('elev'))
        MSL.append(feature.GetField('mSeaLevel'))
        if mins[0] < geom.GetX() < maxs[0] and mins[1] < geom.GetY() < maxs[1]:
            outp = nn_las(geom.GetX(), geom.GetY(), lasdir)
            if len(outp) != 0:
                Z_3DEP_interp.append(float(outp[0]))
                Z_3DEP_nn.append(float(outp[1]))
                dist.append(outp[2])
                las_src.append(outp[3])
            else:
                Z_3DEP_interp.append(-9999)
                Z_3DEP_nn.append(-9999)
                dist.append(-9999)
                las_src.append(-9999)
        else:
            Z_3DEP_interp.append(-9999)
            Z_3DEP_nn.append(-9999)
            dist.append(-9999)
            las_src.append(-9999)
            print('Query point out of ROI.')
        if l % 1000 == 0:
            # print(Z_3DEP_interp)
            Z_3DEP_interp1 = np.asarray(Z_3DEP_interp) * 0.3048
            Z_3DEP_nn1 = np.asarray(Z_3DEP_nn) * 0.3048
            diff_interp1 = np.asarray(Z_GEDI) - np.asarray(MSL) - Z_3DEP_interp1
            diff_nn1 = np.asarray(Z_GEDI) - np.asarray(MSL) - Z_3DEP_nn1
            res = np.dstack((X, Y, Z_GEDI, Z_3DEP_interp1, Z_3DEP_nn1, dist, diff_interp1, diff_nn1, MSL))[0]
            
            print("Start writing to csv...")
            tmp = outcsv.replace('.csv', '_{}.csv'.format(l))
            np.savetxt(tmp,res,delimiter=',',fmt='%s',header="X,Y,Z_GEDI,Z_3DEP_interp,Z_3DEP_nn,dist,diff_interp,diff_nn,MSL_GEDI")         
        # print(len(X),len(Y),len(Z_GEDI),len(Z_3DEP_interp))
    Z_3DEP_interp=np.asarray(Z_3DEP_interp)*0.3048
    Z_3DEP_nn=np.asarray(Z_3DEP_nn)*0.3048
    diff_interp=np.asarray(Z_GEDI)-np.asarray(MSL)-Z_3DEP_interp
    diff_nn=np.asarray(Z_GEDI)-np.asarray(MSL)-Z_3DEP_nn
    res=np.dstack((X,Y,Z_GEDI,Z_3DEP_interp,Z_3DEP_nn,dist,diff_interp,diff_nn,MSL))[0]
    print("Start writing to csv...")
    np.savetxt(outcsv,res,delimiter=',',fmt='%s',header="X,Y,Z_GEDI,Z_3DEP_interp,Z_3DEP_nn,dist,diff_interp,diff_nn,MSL_GEDI")





