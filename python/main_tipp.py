#!/usr/bin/env python
import GEDI
if __name__ == '__main__':
    # Data download:
    # GEDI L2A, slope, aspect (GEE)
    # ATL08 (Python)
#============================================================Define working area========================================================
    area = "tipp"
    cell_size = 30 # 30m DEM
    print("Handeling working area:", area)
    root_dir = "C:/Users/tian133/projects/gedi_icesat2/data/"
    gedi_dir = root_dir + "GEDI/" + area + "/"
    atl08_dir = root_dir + "ATL08/" + area + "/"
    reference_dem_dir = root_dir + "others/" + area + "/dem_3dep.tif"
    srtm_dem_dir = root_dir + "others/" + area + "/srtm.tif"
    egm_path = root_dir + "egm/geoids/egm2008-5.pgm"

    gedi_keys = ['lon', 'lat', 'elev_lowest', 'elev_highestreturn', 'degrade', 'quality', 'sensitivity']
    atl08_keys = ['lon', 'lat', 'h_te_bestfit', 'al_track_slope', 'orbit_orient', 'h_te_uncert']

    in_epsg = "EPSG:4326"
    if 'tipp' in area:
        out_epsg = "EPSG:2968"
        out_epsg = "EPSG:2793" # Tipp
        dist_thresh = 100
    elif "mend" in area:
        out_epsg = "EPSG:2226"
        out_epsg = "EPSG:2767"# Mend
        # ATL08.hdf2csv(root_dir + "ATL08/mend/rawdata/", 2767)
        # ATL08.csv_merge(root_dir + "ATL08/mend/rawdata/")
        dist_thresh = 50
    elif 'piute' in area:
        out_epsg = "EPSG:2851"# piute
        dist_thresh = 50
    elif 'sanborn' in area:
        out_epsg = "EPSG:2842"
        dist_thresh = 50

    if 'mend' in area:
        path_3dep = gedi_dir.replace('/GEDI/', '/others/') + 'CA_30m3DEP.tif'
    if 'tipp' in area:
        path_3dep = gedi_dir.replace('/GEDI/', '/others/') + 'Tipp_30m3DEP.tif'
    if 'piute' in area:
        path_3dep = gedi_dir.replace('/GEDI/', '/others/') + 'piute_30m3DEP_proj.tif'
    if 'sanborn' in area:
        path_3dep = gedi_dir.replace('/GEDI/', '/others/') + 'sanborn_30m3DEP_proj.tif'
    path_srtm = gedi_dir.replace('/GEDI/', '/others/') + 'srtm.tif'
    boundary_fpath = gedi_dir.replace('/GEDI', '/others') + 'boundary.geojson'
    slope_tif_path = gedi_dir.replace('/GEDI/', '/others/') + 'slope_coregistered_30m.tif'

#============================================================Define regreesion method========================================================
    regress_method = "random_forest"
    # regress_method = "svr"
    # regress_method = "svr_custom"
    regress_method = "krr"
    # regress_method = "krr_custom"

#============================================================Run the process========================================================
    feature_list = ['GEDI original height', 'degrade flag', 'quality flag', 'sensitivity', 'slope', 'aspect']
    num_gedi_neighbor_lst = [9]
    for num_gedi_neighbor in num_gedi_neighbor_lst:
        print("===============================================Using number of GEDI neighbors: "+str(num_gedi_neighbor)+"===============================================")
        GEDI.process2(area, gedi_dir, gedi_keys, atl08_dir, atl08_keys, in_epsg, out_epsg, regress_method,
                    num_gedi_neighbor, feature_list, egm_path, path_3dep, path_srtm,
                    cell_size, clip=False, boundary_fpath=boundary_fpath)
        
    print('done')
