import os
import re
import io
import copy
import logging
import numpy as np
import scipy.interpolate
import geopandas as gpd
import h5py
import pandas as pd
from shapely.geometry import box

#this function from icesat2_toolkit
# PURPOSE: read ICESat-2 ATL03 HDF5 data files
def read_granule(FILENAME, ATTRIBUTES=False, **kwargs):
    """
    Reads ICESat-2 ATL03 Global Geolocated Photons data files

    Parameters
    ----------
    FILENAME: str
        full path to ATL03 file
    ATTRIBUTES: bool, default False
        read file, group and variable attributes

    Returns
    -------
    IS2_atl03_mds: dict
        ATL03 variables
    IS2_atl03_attrs: dict
        ATL03 attributes
    IS2_atl03_beams: list
        valid ICESat-2 beams within ATL03 file
    """
    # Open the HDF5 file for reading
    if isinstance(FILENAME, io.IOBase):
        fileID = h5py.File(FILENAME, 'r')
    else:
        fileID = h5py.File(os.path.expanduser(FILENAME), 'r')

    # Output HDF5 file information
    logging.info(fileID.filename)
    logging.info(list(fileID.keys()))

    # allocate python dictionaries for ICESat-2 ATL03 variables and attributes
    IS2_atl03_mds = {}
    IS2_atl03_attrs = {}

    # read each input beam within the file
    IS2_atl03_beams = []
    for gtx in [k for k in fileID.keys() if bool(re.match(r'gt\d[lr]',k))]:
        # check if subsetted beam contains data
        # check in both the geolocation and heights groups
        try:
            fileID[gtx]['geolocation']['segment_id']
            fileID[gtx]['heights']['delta_time']
        except KeyError:
            pass
        else:
            IS2_atl03_beams.append(gtx)

    # for each included beam
    for gtx in IS2_atl03_beams:
        # get each HDF5 variable
        IS2_atl03_mds[gtx] = {}
        IS2_atl03_mds[gtx]['heights'] = {}
        IS2_atl03_mds[gtx]['geolocation'] = {}
        IS2_atl03_mds[gtx]['bckgrd_atlas'] = {}
        IS2_atl03_mds[gtx]['geophys_corr'] = {}
        # ICESat-2 Measurement Group
        for key,val in fileID[gtx]['heights'].items():
            IS2_atl03_mds[gtx]['heights'][key] = val[:]
        # ICESat-2 Geolocation Group
        for key,val in fileID[gtx]['geolocation'].items():
            IS2_atl03_mds[gtx]['geolocation'][key] = val[:]
        # ICESat-2 Background Photon Rate Group
        for key,val in fileID[gtx]['bckgrd_atlas'].items():
            IS2_atl03_mds[gtx]['bckgrd_atlas'][key] = val[:]
        # ICESat-2 Geophysical Corrections Group: Values for tides (ocean,
        # solid earth, pole, load, and equilibrium), inverted barometer (IB)
        # effects, and range corrections for tropospheric delays
        for key,val in fileID[gtx]['geophys_corr'].items():
            IS2_atl03_mds[gtx]['geophys_corr'][key] = val[:]

        # Getting attributes of included variables
        if ATTRIBUTES:
            # Getting attributes of IS2_atl03_mds beam variables
            IS2_atl03_attrs[gtx] = {}
            IS2_atl03_attrs[gtx]['heights'] = {}
            IS2_atl03_attrs[gtx]['geolocation'] = {}
            IS2_atl03_attrs[gtx]['bckgrd_atlas'] = {}
            IS2_atl03_attrs[gtx]['geophys_corr'] = {}
            # Global Group Attributes
            for att_name,att_val in fileID[gtx].attrs.items():
                IS2_atl03_attrs[gtx][att_name] = att_val
            # ICESat-2 Measurement Group
            for key,val in fileID[gtx]['heights'].items():
                IS2_atl03_attrs[gtx]['heights'][key] = {}
                for att_name,att_val in val.attrs.items():
                    IS2_atl03_attrs[gtx]['heights'][key][att_name]=att_val
            # ICESat-2 Geolocation Group
            for key,val in fileID[gtx]['geolocation'].items():
                IS2_atl03_attrs[gtx]['geolocation'][key] = {}
                for att_name,att_val in val.attrs.items():
                    IS2_atl03_attrs[gtx]['geolocation'][key][att_name]=att_val
            # ICESat-2 Background Photon Rate Group
            for key,val in fileID[gtx]['bckgrd_atlas'].items():
                IS2_atl03_attrs[gtx]['bckgrd_atlas'][key] = {}
                for att_name,att_val in val.attrs.items():
                    IS2_atl03_attrs[gtx]['bckgrd_atlas'][key][att_name]=att_val
            # ICESat-2 Geophysical Corrections Group
            for key,val in fileID[gtx]['geophys_corr'].items():
                IS2_atl03_attrs[gtx]['geophys_corr'][key] = {}
                for att_name,att_val in val.attrs.items():
                    IS2_atl03_attrs[gtx]['geophys_corr'][key][att_name]=att_val

    # ICESat-2 spacecraft orientation at time
    IS2_atl03_mds['orbit_info'] = {}
    IS2_atl03_attrs['orbit_info'] = {}
    for key,val in fileID['orbit_info'].items():
        IS2_atl03_mds['orbit_info'][key] = val[:]
        # Getting attributes of group and included variables
        if ATTRIBUTES:
            # Global Group Attributes
            for att_name,att_val in fileID['orbit_info'].attrs.items():
                IS2_atl03_attrs['orbit_info'][att_name] = att_val
            # Variable Attributes
            IS2_atl03_attrs['orbit_info'][key] = {}
            for att_name,att_val in val.attrs.items():
                IS2_atl03_attrs['orbit_info'][key][att_name] = att_val

    # information ancillary to the data product
    # number of GPS seconds between the GPS epoch (1980-01-06T00:00:00Z UTC)
    # and ATLAS Standard Data Product (SDP) epoch (2018-01-01T00:00:00Z UTC)
    # Add this value to delta time parameters to compute full gps_seconds
    # could alternatively use the Julian day of the ATLAS SDP epoch: 2458119.5
    # and add leap seconds since 2018-01-01T00:00:00Z UTC (ATLAS SDP epoch)
    IS2_atl03_mds['ancillary_data'] = {}
    IS2_atl03_attrs['ancillary_data'] = {}
    ancillary_keys = ['atlas_sdp_gps_epoch','data_end_utc','data_start_utc',
        'end_cycle','end_geoseg','end_gpssow','end_gpsweek','end_orbit',
        'end_region','end_rgt','granule_end_utc','granule_start_utc','release',
        'start_cycle','start_geoseg','start_gpssow','start_gpsweek',
        'start_orbit','start_region','start_rgt','version']
    for key in ancillary_keys:
        # get each HDF5 variable
        IS2_atl03_mds['ancillary_data'][key] = fileID['ancillary_data'][key][:]
        # Getting attributes of group and included variables
        if ATTRIBUTES:
            # Variable Attributes
            IS2_atl03_attrs['ancillary_data'][key] = {}
            for att_name,att_val in fileID['ancillary_data'][key].attrs.items():
                IS2_atl03_attrs['ancillary_data'][key][att_name] = att_val

    # transmit-echo-path (tep) parameters
    IS2_atl03_mds['ancillary_data']['tep'] = {}
    IS2_atl03_attrs['ancillary_data']['tep'] = {}
    for key,val in fileID['ancillary_data']['tep'].items():
        # get each HDF5 variable
        IS2_atl03_mds['ancillary_data']['tep'][key] = val[:]
        # Getting attributes of group and included variables
        if ATTRIBUTES:
            # Variable Attributes
            IS2_atl03_attrs['ancillary_data']['tep'][key] = {}
            for att_name,att_val in val.attrs.items():
                IS2_atl03_attrs['ancillary_data']['tep'][key][att_name] = att_val

    # channel dead time and first photon bias derived from ATLAS calibration
    cal1,cal2 = ('ancillary_data','calibrations')
    for var in ['dead_time','first_photon_bias']:
        IS2_atl03_mds[cal1][var] = {}
        IS2_atl03_attrs[cal1][var] = {}
        for key,val in fileID[cal1][cal2][var].items():
            # get each HDF5 variable
            if isinstance(val, h5py.Dataset):
                IS2_atl03_mds[cal1][var][key] = val[:]
            elif isinstance(val, h5py.Group):
                IS2_atl03_mds[cal1][var][key] = {}
                for k,v in val.items():
                    IS2_atl03_mds[cal1][var][key][k] = v[:]
            # Getting attributes of group and included variables
            if ATTRIBUTES:
                # Variable Attributes
                IS2_atl03_attrs[cal1][var][key] = {}
                for att_name,att_val in val.attrs.items():
                    IS2_atl03_attrs[cal1][var][key][att_name] = att_val
                if isinstance(val, h5py.Group):
                    for k,v in val.items():
                        IS2_atl03_attrs[cal1][var][key][k] = {}
                        for att_name,att_val in val.attrs.items():
                            IS2_atl03_attrs[cal1][var][key][k][att_name]=att_val

    # get ATLAS impulse response variables for the transmitter echo path (TEP)
    tep1,tep2 = ('atlas_impulse_response','tep_histogram')
    IS2_atl03_mds[tep1] = {}
    IS2_atl03_attrs[tep1] = {}
    for pce in ['pce1_spot1','pce2_spot3']:
        IS2_atl03_mds[tep1][pce] = {tep2:{}}
        IS2_atl03_attrs[tep1][pce] = {tep2:{}}
        # for each TEP variable
        for key,val in fileID[tep1][pce][tep2].items():
            IS2_atl03_mds[tep1][pce][tep2][key] = val[:]
            # Getting attributes of included variables
            if ATTRIBUTES:
                # Global Group Attributes
                for att_name,att_val in fileID[tep1][pce][tep2].attrs.items():
                    IS2_atl03_attrs[tep1][pce][tep2][att_name] = att_val
                # Variable Attributes
                IS2_atl03_attrs[tep1][pce][tep2][key] = {}
                for att_name,att_val in val.attrs.items():
                    IS2_atl03_attrs[tep1][pce][tep2][key][att_name] = att_val

    # Global File Attributes
    if ATTRIBUTES:
        for att_name,att_val in fileID.attrs.items():
            IS2_atl03_attrs[att_name] = att_val

    # Closing the HDF5 file
    fileID.close()
    # Return the datasets and variables
    return (IS2_atl03_mds,IS2_atl03_attrs,IS2_atl03_beams)
  
# requires that the input gdf has ranged index values i
# will need to change if index is changed to time or something
# this currently checks point in polygon for EVERY point
# would be significantly sped up if evaluated at 10m or something similar
# maybe later, fine for now
def isolate_sea_land_photons(shoreline_data_path, ICESat2_GDF):
    # try loading the shoreline data
    try:
        ICESat2_GDF.insert(0, 'lat', ICESat2_GDF.geometry.y, False)
        ICESat2_GDF.insert(0, 'lon', ICESat2_GDF.geometry.x, False)

        # allocation of to be used arrays
        zero_int_array = np.int64(np.zeros_like(ICESat2_GDF.geometry.x))

        # Land flag initialized as -1
        # If shorelines downloaded already, will be set to 0 or 1
        ICESat2_GDF.insert(0, 'is_land',
                           zero_int_array - 1, False)

        # set the projection
        ICESat2_GDF.set_crs("EPSG:4326", inplace=True)

        # load shoreline dataset to include only the features that intersect the bounding box
        # bbox can be GeoDataFrame or GeoSeries | shapely Geometry, default None
        # Filter features by given bounding box, GeoSeries, GeoDataFrame or a shapely geometry.
        # engine str, 'fiona' or 'pyogrio'
        land_polygon_gdf = gpd.read_file(shoreline_data_path, bbox=ICESat2_GDF, engine='pyogrio')

        # continue with getting a new array of 0-or-1 labels for each photon
        land_point_labels = np.zeros_like(ICESat2_GDF.is_land.values)

        # update labels for points in the land polygons
        pts_in_land = gpd.sjoin(ICESat2_GDF, land_polygon_gdf, predicate='within')

        # get land or not bool value
        land_loc = ICESat2_GDF.index.isin(pts_in_land.index)

        # asigned them to new numpy array
        land_point_labels[land_loc] = 1
        land_point_labels[~land_loc] = 0

        return land_point_labels

    except Exception as e:

        print(e)

        print("Error loading shoreline data, returning -1s for is_land flag")

        # if the shoreline data is not available
        # return the original label array

        return -np.ones_like(ICESat2_GDF.is_land.values)
      
  
