import pandas as pd
import os
import numpy as np
import zipfile
from zipfile import BadZipFile
import h5py
import matplotlib.pylab as plt
from IPython.utils import strdispatch
from create_landmask import read_granule, isolate_sea_land_photons
import re
import io
import copy
import logging
import scipy.interpolate
import geopandas as gpd
from shapely.geometry import box
import math
import statistics as st 


def get_dataframes(h5_file, shoreline_dir):

    shoreline_data_path = shoreline_dir + 'GeoPkgGlobalShoreline.gpkg'

    # read atl03 HDF5 file and extract variables of interest
    IS2_atl03_mds, IS2_atl03_attrs, IS2_atl03_beams = read_granule(h5_file, ATTRIBUTES=True)

    # extract parameters from ICESat-2 ATLAS HDF5 file name
    rx = re.compile(r'(processed_)?(ATL\d{2})_(\d{4})(\d{2})(\d{2})(\d{2})'
                    r'(\d{2})(\d{2})_(\d{4})(\d{2})(\d{2})_(\d{3})_(\d{2})(.*?).h5$')
    SUB, PRD, YY, MM, DD, HH, MN, SS, TRK, CYCL, GRAN, RL, VERS, AUX = rx.findall(h5_file).pop()
    
class seaLand:

  def __init__(self):
    pass  

  def label_seaLand_function(self, shoreline_data_path, ATL03_h5_file,
                              IS2_atl03_mds,
                              IS2_atl03_attrs,
                              IS2_atl03_beams,
                              date_year,
                              date_month,
                              date_day, *vars):
          
      # initialize data storage variable
      # of interest for generating three types classification

      [setattr(self, var, None) for var in vars]

      Segment_ID = {}
      Segment_Index_begin = {}
      Segment_PE_count = {}

      segment_Distance = {}
      segment_Length = {}
      segment_is_land = {}

      # mean geolocation, height and delta time
      # Segment_Elev = {}
      # Segment_Time = {}

      # Segment_ref_elev = {}
      # Segment_ref_azimuth = {}

      # organized beam-by-beam of derived variables
      IS2_atl03_added_derived_parameters = {}
      IS2_at103_metadata = {}

      # for each input beam within the file
      # for gtx in sorted(IS2_atl03_beams[1:2]):
      for gtx in sorted(IS2_atl03_beams):
          # data and attributes for beam gtx
          IS2_val = IS2_atl03_mds[gtx]
          IS2_attrs = IS2_atl03_attrs[gtx]

          # ATL03 Segment ID
          Segment_ID[gtx] = IS2_val['geolocation']['segment_id']
          # number of valid overlapping ATL03 segments
          n_seg = len(Segment_ID[gtx])
          # number of photon events
          n_pe, = IS2_val['heights']['delta_time'].shape

          # first photon ID (1-based) in each segment (convert to 0-based indexing)
          Segment_Index_begin[gtx] = IS2_val['geolocation']['ph_index_beg'] - 1

          # number of photon events in the segment
          Segment_PE_count[gtx] = IS2_val['geolocation']['segment_ph_cnt']

          # along-track distance for each ATL03 segment
          segment_Distance[gtx] = IS2_val['geolocation']['segment_dist_x']

          # along-track length for each ATL03 segment
          segment_Length[gtx] = IS2_val['geolocation']['segment_length']

          # Transmit time of the reference photon
          delta_time = IS2_val['geolocation']['delta_time']

          # get geolocation lat/lon
          segment_lat = IS2_val['geolocation']['reference_photon_lat'][:].copy()
          segment_lon = IS2_val['geolocation']['reference_photon_lon'][:].copy()
          solar_elevation = IS2_val['geolocation']['solar_elevation'][:].copy()
          setattr(self, 'solar_elevation', solar_elevation)
          full_sat_fract = IS2_val['geolocation']['full_sat_fract'][:].copy()
          setattr(self, 'full_sat_fract', full_sat_fract)
          near_sat_fract = IS2_val['geolocation']['near_sat_fract'][:].copy()
          setattr(self, 'near_sat_fract', near_sat_fract)

          # get parameters ref elev and azimuth for refraction correction
          ref_elev = IS2_val['geolocation']['ref_elev'][:].copy()
          ref_azimuth = IS2_val['geolocation']['ref_azimuth'][:].copy()

          # photon event heights
          h_ph = IS2_val['heights']['h_ph'][:].copy()
          lat_ph = IS2_val['heights']['lat_ph'][:].copy()
          lon_ph = IS2_val['heights']['lon_ph'][:].copy()
          #     dist_ph_along = IS2_val['heights']['dist_ph_along'][:].copy()
          signal_conf_ph = IS2_val['heights']['signal_conf_ph'][..., 0].copy()

          # along-track and across-track distance for photon events
          x_atc = IS2_val['heights']['dist_ph_along'][:].copy()
          y_atc = IS2_val['heights']['dist_ph_across'][:].copy()

          # this function is a significant slowdown without pygeos
          segment_is_land['geometry'] = gpd.points_from_xy(segment_lon, segment_lat)

          # create a geo dataframe
          ICESat2_GDF = gpd.GeoDataFrame(segment_is_land, crs="EPSG:4326")

          # Step1: isolate water using landmask
          # spatial joint to determine land/sea mask
          segment_is_land_label = isolate_sea_land_photons(shoreline_data_path, ICESat2_GDF)

          # interpolate is_land labels based on photon lat
          #     SPL = scipy.interpolate.UnivariateSpline(Segment_lat[:],
          #     Segment_Is_Land_Labels[:],k=3,s=0)
          #     Is_Land_Labels_SPL = SPL(lat_ph[:])
          island_interp1d_model = scipy.interpolate.interp1d(segment_lat[:],
                                                            segment_is_land_label[:],
                                                            fill_value="extrapolate")
          # apply interpid model
          is_land_label_interp1d = island_interp1d_model(lat_ph[:])

          # Aggregate label data into dataframe
          Dataframe_added_landsea_label = \
              pd.DataFrame({'is_land_label': is_land_label_interp1d,},
                          columns=['is_land_label',])

          if len(vars)!=0:
            Dataframe_for_metadata = \
                pd.DataFrame({var : getattr(self, var) for var in vars},
                              columns=[var for var in vars])
            
            IS2_at103_metadata[gtx] = Dataframe_for_metadata
          
          #added the land sea label into dataframe
          IS2_atl03_added_derived_parameters[gtx] = Dataframe_added_landsea_label
           
      return IS2_atl03_added_derived_parameters, IS2_at103_metadata  
    
 class masker:

  def __init__(self, landmask_dataframe, metadata_dataframe, var1, var2):
    self.landmask_dataframe = landmask_dataframe
    self.metadata_dataframe = metadata_dataframe
    self.var1 = var1
    self.var2 = var2
    self.beams = ['gt1l','gt1r','gt2l','gt2r','gt3l','gt3r']

  def resize_landmask_array(self, landmask_array, size):

    if type(landmask_array) == "<class 'numpy.ndarray'>":
      listarray = [([pp.x, pp.y]) for pp in landmask_array]
      arr = np.array(listarray)
    else:
      arr = landmask_array

    array_ratio = int(math.floor(arr.size / size))
    scaled_landmask_array = []

    for idx in range(int(size)):

      Slice = [idx for idx in arr[(idx*array_ratio):((idx+1)*array_ratio)]]
      mean_landmask = st.mean(Slice)
      scaled_landmask_array.append(mean_landmask)

    return np.array(scaled_landmask_array)

  def get_masked_data(self, var1, var2, conf=0.5):

    for beam in self.beams:
      setattr(self, beam, {})
      getattr(self, beam)['is_land_label'] = []
      getattr(self, beam)[var1] = [] 
      getattr(self, beam)[var2] = []

      lm_df = self.landmask_dataframe[beam]['is_land_label']
      lm_arr = np.array(lm_df)

      md_df_var1 = self.metadata_dataframe[beam][var1]
      md_arr_var1 = np.array(md_df_var1)

      md_df_var2 = self.metadata_dataframe[beam][var2]
      md_arr_var2 = np.array(md_df_var2)

      assert md_arr_var1.size == md_arr_var2.size, f"Metadata array sizes (var1:{md_arr_var1.size}, var2:{md_arr_var2.size}) must be the same."

      scaled_lm_arr = self.resize_landmask_array(lm_arr, md_arr_var1.size)

      assert((md_arr_var1.size == scaled_lm_arr.size) and (md_arr_var2.size == scaled_lm_arr.size)), f"Metadata array sizes (var1:{md_arr_var1.size}, var2:{md_arr_var2.size}) and scaled landmask array size (lm:{scaled_lm_arr.size})must be the same."

      for idx in range(int(scaled_lm_arr.size)):
        if scaled_lm_arr[idx] >= conf:
          continue
        else:
          getattr(self, beam)[var1].append(md_arr_var1[idx])
          getattr(self, beam)[var2].append(md_arr_var2[idx])

      getattr(self, beam)[var1] = np.array(getattr(self, beam)[self.var1])
      getattr(self, beam)[var2] = np.array(getattr(self, beam)[self.var2])

    return [((getattr(self, beam)[var1]), getattr(self, beam)[var2]) for beam in self.beams]
