import pandas as pd
import os
import numpy as np
import zipfile
from zipfile import BadZipFile
import h5py
import matplotlib.pylab as plt
from IPython.utils import strdispatch

# Extracts h5 file from zipfile, then arrays from h5 file
class ICEsat_2_Extractor:
  def __init__(self, raw_data_dir, dest_dir, var1, var2, laser_id):
    self.raw_data_dir = raw_data_dir
    self.dest_dir = dest_dir
    self.var1 = var1
    self.var2 = var2
    self.laser_id = laser_id

  def zip_extraction(self, file):

    import zipfile
    with zipfile.ZipFile(file, 'r') as zp:
      zp.extractall(self.dest_dir)

  def h5_extraction(self, file):

    with h5py.File(file, 'r') as f1:
        laser = f1.get(self.laser_id)
        if laser.get('geolocation') == None:
          return None

        #xyz_data = laser.get('heights')
        #h_ph = xyz_data.get('h_ph')
        #h_array = np.array(h_ph)

        laser_data = laser.get('geolocation')
        var1 = laser_data.get(self.var1)
        var2 = laser_data.get(self.var2)

        var1_arr = np.array(var1)
        var2_arr = np.array(var2)

        if self.var1 == 'solar_elevation':

          var1_arr_cur = np.delete(var1_arr, (var1_arr > 180))
          #var1_arr_cur = np.delete(var1_arr, (var1_arr < -180))

        if self.var2 == 'near_sat_fract' or self.var2 == 'full_sat_fract':
          for i in np.where(var1_arr > 180):

            list_of_bad_datapoints = []

            if len(i) > 0:
              for j in i:
                list_of_bad_datapoints.append(j)
              var2_arr_cur = np.delete(var2_arr, list_of_bad_datapoints)
            else:
              var2_arr_cur = var2_arr
    
        f1.close()
    
    return np.float64(var1_arr_cur), np.float64(var2_arr_cur)
  
# Takes arrays from h5 files and plots variables
class ICEsat_2_Visualizer:

    def __init__(self, raw_data_dir, dest_dir, var1, var2, loc_id, water_type="ALL", agg_by_intensity=True, split_by_intensity=True, agg_by_gtx=False):
      self.raw_data_dir = raw_data_dir
      self.dest_dir = dest_dir
      self.var1 = var1
      self.var2 = var2
      self.loc_id = loc_id
      self.water_type = water_type
      self.agg_by_intensity = agg_by_intensity 
      self.beams = ['gt1l', 'gt1r', 'gt2l', 'gt2r', 'gt3l', 'gt3r']
      self.ag_beams = ['gt1', 'gt2', 'gt3']
      self.intensities = ['high', 'low']
      self.split_by_intensity = split_by_intensity


    def get_arrays(self):

      for i in os.listdir(self.dest_dir):

        for beam in self.beams:

            visualizer = ICEsat_2_Extractor(self.raw_data_dir, self.dest_dir, self.var1, self.var2, beam)
            h5 = visualizer.h5_extraction((self.dest_dir + i + '/' + os.listdir(self.dest_dir + i)[0]))

            if h5 == None:
              continue
            else:
              
              if not hasattr(self, f'{beam[0:3]}_var1_arr'):
                setattr(self, f'{beam[0:3]}_var1_arr', list())
              else:
                getattr(self, f'{beam[0:3]}_var1_arr').append(np.mean(h5[0]))

              if not hasattr(self, f'{beam[0:3]}_var2_arr'):
                setattr(self, f'{beam[0:3]}_var2_arr', list())
              else:
                getattr(self, f'{beam[0:3]}_var2_arr').append(np.mean(h5[1]))

        for beam in self.beams:
            if getattr(self, f'{beam[0:3]}_var1_arr') % 2 != 0 or getattr(self, f'{beam[0:3]}_var2_arr') % 2 != 0:
              getattr(self, f'{beam[0:3]}_var1_arr').pop()
              getattr(self, f'{beam[0:3]}_var2_arr').pop()

        for i in self.ag_beams:
          for j in getattr(self, f'{i}_var1_arr'):
            getattr(self, f'{i}_var1_arr')[getattr(self, f'{i}_var1_arr').index(j)] = np.mean(np.concatenate([np.array([j]), np.array(getattr(self, f'{i}_var1_arr')[getattr(self, f'{i}_var1_arr').index(j)+1])]))

          for j in getattr(self, f'{i}_var2_arr'):
            getattr(self, f'{i}_var2_arr')[getattr(self, f'{i}_var2_arr').index(j)] = np.mean(np.concatenate([np.array([j]), np.array(getattr(self, f'{i}_var2_arr')[getattr(self, f'{i}_var2_arr').index(j)+1])]))

      return [(getattr(self, f'{beam[0:3]}_var1_arr'), getattr(self, f'{beam[0:3]}_var2_arr')) for beam in self.ag_beams]
      

    def get_12_arrays(self):

      for i in os.listdir(self.dest_dir):

        for beam in self.beams:

            visualizer = ICEsat_2_Extractor(self.raw_data_dir, self.dest_dir, self.var1, self.var2, beam)
            h5 = visualizer.h5_extraction((self.dest_dir + i + '/' + os.listdir(self.dest_dir + i)[0]))

            if h5 == None:
              continue
            else:
              
              if not hasattr(self, f'{beam}_var1_arr'):
                setattr(self, f'{beam}_var1_arr', list())
              else:
                getattr(self, f'{beam}_var1_arr').append(np.mean(h5[0]))

              if not hasattr(self, f'{beam}_var2_arr'):
                setattr(self, f'{beam}_var2_arr', list())
              else:
                getattr(self, f'{beam}_var2_arr').append(np.mean(h5[1]))

      return [(getattr(self, f'{beam}_var1_arr'), getattr(self, f'{beam}_var2_arr')) for beam in self.beams]
    
    def get_12_arrays_w_landmask(self):

      self.high_beams_var1 = []
      self.high_beams_var2 = []
      self.low_beams_var1 = []
      self.low_beams_var2 = []

      progress = 1

      for file in os.listdir(self.dest_dir):

        print(f"Progress: {progress}/{len(os.listdir(self.dest_dir))}") #, self.dest_dir+'/'+file+'/'+os.listdir(self.dest_dir+'/'+file)[0])

        data = get_dataframes((self.dest_dir+'/'+file+'/'+os.listdir(self.dest_dir+'/'+file)[0]), '/content/drive/MyDrive/ICEsat-2/Scripts/')
        IS2_atl03_landmask = seaLand()
        land_mask1 = IS2_atl03_landmask.label_seaLand_function(data[0], data[1], data[2], data[3], data[4], data[5], data[6])[0]
        metadata_mask2 = IS2_atl03_landmask.label_seaLand_function(data[0], data[1], data[2], data[3], data[4], data[5], data[6], self.var1, self.var2)[1]

        mask = masker(land_mask1, metadata_mask2, self.var1, self.var2)
        masked_arrays = mask.get_masked_data(self.var1, self.var2)

        intensities = []          

        for k,v,s,b in masked_arrays:

            intensities.append(s)

            if len(k) == 0:
              continue

            if self.var1 == 'solar_elevation':

              k = np.delete(k, (k > 180))

            if self.var2 == 'near_sat_fract' or self.var2 == 'full_sat_fract':
              for i in np.where(k > 180):

                list_of_bad_datapoints = []

                if len(i) > 0:
                  for j in i:
                    list_of_bad_datapoints.append(j)
                  v = np.delete(v, list_of_bad_datapoints)
                else:
                  v_ = v

                try:
                  setattr(self, f'{b}_var1_arr', np.mean(k))
                  setattr(self, f'{b}_var2_arr', np.mean(v))

                  if not hasattr(self, f'{b}_var1_arr_sb'):
                    setattr(self, f'{b}_var1_arr_sb', [])
                  else:
                    getattr(self, f'{b}_var1_arr_sb').append(np.mean(k))
                  if not hasattr(self, f'{b}_var2_arr_sb'):
                    setattr(self, f'{b}_var2_arr_sb', [])
                  else:
                    getattr(self, f'{b}_var2_arr_sb').append(np.mean(v))
                  

                except:
                    continue
        
        setattr(self, file+'_high_beams', sorted(zip(intensities, self.beams), reverse=True)[:3][0][1][-1])
        setattr(self, file+'_low_beams', sorted(zip(intensities, self.beams), reverse=False)[:3][0][1][-1])

        for b in self.beams:
          if b[-1] == getattr(self, file+'_high_beams'):
            self.high_beams_var1.append(getattr(self, f'{b}_var1_arr')) 
            self.high_beams_var2.append(getattr(self, f'{b}_var2_arr'))
            setattr(self, f'{b}_var1_arr', [])
            setattr(self, f'{b}_var2_arr', [])

          else:
            self.low_beams_var1.append(getattr(self, f'{b}_var1_arr'))
            self.low_beams_var2.append(getattr(self, f'{b}_var2_arr'))
            setattr(self, f'{b}_var1_arr', [])
            setattr(self, f'{b}_var2_arr', [])

        progress += 1

      if self.agg_by_intensity:
        return ((self.high_beams_var1, self.high_beams_var2) , (self.low_beams_var1, self.low_beams_var2))
      else:
        return [(getattr(self, f'{b}_var1_arr'), getattr(self, f'{b}_var2_arr')) for b in self.beams]
    
    
    def plot_2_vars(laser_id, var1, var2, var1_arr, var2_arr, selims='auto'):

      if selims=='auto':
            lower = np.nanmin(np.array(var2_arr))
            upper = np.nanmax(np.array(var2_arr))
            serange = upper - lower
            selims = [lower - 0.2*serange, upper + 0.1*serange]

      lat_lower = np.nanmin(np.array(var1_arr))
      lat_upper = np.nanmax(np.array(var1_arr))
      latrange = lat_upper - lat_lower
      latlims = [lat_lower - 0.2 * latrange, lat_upper + 0.1 * latrange]

      fig = plt.figure(figsize=(9, 6), dpi=80, facecolor='w', edgecolor='k')
      ax = fig.add_subplot(111)
      solar_elevation = np.array(var1_arr)
      saturation = np.array(var2_arr)
      ax.set_title(f'{var1} vs {var2} over 700m ({laser_id})')
      ax.set_xlabel('Solar Elevation (degrees)')
      ax.set_ylabel('Saturation (percentage)')

      atl03_plot = ax.scatter(solar_elevation, saturation, s=10,c='k',edgecolors='none',label=laser_id)
      ax.legend(handles=[atl03_plot], loc='lower left')
      ax.set_xlim(latlims)
      ax.set_ylim(selims)

      if not os.path.exists(f'figs/aggregate/{var1}_vs_{var2}'):
          os.makedirs(f'figs/aggregate/{var1}_vs_{var2}')
      fn = f'figs/aggregate/{var1}_vs_{var2}/{laser_id}.png'
      plt.savefig(fn, dpi=150)
      plt.close()


    def plot_all_lasers(self, arrays:tuple, specific_intensity='high', selims='auto'):

      setattr(self, 'BIG_var1_arr', np.empty(0,dtype=np.float64))
      setattr(self, 'BIG_var2_arr', np.empty(0,dtype=np.float64))

      if self.agg_by_intensity and not self.split_by_intensity:
        for k,v in arrays:
          self.BIG_var1_arr = np.concatenate([self.BIG_var1_arr, np.array([x for x in k if type(x) != type([])])])
          self.BIG_var2_arr = np.concatenate([self.BIG_var2_arr, np.array([x for x in v if type(x) != type([])])])

      elif self.split_by_intensity:
        print(arrays)
        self.BIG_var1_arr = np.concatenate([self.BIG_var1_arr, np.array([x for x in arrays[0] if type(x) != type([])])])
        self.BIG_var2_arr = np.concatenate([self.BIG_var2_arr, np.array([x for x in arrays[1] if type(x) != type([])])])
        print(self.BIG_var1_arr)
        print(self.BIG_var2_arr)

      else:
        for k,v in arrays:
            self.BIG_var1_arr = np.concatenate([self.BIG_var1_arr, np.array([x for x in k if type(x) != type([])])])
            self.BIG_var2_arr = np.concatenate([self.BIG_var2_arr, np.array([x for x in v if type(x) != type([])])])

      if selims=='auto':
            lower = np.nanmin(self.BIG_var2_arr)
            upper = np.nanmax(self.BIG_var2_arr)
            serange = upper - lower
            selims = [lower - 0.2*serange, upper + 0.1*serange]
            print(selims)

      lat_lower = np.nanmin(self.BIG_var1_arr)
      lat_upper = np.nanmax(self.BIG_var1_arr)
      latrange = lat_upper - lat_lower
      latlims = [lat_lower - 0.2 * latrange, lat_upper + 0.1 * latrange]
      print(latlims)

      fig = plt.figure(figsize=(9, 6), dpi=80, facecolor='w', edgecolor='k')
      ax = fig.add_subplot(111)
      ax.set_title(f'Mean {self.var2} vs {self.var1} - {specific_intensity} intensity - NC data')
      ax.set_xlabel('Solar Elevation (degrees)')
      ax.set_ylabel('Saturation (percentage)')

      colors = ['r','g','b','c','m','y']

      if self.agg_by_intensity and not self.split_by_intensity:
        for intensity in self.intensities:

          setattr(self, f'{intensity}_beams_var1', np.array([x for x in getattr(self, f'{intensity}_beams_var1') if type(x) != type([])]))
          setattr(self, f'{intensity}_beams_var2', np.array([x for x in getattr(self, f'{intensity}_beams_var2') if type(x) != type([])]))
          setattr(self, f'{intensity}', ax.scatter(np.array(getattr(self, f'{intensity}_beams_var1')), np.array(getattr(self, f'{intensity}_beams_var2')), s=10,c=colors[self.intensities.index(intensity)],edgecolors='none',label=intensity))

        ax.legend(handles=[self.high,self.low], loc='lower left')
        ax.set_xlim(latlims)
        ax.set_ylim(selims)
      
      elif self.split_by_intensity: 

        setattr(self, f'{specific_intensity}_beams_var1', np.array([x for x in getattr(self, f'{specific_intensity}_beams_var1') if type(x) != type([])]))
        setattr(self, f'{specific_intensity}_beams_var2', np.array([x for x in getattr(self, f'{specific_intensity}_beams_var2') if type(x) != type([])]))
        setattr(self, f'{specific_intensity}', ax.scatter(np.array(getattr(self, f'{specific_intensity}_beams_var1')), np.array(getattr(self, f'{specific_intensity}_beams_var2')), s=10,c=colors[self.intensities.index(specific_intensity)],edgecolors='none',label=specific_intensity))

        ax.legend(handles=[getattr(self, f'{specific_intensity}')], loc='lower left')
        ax.set_xlim(latlims)
        ax.set_ylim(selims)

      else:
        for beam in self.beams:
          setattr(self, f'{beam}', ax.scatter(np.array(getattr(self, f'{beam}_var1_arr')), np.array(getattr(self, f'{beam}_var2_arr')), s=10,c=colors[self.beams.index(beam)],edgecolors='none',label=beam))

        ax.legend(handles=[self.gt1l,self.gt1r,self.gt2l,self.gt2r,self.gt3l,self.gt3r], loc='lower left')
        ax.set_xlim(latlims)
        ax.set_ylim(selims)

      print("done")

      if not os.path.exists(f'figs/aggregate/{self.var1}_vs_{self.var2}'):
          os.makedirs(f'figs/aggregate/{self.var1}_vs_{self.var2}')
      fn = f'figs/aggregate/{self.var1}_vs_{self.var2}/{self.loc_id}_{self.water_type}.png'
      plt.savefig(fn, dpi=150)
      plt.close()
