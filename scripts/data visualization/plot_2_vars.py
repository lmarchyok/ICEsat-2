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

    def __init__(self, raw_data_dir, dest_dir, var1, var2):
      self.raw_data_dir = raw_data_dir
      self.dest_dir = dest_dir
      self.var1 = var1
      self.var2 = var2
      self.beams = ['gt1l', 'gt1r', 'gt2l', 'gt2r', 'gt3l', 'gt3r']
      self.ag_beams = ['gt1', 'gt2', 'gt3']


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


    def plot_all_lasers(self, arrays, selims='auto'):

      setattr(self, 'BIG_var1_arr', np.empty(0,dtype=np.float64))
      setattr(self, 'BIG_var2_arr', np.empty(0,dtype=np.float64))

      for k,v in arrays:
        print(k,v)
        self.BIG_var1_arr = np.concatenate([self.BIG_var1_arr, k])
        self.BIG_var2_arr = np.concatenate([self.BIG_var2_arr, v])

      if selims=='auto':
            lower = np.nanmin(self.BIG_var2_arr)
            upper = np.nanmax(self.BIG_var2_arr)
            serange = upper - lower
            selims = [lower - 0.2*serange, upper + 0.1*serange]

      lat_lower = np.nanmin(self.BIG_var1_arr)
      lat_upper = np.nanmax(self.BIG_var1_arr)
      latrange = lat_upper - lat_lower
      latlims = [lat_lower - 0.2 * latrange, lat_upper + 0.1 * latrange]

      fig = plt.figure(figsize=(9, 6), dpi=80, facecolor='w', edgecolor='k')
      ax = fig.add_subplot(111)
      ax.set_title(f'{self.var1} vs {self.var2} over entire track ALL Lasers')
      ax.set_xlabel('Solar Elevation (degrees)')
      ax.set_ylabel('Saturation (percentage)')

      colors = ['r','g','b','c','m','y']

      for beam in self.beams:
        print(beam)
        setattr(self, f'{beam}', ax.scatter(np.array(getattr(self, f'{beam}_var1_arr')), np.array(getattr(self, f'{beam}_var2_arr')), s=10,c=colors[self.beams.index(beam)],edgecolors='none',label=beam))

      ax.legend(handles=[self.gt1l,self.gt1r,self.gt2l,self.gt2r,self.gt3l,self.gt3r], loc='lower left')
      ax.set_xlim(latlims)
      ax.set_ylim(selims)

      if not os.path.exists(f'figs/aggregate/{self.var1}_vs_{self.var2}'):
          os.makedirs(f'figs/aggregate/{self.var1}_vs_{self.var2}')
      fn = f'figs/aggregate/{self.var1}_vs_{self.var2}/six_plots.png'
      plt.savefig(fn, dpi=150)
      plt.close()



