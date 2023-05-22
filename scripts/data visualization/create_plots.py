import os
import plot_2_vars as pl

def six_beams(dest_dir, var1, var2, loc_id, water_type=None):
  for dir in os.listdir(dest_dir):
    visualizer = pl.ICEsat_2_Visualizer('foo', dest_dir+'/'+dir, var1, var2, loc_id, water_type=water_type)
    arrays = visualizer.get_12_arrays_w_landmask()
    visualizer.plot_all_lasers(arrays)
    
def by_intensity(dest_dir, var1, var2, loc_id, intensity, water_type=None):
   visualizer = pl.ICEsat_2_Visualizer('foo', dest_dir+'/'+dir, var1, var2, loc_id, water_type=water_type, agg_by_intensity=True, split_by_intensity=False)
   arrays = visualizer.get_12_arrays_w_landmask()
    
   if intensity == 'high':
     arrays = arrays[0]
    elif intensity == 'low':
     arrays = arrays[1]
    else:
     print("Error: must select intensity value")

   visualizer.plot_all_lasers(arrays, specific_intensity=intensity)
   
def by_water_type(dest_dir, var1, var2, loc_id, intensity, water_type=None):
  for dir in os.listdir(dest_dir):
    visualizer = pl.ICEsat_2_Visualizer('foo', dest_dir+'/'+dir, var1, var2, loc_id, water_type=water_type, agg_by_intensity=False, split_by_intensity=False)
    arrays = visualizer.get_12_arrays_w_landmask()
    
    if intensity == 'high':
      arrays = arrays[0]
    elif intensity == 'low':
      arrays = arrays[1]
    else:
      pass
    
    visualizer.plot_all_lasers(arrays, specific_intensity=intensity)
