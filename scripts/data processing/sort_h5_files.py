import pandas as pd
import os
import numpy as np
import zipfile

def concat_data(dest_dir, *data_dirs):

  for file in data_dirs:
    print(file)
    for zip in os.listdir(file):
      if zip[:2] == "._":
        continue
      else:
        try:
          with zipfile.ZipFile(file+'/'+zip, 'r') as zp:
            zp.extractall(dest_dir)
            folder_name = zp.namelist()[0][0:9]
          os.rename(dest_dir+'/'+folder_name,dest_dir+'/'+zip[:-4])
        except:
          continue
          
def sort_by_type(data_dir, csv_file, ident):
  df = pd.read_csv(csv_file)
  for data in os.listdir(data_dir):
    try:
      if int(data) not in set(df['Data_ID']):
        continue
      df2=df[df['Data_ID']==int(data)]['Type'].values[0]
      print(df2)
      if df2 == 1:
        os.rename(data_dir+'/'+data,f'/content/drive/MyDrive/ICEsat-2/ICEsat-2_data_downloads/{ident}_sorted/A/'+data)
      elif df2 == 2:
        os.rename(data_dir+'/'+data,f'/content/drive/MyDrive/ICEsat-2/ICEsat-2_data_downloads/{ident}_sorted/B/'+data)
      elif df2 == 3:
        os.rename(data_dir+'/'+data,f'/content/drive/MyDrive/ICEsat-2/ICEsat-2_data_downloads/{ident}_sorted/C/'+data)
      else:
        continue
    except:
      continue
