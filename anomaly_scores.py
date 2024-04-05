import pandas as pd
from tqdm import tqdm
from matplotlib import pyplot as plt
import glob
from statsmodels.tsa.seasonal import STL
import argparse
import json
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from scipy.stats import norm
from scipy.stats import linregress
import os
import numpy as np

def linear_interpolation_predict(series, k):
    anomaly_scores = []
    fitted_line = []
    for i in range(series.shape[0]):
      len = k
      start_ind = i-len
      if(start_ind<0):
        start_ind = 0
        len = i
      if(len==0):
        next_point = series[i]
      elif(len==1):
        next_point = series[i-1]
      else:
        x = range(0, len)
        y = series[start_ind:i] 
        slope, intercept, _, _, _ = linregress(x, y)
        next_index = len
        next_point = slope * next_index + intercept
      fitted_line.append(next_point)
      anomaly_scores.append(series[i]-next_point)
    return np.array(fitted_line), np.array(anomaly_scores)

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument("-i", "--input-data", type=str)
  parser.add_argument("-o", "--output-image", type=str)
  parser.add_argument("-j", "--output-json", type=str)
  args = parser.parse_args()
  if not os.path.exists(args.output_image):
      os.makedirs(args.output_image)
  if not os.path.exists(args.output_json):
      os.makedirs(args.output_json)
  files_pattern = f'{args.input_data}\\*.csv'
  file_paths = glob.glob(files_pattern)
  for file_path in tqdm(file_paths):
    file_name = (file_path.split('\\')[-1]).split('.csv')[0]
    time_series_df = pd.read_csv(file_path)
    preprocessed_data = time_series_df['value']
    preprocessed_data = preprocessed_data.interpolate(method='linear')
    preprocessed_data = preprocessed_data.ffill()
    preprocessed_data = preprocessed_data.bfill()

    res = STL(preprocessed_data, period=6, robust=False).fit()
    trend = getattr(res, "trend")
    residue = preprocessed_data - trend
    max_val = max(abs(residue))
    residue = residue/max_val
    time_series_df['anomaly_score'] = residue

    json_dict = time_series_df.set_index('time')['anomaly_score'].to_dict()
    final_dict = {'anomaly_scores': json_dict, 'start': str(time_series_df['time'].iloc[0]), 'end': str(time_series_df['time'].iloc[-1])}
    path_to_output = f'{args.output_json}\{file_name}.json'
    out_file = open(path_to_output,'w')
    json.dump(final_dict, out_file)

    plt.figure(figsize=(16, 8))
    plt.plot(time_series_df['time'], preprocessed_data, label='Original Time Series', color='blue')
    plt.plot(time_series_df['time'], trend, label='Original Time Series', color='red')
    plt.scatter(time_series_df['time'], preprocessed_data, c=residue, cmap='viridis')
    plt.colorbar(label='Anomaly Score')
    plt.xticks(time_series_df['time'][::2], rotation='vertical')
    plt.grid()
    plt.savefig(f'{args.output_image}\{file_name}.png')
    plt.close()
