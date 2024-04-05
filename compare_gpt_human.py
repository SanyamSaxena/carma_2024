import json
import numpy as np
import argparse
import glob
import utils
from tqdm import tqdm
import time
from scipy.stats import spearmanr, kendalltau


def ranking_correlation(ground_truth_rank, pred_rank):
  ground_truth_ranks_list = [ground_truth_rank[item] for item in ground_truth_rank]
  predicted_ranks_list = [pred_rank[item] for item in ground_truth_rank]
  ground_truth_ranks_list = [float(item) for item in ground_truth_ranks_list]
  predicted_ranks_list = [float(item) for item in predicted_ranks_list]
  spearman_corr, _ = spearmanr(ground_truth_ranks_list, predicted_ranks_list)
  kendall_tau, _ = kendalltau(ground_truth_ranks_list, predicted_ranks_list)
  return spearman_corr, kendall_tau


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument("-c", "--config-path", type=str)
  parser.add_argument("-g", "--ground-truth-gpt", type=str)
  parser.add_argument("-t", "--ground-truth-human", type=str)

  args = parser.parse_args()
  config_data = utils.read_config(args.config_path)

  start_time = time.time()

  loose_score = []
  top_1_score = []

  ground_truth = {}

  human_files_pattern = f'{args.ground_truth_human}/*.json'
  human_file_paths = glob.glob(human_files_pattern)
  
  for file_path in tqdm(human_file_paths):
    in_file = open(file_path)
    all_anomalies = json.load(in_file)
    in_file.close()

    for anomaly_detail in all_anomalies:
      anomaly_trend = anomaly_detail['trend']
      indicator = anomaly_detail['indicator']
      place = anomaly_detail['place']
      if(place):
        place_str = anomaly_detail['place']
      else:
        place_str = ""
      trend_time  = anomaly_detail['time']
      ground_truth[f"{indicator}_{place_str}_{anomaly_trend}_{trend_time}"] = anomaly_detail["human ranking"]


  gpt4_files_pattern = f'{args.ground_truth_gpt}/*.json'
  gpt4_file_paths = glob.glob(gpt4_files_pattern)
  count_errors = 0
  for file_path in tqdm(gpt4_file_paths):
    in_file = open(file_path)
    all_anomalies = json.load(in_file)
    in_file.close()

    for anomaly_detail in all_anomalies:
      anomaly_trend = anomaly_detail['trend']
      indicator = anomaly_detail['indicator']
      place = anomaly_detail['place']
      if(place):
        place_str = anomaly_detail['place']
      else:
        place_str = ""
      trend_time  = anomaly_detail['time']

      print(f"Trend:{anomaly_trend}, Indicator: {indicator}, Place: {place_str}, Time: {trend_time}")#

      initial_events_true = []
      for i,event in enumerate(anomaly_detail[f'events_{anomaly_trend}']):
        event_name = event['event name'].strip().lower()
        if(config_data['place']=="yes"):
          event_loc = event['location'].strip().lower()
        else:
          event_loc = ""
        event_time_start = event['start time'].strip().lower()
        event_time_end = event['end time'].strip().lower()
        initial_events_true.append(str([event_name, event_loc, event_time_start, event_time_end]))

      try:
        anomalies_ground_truth_gpt4 = anomaly_detail["gpt4 ranking"]['sorted_event_prompts']
        anomalies_ground_truth_gpt4 = [str(x) for x in anomalies_ground_truth_gpt4]
      except:
        count_errors+=1
        continue

      anomalies_ground_truth_human = ground_truth[f"{indicator}_{place_str}_{anomaly_trend}_{trend_time}"]['sorted_event_prompts']
      anomalies_ground_truth_human = [str(x) for x in anomalies_ground_truth_human]
      anomalies_ground_truth_human_irrelevant = ground_truth[f"{indicator}_{place_str}_{anomaly_trend}_{trend_time}"]['irrelevant_event_prompts']
      anomalies_ground_truth_human_irrelevant = [str(x) for x in anomalies_ground_truth_human_irrelevant]
      anomalies_ground_truth_human_relevant = [x for x in anomalies_ground_truth_human if x not in anomalies_ground_truth_human_irrelevant]

      if len(anomalies_ground_truth_human_relevant)==0:
        loose_score.append(0)
        continue

      if (anomalies_ground_truth_gpt4[0] in anomalies_ground_truth_human_relevant):
          loose_score.append(1)
      else:
          loose_score.append(0)

      if (anomalies_ground_truth_gpt4[0]==anomalies_ground_truth_human_relevant[0]):
          top_1_score.append(1)
      else:
          top_1_score.append(0)

      print("Loose", anomalies_ground_truth_gpt4[0] in anomalies_ground_truth_human_relevant, "...")
      print("Top 1", anomalies_ground_truth_gpt4[0]==anomalies_ground_truth_human_relevant[0], "...")

      print("Initial Event Order")
      for i, item in enumerate(initial_events_true):
          print(i+1, ":", item)

      print("gpt4 Event Order")
      for i, item in enumerate(anomalies_ground_truth_gpt4):
          print(i+1, ":", item)

      print("Human Event Order")
      for i, item in enumerate(anomalies_ground_truth_human):
          print(i+1, ":", item)

  print("\n")
  print('Accuracy Loose', np.mean(loose_score))
  print('Accuracy Top 1', np.mean(top_1_score))
