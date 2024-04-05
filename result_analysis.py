import json
import numpy as np
import argparse
import glob
import utils
from tqdm import tqdm
import time
from scipy.stats import spearmanr, kendalltau
import pickle


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
  parser.add_argument("-t", "--type", type=str) #manual #train #selfcheck
  parser.add_argument("--type-model", default=None, type=str) #selfcheck method name
  parser.add_argument("-m", "--method", default=None, type=str) #selfcheck method name
  parser.add_argument("-s", "--score-type", default=None, type=str)
  parser.add_argument("-i", "--input-data", type=str) #use train only
  parser.add_argument("-g", "--ground-truth", type=str)
  parser.add_argument("-l", "--labeling-method", type=str) # human, gpt4
  parser.add_argument("--hide", action='store_true')

  args = parser.parse_args()
  config_data = utils.read_config(args.config_path)

  start_time = time.time()

  loose_initial_score = []
  top_1_initial_score = []
  loose_initial_score = []
  loose_reranked_score = []
  top_1_reranked_score = []
  loose_reranked_score = []
  rhos_init = []
  rhos_rerank = []
  taus_init = []
  taus_rerank = []
  gpt4_files_pattern = f'{args.ground_truth}/*.json'
  gpt4_file_paths = glob.glob(gpt4_files_pattern)
  ground_truth = {}

  for file_path in gpt4_file_paths:
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
      if(args.labeling_method=="gpt4"):
        ground_truth[f"{indicator}_{place_str}_{anomaly_trend}_{trend_time}"] = anomaly_detail["gpt4 ranking"]
      if(args.labeling_method=="human"):
        ground_truth[f"{indicator}_{place_str}_{anomaly_trend}_{trend_time}"] = anomaly_detail["human ranking"]

  count_errors = 0
  if args.type=="trained":
    files_pattern = f'{args.input_data}/*.p'
  elif args.type=="manual":
    files_pattern = f'{args.input_data}/*.json'
  elif args.type=="selfcheck":
    files_pattern = f'{args.input_data}/{args.method}/*.json'

  file_paths = glob.glob(files_pattern)
  for file_path in file_paths:
    if args.type=="trained":
      in_file = open(file_path, 'rb')
      all_anomalies = pickle.load(in_file)
    elif args.type=="manual" or args.type =="selfcheck":
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

      initial_events_true = []
      events_ranked_using_technique = {}
      bayes_preds = {}
      if not args.hide:
        print(f"Trend:{anomaly_trend}, Indicator: {indicator}, Place: {place_str}, Time: {trend_time}")#
      for i,event in enumerate(anomaly_detail[f'events_{anomaly_trend}']):
        event_name = event['event name'].strip().lower()
        if(config_data['place']=="yes"):
          event_loc = event['location'].strip().lower()
        else:
          event_loc = ""
        event_time_start = event['start time'].strip().lower()
        event_time_end = event['end time'].strip().lower()
        initial_events_true.append(str([event_name, event_loc, event_time_start, event_time_end]))
        if (args.type=="trained"):
          if (args.type_model=="naive_bayes") or (args.type_model=="bayes_with_prior"):
            events_ranked_using_technique[str([event_name, event_loc, event_time_start, event_time_end])] = event["training score"][1]
            bayes_preds[str([event_name, event_loc, event_time_start, event_time_end])] = int(event['pred class'])
          else:
            events_ranked_using_technique[str([event_name, event_loc, event_time_start, event_time_end])] = event["training score"]          
          if not args.hide:
            print(event_name, event['training score'])
        elif (args.type=="selfcheck"):
          events_ranked_using_technique[str([event_name, event_loc, event_time_start, event_time_end])] = event["score"][args.method]          
          if not args.hide:
            print(event_name, event["score"][args.method])
        elif args.type=="manual":
          if not args.hide:
            print(event_name, event['score']["contradiction"], event['score']["consensus_stl"]["avg"], event['score']["consensus_stl"]["our_max"], event['score']["time"], event['score']["pattern"])
          if(args.score_type=="contradiction_s"):
            events_ranked_using_technique[str([event_name, event_loc, event_time_start, event_time_end])] = event['score']["contradiction s"]
          if(args.score_type=="contradiction"):
            events_ranked_using_technique[str([event_name, event_loc, event_time_start, event_time_end])] = event['score']["contradiction"]
          elif(args.score_type=="consensus_our_max"):
            if not args.hide:
              print(event_name, event['cross_examine']['consensus']['times occured'])
            events_ranked_using_technique[str([event_name, event_loc, event_time_start, event_time_end])] = event['score']["consensus_stl"]["our_max"]
          elif(args.score_type=="consensus_avg"):
            if not args.hide:
              print(event_name, event['cross_examine']['consensus']['times occured'])
            events_ranked_using_technique[str([event_name, event_loc, event_time_start, event_time_end])] = event['score']["consensus_stl"]["avg"]
          elif(args.score_type=="time"):
            events_ranked_using_technique[str([event_name, event_loc, event_time_start, event_time_end])] = event['score']["time"]
          elif(args.score_type=="pattern"):
            events_ranked_using_technique[str([event_name, event_loc, event_time_start, event_time_end])] = event['score']["pattern"]
          elif(args.score_type=="pattern_time"):
            events_ranked_using_technique[str([event_name, event_loc, event_time_start, event_time_end])] = (event['score']["pattern"] + event['score']["time"])/2
          elif(args.score_type=="sum_all"):
            events_ranked_using_technique[str([event_name, event_loc, event_time_start, event_time_end])] = event['score']["pattern"] + event['score']["time"] + event['score']["consensus_stl"]["avg"] +event['score']["contradiction s"]

      initial_events_list = initial_events_true
      rank = 1
      initial_events_dict_rank = {}
      for i, value in enumerate(initial_events_list):
          initial_events_dict_rank[value] = rank
          rank+=1

      if (args.type=="trained"):
        if args.type_model=="naive_bayes" or args.type_model=="bayes_with_prior":
          filtered_events_dict = {key: value for key, value in events_ranked_using_technique.items() if bayes_preds[key]==1}
        else:
          filtered_events_dict = {key: value for key, value in events_ranked_using_technique.items() if value > 0.5}
        sorted_filtered_events_dict = dict(sorted(filtered_events_dict.items(), key=lambda item: item[1], reverse=True))
        sorted_filtered_events_list = list(sorted_filtered_events_dict.keys())
      
      sorted_events_dict = dict(sorted(events_ranked_using_technique.items(), key=lambda item: item[1], reverse=True))
      sorted_events_list = list(sorted_events_dict.keys())
      rank = 1
      sorted_events_dict_rank = {}
      for value in sorted_events_dict:
          sorted_events_dict_rank[value] = rank
          rank+=1

      if(args.labeling_method=="gpt4"):
        try:
          anomalies_ground_truth = ground_truth[f"{indicator}_{place_str}_{anomaly_trend}_{trend_time}"]['sorted_event_prompts']
          anomalies_ground_truth = [str(x) for x in anomalies_ground_truth]
        except:
          count_errors+=1
          continue
      elif args.labeling_method=="human":
        anomalies_ground_truth = ground_truth[f"{indicator}_{place_str}_{anomaly_trend}_{trend_time}"]['sorted_event_prompts']
        anomalies_ground_truth = [str(x) for x in anomalies_ground_truth]
        anomalies_ground_truth_irrelevant = ground_truth[f"{indicator}_{place_str}_{anomaly_trend}_{trend_time}"]['irrelevant_event_prompts']
        anomalies_ground_truth_irrelevant = [str(x) for x in anomalies_ground_truth_irrelevant]
        anomalies_ground_truth_relevant = [x for x in anomalies_ground_truth if x not in anomalies_ground_truth_irrelevant]

      rank = 1
      anomalies_ground_truth_rank = {}
      for value in anomalies_ground_truth:
        anomalies_ground_truth_rank[value] = rank
        rank+=1

      rho_init, tau_init =  ranking_correlation(anomalies_ground_truth_rank, initial_events_dict_rank)
      rhos_init.append(rho_init)
      taus_init.append(tau_init)
      
      rho_rerank, tau_rerank = ranking_correlation(anomalies_ground_truth_rank, sorted_events_dict_rank)
      rhos_rerank.append(rho_rerank)
      taus_rerank.append(tau_rerank)

      if (args.type=="trained"):
        if len(anomalies_ground_truth_relevant)==0 and len(sorted_filtered_events_list)==0:
            loose_initial_score.append(0)
            loose_reranked_score.append(1)
            if not args.hide:
              print("Loose", 0, 1, "...")
            continue
        elif len(anomalies_ground_truth_relevant)==0:
            loose_initial_score.append(0)
            loose_reranked_score.append(0)
            if not args.hide:
              print("Loose", 0, 0, "...")
            continue
        elif len(sorted_filtered_events_list)==0:
            if (initial_events_list[0] in anomalies_ground_truth_relevant):
                loose_initial_score.append(1)
            else:
                loose_initial_score.append(0)
            loose_reranked_score.append(0)
            if not args.hide:
              print("Loose", initial_events_list[0] in anomalies_ground_truth_relevant, 0, "...")
        else:
            if (initial_events_list[0] in anomalies_ground_truth_relevant):
                loose_initial_score.append(1)
            else:
                loose_initial_score.append(0)
            if (sorted_events_list[0] in anomalies_ground_truth_relevant):
                loose_reranked_score.append(1)
            else:
                loose_reranked_score.append(0)
            if not args.hide:
              print("Loose", initial_events_list[0] in anomalies_ground_truth_relevant, sorted_events_list[0] in anomalies_ground_truth_relevant, "...")
      else:
          if(len(anomalies_ground_truth_relevant)==0):
            loose_initial_score.append(0)
            loose_reranked_score.append(0)
            continue
          if (initial_events_list[0] in anomalies_ground_truth_relevant):
              loose_initial_score.append(1)
          else:
              loose_initial_score.append(0) 
          if (sorted_events_list[0] in anomalies_ground_truth_relevant):
              loose_reranked_score.append(1)
          else:
              loose_reranked_score.append(0)
          if not args.hide:
            print("Loose", initial_events_list[0] in anomalies_ground_truth_relevant, sorted_events_list[0] in anomalies_ground_truth_relevant, "...")

      if (initial_events_list[0]==anomalies_ground_truth_relevant[0]):
          top_1_initial_score.append(1)
      else:
          top_1_initial_score.append(0)

      if (sorted_events_list[0]==anomalies_ground_truth_relevant[0]):
          top_1_reranked_score.append(1)
      else:
          top_1_reranked_score.append(0)

      if not args.hide:
        print("Top 1", initial_events_list[0]==anomalies_ground_truth[0], sorted_events_list[0]==anomalies_ground_truth[0], "...")

        print("Initial Event Order")
        for i, item in enumerate(initial_events_list):
            print(i+1, ":", item)

        print("Reranked Event Order")
        for i, item in enumerate(sorted_events_dict):
            print(i+1, ":", item, " Score:", sorted_events_dict[item])

        print("Ground Truth Event Order")
        for i, item in enumerate(anomalies_ground_truth):
            print(i+1, ":", item)
 
  if not args.hide:
    print("\n")
    print('Accuracy Initial Loose', np.mean(loose_initial_score))
    print('Accuracy Initial Top 1', np.mean(top_1_initial_score))
    print('Rho Initial', np.mean(rhos_init))
    print('Tau Initial', np.mean(taus_init))
    print('Accuracy Reranked Loose', np.mean(loose_reranked_score))
    print('Accuracy Reranked Top 1', np.mean(top_1_reranked_score))
    print('Rho Reranked', np.mean(rhos_rerank))
    print('Tau Reranked', np.mean(taus_rerank))
    print("Num Errors", count_errors)
  else:
    print(np.mean(loose_initial_score), np.mean(loose_reranked_score), end=" ")
