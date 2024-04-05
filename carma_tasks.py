import numpy as np
import json
from functools import partial
import multiprocessing
import time
from tqdm import tqdm
import glob
import json
import argparse
import pickle
import utils
import os
import random
import torch
from models import LogisticRegression, NN2Layer, GeneralizedCrossEntropyLoss

dict_finance = {
  "Technology": {
    "Apple Inc.",
    "Microsoft Corporation",
    "Amazon.com Inc.",
    "Alphabet Inc.",
    "NVIDIA Corporation"
  },
  "Healthcare": {
    "Amgen Inc.",
    "Biogen Inc.",
    "Gilead Sciences Inc.",
    "Regeneron Pharmaceuticals Inc.",
    "Vertex Pharmaceuticals Incorporated"
  },
  "Finance": {
    "PayPal Holdings Inc.",
    "The Goldman Sachs Group, Inc.",
    "JPMorgan Chase & Co.",
    "American Express Company",
    "Square, Inc."
  },
  "Consumer Goods": {
    "Tesla, Inc.",
    "The Coca-Cola Company",
    "PepsiCo, Inc.",
    "Nike, Inc.",
    "Procter & Gamble Company"
  },
  "Communication Services": {
    "Meta Platforms, Inc.",
    "Netflix Inc.",
    "T-Mobile US, Inc.",
    "Comcast Corporation",
    "Charter Communications, Inc."
  },
  "Energy": {
     "Marathon Petroleum Corporation",
     "Clean Energy Fuels Corp.",
     "Plug Power Inc.",
     "Renewable Energy Group, Inc.",
     "SunPower Corporation"
  },
  "Industrials": {
     "Boeing Company",
     "Lockheed Martin Corporation",
     "FedEx Corporation",
     "United Parcel Service, Inc.",
     "Caterpillar Inc."
  }
}

def get_events_from_anomaly(anomaly_details, num_events, client_gpt_35, config_data):
  indicator = anomaly_details['indicator']
  place = anomaly_details['place']
  trend_time  = anomaly_details['time']
  for trend in config_data["trends"]:
    events = utils.get_events(trend, indicator, place, trend_time, client_gpt_35, config_data, num_events=num_events, location=True, chain_of_thought=False)
    if(not events):
      return None
    anomaly_details[f'events_{trend}'] = events
  return anomaly_details


def return_events(anomaly_details):
  anomaly_events = {}
  anomaly_events['increase'] = anomaly_details[f'events_increase']
  anomaly_events['decrease'] = anomaly_details[f'events_decrease']
  return anomaly_events


def get_random_negative_events(anomaly_detail, file_name, all_events, separated_trend_events, separated_filename_events, config_data):
  excluded_events = []
  anomaly_trend = anomaly_detail["trend"]
  if(config_data['dataset']=="worldbank"):
    place = file_name.strip().split("_")[0]
    indicator = file_name.strip().split("_")[1]
    for name_of_file in separated_filename_events:
      if place in name_of_file or indicator in name_of_file:
        excluded_events.extend(separated_filename_events[name_of_file])
  elif(config_data['dataset']=="financial"):
    excluded_file_names = None
    for type_ind in dict_finance:
      if(file_name in dict_finance[type_ind]):
        excluded_file_names = dict_finance[type_ind]
        break
    for name_of_file in separated_filename_events:
      if name_of_file in excluded_file_names:
        excluded_events.extend(separated_filename_events[name_of_file])
      excluded_events.append(separated_trend_events[anomaly_trend])
  
  random_negative_events = []
  count_random_event = 0
  while count_random_event<5:
    sample_event = random.sample(all_events, 1)
    if sample_event[0] not in excluded_events:
      random_negative_events.append(sample_event[0])
      count_random_event+=1

  return random_negative_events


def cross_examine_anomaly(anomaly_details, file_name, anomaly_score_path, client_gpt_35, config_data, split):
  anomaly_trend = anomaly_details['trend']
  indicator = anomaly_details['indicator']
  place = anomaly_details['place']
  trend_time  = anomaly_details['time']
  with open(f"{anomaly_score_path}/{file_name}.json", 'r') as json_file:
    anomaly_scores_dict = json.load(json_file)
  if split=="test":
    for event in anomaly_details[f'events_{anomaly_trend}']:
      event['cross_examine'] = utils.cross_examine_event(event, anomaly_trend, indicator, place, trend_time, anomaly_scores_dict, "no", client_gpt_35, config_data)
  elif split=="train":
    for trend in config_data["trends"]:
      for event in anomaly_details[f'events_{trend}']:
        event['cross_examine'] = utils.cross_examine_event(event, anomaly_trend, indicator, place, trend_time, anomaly_scores_dict, "no", client_gpt_35, config_data)
    random_negative_events = anomaly_details["random negative events"]
    for event in random_negative_events:
      event['cross_examine'] = utils.cross_examine_event(event, anomaly_trend, indicator, place, trend_time, anomaly_scores_dict, "no", client_gpt_35, config_data)
  return anomaly_details


def score_anomaly(anomaly_details, file_name, anomaly_score_path, config_data, split):
  with open(f"{anomaly_score_path}/stl/{file_name}.json", 'r') as json_file:
    anomaly_scores_dict_stl = json.load(json_file)

  with open(f"{anomaly_score_path}/lin/{file_name}.json", 'r') as json_file:
    anomaly_scores_dict_lin = json.load(json_file)

  anomaly_trend = anomaly_details['trend']
  indicator = anomaly_details['indicator']
  place = anomaly_details['place']
  anomaly_time  = anomaly_details['time']

  if split=="test":
    for event in anomaly_details[f'events_{anomaly_trend}']:
      score_rel_contradiction_s = event['cross_examine']['contradiction s']['score']
      score_rel_contradiction = event['cross_examine']['contradiction simple']['score']
      score_rel_consensus_stl = utils.calc_consensus_score(event, anomaly_time, anomaly_trend, indicator, place, anomaly_scores_dict_stl, config_data)
      score_rel_consensus_lin = utils.calc_consensus_score(event, anomaly_time, anomaly_trend, indicator, place, anomaly_scores_dict_lin, config_data)
      score_rel_pattern = utils.calc_pattern_score(event['cross_examine']['pattern'], anomaly_trend)
      score_rel_time = utils.calc_score_time(event, anomaly_time, config_data)

      if(score_rel_contradiction=="NA"):
        score_rel_contradiction = 0
      if(score_rel_pattern=="NA"):
        score_rel_pattern = 0
      if(score_rel_time=="NA"):
        score_rel_time = 0.5

      event['score'] = {}
      event['score']['contradiction s'] = score_rel_contradiction_s
      event['score']['contradiction'] = score_rel_contradiction
      event['score']['pattern'] = score_rel_pattern
      event['score']['time'] = score_rel_time
      event['score']['consensus_stl'] = score_rel_consensus_stl      
      event['score']['consensus_lin'] = score_rel_consensus_lin      

  elif split=="train":
    for trend in config_data["trends"]:
      for event in anomaly_details[f'events_{trend}']:
        score_rel_contradiction_s = event['cross_examine']['contradiction s']['score']
        score_rel_contradiction = event['cross_examine']['contradiction simple']['score']
        score_rel_consensus_stl = utils.calc_consensus_score(event, anomaly_time, anomaly_trend, indicator, place, anomaly_scores_dict_stl, config_data)
        score_rel_consensus_lin = utils.calc_consensus_score(event, anomaly_time, anomaly_trend, indicator, place, anomaly_scores_dict_lin, config_data)
        score_rel_pattern = utils.calc_pattern_score(event['cross_examine']['pattern'], anomaly_trend)
        score_rel_time = utils.calc_score_time(event, anomaly_time, config_data)

        if(score_rel_contradiction=="NA"):
          score_rel_contradiction = 0
        if(score_rel_pattern=="NA"):
          score_rel_pattern = 0
        if(score_rel_time=="NA"):
          score_rel_time = 0.5

        event['score'] = {}
        event['score']['contradiction s'] = score_rel_contradiction_s
        event['score']['contradiction'] = score_rel_contradiction
        event['score']['pattern'] = score_rel_pattern
        event['score']['time'] = score_rel_time
        event['score']['consensus_stl'] = score_rel_consensus_stl      
        event['score']['consensus_lin'] = score_rel_consensus_lin      

    random_negative_events = anomaly_details["random negative events"]

    for event in random_negative_events:
      score_rel_contradiction_s = event['cross_examine']['contradiction s']['score']
      score_rel_contradiction = event['cross_examine']['contradiction simple']['score']
      score_rel_consensus_stl = utils.calc_consensus_score(event, anomaly_time, anomaly_trend, indicator, place, anomaly_scores_dict_stl, config_data)
      score_rel_consensus_lin = utils.calc_consensus_score(event, anomaly_time, anomaly_trend, indicator, place, anomaly_scores_dict_lin, config_data)
      score_rel_pattern = utils.calc_pattern_score(event['cross_examine']['pattern'], anomaly_trend)
      score_rel_time = utils.calc_score_time(event, anomaly_time, config_data)

      if(score_rel_contradiction=="NA"):
        score_rel_contradiction = 0
      if(score_rel_pattern=="NA"):
        score_rel_pattern = 0
      if(score_rel_time=="NA"):
        score_rel_time = 0.5

      event['score'] = {}
      event['score']['contradiction s'] = score_rel_contradiction_s
      event['score']['contradiction'] = score_rel_contradiction
      event['score']['pattern'] = score_rel_pattern
      event['score']['time'] = score_rel_time
      event['score']['consensus_stl'] = score_rel_consensus_stl      
      event['score']['consensus_lin'] = score_rel_consensus_lin      

  return anomaly_details


def get_features_anomaly(anomaly_details, feature_level, config_data, split):
  anomaly_trend = anomaly_details['trend']

  for i, event in enumerate(anomaly_details[f'events_{anomaly_trend}']):
    event = utils.get_features_from_event(event, anomaly_trend, feature_level)
    if (i==0):
      event['hard status'] = True
    else:
      event['hard status'] = "NA"
    event['loose status'] = True

  if split=="train":
    opposite_trend = config_data["trends"][anomaly_trend]
    for i, event in enumerate(anomaly_details[f'events_{opposite_trend}']):
      event = utils.get_features_from_event(event, anomaly_trend, feature_level)
      if (i==0):
        event['hard status'] = False
      else:
        event['hard status'] = "NA"
      event['loose status'] = False

    # if(split!=test):
    random_negative_events = anomaly_details["random negative events"]
    for event in random_negative_events:
      event = utils.get_features_from_event(event, anomaly_trend, feature_level)
  return anomaly_details


def gpt4_supervise_anomaly(anomaly_details, explain, client_gpt_4, config_data):
  anomaly_trend = anomaly_details['trend']
  indicator = anomaly_details['indicator']
  place = anomaly_details['place']
  trend_time  = anomaly_details['time']

  events = []
  for event in anomaly_details[f'events_{anomaly_trend}']:
    events.append(event)

  anomaly_details['gpt4 ranking'] = utils.gpt4_rank_event(anomaly_details, events, anomaly_trend, indicator, place, trend_time, client_gpt_4, explain, config_data)
  return anomaly_details


def get_combined_score(anomaly_details, type_model, model_path, feature_level, split):

  if feature_level=="drop_none":
    input_size = 7
  elif feature_level=="drop_pattern":
    input_size = 4
  elif feature_level=="drop_time":
    input_size = 6
  elif feature_level=="drop_consensus":
    input_size = 6
  elif feature_level=="drop_contradiction":
    input_size = 5

  if(type_model=="naive_bayes") or (type_model=="bayes_with_prior"):
    with open(f"{model_path}/best_model.p", 'rb') as file:
      model = pickle.load(file)
  elif(type_model=="logistic"):
    model = LogisticRegression(input_size)
  elif(type_model=="nn"):
    model = NN2Layer(input_size)

  if(type_model!="naive_bayes") and (type_model!="bayes_with_prior"):
    model.load_state_dict(torch.load(f"{model_path}/best_model.pth"))
    model.eval()

  anomaly_trend = anomaly_details['trend']

  if split=="test":
      for event in anomaly_details[f'events_{anomaly_trend}']:
        feature = event['feature']
        feature = torch.from_numpy(event['feature']).to(torch.float)
        if(type_model!="naive_bayes") and (type_model!="bayes_with_prior"):
          event_score = model(feature).item()
          event['training score'] = event_score
        elif (type_model=="bayes_with_prior") or (type_model=="naive_bayes"):
          feat = feature.reshape(-1, 1).T
          pred_class, pred_probs = model.predict(feat)
          event['pred class'] = pred_class[0]
          event['training score'] = pred_probs[0]

  return anomaly_details

if __name__ == '__main__':
  print("Process ID:", os.getpid())  
  parser = argparse.ArgumentParser()
  parser.add_argument("-c", "--config-path", type=str)
  parser.add_argument("-f", "--feature-level", default="drop_none", type=str)  
  parser.add_argument("-s", "--split", default="both", type=str)
  parser.add_argument("-t", "--task", type=str)
  parser.add_argument("-i", "--input-data", type=str)
  parser.add_argument("-o", "--output-data", type=str)
  parser.add_argument("-a", "--anomaly-score-path", default=None, type=str)
  parser.add_argument("-p", "--model-path", default="", type=str) # only if trained
  parser.add_argument("-m", "--type-model", default="", type=str) # only if trained

  random.seed(10)
  np.random.seed(0)

  args = parser.parse_args()
  config_data = utils.read_config(args.config_path)
  client_gpt_35 = utils.setup_openai(config_data, '3.5')
  client_gpt_4 = utils.setup_openai(config_data, '4')
  start_time = time.time()
  if args.split=="test":
    splits = ['test']
  elif args.split=="train":
    splits = ['train']
  elif args.split=="both":
    splits = ['train', 'test']


  for split in splits:
    if args.task == "get_combined_score":
      files_pattern = f'{args.input_data}/{split}/*.p'
    else:
      files_pattern = f'{args.input_data}/{split}/*.json'
    file_paths = glob.glob(files_pattern)
    file_paths.sort()
    if args.task == "create_random_neg_events":
      all_events = []
      separated_trend_events = {"increase":[], "decrease":[]}
      separated_filename_events = {}

      for file_path in tqdm(file_paths):
        file_events = []
        file_name = (file_path.split('/')[-1]).split('.json')[0]
        in_file = open(file_path)
        all_anomalies = json.load(in_file)
        in_file.close()
        for anomaly_detail in all_anomalies:
          events = return_events(anomaly_detail)
          separated_trend_events["increase"].extend(events["increase"])
          separated_trend_events["decrease"].extend(events["decrease"])
          file_events.extend(events["increase"])
          file_events.extend(events["decrease"])
        separated_filename_events[file_name] = file_events
        all_events.extend(file_events)

      for file_path in tqdm(file_paths):
        file_events = []
        file_name = (file_path.split('/')[-1]).split('.json')[0]
        in_file = open(file_path)
        all_anomalies = json.load(in_file)
        in_file.close()
        for anomaly_detail in all_anomalies:
          random_negative_events = get_random_negative_events(anomaly_detail, file_name, all_events, separated_trend_events, separated_filename_events, config_data)
          anomaly_detail['random negative events'] = random_negative_events

        path_to_output_folder = f'{args.output_data}/{split}'    
        if not os.path.exists(path_to_output_folder):
          os.makedirs(path_to_output_folder)
        path_to_output = f'{path_to_output_folder}/{file_name}.json'
        out_file = open(path_to_output,'w')
        json.dump(all_anomalies, out_file)
        out_file.close()
    else:
      
      for file_path in file_paths:
        file_events = []
        file_name = (file_path.split('/')[-1]).split('.json')[0]
        in_file = open(file_path)
        all_anomalies = json.load(in_file)
        in_file.close()
        updated_anomalies = []
        for anomaly_detail in all_anomalies:
          if args.task == "extract_events":
            updated_anomaly = get_events_from_anomaly(anomaly_detail, num_events=args.num_events, client_gpt_35=client_gpt_35, config_data=config_data)
          elif args.task == "cross_examine_random_events":
            updated_anomaly = cross_examine_anomaly(anomaly_detail, file_name, args.anomaly_score_path, client_gpt_35=client_gpt_35, config_data=config_data, split=split)
          elif args.task == "score_random_events":
            updated_anomaly = score_anomaly(anomaly_detail, file_name, args.anomaly_score_path, config_data=config_data, split=split)
          elif args.task == "get_features_random_events":
            updated_anomaly = get_features_anomaly(anomaly_detail, feature_level=args.feature_level, config_data=config_data, split=split)
          elif args.task == "get_combined_score":
            updated_anomaly = get_combined_score(anomaly_detail, type_model=args.type_model, model_path=args.model_path, feature_level=args.feature_level, split=split)          
          elif args.task == "gpt4_supervise_events":
            updated_anomaly = gpt4_supervise_anomaly(anomaly_detail, explain=args.explain, client_gpt_4=client_gpt_4, config_data=config_data)
          updated_anomalies.append(updated_anomaly)
        path_to_output_folder = f'{args.output_data}/{split}'    
        if not os.path.exists(path_to_output_folder):
          os.makedirs(path_to_output_folder)
        if args.task == "get_features_random_events" or args.task == 'get_combined_score':
          path_to_output = f'{path_to_output_folder}/{file_name}.p'
          out_file = open(path_to_output,'wb')
          pickle.dump(updated_anomalies, out_file)
          out_file.close()
        else:
          path_to_output = f'{path_to_output_folder}/{file_name}.json'
          out_file = open(path_to_output,'w')
          json.dump(updated_anomalies, out_file)
          out_file.close()
        # break
    end_time = time.time()
    print('total time:',end_time-start_time)
