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
import utils as utils
import os
import torch
from selfcheckgpt.modeling_selfcheck import SelfCheckMQAG, SelfCheckBERTScore, SelfCheckNgram, SelfCheckNLI
import spacy
import torch.multiprocessing as mp


def self_check_gpt_save_samples(anomaly_details, num_samples, client_gpt_35, config_data):
    anomaly_trend = anomaly_details['trend']
    indicator = anomaly_details['indicator']
    place = anomaly_details['place']
    trend_time  = anomaly_details['time']

    sampled_events = []
    sampled_passages = []
    count_samples = 0
    while count_samples<num_samples:
        event = utils.get_event_self_check(anomaly_trend, indicator, place, trend_time, client_gpt_35, config_data, location=True, chain_of_thought=False, temp=1.0)
        if(not event):
            continue
        try:
            _, sampled_passage = utils.convert_event_to_passage(event, anomaly_trend, indicator, place, trend_time)
            sampled_events.append(event)
            sampled_passages.append(sampled_passage)
            count_samples+=1
        except:
            continue
    anomaly_details["sampled events"] = sampled_events
    anomaly_details["sampled passages"] = sampled_passages
    return anomaly_details


def self_check_gpt_score(anomaly_details, self_model, num_samples, method, client_gpt_35, config_data):
    anomaly_trend = anomaly_details['trend']
    indicator = anomaly_details['indicator']
    place = anomaly_details['place']
    trend_time  = anomaly_details['time']

    for i, event in enumerate(anomaly_details[f'events_{anomaly_trend}']):
        sentences, passage = utils.convert_event_to_passage(event, anomaly_trend, indicator, place, trend_time)
        event['sentences'] = sentences
        event['passage'] = passage
        sampled_events = anomaly_details["sampled events"]
        sampled_passages = []
        for event_sampled in sampled_events:
          _, sampled_passage  = utils.convert_event_to_passage(event_sampled, anomaly_trend, indicator, place, trend_time)
          sampled_passages.append(sampled_passage)
        return_val = None
        if method == "prompt":
          return_val = utils.self_check_prompt(sentences, sampled_passages, client_gpt_35, config_data)
          score = np.mean(return_val['score'])
        elif method == "nli":
          return_val = utils.self_check_nli(sentences, sampled_passages, self_model)
          score = -1 * np.mean(return_val['score'])
        elif method == "bertscore":
          return_val = utils.self_check_bertscore(sentences, sampled_passages, self_model)
          score = -1 * np.mean(return_val['score'])
        elif method == "mqag":
          return_val = utils.self_check_mqag(sentences, passage, sampled_passages, self_model)
          score = -1 * np.mean(return_val['score'])
        elif method == "ngram":
          return_val = utils.self_check_ngram(sentences, passage, sampled_passages, self_model)
          score = -1 * np.mean(return_val['score']['sent_level']['max_neg_logprob'])
        else:
          print("Invalid Method")
          return None
        event['score'] = {method: score, "complete answer": return_val}
    return anomaly_details


if __name__ == '__main__':
  pid = os.getpid()
  print("Process ID:", pid)
  parser = argparse.ArgumentParser()
  parser.add_argument("-c", "--config-path", type=str)
  parser.add_argument("-t", "--task", type=str)
  parser.add_argument("-s", "--split", default="test", type=str)
  parser.add_argument("-g", "--gpu", default=0, type=int)
  parser.add_argument("-n", "--num-samples", default=20, type=int)
  parser.add_argument("-i", "--input-data", type=str)
  parser.add_argument("-o", "--output-data", type=str)
  parser.add_argument("-m", "--method", default="prompt", type=str)

  args = parser.parse_args()
  config_data = utils.read_config(args.config_path)
  client_gpt_35 = utils.setup_openai(config_data, '3.5')

  start_time = time.time()

  if args.split=="test":
    splits = ['test']
  elif args.split=="train":
    splits = ['train']
  elif args.split=="both":
    splits = ['train', 'test']
  
  if args.task == "get_scores":
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    if args.method=="prompt":
      self_model = None
    elif args.method == "nli":
      self_model = SelfCheckNLI(device=device)
    elif args.method == "mqag":
      self_model = SelfCheckMQAG(device=device)
    elif args.method == "bertscore":
      nlp = spacy.load("en_core_web_sm")
      self_model = utils.SelfCheckBERTScore(nlp, rescale_with_baseline=True)
    elif args.method == "ngram":             
      self_model = SelfCheckNgram(n=1)


  for split in splits:
    files_pattern = f'{args.input_data}/{split}/*.json'
    file_paths = glob.glob(files_pattern)
    file_paths.sort()
    for i, file_path in tqdm(enumerate(file_paths)):
      file_name = (file_path.split('/')[-1]).split('.json')[0]
      in_file = open(file_path)
      all_anomalies = json.load(in_file)
      in_file.close()
      updated_anomalies = []
      for j, anomaly_detail in enumerate(all_anomalies):
          if args.task == "save_samples":
              updated_anomaly = self_check_gpt_save_samples(anomaly_detail, num_samples=args.num_samples, client_gpt_35=client_gpt_35, config_data=config_data)        
          elif args.task == "get_scores":
              updated_anomaly = self_check_gpt_score(anomaly_detail, self_model, num_samples=args.num_samples, method=args.method, client_gpt_35=client_gpt_35, config_data=config_data)        
          updated_anomalies.append(updated_anomaly)
      if args.task == "get_scores":
        path_to_output_folder = f'{args.output_data}/{split}/{args.method}'
      elif args.task == "save_samples":
        path_to_output_folder = f'{args.output_data}/{split}'
      if not os.path.exists(path_to_output_folder):
        os.makedirs(path_to_output_folder)
      path_to_output = f'{path_to_output_folder}/{file_name}.json'
      out_file = open(path_to_output,'w')
      json.dump(updated_anomalies, out_file)
      out_file.close()
      # break
  end_time = time.time()
  print('total time:',end_time-start_time)
