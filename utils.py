import numpy as np
import json
from openai import AzureOpenAI
from datetime import datetime
import torch
import bert_score
import spacy 
import ast
import re
import pprint

def read_config(config_file_path):
    with open(config_file_path, 'r') as file:
        config_data = json.load(file)
    return config_data

# Openai setup.
def setup_openai(config_data, version):
    if version=='3.5':
        client = AzureOpenAI(
          azure_endpoint = config_data["openai 3.5 api_base"],
          api_key=config_data["openai 3.5 api_key"],
          api_version=config_data["openai api_version"]
        )
    elif version=='4':
        client = AzureOpenAI(
          azure_endpoint = config_data["openai 4 api_base"],
          api_key=config_data["openai 4 api_key"],
          api_version=config_data["openai api_version"]
        )
    return client


# Openai Request
def get_gpt_response(client, prompt, deployment_name, temp=0.0):
    msgs=[{"role": "user", "content": prompt}]
    response=client.chat.completions.create(
      model=deployment_name,
      messages=msgs,
      temperature=temp
    )
    try:
      answer = response.choices[0].message.content
    except:
      answer = None
    return answer


def create_event_type_string(event_types):
    event_string = ", ".join([f"'{event_type}'" for event_type in event_types[:-1]])
    return f"{event_string} or '{event_types[-1]}'" if len(event_types) > 1 else f"'{event_types[0]}'"


# Event Extraction.
def get_events(trend, indicator, place, time, client, config_data, num_events=3, location=True, chain_of_thought=False, temp=0.0):
    if place:
      place_str = f' at {place}'
    else:
      place_str = ""
    name_key = "event name"

    deployment_name = config_data["deployment 3.5 name"]
    event_types = create_event_type_string(config_data["event types"])
    time_format = config_data["event time format"]
    if (config_data["anomaly time format"]=="yyyyQx"):
        time  = f"{time.split('Q')[0]} Quarter {time.split('Q')[1]}"
    if(location and chain_of_thought):
        prompt = f"""
You are a helpful assistant for causal relationship understanding. Think about the cause-and-effect relationships between the events and its effect on the timeseries.
According to you, what important events could have caused {trend} in {indicator}{place_str} around {time}?.
Return only python list of top {num_events} events in descending order of relevance as answer where each event is in a json parsable dictionary form (all values should be in string format) with keys "{name_key}", location (country name or "world" if event is global), start time in format {time_format}, end time in format {time_format}, type of event (one from {event_types}) and explanation.
Important Note: The key names of the dictionary must be "{name_key}", "location", "start time", "end time", "type of event" and "explanation" strictly. Do not return any text other than this list of dictionaries.'
"""

    elif(location and not chain_of_thought):
        prompt = f"""
You are a helpful assistant for causal relationship understanding. Think about the cause-and-effect relationships between the events and its effect on the timeseries.
According to you, what important events could have caused {trend} in {indicator}{place_str} around {time}?.
Return only python list of top {num_events} events in descending order of relevance as answer where each event is in a json parsable dictionary form (all values should be in string format) with keys {name_key}, location (country name or "world" if event is global), start time in format {time_format}, end time in format {time_format} and type of event (one from {event_types}).
Important Note: The key names of the dictionary must be "{name_key}", "location", "start time", "end time" and "type of event" strictly. Do not return any text other than this list of dictionaries.'
"""
    elif(not location and chain_of_thought):
        prompt = f"""
You are a helpful assistant for causal relationship understanding. Think about the cause-and-effect relationships between the events and its effect on the timeseries.
According to you, what important events could have caused {trend} in {indicator}{place_str} around {time}?.
Return only python list of top {num_events} events in descending order of relevance as answer where each event is in a json parsable dictionary form (all values should be in string format) with keys {name_key}, start time in format {time_format}, end time in format {time_format}, type of event (one from {event_types}) and explanation.
Important Note: The key names of the dictionary must be "{name_key}", "start time", "end time", "type of event" and "explanation" strictly. Do not return any text other than this list of dictionaries.'
"""
    elif(not location and not chain_of_thought):
        prompt = f"""
You are a helpful assistant for causal relationship understanding. Think about the cause-and-effect relationships between the events and its effect on the timeseries.
According to you, what important events could have caused {trend} in {indicator}{place_str} around {time}?.
Return only python list of top {num_events} events in descending order of relevance as answer where each event is in a json parsable dictionary form (all values should be in string format) with keys {name_key}, start time in format {time_format}, end time in format {time_format}, and type of event (one from {event_types}).
Important Note: The key names of the dictionary must be "{name_key}", "start time", "end time", and "type of event" strictly. Do not return any text other than this list of dictionaries.'
"""
    print(prompt)
    response = get_gpt_response(client, prompt, deployment_name, temp)
    try:
        events = json.loads(response)
    except:
        events = None
    return events


def get_event_self_check(trend, indicator, place, time, client, config_data, location=True, chain_of_thought=False, temp=1.0):
    if place:
      place_str = f' at {place}'
    else:
      place_str = ""
    name_key = "event name"
    deployment_name = config_data["deployment 3.5 name"]
    event_types = create_event_type_string(config_data["event types"])
    time_format = config_data["event time format"]
    if (config_data["anomaly time format"]=="yyyyQx"):
        time  = f"{time.split('Q')[0]} Quarter {time.split('Q')[1]}"
    if(location and chain_of_thought):
        prompt = f"""
You are a helpful assistant for causal relationship understanding. Think about the cause-and-effect relationships between the events and its effect on the timeseries.
According to you, what important event could have caused {trend} in {indicator}{place_str} around {time}?.
Return most relevant event as a json parsable dictionary form (all values should be in string format) with keys {name_key}, location, start time in format {time_format}, end time in format {time_format}, type of event (one from {event_types}) and explanation.
Important Note: The key names of the dictionary must be "{name_key}", "location", "start time", "end time", "type of event" and "explanation" strictly. Do not return any text other than this dictionary.'
"""
    elif(location and not chain_of_thought):
        prompt = f"""
You are a helpful assistant for causal relationship understanding. Think about the cause-and-effect relationships between the events and its effect on the timeseries.
According to you, what important event could have caused {trend} in {indicator}{place_str} around {time}?.
Return most relevant event as a json parsable dictionary form (all values should be in string format) with keys {name_key}, location (country name or "world" if event is global), start time in format {time_format}, end time in format {time_format} and type of event (one from {event_types}).
Important Note: The key names of the dictionary must be "{name_key}", "location", "start time", "end time" and "type of event" strictly. Do not return any text other than this dictionary.'
"""
    elif(not location and chain_of_thought):
        prompt = f"""
You are a helpful assistant for causal relationship understanding. Think about the cause-and-effect relationships between the events and its effect on the timeseries.
According to you, what important event could have caused {trend} in {indicator}{place_str} around {time}?.
Return most relevant event as a json parsable dictionary form (all values should be in string format) with keys {name_key}, location, start time in format {time_format}, end time in format {time_format}, type of event (one from {event_types}) and explanation.
Important Note: The key names of the dictionary must be "{name_key}", "start time", "end time", "type of event" and "explanation" strictly. Do not return any text other than this dictionary.'
"""
    elif(not location and not chain_of_thought):
        prompt = f"""
You are a helpful assistant for causal relationship understanding. Think about the cause-and-effect relationships between the events and its effect on the timeseries.
According to you, what important event could have caused {trend} in {indicator}{place_str} around {time}?.
Return most relevant event as a json parsable dictionary form (all values should be in string format) with keys {name_key}, start time in format {time_format}, end time in format {time_format} and type of event (one from {event_types}).
Important Note: The key names of the dictionary must be "{name_key}", "start time", "end time", and "type of event" strictly. Do not return any text other than this dictionary.'
"""
    response = get_gpt_response(client, prompt, deployment_name, temp)
    try:
        event = json.loads(response)
    except:
        event = None
    return event


def get_contradiction_prompt_s(event, indicator, place, trend, time, explain, config_data):
    if (config_data["anomaly time format"]=="yyyyQx"):
        time  = f"{time.split('Q')[0]} Quarter {time.split('Q')[1]}"
    if place:
      place_str = f' at {place}'
    else:
      place_str = ""
    event_name = event['event name']
    if(config_data['place']=="yes"):
        event_loc = f" in {event['location']}"
    else:
        event_loc = ""
    event_start_time = event['start time']
    event_end_time = event['end time']

    if explain == "yes":
        prompt = f"""
You are a helpful assistant for causal relationship understanding. Think about the cause-and-effect relationships between the event and its effect on the indicator.
Event: {event_name} which happened from {event_start_time} to {event_end_time}{event_loc}
Effect: {trend} in {indicator}{place_str} around {time}

Could the event create this effect? Answer from one of the following options.
Yes: Event could cause this effect.
No: Event cannot cause this effect. 

Let’s work this out in a step-by-step way to be sure that we have the right answer.
Then provide your final answer within the tags at the end separately, <Answer>answer</Answer>.
Important Note: You must return the final answers and put it between the tags <Answer>answer</Answer>. Answer must from one of the 2 options provided.
"""
    elif explain == "no":
        prompt = f"""
You are a helpful assistant for causal relationship understanding. Think about the cause-and-effect relationships between the event and its effect on the indicator.
Event: {event_name} which happened from {event_start_time} to {event_end_time}{event_loc}
Effect: {trend} in {indicator}{place_str} around {time}

Could the event create this effect? Answer from one of the following options.
Yes: Event could cause this effect.
No: Event cannot cause this effect.

Answer should be one of the options 'Yes', 'No'.
Important Note: Return just the answer from the options and nothing else.
"""
    return prompt



def cross_examine_relevance_contradiction_s(event, trend, indicator, place, time, explain, client, config_data):
    deployment_name = config_data["deployment 3.5 name"]
    causal_score_map = {'yes': 1,
                        'no': -1
                        }

    prompt = get_contradiction_prompt_s(event, indicator, place, trend, time, explain, config_data)
    response = get_gpt_response(client, prompt, deployment_name)
    print(prompt)
    print(response)
    print()

    try:
        if(explain=="yes"):
            pattern = re.compile(r'<Answer>(.*?)<\/Answer>', re.DOTALL)
            match = pattern.search(response)
            if match:
                extracted_value_pos = match.group(1)
        elif(explain=="no"):
            extracted_value_pos = response
        positive_causal = extracted_value_pos.lower().strip()
        positive_causality_score = causal_score_map[positive_causal]
    except:
        positive_causality_score = 0
    print(positive_causality_score)

    inverse_trend = config_data["trends"][trend]
    prompt_inverse = get_contradiction_prompt_s(event, indicator, place, inverse_trend, time, explain, config_data)
    response_inverse = get_gpt_response(client, prompt_inverse, deployment_name)
    print(prompt_inverse)
    print(response_inverse)
    print()

    try:
        if(explain=="yes"):
            pattern = re.compile(r'<Answer>(.*?)<\/Answer>', re.DOTALL)
            match = pattern.search(response_inverse)
            if match:
                extracted_value_neg = match.group(1)
        elif(explain=="no"):
            extracted_value_neg = response_inverse
        negative_causal = extracted_value_neg.lower().strip()
        negative_causality_score = causal_score_map[negative_causal]
    except:
        negative_causality_score = 0
    print(negative_causality_score)

    strength = abs(positive_causality_score-negative_causality_score)/2

    return_dict = {}
    return_dict['raw'] = [response, response_inverse]
    return_dict['+ve score'] = positive_causality_score
    return_dict['-ve score'] = negative_causality_score
    return_dict['strength'] = strength
    return_dict['score'] = (positive_causality_score-negative_causality_score)/2
    return return_dict


def get_contradiction_prompt(event, indicator, place, trend, time, explain, config_data):
    if (config_data["anomaly time format"]=="yyyyQx"):
        time  = f"{time.split('Q')[0]} Quarter {time.split('Q')[1]}"
    if place:
      place_str = f' at {place}'
    else:
      place_str = ""
    event_name = event['event name']
    if(config_data['place']=="yes"):
        event_loc = f" in {event['location']}"
    else:
        event_loc = ""
    event_start_time = event['start time']
    event_end_time = event['end time']

    if explain == "yes":
        prompt = f"""
You are a helpful assistant for causal relationship understanding. Think about the cause-and-effect relationships between the event and its effect on the indicator.
Event: {event_name} which happened from {event_start_time} to {event_end_time}{event_loc}
Effect: {trend} in {indicator}{place_str} around {time}

Could the event create this effect? Answer from one of the following options.
Strong yes: Event is definitely responsible for this effect.
Weak yes: Event might be responsible for this effect.
Weak no: Event might not to be responsible for this effect.
Strong no: Event is definitely not responsible for this effect.

Let’s work this out in a step-by-step way to be sure that we have the right answer.
Then provide your final answer within the tags at the end separately, <Answer>answer</Answer>.
Important Note: You must return the final answers and put it between the tags <Answer>answer</Answer>. Answer must from one of the 4 options provided.
"""
    elif explain == "no":
        prompt = f"""
You are a helpful assistant for causal relationship understanding. Think about the cause-and-effect relationships between the event and its effect on the indicator.
Event: {event_name} which happened from {event_start_time} to {event_end_time}{event_loc}
Effect: {trend} in {indicator}{place_str} around {time}

Could the event create this effect? Answer from one of the following options.
Strong yes: Event is definitely responsible for this effect.
Weak yes: Event might be responsible for this effect.
Weak no: Event might not to be responsible for this effect.
Strong no: Event is definitely not responsible for this effect.

Answer should be one of the options 'Strong yes', 'Weak yes', 'Weak no', 'Strong no'.
Important Note: Return just the answer from the options and nothing else.
"""
    return prompt


def cross_examine_relevance_contradiction(event, trend, indicator, place, time, explain, client, config_data):
    deployment_name = config_data["deployment 3.5 name"]
    causal_score_map = {'strong yes': 2,
                        'weak yes': 1,
                        'weak no': -1,
                        'strong no': -2,
                        }


    prompt = get_contradiction_prompt(event, indicator, place, trend, time, explain, config_data)
    response = get_gpt_response(client, prompt, deployment_name)
    print(prompt)
    print(response)
    print()

    try:
        if(explain=="yes"):
            pattern = re.compile(r'<Answer>(.*?)<\/Answer>', re.DOTALL)
            match = pattern.search(response)
            if match:
                extracted_value_pos = match.group(1)
        elif(explain=="no"):
            extracted_value_pos = response
        positive_causal = extracted_value_pos.lower().strip()
        positive_causality_score = causal_score_map[positive_causal]
    except:
        positive_causality_score = 0
    print(positive_causality_score)

    inverse_trend = config_data["trends"][trend]
    prompt_inverse = get_contradiction_prompt(event, indicator, place, inverse_trend, time, explain, config_data)
    response_inverse = get_gpt_response(client, prompt_inverse, deployment_name)
    print(prompt_inverse)
    print(response_inverse)
    print()

    try:
        if(explain=="yes"):
            pattern = re.compile(r'<Answer>(.*?)<\/Answer>', re.DOTALL)
            match = pattern.search(response_inverse)
            if match:
                extracted_value_neg = match.group(1)
        elif(explain=="no"):
            extracted_value_neg = response_inverse
        negative_causal = extracted_value_neg.lower().strip()
        negative_causality_score = causal_score_map[negative_causal]
    except:
        negative_causality_score = 0
    print(negative_causality_score)

    strength = abs(positive_causality_score-negative_causality_score)/4

    return_dict = {}
    return_dict['raw'] = [response, response_inverse]
    return_dict['+ve score'] = positive_causality_score
    return_dict['-ve score'] = negative_causality_score
    return_dict['strength'] = strength
    return_dict['score'] = (positive_causality_score-negative_causality_score)/4
    return return_dict


def cross_examine_relevance_pattern(event, trend, indicator, place, time, explain, client, config_data):
    if place:
      place_str = f' at {place}'
    else:
      place_str = ""
    if (config_data["anomaly time format"]=="yyyyQx"):
        time  = f"{time.split('Q')[0]} Quarter {time.split('Q')[1]}"
    deployment_name = config_data["deployment 3.5 name"]
    event_name = event['event name']
    if(config_data['place']=="yes"):
        event_loc = f" in {event['location']}"
    else:
        event_loc = ""
    event_start_time = event['start time']
    event_end_time = event['end time']

    if explain=="yes":
        prompt = f"""
You are a helpful assistant for causal relationship understanding. Think about the cause-and-effect relationships between the event and its effect on the indicator.
Event: {event_name} which happened from {event_start_time} to {event_end_time}{event_loc}
Indicator: {indicator}{place_str} around {time}

Event's effect on the Indicator is:
Increase: Event could increase the indicator. Choose this option if event has positive impact on indicator.
Decrease: Event could decrease the indicator. Choose this option if event has negative impact on indicator.
No effect: Event could not affect the indicator. Choose this option if event has no impact on indicator.

Magnitude of this effect is measured using a strength score from 0 to 100. (In case of No Effect return 0)
Score above 80: Event is related to this indicator and will definitely affect it.
Score between 50 and 80: Event is related to this indicator and might affect it.
Score between 20 and 50: Event might be related to this indicator but is less likely to affect it.
Score below 20: Event is not related to this indicator and will not affect it.

Let’s work this out in a step-by-step way to be sure that we have the right answer.
Then provide your final answer within the tags, <Effect>effect</Effect> and <Magnitude>magnitude</Magnitude>. Effect must be from one of the 3 options provided. Magnitude must be a single integer score from 0 to 100.
Important Note: You must put the final answers within the tags <Effect>effect</Effect> and <Magnitude>magnitude</Magnitude>. This is very important for parsing the answer.
"""
    elif explain=="no":
        prompt = f"""
You are a helpful assistant for causal relationship understanding. Think about the cause-and-effect relationships between the event and its effect on the indicator.
Event: {event_name} which happened from {event_start_time} to {event_end_time}{event_loc}
Indicator: {indicator}{place_str} around {time}

Event's effect on the Indicator is:
Increase: Event could increase the indicator. Choose this option if event has positive impact on indicator.
Decrease: Event could decrease the indicator. Choose this option if event has negative impact on indicator.
No effect: Event could not affect the indicator. Choose this option if event has no impact on indicator.

Magnitude of this effect is measured using a strength score from 0 to 100. (In case of No Effect return 0)
Score above 80: Event is related to this indicator and will definitely affect it.
Score between 50 and 80: Event is related to this indicator and might affect it.
Score between 20 and 50: Event might be related to this indicator but is less likely to affect it.
Score below 20: Event is not related to this indicator and will not affect it.

Return your answer as a python list of strings ["Effect", "Magnitude"]. Effect must be from one of the 3 options provided. Magnitude must be a single integer score from 0 to 100.
Important Note: Return just this list as answer and nothing else.
"""

    response = get_gpt_response(client, prompt, deployment_name)
    print(prompt)
    print(response)

    try:
        if explain=="yes":
            pattern_effect = re.compile(r'<Effect>(.*?)<\/Effect>', re.DOTALL)
            match = pattern_effect.search(response)
            if match:
                effect = str(match.group(1)).strip().lower()
            pattern_magnitude = re.compile(r'<Magnitude>(.*?)<\/Magnitude>', re.DOTALL)
            match = pattern_magnitude.search(response)
            if match:
                magnitude = int(match.group(1).strip())
        elif explain=="no":
            effect_mag_list = json.loads(response)
            effect = effect_mag_list[0]
            magnitude = effect_mag_list[1]
            
        print(effect, magnitude) 
    except:
        return_dict = {}
        return_dict['raw'] = response
        return_dict['score'] = 'NA'
        return return_dict

    return_dict = {}
    return_dict['raw'] = response
    return_dict['type'] = effect
    return_dict['strength'] = magnitude
    return return_dict


def cross_examine_relevance_consensus(event, trend, indicator, place, time, anomaly_scores_dict, client, config_data):
  anomaly_scores = anomaly_scores_dict['anomaly_scores']
  series_start = anomaly_scores_dict['start']
  series_end = anomaly_scores_dict['end']
  if (config_data['anomaly time format']=="yyyyQx"):
    start_y =  series_start.split("Q")[0]
    start_q = series_start.split("Q")[1]
    end_y = series_end.split("Q")[0]
    end_q = series_end.split("Q")[1]
    series_start = f"{start_y} quarter {start_q}"
    series_end = f"{end_y} quarter {end_q}"
  deployment_name = config_data["deployment 3.5 name"]
  event_name = event['event name']
  event_loc = event['location']
  if place:
    place_str = f' at {place}'
  else:
    place_str = ""

  if (config_data['anomaly time format']=="yyyyQx"):
    example_str = """ 
Sample Answer 1: <Answer>[["2007Q3", "2007Q3"]]</Answer>
Sample Answer 2: <Answer>[["2007Q4", "2008Q1"]]</Answer>
Sample Answer 3: <Answer>[["2007Q4", "2007Q4"], ["2010Q4", "2011Q2"]]</Answer>
Sample Answer 4: <Answer>[["2007Q4", "2007Q4"], ["2010Q4", "2011Q2"], ["2013Q4", "2014Q1"]]</Answer>
"""
  elif (config_data['anomaly time format']=="yyyy"):
    example_str = """ 
Sample Answer 1: <Answer>[["2007", "2007"]]</Answer>
Sample Answer 2: <Answer>[["2007", "2008"]]</Answer>
Sample Answer 3: <Answer>[["2007", "2007"], ["2010", "2011"]]</Answer>
Sample Answer 4: <Answer>[["2007", "2007"], ["2010", "2011"], ["2013", "2014"]]</Answer>
"""

  prompt = f"""
You are a helpful assistant who has good knowledge of history and important events. Use this knowledge to answer the following question.
Event: {event_name} which happened in {event_loc}
Related Indicator: {indicator}{place_str}
Between {series_start} and {series_end}, return the time periods when this event happened.

Return answer as a list of these time periods in the format:

[[start_time_1, end_time_1], [start_time_2, end_time_2], [start_time_3, end_time_3]...]

Some sample answers are shown below (each line is a sample answer):
{example_str}
Give the best answer as per your knowledge.
Important Note: Return the final answer between the tags <Answer>answer</Answer>.
"""

  response = get_gpt_response(client, prompt, deployment_name)
  print(prompt)
  print(response)
  try:
    pattern = re.compile(r'<Answer>(.*?)<\/Answer>', re.DOTALL)
    match = pattern.search(response)
    if match:
        answer = json.loads(match.group(1))
    else:
        answer = "NA"
  except:
    print("Error")
    answer = 'NA'
  print(answer)
  return_dict = {}
  return_dict['raw'] = response
  return_dict['times occured'] = answer
  return return_dict


def cross_examine_event(event, trend, indicator, place, time, anomaly_scores_dict, explain, client, config_data):
    cross_examination_info = {}
    cross_examination_info = {}
    cross_examination_info['contradiction s'] = cross_examine_relevance_contradiction_s(event, trend, indicator, place, time, explain, client, config_data)
    cross_examination_info['contradiction simple'] = cross_examine_relevance_contradiction(event, trend, indicator, place, time, explain, client, config_data)
    cross_examination_info['pattern'] = cross_examine_relevance_pattern(event, trend, indicator, place, time, explain, client, config_data)
    cross_examination_info['consensus'] = cross_examine_relevance_consensus(event, trend, indicator, place, time, anomaly_scores_dict, client, config_data)
    return cross_examination_info


# Scoring
def calc_score_time(event, anomaly_time, config_data):
    event_start_time = event['start time']#2019-10
    event_end_time = event['end time']
    anomaly_time_format = config_data["anomaly time format"]
    try:
      date_object_start = datetime.strptime(event_start_time, "%Y-%m")
    except:
      date_object_start = None
    try:
      date_object_end = datetime.strptime(event_end_time, "%Y-%m")
    except:
        if(event_end_time.strip().lower()=="present" or event_end_time.strip().lower()=="ongoing"):
            date_object_end = datetime.strptime("2021-12", "%Y-%m")
        else:
            date_object_end = None
    start_time_pred, end_time_pred =  None, None
    if anomaly_time_format == "yyyyQx" and date_object_start and date_object_end:
        anomaly_time = int(anomaly_time.split('Q')[0]) + int(anomaly_time.split('Q')[1])*0.25
        year_start = date_object_start.year
        quarter_start = (date_object_start.month - 1) // 3 + 1
        start_time_pred = year_start + quarter_start*0.25
        year_end = date_object_end.year
        quarter_end = (date_object_end.month - 1) // 3 + 1
        end_time_pred = year_end + quarter_end*0.25
    elif anomaly_time_format=="yyyy" and date_object_start and date_object_end:
        anomaly_time = anomaly_time
        start_time_pred =  date_object_start.year
        end_time_pred =  date_object_end.year
    if not start_time_pred or not end_time_pred:
        return 0.5
    if (anomaly_time<start_time_pred):
        return 0
    elif (anomaly_time>=start_time_pred and anomaly_time <= end_time_pred):
        return 1
    else:
        if config_data['dataset']=="worldbank":
          return max(0, 1-(anomaly_time - end_time_pred)/5)
        elif config_data['dataset']=="financial":
          return max(0, 1-(anomaly_time - end_time_pred))#delta will already be 0.25


def calc_pattern_score(return_dict, trend):
    if return_dict['type'].strip().lower()=='increase':
        if(trend=='peak' or trend=="increase"):
          effect_on_indicator = 1
        elif(trend=="dip" or trend=="decrease"):
          effect_on_indicator = -1 
    elif return_dict['type'].strip().lower()=='no effect':
        effect_on_indicator = 0
    elif return_dict['type'].strip().lower()=='decrease':
        if(trend=='peak' or trend=="increase"):
          effect_on_indicator = -1
        elif(trend=="dip" or trend=="decrease"):
          effect_on_indicator = 1
    else:
       effect_on_indicator = 0

    try:
        score = int(return_dict['strength']) * effect_on_indicator / 100
    except:
        score = 0
    return score


def calc_consensus_score(event, anomaly_time, anomaly_trend, indicator, place, anomaly_scores_dict, config_data):
  anomaly_scores = anomaly_scores_dict['anomaly_scores']
  anomaly_scores_list = list(anomaly_scores.values())
  positive_scores = [x for x in anomaly_scores_list if x >= 0]
  negative_scores = [x for x in anomaly_scores_list if x < 0]
  positive_avg = np.mean(positive_scores)
  negative_avg = abs(np.mean(negative_scores))
  anomaly_avg = np.mean(anomaly_scores_list)
  anomaly_times_list = list(anomaly_scores.keys())

  if config_data['anomaly time format'] == "yyyyQx":
      series_start_time = int(anomaly_times_list[0].split('Q')[0]) + int(anomaly_times_list[0].split('Q')[1])*0.25
      series_end_time = int(anomaly_times_list[-1].split('Q')[0]) + int(anomaly_times_list[-1].split('Q')[1])*0.25
  elif config_data['anomaly time format']=="yyyy":
      series_start_time = int(anomaly_times_list[0])
      series_end_time = int(anomaly_times_list[-1])

  periods_list = event['cross_examine']['consensus']['times occured']
  answer_dict = {}
  if(periods_list=="NA"):
      if(anomaly_trend=="increase"):
          answer_dict['our_max'] = positive_avg
          answer_dict['our_max_minus'] = positive_avg
      elif (anomaly_trend=="decrease"):
          answer_dict['our_max'] = negative_avg
          answer_dict['our_max_minus'] = negative_avg
      answer_dict['avg'] = anomaly_avg
      answer_dict['avg_minus'] = anomaly_avg
      answer_dict['max'] = anomaly_avg
      answer_dict['max_minus'] = anomaly_avg
      return answer_dict

  selected_anomaly_scores = []
  mask_inside = np.zeros(len(anomaly_scores_list), dtype=bool)
  try:
      if(len(periods_list)==0):
          inside_average = 'empty'
      else:
          for start_period, end_period in periods_list:
              try:
                  if config_data['anomaly time format'] == "yyyyQx":
                      start_time = int(start_period.split('Q')[0]) + int(start_period.split('Q')[1])*0.25
                      end_time = int(end_period.split('Q')[0]) + int(end_period.split('Q')[1])*0.25
                  elif config_data['anomaly time format']=="yyyy":
                      start_time = int(start_period)
                      end_time = int(end_period)
                  if(start_time<series_start_time):
                      start_period = anomaly_times_list[0]
                  if(end_time>series_end_time):
                      end_period = anomaly_times_list[-1]                       
                  start_index = anomaly_times_list.index(start_period)
                  end_index = anomaly_times_list.index(end_period) + 1
              except:
                  continue
              
              mask_inside[start_index:end_index] = True
              selected_anomaly_scores.append(anomaly_scores_list[start_index:end_index])
          if(len(selected_anomaly_scores)==0):
              inside_average = 'empty'
          else:
              average_scores = [sum(scores)/len(scores) for scores in selected_anomaly_scores]
              inside_average = sum(average_scores) / len(average_scores)
              max_anomalies = [max(scores, key=abs) for scores in selected_anomaly_scores]
              max_average = sum(max_anomalies) / len(max_anomalies)
              if(anomaly_trend=="increase"):
                  our_max_anomalies = [max(scores) for scores in selected_anomaly_scores]
              elif (anomaly_trend=="decrease"):
                  our_max_anomalies = [min(scores) for scores in selected_anomaly_scores]
              our_max_average = sum(our_max_anomalies) / len(our_max_anomalies)
  except:
      inside_average = "NA"

  if inside_average=="NA" or inside_average=="empty":
      if(anomaly_trend=="increase"):
          answer_dict['our_max'] = positive_avg
          answer_dict['our_max_minus'] = positive_avg
      elif (anomaly_trend=="decrease"):
          answer_dict['our_max'] = negative_avg
          answer_dict['our_max_minus'] = negative_avg
      answer_dict['avg'] = anomaly_avg
      answer_dict['avg_minus'] = anomaly_avg
      answer_dict['max'] = anomaly_avg
      answer_dict['max_minus'] = anomaly_avg
  elif anomaly_trend == 'increase':
      rem = np.array(anomaly_scores_list)[~mask_inside]
      if(rem.shape[0]==0):
          outside_average = 0
      else:
          outside_average = np.mean(rem)
      answer_dict['avg'] = inside_average
      answer_dict['avg_minus'] = inside_average - outside_average
      answer_dict['max'] = max_average
      answer_dict['max_minus'] = max_average - outside_average
      answer_dict['our_max'] = our_max_average
      answer_dict['our_max_minus'] = our_max_average - outside_average
  elif anomaly_trend=="decrease":
      rem = np.array(anomaly_scores_list)[~mask_inside]
      if(rem.shape[0]==0):
          outside_average = 0
      else:
          outside_average = np.mean(rem)
      answer_dict['avg'] = -inside_average
      answer_dict['avg_minus'] = -(inside_average - outside_average)
      answer_dict['max'] = -max_average
      answer_dict['max_minus'] = -(max_average - outside_average)
      answer_dict['our_max'] = -our_max_average
      answer_dict['our_max_minus'] = -(our_max_average - outside_average)
  return answer_dict


def get_features_from_event(event, trend, feature_level):
    feature = []

    if (feature_level != "drop_contradiction"):
        feature.append(event['cross_examine']['contradiction s']["+ve score"])
        feature.append(event['cross_examine']['contradiction s']["-ve score"])

    if(feature_level != "drop_pattern"):
        if event['cross_examine']['pattern']['type'].strip().lower()=='increase':
            if(trend=="increase"):
              effect_on_indicator = 1
            elif(trend=="decrease"):
              effect_on_indicator = -1 
        elif event['cross_examine']['pattern']['type'].strip().lower()=='no effect':
            effect_on_indicator = 0
        elif event['cross_examine']['pattern']['type'].strip().lower()=='decrease':
            if(trend=="increase"):
              effect_on_indicator = -1
            elif(trend=="decrease"):
              effect_on_indicator = 1
        else:
           effect_on_indicator = 0
        feature.append(effect_on_indicator)

        try:
            magnitude = int(event['cross_examine']['pattern']['strength'])/100
            strength = magnitude * effect_on_indicator
            feature.append(magnitude)
            feature.append(strength)
        except:
            feature.append(0.5)
            feature.append(0)

    if(feature_level != "drop_time"):
        feature.append(event['score']['time'])
    if(feature_level != "drop_consensus"):
        feature.append(event['score']['consensus_stl']['avg'])

    event['feature'] = np.array(feature)
    return event


def gpt4_rank_event(anomaly_details, events, trend, indicator, place, time, client, explain, config_data):    
    deployment_name = config_data["deployment 4 name"]
    if place:
      place_str = f' at {place}'
    else:
      place_str = ""
    events_prompts = []
    for i, event in enumerate(events):
      event_name = event['event name'].strip().lower()
      if(config_data['place']=="yes"):
        event_loc = event['location'].strip().lower()
      else:
        event_loc = ""
      event_time_start = event['start time'].strip().lower()
      event_time_end = event['end time'].strip().lower()
      events_prompts.append([event_name, event_loc, event_time_start, event_time_end])

    prompt = f"""
You are a helpful assistant for causal relationship understanding. Think about the cause-and-effect relationships between the event and its effect.

Following is a list of events:
"""
    for i, event_prompt in enumerate(events_prompts):
      if(config_data['place']=="yes"):
          prompt+=f'Event Number {i+1}: {event_prompt[0]} which happened from {event_prompt[2]} to {event_prompt[3]} in {event_prompt[1]}\n'
      else:
          prompt+=f'Event Number {i+1}: {event_prompt[0]} which happened from {event_prompt[2]} to {event_prompt[3]}\n'
          
    prompt+="\n"

    if(explain=="no"):
        print(events_prompts)
        if (len(events_prompts)==5):
            example_rank = "[1,2,4,3,5], [2,5,4,1,3], [5,4,2,1,3], [3,1,2,4,5]"
        elif (len(events_prompts)==3):
            example_rank = "[1,2,3], [2,1,3], [2,3,1], [3,1,2]"
        prompt += f"""
Effect: {trend} in {indicator}{place_str} around {time}

Rank these events in descending order of their ability to cause this effect.
Return answer as a list of integers which correspond to the the event numbers of the events.
Important Note: The answer must be in format of a list of integers. Some examples of answers are {example_rank}. Do not return any other text.
"""
    elif(explain=="yes"):
        print(events_prompts)
        if (len(events_prompts)==5):
            example_rank = "<Answer>[1,2,4,3,5]</Answer>, <Answer>[2,5,4,1,3]</Answer>, <Answer>[5,4,2,1,3]</Answer>, <Answer>[3,1,2,4,5]</Answer>"
        elif (len(events_prompts)==3):
            example_rank = "<Answer>[1,2,3]</Answer>, <Answer>[2,1,3]</Answer>, <Answer>[2,3,1]</Answer>, <Answer>[3,1,2]</Answer>"
        prompt += f"""
Effect: {trend} in {indicator}{place_str} around {time}

Rank these events in descending order of their ability to cause this effect.
Let’s work this out in a step-by-step way to be sure that we have the right answer.
Then return your final answer within the tags, <Answer>ranking</Answer> where ranking is a list of integers which correspond to the the event numbers of the events.
Important Note: The final answer must be within tags and in format of a list of integers. Some examples of answers are {example_rank}. Do not return any other text.
"""

    print(prompt)
    response = get_gpt_response(client, prompt, deployment_name)
    try:
        if explain=="yes":
            print(response)
            pattern = re.compile(r'<Answer>(.*?)<\/Answer>', re.DOTALL)
            match = pattern.search(response)
            if match:
                list_sorted_events = json.loads(match.group(1))
        elif explain=="no":
            print(response)
            list_sorted_events = json.loads(response)
        print(list_sorted_events)
        print()
    except:
        print("Error")
        answer_dict = {}
        answer_dict['raw response'] = response 
        answer_dict['original_event_prompts'] = events_prompts
        return answer_dict

    sorted_event_prompts = []
    for event_number in list_sorted_events:
        sorted_event_prompts.append(events_prompts[int(event_number)-1])

    answer_dict = {}
    answer_dict['raw response'] = response 
    answer_dict['raw sorted'] = list_sorted_events
    answer_dict['original_event_prompts'] = events_prompts
    answer_dict['sorted_event_prompts'] = sorted_event_prompts
    return answer_dict


def gpt4_score_one_event(anomaly_details, events, trend, indicator, place, time, client, config_data):
    deployment_name = config_data["deployment 4 name"]
    if (config_data["anomaly time format"]=="yyyyQx"):
        time  = f"{time.split('Q')[0]} Quarter {time.split('Q')[1]}"
    if place:
        place_str = f' at {place}'
    else:
        place_str = ""
    for i, event in enumerate(events):
        event_name = event['event name'].strip().lower()
        if(config_data['place']=="yes"):
            event_loc = event['location'].strip().lower()
            event_loc = f" in {event_loc}"
        else:
            event_loc = ""
        event_time_start = event['start time'].strip().lower()
        event_time_end = event['end time'].strip().lower()
        prompt = f"""
You are a helpful assistant for causal relationship understanding. Think about the cause-and-effect relationships between the event and its effect.
Event: {event_name} which happened from {event_time_start} to {event_time_end}{event_loc}
Effect: {trend} in {indicator}{place_str} around {time}
What is the causality score for this event to cause this effect from 0 to 100?

Score above 80: Event is related to the effect and is definitely responsible for effect.
Score between 50-80: Event is related to the effect and might be responsible for effect.
Score between 20-50: Event might be related to the effect but less likely to cause this effect.
Score below 20: Event is not related to the effect and cannot cause this effect.

Let’s work this out in a step-by-step way to be sure that we have the right answer.
Then provide your final answer within the tags, <Answer>causality_score</Answer> where causality score is a value between 0 to 100.
"""

        try:
            print(prompt)
            response = get_gpt_response(client, prompt, deployment_name)
            print(response)

            pattern = re.compile(r'<Answer>(.*?)<\/Answer>', re.DOTALL)
            match = pattern.search(response)
            if match:
                extracted_value = int(match.group(1))
            print(extracted_value)
            event['gpt4 supervision'] = {'score one event': extracted_value}
        except:
            event['gpt4 supervision'] = {'score one event': 'NA'}
    return anomaly_details

#SelfCheckGPT

def convert_event_to_passage(event, trend, indicator, place, trend_time):
    if place:
      place_str = f' at {place}'
    else:
      place_str = ""
    key_name = "event name"
    event_name = event[key_name].strip()
    event_loc = event['location'].strip()
    event_time_start = event['start time'].strip()
    event_time_end = event['end time'].strip()
    sentence_1 = f"Event \"{event_name}\" can {trend} {indicator}{place_str} around {trend_time}."
    sentence_2 = f"Event \"{event_name}\" started in {event_time_start} and ended in {event_time_end}."
    sentence_3 = f"Event \"{event_name}\" happened in {event_loc}."
    sentences = [sentence_1, sentence_2, sentence_3]
    passage = " ".join(sentences)
    return sentences, passage


def self_check_prompt(sentences, samples, client, config_data):
    deployment_name = config_data["deployment 3.5 name"]
    score_sentences = []
    returned_raw = {}
    for sentence in sentences:
        returned_raw[sentence] = {}
        count_yes = 0
        for sample in samples:
            returned_raw[sentence][sample] = []
            prompt = f"Sentence: \"{sentence}\"\n\n"\
                     f"Context: \"{sample}\"\n\n"\
                     f"Is the sentence supported by the context above?"\
                     f"Answer from one of the options \'yes\', \'no\' and nothing else at all."
            try:
                response = get_gpt_response(client, prompt, deployment_name)
            except:
                print("Error")
                count_yes+=0.5
                continue
            returned_raw[sentence][sample].append(response)
            if(response.strip().lower()=='yes'):
                count_yes+=1
            elif(response.strip().lower()=='no'):
                count_yes+=0
            else:
                count_yes+=0.5
        score_sentence = count_yes/len(samples)
        score_sentences.append(score_sentence)
    answer_dict = {}
    answer_dict["raw"] =  returned_raw
    answer_dict["score"] =  score_sentences
    return answer_dict


def self_check_mqag(sentences, passage, samples, selfcheck_mqag):
    sent_scores_mqag = selfcheck_mqag.predict(
        sentences = sentences,               # list of sentences
        passage = passage,                   # passage (before sentence-split)
        sampled_passages = samples, # list of sampled passages
        num_questions_per_sent = 5,          # number of questions to be drawn
        scoring_method = 'bayes_with_alpha', # options = 'counting', 'bayes', 'bayes_with_alpha'
        beta1 = 0.8, beta2 = 0.8,            # additional params depending on scoring_method
    )
    answer_dict = {}
    answer_dict["score"] =  sent_scores_mqag.tolist()
    return answer_dict


def expand_list1(mylist, num):
    expanded = []
    for x in mylist:
        for _ in range(num):
            expanded.append(x)
    return expanded


def expand_list2(mylist, num):
    expanded = []
    for _ in range(num):
        for x in mylist:
            expanded.append(x)
    return expanded


class SelfCheckBERTScore:
    def __init__(self, nlp, default_model="en", rescale_with_baseline=True):
        self.nlp = nlp
        self.default_model = default_model # en => roberta-large
        self.rescale_with_baseline = rescale_with_baseline
        print("SelfCheck-BERTScore initialized")

    # @torch.no_grad()
    def predict(
        self,
        sentences,
        sampled_passages
    ):
        num_sentences = len(sentences)
        num_samples = len(sampled_passages)
        bertscore_array = np.zeros((num_sentences, num_samples))
        for s in range(num_samples):
            sample_passage = sampled_passages[s]
            sentences_sample = [sent for sent in self.nlp(sample_passage).sents] # List[spacy.tokens.span.Span]
            sentences_sample = [sent.text.strip() for sent in sentences_sample if len(sent) > 3]
            num_sentences_sample  = len(sentences_sample)

            refs  = expand_list1(sentences, num_sentences_sample) # r1,r1,r1,....
            cands = expand_list2(sentences_sample, num_sentences) # s1,s2,s3,...

            P, R, F1 = bert_score.score(
                    cands, refs,
                    lang=self.default_model, verbose=False,
                    rescale_with_baseline=self.rescale_with_baseline,
                    device="cuda:2"
            )
            F1_arr = F1.reshape(num_sentences, num_sentences_sample)
            F1_arr_max_axis1 = F1_arr.max(axis=1).values
            F1_arr_max_axis1 = F1_arr_max_axis1.numpy()

            bertscore_array[:,s] = F1_arr_max_axis1
        
        bertscore_mean_per_sent = bertscore_array.mean(axis=-1)
        one_minus_bertscore_mean_per_sent = 1.0 - bertscore_mean_per_sent
        return one_minus_bertscore_mean_per_sent


def self_check_bertscore(sentences, samples, selfcheck_bertscore):
    sent_scores_bertscore = selfcheck_bertscore.predict(
        sentences = sentences,                          # list of sentences
        sampled_passages = samples, # list of sampled passages
    )
    answer_dict = {}
    answer_dict["score"] =  sent_scores_bertscore.tolist()
    return answer_dict


def self_check_ngram(sentences, passage, samples, selfcheck_ngram):
    sent_scores_ngram = selfcheck_ngram.predict(
        sentences = sentences,
        passage = passage,
        sampled_passages = samples,
    )
    answer_dict = {}
    answer_dict["score"] =  sent_scores_ngram
    return answer_dict


def self_check_nli(sentences, samples, selfcheck_nli):
    sent_scores_nli = selfcheck_nli.predict(
        sentences = sentences,                          # list of sentences
        sampled_passages = samples, # list of sampled passages
    )
    answer_dict = {}
    answer_dict["score"] = sent_scores_nli.tolist()
    return answer_dict
