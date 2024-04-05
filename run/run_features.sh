nohup python all_tasks.py -c configs/config_worldbank.json -t extract_features -i 5_events/scoring/worldbank/ -o 5_events/features/worldbank &
nohup python all_tasks.py -c configs/config_worldbank.json -t extract_features -i 3_events/scoring/worldbank/ -o 3_events/features/worldbank &
nohup python all_tasks.py -c configs/config_financial.json -t extract_features -i 5_events/scoring/financial/ -o 5_events/features/financial &
nohup python all_tasks.py -c configs/config_financial.json -t extract_features -i 3_events/scoring/financial/ -o 3_events/features/financial &
