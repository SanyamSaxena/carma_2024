nohup python create_dataset.py -c configs/config_financial.json -t create_events -i 3_events/extracted_events/financial/ -o 3_events/events_with_random_negative/financial -a anomaly_scores/financial_stl/ &
nohup python create_dataset.py -c configs/config_financial.json -t create_events -i 5_events/extracted_events/financial/ -o 5_events/events_with_random_negative/financial -a anomaly_scores/financial_stl/ &
nohup python create_dataset.py -c configs/config_worldbank.json -t create_events -i 3_events/extracted_events/worldbank/ -o 3_events/events_with_random_negative/worldbank -a anomaly_scores/worldbank_stl/ &
nohup python create_dataset.py -c configs/config_worldbank.json -t create_events -i 5_events/extracted_events/worldbank/ -o 5_events/events_with_random_negative/worldbank -a anomaly_scores/worldbank_stl/ &