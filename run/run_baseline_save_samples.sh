nohup python baseline.py -c configs/config_worldbank.json -t save_samples -i  3_events/extracted_events/worldbank -o 3_events/extracted_samples/worldbank > 3_events/logs/worldbank_save_samples.txt &
nohup python baseline.py -c configs/config_worldbank.json -t save_samples -i  5_events/extracted_events/worldbank -o 5_events/extracted_samples/worldbank > 5_events/logs/worldbank_save_samples.txt &
nohup python baseline.py -c configs/config_financial.json -t save_samples -i  3_events/extracted_events/financial -o 3_events/extracted_samples/financial > 3_events/logs/financial_save_samples.txt &
nohup python baseline.py -c configs/config_financial.json -t save_samples -i  5_events/extracted_events/financial -o 5_events/extracted_samples/financial > 5_events/logs/financial_save_samples.txt &