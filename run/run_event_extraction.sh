# nohup python all_tasks.py -s test -c configs/config_worldbank.json -t extract_events -i anomalies/worldbank -o 5_events/extracted_events/worldbank/ -n 5 > 5_events/logs/worldbank_extraction_test.txt &
# nohup python all_tasks.py -s test -c configs/config_worldbank.json -t extract_events -i anomalies/worldbank -o 3_events/extracted_events/worldbank/ -n 3 > 3_events/logs/worldbank_extraction_test.txt &
# nohup python all_tasks.py -s test -c configs/config_financial.json -t extract_events -i anomalies/financial -o 5_events/extracted_events/financial/ -n 5 > 5_events/logs/financial_extraction_test.txt &
# nohup python all_tasks.py -s test -c configs/config_financial.json -t extract_events -i anomalies/financial -o 3_events/extracted_events/financial/ -n 3 > 3_events/logs/financial_extraction_test.txt &
# nohup python all_tasks.py -s train -c configs/config_worldbank.json -t extract_events -i anomalies/worldbank -o 5_events/extracted_events/worldbank/ -n 5 > 5_events/logs/worldbank_extraction_train.txt &
# nohup python all_tasks.py -s train -c configs/config_worldbank.json -t extract_events -i anomalies/worldbank -o 3_events/extracted_events/worldbank/ -n 3 > 3_events/logs/worldbank_extraction_train.txt &
# nohup python all_tasks.py -s train -c configs/config_financial.json -t extract_events -i anomalies/financial -o 5_events/extracted_events/financial/ -n 5 > 5_events/logs/financial_extraction_train.txt &
# nohup python all_tasks.py -s train -c configs/config_financial.json -t extract_events -i anomalies/financial -o 3_events/extracted_events/financial/ -n 3 > 3_events/logs/financial_extraction_train.txt &

nohup python all_tasks.py -s test -c configs/config_worldbank.json -t extract_events -i anomalies/worldbank -o 5_events/extracted_events/worldbank/ -n 5 > 5_events/logs/worldbank_extraction_test_temp.txt &
nohup python all_tasks.py -s test -c configs/config_worldbank.json -t extract_events -i anomalies/worldbank -o 3_events/extracted_events/worldbank/ -n 3 > 3_events/logs/worldbank_extraction_test_temp.txt &
nohup python all_tasks.py -s train -c configs/config_worldbank.json -t extract_events -i anomalies/worldbank -o 5_events/extracted_events/worldbank/ -n 5 > 5_events/logs/worldbank_extraction_train_temp.txt &
nohup python all_tasks.py -s train -c configs/config_worldbank.json -t extract_events -i anomalies/worldbank -o 3_events/extracted_events/worldbank/ -n 3 > 3_events/logs/worldbank_extraction_train_temp.txt &
