nohup python all_tasks.py --explain no -s test -c configs/config_worldbank.json -t cross_examine_events -i 5_events/extracted_events/worldbank/ -a anomaly_scores/worldbank_lin/ -o 5_events/cross_examined_events/worldbank > 5_events/logs/worldbank_cross_examine_test.txt &
nohup python all_tasks.py --explain no -s test -c configs/config_worldbank.json -t cross_examine_events -i 3_events/extracted_events/worldbank/ -a anomaly_scores/worldbank_lin/ -o 3_events/cross_examined_events/worldbank > 3_events/logs/worldbank_cross_examine_test.txt &
nohup python all_tasks.py --explain no -s test -c configs/config_financial.json -t cross_examine_events -i 5_events/extracted_events/financial/ -a anomaly_scores/financial_lin/ -o 5_events/cross_examined_events/financial > 5_events/logs/financial_cross_examine_test.txt &
nohup python all_tasks.py --explain no -s test -c configs/config_financial.json -t cross_examine_events -i 3_events/extracted_events/financial/ -a anomaly_scores/financial_lin/ -o 3_events/cross_examined_events/financial > 3_events/logs/financial_cross_examine_test.txt &
nohup python all_tasks.py --explain no -s train -c configs/config_worldbank.json -t cross_examine_events -i 5_events/extracted_events/worldbank/ -a anomaly_scores/worldbank_lin/ -o 5_events/cross_examined_events/worldbank > 5_events/logs/worldbank_cross_examine_train.txt &
nohup python all_tasks.py --explain no -s train -c configs/config_worldbank.json -t cross_examine_events -i 3_events/extracted_events/worldbank/ -a anomaly_scores/worldbank_lin/ -o 3_events/cross_examined_events/worldbank > 3_events/logs/worldbank_cross_examine_train.txt &
nohup python all_tasks.py --explain no -s train -c configs/config_financial.json -t cross_examine_events -i 5_events/extracted_events/financial/ -a anomaly_scores/financial_lin/ -o 5_events/cross_examined_events/financial > 5_events/logs/financial_cross_examine_train.txt &
nohup python all_tasks.py --explain no -s train -c configs/config_financial.json -t cross_examine_events -i 3_events/extracted_events/financial/ -a anomaly_scores/financial_lin/ -o 3_events/cross_examined_events/financial > 3_events/logs/financial_cross_examine_train.txt &
nohup python all_tasks.py --explain yes -s test -c configs/config_worldbank.json -t cross_examine_events -i 5_events/extracted_events/worldbank/ -a anomaly_scores/worldbank_lin/ -o 5_events/cross_examined_events_explained/worldbank > 5_events/logs/worldbank_cross_examine_test_explained.txt &
nohup python all_tasks.py --explain yes -s test -c configs/config_worldbank.json -t cross_examine_events -i 3_events/extracted_events/worldbank/ -a anomaly_scores/worldbank_lin/ -o 3_events/cross_examined_events_explained/worldbank > 3_events/logs/worldbank_cross_examine_test_explained.txt &
nohup python all_tasks.py --explain yes -s test -c configs/config_financial.json -t cross_examine_events -i 5_events/extracted_events/financial/ -a anomaly_scores/financial_lin/ -o 5_events/cross_examined_events_explained/financial > 5_events/logs/financial_cross_examine_test_explained.txt &
nohup python all_tasks.py --explain yes -s test -c configs/config_financial.json -t cross_examine_events -i 3_events/extracted_events/financial/ -a anomaly_scores/financial_lin/ -o 3_events/cross_examined_events_explained/financial > 3_events/logs/financial_cross_examine_test_explained.txt &
nohup python all_tasks.py --explain yes -s train -c configs/config_worldbank.json -t cross_examine_events -i 5_events/extracted_events/worldbank/ -a anomaly_scores/worldbank_lin/ -o 5_events/cross_examined_events_explained/worldbank > 5_events/logs/worldbank_cross_examine_train_explained.txt &
nohup python all_tasks.py --explain yes -s train -c configs/config_worldbank.json -t cross_examine_events -i 3_events/extracted_events/worldbank/ -a anomaly_scores/worldbank_lin/ -o 3_events/cross_examined_events_explained/worldbank > 3_events/logs/worldbank_cross_examine_train_explained.txt &
nohup python all_tasks.py --explain yes -s train -c configs/config_financial.json -t cross_examine_events -i 5_events/extracted_events/financial/ -a anomaly_scores/financial_lin/ -o 5_events/cross_examined_events_explained/financial > 5_events/logs/financial_cross_examine_train_explained.txt &
nohup python all_tasks.py --explain yes -s train -c configs/config_financial.json -t cross_examine_events -i 3_events/extracted_events/financial/ -a anomaly_scores/financial_lin/ -o 3_events/cross_examined_events_explained/financial > 3_events/logs/financial_cross_examine_train_explained.txt &
