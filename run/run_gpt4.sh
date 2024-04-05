nohup python all_tasks.py --explain no -s test -c configs/config_worldbank.json -t gpt4_supervise_events -i 5_events/extracted_events/worldbank/ -o 5_events/gpt4_supervised_events/worldbank > 5_events/logs/worldbank_gpt4_supervision_test.txt &
nohup python all_tasks.py --explain no -s test -c configs/config_worldbank.json -t gpt4_supervise_events -i 3_events/extracted_events/worldbank/ -o 3_events/gpt4_supervised_events/worldbank > 3_events/logs/worldbank_gpt4_supervision_test.txt &
nohup python all_tasks.py --explain no -s test -c configs/config_financial.json -t gpt4_supervise_events -i 5_events/extracted_events/financial/ -o 5_events/gpt4_supervised_events/financial > 5_events/logs/financial_gpt4_supervision_test.txt &
nohup python all_tasks.py --explain no -s test -c configs/config_financial.json -t gpt4_supervise_events -i 3_events/extracted_events/financial/ -o 3_events/gpt4_supervised_events/financial > 3_events/logs/financial_gpt4_supervision_test.txt &
nohup python all_tasks.py --explain no -s train -c configs/config_worldbank.json -t gpt4_supervise_events -i 5_events/extracted_events/worldbank/ -o 5_events/gpt4_supervised_events/worldbank > 5_events/logs/worldbank_gpt4_supervision_train.txt &
nohup python all_tasks.py --explain no -s train -c configs/config_worldbank.json -t gpt4_supervise_events -i 3_events/extracted_events/worldbank/ -o 3_events/gpt4_supervised_events/worldbank > 3_events/logs/worldbank_gpt4_supervision_train.txt &
nohup python all_tasks.py --explain no -s train -c configs/config_financial.json -t gpt4_supervise_events -i 5_events/extracted_events/financial/ -o 5_events/gpt4_supervised_events/financial > 5_events/logs/financial_gpt4_supervision_train.txt &
nohup python all_tasks.py --explain no -s train -c configs/config_financial.json -t gpt4_supervise_events -i 3_events/extracted_events/financial/ -o 3_events/gpt4_supervised_events/financial > 3_events/logs/financial_gpt4_supervision_train.txt &
nohup python all_tasks.py --explain yes -s test -c configs/config_worldbank.json -t gpt4_supervise_events -i 5_events/extracted_events/worldbank/ -o 5_events/gpt4_supervised_events_explained/worldbank > 5_events/logs/worldbank_gpt4_supervision_test_explained.txt &
nohup python all_tasks.py --explain yes -s test -c configs/config_worldbank.json -t gpt4_supervise_events -i 3_events/extracted_events/worldbank/ -o 3_events/gpt4_supervised_events_explained/worldbank > 3_events/logs/worldbank_gpt4_supervision_test_explained.txt &
nohup python all_tasks.py --explain yes -s test -c configs/config_financial.json -t gpt4_supervise_events -i 5_events/extracted_events/financial/ -o 5_events/gpt4_supervised_events_explained/financial > 5_events/logs/financial_gpt4_supervision_test_explained.txt &
nohup python all_tasks.py --explain yes -s test -c configs/config_financial.json -t gpt4_supervise_events -i 3_events/extracted_events/financial/ -o 3_events/gpt4_supervised_events_explained/financial > 3_events/logs/financial_gpt4_supervision_test_explained.txt &
nohup python all_tasks.py --explain yes -s train -c configs/config_worldbank.json -t gpt4_supervise_events -i 5_events/extracted_events/worldbank/ -o 5_events/gpt4_supervised_events_explained/worldbank > 5_events/logs/worldbank_gpt4_supervision_train_explained.txt &
nohup python all_tasks.py --explain yes -s train -c configs/config_worldbank.json -t gpt4_supervise_events -i 3_events/extracted_events/worldbank/ -o 3_events/gpt4_supervised_events_explained/worldbank > 3_events/logs/worldbank_gpt4_supervision_train_explained.txt &
nohup python all_tasks.py --explain yes -s train -c configs/config_financial.json -t gpt4_supervise_events -i 5_events/extracted_events/financial/ -o 5_events/gpt4_supervised_events_explained/financial > 5_events/logs/financial_gpt4_supervision_train_explained.txt &
nohup python all_tasks.py --explain yes -s train -c configs/config_financial.json -t gpt4_supervise_events -i 3_events/extracted_events/financial/ -o 3_events/gpt4_supervised_events_explained/financial > 3_events/logs/financial_gpt4_supervision_train_explained.txt &

