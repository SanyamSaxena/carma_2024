nohup python compare_gpt_human.py -c configs/config_financial.json -g 3_events/gpt4_supervised_events/financial/test -t 3_events/human_ranked_events/financial/test > 3_events/results/compare_financial.txt &
nohup python compare_gpt_human.py -c configs/config_financial.json -g 5_events/gpt4_supervised_events/financial/test -t 5_events/human_ranked_events/financial/test > 5_events/results/compare_financial.txt &
nohup python compare_gpt_human.py -c configs/config_worldbank.json -g 3_events/gpt4_supervised_events/worldbank/test -t 3_events/human_ranked_events/worldbank/test > 3_events/results/compare_worldbank.txt &
nohup python compare_gpt_human.py -c configs/config_worldbank.json -g 5_events/gpt4_supervised_events/worldbank/test -t 5_events/human_ranked_events/worldbank/test > 5_events/results/compare_worldbank.txt &
