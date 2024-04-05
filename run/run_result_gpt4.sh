nohup python result_analysis.py -t manual -c configs/config_financial.json -s contradiction -i 3_events/scoring/financial/test -g 3_events/gpt4_supervised_events/financial/test -l gpt4 > 3_events/results/financial_contradiction.txt &
nohup python result_analysis.py -t manual -c configs/config_financial.json -s contradiction -i 5_events/scoring/financial/test -g 5_events/gpt4_supervised_events/financial/test -l gpt4 > 5_events/results/financial_contradiction.txt &
nohup python result_analysis.py -t manual -c configs/config_worldbank.json -s contradiction -i 3_events/scoring/worldbank/test -g 3_events/gpt4_supervised_events/worldbank/test -l gpt4 > 3_events/results/worldbank_contradiction.txt &
nohup python result_analysis.py -t manual -c configs/config_worldbank.json -s contradiction -i 5_events/scoring/worldbank/test -g 5_events/gpt4_supervised_events/worldbank/test -l gpt4 > 5_events/results/worldbank_contradiction.txt &

nohup python result_analysis.py -t manual -c configs/config_financial.json -s consensus_our_max -i 3_events/scoring/financial/test -g 3_events/gpt4_supervised_events/financial/test -l gpt4 > 3_events/results/financial_consensus_our_max.txt &
nohup python result_analysis.py -t manual -c configs/config_financial.json -s consensus_our_max -i 5_events/scoring/financial/test -g 5_events/gpt4_supervised_events/financial/test -l gpt4 > 5_events/results/financial_consensus_our_max.txt &
nohup python result_analysis.py -t manual -c configs/config_worldbank.json -s consensus_our_max -i 3_events/scoring/worldbank/test -g 3_events/gpt4_supervised_events/worldbank/test -l gpt4 > 3_events/results/worldbank_consensus_our_max.txt &
nohup python result_analysis.py -t manual -c configs/config_worldbank.json -s consensus_our_max -i 5_events/scoring/worldbank/test -g 5_events/gpt4_supervised_events/worldbank/test -l gpt4 > 5_events/results/worldbank_consensus_our_max.txt &

nohup python result_analysis.py -t manual -c configs/config_financial.json -s consensus_avg -i 3_events/scoring/financial/test -g 3_events/gpt4_supervised_events/financial/test -l gpt4 > 3_events/results/financial_consensus_avg.txt &
nohup python result_analysis.py -t manual -c configs/config_financial.json -s consensus_avg -i 5_events/scoring/financial/test -g 5_events/gpt4_supervised_events/financial/test -l gpt4 > 5_events/results/financial_consensus_avg.txt &
nohup python result_analysis.py -t manual -c configs/config_worldbank.json -s consensus_avg -i 3_events/scoring/worldbank/test -g 3_events/gpt4_supervised_events/worldbank/test -l gpt4 > 3_events/results/worldbank_consensus_avg.txt &
nohup python result_analysis.py -t manual -c configs/config_worldbank.json -s consensus_avg -i 5_events/scoring/worldbank/test -g 5_events/gpt4_supervised_events/worldbank/test -l gpt4 > 5_events/results/worldbank_consensus_avg.txt &

nohup python result_analysis.py -t manual -c configs/config_financial.json -s pattern -i 3_events/scoring/financial/test -g 3_events/gpt4_supervised_events/financial/test -l gpt4 > 3_events/results/financial_pattern.txt &
nohup python result_analysis.py -t manual -c configs/config_financial.json -s pattern -i 5_events/scoring/financial/test -g 5_events/gpt4_supervised_events/financial/test -l gpt4 > 5_events/results/financial_pattern.txt &
nohup python result_analysis.py -t manual -c configs/config_worldbank.json -s pattern -i 3_events/scoring/worldbank/test -g 3_events/gpt4_supervised_events/worldbank/test -l gpt4 > 3_events/results/worldbank_pattern.txt &
nohup python result_analysis.py -t manual -c configs/config_worldbank.json -s pattern -i 5_events/scoring/worldbank/test -g 5_events/gpt4_supervised_events/worldbank/test -l gpt4 > 5_events/results/worldbank_pattern.txt &

nohup python result_analysis.py -t manual -c configs/config_financial.json -s pattern_time -i 3_events/scoring/financial/test -g 3_events/gpt4_supervised_events/financial/test -l gpt4 > 3_events/results/financial_pattern_time.txt &
nohup python result_analysis.py -t manual -c configs/config_financial.json -s pattern_time -i 5_events/scoring/financial/test -g 5_events/gpt4_supervised_events/financial/test -l gpt4 > 5_events/results/financial_pattern_time.txt &
nohup python result_analysis.py -t manual -c configs/config_worldbank.json -s pattern_time -i 3_events/scoring/worldbank/test -g 3_events/gpt4_supervised_events/worldbank/test -l gpt4 > 3_events/results/worldbank_pattern_time.txt &
nohup python result_analysis.py -t manual -c configs/config_worldbank.json -s pattern_time -i 5_events/scoring/worldbank/test -g 5_events/gpt4_supervised_events/worldbank/test -l gpt4 > 5_events/results/worldbank_pattern_time.txt &

