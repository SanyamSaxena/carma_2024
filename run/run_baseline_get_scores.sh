nohup python baseline.py -s test -c configs/config_financial.json -g 0  -t get_scores -m "prompt" -i 3_events/extracted_samples/financial -o 3_events/selfchecked_events/financial > 3_events/logs/financial_get_scores_prompt.txt &
nohup python baseline.py -s test -c configs/config_financial.json -g 0  -t get_scores -m "prompt" -i 5_events/extracted_samples/financial -o 5_events/selfchecked_events/financial > 5_events/logs/financial_get_scores_prompt.txt &
nohup python baseline.py -s test -c configs/config_worldbank.json -g 0  -t get_scores -m "prompt" -i 3_events/extracted_samples/worldbank -o 3_events/selfchecked_events/worldbank > 3_events/logs/worldbank_get_scores_prompt.txt &
nohup python baseline.py -s test -c configs/config_worldbank.json -g 0  -t get_scores -m "prompt" -i 5_events/extracted_samples/worldbank -o 5_events/selfchecked_events/worldbank > 5_events/logs/worldbank_get_scores_prompt.txt &

# nohup python baseline.py -s test -c configs/config_financial.json -g 0 -t get_scores -m "nli" -i 3_events/extracted_samples/financial -o 3_events/selfchecked_events/financial > 3_events/logs/financial_get_scores_nli.txt &
# nohup python baseline.py -s test -c configs/config_financial.json -g 0 -t get_scores -m "nli" -i 5_events/extracted_samples/financial -o 5_events/selfchecked_events/financial > 5_events/logs/financial_get_scores_nli.txt &
# nohup python baseline.py -s test -c configs/config_worldbank.json -g 0 -t get_scores -m "nli" -i 3_events/extracted_samples/worldbank -o 3_events/selfchecked_events/worldbank > 3_events/logs/worldbank_get_scores_nli.txt &
# nohup python baseline.py -s test -c configs/config_worldbank.json -g 0 -t get_scores -m "nli" -i 5_events/extracted_samples/worldbank -o 5_events/selfchecked_events/worldbank > 5_events/logs/worldbank_get_scores_nli.txt &

# nohup python baseline.py -s test -c configs/config_financial.json -g 4 -t get_scores -m "ngram" -i 3_events/extracted_samples/financial -o 3_events/selfchecked_events/financial > 3_events/logs/financial_get_scores_ngram.txt &
# nohup python baseline.py -s test -c configs/config_financial.json -g 4 -t get_scores -m "ngram" -i 5_events/extracted_samples/financial -o 5_events/selfchecked_events/financial > 5_events/logs/financial_get_scores_ngram.txt &
# nohup python baseline.py -s test -c configs/config_worldbank.json -g 4 -t get_scores -m "ngram" -i 3_events/extracted_samples/worldbank -o 3_events/selfchecked_events/worldbank > 3_events/logs/worldbank_get_scores_ngram.txt &
# nohup python baseline.py -s test -c configs/config_worldbank.json -g 4 -t get_scores -m "ngram" -i 5_events/extracted_samples/worldbank -o 5_events/selfchecked_events/worldbank > 5_events/logs/worldbank_get_scores_ngram.txt &








# nohup python baseline.py -s test -c configs/config_financial.json -g 6 -t get_scores -m "mqag" -i 3_events/extracted_samples/financial/seg_1 -o 3_events/selfchecked_events/financial/seg_1 > 3_events/logs/financial_get_scores_mqag_seg_1.txt &
# nohup python baseline.py -s test -c configs/config_financial.json -g 6 -t get_scores -m "mqag" -i 5_events/extracted_samples/financial/seg_1 -o 5_events/selfchecked_events/financial/seg_1 > 5_events/logs/financial_get_scores_mqag_seg_1.txt &
# nohup python baseline.py -s test -c configs/config_financial.json -g 6 -t get_scores -m "mqag" -i 3_events/extracted_samples/financial/seg_2 -o 3_events/selfchecked_events/financial/seg_2 > 3_events/logs/financial_get_scores_mqag_seg_2.txt &
# nohup python baseline.py -s test -c configs/config_financial.json -g 6 -t get_scores -m "mqag" -i 5_events/extracted_samples/financial/seg_2 -o 5_events/selfchecked_events/financial/seg_2 > 5_events/logs/financial_get_scores_mqag_seg_2.txt &
# nohup python baseline.py -s test -c configs/config_financial.json -g 6 -t get_scores -m "mqag" -i 3_events/extracted_samples/financial/seg_3 -o 3_events/selfchecked_events/financial/seg_3 > 3_events/logs/financial_get_scores_mqag_seg_3.txt &
# nohup python baseline.py -s test -c configs/config_financial.json -g 6 -t get_scores -m "mqag" -i 5_events/extracted_samples/financial/seg_3 -o 5_events/selfchecked_events/financial/seg_3 > 5_events/logs/financial_get_scores_mqag_seg_3.txt &
# nohup python baseline.py -s test -c configs/config_worldbank.json -g 6 -t get_scores -m "mqag" -i 3_events/extracted_samples/worldbank/seg_1 -o 3_events/selfchecked_events/worldbank/seg_1 > 3_events/logs/worldbank_get_scores_mqag_seg_1.txt &
# nohup python baseline.py -s test -c configs/config_worldbank.json -g 6 -t get_scores -m "mqag" -i 5_events/extracted_samples/worldbank/seg_1 -o 5_events/selfchecked_events/worldbank/seg_1 > 5_events/logs/worldbank_get_scores_mqag_seg_1.txt &
# nohup python baseline.py -s test -c configs/config_worldbank.json -g 6 -t get_scores -m "mqag" -i 3_events/extracted_samples/worldbank/seg_2 -o 3_events/selfchecked_events/worldbank/seg_2 > 3_events/logs/worldbank_get_scores_mqag_seg_2.txt &
# nohup python baseline.py -s test -c configs/config_worldbank.json -g 6 -t get_scores -m "mqag" -i 5_events/extracted_samples/worldbank/seg_2 -o 5_events/selfchecked_events/worldbank/seg_2 > 5_events/logs/worldbank_get_scores_mqag_seg_2.txt &
# nohup python baseline.py -s test -c configs/config_worldbank.json -g 6 -t get_scores -m "mqag" -i 3_events/extracted_samples/worldbank/seg_3 -o 3_events/selfchecked_events/worldbank/seg_3 > 3_events/logs/worldbank_get_scores_mqag_seg_3.txt &
# nohup python baseline.py -s test -c configs/config_worldbank.json -g 6 -t get_scores -m "mqag" -i 5_events/extracted_samples/worldbank/seg_3 -o 5_events/selfchecked_events/worldbank/seg_3 > 5_events/logs/worldbank_get_scores_mqag_seg_3.txt &

# nohup python baseline.py -s test -c configs/config_financial.json -g 5 -t get_scores -m "bertscore" -i 3_events/extracted_samples/financial/seg_1 -o 3_events/selfchecked_events/financial/seg_1 > 3_events/logs/financial_get_scores_bertscore_seg_1.txt &
# nohup python baseline.py -s test -c configs/config_financial.json -g 5 -t get_scores -m "bertscore" -i 5_events/extracted_samples/financial/seg_1 -o 5_events/selfchecked_events/financial/seg_1 > 5_events/logs/financial_get_scores_bertscore_seg_1.txt &
# nohup python baseline.py -s test -c configs/config_financial.json -g 5 -t get_scores -m "bertscore" -i 3_events/extracted_samples/financial/seg_2 -o 3_events/selfchecked_events/financial/seg_2 > 3_events/logs/financial_get_scores_bertscore_seg_2.txt &
# nohup python baseline.py -s test -c configs/config_financial.json -g 5 -t get_scores -m "bertscore" -i 5_events/extracted_samples/financial/seg_2 -o 5_events/selfchecked_events/financial/seg_2 > 5_events/logs/financial_get_scores_bertscore_seg_2.txt &
# nohup python baseline.py -s test -c configs/config_financial.json -g 5 -t get_scores -m "bertscore" -i 3_events/extracted_samples/financial/seg_3 -o 3_events/selfchecked_events/financial/seg_3 > 3_events/logs/financial_get_scores_bertscore_seg_3.txt &
# nohup python baseline.py -s test -c configs/config_financial.json -g 5 -t get_scores -m "bertscore" -i 5_events/extracted_samples/financial/seg_3 -o 5_events/selfchecked_events/financial/seg_3 > 5_events/logs/financial_get_scores_bertscore_seg_3.txt &
# nohup python baseline.py -s test -c configs/config_worldbank.json -g 5 -t get_scores -m "bertscore" -i 3_events/extracted_samples/worldbank/seg_1 -o 3_events/selfchecked_events/worldbank/seg_1 > 3_events/logs/worldbank_get_scores_bertscore_seg_1.txt &
# nohup python baseline.py -s test -c configs/config_worldbank.json -g 5 -t get_scores -m "bertscore" -i 5_events/extracted_samples/worldbank/seg_1 -o 5_events/selfchecked_events/worldbank/seg_1 > 5_events/logs/worldbank_get_scores_bertscore_seg_1.txt &
# nohup python baseline.py -s test -c configs/config_worldbank.json -g 5 -t get_scores -m "bertscore" -i 3_events/extracted_samples/worldbank/seg_2 -o 3_events/selfchecked_events/worldbank/seg_2 > 3_events/logs/worldbank_get_scores_bertscore_seg_2.txt &
# nohup python baseline.py -s test -c configs/config_worldbank.json -g 5 -t get_scores -m "bertscore" -i 5_events/extracted_samples/worldbank/seg_2 -o 5_events/selfchecked_events/worldbank/seg_2 > 5_events/logs/worldbank_get_scores_bertscore_seg_2.txt &
# nohup python baseline.py -s test -c configs/config_worldbank.json -g 5 -t get_scores -m "bertscore" -i 3_events/extracted_samples/worldbank/seg_3 -o 3_events/selfchecked_events/worldbank/seg_3 > 3_events/logs/worldbank_get_scores_bertscore_seg_3.txt &
# nohup python baseline.py -s test -c configs/config_worldbank.json -g 5 -t get_scores -m "bertscore" -i 5_events/extracted_samples/worldbank/seg_3 -o 5_events/selfchecked_events/worldbank/seg_3 > 5_events/logs/worldbank_get_scores_bertscore_seg_3.txt &
