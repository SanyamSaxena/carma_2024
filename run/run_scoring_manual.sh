nohup python scoring.py -c configs/config_worldbank.json -t manual -i 3_events/cross_examined_events/worldbank/ -a anomaly_scores/worldbank_stl/ -o 3_events/scoring/worldbank &
nohup python scoring.py -c configs/config_worldbank.json -t manual -i 5_events/cross_examined_events/worldbank/ -a anomaly_scores/worldbank_stl/ -o 5_events/scoring/worldbank &
nohup python scoring.py -c configs/config_financial.json -t manual -i 3_events/cross_examined_events/financial/ -a anomaly_scores/financial_stl/ -o 3_events/scoring/financial &
nohup python scoring.py -c configs/config_financial.json -t manual -i 5_events/cross_examined_events/financial/ -a anomaly_scores/financial_stl/ -o 5_events/scoring/financial &
# nohup python scoring.py -c configs/config_worldbank.json -t manual -i 3_events/cross_examined_events_explained/worldbank/ -a anomaly_scores/worldbank_lin/ -o 3_events/scoring_explained/worldbank &
# nohup python scoring.py -c configs/config_worldbank.json -t manual -i 5_events/cross_examined_events_explained/worldbank/ -a anomaly_scores/worldbank_lin/ -o 5_events/scoring_explained/worldbank &
# nohup python scoring.py -c configs/config_financial.json -t manual -i 3_events/cross_examined_events_explained/financial/ -a anomaly_scores/financial_lin/ -o 3_events/scoring_explained/financial &
# nohup python scoring.py -c configs/config_financial.json -t manual -i 5_events/cross_examined_events_explained/financial/ -a anomaly_scores/financial_lin/ -o 5_events/scoring_explained/financial &
