nohup python model.py -c configs/config_financial.json -f 3_events/features_with_random_neg/financial/ -s 3_events/models/financial/logistic_regression/ > 3_events/logs/financial_model_train.txt &
nohup python model.py -c configs/config_financial.json -f 5_events/features_with_random_neg/financial/ -s 5_events/models/financial/logistic_regression/ > 5_events/logs/financial_model_train.txt &
nohup python model.py -c configs/config_worldbank.json -f 3_events/features_with_random_neg/worldbank/ -s 3_events/models/worldbank/logistic_regression/ > 3_events/logs/worldbank_model_train.txt &
nohup python model.py -c configs/config_worldbank.json -f 5_events/features_with_random_neg/worldbank/ -s 5_events/models/worldbank/logistic_regression/ > 5_events/logs/worldbank_model_train.txt &

nohup python model.py -m nn -c configs/config_financial.json -f 3_events/features_with_random_neg/financial/ -s 3_events/models/financial/nn2layer/ > 3_events/logs/financial_model_train_nn.txt &
nohup python model.py -m nn -c configs/config_financial.json -f 5_events/features_with_random_neg/financial/ -s 5_events/models/financial/nn2layer/ > 5_events/logs/financial_model_train_nn.txt &
nohup python model.py -m nn -c configs/config_worldbank.json -f 3_events/features_with_random_neg/worldbank/ -s 3_events/models/worldbank/nn2layer/ > 3_events/logs/worldbank_model_train_nn.txt &
nohup python model.py -m nn -c configs/config_worldbank.json -f 5_events/features_with_random_neg/worldbank/ -s 5_events/models/worldbank/nn2layer/ > 5_events/logs/worldbank_model_train_nn.txt &
