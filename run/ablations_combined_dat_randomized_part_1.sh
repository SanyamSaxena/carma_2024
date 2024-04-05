#!/bin/bash

# # datasets=("financial" "worldbank")
# datasets=("financial" "worldbank")
# num_ks=("3" "5")
# seeds=("0" "10" "20" "30" "40" "50" "60" "70" "80" "90")
# proportions=("0.001" "0.005" "0.01" "0.05" "0.1" "0.2" "0.3" "0.4" "0.5" "0.6" "0.7" "0.8" "0.9" "1")

# features=("drop_none" "drop_contradiction" "drop_pattern" "drop_time" "drop_consensus")

# # model_types=("naive_bayes" "bayes_with_prior" "logistic" "nn")
# model_types=("naive_bayes" "logistic" "nn")

model_types=("naive_bayes")
datasets=("financial" "worldbank")
num_ks=("3")
seeds=("0" "10" "20" "30" "40" "50" "60" "70" "80" "90")
proportions=("0.001" "0.01" "0.05" "0.1" "0.2" "0.3" "0.4" "0.5" "0.6" "0.7" "0.8" "0.9" "1")
features=("drop_none")

for k in "${num_ks[@]}"; do
    for feat in "${features[@]}"; do
        for dat in "${datasets[@]}"; do
            python create_dataset.py -c configs/config_"$dat".json -t get_features_random_events -i "$k"_events/events_with_scored_random_negative/"$dat"/ -o "$k"_events/features_with_random_neg/"$feat"/"$dat" -a anomaly_scores/financial_stl/ -f "$feat" > debug/dump_features_"$feat"_"$dat".txt &
        done
    done
done

wait