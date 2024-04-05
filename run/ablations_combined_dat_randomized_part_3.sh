#!/bin/bash

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

# Evaluations
for proportion in "${proportions[@]}"; do
    for model in "${model_types[@]}"; do
        for k in "${num_ks[@]}"; do
            for feat in "${features[@]}"; do
                for dat in "${datasets[@]}"; do
                    echo -n "$k" "$feat" "$proportion" "$model" "$dat" ""
                    for seed in "${seeds[@]}"; do
                        python scoring.py -c configs/config_"$dat".json -t trained -p "$k"_events/models/"$feat"/"$model"/"$seed"/"$proportion" -m "$model" -i "$k"_events/features_with_random_neg/"$feat"/"$dat" -a anomaly_scores/"$dat"_stl/ -o "$k"_events/scoring_model/"$feat"/"$dat"/"$model"/"$seed"/"$proportion" -f "$feat" > debug/dump_scoring_model.txt &
                    done
                    echo ""
                    echo "waiting"
                    wait
                done
            done
        done
    done
done
wait