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


for proportion in "${proportions[@]}"; do
    for model in "${model_types[@]}"; do
        for k in "${num_ks[@]}"; do
            for feat in "${features[@]}"; do
                for dat in "${datasets[@]}"; do
                    echo -n "$k" "$feat" "$proportion" "$model" "$dat" ""
                    for seed in "${seeds[@]}"; do
                        python result_analysis.py -t trained --type-model "$model" -c configs/config_"$dat".json -i "$k"_events/scoring_model/"$feat"/"$dat"/"$model"/"$seed"/"$proportion"/test -g "$k"_events/human_ranked_events/"$dat"/test -l human --hide
                    done
                    echo ""
                done
            done
        done
    done
done
