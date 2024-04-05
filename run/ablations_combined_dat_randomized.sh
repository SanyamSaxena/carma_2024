#!/bin/bash

datasets=("financial" "worldbank")
num_ks=("3")
seeds=("0")
proportions=("1")

# features=("drop_none" "drop_contradiction" "drop_pattern" "drop_time" "drop_consensus")
features=("drop_none")

# model_types=("naive_bayes" "bayes_with_prior" "logistic" "nn")
# model_types=("naive_bayes" "logistic" "nn")
model_types=("naive_bayes")

for k in "${num_ks[@]}"; do
    for feat in "${features[@]}"; do
        for dat in "${datasets[@]}"; do
            python create_dataset.py -c configs/config_"$dat".json -t get_features_random_events -i "$k"_events/events_with_scored_random_negative/"$dat"/ -o "$k"_events/features_with_random_neg/"$feat"/"$dat" -a anomaly_scores/financial_stl/ -f "$feat" > debug/dump_features_"$feat"_"$dat".txt
        done
    done
done

for k in "${num_ks[@]}"; do
    for feat in "${features[@]}"; do
        for proportion in "${proportions[@]}"; do
            for model in "${model_types[@]}"; do
                for seed in "${seeds[@]}"; do
                    python model.py -l "$feat" -p "$proportion" --use-data both --seed "$seed" -m "$model" -c configs/config_worldbank.json -f "$k"_events/features_with_random_neg/"$feat"/ -s "$k"_events/models/"$feat"/"$model"/"$seed"/"$proportion" > debug/dump_"$model"_train_"$proportion"_"$seed".txt
                done
            done
        done
    done
done

# Evaluations
for proportion in "${proportions[@]}"; do
    for model in "${model_types[@]}"; do
        for k in "${num_ks[@]}"; do
            for feat in "${features[@]}"; do
                for dat in "${datasets[@]}"; do
                    echo -n "$k" "$feat" "$proportion" "$model" "$dat" ""
                    for seed in "${seeds[@]}"; do
                        python scoring.py -c configs/config_"$dat".json -t trained -p "$k"_events/models/"$feat"/"$model"/"$seed"/"$proportion" -m "$model" -i "$k"_events/features_with_random_neg/"$feat"/"$dat" -a anomaly_scores/"$dat"_stl/ -o "$k"_events/scoring_model/"$feat"/"$dat"/"$model"/"$seed"/"$proportion" -f "$feat" > debug/dump_scoring_model.txt
                        python result_analysis.py -t trained --type-model "$model" -c configs/config_"$dat".json -i "$k"_events/scoring_model/"$feat"/"$dat"/"$model"/"$seed"/"$proportion"/test -g "$k"_events/human_ranked_events/"$dat"/test -l human
                    done
                    echo ""
                done
            done
        done
    done
done
