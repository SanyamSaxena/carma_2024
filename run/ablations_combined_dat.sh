#!/bin/bash

# datasets=("financial" "worldbank")
# num_ks=("3" "5")
datasets=("financial")
num_ks=("3")
# proportions=("0.001" "0.002" "0.003" "0.004" "0.005" "0.0075" "0.01" "0.015" "0.02" "0.025" "0.03" "0.035")
proportions=("0.001" "0.005" "0.01" "0.05" "0.1" "0.2" "0.3" "0.4" "0.5" "0.75" "1")
# proportions=("0.00225" "0.0025" "0.00275")
# proportions=("1")

# features=("drop_none" "drop_contradiction" "drop_pattern" "drop_time" "drop_consensus")
# features=("drop_contradiction" "drop_pattern" "drop_time" "drop_consensus")
features=("drop_none")
# model_types=("naive_bayes" "bayes_with_prior" "logistic" "nn")
model_types=("nn")
# model_types=("bayes_with_prior")
# noise_levels=("000" "025" "050" "075" "100")
noise_levels=("050")

for proportion in "${proportions[@]}"; do
    for k in "${num_ks[@]}"; do
        for feat in "${features[@]}"; do
            for dat in "${datasets[@]}"; do
                python create_dataset.py -c configs/config_"$dat".json -t get_features_random_events -i "$k"_events/events_with_scored_random_negative/"$dat"/ -o "$k"_events/features_with_random_neg/"$dat" -a anomaly_scores/financial_stl/ -f "$feat" >> debug/dump_features.txt
            done
            for model in "${model_types[@]}"; do
                if [[ "$model" == "naive_bayes" || "$model" == "bayes_with_prior" ]]; then
                    python model.py -m "$model" -c configs/config_worldbank.json -f "$k"_events/features_with_random_neg/ -s "$k"_events/models/"$model"/  -l "$feat" -p "$proportion" >> debug/dump_model_train_"$proportion".txt
                    for dat in "${datasets[@]}"; do
                        python scoring.py -c configs/config_"$dat".json -t trained -p "$k"_events/models/"$model" -m "$model" -i "$k"_events/features_with_random_neg/"$dat"/ -a anomaly_scores/"$dat"_stl/ -o "$k"_events/scoring_model/"$dat" -f "$feat" >> debug/dump_scoring_model.txt
                        echo -n "$dat" "$k" "$proportion" "$feat" "$model" ""
                        python result_analysis.py -t trained --type-model "$model" -c configs/config_"$dat".json -i "$k"_events/scoring_model/"$dat"/test -g "$k"_events/human_ranked_events/"$dat"/test -l human --hide
                    done
                else
                    for noise_level in "${noise_levels[@]}"; do
                        python model.py -m "$model" -n "$noise_level" -c configs/config_"$dat".json -f "$k"_events/features_with_random_neg/ -s "$k"_events/models/"$model"/  -l "$feat" -p "$proportion" >> debug/dump_model_train_"$proportion".txt
                        for dat in "${datasets[@]}"; do
                            python scoring.py -c configs/config_"$dat".json -t trained -p "$k"_events/models/"$model" -m "$model" -i "$k"_events/features_with_random_neg/"$dat"/ -a anomaly_scores/"$dat"_stl/ -o "$k"_events/scoring_model/"$dat" -f "$feat" >> debug/dump_scoring_model.txt
                            echo -n "$dat" "$k" "$proportion" "$feat" "$model" "$noise_level" ""
                            python result_analysis.py -t trained --type-model "$model" -c configs/config_"$dat".json -i "$k"_events/scoring_model/"$dat"/test -g "$k"_events/human_ranked_events/"$dat"/test -l human --hide
                        done
                    done
                fi
            done
        done
    done
done