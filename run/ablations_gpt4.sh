#!/bin/bash

datasets=("financial" "worldbank")
num_ks=("3" "5")
# features=("drop_none" "drop_contradiction" "drop_pattern" "drop_time" "drop_consensus")
features=("drop_none")
# datasets=("worldbank")
# num_ks=("5")
# features=("drop_pattern" "drop_time" "drop_consensus")
# features=("drop_pattern" "drop_time" "drop_consensus")
model_types=("naive_bayes" "bayes_with_prior" "logistic" "nn")
# model_types=("naive_bayes")
noise_levels=("000" "025" "050" "075" "100")

for dat in "${datasets[@]}"; do
    for k in "${num_ks[@]}"; do
        for feat in "${features[@]}"; do
            python create_dataset.py -c configs/config_"$dat"_gpt4.json -t get_features_random_events -i "$k"_events/events_with_scored_random_negative_gpt4/"$dat"/ -o "$k"_events/features_with_random_neg_gpt4/"$dat" -a anomaly_scores/financial_stl/ -f "$feat" >> dump_features_gpt4.txt
            for model in "${model_types[@]}"; do
                if [[ "$model" == "naive_bayes" || "$model" == "bayes_with_prior" ]]; then
                    python model.py -m "$model" -c configs/config_"$dat"_gpt4.json -f "$k"_events/features_with_random_neg_gpt4/"$dat"/ -s "$k"_events/models_gpt4/"$dat"/"$model"/ -l "$feat" >> dump_model_train_gpt4.txt
                    python scoring.py -c configs/config_"$dat"_gpt4.json -t trained -p "$k"_events/models_gpt4/"$dat"/"$model" -m "$model" -i "$k"_events/features_with_random_neg_gpt4/"$dat"/ -a anomaly_scores/"$dat"_stl/ -o "$k"_events/scoring_model_gpt4/"$dat" -f "$feat" >> dump_scoring_model_gpt4.txt
                    echo -n "$dat" "$k" "$feat" "$model" ""
                    python result_analysis.py -t trained --type-model "$model" -c configs/config_"$dat"_gpt4.json -i "$k"_events/scoring_model_gpt4/"$dat"/test -g "$k"_events/human_ranked_events/"$dat"/test -l human --hide
                else
                    for noise_level in "${noise_levels[@]}"; do
                        python model.py -m "$model" -n "$noise_level" -c configs/config_"$dat"_gpt4.json -f "$k"_events/features_with_random_neg_gpt4/"$dat"/ -s "$k"_events/models_gpt4/"$dat"/"$model"/  -l "$feat" >> dump_model_train_gpt4.txt
                        python scoring.py -c configs/config_"$dat"_gpt4.json -t trained -p "$k"_events/models_gpt4/"$dat"/"$model" -m "$model" -i "$k"_events/features_with_random_neg_gpt4/"$dat"/ -a anomaly_scores/"$dat"_stl/ -o "$k"_events/scoring_model_gpt4/"$dat" -f "$feat" >> dump_scoring_model_gpt4.txt
                        echo -n "$dat" "$k" "$feat" "$model" "$noise_level" ""
                        python result_analysis.py -t trained --type-model "$model" -c configs/config_"$dat"_gpt4.json -i "$k"_events/scoring_model_gpt4/"$dat"/test -g "$k"_events/human_ranked_events/"$dat"/test -l human --hide
                    done
                fi
            done
        done
    done
done
