#!/bin/bash

datasets=("financial" "worldbank")
num_ks=("3" "5")
proportions=("0.1" "0.25" "0.5" "0.75" "1")
# proportions=("1")

# features=("drop_none" "drop_contradiction" "drop_pattern" "drop_time" "drop_consensus")
features=("drop_none")
model_types=("naive_bayes" "bayes_with_prior" "logistic" "nn")
# model_types=("naive_bayes")
# noise_levels=("000" "025" "050" "075" "100")
noise_levels=("050")

for proportion in "${proportions[@]}"; do
    for k in "${num_ks[@]}"; do
        for feat in "${features[@]}"; do
            for dat in "${datasets[@]}"; do
                python create_dataset.py -c configs/config_"$dat"_gpt4.json -t get_features_random_events -i "$k"_events/events_with_scored_random_negative_gpt4/"$dat"/ -o "$k"_events/features_with_random_neg_gpt4/"$dat" -a anomaly_scores/financial_stl/ -f "$feat" >> debug/dump_features_gpt4.txt
            done
            for model in "${model_types[@]}"; do
                if [[ "$model" == "naive_bayes" || "$model" == "bayes_with_prior" ]]; then
                    python model.py -m "$model" -c configs/config_worldbank_gpt4.json -f "$k"_events/features_with_random_neg_gpt4/ -s "$k"_events/models_gpt4/"$dat"/"$model"/  -l "$feat" -p "$proportion" >> debug/dump_model_train_"$proportion"_gpt4.txt
                    for dat in "${datasets[@]}"; do
                        python scoring.py -c configs/config_"$dat"_gpt4.json -t trained -p "$k"_events/models_gpt4/"$dat"/"$model" -m "$model" -i "$k"_events/features_with_random_neg_gpt4/"$dat"/ -a anomaly_scores/"$dat"_stl/ -o "$k"_events/scoring_model_gpt4/"$dat" -f "$feat" >> debug/dump_scoring_model_gpt4.txt
                        echo -n "$dat" "$k" "$proportion" "$feat" "$model" ""
                        python result_analysis.py -t trained --type-model "$model" -c configs/config_"$dat"_gpt4.json -i "$k"_events/scoring_model_gpt4/"$dat"/test -g "$k"_events/human_ranked_events/"$dat"/test -l human --hide
                    done
                else
                    for noise_level in "${noise_levels[@]}"; do
                        python model.py -m "$model" -n "$noise_level" -c configs/config_"$dat".json -f "$k"_events/features_with_random_neg_gpt4/ -s "$k"_events/models_gpt4/"$dat"/"$model"/  -l "$feat" -p "$proportion" >> debug/dump_model_train_"$proportion"_gpt4.txt
                        for dat in "${datasets[@]}"; do
                            python scoring.py -c configs/config_"$dat"_gpt4.json -t trained -p "$k"_events/models_gpt4/"$dat"/"$model" -m "$model" -i "$k"_events/features_with_random_neg_gpt4/"$dat"/ -a anomaly_scores/"$dat"_stl/ -o "$k"_events/scoring_model_gpt4/"$dat" -f "$feat" >> debug/dump_scoring_model_gpt4.txt
                            echo -n "$dat" "$k" "$proportion" "$feat" "$model" "$noise_level" ""
                            python result_analysis.py -t trained --type-model "$model" -c configs/config_"$dat"_gpt4.json -i "$k"_events/scoring_model_gpt4/"$dat"/test -g "$k"_events/human_ranked_events/"$dat"/test -l human --hide
                        done
                    done
                fi
            done
        done
    done
done