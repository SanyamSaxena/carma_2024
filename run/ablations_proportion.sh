#!/bin/bash

# datasets=("financial" "worldbank")
datasets=("worldbank")
num_ks=("3" "5")
proportions=("0.25" "0.5" "0.75" "1")
# num_ks=("3")
# features=("drop_none" "drop_contradiction" "drop_pattern" "drop_time" "drop_consensus")
features=("drop_none")
model_types=("naive_bayes" "bayes_with_prior" "logistic" "nn")
# model_types=("naive_bayes")
# noise_levels=("000" "025" "050" "075" "100")
noise_levels=("050")

for proportion in "${proportions[@]}"; do
    for dat in "${datasets[@]}"; do
        for k in "${num_ks[@]}"; do
            for feat in "${features[@]}"; do
                python create_dataset.py -c configs/config_"$dat".json -t get_features_random_events -i "$k"_events/events_with_scored_random_negative/"$dat"/ -o "$k"_events/features_with_random_neg/"$dat" -a anomaly_scores/financial_stl/ -f "$feat" >> dump_features.txt
                for model in "${model_types[@]}"; do
                    if [[ "$model" == "naive_bayes" || "$model" == "bayes_with_prior" ]]; then
                        python model.py -m "$model" -c configs/config_"$dat".json -f "$k"_events/features_with_random_neg/"$dat"/ -s "$k"_events/models/"$dat"/"$model"/  -l "$feat" -p "$proportion" >> dump_model_train_"$proportion".txt
                        python scoring.py -c configs/config_"$dat".json -t trained -p "$k"_events/models/"$dat"/"$model" -m "$model" -i "$k"_events/features_with_random_neg/"$dat"/ -a anomaly_scores/"$dat"_stl/ -o "$k"_events/scoring_model/"$dat" -f "$feat" >> dump_scoring_model.txt
                        echo -n "$dat" "$k" "$feat" "$proportion" "$model" ""
                        python result_analysis.py -t trained --type-model "$model" -c configs/config_"$dat".json -i "$k"_events/scoring_model/"$dat"/test -g "$k"_events/human_ranked_events/"$dat"/test -l human
                    else
                        for noise_level in "${noise_levels[@]}"; do
                            python model.py -m "$model" -n "$noise_level" -c configs/config_"$dat".json -f "$k"_events/features_with_random_neg/"$dat"/ -s "$k"_events/models/"$dat"/"$model"/  -l "$feat" -p "$proportion" >> dump_model_train_"$proportion".txt
                            python scoring.py -c configs/config_"$dat".json -t trained -p "$k"_events/models/"$dat"/"$model" -m "$model" -i "$k"_events/features_with_random_neg/"$dat"/ -a anomaly_scores/"$dat"_stl/ -o "$k"_events/scoring_model/"$dat" -f "$feat" >> dump_scoring_model.txt
                            echo -n "$dat" "$k" "$feat" "$model" "$proportion" "$noise_level" ""
                            python result_analysis.py -t trained --type-model "$model" -c configs/config_"$dat".json -i "$k"_events/scoring_model/"$dat"/test -g "$k"_events/human_ranked_events/"$dat"/test -l human
                        done
                    fi
                done
            done
        done
    done
done    