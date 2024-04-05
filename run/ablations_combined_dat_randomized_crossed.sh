#!/bin/bash

datasets=("financial" "worldbank")
num_ks=("3" "5")
seeds=("0")
# seeds=("0")
# proportions=("0.001" "0.005" "0.01" "0.05" "0.1" "0.2" "0.3" "0.4" "0.5" "0.75" "1")
proportions=("1")

# features=("drop_none" "drop_contradiction" "drop_pattern" "drop_time" "drop_consensus")
features=("drop_none")

# model_types=("naive_bayes" "bayes_with_prior" "logistic" "nn")
model_types=("naive_bayes" "logistic" "nn")
# model_types=("logistic")

for k in "${num_ks[@]}"; do
    for feat in "${features[@]}"; do
        for dat in "${datasets[@]}"; do
            python create_dataset.py -c configs/config_"$dat".json -t get_features_random_events -i "$k"_events/events_with_scored_random_negative/"$dat"/ -o "$k"_events/features_with_random_neg/"$feat"/"$dat" -a anomaly_scores/financial_stl/ -f "$feat" > debug/dump_features_"$feat"_"$dat".txt
        done
    done
done

# train financial
for k in "${num_ks[@]}"; do
    for feat in "${features[@]}"; do
        for proportion in "${proportions[@]}"; do
            for model in "${model_types[@]}"; do
                for seed in "${seeds[@]}"; do
                    python model.py -l "$feat" -p "$proportion" --use-data financial --seed "$seed" -m "$model" -c configs/config_financial.json -f "$k"_events/features_with_random_neg/"$feat"/ -s "$k"_events/models/"$feat"/"$model"/"$seed"/"$proportion" > debug/dump_"$model"_train_"$proportion"_"$seed".txt
                done
            done
        done
    done
done

# evaluate on worldbank
for proportion in "${proportions[@]}"; do
    for model in "${model_types[@]}"; do
        for k in "${num_ks[@]}"; do
            for feat in "${features[@]}"; do
                echo -n "$k" "$feat" "$proportion" "$model" "worldbank" ""
                for seed in "${seeds[@]}"; do
                    python scoring.py -c configs/config_worldbank.json -t trained -p "$k"_events/models/"$feat"/"$model"/"$seed"/"$proportion"/financial -m "$model" -i "$k"_events/features_with_random_neg/"$feat"/worldbank -a anomaly_scores/worldbank_stl/ -o "$k"_events/scoring_model/"$feat"/worldbank_using_financial/"$model"/"$seed"/"$proportion" -f "$feat" > debug/dump_scoring_model.txt
                    python result_analysis.py -t trained --type-model "$model" -c configs/config_worldbank.json -i "$k"_events/scoring_model/"$feat"/worldbank_using_financial/"$model"/"$seed"/"$proportion"/test -g "$k"_events/human_ranked_events/worldbank/test -l human --hide
                done
                echo ""
            done
        done
    done
done


# train worldbank
for k in "${num_ks[@]}"; do
    for feat in "${features[@]}"; do
        for proportion in "${proportions[@]}"; do
            for model in "${model_types[@]}"; do
                for seed in "${seeds[@]}"; do
                    python model.py -l "$feat" -p "$proportion" --use-data worldbank --seed "$seed" -m "$model" -c configs/config_worldbank.json -f "$k"_events/features_with_random_neg/"$feat"/ -s "$k"_events/models/"$feat"/"$model"/"$seed"/"$proportion" > debug/dump_"$model"_train_"$proportion"_"$seed".txt
                done
            done
        done
    done
done

# evaluate on financial
for proportion in "${proportions[@]}"; do
    for model in "${model_types[@]}"; do
        for k in "${num_ks[@]}"; do
            for feat in "${features[@]}"; do
                echo -n "$k" "$feat" "$proportion" "$model" "financial" ""
                for seed in "${seeds[@]}"; do
                    python scoring.py -c configs/config_financial.json -t trained -p "$k"_events/models/"$feat"/"$model"/"$seed"/"$proportion"/worldbank -m "$model" -i "$k"_events/features_with_random_neg/"$feat"/financial -a anomaly_scores/financial_stl/ -o "$k"_events/scoring_model/"$feat"/financial_using_worldbank/"$model"/"$seed"/"$proportion" -f "$feat" > debug/dump_scoring_model.txt
                    python result_analysis.py -t trained --type-model "$model" -c configs/config_financial.json -i "$k"_events/scoring_model/"$feat"/financial_using_worldbank/"$model"/"$seed"/"$proportion"/test -g "$k"_events/human_ranked_events/financial/test -l human --hide
                done
                echo ""
            done
        done
    done
done
