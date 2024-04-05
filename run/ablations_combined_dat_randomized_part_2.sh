#!/bin/bash

# datasets=("financial" "worldbank")
# num_ks=("3")
# seeds=("0" "10" "20" "30" "40")
# proportions=("0.001" "0.01" "0.05" "0.1" "0.2" "0.3" "0.4" "0.5" "0.6" "0.7" "0.8" "0.9" "1")

# features=("drop_none")

# # model_types=("naive_bayes" "bayes_with_prior" "logistic" "nn")

model_types=("naive_bayes")
datasets=("financial" "worldbank")
num_ks=("3")
# seeds=("0" "10" "20" "30" "40")
seeds=("50" "60" "70" "80" "90")
# proportions=("0.001" "0.01" "0.05" "0.1" "0.2")
proportions=("0.3" "0.4" "0.5" "0.6" "0.7" "0.8" "0.9" "1")
features=("drop_none")


for k in "${num_ks[@]}"; do
    for feat in "${features[@]}"; do
        for proportion in "${proportions[@]}"; do
            for model in "${model_types[@]}"; do
                for seed in "${seeds[@]}"; do
                    python model.py -l "$feat" -p "$proportion" --use-data both --seed "$seed" -m "$model" -c configs/config_worldbank.json -f "$k"_events/features_with_random_neg/"$feat"/ -s "$k"_events/models/"$feat"/"$model"/"$seed"/"$proportion" > debug/dump_"$model"_train_"$proportion"_"$seed".txt &
                done
                echo "waiting..."
                wait
            done
        done
    done
done
