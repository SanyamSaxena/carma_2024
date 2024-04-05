import torch
import torch.nn as nn
import torch.optim as optim
import pickle
import numpy as np
import argparse
import glob
import utils
import os
from models import LogisticRegression, NN2Layer, GeneralizedCrossEntropyLoss, CustomNaiveBayesClassifier
from sklearn.model_selection import train_test_split

data_split_info = {
   "companies_test" : ['Clean Energy Fuels Corp.', 'Comcast Corporation', 'Gilead Sciences Inc.', 'Lockheed Martin Corporation', 'Microsoft Corporation', 'NVIDIA Corporation', 'PayPal Holdings Inc.', 'Procter & Gamble Company', 'Regeneron Pharmaceuticals Inc.', 'Square, Inc.', 'SunPower Corporation', 'T-Mobile US, Inc.', 'Tesla, Inc.', 'United Parcel Service, Inc.'],
   "companies_val" : ['Alphabet Inc.', 'American Express Company', 'Biogen Inc.', 'Boeing Company', 'Charter Communications, Inc.', 'Marathon Petroleum Corporation', 'The Coca-Cola Company'],
   "companies_train" : ['Amazon.com Inc.', 'Amgen Inc.', 'Apple Inc.', 'Caterpillar Inc.', 'FedEx Corporation', 'JPMorgan Chase & Co.', 'Meta Platforms, Inc.', 'Netflix Inc.', 'Nike, Inc.', 'PepsiCo, Inc.', 'Plug Power Inc.', 'Renewable Energy Group, Inc.', 'The Goldman Sachs Group, Inc.', 'Vertex Pharmaceuticals Incorporated'],
   "countries_test" : ['Brazil', 'China', 'Congo, Dem. Rep.', 'Iran, Islamic Rep.', 'Peru', 'Saudi Arabia', 'Sudan', 'United States'],
   "countries_val" : ['Argentina', 'Indonesia', 'Libya', 'Russian Federation'],
   "countries_train" : ['Algeria', 'Australia', 'Canada', 'Greenland', 'India', 'Kazakhstan', 'Mexico', 'Mongolia'],
}


def get_torch_dataset_binary(feature_path, split, config_data, dataset, proportion=1, type_data="hard"):
    companies_train = data_split_info["companies_train"]
    countries_train = data_split_info["countries_train"]
    companies_val = data_split_info["companies_val"]
    countries_val = data_split_info["countries_val"]

    X = []
    Y = []
    split_load = split
    if split=="val":
        split_load = "train"
    files_pattern = f'{feature_path}/{split_load}/*.p'
    file_paths = glob.glob(files_pattern)
    file_paths.sort()
    for file_path in file_paths:
        if(split=="train"):
            file_name = (file_path.split('/')[-1]).split('.p')[0]
            if(dataset=="worldbank"):
                place = file_name.strip().split("_")[0]
                if place not in countries_train:
                    continue
            elif(dataset=="financial"):
                company_name = file_name
                if company_name not in companies_train:
                    continue
        elif(split=="val"):
            file_name = (file_path.split('/')[-1]).split('.p')[0]
            if(dataset=="worldbank"):
                place = file_name.strip().split("_")[0]
                if place not in countries_val:
                    continue
            elif(dataset=="financial"):
                company_name = file_name
                if company_name not in companies_val:
                    continue

        file = open(file_path,'rb')
        all_anomalies = pickle.load(file)
        for anomaly in all_anomalies:
            anomaly_trend = anomaly['trend']

            for event in anomaly[f'events_{anomaly_trend}']:
                X.append(event['feature'])
                Y.append(1)

            inverse_trend = config_data["trends"][anomaly_trend]
            for event in anomaly[f'events_{inverse_trend}']:
                X.append(event['feature'])
                Y.append(0)

            count = 0
            for event in anomaly[f'random negative events']:
                X.append(event['feature'])
                Y.append(0)
                count+=1
                if("5" in feature_path):
                    if count==1:
                        break
                elif "3" in feature_path:
                    if count==1:
                        break

    X_np = np.array(X)
    Y_np = np.array(Y)
    if(X_np.shape[0]==0):
        print(file_paths, files_pattern)
    print(X_np.shape, X_np.shape[0], (Y_np==1).sum(), (Y_np==0).sum())
    return X_np, Y_np


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config-data", type=str)
    parser.add_argument("-f", "--feature-path", type=str)
    parser.add_argument("-l", "--feature-level", default="drop_none", type=str)  
    parser.add_argument("-p", "--proportion", default=1, type=float)  
    parser.add_argument("-s", "--save-folder", type=str)
    parser.add_argument("-t", "--type", default="binary", type=str)
    parser.add_argument("-m", "--type-model", default="logistic", type=str) # only if trained
    parser.add_argument("-n", "--noise-level", default="050", type=str)
    parser.add_argument("-u", "--use-data", default="both", type=str)
    parser.add_argument("--seed", default=0, type=int)
    
    args = parser.parse_args()

    if args.use_data=="both":
        if not os.path.exists(args.save_folder):
          os.makedirs(args.save_folder)
    else:
        if not os.path.exists(f"{args.save_folder}/{args.use_data}"):
          os.makedirs(f"{args.save_folder}/{args.use_data}")

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True

    config_data = utils.read_config(args.config_data)
    if(args.use_data=="both"):
        feature_path = f"{args.feature_path}/worldbank"
        X_train_world, y_train_world = get_torch_dataset_binary(feature_path, 'train', config_data, dataset="worldbank", proportion=args.proportion, type_data="hard")
        X_val_world, y_val_world = get_torch_dataset_binary(feature_path, 'val', config_data, dataset="worldbank", type_data="hard")

        feature_path = f"{args.feature_path}/financial"
        X_train_fin, y_train_fin = get_torch_dataset_binary(feature_path, 'train', config_data, dataset="financial", proportion=args.proportion, type_data="hard")
        X_val_fin, y_val_fin = get_torch_dataset_binary(feature_path, 'val', config_data, dataset="financial", type_data="hard")
    
        X_train = np.concatenate((X_train_world, X_train_fin), axis=0)
        y_train = np.concatenate((y_train_world, y_train_fin), axis=0)
        X_val = np.concatenate((X_val_world, X_val_fin), axis=0)
        y_val = np.concatenate((y_val_world, y_val_fin), axis=0)

    elif(args.use_data=="financial"):
        feature_path = f"{args.feature_path}/financial"
        X_train, y_train = get_torch_dataset_binary(feature_path, 'train', config_data, dataset="financial", proportion=args.proportion, type_data="hard")
        X_val, y_val = get_torch_dataset_binary(feature_path, 'val', config_data, dataset="financial", type_data="hard")

    elif(args.use_data=="worldbank"):
        feature_path = f"{args.feature_path}/worldbank"
        X_train, y_train = get_torch_dataset_binary(feature_path, 'train', config_data, dataset="worldbank", proportion=args.proportion, type_data="hard")
        X_val, y_val = get_torch_dataset_binary(feature_path, 'val', config_data, dataset="worldbank", type_data="hard")

    if args.proportion!=1:
        np.random.seed(args.seed)
        _, X_train, _, y_train = train_test_split(X_train, y_train, test_size=args.proportion)
    else:
        np.random.seed(0)

    print(args.proportion, X_train.shape[0], y_train.shape[0],np.sum(y_train == 1), np.sum(y_train == 0))

    X_train = torch.FloatTensor(X_train)
    y_train = torch.LongTensor(y_train)
    X_val = torch.FloatTensor(X_val)
    y_val = torch.LongTensor(y_val)

    num_samples = y_train.shape[0]
    strength_prior = 1
    input_size = X_train.shape[1]
    if args.type_model == "naive_bayes" or args.type_model == "bayes_with_prior":
        boolean_feature_indices = []
        three_cat_feature_indices = []
        continuous_feature_indices = []
        boolean_priors = []
        three_cat_priors = []
        continuous_priors = []
        running_index = 0
        if(args.feature_level!="drop_contradiction"):
            boolean_feature_indices.extend([running_index,running_index+1])
            if args.type_model == "naive_bayes":
                boolean_priors.extend([
                    [[0, 0],[0, 0]],
                    [[0, 0],[0, 0]],
                    ])
            elif args.type_model == "bayes_with_prior":
                boolean_priors.extend([
                    [[1,1],[1, 1]],
                    [[1,1],[1, 1]],
                    ])
            running_index+=2
        if(args.feature_level!="drop_pattern"):
            three_cat_feature_indices.extend([running_index])
            if args.type_model == "naive_bayes":
                three_cat_priors.extend([
                    [[0,0,0],[0,0,0]]
                    ])
            elif args.type_model == "bayes_with_prior":
                three_cat_priors.extend([
                    [[1,1,1],[1,1,1]]
                    ])
            continuous_feature_indices.extend([running_index+1,running_index+2])
            if args.type_model == "naive_bayes":
                continuous_priors.extend([
                    [[0,0],[0,0]],
                    [[0,0],[0,0]]
                ])
            elif args.type_model == "bayes_with_prior":
                continuous_priors.extend([
                    [[0,1],[0.7,1]],
                    [[0,1],[0.7,1]]
                    ])
            running_index+=3
        if(args.feature_level!="drop_time"):
            continuous_feature_indices.extend([running_index])
            if args.type_model == "naive_bayes":
                continuous_priors.extend([
                    [[0,0],[0,0]]
                ])
            elif args.type_model == "bayes_with_prior":
                continuous_priors.extend([
                    [[0,1],[0.9,1]]
                ])
            running_index+=1
        if(args.feature_level!="drop_consensus"):
            continuous_feature_indices.extend([running_index])
            if args.type_model == "naive_bayes":
                continuous_priors.extend([
                    [[0,0],[0,0]]
                ])
            elif args.type_model == "bayes_with_prior":
                continuous_priors.extend([
                    [[0,1],[0,1]]
                ])
            running_index+=1

        model = CustomNaiveBayesClassifier(
            num_classes=2,
            boolean_feature_indices=boolean_feature_indices,
            three_cat_feature_indices=three_cat_feature_indices,
            continuous_feature_indices=continuous_feature_indices,
            boolean_priors=boolean_priors,
            three_cat_priors=three_cat_priors,
            continuous_priors=continuous_priors
        )
        model.fit(X_train, y_train)

        if args.use_data=="both":
            place_to_save = f"{args.save_folder}/best_model.p"
        else:
            place_to_save = f"{args.save_folder}/{args.use_data}/best_model.p"
        with open(place_to_save, 'wb') as file:
            pickle.dump(model, file)
        preds, _ = model.predict(X_val)
        preds = torch.tensor(preds, dtype=torch.int64)
        val_accuracy = (preds == y_val).sum().item() / len(y_val)
        print("Val Accuracy", val_accuracy)
    else:
        if(args.type_model=="logistic"):
            model = LogisticRegression(input_size)
        elif args.type_model=="nn":
            model = NN2Layer(input_size)
        if(args.noise_level=="100"):
            gamma_value = 1
            criterion = GeneralizedCrossEntropyLoss(gamma=gamma_value)
        elif(args.noise_level=="075"):
            gamma_value = 0.75
            criterion = GeneralizedCrossEntropyLoss(gamma=gamma_value)
        elif(args.noise_level=="050"):
            gamma_value = 0.5
            criterion = GeneralizedCrossEntropyLoss(gamma=gamma_value)
        elif(args.noise_level=="025"):
            gamma_value = 0.25
            criterion = GeneralizedCrossEntropyLoss(gamma=gamma_value)
        elif(args.noise_level=="000"):
            criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.1)
        best_val_accuracy = 0.0
        best_val_loss = 10000
        epochs = 100
        for epoch in range(epochs):
            model.train()
            optimizer.zero_grad()
            outputs = model(X_train)
            loss = criterion(outputs.squeeze(), y_train.float())
            loss.backward()
            optimizer.step()
            model.eval()
            with torch.no_grad():
                predictions = model(X_train)
                predictions = predictions.squeeze()
                train_loss = criterion(predictions.squeeze(), y_train.float())
                train_accuracy = ((predictions > 0.5).int() == y_train).sum().item() / len(y_train)
                print(f'Epoch: {epoch}, Train Loss: {train_loss.item()}, Train Accuracy: {train_accuracy}')
                predictions = model(X_val)
                predictions = predictions.squeeze()
                val_loss = criterion(predictions.squeeze(), y_val.float())
                val_accuracy = ((predictions > 0.5).int() == y_val).sum().item() / len(y_val)
                print(f'Epoch: {epoch}, Val Loss: {val_loss.item()}, Val Accuracy: {val_accuracy}')
            if val_accuracy > best_val_accuracy:
                print("Val Accuracy", X_train.shape, args.feature_path, args.feature_level, args.type_model, best_val_accuracy, args.save_folder)
                best_val_accuracy = val_accuracy
                if args.use_data=="both":
                    torch.save(model.state_dict(), f'{args.save_folder}/best_model.pth')
                else:
                    torch.save(model.state_dict(), f'{args.save_folder}/{args.use_data}/best_model.pth')
