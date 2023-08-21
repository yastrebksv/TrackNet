import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
import catboost as ctb
import argparse


def create_features(path_dataset, num_frames):
    games = os.listdir(path_dataset)
    games.remove('Readme.docx')
    df = pd.DataFrame()
    for game in tqdm(games):
        clips = os.listdir(os.path.join(path_dataset, game))
        for clip in clips:
            labels = pd.read_csv(os.path.join(path_dataset, game, clip, 'Label.csv'))

            eps = 1e-15
            for i in range(1, num_frames):
                labels['x_lag_{}'.format(i)] = labels['x-coordinate'].shift(i)
                labels['x_lag_inv_{}'.format(i)] = labels['x-coordinate'].shift(-i)
                labels['y_lag_{}'.format(i)] = labels['y-coordinate'].shift(i)
                labels['y_lag_inv_{}'.format(i)] = labels['y-coordinate'].shift(-i) 
                labels['x_diff_{}'.format(i)] = abs(labels['x_lag_{}'.format(i)] - labels['x-coordinate'])
                labels['y_diff_{}'.format(i)] = labels['y_lag_{}'.format(i)] - labels['y-coordinate']
                labels['x_diff_inv_{}'.format(i)] = abs(labels['x_lag_inv_{}'.format(i)] - labels['x-coordinate'])
                labels['y_diff_inv_{}'.format(i)] = labels['y_lag_inv_{}'.format(i)] - labels['y-coordinate']
                labels['x_div_{}'.format(i)] = abs(labels['x_diff_{}'.format(i)]/(labels['x_diff_inv_{}'.format(i)] + eps))
                labels['y_div_{}'.format(i)] = labels['y_diff_{}'.format(i)]/(labels['y_diff_inv_{}'.format(i)] + eps)

            labels['target'] = (labels['status'] == 2).astype(int)         
            for i in range(1, num_frames):    
                labels = labels[labels['x_lag_{}'.format(i)].notna()]
                labels = labels[labels['x_lag_inv_{}'.format(i)].notna()]
            labels = labels[labels['x-coordinate'].notna()]  

            labels['status'] = labels['status'].astype(int)
            df = df.append(labels)
    return df

def create_train_test(df, num_frames):
    colnames_x = ['x_diff_{}'.format(i) for i in range(1, num_frames)] + \
                 ['x_diff_inv_{}'.format(i) for i in range(1, num_frames)] + \
                 ['x_div_{}'.format(i) for i in range(1, num_frames)]
    colnames_y = ['y_diff_{}'.format(i) for i in range(1, num_frames)] + \
                 ['y_diff_inv_{}'.format(i) for i in range(1, num_frames)] + \
                 ['y_div_{}'.format(i) for i in range(1, num_frames)]
    colnames = colnames_x + colnames_y 
    df_train, df_test = train_test_split(df, test_size=0.25, random_state=5)
    X_train = df_train[colnames]
    X_test = df_test[colnames]
    y_train = df_train['target']
    y_test = df_test['target']
    return X_train, y_train, X_test, y_test


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_dataset', type=str, help='path to the TrackNet dataset')
    parser.add_argument('--path_save_model', type=str, help='path for saving model with .cbm format')
    args = parser.parse_args()    
    
    NUM_FEATURE_FRAMES = 3
    df_features = create_features(args.path_dataset, NUM_FEATURE_FRAMES)
    X_train, y_train, X_test, y_test = create_train_test(df_features, NUM_FEATURE_FRAMES)
    
    train_dataset = ctb.Pool(X_train, y_train)
    model_ctb = ctb.CatBoostRegressor(loss_function='RMSE')
    grid = {'iterations': [150, 200, 250],
            'learning_rate': [0.03, 0.1],
            'depth': [2, 4, 6],
            'l2_leaf_reg': [0.2, 0.5, 1, 3]}
    model_ctb.grid_search(grid, train_dataset)
    
    pred_ctb = model_ctb.predict(X_test)
    y_pred_bin = (pred_ctb > 0.5).astype(int)
    
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred_bin).ravel()
    print('tn = {}, fp = {}, fn = {}, tp = {}'.format(tn, fp, fn, tp))
    print('accuracy = {}'.format(accuracy_score(y_test, y_pred_bin)))
    
    model_ctb.save_model(args.path_save_model)

