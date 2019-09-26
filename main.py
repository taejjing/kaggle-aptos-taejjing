from src.train import train
from src import config
from src.utils import seed_everything
import pandas as pd
import os
from sklearn.model_selection import train_test_split, StratifiedKFold
from datetime import datetime


if __name__ == '__main__':

    print("SEED number : %d" % config.SEED)
    seed_everything(config.SEED)
    now = datetime.now()
    now = f'{now.year}{now.month}{now.day}{now.hour}{now.minute}'

    # Define param
    nfold = 5
    model_name = 'b4'
    n_epochs = 25
    pretraining = True
#     weight_path = "input/efn_b4_nfNone_ep13_vl0.3339_vk0.7768_acc0.7882.pt"
    weight_path = None

    print("N Fold : {}, Model : EFN_{}, N_epochs : {}".format(nfold, model_name, n_epochs))

    # load Dataset
    # train_csv = pd.read_csv(os.path.join(config.DATA_PATH, 'prev_curr_train_v2.csv'))
    train_csv = pd.read_csv(os.path.join(config.DATA_PATH, 'prev_curr_train.csv'))
    # train_csv = pd.read_csv(os.path.join(config.DATA_PATH, 'prev_curr_train_v1_1.csv'))
    # train_csv = pd.read_csv(os.path.join(config.DATA_PATH, 'train.csv'))
    labels = train_csv['diagnosis'].values
   
    if pretraining == False:

        EXPERIMENT_NAME = "Plateau_onlyresize_addpre_v1_trans_v2"
        path = os.path.join(config.RES_PATH, f'{EXPERIMENT_NAME}_{model_name}-{nfold}fold-{now}')
        print(EXPERIMENT_NAME)

        if not os.path.exists(path):
            os.mkdir(path)    

        # Straitfied KFold
        skf = StratifiedKFold(n_splits=nfold, shuffle=True, random_state=config.SEED)    
        
        for i, (train_index, val_index) in enumerate(skf.split(labels, labels)):
            cur_fold = i+1

            print("Current fold : {}".format(cur_fold))
            
            train_df = train_csv.iloc[train_index, :]
            val_df = train_csv.iloc[val_index, :]

            print(train_df['diagnosis'].value_counts())
            print(val_df['diagnosis'].value_counts())
                
            train_df.reset_index(drop=True, inplace=True)
            val_df.reset_index(drop=True, inplace=True)
            train(train_df, val_df, path, weight_path, cur_fold, model_num=model_name[-1], n_epochs=n_epochs)

    elif pretraining == True:

        weight_path = None
        EXPERIMENT_NAME = "b4_384_Plateau_pretrain_only_resize_trans_v2"
        path = os.path.join(config.RES_PATH, f'{EXPERIMENT_NAME}_{model_name}-{nfold}fold-{now}')
        print(EXPERIMENT_NAME)
        
        if not os.path.exists(path):
            os.mkdir(path)    

        test_size = 0.1
        print("No fold, test_size : {}".format(test_size))
        train_csv = pd.read_csv(os.path.join(config.DATA_PATH, 'prev_train.csv'))
        train_csv.columns = ['id_code', 'diagnosis']
        labels = train_csv['diagnosis'].values

        train_df, val_df = train_test_split(train_csv, test_size=test_size, random_state=config.SEED, stratify=labels)    
        train_df.reset_index(drop=True, inplace=True)
        val_df.reset_index(drop=True, inplace=True)

        print(train_df['diagnosis'].value_counts())
        print(val_df['diagnosis'].value_counts())

        train(train_df, val_df, path, weight_path, model_num=model_name[-1], n_epochs=n_epochs)

    else:
        print("Exit..")

    fin = datetime.now()
    finish_time = f'{fin.year}{fin.month}{fin.day}{fin.hour}{fin.minute}'
    print("Finish time : {}".format(finish_time))