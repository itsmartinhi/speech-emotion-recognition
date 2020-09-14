import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


class DataManager:
    DATA_PATH = '../data/'
    RAW_DATA_PATH = DATA_PATH + 'Audio_Speech_Actors_01-24/'

    PATH_DF_NORMAL = DATA_PATH + 'df_normal.pkl'
    PATH_DF_NOISE = DATA_PATH + 'df_noise.pkl'
    PATH_DF_PITCH = DATA_PATH + 'df_pitch.pkl'
    PATH_DF_SHIFT = DATA_PATH + 'df_shift.pkl'
    PATH_DF_STRETCH = DATA_PATH + 'df_stretch.pkl'

    def save_df(self, path, df):
        print('\nSaving {} file...'.format(path))
        df.to_pickle(path)
        print("File size: {} MB".format(os.path.getsize(path) / (1024 ** 2)))

    def load_df(self, path):
        return pd.read_pickle(path)

    def get_path_to_aug_pkl(self, aug_name):
        return self.DATA_PATH + 'df_{}.pkl'.format(aug_name)

    def get_augmented_data(self, augmentations=[]):

        df_normal = self.load_df(self.PATH_DF_NORMAL)

        df_train, df_test = train_test_split(df_normal, test_size=0.2, random_state=42, shuffle=True)

        indices = df_train.index.values.tolist()

        for augmentation in augmentations:
            df_aug = self.load_df(self.get_path_to_aug_pkl(augmentation))
            df_aug = df_aug.iloc[indices]
            df_train = pd.concat([df_train, df_aug])
            df_train.reset_index(drop=True, inplace=True)

        X_train = np.stack(df_train['mfcc'].to_numpy())
        y_train = np.stack(df_train['emotion'].to_numpy())

        X_test = np.stack(df_test['mfcc'].to_numpy())
        y_test = np.stack(df_test['emotion'].to_numpy())

        print(len(df_train))
        print(len(df_test))

        return X_train, y_train, X_test, y_test
