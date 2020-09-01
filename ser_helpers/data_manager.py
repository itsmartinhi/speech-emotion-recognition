import os
import pandas as pd


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

    # todo: mix and match function

    def add_augmentation(self, df, aug_df):


        return feature_df
