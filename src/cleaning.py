import numpy as np
import pandas as pd


class CleanData:
    @staticmethod
    def drop_columns(data, drop_cols):
        data.drop(drop_cols, axis='columns', inplace=True)
        return data

    @staticmethod
    def convert_rows_with_negative_values_to_positive_values(data, column_name='Caudal_formula'):
        dataset = data.copy()
        dataset_with_value_less_than_zero = dataset[dataset[column_name] < 0]
        if dataset_with_value_less_than_zero.shape[0] == 0:
            return dataset
        # dataset.loc[dataset[column_name] < 0, column_name] = 0
        dataset[column_name] = np.abs(dataset[column_name])
        return dataset

    @staticmethod
    def drop_rows_with_negative_or_zero_values_in_column(data, column_name='Caudal_formula'):
        dataset = data.copy()
        dataset_with_value_less_than_zero = dataset[dataset[column_name] <= 0]
        if dataset_with_value_less_than_zero.shape[0] == 0:
            return dataset
        return dataset[dataset[column_name] > 0]

    @staticmethod
    def get_dataset_for_test_set(data, test_size=185):
        dataset = data.copy()
        dataset_with_b_d = dataset[pd.notnull(dataset['D']) & pd.notnull(dataset['D']) & dataset['hasrq'] == 1]
        dataset_with_b_d = dataset_with_b_d.sample(n=test_size)
        print(f'dataset_with_b_d { dataset_with_b_d.columns.tolist() }')
        return dataset_with_b_d

    @staticmethod
    def get_dataset_for_train_set(data, test_indexes, drop_cols):
        dataset = data.copy()
        dataset_train = dataset[~dataset.index.isin(test_indexes)].drop(drop_cols, axis=1)
        return dataset_train

    @staticmethod
    def delete_outlier(data):
        dataset = data.copy()
        dataset = dataset[dataset['Porosidad'] < 0.8]
        dataset = dataset[dataset['Viscosidad_petroleo'] < 10]
        dataset = dataset[dataset['Factor_volumetrico_petroleo'] < 1.4]
        return dataset
