import joblib as joblib
import pandas as pd

from utils.validations import test_if_necessary_columns_exists


class Files:
    @staticmethod
    def import_from_csv(url, required_columns=None):
        """
        Function to import files with extension .csv as pandas dataframe
        """
        data = pd.read_csv(url)
        if required_columns:
            test_if_necessary_columns_exists(data.columns.tolist(), required_columns)
        print(f'Nr rows: { data.shape[0] } Attributes: { data.shape[1] }')
        return data

    @staticmethod
    def import_from_excel(url, sheet_name, required_columns=None):
        """
        Function to import files with extension .xlsx as pandas dataframe
        """
        data = pd.read_excel(url, engine='openpyxl', sheet_name=sheet_name)
        if required_columns:
            test_if_necessary_columns_exists(data.columns.tolist(), required_columns)
        print(f'Nr rows: { data.shape[0] } Attributes: { data.shape[1] }')
        return data

    @staticmethod
    def export_model(model, filename):
        """
        Function to save model
        """
        joblib.dump(model, filename)

    @staticmethod
    def import_model(filename):
        """
        Function to import model
        """
        try:
            with open(filename, 'rb') as f:
                data = joblib.load(f)
            return data
        except (OSError, IOError) as e:
            return False



