import unittest
import joblib
import pandas as pd
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import Pipeline

from src.cleaning import CleanData
from src.load_data import Files
from src.preprocessing import PreprocessingData
from src.modelling import ModellingData
from utils.validations import log_transform

if __name__ == '__main__':
    unittest.main()


class TestFileImportAndExport(unittest.TestCase):
    def setUp(self) -> None:
        self.files = Files()

    def test_import_from_csv(self):
        data = self.files.import_from_csv('../database/requirements_encoded.csv')
        self.assertEqual(data.shape, (1921, 29))

    def test_if_imported_table_has_necessary_columns(self):
        data = self.files.import_from_csv('../database/requirements_encoded.csv')
        self.assertEqual(data.shape, (1921, 29))

    def test_import_from_excel(self):
        data = self.files.import_from_excel('../database/requirements_encoded.xlsx', 'requirements_encoded')
        self.assertEqual(data.shape, (1921, 29))

    def test_export_model(self):
        model = {}
        self.files.export_model(model, '../models/test_model.pkl')
        model_got = joblib.load('../models/test_model.pkl')
        self.assertEqual(model, model_got)


class TestCleanData(unittest.TestCase):
    def setUp(self) -> None:
        self.files = Files()
        self.data = self.files.import_from_csv('../database/requirements_encoded.csv')
        self.clean_data = CleanData()

    def test_if_columns_area_dropped(self):
        dataset = self.data.copy()
        drop_cols = ['Area_drenaje', 'Pozo', 'Reservas', 'Miembro', 'Base', 'Tope', 'YEAR', 'id', 'Caudal_formula', 'Permeabilidad_ajustada', 'hasrq']
        dataset = self.clean_data.drop_columns(data=dataset, drop_cols=drop_cols)
        cols = dataset.columns.tolist()
        for col in drop_cols:
            self.assertFalse(col in cols)

    def test_if_drop_rows_with_negative_or_zero_values_in_column(self):
        dataset = self.data.copy()
        column_name = 'Espesor_neto'
        dataset = self.clean_data.drop_rows_with_negative_or_zero_values_in_column(data=dataset, column_name=column_name)
        self.assertEqual(dataset[dataset[column_name] <= 0].shape[0], 0)

    def test_if_convert_rows_with_negative_values_to_positive_values(self):
        dataset = self.data.copy()
        column_name = 'Compresibilidad_fluidos'
        dataset = self.clean_data.convert_rows_with_negative_values_to_positive_values(data=dataset, column_name=column_name)
        self.assertEqual(dataset[dataset[column_name] < 0].shape[0], 0)

    def test_if_get_dataset_for_test_set(self):
        dataset = self.data.copy()
        N_TEST = 250
        dataset_width_b_d = self.clean_data.get_dataset_for_test_set(dataset, test_size=N_TEST)
        self.assertEqual(dataset_width_b_d.shape[0], N_TEST)

    def test_if_get_dataset_for_train_set(self):
        dataset = self.data.copy()
        test_indexes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 10]
        drop_cols = ["b", "D", "hasrq"]
        dataset_train = self.clean_data.get_dataset_for_train_set(dataset, test_indexes=test_indexes, drop_cols=drop_cols)
        self.assertEqual(dataset_train[dataset_train.index.isin(test_indexes)].shape[0], 0)
        self.assertTrue(not any([c in drop_cols for c in dataset_train.columns.tolist()]))

    def test_delete_outlier(self):
        dataset = self.data.copy()
        dataset_without_outliers = self.clean_data.delete_outlier(dataset)
        self.assertEqual(dataset_without_outliers[dataset_without_outliers['Porosidad'] >= 0.8].shape[0], 0)
        self.assertEqual(dataset_without_outliers[dataset_without_outliers['Viscosidad_petroleo'] >= 10].shape[0], 0)
        self.assertEqual(dataset_without_outliers[dataset_without_outliers['Factor_volumetrico_petroleo'] >= 1.4].shape[0], 0)


class TestPreprocessingData(unittest.TestCase):
    def setUp(self) -> None:
        self.files = Files()
        self.data = self.files.import_from_csv('../database/requirements_encoded.csv')
        self.preprocessing_data = PreprocessingData()

    def test_get_numeric_and_categorical_columns(self):
        dataset_random = pd.DataFrame({
            'Item': {0: 'A', 1: 'A', 2: 'B', 3: 'B'},
            'Variable1': {0: 21.3, 1: 18.4, 2: 12.3, 3: 9.4},
            'Variable2': {0: 19.4, 1: 17.2, 2: 11.6, 3: 10.2}
        })
        numeric_cols, cat_cols = self.preprocessing_data.get_numeric_and_categorical_columns(dataset_random)
        self.assertEqual(numeric_cols.tolist(), ['Variable1', 'Variable2'])
        self.assertEqual(cat_cols.tolist(), ['Item'])

    def test_function_transformer(self):
        dataset_random_1 = pd.DataFrame({
            'Variable1': {0: 21.3, 1: 18.4, 2: 12.3, 3: 9.4, 4: 10, 5: 100}
        })

        transformer = FunctionTransformer(log_transform)
        model = Pipeline(steps=[('k_transformer', transformer)])

        dataset_random_1_logarithm = model.transform(dataset_random_1)

        self.assertEqual(dataset_random_1_logarithm.iloc[5, 0], 2)
        self.assertEqual(dataset_random_1_logarithm.iloc[4, 0], 1)


class TestModellingData(unittest.TestCase):
    def setUp(self) -> None:
        self.files = Files()
        self.X_train = self.files.import_from_excel(self.PATH + '//database//train_features.xlsx', sheet_name='features')
        self.modelling_data = ModellingData()

