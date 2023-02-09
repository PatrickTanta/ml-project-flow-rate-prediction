import os
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


class PreprocessingData:
    def __init__(self):
        self.PATH = os.path.realpath(os.path.join(os.path.dirname(__file__), '..')).replace('\\', '//')

    @staticmethod
    def get_train_test_split(dataset_train, dataset_test, target_col):
        X_train, y_train = dataset_train.drop(target_col, axis=1), dataset_train[target_col]
        X_test, y_test = dataset_test.drop(target_col, axis=1), dataset_test[target_col]
        return (X_train, y_train), (X_test, y_test)

    def save_data_splitted(self, x_train, y_train, x_test, y_test):
        x_train.to_excel(self.PATH + "//database//train_features.xlsx", sheet_name="features")
        y_train.to_excel(self.PATH + "//database//train_target.xlsx", sheet_name="target")
        x_test.to_excel(self.PATH + "//database//test_features.xlsx", sheet_name="features")
        y_test.to_excel(self.PATH + "//database//test_target.xlsx", sheet_name="target")

    @staticmethod
    def get_numeric_and_categorical_columns(x_train):
        numeric_cols = x_train.select_dtypes(include=["int64", "float64"]).columns
        cat_cols = x_train.select_dtypes(include=["object", "category"]).columns
        return numeric_cols, cat_cols

    @staticmethod
    def build_column_transformer_for_pipe(
        numeric_steps,
        numeric_cols,
        categorical_steps,
        categorical_cols,
        permeability_col,
        permeability_steps
    ):
        """
        Function that returns column transformer with preprocessing steps for
        numeric and categorical columns
        """

        numeric_transformer = Pipeline(steps=numeric_steps)
        categorical_transformer = Pipeline(steps=categorical_steps)
        # is preferable having permeability on log scale
        k_transformer = Pipeline(steps=permeability_steps)

        preprocessor = ColumnTransformer(
            transformers=[
                ('numeric', numeric_transformer, numeric_cols),
                ('cat', categorical_transformer, categorical_cols),
                ('k',  k_transformer, permeability_col)
            ]
        )

        return preprocessor

