import json
import os

import pandas as pd
from tqdm import tqdm
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.kernel_ridge import KernelRidge
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import (
    RandomForestRegressor,
    ExtraTreesRegressor,
    AdaBoostRegressor,
    GradientBoostingRegressor,
)
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor


class ModellingData:

    def __init__(self):
        self.PATH = os.path.realpath(os.path.join(os.path.dirname(__file__), '..')).replace('\\', '//')
        data = json.load(open(file=self.PATH + "//utils//grid_params.json", encoding="utf-8"))

        self.params_grid_develop = data["params_grid_develop"]
        self.params_grid_production = data["params_grid_production"]

        self.SEED = 123

    def export_initialized_models(self):
        """
        Function to export initialized models, with basic hyparameters configuration
        """

        models = dict()

        models["linear"] = LinearRegression(n_jobs=-1)
        models["lasso"] = Lasso(random_state=self.SEED)
        models["ridge"] = Ridge(random_state=self.SEED)
        models["kr"] = KernelRidge()
        models["elnt"] = ElasticNet(random_state=self.SEED)
        models["dt"] = DecisionTreeRegressor(random_state=self.SEED)
        models["svm"] = SVR()
        models["knn"] = KNeighborsRegressor(n_jobs=-1)
        models["rf"] = RandomForestRegressor(n_jobs=-1, random_state=self.SEED)
        models["et"] = ExtraTreesRegressor(n_jobs=-1, random_state=self.SEED)
        models["ab"] = AdaBoostRegressor(random_state=self.SEED)
        models["gb"] = GradientBoostingRegressor(random_state=self.SEED)
        models["xgb"] = XGBRegressor(random_state=self.SEED, n_jobs=-1)
        models["lgb"] = LGBMRegressor(random_state=self.SEED, n_jobs=-1)
        models["mlpr"] = MLPRegressor(random_state=self.SEED, max_iter=500)

        return models

    @staticmethod
    def export_models_customized(params):
        """
        Function to export models with specific params
        """

        models = dict()

        super_params_formatted = {}
        for k, v in params.items():
            params_formatted = {}
            for ki, vi in v.items():
                params_formatted[ki.replace("model__", "")] = vi
            super_params_formatted[k] = params_formatted

        try:
            models["linear"] = LinearRegression(**super_params_formatted["linear"])
        except:
            raise "Error initializing LinearRegression"
        try:
            models["lasso"] = Lasso(**super_params_formatted["lasso"])
        except:
            raise "Error initializing Lasso"
        try:
            models["ridge"] = Ridge(**super_params_formatted["ridge"])
        except:
            raise "Error initializing Ridge"
        try:
            models["kr"] = KernelRidge(**super_params_formatted["kr"])
        except:
            raise "Error initializing KernelRidge"
        try:
            models["elnt"] = ElasticNet(**super_params_formatted["elnt"])
        except:
            raise "Error initializing ElasticNet"
        try:
            models["dt"] = DecisionTreeRegressor(**super_params_formatted["dt"])
        except:
            raise "Error initializing DecisionTreeRegressor"
        try:
            models["svm"] = SVR(**super_params_formatted["svm"])
        except:
            raise "Error initializing SVR"
        try:
            models["knn"] = KNeighborsRegressor(**super_params_formatted["knn"])
        except:
            raise "Error initializing KNeighborsRegressor"
        try:
            models["rf"] = RandomForestRegressor(**super_params_formatted["rf"])
        except:
            raise "Error initializing RandomForestRegressor"
        try:
            models["et"] = ExtraTreesRegressor(**super_params_formatted["et"])
        except:
            raise "Error initializing ExtraTreesRegressor"
        try:
            models["ab"] = AdaBoostRegressor(**super_params_formatted["ab"])
        except:
            raise "Error initializing AdaBoostRegressor"
        try:
            models["gb"] = GradientBoostingRegressor(**super_params_formatted["gb"])
        except:
            raise "Error initializing GradientBoostingRegressor"
        try:
            models["xgb"] = XGBRegressor(**super_params_formatted["xgb"])
        except:
            raise "Error initializing XGBRegressor"
        try:
            models["lgb"] = LGBMRegressor(**super_params_formatted["lgb"])
        except:
            raise "Error initializing LGBMRegressor"
        try:
            models["mlpr"] = MLPRegressor(**super_params_formatted["mlpr"])
        except:
            raise "Error initializing MLPRegressor"

        return models

    def run_basic_model_and_show_results(self, preprocessor, X_train, y_train, X_test, y_test, chart):
        pipe_collection = {}
        initialized_models = self.export_initialized_models()

        for key, model in tqdm(initialized_models.items()):
            pipe_collection[key] = Pipeline(
                [('preprocessing', preprocessor), ('model', model)]
            )

        print(X_train.columns)

        mae_score = {}
        for key, pipe in tqdm(pipe_collection.items()):
            print('Executing ... ')
            _ = pipe.fit(X=X_train, y=y_train)
            mae_score[key] = mean_absolute_error(y_test, pipe.predict(X_test))
            print('model fited: ', key)

        df_basic_models = pd.DataFrame([mae_score])

        chart.bar_plot_plt(
            "Models_initialized_default_params Modelos inicializados con los parámetros por default",
            df_basic_models.columns,
            df_basic_models.iloc[0, :],
            "Modelos",
            "Modelos",
            "Métrica (Error absoluto medio)"
        )

        # TODO: investigate why permeability is missing in dataframe

