import json
import multiprocessing
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedKFold, cross_validate, GridSearchCV
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

    def __init__(self, load_data):
        self.PATH = os.path.realpath(os.path.join(os.path.dirname(__file__), '..')).replace('\\', '//')
        data = json.load(open(file=self.PATH + "//utils//grid_params.json", encoding="utf-8"))
        self.params_grid_develop = data["params_grid_develop"]
        self.params_grid_production = data["params_grid_production"]
        self.SEED = 123
        self.load_data = load_data

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

    def generate_pipe_collection(self, preprocessor):
        pipe_collection = {}
        initialized_models = self.export_initialized_models()

        for key, model in tqdm(initialized_models.items()):
            pipe_collection[key] = Pipeline(
                [('preprocessing', preprocessor), ('model', model)]
            )

        return pipe_collection

    def run_basic_model_and_show_results(self, preprocessor, X_train, y_train, X_test, y_test, chart):
        pipe_collection = self.generate_pipe_collection(preprocessor)
        print(X_train.columns)
        mae_score = {}
        print('Executing ... ')
        for key, pipe in tqdm(pipe_collection.items()):
            _ = pipe.fit(X=X_train, y=y_train)
            mae_score[key] = mean_absolute_error(y_test, pipe.predict(X_test))
            print('Model fitted: ', key)

        df_basic_models = pd.DataFrame([mae_score])

        chart.bar_plot_plt(
            "Modelos inicializados con los parámetros por default",
            df_basic_models.columns,
            df_basic_models.iloc[0, :],
            "Modelos inicializados con los parámetros por default",
            "Modelos",
            "Métrica (Error absoluto medio)"
        )

    def run_cross_validation_model_and_show_results(self, preprocessor, X_train, y_train, X_test, y_test, chart):
        pipe_collection = self.generate_pipe_collection(preprocessor)
        mae_score_cv = {}
        print('Executing Cross validations... ')
        for key, pipe in tqdm(pipe_collection.items()):
            cv = RepeatedKFold(n_splits=5, n_repeats=5, random_state=123)
            cv_scores = cross_validate(
                estimator=pipe,
                X=X_train,
                y=y_train,
                scoring=('r2', 'neg_mean_absolute_error'),
                cv=cv,
                return_train_score=True
            )
            mae_score_cv[key] = np.abs(cv_scores['test_neg_mean_absolute_error'].mean())
            print('Cross validated models fitted: ', key)

        df_cv_basic_models = pd.DataFrame([mae_score_cv])

        chart.bar_plot_plt(
            "Modelos inicializados con los parámetros por default usando Cross Validation",
            df_cv_basic_models.columns,
            df_cv_basic_models.iloc[0, :],
            "Modelos inicializados con los parámetros por default usando Cross Validation",
            "Modelos",
            "Métrica (Error absoluto medio)"
        )

    def get_and_save_optimized_model_parameters(self, pipe_collection, X_train, y_train):
        print("\nRun grid search optimization...")
        models_parameters_optimized = {}
        for name, pipe in tqdm(pipe_collection.items()):
            print(f"\nOptimizing  {name}...")
            new_param_grid = {}
            for k, v in self.params_grid_production[name].items():
                new_param_grid["model__{}".format(k)] = v
            grid = GridSearchCV(
                estimator=pipe,
                param_grid=new_param_grid,
                scoring="neg_mean_absolute_error",
                n_jobs=multiprocessing.cpu_count() - 1,
                cv=RepeatedKFold(n_splits=5, n_repeats=5),
                verbose=0,
            )
            _ = grid.fit(X=X_train, y=y_train)
            print("-----------------------------------")
            print(f"Mejores hiperparámetros encontrados para {name}")
            print(grid.best_params_, ":", grid.best_score_, grid.scoring)
            best_params = grid.best_params_
            models_parameters_optimized[name] = best_params

            try:
                self.load_data.export_model(models_parameters_optimized, "models//models_parameters_optimized.pkl")
            except Exception as ex:
                print(ex)

    def run_cross_validation_optimized_model_and_show_results(self, preprocessor, X_train, y_train, X_test, y_test, chart):
        pipe_collection = self.generate_pipe_collection(preprocessor)

        models_parameters_optimized = self.load_data.import_model('models//models_parameters_optimized.pkl')

        # if models has already been optimized, use it
        if not models_parameters_optimized:
            self.get_and_save_optimized_model_parameters(pipe_collection, X_train, y_train)
            models_parameters_optimized = self.load_data.import_model('models//models_parameters_optimized.pkl')

        models_optimized = self.export_models_customized(models_parameters_optimized)

        optimized_pipe_collection = {}
        for key, model in tqdm(models_optimized.items()):
            optimized_pipe_collection[key] = Pipeline(
                [("preprocessing", preprocessor), ("model", model)]
            )

        # predict in test data using every model and obtain what is the model with the less error
        model_score_test = {}
        for key, pipe in tqdm(optimized_pipe_collection.items()):
            best_score = 999
            best_pipe = None
            pipe.fit(X_train, y_train)
            y_pred = pipe.predict(X_test)
            model_score_test[key] = mean_absolute_error(y_test, y_pred)
            score = model_score_test[key]
            if score < best_score:
                best_score = score
                filename = "models//best_model.sav"
                self.load_data.export_model(pipe, filename)

        self.load_data.export_model(model_score_test, "models//model_score_test.pkl")

        df = pd.DataFrame([model_score_test])

        print(df)

        chart.bar_plot_plt(
            "Modelos optimizados usando Cross Validation",
            df.columns,
            df.iloc[0, :],
            "Modelos optimizados usando Cross Validation",
            "Modelos",
            "Métricas Optimizadas",
        )





