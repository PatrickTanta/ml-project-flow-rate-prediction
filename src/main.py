from sklearn.preprocessing import RobustScaler, OneHotEncoder, FunctionTransformer
from src import cleaning, load_data, preprocessing, chart, modelling
from utils.validations import log_transform

if __name__ == '__main__':

    file_data = load_data.Files.import_from_csv(
        '../database/requirements_encoded.csv',
        ['id', 'Pozo', 'Formacion']
    )

    # import file dependencies
    dataset = file_data.copy()
    cleaning = cleaning.CleanData()
    preprocessing = preprocessing.PreprocessingData()
    modelling = modelling.ModellingData(load_data=load_data.Files)
    plotting = chart.Chart()

    # drop unnecessary columns
    dataset = cleaning.drop_columns(
        data=dataset,
        drop_cols=['Area_drenaje', 'Pozo', 'Reservas', 'Miembro', 'Base', 'Tope', 'YEAR', 'id', 'Caudal_formula', 'Permeabilidad_ajustada']
    )

    # convert zero and negative values
    dataset = cleaning.drop_rows_with_negative_or_zero_values_in_column(data=dataset, column_name='Caudal')
    dataset = cleaning.drop_rows_with_negative_or_zero_values_in_column(data=dataset, column_name='Espesor_neto')
    dataset = cleaning.convert_rows_with_negative_values_to_positive_values(data=dataset, column_name='Compresibilidad_fluidos')

    # clean outliers with criteria from initial EDA
    dataset = cleaning.delete_outlier(dataset)

    # set a number of rows that is going to be used for test our model
    N_TEST = 288

    # hasrq:
    # value -> 1: point out row has requirement information including b and d parameters
    # value -> 0: point out row hasn't requirement information including b and d parameters
    dataset_test = cleaning.get_dataset_for_test_set(data=dataset, test_size=N_TEST)
    dataset_train = cleaning.get_dataset_for_train_set(data=dataset, test_indexes=dataset_test.index.tolist(), drop_cols=['b', 'D', 'hasrq'])

    (X_train, y_train), (X_test, y_test) = preprocessing.get_train_test_split(dataset_train=dataset_train, dataset_test=dataset_test, target_col='Caudal')

    plotting.hist_plot_plt(
        name='Comparison_y_train_test.png',
        features=[y_train, y_test],
        labels=['Data de entrenamiento', 'Data de prueba'],
        colors=['green', 'red'],
        xlabel='Frecuencia',
        ylabel='Caudal de producción'
    )

    preprocessing.save_data_splitted(x_train=X_train, y_train=y_train, x_test=X_test, y_test=y_test)

    # preprocessing step with data normalization and encoding categorical values
    numeric_cols, cat_cols = [cols.tolist() for cols in preprocessing.get_numeric_and_categorical_columns(x_train=X_train)]

    print("\nnumeric_cols: ", numeric_cols)
    print("\ncat_cols: ", cat_cols, "\n")

    # only apply logarithmic function for permeability
    permeability_column = 'Permeabilidad'
    k_transformer = FunctionTransformer(log_transform)

    if permeability_column in numeric_cols:
        numeric_cols.remove(permeability_column)

    print('perm removed: ', permeability_column)
    print(numeric_cols)

    preprocessor = preprocessing.build_column_transformer_for_pipe(
        numeric_steps=[('scaler', RobustScaler())],
        numeric_cols=numeric_cols,
        categorical_steps=[('onehot', OneHotEncoder(handle_unknown='ignore'))],
        categorical_cols=cat_cols,
        permeability_steps=[('k_transformer', k_transformer)],
        permeability_col=permeability_column
    )

    print(f'X_train {X_train.columns}')
    print(f'X_train {X_test.columns}')

    # modelling.run_basic_model_and_show_results(preprocessor, X_train, y_train, X_test, y_test, plotting)
    # modelling.run_cross_validation_model_and_show_results(preprocessor, X_train, y_train, X_test, y_test, plotting)
    modelling.run_cross_validation_optimized_model_and_show_results(preprocessor, X_train, y_train, X_test, y_test, plotting)




