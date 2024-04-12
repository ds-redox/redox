import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.svm import SVC
from sklearn.feature_selection import SelectKBest, f_classif
import pickle

def get_data_2022(folder: str):
    exclude_features = ['TIMESTAMP', 'pit_number', 'Redox_error_flag']
    data = pd.read_pickle(open(folder, 'rb'))
    return data.loc[:,~data.columns.isin(exclude_features)]

def get_data_2022_sensors(folder: str, sensor: int):
    features = [f'Redox_Avg({sensor})', f'EC_Avg({sensor})', f'Matric_potential_Avg({sensor})',
                f'Temp_T12_Avg({sensor})', 'Water_level_Avg', 'Temp_ottpls_Avg', 'BatterymV_Min',
                f'WC{sensor}', f'Redox_Avg({sensor})_sigma_b_24', f'Redox_Avg({sensor})_sigma_f_24',
                f'Redox_Avg({sensor})_sigma_b_12', f'Redox_Avg({sensor})_sigma_f_12']
    data = pd.read_pickle(open(folder, 'rb'))
    return data.loc[:,features]

def top_10_features(X: pd.DataFrame, y: pd.DataFrame):
    selector = SelectKBest(f_classif, k=10)
    selector.fit(X, np.ravel(y))
    return list(selector.get_feature_names_out())

def get_full_data(train, test):
    return train.append(test, ignore_index=True)

def gridsearchcv(X: pd.DataFrame, y: pd.DataFrame, file_name: str, get_model: bool):
    svc = SVC(kernel="poly", random_state=0)
    cv = KFold(n_splits=10, shuffle=True, random_state=0)
    param_grids = {"degree": [3,4,5,6,7], "C": [0.5,1,2,3,4,5,6,7]}

    grid_search = GridSearchCV(
        estimator=svc,
        param_grid=param_grids,
        return_train_score=True,
        refit=get_model,
        n_jobs=10,
        cv=cv,
    ).fit(X, np.ravel(y))

    if get_model:
        pickle.dump(grid_search, open(file_name, 'wb'))
    else:
        result = pd.DataFrame(grid_search.cv_results_)
        result.to_csv(file_name)

def gridsearch_2022(feature_selection: bool, get_model: bool):
    
    # Get train and test data for 2022
    # Train X
    folder = f'./data/2022/X_train.pkl'
    train_X = get_data_2022(folder)
    # Test X
    folder = f'./data/2022/X_test.pkl'
    test_X = get_data_2022(folder)

    # Train X scaled
    folder = f'./data/2022/Scaled/X_train_scaled.pkl'
    train_X_scaled = get_data_2022(folder)
    # Test X scaled
    folder = f'./data/2022/Scaled/X_test_scaled.pkl'
    test_X_scaled = get_data_2022(folder)

    # Train y
    folder = f'./data/2022/y_train.pkl'
    train_y = pd.read_pickle(open(folder, 'rb'))
    # Test y
    folder = f'./data/2022/y_test.pkl'
    test_y = pd.read_pickle(open(folder, 'rb'))

    X = pd.DataFrame
    y = pd.DataFrame

    if get_model:
        y = train_y
        if feature_selection:
            top_10_columns = top_10_features(train_X, train_y)
            X = train_X_scaled.loc[:,top_10_columns]
        else:
            X = train_X_scaled
    else:
        y = get_full_data(train_y, test_y)
        if feature_selection:
            top_10_columns = top_10_features(get_full_data(train_X, test_X), y)
            X = get_full_data(train_X_scaled, test_X_scaled).loc[:,top_10_columns]
        X = get_full_data(train_X_scaled, test_X_scaled)


    # Cross validation
    if get_model:
        if feature_selection:
            gridsearchcv(X, y, 'gs_2022_fs', get_model)
        else:
            gridsearchcv(X, y, 'gs_2022', get_model)
    else:
        if feature_selection:
            gridsearchcv(X, y, './Results_fs.csv', get_model)
        else:
            gridsearchcv(X, y, './Results.csv', get_model)
        
def gridsearch_2022_sensors(feature_selection: bool, get_model: bool):
    # Loop through sensors and train individual models for each sensor
    for sensor in range(1,6):

        # Get train and test data for 2022_sensors
        # Train X
        folder = f'./data/2022_sensors/X_train_sensor_{sensor}.pkl'
        train_X = get_data_2022_sensors(folder, sensor)
        # Test X
        folder = f'./data/2022_sensors/X_test_sensor_{sensor}.pkl'
        test_X = get_data_2022_sensors(folder, sensor)

        # Train X scaled
        folder = f'./data/2022_sensors/Scaled/X_train_scaled_sensor_{sensor}.pkl'
        train_X_scaled = get_data_2022_sensors(folder, sensor)
        # Test X scaled
        folder = f'./data/2022_sensors/Scaled/X_test_scaled_sensor_{sensor}.pkl'
        test_X_scaled = get_data_2022_sensors(folder, sensor)

        # Train y
        folder = f'./data/2022_sensors/y_train_sensor_{sensor}.pkl'
        train_y = pd.read_pickle(open(folder, 'rb'))
        # Test y
        folder = f'./data/2022_sensors/y_test_sensor_{sensor}.pkl'
        test_y = pd.read_pickle(open(folder, 'rb'))

        X = pd.DataFrame
        y = pd.DataFrame

        if get_model:
            y = train_y
            if feature_selection:
                top_10_columns = top_10_features(train_X, train_y)
                X = train_X_scaled.loc[:,top_10_columns]
            else:
                X = train_X_scaled
        else:
            y = get_full_data(train_y, test_y)
            if feature_selection:
                top_10_columns = top_10_features(get_full_data(train_X, test_X), y)
                X = get_full_data(train_X_scaled, test_X_scaled).loc[:,top_10_columns]
            X = get_full_data(train_X_scaled, test_X_scaled)

        # Cross validation
        if get_model:
            if feature_selection:
                name = f'gs_2022_fs_sensor_{sensor}'
                gridsearchcv(X, y, name, get_model)
            else:
                name = f'gs_2022_sensor_{sensor}'
                gridsearchcv(X, y, name, get_model)
        else:
            if feature_selection:
                name = f'./Results_fs_{sensor}.csv'
                gridsearchcv(X, y, name, get_model)
            else:
                name = f'./Results_{sensor}.csv'
                gridsearchcv(X, y, name, get_model)
    

##### 2022 data #####
# get best model from gridsearch
gridsearch_2022(False, True)

# get best model from gridsearch with feature selection
gridsearch_2022(True, True)

# get results from gridsearch
gridsearch_2022(False, False)

# get result from gridsearch with feature selection
gridsearch_2022(True, False)


##### 2022_sensors data #####
# get best model from gridsearch
gridsearch_2022_sensors(False, True)

# get best model from gridsearch with feature selection
gridsearch_2022_sensors(True, True)

# get results from gridsearch
gridsearch_2022_sensors(False, False)

# get result from gridsearch with feature selection
gridsearch_2022_sensors(True, False)