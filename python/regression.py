from sklearn.ensemble import RandomForestRegressor as rf
from sklearn.ensemble import AdaBoostRegressor as adb
from sklearn.model_selection import GridSearchCV
from xgboost import XGBRegressor
from sklearn import svm
from sklearn.svm import SVR
import os
# os.add_dll_directory('C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.2/bin')
# from thundersvm import SVR
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler, Normalizer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures, SplineTransformer
from sklearn.pipeline import make_pipeline
import matplotlib.pyplot as plt
import numpy as np
import mlflow
import time
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import termtables as tt
import pandas as pd
# import datashader as ds
# from datashader.mpl_ext import dsshow
from matplotlib.colors import LinearSegmentedColormap
from matplotlib import cm
from matplotlib.colors import Normalize 
from scipy.interpolate import interpn
import math
from mgwr.gwr import GWR
import general
from scipy.interpolate import griddata, RBFInterpolator

white_viridis = LinearSegmentedColormap.from_list('white_viridis', [
    (0, '#ffffff'),
    (1e-20, '#440053'),
    (0.2, '#404388'),
    (0.4, '#2a788e'),
    (0.6, '#21a784'),
    (0.8, '#78d151'),
    (1, '#fde624'),
], N=256)

# from cuml.ensemble import RandomForestRegressor as rf_rapids
# from dask import persist
# import cudf


def data_split(features, labels, indices, p_test):
    headers = ['# total', '# training', '# testing']
    train_features, test_features, train_labels, test_labels, train_indices, test_indices = train_test_split(
        features, labels, indices, test_size=p_test, random_state=42)
    t_lst = [[features.shape[0], train_features.shape[0], test_features.shape[0]]]
    string = tt.to_string(
        t_lst,
        header = headers,
        style=tt.styles.ascii_thin_double,
        padding=(0,1),
        alignment='c'*len(headers)
    )
    # print(string)
    return train_features, test_features, train_labels, test_labels, train_indices, test_indices


# def using_datashader(ax, x, y):
#     df = pd.DataFrame(dict(x=x, y=y))
#     dsartist = dsshow(
#         df,
#         ds.Point("x", "y"),
#         ds.count(),
#         vmin=0,
#         vmax=35,
#         norm="linear",
#         aspect="auto",
#         ax=ax,
#     )
#     plt.colorbar(dsartist)


def using_mpl_scatter_density(fig, x, y):
    ax = fig.add_subplot(1, 1, 1, projection='scatter_density')
    density = ax.scatter_density(x, y, cmap=white_viridis)
    fig.colorbar(density, label='Number of points per pixel')


def density_scatter( x , y, ax = None, sort = True, bins = 20, **kwargs )   :
    """
    Scatter plot colored by 2d histogram
    """
    if ax is None :
        fig , ax = plt.subplots()
    data , x_e, y_e = np.histogram2d( x, y, bins = bins, density = True )
    z = interpn( ( 0.5*(x_e[1:] + x_e[:-1]) , 0.5*(y_e[1:]+y_e[:-1]) ) , data , np.vstack([x,y]).T , method = "splinef2d", bounds_error = False)

    #To be sure to plot all data
    z[np.where(np.isnan(z))] = 0.0

    # Sort the points by density, so that the densest points are plotted last
    if sort :
        idx = z.argsort()
        x, y, z = x[idx], y[idx], z[idx]

    ax.scatter( x, y, c=z, s=1, **kwargs )

    norm = Normalize(vmin = np.min(z), vmax = np.max(z))
    cbar = fig.colorbar(cm.ScalarMappable(norm = norm), ax=ax)
    cbar.ax.set_ylabel('Density')

    return ax


def plot_regression_results_R(y_true, y_pred, title, r2, elapsed_time):
    """Scatter plot of the predicted vs true targets."""
    # fig, ax = plt.subplots()
    ax = density_scatter(y_true, y_pred, bins=[30,30])
    ax.plot(
        [y_true.min(), y_true.max()], [y_true.min(), y_true.max()], "--r", linewidth=2
    )
    
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    ax.spines["left"].set_position(("outward", 10))
    ax.spines["bottom"].set_position(("outward", 10))
    ax.set_xlim([y_true.min(), y_true.max()])
    ax.set_ylim([y_true.min(), y_true.max()])
    ax.set_xlabel("Measured")
    ax.set_ylabel("Predicted")
    extra = plt.Rectangle(
        (0, 0), 0, 0, fc="w", fill=False, edgecolor="none", linewidth=0
    )
    ax.legend([extra], [r2], loc="upper left")
    title = title + "\n Evaluation in {:.2f} seconds".format(elapsed_time)
    ax.set_title(title)


def RF_tunning(features, labels, n_trees_list, EXPERIMENT_NAME):
    # EXPERIMENT_NAME = "GEDI IN random forest tunning mend sub"
    EXPERIMENT_ID = mlflow.create_experiment(EXPERIMENT_NAME)
    n_samples = np.max(labels.shape)
    indices = np.arange(n_samples)
    train_features, test_features, train_labels, test_labels, train_indices, test_indices = \
        data_split(features, labels, indices, 0.25)
    scaler = StandardScaler()
    scaler.fit(train_features)
    train_features = scaler.transform(train_features)
    test_features = scaler.transform(test_features)
    features = scaler.transform(features)
    for idx, n_trees in enumerate(n_trees_list):
        regressor = rf(n_estimators=n_trees, random_state=42, n_jobs=-1)
        regressor.fit(train_features, train_labels)
        prediction = regressor.predict(test_features)
        mse_test = mean_squared_error(test_labels, prediction)
        mae_test = mean_absolute_error(test_labels, prediction)
        r2_test = r2_score(test_labels, prediction)

        prediction_train = regressor.predict(train_features)
        mse_train = mean_squared_error(train_labels, prediction_train)
        mae_train = mean_absolute_error(train_labels, prediction_train)
        r2_train = r2_score(train_labels, prediction_train)

        # Start MLflow
        RUN_NAME = f"run_{idx}"
        with mlflow.start_run(experiment_id=EXPERIMENT_ID, run_name=RUN_NAME) as run:
            # Retrieve run id
            RUN_ID = run.info.run_id
            # Track parameters
            mlflow.log_param("n_estimator", n_trees)
            # Track metrics
            mlflow.log_metric("mean_squared_error_test", mse_test)
            mlflow.log_metric("mean_absolute_error_test", mae_test)
            mlflow.log_metric("r2_score_test", r2_test)
            mlflow.log_metric("mean_squared_error_train", mse_train)
            mlflow.log_metric("mean_absolute_error_train", mae_train)
            mlflow.log_metric("r2_score_train", r2_train)
            # Track model
            mlflow.sklearn.log_model(regressor, "regressor")


def regressor_gb(features, labels, feature_list, save_path):
    n_samples = np.max(labels.shape)
    indices = np.arange(n_samples)
    train_features, test_features, train_labels, test_labels, train_indices, test_indices = \
        data_split(features, labels, indices, 0.15)
    scaler = StandardScaler()
    scaler.fit(train_features)
    train_features = scaler.transform(train_features)
    test_features = scaler.transform(test_features)
    features = scaler.transform(features)

    start_time = time.time()
    regressor = XGBRegressor(max_depth=100,random_state=2, gamma=10,alpha=1, tree_method='approx')
    # cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
    # scores = cross_val_score(regressor, train_features, train_labels, cv=10)
    # print("Mean cross-validation score: %.2f" % scores.mean())

    regressor.fit(train_features, train_labels)
    prediction = regressor.predict(test_features)
    elapsed_time = time.time() - start_time

    prediction_training = regressor.predict(train_features)
    rmse_train = mean_squared_error(train_labels, prediction_training, squared=False)
    errors_training = prediction_training - train_labels
    t_lst = []
    t_lst.append(['Training', '{:.4f}'.format(rmse_train), '{:.4f}'.format(errors_training.mean()), '{:.4f}'.format(np.median(errors_training)), '{:.4f}'.format(errors_training.std())])

    rmse_test = mean_squared_error(test_labels, prediction, squared=False)
    errors_testing = prediction - test_labels
    t_lst.append(['Testing', '{:.4f}'.format(rmse_test), '{:.4f}'.format(errors_testing.mean()), '{:.4f}'.format(np.median(errors_testing)), '{:.4f}'.format(errors_testing.std())])

    prediction_all = regressor.predict(features)
    errors_all = prediction_all - labels
    rmse_all = mean_squared_error(labels, prediction_all, squared=False)
    t_lst.append(['All', '{:.4f}'.format(rmse_all), '{:.4f}'.format(errors_all.mean()), '{:.4f}'.format(np.median(errors_all)), '{:.4f}'.format(errors_all.std())])
    headers = [' ', 'RMSE', 'mean', 'median', 'STD']
    string = tt.to_string(
        t_lst,
        header=headers,
        style=tt.styles.ascii_thin_double,
        padding=(0,1),
        alignment='c'*len(headers)
    )
    print(string)
    tmp_df = pd.DataFrame(t_lst, columns=headers)
    tmp_path = save_path.replace('variable_importance', 'training_analysis')
    tmp_path = tmp_path.replace('.png', '.csv')
    tmp_df.to_csv(tmp_path)

    r2 = r2_score(train_labels, prediction_training)
    plot_regression_results_R(
        train_labels,
        prediction_training,
        "Random forest training dataset",
        (r"$R^2={:.2f}$").format(
            r2
        ),
        elapsed_time,
    )
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.savefig(save_path.replace('variable_importance', 'relationship_training'))

    r2 = r2_score(test_labels, prediction)
    plot_regression_results_R(
        test_labels,
        prediction,
        "Random forest testing dataset",
        (r"$R^2={:.2f}$").format(
            r2
        ),
        elapsed_time,
    )
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.savefig(save_path.replace('variable_importance', 'relationship_testing'))

    r2 = r2_score(labels, prediction_all)
    plot_regression_results_R(
        labels,
        prediction_all,
        "Random forest",
        (r"$R^2={:.2f}$").format(
            r2
        ),
        elapsed_time,
    )
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.savefig(save_path.replace('variable_importance', 'relationship_all'))

    # Get numerical feature importances
    importances = list(regressor.feature_importances_)
    np.savetxt(save_path.replace('png', 'csv'), importances)
    # print("Importances:", importances)

    plt.figure()
    plt.style.use('seaborn-paper')
    x_values = list(range(len(importances)))
    plt.bar(x_values, importances, orientation='vertical')
    plt.xticks(x_values, feature_list, rotation='vertical')
    plt.ylabel('Importance')
    plt.xlabel('Variable')
    plt.title('Variable Importance')
    plt.tight_layout()
    plt.savefig(save_path)
    # plt.show(block=True)
    # return regressor, scaler, prediction_training, train_labels, test_indices
    return regressor, scaler, prediction, test_labels, test_indices


def regressor_adb(features, labels, feature_list, save_path):
    n_samples = np.max(labels.shape)
    indices = np.arange(n_samples)
    train_features, test_features, train_labels, test_labels, train_indices, test_indices = \
        data_split(features, labels, indices, 0.25)
    scaler = StandardScaler()
    scaler.fit(train_features)
    train_features = scaler.transform(train_features)
    test_features = scaler.transform(test_features)
    features = scaler.transform(features)

    start_time = time.time()
    regressor = adb(n_estimators=2000, random_state=1, loss="exponential")
    regressor.fit(train_features, train_labels)
    prediction = regressor.predict(test_features)
    errors = (prediction - test_labels)
    elapsed_time = time.time() - start_time

    prediction_training = regressor.predict(train_features)
    rmse_train = mean_squared_error(train_labels, prediction_training, squared=False)
    errors_training = prediction_training - train_labels
    t_lst = []
    t_lst.append(['Training', '{:.4f}'.format(rmse_train), '{:.4f}'.format(errors_training.mean()), '{:.4f}'.format(np.median(errors_training)), '{:.4f}'.format(errors_training.std())])

    rmse_test = mean_squared_error(test_labels, prediction, squared=False)
    errors_testing = prediction - test_labels
    t_lst.append(['Testing', '{:.4f}'.format(rmse_test), '{:.4f}'.format(errors_testing.mean()), '{:.4f}'.format(np.median(errors_testing)), '{:.4f}'.format(errors_testing.std())])

    prediction_all = regressor.predict(features)
    errors_all = prediction_all - labels
    rmse_all = mean_squared_error(labels, prediction_all, squared=False)
    t_lst.append(['All', '{:.4f}'.format(rmse_all), '{:.4f}'.format(errors_all.mean()), '{:.4f}'.format(np.median(errors_all)), '{:.4f}'.format(errors_all.std())])
    headers = [' ', 'RMSE', 'mean', 'median', 'STD']
    string = tt.to_string(
        t_lst,
        header=headers,
        style=tt.styles.ascii_thin_double,
        padding=(0,1),
        alignment='c'*len(headers)
    )
    print(string)
    tmp_df = pd.DataFrame(t_lst, columns=headers)
    tmp_path = save_path.replace('variable_importance', 'training_analysis')
    tmp_path = tmp_path.replace('.png', '.csv')
    tmp_df.to_csv(tmp_path)

    r2 = r2_score(labels, prediction_all)
    plot_regression_results_R(
        labels,
        prediction_all,
        "Random forest",
        (r"$R^2={:.2f}$").format(
            r2
        ),
        elapsed_time,
    )
    # plt.suptitle("Single predictors versus stacked predictors")
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.savefig(save_path.replace('variable_importance', 'relationship_between_obs_calc'))
    # plt.show(block=True)

    # Get numerical feature importances
    importances = list(regressor.feature_importances_)
    np.savetxt(save_path.replace('png', 'csv'), importances)
    # print("Importances:", importances)

    plt.figure()
    plt.style.use('seaborn-paper')
    x_values = list(range(len(importances)))
    plt.bar(x_values, importances, orientation='vertical')
    plt.xticks(x_values, feature_list, rotation='vertical')
    plt.ylabel('Importance')
    plt.xlabel('Variable')
    plt.title('Variable Importance')
    plt.tight_layout()
    plt.savefig(save_path)
    # plt.show(block=True)
    return regressor, scaler, prediction_all, train_indices, test_indices


def regressor_RF(features, labels, feature_list, save_path, num):
    print('performing Random Forest Regression...')
    n_samples = np.max(labels.shape)
    indices = np.arange(n_samples)
    train_features, test_features, train_labels, test_labels, train_indices, test_indices = \
        data_split(features, labels, indices, 0.25)
    scaler = StandardScaler()
    # scaler.fit(train_features)
    # train_features = scaler.transform(train_features)
    # # norm = Normalizer().fit(train_features)
    # # train_features = norm.transform(train_features)
    # test_features = scaler.transform(test_features)
    # features = scaler.transform(features)

    # scaler_label = StandardScaler().fit(train_labels.reshape((-1, 1)))
    # train_labels = scaler_label.transform(train_labels.reshape((-1, 1)))
    # norm_label = Normalizer().fit(train_labels)
    # train_labels = norm_label.transform(train_labels)
    # train_labels = train_labels.flatten()

    start_time = time.time()
    print('start training')
    # regressor = rf(n_estimators=30, random_state=42, n_jobs=-1)
    n_features = features.shape[1]
    regressor = rf(criterion="friedman_mse", n_estimators=num,max_features=math.ceil(n_features/3), random_state=42, n_jobs=-1)
    regressor.fit(train_features, train_labels)
    prediction = regressor.predict(test_features)
    # prediction = (scaler_label.inverse_transform(prediction.reshape((-1, 1))))
    elapsed_time = time.time() - start_time
    print('stop training')

    prediction_training = regressor.predict(train_features)
    # prediction_training = scaler_label.inverse_transform(prediction_training.reshape((-1, 1)))
    # train_labels = scaler_label.inverse_transform(train_labels.reshape((-1, 1)))
    rmse_train = mean_squared_error(train_labels, prediction_training, squared=False)
    errors_training = prediction_training.flatten() - train_labels.flatten()
    t_lst = []
    t_lst.append(['Training', '{:.4f}'.format(rmse_train), '{:.4f}'.format(errors_training.mean()), '{:.4f}'.format(np.median(errors_training)), '{:.4f}'.format(errors_training.std())])

    rmse_test = mean_squared_error(test_labels.flatten(), prediction.flatten(), squared=False)
    errors_testing = prediction.flatten() - test_labels.flatten()
    t_lst.append(['Testing', '{:.4f}'.format(rmse_test), '{:.4f}'.format(errors_testing.mean()), '{:.4f}'.format(np.median(errors_testing)), '{:.4f}'.format(errors_testing.std())])

    prediction_all = regressor.predict(features)
    # prediction_all = scaler_label.inverse_transform(prediction_all.reshape((-1, 1)))
    errors_all = prediction_all.flatten() - labels.flatten()
    rmse_all = mean_squared_error(labels, prediction_all, squared=False)
    t_lst.append(['All', '{:.4f}'.format(rmse_all), '{:.4f}'.format(errors_all.mean()), '{:.4f}'.format(np.median(errors_all)), '{:.4f}'.format(errors_all.std())])
    headers = [' ', 'RMSE', 'mean', 'median', 'STD']
    string = tt.to_string(
        t_lst,
        header=headers,
        style=tt.styles.ascii_thin_double,
        padding=(0,1),
        alignment='c'*len(headers)
    )
    print(string)
    tmp_df = pd.DataFrame(t_lst, columns=headers)
    tmp_path = save_path.replace('variable_importance', 'training_analysis')
    tmp_path = tmp_path.replace('.png', '.csv')
    tmp_df.to_csv(tmp_path)

    r2 = r2_score(train_labels, prediction_training)
    plot_regression_results_R(
        train_labels.flatten(),
        prediction_training.flatten(),
        "Random forest training dataset",
        (r"$R^2={:.2f}$").format(
            r2
        ),
        elapsed_time,
    )
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.savefig(save_path.replace('variable_importance', 'relationship_training'))
    print("Figure saved to: ", save_path.replace('variable_importance', 'relationship_training'))

    r2 = r2_score(test_labels, prediction)
    plot_regression_results_R(
        test_labels.flatten(),
        prediction.flatten(),
        "Random forest testing dataset",
        (r"$R^2={:.2f}$").format(
            r2
        ),
        elapsed_time,
    )
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.savefig(save_path.replace('variable_importance', 'relationship_testing'))
    print("Figure saved to: ", save_path.replace('variable_importance', 'relationship_testing'))

    r2 = r2_score(labels, prediction_all)
    plot_regression_results_R(
        labels.flatten(),
        prediction_all.flatten(),
        "Random forest",
        (r"$R^2={:.2f}$").format(
            r2
        ),
        elapsed_time,
    )
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.savefig(save_path.replace('variable_importance', 'relationship_all'))
    print("Figure saved to: ", save_path.replace('variable_importance', 'relationship_all'))

    # Get numerical feature importances
    importances = list(regressor.feature_importances_)
    np.savetxt(save_path.replace('png', 'csv'), importances)
    # print("Importances:", importances)

    plt.figure()
    plt.style.use('seaborn-paper')
    x_values = list(range(len(importances)))
    plt.bar(x_values, importances, orientation='vertical')
    plt.xticks(x_values, feature_list, rotation='vertical')
    plt.ylabel('Importance')
    plt.xlabel('Variable')
    plt.title('Variable Importance')
    plt.tight_layout()
    plt.savefig(save_path)
    print("Figure saved to: ", save_path)
    # plt.show(block=True)
    return regressor, scaler, prediction_all, train_indices, test_indices

def regressor_svm(features, labels, kernel_func, save_path):
    print(f"performing Support Vector Regression with {kernel_func} kernel function ...")
    n_samples = np.max(labels.shape)
    indices = np.arange(n_samples)
    train_features, test_features, train_labels, test_labels, train_indices, test_indices = \
        data_split(features, labels, indices, 0.25)
    
    dh = train_features[:, 2].flatten() - train_labels.flatten()
    weights = np.exp(-abs(dh)/(50**2))
    weights = np.ones_like(train_labels.flatten())
    
    scaler = StandardScaler()
    scaler.fit(train_features)
    train_features = scaler.transform(train_features)
    # # norm = Normalizer().fit(train_features)
    # # train_features = norm.transform(train_features)
    test_features = scaler.transform(test_features)
    features = scaler.transform(features)

    scaler_label = StandardScaler().fit(train_labels.reshape(-1, 1))
    train_labels = scaler_label.transform(train_labels.reshape(-1, 1))

    # norm_label = Normalizer().fit(train_labels)
    # train_labels = norm_label.transform(train_labels)
    # train_labels = train_labels.flatten()

    print('start training')
    start_time = time.time()
    if kernel_func == 'rbf':
        # from sklearn.kernel_approximation import RBFSampler
        # best_gamma = 0.001
        # best_score = 10000
        # best_C = 0
        # for gamma in [1e-4]:
        #     rbf_transformer = RBFSampler(gamma=gamma, random_state=1)
        #     train_features_ = rbf_transformer.fit_transform(train_features)
        #     # test_features_ = rbf_transformer.fit_transform(test_features)
        #     # features = rbf_transformer.fit_transform(features)
        #     svr = GridSearchCV(SVR(kernel="linear", epsilon=0.01),
        #                     param_grid={"C": np.linspace(150, 250, 5)}, scoring='neg_root_mean_squared_error', verbose=2)
        #     svr.fit(train_features_, train_labels, sample_weight=weights)
        #     if abs(svr.best_score_) < best_score:
        #         best_score = svr.best_score_
        #         best_gamma = gamma
        #         best_C = svr.best_estimator_.C
        #     else:
        #         continue
        
        # print("Best gamma: ", best_gamma)
        # print("Best C: ", best_C)
        # print(f"RMSE score: {svr.best_score_:.3f}")
        
        svr = GridSearchCV(SVR(kernel="rbf", epsilon=0.01),
                        param_grid={"C": np.linspace(1, 500, 7), "gamma": [0.0017782794100389228]}, scoring='neg_root_mean_squared_error', verbose=0)
        svr.fit(train_features, train_labels, sample_weight=weights)
        print(f"Best SVR with params: {svr.best_params_} and RMSE score: {svr.best_score_:.3f}")
        # np.logspace(-2, 5, 5)
    elif kernel_func == 'linear':
        svr = GridSearchCV(SVR(kernel="linear", epsilon=0.01),
                        param_grid={"C": np.linspace(0, 2, 2)}, scoring='neg_root_mean_squared_error', verbose=3)
        svr.fit(train_features, train_labels, sample_weight=weights)
        print(f"Best SVR with params: {svr.best_params_} and RMSE score: {svr.best_score_:.3f}")
    elif kernel_func == 'custom':
        print('using self-defined kernel')
        svr = GridSearchCV(svm.SVR(kernel=general.my_kernel),
                        param_grid={"C": np.linspace(1, 20, 5)})
        svr.fit(train_features, train_labels, sample_weight=weights)
        print(f"Best SVR with params: {svr.best_params_} and RMSE score: {svr.best_score_:.3f}")
    if kernel_func == 'rbf':
        regressor = SVR(kernel='rbf', gamma=svr.best_estimator_.gamma, C=svr.best_estimator_.C, verbose=False)
        # regressor = SVR(kernel='linear', C=best_C, gamma=best_gamma, verbose=False)
        # rbf_transformer = RBFSampler(gamma=best_gamma, random_state=1)
        # train_features = rbf_transformer.fit_transform(train_features)
        # test_features = rbf_transformer.fit_transform(test_features)
        # features = rbf_transformer.fit_transform(features)
    elif kernel_func == 'linear':
        regressor = SVR(kernel='linear', C=svr.best_estimator_.C, verbose=False)

    regressor.fit(train_features, train_labels, sample_weight=weights)
    train_labels = scaler_label.inverse_transform(train_labels)
    prediction = scaler_label.inverse_transform(regressor.predict(test_features).reshape(-1, 1))
    # prediction = regressor.predict(test_features).reshape(-1, 1)
    elapsed_time = time.time() - start_time
    
    prediction_training = scaler_label.inverse_transform(regressor.predict(train_features).reshape(-1, 1))
    # prediction_training = regressor.predict(train_features).reshape(-1, 1)
    rmse_train = mean_squared_error(train_labels, prediction_training, squared=False)
    errors_training = prediction_training.flatten() - train_labels.flatten()
    print('stop training')
    t_lst = []
    t_lst.append(['Training', '{:.4f}'.format(rmse_train), '{:.4f}'.format(errors_training.mean()), '{:.4f}'.format(np.median(errors_training)), '{:.4f}'.format(errors_training.std())])

    rmse_test = mean_squared_error(test_labels.flatten(), prediction.flatten(), squared=False)
    errors_testing = prediction.flatten() - test_labels.flatten()
    t_lst.append(['Testing', '{:.4f}'.format(rmse_test), '{:.4f}'.format(errors_testing.mean()), '{:.4f}'.format(np.median(errors_testing)), '{:.4f}'.format(errors_testing.std())])

    prediction_all = scaler_label.inverse_transform(regressor.predict(features).reshape(-1, 1))
    # prediction_all = regressor.predict(features).reshape(-1, 1)
    errors_all = prediction_all.flatten() - labels.flatten()
    rmse_all = mean_squared_error(labels, prediction_all, squared=False)
    t_lst.append(['All', '{:.4f}'.format(rmse_all), '{:.4f}'.format(errors_all.mean()), '{:.4f}'.format(np.median(errors_all)), '{:.4f}'.format(errors_all.std())])
    headers = [' ', 'RMSE', 'mean', 'median', 'STD']
    string = tt.to_string(
        t_lst,
        header=headers,
        style=tt.styles.ascii_thin_double,
        padding=(0,1),
        alignment='c'*len(headers)
    )
    print(string)
    tmp_df = pd.DataFrame(t_lst, columns=headers)
    tmp_path = save_path + 'training_analysis.csv'
    tmp_df.to_csv(tmp_path)

    return regressor, scaler, scaler_label, prediction_all, train_indices, test_indices


def regressor_krr(features, labels, kernel_func, save_path):
    print(f"performing Kernel Ridge Regression with {kernel_func} kernel function ...")
    n_samples = np.max(labels.shape)
    indices = np.arange(n_samples)
    train_features, test_features, train_labels, test_labels, train_indices, test_indices = \
        data_split(features, labels, indices, 0.25)
    
    dh = train_features[:, 2].flatten() - train_labels.flatten()
    weights = np.exp(-abs(dh)/(50**2))
    # plt.figure()
    # plt.scatter(abs(dh), weights)
    # plt.show()
    
    scaler = StandardScaler()
    scaler.fit(train_features)
    train_features = scaler.transform(train_features)
    # # norm = Normalizer().fit(train_features)
    # # train_features = norm.transform(train_features)
    test_features = scaler.transform(test_features)
    features = scaler.transform(features)

    scaler_label = StandardScaler().fit(train_labels.reshape(-1, 1))
    train_labels = scaler_label.transform(train_labels.reshape(-1, 1))
    # norm_label = Normalizer().fit(train_labels)
    # train_labels = norm_label.transform(train_labels)
    # train_labels = train_labels.flatten()

    print('start training')
    start_time = time.time()
    if kernel_func == 'rbf':
        from sklearn.kernel_approximation import RBFSampler
        # rbf_transformer = RBFSampler(gamma=1e-4, random_state=1)
        # train_features = rbf_transformer.fit_transform(train_features)
        # test_features = rbf_transformer.fit_transform(test_features)
        # features = rbf_transformer.fit_transform(features)

        # kr = make_pipeline(SplineTransformer(n_knots=5, degree=1), RBFSampler(gamma=1e-4, random_state=1), Ridge(alpha=1e-3))
        # kwargs = {kr.steps[-1][0] + '__sample_weight': weights}
        # kr.fit(train_features, train_labels, **kwargs)

        # kr = GridSearchCV(KernelRidge(kernel="linear"),
        #                 param_grid={"alpha": [1e0, 0.1, 1e-2, 1e-3]})
        kr = GridSearchCV(KernelRidge(kernel="rbf"),
                        param_grid={"alpha": [0.1, 1e-2, 1e-3]  , "gamma": np.logspace(-6, 0.5, 3)}, verbose=0)
        
        kr.fit(train_features, train_labels, sample_weight=weights)
        kr = KernelRidge(alpha=kr.best_estimator_.alpha, gamma=kr.best_estimator_.gamma)
        kr.fit(train_features, train_labels, sample_weight=weights)
    elif kernel_func == 'custom':
        print('using self-defined kernel')
        kr = GridSearchCV(KernelRidge(kernel=general.my_kernel),
                        param_grid={"alpha": [0.1]})
        # kr = GridSearchCV(KernelRidge(kernel="rbf", gamma=0.1),
        #                   param_grid={"alpha": [1e0, 0.1, 1e-2, 1e-3], "gamma": np.logspace(-2, 2, 5)})
        kr.fit(train_features, train_labels, sample_weight=weights)
        # kr.fit(train_features, train_labels)
        print(f"Best KRR with params: {kr.best_params_} and R2 score: {kr.best_score_:.3f}")
        kr = KernelRidge(alpha=kr.best_estimator_.alpha, gamma=kr.best_estimator_.gamma)
        kr.fit(train_features, train_labels, sample_weight=weights)
    
    regressor = kr

    train_labels = scaler_label.inverse_transform(train_labels)
    prediction = scaler_label.inverse_transform(regressor.predict(test_features).reshape(-1, 1))
    elapsed_time = time.time() - start_time
    
    prediction_training = scaler_label.inverse_transform(regressor.predict(train_features).reshape(-1, 1))
    rmse_train = mean_squared_error(train_labels, prediction_training, squared=False)
    errors_training = prediction_training.flatten() - train_labels.flatten()
    print('stop training')
    t_lst = []
    t_lst.append(['Training', '{:.4f}'.format(rmse_train), '{:.4f}'.format(errors_training.mean()), '{:.4f}'.format(np.median(errors_training)), '{:.4f}'.format(errors_training.std())])

    rmse_test = mean_squared_error(test_labels.flatten(), prediction.flatten(), squared=False)
    errors_testing = prediction.flatten() - test_labels.flatten()
    t_lst.append(['Testing', '{:.4f}'.format(rmse_test), '{:.4f}'.format(errors_testing.mean()), '{:.4f}'.format(np.median(errors_testing)), '{:.4f}'.format(errors_testing.std())])

    prediction_all = scaler_label.inverse_transform(regressor.predict(features).reshape(-1, 1))
    errors_all = prediction_all.flatten() - labels.flatten()
    rmse_all = mean_squared_error(labels, prediction_all, squared=False)
    t_lst.append(['All', '{:.4f}'.format(rmse_all), '{:.4f}'.format(errors_all.mean()), '{:.4f}'.format(np.median(errors_all)), '{:.4f}'.format(errors_all.std())])
    headers = [' ', 'RMSE', 'mean', 'median', 'STD']
    string = tt.to_string(
        t_lst,
        header=headers,
        style=tt.styles.ascii_thin_double,
        padding=(0,1),
        alignment='c'*len(headers)
    )
    print(string)
    tmp_df = pd.DataFrame(t_lst, columns=headers)
    tmp_path = save_path + 'training_analysis.csv'
    tmp_df.to_csv(tmp_path)

    # r2 = r2_score(train_labels, prediction_training)
    # plot_regression_results_R(
    #     train_labels.flatten(),
    #     prediction_training.flatten(),
    #     "KRR training dataset",
    #     (r"$R^2={:.2f}$").format(
    #         r2
    #     ),
    #     elapsed_time,
    # )
    # plt.tight_layout()
    # plt.subplots_adjust(top=0.9)
    # plt.savefig(save_path+'relationship_training.png')

    # r2 = r2_score(test_labels, prediction)
    # plot_regression_results_R(
    #     test_labels.flatten(),
    #     prediction.flatten(),
    #     "KRR testing dataset",
    #     (r"$R^2={:.2f}$").format(
    #         r2
    #     ),
    #     elapsed_time,
    # )
    # plt.tight_layout()
    # plt.subplots_adjust(top=0.9)
    # plt.savefig(save_path+'relationship_testing.png')

    # r2 = r2_score(labels, prediction_all)
    # plot_regression_results_R(
    #     labels.flatten(),
    #     prediction_all.flatten(),
    #     "KRR all dataset",
    #     (r"$R^2={:.2f}$").format(
    #         r2
    #     ),
    #     elapsed_time,
    # )
    # plt.tight_layout()
    # plt.subplots_adjust(top=0.9)
    # plt.savefig(save_path+'relationship_all.png')
    # if kernel_func == 'rbf':
    #     return regressor, scaler, scaler_label, prediction_all, train_indices, test_indices, rbf_transformer
    # else:
    return regressor, scaler, scaler_label, prediction_all, train_indices, test_indices


def regressor_gwr(XY, features, labels, save_path):
    labels = labels.reshape(-1, 1)
    np.random.seed(908)
    start_time = time.time()
    # gwr_selector = Sel_BW(XY, labels, features)
    # gwr_bw = gwr_selector.search(bw_min=2)
    gwr_bw = 12060
    print('GWR bandwidth =', gwr_bw)
    model = GWR(XY, labels, features, gwr_bw)
    # gwr_results = model.fit()
    # print(gwr_results.summary())
    # print('Mean R2 =', gwr_results.R2)
    # print('AIC =', gwr_results.aic)
    # print('AICc =', gwr_results.aicc)
    elapsed_time = time.time() - start_time

    # scale = gwr_results.scale
    # residuals = gwr_results.resid_response
    # pred_results = model.predict(XY, features, scale, residuals)
    # pred_results = model.predict(XY, features)
    # prediction_all = pred_results.predictions
    # r2 = r2_score(labels, prediction_all)
    # plot_regression_results_R(
    #     labels.flatten(),
    #     prediction_all.flatten(),
    #     "GWR all dataset",
    #     (r"$R^2={:.2f}$").format(
    #         r2
    #     ),
    #     elapsed_time,
    # )
    # plt.tight_layout()
    # plt.subplots_adjust(top=0.9)
    # plt.savefig(save_path+'relationship_all.png')

    return GWR(XY, labels, features, gwr_bw, fixed=False)


def regressor_polyn(features, labels, degree, save_path):
    n_samples = np.max(labels.shape)
    indices = np.arange(n_samples)
    train_features, test_features, train_labels, test_labels, train_indices, test_indices = \
        data_split(features, labels, indices, 0.25)
    
    scaler = StandardScaler()
    scaler.fit(train_features)
    train_features = scaler.transform(train_features)
    test_features = scaler.transform(test_features)
    features = scaler.transform(features)

    scaler_label = StandardScaler().fit(train_labels.reshape(-1, 1))
    train_labels = scaler_label.transform(train_labels.reshape(-1, 1))

    start_time = time.time()

    regressor = Pipeline([('poly', PolynomialFeatures(degree=degree)),
                      ('linear', LinearRegression(fit_intercept=True))])
    regressor.fit(train_features, train_labels)
    train_labels = scaler_label.inverse_transform(train_labels)
    prediction = scaler_label.inverse_transform(regressor.predict(test_features).reshape(-1, 1))
    # prediction = regressor.predict(test_features).reshape(-1, 1)
    elapsed_time = time.time() - start_time
    
    prediction_training = scaler_label.inverse_transform(regressor.predict(train_features).reshape(-1, 1))
    # prediction_training = regressor.predict(train_features).reshape(-1, 1)
    rmse_train = mean_squared_error(train_labels, prediction_training, squared=False)
    errors_training = prediction_training.flatten() - train_labels.flatten()
    print('stop training')
    t_lst = []
    t_lst.append(['Training', '{:.4f}'.format(rmse_train), '{:.4f}'.format(errors_training.mean()), '{:.4f}'.format(np.median(errors_training)), '{:.4f}'.format(errors_training.std())])

    rmse_test = mean_squared_error(test_labels.flatten(), prediction.flatten(), squared=False)
    errors_testing = prediction.flatten() - test_labels.flatten()
    t_lst.append(['Testing', '{:.4f}'.format(rmse_test), '{:.4f}'.format(errors_testing.mean()), '{:.4f}'.format(np.median(errors_testing)), '{:.4f}'.format(errors_testing.std())])

    prediction_all = scaler_label.inverse_transform(regressor.predict(features).reshape(-1, 1))
    # prediction_all = regressor.predict(features).reshape(-1, 1)
    errors_all = prediction_all.flatten() - labels.flatten()
    rmse_all = mean_squared_error(labels, prediction_all, squared=False)
    t_lst.append(['All', '{:.4f}'.format(rmse_all), '{:.4f}'.format(errors_all.mean()), '{:.4f}'.format(np.median(errors_all)), '{:.4f}'.format(errors_all.std())])
    headers = [' ', 'RMSE', 'mean', 'median', 'STD']
    string = tt.to_string(
        t_lst,
        header=headers,
        style=tt.styles.ascii_thin_double,
        padding=(0,1),
        alignment='c'*len(headers)
    )
    print(string)
    tmp_df = pd.DataFrame(t_lst, columns=headers)
    tmp_path = save_path + 'training_analysis.csv'
    tmp_df.to_csv(tmp_path)

    r2 = r2_score(train_labels, prediction_training)
    plot_regression_results_R(
        train_labels.flatten(),
        prediction_training.flatten(),
        "Polynomial training dataset",
        (r"$R^2={:.2f}$").format(
            r2
        ),
        elapsed_time,
    )
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.savefig(save_path+'relationship_training.png')

    r2 = r2_score(test_labels, prediction)
    plot_regression_results_R(
        test_labels.flatten(),
        prediction.flatten(),
        "Polynomial testing dataset",
        (r"$R^2={:.2f}$").format(
            r2
        ),
        elapsed_time,
    )
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.savefig(save_path+'relationship_testing.png')

    r2 = r2_score(labels, prediction_all)
    plot_regression_results_R(
        labels.flatten(),
        prediction_all.flatten(),
        "SVR all dataset",
        (r"$R^2={:.2f}$").format(
            r2
        ),
        elapsed_time,
    )
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.savefig(save_path+'relationship_all.png')

    return regressor, scaler, scaler_label, prediction_all, train_indices, test_indices


def regressor_linear(features, labels, save_path):
    n_samples = np.max(labels.shape)
    indices = np.arange(n_samples)
    train_features, test_features, train_labels, test_labels, train_indices, test_indices = \
        data_split(features, labels, indices, 0.25)
    
    scaler = StandardScaler()
    scaler.fit(train_features)
    train_features = scaler.transform(train_features)
    test_features = scaler.transform(test_features)
    features = scaler.transform(features)

    scaler_label = StandardScaler().fit(train_labels.reshape(-1, 1))
    train_labels = scaler_label.transform(train_labels.reshape(-1, 1))

    start_time = time.time()

    regressor = LinearRegression().fit(train_features, train_labels)
    train_labels = scaler_label.inverse_transform(train_labels)
    prediction = scaler_label.inverse_transform(regressor.predict(test_features).reshape(-1, 1))
    # prediction = regressor.predict(test_features).reshape(-1, 1)
    elapsed_time = time.time() - start_time
    
    prediction_training = scaler_label.inverse_transform(regressor.predict(train_features).reshape(-1, 1))
    # prediction_training = regressor.predict(train_features).reshape(-1, 1)
    rmse_train = mean_squared_error(train_labels, prediction_training, squared=False)
    errors_training = prediction_training.flatten() - train_labels.flatten()
    print('stop training')
    t_lst = []
    t_lst.append(['Training', '{:.4f}'.format(rmse_train), '{:.4f}'.format(errors_training.mean()), '{:.4f}'.format(np.median(errors_training)), '{:.4f}'.format(errors_training.std())])

    rmse_test = mean_squared_error(test_labels.flatten(), prediction.flatten(), squared=False)
    errors_testing = prediction.flatten() - test_labels.flatten()
    t_lst.append(['Testing', '{:.4f}'.format(rmse_test), '{:.4f}'.format(errors_testing.mean()), '{:.4f}'.format(np.median(errors_testing)), '{:.4f}'.format(errors_testing.std())])

    prediction_all = scaler_label.inverse_transform(regressor.predict(features).reshape(-1, 1))
    # prediction_all = regressor.predict(features).reshape(-1, 1)
    errors_all = prediction_all.flatten() - labels.flatten()
    rmse_all = mean_squared_error(labels, prediction_all, squared=False)
    t_lst.append(['All', '{:.4f}'.format(rmse_all), '{:.4f}'.format(errors_all.mean()), '{:.4f}'.format(np.median(errors_all)), '{:.4f}'.format(errors_all.std())])
    headers = [' ', 'RMSE', 'mean', 'median', 'STD']
    string = tt.to_string(
        t_lst,
        header=headers,
        style=tt.styles.ascii_thin_double,
        padding=(0,1),
        alignment='c'*len(headers)
    )
    print(string)
    tmp_df = pd.DataFrame(t_lst, columns=headers)
    tmp_path = save_path + 'training_analysis.csv'
    tmp_df.to_csv(tmp_path)

    r2 = r2_score(train_labels, prediction_training)
    plot_regression_results_R(
        train_labels.flatten(),
        prediction_training.flatten(),
        "Linear training dataset",
        (r"$R^2={:.2f}$").format(
            r2
        ),
        elapsed_time,
    )
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.savefig(save_path+'relationship_training.png')

    r2 = r2_score(test_labels, prediction)
    plot_regression_results_R(
        test_labels.flatten(),
        prediction.flatten(),
        "Linear testing dataset",
        (r"$R^2={:.2f}$").format(
            r2
        ),
        elapsed_time,
    )
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.savefig(save_path+'relationship_testing.png')

    r2 = r2_score(labels, prediction_all)
    plot_regression_results_R(
        labels.flatten(),
        prediction_all.flatten(),
        "Linear all dataset",
        (r"$R^2={:.2f}$").format(
            r2
        ),
        elapsed_time,
    )
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.savefig(save_path+'relationship_all.png')

    return regressor, scaler, scaler_label, prediction_all, train_indices, test_indices

def regressor_spline(features, labels, save_path):
    n_samples = np.max(labels.shape)
    indices = np.arange(n_samples)
    train_features, test_features, train_labels, test_labels, train_indices, test_indices = \
        data_split(features, labels, indices, 0.25)
    
    dh = train_features[:, 2].flatten() - train_labels.flatten()
    weights = np.exp(-abs(dh)/(50**2))

    # train_features = train_features[:, 1:]
    # test_features = test_features[:, 1:]
    # features = features[:, 1:]
    
    scaler = StandardScaler()
    scaler.fit(train_features)
    train_features = scaler.transform(train_features)
    test_features = scaler.transform(test_features)
    features = scaler.transform(features)

    scaler_label = StandardScaler().fit(train_labels.reshape(-1, 1))
    train_labels = scaler_label.transform(train_labels.reshape(-1, 1))

    start_time = time.time()

    # pipe = make_pipeline(SplineTransformer(), Ridge(alpha=1e-3))
    # spline = GridSearchCV(pipe(),
    #                     param_grid={"n_knots": np.linspace(2, 10, 5), "degree": np.linspace(1, 10, 5)}, scoring='neg_root_mean_squared_error', verbose=3)
    # spline.fit(train_features, train_labels)
    # print(f"Best SVR with params: {spline.best_params_} and RMSE score: {spline.best_score_:.3f}")
    # regressor = make_pipeline(SplineTransformer(n_knots=spline.best_estimator_.n_knots, degree=spline.best_estimator_.degree), Ridge(alpha=1e-3))

    regressor = make_pipeline(SplineTransformer(n_knots=5, degree=3),Ridge(alpha=1e-3))
    kwargs = {regressor.steps[-1][0] + '__sample_weight': weights}
    regressor.fit(train_features, train_labels, **kwargs)
    # regressor.fit(train_features, train_labels)
    train_labels = scaler_label.inverse_transform(train_labels)
    prediction = scaler_label.inverse_transform(regressor.predict(test_features).reshape(-1, 1))
    # prediction = regressor.predict(test_features).reshape(-1, 1)
    elapsed_time = time.time() - start_time
    
    prediction_training = scaler_label.inverse_transform(regressor.predict(train_features).reshape(-1, 1))
    # prediction_training = regressor.predict(train_features).reshape(-1, 1)
    rmse_train = mean_squared_error(train_labels, prediction_training, squared=False)
    errors_training = prediction_training.flatten() - train_labels.flatten()
    print('stop training')
    t_lst = []
    t_lst.append(['Training', '{:.4f}'.format(rmse_train), '{:.4f}'.format(errors_training.mean()), '{:.4f}'.format(np.median(errors_training)), '{:.4f}'.format(errors_training.std())])

    rmse_test = mean_squared_error(test_labels.flatten(), prediction.flatten(), squared=False)
    errors_testing = prediction.flatten() - test_labels.flatten()
    t_lst.append(['Testing', '{:.4f}'.format(rmse_test), '{:.4f}'.format(errors_testing.mean()), '{:.4f}'.format(np.median(errors_testing)), '{:.4f}'.format(errors_testing.std())])

    prediction_all = scaler_label.inverse_transform(regressor.predict(features).reshape(-1, 1))
    # prediction_all = regressor.predict(features).reshape(-1, 1)
    errors_all = prediction_all.flatten() - labels.flatten()
    rmse_all = mean_squared_error(labels, prediction_all, squared=False)
    t_lst.append(['All', '{:.4f}'.format(rmse_all), '{:.4f}'.format(errors_all.mean()), '{:.4f}'.format(np.median(errors_all)), '{:.4f}'.format(errors_all.std())])
    headers = [' ', 'RMSE', 'mean', 'median', 'STD']
    string = tt.to_string(
        t_lst,
        header=headers,
        style=tt.styles.ascii_thin_double,
        padding=(0,1),
        alignment='c'*len(headers)
    )
    print(string)
    tmp_df = pd.DataFrame(t_lst, columns=headers)
    tmp_path = save_path + 'training_analysis.csv'
    tmp_df.to_csv(tmp_path)

    # r2 = r2_score(train_labels, prediction_training)
    # plot_regression_results_R(
    #     train_labels.flatten(),
    #     prediction_training.flatten(),
    #     "Spline training dataset",
    #     (r"$R^2={:.2f}$").format(
    #         r2
    #     ),
    #     elapsed_time,
    # )
    # plt.tight_layout()
    # plt.subplots_adjust(top=0.9)
    # plt.savefig(save_path+'relationship_training.png')

    # r2 = r2_score(test_labels, prediction)
    # plot_regression_results_R(
    #     test_labels.flatten(),
    #     prediction.flatten(),
    #     "Spline testing dataset",
    #     (r"$R^2={:.2f}$").format(
    #         r2
    #     ),
    #     elapsed_time,
    # )
    # plt.tight_layout()
    # plt.subplots_adjust(top=0.9)
    # plt.savefig(save_path+'relationship_testing.png')

    # r2 = r2_score(labels, prediction_all)
    # plot_regression_results_R(
    #     labels.flatten(),
    #     prediction_all.flatten(),
    #     "Spline all dataset",
    #     (r"$R^2={:.2f}$").format(
    #         r2
    #     ),
    #     elapsed_time,
    # )
    # plt.tight_layout()
    # plt.subplots_adjust(top=0.9)
    # plt.savefig(save_path+'relationship_all.png')

    return regressor, scaler, scaler_label, prediction_all, train_indices, test_indices


def regressor_MLP(features, labels, save_path):
    n_samples = np.max(labels.shape)
    indices = np.arange(n_samples)
    train_features, test_features, train_labels, test_labels, train_indices, test_indices = \
        data_split(features, labels, indices, 0.25)
    
    scaler = StandardScaler()
    scaler.fit(train_features)
    train_features = scaler.transform(train_features)
    # # norm = Normalizer().fit(train_features)
    # # train_features = norm.transform(train_features)
    test_features = scaler.transform(test_features)
    features = scaler.transform(features)

    scaler_label = StandardScaler().fit(train_labels.reshape(-1, 1))
    train_labels = scaler_label.transform(train_labels.reshape(-1, 1))

    # norm_label = Normalizer().fit(train_labels)
    # train_labels = norm_label.transform(train_labels)
    # train_labels = train_labels.flatten()

    print('start training')
    start_time = time.time()
    regressor = MLPRegressor(random_state=1, hidden_layer_sizes=(1024, 1024, 256,256,30, 20, 10), activation='relu', alpha=1, early_stopping=True)

    regressor.fit(train_features, train_labels)
    train_labels = scaler_label.inverse_transform(train_labels)
    prediction = scaler_label.inverse_transform(regressor.predict(test_features).reshape(-1, 1))
    # prediction = regressor.predict(test_features).reshape(-1, 1)
    elapsed_time = time.time() - start_time
    
    prediction_training = scaler_label.inverse_transform(regressor.predict(train_features).reshape(-1, 1))
    # prediction_training = regressor.predict(train_features).reshape(-1, 1)
    rmse_train = mean_squared_error(train_labels, prediction_training, squared=False)
    errors_training = prediction_training.flatten() - train_labels.flatten()
    print('stop training')
    t_lst = []
    t_lst.append(['Training', '{:.4f}'.format(rmse_train), '{:.4f}'.format(errors_training.mean()), '{:.4f}'.format(np.median(errors_training)), '{:.4f}'.format(errors_training.std())])

    rmse_test = mean_squared_error(test_labels.flatten(), prediction.flatten(), squared=False)
    errors_testing = prediction.flatten() - test_labels.flatten()
    t_lst.append(['Testing', '{:.4f}'.format(rmse_test), '{:.4f}'.format(errors_testing.mean()), '{:.4f}'.format(np.median(errors_testing)), '{:.4f}'.format(errors_testing.std())])

    prediction_all = scaler_label.inverse_transform(regressor.predict(features).reshape(-1, 1))
    # prediction_all = regressor.predict(features).reshape(-1, 1)
    errors_all = prediction_all.flatten() - labels.flatten()
    rmse_all = mean_squared_error(labels, prediction_all, squared=False)
    t_lst.append(['All', '{:.4f}'.format(rmse_all), '{:.4f}'.format(errors_all.mean()), '{:.4f}'.format(np.median(errors_all)), '{:.4f}'.format(errors_all.std())])
    headers = [' ', 'RMSE', 'mean', 'median', 'STD']
    string = tt.to_string(
        t_lst,
        header=headers,
        style=tt.styles.ascii_thin_double,
        padding=(0,1),
        alignment='c'*len(headers)
    )
    print(string)
    tmp_df = pd.DataFrame(t_lst, columns=headers)
    tmp_path = save_path + 'training_analysis.csv'
    tmp_df.to_csv(tmp_path)

    # r2 = r2_score(train_labels, prediction_training)
    # plot_regression_results_R(
    #     train_labels.flatten(),
    #     prediction_training.flatten(),
    #     "MLP training dataset",
    #     (r"$R^2={:.2f}$").format(
    #         r2
    #     ),
    #     elapsed_time,
    # )
    # plt.tight_layout()
    # plt.subplots_adjust(top=0.9)
    # plt.savefig(save_path+'relationship_training.png')

    # r2 = r2_score(test_labels, prediction)
    # plot_regression_results_R(
    #     test_labels.flatten(),
    #     prediction.flatten(),
    #     "MLP testing dataset",
    #     (r"$R^2={:.2f}$").format(
    #         r2
    #     ),
    #     elapsed_time,
    # )
    # plt.tight_layout()
    # plt.subplots_adjust(top=0.9)
    # plt.savefig(save_path+'relationship_testing.png')

    # r2 = r2_score(labels, prediction_all)
    # plot_regression_results_R(
    #     labels.flatten(),
    #     prediction_all.flatten(),
    #     "MLP all dataset",
    #     (r"$R^2={:.2f}$").format(
    #         r2
    #     ),
    #     elapsed_time,
    # )
    # plt.tight_layout()
    # plt.subplots_adjust(top=0.9)
    # plt.savefig(save_path+'relationship_all.png')

    return regressor, scaler, scaler_label, prediction_all, train_indices, test_indices


def plot_regression_results(errors_original, errors_prediction, errors_reference_data, save_path):
    plt.figure(figsize=(10, 4))
    plt.style.use('seaborn-paper')
    ax1 = plt.subplot(132)
    bins = np.linspace(0, 20, 40)
    # errors = errors[np.where((np.quantile(errors, 0.95) >= errors) & (errors >= np.quantile(errors, 0.05)))[0]]

    ax1.hist(errors_prediction, bins=20, label=['prediction error'])
    mu = errors_prediction.mean()
    media = np.median(errors_prediction)
    sigma = errors_prediction.std()
    textstr = '\n'.join((
        r'$\mu=%.2f$m' % (mu,),
        r'$\sigma=%.2f$m' % (sigma,),
        r'$\mathrm{median}=%.2f$m' % (media,)))
    props = dict(boxstyle='round', facecolor='white', alpha=0.5)
    ax1.text(0.6, 0.9, textstr, transform=ax1.transAxes, fontsize=7,
            verticalalignment='top', bbox=props)
    ax1.set_xlabel('Height error (m)')
    # ax1.set_ylabel('Frequency')
    ax1.legend()

    ax2 = plt.subplot(131, sharey=ax1)
    # errors = test_features[:, 0] - test_labels
    # errors = errors[errors >= np.quantile(errors, 0.05)]
    # errors = errors[errors <= np.quantile(errors, 0.95)]
    ax2.hist(errors_original, bins=30, label=['original GEDI error'])
    ax2.set_xlabel('Height error (m)')
    ax2.set_ylabel('Frequency')
    mu = errors_original.mean()
    med = np.median(errors_original)
    sigma = errors_original.std()
    textstr = '\n'.join((
        r'$\mu=%.2f $m' % (mu,),
        r'$\sigma=%.2f $m' % (sigma,),
        r'$\mathrm{median}=%.2f $m' % (med,)))
    props = dict(boxstyle='round', facecolor='white', alpha=0.5)
    ax2.text(0.5, 0.9, textstr, transform=ax2.transAxes, fontsize=7,
             verticalalignment='top', bbox=props)
    ax2.legend()

    ax3 = plt.subplot(133)
    # atl08_z_3dep_interp = atl08_z_3dep[:, 0] * 0.3048 + np.mean(gedi_data[:, 3])
    # errors = atl08_data[atl08_z_3dep_interp >= 0, 2] - atl08_z_3dep_interp[atl08_z_3dep_interp >= 0]
    ax3.hist(errors_reference_data, bins=30, label=['ATL08 error'])
    ax3.set_xlabel('Height error (m)')
    # ax3.set_ylabel('Frequency')
    mu = errors_reference_data.mean()
    med = np.median(errors_reference_data)
    sigma = errors_reference_data.std()
    textstr = '\n'.join((
        r'$\mu=%.2f $m' % (mu,),
        r'$\sigma=%.2f $m' % (sigma,),
        r'$\mathrm{median}=%.2f $m' % (med,)))
    props = dict(boxstyle='round', facecolor='white', alpha=0.5)
    ax3.text(0.5, 0.9, textstr, transform=ax3.transAxes, fontsize=7,
             verticalalignment='top', bbox=props)
    ax3.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    # plt.show(block=True)


def trandition_interpolation(pointsXYZ, raster, save_path):
    if not os.path.exists(save_path):
            os.makedirs(save_path)
    grid_z0 = griddata(pointsXYZ[:, 0:2], pointsXYZ[:, 2], (raster.x_coords, raster.y_coords), method='nearest')
    grid_z1 = griddata(pointsXYZ[:, 0:2], pointsXYZ[:, 2], (raster.x_coords, raster.y_coords), method='linear')
    grid_z2 = griddata(pointsXYZ[:, 0:2], pointsXYZ[:, 2], (raster.x_coords, raster.y_coords), method='cubic')
    raster.save_raster(raster.empty_raster_path, grid_z0, save_path+'nearest_neighbor_interpolation_'+str(raster.resolution)+'m.tif')
    raster.save_raster(raster.empty_raster_path, grid_z0-raster.reference_raster, save_path+'nearest_neighbor_interpolation_'+str(raster.resolution)+'m_error.tif')
    raster.save_raster(raster.empty_raster_path, grid_z1, save_path+'piecewise_linear_interpolation_'+str(raster.resolution)+'m.tif')
    raster.save_raster(raster.empty_raster_path, grid_z1-raster.reference_raster, save_path+'piecewise_linear_interpolation_'+str(raster.resolution)+'m_error.tif')
    raster.save_raster(raster.empty_raster_path, grid_z2, save_path+'piecewise_cubic_interpolation_'+str(raster.resolution)+'m.tif')
    raster.save_raster(raster.empty_raster_path, grid_z2-raster.reference_raster, save_path+'piecewise_cubic_interpolation_'+str(raster.resolution)+'m_error.tif')

    xflat = np.stack((raster.x_coords.flatten(), raster.y_coords.flatten()), axis=-1)
    yflat = RBFInterpolator(pointsXYZ[:, 0:2], pointsXYZ[:, 2])(xflat)
    grid_z3 = yflat.reshape(raster.shape)
    raster.save_raster(raster.empty_raster_path, grid_z3, save_path+'rbf_interpolation_'+str(raster.resolution)+'m.tif')
    raster.save_raster(raster.empty_raster_path, grid_z3-raster.reference_raster, save_path+'rbf_interpolation_'+str(raster.resolution)+'m_error.tif')
    print('done')