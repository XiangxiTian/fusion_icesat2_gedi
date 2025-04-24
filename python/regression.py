from sklearn.ensemble import RandomForestRegressor as rf
from sklearn import svm
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures, SplineTransformer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_validate, cross_val_predict
# from patsy import dmatrix
# import statsmodels.api as sm
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText
import numpy as np
import mlflow
import time
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def data_split(features, labels, indices, p_test):
    train_features, test_features, train_labels, test_labels, train_indices, test_indices = train_test_split(
        features, labels, indices, test_size=p_test, random_state=42)
    print('Training Features Shape:', train_features.shape)
    print('Training Labels Shape:', train_labels.shape)
    print('Testing Features Shape:', test_features.shape)
    print('Testing Labels Shape:', test_labels.shape)
    return train_features, test_features, train_labels, test_labels, train_indices, test_indices


def plot_regression_results_R(y_true, y_pred, title, scores, elapsed_time):
    """Scatter plot of the predicted vs true targets."""
    fig, ax = plt.subplots()
    ax.plot(
        [y_true.min(), y_true.max()], [y_true.min(), y_true.max()], "--r", linewidth=2
    )
    ax.scatter(y_true, y_pred, alpha=0.2)

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
    ax.legend([extra], [scores], loc="upper left")
    title = title + "\n Evaluation in {:.2f} seconds".format(elapsed_time)
    ax.set_title(title)


def RF_tunning(features, labels, n_trees_list):
    EXPERIMENT_NAME = "GEDI IN random forest tunning CA3"
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
        regressor = rf(n_estimators=n_trees, random_state=42)
        regressor.fit(train_features, train_labels)
        prediction = regressor.predict(test_features)
        mse = mean_squared_error(test_labels, prediction)
        mae = mean_absolute_error(test_labels, prediction)
        r2 = r2_score(test_labels, prediction)

        # Start MLflow
        RUN_NAME = f"run_{idx}"
        with mlflow.start_run(experiment_id=EXPERIMENT_ID, run_name=RUN_NAME) as run:
            # Retrieve run id
            RUN_ID = run.info.run_id
            # Track parameters
            mlflow.log_param("n_estimator", n_trees)
            # Track metrics
            mlflow.log_metric("mean_squared_error", mse)
            mlflow.log_metric("mean_absolute_error", mae)
            mlflow.log_metric("r2_score", r2)
            # Track model
            mlflow.sklearn.log_model(regressor, "regressor")


def regressor_RF(features, labels, feature_list, save_path):
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
    regressor = rf(n_estimators=500, random_state=42)
    regressor.fit(train_features, train_labels)
    prediction = regressor.predict(test_features)
    errors = (prediction - test_labels)
    elapsed_time = time.time() - start_time

    prediction_training = regressor.predict(train_features)
    rmse_train = mean_squared_error(train_labels, prediction_training, squared=False)
    errors_training = prediction_training - train_labels
    print('RMSE for training:', rmse_train)
    print('error for training:', errors_training.mean(), errors_training.std(), np.median(errors_training))

    rmse_test = mean_squared_error(test_labels, prediction, squared=False)
    errors_testing = prediction - test_labels
    print('RMSE for testing:', rmse_test)
    print('error for testing:', errors_testing.mean(), errors_testing.std(), np.median(errors_testing))

    prediction_all = regressor.predict(features)
    errors_all = prediction_all - labels
    rmse_all = mean_squared_error(labels, prediction_all, squared=False)
    print('RMSE for all:', rmse_all)
    print('error for all:', errors_all.mean(), errors_all.std(), np.median(errors_all))
    score = cross_validate(
        regressor, features, labels, scoring=["r2", "neg_mean_absolute_error"], n_jobs=2, verbose=0
    )
    plot_regression_results_R(
        labels,
        prediction_all,
        "Random forest",
        (r"$R^2={:.2f} \pm {:.2f}$" + "\n" + r"$MAE={:.2f} \pm {:.2f}$").format(
            np.mean(score["test_r2"]),
            np.std(score["test_r2"]),
            -np.mean(score["test_neg_mean_absolute_error"]),
            np.std(score["test_neg_mean_absolute_error"]),
        ),
        elapsed_time,
    )
    # plt.suptitle("Single predictors versus stacked predictors")
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    # plt.savefig(save_path)
    plt.show(block=True)


    # mean absolute error (mae)
    mae = np.mean(errors)
    print('Mean Absolute Error:', round(mae, 2), 'm.')
    # mean absolute percentage error (MAPE)
    mape = 100 * (errors / test_labels)
    accuracy = 100 - np.mean(mape)
    print('Accuracy:', round(accuracy, 2), '%.')

    # Get numerical feature importances
    importances = list(regressor.feature_importances_)
    # List of tuples with variable and importance
    # feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(feature_list, importances)]
    # # Sort the feature importances by most important first
    # feature_importances = sorted(feature_importances, key=lambda x: x[1], reverse=True)

    plt.style.use('seaborn-paper')
    x_values = list(range(len(importances)))
    plt.bar(x_values, importances, orientation='vertical')
    plt.xticks(x_values, feature_list, rotation='vertical')
    plt.ylabel('Importance')
    plt.xlabel('Variable')
    plt.title('Variable Importance')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show(block=True)
    return regressor, scaler, prediction, errors, \
           train_features, test_features, train_labels, test_labels, train_indices, test_indices


def regressor_svm(features, labels):
    n_samples = np.max(labels.shape)
    indices = np.arange(n_samples)
    train_features, test_features, train_labels, test_labels, train_indices, test_indices = \
        data_split(features, labels, indices, 0.25)
    start_time = time.time()
    regressor = svm.SVR()
    regressor.fit(train_features, train_labels)
    prediction = regressor.predict(test_features)
    errors = (prediction - test_labels)

    elapsed_time = time.time() - start_time

    prediction_training = regressor.predict(train_features)
    rmse_train = mean_squared_error(train_labels, prediction_training, squared=False)
    errors_training = prediction_training - train_labels
    print('RMSE for training:', rmse_train)
    print('error for training:', errors_training.mean(), errors_training.std(), np.median(errors_training))

    rmse_test = mean_squared_error(test_labels, prediction, squared=False)
    errors_testing = prediction - test_labels
    print('RMSE for testing:', rmse_test)
    print('error for training:', errors_testing.mean(), errors_testing.std(), np.median(errors_testing))

    prediction_all = regressor.predict(features)
    errors_all = prediction_all - labels
    rmse_all = mean_squared_error(labels, prediction_all, squared=False)
    print('RMSE for all:', rmse_all)
    print('error for all:', errors_all.mean(), errors_all.std(), np.median(errors_all))
    score = cross_validate(
        regressor, features, labels, scoring=["r2", "neg_mean_absolute_error"], n_jobs=2, verbose=0
    )
    plot_regression_results_R(
        labels,
        prediction_all,
        "SVM",
        (r"$R^2={:.2f} \pm {:.2f}$" + "\n" + r"$MAE={:.2f} \pm {:.2f}$").format(
            np.mean(score["test_r2"]),
            np.std(score["test_r2"]),
            -np.mean(score["test_neg_mean_absolute_error"]),
            np.std(score["test_neg_mean_absolute_error"]),
        ),
        elapsed_time,
    )
    # plt.suptitle("Single predictors versus stacked predictors")
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    # plt.savefig(save_path)
    plt.show(block=True)

    mae = np.mean(errors)
    print('Mean Absolute Error:', round(mae, 2), 'm.')
    # mean absolute percentage error (MAPE)
    mape = 100 * (errors / test_labels)
    accuracy = 100 - np.mean(mape)
    print('Accuracy:', round(accuracy, 2), '%.')

    return regressor, prediction, errors, \
           train_features, test_features, train_labels, test_labels, train_indices, test_indices


def regressor_polyn(features, labels):
    n_samples = np.max(labels.shape)
    indices = np.arange(n_samples)
    train_features, test_features, train_labels, test_labels, train_indices, test_indices = \
        data_split(features, labels, indices, 0.25)
    start_time = time.time()
    regressor = Pipeline([('poly', PolynomialFeatures(degree=6)),
                      ('linear', LinearRegression(fit_intercept=False))])
    regressor = regressor.fit(features, labels)
    print('coef:', regressor.named_steps['linear'].coef_)
    print('score:', regressor.score(test_features, test_labels))
    prediction = regressor.predict(test_features)
    errors = (prediction - test_labels)

    elapsed_time = time.time() - start_time

    prediction_training = regressor.predict(train_features)
    rmse_train = mean_squared_error(train_labels, prediction_training, squared=False)
    errors_training = prediction_training - train_labels
    print('RMSE for training:', rmse_train)
    print('error for training:', errors_training.mean(), errors_training.std(), np.median(errors_training))

    rmse_test = mean_squared_error(test_labels, prediction, squared=False)
    errors_testing = prediction - test_labels
    print('RMSE for testing:', rmse_test)
    print('error for training:', errors_testing.mean(), errors_testing.std(), np.median(errors_testing))

    prediction_all = regressor.predict(features)
    errors_all = prediction_all - labels
    rmse_all = mean_squared_error(labels, prediction_all, squared=False)
    print('RMSE for all:', rmse_all)
    print('error for all:', errors_all.mean(), errors_all.std(), np.median(errors_all))
    score = cross_validate(
        regressor, features, labels, scoring=["r2", "neg_mean_absolute_error"], n_jobs=2, verbose=0
    )
    plot_regression_results_R(
        labels,
        prediction_all,
        "Polynomial",
        (r"$R^2={:.2f} \pm {:.2f}$" + "\n" + r"$MAE={:.2f} \pm {:.2f}$").format(
            np.mean(score["test_r2"]),
            np.std(score["test_r2"]),
            -np.mean(score["test_neg_mean_absolute_error"]),
            np.std(score["test_neg_mean_absolute_error"]),
        ),
        elapsed_time,
    )
    # plt.suptitle("Single predictors versus stacked predictors")
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    # plt.savefig(save_path)
    plt.show(block=True)

    mse = mean_squared_error(test_labels, prediction)
    mae = mean_absolute_error(test_labels, prediction)
    r2 = r2_score(test_labels, prediction)

    mae = np.mean(errors)
    print('Mean Absolute Error:', round(mae, 2), 'm.')
    # mean absolute percentage error (MAPE)
    mape = 100 * (errors / test_labels)
    accuracy = 100 - np.mean(mape)
    print('Accuracy:', round(accuracy, 2), '%.')

    return regressor, prediction, errors, \
           train_features, test_features, train_labels, test_labels, train_indices, test_indices


def regressor_spline(features, labels):
    n_samples = np.max(labels.shape)
    indices = np.arange(n_samples)
    train_features, test_features, train_labels, test_labels, train_indices, test_indices = \
        data_split(features, labels, indices, 0.25)
    start_time = time.time()
    regressor = make_pipeline(SplineTransformer(n_knots=6, degree=3), Ridge(alpha=1e-3))
    regressor.fit(train_features, train_labels)
    prediction = regressor.predict(test_features)
    errors = (prediction - test_labels)

    elapsed_time = time.time() - start_time

    prediction_training = regressor.predict(train_features)
    rmse_train = mean_squared_error(train_labels, prediction_training, squared=False)
    errors_training = prediction_training - train_labels
    print('RMSE for training:', rmse_train)
    print('error for training:', errors_training.mean(), errors_training.std(), np.median(errors_training))

    rmse_test = mean_squared_error(test_labels, prediction, squared=False)
    errors_testing = prediction - test_labels
    print('RMSE for testing:', rmse_test)
    print('error for training:', errors_testing.mean(), errors_testing.std(), np.median(errors_testing))

    prediction_all = regressor.predict(features)
    errors_all = prediction_all - labels
    rmse_all = mean_squared_error(labels, prediction_all, squared=False)
    print('RMSE for all:', rmse_all)
    print('error for all:', errors_all.mean(), errors_all.std(), np.median(errors_all))
    score = cross_validate(
        regressor, features, labels, scoring=["r2", "neg_mean_absolute_error"], n_jobs=2, verbose=0
    )
    plot_regression_results_R(
        labels,
        prediction_all,
        "Cubic spline regression",
        (r"$R^2={:.2f} \pm {:.2f}$" + "\n" + r"$MAE={:.2f} \pm {:.2f}$").format(
            np.mean(score["test_r2"]),
            np.std(score["test_r2"]),
            -np.mean(score["test_neg_mean_absolute_error"]),
            np.std(score["test_neg_mean_absolute_error"]),
        ),
        elapsed_time,
    )
    # plt.suptitle("Single predictors versus stacked predictors")
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    # plt.savefig(save_path)
    plt.show(block=True)

    mae = np.mean(errors)
    print('Mean Absolute Error:', round(mae, 2), 'm.')
    # mean absolute percentage error (MAPE)
    mape = 100 * (errors / test_labels)
    accuracy = 100 - np.mean(mape)
    print('Accuracy:', round(accuracy, 2), '%.')

    return regressor, prediction, errors, \
           train_features, test_features, train_labels, test_labels, train_indices, test_indices


def regressor_MLP(features, labels):
    n_samples = np.max(labels.shape)
    indices = np.arange(n_samples)
    train_features, test_features, train_labels, test_labels, train_indices, test_indices = \
        data_split(features, labels, indices, 0.25)
    # train_features_raw = np.copy(train_features)
    # test_features_raw = np.copy(test_features)
    start_time = time.time()
    scaler = StandardScaler()
    scaler.fit(train_features)
    train_features = scaler.transform(train_features)
    test_features = scaler.transform(test_features)
    features = scaler.transform(features)
    regressor = MLPRegressor(random_state=1, hidden_layer_sizes=(100, 100), max_iter=10000, alpha=1.5, early_stopping=True)
    regressor.fit(train_features, train_labels)
    print('score:', regressor.score(test_features, test_labels))
    prediction = regressor.predict(test_features)
    errors = (prediction - test_labels)

    elapsed_time = time.time() - start_time

    prediction_training = regressor.predict(train_features)
    rmse_train = mean_squared_error(train_labels, prediction_training, squared=False)
    errors_training = prediction_training - train_labels
    print('RMSE for training:', rmse_train)
    print('error for training:', errors_training.mean(), errors_training.std(), np.median(errors_training))

    rmse_test = mean_squared_error(test_labels, prediction, squared=False)
    errors_testing = prediction - test_labels
    print('RMSE for testing:', rmse_test)
    print('error for training:', errors_testing.mean(), errors_testing.std(), np.median(errors_testing))

    prediction_all = regressor.predict(features)
    errors_all = prediction_all - labels
    rmse_all = mean_squared_error(labels, prediction_all, squared=False)
    print('RMSE for all:', rmse_all)
    print('error for all:', errors_all.mean(), errors_all.std(), np.median(errors_all))
    score = cross_validate(
        regressor, features, labels, scoring=["r2", "neg_mean_absolute_error"], n_jobs=2, verbose=0
    )
    plot_regression_results_R(
        labels,
        prediction_all,
        "MLP",
        (r"$R^2={:.2f} \pm {:.2f}$" + "\n" + r"$MAE={:.2f} \pm {:.2f}$").format(
            np.mean(score["test_r2"]),
            np.std(score["test_r2"]),
            -np.mean(score["test_neg_mean_absolute_error"]),
            np.std(score["test_neg_mean_absolute_error"]),
        ),
        elapsed_time,
    )
    # plt.suptitle("Single predictors versus stacked predictors")
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    # plt.savefig(save_path)
    plt.show(block=True)

    mse = mean_squared_error(test_labels, prediction)
    mae = mean_absolute_error(test_labels, prediction)
    r2 = r2_score(test_labels, prediction)

    mae = np.mean(errors)
    print('Mean Absolute Error:', round(mae, 2), 'm.')
    # mean absolute percentage error (MAPE)
    mape = 100 * (errors / test_labels)
    accuracy = 100 - np.mean(mape)
    print('Accuracy:', round(accuracy, 2), '%.')

    return regressor, scaler, prediction, errors, \
           train_features, test_features, train_labels, test_labels, train_indices, test_indices


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
    plt.show(block=True)

