import numpy as np
import pandas as pd
import xgboost as xgb
from feature_selection import FeatureSelectionModel, run_feature_selection
from fine_tune import Model, run_fine_tune
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (balanced_accuracy_score, f1_score, log_loss,
                             precision_score, recall_score, roc_auc_score)
from synthesizer import get_balanced_data
from train_models import run_training_all_models

from prepare_data import prepare_data


def main():
    X_train = pd.read_csv("X_train.csv")
    X_test = pd.read_csv("X_test.csv")
    y_train = pd.read_csv("y_train.csv")
    y_test = pd.read_csv("y_test.csv")

    model = Model.XGB

    # Prepare data
    print("Preparing data")
    X_train_prepared, feature_names = prepare_data(X_train)

    # Perform feature selection
    print("Selecting features")
    X_train_selected = run_feature_selection(
        X_train_prepared, y_train, feature_names, FeatureSelectionModel.SELECT_K_BEST
    )
    used_features = X_train_selected.columns

    # Synthesize samples
    print("Synthesising data")
    data_cv_balanced, folds_idx = get_balanced_data(X_train_selected, y_train, cuda=True)

    # Train models - run the training on all models to see how they perform before hyperparam tuning
    print("Training basic models")
    run_training_all_models(
        data_cv_balanced.drop("readmitted", axis=1), data_cv_balanced["readmitted"]
    )

    # Hyper parameter tuning
    print("Tuning params")
    best_params = run_fine_tune(data_cv_balanced, folds_idx, model)

    # Test model
    print("Preparing test data")
    X_test_prepared, feature_names = prepare_data(X_train, X_test, is_test=True)
    X_test_selected = X_test_prepared[used_features]

    print("Running tests")
    run_test(model, best_params, data_cv_balanced, X_test_selected, y_test)


def run_test(model, params, data_cv_balanced, X_test, y_test):
    if model == Model.XGB:
        num_estimators = params.pop("num_estimators")

    fscores = []
    precisions = []
    recalls = []
    balanced_accuracies = []
    logloss = []
    aucs = []
    seeds = np.arange(0, 15)
    for seed in seeds:
        if model == Model.XGB:
            dtest = xgb.DMatrix(X_test)
            dtrain = xgb.DMatrix(
                data_cv_balanced.drop("readmitted", axis=1),
                data_cv_balanced["readmitted"],
            )
            final_model = xgb.train(
                dtrain=dtrain, params=params, num_boost_round=num_estimators
            )
            final_predictions = final_model.predict(dtest)
            final_predictions = np.round(final_predictions).astype(int)
        else:
            final_model = RandomForestClassifier(**params)
            final_model.fit(
                data_cv_balanced.drop("readmitted", axis=1),
                data_cv_balanced["readmitted"],
            )
            final_predictions = final_model.predict(X_test)

        fscores.append(f1_score(y_test, final_predictions))
        precisions.append(precision_score(y_test, final_predictions))
        recalls.append(recall_score(y_test, final_predictions))
        balanced_accuracies.append(balanced_accuracy_score(y_test, final_predictions))
        logloss.append(log_loss(y_test, final_predictions))
        aucs.append(roc_auc_score(y_test, final_predictions))

    print("Mean f1-score: {:.5f}".format(np.mean(fscores)))
    print("Mean precision: {:.5f}".format(np.mean(precisions)))
    print("Mean recall: {:.5f}".format(np.mean(recalls)))
    print("Mean balanced accuracies: {:.5f}".format(np.mean(balanced_accuracies)))
    print("Mean log loss: {:.5f}".format(np.mean(logloss)))
    print("Mean auc: {:.5f}".format(np.mean(aucs)))


if __name__ == "__main__":
    main()
