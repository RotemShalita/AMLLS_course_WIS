from enum import Enum, auto
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.feature_selection import SelectKBest, f_classif
import lightgbm as lgbm
from scipy.stats import randint


class FeatureSelectionModel(Enum):
    RANDOM_FOREST_CLASSIFIER = auto()
    LGBM_CLASSIFIER = auto()
    SELECT_K_BEST = auto()


def run_feature_selection(X_train, y_train, feat_names, model):
    if model == FeatureSelectionModel.RANDOM_FOREST_CLASSIFIER:
        X_train_top_features = select_with_random_forest(X_train, y_train, feat_names)

    elif model == FeatureSelectionModel.LGBM_CLASSIFIER:
        X_train_top_features = select_with_lgbm(X_train, y_train, feat_names)

    elif model == FeatureSelectionModel.SELECT_K_BEST:
        X_train_top_features = select_with_k_best(X_train, y_train, feat_names)

    print_correlations(X_train_top_features, y_train)
    return X_train_top_features


def print_correlations(X_train_top_features, y_train):
    for column in X_train_top_features.columns:
        print(pd.concat([X_train_top_features[column], y_train], axis=1).corr())


def select_with_random_forest(X_train, y_train, feat_names):
    param_grid = [
        {
            "n_estimators": [3, 10, 30],
            "max_features": [2, 4, 6, 8],
        },  # trying 12 (3×4) combinations of hyperparameters
        {"n_estimators": [3, 10], "max_features": [2, 3, 4]},
    ]  # then trying 6 (2×3) combinations

    forest_clf = RandomForestClassifier(random_state=42)
    # train across 5 folds, that's a total of (12+6)*5=90 rounds of training
    grid_search = GridSearchCV(forest_clf, param_grid, cv=5, scoring="f1")

    # Pipeline combining preprocessing and model training
    pipe_prep_train = Pipeline(
        steps=[
            ("train", grid_search),
        ]
    )
    # fit data and avoid leaking the test set into the train set
    pipe_prep_train.fit(X_train, y_train)

    best_est = grid_search.best_estimator_
    feat_impo = best_est.feature_importances_

    s_feat_impo = pd.Series(feat_impo, feat_names).sort_values(ascending=False)

    rf_importances = s_feat_impo[:20]
    indices = np.argsort(rf_importances)
    tick_count = list(reversed(range(len(rf_importances))))

    plt.title("Random Forest Feature Importances (seed=42)")
    plt.barh(range(len(indices)), rf_importances[indices], color="g", align="center")
    plt.yticks(tick_count, rf_importances.keys())
    plt.tight_layout(rect=(0, 0.1, 1, 1))
    plt.xlabel("Relative Importance")


def select_with_lgbm(X_train, y_train, feat_names):
    param_distribs = {
        "n_estimators": randint(low=1, high=200),
        "max_features": randint(low=1, high=8),
    }

    lgbm_clf = lgbm.LGBMClassifier(
        random_state=42, importance_type="gain", verbosity=-1
    )

    lgbm_search = RandomizedSearchCV(
        lgbm_clf,
        param_distributions=param_distribs,
        n_iter=10,
        cv=5,
        scoring="f1",
        random_state=42,
        verbose=1,
        n_jobs=1,
    )  # n_jobs = -1
    lgbm_search.fit(X_train, y_train)

    cvres = lgbm_search.cv_results_
    for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
        print(np.sqrt(-mean_score), params)

    print(lgbm_search.best_params_)

    feature_importances = lgbm_search.best_estimator_.feature_importances_

    final_model = lgbm.LGBMClassifier(
        **lgbm_search.best_params_,
        random_state=42,
        importance_type="gain",
        verbosity=-1
    )
    final_model.fit(X_train, y_train, feature_name=feat_names)
    lgbm.plot_importance(
        final_model, figsize=(7, 6), max_num_features=20, title="seed 42"
    )


def select_with_k_best(X_train, y_train, feat_names):
    selector = SelectKBest(f_classif, k=10)
    X_train_new = selector.fit_transform(X_train, y_train)

    mask = selector.get_support()
    X_train_top_features = pd.DataFrame(
        X_train_new.toarray(), columns=pd.DataFrame(feat_names)[mask][0]
    )

    return X_train_top_features
