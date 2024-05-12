from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import lightgbm as lgbm
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.metrics import (
    make_scorer,
    f1_score,
    precision_score,
    recall_score,
    balanced_accuracy_score,
    roc_auc_score
)

from sklearn.tree import DecisionTreeClassifier
import xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.model_selection import cross_validate


def display_performance_scores(y_true, y_pred):
    print("f1-score: {:.5f}".format(f1_score(y_true, y_pred)))
    print("Precision: {:.5f}".format(precision_score(y_true, y_pred)))
    print("Recalls: {:.5f}".format(recall_score(y_true, y_pred)))
    print("balanced accuracy: {:.5f}".format(balanced_accuracy_score(y_true, y_pred)))
    print("auc: {:.5f}".format(roc_auc_score(y_true, y_pred)))


def run_model_training(model, X_train, y_train):
    model.fit(X_train, y_train)
    predictions = model.predict(X_train)
    display_performance_scores(y_train, predictions)


def run_model_cv(model, X_train, y_train, cv):
    scoring = {
        'precision': make_scorer(precision_score, average='weighted'),
        'recall': make_scorer(recall_score, average='weighted'),
        'f1_score': make_scorer(f1_score, average='weighted')
    }
    scores = cross_validate(model, X_train, y_train, scoring=scoring, cv=cv, return_train_score=True)
    print(scores)


def run_training_all_models(X_train, y_train):
    models = [
        LogisticRegression(),
        DecisionTreeClassifier(random_state=42),
        RandomForestClassifier(n_estimators=10, random_state=42),
        BalancedRandomForestClassifier(
            sampling_strategy="not majority", replacement=True, random_state=42
        ),
        lgbm.LGBMClassifier(random_state=42, importance_type="gain", verbosity=-1),
        xgb.XGBRFClassifier(random_state=42, importance_type="gain"),
        CatBoostClassifier(random_state=42, verbose=False),
    ]

    for model in models:
        run_model_cv(model, X_train, y_train, 10)


def model_validation_no_synthesis(model, X_train, y_train):
    scores = cross_val_score(
        balanced_forest_clf,
        X_train_without_sample_synthesis,
        y_train_without_sample_synthesis,
        scoring="f1",
        cv=10,
    )
    display_cv_scores(scores)
