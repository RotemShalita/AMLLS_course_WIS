from enum import Enum, auto
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import xgboost as xgb
import optuna 
from optuna.samplers import RandomSampler


class Model(Enum):
    XGB = auto()
    RANDOM_FORREST = auto()


LOG_LOSS = 'logloss'
AUC = 'auc'
ERROR = 'error'


def run_fine_tune(data_cv_balanced, folds_idx, model, nfolds=5, n_trials=100, device='cpu', tree_method='auto'):
    if model == Model.XGB:
        params = {"objective":"binary:logistic", 'max_depth':5, 'subsample':0.8, 'gamma':0, 'colsample_bytree':0.8, "seed":42, 'device':device, 'tree_method':tree_method}
    else:
        params = {}

    study = optuna.create_study(direction=get_eval_metric_opt(AUC), sampler=RandomSampler(seed=0))
    func = lambda trial: objective_tune_all(trial, data_cv_balanced, model, params, AUC, nfolds, folds_idx)
    study.optimize(func, n_trials= n_trials,  show_progress_bar=True, callbacks=[print_best_callback]) 

    # Get the best hyperparameters and corresponding score
    best_params = study.best_params
    best_score = study.best_value

    print("Best parameters:", best_params)
    print("Best score:", best_score)

    return best_params


def objective_tune_all(trial, data_cv_balanced, model, params_train, eval_metric, nfold, folds_idx): 
    if model == Model.XGB:
        dtrain = xgb.DMatrix(data_cv_balanced.drop('readmitted', axis=1), data_cv_balanced['readmitted'])
        params_search = {"num_estimators" :trial.suggest_int("num_estimators", 1, 1000),  
                       "max_depth" : trial.suggest_int('max_depth', 3, 20),
                       "learning_rate" :trial.suggest_float("learning_rate", 0.001, 10),
                       "subsample" :trial.suggest_float("subsample", 0.1, 1),
                       "gamma" :trial.suggest_float("gamma", 0, 5),
                       "min_child_weight": trial.suggest_int("min_child_weight", 1, 50), 
                       "lambda": trial.suggest_float("lambda", 0.0, 0.3),
                       "alpha": trial.suggest_float("alpha", 0.0, 0.3), 
                       "scale_pos_weight":trial.suggest_float("scale_pos_weight", 0.0, 1.0)  # for classification and for unbalanced datasets 
                    }
    
        params_train.update(params_search)
        num_boost_round = params_train.pop('num_estimators')
        cls_cv = xgb.cv(params=params_train, num_boost_round = num_boost_round, dtrain=dtrain,  nfold=nfold, shuffle=True,  metrics=eval_metric,  stratified=True,  seed=42, folds = folds_idx) 

        return (cls_cv.iloc[-1, 2])

    else:
        params_search = {
            'n_estimators': trial.suggest_int('n_estimators', 0, 50),
            'max_depth': trial.suggest_int('max_depth', 3, 20),
            'max_features': trial.suggest_int('max_features', 3, 10),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10)
        }
        clf = RandomForestClassifier(**params_search)

        cv_scores = cross_val_score(clf, data_cv_balanced.drop('readmitted', axis=1), data_cv_balanced['readmitted'], cv=folds_idx, scoring=eval_metric)  # cross-validation by testing indices that are not synthesized
    
        return cv_scores.mean()
    

def print_best_callback(study, trial):
    print(f"Best value: {study.best_value}, Best params: {study.best_trial.params}")


def get_eval_metric_opt(eval_metric):
    print(f'Tuning with respect to {eval_metric}')

    if eval_metric == AUC:
        return 'maximize'
    else:
        return 'minimize'