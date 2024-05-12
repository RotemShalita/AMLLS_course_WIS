from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def prepare_data(X_train, X_test=None, is_test=False):
    cat_features = list(X_train.select_dtypes(["object"]).columns)
    num_features = list(X_train.select_dtypes(["int64"]).columns)

    num_prep_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("std_scaler", StandardScaler()),
        ]
    )

    full_prep_pipeline = ColumnTransformer(
        transformers=[
            ("num", num_prep_pipeline, num_features),
            ("cat", OneHotEncoder(handle_unknown='ignore'), cat_features),
        ]
    )

    if is_test:
        X_train_prepd = full_prep_pipeline.fit_transform(X_train)
        X_prepd = full_prep_pipeline.transform(X_test)    
    else:
        X_prepd = full_prep_pipeline.fit_transform(X_train)

    ohe_feat_names = list(
        full_prep_pipeline.transformers_[1][1].get_feature_names_out()
    )
    feat_names = num_features + ohe_feat_names
    
    return X_prepd, feat_names