from enum import Enum, auto
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sdv.single_table import GaussianCopulaSynthesizer, CopulaGANSynthesizer
from sdv.metadata import SingleTableMetadata
from sdv.evaluation.single_table import run_diagnostic, evaluate_quality, get_column_plot
from sdv.sampling import Condition


class Synthesizer(Enum):
    GAN = auto()
    GAUSSIAN = auto()


def get_balanced_data(
    X_train,
    y_train,
    nfolds=5,
    synthesize_data=True,
    synthesizer_type=Synthesizer.GAN,
    default_distribution="beta",
    numerical_disributions=None,
    cuda=False,
):
    data_train = pd.concat([X_train, y_train], axis=1)
    synthesizer = get_synthesizer(
        data_train, synthesizer_type, default_distribution, numerical_disributions, cuda=cuda
    )

    skf = StratifiedKFold(n_splits=nfolds, shuffle=True, random_state=42)
    folds_idx = []
    start = 0
    # for fold, (train_index, test_index) in enumerate(skf.split(X_train, y_train), 1):
    for fold, (train_index, test_index) in enumerate(skf.split(X_train, y_train)):
        print(fold)
        X_train_cv = X_train.iloc[train_index]
        y_train_cv = y_train.iloc[train_index]
        X_test_cv = X_train.iloc[test_index]
        y_test_cv = y_train.iloc[test_index]
        data_train_cv = pd.concat((X_train_cv, y_train_cv), axis=1)
        data_test_cv = pd.concat((X_test_cv, y_test_cv), axis=1)

        minority_data_train_cv = data_train_cv.loc[data_train_cv["readmitted"] == 1]
        unique, counts = np.unique(y_train_cv, return_counts=True)
        print("difference between minority and majority", counts[0] - counts[1])
        synthesizer.fit(minority_data_train_cv)
        # Note you might not want to make the dataset balanced but only reduce the ratio between the majority and minority classes or even oversample both minority and majority to reduce overfitting
        synthetic_data = synthesizer.sample(counts[0] - counts[1]) if synthesize_data else pd.DataFrame([])

        if fold == 0:
            data_cv_balanced = data_train_cv
        else:
            data_cv_balanced = pd.concat(
                (data_cv_balanced, data_train_cv), ignore_index=True
            )

        data_cv_balanced = pd.concat(
            (data_cv_balanced, synthetic_data), ignore_index=True
        )

        folds_idx.append(
            (
                np.arange(
                    start, start + data_train_cv.shape[0] + synthetic_data.shape[0]
                ),
                np.arange(
                    start + data_train_cv.shape[0] + synthetic_data.shape[0],
                    start
                    + data_train_cv.shape[0]
                    + synthetic_data.shape[0]
                    + data_test_cv.shape[0],
                ),
            )
        )
        data_cv_balanced = pd.concat((data_cv_balanced, data_test_cv), ignore_index = True)
        start =+ data_cv_balanced.shape[0]

    return data_cv_balanced, folds_idx


def get_metadata(data_train):
    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(data_train)
    metadata.validate()
    metadata.validate_data(data=data_train)
    return metadata


def get_synthesizer(
    data, synthesizer_type, default_distribution="beta", numerical_disributions=None, cuda=False
):
    if synthesizer_type == Synthesizer.GAUSSIAN:
        synthesizer = get_gaussian_synthesizer(
            data, get_metadata(data), default_distribution, numerical_disributions
        )
    elif synthesizer_type == Synthesizer.GAN:
        synthesizer = get_gan_synthesizer(data, get_metadata(data), cuda)

    return synthesizer


def get_gaussian_synthesizer(
    real_data, metadata, default_distribution="beta", numerical_disributions=None
):
    synthesizer = GaussianCopulaSynthesizer(
        metadata,
        default_distribution=default_distribution,
        numerical_disributions=numerical_disributions,
    )  # Creating the synthesizer

    synthesizer.fit(real_data)  # training the synthesizer

    evaluate_synthetic_data(synthesizer, real_data, metadata)

    return synthesizer


def get_gan_synthesizer(real_data, metadata, cuda=False):
    synthesizer = CopulaGANSynthesizer(metadata, cuda=cuda)
    synthesizer.fit(real_data)

    evaluate_synthetic_data(synthesizer, real_data, metadata)

    return synthesizer


def evaluate_synthetic_data(synthesizer, real_data, metadata):
    synthetic_data = synthesizer.sample(num_rows=100)

    diagnostic = run_diagnostic(
        real_data=real_data, synthetic_data=synthetic_data, metadata=metadata
    )

    quality_report = evaluate_quality(real_data, synthetic_data, metadata)

    print(quality_report.get_details("Column Shapes"))
    plot_column(
        real_data,
        synthetic_data,
        metadata,
        "discharge_disposition_id_Discharged to Home",
    )
    plot_column(real_data, synthetic_data, metadata, "number_inpatient")


def plot_column(real_data, synthetic_data, metadata, column_name):
    fig = get_column_plot(
        real_data=real_data,
        synthetic_data=synthetic_data,
        column_name=column_name,
        metadata=metadata,
    )

    fig.show()


def get_performance_measures(dataset, real_data):
    X_train_balanced = dataset.drop(["readmitted"], axis=1)  # drop labels for X train
    y_train_balanced = dataset[["readmitted"]].copy()

    train_without_sample_synthesis = real_data
