"""
Original source code for manipulating datasets was taken from the official
DECAF repository: https://github.com/vanderschaarlab/DECAF

We made small changes and added preprocessing.
"""

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from torch.utils.data import DataLoader

import logger as log


class Dataset(torch.utils.data.Dataset):
    def __init__(self, data: list) -> None:
        data = np.array(data, dtype="float32")
        self.x = torch.from_numpy(data)
        self.n_samples = self.x.shape[0]
        log.info("***** DATA ****")
        log.info(f"n_samples = {self.n_samples}")

    def __getitem__(self, index: int) -> Any:
        return self.x[index]

    def __len__(self) -> int:
        return self.n_samples


class DataModule(pl.LightningDataModule):
    def __init__(
        self,
        data: list,
        data_dir: Path = Path.cwd(),
        batch_size: int = 64,
        num_workers: int = 0,
    ) -> None:
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.dataset = Dataset(data)
        self.dims = self.dataset.x.shape[1:]

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.data_val, batch_size=self.batch_size, num_workers=self.num_workers
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.data_test, batch_size=self.batch_size, num_workers=self.num_workers
        )


def load_adult() -> pd.DataFrame:
    """Load the Adult dataset in a pandas dataframe"""

    path = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
    test_path = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test"

    names = [
        "age",
        "workclass",
        "fnlwgt",
        "education",
        "education-num",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "capital-gain",
        "capital-loss",
        "hours-per-week",
        "native-country",
        "income",
    ]
    train_df = pd.read_csv(path, names=names, index_col=False)
    test_df = pd.read_csv(test_path, names=names, index_col=False)[1:]
    df = pd.concat([train_df, test_df])
    df = df.applymap(lambda x: x.strip() if type(x) is str else x)

    for col in df:
        if df[col].dtype == "object":
            df = df[df[col] != "?"]
    
    df["income"].replace({'<=50K.': '<=50K', '>50K.': '>50K'}, inplace=True)

    return df


def preprocess_adult(dataset: pd.DataFrame) -> pd.DataFrame:
    """Preprocess adult data set."""

    replace = [
        [
            "Private",
            "Self-emp-not-inc",
            "Self-emp-inc",
            "Federal-gov",
            "Local-gov",
            "State-gov",
            "Without-pay",
            "Never-worked",
        ],
        [
            "Bachelors",
            "Some-college",
            "11th",
            "HS-grad",
            "Prof-school",
            "Assoc-acdm",
            "Assoc-voc",
            "9th",
            "7th-8th",
            "12th",
            "Masters",
            "1st-4th",
            "10th",
            "Doctorate",
            "5th-6th",
            "Preschool",
        ],
        [
            "Married-civ-spouse",
            "Divorced",
            "Never-married",
            "Separated",
            "Widowed",
            "Married-spouse-absent",
            "Married-AF-spouse",
        ],
        [
            "Tech-support",
            "Craft-repair",
            "Other-service",
            "Sales",
            "Exec-managerial",
            "Prof-specialty",
            "Handlers-cleaners",
            "Machine-op-inspct",
            "Adm-clerical",
            "Farming-fishing",
            "Transport-moving",
            "Priv-house-serv",
            "Protective-serv",
            "Armed-Forces",
        ],
        [
            "Wife",
            "Own-child",
            "Husband",
            "Not-in-family",
            "Other-relative",
            "Unmarried",
        ],
        ["White", "Asian-Pac-Islander", "Amer-Indian-Eskimo", "Other", "Black"],
        ["Female", "Male"],
        [
            "United-States",
            "Cambodia",
            "England",
            "Puerto-Rico",
            "Canada",
            "Germany",
            "Outlying-US(Guam-USVI-etc)",
            "India",
            "Japan",
            "Greece",
            "South",
            "China",
            "Cuba",
            "Iran",
            "Honduras",
            "Philippines",
            "Italy",
            "Poland",
            "Jamaica",
            "Vietnam",
            "Mexico",
            "Portugal",
            "Ireland",
            "France",
            "Dominican-Republic",
            "Laos",
            "Ecuador",
            "Taiwan",
            "Haiti",
            "Columbia",
            "Hungary",
            "Guatemala",
            "Nicaragua",
            "Scotland",
            "Thailand",
            "Yugoslavia",
            "El-Salvador",
            "Trinadad&Tobago",
            "Peru",
            "Hong",
            "Holand-Netherlands",
        ],
        [">50K", "<=50K"],
    ]

    df = dataset

    for row in replace:
        df = df.replace(row, range(len(row)))

    df = pd.DataFrame(MinMaxScaler().fit_transform(df),
                      index=df.index, columns=df.columns)

    return df


def load_credit() -> pd.DataFrame:
    """Load the Credit dataset."""
    
    path = "https://archive.ics.uci.edu/ml/machine-learning-databases/credit-screening/crx.data"
    names = ['male', 'age', 'debt', 'married', 'bankcustomer', 'educationlevel',
             'ethnicity', 'yearsemployed', 'priordefault', 'employed',
             'creditscore', 'driverslicense', 'citizen', 'zip', 'income',
             'approved']

    df = pd.read_csv(path, names=names, index_col=False)
    df.reset_index(drop=True, inplace=True) 
    df = df.dropna(how='all')
    df = df[df.age != '?']

    return df


def preprocess_credit(dataset: pd.DataFrame) -> pd.DataFrame:
    cat_features = ['male', 'married','bankcustomer', 'educationlevel',
                    'ethnicity','priordefault', 'employed', 'driverslicense',
                    'citizen', 'zip', 'approved']
    for feat in cat_features:
        dataset[feat] = LabelEncoder().fit_transform(dataset[feat])
    dataset['age'] = pd.to_numeric(dataset['age'], errors='coerce')

    # binarise protected variable
    dataset.loc[dataset['ethnicity'] <= 4, 'ethnicity'] = 0
    dataset.loc[dataset['ethnicity'] > 4, 'ethnicity']= 1
    dataset.loc[dataset['ethnicity'] == 1 , 'employed'] =  1

    dataset[dataset.columns] = MinMaxScaler().fit_transform(dataset)

    return dataset


def inject_synth_bias(dataset: pd.DataFrame, bias=0.2) -> pd.DataFrame:
    """Inject synthetic bias into a dataset."""

    biased_dataset = dataset.copy()
    biased_dataset.loc[biased_dataset['ethnicity'] > 0.5, 'approved'] = np.logical_and(
        biased_dataset.loc[biased_dataset['ethnicity'] > 0.5, 'approved'].values,
        np.random.binomial(
            1, bias,
            len(biased_dataset.loc[biased_dataset['ethnicity'] > 0.5, 'approved'])
        )
    ).astype(int)

    return biased_dataset
