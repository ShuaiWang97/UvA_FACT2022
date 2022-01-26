import numpy as np
import pandas as pd


columns = [
    "age",
    "workclass",
    # "fnlwgt",
    "education",
    # "education-num",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    # "capital-gain",
    # "capital-loss",
    "hours-per-week",
    "native-country",
    "income",
]


def DP(mlp, X_test):
    """Calculate fairness metric DP"""

    X_test_df = pd.DataFrame(X_test, columns=columns[:-1]) 
    X_test_0 = X_test_df[X_test_df["sex"] < 0.5]
    X_test_1 = X_test_df[X_test_df["sex"] > 0.5]
    dp = abs(np.mean(mlp.predict(X_test_0)) - np.mean(mlp.predict(X_test_1)))

    return dp


def FTU(mlp, X_test):
    """Calculate fairness metric FTU"""

    X_test_df = pd.DataFrame(X_test, columns=columns[:-1])
    X_test_0 = X_test_df.assign(sex = 0)
    X_test_1 = X_test_df.assign(sex = 1)

    ftu = abs(np.mean(mlp.predict(X_test_0)) - np.mean(mlp.predict(X_test_1)))

    return ftu
