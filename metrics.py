import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier


def DP(mlp, X_test):
    """Calculate fairness metric DP"""

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

    X_test_df = pd.DataFrame(X_test, columns = names[:-1])
    X_test_0 = X_test_df.assign(sex=0)
    X_test_1 = X_test_df.assign(sex=1)
    
    dp = abs(mlp.predict_proba(X_test_0) - mlp.predict_proba(X_test_1))

    return np.sum(dp)/len(X_test)


def FTU(X_synth, y_synth, X_test):
    """Calculate fairness metric FTU"""
    X_synth_0 = np.delete(X_synth,9,1) #Delete column of protected attribute
    X_test_0 = np.delete(X_test,9,1)
    
    mlp_0 = MLPClassifier().fit(X_synth_0, y_synth)
    mlp_1 = MLPClassifier().fit(X_synth, y_synth)

    ftu = abs(mlp_0.predict_proba(X_test_0) - mlp_1.predict_proba(X_test))

    return np.sum(ftu)/len(X_test)
