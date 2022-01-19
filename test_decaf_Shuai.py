from typing import Tuple

import networkx as nx
import numpy as np
import pandas as pd
import pytest
import pytorch_lightning as pl
import torch
from sklearn.metrics import precision_score, recall_score, roc_auc_score
from utils import gen_data_nonlinear, load_adult
from sklearn.neural_network import MLPClassifier

import argparse
from DECAF import DECAF
from data import DataModule


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--h_dim", type=int, default=200)
    parser.add_argument("--lr", type=float, default=0.5e-3)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--d_updates", type=int, default=10)
    parser.add_argument("--threshold", type=float, default=0.1)
    parser.add_argument("--alpha", type=float, default=2)
    parser.add_argument("--rho", type=float, default=2)
    parser.add_argument("--l1_W", type=float, default=1e-4)
    parser.add_argument("--logfile", type=str, default="default_log.txt")
    parser.add_argument("--datasize", type=int, default=1000)
    args = parser.parse_args()

    # causal structure is in dag_seed
    dag_seed = [
    [9, 8], [3, 6], [0, 6], [9, 12], [4, 11], [0, 10], [12, 5], [8, 7], [8, 5], [4, 6], [0, 5], [0, 4], [12, 1], [4, 10], [9, 3], [2, 4], [4, 3], [0, 11], [2, 0], [1, 6], [7, 6],  [3, 5], [9, 4], [8, 1], [12, 10], [7, 12], [2, 9], [11, 10], [7, 4],  [4, 1], [13, 8], [0, 7], [9, 1], [1, 5],  [9, 0], [13, 3], [13, 2], [9, 7], [13, 4], [9, 6], [12, 4], [7, 5]
    ]
    # edge removal dictionary
    bias_dict = {9: [13]}

    #new_data = pd.read_csv("adult_data.csv")
    #new_data = new_data.values
    #X = new_data[:, :14].astype(np.uint32)
    #y = new_data[:, 14].astype(np.uint8)
    
    X,y = load_adult()
    
    baseline_clf = MLPClassifier().fit(X, y)  #Train and test uses same data??
    y_pred = baseline_clf.predict(X) # Training performance
    
    np.savetxt('X.csv', X, delimiter=',')

    print(
        "baseline scores",
        precision_score(y, y_pred),
        recall_score(y, y_pred),
        roc_auc_score(y, y_pred),
    )
    
    dm = DataModule(X)

    # sample default hyperparameters
    x_dim = dm.dims[0]
    z_dim = x_dim  # noise dimension for generator input. For the causal system, this should be equal to x_dim
    lambda_privacy = 0  # privacy used for ADS-GAN, not sure if necessary for us tbh
    lambda_gp = 10  # gradient penalisation used in WGAN-GP
    l1_g = 0  # l1 reg on sum of all parameters in generator
    weight_decay = 1e-2  # used by AdamW to regularise all network weights. Similar to L2 but for momentum-based optimization
    p_gen = (
        -1
    )  # proportion of points to generate (instead of copy from input) # Has to be negative for sequential sampling!
    use_mask = True

    # causality settings
    grad_dag_loss = False

    number_of_gpus = 0

    # model initialisation and train
    model = DECAF(
        dm.dims[0],
        dag_seed=dag_seed,
        h_dim=args.h_dim,
        lr=args.lr,
        batch_size=args.batch_size,
        lambda_privacy=lambda_privacy,
        lambda_gp=lambda_gp,
        d_updates=args.d_updates,
        alpha=args.alpha,
        rho=args.rho,
        weight_decay=weight_decay,
        grad_dag_loss=grad_dag_loss,
        l1_g=l1_g,
        l1_W=args.l1_W,
        p_gen=p_gen,
        use_mask=use_mask,
    )
    trainer = pl.Trainer(
        gpus=number_of_gpus,
        max_epochs=args.epochs,
        progress_bar_refresh_rate=1,
        profiler=False,
        callbacks=[],
    )
    trainer.fit(model, dm)
    synth_data = (
        model.gen_synthetic(
            dm.dataset.x, gen_order=model.get_gen_order(), biased_edges=bias_dict
        )
        .detach()
        .numpy()
    )
    print("Data generated successfully!")
    
    y_synth = baseline_clf.predict(synth_data)

    synth_clf = MLPClassifier().fit(synth_data, y_synth)
    y_pred = synth_clf.predict(synth_data)
    np.savetxt('synth_data.csv', synth_data, delimiter=',')
    
    print("accuracy : ",synth_clf.score(synth_data, y))
    print(
        "synth scores",
        precision_score(y_synth, y_pred),
        recall_score(y_synth, y_pred),
        roc_auc_score(y_synth, y_pred),
    )
    
