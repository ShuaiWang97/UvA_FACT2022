"""Module containing training functions for the various models evaluated in the DECAF paper."""

import numpy as np
import pytorch_lightning as pl
from ydata_synthetic.synthesizers import ModelParameters, TrainParameters
from ydata_synthetic.synthesizers.regular import VanilllaGAN, WGAN_GP

from data import DataModule, load_adult
from models.DECAF import DECAF


def train_vanilla_gan(noise_dim=32, dim=128, batch_size=128, log_step=100,
                      epochs=10, learning_rate=5e-4, beta_1=0.5, beta_2=0.9):
    model = VanilllaGAN

    # Load data and define the data processor parameters
    data = load_adult()
    num_cols = ['age', 'fnlwgt', 'capital-gain', 'capital-loss',
                'hours-per-week']
    cat_cols = ['workclass','education', 'education-num', 'marital-status',
                'occupation', 'relationship', 'race', 'sex', 'native-country',
                'income']

    gan_args = ModelParameters(batch_size=batch_size,
                               lr=learning_rate,
                               betas=(beta_1, beta_2),
                               noise_dim=noise_dim,
                               layers_dim=dim)

    train_args = TrainParameters(epochs=epochs,
                                 sample_interval=log_step)

    synthesizer = model(gan_args)
    synthesizer.train(data=data, train_arguments=train_args,
                      num_cols=num_cols, cat_cols=cat_cols)

    return synthesizer


def train_wgan_gp(noise_dim=128, dim=128, batch_size=500, log_step=100,
                  epochs=10, learning_rate=[5e-4, 3e-3], beta_1=0.5, beta_2=0.9):
    model = WGAN_GP

    #Load data and define the data processor parameters
    data = load_adult()
    num_cols = ['age', 'fnlwgt', 'capital-gain', 'capital-loss',
                'hours-per-week']
    cat_cols = ['workclass','education', 'education-num', 'marital-status',
                'occupation', 'relationship', 'race', 'sex', 'native-country',
                'income']

    gan_args = ModelParameters(batch_size=batch_size,
                               lr=learning_rate,
                               betas=(beta_1, beta_2),
                               noise_dim=noise_dim,
                               layers_dim=dim)

    train_args = TrainParameters(epochs=epochs,
                                 sample_interval=log_step)

    synthesizer = model(gan_args, n_critic=2)
    synthesizer.train(data, train_args, num_cols, cat_cols)

    return synthesizer


def train_decaf(X_train, y_train, dag_seed, h_dim=200, lr=0.5e-3,
                batch_size=64, lambda_privacy=0, lambda_gp=10, d_updates=10,
                alpha=2, rho=2, weight_decay=1e-2, grad_dag_loss=False,
                l1_g=0, l1_W=1e-4, p_gen=-1, use_mask=True, epochs=10):
    train_data = np.column_stack((X_train, y_train))
    dm = DataModule(train_data)

    model = DECAF(
        dm.dims[0],
        dag_seed=dag_seed,
        h_dim=h_dim,
        lr=lr,
        batch_size=batch_size,
        lambda_privacy=lambda_privacy,
        lambda_gp=lambda_gp,
        d_updates=d_updates,
        alpha=alpha,
        rho=rho,
        weight_decay=weight_decay,
        grad_dag_loss=grad_dag_loss,
        l1_g=l1_g,
        l1_W=l1_W,
        p_gen=p_gen,
        use_mask=use_mask,
    )

    trainer = pl.Trainer(max_epochs=epochs, logger=False)
    trainer.fit(model, dm)

    return model, dm
