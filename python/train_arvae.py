import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns

import t_VAE
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TestTubeLogger

sns.set(font_scale=1.25)


def train(data, hyper_params, input_checkpoint_path=None, output_checkpoint_path='ar_vae.ckpt', M_N=1):
    # TRAINING FUNCTION
    max_epochs = hyper_params['epochs']
    lag = hyper_params['lag']
    latent_dim = hyper_params['latent_dim']
    hidden_dims = hyper_params['hidden_dims']

    vae_model = t_VAE.AR_VAE(lag=lag,
                             latent_dim=latent_dim,
                             X=torch.tensor(data).float(),
                             hidden_dims=hidden_dims,
                             ).float()
    if input_checkpoint_path is not None:
        vae_model = t_VAE.AR_VAE.load_from_checkpoint(input_checkpoint_path,
                                                      lag=lag,
                                                      latent_dim=latent_dim,
                                                      X=torch.tensor(data).float(),
                                                      hidden_dims=hidden_dims,
                                                      ).float()

    print('Loss Before Training')
    res = vae_model.forward(torch.tensor(data).float())
    print(vae_model.loss_function(*res, M_N=1))

    # Logger
    tt_logger = TestTubeLogger(
        save_dir=os.getcwd(),
        name='t_VAE_log',
        debug=False,
        create_git_tag=False)

    # Trainer
    runner = Trainer(max_epochs=max_epochs,
                     logger=tt_logger)

    runner.fit(vae_model)

    runner.save_checkpoint(output_checkpoint_path)

    print('Loss After Training')
    res = vae_model.forward(torch.tensor(data).float())
    print(vae_model.loss_function(*res, M_N=M_N))

    return vae_model, runner


# Function for Fetching the Interpretable Transformation Map
def fetch_ITM(vae_model):
    weights = None
    bias = None
    i = 0
    for p in vae_model.final_layer.parameters():
        if i == 0:
            weights = p
        else:
            bias = p
        i += 1
    return weights[:, :], bias[:]


# Function to encode Interventions
def intervene_raw(target_idx, feature_idx, bias, intervention, checkpoint_path, hyper_params, data):
    lag = hyper_params['lag']
    latent_dim = hyper_params['latent_dim']
    hidden_dims = hyper_params['hidden_dims']
    vae_model_intv = t_VAE.AR_VAE.load_from_checkpoint(checkpoint_path,
                                                       lag=lag,
                                                       latent_dim=latent_dim,
                                                       hidden_dims=hidden_dims,
                                                       X=torch.tensor(data).float(),
                                                       ).float()
    w_i, b_i = fetch_ITM(vae_model_intv)
    print(w_i.shape)
    print(w_i[target_idx, :][:, feature_idx])
    if bias:
        intv = intervention((w_i[target_idx, :][:, feature_idx], b_i[target_idx]))
    else:
        intv = intervention(w_i[target_idx, :][:, feature_idx])
    if bias:
        b_i[target_idx] = intv[1]
        for i in range(len(target_idx)):
            for j in range(len(feature_idx)):
                w_i[target_idx[i], feature_idx[j]] = intv[0][i, j]
        print((w_i[target_idx, :][:, feature_idx], b_i[target_idx]))
    else:
        for i in range(len(target_idx)):
            for j in range(len(feature_idx)):
                w_i[target_idx[i], feature_idx[j]] = intv[0][i, j]
        print(w_i[target_idx, :][:, feature_idx])
    return vae_model_intv


# Function to generate example samples
def generate_example_sample(data, vae_model, post_intervention_vae_model=None, T0=None, eps=None):
    # DATA -> PI -> Z -> W -> S(prime) ###
    pi = vae_model.encode(torch.tensor(data).float())  # calculate vector in sampling space; pi = (mean, log variance)
    if eps is None:
        mu, logvar = pi
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
    Z = vae_model.reparameterize(pi, eps)  # drawing sample from the sampling space

    S_init = torch.tensor(data[0:vae_model.lag, :, :]).float()  # initializing lag terms

    T = Z.shape[0]  # length of timeseries
    B = Z.shape[1]  # number of bundles
    S = S_init

    # Hardcoding few dimensions in latent space
    peaks = vae_model.make_peak(T, B)
    trend = vae_model.make_trend(T, B, 0.005)
    seasonality_y = vae_model.make_seasonality(T, B, 365)
    seasonality_m = vae_model.make_seasonality(T, B, 30)
    w1 = torch.cat((peaks, trend, seasonality_y, seasonality_m), axis=2)

    # LOOPING OVER EACH STEP
    for t in range(T):
        # z = torch.cat([Z[t]] + [S[-i] for i in range(vae_model.lag)], dim=1)
        z = Z[t]
        result = vae_model.decoder_input(z)  # calculating input to decoder
        w = vae_model.decoder(result)  # calculating latent vector
        # concat latent vector with addition trends (seasonality+peaks+linear)
        w_appended = torch.cat((w, w1[t, :, :]), axis=1)
        w_lag = torch.cat([w_appended] + [S[-i] for i in range(vae_model.lag)], dim=1)

        # POST INTERVENTION PART
        if((post_intervention_vae_model is not None) and (T0 is not None)):
            if t >= (T0-vae_model.lag):
                if t == (T0-vae_model.lag):
                    Sprime = S.clone()  # copy the pre-intervention model
                sprime = post_intervention_vae_model.final_layer(w_lag)  # calculate the outcome
                # collect the outcome
                Sprime = torch.cat([Sprime, sprime.reshape([1, sprime.shape[0], sprime.shape[1]])])

        # PRE INTERVENTION PART/ COUNTERFACTUAL
        s = vae_model.final_layer(w_lag)
        S = torch.cat([S, s.reshape([1, s.shape[0], s.shape[1]])])

        # COLLECTING LATENT VECTORS
        if t == 0:
            W = w.reshape([1, w.shape[0], w.shape[1]])
        else:
            W = torch.cat([W, w.reshape([1, w.shape[0], w.shape[1]])])

    if((post_intervention_vae_model is not None) and (T0 is not None)):
        return S, Sprime, T0, pi, Z, W
    return S, pi, Z, W


# Function to plot data+samples
def plot(data, vae_model, post_intervention_vae_model=None, T0=None, out_folder=None):
    sns.set(font_scale=1.25)

    if((post_intervention_vae_model is not None) and (T0 is not None)):
        S, Sprime, T0, pi, Z, W = generate_example_sample(data, vae_model,
                                                          post_intervention_vae_model=post_intervention_vae_model,
                                                          T0=T0)
    else:
        S, pi, Z, W = generate_example_sample(data, vae_model)
    mu, logvar = pi
    T = mu.shape[0]
    for i in range(data.shape[1]):
        fig, ax = plt.subplots(nrows=3, ncols=2, figsize=(35, 20))

        ax[0, 0].plot(np.arange(0, T+vae_model.lag), (S).detach().numpy()[:, i, :])
        ax[0, 0].set_ylabel('Normalized (Outcome)')
        ax[0, 0].set_xlabel('Time Steps')
        ax[0, 0].title.set_text('Simulated X')

        ax[0, 1].plot(np.arange(0, T+vae_model.lag), data[:, i, :])
        ax[0, 1].set_ylabel('Normalized (Outcome)')
        ax[0, 1].set_xlabel('Time Steps')
        ax[0, 1].title.set_text('Observed X')

        ax[1, 0].plot(np.arange(vae_model.lag, vae_model.lag+T), (mu).detach().numpy()[:, i, :])
        ax[1, 0].set_ylabel('Latent Value')
        ax[1, 0].set_xlabel('Time Steps')
        ax[1, 0].title.set_text('Mean Z')

        ax[1, 1].plot(np.arange(vae_model.lag, vae_model.lag+T), (logvar).detach().numpy()[:, i, :])
        ax[1, 1].set_ylabel('Latent Value')
        ax[1, 1].set_xlabel('Time Steps')
        ax[1, 1].title.set_text('Log-Std Z')

        ax[2, 0].plot(np.arange(0, T+vae_model.lag),
                      (S).detach().numpy()[:, i, :]-(S).detach().numpy()[:, i, :], alpha=0.5)
        ax[2, 0].set_ylabel('Normalized (Outcome)')
        ax[2, 0].set_xlabel('Time Steps')
        ax[2, 0].title.set_text('Difference: X(Intervened) - X')

        ax[2, 1].plot(np.arange(vae_model.lag, T+vae_model.lag), (W).detach().numpy()[:, i, :])
        ax[2, 1].set_ylabel('Outcome (Penultimate Layer)')
        ax[2, 1].set_xlabel('Time Steps')
        ax[2, 1].title.set_text('Simulated W')

        if post_intervention_vae_model is not None:
            if T0 is not None:
                ax[0, 0].set_prop_cycle(None)
                ax[0, 0].plot(np.arange(0, T+post_intervention_vae_model.lag),
                              (Sprime).detach().numpy()[:, i, :])
                ax[0, 0].axvline(x=T0)
                ax[0, 0].set_prop_cycle(None)
                ax[2, 0].plot(np.arange(0, T+post_intervention_vae_model.lag),
                              (Sprime).detach().numpy()[:, i, :]-(S).detach().numpy()[:, i, :])
                ax[2, 0].axvline(x=T0)

        if out_folder is not None:
            fig.savefig(out_folder+'example_sample_%d.png' % (i))
