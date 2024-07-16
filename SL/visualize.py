import pickle
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
import utm
from matplotlib.ticker import ScalarFormatter
from tqdm import tqdm

from train_supervised import CPANet


def single_forecast(data : dict, n_obs_est : int, n_forecasts : int, n_obs_pred : int, model : CPANet) -> tuple[np.ndarray]:
    """Performs predictions using the 'model' for trajectory prediction. 
    Selects randomly a part of a trajectory from 'data', uses 'n_obs_est' points for estimation,
    and predicts the next 'n_obs_pred' points via 'n_forecasts'-step ahead forecasts, 
    using the last 'n_obs_est' each time.

    Returns:
        np.ndarray(n_obs_est,) -> e_base 
        np.ndarray(n_obs_est,) -> n_base 
        np.ndarray(n_obs_pred,) -> e_predicted
        np.ndarray(n_obs_pred,) -> n_predicted 
        np.ndarray(n_obs_pred,) -> e_truth
        np.ndarray(n_obs_pred,) -> n_truth
    """
    assert n_obs_pred % n_forecasts == 0, "The number of points you want to estimate is not a multiple of the single-step forecasts."
    n_steps = int(n_obs_pred / n_forecasts)

    # Select traj
    while True:
        idx = random.randrange(0, len(data))
        tr = data[idx]
        tr_len = len(tr["ts"])

        if tr_len > n_obs_est + n_obs_pred:
            break
    n = tr["n"]
    e = tr["e"]
    
    tp = np.argmin(np.abs(np.diff(e)))

    # Curve
    if np.random.binomial(n=1, p=0):
        t_start = random.randrange(max([tp-200, 0]), tp)

    # Any (most likely straight though)
    else:
        t_start = random.randrange(0, tr_len-1-n_obs_est-n_obs_pred)

    n_base = n[t_start : t_start + n_obs_est]
    e_base = e[t_start : t_start + n_obs_est]

    n_pred_truth = n[t_start + n_obs_est : t_start + n_obs_est + n_obs_pred]
    e_pred_truth = e[t_start + n_obs_est : t_start + n_obs_est + n_obs_pred]

    # Iteratively compute one-step ahead forecasts
    n_for_est = n_base.copy()
    e_for_est = e_base.copy()

    n_predicted = np.array([])
    e_predicted = np.array([])

    if model is not None:

        for _ in range(n_steps):   

            if DIFF:
                # Take last 'n_obs_est' elements for prediction
                n_diff = torch.tensor(np.diff(n_for_est[-n_obs_est:]), dtype=torch.float32)
                e_diff = torch.tensor(np.diff(e_for_est[-n_obs_est:]), dtype=torch.float32)

                n_diff = torch.reshape(n_diff, (1, n_obs_est-1, 1))
                e_diff = torch.reshape(e_diff, (1, n_obs_est-1, 1))

                # Add absolute position
                pos = torch.reshape(torch.tensor([e_for_est[-1], n_for_est[-1]], dtype=torch.float32) / 100., shape=(1, 2))

                # Forward pass
                est = model(e_diff, n_diff, pos)
                e_add = est[0][:n_forecasts].detach().numpy()
                n_add = est[0][n_forecasts:].detach().numpy()
                
                new_e = e_for_est[-1] + np.cumsum(e_add)
                new_n = n_for_est[-1] + np.cumsum(n_add)     

            else:
                # Take last 'n_obs_est' elements for prediction
                n_in = torch.tensor(n_for_est[-n_obs_est:], dtype=torch.float32)
                e_in = torch.tensor(e_for_est[-n_obs_est:], dtype=torch.float32)

                n_in = torch.reshape(n_in, (1, n_obs_est, 1))
                e_in = torch.reshape(e_in, (1, n_obs_est, 1))

                # Forward pass
                est = model(e_in, n_in)
                new_e = est[0][:n_forecasts].detach().numpy()
                new_n = est[0][n_forecasts:].detach().numpy()
            
            # Append estimates
            n_predicted = np.concatenate((n_predicted, new_n))
            e_predicted = np.concatenate((e_predicted, new_e))

            # Update estimation data
            n_for_est = np.concatenate((n_for_est, new_n))
            e_for_est = np.concatenate((e_for_est, new_e))
    
    return e_base, n_base, e_predicted, n_predicted, e_pred_truth, n_pred_truth

def renorm_trajs(east:np.ndarray, north:np.ndarray) -> tuple[np.ndarray|np.ndarray]:
    norm_e, norm_n, *_  = utm.from_latlon(56.13, 10.533333)

    new_e = east + norm_e
    new_n = north + norm_n
    lat, lon = utm.to_latlon(easting=new_e, northing=new_n, zone_number=32, northern=True)
    return lat, lon

def viz_forecast(data : dict, n_obs_est : int, n_obs_pred : int, n_forecasts : int, model : CPANet, i : int) -> None:
    """Visualizes a single trajectory forecast."""
    plt.rcParams.update({'font.size': 12})
    plt.rcParams['axes.titlesize'] = 16
    plt.rcParams['axes.labelsize'] = 13
    plt.rcParams['xtick.labelsize'] = 13
    plt.rcParams['ytick.labelsize'] = 13
    plt.rcParams['legend.fontsize'] = 12
    plt.rcParams['figure.titlesize'] = 18

    # Forecast
    e_base, n_base, e_predicted, n_predicted, e_pred_truth, n_pred_truth = single_forecast(data=data,
                                                                                           n_obs_est=n_obs_est, n_obs_pred=n_obs_pred,
                                                                                           n_forecasts=n_forecasts, model=model)
    # To latlon
    lat_pred_truth, lon_pred_truth = renorm_trajs(north=n_pred_truth * DATA_NORM, east=e_pred_truth * DATA_NORM)
    lat_predicted, lon_predicted = renorm_trajs(north=n_predicted * DATA_NORM, east=e_predicted * DATA_NORM)
    lat_base, lon_base = renorm_trajs(north=n_base * DATA_NORM, east=e_base * DATA_NORM)

    # Viz
    #plt.scatter(e_pred_truth, n_pred_truth, label="real trajectory")
    #plt.scatter(e_predicted, n_predicted, label="predicted trajectory")
    #plt.scatter(e_base, n_base, label="observations")
    #plt.scatter(e_base[-1], n_base[-1], label="current position")

    plt.scatter(lon_pred_truth, lat_pred_truth, label="real trajectory")
    plt.scatter(lon_predicted, lat_predicted, label="predicted trajectory")
    plt.scatter(lon_base, lat_base, label="observations")
    plt.scatter(lon_base[-1], lat_base[-1], label="current position")

    plt.xlabel("Longitude [°]")
    plt.ylabel("Latitude [°]")

    ax = plt.gca()
    ax.xaxis.set_major_formatter(ScalarFormatter(useOffset=False, useMathText=False))
    ax.yaxis.set_major_formatter(ScalarFormatter(useOffset=False, useMathText=False))
    
    plt.legend() 
    plt.savefig(f"Traj_estimate_{i+1}.pdf", bbox_inches="tight")
    #plt.show()
    plt.close()

def get_RMSE(actual_values : np.ndarray, predicted_values : np.ndarray) -> float:
    """Compute Root Mean Squared Error (RMSE).

    Args:
        actual_values (array-like): Array of actual target values.
        predicted_values (array-like): Array of predicted target values.

    Returns:
        float: The RMSE value.
    """
    # Compute squared differences between actual and predicted values
    squared_errors = (actual_values - predicted_values) ** 2

    # Compute mean of squared errors
    mean_squared_error = np.mean(squared_errors)

    # Compute square root of mean squared error to get RMSE
    return np.sqrt(mean_squared_error)

def compute_quant_study(n_traj : int, n_obs_est : int, n_obs_pred : int, n_forecasts : int, model : CPANet):

    # Setup data    
    e_predicted = np.empty((n_traj, n_obs_pred))
    n_predicted = np.empty((n_traj, n_obs_pred))
    e_truth     = np.empty((n_traj, n_obs_pred))
    n_truth     = np.empty((n_traj, n_obs_pred))

    # Computation
    for i in tqdm(range(n_traj)):
        _, _, e_pred_tmp, n_pred_tmp, e_truth_tmp, n_truth_tmp = single_forecast(data=data, n_obs_est=n_obs_est, 
                                                                                 n_obs_pred=n_obs_pred, n_forecasts=n_forecasts, model=model)
        e_predicted[i] = e_pred_tmp
        n_predicted[i] = n_pred_tmp
        e_truth[i] = e_truth_tmp
        n_truth[i] = n_truth_tmp

    # Aggregation
    dist = np.sqrt((e_predicted-e_truth)**2 + (n_predicted-n_truth)**2)

    # Quantiles
    q25 = np.quantile(dist, q=0.25, axis=0)
    q50 = np.quantile(dist, q=0.50, axis=0)
    q75 = np.quantile(dist, q=0.75, axis=0)
    q90 = np.quantile(dist, q=0.90, axis=0)

    # Viz
    x = np.arange(1, n_obs_pred+1)
    plt.fill_between(x, 0, q25, facecolor="#08306b", alpha=0.7, label="25%")
    plt.fill_between(x, q25, q50, facecolor="#08519c", alpha=0.7, label="50%")
    plt.fill_between(x, q50, q75, facecolor="#1f78b4", alpha=0.7, label="75%")   
    plt.fill_between(x, q75, q90, facecolor="#a6cee3", alpha=0.7, label="90%")

    # Plot the lines
    plt.plot(x, q90, color="black", linewidth=0.5)
    plt.plot(x, q75, color="black", linewidth=0.5)
    plt.plot(x, q50, color="black", linewidth=0.5)
    plt.plot(x, q25, color="black", linewidth=0.5)

    # Get the handles and labels, and reverse them
    handles, labels = plt.gca().get_legend_handles_labels()
    plt.legend(handles[::-1], labels[::-1])

    plt.xlim(-1, n_obs_pred)
    plt.ylim(bottom=0)
    plt.xlabel("Forecast time horizon [time steps]")
    plt.ylabel("Prediction RMSE [m]")
    #plt.title(f"Forecast window: {N_FORECASTS} points")
    plt.savefig(f"RMSE_{N_FORECASTS}.pdf", bbox_inches="tight")
    #plt.show()
    plt.close()

if __name__ == "__main__":
    from train import DATA_NORM, DIFF, N_FORECASTS

    # Load data
    with open("Ebeltoft.dict","rb") as f:
        orig_data = pickle.load(f)    
        
    # Normalize data
    data = {}
    for key, value in orig_data.items():
        data[key] = {"ts" : value["ts"], "e" : value["e"] / DATA_NORM, "n" : value["n"] / DATA_NORM}

    # Construct net and optimizer
    model = CPANet(n_forecasts=N_FORECASTS)
    model.load_state_dict(torch.load(f"cpa_weights_{N_FORECASTS}_1200.pth"))
    
    # Evaluation mode
    model.eval()

    # Viz
    for i in range(20):
        viz_forecast(data=data, n_obs_est=75, n_obs_pred=200, n_forecasts=N_FORECASTS, model=model, i=i)

    #random.seed(0)
    #np.random.seed(0)

    # Quantitative analysis
    #compute_quant_study(n_traj=100, n_obs_est=75, n_forecasts=N_FORECASTS, n_obs_pred=200, model=model)
