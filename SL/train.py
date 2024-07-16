import pickle
import random
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


#--------------------- Classes & Functions ---------------------------
class CPANet(nn.Module):
    """Defines a recurrent network."""
    
    def __init__(self, n_forecasts : int) -> None:
        super(CPANet, self).__init__()

        # memory
        self.mem_LSTM = nn.LSTM(input_size=2, hidden_size=256, num_layers=1, batch_first=True)
        self.dense1 = nn.Linear(256 + 2, 256)
        
        # post combination
        self.dense_add = nn.Linear(256, 256)
        self.dense2 = nn.Linear(256, 2 * n_forecasts)

    def forward(self, x0, y0, pos) -> tuple:
        """Shapes:
        x0:   torch.Size([batch_size, history_length, 1])
        y0:   torch.Size([batch_size, history_length, 1])
        pos:  torch.Size([batch_size, 2])
 
        Returns: 
        torch.Size([batch_size, 2 * n_forecasts])
        """

        #------ memory ------
        # LSTM
        _, (mem, _) = self.mem_LSTM(torch.cat([x0, y0], dim=2))
        mem = mem[0]

        # dense
        combo = torch.cat([mem, pos], dim=1)
        x = F.leaky_relu(self.dense1(combo))
        x = F.leaky_relu(self.dense_add(x))
        
        # final dense layers
        x = self.dense2(x)
        return x

def augment_traj(e : np.ndarray, n : np.ndarray) -> tuple[np.ndarray|np.ndarray]:
    """Augments a given trajectory."""
    # Add noise to traj
    n += np.random.randn(len(n)) * 0.2 / DATA_NORM
    e += np.random.randn(len(e)) * 0.2 / DATA_NORM     
    return e, n

def sample_AIS_trajectory(data : dict, n_obs : int, n_forecasts : int, diff : bool) -> tuple:
    """Samples a single AIS-trajectory. 'n_obs' is the number of observations used for prediction. 
    'n_forecasts' is the number of predicted points.
    
    Returns:
        np.array(n_obs-1,) -> e_in
        np.array(n_obs-1,) -> n_in
        np.array(n_forecasts,) -> e_tar
        np.array(n_forecasts,) -> n_tar
    """
    # Select traj
    while True:
        tr = data[random.randrange(0, len(data))]
        tr_len = len(tr["ts"])

        if tr_len > n_obs + n_forecasts:
            break

    n = tr["n"]
    e = tr["e"]

    # ---- Starting point of traj part ----
    tp = np.argmin(e)
    
    # Curve
    if np.random.binomial(n=1, p=0.2) and (tp-n_obs-n_forecasts >= 0):
        t_start = random.randrange(max([tp-200, 0]), tp)
    
    # Any (most likely straight though)
    else:
        t_start = random.randrange(0, tr_len-1-n_obs-n_forecasts)

    # Data augmentation
    e, n = augment_traj(e=e, n=n)
    
    if diff:
        # Add absolute position
        n_pos = n[t_start + n_obs - 1] / 100.
        e_pos = e[t_start + n_obs - 1] / 100.

        # First-order difference
        n_diff = np.diff(n)
        e_diff = np.diff(e)

        # Select observations from the difference
        n_diff_in  = n_diff[t_start : t_start + n_obs - 1]
        e_diff_in  = e_diff[t_start : t_start + n_obs - 1]

        n_diff_tar = n_diff[t_start + n_obs - 1 : t_start + n_obs - 1 + n_forecasts]
        e_diff_tar = e_diff[t_start + n_obs - 1 : t_start + n_obs - 1 + n_forecasts]
        return e_diff_in, n_diff_in, e_diff_tar, n_diff_tar, np.array([e_pos, n_pos])
    
    else:
        # Select observations
        n_in  = n[t_start : t_start + n_obs]
        e_in  = e[t_start : t_start + n_obs]
        n_tar = n[t_start + n_obs : t_start + n_obs + n_forecasts]
        e_tar = e[t_start + n_obs : t_start + n_obs  + n_forecasts]
        return e_in, n_in, e_tar, n_tar

def compute_trajectories(data : dict, n_traj : int, n_obs : int, n_forecasts : int, diff : bool) -> tuple: 
    """Samples 'n_trajectories' each having 'n_observations' points from the 'data' dict.
    
    Returns:
        torch.Size([n_trajectories, n_observerations, 1]) -> x_diff_in
        torch.Size([n_trajectories, n_observerations, 1]) -> y_diff_in
        torch.Size([n_trajectories, n_forecasts]) -> x_diff_tar
        torch.Size([n_trajectories, n_forecasts]) -> y_diff_tar
    """
    if diff:
        x_diff_in  = np.empty((n_traj, n_obs-1, 1))
        y_diff_in  = np.empty((n_traj, n_obs-1, 1))
        x_diff_tar = np.empty((n_traj, n_forecasts))
        y_diff_tar = np.empty((n_traj, n_forecasts))
        pos_in     = np.empty((n_traj, 2))

        for i in range(n_traj):
            x_diff_in_tmp, y_diff_in_tmp, x_diff_tar_tmp, y_diff_tar_tmp, pos_tmp = sample_AIS_trajectory(data=data,
                                                                                                          n_obs=n_obs,n_forecasts=n_forecasts, diff=DIFF)

            x_diff_in[i]  = x_diff_in_tmp[..., np.newaxis]
            y_diff_in[i]  = y_diff_in_tmp[..., np.newaxis]
            x_diff_tar[i] = x_diff_tar_tmp
            y_diff_tar[i] = y_diff_tar_tmp
            pos_in[i] = pos_tmp
        return torch.tensor(x_diff_in, dtype=torch.float32), torch.tensor(y_diff_in, dtype=torch.float32), torch.tensor(x_diff_tar, dtype=torch.float32), torch.tensor(y_diff_tar, dtype=torch.float32), torch.tensor(pos_in, dtype=torch.float32)

    else:
        x_in  = np.empty((n_traj, n_obs, 1))
        y_in  = np.empty((n_traj, n_obs, 1))
        x_tar = np.empty((n_traj, n_forecasts))
        y_tar = np.empty((n_traj, n_forecasts))

        for i in range(n_traj):
            x_in_tmp, y_in_tmp, x_tar_tmp, y_tar_tmp = sample_AIS_trajectory(data=data, n_obs=n_obs, 
                                                                            n_forecasts=n_forecasts, diff=DIFF)

            x_in[i]  = x_in_tmp[..., np.newaxis]
            y_in[i]  = y_in_tmp[..., np.newaxis]
            x_tar[i] = x_tar_tmp
            y_tar[i] = y_tar_tmp
        return torch.tensor(x_in, dtype=torch.float32), torch.tensor(y_in, dtype=torch.float32), torch.tensor(x_tar, dtype=torch.float32), torch.tensor(y_tar, dtype=torch.float32)


#--------------------- Computation -----------------------
# trajectory parameter
N_FORECASTS = 10
DATA_NORM = 10.
DIFF = True

# learning parameters
LR = 0.00001
BATCH_SIZE = 32
ITERATIONS = 5_000_000
TEST_ITERS = 100
EPOCH_LENGTH = 100
SEED = 125

if __name__ == "__main__":
    # seeding
    random.seed(SEED)
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    # Load data
    with open("Ebeltoft.dict","rb") as f:
        orig_data : dict = pickle.load(f)

    # Normalize data
    data = {}
    for key, value in orig_data.items():
        data[key] = {"ts" : value["ts"], "e" : value["e"] / DATA_NORM, "n" : value["n"] / DATA_NORM}

    # construct net and optimizer
    net = CPANet(n_forecasts=N_FORECASTS)
    opt = optim.Adam(net.parameters(), lr=LR)

    # save losses for plotting
    losses = []
    epoch_losses = []

    # time measurement
    start = time.time()

    # clean file
    open(f"train_stats_{N_FORECASTS}.txt", 'w').close()  

    # train
    for iter in range(ITERATIONS):
        if DIFF:
            x_diff_in, y_diff_in, x_diff_tar, y_diff_tar, pos_in = compute_trajectories(data=data, n_traj=BATCH_SIZE, n_obs=50, n_forecasts=N_FORECASTS, diff=DIFF)
        else:
            x_in, y_in, x_tar, y_tar = compute_trajectories(data=data, n_traj=BATCH_SIZE, n_obs=50, n_forecasts=N_FORECASTS, diff=DIFF)

        # clear gradients
        opt.zero_grad()

        # forward
        if DIFF:
            est = net(x_diff_in, y_diff_in, pos_in)
        else:
            est = net(x_in, y_in)

        # compute loss
        if DIFF:
            loss = F.mse_loss(est[:, :N_FORECASTS], x_diff_tar) + F.mse_loss(est[:, N_FORECASTS:], y_diff_tar)
        else:
            loss = F.mse_loss(est[:, :N_FORECASTS], x_tar) + F.mse_loss(est[:, N_FORECASTS:], y_tar)

        # gradients
        loss.backward()

        # parameter update
        opt.step()

        # store loss for epoch
        epoch_losses.append(loss.item())

        # epoch end
        if iter % EPOCH_LENGTH == 0:
            crt_loss = np.mean(epoch_losses)
            losses.append(crt_loss)

            print(f"Iteration: {iter}")
            print(f"Train loss: {crt_loss}")
            print(f"Runtime [min]: {(time.time()-start) / 60}")
            print("---------------------------------")

            with open(f"train_stats_{N_FORECASTS}.txt", "ab") as f:
                np.savetxt(f, [crt_loss])

            epoch_losses = []

            # save trained model
            torch.save(net.state_dict(), f"cpa_weights_{N_FORECASTS}_{iter}.pth")

            # show training behaviour
            plt.plot(losses, color="blue", label="train loss")
            plt.ylim(0., 0.1)
            plt.savefig(f"cpa_train_{N_FORECASTS}.pdf")
