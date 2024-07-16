import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tud_rl.common.helper_fnc import exponential_smoothing

OBS_TRAJ = "AR1"
# OBS_TRAJ = "sinus"


ACTIVATIONS = {"relu"     : F.relu,
               "identity" : nn.Identity(),
               "tanh"     : torch.tanh}

#--------------------- Classes & Functions ---------------------------
class CPANet(nn.Module):
    """Defines a recurrent network to predict DCPA and TCPA."""
    
    def __init__(self) -> None:
        super(CPANet, self).__init__()

        # memory
        self.mem_LSTM = nn.LSTM(input_size=2, hidden_size=64, num_layers=1, batch_first=True)
        self.dense1 = nn.Linear(64, 64)
        
        # post combination
        self.dense2 = nn.Linear(64, 2)
        

    def forward(self, x0, y0) -> tuple:
        """Shapes:
        x0:   torch.Size([batch_size, 10, 1])
        y0:   torch.Size([batch_size, 10, 1])
        vx:   torch.Size([batch_size, 1])
        vy:   torch.Size([batch_size, 1])
 
        Returns: 
        torch.Size([batch_size, 2])
        
        Note: 
        The one-layer LSTM is defined with batch_first=True, hence it expects input in form of:
        x = (batch_size, seq_length, state_shape)
        
        The call <out, (hidden, cell) = LSTM(x)> results in: 
        out:    Output (= hidden state) of LSTM for each time step with shape (batch_size, seq_length, hidden_size).
        hidden: The hidden state of the last time step in each sequence with shape (1, batch_size, hidden_size).
        cell:   The cell state of the last time step in each sequence with shape (1, batch_size, hidden_size).
        """

        #------ memory ------
        # LSTM
        _, (mem, _) = self.mem_LSTM(torch.cat([x0, y0], dim=2))
        mem = mem[0]

        # dense
        x = F.relu(self.dense1(mem))
        
        # final dense layers
        x = self.dense2(x)

        return x

def compute_trajectories(n_trajectories, n_observations): 
    """Samples 'n_trajectories' each having 'n_observations' points.
    
    Returns:
        torch.Size([n_trajectories, n_observerations-1, 1]) -> x_diff_in
        torch.Size([n_trajectories, n_observerations-1, 1]) -> y_diff_in
        torch.Size([n_trajectories, 1]) -> x_diff_target
        torch.Size([n_trajectories, 1]) -> y_diff_target
    """
     # compute random trajectories
    vx = (1 - 2 * torch.rand(1,n_trajectories)) * VX_MAX
    vy = (1 - 2 * torch.rand(1,n_trajectories)) * VY_MAX

    t_vector = torch.linspace(0,n_observations+1, n_observations+2) 
    t_vector = torch.reshape(t_vector,(t_vector.size(0),1))

    # constant trajectory part
    x_const = torch.mul(t_vector, DELTA_T * vx)
    y_const = torch.mul(t_vector, DELTA_T * vy)

    # stochastic trajectory part
    if OBS_TRAJ == "sinus":

        phase_offset = torch.rand(1,n_trajectories)*np.pi

        # generate noise 
        noise_const = 4
        ones_temp = torch.ones(n_observations+2,n_trajectories)    
        x_noise = vx * torch.normal(0*ones_temp, noise_const*ones_temp)
        y_noise = vy * torch.normal(0*ones_temp, noise_const*ones_temp)

        x_stoch =  15 * vy * torch.sin(t_vector.repeat(1,n_trajectories) * 2 * np.pi/(2*10) + phase_offset) + x_noise 
        y_stoch = -15 * vx * torch.sin(t_vector.repeat(1,n_trajectories) * 2 * np.pi/(2*10) + phase_offset) + y_noise

    if OBS_TRAJ == "AR1":
        # initialize trajectory
        x_AR1 = np.zeros_like(x_const) 
        y_AR1 = np.zeros_like(y_const)                       

        # compute stochastic trajectory   
        for i in range(x_AR1.shape[0]-1):
            # stochastic part
            x_AR1[i+1,:] = x_AR1[i,:] * 0.9 + np.random.normal(0,15, size=n_trajectories)
            y_AR1[i+1,:] = y_AR1[i,:] * 0.9 + np.random.normal(0,15, size=n_trajectories) 

        # smoothing and scaling trajectory
        alpha = np.sqrt(vx**2 + vy**2)/ np.sqrt(VX_MAX**2 + VY_MAX**2)
        x_stoch = alpha * exponential_smoothing(x_AR1, 0.8) 
        y_stoch = alpha * exponential_smoothing(y_AR1, 0.8)

    x = x_const + x_stoch
    y = y_const + y_stoch

    x_diff = torch.diff(x,1,0)
    y_diff = torch.diff(y,1,0)        

    x_diff_in = x_diff[:-1]
    y_diff_in = y_diff[:-1]

    x_diff_tar = x_diff[-1]
    y_diff_tar = y_diff[-1]

    x_diff_in = x_diff_in.permute(1,0)
    y_diff_in = y_diff_in.permute(1,0)
    x_diff_in = torch.reshape(x_diff_in,[n_trajectories, n_observations,1])
    y_diff_in = torch.reshape(y_diff_in,[n_trajectories, n_observations,1])
    x_diff_tar = torch.reshape(x_diff_tar, [n_trajectories,1])
    y_diff_tar = torch.reshape(y_diff_tar, [n_trajectories,1])

    return  x_diff_in, y_diff_in, x_diff_tar, y_diff_tar

def exponential_smoothing(x, alpha=0.03):
    s = np.zeros_like(x)

    for idx, x_val in enumerate(x):
        if idx == 0:
            s[idx] = x[idx]
        else:
            s[idx] = alpha * x_val + (1-alpha) * s[idx-1]

    return s
#--------------------- Computation -----------------------

# trajectory parameters
VX_MAX = 5
VY_MAX = 5
N_OBS = 10
DELTA_T = 5

# learning parameters
LR = 0.0001
BATCH_SIZE = 32
ITERATIONS = 1_000_000
TEST_ITERS = 100
EPOCH_LENGTH = 500

if __name__ == "__main__":
    # seeding
    torch.manual_seed(10)
    np.random.seed(10)

    # construct net and optimizer
    net = CPANet()
    opt = optim.Adam(net.parameters(), lr=LR)

    # save losses for plotting
    losses = []
    test_losses = []
    epoch_losses = []

    # time measurement
    start = time.time()

    # clean file
    open('train_stats.txt', 'w').close()  

    # train
    for iter in range(ITERATIONS):
        n_obs = np.random.randint(10,100)
        x_diff_in, y_diff_in, x_diff_tar, y_diff_tar = compute_trajectories(BATCH_SIZE, n_obs)

        # clear gradients
        opt.zero_grad()

        # forward
        est = net(x_diff_in, y_diff_in)

        # compute loss
        loss = F.mse_loss(est[:,0].unsqueeze(-1), x_diff_tar) + F.mse_loss(est[:,1].unsqueeze(-1), y_diff_tar)

        # gradients
        loss.backward()

        # parameter update
        opt.step()

        # store loss for epoch
        epoch_losses.append(loss.item())

        # epoch end
        if iter % EPOCH_LENGTH == 0:
            crt_loss = np.mean(epoch_losses)
            #test_loss = test(net=net, X0=X0, Y0=Y0, VX=VX, VY=VY, TCPA=TCPA, DCPA=DCPA)
            #test_losses.append(test_loss)
            losses.append(crt_loss)

            print(f"Iteration: {iter}")
            print(f"Train loss: {crt_loss}")
            #print(f"Test loss: {test_loss}")
            print(f"Runtime [min]: {(time.time()-start) / 60}")
            print("---------------------------------")

            with open("train_stats.txt", "ab") as f:
                np.savetxt(f, [crt_loss])

            epoch_losses = []

            # save trained model
            torch.save(net.state_dict(), "cpa_weights.pth")

    # show training behaviour
    steps = [i * EPOCH_LENGTH for i in range(len(losses))]
    plt.plot(steps, test_losses, color="orange", label="test loss")
    plt.plot(steps, losses, color="blue", label="train loss")
    plt.legend()
    plt.savefig("cpa_train.pdf")
    print(test_losses)
    print(losses)
