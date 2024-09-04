# Authors' code to 'Two-Step Dynamic Obstacle Avoidance'

## Overview
This repository contains the python code to:

Hart, Fabian, Martin Waltz, and Ostap Okhrin. "Two-step dynamic obstacle avoidance." arXiv preprint arXiv:2311.16841 (2023).

The paper proposes a two-step approach to dynamic obstacle avoidance that can handle highly nonlinear obstacle dynamics. The first step consists of a supervised learning procedure, which performs trajectory prediction of obstacles and numerically derives collision risk estimates. The second step is deep reinforcement learning, where the agent processes the collision risk estimates from step one.

## Computation
The scripts for the supervised learning part, including training and visualization, are in the folder `SL`. The training script generates obstacle trajectories based on the nonlinear dynamics outlined in the paper. We emphasize that the real-world trajectory data for the generalization study to maritime traffic cannot be shared since the data is confidential.


The deep reinforcement learning computations are performed using the [TUD_RL](https://github.com/MarWaltz/TUD_RL) package. The package requires a `.py` file containing the environmental dynamics and a `.yaml` file with the hyperparameter configuration. The folder `RL` contains the corresponding files for both considered reinforcement learning environments of the paper.

## Citation
If you find the work useful, please cite it as follows:

```bibtex
@article{hart2023two,
  title={Two-step dynamic obstacle avoidance},
  author={Hart, Fabian and Waltz, Martin and Okhrin, Ostap},
  journal={Knowledge-Based Systems},
  pages={112402},
  year={2024},
  volume={302},
  publisher={Elsevier}
}
```
