from pathlib import Path

import flwr as fl
import hydra
import numpy as np
import torch
from omegaconf import DictConfig
from agent import Agent
from fedavg_rl import FedAvg

import client
import utils

print(f"cuda.is_available:{torch.cuda.is_available()}")
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
STATE_DIM = 4
ACTION_DIM = 10
REPLAY_SIZE = 200
LEARNING_RATE = 0.1

@hydra.main(config_path="docs/conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    """Main function to run CNN federated learning on MNIST.

    Parameters
    ----------
    cfg : DictConfig
        An omegaconf object that stores the hydra config.
    """

    agent = Agent(STATE_DIM, ACTION_DIM, REPLAY_SIZE, LEARNING_RATE)
    client_fn = client.gen_client_fn(
        num_epochs=cfg.num_epochs,
        batch_size=cfg.batch_size,
        device=DEVICE,
        num_clients=cfg.num_clients,
        learning_rate=cfg.learning_rate,
        agent=agent
    )

    # evaluate_fn = utils.gen_evaluate_fn(testloader, DEVICE)
    # 采用去中心评估策略
    strategy = FedAvg(
        fraction_fit=cfg.client_fraction,
        fraction_evaluate=1,
        min_fit_clients=int(cfg.num_clients * cfg.client_fraction),
        min_evaluate_clients=cfg.num_clients,
        min_available_clients=cfg.num_clients,
        # evaluate_fn=evaluate_fn,
        evaluate_metrics_aggregation_fn=utils.weighted_average,
        agent=agent
    )

    # Start simulation
    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=cfg.num_clients,
        config=fl.server.ServerConfig(num_rounds=cfg.num_rounds),
        strategy=strategy,
    )

    file_suffix: str = (
        f"_C={cfg.num_clients}"
        f"_B={cfg.batch_size}"
        f"_E={cfg.num_epochs}"
        f"_R={cfg.num_rounds}"
    )

    np.save(
        Path(cfg.save_path) / Path(f"hist{file_suffix}"),
        history,  # type: ignore
    )

    utils.plot_metric_from_history(
        history,
        cfg.save_path,
        cfg.expected_maximum,
        file_suffix,
    )


if __name__ == "__main__":
    main()
