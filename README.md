# FRLCD

This is the code for the "FRLCD: Epidemic close contact detection method based on personalized federated reinforcement learning" and baselines.

Federated learning framework used: [Flower](https://github.com/adap/flower)

baselines:

- [FedAvg](https://github.com/adap/flower/tree/main/baselines/flwr_baselines/flwr_baselines/publications/fedavg_mnist)
- [FedProx](https://github.com/adap/flower/tree/main/baselines/fedprox)
- [FedALA](https://github.com/TsingZ0/FedALA)

**Pre-request**

> python 3.9.16, pytorch 2.0.0, flwr 1.4.0

**Usage**

- Clone this repo.
- Prepare your train/val/test data and preprocess the data.
- Refer to the codes of corresponding sections for specific purposes.

**Model**

- model.py

> The model code of this method, including model structure, model training and testing, etc.

- client.py

> Client code for this method.

- main.py

> Method startup code.

- agent.py

> Reinforcement learning agent.

- fedavg_rl.py

> Central Server Configuration Policy.

- aggregate.py

> Aggregation methods, including parameter aggregation, etc.