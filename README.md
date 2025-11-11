# Tabular-Multi-Agent-Reinforcement-algorithms
This repository contains some basic tabular algorithms and some environments to test them

This repository includes implementations of the following tabular multi-agent reinforcement learning algorithms: IMAQ, NashQ, PHC, and WoLFPHC.

    IMAQ and NashQ are compatible with all environments.
    PHC and WoLFPHC can only be used with the Matrix and GridV1 environments.

Use train.py to run the learning algorithm.
In this file, you can select the agent and the agent configuration for training.
To configure the training environment, specify the environment name, num_agents, num_states, and num_actions in the agent configuration file.
After training, you will be able to access the models, logs, and configuration files for both the environment and the agent.

Use plot.py to visualize the learning results.
