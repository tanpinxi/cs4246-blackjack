# Black Jack Reinforcement Learning Agents

This repository contains a set of reinforcement learning agents trained to play the game of Black Jack. The agents are trained using policy gradient and Deep Q-Network. 

## Setup

Install requirements:

```bash
pip install -r requirements.txt
```

## Project Structure

The project is structured as follows:

```bash
.
├── game  # contains the Black Jack game environment
├── models  # contains the trained models and training logs
├── training  # contains notebooks used for training and evaluation
```

## Agents

All trained models are located in the `models` directory. The following models are available:

| Agent                      | Description                                                  |
|----------------------------|--------------------------------------------------------------|
| `dqn_final`                | Deep Q-Network agent                                         |
| `dqn_final_no_counting`    | DQN agent trained without card counting features             |
| `policy_steps`             | Policy Gradient agent                                        |
| `policy_steps_no_counting` | Policy Gradient agent trained without card counting features |

Training logs are also available in the `models` directory, named as `*_training_scores.txt`
