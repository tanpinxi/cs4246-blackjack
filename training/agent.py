import random
from typing import Tuple, List

import torch
from torch import nn


class BlackjackPolicyModel(nn.Module):
    """
    Model that accepts a flattened state and outputs 2 values:
    1. Bet percentage from 0 to 1
    2. Probability of taking a card (hit) from 0 to 1
    """

    def __init__(self, in_features: int, device):
        super().__init__()
        self.device = device
        # common layers shared by both outputs
        self.init_layers = nn.Sequential(
            nn.Linear(in_features, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 1024),
            nn.LeakyReLU(),
            nn.Linear(1024, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 32),
            nn.LeakyReLU(),
        )
        # layers for bet percentage output
        self.bet_layers = nn.Sequential(
            nn.Linear(32, 8),
            nn.LeakyReLU(),
            nn.Linear(8, 1),
            nn.Sigmoid(),
        )
        # layers for card action output
        self.card_layers = nn.Sequential(
            nn.Linear(32, 16),
            nn.LeakyReLU(),
            nn.Linear(16, 8),
            nn.LeakyReLU(),
            nn.Linear(8, 4),
            nn.LeakyReLU(),
            nn.Linear(4, 1),
            nn.Sigmoid(),
        )

    def forward(self, x) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.init_layers(x)
        x1 = self.bet_layers(x)
        x2 = self.card_layers(x)
        return x1, x2

    def get_bet_percent(self, normalized_state) -> torch.Tensor:
        state = torch.from_numpy(normalized_state).float().unsqueeze(0).to(self.device)
        bet_percent, _ = self.forward(state)
        return bet_percent.cpu()

    def get_card_action(self, normalized_state) -> torch.Tensor:
        state = torch.from_numpy(normalized_state).float().unsqueeze(0).to(self.device)
        _, card_prob = self.forward(state)
        return card_prob.cpu()


class BlackjackDQN(nn.Module):
    """
    Model that accepts a flattened state and outputs 12 values:
    1. Q-values of bet percentages from 0.1 to 1.0 (increments of 0.1)
    2. Q-value of taking a card (hit) or not (stand)
    """

    def __init__(
        self,
        in_features: int,
        bet_choices: List[float],
        card_choices: List[bool],
        epsilon: float,
        min_epsilon: float,
    ):
        """
        epsilon and min_epsilon are used for epsilon-greedy action selection
        the higher their values, the more likely the model will choose a random action
        """
        super().__init__()
        self.bet_choices = bet_choices
        self.card_choices = card_choices
        self.epsilon = epsilon
        self.min_epsilon = min_epsilon
        # common layers shared by both outputs
        self.init_layers = nn.Sequential(
            nn.Linear(in_features, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 32),
            nn.LeakyReLU(),
        )
        # layers for bet percentage Q-value output
        self.bet_layers = nn.Sequential(
            nn.Linear(32, 16),
            nn.LeakyReLU(),
            nn.Linear(16, len(bet_choices)),
        )
        # layers for card action Q-value output
        self.card_layers = nn.Sequential(
            nn.Linear(32, 16),
            nn.LeakyReLU(),
            nn.Linear(16, 8),
            nn.LeakyReLU(),
            nn.Linear(8, len(card_choices)),
        )

    def forward(self, x) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.init_layers(x)
        x1 = self.bet_layers(x)
        x2 = self.card_layers(x)
        return x1, x2

    def batched_forward_with_concat(self, x) -> torch.Tensor:
        x = self.init_layers(x)
        x1 = self.bet_layers(x)
        x2 = self.card_layers(x)
        return torch.cat((x1, x2), dim=-1)

    def get_bet_percent(
        self, normalized_state, allow_explore: bool, num_steps: int
    ) -> Tuple[float, int, torch.Tensor]:
        bet_values, card_values = self.forward(normalized_state)
        if allow_explore and random.random() < max(
            self.epsilon ** num_steps, self.min_epsilon
        ):
            # explore
            idx = random.randint(0, len(self.bet_choices) - 1)
        else:
            # exploit
            idx = bet_values.argmax().item()
        action = self.bet_choices[idx]
        mask = torch.tensor(
            len(self.bet_choices) * [1] + len(self.card_choices) * [0]
        ).unsqueeze(0)
        return action, idx, mask

    def get_card_action(
        self, normalized_state, allow_explore: bool, num_steps: int
    ) -> Tuple[bool, int, torch.Tensor]:
        bet_values, card_values = self.forward(normalized_state)
        if allow_explore and random.random() < max(
            self.epsilon ** num_steps, self.min_epsilon
        ):
            # explore
            idx = random.randint(0, len(self.card_choices) - 1)
        else:
            # exploit
            idx = card_values.argmax().item()
        action = self.card_choices[idx]
        mask = torch.tensor(
            len(self.bet_choices) * [0] + len(self.card_choices) * [1]
        ).unsqueeze(0)
        return action, idx + len(self.bet_choices), mask
