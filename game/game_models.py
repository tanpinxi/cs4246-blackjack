from collections import Counter
from enum import IntEnum
from typing import Optional, List

import numpy as np
import torch
from pydantic import BaseModel


class Card(IntEnum):
    ace = 1
    two = 2
    three = 3
    four = 4
    five = 5
    six = 6
    seven = 7
    eight = 8
    nine = 9
    ten = 10
    jack = 11
    queen = 12
    king = 13


class GameState(BaseModel):
    hand: Counter[Card]
    discarded: Counter[Card]
    # % of my remaining cash I am betting
    bet_percent: Optional[float]
    # total cash I have left
    remaining_cash: int

    @staticmethod
    def get_state_size() -> int:
        # number of cards in hand + number of cards discarded + bet percent + remaining cash
        return len(Card) * 2 + 2

    def flatten(self) -> np.ndarray:
        output = []
        for card in Card:
            output.append(self.hand[card] if card in self.hand else 0)
        for card in Card:
            output.append(self.discarded[card] if card in self.discarded else 0)
        output.append(self.bet_percent or 0)
        output.append(self.remaining_cash)
        return np.array(output)

    def torch_flatten(self, device) -> torch.Tensor:
        return torch.Tensor(self.flatten()).unsqueeze(0).to(device)


class ActionOutcome(BaseModel):
    new_state: GameState
    # amount of cash won/loss
    # should be int but left as float for convenience
    reward: float
    # single game has ended
    terminated: bool
