from collections import Counter
from typing import Optional, List

import numpy as np
import torch
from pydantic import BaseModel

from constant import Card


class GameState(BaseModel):
    hand: Counter[Card]
    discarded: Counter[Card]
    bet_percent: Optional[float] # % of my remaining cash I am betting 
    remaining_cash: int # total cash I have left

    @staticmethod
    def get_state_size() -> int:
        # number of cards in hand + number of cards discarded + bet percent + remaining cash
        return len(Card) * 2 + 2

    def flatten(self) -> np.ndarray:
        """
        Flattens the response into a 1D numpy array. Used as input for backend ML training.
        """

        """
        TODO: normalize this np array
        """

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
