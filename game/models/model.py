from collections import Counter
from typing import Optional, List

import numpy as np
import torch
from pydantic import BaseModel

from game.models.constant import Card, PlayerType


class GameState(BaseModel):
    deck_nums: int
    initial_cash: int
    turn: PlayerType
    hand: Counter[Card]
    discarded: Counter[Card]
    bet_percent: Optional[float]  # % of my remaining cash I am betting
    remaining_cash: int  # total cash I have left

    @staticmethod
    def get_state_size(add_steps: bool = False) -> int:
        # number of cards in hand + number of cards discarded + bet percent + remaining cash
        return len(Card) * 2 + 2 + (1 if add_steps else 0)

    def flatten(self, include_discarded: bool = True, step_num: Optional[float] = None) -> np.ndarray:
        """
        Flattens the response into a 1D numpy array. Used as input for backend ML training.
        """
        output = []
        for card in Card:
            output.append(self.hand[card] / self.deck_nums if card in self.hand else 0)
        for card in Card:
            output.append(
                self.discarded[card] / self.deck_nums
                if include_discarded and card in self.discarded
                else 0
            )
        output.append(self.bet_percent or 0)
        output.append(self.remaining_cash / self.initial_cash)
        if step_num is not None:
            output.append(step_num)
        return np.array(output)

    def torch_flatten(self, device, include_discarded: bool = True, step_num: Optional[float] = None) -> torch.Tensor:
        return torch.Tensor(self.flatten(include_discarded=include_discarded, step_num=step_num)).unsqueeze(0).to(device)


class ActionOutcome(BaseModel):
    new_state: GameState
    # amount of cash won/loss
    # should be int but left as float for convenience
    reward: float
    # single game has ended
    terminated: bool
