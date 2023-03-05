from collections import Counter
from enum import IntEnum
from typing import List, Optional

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
    hand: List[Card]
    discarded: Counter[Card]
    # % of my remaining cash I am betting
    bet_percent: Optional[float]
    # total cash I have left
    remaining_cash: int


class ActionOutcome(BaseModel):
    new_state: GameState
    # amount of cash won/loss
    # should be int but left as float for convenience
    reward: float
    # single game has ended
    terminated: bool
