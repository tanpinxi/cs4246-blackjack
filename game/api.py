from collections import Counter
import math

from game.models.game_models import ActionOutcome, GameState
from game.models.constant import Card


class BlackjackWrapper:
    """
    Mid-layer APIs to be implemented for RL training
    Wraps around the blackjack game implementation
    Processes input from RL training process and returns output from blackjack game
    """

    def __init__(self):
        ...

    def reset(self) -> None:
        ...
    
    def get_state(self) -> GameState:
        return GameState(
            hand=Counter([Card.ace, Card.ace]),
            discarded=Counter([]),
            bet_percent=0.5,
            remaining_cash=100,
        )

    def bet_step(self, bet_percent: float) -> ActionOutcome:
        return ActionOutcome(
            new_state=GameState(
                hand=Counter([Card.ace, Card.ace]),
                discarded=Counter([]),
                bet_percent=bet_percent,
                remaining_cash=math.floor(100 * bet_percent),
            ),
            reward=0,
            terminated=False,
        )

    def card_step(self, take_card: bool) -> ActionOutcome:
        return ActionOutcome(
            new_state=GameState(
                hand=Counter([Card.ace, Card.ace]),
                discarded=Counter([]),
                bet_percent=0.5,
                remaining_cash=100,
            ),
            reward=1.0,
            terminated=True,
        )
