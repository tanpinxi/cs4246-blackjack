from collections import Counter

from game.game_models import ActionOutcome, GameState, Card


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
            bet_amount=0.5,
            remaining_cash=100,
        )

    def bet_step(self, bet_percent: float) -> ActionOutcome:
        return ActionOutcome(
            new_state=GameState(
                hand=Counter([Card.ace, Card.ace]),
                discarded=Counter([]),
                bet_amount=bet_percent,
                remaining_cash=100 * bet_percent,
            ),
            reward=0,
            terminated=False,
        )

    def card_step(self, take_card: bool) -> ActionOutcome:
        return ActionOutcome(
            new_state=GameState(
                hand=Counter([Card.ace, Card.ace]),
                discarded=Counter([]),
                bet_amount=0.5,
                remaining_cash=100,
            ),
            reward=1.0,
            terminated=True,
        )
