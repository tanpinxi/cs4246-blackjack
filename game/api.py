from collections import Counter
import math

from game.models.model import ActionOutcome, GameState
from game.models.player import Player
from game.models.deck import Deck
from game.models.constant import Card


class BlackjackWrapper:
    """
    Mid-layer APIs to be implemented for RL training
    Wraps around the blackjack game implementation
    Processes input from RL training process and returns output from blackjack game
    """

    def __init__(self):
        self.dealer = Player()
        self.player = Player()
        
        self.deck = Deck(deck_nums=4)
        self.discarded: Counter[Card] = Counter()

        self.deck.shuffle()

    def next_game(self) -> None:
        """
        Next game reshuffles the deck, but don't collect the discarded cards
        """

        dealer_discarded = self.dealer.reset_hand()
        player_discarded = self.player.reset_hand()
        for card, amount in dealer_discarded.items():
            self.discarded[card] += amount
        for card, amount in player_discarded.items():
            self.discarded[card] += amount

        self.deck.shuffle()

    def reset(self) -> None:
        """
        Next game reshuffles the deck and recollects discarded cards
        """

        self.dealer.reset_hand()
        self.player.reset_hand()
        self.discarded = Counter()
        self.deck.shuffle()
    
    def get_state(self) -> GameState:
        """
        Returns state of player (not dealer!)
        """

        return GameState(
            hand=self.player.hand,
            discarded=self.discarded,
            bet_percent=0.5,
            remaining_cash=100,
        )

    def bet_step(self, bet_percent: float) -> ActionOutcome:

        # TODO: bet logic

        return ActionOutcome(
            new_state=GameState(
                hand=self.player.hand,
                discarded=self.discarded,
                bet_percent=bet_percent,
                remaining_cash=math.floor(100 * bet_percent),
            ),
            reward=0,
            terminated=False,
        )

    def card_step(self, take_card: bool) -> ActionOutcome:

        # TODO: draw logic

        return ActionOutcome(
            new_state=GameState(
                hand=self.player.hand,
                discarded=self.discarded,
                bet_percent=0.5,
                remaining_cash=100,
            ),
            reward=1.0,
            terminated=True,
        )
