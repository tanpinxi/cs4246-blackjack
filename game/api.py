from collections import Counter
import math

from game.models.model import ActionOutcome, GameState
from game.models.player import Player
from game.models.deck import Deck
from game.models.constant import Card, PlayerType


class BlackjackWrapper:
    """
    Mid-layer APIs to be implemented for RL training
    Wraps around the blackjack game implementation
    Processes input from RL training process and returns output from blackjack game
    """

    def __init__(self, player_cash=100, deck_nums=4):
        self.player_cash = player_cash
        self.deck_nums = deck_nums

        self.dealer = Player(player_type=PlayerType.dealer)
        self.player = Player(player_type=PlayerType.player)
        
        self.deck = Deck(deck_nums=self.deck_nums)
        self.discarded: Counter[Card] = Counter()

        self.deck.shuffle()

    def reset(self) -> None:
        """
        Next game reshuffles the deck and recollects discarded cards
        """

        dealer_discarded = self.dealer.reset_hand()
        player_discarded = self.player.reset_hand()
        for card, amount in dealer_discarded.items():
            self.discarded[card] += amount
        for card, amount in player_discarded.items():
            self.discarded[card] += amount

        if self.deck.get_remaining_cards() < 10:
            # when deck has less than 10 cards, reset deck
            self.discarded = Counter()
            self.deck = Deck(deck_nums=self.deck_nums)

        self.deck.shuffle()
    
    def get_state(self) -> GameState:
        """
        Returns state of player (not dealer!)
        """

        return GameState(
            hand=self.player.hand,
            discarded=self.discarded,
            bet_percent=0.5,
            remaining_cash=self.player_cash,
        )

    def bet_step(self, bet_percent: float) -> ActionOutcome:

        # TODO: bet logic
        self.player_cash -= math.floor(self.player_cash * bet_percent)

        # Question: how do we get bet_percent?

        return ActionOutcome(
            new_state=GameState(
                hand=self.player.hand,
                discarded=self.discarded,
                bet_percent=bet_percent,
                remaining_cash=self.player_cash,
            ),
            reward=0,
            terminated=False,
        )

    def card_step(self, player_type: PlayerType, take_card: bool) -> ActionOutcome:

        # TODO: draw logic
        game_terminated = False

        if player_type == PlayerType.player:
            if take_card:
                self.player.draw(self.deck)
                if self.player.get_hand_value() > 21:
                    game_terminated = True
        
        if player_type == PlayerType.dealer:
            if take_card:
                self.dealer.draw(self.deck)
                if self.dealer.get_hand_value() > 21:
                    game_terminated = True

        return ActionOutcome(
            new_state=GameState(
                hand=self.player.hand,
                discarded=self.discarded,
                bet_percent=0.5, # how do get this?
                remaining_cash=self.player_cash,
            ),
            reward=1.0,
            terminated=game_terminated,
        )
