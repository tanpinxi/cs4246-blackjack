from collections import Counter
import math

from game.models.model import ActionOutcome, GameState
from game.models.player import Player
from game.models.deck import Deck
from game.models.constant import Card, PlayerType, Action


class BlackjackWrapper:
    """
    Mid-layer APIs to be implemented for RL training
    Wraps around the blackjack game implementation
    Processes input from RL training process and returns output from blackjack game
    """

    def __init__(self, initial_cash:int = 100, deck_nums:int = 4):
        self.initial_cash: int = initial_cash
        self.remaining_cash: int = initial_cash
        self.player_bet_percent: float = 0
        self.deck_nums: int = deck_nums

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
            deck_nums = self.deck_nums,
            initial_cash = self.initial_cash,
            hand=self.player.hand,
            discarded=self.discarded,
            bet_percent=self.player_bet_percent,
            remaining_cash=self.remaining_cash,
        )

    def bet_step(self, bet_percent: float) -> ActionOutcome:
        """
        Must call this first at the start of each round
        """

        # TODO: bet logic
        self.player_bet_percent = bet_percent

        return ActionOutcome(
            new_state=GameState(
                deck_nums = self.deck_nums,
                initial_cash = self.initial_cash,
                hand=self.player.hand,
                discarded=self.discarded,
                bet_percent=self.player_bet_percent,
                remaining_cash=self.remaining_cash,
            ),
            reward=0,
            terminated=False,
        )

    def card_step(self, player_type: PlayerType) -> ActionOutcome:

        # TODO: draw logic
        game_terminated: bool = False
        game_reward: float = 0

        if player_type == PlayerType.player:
            if self.player.get_action() == Action.hit:
                self.player.draw(self.deck)
                if self.player.get_hand_value() > 21:
                    game_terminated = True
                    self.remaining_cash -= math.floor(self.remaining_cash * self.player_bet_percent)
                    game_reward = -1.0 # player lost
                else:
                    game_reward = 0.5 # player maybe win or lose
            # else, do nothing and reward is 0 
        
        if player_type == PlayerType.dealer:
            if self.dealer.get_action() == Action.hit:
                self.dealer.draw(self.deck)
                if self.dealer.get_hand_value() > 21:
                    game_terminated = True
                    self.remaining_cash += math.floor(self.remaining_cash * self.player_bet_percent)
                    game_reward = 1.0 # player wins
            # else, do nothing and reward is 0

        return ActionOutcome(
            new_state=GameState(
                deck_nums = self.deck_nums,
                initial_cash = self.initial_cash,
                hand=self.player.hand,
                discarded=self.discarded,
                bet_percent=self.player_bet_percent,
                remaining_cash=self.remaining_cash,
            ),
            reward=game_reward,
            terminated=game_terminated, # if Ture, have to call reset()
        )
