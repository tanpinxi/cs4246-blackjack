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

    def __init__(self, initial_cash: int = 100, deck_nums: int = 4):
        """
        Initialise the game.
        Will draw 2 cards for each player and dealer.
        """
        self.initial_cash: int = initial_cash
        self.remaining_cash: int = initial_cash
        self.player_bet_percent: float = 0
        self.deck_nums: int = deck_nums

        self.dealer = Player(player_type=PlayerType.dealer)
        self.player = Player(player_type=PlayerType.player)

        self.deck = Deck(deck_nums=self.deck_nums)
        self.discarded: Counter[Card] = Counter()

        self.deck.shuffle()

        # draw 2 cards for each player and dealer, sequence matters
        self.player.draw(self.deck)
        self.dealer.draw(self.deck)
        self.player.draw(self.deck)
        self.dealer.draw(self.deck)

        # always starts with player turn
        self.turn = PlayerType.player

    def reset(self) -> None:
        """
        Next game reshuffles the deck and recollects discarded cards
        Will draw 2 cards for each player and dealer.
        """
        dealer_discarded = self.dealer.reset_hand()
        player_discarded = self.player.reset_hand()
        for card, amount in dealer_discarded.items():
            self.discarded[card] += amount
        for card, amount in player_discarded.items():
            self.discarded[card] += amount

        if self.deck.get_remaining_cards() < self.deck_nums * len(Card) * 4 // 2:
            # used more than half the deck, reset deck
            self.discarded = Counter()
            self.deck = Deck(deck_nums=self.deck_nums)

        self.deck.shuffle()
        self.player_bet_percent = 0

        # draw 2 cards for each player and dealer, sequence matters
        self.player.draw(self.deck)
        self.dealer.draw(self.deck)
        self.player.draw(self.deck)
        self.dealer.draw(self.deck)

        # always starts with player turn
        self.turn = PlayerType.player

    def get_state(self) -> GameState:
        """
        Returns state of player (not dealer!)
        """
        return GameState(
            deck_nums=self.deck_nums,
            initial_cash=self.initial_cash,
            turn=self.turn,
            hand=self.player.hand,
            discarded=self.discarded,
            bet_percent=self.player_bet_percent,
            remaining_cash=self.remaining_cash,
        )

    def bet_step(self, bet_percent: float) -> ActionOutcome:
        """
        Must call this first at the start of each round
        """
        self.player_bet_percent = bet_percent

        return ActionOutcome(
            new_state=GameState(
                deck_nums=self.deck_nums,
                initial_cash=self.initial_cash,
                turn=self.turn,
                hand=self.player.hand,
                discarded=self.discarded,
                bet_percent=self.player_bet_percent,
                remaining_cash=self.remaining_cash,
            ),
            reward=0,
            terminated=False,
        )

    def card_step(self, take_card: bool) -> ActionOutcome:
        game_terminated = False
        game_reward = 0
        if take_card:
            self.player.draw(self.deck)
            if self.player.get_hand_value() > 21:
                # bust, immediately loss
                game_terminated = True
                loss_cash = max(int(self.remaining_cash * self.player_bet_percent), 1)
                self.remaining_cash -= loss_cash
                game_reward = -loss_cash
            else:
                # not bust, continue
                game_terminated = False
        else:
            # stand, end turn
            game_terminated = True
            player_score = self.player.get_hand_value()
            while True:
                dealer_score = self.dealer.get_hand_value()
                if dealer_score > 21:
                    # dealer bust, player wins
                    win_cash = int(self.remaining_cash * self.player_bet_percent)
                    self.remaining_cash += win_cash
                    game_reward = win_cash
                    break
                elif dealer_score > player_score:
                    # dealer score is higher than player, player loses
                    loss_cash = max(
                        int(self.remaining_cash * self.player_bet_percent), 1
                    )
                    self.remaining_cash -= loss_cash
                    game_reward = -loss_cash
                    break
                self.dealer.draw(self.deck)

        return ActionOutcome(
            new_state=GameState(
                deck_nums=self.deck_nums,
                initial_cash=self.initial_cash,
                turn=self.turn,
                hand=self.player.hand,
                discarded=self.discarded,
                bet_percent=self.player_bet_percent,
                remaining_cash=self.remaining_cash,
            ),
            reward=game_reward,
            terminated=game_terminated,
        )
