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

    def __init__(self, initial_cash: int = 100, deck_nums: int = 4, min_bet: int = 10):
        """
        Initialise the game.
        """
        self.initial_cash: int = initial_cash
        self.max_attained_cash: int = initial_cash
        self.min_bet: int = min_bet
        self.remaining_cash: int = initial_cash
        self.player_bet_percent: float = 0
        self.deck_nums: int = deck_nums

        self.dealer = Player(player_type=PlayerType.dealer)
        self.player = Player(player_type=PlayerType.player)

        self.deck = Deck(deck_nums=self.deck_nums)
        self.discarded: Counter[Card] = Counter()

        self.deck.shuffle()

        # always starts with player turn
        self.turn = PlayerType.player

    def reset(self) -> "BlackjackWrapper":
        """
        Next game reshuffles the deck and recollects discarded cards
        """
        if self.remaining_cash < self.min_bet:
            # ran out of cash, restart entire game
            return BlackjackWrapper(
                initial_cash=self.initial_cash, deck_nums=self.deck_nums
            )

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

        # always starts with player turn
        self.turn = PlayerType.player
        return self

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
        reward = 0
        game_terminated = False
        self.player_bet_percent = bet_percent

        # draw 2 cards for each player and dealer, sequence matters
        self.player.draw(self.deck)
        self.dealer.draw(self.deck)
        self.player.draw(self.deck)
        self.dealer.draw(self.deck)

        # Implement natural blackjack rules for the dealer
        # Case 1 - Both have natural blackjack
        if self.player.get_hand_value() == 21 and self.dealer.get_hand_value() == 21:
            # push, no cash change
            game_terminated = True
        # Case 2 - Dealer has natural blackjack, loss is immediately
        elif self.dealer.get_hand_value() == 21:
            # loss, immediately lose
            game_terminated = True
            loss_cash = max(
                int(self.remaining_cash * self.player_bet_percent), self.min_bet
            )
            self.remaining_cash -= loss_cash
            reward = (
                self.remaining_cash
                if self.remaining_cash >= self.min_bet
                else -self.max_attained_cash
            ) / self.initial_cash

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
            reward=reward,
            terminated=game_terminated,
        )

    def card_step(self, take_card: bool) -> ActionOutcome:
        game_terminated = False
        reward = None
        bet_amount = max(
            int(self.remaining_cash * self.player_bet_percent), self.min_bet
        )

        if take_card:
            self.player.draw(self.deck)
            if self.player.get_hand_value() > 21:
                # bust, immediately lose
                game_terminated = True
                loss_cash = bet_amount
                self.remaining_cash -= loss_cash
            else:
                # not bust, continue
                game_terminated = False
        else:
            # stand, players turn ends
            player_score = self.player.get_hand_value()
            # Penalize invalid action when the player tries to stand with score < 16
            if player_score < 16:
                reward = -1.0
            else:
                game_terminated = True
                # Player has natural blackjack, outcome is immediate and wins 2 times the bet
                if player_score == 21 and len(self.player.hand) == 2:
                    win_cash = bet_amount * 2
                    self.remaining_cash += win_cash
                else:
                    # Draw card until score is greater than or equal to 17 for dealer (house rules)
                    while self.dealer.get_hand_value() < 17:
                        self.dealer.draw(self.deck)

                    # Check if dealer busts or has a score greater than player here and update game_reward accordingly
                    dealer_score = self.dealer.get_hand_value()
                    if dealer_score > 21:
                        # dealer bust, player wins
                        win_cash = bet_amount
                        self.remaining_cash += win_cash
                    elif dealer_score > player_score:
                        # dealer score is higher than player, player loses
                        loss_cash = bet_amount
                        self.remaining_cash -= loss_cash
                    elif dealer_score == player_score:
                        # no difference in score, zero change
                        ...
                    else:
                        # player score is higher than dealer, player wins
                        win_cash = bet_amount
                        self.remaining_cash += win_cash
        self.max_attained_cash = max(self.max_attained_cash, self.remaining_cash)
        reward = reward or (
            (
                (
                    self.remaining_cash
                    if self.remaining_cash >= self.min_bet
                    else -self.max_attained_cash
                )
                / self.initial_cash
            )
            if game_terminated
            else 0
        )
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
            reward=reward,
            terminated=game_terminated,
        )
