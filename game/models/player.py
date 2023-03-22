from typing import List, Union
from collections import Counter

from game.models.constant import Card, Action, PlayerType


class Player:
    """
    Encapsulates a player's action
    """

    def __init__(self, player_type: PlayerType):
        self.player_type = player_type
        self.first_card: Union[None, Card] = None
        self.hand: Counter[Card] = Counter()

    def get_action(self, state=None) -> Action:
        """
        Obtain player's next action, should perform Q-Learning on this
        """

        if self.get_hand_value() < 15:
            return Action.hit
        else:
            return Action.stay

    def get_hand_value(self) -> int:
        hand_value = 0
        total_cards = sum(self.hand.values())
        for card, count in self.hand.items():
            if card in [Card.jack, Card.queen, Card.king]:
                hand_value += 10 * count
            else:
                hand_value += card * count

        # blackjack case
        if hand_value == 11 and total_cards == 2 and self.hand[Card.ace] == 1:
            hand_value = 21

        return hand_value

    def get_showing_value(self) -> int:
        """
        Get player's showing value, only applies for dealer
        """
        if not self.first_card:
            return 0
        return self.get_hand_value() - self.first_card

    def draw(self, deck) -> None:
        card = deck.draw()
        self.hand[card] += 1

        if not self.first_card:
            self.first_card = card

    def reset_hand(self) -> Counter[Card]:
        """
        Resets player's hand and return discarded cards
        """

        discarded = self.hand
        self.hand = Counter()
        return discarded

    def update(self, new_state, reward):
        ...
