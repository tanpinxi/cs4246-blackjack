from typing import List, Union
from collections import Counter

from constant import Card, Action, PlayerType

class Player:
    """
    Encapsulates a player's action
    """

    def __init__(self, player_type:PlayerType):
        self.player_type = player_type
        self.first_card: Union[None, Card] = None
        self.hand: Counter[Card] = Counter()

    def get_action(self, state = None) -> Action:
        """
        Obtain player's next action, should perform Q-Learning on this
        """

        if self.get_hand_value() < 15:
            return Action.hit
        else:
            return Action.stay

    def get_hand_value(self) -> int:
        return sum(self.hand)

    def get_showing_value(self) -> int:
        """
        Get player's showing value, only applies for dealer
        """
        if not self.first_card:
            return 0
        return sum(self.hand) - self.first_card

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

