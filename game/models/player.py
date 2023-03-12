from constant import Card, Action
from typing import List

class Player:
    """
    Encapsulates a player's action
    """

    def __init__(self):
        self.hand: List[Card] = []

    def get_action(self, state = None) -> Action:
        """
        Obtain player's next action, should perform Q-Learning on this
        """

        if self.get_hand_value() < 15:
            return Action.draw
        else:
            return Action.stay

    def get_hand_value(self) -> int:
        return sum(self.hand)

    def draw(self, deck) -> None:
        card = deck.draw()
        self.hand.append(card)

    def reset_hand(self) -> List[Card]:
        """
        Resets player's hand and return discarded cards
        """

        discarded = self.hand
        self.hand = []
        return discarded

    def update(self, new_state, reward):
        ...

