from typing import List, Union
from collections import Counter

from game.models.constant import Card, PlayerType


class Player:
    """
    Encapsulates a player's action
    """

    def __init__(self, player_type: PlayerType):
        self.player_type = player_type
        self.hand: Counter[Card] = Counter()

    def get_hand_value(self) -> int:
        def get_value_with_aces(hand):
            # This method deals with hands that have an ace.
            # First case is where there are only 2 cards and they are both aces
            total_cards = sum(hand.values())
            if total_cards == 2 and hand[Card.ace] == 2:
                return 21
            # Second case is where there is only 1 ace and the other card is a 10 value card
            elif total_cards == 2 and hand[Card.ace] == 1 and (hand[Card.jack] == 1 or
                                                               hand[Card.queen] == 1 or
                                                               hand[Card.king] == 1 or
                                                               hand[Card.ten] == 1):
                return 21
            # Third case is where there are an arbitrary number of aces and arbitrary number of other cards
            # In this case, there are 2 possibilities: The aces are either 1 or 11
            # We will try to make the aces 11 first, and if the value of the hand is > 21, we will make the ace 1
            else:
                hand_value = 0
                for card, count in hand.items():
                    if card in [Card.jack, Card.queen, Card.king]:
                        hand_value += 10 * count
                    else:
                        hand_value += card * count
                # Now we will try to make the aces 11
                for i in range(hand[Card.ace]):
                    if hand_value + 10 <= 21:
                        hand_value += 10
                return hand_value
        hand_value = 0
        # Separate the ace cases from the rest of the cards
        if Card.ace not in self.hand:
            for card, count in self.hand.items():
                if card in [Card.jack, Card.queen, Card.king]:
                    hand_value += 10 * count
                else:
                    hand_value += card * count
            return hand_value
        else:
            return get_value_with_aces(self.hand)

    def draw(self, deck) -> None:
        card = deck.draw()
        self.hand[card] += 1

    def reset_hand(self) -> Counter[Card]:
        """
        Resets player's hand and return discarded cards
        """

        discarded = self.hand
        self.hand = Counter()
        return discarded
