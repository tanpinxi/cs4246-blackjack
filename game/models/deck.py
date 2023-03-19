import numpy as np

from constant import Card
from typing import List

class Deck:
    """
    A Deck simulates the behaviour of a Deck. We allow multiple decks to be used, but at least 1 deck must be used.
    """

    def __init__(self, deck_nums:int):
        if deck_nums < 1:
            raise ValueError("deck_nums has to be >= 1")
        
        one_suit_of_cards = [card for card in Card]
        one_deck_of_cards = one_suit_of_cards * 4
        self.cards = one_deck_of_cards * deck_nums
        self.shuffle()

    def shuffle(self):
        np.random.shuffle(self.cards)

    def draw(self):
        return self.cards.pop()
    
    def get_remaining_cards(self) -> int:
        return len(self.cards)