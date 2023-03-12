from enum import IntEnum

class Card(IntEnum):
    # needs to ace's special case when it's considered to be 1, 10 or 11
    ace = 1
    
    two = 2
    three = 3
    four = 4
    five = 5
    six = 6
    seven = 7
    eight = 8
    nine = 9
    ten = 10

    # these are considered to be 10
    jack = 11
    queen = 12
    king = 13

class Action(IntEnum):
    draw = 1
    stay = 2