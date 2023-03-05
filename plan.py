"""
Q-learning (VI) - covered in lec

state + possible action -> NN -> Q value (quality of action in state)
state: current state (normalization)
possible action: discrete (e.g. 0, 1, 2, 3 is fine, continuous 0 to 3 is not)
- bet percentage: make it discrete by intervals of 0.1 (10 possible values)
- hit/stand: discrete 0 or 1
inference: have trained NN
- test every possible action
- choose action with highest Q value
training: learn NN
- explore the game (with eps randomization: eps times you randomly move, 1 - eps you use NN)
- training loss: MSE between predicted Q value (from NN) and target Q value (calculated)

policy gradient (PI) - not covered yet

state -> NN -> softmax prob of action
diff vs Q-learning:
- no utility (Q-value)
- don't need to test all possible actions
- can have continuous actions
state: current state (normalization)
actions:
- bet percentage: continuous 0 to 1
- hit/stand: discrete 0 or 1
inference: have trained NN
- put state into NN
- sample action based on softmax prob
training: learn NN
- explore the game
- training loss: negative discounted reward
"""
