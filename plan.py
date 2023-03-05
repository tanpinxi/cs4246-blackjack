"""
deep Q-learning (VI) - covered in lec
https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html

state -> NN -> Q-value of each possible action (quality of action in state)
state: current state (normalization)
possible action: must be discrete actions
- bet percentage: make it discrete by intervals of 0.1, from 0.1 to 1.0 (10 possible values)
- hit/stand: discrete 0 or 1
- 12 actions in total -> output of NN is 12 Q-values
inference: have trained NN
- run NN at each state to get Q-value array
- choose action with highest Q-value
training: learn NN
- explore the game
- with eps randomization:
    - eps times you randomly move
    - 1 - eps times you use NN and select action with highest Q-value
    - reduce eps as training progresses
- training loss:
    - MSE between predicted Q-value (from NN) and target Q value (calculated)
    - ONLY apply loss on Q-value of the action that you took

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
