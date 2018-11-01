
'''
Given a board...

1. Generate all pseudo-legal moves
2. Am I currently in check?  If so, which pseudo-legal moves keep me in check?
   Remove those moves from the set.
3. Do any remaining moves put me in check?  Cull those.
   (absolute pin check + en passant?)
4. Am I castling?  If so, is the castling square under attack?

Neural Network
1. play through a game
2. update value and policy networks

How to update the value and policy networks:
----given a current position:
1. value (tanh [-1, 1]) is updated to be minimax value of the search tree.
   loss function is mse(v(s) - v(terminal state) (could be end of game))*entropy
   entropy is to encourage exploration
2. policy (softmax [0, 1]) is updated to
https://github.com/unixpickle/anyrl-py/blob/953ad68d6507b83583e342b3210ed98e03a86a4f/anyrl/algos/ppo.py#L149-L155
'''
