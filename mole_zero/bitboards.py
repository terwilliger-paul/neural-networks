import numpy as np
import board_representation_functions as brf
import time

class ChessBitboard:

    def __init__(self):

        # White pieces
        self.wk = np.uint64(0)
        self.wq = np.uint64(0)
        self.wr = np.uint64(0)
        self.wb = np.uint64(0)
        self.wn = np.uint64(0)
        self.wp = np.uint64(0)

        # Black pieces
        self.bk = np.uint64(0)
        self.bq = np.uint64(0)
        self.br = np.uint64(0)
        self.bb = np.uint64(0)
        self.bn = np.uint64(0)
        self.bp = np.uint64(0)

        # Other booleans
        self.wtm = np.bool(True)
        self.wk_castle = np.bool(True)
        self.wq_castle = np.bool(True)
        self.bk_castle = np.bool(True)
        self.bq_castle = np.bool(True)

        # En passant files
        self.ep = np.uint8(0)

        # Moves to 50 move draw
        self.to_draw = np.uint(0)

    def vectorize(self):
        pass

print(brf.flip(np.uint64(12345678)))
print((np.uint64(12345678)).byteswap())

trials = 100000
start = time.time()
for _ in range(trials):
    foo = brf.flip(np.uint64(1234567))
print(time.time() - start)
start = time.time()
for _ in range(trials):
    foo = (np.uint64(1234567)).byteswap()
print(time.time() - start)
