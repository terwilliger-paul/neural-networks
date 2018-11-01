import numpy as np
import time

class Board:

    def __init__(self):

        # Piece boards
        self.wk = np.zeros((8, 8), dtype=bool)
        self.wq = np.zeros((8, 8), dtype=bool)
        self.wr = np.zeros((8, 8), dtype=bool)
        self.wb = np.zeros((8, 8), dtype=bool)
        self.wn = np.zeros((8, 8), dtype=bool)
        self.wp = np.zeros((8, 8), dtype=bool)

        self.bk = np.zeros((8, 8), dtype=bool)
        self.bq = np.zeros((8, 8), dtype=bool)
        self.br = np.zeros((8, 8), dtype=bool)
        self.bb = np.zeros((8, 8), dtype=bool)
        self.bn = np.zeros((8, 8), dtype=bool)
        self.bp = np.zeros((8, 8), dtype=bool)

        # Who's turn to move
        self.w_move = True

        # Castling rights
        self.wk_castle = True
        self.wq_castle = True
        self.bk_castle = True
        self.bq_castle = True

        # En passant: the file that you can capture onto
        self.ep = np.zeros(8, dtype=bool)

        # Halfmove clock
        self.halfmove = 0

        # Pre-loaded constants
        self.diags = np.array([[0+i+1, 0+i+1] for i in range(7)] + \
                              [[0-i-1, 0-i-1] for i in range(7)] + \
                              [[0+i+1, 0-i-1] for i in range(7)] + \
                              [[0-i-1, 0+i+1] for i in range(7)])
        self.files = [[0+i+1, 0] for i in range(7)] + \
                     [[0-i-1, 0] for i in range(7)] + \
                     [[0, 0+i+1] for i in range(7)] + \
                     [[0, 0-i-1] for i in range(7)]

    def _w_board(self):
        return np.any([self.wk, self.wq, self.wr,
                       self.wb, self.wn, self.wp], axis=0)

    def _b_board(self):
        return np.any([self.bk, self.bq, self.br,
                       self.bb, self.bn, self.bp], axis=0)

    def _board(self):
        return np.any([self._b_board(), self._w_board()], axis=0)

    def print_board(self):

        # Create the board
        to_print = np.full((8, 8), "- ")

        # Fill the board
        to_print[self.wk] = "K "
        to_print[self.wq] = "Q "
        to_print[self.wr] = "R "
        to_print[self.wb] = "B "
        to_print[self.wn] = "N "
        to_print[self.wp] = "P "

        to_print[self.bk] = "k "
        to_print[self.bq] = "q "
        to_print[self.br] = "r "
        to_print[self.bb] = "b "
        to_print[self.bn] = "n "
        to_print[self.bp] = "p "

        # Print the board
        print()
        for i in range(8):
            for j in range(8):
                print(to_print[i, j], end="")
            print()

    def new_game(self):
        self.wk[7, 4] = 1
        self.wq[7, 3] = 1
        self.wr[7, [0, 7]] = 1
        self.wb[7, [2, 5]] = 1
        self.wn[7, [1, 6]] = 1
        self.wp[6, 0:8] = 1

        self.bk[0, 4] = 1
        self.bq[0, 3] = 1
        self.br[0, [0, 7]] = 1
        self.bb[0, [2, 5]] = 1
        self.bn[0, [1, 6]] = 1
        self.bp[1, 0:8] = 1

    def _gen_rook_moves(self, coor):
        pass


B = Board()
B.new_game()
B.print_board()
