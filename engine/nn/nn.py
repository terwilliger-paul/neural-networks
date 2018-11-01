import numpy as np
import chess
from itertools import chain
import re
import time

class FenParser():
  def __init__(self, fen_str):
    self.fen_str = fen_str

  def parse(self):
    ranks = self.fen_str.split(" ")[0].split("/")
    pieces_on_all_ranks = [self.parse_rank(rank) for rank in ranks]
    return pieces_on_all_ranks

  def parse_rank(self, rank):
    rank_re = re.compile("(\d|[kqbnrpKQBNRP])")
    piece_tokens = rank_re.findall(rank)
    pieces = self.flatten(map(self.expand_or_noop, piece_tokens))
    return pieces

  def flatten(self, lst):
    return list(chain(*lst))

  def expand_or_noop(self, piece_str):
    piece_re = re.compile("([kqbnrpKQBNRP])")
    retval = ""
    if piece_re.match(piece_str):
      retval = piece_str
    else:
      retval = self.expand(piece_str)
    return retval

  def expand(self, num_str):
    return int(num_str)*" "

def vectorize_board(chess_board):

    wk = np.zeros((8, 8), dtype=bool)
    wq = np.zeros((8, 8), dtype=bool)
    wr = np.zeros((8, 8), dtype=bool)
    wb = np.zeros((8, 8), dtype=bool)
    wn = np.zeros((8, 8), dtype=bool)
    wp = np.zeros((8, 8), dtype=bool)

    bk = np.zeros((8, 8), dtype=bool)
    bq = np.zeros((8, 8), dtype=bool)
    br = np.zeros((8, 8), dtype=bool)
    bb = np.zeros((8, 8), dtype=bool)
    bn = np.zeros((8, 8), dtype=bool)
    bp = np.zeros((8, 8), dtype=bool)

    location = np.array([8, 8], dtype=np.int)

    # Parse out FEN
    fen_fields = chess_board.fen().split(" ")
    assert len(fen_fields) == 6

    matrix_board = FenParser(chess_board.fen()).parse()
    print(matrix_board)

start = time.time()
for _ in range(100):
    # Create chess board
    board = chess.Board()
    board.push_san("d4")
    board.push_san("Nf6")
    print(board)
    print(board.fen())

    # Turn chess board into input vector
    print(vectorize_board(board))
print(time.time() - start)
