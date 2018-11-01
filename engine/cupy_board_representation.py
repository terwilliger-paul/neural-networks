import board_representation_functions as bf

import numpy as np
import cupy as cp
import time

def view_bitboard(b64):
    b64_array = np.array([b64], dtype=np.uint64)
    return np.unpackbits(b64_array.byteswap().view(np.uint8)).reshape((8, 8))

v56 = np.uint64(56)
v40 = np.uint64(40)
v24 = np.uint64(24)
v8_ = np.uint64(8)
v_flip1 = np.uint64(0x00ff000000000000)
v_flip2 = np.uint64(0x0000ff0000000000)
v_flip3 = np.uint64(0x000000ff00000000)
v_flip4 = np.uint64(0x00000000ff000000)
v_flip5 = np.uint64(0x0000000000ff0000)
v_flip6 = np.uint64(0x000000000000ff00)

def v_flip_slow(b64):
    return ((b64 << v56) |
            ((b64 << v40) & v_flip1) |
            ((b64 << v24) & v_flip2) |
            ((b64 << v8_) & v_flip3) |
            ((b64 >> v8_) & v_flip4) |
            ((b64 >> v24) & v_flip5) |
            ((b64 >> v40) & v_flip6) |
            (b64 >> v56))

def v_flip(b64):
    return b64.byteswap()

np_h_flip1 = np.uint64(0x5555555555555555)
np_h_flip2 = np.uint64(0x3333333333333333)
np_h_flip3 = np.uint64(0x0f0f0f0f0f0f0f0f)
np_h_flip4 = np.uint64(1)
np_h_flip5 = np.uint64(2)
np_h_flip6 = np.uint64(4)
def np_h_flip(b64):
   b64 = ((b64 >> np_h_flip4) & np_h_flip1) | ((b64 & np_h_flip1) << np_h_flip4)
   b64 = ((b64 >> np_h_flip5) & np_h_flip2) | ((b64 & np_h_flip2) << np_h_flip5)
   b64 = ((b64 >> np_h_flip6) & np_h_flip3) | ((b64 & np_h_flip3) << np_h_flip6)
   return b64

a1h8_1 = np.uint64(0x5500550055005500)
a1h8_2 = np.uint64(0x3333000033330000)
a1h8_3 = np.uint64(0x0f0f0f0f00000000)
def a1h8_axis(b64):
   t  = a1h8_3 & (b64 ^ (b64 << 28))
   b64 ^=       t ^ (t >> 28)
   t  = a1h8_2 & (b64 ^ (b64 << 14))
   b64 ^=       t ^ (t >> 14)
   t  = a1h8_1 & (b64 ^ (b64 <<  7))
   b64 ^=       t ^ (t >>  7)
   return b64;

################

np_board = np.uint64(2**62 + 100)
np_foo = np.uint64(100)
print(np_foo | np_board)
print()
print(np_board)

print(view_bitboard(np_board))
print(view_bitboard(v_flip(np_board)))

print(v_flip(np_board).flags)

################

bpack = cp.array([[0, 1, 1, 1, 1, 0, 0, 0],
                  [0, 1, 0, 0, 0, 1, 0, 0],
                  [0, 1, 0, 0, 0, 1, 0, 0],
                  [0, 1, 0, 0, 1, 0, 0, 0],
                  [0, 1, 1, 1, 0, 0, 0, 0],
                  [0, 1, 0, 1, 0, 0, 0, 0],
                  [0, 1, 0, 0, 1, 0, 0, 0],
                  [0, 1, 0, 0, 0, 1, 0, 0]], dtype=cp.uint8)

board = (cp.packbits(bpack.reshape(64)).view(cp.uint64))[0]
foo = cp.uint64(2**62 + 100)
print(type(board))
print(type(foo))


print(bf.view_bitboard(board))
print()
print(bf.view_bitboard(bf.mirror(board)))

trials = 100000

start = time.time()
for _ in range(trials):
    np_h_flip(np_board)
print(time.time() - start)

cp.cuda.stream.get_current_stream().synchronize()
start = time.time()
for _ in range(trials):
    #x_gpu = bf.flip(board)
    pass
cp.cuda.stream.get_current_stream().synchronize()
print(time.time() - start)

cp.cuda.stream.get_current_stream().synchronize()
start = time.time()
for _ in range(trials):
    foo_gpu = cp.array([foo], dtype=cp.uint64)
cp.cuda.stream.get_current_stream().synchronize()
print(time.time() - start)

cp.cuda.stream.get_current_stream().synchronize()
start = time.time()
for _ in range(trials):
    np_foo = cp.asnumpy(foo_gpu)
cp.cuda.stream.get_current_stream().synchronize()
print(time.time() - start)
