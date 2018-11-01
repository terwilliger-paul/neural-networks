import numpy as np

# Flip
FLIP_MASK_1 = np.array(0x00FF00FF00FF00FF, dtype=np.uint64)
FLIP_MASK_2 = np.array(0x0000FFFF0000FFFF, dtype=np.uint64)
CST32 = np.array(32, dtype=np.uint64)
CST16 = np.array(16, dtype=np.uint64)
CST8 = np.array(8, dtype=np.uint64)

# Mirror
H_MASK1 = np.array(0x5555555555555555, dtype=np.uint64)
H_MASK2 = np.array(0x3333333333333333, dtype=np.uint64)
H_MASK3 = np.array(0x0f0f0f0f0f0f0f0f, dtype=np.uint64)
CST1 = np.array(1, dtype=np.uint64)
CST2 = np.array(2, dtype=np.uint64)
CST4 = np.array(4, dtype=np.uint64)

# a1h8 axis flip
A1H8_1 = np.array(0x5500550055005500, dtype=np.uint64)
A1H8_2 = np.array(0x3333000033330000, dtype=np.uint64)
A1H8_3 = np.array(0x0f0f0f0f00000000, dtype=np.uint64)
CST28 = np.array(28, dtype=np.uint64)
CST14 = np.array(14, dtype=np.uint64)
CST7 = np.array(7, dtype=np.uint64)

# a8h1 axis flip
A8H1_1 = np.array(0xaa00aa00aa00aa00, dtype=np.uint64)
A8H1_2 = np.array(0xcccc0000cccc0000, dtype=np.uint64)
A8H1_4 = np.array(0xf0f0f0f00f0f0f0f, dtype=np.uint64)
CST36 = np.array(36, dtype=np.uint64)
CST18 = np.array(18, dtype=np.uint64)
CST9 = np.array(9, dtype=np.uint64)

def mirror(b64):
    b64 = ((b64 >> CST1) & H_MASK1) | ((b64 & H_MASK1) << CST1)
    b64 = ((b64 >> CST2) & H_MASK2) | ((b64 & H_MASK2) << CST2)
    b64 = ((b64 >> CST4) & H_MASK3) | ((b64 & H_MASK3) << CST4)
    return b64

def flip(b64):
    '''
    Flip a bitboard vertically about the centre ranks.
    Rank 1 is mapped to rank 8 and vice versa.
    b64 is any bitboard
    return bitboard x flipped vertically
    '''
    '''
    b64 = ((b64 >>  CST8) & FLIP_MASK_1) | ((b64 & FLIP_MASK_1) <<  CST8)
    b64 = ((b64 >> CST16) & FLIP_MASK_2) | ((b64 & FLIP_MASK_2) << CST16)
    b64 = ( b64 >> CST32) | ( b64 << CST32)
    return b64
    '''
    return b64.byteswap()

def a1h8_axis(b64):
   t = A1H8_3 & (b64 ^ (b64 << CST28))
   b64 ^= t ^ (t >> CST28)
   t = A1H8_2 & (b64 ^ (b64 << CST14))
   b64 ^= t ^ (t >> CST14)
   t = A1H8_1 & (b64 ^ (b64 <<  CST7))
   b64 ^= t ^ (t >>  CST7)
   return b64;

def a8h1_axis(b64):
    t = b64 ^ (b64 << CST36)
    b64 ^= A8H1_4 & (t ^ (b64 >> CST36))
    t = A8H1_2 & (b64 ^ (b64 << CST18))
    b64 ^= t ^ (t >> CST18)
    t = A8H1_1 & (b64 ^ (b64 <<  CST9))
    b64 ^= t ^ (t >>  CST9)
    return b64;

def rot90(b64):
    return flip(a1h8_axis(b64))

def rot180(b64):
    return mirror(flip(b64))

def view_bitboard(b64):
    b64_array = np.array([flip(b64)], dtype=np.uint64)
    return np.unpackbits(b64_array.view(np.uint8)).reshape((8, 8))

def gen_policy_dict():

    letters = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
    num_to_alg = {(i, j): letters[i]+str(j+1)
                     for i in range(8) for j in range(8)}
    policy_dict = {}
    inc = 0
    for i in range(8):
        for j in range(8):
            range8 = np.arange(8) + 1

            # Generate knight moves
            k = [(i+1, j+2), (i+1, j-2), (i+2, j+1), (i+2, j-1),
                 (i-1, j+2), (i-1, j-2), (i-2, j+1), (i-2, j-1)]
            # Generate diagonals
            d = [(i+di, j+di) for di in range8] +\
                [(i+di, j-di) for di in range8] +\
                [(i-di, j+di) for di in range8] +\
                [(i-di, j-di) for di in range8]
            # Generate files and ranks
            r = [(i+di, j) for di in range8] +\
                [(i-di, j) for di in range8] +\
                [(i, j+di) for di in range8] +\
                [(i, j-di) for di in range8]

            # Combine and purge
            end_loc = k + d + r
            purged = [e_loc for e_loc in end_loc
                      if e_loc in num_to_alg.keys()]

            for purge in purged:
                policy_dict[inc] = num_to_alg[(i, j)] + num_to_alg[purge]
                inc += 1

    return policy_dict, num_to_alg

'''
12 bitboards each with 64 bits + 1 white_to_move bit + 4 castling bits + 8 ep_bits
(12*64) + 1 + 4 + 8 = 781 inputs
1792 outputs
'''

print(max(gen_policy_dict()[0].keys()))
