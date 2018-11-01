import cupy as cp

# Flip
FLIP_MASK_1 = cp.array(0x00FF00FF00FF00FF, dtype=cp.uint64)
FLIP_MASK_2 = cp.array(0x0000FFFF0000FFFF, dtype=cp.uint64)
CST32 = cp.array(32, dtype=cp.uint64)
CST16 = cp.array(16, dtype=cp.uint64)
CST8 = cp.array(8, dtype=cp.uint64)

# Mirror
H_MASK1 = cp.array(0x5555555555555555, dtype=cp.uint64)
H_MASK2 = cp.array(0x3333333333333333, dtype=cp.uint64)
H_MASK3 = cp.array(0x0f0f0f0f0f0f0f0f, dtype=cp.uint64)
CST1 = cp.array(1, dtype=cp.uint64)
CST2 = cp.array(2, dtype=cp.uint64)
CST4 = cp.array(4, dtype=cp.uint64)

# a1h8 axis flip
A1H8_1 = cp.array(0x5500550055005500, dtype=cp.uint64)
A1H8_2 = cp.array(0x3333000033330000, dtype=cp.uint64)
A1H8_3 = cp.array(0x0f0f0f0f00000000, dtype=cp.uint64)
CST28 = cp.array(28, dtype=cp.uint64)
CST14 = cp.array(14, dtype=cp.uint64)
CST7 = cp.array(7, dtype=cp.uint64)

# a8h1 axis flip
A8H1_1 = cp.array(0xaa00aa00aa00aa00, dtype=cp.uint64)
A8H1_2 = cp.array(0xcccc0000cccc0000, dtype=cp.uint64)
A8H1_4 = cp.array(0xf0f0f0f00f0f0f0f, dtype=cp.uint64)
CST36 = cp.array(36, dtype=cp.uint64)
CST18 = cp.array(18, dtype=cp.uint64)
CST9 = cp.array(9, dtype=cp.uint64)

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
    b64 = ((b64 >>  CST8) & FLIP_MASK_1) | ((b64 & FLIP_MASK_1) <<  CST8)
    b64 = ((b64 >> CST16) & FLIP_MASK_2) | ((b64 & FLIP_MASK_2) << CST16)
    b64 = ( b64 >> CST32) | ( b64 << CST32)
    return b64
#return b64.byteswap()

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
    b64_array = cp.array([flip(b64)], dtype=cp.uint64)
    return cp.unpackbits(b64_array.view(cp.uint8)).reshape((8, 8))
