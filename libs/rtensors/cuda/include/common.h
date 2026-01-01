#pragma once
#include <stdint.h>
#include <stddef.h>

// Operation codes for elementwise operations
enum OpCode : uint8_t {
    OP_ADD = 0,
    OP_SUB = 1,
    OP_MUL = 2,
};

enum ReductionOpCode: uint8_t {
    OP_SUM = 1,
    OP_PROD = 2,
    OP_MAX = 3,
    OP_MIN = 4,
    OP_MEAN = 5,
    OP_SOFTMAX = 6
};

// Contiguity types for matrix layouts
enum ContiguityType : uint8_t {
    ROW_MAJOR = 0,
    COLUMN_MAJOR = 1,
};

#define WARP_SIZE 32
#define MIN_BLOCK_SIZE 32
#define MAX_BLOCK_SIZE 1024

// clamp and align block size to warp boundaries
#define ALIGN_BLOCK_SIZE(block_size) \
    ((block_size) < MIN_BLOCK_SIZE ? MIN_BLOCK_SIZE : \
     (block_size) > MAX_BLOCK_SIZE ? MAX_BLOCK_SIZE : \
     ((block_size) / WARP_SIZE) * WARP_SIZE)