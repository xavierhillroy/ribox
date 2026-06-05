#ifndef LGP_CONFIG_H
#define LGP_CONFIG_H

#include <cstdint>
#include <array>

// =============================================================================
// LGPConfig: Compile-time configuration for the LGP system.
//
// All values here are `constexpr` -- they are baked into the binary at compile
// time. Changing any of these values requires a rebuild, but in exchange the
// compiler can use them for sizing, masking, and loop unrolling, and the CPU
// pays no runtime cost to read them.
//
// This file is intentionally parameter-only: it answers "how big is the
// population?", "how many registers?", "which constants can programs use?".
// The bit layout that turns these parameters into an instruction encoding
// lives in ISA.h.
// =============================================================================

namespace LGPConfig {

    // -------------------------------------------------------------------------
    // Utility: compile-time integer log2.
    // Used to derive bit-widths from pool sizes (e.g. 8 registers -> 3 bits).
    // Recursive because C++14 constexpr functions can't use loops portably;
    // by C++17 we could rewrite as a loop, but this is fine and runs at
    // compile time only.
    // -------------------------------------------------------------------------
    constexpr int log2(int n) {
        return (n <= 1) ? 0 : 1 + log2(n / 2);
    }

    // =========================================================================
    // Evolutionary loop parameters
    // =========================================================================

    constexpr int MAX_GENERATIONS = 1000;  // Outer loop cap on the evolutionary run.

    // =========================================================================
    // Population and program sizing
    //
    // Memory layout: all programs are stored as one flat uint32_t buffer of
    // length TOTAL_INSTRUCTIONS. Program i occupies the slice
    //   [i * MAX_PROGRAM_SIZE, i * MAX_PROGRAM_SIZE + program_lengths[i])
    // Slots past program_lengths[i] exist but are not executed by the
    // interpreter. This "program-major" layout is warp-friendly: on the GPU,
    // one warp evaluating one program reads its instruction stream
    // contiguously.
    // =========================================================================

    constexpr int POPULATION_SIZE = 1000;    // Number of programs per generation. MUST BE MULTIPLE OF 2... 
    static_assert((POPULATION_SIZE%2) == 0, "POPULATION_SIZE must be mult of 2");
                                                 // Small for now while testing;
                                                 // will be bumped up later.
    constexpr int MAX_PROGRAM_SIZE = 50;   // Hard cap on instructions per program. MAX MAX FOR NOW IS 255 because using uint8 att some places
                                                 // All programs reserve this much
                                                 // space; variable length is tracked
                                                 // separately in program_lengths[].
    constexpr int SIZE_SENTINEL = 255; // max size for uint8 acts as a flag for invalid length 
    constexpr int STARTING_PROGRAM_SIZE = 8;    // Initial length of every program
                                                 // at generation 0.
    constexpr int TOTAL_INSTRUCTIONS = POPULATION_SIZE * MAX_PROGRAM_SIZE;
                                                 // Total slots in the flat buffer.

    // =========================================================================
    // Register file
    //
    // Each program has its own set of scalar float registers. Register IDs
    // are packed into the instruction word, so NUM_REGISTERS must be a power
    // of 2 and must fit in the index field (currently 7 bits -> max 128).
    // =========================================================================
    // IMPORTANT MASK SIZE WILL BASICALLY SET THE BITS ACTUALLY USED TO REPRESENT THE NUM REG IN BINARY 
    constexpr int NUM_REGISTERS = 8;                   // MUST be a power of 2.
    constexpr int REGISTER_BITS = log2(NUM_REGISTERS); // Bits needed to encode an ID.
    constexpr int REGISTER_MASK = NUM_REGISTERS - 1;   // AND-mask that clips any value
                                                        // to a valid register index.
                                                        // For NUM_REGISTERS=8 this is 0b111.

    // =========================================================================
    // Constant pool
    //
    // When an instruction's src2 field is flagged as "constant mode", its
    // index selects into CONSTANTS[] instead of the register file. These are
    // fixed at compile time -- programs cannot evolve new constants, only
    // pick from this list.
    // =========================================================================
    //
    constexpr int NUM_CONSTANTS = NUM_REGISTERS;
    constexpr int CONSTANT_BITS = log2(NUM_CONSTANTS);
    constexpr int CONSTANT_MASK = NUM_CONSTANTS - 1; // produces 7 which is 111 


    constexpr auto CONSTANTS = std::to_array<float>({
    0.0f, 1.0f, -1.0f, 3.14159f, 0.5f, 2.0f, 10.0f, 100.0f
    });
    static_assert(CONSTANTS.size() == NUM_CONSTANTS,
              "CONSTANTS pool size doesn't match NUM_CONSTANTS");

    // The src2 index field in an instruction is 7 bits wide and is used for
    // BOTH registers (when mode=0) and constants (when mode=1). Keeping the
    // two pools equal-sized means one mask suffices for both cases. If you
    // ever want asymmetric pools, this assumption breaks and the encoder/
    // decoder need reworking -- the static_assert will catch it at compile
    // time.
    static_assert(NUM_REGISTERS == NUM_CONSTANTS,
                  "src2 index field assumes NUM_REGISTERS == NUM_CONSTANTS");

    // =========================================================================
    // Instruction set size
    //
    // The opcode occupies the low byte of the instruction word. Only
    // OPERATIONS_BITS of that byte are meaningful; the rest must be masked.
    // The concrete opcode enum lives in ISA.h (ADD, SUB, MUL, ...).
    // =========================================================================

    constexpr int NUM_OPERATIONS = 8;                    // Max 2^8 = 256 if we used the whole byte.
    constexpr int OPERATIONS_BITS = log2(NUM_OPERATIONS); // Active bits in the opcode field.
    constexpr int OPERATION_MASK = NUM_OPERATIONS - 1;   // AND-mask that clips a random byte
                                                           // to a valid opcode index.
                                                           // Example: 8 ops -> mask = 0b0000_0111.

    // =========================================================================
    // Variation rates
    //
    // These control the probabilities applied during mutation and crossover.
    // Names correspond to specific operators that will be implemented later;
    // documenting semantics here as placeholders.
    // =========================================================================

    constexpr float INSERT_TAIL_RATE = 0.8f;   // P(insert at end vs. random position) in macro-mutation.
    constexpr float DELETE_TAIL_RATE = 0.75f;  // P(swap operands) micro-mutation rate.
    constexpr float REPLACE = 0.8f;   // P(replace instruction) macro-mutation rate.
    constexpr float MICRO_RATE = 0.8f;   // P(apply micro-mutation to a given instruction).

    constexpr float CROSSOVER_RATE = 1.0f;   // P(apply crossover) per offspring.
    constexpr float ELITES = 0.1f;   // Fraction of population preserved as elites.
    constexpr int ELITE_COUNT = static_cast<int>(LGPConfig::POPULATION_SIZE * LGPConfig::ELITES);
    constexpr int NON_ELITE_COUNT = LGPConfig::POPULATION_SIZE - ELITE_COUNT;
    static_assert(NON_ELITE_COUNT % 2 == 0,
              "Non-elite slots must be even -- vary_pair produces two children");
    constexpr int TOURNAMENT_SIZE = 3;


    // =========================================================================
    // RNG
    // =========================================================================

    constexpr uint32_t SEED = 42;  // Fixed seed -> deterministic runs for testing.

    constexpr int NUM_CONTEXTS = 32; // number of contexts / environments...

}  // namespace LGPConfig

#endif  // LGP_CONFIG_H