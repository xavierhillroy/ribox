# Ribox — Sequential LGP Reference: Design Report

*Deliverable 1. First step toward a warp-level GPU implementation.*

## 1. What it is, what it isn't

Ribox-sequential is a single-threaded C++ implementation of Linear Genetic Programming (LGP). It runs the full evolutionary loop — initialize, evaluate, select, vary, repeat — against symbolic-regression targets, and it is covered end-to-end by a unit-test suite.

It is not a competitive symbolic-regression engine, and it is not trying to be. There are no benchmark problem libraries, no train/test splits, no R² reporting, no large datasets. Adding any of that would serve SR performance, which is not what this deliverable is testing.

What it is is two things at once:

A correctness oracle. The GPU version will be validated against it. Every design choice here is the algorithmic blueprint the device code is derived from.

A hardware-aware rehearsal. The whole point of writing the CPU version first is to make the GPU port mechanical rather than exploratory. So the CPU code is written in the shape the GPU wants, even where a more idiomatic C++ choice existed.

### Set up for warp-level execution

The near-term hypothesis the larger project tests is warp-level execution: map one GP individual to one CUDA warp of 32 threads, with all 32 lanes running the same program in lockstep, each lane evaluating a different context (a fitness case in SR; later, an environment instance in RL). This sits between block-level mappings (which cap how many distinct individuals stay resident) and thread-level mappings (which expose heavy divergence).

The CPU reference is built so this mapping is the natural reading of the code, not a later retrofit. Concretely, the design rules that hold throughout:

Flat contiguous arrays, no pointer chasing. Programs live in one flat instruction buffer; per-program length is a parallel array. No `vector<vector<T>>`.

No runtime polymorphism on hot paths. The interpreter is a plain free function, not a virtual method. Virtuals don't port to the device.

Compile-time constants for anything affecting layout. Register count, max program size, context count. On GPU these become template parameters or `__constant__` memory.

The interpreter's "decode-once, apply-across-contexts" loop is the warp broadcast. Same logical structure on both sides; only the expression of parallelism changes.

### First steps toward RL

This deliverable is SR-only and the interpreter is stateless. The RL extension (GPU-resident, trajectory-dependent environments) is where warp-level execution is expected to dominate, but it is deliberately deferred. The seams are left in the right places: the chunked context loop (below) is exactly the pattern RL needs for multiple environment instances per agent, and the interpreter is a free function that will factor cleanly into a stateful variant when the time comes. Nothing here forecloses RL; nothing here pretends to implement it yet.

## 2. ISA — bit-packing

Each instruction is a single 32-bit word, four 8-bit fields:

```
 bit:  31 ........ 24 | 23 ........ 16 | 15 .......... 8 | 7 ........... 0
       [    SRC2     ] [    SRC1      ] [     DEST       ] [      OP      ]
       [mode | index ] [  reg index   ] [   reg index    ] [   opcode     ]
```

A program is just an array of these words plus a length. This is the representation that makes the whole project tractable: a flat, fixed-width, contiguous executable form maps cleanly onto accelerator memory and copies as raw bytes with no serialization.

Field decisions worth recording:

SRC2 carries a mode bit. Its high bit selects register (0) vs constant (1); its low bits index either the register file or the constant pool. The invariant NUM_REGISTERS == NUM_CONSTANTS lets a single mask handle both modes — no separate decode path for constants.

Only the low log2(N) bits of each field are meaningful. The remaining bits in each byte are reserved, leaving room to widen the register file or opcode set later without changing the word layout.

Eight operations: ADD SUB MUL DIV SIN COS LT GT. DIV is protected (returns 1.0 on divide-by-zero), written branchlessly so it lowers to a predicated select rather than a divergent branch on GPU. LT/GT write 1.0/0.0 into a float register. SIN/COS are unary — they read SRC2 (so a mutated unary instruction can still reach both registers and constants) and ignore SRC1.

The unary ops reading SRC2 is a small but deliberate call: it means SRC1 on a SIN/COS instruction is an intron. That's fine, and arguably useful — neutral mutation drift on introns is harmless and there's schema-protection literature suggesting it helps. Special-casing arity inside the variation operators would couple ISA structure to mutation logic for no real gain.

## 3. Design decisions — variation

The variation pipeline is fixed: select two parents, crossover into two children, mutate each child. Crossover and mutation are each gated by an independent rate, so any offspring may pass through unchanged.

### Crossover: single-point

Single-point, chosen over two-point, uniform, and Brameier-style segment crossover. The reasoning is hardware-first:

Uniform needs MAX_PROGRAM_SIZE independent random draws per offspring — on GPU that's orders of magnitude more RNG work per warp. Single-point needs one draw.

Two-point and segment crossover add arithmetic and, in the segment case, variable-length blocks that need clamping — for a benefit that's negligible at these population sizes and program lengths.

Single-point is one random cut and two contiguous copies per child. The cut is drawn from [1, min(len_a, len_b)) so both children inherit at least one instruction from each parent; parents shorter than length 2 are cloned rather than cut. Each child inherits the length of the parent whose suffix it takes, so two parents of different lengths naturally produce length variation — no padding, no explicit length-mutation operator needed. The copies are memcpy on CPU, which becomes a coalesced/warp copy on GPU: same operation, different machinery.

### Mutation: micro-mutation, tail insert/delete

We have 3 types of mutation, all memory friendly that follow the classical Brameier macro + micro scheme with multiple mutations for selected offsprings.

Micro-mutation regenerates one field of one instruction: it picks one of five fields (op, dest, src1, src2-index, src2-mode) uniformly, then rewrites just that field with correct field-width masking. The mode field is a toggle, not a re-roll, so it always produces a real semantic change rather than a 50% no-op.

To allow size to change and evolve we added tail insert and delete. Removing and adding from the tail means we dont have to shift the instructions within the instruction buffer.

It is important to note when porting to GPU that having 3 random mutation selection each thread chooses can cause divergence. However, the deliverables are only focusing on porting evaluation to the GPU, and mutation is already likely just a small portion of the time.

### Selection and survival

K-tournament with replacement (one parent index per call). Elitism copies the top-E verbatim into the next generation with fitness carried forward, so elites aren't re-evaluated. Two population buffers ping-pong each generation — a single flag selects which is live, so there's no per-generation reallocation or copying. That double-buffer is itself a GPU rehearsal: it's how the device version will manage its two population buffers.

## 4. The num-context loop

This is the structural heart of the warp-level rehearsal, so it's worth being explicit about.

The interpreter does not loop one-context-at-a-time. It decodes each instruction once, then applies that decoded instruction across all NUM_CONTEXTS contexts before moving to the next instruction:

```
for each instruction in program:
    decode instruction once
    for c in 0 .. NUM_CONTEXTS:
        apply decoded instruction to context c's registers
```

On CPU this is just an inner loop. On GPU it is the warp: the "decode once" happens implicitly because all 32 lanes fetch the same instruction word, and the inner loop over contexts becomes the 32 lanes each operating on their own context in lockstep. Writing the CPU loop in this order — decode outer, context inner — is what makes the port a near-transcription instead of a rewrite.

NUM_CONTEXTS is a compile-time constant fixed at the warp width of 32. To allow context counts that aren't exactly 32, evaluation is chunked: the dataset pads its fitness-case count up to a multiple of NUM_CONTEXTS, and the evaluator issues full-width interpreter calls over each chunk, accumulating only the valid (non-padding) outputs into the fitness sum. Two layout notes carried forward to GPU work, recorded so they aren't mis-remembered as settled:

The CPU register file is registers[c * NUM_REGISTERS + i]. This is a convenient CPU layout, not a literal mirror of the GPU layout. In warp-per-program mode the natural device layout is per-lane registers; if the register file ends up in shared memory, the bank-conflict-free stride is the opposite one, registers[i * NUM_CONTEXTS + c].

Variable program length is divergence-free within a warp (all 32 lanes run the same program, so instruction counts match by construction). Different- length programs across different warps is just inter-warp scheduling, which the hardware absorbs. This is precisely why the warp-level mapping is the one the project bets on.

## Status

Deliverable 1 is functionally complete: the full loop runs and is covered by tests (encode/decode, every opcode, interpreter contract, dataset padding, chunk-boundary evaluation, crossover, mutation, multi-generation stability). Open items deferred to the GPU work: the precise validation tolerance (bit-exact agreement is off the table — FP non-associativity, mt19937 vs curand, SFU-approximate transcendentals — so the contract is per-program, per-input agreement within epsilon); the stateful interpreter for RL; and Nsight profiling, where warp efficiency will be the empirical evidence for the central claim.s