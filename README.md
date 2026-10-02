# nnue-rs

A small, dependency-free Rust library for loading and evaluating NNUE
(Efficiently Updatable Neural Network) chess networks.

[![Crates.io](https://img.shields.io/crates/v/nnue-rs.svg)](https://crates.io/crates/nnue-rs)
[![Documentation](https://docs.rs/nnue-rs/badge.svg)](https://docs.rs/nnue-rs)
[![License](https://img.shields.io/crates/l/nnue-rs.svg)](https://github.com/hedgeg0d/nnue-rs#license)

## Features

- **Load Stockfish networks**: Read `.nnue` files (or embedded bytes); the
  architecture (`SFNNv10`, `HalfKAv2_hm`, `HalfKAv2` or classic `HalfKP`) is detected
  automatically from the file header.
- **FEN support**: Evaluate a position directly from a FEN string.
- **Generic board interface**: Integrate with any engine via the `Board` trait —
  no board conversions needed.
- **Incremental evaluation**: Advance an accumulator across moves instead of
  recomputing, for fast search.
- **Cross-platform with SIMD**: A runtime-detected AVX2 fast path on x86-64, with
  a portable scalar fallback everywhere else. No dependencies.

## 0.4.2 backend changes

- Fuse common move updates into one accumulator pass, retaining wrapping i16
  arithmetic and the generic path for other change lists.
- Prepack the first affine layer of `HalfKAv2_hm` and `SFNNv10` networks for an
  AVX2 kernel that processes all 16 outputs together and skips zero input blocks.
- Preserve public APIs, network-file formats and integer evaluation results.
  Row-major weights remain available for the portable fallback. For a 3072-wide
  network, packed weights add 384 KiB; no second network is needed.
- No new CPU requirement: AVX2 remains runtime-detected. SSE/NEON backends,
  refresh caches and incremental threat discovery are not added in this release.

Performance depends on the network, position and hardware. Benchmarks are not
an Elo estimate.

## Quick Start

Add this to your `Cargo.toml`:

```toml
[dependencies]
nnue-rs = "0.4.2"
```

### Basic Usage

```rust
use nnue_rs::Network;

// Load a network (e.g. a Stockfish .nnue file)
let net = Network::from_file("net.nnue")?;

// Evaluate by FEN. The score is in internal units from the side-to-move
// perspective.
let start = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
let score = net.evaluate_fen(start)?;
println!("score: {score}");
```

### Integration with a Custom Board

Implement the `Board` trait for your own position type. Squares are `0..64`
with `a1 = 0`, `b1 = 1`, ..., `h8 = 63`.

```rust
use nnue_rs::{Board, Color, Piece, Network};

struct MyPosition {
    // your board representation
}

impl Board for MyPosition {
    fn side_to_move(&self) -> Color {
        // whose turn it is
    }

    fn king_square(&self, color: Color) -> u8 {
        // square (0-63) of `color`'s king
    }

    fn for_each_piece(&self, f: &mut dyn FnMut(u8, Piece)) {
        // call `f(square, piece)` for every piece on the board
    }
}

let net = Network::from_file("net.nnue")?;
let pos = MyPosition::new();
let score = net.evaluate(&pos);
```

### Incremental Evaluation

In a search, advancing the accumulator move-by-move is much faster than a full
recompute. Keep an `Accumulator` alongside each node:

```rust
use nnue_rs::Network;

let net = Network::from_file("net.nnue")?;

// Compute the accumulator once for the root position.
let root_acc = net.accumulator(&parent);

// For each child, advance into a fresh accumulator slot.
let mut child_acc = net.empty_accumulator();
net.update(&parent, &child, &root_acc, &mut child_acc);

// Evaluate. Side to move is passed separately so the same accumulator can be
// reused across a null move.
let score = net.evaluate_accumulator(&child_acc, child.side_to_move());
```

`update` derives the changed features by diffing the two boards, so every move
type (captures, en passant, promotions, castling) is handled, and a king move
transparently triggers a refresh of that side.

If the engine already knows the move, `update_changes` accepts complete
`removed` and `added` piece lists instead, avoiding the board-diff scans.

## Architecture Support

| Architecture | Networks | Load | Evaluate |
|--------------|----------|------|----------|
| `SFNNv10` | Stockfish 18 big nets (threat inputs) | yes | yes |
| `HalfKAv2_hm` | Stockfish SFNNv5-v9 (SF 16/17, SF 18 small) | yes | yes |
| `HalfKAv2` | Stockfish SFNNv2-v4 (SF 14) | yes | yes |
| `HalfKP` | classic Stockfish NNUE (SF 12-14) | yes | yes |

The architecture is selected automatically from the network header; query it
with `Network::arch()`. More feature sets are planned.

## API Reference

### `Network`

- `from_file(path)` / `from_bytes(bytes)` / `from_reader(reader)` — load a network
- `evaluate(&board)` — evaluate any `Board`
- `evaluate_fen(fen)` — evaluate a FEN string
- `accumulator(&board)` — fresh accumulator for a position
- `empty_accumulator()` — zeroed accumulator for reuse pools
- `refresh(&board, &mut acc)` — recompute an accumulator
- `update(&parent_board, &child_board, &parent_acc, &mut child_acc)` — incremental step
- `update_changes(&parent_board, &child_board, removed, added, &parent_acc, &mut child_acc)` — incremental step from known piece changes
- `evaluate_accumulator(&acc, stm)` — evaluate a ready accumulator
- `arch()` — the detected feature-set architecture

### Traits & Types

- `Board` — implement for your position (`side_to_move`, `king_square`, `for_each_piece`)
- `FenBoard` — a `Board` parsed from a FEN string
- `Arch`, `Color`, `Piece`, `PieceKind`, `Accumulator`, `Error`

## Testing

Kernel tests compare portable and AVX2 arithmetic against independent dense
and wrapping references, including unaligned slices, tails and sparse inputs.
Synthetic accumulator tests cover all legacy feature sets without external files.
Optional real-network regressions compare every accumulator field:

```bash
NNUE_TEST_NET=/path/to/sf17.nnue \
SFNNV10_NET=/path/to/sf18-big.nnue \
NNUE_SMALL_NET=/path/to/small.nnue \
cargo test --release
```

Real-network checks are skipped when their fixture variables are unset; they
are not a replacement for the always-on synthetic and kernel tests.

## License

Licensed under the [MIT license](LICENSE).
