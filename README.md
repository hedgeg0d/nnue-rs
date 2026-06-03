# nnue-rs

A small, dependency-free Rust library for loading and evaluating NNUE
(Efficiently Updatable Neural Network) chess networks.

[![Crates.io](https://img.shields.io/crates/v/nnue-rs.svg)](https://crates.io/crates/nnue-rs)
[![Documentation](https://docs.rs/nnue-rs/badge.svg)](https://docs.rs/nnue-rs)
[![License](https://img.shields.io/crates/l/nnue-rs.svg)](https://github.com/hedgeg0d/nnue-rs#license)

## Features

- **Load Stockfish networks**: Read `.nnue` files (or embedded bytes); the
  architecture (`HalfKAv2_hm`, `HalfKAv2` or classic `HalfKP`) is detected
  automatically from the file header.
- **FEN support**: Evaluate a position directly from a FEN string.
- **Generic board interface**: Integrate with any engine via the `Board` trait —
  no board conversions needed.
- **Incremental evaluation**: Advance an accumulator across moves instead of
  recomputing, for fast search.
- **Cross-platform with SIMD**: A runtime-detected AVX2 fast path on x86-64, with
  a portable scalar fallback everywhere else. No dependencies.

## Quick Start

Add this to your `Cargo.toml`:

```toml
[dependencies]
nnue-rs = "0.3.1"
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

## Architecture Support

| Architecture | Networks | Load | Evaluate |
|--------------|----------|------|----------|
| `HalfKAv2_hm` | Stockfish SFNNv5+ (SF 16/17/18) | yes | yes |
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
- `evaluate_accumulator(&acc, stm)` — evaluate a ready accumulator
- `arch()` — the detected feature-set architecture

### Traits & Types

- `Board` — implement for your position (`side_to_move`, `king_square`, `for_each_piece`)
- `FenBoard` — a `Board` parsed from a FEN string
- `Arch`, `Color`, `Piece`, `PieceKind`, `Accumulator`, `Error`

## License

Licensed under the [MIT license](LICENSE).
