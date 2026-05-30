# nnue-rs

A Rust library for loading and evaluating NNUE (Efficiently Updatable Neural
Network) networks.

The goal of the crate is to be a small, dependency-free, cross-platform engine
for NNUE inference that can grow to support multiple network architectures.

## Status

| Architecture | Load | Evaluate |
|--------------|------|----------|
| HalfKAv2_hm (Stockfish SFNNv5+) | yes | yes |

More feature sets and architectures are planned.

## Usage

Two ways to feed positions, mirroring `polyglot-book-rs`:

By FEN:

```rust
use nnue_rs::Network;

let net = Network::from_file("net.nnue")?;
let cp = net.evaluate_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")?;
```

By implementing the `Board` trait on your own position type:

```rust
use nnue_rs::{Board, Color, Piece, Network};

impl Board for MyPosition {
    fn side_to_move(&self) -> Color { /* ... */ }
    fn king_square(&self, color: Color) -> u8 { /* ... */ }
    fn for_each_piece(&self, f: &mut dyn FnMut(u8, Piece)) { /* ... */ }
}

let score = net.evaluate(&my_position);
```

Squares are `0..64` with `a1 = 0`. The returned score is in internal network
units from the side-to-move perspective.

## License

MIT
