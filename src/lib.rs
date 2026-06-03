//! Load and evaluate NNUE (Efficiently Updatable Neural Network) chess networks.
//!
//! The crate is small, dependency-free and cross-platform, with an optional
//! AVX2 fast path (used automatically at runtime when available, with a scalar
//! fallback everywhere else).
//!
//! # Architectures
//!
//! The architecture is detected automatically from the network file's header:
//!
//! - `HalfKAv2_hm` — Stockfish SFNNv5 and later (Stockfish 16/17/18 nets).
//! - `HalfKAv2` — the non-mirrored predecessor (Stockfish SFNNv2-v4, SF 14).
//! - `HalfKP` — the classic Stockfish NNUE feature set (Stockfish 12-14 nets).
//!
//! # Integration
//!
//! Feed positions either by FEN or by implementing the [`Board`] trait on your
//! own position type.
//!
//! By FEN:
//!
//! ```no_run
//! use nnue_rs::Network;
//!
//! let net = Network::from_file("net.nnue")?;
//! let cp = net.evaluate_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")?;
//! # Ok::<(), nnue_rs::Error>(())
//! ```
//!
//! By implementing [`Board`]:
//!
//! ```no_run
//! use nnue_rs::{Board, Color, Piece, Network};
//!
//! # struct MyPosition;
//! impl Board for MyPosition {
//!     fn side_to_move(&self) -> Color { /* ... */ todo!() }
//!     fn king_square(&self, color: Color) -> u8 { /* ... */ todo!() }
//!     fn for_each_piece(&self, f: &mut dyn FnMut(u8, Piece)) { /* ... */ todo!() }
//! }
//!
//! # let net: Network = unreachable!();
//! # let position = MyPosition;
//! let score = net.evaluate(&position);
//! ```
//!
//! # Incremental evaluation
//!
//! For a search, advance an [`Accumulator`] as moves are made instead of
//! recomputing from scratch. See [`Network::accumulator`], [`Network::update`]
//! and [`Network::evaluate_accumulator`].

mod error;
mod feature;
mod fen;
mod leb128;
mod network;
mod simd;
mod types;

pub use error::{Error, Result};
pub use feature::Arch;
pub use fen::FenBoard;
pub use network::{Accumulator, Network};
pub use types::{Board, Color, Piece, PieceKind};

impl Network {
    /// Evaluate a position given as a FEN string.
    ///
    /// Convenience wrapper over [`FenBoard`] + [`Network::evaluate`].
    pub fn evaluate_fen(&self, fen: &str) -> Result<i32> {
        Ok(self.evaluate(&FenBoard::parse(fen)?))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_startpos() {
        let board = FenBoard::parse(
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        )
        .unwrap();
        assert_eq!(board.side_to_move(), Color::White);
        assert_eq!(board.king_square(Color::White), 4);
        assert_eq!(board.king_square(Color::Black), 60);
    }

    #[test]
    fn collects_active_features() {
        let board =
            FenBoard::parse("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1").unwrap();
        let arch = feature::Arch::HalfKAv2Hm;
        let mut white = Vec::new();
        let mut black = Vec::new();
        feature::active_indices(arch, &board, Color::White, &mut white);
        feature::active_indices(arch, &board, Color::Black, &mut black);
        assert_eq!(white.len(), 32);
        assert_eq!(black.len(), 32);
        for &idx in white.iter().chain(black.iter()) {
            assert!(idx < arch.input_dimensions());
        }
    }
}
