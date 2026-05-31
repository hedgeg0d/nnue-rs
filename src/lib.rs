mod feature;
mod fen;
mod leb128;
mod network;
mod types;

pub use fen::FenBoard;
pub use network::{Accumulator, Network};
pub use types::{Board, Color, Piece, PieceKind};

impl Network {
    pub fn evaluate_fen(&self, fen: &str) -> Result<i32, String> {
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
        let mut white = Vec::new();
        let mut black = Vec::new();
        feature::active_indices(&board, Color::White, &mut white);
        feature::active_indices(&board, Color::Black, &mut black);
        assert_eq!(white.len(), 32);
        assert_eq!(black.len(), 32);
        for &idx in white.iter().chain(black.iter()) {
            assert!(idx < feature::INPUT_DIMENSIONS);
        }
    }
}
