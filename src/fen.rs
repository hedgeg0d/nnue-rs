use crate::error::Error;
use crate::types::{Board, Color, Piece, PieceKind};

/// A minimal board built from a FEN string, implementing [`Board`].
///
/// Use it for one-off evaluations via [`crate::Network::evaluate_fen`], or as a
/// reference implementation of the [`Board`] trait.
pub struct FenBoard {
    squares: [Option<Piece>; 64],
    white_king: u8,
    black_king: u8,
    side_to_move: Color,
}

impl FenBoard {
    /// Parse the placement and side-to-move fields of a FEN string.
    pub fn parse(fen: &str) -> Result<Self, Error> {
        let mut squares = [None; 64];
        let mut parts = fen.split_whitespace();
        let placement = parts.next().ok_or(Error::Fen("empty fen"))?;
        let mut white_king = 64u8;
        let mut black_king = 64u8;

        let mut rank: i32 = 7;
        let mut file: i32 = 0;
        for c in placement.chars() {
            match c {
                '/' => {
                    rank -= 1;
                    file = 0;
                }
                '1'..='8' => file += c as i32 - '0' as i32,
                _ => {
                    let piece = parse_piece(c).ok_or(Error::Fen("bad piece char"))?;
                    if rank < 0 || file > 7 {
                        return Err(Error::Fen("fen out of range"));
                    }
                    let sq = (rank * 8 + file) as u8;
                    if piece.kind == PieceKind::King {
                        match piece.color {
                            Color::White => white_king = sq,
                            Color::Black => black_king = sq,
                        }
                    }
                    squares[sq as usize] = Some(piece);
                    file += 1;
                }
            }
        }

        let side_to_move = match parts.next() {
            Some("b") => Color::Black,
            _ => Color::White,
        };

        if white_king == 64 || black_king == 64 {
            return Err(Error::Fen("missing king"));
        }

        Ok(Self {
            squares,
            white_king,
            black_king,
            side_to_move,
        })
    }
}

fn parse_piece(c: char) -> Option<Piece> {
    let color = if c.is_ascii_uppercase() {
        Color::White
    } else {
        Color::Black
    };
    let kind = match c.to_ascii_lowercase() {
        'p' => PieceKind::Pawn,
        'n' => PieceKind::Knight,
        'b' => PieceKind::Bishop,
        'r' => PieceKind::Rook,
        'q' => PieceKind::Queen,
        'k' => PieceKind::King,
        _ => return None,
    };
    Some(Piece::new(color, kind))
}

impl Board for FenBoard {
    fn side_to_move(&self) -> Color {
        self.side_to_move
    }

    fn king_square(&self, color: Color) -> u8 {
        match color {
            Color::White => self.white_king,
            Color::Black => self.black_king,
        }
    }

    fn for_each_piece(&self, f: &mut dyn FnMut(u8, Piece)) {
        for sq in 0..64u8 {
            if let Some(piece) = self.squares[sq as usize] {
                f(sq, piece);
            }
        }
    }
}
