/// Side of a piece or the side to move.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Color {
    White,
    Black,
}

impl Color {
    #[inline]
    pub(crate) fn index(self) -> usize {
        self as usize
    }
}

/// Type of a chess piece, color-independent.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum PieceKind {
    Pawn,
    Knight,
    Bishop,
    Rook,
    Queen,
    King,
}

/// A colored chess piece.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Piece {
    pub color: Color,
    pub kind: PieceKind,
}

impl Piece {
    /// Construct a piece from a color and kind.
    #[inline]
    pub fn new(color: Color, kind: PieceKind) -> Self {
        Self { color, kind }
    }

    #[inline]
    pub(crate) fn sf_index(self) -> usize {
        let base = match self.kind {
            PieceKind::Pawn => 1,
            PieceKind::Knight => 2,
            PieceKind::Bishop => 3,
            PieceKind::Rook => 4,
            PieceKind::Queen => 5,
            PieceKind::King => 6,
        };
        match self.color {
            Color::White => base,
            Color::Black => base + 8,
        }
    }
}

#[inline]
pub(crate) fn file_of(square: u8) -> u8 {
    square & 7
}

/// A chess position, as seen by the network.
///
/// Implement this for your engine's board to evaluate it without conversions.
/// Squares are `0..64` with `a1 = 0`, `b1 = 1`, ..., `h8 = 63`.
pub trait Board {
    /// The side to move in this position.
    fn side_to_move(&self) -> Color;

    /// The square (`0..64`) of `color`'s king.
    fn king_square(&self, color: Color) -> u8;

    /// Invoke `f` once for every piece on the board, with its square and piece.
    fn for_each_piece(&self, f: &mut dyn FnMut(u8, Piece));
}
