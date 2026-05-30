#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Color {
    White,
    Black,
}

impl Color {
    #[inline]
    pub fn flip(self) -> Color {
        match self {
            Color::White => Color::Black,
            Color::Black => Color::White,
        }
    }

    #[inline]
    pub fn index(self) -> usize {
        self as usize
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum PieceKind {
    Pawn,
    Knight,
    Bishop,
    Rook,
    Queen,
    King,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Piece {
    pub color: Color,
    pub kind: PieceKind,
}

impl Piece {
    #[inline]
    pub fn new(color: Color, kind: PieceKind) -> Self {
        Self { color, kind }
    }

    #[inline]
    pub fn sf_index(self) -> usize {
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
pub fn file_of(square: u8) -> u8 {
    square & 7
}

pub trait Board {
    fn side_to_move(&self) -> Color;
    fn king_square(&self, color: Color) -> u8;
    fn for_each_piece(&self, f: &mut dyn FnMut(u8, Piece));
}
