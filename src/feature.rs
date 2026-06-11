use crate::types::{file_of, Board, Color};

/// The feature set (input layer) an NNUE network is built on.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Arch {
    /// `SFNNv10`, `HalfKAv2_hm` piece-square features combined with
    /// `FullThreats` threat features (Stockfish 18 big nets).
    Sfnnv10,
    /// `HalfKAv2_hm`, the horizontally-mirrored feature set used by Stockfish
    /// SFNNv5-v9 (SF 16/17 and SF 18 small nets).
    HalfKAv2Hm,
    /// `HalfKAv2`, the non-mirrored predecessor used by Stockfish SFNNv2-v4
    /// (SF 14).
    HalfKAv2,
    /// `HalfKP`, the classic Stockfish NNUE feature set (SF 12-14).
    HalfKP,
}

impl Arch {
    /// Number of input features per perspective.
    pub fn input_dimensions(self) -> usize {
        match self {
            Arch::Sfnnv10 | Arch::HalfKAv2Hm => 22528,
            Arch::HalfKAv2 => 45056,
            Arch::HalfKP => 41024,
        }
    }

    /// Whether kings are encoded as features (true for the `HalfKA` family,
    /// false for `HalfKP`, which only uses kings to bucket the other pieces).
    pub fn kings_are_features(self) -> bool {
        matches!(self, Arch::Sfnnv10 | Arch::HalfKAv2Hm | Arch::HalfKAv2)
    }
}

const PS_NB: usize = 704;

const WHITE_KING_BUCKET: [usize; 64] = [
    28, 29, 30, 31, 31, 30, 29, 28,
    24, 25, 26, 27, 27, 26, 25, 24,
    20, 21, 22, 23, 23, 22, 21, 20,
    16, 17, 18, 19, 19, 18, 17, 16,
    12, 13, 14, 15, 15, 14, 13, 12,
    8, 9, 10, 11, 11, 10, 9, 8,
    4, 5, 6, 7, 7, 6, 5, 4,
    0, 1, 2, 3, 3, 2, 1, 0,
];

const HALFKA_PIECE_SQUARE_INDEX: [[usize; 16]; 2] = [
    [0, 0, 128, 256, 384, 512, 640, 0, 0, 64, 192, 320, 448, 576, 640, 0],
    [0, 64, 192, 320, 448, 576, 640, 0, 0, 0, 128, 256, 384, 512, 640, 0],
];

const PS_END: usize = 641;

const HALFKP_PIECE_SQUARE_INDEX: [[usize; 16]; 2] = [
    [0, 1, 129, 257, 385, 513, 641, 0, 0, 65, 193, 321, 449, 577, 641, 0],
    [0, 65, 193, 321, 449, 577, 641, 0, 0, 1, 129, 257, 385, 513, 641, 0],
];

#[inline]
fn halfka_orient(perspective: Color, king_square: u8) -> u8 {
    let base = if file_of(king_square) <= 3 { 7 } else { 0 };
    match perspective {
        Color::White => base,
        Color::Black => base ^ 56,
    }
}

#[inline]
fn halfka_king_bucket(perspective: Color, king_square: u8) -> usize {
    let sq = match perspective {
        Color::White => king_square as usize,
        Color::Black => (king_square ^ 56) as usize,
    };
    WHITE_KING_BUCKET[sq] * PS_NB
}

#[inline]
fn halfka_index(perspective: Color, square: u8, piece_sf_index: usize, king_square: u8) -> usize {
    let o = halfka_orient(perspective, king_square);
    (square ^ o) as usize
        + HALFKA_PIECE_SQUARE_INDEX[perspective.index()][piece_sf_index]
        + halfka_king_bucket(perspective, king_square)
}

#[inline]
fn halfka_v2_index(perspective: Color, square: u8, piece_sf_index: usize, king_square: u8) -> usize {
    let o = match perspective {
        Color::White => 0u8,
        Color::Black => 56,
    };
    let ksq = (king_square ^ o) as usize;
    (square ^ o) as usize
        + HALFKA_PIECE_SQUARE_INDEX[perspective.index()][piece_sf_index]
        + PS_NB * ksq
}

#[inline]
fn halfkp_orient(perspective: Color, square: u8) -> u8 {
    match perspective {
        Color::White => square,
        Color::Black => square ^ 63,
    }
}

#[inline]
fn halfkp_index(perspective: Color, square: u8, piece_sf_index: usize, king_square: u8) -> usize {
    let ksq = halfkp_orient(perspective, king_square) as usize;
    halfkp_orient(perspective, square) as usize
        + HALFKP_PIECE_SQUARE_INDEX[perspective.index()][piece_sf_index]
        + PS_END * ksq
}

#[inline]
pub fn make_index(
    arch: Arch,
    perspective: Color,
    square: u8,
    piece_sf_index: usize,
    king_square: u8,
) -> usize {
    match arch {
        Arch::Sfnnv10 | Arch::HalfKAv2Hm => {
            halfka_index(perspective, square, piece_sf_index, king_square)
        }
        Arch::HalfKAv2 => halfka_v2_index(perspective, square, piece_sf_index, king_square),
        Arch::HalfKP => halfkp_index(perspective, square, piece_sf_index, king_square),
    }
}

pub fn active_indices(arch: Arch, board: &impl Board, perspective: Color, out: &mut Vec<usize>) {
    out.clear();
    let ksq = board.king_square(perspective);
    let kings_are_features = arch.kings_are_features();
    board.for_each_piece(&mut |square, piece| {
        if kings_are_features || piece.kind != crate::types::PieceKind::King {
            out.push(make_index(arch, perspective, square, piece.sf_index(), ksq));
        }
    });
}
