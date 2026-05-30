use crate::types::{file_of, Board, Color};

pub const INPUT_DIMENSIONS: usize = 22528;

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

const PIECE_SQUARE_INDEX: [[usize; 16]; 2] = [
    [0, 0, 128, 256, 384, 512, 640, 0, 0, 64, 192, 320, 448, 576, 640, 0],
    [0, 64, 192, 320, 448, 576, 640, 0, 0, 0, 128, 256, 384, 512, 640, 0],
];

#[inline]
fn orient(perspective: Color, king_square: u8) -> u8 {
    let base = if file_of(king_square) <= 3 { 7 } else { 0 };
    match perspective {
        Color::White => base,
        Color::Black => base ^ 56,
    }
}

#[inline]
fn king_bucket(perspective: Color, king_square: u8) -> usize {
    let sq = match perspective {
        Color::White => king_square as usize,
        Color::Black => (king_square ^ 56) as usize,
    };
    WHITE_KING_BUCKET[sq] * PS_NB
}

#[inline]
pub fn make_index(perspective: Color, square: u8, piece_sf_index: usize, king_square: u8) -> usize {
    let o = orient(perspective, king_square);
    (square ^ o) as usize + PIECE_SQUARE_INDEX[perspective.index()][piece_sf_index]
        + king_bucket(perspective, king_square)
}

pub fn active_indices(board: &impl Board, perspective: Color, out: &mut Vec<usize>) {
    out.clear();
    let ksq = board.king_square(perspective);
    board.for_each_piece(&mut |square, piece| {
        out.push(make_index(perspective, square, piece.sf_index(), ksq));
    });
}
