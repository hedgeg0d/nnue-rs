use nnue_rs::{Board, Color, Network, Piece, PieceKind};

struct Position {
    squares: [Option<Piece>; 64],
    side_to_move: Color,
}

impl Position {
    fn startpos() -> Self {
        let mut squares = [None; 64];
        let back = [
            PieceKind::Rook,
            PieceKind::Knight,
            PieceKind::Bishop,
            PieceKind::Queen,
            PieceKind::King,
            PieceKind::Bishop,
            PieceKind::Knight,
            PieceKind::Rook,
        ];
        for (file, &kind) in back.iter().enumerate() {
            squares[file] = Some(Piece::new(Color::White, kind));
            squares[8 + file] = Some(Piece::new(Color::White, PieceKind::Pawn));
            squares[48 + file] = Some(Piece::new(Color::Black, PieceKind::Pawn));
            squares[56 + file] = Some(Piece::new(Color::Black, kind));
        }
        Self {
            squares,
            side_to_move: Color::White,
        }
    }
}

impl Board for Position {
    fn side_to_move(&self) -> Color {
        self.side_to_move
    }

    fn king_square(&self, color: Color) -> u8 {
        for sq in 0..64u8 {
            if let Some(piece) = self.squares[sq as usize] {
                if piece.color == color && piece.kind == PieceKind::King {
                    return sq;
                }
            }
        }
        64
    }

    fn for_each_piece(&self, f: &mut dyn FnMut(u8, Piece)) {
        for sq in 0..64u8 {
            if let Some(piece) = self.squares[sq as usize] {
                f(sq, piece);
            }
        }
    }
}

fn main() {
    let path = std::env::args()
        .nth(1)
        .expect("usage: custom_board <net.nnue>");
    let net = Network::from_file(&path).expect("load network");

    let pos = Position::startpos();
    println!("startpos score: {}", net.evaluate(&pos));
}
