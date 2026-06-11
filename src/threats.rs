use crate::types::{Board, Color};

pub(crate) const THREAT_DIMENSIONS: usize = 79856;
pub(crate) const EXCLUDED: u32 = THREAT_DIMENSIONS as u32;

const NUM_VALID_TARGETS: [u32; 16] = [0, 6, 12, 10, 10, 12, 8, 0, 0, 6, 12, 10, 10, 12, 8, 0];

const MAP: [[i8; 6]; 6] = [
    [0, 1, -1, 2, -1, -1],
    [0, 1, 2, 3, 4, 5],
    [0, 1, 2, 3, -1, 4],
    [0, 1, 2, 3, -1, 4],
    [0, 1, 2, 3, 4, 5],
    [0, 1, 2, 3, -1, -1],
];

const ALL_PIECES: [usize; 12] = [1, 2, 3, 4, 5, 6, 9, 10, 11, 12, 13, 14];

const fn knight_attacks_table() -> [u64; 64] {
    let deltas: [(i8, i8); 8] = [
        (-2, -1), (-2, 1), (-1, -2), (-1, 2),
        (1, -2), (1, 2), (2, -1), (2, 1),
    ];
    let mut table = [0u64; 64];
    let mut sq = 0;
    while sq < 64 {
        let r = (sq / 8) as i8;
        let f = (sq % 8) as i8;
        let mut i = 0;
        while i < 8 {
            let nr = r + deltas[i].0;
            let nf = f + deltas[i].1;
            if nr >= 0 && nr < 8 && nf >= 0 && nf < 8 {
                table[sq] |= 1u64 << (nr * 8 + nf);
            }
            i += 1;
        }
        sq += 1;
    }
    table
}

const fn king_attacks_table() -> [u64; 64] {
    let mut table = [0u64; 64];
    let mut sq = 0;
    while sq < 64 {
        let r = (sq / 8) as i8;
        let f = (sq % 8) as i8;
        let mut dr = -1i8;
        while dr <= 1 {
            let mut df = -1i8;
            while df <= 1 {
                if dr != 0 || df != 0 {
                    let nr = r + dr;
                    let nf = f + df;
                    if nr >= 0 && nr < 8 && nf >= 0 && nf < 8 {
                        table[sq] |= 1u64 << (nr * 8 + nf);
                    }
                }
                df += 1;
            }
            dr += 1;
        }
        sq += 1;
    }
    table
}

const fn pawn_attacks_table() -> [[u64; 64]; 2] {
    let mut table = [[0u64; 64]; 2];
    let mut sq = 0i32;
    while sq < 64 {
        let f = sq % 8;
        if sq + 7 < 64 && f > 0 {
            table[0][sq as usize] |= 1u64 << (sq + 7);
        }
        if sq + 9 < 64 && f < 7 {
            table[0][sq as usize] |= 1u64 << (sq + 9);
        }
        if sq - 7 >= 0 && f < 7 {
            table[1][sq as usize] |= 1u64 << (sq - 7);
        }
        if sq - 9 >= 0 && f > 0 {
            table[1][sq as usize] |= 1u64 << (sq - 9);
        }
        sq += 1;
    }
    table
}

const fn ray_table(diag: bool, orth: bool) -> [u64; 64] {
    let mut table = [0u64; 64];
    let dirs: [(i8, i8); 8] = [
        (1, 1), (1, -1), (-1, 1), (-1, -1),
        (1, 0), (-1, 0), (0, 1), (0, -1),
    ];
    let mut sq = 0;
    while sq < 64 {
        let mut d = 0;
        while d < 8 {
            let use_dir = if d < 4 { diag } else { orth };
            if use_dir {
                let mut r = (sq / 8) as i8 + dirs[d].0;
                let mut f = (sq % 8) as i8 + dirs[d].1;
                while r >= 0 && r < 8 && f >= 0 && f < 8 {
                    table[sq] |= 1u64 << (r * 8 + f);
                    r += dirs[d].0;
                    f += dirs[d].1;
                }
            }
            d += 1;
        }
        sq += 1;
    }
    table
}

const KNIGHT_ATK: [u64; 64] = knight_attacks_table();
const KING_ATK: [u64; 64] = king_attacks_table();
const PAWN_ATK: [[u64; 64]; 2] = pawn_attacks_table();
const BISHOP_PSEUDO: [u64; 64] = ray_table(true, false);
const ROOK_PSEUDO: [u64; 64] = ray_table(false, true);
const QUEEN_PSEUDO: [u64; 64] = ray_table(true, true);

const fn pseudo_attacks(code: usize, sq: usize) -> u64 {
    match code & 7 {
        1 => PAWN_ATK[(code >> 3) & 1][sq],
        2 => KNIGHT_ATK[sq],
        3 => BISHOP_PSEUDO[sq],
        4 => ROOK_PSEUDO[sq],
        5 => QUEEN_PSEUDO[sq],
        _ => KING_ATK[sq],
    }
}

struct OffsetTables {
    offsets: [[u32; 64]; 16],
    cum_piece: [u32; 16],
    cum_total: [u32; 16],
}

const fn build_offsets() -> OffsetTables {
    let mut offsets = [[0u32; 64]; 16];
    let mut cum_piece = [0u32; 16];
    let mut cum_total = [0u32; 16];
    let mut cumulative = 0u32;
    let mut p = 0;
    while p < 12 {
        let code = ALL_PIECES[p];
        let mut piece_cum = 0u32;
        let mut from = 0usize;
        while from < 64 {
            offsets[code][from] = piece_cum;
            if code & 7 != 1 {
                piece_cum += pseudo_attacks(code, from).count_ones();
            } else if from >= 8 && from < 56 {
                piece_cum += pseudo_attacks(code, from).count_ones();
            }
            from += 1;
        }
        cum_piece[code] = piece_cum;
        cum_total[code] = cumulative;
        cumulative += NUM_VALID_TARGETS[code] * piece_cum;
        p += 1;
    }
    OffsetTables { offsets, cum_piece, cum_total }
}

const OFFSET_TABLES: OffsetTables = build_offsets();

const fn build_lut1() -> [[[u32; 2]; 16]; 16] {
    let mut lut = [[[EXCLUDED; 2]; 16]; 16];
    let mut a = 0;
    while a < 12 {
        let attacker = ALL_PIECES[a];
        let mut d = 0;
        while d < 12 {
            let attacked = ALL_PIECES[d];
            let enemy = (attacker ^ attacked) == 8;
            let at = attacker & 7;
            let dt = attacked & 7;
            let m = MAP[at - 1][dt - 1];
            let semi_excluded = at == dt && (enemy || at != 1);
            if m >= 0 {
                let attacked_color = (attacked >> 3) as u32;
                let feature = OFFSET_TABLES.cum_total[attacker]
                    + (attacked_color * (NUM_VALID_TARGETS[attacker] / 2) + m as u32)
                        * OFFSET_TABLES.cum_piece[attacker];
                lut[attacker][attacked][0] = feature;
                if !semi_excluded {
                    lut[attacker][attacked][1] = feature;
                }
            }
            d += 1;
        }
        a += 1;
    }
    lut
}

const LUT1: [[[u32; 2]; 16]; 16] = build_lut1();

const fn build_lut2() -> [[[u8; 64]; 64]; 16] {
    let mut lut = [[[0u8; 64]; 64]; 16];
    let mut p = 0;
    while p < 12 {
        let code = ALL_PIECES[p];
        let mut from = 0usize;
        while from < 64 {
            let attacks = pseudo_attacks(code, from);
            let mut to = 0usize;
            while to < 64 {
                lut[code][from][to] = (((1u64 << to) - 1) & attacks).count_ones() as u8;
                to += 1;
            }
            from += 1;
        }
        p += 1;
    }
    lut
}

static LUT2: [[[u8; 64]; 64]; 16] = build_lut2();

#[inline]
pub(crate) fn threat_index(
    perspective: Color,
    attacker: u8,
    from: u8,
    to: u8,
    attacked: u8,
    king_square: u8,
) -> u32 {
    let p = perspective.index() as u8;
    let orientation = (if king_square & 7 >= 4 { 7u8 } else { 0 }) ^ (56 * p);
    let from_o = (from ^ orientation) as usize;
    let to_o = (to ^ orientation) as usize;
    let swap = 8 * p;
    let atk = (attacker ^ swap) as usize;
    let atkd = (attacked ^ swap) as usize;
    let base = LUT1[atk][atkd][(from_o < to_o) as usize];
    if base >= EXCLUDED {
        return EXCLUDED;
    }
    base + OFFSET_TABLES.offsets[atk][from_o] + LUT2[atk][from_o][to_o] as u32
}

pub(crate) struct PosInfo {
    pub piece_on: [u8; 64],
    pub occ: u64,
    pub kings: [u8; 2],
}

impl PosInfo {
    pub fn from_board(board: &impl Board) -> Self {
        let mut piece_on = [0u8; 64];
        let mut occ = 0u64;
        board.for_each_piece(&mut |sq, piece| {
            piece_on[sq as usize] = piece.sf_index() as u8;
            occ |= 1u64 << sq;
        });
        Self {
            piece_on,
            occ,
            kings: [
                board.king_square(Color::White),
                board.king_square(Color::Black),
            ],
        }
    }
}

fn slider_attacks(from: u8, occ: u64, diag: bool, orth: bool) -> u64 {
    let dirs: [(i8, i8); 8] = [
        (1, 1), (1, -1), (-1, 1), (-1, -1),
        (1, 0), (-1, 0), (0, 1), (0, -1),
    ];
    let (lo, hi) = if diag && orth {
        (0, 8)
    } else if diag {
        (0, 4)
    } else {
        (4, 8)
    };
    let mut attacks = 0u64;
    for &(dr, df) in &dirs[lo..hi] {
        let mut r = (from / 8) as i8 + dr;
        let mut f = (from % 8) as i8 + df;
        while r >= 0 && r < 8 && f >= 0 && f < 8 {
            let bit = 1u64 << (r * 8 + f);
            attacks |= bit;
            if occ & bit != 0 {
                break;
            }
            r += dr;
            f += df;
        }
    }
    attacks
}

fn attacks_of(code: u8, from: u8, occ: u64) -> u64 {
    match code & 7 {
        1 => PAWN_ATK[((code >> 3) & 1) as usize][from as usize],
        2 => KNIGHT_ATK[from as usize],
        3 => slider_attacks(from, occ, true, false),
        4 => slider_attacks(from, occ, false, true),
        5 => slider_attacks(from, occ, true, true),
        _ => KING_ATK[from as usize],
    }
}

#[inline]
fn pack_pair(from: u8, to: u8, attacker: u8, attacked: u8) -> u32 {
    ((from as u32) << 14) | ((to as u32) << 8) | ((attacker as u32) << 4) | attacked as u32
}

#[inline]
pub(crate) fn unpack_pair(pair: u32) -> (u8, u8, u8, u8) {
    (
        (pair >> 14) as u8,
        ((pair >> 8) & 0x3f) as u8,
        ((pair >> 4) & 0xf) as u8,
        (pair & 0xf) as u8,
    )
}

pub(crate) fn threat_pairs(pos: &PosInfo, out: &mut Vec<u32>) {
    out.clear();
    let mut bb = pos.occ;
    while bb != 0 {
        let from = bb.trailing_zeros() as u8;
        let attacker = pos.piece_on[from as usize];
        let mut attacks = attacks_of(attacker, from, pos.occ) & pos.occ;
        while attacks != 0 {
            let to = attacks.trailing_zeros() as u8;
            out.push(pack_pair(from, to, attacker, pos.piece_on[to as usize]));
            attacks &= attacks - 1;
        }
        bb &= bb - 1;
    }
}
