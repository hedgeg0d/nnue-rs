use std::cell::RefCell;
use std::fs::File;
use std::io::{self, BufReader, Read};

use crate::error::{Error, Result};
use crate::feature::{active_indices, make_index, Arch};
use crate::leb128;
use crate::types::{Board, Color, Piece, PieceKind};

#[derive(Default)]
struct Scratch {
    white: Vec<usize>,
    input: Vec<u8>,
}

thread_local! {
    static SCRATCH: RefCell<Scratch> = RefCell::new(Scratch::default());
}

const VERSION_HALFKA: u32 = 0x7AF32F20;
const VERSION_HALFKP: u32 = 0x7AF32F16;
const FEATURE_HASH: u32 = 0x7f234cb8;
const PSQT_BUCKETS: usize = 8;
const LAYER_STACKS: usize = 8;
const L2: usize = 15;
const L3: usize = 32;
const FC0_OUT: usize = L2 + 1;
const FC1_IN: usize = L2 * 2;
const OUTPUT_SCALE: i32 = 16;
const MAX_CHANGED: usize = 4;

const HALFKP_HALF_DIMENSIONS: usize = 256;
const HALFKP_L1_OUT: usize = 32;
const HALFKP_L2_OUT: usize = 32;
const HALFKP_WEIGHT_SCALE_BITS: i32 = 6;
const HALFKP_FV_SCALE: i32 = 16;

struct Bucket {
    fc0_bias: Vec<i32>,
    fc0_weight: Vec<i8>,
    fc1_bias: Vec<i32>,
    fc1_weight: Vec<i8>,
    fc2_bias: i32,
    fc2_weight: Vec<i8>,
}

struct HalfKPLayers {
    fc0_bias: Vec<i32>,
    fc0_weight: Vec<i8>,
    fc1_bias: Vec<i32>,
    fc1_weight: Vec<i8>,
    fc2_bias: i32,
    fc2_weight: Vec<i8>,
}

enum Layers {
    HalfKAv2Hm {
        ft_psqt: Vec<i32>,
        buckets: Vec<Bucket>,
    },
    HalfKP(HalfKPLayers),
}

/// A loaded NNUE network.
///
/// Load one with [`Network::from_file`], [`Network::from_bytes`] or
/// [`Network::from_reader`], then evaluate positions with [`Network::evaluate`],
/// [`Network::evaluate_fen`], or the incremental accumulator API. The
/// architecture is detected automatically from the file header; see
/// [`Network::arch`].
pub struct Network {
    arch: Arch,
    l1: usize,
    ft_bias: Vec<i16>,
    ft_weight: Vec<i16>,
    layers: Layers,
}

/// A pair of feature-transformer accumulators (one per side-to-move
/// perspective) plus the piece count for the position they describe.
///
/// Obtain one with [`Network::accumulator`] or [`Network::empty_accumulator`],
/// keep it alongside your board, and advance it incrementally with
/// [`Network::update`] as moves are made. Evaluate it with
/// [`Network::evaluate_accumulator`].
#[derive(Clone)]
pub struct Accumulator {
    white: Vec<i16>,
    black: Vec<i16>,
    psqt_white: [i32; PSQT_BUCKETS],
    psqt_black: [i32; PSQT_BUCKETS],
    piece_count: usize,
}

fn read_u32(reader: &mut impl Read) -> io::Result<u32> {
    let mut buf = [0u8; 4];
    reader.read_exact(&mut buf)?;
    Ok(u32::from_le_bytes(buf))
}

fn read_i32_raw(reader: &mut impl Read, count: usize) -> io::Result<Vec<i32>> {
    let mut out = Vec::with_capacity(count);
    let mut buf = [0u8; 4];
    for _ in 0..count {
        reader.read_exact(&mut buf)?;
        out.push(i32::from_le_bytes(buf));
    }
    Ok(out)
}

fn read_i16_raw(reader: &mut impl Read, count: usize) -> io::Result<Vec<i16>> {
    let mut out = Vec::with_capacity(count);
    let mut buf = [0u8; 2];
    for _ in 0..count {
        reader.read_exact(&mut buf)?;
        out.push(i16::from_le_bytes(buf));
    }
    Ok(out)
}

fn read_i8_raw(reader: &mut impl Read, count: usize) -> io::Result<Vec<i8>> {
    let mut bytes = vec![0u8; count];
    reader.read_exact(&mut bytes)?;
    Ok(bytes.into_iter().map(|b| b as i8).collect())
}

fn count_pieces(board: &impl Board) -> usize {
    let mut n = 0;
    board.for_each_piece(&mut |_, _| n += 1);
    n
}

fn diff_boards(
    parent: &impl Board,
    child: &impl Board,
    removed: &mut [(u8, Piece)],
    added: &mut [(u8, Piece)],
) -> (usize, usize) {
    let mut p = [None; 64];
    let mut c = [None; 64];
    parent.for_each_piece(&mut |sq, piece| p[sq as usize] = Some(piece));
    child.for_each_piece(&mut |sq, piece| c[sq as usize] = Some(piece));
    let mut nr = 0;
    let mut na = 0;
    for sq in 0..64u8 {
        let (a, b) = (p[sq as usize], c[sq as usize]);
        if a != b {
            if let Some(piece) = a {
                if nr < removed.len() {
                    removed[nr] = (sq, piece);
                }
                nr += 1;
            }
            if let Some(piece) = b {
                if na < added.len() {
                    added[na] = (sq, piece);
                }
                na += 1;
            }
        }
    }
    (nr.min(removed.len()), na.min(added.len()))
}

impl Network {
    /// Load a network from a `.nnue` file on disk.
    pub fn from_file(path: &str) -> Result<Self> {
        Self::from_reader(&mut BufReader::new(File::open(path)?))
    }

    /// Load a network from an in-memory byte slice (e.g. an embedded net).
    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        Self::from_reader(&mut &bytes[..])
    }

    /// Load a network from any reader.
    ///
    /// The architecture is selected from the file's version header.
    pub fn from_reader(reader: &mut impl Read) -> Result<Self> {
        let version = read_u32(reader)?;
        let _arch_hash = read_u32(reader)?;
        let desc_len = read_u32(reader)? as usize;
        let mut desc = vec![0u8; desc_len];
        reader.read_exact(&mut desc)?;

        match version {
            VERSION_HALFKA => Self::load_halfka(reader),
            VERSION_HALFKP => Self::load_halfkp(reader),
            other => Err(Error::UnsupportedVersion(other)),
        }
    }

    /// The feature-set architecture this network was loaded as.
    pub fn arch(&self) -> Arch {
        self.arch
    }

    fn load_halfka(reader: &mut impl Read) -> Result<Self> {
        let arch = Arch::HalfKAv2Hm;
        let input_dims = arch.input_dimensions();

        let ft_hash = read_u32(reader)?;
        let l1 = ((ft_hash ^ FEATURE_HASH) / 2) as usize;
        if l1 == 0 || l1 % 2 != 0 {
            return Err(Error::InvalidData("bad feature-transformer width"));
        }

        let mut ft_bias = leb128::read_i16(reader, l1)?;
        let mut ft_weight = leb128::read_i16(reader, l1 * input_dims)?;
        let ft_psqt = leb128::read_i32(reader, PSQT_BUCKETS * input_dims)?;

        for b in ft_bias.iter_mut() {
            *b = b.wrapping_mul(2);
        }
        for w in ft_weight.iter_mut() {
            *w = w.wrapping_mul(2);
        }

        let mut buckets = Vec::with_capacity(LAYER_STACKS);
        for _ in 0..LAYER_STACKS {
            let _bucket_hash = read_u32(reader)?;
            let fc0_bias = read_i32_raw(reader, FC0_OUT)?;
            let fc0_weight = read_i8_raw(reader, FC0_OUT * l1)?;
            let fc1_bias = read_i32_raw(reader, L3)?;
            let fc1_weight = read_i8_raw(reader, L3 * 32)?;
            let fc2_bias = read_i32_raw(reader, 1)?[0];
            let fc2_weight = read_i8_raw(reader, 32)?;
            buckets.push(Bucket {
                fc0_bias,
                fc0_weight,
                fc1_bias,
                fc1_weight,
                fc2_bias,
                fc2_weight,
            });
        }

        Ok(Self {
            arch,
            l1,
            ft_bias,
            ft_weight,
            layers: Layers::HalfKAv2Hm { ft_psqt, buckets },
        })
    }

    fn load_halfkp(reader: &mut impl Read) -> Result<Self> {
        let arch = Arch::HalfKP;
        let l1 = HALFKP_HALF_DIMENSIONS;
        let input_dims = arch.input_dimensions();

        let _ft_hash = read_u32(reader)?;
        let ft_bias = read_i16_raw(reader, l1)?;
        let ft_weight = read_i16_raw(reader, l1 * input_dims)?;

        let _net_hash = read_u32(reader)?;
        let input = 2 * l1;
        let fc0_bias = read_i32_raw(reader, HALFKP_L1_OUT)?;
        let fc0_weight = read_i8_raw(reader, HALFKP_L1_OUT * input)?;
        let fc1_bias = read_i32_raw(reader, HALFKP_L2_OUT)?;
        let fc1_weight = read_i8_raw(reader, HALFKP_L2_OUT * HALFKP_L1_OUT)?;
        let fc2_bias = read_i32_raw(reader, 1)?[0];
        let fc2_weight = read_i8_raw(reader, HALFKP_L2_OUT)?;

        Ok(Self {
            arch,
            l1,
            ft_bias,
            ft_weight,
            layers: Layers::HalfKP(HalfKPLayers {
                fc0_bias,
                fc0_weight,
                fc1_bias,
                fc1_weight,
                fc2_bias,
                fc2_weight,
            }),
        })
    }

    fn ft_psqt(&self) -> &[i32] {
        match &self.layers {
            Layers::HalfKAv2Hm { ft_psqt, .. } => ft_psqt,
            Layers::HalfKP(_) => &[],
        }
    }

    fn accumulate(&self, indices: &[usize], acc: &mut [i16], psqt: &mut [i32; PSQT_BUCKETS]) {
        acc.copy_from_slice(&self.ft_bias);
        *psqt = [0i32; PSQT_BUCKETS];
        let ft_psqt = self.ft_psqt();
        let has_psqt = !ft_psqt.is_empty();
        for &feat in indices {
            let base = feat * self.l1;
            crate::simd::add_i16(acc, &self.ft_weight[base..base + self.l1]);
            if has_psqt {
                let pbase = feat * PSQT_BUCKETS;
                for b in 0..PSQT_BUCKETS {
                    psqt[b] += ft_psqt[pbase + b];
                }
            }
        }
    }

    /// Allocate a zeroed accumulator sized for this network.
    ///
    /// Useful for pre-allocating a stack of accumulators in a search; fill each
    /// with [`Network::refresh`] or [`Network::update`] before evaluating.
    pub fn empty_accumulator(&self) -> Accumulator {
        Accumulator {
            white: vec![0i16; self.l1],
            black: vec![0i16; self.l1],
            psqt_white: [0i32; PSQT_BUCKETS],
            psqt_black: [0i32; PSQT_BUCKETS],
            piece_count: 0,
        }
    }

    /// Allocate and fully compute an accumulator for `board`.
    pub fn accumulator(&self, board: &impl Board) -> Accumulator {
        let mut acc = self.empty_accumulator();
        self.refresh(board, &mut acc);
        acc
    }

    fn refresh_side(
        &self,
        board: &impl Board,
        color: Color,
        acc: &mut [i16],
        psqt: &mut [i32; PSQT_BUCKETS],
    ) {
        SCRATCH.with(|cell| {
            let mut s = cell.borrow_mut();
            active_indices(self.arch, board, color, &mut s.white);
            self.accumulate(&s.white, acc, psqt);
        });
    }

    /// Recompute `acc` from scratch for `board`.
    ///
    /// Call at the root of a search; [`Network::update`] calls it internally
    /// when a king moves.
    pub fn refresh(&self, board: &impl Board, acc: &mut Accumulator) {
        self.refresh_side(board, Color::White, &mut acc.white, &mut acc.psqt_white);
        self.refresh_side(board, Color::Black, &mut acc.black, &mut acc.psqt_black);
        acc.piece_count = count_pieces(board);
    }

    fn apply_side(
        &self,
        parent: &[i16],
        parent_psqt: &[i32; PSQT_BUCKETS],
        child: &mut [i16],
        child_psqt: &mut [i32; PSQT_BUCKETS],
        king_square: u8,
        color: Color,
        removed: &[(u8, Piece)],
        added: &[(u8, Piece)],
    ) {
        child.copy_from_slice(parent);
        *child_psqt = *parent_psqt;
        let ft_psqt = self.ft_psqt();
        let has_psqt = !ft_psqt.is_empty();
        let kings_are_features = self.arch.kings_are_features();
        for &(sq, piece) in removed {
            if !kings_are_features && piece.kind == PieceKind::King {
                continue;
            }
            let feat = make_index(self.arch, color, sq, piece.sf_index(), king_square);
            let base = feat * self.l1;
            crate::simd::sub_i16(child, &self.ft_weight[base..base + self.l1]);
            if has_psqt {
                let pbase = feat * PSQT_BUCKETS;
                for b in 0..PSQT_BUCKETS {
                    child_psqt[b] -= ft_psqt[pbase + b];
                }
            }
        }
        for &(sq, piece) in added {
            if !kings_are_features && piece.kind == PieceKind::King {
                continue;
            }
            let feat = make_index(self.arch, color, sq, piece.sf_index(), king_square);
            let base = feat * self.l1;
            crate::simd::add_i16(child, &self.ft_weight[base..base + self.l1]);
            if has_psqt {
                let pbase = feat * PSQT_BUCKETS;
                for b in 0..PSQT_BUCKETS {
                    child_psqt[b] += ft_psqt[pbase + b];
                }
            }
        }
    }

    /// Advance `parent` into `child` for the move that turns `parent_board` into
    /// `child_board`, writing the result into `child`.
    ///
    /// The changed pieces are derived by diffing the two boards, so every move
    /// type (captures, en passant, promotions, castling) is handled. When a king
    /// moves, that perspective is recomputed from scratch automatically.
    pub fn update(
        &self,
        parent_board: &impl Board,
        child_board: &impl Board,
        parent: &Accumulator,
        child: &mut Accumulator,
    ) {
        let dummy = (0u8, Piece::new(Color::White, PieceKind::Pawn));
        let mut removed = [dummy; MAX_CHANGED];
        let mut added = [dummy; MAX_CHANGED];
        let (nr, na) = diff_boards(parent_board, child_board, &mut removed, &mut added);
        let removed = &removed[..nr];
        let added = &added[..na];

        let white_king_moved =
            parent_board.king_square(Color::White) != child_board.king_square(Color::White);
        let black_king_moved =
            parent_board.king_square(Color::Black) != child_board.king_square(Color::Black);

        if white_king_moved {
            self.refresh_side(child_board, Color::White, &mut child.white, &mut child.psqt_white);
        } else {
            let wk = child_board.king_square(Color::White);
            self.apply_side(
                &parent.white, &parent.psqt_white,
                &mut child.white, &mut child.psqt_white,
                wk, Color::White, removed, added,
            );
        }
        if black_king_moved {
            self.refresh_side(child_board, Color::Black, &mut child.black, &mut child.psqt_black);
        } else {
            let bk = child_board.king_square(Color::Black);
            self.apply_side(
                &parent.black, &parent.psqt_black,
                &mut child.black, &mut child.psqt_black,
                bk, Color::Black, removed, added,
            );
        }

        child.piece_count = parent.piece_count + na - nr;
    }

    /// Evaluate a ready accumulator for the given side to move.
    ///
    /// Returns an internal score in roughly centipawn-scaled units from `stm`'s
    /// perspective. `stm` is passed separately so the same accumulator can be
    /// reused across a null move (which flips the side to move without changing
    /// any features).
    pub fn evaluate_accumulator(&self, acc: &Accumulator, stm: Color) -> i32 {
        match &self.layers {
            Layers::HalfKAv2Hm { .. } => self.evaluate_halfka(acc, stm),
            Layers::HalfKP(layers) => self.evaluate_halfkp(acc, stm, layers),
        }
    }

    fn evaluate_halfka(&self, acc: &Accumulator, stm: Color) -> i32 {
        SCRATCH.with(|cell| {
            let mut s = cell.borrow_mut();
            if s.input.len() != self.l1 {
                s.input = vec![0u8; self.l1];
            }
            let input = &mut s.input;

            let (acc_stm, acc_opp, psqt_stm, psqt_opp) = match stm {
                Color::White => (&acc.white, &acc.black, &acc.psqt_white, &acc.psqt_black),
                Color::Black => (&acc.black, &acc.white, &acc.psqt_black, &acc.psqt_white),
            };

            let half = self.l1 / 2;
            for (p, side) in [acc_stm, acc_opp].iter().enumerate() {
                let offset = half * p;
                for j in 0..half {
                    let s0 = (side[j] as i32).clamp(0, 254);
                    let s1 = (side[j + half] as i32).clamp(0, 254);
                    input[offset + j] = ((s0 * s1) as u32 / 512) as u8;
                }
            }

            let bucket = (acc.piece_count - 1) / 4;
            let psqt = (psqt_stm[bucket] - psqt_opp[bucket]) / 2;
            let positional = self.propagate_halfka(input, bucket);

            psqt / OUTPUT_SCALE + positional / OUTPUT_SCALE
        })
    }

    fn evaluate_halfkp(&self, acc: &Accumulator, stm: Color, layers: &HalfKPLayers) -> i32 {
        SCRATCH.with(|cell| {
            let mut s = cell.borrow_mut();
            let want = 2 * self.l1;
            if s.input.len() != want {
                s.input = vec![0u8; want];
            }
            let input = &mut s.input;

            let (acc_stm, acc_opp) = match stm {
                Color::White => (&acc.white, &acc.black),
                Color::Black => (&acc.black, &acc.white),
            };

            for (p, side) in [acc_stm, acc_opp].iter().enumerate() {
                let offset = self.l1 * p;
                for j in 0..self.l1 {
                    input[offset + j] = (side[j] as i32).clamp(0, 127) as u8;
                }
            }

            self.propagate_halfkp(input, layers) / HALFKP_FV_SCALE
        })
    }

    /// Evaluate a position directly, computing a fresh accumulator.
    ///
    /// This is the simple, stateless entry point used by both the FEN and trait
    /// integration styles. For a search, prefer the incremental
    /// [`Network::accumulator`] + [`Network::update`] path.
    pub fn evaluate(&self, board: &impl Board) -> i32 {
        let acc = self.accumulator(board);
        self.evaluate_accumulator(&acc, board.side_to_move())
    }

    fn propagate_halfka(&self, input: &[u8], bucket: usize) -> i32 {
        let b = match &self.layers {
            Layers::HalfKAv2Hm { buckets, .. } => &buckets[bucket],
            Layers::HalfKP(_) => unreachable!(),
        };

        let mut fc0_out = [0i32; FC0_OUT];
        let inp = &input[..self.l1];
        for (o, out) in fc0_out.iter_mut().enumerate() {
            let wbase = o * self.l1;
            *out = b.fc0_bias[o] + crate::simd::dot_u8_i8(inp, &b.fc0_weight[wbase..wbase + self.l1]);
        }

        let mut concat = [0u8; FC1_IN];
        for i in 0..L2 {
            let x = fc0_out[i] as i64;
            concat[i] = (x * x >> 19).min(127) as u8;
            concat[L2 + i] = (fc0_out[i] >> 6).clamp(0, 127) as u8;
        }

        let mut fc1_out = [0i32; L3];
        for (o, out) in fc1_out.iter_mut().enumerate() {
            let mut sum = b.fc1_bias[o];
            let wbase = o * 32;
            for i in 0..FC1_IN {
                sum += b.fc1_weight[wbase + i] as i32 * concat[i] as i32;
            }
            *out = sum;
        }

        let mut ac1 = [0u8; L3];
        for i in 0..L3 {
            ac1[i] = (fc1_out[i] >> 6).clamp(0, 127) as u8;
        }

        let mut fc2 = b.fc2_bias;
        for i in 0..L3 {
            fc2 += b.fc2_weight[i] as i32 * ac1[i] as i32;
        }

        let fwd_out = fc0_out[L2] * (600 * OUTPUT_SCALE) / (127 * (1 << 6));
        fc2 + fwd_out
    }

    fn propagate_halfkp(&self, input: &[u8], layers: &HalfKPLayers) -> i32 {
        let inp = &input[..2 * self.l1];

        let mut fc0_out = [0u8; HALFKP_L1_OUT];
        for o in 0..HALFKP_L1_OUT {
            let wbase = o * inp.len();
            let sum =
                layers.fc0_bias[o] + crate::simd::dot_u8_i8(inp, &layers.fc0_weight[wbase..wbase + inp.len()]);
            fc0_out[o] = (sum >> HALFKP_WEIGHT_SCALE_BITS).clamp(0, 127) as u8;
        }

        let mut fc1_out = [0u8; HALFKP_L2_OUT];
        for o in 0..HALFKP_L2_OUT {
            let wbase = o * HALFKP_L1_OUT;
            let sum = layers.fc1_bias[o]
                + crate::simd::dot_u8_i8(&fc0_out, &layers.fc1_weight[wbase..wbase + HALFKP_L1_OUT]);
            fc1_out[o] = (sum >> HALFKP_WEIGHT_SCALE_BITS).clamp(0, 127) as u8;
        }

        layers.fc2_bias + crate::simd::dot_u8_i8(&fc1_out, &layers.fc2_weight)
    }
}
