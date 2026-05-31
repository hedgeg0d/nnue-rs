use std::cell::RefCell;
use std::fs::File;
use std::io::{self, BufReader, Read};

use crate::feature::{active_indices, make_index, INPUT_DIMENSIONS};
use crate::leb128;
use crate::types::{Board, Color};

#[derive(Default)]
struct Scratch {
    white: Vec<usize>,
    input: Vec<u8>,
}

thread_local! {
    static SCRATCH: RefCell<Scratch> = RefCell::new(Scratch::default());
}

const VERSION: u32 = 0x7AF32F20;
const FEATURE_HASH: u32 = 0x7f234cb8;
const PSQT_BUCKETS: usize = 8;
const LAYER_STACKS: usize = 8;
const L2: usize = 15;
const L3: usize = 32;
const FC0_OUT: usize = L2 + 1;
const FC1_IN: usize = L2 * 2;
const OUTPUT_SCALE: i32 = 16;

struct Bucket {
    fc0_bias: Vec<i32>,
    fc0_weight: Vec<i8>,
    fc1_bias: Vec<i32>,
    fc1_weight: Vec<i8>,
    fc2_bias: i32,
    fc2_weight: Vec<i8>,
}

pub struct Network {
    l1: usize,
    ft_bias: Vec<i16>,
    ft_weight: Vec<i16>,
    ft_psqt: Vec<i32>,
    buckets: Vec<Bucket>,
}

#[derive(Clone)]
pub struct Accumulator {
    white: Vec<i16>,
    black: Vec<i16>,
    psqt_white: [i32; PSQT_BUCKETS],
    psqt_black: [i32; PSQT_BUCKETS],
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

fn read_i8_raw(reader: &mut impl Read, count: usize) -> io::Result<Vec<i8>> {
    let mut bytes = vec![0u8; count];
    reader.read_exact(&mut bytes)?;
    Ok(bytes.into_iter().map(|b| b as i8).collect())
}

impl Network {
    pub fn from_file(path: &str) -> io::Result<Self> {
        Self::from_reader(&mut BufReader::new(File::open(path)?))
    }

    pub fn from_bytes(bytes: &[u8]) -> io::Result<Self> {
        Self::from_reader(&mut &bytes[..])
    }

    pub fn from_reader(reader: &mut impl Read) -> io::Result<Self> {
        let version = read_u32(reader)?;
        if version != VERSION {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "unsupported NNUE version",
            ));
        }
        let _arch_hash = read_u32(reader)?;
        let desc_len = read_u32(reader)? as usize;
        let mut desc = vec![0u8; desc_len];
        reader.read_exact(&mut desc)?;

        let ft_hash = read_u32(reader)?;
        let l1 = ((ft_hash ^ FEATURE_HASH) / 2) as usize;
        if l1 == 0 || l1 % 2 != 0 {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "bad L1"));
        }

        let mut ft_bias = leb128::read_i16(reader, l1)?;
        let mut ft_weight = leb128::read_i16(reader, l1 * INPUT_DIMENSIONS)?;
        let ft_psqt = leb128::read_i32(reader, PSQT_BUCKETS * INPUT_DIMENSIONS)?;

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
            l1,
            ft_bias,
            ft_weight,
            ft_psqt,
            buckets,
        })
    }

    fn accumulate(&self, indices: &[usize], acc: &mut [i16], psqt: &mut [i32]) {
        acc.copy_from_slice(&self.ft_bias);
        psqt.iter_mut().for_each(|p| *p = 0);
        for &feat in indices {
            let base = feat * self.l1;
            crate::simd::add_i16(acc, &self.ft_weight[base..base + self.l1]);
            let pbase = feat * PSQT_BUCKETS;
            for b in 0..PSQT_BUCKETS {
                psqt[b] += self.ft_psqt[pbase + b];
            }
        }
    }

    pub fn new_accumulator(&self) -> Accumulator {
        Accumulator {
            white: vec![0i16; self.l1],
            black: vec![0i16; self.l1],
            psqt_white: [0i32; PSQT_BUCKETS],
            psqt_black: [0i32; PSQT_BUCKETS],
        }
    }

    fn refresh_side(&self, board: &impl Board, color: Color, acc: &mut [i16], psqt: &mut [i32; PSQT_BUCKETS]) {
        SCRATCH.with(|cell| {
            let mut s = cell.borrow_mut();
            active_indices(board, color, &mut s.white);
            self.accumulate(&s.white, acc, psqt);
        });
    }

    pub fn refresh(&self, board: &impl Board, acc: &mut Accumulator) {
        self.refresh_side(board, Color::White, &mut acc.white, &mut acc.psqt_white);
        self.refresh_side(board, Color::Black, &mut acc.black, &mut acc.psqt_black);
    }

    fn apply_side(
        &self,
        parent: &[i16],
        parent_psqt: &[i32; PSQT_BUCKETS],
        child: &mut [i16],
        child_psqt: &mut [i32; PSQT_BUCKETS],
        king_square: u8,
        color: Color,
        removed: &[(u8, usize)],
        added: &[(u8, usize)],
    ) {
        child.copy_from_slice(parent);
        *child_psqt = *parent_psqt;
        for &(sq, piece) in removed {
            let feat = make_index(color, sq, piece, king_square);
            let base = feat * self.l1;
            crate::simd::sub_i16(child, &self.ft_weight[base..base + self.l1]);
            let pbase = feat * PSQT_BUCKETS;
            for b in 0..PSQT_BUCKETS {
                child_psqt[b] -= self.ft_psqt[pbase + b];
            }
        }
        for &(sq, piece) in added {
            let feat = make_index(color, sq, piece, king_square);
            let base = feat * self.l1;
            crate::simd::add_i16(child, &self.ft_weight[base..base + self.l1]);
            let pbase = feat * PSQT_BUCKETS;
            for b in 0..PSQT_BUCKETS {
                child_psqt[b] += self.ft_psqt[pbase + b];
            }
        }
    }

    pub fn update(
        &self,
        parent: &Accumulator,
        child: &mut Accumulator,
        board: &impl Board,
        white_king_moved: bool,
        black_king_moved: bool,
        removed: &[(u8, usize)],
        added: &[(u8, usize)],
    ) {
        if white_king_moved {
            self.refresh_side(board, Color::White, &mut child.white, &mut child.psqt_white);
        } else {
            let wk = board.king_square(Color::White);
            self.apply_side(
                &parent.white, &parent.psqt_white,
                &mut child.white, &mut child.psqt_white,
                wk, Color::White, removed, added,
            );
        }
        if black_king_moved {
            self.refresh_side(board, Color::Black, &mut child.black, &mut child.psqt_black);
        } else {
            let bk = board.king_square(Color::Black);
            self.apply_side(
                &parent.black, &parent.psqt_black,
                &mut child.black, &mut child.psqt_black,
                bk, Color::Black, removed, added,
            );
        }
    }

    pub fn evaluate_accumulator(&self, acc: &Accumulator, stm: Color, piece_count: usize) -> i32 {
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

            let bucket = (piece_count - 1) / 4;
            let psqt = (psqt_stm[bucket] - psqt_opp[bucket]) / 2;
            let positional = self.propagate(input, bucket);

            psqt / OUTPUT_SCALE + positional / OUTPUT_SCALE
        })
    }

    pub fn evaluate(&self, board: &impl Board) -> i32 {
        let mut acc = self.new_accumulator();
        self.refresh(board, &mut acc);
        let mut piece_count = 0usize;
        board.for_each_piece(&mut |_, _| piece_count += 1);
        self.evaluate_accumulator(&acc, board.side_to_move(), piece_count)
    }

    fn propagate(&self, input: &[u8], bucket: usize) -> i32 {
        let b = &self.buckets[bucket];

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
}
