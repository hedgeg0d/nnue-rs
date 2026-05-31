use std::cell::RefCell;
use std::fs::File;
use std::io::{self, BufReader, Read};

use crate::feature::{active_indices, INPUT_DIMENSIONS};
use crate::leb128;
use crate::types::{Board, Color};

#[derive(Default)]
struct Scratch {
    white: Vec<usize>,
    black: Vec<usize>,
    acc_white: Vec<i16>,
    acc_black: Vec<i16>,
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
        for &feat in indices {
            let base = feat * self.l1;
            let weights = &self.ft_weight[base..base + self.l1];
            for (a, &w) in acc.iter_mut().zip(weights) {
                *a = a.wrapping_add(w);
            }
            let pbase = feat * PSQT_BUCKETS;
            for b in 0..PSQT_BUCKETS {
                psqt[b] += self.ft_psqt[pbase + b];
            }
        }
    }

    pub fn evaluate(&self, board: &impl Board) -> i32 {
        SCRATCH.with(|cell| {
            let mut s = cell.borrow_mut();
            if s.acc_white.len() != self.l1 {
                s.acc_white = vec![0i16; self.l1];
                s.acc_black = vec![0i16; self.l1];
                s.input = vec![0u8; self.l1];
            }

            let stm = board.side_to_move();
            active_indices(board, Color::White, &mut s.white);
            active_indices(board, Color::Black, &mut s.black);
            let piece_count = s.white.len();

            let mut psqt_white = [0i32; PSQT_BUCKETS];
            let mut psqt_black = [0i32; PSQT_BUCKETS];
            let Scratch {
                white,
                black,
                acc_white,
                acc_black,
                input,
            } = &mut *s;
            self.accumulate(white, acc_white, &mut psqt_white);
            self.accumulate(black, acc_black, &mut psqt_black);

            let (acc_stm, acc_opp, psqt_stm, psqt_opp) = match stm {
                Color::White => (&*acc_white, &*acc_black, &psqt_white, &psqt_black),
                Color::Black => (&*acc_black, &*acc_white, &psqt_black, &psqt_white),
            };

            let half = self.l1 / 2;
            for (p, acc) in [acc_stm, acc_opp].iter().enumerate() {
                let offset = half * p;
                for j in 0..half {
                    let s0 = (acc[j] as i32).clamp(0, 254);
                    let s1 = (acc[j + half] as i32).clamp(0, 254);
                    input[offset + j] = ((s0 * s1) as u32 / 512) as u8;
                }
            }

            let bucket = (piece_count - 1) / 4;
            let psqt = (psqt_stm[bucket] - psqt_opp[bucket]) / 2;
            let positional = self.propagate(input, bucket);

            psqt / OUTPUT_SCALE + positional / OUTPUT_SCALE
        })
    }

    fn propagate(&self, input: &[u8], bucket: usize) -> i32 {
        let b = &self.buckets[bucket];

        let mut fc0_out = [0i32; FC0_OUT];
        for (o, out) in fc0_out.iter_mut().enumerate() {
            let mut sum = b.fc0_bias[o];
            let wbase = o * self.l1;
            for i in 0..self.l1 {
                sum += b.fc0_weight[wbase + i] as i32 * input[i] as i32;
            }
            *out = sum;
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
