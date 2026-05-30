use std::io::{self, Read};

const MAGIC: &[u8] = b"COMPRESSED_LEB128";

pub fn read_i16(reader: &mut impl Read, count: usize) -> io::Result<Vec<i16>> {
    read_signed::<i16>(reader, count, 16)
}

pub fn read_i32(reader: &mut impl Read, count: usize) -> io::Result<Vec<i32>> {
    read_signed::<i32>(reader, count, 32)
}

trait FromI64 {
    fn from_i64(v: i64) -> Self;
}
impl FromI64 for i16 {
    fn from_i64(v: i64) -> Self {
        v as i16
    }
}
impl FromI64 for i32 {
    fn from_i64(v: i64) -> Self {
        v as i32
    }
}

fn read_signed<T: FromI64>(reader: &mut impl Read, count: usize, bits: u32) -> io::Result<Vec<T>> {
    let mut magic = [0u8; 17];
    reader.read_exact(&mut magic)?;
    if &magic != MAGIC {
        return Err(io::Error::new(io::ErrorKind::InvalidData, "bad LEB128 magic"));
    }

    let mut len_buf = [0u8; 4];
    reader.read_exact(&mut len_buf)?;
    let mut bytes_left = u32::from_le_bytes(len_buf) as i64;

    let mut out = Vec::with_capacity(count);
    let mut byte = [0u8; 1];
    for _ in 0..count {
        let mut result: i64 = 0;
        let mut shift: u32 = 0;
        loop {
            reader.read_exact(&mut byte)?;
            bytes_left -= 1;
            let b = byte[0];
            result |= ((b & 0x7f) as i64) << shift;
            shift += 7;
            if b & 0x80 == 0 {
                if shift < bits && b & 0x40 != 0 {
                    result |= -1i64 << shift;
                }
                break;
            }
        }
        out.push(T::from_i64(result));
    }

    while bytes_left > 0 {
        reader.read_exact(&mut byte)?;
        bytes_left -= 1;
    }
    Ok(out)
}
