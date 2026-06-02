use std::fmt;
use std::io;

/// Errors returned when loading a network or parsing a position.
#[derive(Debug)]
pub enum Error {
    /// The network file used an unsupported format version.
    UnsupportedVersion(u32),
    /// The network data was malformed (bad header, truncated, wrong sizes).
    InvalidData(&'static str),
    /// A FEN string could not be parsed.
    Fen(&'static str),
    /// An underlying I/O error occurred while reading the network.
    Io(io::Error),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::UnsupportedVersion(v) => write!(f, "unsupported NNUE version: {:#x}", v),
            Error::InvalidData(m) => write!(f, "invalid NNUE data: {}", m),
            Error::Fen(m) => write!(f, "invalid FEN: {}", m),
            Error::Io(e) => write!(f, "io error: {}", e),
        }
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Error::Io(e) => Some(e),
            _ => None,
        }
    }
}

impl From<io::Error> for Error {
    fn from(e: io::Error) -> Self {
        Error::Io(e)
    }
}

/// Convenience alias for results returned by this crate.
pub type Result<T> = std::result::Result<T, Error>;
