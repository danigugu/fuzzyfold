use crate::NAIDX;
use crate::P1KEY;

/// A base pair (i, j) with i < j.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Pair {
    i: NAIDX,
    j: NAIDX,
}

impl Pair {
    /// Create a new pair (i, j). Panics in debug if i >= j.
    pub fn new(i: NAIDX, j: NAIDX) -> Self {
        debug_assert!(i < j);
        debug_assert!(j < NAIDX::MAX);
        Pair { i, j }
    }

    /// Return the 5'-side index.
    pub fn i(&self) -> NAIDX {
        self.i
    }

    /// Return the 3'-side index.
    pub fn j(&self) -> NAIDX {
        self.j
    }

    /// Compact 32-bit key encoding both indices.
    pub fn key(&self) -> P1KEY {
        ((self.i as P1KEY) << 16) | (self.j as P1KEY)
    }

    /// Decode a key back into a `Pair`.
    pub fn from_key(key: P1KEY) -> Self {
        let i = (key >> 16) as NAIDX;
        let j = (key & 0xFFFF) as NAIDX;
        debug_assert!(i < j);
        Pair { i, j }
    }
}