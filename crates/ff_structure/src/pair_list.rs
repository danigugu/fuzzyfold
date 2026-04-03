//! Pair and PairList definitions. 
//!
//! Compact integer-based representation of base pairs, can 
//! be used as alternative to PairTable representations.
//!
//! A `Pair` is defined by two 16-bit indices (`NAIDX`) packed into a vector
//!  

use std::fmt;
use std::ops::Deref;

use crate::PairTable;
use crate::NAIDX;
use crate::pair::Pair;

/// A collection of base pairs stored in a vector.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Default)]
pub struct PairList {
    pairs: Vec<(NAIDX, NAIDX)>,
}

impl PairList {
    /// Create an empty pair list.
    pub fn new() -> Self {
        Self {
            pairs: Vec::new()
        }
    }

    /// Number of pairs contained in the list.
    pub fn len(&self) -> usize {
        self.pairs.len()
    }

    /// Returns true if there are no pairs.
    pub fn is_empty(&self) -> bool {
        self.pairs.is_empty()
    }

    /// Insert a new pair; returns true if it was newly inserted.
    pub fn insert(&mut self, pair: Pair) -> bool {
        let entry = (pair.i(), pair.j());
        if self.pairs.contains(&entry) {
            return false
        }else {
            self.pairs.push(entry);
            true
        }
    }

    /// Check if a pair exists in the list.
    pub fn contains(&self, pair: &Pair) -> bool {
        self.pairs.contains(&(pair.i(), pair.j()))
    }

}

impl Deref for PairList {
    type Target = Vec<(NAIDX, NAIDX)>;
    fn deref(&self) -> &Self::Target {
        &self.pairs
    }
}


impl From<&PairTable> for PairList {
    fn from(pt: &PairTable) -> Self {
        let mut pairs = Vec::new();
        for (i, &j_opt) in pt.iter().enumerate() {
            let i = i as NAIDX;
            if let Some(j) = j_opt {
                if i < j {
                    pairs.push((i,j));
                }
            }
        }
        Self {
            pairs,
        }
    }
}

impl fmt::Display for PairList {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut first = true;
        for &(i,j) in &self.pairs {
            if !first {
                write!(f, ",")?;
            }
            // Only here we show 1-based values for readability.
            write!(f, "({},{})", i, j)?;
            first = false;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pair_key_roundtrip() {
        let p = Pair::new(1, 42);
        let k = p.key();
        let q = Pair::from_key(k);
        assert_eq!(p, q);
    }

    #[test]
    fn test_pair_list_from_pair_table() {
        let pt = PairTable::try_from("((..))").unwrap();
        let pl = PairList::from(&pt);

        let expected = vec![Pair::new(0, 5), Pair::new(1, 4)];
        
        for p in &expected {
            assert!(pl.contains(p));
        }
        assert!(!pl.contains(&Pair::new(0, 4)));
    }

    #[test]
    fn test_display() {
        let pt = PairTable::try_from("((..))").unwrap();
        let pl = PairList::from(&pt);
        println!("PairList:{}", pl);
        let s = format!("{}", pl);
        assert!(s.contains("(0,5)"));
        assert!(s.contains("(1,4)"));
    }
}
