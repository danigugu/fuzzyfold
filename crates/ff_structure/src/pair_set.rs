//! Pair and PairSet definitions. 
//!
//! Compact integer-based representation of base pairs, can 
//! be used as alternative to PairTable representations.
//!
//! A `Pair` is defined by two 16-bit indices (`NAIDX`) packed into a
//! 32-bit integer key (`P1KEY`) for efficient set and map storage.
//!
//! We currently do not povide the conversions from PairSet to 
//! PairTable, mainly because at this stage it is not clear if
//! PairSets may be used in the future to include pseudoknots. 
//! 

use std::fmt;
use nohash_hasher::IntSet;
use std::hash::{Hash, Hasher};

use crate::PairTable;
use crate::NAIDX;
use crate::P1KEY;
use crate::pair::Pair;

/// A collection of base pairs represented as compact integer keys.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PairSet {
    length: usize,
    pairs: IntSet<P1KEY>,
}

impl PairSet {
    /// Create an empty pair set for a given sequence length.
    pub fn new(length: usize) -> Self {
        Self {
            length,
            pairs: IntSet::default(),
        }
    }

    /// Number of pairs contained in the set.
    pub fn len(&self) -> usize {
        self.pairs.len()
    }

    /// Returns true if there are no pairs.
    pub fn is_empty(&self) -> bool {
        self.pairs.is_empty()
    }

    /// Insert a new pair; returns true if it was newly inserted.
    pub fn insert(&mut self, pair: Pair) -> bool {
        debug_assert!((pair.j() as usize) < self.length);
        self.pairs.insert(pair.key())
    }

    /// Check if a pair exists in the set.
    pub fn contains(&self, pair: &Pair) -> bool {
        self.pairs.contains(&pair.key())
    }

    /// Iterator over all pairs in arbitrary order.
    pub fn iter(&self) -> impl Iterator<Item = Pair> + '_ {
        self.pairs.iter().map(|&k| Pair::from_key(k))
    }

    /// Return all pairs as a Vec (for deterministic inspection).
    pub fn to_vec(&self) -> Vec<Pair> {
        let mut v: Vec<_> = self.iter().collect();
        v.sort_unstable_by_key(|p| (p.i(), p.j()));
        v
    }

    /// Underlying sequence length (from the originating `PairTable`).
    pub fn length(&self) -> usize {
        self.length
    }

    /// Returns a new PairSet containing only the pairs present in both sets.
    pub fn intersect(&self, other: &Self) -> Self {
        // 1. Optimization: Find which set is smaller.
        // It's faster to loop 10 times and check a set of 1000 
        // than to loop 1000 times and check a set of 10.
        let (smaller, larger) = if self.pairs.len() <= other.pairs.len() {
            (&self.pairs, &other.pairs)
        } else {
            (&other.pairs, &self.pairs)
        };

        // 2. Create a new set. 
        // Pre-allocating capacity prevents multiple "re-sizes" during the loop.
        let mut intersected_pairs = IntSet::with_capacity_and_hasher(
            smaller.len(), 
            Default::default()
        );

        // 3. The actual intersection logic
        for &key in smaller {
            if larger.contains(&key) {
                intersected_pairs.insert(key);
            }
        }

        Self {
            length: self.length, // Assumes both structures have same seq length
            pairs: intersected_pairs,
        }
    }

}

impl Hash for PairSet {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.length.hash(state);
        self.to_vec().hash(state); 
    }
}

impl From<&PairTable> for PairSet {
    fn from(pt: &PairTable) -> Self {
        let mut pairs = IntSet::default();
        for (i, &j_opt) in pt.iter().enumerate() {
            let i = i as NAIDX;
            if let Some(j) = j_opt {
                if i < j {
                    pairs.insert(Pair::new(i, j).key());
                }
            }
        }
        Self {
            length: pt.len(),
            pairs,
        }
    }
}

impl fmt::Display for PairSet {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut first = true;
        for pair in self.to_vec() {
            if !first {
                write!(f, ",")?;
            }
            // Only here we show 1-based values for readability.
            write!(f, "({},{})", pair.i(), pair.j())?;
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
        let pl = PairSet::from(&pt);

        let expected = vec![Pair::new(0, 5), Pair::new(1, 4)];
        assert_eq!(pl.length(), 6);
        assert_eq!(pl.to_vec(), expected);

        for p in &expected {
            assert!(pl.contains(p));
        }
        assert!(!pl.contains(&Pair::new(0, 4)));
    }

    #[test]
    fn test_display() {
        let pt = PairTable::try_from("((..))").unwrap();
        let pl = PairSet::from(&pt);
        let s = format!("{}", pl);
        assert!(s.contains("(0,5)"));
        assert!(s.contains("(1,4)"));
    }
}
