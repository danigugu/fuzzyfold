// This Data Structure is used to store and map the constrained positions of
// a constrained dot bracket notation to either it's pairing nucleotide position (Pair(position)) or X if unpaired.
// It is similar to PairTable, but only stores the mapping of the specified constrained positions, as this reduces the number
// of positions stored and speeds up the structure motif comparisons in motif.rs.
//
// Example:
// 0123456789
// x((xxx))..
//
// 0 -> X; 1 -> Pair(7); 2 -> Pair(6); 3 -> X; 4 -> X; 5 -> X;
//
// Positions with . and ) are ignored, as they do not provide additional information and we don't need to query them
// when comparing a given structure to a motif.

use std::ops::{Deref, DerefMut};
use std::convert::TryFrom;

use crate::NAIDX;
use crate::StructureError;
use std::collections::HashMap;
use crate::{Constraints, ConstraintsVec};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ConstrPos {
    Pair(NAIDX),
    X,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConstrPosMap(pub HashMap<usize, ConstrPos>);

impl ConstrPosMap {
    pub fn new() -> Self {
        ConstrPosMap(HashMap::new())
    }

    pub fn to_sorted_list(&self) -> Vec<(usize, ConstrPos)> {
        let mut sorted_list: Vec<_> = self.0.iter().map(|(&key, value)| (key, value.clone())).collect();
        sorted_list.sort_by(|a, b| a.0.cmp(&b.0)); // Sort by the key (usize)
        sorted_list // Return the sorted list of tuples
    }
}

impl Deref for ConstrPosMap {
    type Target = HashMap<usize, ConstrPos>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl DerefMut for ConstrPosMap {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl TryFrom<&str> for ConstrPosMap {
    type Error = StructureError;

    fn try_from(s: &str) -> Result<Self, Self::Error> {
        let mut map = HashMap::new();
        let mut stack: Vec<usize> = Vec::new();

        for (i, c) in s.chars().enumerate() {
            match c {
                '(' => stack.push(i),

                ')' => {
                    let j = stack.pop().ok_or(StructureError::UnmatchedClose(i))?;

                    let i_idx: NAIDX = i
                        .try_into()
                        .map_err(|_| StructureError::InvalidToken(
                        "index too large".to_string(),
                        "structure".to_string(),
                        i,
                    ))?;

                    map.insert(j, ConstrPos::Pair(i_idx));
                }

                'x' => {
                    map.insert(i, ConstrPos::X);
                }

                '.' => {}

                _ => {
                    return Err(StructureError::InvalidToken(
                        format!("character '{}'", c),
                        "structure".to_string(),
                        i,
                    ));
                }
            }
        }

        // unmatched '('
        if let Some(unmatched) = stack.pop() {
            return Err(StructureError::UnmatchedOpen(unmatched as usize));
        }

        Ok(ConstrPosMap(map))
    }
}

impl TryFrom<&ConstraintsVec> for ConstrPosMap {
    type Error = StructureError;

    fn try_from(db: &ConstraintsVec) -> Result<Self, Self::Error> {
        let mut map = HashMap::new();
        let mut stack: Vec<usize> = Vec::new();

        for (i, dot) in db.iter().enumerate() {
            match dot {
                Constraints::Unspecified => continue,

                Constraints::Open => stack.push(i),

                Constraints::Close => {
                    let j = stack.pop().ok_or(StructureError::UnmatchedClose(i))?;

                    let i_idx: NAIDX = i
                        .try_into()
                        .map_err(|_| StructureError::InvalidToken(
                        "index too large".to_string(),
                        "structure".to_string(),
                        i,
                    ))?;

                    map.insert(j, ConstrPos::Pair(i_idx));
                }

                Constraints::Unpaired => {
                    map.insert(i, ConstrPos::X);
                }

                Constraints::Break => unreachable!("unexpected Break in single-stranded case"),
            }
        }

        // Handle unmatched open parentheses
        if let Some(i) = stack.pop() {
            return Err(StructureError::UnmatchedOpen(i));
        }

        Ok(ConstrPosMap(map))
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_valid_contr_pos_map_string() {
        let cpm: ConstrPosMap = ConstrPosMap::try_from("((xxx)).(xxx)").unwrap();

        for (k, v) in &*cpm {
            println!("{} -> {:?}", k, v);
        }

        assert_eq!(cpm.len(), 9);

        assert_eq!(cpm.get(&0), Some(&ConstrPos::Pair(6)));
        assert_eq!(cpm.get(&1), Some(&ConstrPos::Pair(5)));

        assert_eq!(cpm.get(&2), Some(&ConstrPos::X));
        assert_eq!(cpm.get(&3), Some(&ConstrPos::X));
        assert_eq!(cpm.get(&4), Some(&ConstrPos::X));

        assert_eq!(cpm.get(&5), None); // ) doesn't have to be checked
        assert_eq!(cpm.get(&6), None); // ) doesn't have to be checked

        assert_eq!(cpm.get(&7), None); // .

        assert_eq!(cpm.get(&8), Some(&ConstrPos::Pair(12)));

        assert_eq!(cpm.get(&9), Some(&ConstrPos::X));
        assert_eq!(cpm.get(&10), Some(&ConstrPos::X));
        assert_eq!(cpm.get(&11), Some(&ConstrPos::X));

        assert_eq!(cpm.get(&12), None); 
    }


    #[test]
    fn test_valid_contr_pos_map_constraint_vec() {
        let cv = ConstraintsVec::try_from("((xxx)).(xxx)").unwrap();
        let cpm: ConstrPosMap = ConstrPosMap::try_from(&cv).unwrap();

        for (k, v) in &*cpm {
            println!("{} -> {:?}", k, v);
        }

        assert_eq!(cpm.len(), 9);

        assert_eq!(cpm.get(&0), Some(&ConstrPos::Pair(6)));
        assert_eq!(cpm.get(&1), Some(&ConstrPos::Pair(5)));

        assert_eq!(cpm.get(&2), Some(&ConstrPos::X));
        assert_eq!(cpm.get(&3), Some(&ConstrPos::X));
        assert_eq!(cpm.get(&4), Some(&ConstrPos::X));

        assert_eq!(cpm.get(&5), None); // ) doesn't have to be checked
        assert_eq!(cpm.get(&6), None); // ) doesn't have to be checked

        assert_eq!(cpm.get(&7), None); // .

        assert_eq!(cpm.get(&8), Some(&ConstrPos::Pair(12)));

        assert_eq!(cpm.get(&9), Some(&ConstrPos::X));
        assert_eq!(cpm.get(&10), Some(&ConstrPos::X));
        assert_eq!(cpm.get(&11), Some(&ConstrPos::X));

        assert_eq!(cpm.get(&12), None); 
    }
}