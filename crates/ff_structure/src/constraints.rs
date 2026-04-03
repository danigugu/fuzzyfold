//! Constraints stores a vector representing structural constraints:
//! '()': fixed paired
//! 'x': fixed unpaired
//! '.': unspecified => either paired or unpaired
//! '+': break
//!
//! Used for motifs, where contraints can be defined, but not the full sequence has to be specified


use std::fmt;
use std::ops::Deref;
use std::ops::DerefMut;
use std::convert::TryFrom;

use crate::PairTable;
use crate::MultiPairTable;
use crate::MultiStruct;
use crate::StrandPairTable;
use crate::StructureError;
use crate::DotBracket;
use crate::DotBracketVec;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
/// Characters allowed for 'Constraints'
pub enum Constraints {
    Unpaired, // 'x' (forced unpaired)
    Open,     // '(' (forced paired)
    Close,    // ')' (forced paired)
    Unspecified, // '.' (either paired or unpaired)
    Break,    // '+' or '&'
}

impl TryFrom<char> for Constraints {
    type Error = StructureError;

    fn try_from(c: char) -> Result<Self, Self::Error> {
        match c {
            'x' => Ok(Constraints::Unpaired),
            '(' => Ok(Constraints::Open),
            ')' => Ok(Constraints::Close),
            '.' => Ok(Constraints::Unspecified),
            '+' | '&' => Ok(Constraints::Break),
            _ => Err(StructureError::InvalidToken(c.to_string(), "constraints".into(), 0)),
        }
    }
}

impl From<Constraints> for char {
    fn from(con: Constraints) -> Self {
        match con {
            Constraints::Open => '(',
            Constraints::Close => ')',
            Constraints::Unpaired => 'x',
            Constraints::Unspecified => '.',
            Constraints::Break => '+',
        }
    }
}

/// ConstraintsVec is a compact representation of constraints for a secondary structure. 
/// Note that the field is public, to allow unsafe modifications. Thus, ConstraintVecs 
/// can be malformed and should be converted using the TryFrom trait.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ConstraintsVec(pub Vec<Constraints>);

impl Deref for ConstraintsVec {
    type Target = [Constraints];
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl DerefMut for ConstraintsVec {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl TryFrom<&str> for ConstraintsVec {
    type Error = StructureError;

    fn try_from(s: &str) -> Result<Self, Self::Error> {
        let mut vec = Vec::with_capacity(s.len());
        for (i, c) in s.chars().enumerate() {
            match Constraints::try_from(c) {
                Ok(con) => vec.push(con),
                Err(StructureError::InvalidToken(tok, src, _)) => {
                    return Err(StructureError::InvalidToken(tok, src, i));
                }
                Err(e) => return Err(e),
            }
        }
        Ok(ConstraintsVec(vec))
    }
}


impl TryFrom<&DotBracketVec> for ConstraintsVec {
    type Error = StructureError;

    fn try_from(dbv: &DotBracketVec) -> Result<Self, Self::Error> {
        let mut vec = Vec::with_capacity(dbv.len());
        for (_, db) in dbv.iter().enumerate() {
            let con = match db {
                DotBracket::Open     => Constraints::Open,
                DotBracket::Close    => Constraints::Close,
                DotBracket::Unpaired => Constraints::Unpaired,
                DotBracket::Break    => Constraints::Break,
            };
            vec.push(con);
        }
        Ok(ConstraintsVec(vec))
    }
}
     


impl From<&PairTable> for ConstraintsVec {
    fn from(pt: &PairTable) -> Self {
        let mut result: Vec<Constraints> = Vec::with_capacity(pt.len());
        for (i, &j_opt) in pt.iter().enumerate() {
            match j_opt {
                None => result.push(Constraints::Unpaired),
                Some(j) if (j as usize) > i => result.push(Constraints::Open),
                Some(j) if (j as usize) < i => result.push(Constraints::Close),
                Some(j) if (j as usize) == i => {
                    unreachable!("PairTable construction prevents self-pairing! ({})", i);
                }
                _ => unreachable!(),
            }
        }
        ConstraintsVec(result)
    }
}


impl From<&MultiPairTable> for ConstraintsVec {
    fn from(mpt: &MultiPairTable) -> Self {
        let mut con = Vec::with_capacity(mpt.len());

        for (i, entry) in mpt.iter().enumerate() {
            match entry {
                MultiStruct::Unpaired => {
                    con.push(Constraints::Unpaired);
                }
                MultiStruct::StrandBreak => {
                    con.push(Constraints::Break);
                }
                MultiStruct::Paired(j) => {
                    let j = *j as usize;
                    if i < j {
                        con.push(Constraints::Open);
                    } else {
                        con.push(Constraints::Close);
                    }
                }
            }
        }
        ConstraintsVec(con)
    }
}


impl From<&StrandPairTable> for ConstraintsVec {

    fn from(pt: &StrandPairTable) -> Self {
        let mut result: Vec<Constraints> = Vec::with_capacity(pt.len() + pt.num_strands());

        for (si, strand) in pt.iter().enumerate() {
            for (di, &pair) in strand.iter().enumerate() {
                match pair {
                    None => result.push(Constraints::Unpaired),
                    Some((sj, dj)) => {
                        let sj = sj as usize;
                        let dj = dj as usize;
                        if (sj, dj) > (si, di) {
                            result.push(Constraints::Open);
                        } else if (sj, dj) < (si, di) {
                            result.push(Constraints::Close);
                        } else {
                            panic!("Invalid self-pairing at strand {si}, domain {di}");
                        }
                    }
                }
            }
            // NOTE:: pushes strand break at the end of the ConstraintsVec,
            // intentionally!
            result.push(Constraints::Break);
        }

        ConstraintsVec(result)
    }
}

impl fmt::Display for ConstraintsVec {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for db in &self.0 {
            write!(f, "{}", char::from(*db))?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_constraints_from_char() {
        assert_eq!(Constraints::try_from('x').unwrap(), Constraints::Unpaired);
        assert_eq!(Constraints::try_from('(').unwrap(), Constraints::Open);
        assert_eq!(Constraints::try_from(')').unwrap(), Constraints::Close);
        assert_eq!(Constraints::try_from('.').unwrap(), Constraints::Unspecified);
    }

    #[test]
    fn test_char_from_constraints() {
        assert_eq!(char::from(Constraints::Unpaired), 'x');
        assert_eq!(char::from(Constraints::Open), '(');
        assert_eq!(char::from(Constraints::Close), ')');
        assert_eq!(char::from(Constraints::Unspecified), '.');
    }

    #[test]
    fn test_constraints_from_invalid_char() {
        let res = Constraints::try_from('y');
        assert!(matches!(res, Err(StructureError::InvalidToken(_, src, _)) if src == "constraints"));
    }

    #[test]
    fn test_constraints_vec_from_str() {
        let dbv = ConstraintsVec::try_from("(x).").unwrap();
        assert_eq!(format!("{}", dbv), "(x).");
        assert_eq!(dbv.len(), 4);
        assert_eq!(dbv[0], Constraints::Open);
        assert_eq!(dbv[1], Constraints::Unpaired);
        assert_eq!(dbv[2], Constraints::Close);
        assert_eq!(dbv[3], Constraints::Unspecified);
    }

    #[test]
    fn test_constraints_from_pair_table() {
        let pt = PairTable::try_from("((..))").unwrap();
        let con = ConstraintsVec::from(&pt);
        assert_eq!(format!("{}", con), "((xx))");
    }

    #[test]
    fn test_constraints_from_multi_pair_table_hack() {
        let pt = StrandPairTable::try_from("((..))+").unwrap();
        let con = ConstraintsVec::from(&pt);
        assert_eq!(format!("{}", con), "((xx))+");
    }

    #[test]
    fn test_constraints_vec_from_multi_pair_table() {
        let pt = StrandPairTable::try_from("((..)+)").unwrap();
        let con = ConstraintsVec::from(&pt);
        assert_eq!(format!("{}", con), "((xx)+)+");
    }

    #[test]
    fn test_constraints_from_dot_bracket_vec() {
        let dbv = DotBracketVec::try_from("((..))").unwrap();
        let con = ConstraintsVec::try_from(&dbv).unwrap();
        assert_eq!(format!("{}", con), "((xx))");
}

}