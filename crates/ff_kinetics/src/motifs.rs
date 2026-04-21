use std::fs::File;
use std::io::BufRead;
use std::io::BufReader;
use std::io;
use std::path::PathBuf;
use std::sync::Arc;
use std::cmp::min;

use ff_structure::DotBracketVec;
use ff_structure::PairTable;
use ff_structure::ConstrPosMap;
use ff_structure::ConstrPos;
use ff_structure::ConstraintsVec;
use ff_structure::NAIDX;
use ff_energy::NucleotideVec;
use ff_energy::EnergyModel;

use crate::{K0, KB};

#[derive(Debug)]
pub struct Motif {
    name: String,
    constr_pos_map: ConstrPosMap,
    motif_energy: Option<f64>,
    motif_prob: Option<f64>,
    allowed_distance: NAIDX
}

impl Motif {
    /// Initialize a catch-all motif. 
    /// (This is the default motif when initializing a MotifRegistry.)
    pub fn new_catch_all(name: &str) -> Self {
        Motif { 
            name: name.to_owned(),
            constr_pos_map: ConstrPosMap::new(),
            motif_energy: None,
            motif_prob: None,
            allowed_distance: 0
        }
    }

    // Version 1: For `&ConstraintsVec`
    pub fn from_constr_vec<E: EnergyModel>(
        name: &str,
        sequence: Arc<NucleotideVec>,
        structure: &ConstraintsVec,
        energy_model: Arc<E>,
        allowed_distance: NAIDX,
    ) -> Self {
        let _rt = KB * (K0 + energy_model.temperature());

        let _ = sequence;

        Self {
            name: name.to_owned(),
            constr_pos_map: ConstrPosMap::try_from(structure).unwrap(),
            motif_energy: Some(0.0),
            motif_prob: Some(0.0),
            allowed_distance: allowed_distance,
        }
    }

    // Version 2: For `&str`
    pub fn from_str<E: EnergyModel>(
        name: &str,
        sequence: Arc<NucleotideVec>,
        structure: &str,
        energy_model: Arc<E>,
        allowed_distance: NAIDX,
    ) -> Self {
        let _rt = KB * (K0 + energy_model.temperature());

        let _ = sequence;

        Self {
            name: name.to_owned(),
            constr_pos_map: ConstrPosMap::try_from(structure).unwrap(),
            motif_energy: Some(0.0),
            motif_prob: Some(0.0),
            allowed_distance: allowed_distance,
        }
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn allowed_distance(&self) -> &NAIDX {
        &self.allowed_distance
    }

    pub fn energy(&self) -> Option<f64> {
        self.motif_energy
    }

    pub fn prob(&self) -> Option<f64> {
        self.motif_prob
    }

    /// Check if a secondary structure is contained in this motif.
    pub fn contains(&self, structure: &PairTable) -> bool {
        let mut dist_counter = 0;

        for (key, value) in &self.constr_pos_map.0 {
            if *key as usize >= structure.len() {
                return false; 
            }

            let entry: Option<NAIDX> = structure.get(key);

            match value {
                // Pair is mismatched -> add distance of 2
                ConstrPos::Pair(expected) => {
                    if entry != Some(*expected) {
                        dist_counter += 2;
                    }
                }
                // Unpaired Position is paired -> add distance of 1
                ConstrPos::X => {
                    if entry.is_some() {
                        dist_counter += 1;
                    }
                }
            }

            // If distance exceeds the allowed threshold, return false
            if dist_counter > self.allowed_distance {
                return false;
            }
        }
        true
    }

    pub fn distance(&self, structure: &PairTable) -> usize {
        let mut dist_counter = 0;

        for (key, value) in &self.constr_pos_map.0 {
            if *key as usize >= structure.len() {
                return 0;
            }

            let entry: Option<NAIDX> = structure.get(key);

            match value {
                // Pair is mismatched -> add distance of 2
                ConstrPos::Pair(expected) => {
                    if entry != Some(*expected) {
                        dist_counter += 2;
                    }
                }
                // Unpaired Position is paired -> add distance of 1
                ConstrPos::X => {
                    if entry.is_some() {
                        dist_counter += 1;
                    }
                }
            }

        }
        return dist_counter;
    }

}



/// A registry to collect macrostate definitions.
pub struct MotifRegistry<E: EnergyModel> {
    sequence: Arc<NucleotideVec>,
    energy_model: Arc<E>,
    /// By convention: motifs[0] = unassigned.
    motifs: Vec<Motif>,
    min_motif_length: usize
}

impl<E: EnergyModel> From<(Arc<NucleotideVec>, Arc<E>)> for MotifRegistry<E> {
    fn from((sequence, energy_model): (Arc<NucleotideVec>, Arc<E>)) -> Self {
        let motifs = vec![Motif::new_catch_all("Unassigned")];

        MotifRegistry {
            sequence,
            energy_model,
            motifs,
            min_motif_length: usize::MAX,
        }
    }
}

fn io_err(msg: &str, src: &str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, format!("{} in {}", msg, src))
}

impl<E: EnergyModel> MotifRegistry<E> {

    /// High-level entry: read one or more macrostate files from disk.
    pub fn insert_from_file(&mut self, file: &PathBuf) -> io::Result<()> {
        let fh = File::open(file)?;
        let reader = BufReader::new(fh);
        self.insert_from_reader(reader, &file.display().to_string())?;
        Ok(())
    }

    pub fn insert_from_reader<R: BufRead>(&mut self, reader: R, source: &str) -> io::Result<()> {
        let mut lines = reader.lines();

        let mut seq_line = String::new();
        while let Some(line_res) = lines.next() {
            let l = line_res?.trim().to_string();
            if !l.is_empty() {
                seq_line = l;
                break;
            }
        }

        if seq_line.is_empty() {
            return Err(io_err("File is empty or contains only whitespace", source));
        }

        // 2. Parse the sequence (same as before)
        let file_seq = NucleotideVec::try_from(seq_line.as_str())
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
        
        if &file_seq != self.sequence.as_ref() {
            return Err(io_err("Sequence does not match input sequence", source));
        }

        let mut motifs = Vec::new();

        // Read the motif lines
        while let Some(line_result) = lines.next() {
            let line = line_result?.trim().to_string();

            // Skip empty lines/comments
            if line.is_empty() || line.starts_with('#') {
                continue;
            }

            // Handle the Header Line (e.g., ">motif_name1 0")
            if !line.starts_with('>') {
                return Err(io_err("Expected motif header starting with '>'", source));
            }
            
            let header = line.trim_start_matches('>');
            let mut parts = header.split_whitespace();
            
            let motif_name = parts.next()
                .ok_or_else(|| io_err("Missing motif name", source))?.to_string();
            
            let distance_str = parts.next()
                .ok_or_else(|| io_err("Missing distance", source))?;
                
            let allowed_distance = distance_str.parse::<NAIDX>()
                .map_err(|_| io_err("Invalid distance", source))?;


            // Handle the Structure Line (The VERY NEXT line)
            let structure_line = lines.next()
                .ok_or_else(|| io_err(&format!("Missing structure for motif {}", motif_name), source))??;
            
            let mut parts = structure_line.split_whitespace();
            // The first part is the structure
            let structure_str = parts.next().ok_or_else(|| io_err("Empty structure", source))?;

            // If there is a NEXT part, that is "trailing data"
            if parts.next().is_some(){
                eprintln!("Warning: trailing data (like extra numbers) ignored in {}.", source);
            }

            // Create the Motif
            let motif = Motif::from_str(
                &motif_name,
                Arc::clone(&self.sequence),
                structure_str,
                Arc::clone(&self.energy_model),
                allowed_distance,
            );

            motifs.push(motif);
            self.min_motif_length = min(structure_str.len(), self.min_motif_length);
        }
    

        // If no motifs are found, return an error
        if motifs.is_empty() {
            return Err(io_err("No motifs found", source));
        }

        // Add the motifs to the registry
        self.motifs.extend(motifs);
        Ok(())
    }


    pub fn classify(&self, structure: &DotBracketVec) -> Vec<usize> {
        let mut matches: Vec<usize> = Vec::new();
        let structure_pt = PairTable::try_from(structure).unwrap();


        for (i, ms) in self.motifs.iter().enumerate().skip(1) {

            if ms.contains(&structure_pt) {
                matches.push(i);
            }
        }

        match matches.len() {
            0 => vec![0usize],
            _ => matches
        }
    }

    pub fn sequence(&self) -> &NucleotideVec {
        self.sequence.as_ref()
    }

    pub fn min_motif_length(&self) -> &usize {
        &self.min_motif_length
    }

    pub fn energy_model(&self) -> &E {
        self.energy_model.as_ref()
    }

    pub fn motifs(&self) -> &Vec<Motif> {
        &self.motifs
    }

    /// Number of motifs, including the catch-all unassigned motif.
    pub fn len(&self) -> usize {
        self.motifs.len()
    }


    /// Iterate over all macrostates
    pub fn iter(&self) -> impl Iterator<Item = (usize, &Motif)> {
        self.motifs.iter().enumerate()
    }


}


#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;
    use std::sync::Arc;
    use ff_energy::{parameters::RNA_TURNER_2004, ViennaRNA};

    #[test]
    fn test_motif_init_constr_vec() {
        /*        
        >lmin=lm3_bh=3.0
        UCAGUCUUCGCUGCGCUGUAUCGAUUCGGUUUCAGUUUUUAUUGC
        .((((....)))).((((........))))...............
        */        

        let energy_model = Arc::new(ViennaRNA::from_thermo_params(&RNA_TURNER_2004, 37.0));
        let seq = Arc::new(NucleotideVec::try_from("UCAGUCUUCGCUGCGCUGUAUCGAUUCGGUUUCAGUUUUUAUUGC").unwrap());

        let cv1 = ConstraintsVec::try_from(".((((xxxx)))).((((xxxxxxxx))))...............").unwrap(); // REFERENCE
        let cv2 = ConstraintsVec::try_from(".((((....)))).((((........))))...(......)....").unwrap(); // MATCHING
        let cv3 = ConstraintsVec::try_from(".((((....)))).(((((......)))))...............").unwrap(); // NON-MATCHING

        let motif = Motif::from_constr_vec(
            "lmin=lm3_bh=3.0",
            Arc::clone(&seq),
            &cv1,
            Arc::clone(&energy_model),
            0
        );

        println!("USING CONSTR VECTOR");
        println!("Motif '{}':", motif.name());
        println!("E(motif) = {:.4}, P(motif) = {:.4}", motif.energy().unwrap(), motif.prob().unwrap());
        println!();


        println!("{}", &seq);
        println!("{}", &cv2);
        println!("{}", &cv3);
        println!();

        let pt = PairTable::try_from(&cv2).unwrap();
        for (k, v) in &motif.constr_pos_map.to_sorted_list() {
            println!("Pos: {} -> Motif: {:?}    Observed: {:?} ", k, v, pt.get(k));
        }
        println!();

        assert!(motif.contains(&pt));


        println!("{}", &seq);
        println!("{}", &cv1);
        println!("{}", &cv3);
        println!();

        let pt = PairTable::try_from(&cv3).unwrap();
        for (k, v) in &motif.constr_pos_map.to_sorted_list() {
            println!("Pos: {} -> Motif: {:?}    Observed: {:?} ", k, v, pt.get(k));
        }
        println!();

        assert!(!motif.contains(&pt));

    }



    #[test]
    fn test_motif_init_str() {
        /*        
        >lmin=lm3_bh=3.0
        UCAGUCUUCGCUGCGCUGUAUCGAUUCGGUUUCAGUUUUUAUUGC
        .((((....)))).((((........))))...............
        */ 

        let energy_model = Arc::new(ViennaRNA::from_thermo_params(&RNA_TURNER_2004, 37.0));
        let seq = Arc::new(NucleotideVec::try_from("UCAGUCUUCGCUGCGCUGUAUCGAUUCGGUUUCAGUUUUUAUUGC").unwrap());

        let dbs1 = ".((((xxxx)))).((((xxxxxxxx))))..............."; // REFERENCE
        let dbs2 = ".((((....)))).((((........))))...(......)...."; // MATCHING
        let dbs3 = ".((((....)))).(((((......)))))..............."; // NON-MATCHING

        let motif = Motif::from_str(
            "lmin=lm3_bh=3.0",
            Arc::clone(&seq),
            &dbs1,
            Arc::clone(&energy_model),
            0
        );

        println!("USING STRING");
        println!("Motif '{}':", motif.name());
        println!("E(motif) = {:.4}, P(motif) = {:.4}", motif.energy().unwrap(), motif.prob().unwrap());
        println!();


        println!("{}", &seq);
        println!("{}", dbs1);
        println!("{}", dbs2);
        println!();

        let pt = PairTable::try_from(dbs2).unwrap();
        for (k, v) in &motif.constr_pos_map.to_sorted_list() {
            println!("Pos: {} -> Motif: {:?}    Observed: {:?} ", k, v, pt.get(k));
        }
        println!();

        assert!(motif.contains(&pt));


        println!("{}", &seq);
        println!("{}", dbs1);
        println!("{}", dbs3);
        println!();

        let pt = PairTable::try_from(dbs3).unwrap();
        for (k, v) in &motif.constr_pos_map.to_sorted_list() {
            println!("Pos: {} -> Motif: {:?}    Observed: {:?} ", k, v, pt.get(k));
        }
        println!();

        assert!(!motif.contains(&pt));

    }




    #[test]
    fn test_motifregistry_init_and_classify() {
        let energy_model = Arc::new(ViennaRNA::from_thermo_params(&RNA_TURNER_2004, 37.0));
        let seq = Arc::new(NucleotideVec::try_from("UCAGUCUUCGCUGCGCUGUAUCGAUUCGGUUUCAGUUUUUAUUGC").unwrap());

        let mut registry = MotifRegistry::from((Arc::clone(&seq), Arc::clone(&energy_model)));

        assert_eq!(registry.len(), 1);
        assert_eq!(registry.motifs()[0].name(), "Unassigned");

        let input = 
     r#"UCAGUCUUCGCUGCGCUGUAUCGAUUCGGUUUCAGUUUUUAUUGC
        >motif_name1 0
        .((((xxxx)))).((((xxxxxxxx))))...............
        >motif_name2 0
        .((((....)))).((((........))))...............
        >motif_name3 0
        .((((xxxx)))).((((........))))xxxxxxxxxxxxxxx
        >motif_name4 0
        xxxxxxxxxxxxx"#.as_bytes();

        registry.insert_from_reader(Cursor::new(input), "manual").unwrap();
        // Can also use a file
        // let file_path = PathBuf::from("path_to_file/test.txt");
        // registry.insert_from_file(&file_path).unwrap();
        assert_eq!(registry.len(), 5);

        let s1 = DotBracketVec::try_from(".((((....)))).((((........)))).....(.....)...").unwrap();
        assert_eq!(registry.classify(&s1), vec![1, 2]);
        let s2 = DotBracketVec::try_from(".((((....)))).((((((....)))))).....(.....)...").unwrap();
        assert_eq!(registry.classify(&s2), vec![2]);
        let s3 = DotBracketVec::try_from(".((((....)))).((((........))))...............").unwrap();
        assert_eq!(registry.classify(&s3), vec![1, 2, 3]);
        let s4 = DotBracketVec::try_from("..(((....)))..((((((....)))))).....(.....)...").unwrap();
        assert_eq!(registry.classify(&s4), vec![0]);
        let s5 = DotBracketVec::try_from(".((((....)))).((((........))))").unwrap();
        assert_eq!(registry.classify(&s5), vec![1, 2]);
        let s6 = DotBracketVec::try_from("..............((((((....)))))).....(.....)...").unwrap();
        assert_eq!(registry.classify(&s6), vec![4]);

        // Iteration test
        let all_names: Vec<_> = registry.iter().map(|(_, ms)| ms.name().to_string()).collect();
        assert!(all_names.contains(&"Unassigned".to_string()));
        assert!(all_names.contains(&"motif_name1".to_string()));
        assert!(all_names.contains(&"motif_name2".to_string()));
        assert!(all_names.contains(&"motif_name3".to_string()));
        assert!(all_names.contains(&"motif_name4".to_string()));
    }

    #[test]
    fn test_registry_with_varied_distances() {
        let energy_model = Arc::new(ViennaRNA::from_thermo_params(&RNA_TURNER_2004, 37.0));
        let seq = Arc::new(NucleotideVec::try_from("UCAGUCUUCGCUGCGCUGUAUCGAUUCGGUUUCAGUUUUUAUUGC").unwrap());
        let mut registry = MotifRegistry::from((Arc::clone(&seq), Arc::clone(&energy_model)));

        let input = r#"UCAGUCUUCGCUGCGCUGUAUCGAUUCGGUUUCAGUUUUUAUUGC
                              >motif_exact 0
                              x((((xxxx))))x((((xxxxxxxx))))xxxxxxxx.......
                              >motif_dist1 1
                              x((((xxxx))))x((((xxxxxxxx))))xxxxxxxx.......
                              >motif_dist2 2
                              x((((xxxx))))x((((xxxxxxxx))))xxxxxxxx.......
                              >motif_dist3 3
                              x((((xxxx))))x((((xxxxxxxx))))xxxxxxxx.......
                              >motif_dist4 4
                              x((((xxxx))))x((((xxxxxxxx))))xxxxxxxx......."#.as_bytes();

        registry.insert_from_reader(Cursor::new(input), "test").unwrap();

        let exact_structure_dist00 = DotBracketVec::try_from(".((((....)))).((((........))))...............").unwrap();
        assert_eq!(registry.classify(&exact_structure_dist00), vec![1, 2, 3, 4, 5]);
        let faulty_structure_dist3 = DotBracketVec::try_from(".((((....)))).(((((......)))))...(........)..").unwrap();
        assert_eq!(registry.classify(&faulty_structure_dist3), vec![4, 5]);
        let faulty_structure_dist5 = DotBracketVec::try_from(".((((....)))).((((((....))))))...(........)..").unwrap();
        assert_eq!(registry.classify(&faulty_structure_dist5), vec![0]);

    }
}