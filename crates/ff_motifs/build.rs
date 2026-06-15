use std::env;
use std::path::PathBuf;

fn main() {
    // Determine the directory where the crate is located
    let crate_dir = env::var("CARGO_MANIFEST_DIR").unwrap();

    // Define the output path to match your Makefile (-I$(RUST_CRATE_DIR)/src)
    let output_file = PathBuf::from(&crate_dir).join("src").join("motifs.h");

    // Generate the bindings
    cbindgen::Builder::new()
        .with_crate(crate_dir)
        .with_config(cbindgen::Config::from_file("cbindgen.toml").unwrap_or_default())
        .generate()
        .expect("Unable to generate bindings")
        .write_to_file(output_file);
}