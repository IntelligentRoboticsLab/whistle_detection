use std::path::Path;

/// Loads an audio file, assuming 16 bits per sample. Not very rust-esque as it crashes
/// whenever it wants, but I couldn't care less rn.
pub fn load<P>(path: &P) -> Vec<f32>
    where P: AsRef<Path>
{
    hound::WavReader::open(path).expect("Could not load file!")
        .samples::<i16>()
        // normalization
        .map(|s| s.expect("Not enough bits per sample!") as f32 / i16::MAX as f32)
        .collect::<Vec<_>>()
}