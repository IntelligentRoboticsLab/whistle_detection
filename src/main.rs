use std::{path::Path, sync::Arc};
use rustfft::{num_complex::Complex, num_traits::Zero, FftPlanner};
use serde::Serialize;
use serde_json::json;

mod audio;

const WINDOW_SIZE: usize = 512;
const HOP_SIZE: usize = 256;
const MEAN_WINDOWS: usize = 7;

#[derive(Debug, Serialize)]
pub struct Spectrogram {
    pub powers: Vec<f32>,
    pub freq_bins: usize
}

impl Spectrogram {
    /// Returns the mean of all windows.
    pub fn windows_mean(mut self) -> Self {
        let mut powers = self.powers[0..self.freq_bins].to_vec();
        for (i, p) in self.powers.iter().skip(self.freq_bins).enumerate() {
            powers[i % self.freq_bins] += p;
        }

        let windows = (self.powers.len() / self.freq_bins) as f32;
        for i in 0..powers.len() {
            powers[i] /= windows;
        }
        return Self { powers, freq_bins: self.freq_bins }
    }

    /// Scales each window vector with the inverse euclidian norm: x = x / |x|
    pub fn normalize_euclid(mut self) -> Self {
        let window_len = self.freq_bins;

        // loop over window vectors
        self.powers.chunks_mut(window_len)
            .for_each(|vec| {
                // normalize
                let norm = vec.iter().fold(0.0, |acc, x| acc + x*x).sqrt();
                
                if !norm.is_zero() {
                    vec.iter_mut().for_each(|s| *s = *s / norm);
                }
            });
        self
    }

    /// Returns the power per frequency for Hz in the range \[`min_freq`, `max_freq`],
    /// assuming the original audio had the given `sample_rate`.
    pub fn hz_power_iter(
        &self, window: usize, sample_rate: usize, min_freq: f32, max_freq: f32
    ) -> impl Iterator<Item = (f32, f32)> + '_ {
        let nyquist = (sample_rate / 2 + 1) as f32;

        self.powers.iter()
            // samples of the given window
            .skip(self.freq_bins * window).take(self.freq_bins)
            .enumerate()
            // frequencies range from [0, nyquist)
            .map(move |(i, p)| (i as f32 / self.freq_bins as f32 * nyquist, *p))
    }
}

/// Short time fourier transform.
/// Manages internal state to avoid repeat allocations over multiple calls.
pub struct Stft {
    fft: Arc<dyn rustfft::Fft<f32>>,
    window_size: usize,
    hop_size: usize,

    /// Reusable internal complex fft output buffer.
    window_buff: Vec<Complex<f32>>,
    /// Reusable internal fft scratch buffer.
    window_scratch: Vec<Complex<f32>>   
}

impl Stft {
    pub fn new(window_size: usize, hop_size: usize) -> Self {
        let fft = FftPlanner::<f32>::new().plan_fft_forward(window_size);
        let scratch_len = fft.get_inplace_scratch_len();

        Self {
            fft, window_size, hop_size,

            window_buff: vec![Complex::zero(); window_size],
            window_scratch: vec![Complex::zero(); scratch_len]
        }
    }

    /// Computes the short time fourier transform with hann window smoothing
    /// for `windows` windows starting from `offset`.
    pub fn compute(&mut self, audio_pwr: &[f32], offset: usize, windows: usize) -> Spectrogram {
        let mut fft_outputs = Vec::with_capacity(windows * self.fft.len());
        let unique_freqs = self.window_size / 2 + 1;
        
        // compute windowed fft for every window
        for i in 0..windows {
            fft_outputs.extend(
                self.windowed_fft(audio_pwr, offset + i * self.hop_size)
            );
        }
        return Spectrogram { powers: fft_outputs, freq_bins: unique_freqs };
    }

    /// Computes the fast fourier transform with hann window smoothing
    /// starting from `offset`.
    fn windowed_fft(&mut self, audio_pwr: &[f32], offset: usize) -> impl Iterator<Item = f32> + '_ {
        // apply window smoothing
        for (i, w) in apodize::hanning_iter(self.window_size).enumerate() {
            self.window_buff[i] = Complex::new(audio_pwr[offset + i] * w as f32, 0.0);
        }
    
        // compute fft
        self.fft.process_with_scratch(&mut self.window_buff, &mut self.window_scratch);

        return self.window_buff.iter().cloned()
            // ft result is symmetric, only first window_size / 2 + 1 samples are unique
            .take(self.window_size / 2 + 1)
            // square norm of complex fft output
            .map(|c| c.norm_sqr());
    } 
}

/// Saves the fourier transforms of the dataset in
/// `data/fouriers`.
fn serialize_transforms() {
    #[derive(Serialize)]
    struct Fourier {
        powers: Vec<f32>,
        freq_bins: usize,
        label: bool
    }

    // load stft
    let mut stft = Stft::new(WINDOW_SIZE, HOP_SIZE);

    // prepare files
    let paths = std::fs::read_dir("data/true").unwrap().map(|p| (p.unwrap().path(), true))
        .chain(
            std::fs::read_dir("data/false").unwrap().map(|p| (p.unwrap().path(), false))
        );

    for (i, (path, label)) in paths.enumerate() {
        let audio_pwr = audio::load(&path);

        let spectral = stft.compute(&audio_pwr, 0, MEAN_WINDOWS)
            .windows_mean();
            //.normalize_euclid();

        let json_str = json!(Fourier { powers: spectral.powers, freq_bins: spectral.freq_bins, label }).to_string();
        std::fs::write(format!("data/fouriers/{i}.json"), json_str).unwrap();
    }
}

/// Computes the stft spectrogram for each timestep with interval `lag`.
fn process_audio<P>(path: P, lag: usize) -> Vec<Spectrogram>
    where P: AsRef<Path>
{
    let mut stft = Stft::new(WINDOW_SIZE, HOP_SIZE);
    let audio_pwr = audio::load(&path);

    let mut spectrals = Vec::new();
    // number of time steps for which model input is generated
    let steps = (audio_pwr.len() - stft.window_size - stft.hop_size * MEAN_WINDOWS) / lag;
    for i in 0..steps {
        spectrals.push(
            stft.compute(&audio_pwr, i * lag, MEAN_WINDOWS)
                .windows_mean()
                // NOTE: euclid normalization works poorly for quiets
                //.normalize_euclid()
        );
    }
    return spectrals;
}

fn main() {
    let mut args = std::env::args();
    args.next();
    let op = args.next().unwrap();

    match op.as_str() {
        "process_dataset" => {
            serialize_transforms()
        }
        "process_audio" => {
            let in_path = args.next().unwrap();
            let out_path = args.next().unwrap();
    
            let spectrals = process_audio(in_path, HOP_SIZE);
            std::fs::write(out_path, json!(spectrals).to_string());
        }
        _ => {}
    }

    /*println!("cwd: {:?}", std::env::current_dir());
    
    let audio_pwr = audio::load(&"data/true/2841.wav".to_string());

    let mut stft = Stft::new(512, 256);
    let out = stft.compute(&audio_pwr, 0, 7).windows_mean().normalize_euclid();

    for (hz, pwr) in out.hz_power_iter(0, 48000, 0.0, 100000.0).collect::<Vec<_>>() {
        println!("{hz} : {pwr}");
    }*/
}
