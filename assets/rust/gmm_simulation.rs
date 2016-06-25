extern crate rusty_machine as rm;
extern crate rand;

use rm::stats::dist::gaussian::Gaussian;
use rm::learning::gmm::GaussianMixtureModel;
use rm::prelude::*;

use rand::distributions::IndependentSample;
use rand::{Rng, thread_rng};

pub fn simulate_gmm_1d_data(count: usize,
                            means: Vec<f64>,
                            vars: Vec<f64>,
                            mixture_weights: Vec<f64>)
                            -> Vec<f64> {
    assert_eq!(means.len(), vars.len());
    assert_eq!(means.len(), mixture_weights.len());

    let gmm_count = means.len();

    let mut gaussians = Vec::with_capacity(gmm_count);

    for i in 0..gmm_count {
        // Create a gaussian with mean and var
        gaussians.push(Gaussian::new(means[i], vars[i]));
    }

    let mut rng = thread_rng();
    let mut out_samples = Vec::with_capacity(count);

    for _ in 0..count {
        // Pick a gaussian from the mixture weights
        let chosen_gaussian = gaussians[pick_gaussian_idx(&mixture_weights, &mut rng)];

        // Draw a sample from it
        let sample = chosen_gaussian.ind_sample(&mut rng);

        // Add to data
        out_samples.push(sample);
    }

    out_samples
}

fn pick_gaussian_idx<R: Rng>(unnorm_dist: &[f64], rng: &mut R) -> usize {
    assert!(unnorm_dist.len() > 0);

    // Get the sum of the unnormalized distribution
    let sum = unnorm_dist.iter().fold(0f64, |acc, &x| acc + x);

    // Sum must be positive
    assert!(sum > 0);
    
    // A random number between 0 and sum
    let rand = rng.gen_range(0.0f64, sum);

    let mut unnorm_pmf = 0.0;
    for (i, p) in unnorm_dist.iter().enumerate() {
        // Add the current probability to the pmf
        unnorm_pmf += *p;

        // Return i if rand falls in the correct interval
        if rand < unnorm_pmf {
            return i;
        }
    }

    panic!("No random value was sampled!");
}

fn main() {
    // Number of Gaussians and total samples
    let gmm_count = 3;
    let data_count = 1000;

    // Parameters for our model
    let means = vec![-3.0, 0.0, 3.0];
    let vars = vec![1.0, 0.5, 0.25];
    let weights = vec![0.5, 0.25, 0.25];

    // Simulate some data using the parameters above
    let samples: Vec<f64> = simulate_gmm_1d_data(data_count, means, vars, weights);

    // Create a GMM with the same number of Gaussians
    let mut gmm = GaussianMixtureModel::new(gmm_count);

    // Train the model on the samples
    gmm.train(&Matrix::new(data_count, 1, samples));

    println!("Means = {:?}", gmm.means());
    println!("Covs = {:?}", gmm.covariances());
    println!("Mix Weights = {:?}", gmm.mixture_weights());
}
