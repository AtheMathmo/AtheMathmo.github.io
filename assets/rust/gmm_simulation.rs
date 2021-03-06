extern crate rusty_machine;
extern crate rand;

use rusty_machine::linalg::Matrix;
use rusty_machine::learning::k_means::KMeansClassifier;
use rusty_machine::learning::UnSupModel;

use rand::{Rng, thread_rng};

fn generate_data(centroids: &Matrix<f64>,
                 points_per_centroid: usize,
                 noise_range: f64)
                 -> Matrix<f64> {
    assert!(centroids.cols() > 0, "Centroids cannot be empty.");
    assert!(centroids.rows() > 0, "Centroids cannot be empty.");
    assert!(noise_range >= 0f64, "Noise must be non-negative.");
    let mut raw_cluster_data = Vec::with_capacity(centroids.rows() * points_per_centroid *
                                                  centroids.cols());
    let mut rng = thread_rng();

    for _ in 0..points_per_centroid {
        // Generate points from each centroid
        for centroid in centroids.iter_rows() {
            // Generate a point randomly around the centroid
            let mut point = Vec::with_capacity(centroids.cols());
            for feature in centroid {
                point.push(feature + rng.gen_range(-noise_range, noise_range));
            }

            // Push point to raw_cluster_data
            raw_cluster_data.extend(point);
        }
    }

    Matrix::new(centroids.rows() * points_per_centroid,
                centroids.cols(),
                raw_cluster_data)
}

fn main() {
    println!("K-Means clustering example:");

    const SAMPLES_PER_CENTROID: usize = 2000;

    println!("Generating {0} samples from each centroids:",
             SAMPLES_PER_CENTROID);
    // Choose two cluster centers, at (-0.5, -0.5) and (0, 0.5).
    let centroids = Matrix::new(2, 2, vec![-0.5, -0.5, 0.0, 0.5]);
    println!("{}", centroids);

    // Generate some data randomly around the centroids
    let samples = generate_data(&centroids, SAMPLES_PER_CENTROID, 0.4);

    // Create a new model with 2 clusters
    let mut model = KMeansClassifier::new(2);

    println!("Training the model...");
    // Train the model
    model.train(&samples);

    let centroids = model.centroids().as_ref().unwrap();
    println!("Model Centroids:\n{:.3}", centroids);

    // Predict the classes and partition into
    println!("Classifying the samples...");
    let classes = model.predict(&samples);
    let (first, second): (Vec<usize>, Vec<usize>) = classes.data().iter().partition(|&x| *x == 0);

    println!("Samples closest to first centroid: {}", first.len());
    println!("Samples closest to second centroid: {}", second.len());
}
