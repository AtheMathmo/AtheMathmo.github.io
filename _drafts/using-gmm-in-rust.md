---
layout: post
title: Using GMMs in Rust
excerpt: "What is a GMM? How can we use one in Rust?"
comments: true
---

This post aims to introduce Gaussian Mixture Models (from now on referred to as GMMs) and explain what they can be used for. To do that I'll be creating some synthetic data and training a GMM on them.

I'll be assuming some familiarity with the following:

- Basic probability concepts
- Bayesian statistics
- Gaussian distributions
- Basic machine learning concepts

This post follows a similar vein to the one on [Naive Bayes Classifiers](/2016/04/08/naive-bayes-rusty-machine.html). It may also be helpful to read my [first post on rusty-machine](/2016/03/07/rusty-machine.html).

## What are you talking about?

Before jumping into GMMs let's define general Mixture Models. A Mixture Model is a probabilistic model used to represent subclasses within a whole population. We use Mixture Models to make sense of datasets that we believe are composed of a mixture of different groups. We want to learn the composition of the wider population from which the samples were drawn.

GMMs can be used when the population we are considering is a dataset made up of continuous measurements (real numbers). We attempt to model the wider population from which the samples were drawn as a combination of Gaussian random variables. For example, imagine that we have an infinitely large room filled with males aged 10, 15 and 20. We could reasonably use a GMM to model this. We'd expect the heights of each age group to be roughly normally distributed. And so to model the entire population we would use a mixture of height distributions of each group.

We _train_ the GMM using some data that we have gathered - this means determining the best model parameters to describe the data. This involves using an algorithm - generally Expectation Maximization<sup>[2](#references)</sup>.

## A bit more formally...

Ok, mixture models tend to have some pretty hefty notation<sup>[1](#references)</sup>. I'll try to introduce things in a sensible order - take solace in the fact that you _probably_ don't need to understand this section.

First we have `K` - the number of Gaussians we mix in the model. And `N` be the total number of samples.

For each of our `K` Gaussian's we will have some mean and variance, we'll denote those: <code class="highlighter-rouge">&micro;<sub>1..K</sub></code> and <code class="highlighter-rouge">&sigma;<sup>2</sup><sub>1..K</sub></code> , respectively.

And the last thing we'll need is the mixture weights, which we'll denote <code class="highlighter-rouge">&phi;<sub>1..K</sub></code> . Where <code class="highlighter-rouge">&phi;<sub>i</sub></code> is the (prior) probability that a sample belongs to sub group `i`.

Putting this back into comprehensible language: we have `K` Gaussian random variables which represent some subcategories in our data. And we have some belief (<code class="highlighter-rouge">&phi;</code>) of how often each subcategory appears in our data.

If you've kept up so far, great! Next I'm going to write about a great way to test probabilistic models (and other more general models).

## Using simulated data to test models

This is a general technique we can use to validate our models. It's often a valuable tool when debugging model code and is also a great way to learn about the model.

We simulate some data using some model assumptions. So in the case of GMMs: we define some Gaussian random variables. We define the mixture probabilities for our model. We then draw samples from the model by choosing a Gaussian (according to the mixture probabilities) and then drawing a sample from it.

By repeating this process we get a data set which is indicative of the population defined by our model. The hope is that by training a GMM on this data set we should be able to recover the parameters used to generate our data.

## Finally, some code!

As promised, we'll be using Rust. In this section we'll walk through some code for simulating samples from a GMM. We'll then use [rusty-machine](https://github.com/AtheMathmo/rusty-machine) to train a GMM on this simulated data - and verify that we can learn the underlying model parameters.

```rust
pub fn simulate_gmm_1d_data(n: usize, means: Vec<f64>,  vars: Vec<f64>, mixture_weights: Vec<f64>)-> Vec<f64> {
	assert_eq!(means.len(), vars.len());
	assert_eq!(means.len(), mixture_weights.len());

	let gmm_count = means.len();

	// More to come...
}
```

This is the start of our data simulation function. We take in the various model parameters and check that they are the same length (`K`).

Rusty-machine has some built in support for Gaussian random variables, so we can add the following:

```rust
pub fn simulate_gmm_1d_data(n: usize, means: Vec<f64>,  vars: Vec<f64>, mixture_weights: Vec<f64>)-> Vec<f64> {
	// Checking lengths above...

	let gmm_count = means.len();
	let mut gaussians = Vec::with_capacity(gmm_count);

	use rusty_machine::stats::dist::gaussian::Gaussian;
	for i in 0..gmm_count {
		// Create a gaussian with mean and var
		gaussians.push(Gaussian::new(means[i], vars[i]));
	}

	// Sampling the right Gaussians
	use rand::thread_rng;
	let mut rng = thread_rng();
	let mut out_samples = Vec::with_capacity(n);

	for _ in 0..n {
		// We'll write this next
	}

	out_samples
}
```

Here we fill a vector with the Gaussians specified by the input arguments. Inside the last loop we need to pick the Gaussian we're sampling from based on the mixture weights. And then draw the sample. 

```rust
pub fn simulate_gmm_1d_data(n: usize, means: Vec<f64>,  vars: Vec<f64>, mixture_weights: Vec<f64>)-> Vec<f64> {
	// Setting up the Gaussians above...

	for _ in 0..n {
		// Pick a gaussian from the mixture weights
		let chosen_gaussian = gaussians[pick_gaussian(&mixture_weights, &mut rng)];

		// Draw sample from it
		use rand::dist::IndependentSample;
		let sample = chosen_gaussian.ind_sample(&mut rng);

		// Add to data
		out_samples.push(sample);
	}

	out_samples
}
```

And here's the helper function `pick_gaussian`:

```rust
use rand::Rng;

fn pick_gaussian<R: Rng>(unnorm_dist: &[f64], rng: &mut R) -> usize {
    assert!(unnorm_dist.len() > 0);

    // Get the sum of the unnormalized distribution
    let sum = unnorm_dist.iter().fold(0f64, |acc, &x| acc + x);

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
```

This function implements a standard method for generating samples from a discrete distribution.

### So what is all of the above actually good for?

Using the code above we can generate some samples from a GMM. We can then train a GMM using rusty-machine on the generated samples. The hope is that - we should be able to recover the original parameters that generated the samples (the mean, variance, and mixture probabilities).

Luckily, rusty-machine makes training a GMM very easy!

```rust
use rusty_machine::learning::gmm::GaussianMixtureModel;
use rusty_machine::prelude::*;

// Number of Gaussians and Samples
let gmm_count = 3;
let count = 1000;

// Parameters for our model
let means = vec![-3f64, 0., 3.];
let vars = vec![1f64, 0.5, 0.25];
let weights = vec![0.5, 0.25, 0.25];

// Simulate some samples used the parameters
let samples = simulate_gmm_1d_data(count, means, vars, weights);

// Create a GMM with the same number of Gaussians
let mut gmm = GaussianMixtureModel::new(gmm_count);

// Train the model on the samples
gmm.train(&Matrix::new(count, 1, samples));

println!("Means = {:?}", gmm.means());
println!("Covs = {:?}", gmm.covariances());
println!("Mix Weights = {:?}", gmm.mixture_weights());
```
And if we run this code, we get the following output (after cleaning up a little):

```
Means = [-3.035455788753389, -0.04297417671919327, 2.985399452537148]
Covs = [0.8411235714344266, 0.4789527951624592, 0.25361125042304056]
Mix Weights = [0.4850802910597797, 0.2552003495199645, 0.2597193594202557]
```

These look like good estimates of the initial parameters! We could likely get better estimates by increasing the number of iterations, and the sample count (`N`).

[Check out the full source code](https://github.com/AtheMathmo/AtheMathmo.github.io/tree/master/assets/rust/gmm_simulation.rs).

## Peeking under the hood

- We train the model using the Expectation Maximization Algorithm<sup>[2](#references)</sup>
- How we write the EM algorithm in Rust

---

### References

[1] - [Structure of a Mixture Model](https://en.wikipedia.org/wiki/Mixture_model#Structure_of_a_mixture_model). As you can see, there are a lot of parameters... In the definition above I've simplified a few things. 

[2] - [The EM Algorithm](http://www.ics.uci.edu/~smyth/courses/cs274/notes/EMnotes.pdf). These were the notes I used when writing the GMM model code.
