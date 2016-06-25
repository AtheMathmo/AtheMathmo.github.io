---
layout: post
title: Using GMMs in Rust
excerpt: "What is a GMM? How can we use one in Rust?"
comments: true
---

This post aims to introduce Gaussian Mixture Models (from now on referred to as GMMs) and explain what they can be used for. To do that I'll be creating some synthetic data and training a GMM on that using [rusty-machine](https://github.com/AtheMathmo/rusty-machine). This post is fairly heavy on theory but I promise there is _some_ code.

Some familiarity with the following will be helpful (though not entirely necessary):

- Basic probability concepts
- Bayesian statistics
- Gaussian distributions
- Basic machine learning concepts

## What are you talking about?

Before jumping into GMMs let's define a more general Mixture Model. A Mixture Model is a probabilistic model used to represent subclasses within a whole population. We use Mixture Models to make sense of datasets that we believe are composed of a mixture of different groups. We want to learn the composition of the wider population from which our data has been drawn.

GMMs are a specific case of a Mixture Model which can be used when the population we are considering is made up of continuous measurements (real numbers). We attempt to model this population as a combination of [Gaussian random variables](https://en.wikipedia.org/wiki/Normal_distribution) each representing a sub-group. A more formal definition will follow the example.

#### A GMM Example

Imagine that we have a large room filled with males aged 10, 15 and 20. We'd expect the heights within each age group to be roughly normally distributed<sup>[*](#remarks)</sup> - with perhaps some overlap between the groups. This lends itself naturally to a GMM - we consider the whole population to be made up of sub groups with each having height normally distributed around some average.

What does all this mean in practice? We want to determine the _mean_ and _variance_ of these underlying Gaussians to learn something about our population. To do this we _train_ the GMM using the data that we have gathered. This involves using an algorithm - commonly Expectation Maximization<sup>[2](#references)</sup>.

## A bit more formally...

Mixture models tend to have some pretty hefty notation<sup>[1](#references)</sup>. I'll try to introduce things in a sensible order...

First we have `K` - the number of Gaussians we mix in the model; in our above example this was 3. And also `N`, the total number of samples.

For each of our `K` Gaussians we will have some _mean_ and _variance_, we'll denote those: <code class="highlighter-rouge">&micro;<sub>1..K</sub></code> and <code class="highlighter-rouge">&sigma;<sup>2</sup><sub>1..K</sub></code> , respectively.

And the last thing we'll need is the _mixture weights_, which we'll denote <code class="highlighter-rouge">&phi;<sub>1..K</sub></code> . Where <code class="highlighter-rouge">&phi;<sub>i</sub></code> is the (_prior_) probability that a sample belongs to sub group `i`. In many cases we may be content assuming each subcategory is equally likely.

Putting this back into comprehensible language: we have `K` Gaussian random variables which represent some subcategories in our data. And we have some belief (<code class="highlighter-rouge">&phi;</code>) of how often each subcategory appears in our data.

If you've kept up so far, great! Next I'm going to write about a great way to test a broad class of models.

## Using simulated data to test models

Suppose we've just finished writing some code that trains a GMM on a dataset. How can we be sure that our code works? We probably want to test our code out on some data that we believe contains subclasses that can be represented by Gaussians. We could try to find such a dataset - or we can create one!<sup>[**](#remarks)</sup>

This is a general technique that can be applied to a broad class of models. The plan is to construct some data which explicitly contains the properties described by our model. We then train the model on this dataset and hope to recover the properties as we defined them.

In the case of GMMs: we define some Gaussian random variables by choosing pairs of means and variances, (<code class="highlighter-rouge">&micro;<sub>k</sub>, &sigma;<sup>2</sup><sub>k</sub></code>). We also define the mixture probabilities for our model which define how much each Gaussian will contribute to the population as a whole. We then draw samples from the model by choosing a Gaussian (according to the mixture probabilities) and, in turn, drawing a sample from it.

By repeating this process we get a data set which should contain the properties defined by the means, variances, and mixture weights chosen. Now we train our GMM on the generated data and try to recover the means, variance and mixture weights we chose above.

## Finally, some code!

As promised, we'll be using Rust. In this section we'll walk through some code for simulating samples from a GMM. We'll then use [rusty-machine](https://github.com/AtheMathmo/rusty-machine) to train a GMM on this simulated data - and verify that we can learn the underlying model parameters.

```rust
use rusty_machine::stats::dist::gaussian::Gaussian;

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
        // We'll write this part next
    }

    out_samples
}
```

This is our function for generating a dataset with the properties of a GMM. We do some sensible length checking and then we fill a vector with Gaussians according to the incoming means and variances. Next we will be picking one of these Gaussians according to the mixture weights and taking a sample from it.

```rust
pub fn simulate_gmm_1d_data(n: usize, means: Vec<f64>,  vars: Vec<f64>, mixture_weights: Vec<f64>)-> Vec<f64> {
    // Setting up the Gaussians above...

    let mut rng = thread_rng();
    let mut out_samples = Vec::with_capacity(count);

    for _ in 0..n {
        // Pick a gaussian from the mixture weights
        let chosen_gaussian = gaussians[pick_gaussian_idx(&mixture_weights, &mut rng)];

        // Draw a sample from it
        use rand::dist::IndependentSample;
        let sample = chosen_gaussian.ind_sample(&mut rng);

        // Add to data
        out_samples.push(sample);
    }

    out_samples
}
```

And, just for good measure, here's the helper function `pick_gaussian_idx`:

```rust
use rand::Rng;

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
```

This function implements a standard method for generating samples from a discrete distribution.

### What is all of the above actually good for?

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

// Simulate some data using the parameters above
let samples = simulate_gmm_1d_data(count, means, vars, weights);

// Create a GMM with the same number of Gaussians
let mut gmm = GaussianMixtureModel::new(gmm_count);

// Train the model on the samples
gmm.train(&Matrix::new(count, 1, samples));

println!("Means = {:?}", gmm.means());
println!("Covs = {:?}", gmm.covariances());
println!("Mix Weights = {:?}", gmm.mixture_weights());
```
_[Check out the full source code](https://github.com/AtheMathmo/AtheMathmo.github.io/tree/master/assets/rust/gmm_simulation.rs)._

And if we run this code, we get the following output (after cleaning up a little):

```
Means = [-3.035455788753389, -0.04297417671919327, 2.985399452537148]
Covs = [0.8411235714344266, 0.4789527951624592, 0.25361125042304056]
Mix Weights = [0.4850802910597797, 0.2552003495199645, 0.2597193594202557]
```

These look like good estimates of the initial parameters! We could likely get better estimates by increasing the number of iterations, and the sample count (`N`).

Fortunately it looks like rusty-machine's implementation of GMMs is at least sort-of-sensible.<sup>[***](#remarks)</sup>

---

## Peeking under the hood (a grueling detour)

As [one last grueling detour](https://www.youtube.com/watch?v=QRJ38y4Jn6k) I'll talk briefly about what's happening under the hood.

As I mentioned above it is common to use the Expectation Maximization Algorithm<sup>[2](#references)</sup> to train GMMs. I'm not going to go into too much detail here but will touch on the basic steps.

The EM algorithm is split into two parts - an E-step and an M-step.

For GMMs the E-Step consists of computing the _posterior probability_ (called _membership weights_ in this case) for each data step lying in each class. This means - roughly - given our current estimates of the parameters and the data we have - how likely is it for each data point _i_ to be in subgroup _k_ (for each _i_ and _k_).

The M-Step involves using the _posterior probabilities_ computed above with the data to compute new parameter estimates. We compute new updates for the mixture weights, the means, and the variances. [These notes](http://www.ics.uci.edu/~smyth/courses/cs274/notes/EMnotes.pdf) provide a very good introduction and explanation.

The current rusty-machine implementation is fairly basic but does allow for some type-based-niceness. A common issue with GMMs is numerical instability around computing covariance inverses. The rusty-machine implementation uses a `CovOption` enum which allows the user to specify how the covariance updates should be computed.

```rust
pub enum CovOption {
    /// The full covariance structure.
    Full,
    /// Adds a regularization constant to the covariance diagonal.
    Regularized(f64),
    /// Only the diagonal covariance structure.
    Diagonal,
}
```

Though an exceedingly simple example this follows the general mandate of trying to keep rusty-machine's models simple (and obvious) whilst fully customisable.

---

### Remarks

Thanks for reading! Please give feedback on this post and [rusty-machine](https://github.com/AtheMathmo/rusty-machine). If you see some obvious improvements I'd love to hear them!

[*](#a-gmm-example) - I have absolutely no idea if they actually are. But it sounds sort of right?

[**](#using-simulated-data-to-test-models) - Of course it is also a good idea to test on real data _and_ test out some other edge cases.

[***](#peeking-under-the-hood-a-gruelling-detour) - Though, in my excitement to do machine learning I knowingly neglected some error handling among other things. :(


### References

[[1]](#a-bit-more-formally) - [Structure of a Mixture Model](https://en.wikipedia.org/wiki/Mixture_model#Structure_of_a_mixture_model). As you can see, there are a lot of parameters... In the definition above I've simplified a few things. 

[2] - [The EM Algorithm](http://www.ics.uci.edu/~smyth/courses/cs274/notes/EMnotes.pdf). These were the notes I used when writing the GMM model code.
