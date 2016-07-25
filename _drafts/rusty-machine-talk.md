---
layout: slide
title: Rusty-machine talk from SF Meetup
excerpt: Talking about Rust and machine learning with rusty-machine
theme: night
transition: slide
---

<section data-markdown>
# Rusty-machine

## James Lucas
</section>

<section data-markdown>
## This talk

- What is machine learning?
- How does rusty-machine work?
- Why is rusty-machine great?

</section>

<section>
<h2>What is rusty-machine?</h2>

<p><a href="https://github.com/AtheMathmo/rusty-machine">Rusty-machine</a> is a machine learning library written entirely in Rust.</p>

<p class="fragment" data-fragment-index="1">It focuses on the following:</p>

<ul class="fragment" data-fragment-index="1">
    <li>Works out-of-the-box without relying on external dependencies.</li>
    <li>Simple and easy to understand API.</li>
    <li>Extendible and easy to configure.</li>
</ul>

<aside class="notes">
Installing machine learning libraries can often be made a pain if we also <b>need</b> to install, BLAS, LAPACK, CUDA, and more. Especially for new users.

Try and keep things modular and reuse the API across all models. Some examples of this later.
</aside>
</section>

<section data-markdown>
## Another machine learning library?

Note:
- Machine learning is already in every other language, multiple times each. Are we just rewriting stuff?
- Rusty-machine is more than deep learning.
- Rust is a good choice: it seemed like it would be rewarding to explore.

</section>

<section data-markdown>
## Machine Learning

> "Field of study that gives computers the ability to learn without being explicitly programmed." - Arthur Samuel

Note:
We'll walk through some basic concepts in machine learning that help us to understand why rusty-machine is built as it is.
</section>

<section>
<h2>How do machines learn?</h2>

<p class="fragment fade-up">With data.</p>
</section>

<section>
<h2>Some examples</h2>

<ul>
    <li class="fragment down" data-fragment-index="1">Predicting whether an image contains a cat or a dog</li>
    <li class="fragment down" data-fragment-index="3">Predicting house prices given related data</li>
    <li class="fragment down" data-fragment-index="5">Separating two sources of audio from a single track</li>
</ul>

<br><br><br><br><br><br><br><br>
</section>

<!-- Skipping the slides about learning types..
<section>

In machine learning we have <b>Supervised</b> and <b>Unsupervised</b> learning.

<p class="fragment fade-up" data-fragment-index="1">Supervised - We have labelled <i>input data</i></p>
<p class="fragment fade-up" data-fragment-index="3">Unsupervised - We have unlabelled <i>input data</i></p>

<span style="position:relative; height: 800px;">
    <span class="fragment fade-up" data-fragment-index="2" style="position: absolute; display: block; height: 400px; width: 800px;">
        <span class="fragment fade-out" data-fragment-index="3">
            <img src="{{ site.url }}/assets/cat-in-suit.jpg" style="border-radius: 20px;" >
        </span>
    </span>
    <span class="fragment fade-up" data-fragment-index="4" style="position: absolute; display: block; height: 400px; width: 800px;">
        <img src="{{ site.url }}/assets/dog-headphones.jpg" style="border-radius: 20px;" >
    </span>
</span>

<br><br><br><br><br><br><br><br>

<aside class="notes">
Supervised example - We have pictures of cats and dogs which are labelled as such. Maybe we want to teach the machine to identify new cats and dogs from just their pictures.

UnSupervised example - We have just the pictures without labels. Maybe we want the machine to separate the set of photos in two groups.

We also have some others. Semi-Supervised, Reinforcement - we wont go into these.
</aside>
</section>
 -->

<section>
<h2>Some terminology</h2>

<ul>
<li><b>Model</b> : An object that transforms <i>inputs</i> into <i>outputs</i> based on information in data.</li>
<li><b>Train/Fit</b> : Teaching a model how it should transform <i>inputs</i> using data.</li>
<li><b>Predict</b> : Feeding <i>inputs</i> into a model to receive <i>outputs</i>.</li>
</ul>
<br><br>
<p class="fragment">To predict house prices we may use a <i>Linear Regression</i> <b>Model</b>. We'd <b>train</b> the model on some house prices and facts about the houses. Then we'd <b>predict</b> the price of houses we do not yet know.</p>

<aside class="notes">
There is a _lot_ of terminology in ML. This is just a handful of things I'll use going forwards.

In the example I've used the terminology to illustrate a little more clearly what each means.

We've now got a very basic idea of what machine learning is - so let's start talking about rusty-machine!
</aside>

</section>


<section data-markdown>

## Back to rusty-machine

</section>

<!-- For some reason we must use an explicit code block somewhere for the highlighter to work with markdown... -->
<section>
<h2>The base of rusty-machine</h2>
<pre><code class="hljs rust">
pub mod learning {
    pub trait Model<T, U> {
        fn train(&mut self, inputs: &T, targets: &U);

        fn predict(&self, inputs: &T) -> U;
    }
}
</code></pre>

<aside class="notes">
This trait is used to represent a model.

It is simplified a little from the actual traits used.
</aside>
</section>

<section data-markdown>
## An example

Before we go any further we should see an example.
</section>

<section>
<h2>K-Means</h2>

<p>A model for <i>clustering</i>.</p>

<img src="{{ site.url }}/assets/k_means_samples.jpg" class="stretch" style="border-radius: 20px;">

</section>

<section>

<section data-markdown>
## Using a K-Means Model

```
// ... Get the data samples

// Create a new model with 2 clusters
let mut model = KMeansClassifier::new(2);

// Train the model
model.train(&samples);

// Get the model centroids (the center of each cluster)
let centroids : Matrix&lt;T> = model.centroids().as_ref().unwrap();

// Predict which cluster each point belongs to
println!("Classifying the samples...");
let classes = model.predict(&samples);
```

_You can run the full example in the [rusty-machine repo](https://github.com/AtheMathmo/rusty-machine/tree/master/examples)._

</section>

<section data-markdown>
## Under the hood

K-Means works in roughly the following way:

1. Get some initial guesses for the centroids (cluster centers)
2. Assign each point to the centroid it is closest to.
3. Update the centroids by taking the average of all points assigned to it.
4. Repeat 2 and 3 until convergence.

</section>
</section>

<section>

<h2>K-Means Classification</h2>

<img src="{{ site.url }}/assets/k_means_samples_with_model.jpg" class="stretch" style="border-radius: 20px;">

</section>

<section>
<h2>Simple but complicated</h2>

<p>The API for other models aim to be as simple as that one. However... <div class="fragment">Machine learning is complicated.</div></p>

<p class="fragment">Rusty-machine aims for ease of use.</p>

<aside class="notes">
There are lots of different ways to train models and on top of that many ways to configure and adapt them.
</aside>

</section>

<section data-markdown>
## Using traits

- A clean, simple model API
- Extensibility at the user level
- Reusable components within the library

Note:
As seen before, rusty-machine uses traits as its foundation.

In Rust a trait defines an interface - a set of functions which the implementor should define.

</section>

<section data-markdown>
## Extensibility

We use traits to define parts of the models.

Users can swap in different implementations from within rusty-machine.

Users can write their own implementations and plug them in.

</section>

<section>
<section>
<h2>Extensibility Example</h2>
<h4>Support Vector Machine</h4>

<pre class="fragment"><code class="hljs rust">
pub struct SVM&lt;K: Kernel> {
    ker: K,
    /// Number of iterations for training.
    pub optim_iters: usize,
    /* ... */
}
</code></pre>

<pre class="fragment"><code class="hljs rust">
pub trait Kernel {
    /// The kernel function.
    ///
    /// Takes two equal length slices and returns a scalar.
    fn kernel(&amp;self, x1: &amp;[f64], x2: &amp;[f64]) -> f64;
}
</code></pre>
</section>

<section>
<h2>Combining kernels</h2>

(K<sub>1</sub> + K<sub>2</sub>)(x<sub>1</sub>, x<sub>2</sub>) = K<sub>1</sub>(x<sub>1</sub>, x<sub>2</sub>) + K<sub>2</sub>(x<sub>1</sub>, x<sub>2</sub>)

<pre class="fragment"><code class="hljs rust">
pub struct KernelSum&lt;T, U>
    where T: Kernel,
          U: Kernel
{
    k1: T,
    k2: U,
}
</code></pre>

<pre class="fragment"><code class="hljs rust">
/// Computes the sum of the two associated kernels.
impl&lt;T, U> Kernel for KernelSum&lt;T, U>
    where T: Kernel,
          U: Kernel
{
    fn kernel(&amp;self, x1: &amp;[f64], x2: &amp;[f64]) -> f64 {
        self.k1.kernel(x1, x2) + self.k2.kernel(x1, x2)
    }
}
</code></pre>

</section>
</section>

<section data-markdown>
## Reusability

We use traits to define common components, e.g. _Gradient Descent Solvers_.

These components can be swapped in and out of models.

New models can easily make use of these common components.

Note:
And of course - users can write their own versions of such components as in previous slide.

</section>

<section data-markdown>
## Reusability Example

All _Gradient Descent Solvers_ implement this trait.

```
/// Trait for optimization algorithms. (Some things omitted)
pub trait OptimAlgorithm&lt;M> {
    /// Return the optimized parameter using gradient optimization.
    fn optimize(&amp;self, model: &amp;M, ...) -> Vec&lt;f64>;
}
```

This means they can be reused across multiple models.

</section>

<section data-markdown>
## Error Handling

Though I've not made good use of it - Rust's error handling is fantastic.

```rust
impl Matrix&lt;T> {
    pub fn inverse(&amp;self) -> Result&lt;Matrix&lt;T>, Error> {
        // Fun stuff goes here
    }
}

```
</section>

<section data-markdown>

## What can rusty-machine do?

- K-Means Clustering
- Linear Regression
- Logistic Regression
- Generalized Linear Models
- Neural Networks
- Gaussian Process Regression
- Support Vector Machines
- Gaussian Mixture Models
- Naive Bayes Classifiers

</section>

<section data-markdown>

## Linear Algebra - [Rulinalg](https://github.com/AtheMathmo/rulinalg)

Rusty-machine works without any external dependencies.

Rulinalg provides linear algebra implemented entirely in Rust.

</section>

<section data-markdown>

## What does Rulinalg do?

- Data structures (`Matrix`, `Vector`)
- Basic operators (with in-place allocation where possible)
- Decompositions (Inverse, Eigendecomp, SVD, etc.)
- And more...


</section>

<section data-markdown>
## Why is Rust a good choice?

- Trait system is amazing.
- Error handling is amazing.
- Performance focused code without relying on heavy dependencies*.
- (Historically we prototype in high level languages and then rewrite performance critical parts.)

\* Not so performant right now, but the future looks bright!

Note:
Traits - Clean, extensible, homogenous API.
Performance - A bold claim right now... But the potential is there for us to prototype and achieve high performance code in the same environment.
Insights - More from a developers points of view; it is useful to have to think about how the model should be structured. What data does it need to own, which parts can be made modular without adding unneeded complexity, etc.
</section>

<section data-markdown>
## Why is Rust a good choice?

Most importantly for me - we gain more control over data.

Note:
We choose when a model needs ownership. When to allocate new memory for operations. These are things
that are much harder to achieve in other languages as pleasant-to-use as Rust.

</section>

<section data-markdown>
## When would you use rusty-machine?

At the moment - experimentation, non-performance critical applications.

In the future - quick, extensible modeling.

Note:
For now it would be unwise to use this for anything serious. Except maybe if the benefits of Rust outweigh performance and accuracy.

In the future, rusty-machine will try to enable rapid prototyping that can be easily extended into a finished product.

</section>

<section>
<h2>What's next?</h2>
<ul>
<li class="fragment fade-up">Optimizing and stablizing existing models.</li>
<li class="fragment fade-up">Providing optional use of BLAS/LAPACK/CUDA/etc.</li>
</ul>

</section>

<section data-markdown>
## What would I like to see from Rust?

- Specialization
- Growth of Float/Complex generics

</section>

<section data-markdown>
## Summary

- Machine learning (done quickly)
- Rusty-machine
- Rulinalg

</section>

<section data-markdown>
## Thanks!

#### Some Links

- [Rusty-machine](https://github.com/AtheMathmo/rusty-machine)
- [My Blog](http://athemathmo.github.io/)

</section>

<section>

<section data-markdown>
## Some FAQs

</section>

<section>
<h2>Why no GPU support</h2>

<img src="{{ site.url }}/assets/why-no-gpu.png" style="border-radius: 20px;" >

<p>From Scikit-learn's FAQs.</p>

<aside class="notes">
I do think it is worth having, but the concerns expressed in this
excerpt from Scikit learn are valid.
</aside>
</section>

<section data-markdown>

## BLAS/LAPACK

Hopefully soon!

</section>

<section data-markdown>

## Integrating with other languages

Nothing planned yet, but some good choices.

Python is especially exciting.

</section>

</section>


