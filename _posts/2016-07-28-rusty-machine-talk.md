---
layout: slide
title: Rusty-machine talk from SF Meetup
excerpt: Talking about Rust and machine learning with rusty-machine. (Press 's' for notes).
theme: night
transition: slide
---

<section data-markdown>
# Rusty-machine

## James Lucas

Note:
Disclaimer: I'm a mathematician by training so things may get heavy.
I'll do my best to explain but please interrupt me if I'm not making sense.
</section>

<section data-markdown>
## This talk

- What is machine learning?
- How does rusty-machine work?
- Why is rusty-machine great?

</section>

<section>
<h2>What is rusty-machine?</h2>

<p><a href="https://github.com/AtheMathmo/rusty-machine">Rusty-machine</a> is a machine learning library written <b>entirely</b> in Rust.</p>

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
    <li class="fragment fade-down" data-fragment-index="1">Predicting rent increase</li>
    <li class="fragment fade-down" data-fragment-index="3">Predicting whether an image contains a cat or a dog</li>
    <li class="fragment fade-down" data-fragment-index="5">Understanding hand written digits</li>
</ul>

<br><br>

<span style="position:relative; height: 800px;">
    <h4 class="fragment fade-up" data-fragment-index="2">Data set might be:</h4>
    <span class="fragment fade-up" data-fragment-index="2" style="position: absolute; display: block; height: 400px; width: 800px;">
            <p class="fragment fade-out" data-fragment-index="3">rent prices and other facts about the residence.</p>
    </span>
    <span class="fragment fade-up" data-fragment-index="4" style="position: absolute; display: block; height: 400px; width: 800px;">
        <p class="fragment fade-out" data-fragment-index="5">
            <i>labelled</i> pictures of cats and dogs.
        </p>
    </span>
    <p class="fragment fade-up" data-fragment-index="6" style="position: absolute; display: block; height: 400px; width: 800px;">
        many examples of hand written digits.
    </p>
</span>

<br><br>

<aside class="notes">
Define the problem first - then the data - then how machine learning could solve it.

For the second problem - imagine you want to predict what your rent will be when you renew your lease.
You have data from craigs list of the rent listings in your neighbourhood. And data from some ordinance
service with facts like, windows on each apartment, # chimneys, sq footage, etc. You want to use this data
to predict what your rent will be.
</aside>

</section>

<section>
<h2>Some terminology</h2>

<ul>
<li class="fragment"><b>Model</b> : An object that transforms <i>inputs</i> into <i>outputs</i> based on information in data.</li>
<li class="fragment"><b>Train/Fit</b> : Teaching a model how it should transform <i>inputs</i> using data.</li>
<li class="fragment"><b>Predict</b> : Feeding <i>inputs</i> into a model to receive <i>outputs</i>.</li>
</ul>
<br><br>
<p class="fragment">To predict rent increases we may use a <i>Linear Regression</i> <b>Model</b>. We'd <b>train</b>
the model on some rent prices and facts about the residence. Then we'd <b>predict</b> the rent of unlisted places.</p>

<aside class="notes">
There is a _lot_ of terminology in ML. This is just a handful of things I'll use going forwards.

In the example I've used the terminology to illustrate a little more clearly what each means.

We've now got a very basic idea of what machine learning is - so let's start talking about rusty-machine!
</aside>

</section>

<section>
<h2>Why is machine learning hard?</h2>

<p class="fragment">There are many, many models to choose from.</p>

<p class="fragment">There are many, many ways to use each model.</p>

<aside class="notes">
Machine learning is inherently difficult - those described here certainly aren't the only challenges.

Rusty-machine doesn't so much try to solve these problems. Instead it aims to make it easy
to navigate the solutions yourself.
</aside>
</section>

<section data-markdown>

## Back to rusty-machine

</section>

<!-- For some reason we must use an explicit code block somewhere for the highlighter to work with markdown... -->
<section>
<h2>The foundation of rusty-machine</h2>
<pre><code class="hljs rust">
pub trait Model<T, U> {
    fn train(&mut self, inputs: &T, targets: &U);

    fn predict(&self, inputs: &T) -> U;
}
</code></pre>

<aside class="notes">
In Rust a trait defines an interface - a set of functions which the implementor should define.

This trait is used to represent a model. It is simplified a little from the actual traits used.
</aside>
</section>

<section data-markdown>
## An example

Before we go any further we should see an example.

Note:
The example will show how we use these functions from the Model trait.
</section>

<section>
<h2>K-Means</h2>

<p>A model for <i>clustering</i>.</p>

<img src="{{ site.url }}/assets/k_means_samples.jpg" class="stretch" style="border-radius: 20px;">

<aside class="notes">
Clustering is essentially grouping together similar items. Where <i>similar</i> may mean close together in space, or share similar features, etc.
</aside>

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

// Predict which cluster each point belongs to
let clusters : Vector&lt;usize> = model.predict(&samples);
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

<img src="{{ site.url }}/assets/k_means_classified_samples.jpg" class="stretch" style="border-radius: 20px;">

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
## How does rusty-machine (try to) keep things simple?
</section>

<section data-markdown>
## Using traits

- A clean, simple model API
- Extensibility at the user level
- Reusable components within the library

Note:
As seen before, rusty-machine uses the `Model` trait as its foundation.
This is the primary way we keep things clean and simple.

We use traits to try and _hide_ as much of the machine learning complexity as possible.
This is while keeping it in reach for users who need it.

</section>

<section data-markdown>
## Extensibility

We use traits to define parts of the models.

While rusty-machine provides common defaults - users can write their own implementations and plug them in.

</section>

<section>
<section>
<h2>Extensibility Example</h2>
<h4>Support Vector Machine</h4>

<pre class="fragment"><code class="hljs rust">
/// A Support Vector Machine
pub struct SVM&lt;K: Kernel> {
    ker: K,
    /// Some other fields
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

<aside class="notes">
An SVM is a model which is generally used for classification. The behaviour of the SVM is governed by a <i>kernel</i>.
A kernel is essentially a function which obeys some properties (which I won't go into here,
there are good resources online).

Here we allow the kernel to be generic while providing some sensible defaults.

This is accessible in other languages but Rust helps us enforce this with the compiler.
</aside>
</section>

<section>
<h2>Combining kernels</h2>

K<sub>1</sub>(x<sub>1</sub>, x<sub>2</sub>) + K<sub>2</sub>(x<sub>1</sub>, x<sub>2</sub>) = K(x<sub>1</sub>, x<sub>2</sub>)

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

<aside class="notes">
One property of kernels is that the sum of two kernels is also a kernel.
i.e. K on the right also has all the properties of a kernel itself.

We can override the `Add` trait to allow complex combinations of kernels.
</aside>
</section>

<section>
<h2>Combining kernels</h2>

K<sub>1</sub>(x<sub>1</sub>, x<sub>2</sub>) + K<sub>2</sub>(x<sub>1</sub>, x<sub>2</sub>) = K(x<sub>1</sub>, x<sub>2</sub>)

<pre><code class="hljs rust">
let poly_ker = kernel::Polynomial::new(...);
let hypert_ker = kernel::HyperTan::new(...);

let sum_kernel = poly_ker + hypert_ker;

let mut model = SVM::new(sum_kernel);
</code></pre>

<aside class="notes">
We can override the `Add` trait to allow complex combinations of kernels.
</aside>
</section>
</section>

<section>
<h2>Reusability</h2>

<p>We use traits to define common components, e.g. <i>Kernels</i>.</p>

<p class="fragment" data-fragment-index="1">These components can be swapped in and out of models.</p>

<p class="fragment" data-fragment-index="1">New models can easily make use of these common components.</p>

<aside class="notes">
Similar to Extensibility - but by this I mean we can move common components across different models.
Of course this is possible with other languages and frameworks but Rust helps us do this while enforcing
the requirements with the compiler.

For example - in other languages how can we be sure that the kernel function won't consume the input data?
</aside>

</section>

<section>
<h2>Reusability Example</h2>
<h4>Gradient Descent Solvers</h4>

<p>We use Gradient Descent to minimize a <i>cost</i> function.</p>

<span class="fragment">
All <i>Gradient Descent Solvers</i> implement this trait.

<pre><code class="hljs rust">
/// Trait for gradient descent algorithms. (Some things omitted)
pub trait OptimAlgorithm&lt;M: Optimizable> {
    /// Return the optimized parameters using gradient optimization.
    fn optimize(&amp;self, model: &amp;M, ...) -> Vec&lt;f64>;
}
</code></pre>
</span>

<p class="fragment">The <b>Optimizable</b> trait is implemented by a model which is differentiable.</p>


<aside class="notes">
Our models have a cost function - e.g. for predicting rent our cost might be the squared distance
between our models estimate and the actual value. When our cost function is differentiable we can
use gradient descent.

The idea is that by taking steps down the steepest slope we get closer to the minimum cost.

The OptimAlgorithm trait specifies how we shall do this downward stepping towards the minimum.
The Optimizable trait specifies how the derivative of the cost function will be computed.
</aside>
</section>

<section>
<h2>Creating a new model</h2>
<h3>With gradient descent optimization</h3>

<p class="fragment" data-fragment-index="1">Define the model.</p>
<pre class="fragment" data-fragment-index="2"><code class="hljs rust">
/// Cost function is: f(x) = (x-c)^2
struct XSqModel {
    c: f64,
}
</code></pre>

<p class="fragment" data-fragment-index="3">You can think of this model as <i>learning</i> the value <b>c</b>.</p>

<aside class="notes">
The bulk of the work will be in step 2 - which is where we compute the gradient of the model.
</aside>

</section>

<section>
<h2>Creating a new model</h2>
<h3>With gradient descent optimization</h3>

<p class="fragment">Implement <b>Optimizable</b> for model.</p>

<pre class="fragment" ><code class="hljs rust">
/// Cost function is: f(x) = (x-c)^2
struct XSqModel {
    c: f64,
}

impl Optimizable for XSqModel {
    /// 'params' here is 'x'
    fn compute_grad(&amp;self, params: &amp;[f64], ...) -> Vec&lt;f64> {
         vec![2f64 * (params[0] - self.c)]
    }
}
</code></pre>

</section>

<section>
<h2>Creating a new model</h2>
<h3>With gradient descent optimization</h3>

<p class="fragment">Use an <b>OptimAlgorithm</b> to compute the optimized parameters.</p>

<pre class="fragment" ><code class="hljs rust">
/// Cost function is: f(x) = (x-c)^2
struct XSqModel {
    c: f64,
}

impl Optimizable for XSqModel {
    fn compute_grad(&amp;self, params: &amp;[f64], ...) -> Vec&lt;f64> {
         vec![2f64 * (params[0] - self.c)]
    }
}

let x_sq = XSqModel { c : 1.0 };
let x_start = vec![30.0];
let gd = GradientDesc::default();
let optimal = gd.optimize(&amp;x_sq, &amp;x_start, ...);
</code></pre>

<aside class="notes">
The optimal value should be close to 1.0.
</aside>

</section>

<section data-markdown>

## What can rusty-machine do?

- K-Means Clustering
- DBSCAN Clustering
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

<section>

<h2>Why Rulinalg?</h2>

<p class="fragment">Ease of use</p>

<aside class="notes">
Some history behind why this exists - when I started development it was unclear whether any other options would be a good fit.

And of course Rust is a great choice for implementing linear algebra.
</aside>
</section>

<section data-markdown>
## A quick note on error handling

Rust's error handling is fantastic.

```rust
impl Matrix&lt;T> {
    pub fn inverse(&amp;self) -> Result&lt;Matrix&lt;T>, Error> {
        // Fun stuff goes here
    }
}
```

Note:
Using Results to communicate that a method may fail provides more freedom whilst being more explicit.

I could certainly use the error handling more frequently - especially within rusty-machine (rulinalg is pretty good).

</section>

<section data-markdown>

## What does Rulinalg do?

- Data structures (`Matrix`, `Vector`)
- Basic operators (with in-place allocation where possible)
- Decompositions (Inverse, Eigendecomp, SVD, etc.)
- And more...

</section>

<section>
<h2>Why is Rust a good choice?</h2>

<ul>
<li>Trait system is amazing.</li>
<li>Error handling is amazing.</li>
<li>Performance focused code*.</li>
</ul>

<p class="fragment">* Rusty-machine needs some work, but the future looks bright!</p>

<aside class="notes">
Historically we prototype in high level languages and then rewrite performance critical parts.

Traits - Clean, extensible, homogenous API.
Performance - A bold claim right now... But the potential is there for us to prototype
and achieve high performance code in the same environment.
Insights - More from a developers points of view; it is useful to have to think about how the
model should be structured. What data does it need to own, which parts can be made modular without
adding unneeded complexity, etc.
</aside>
</section>

<section data-markdown>
## Why is Rust a good choice?

Most importantly for me - safe control over memory.

Note:
Specifically with the ownership/lifetimes mechanic.

We choose when a model needs ownership. When to allocate new memory for operations. These are things
that are much harder to achieve in other languages as pleasant-to-use as Rust.

</section>

<section data-markdown>
## When would you use rusty-machine?

At the moment - experimentation, non-performance critical applications.

In the future - quick, safe and powerful modeling.

Note:
For now it would be unwise to use this for anything serious. Except maybe if the benefits of Rust outweigh performance and accuracy.

In the future, rusty-machine will try to enable rapid prototyping that can be easily extended into a finished product.

</section>

<section data-markdown>
## Rust and ML in general

Note:
Rust is well poised to make an impact in the machine learning space.

It's excellent tooling and modern design are valuable for ML - and the
benefit of performance with minimal effort (once you're past wrestling with the
borrow checker) is huge.

Some difficulty doing 'exploratory analysis' in Rust compared to say Python.
But I think in the future Rust could definitely hold it's own.
</section>

<section>
<h2>What's next?</h2>
<ul>
<li class="fragment fade-up">Optimizing and stabilizing existing models.</li>
<li class="fragment fade-up">Providing optional use of BLAS/LAPACK/CUDA/etc.</li>
<li class="fragment fade-up">Addressing lack of tooling.</li>
</ul>

<aside class="notes">
By lack of tooling I mean for data handling mostly.
</aside>

</section>

<section data-markdown>
## What would I like to see from Rust?

- Specialization
- Growth of Float/Complex generics
- Continued effort from community

Note:
I really like the direction of the language so far and look forward to what will follow.

The community is great as I'm sure most would confirm. That drive and enthusiasm will create great things.
</section>

<section data-markdown>
## Summary

- Machine learning (done quickly)
- Rusty-machine
- Rulinalg

</section>

<section data-markdown>

## Contributors

|||
 --- | --- | --- 
[zackmdavis](https://github.com/zackmdavis) | [DarkDrek](https://github.com/DarkDrek) | [tafia](https://github.com/tafia)
[ic](https://github.com/ic) | [rrichardson](https://github.com/rrichardson) | [vishalsodani](https://github.com/vishalsodani)
[raulsi](https://github.com/raulsi) | [danlrobertson](https://github.com/danlrobertson) | [brendan-rius](https://github.com/brendan-rius)
 | [andrewcsmith](https://github.com/andrewcsmith) | |

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

Python is especially exciting as we gain access to lots of tooling.

</section>

</section>


