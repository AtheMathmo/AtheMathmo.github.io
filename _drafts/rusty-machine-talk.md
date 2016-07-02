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
# This talk

- What is rusty-machine?
- What is machine learning?
- Why is rusty-machine great?

</section>

<section data-markdown>
## What is rusty-machine?

[Rusty-machine](https://github.com/AtheMathmo/rusty-machine) is a machine learning library written entirely in Rust.

It focuses on the following:

- Works out-of-the-box without relying on external dependencies.
- Simple and easy to understand API.
- Extendible and easy to configure.

Note:
Installing machine learning libraries can often be made a pain if we also <b>need</b> to install, BLAS, LAPACK, CUDA, and more. Especially for new users.

Try and keep things modular and reuse the API across all models. Some examples of this later.

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

<section data-markdown>
## A broad overview

In machine learning we have *Supervised* and *UnSupervised* learning.

Supervised - We have both *input data* and *targets*

UnSupervised - We have only *input data*

Note:
Supervised example - We have pictures of cats and dogs which are labelled as such. Maybe we want to teach the machine to identify new cats and dogs from just their pictures.

UnSupervised - We have just the pictures without labels. Maybe we want the machine to separate the set of photos in two groups.

We also have some others. SemiSupervised, Reinforcement - we wont go into these.

</section>

<!-- For some reason we must use an explicit code block somewhere for the highlighter to work with markdown... -->
<section>
<h2>The base of rusty-machine</h2>
<pre><code class="hljs rust">
pub mod learning {
    /// For supervised learning
    pub trait SupModel<T, U> {
        fn train(&mut self, inputs: &T, targets: &U);

        fn predict(&self, inputs: &T) -> U;
    }

    /// For unsupervised learning
    pub trait UnSupModel {
        fn train(&mut self, inputs: &T);

        fn predict(&self, inputs: &T) -> U;
    }
}
</code></pre>

<aside class="notes">
SupModel trait is for the supervised learning algorithms. UnSupModel trait for unsupervised learning algorithnms.
</aside>
</section>

<section data-markdown>
## An example

Before we go any further we should see an example.
</section>

<section data-markdown>
## Simple but complicated

The API for other models aim to be as simple as that one. However - machine learning is complicated.

Rusty-machine tries to hide as much of this gunk from the average user whilst keeping it easily accessible to those who need it.

Note:
There are lots of different ways to train models and on top of that many ways to configure and adapt them.

</section>

<section>
<h2>What's next?</h2>
<ul>
<li class="fragment fade-up">Optimizing and stablizing existing models.</li>
<li class="fragment fade-up">Pulling linear algebra into a new crate.</li>
<li class="fragment fade-up">Providing optional use of BLAS/LAPACK/CUDA/etc.</li>
</ul>

</section>

<section data-markdown>
## Thanks!

#### Some Links

- [Rusty-machine](https://github.com/AtheMathmo/rusty-machine)
- [My Blog](http://athemathmo.github.io/)

</section>