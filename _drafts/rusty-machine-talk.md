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

<section>
In machine learning we have <b>Supervised</b> and <b>Unsupervised</b> learning.

<p class="fragment fade-up" data-fragment-index="1">Supervised - We have labelled <i>input data</i></p>
<p class="fragment fade-up" data-fragment-index="2">Unsupervised - We have unlabelled <i>input data</i></p>

<span style="position:relative; height: 800px;">
    <span class="fragment fade-up" data-fragment-index="1" style="position: absolute; display: block; height: 400px; width: 800px;">
        <span class="fragment fade-out" data-fragment-index="2">
            <img src="{{ site.url }}/assets/cat-in-suit.jpg" style="border-radius: 20px;" >
        </span>
    </span>
    <span class="fragment fade-up" data-fragment-index="2" style="position: absolute; display: block; height: 400px; width: 800px;">
        <img src="{{ site.url }}/assets/dog-headphones.jpg" style="border-radius: 20px;" >
    </span>
</span>

<br><br><br><br><br><br><br><br>

<aside class="notes">
Supervised example - We have pictures of cats and dogs which are labelled as such. Maybe we want to teach the machine to identify new cats and dogs from just their pictures.

UnSupervised - We have just the pictures without labels. Maybe we want the machine to separate the set of photos in two groups.

We also have some others. Semi-Supervised, Reinforcement - we wont go into these.
</aside>
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

Rusty-machine tries to hide as much of this junk from the average user whilst keeping it easily accessible to those who need it.

Note:
There are lots of different ways to train models and on top of that many ways to configure and adapt them.

</section>

<section data-markdown>
## Why is Rust a good choice?

- Trait system is amazing.
- Performance focused focused code without relying on heavy dependencies*.
- Provides insights into models.
- (Historically we prototype in high level languages and then rewrite performance critical parts.)

\* Not so performant right now, but the future looks bright!

Note:
Traits - Clean, extensible, homogenous API.
Performance - A bold claim right now... But the potential is there for us to prototype and achieve high performance code in the same environment.
Insights - More from a developers points of view; it is useful to have to think about how the model should be structured. What data does it need to own, which parts can be made modular without adding unneeded complexity, etc.
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