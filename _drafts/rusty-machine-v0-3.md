---
layout: post
title: Rusty-Machine - Version 0.3
excerpt: "A lot has changed since v0.2!"
comments: true
---

[Rusty-machine](https://github.com/AtheMathmo/rusty-machine) has just reached version 0.3!

## What is rusty-machine?

For those who haven't heard of rusty-machine you can check out the [repo](https://github.com/AtheMathmo/rusty-machine) and [crate](https://crates.io/crates/rusty-machine).

Rusty-machine is a general purpose machine learning library written entirely in Rust. I wanted to create a library which could support both performance and ease of use - without a large number of external dependencies. Rust let's us write a modular library with a simple interface which can be easily extended by the user. For a more concrete explanation check out my previous blog post. 

## What's new?

A lot has changed since v0.2. Probably too much to list everything so I'll stick with the important stuff.

A lot of work went into optimizing the linear algebra. It's no longer terrible (but still not state of the art). Thanks to [matrixmultiply](https://github.com/bluss/matrixmultiply) we now have a fairly competitive matrix multiplication implementation. Along the same lines we've also introduced the idea of a `MatrixSlice`. This behaves like a [slice](https://doc.rust-lang.org/std/slice/) in the standard library but for matrices! It has already helped us remove a lot of copying and allocation - with more to come.

There have also been some additions to the learning module. We've introduce two new machine learning algorithms: Naive Bayes and GMMs. Both follow the same pattern as the other models - focusing on modularity and customizability. 

Version 0.3 also brings some new ML tools: Regularization and AdaGrad. Regularization hasn't been fully implemented yet but is in place for Neural Nets - the other relevant models will follow soon! 

You can read the full changelog [here](https://github.com/AtheMathmo/rusty-machine/blob/master/CHANGELOG.md).

## What's next?

This section will hopefully change a lot as more people try out the library. For now my goals look something like this:

- Separating out the linear algebra. For now in the same repository but a different crate.
- More data processing - normalization first probably.
- Building on existing models - more support for regularization, more implemented algorithms.
- Serialization (not sure how to handle this one because of nightly).
- Some new models - maybe random trees?
