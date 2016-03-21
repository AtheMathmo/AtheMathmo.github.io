---
layout: post
title: What is rusty-machine?
excerpt: "An overview of rusty-machine and how it fits together."
comments: false
---

In [my last post]() I attempted to explain a little about machine learning and some of the problems we face as practitioners. I tried to show how I believe that rust can help to solve this problem and introduced rusty-machine as a means to do so.

In this post I'll talk a little more about [rusty-machine](https://github.com/AtheMathmo/rusty-machine) and how the project is structured. This will include mostly high-level discussion but may get a little gritty in places. Knowledge of machine learning or rust shouldn't be necessary but may help!

# [Rusty-machine](https://github.com/AtheMathmo/rusty-machine)

[Rusty-machine](https://github.com/AtheMathmo/rusty-machine) is a general purpose machine learning library. Implemented entirely in Rust. This library includes two core modules; the `linalg` module and the `learning` module.

## Linalg

The linalg module contains most of the linear algebra you can expect to find within a scientific computing library. A brief overview includes:

- Generic Matrix and Vector types.
- Standard Matrix and Vector arithmetic.
- Some basic data manipulation and selection.
- Matrix decompositions (cholesky, lup, qr, eigendecomp).
- Inverses, determinants.

I won't spend too much time talking about this. Though it took a huge amount of effort it is still not close to the state of the art.

Linear algebra is a tricky and long-studied topic. For now I've decided it is best to focus on the other (more interesting) module and switch to a common linear algebra library later.

## Learning

This module makes up the machine learning portion of rusty-machine. The module currently has support for the following models:

- Linear Regression
- Logistic Regression
- Generalized Linear Models
- K-Means Clustering
- Neural Networks
- Gaussian Process Regression
- Support Vector Machines
- Gaussian Mixture Models





## Future development