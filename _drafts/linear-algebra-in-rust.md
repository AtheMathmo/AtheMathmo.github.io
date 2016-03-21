---
layout: post
title: Linear Algebra in Rust
excerpt: ""
comments: false
---

In my previous post I focused on the learning module of rusty machine. This time I'll be focusing on the linalg module.

This post is aimed primarily at readers who are familiar with Rust.

## [Rusty-machine](https://github.com/AtheMathmo/rusty-machine)

[Rusty-machine](https://github.com/AtheMathmo/rusty-machine) is a general purpose machine learning library implemented entirely in rust.

It is important for a machine learning library to have a strong linear algebra backbone. But as Rust is an immature language there was no clear contender for this space when I began development. As a result I decided that I would implement this myself!

It should be said that BLAS/LAPACK are the real winner in this linear algebra space - but we'll suspend disbelief for a moment.

### The linalg model

The linalg module is made up mostly of two parts - the `vector` module and the `matrix` module. The `Vector` struct within the `vector` module exists primarily to optimize matrix * vector operations. This feels a little clunky and is likely to change in the future. For the remainder of this post I'll focus on the `matrix` module.

```rust
pub struct Matrix<T> {
    rows: usize,
    cols: usize,
    data: Vec<T>,
}
```

This `Matrix` is the core data structure behind the linear algebra module.

Within the matrix module we implement a selection of helper methods to manipulate and extract data from the `Matrix`. We also implement some operation overloading (as you'd expect for matrices), and some standard decompositions.

### Optimizing in Rust

- Vectorization, slicing.

### Other community libraries
