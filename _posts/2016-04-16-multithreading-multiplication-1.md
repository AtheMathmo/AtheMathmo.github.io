---
layout: post
title: Multithreaded matrix multiplication in Rust - Part I
excerpt: "A first step to parallelized linear algebra."
comments: true
---

<h3><a href="/2016/04/25/multithreading-multiplication-2.html">Part II</a></h3>

---

This will be a fairly short post. I'm labeling this **Part I** as I'm hoping to give a more full report once I see this work through. It would be good to get some feedback and more eyes on this in the mean time though! Lots of credit to [bluss](https://github.com/bluss) here - he provided an awesome library and lots of support (and code to steal) for this work.


Recently I implemented [bluss](https://github.com/bluss)' [matrixmultiply](https://github.com/bluss/matrixmultiply) in [rusty-machine](https://github.com/AtheMathmo/rusty-machine). Matrixmultiply is an awesome, lightweight library that provides fast, [native implementations](http://bluss.github.io/rust/2016/03/28/a-gemmed-rabbit-hole/) of matrix multiplication in Rust.

The gains were pretty huge (upwards of 10x improvements on my naive implementation) and this inspired me to see if we could get even better by using multithreading.

Parallelizing matrix multiplication isn't easy and there are a few different approaches. For now I have been playing around with a pretty simple divide and conquer implementation. The idea is that we find the largest dimension and if it is greater than some threshold we split the matrix along it and repeat recursively on the two new halves. Once we are below the threshold we perform our _usual_ matrix multiplication. The algorithm is summarized nicely [here](https://en.wikipedia.org/wiki/Matrix_multiplication_algorithm#Non-square_matrices). I'm using [rayon](https://github.com/nikomatsakis/rayon) to achieve the parallelism.

Initial results are looking pretty good!

```
test linalg::matrix::mat_mul_128_100        ... bench:     221,813 ns/iter (+/- 28,576)
test linalg::matrix::mat_paramul_128_100    ... bench:     213,257 ns/iter (+/- 16,667)
test linalg::matrix::mat_blasmul_128_100    ... bench:     107,305 ns/iter (+/- 14,451)

test linalg::matrix::mat_mul_128_1000       ... bench:   1,994,442 ns/iter (+/- 79,774)
test linalg::matrix::mat_paramul_128_1000   ... bench:   1,147,764 ns/iter (+/- 136,592)
test linalg::matrix::mat_blasmul_128_1000   ... bench:     996,405 ns/iter (+/- 109,778)

test linalg::matrix::mat_mul_128_10000      ... bench:  21,185,583 ns/iter (+/- 794,584)
test linalg::matrix::mat_paramul_128_10000  ... bench:  11,687,473 ns/iter (+/- 638,582)
test linalg::matrix::mat_blasmul_128_10000  ... bench:  10,278,981 ns/iter (+/- 973,273)

test linalg::matrix::mat_mul_128_100000     ... bench: 210,618,866 ns/iter (+/- 4,908,516)
test linalg::matrix::mat_paramul_128_100000 ... bench: 112,120,346 ns/iter (+/- 6,052,281)
test linalg::matrix::mat_blasmul_128_100000 ... bench: 102,699,089 ns/iter (+/- 9,024,207)
```

The BLAS benchmarks are taken from [ndarray](https://github.com/bluss/rust-ndarray) using openblas. For comparison, here is the old implementation:

```
test linalg::matrix::mat_mul_128_100        ... bench:   2,078,298 ns/iter (+/- 132,209)
test linalg::matrix::mat_mul_128_1000       ... bench:  20,901,834 ns/iter (+/- 576,653)
test linalg::matrix::mat_mul_128_10000      ... bench: 228,113,515 ns/iter (+/- 2,512,868)
test linalg::matrix::mat_mul_128_100000     ... bench: too damn long /iter (+/- ____)
```

This was run on my pretty average laptop with 4 cores. The implementation is still basic and so there is some overhead that can be removed quite easily.

### What next?

First I'm going to try trimming off some of this overhead. In the process I'll hopefully make some of the code a little safer to use. I've had to break a couple rules to get this implementation up and running (allowing cloning of mutable raw pointers, and `Sync`/`Send`ing them too...).

I'll also take a deeper look at the [matrixmultiply](https://github.com/bluss/matrixmultiply) library to see if we can introduce the [BLIS multithreading](https://github.com/flame/blis/wiki/Multithreading) - this may give us better performance.

After this I'm going to see if I can optimize some of the other native linear algebra within [rusty-machine](https://github.com/AtheMathmo/rusty-machine). Starting with inverses (probably). _This won't be included as part of this series of posts._