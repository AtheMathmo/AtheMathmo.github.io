---
layout: post
title: Multithreaded matrix multiplication in Rust - Part I
excerpt: "A first step to parallelized linear algebra."
comments: true
---

This will be a fairly short post. I'm labeling this **Part I** as I'm hoping to give a more full report once I see this work through. It would be good to get some feedback and more eyes on this in the mean time though! Lots of credit to [bluss](https://github.com/bluss) here - he provided an awesome library and lots of support (and code to steal) for this work.


Recently I implemented [bluss](https://github.com/bluss)' [matrixmultiply](https://github.com/bluss/matrixmultiply) in [rusty-machine](https://github.com/AtheMathmo/rusty-machine). Matrixmultiply is an awesome, lightweight library that provides fast, [native implementations](http://bluss.github.io/rust/2016/03/28/a-gemmed-rabbit-hole/) of matrix multiplication in Rust.

The gains were pretty huge (up to 10x improvements on my naive implementation) and this inspired me to see if we could get even better by using multithreading.

Parallelizing matrix multiplication isn't easy and there are a few different approaches. For now I have been playing around with a pretty simple divide and conquer implementation. The idea is that we find the largest dimension and if it is greater than some threshold we split the matrix along it and repeat recursively on the two new halves. Once we are below the threshold we perform our _usual_ matrix multiplication. The algorithm is summarized nicely [here](https://en.wikipedia.org/wiki/Matrix_multiplication_algorithm#Non-square_matrices). I'm using [rayon](https://github.com/nikomatsakis/rayon) to achieve the parallelism.

Initial results are looking pretty good!

```
test linalg::matrix::mat_mul_128_100        ... bench:     213,072 ns/iter (+/- 47,042)
test linalg::matrix::mat_paramul_128_100    ... bench:     270,320 ns/ iter (+/- 167,244)
test linalg::matrix::mat_mul_128_1000       ... bench:   1,996,190 ns/iter (+/- 66,789)
test linalg::matrix::mat_paramul_128_1000   ... bench:   1,253,315 ns/iter (+/- 1,715,936)
test linalg::matrix::mat_mul_128_10000      ... bench:  21,215,542 ns/iter (+/- 793,933)
test linalg::matrix::mat_paramul_128_10000  ... bench:  11,659,379 ns/iter (+/- 2,533,431)
test linalg::matrix::mat_mul_128_100000     ... bench: 211,979,455 ns/iter (+/- 13,750,261)
test linalg::matrix::mat_paramul_128_100000 ... bench: 112,330,287 ns/iter (+/- 4,083,264)
```

_If requested I'll try to add some benchmarks for parallel BLAS._

This was run on my pretty average laptop with 4 cores. The implementation is still basic and so there is some overhead that can be removed quite easily.

### What next?

First I'm going to try trimming off some of this overhead. In the process I'll hopefully make some of the code a little safer to use. I've had to break a couple rules to get this implementation up and running (allowing cloning of mutable raw pointers, and `Sync`/`Send`ing them too...).

I'll also take a deeper look at the [matrixmultiply](https://github.com/bluss/matrixmultiply) library to see if we can introduce the [BLIS multithreading](https://github.com/flame/blis/wiki/Multithreading) - this may give us better performance.

After this I'm going to see if I can optimize some of the other native linear algebra within [rusty-machine](https://github.com/AtheMathmo/rusty-machine). Starting with inverses (probably). _This won't be included as part of this series of posts._