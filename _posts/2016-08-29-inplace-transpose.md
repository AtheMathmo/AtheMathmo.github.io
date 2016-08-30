---
layout: post
title: In-place Transposition in Rust
excerpt: "Should be simple..."
comments: true
---

# In-place Transposition in Rust

In this blog post I'm going to talk about getting an optimized in-place transposition algorithm working in [rulinalg](https://github.com/AtheMathmo/rulinalg) - a pure Rust linear algebra library. I'm not going to go into much detail on how the algorithm works - but will instead focus on my efforts to optimize it in Rust.

## What is rulinalg?

[Rulinalg](https://github.com/AtheMathmo/rulinalg) is a linear algebra library written entirely in Rust. It was originally created as part of [rusty-machine](https://github.com/AtheMathmo/rusty-machine) -
a machine learning library providing implementations of common machine learning algorithms.
For a linear algebra library to support machine learning applications it has to be able to deal with large amounts of data effectively.

### Transposing?

Transposing is an essential part of any linear algebra framework. Transposing is a linear algebra operation which reflects a matrix along its diagonal. This means swapping each element above the diagonal with its counterparts below.

<figure style="width: 80%; display: block; margin-right: auto; margin-left: auto;">
    <img src="{{ site.url }}/assets/transpose.svg" alt="Transposition" style="border-radius: 30px; width: 100%; background: white;"  preserveAspectRatio="xMinYMin slice"/>
    <figcaption>Transposing a 2x3 matrix into a 3x2 matrix. The raw data in row-major order is shown above each matrix.</figcaption>
</figure>

This is an exceedingly common operation and will arise very often in machine learning. In rusty-machine you'll see transposing in _Neural Nets_, _Linear Models_, _Gaussian Mixture Models_ and more. Because of this it is important to provide an efficient implementation.

### Transposing is easy, right?

Transposition has been supported in rulinalg for quite some time. However in rulinalg we only provide a transpose method for doing _out-of-place_ transposition. This means we allocate a new block of memory for the transpose and copy the matrix data there whilst transposing it.

Most of the time this works fine - but what if our matrix is so big that we can barely hold it in memory? In machine learning this isn't as ridiculous as it sounds - big data is the name of the game. In this case we cannot allocate new memory and we resort to something called _in-place transposition_.

_Note: Some frameworks like [numpy](http://www.numpy.org/) have clever ways to get around this problem. I discuss this a little at the end._

## In-place transposing

Out-of-place transposition is a mostly simple thing. In rulinalg it looks like this:

```rust
impl<T: Copy> Matrix<T> {
    pub fn transpose(&self) {
        // Allocate new 'out-of-place' memory
        let mut new_data = Vec::with_capacity(self.rows * self.cols);

        unsafe {
            new_data.set_len(self.rows * self.cols);
        }

        unsafe {
            for i in 0..self.cols {
                for j in 0..self.rows {
                    // Fill the columns with `self`s rows
                    *new_data.get_unchecked_mut(i * self.rows + j) = *self.get_unchecked([j, i]);
                }
            }
        }

        Matrix {
            cols: self.rows,
            rows: self.cols,
            data: new_data,
        }
    }
}
```

_Note that rulinalg uses row-major order. Additionally the above algorithm is naive and can also be improved!_

Here we are copying the rows from `self` into the columns of our new matrix. We have some unsafe code here too to speed things along.

So _out-of-place_ is fairly simple, how about _in-place_? As it turns out, not so much...
For square matrices it is simple enough - we can just `swap` the elements above the diagonal with those below.
But for arbitrary rectangular matrices this is not the case.

### So how?

There are a few ways we can achieve in-place transposing. One of the more common techniques is a method known as cycle following.
This stems from the idea that a transposition is simply a permutation of the underlying data -
and [all permutations can be represented as disjoint cycles](https://en.wikipedia.org/wiki/Permutation#Cycle_notation).
If we can find all of these independent cycles we can rotate the underlying elements and achieve our transposition.
You can view square matrices in the same way too. Each of the diagonal elements lies in it's own unit cycle (so they do not move).
And all other elements lie in a 2-cycle with their mirror elements above/below the diagonal.

When looking into implementing this method it turned out it was about as difficult as it sounded.
Fortunately a [recent paper by Cantanzaro et al.](https://research.nvidia.com/sites/default/files/publications/ppopp2014.pdf) provided a simple (and perhaps more powerful) alternative.

In the paper they describe an algorithm which operates on rows and columns in turn -
rotating and shuffling them with [_gather_ and _scatter_ operations](https://en.wikipedia.org/wiki/Gather-scatter_(vector_addressing)).
And it gets even better - it is simple to extend this to work in parallel!

## Implementing it

I strongly recommend checking out the paper - in particular _Algorithm 1_ and _Figure 2_ give a solid overview of how things work.

The algorithm is very simple to put into code:

```rust
let m = self.rows;
let n = self.cols;
let c = gcd(m, n);

let a = m / c;
let b = n / c;

// Create a temporary storage vector
let larger = cmp::max(m, n);
let mut tmp = Vec::with_capacity(larger);
unsafe { tmp.set_len(larger) };

if c > 1 {
    // Rotate the columns
    for j in 0..n {
        for i in 0..m {
            tmp[i] = self[[gather_rot_col(i, j, b, m), j]];
        }

        for i in 0..m {
            self[[i, j]] = tmp[i];
        }
    }
}

// Permute the rows
for i in 0..m {
    for j in 0..n {
        tmp[scatter_row(i, j, b, m, n)] = self[[i, j]];
    }

    for j in 0..n {
        self[[i, j]] = tmp[j];
    }
}

// Permute the columns
for j in 0..n {
    for i in 0..m {
        tmp[i] = self[[gather_shuffle_col(i, j, a, m, n), j]];
    }

    for i in 0..m {
        self[[i, j]] = tmp[i];
    }
}

self.rows = n;
self.cols = m;
```

In the above I haven't included the various indexing functions -
though the equations can be found in the paper. Here is one example:

```rust
fn gather_rot_col(i: usize, j: usize, b: usize, rows: usize) -> usize {
    (i + j / b) % rows
}
```

Sadly this simple implementation has pretty terrible performance.

```
test linalg::transpose::allocating_transpose_100_100   ... bench:       5,272 ns/iter (+/- 120)
test linalg::transpose::allocating_transpose_10000_100 ... bench:   7,476,689 ns/iter (+/- 1,531,109)
test linalg::transpose::allocating_transpose_1000_1000 ... bench:  12,543,815 ns/iter (+/- 1,609,386)
test linalg::transpose::inplace_transpose_100_100      ... bench:     769,233 ns/iter (+/- 76,096)
test linalg::transpose::inplace_transpose_10000_100    ... bench:  95,481,354 ns/iter (+/- 6,820,747)
test linalg::transpose::inplace_transpose_1000_1000    ... bench:  98,542,938 ns/iter (+/- 7,459,741)
```

The `allocating_transpose` benchmarks here are the out-of-place algorithm shown above.

## Improving it

Fortunately the paper also has a large and very well written section on optimizing the algorithm.
In addition to this there are some [open source implementations](https://github.com/bryancatanzaro/inplace) for CUDA and OpenMP.
Below I'll be mostly describing implementing these things in Rust and some other things I snuck in along the way.

There are a couple of pieces of low hanging fruit here.
Most obvious perhaps are the bound checks on the matrix indexing and the functions used to compute the indices should be inlined.

```
test linalg::transpose::inplace_transpose_100_100      ... bench:     555,059 ns/iter (+/- 7,991)
test linalg::transpose::inplace_transpose_10000_100    ... bench:  71,368,279 ns/iter (+/- 7,489,322)
test linalg::transpose::inplace_transpose_1000_1000    ... bench:  72,898,722 ns/iter (+/- 6,151,680)
```

Now onto the slightly more difficult things...

In our row shuffle loop we are indexing the temporary storage.
To aid with vectorization we can swap the scatter indexing to gather indexing:

```rust
// Permute the row
for i in 0..m {
    for j in 0..n {
        *tmp.get_unchecked_mut(j) =
            *self.get_unchecked([i, d_inverse(i, j, b, a_inv, m, n, c)]);
    }

    // This ensures the assignment is vectorized
    utils::in_place_vec_bin_op(self.get_row_unchecked_mut(i),
                               &tmp[..n],
                               |x, &y| *x = y);
}
```

Above is the second loop from the original algorithm. This code looks a little different to the last time we saw it.
We're no longer indexing `tmp` but are instead indexing into the `Matrix` (gather instead of scatter).
I've also snuck in an extra optimization - that `utils` function just helps the compiler vectorize the assignment.
All of the `unchecked` code returns the values without doing any bounds checking.

```
test linalg::transpose::inplace_transpose_100_100      ... bench:     537,710 ns/iter (+/- 53,171)
test linalg::transpose::inplace_transpose_10000_100    ... bench:  69,080,843 ns/iter (+/- 1,811,604)
test linalg::transpose::inplace_transpose_1000_1000    ... bench:  70,444,580 ns/iter (+/- 1,108,286)
```

This is a pretty minor improvement but it is an improvement!
It looks like things were already behaving pretty nicely for me - but this change is still worth including.

## Arithmetic strength reduction

Our indexing operations have a lot of division and mod computation. We can use a technique known as arithmetic strength reduction to improve things here.
I wont spend too much time on this but if people are interested I'd be happy to share a little more of what I learned.
The core idea is that we can replace repeated division and modulus operations with multiplication which is more efficient.

The implementation was mostly straight forward in rust with a minor hiccup where I relied on some `u128` computations.
To get an idea of how this looked I used a combination of inline assembly and the [extprim](https://crates.io/crates/extprim) crate.

```
test linalg::transpose::inplace_transpose_10000_100    ... bench:  32,419,713 ns/iter (+/- 3,563,752)
test linalg::transpose::inplace_transpose_1000_1000    ... bench:  24,207,849 ns/iter (+/- 1,500,808)
test linalg::transpose::inplace_transpose_100_100      ... bench:     188,893 ns/iter (+/- 18,314)
```

However, I want this to work without a nightly compiler and I'd prefer not to introduce the extprim dependency (I only need some very limited functionality for my use case).
Removing the inline assembly leads to the following numbers:

```
test linalg::transpose::inplace_transpose_10000_100    ... bench:  45,097,359 ns/iter (+/- 915,984)
test linalg::transpose::inplace_transpose_1000_1000    ... bench:  34,543,206 ns/iter (+/- 986,373)
test linalg::transpose::inplace_transpose_100_100      ... bench:     276,375 ns/iter (+/- 5,072)
```

Still a big improvement on our earlier attempts! And the good news is that [128 bit integer support is on the way](https://github.com/rust-lang/rust/issues/35118) ([there's even a PR](https://github.com/rust-lang/rust/pull/35954)!).

Now at this point I should say that I was a little concerned.
I came into this with very little background knowledge and it seemed iffy to me that my implementation was still so much slower than the out-of-place transposition.
I figured the time taken to allocate those large chunks of memory should add up. It turns out this isn't totally unexpected - in-place transposition is just a more expensive operation.
Fortunately I can close the gap on square matrices by implementing a simple `ptr::swap` algorithm:

```
test linalg::transpose::inplace_transpose_100_100      ... bench:      17,863 ns/iter (+/- 3,015)
test linalg::transpose::inplace_transpose_1000_1000    ... bench:   3,103,315 ns/iter (+/- 404,371)
```

## How does this compare?



## What next?

For now I'm pretty happy with this code and will be merging it into rulinalg soon.
But in the future there are some other optimizations I'd like to try out.

For both out-of-place and in-place my implementations incur unneccessary cache misses.
The Catanzaro paper describes a technique to improve this and there are simple extensions to the more naive algorithms too.

I suspect the in-place algorithm really comes into its own when it's parallelized.
I didn't want to try fiddling with this on a first implementation but it will be exciting to see how big an improvement we can get!

I think it's also quite important that I talk about an alternative approach to the above.
If you `transpose` in [numpy](http://www.numpy.org/) it is an effectively free operation. Because of the way arrays are represented in numpy the strides and dimensions are simply swapped.
I think this is the first time that not leaning more heavily on the strided view system has felt like a significant hindrance.
This is an approach that I may look to adopt in the future but am choosing not to for now.
If we go down this road we will have a much harder time optimizing existing algorithms -
for example we can no longer guarantee that our rows are contiguous (an assumption which is used for vectorization and cache utilization). In addition to this the existing code will have to become more complex.

If you do need _free_ in-place transposing in rust you could take a look at [ndarray](https://github.com/bluss/rust-ndarray) - which uses an approach similar to numpy.

## Summary

So what did I learn? Mostly that this was far more difficult than I expected and not quite as useful as hoped. But I think that along the way there were a lot of valuable lessons about optimization - many of which I will be able to carry with me when exploring optimization in future.

Additionally I'm reminded that trying to compete with long-standing linear algebra frameworks like BLAS and LAPACK is very, very difficult.

## References

- [Catanzaro et al. : A Decomposition for In-place Matrix Transposition](https://research.nvidia.com/sites/default/files/publications/ppopp2014.pdf)

