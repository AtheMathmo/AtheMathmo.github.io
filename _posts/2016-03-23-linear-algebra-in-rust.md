---
layout: post
title: Linear Algebra in Rust
excerpt: "Pure rust linear algebra in rusty-machine."
comments: false
---

In my previous post I focused on the `learning` module of rusty machine. This time I'll be focusing on the `linalg` module. I'll be covering:

- [Rusty-machine](#rusty-machine)
- [The linalg module](#the-linalg-module)
- [Feature summary](#feature-summary)
- [Optimizing linalg](#optimizing-matrix-arithmetic-in-rust)
- [What's next?](#whats-next)

This post is aimed primarily at readers who are familiar with Rust.

<a name="rusty-machine"></a>

## [Rusty-machine](https://github.com/AtheMathmo/rusty-machine)

[Rusty-machine](https://github.com/AtheMathmo/rusty-machine) is a general purpose machine learning library implemented entirely in rust.

It is vital for a machine learning library to have a strong linear algebra backbone. But as Rust is an immature language there was no clear contender for this space when I began development\*. As a result I decided that I would implement this myself.

Before I go on it should be said that this isn't the smartest thing to do. There's a reason [BLAS](http://www.netlib.org/blas/)/[LAPACK](http://www.netlib.org/lapack/) are so common in this space. On top of this there are other considerations like GPU utilization and parallelization. But for now we'll suspend disbelief and jump in!

\* There are in fact [some good alternatives](#other-community-libraries).

### The linalg module

The `linalg` module is made up mostly of two parts - the `vector` module and the `matrix` module. The `Vector` struct within the `vector` module exists primarily to optimize matrix * vector operations. This feels a little clunky and is likely to change in the future. For the remainder of this post I'll focus on the `matrix` module.

```rust
pub struct Matrix<T> {
    rows: usize,
    cols: usize,
    data: Vec<T>,
}
```

This `Matrix` is the core data structure behind the linear algebra module.

Within the matrix module we implement a selection of helper methods to manipulate and extract data from the `Matrix`. We also implement some operation overloading (as you'd expect for matrices), and some standard decompositions.

### Feature summary

#### Data manipulation

The linalg module attempts to provide the same support that is found in most modern linear algebra libraries. This includes:

- Matrix concatenation.
- Copying chunks from matrices.
- Transposing.

These are just a few examples and this area is always in active development.

#### Operation overloading

We use Rust's inbuilt operation overloading for addition, subtraction, multiplication, division and indexing. We also provide implementations of element-wise multiplication and division. And `apply(mut self: Matrix<T>, f: &Fn(T) -> T)` which lets us mutate the matrix with some general function.

I'll come back to the operation overloading in a moment.

#### Decompositions

The library currently supports common decompositions:

- LUP decomposition (and as a result, determinants and inverses).
- Cholesky decomposition.
- QR decomposition.
- Upper Hessenberg decomposition.
- Eigendecomposition.

These are far from state of the art performance. There is also some room for improvement in terms of error handling.

### Optimizing Matrix Arithmetic in Rust

Rust compiles with LLVM - this means we get some really nice optimizations but sometimes we need to ask the compiler just right to get there. For the remainder of this post I'll go over some of the techniques used within the `linalg` module to achieve vectorized code. All of these techniques were adapted from bluss' [ndarray](https://github.com/bluss/rust-ndarray) - which is awesome. For those of you who caught [bluss' talk](http://bluss.github.io/rust-ndarray/talk1/) at the bay area meetup this will be similar to his _Performance Secrets_.

Here's what addition overloading looks like:

```rust
/// Adding and consuming two matrices
impl<T: Copy + One + Zero + Add<T, Output = T>> Add<Matrix<T>> for Matrix<T> {
    type Output = Matrix<T>;

    fn add(self, f: Matrix<T>) -> Matrix<T> { ... }


/// Adding but not consuming two matrices
impl<'a, 'b, T: Copy + One + Zero + Add<T, Output = T>> Add<&'b Matrix<T>> for &'a Matrix<T> {
    type Output = Matrix<T>;

    fn add(self, m: &Matrix<T>) -> Matrix<T> { ... }
```

In the first case above we are consuming both matrices - as a result we can reuse the existing allocated memory. In the second example we are using two references and so we must allocate new memory. 

```rust
fn in_place_vec_bin_op<F, T: Copy>(u: &mut [T], v: &[T], mut f: F)
    where F: FnMut(&mut T, &T) {
        debug_assert_eq!(u.len(), v.len());
        let len = cmp::min(u.len(), v.len());
        
        let ys = &v[..len];
        let xs = &mut u[..len];

        for i in 0..len {
            f(&mut xs[i], &ys[i])
        }
}
```

In the above function we require `u` to be `&mut [T]` so that we can mutate the data and reuse the memory that is allocated. To help the compiler avoid bound checks we slice the incoming slices - this convinces the compiler that any indexing up to `len` is safe. It also tells the compiler these slices are the same length and so we can vecorize the operations on them.

For example when adding:

```rust
fn add(mut self, f: Matrix<T>) -> Matrix<T> {
    utils::in_place_vec_bin_op(&mut self.data, &f.data, |x,&y| {*x = *x + y});
    self
}
```

When we want to allocate new memory we use similar tricks with a few more intricacies. I'll exclude the details for brevity but the only real difference is that we allocate new memory and assign to this instead of mutating.

We utilize these functions when we implement `Matrix` and `Vector` operations. This provides clean reusable code which is vectorized. It also worth noting briefly that this approach opens up an easy avenue into parallelization - we can easily break up the matrix data (using `split_at` and `split_at_mut` for slices) and run the above functions in parallel. This will come in handy when doing divide and conquer matrix multiplication for example.

### Disclaimer

Rusty-machine is still very immature and this is especially true of the linear algebra. I have only recently turned towards optimizing this module and a lot more work is needed. In the future it is likely that this component will be replaced by a stronger library from the community - though I do intend to keep a pure Rust library available without dependencies on BLAS/LAPACK (similar to [numpy](http://www.numpy.org/)).

### What's next?

There are a few things coming up. I'm currently working on implementing `MatrixSlice` - which allows us to do operations on a small chunk of the `Matrix`. This should also be accompanied by `MatrixSliceMut` so that we can mutate these chunks. In the future I'll be looking to:

- Separate `linalg` module from rusty-machine.
- Parallelization - work has been done on multiplication.
- Optimizing decomposition and data manipulation.
- Adding [SVD](https://en.wikipedia.org/wiki/Singular_value_decomposition) and improving eigendecomposition.
- Evaluate the future of `Vector`.

This last point is probably the most complicated. There are two immediate solutions: remove `Vector` entirely; implement a shared trait for `Matrix` and `Vector`. There are other options as well (i.e. misusing specialization).

As I mentioned briefly above I do plan on maintaining and developing the core linear algebra stuff. I think it is extremely useful for Rust to have it's own linear algebra library without depending on BLAS, etc. This is especially useful for bringing new people into libraries like [rusty-machine](https://github.com/AtheMathmo/rusty-machine) without throwing them into _dependency hell_. And of course - Rust is an awesome language and can potentially compete with these long standing libraries.

### Other community libraries

- [ndarray](https://github.com/bluss/rust-ndarray) : Arrays inspired by numpy.
- [blas-sys](https://github.com/stainless-steel/blas-sys) : Generic blas bindings used by ndarray.
- [rust-blas](https://github.com/mikkyang/rust-blas) : BLAS bindings for Rust.
- [nalgebra](https://github.com/sebcrozet/nalgebra/) : Low dimensional linear algebra.
- [servo/euclid](https://github.com/servo/euclid) : Basic linear algebra used by Servo.

There are probably a lot of others. This is a pretty active area of development for the Rust community.
