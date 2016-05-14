---
layout: post
title: A blunder with borrowing
excerpt: "After 6 months I'm still learning how to think in Rust."
comments: true
---

For most readers this will be a pretty dull post. I feel compelled to publicly document my mistake in the hopes that the embarrassment may prevent it happening again.

I've been writing Rust code, almost every day, for about 6 months now. I've begun to feel pretty comfortable with a lot of the core principles and thought I was doing a good job of extending myself to more advanced concepts. Recently I was reminded that I still have a lot to learn...

## The old code

Before I jump into the problem (and solution) I should explain the context. In my library I have a `Matrix` data structure which owns it's underlying data in a `Vec`. It looks like this:

```rust
pub struct Matrix<T> {
   rows: usize,
   cols: usize,
   data: Vec<T>,
}
```

Fairly recently I added the concept of a `MatrixSlice`, which allows the user to work with a chunk of the matrix (without copying data). The struct looked like this:

```rust
pub struct MatrixSlice<T> {
   ptr: *const T,
   rows: usize,
   cols: usize,
   row_stride: usize,
}
```

Now perhaps the more experienced among you can already see the problem. I'll keep the suspense for a little longer though.

The struct as written above does allow me to do all of the things I wanted to do. I can take a slice of a `Matrix`, split a `Matrix` into multiple slices (similar to `split_at` in the standard library's slices).

However, it also lets us do some pretty terrible things.

## Uh-oh

```rust
// Create a new 3x3 matrix filled with ones.
let mat = Matrix::new(3, 3, vec![1u32; 9]);

// Slice the top left 2x2 block.
let slice = MatrixSlice::from_matrix(&mat, [0,0], 2, 2)

// Consume `mat` to get its data.
let data = mat.into_vec();

// Now do whatever we want with `b`!
let nonsense = slice + 2u32;
```

The above code shows that there is relatively little stopping us doing very unsafe things. We can consume the underlying data which our slice is still pointing too and the compiler will have no idea that things are going wrong.

## Lifetimes!

So here is my blunder. I forgot to give the slice the same life time as the underlying data. Once we do that the compiler can do its usual shouting when things are about to go terribly wrong!

```rust
pub struct MatrixSlice<'a, T: 'a> {
   ptr: *const T,
   rows: usize,
   cols: usize,
   row_stride: usize,
   marker: PhantomData<&'a T>
}
```

Fortunately I hadn't done anything terrible (like the "uh-oh" example) within the library. But I did have to annotate large amounts of code with lifetimes and rewrite/add a lot of macros. All in all this was just under 1000 lines but it's a small price to pay for the gains in safety!

## What have I learnt?

Primarily that `unsafe` truly is unsafe. I had read a while ago that introducing an `unsafe` block affects far more than just the block itself - I think that now I finally understand that. In hindsight this all seems incredibly obvious but it certainly wasn't a couple of days ago.