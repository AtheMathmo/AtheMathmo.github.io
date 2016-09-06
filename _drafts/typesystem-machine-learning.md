---
layout: post
title: A type system for Machine Learning
excerpt: "What else can Rust do to help machine learning?"
comments: false
---

This post is about [rusty-machine](https://github.com/AtheMathmo/rusty-machine), a pure rust machine learning library.
In particular I'll be talking about the API and how we can leverage Rust's type system to improve it.

## Why [rusty-machine](https://github.com/AtheMathmo/rusty-machine)?

Machine learning is already huge and there's no lack of tools to support that. So the question is, what can Rust bring to the party?

Rust offers high performance alongside the promise of memory safety. There's many reasons why this is valuable for machine learning but I don't want to focus on them in this post. I've [written](http://athemathmo.github.io/2016/03/07/rusty-machine.html) and [spoken](http://athemathmo.github.io/2016/07/28/rusty-machine-talk.html#/) about them a few times before. What I will add here is that the Rust type system can be a very valuable tool to help guide users into understanding machine learning.

## The `Model` trait

<a name="note-1-origin"></a>

Right now the idea of a _model_ is encapsulated in Rust using a trait like this<sup>[1](#note-1)</sup>:

```rust
pub trait Model<T, U> {
    fn train(&mut self, inputs: T, targets: U);

    fn predict(&self, inputs: T) -> U;
}
```

<a name="note-2-origin"></a>

To those of you familiar with machine learning this probably looks quite familiar. This trait system lets us describe the behaviour of a machine learning model.
All machine learning models follow the pattern of first training them with data and then asking them to predict outcomes from what they've learnt<sup>[2](#note-2)</sup>.

In this trait, type `T` represents the type of the input data. Something like `Matrix` usually. And type `U` is the type of the output data produced by the model - this might be a `Vector` of class labels for each data point.

<a name="note-3-origin"></a>

But this leaves a lot to be desired. For almost all<sup>[3](#note-3)</sup> of our models we cannot use the `predict` function without first `train`ing. If a user tries to do this then our only choice is to panic and abort the program.

### What does this look like?

```rust
// Create a new K-Means classifier
let mut model = KMeansClassifier::new(2);

// Train the model on some input data
model.train(&inputs);

// Predict the classes of some new data
let outputs = model.predict(&new_data);
```

The above shows how we use a simple model in rusty-machine. Here both `inputs` and `new_data` are `Matrix` types.

## Improving the `Model` trait

This core part of the API hasn't really changed since rusty-machine was created. Sadly this is in spite of the issues and my improved knowledge of Rust. Recently I decided that I wanted to address some of these issues - starting with the back bone.

```rust
pub trait Model<T, U> {
    fn train(&mut self, inputs: T, targets: U) -> Result<(), Error>;

    fn predict(&self, inputs: T) -> Result<U, Error>;
}
```

#### What's changed here?

Our functions now return `Result`s! `Result`s are Rust's form of error handling. Though at first they may seem clunky and complex they are a fantastic tool that gives the user much more control over how errors should be processed.

In our case it is common for training to fail due to invalid data, parameter settings, or something else outside of the user's control. `Result`s let the user handle all of these and choose what should happen next.

<a name="note-4-origin"></a>

As a small side note I'm not really a fan of returning `Result<(), E>`<sup>[4](#note-4)</sup> but it feels worthwhile here.

### What does this look like?

```rust
// Create a new K-Means classifier
let mut model = KMeansClassifier::new(2);

// Train the model on some input data
model.train(&inputs).expect("Failed to train k-means");

// Predict the classes of some new data
let outputs = model.predict(&new_data).expect("Failed to predict new classes");
```

## The future of the `Model` trait

With the above we still haven't really solved our problems. The errors are made more explicit and the user gets more control over how to handle them.
But it would be nice if we could prevent some of these errors from ever arising.

```rust
/// Trainer trait
pub trait Trainer<M>
    where M: Model
{
    /// Input type
    type Input;
    /// Target type
    type Target;

    /// Train the model
    fn train(self, inputs: &Self::Input) -> Result<M, Error>;
}

/// A model which can be predicted from
pub trait Model {
    /// Input type
    type Input;
    /// Output type
    type Output;

    /// Predict from the model
    fn predict(&self, inputs: &Self::Input) -> Result<Self::Output, Error>;
}
```

#### What's changed here?

Split out into two traits - one will describes the training process and we change the notion of a model to be a trained model which is ready to predict from new data.

Note that we've also left the associative types on `M` unconstrained in the `Trainer` trait. This is intentional (though not necessary) - we may want to train a model on a `Matrix` but then predict from a `MatrixSlice` (which is synonymous to the `Vec<T>` - `&[T]` relationship). Until we have [custom DSTs](https://github.com/rust-lang/rfcs/pull/1524#issuecomment-241809441) we have to treat these as different types within [rulinalg](https://github.com/AtheMathmo/rulinalg).

### What does this look like?

```rust
// Create a new K-Means trainer
let trainer = KMeansTrainer::new(2);

// Consume the trainer to get a model
let mut model = trainer.train(&inputs).expect("Failed to train k-means");

// Predict from the model
let outputs = model.predict(&new_data).expect("Failed to predict new classes");
```

This looks almost identical to the previous example - the main advantage is that our `model.predict` call can no longer fail if the model has not been trained. But there are some other more subtle advantages here too.

Firstly this _feels_ a lot more Rust-like, and as such promotes some patterns that felt a little unnatural before. One common example is the [builder pattern](https://aturon.github.io/ownership/builders.html). With the builder pattern we consume a dedicated struct to construct the object of interest. This works well for machine learning models as we provide default parameter settings and want our users to be able to pick and choose which settings to modify.

```rust
// Create a new K-Means model with 100 iterations and forgy initialization
let mut model = KMeansTrainer::new(2).iters(100).init(Forgy).train(&inputs).unwrap();

// Predict from the model
let outputs = model.predict(&new_data).expect("Failed to predict new classes");
```

### Some other advantages

Serialization becomes more natural. We can now serialize a model once is has been trained and reuse the model to make future predictions. This is possible currently but we have to worry about whether the model has been trained previously after we deserialize.

In machine learning we sometimes want to incrementally update our model using incoming data - this is sometimes called [online learning](https://en.wikipedia.org/wiki/Online_machine_learning). We can capture things like online learning more easily by implementing traits on the new `Model`s. You can imagine adding an `update` function which only works on `Model`s.

Our new code will be generally cleaner. Current models have a lot of `Option` fields which are filled during training - we can move these out. However there are often functions used by both training and prediction functions which we will need to adapt for both, this could get messy sometimes.

### The disadvantages

This is less familiar to users coming from other machine learning frameworks. It requires a better understanding of Rust before the user can jump in - though hopefully this hurdle can be tackled with good documentation. With that said - I do think this approach will feel more natural to those familiar with Rust.

There are also some added difficulties with code organization. We often require some code used in the training to help with prediction. For example with k-means in both training and classification we need to find the closest centroids to a given point. This presents some new code organization challenges - but these shouldn't be too difficult to overcome.

## Summary

I started rusty-machine about 10 months ago and it was my first ever Rust project. Although the API has worked well so far - it could certainly do with an update (along with quite a few other areas of the library).

Here I've spoken about some changes to rusty-machine. Adding `Result`s seems like an obvious improvement with no real down sides (compared to our current API). The second change I suggested is something I'm not so sure about. It offers a lot in terms of the API but I don't want to under estimate the effect of raising the complexity for new users. I'd love to get some feedback on whether this is the best approach to take.

## Notes

<a name="note-1"></a>

[1](#note-1-origin) - There are actually two traits, one for Supervised learning and one for Unsupervised.

<a name="note-2"></a>

[2](#note-2-origin) - Clustering is arguably an exception. Models like DBSCAN don't have a natural notion of prediction, but we can use a nearest neighbours approach (for example) to predict the class of new data.

<a name ="note-3"></a>

[3](#note-3-origin) - For some models we can make bad predictions from the initial states without training - for example neural networks when we randomize the weights.

<a name="note-4"></a>

[4](#note-4-origin) - The compiler does warns against not consuming the result. However, it feels easy for the user to overlook this and continue about their business with a broken state. Fortunately the final proposal here avoids this issue too!
