---
layout: post
title: Learning in Rust
excerpt: "What are Rust and rusty-machine?"
comments: true
---

What is the aim of this post:

- Why is Rust well suited for Machine Learning?
- What is rusty-machine?
- What's next?
- A brief highlight of other community efforts.

I'll be describing very briefly the work I've been doing on rusty-machine. I'd love to get some feedback - both on this post and on the library.

### What is [Rust](https://www.rust-lang.org/)?

I'm not going to discuss all of the wonderful and difficult features of Rust - but will give a very brief overview of some parts.

Rust is a Systems Programming language emphasizing speed and safety. This is coupled with some nice high level code features. One of which is Traits. 


```rust
pub trait Mammal {
	pub fn speak();

	pub fn eat() {
		println!("Yum yum yum.");
	}
}
```

The above is an example of a trait. It is a contract which can be implemented by an object - similar to an interface in other languages.

```rust
pub struct Dog;

impl Mammal for Dog {
	fn speak() {
		println!("Woof!");
	}
}
```

Our dog can now speak by saying "Woof!" and adopts the default behaviour for eating.

Though there's plenty more to say, we'll leave Rust there for now. As we'll see later this trait structure can help us provide flexible contracts for our Machine Learning models.

### What is Machine Learning?

```
"Field of study that gives computers the ability to learn without being explicitly programmed." - Arthur Samuel
```

I think nowadays most programmers have a rough idea of what Machine Learning means. I wont spend too much time explaining the concept but will instead explain some terminology and give a concrete example.

In Machine Learning we have `Models`. These models represent a set of hypotheses which can be used to explain a pattern which is present in some data. By giving the model some data we can `train` it and have it learn this pattern - the best hypothesis to fit the data. The model can then be used to `predict` new patterns from data it hasn't seen before.

The key take-aways are:

- `Model` : An object representing a set of possible explanations for some data patterns.
- `Train` : The act of teaching a model the best explanation from some data (learning).
- `Predict` : Using the learned model to predict the pattern of unseen data.

Let's consider a concrete example: [The Logistic Regression Model](https://en.wikipedia.org/wiki/Logistic_regression). This model is used for classification. For example we might want to classify whether a tumor is malignant or benign using features of the tumor such as size, symmetry, compactness, etc.

The logistic regression model will contain a vector of parameters, β. The model takes in a row of data x<sup>T</sup> and computes h(x<sup>T</sup>β) where h is the [Sigmoid Function](https://en.wikipedia.org/wiki/Sigmoid_function). Now if this values is greater than 0.5 we classify as positive, and otherwise negative. Going back to our terminology, different values of β represent different hypotheses.

We can train our logistic regression model using a technique known as Gradient Descent Optimization. This iteratively updates the parameters to provide a better explanation of the data (more of the data is classified correctly).

### So, how does Rust help with Machine Learning?

Traits allow us to create well structured high-level code whilst maintaining low-level performance. Hopefully we can find a way to provide developers access to fast iterative machine learning with good performance and scaling.

Of course this does exist in places now, e.g. [scikit-learn](http://scikit-learn.org/). But hey, more of a good thing can't be bad! And Rust provides the unique opportunity to achieve all of this within a single language (maybe). And hopefully we can achieve this with a nice clean API to top it all off.

---

## [Rusty-Machine](https://github.com/AtheMathmo/rusty-machine)

[Rusty-Machine](https://github.com/AtheMathmo/rusty-machine) is a general purpose machine learning library. Implemented entirely in Rust. It is still very much in the early stages of development and so the following information is likely to be outdated in future. I hope to provide an overview of what rusty-machine is trying to achieve.

Rust aims to provide a consistent, high-level API for users without compromising performance. This consistency is achieved through Rust's trait system.

### How do we use traits?

The trait set up allows users to easily implement a model with clean access to the necessary components. It is also very easy for the user to modify the models; updating parameters and swapping out components (such as the Gradient Descent algorithm).

```rust
/// Trait for supervised models.
pub trait SupModel<T,U> {

    /// Train the model using inputs and targets.
    fn train(&mut self, inputs: &T, targets: &U);

    /// Predict output from inputs.
    fn predict(&self, inputs: &T) -> U;
}
```

The above is the trait for a specific type of model - a `Supervised` model. This means that the model is trained using some example outputs as well as the input data. There is also a trait for `UnSupervised` models which looks very similar (except that we do not have any targets).

When a user wants to use a model they `train` and `predict` from the model via this trait. This is the essence of our consistent API - all models are accessed via the same core methods. We balance this rigid access by allowing the models themselves to be very customisable. Let's consider our Logistic Regression model.

```rust
pub struct LogisticRegressor<A>
    where A: OptimAlgorithm<BaseLogisticRegressor>
{
    base: BaseLogisticRegressor,
    alg: A,
}
```

This looks a little messy... Here `A` is a generic type. The line beginning `where` specifies that `A` must implement the `OptimAlgorithm` trait. This is the trait for Gradient Descent algorithms (which is poorly named, my apologies!). The `BaseLogisticRegressor` contains the parameters, β, and other core methods for logistic regression.

The `LogisticRegressor` struct allows any Gradient Descent algorithm that fits the base struct to be used. There are a number of built-in algorithms (Stochastic GD, etc.) or alternatively the user can create their own and plug them in. This relationship is two-fold - developers can create their own models and utilize the existing gradient descent algorithms.

Of course this doesn't end with logistic regression and gradient descent. This flexibility and customisation is an aim throughout.

## The down sides

Maybe this is all sounding great. However, it is still early days and lot's of work still needs to be done.

The library is very immature. Though I think the vision is strong we're a long way off and lack a lot of key components for a Machine Learning library. Consistent data handling, visualizations and performance are all core areas that need a lot of work. Even after this many would consider validation and pipelines too important to miss.

I'd be naive also to ignore the fact that I have been the sole developer on this project for a while\*. There's likely some bad choices that seem good to me - I'd love to have those pointed out!

\* I have had some help in places. Thanks raulsi and all of the amazing people at [/r/rust](https://www.reddit.com/r/rust/) and SO!

## Next steps

There is definitely room for improvement on existing algorithms - both for performance and introducing some more modern techniques. There will also need to be:

- Some restructuring of the current library.
- Separation of linear algebra into a new crate.
- Addition of data tools.

Among other things.

In the slightly more distant future we'll need to decide how to proceed with the linear algebra as well. This will most likely mean using bindings to BLAS and LAPACK - probably via a community adopted linear algebra library.

### Call for help

[Rusty-Machine](https://github.com/AtheMathmo/rusty-machine) is an open-source project and is always looking for more people to jump in. If anything in this post has stood out to you please check out the [collaborating page](https://github.com/AtheMathmo/rusty-machine/blob/master/CONTRIBUTING.md).

### Rust sounds cool... But rusty-machine not so much

Rusty-machine isn't the only machine learning tool written in Rust. There are other libraries and tools which you should take a look at.

- [rustlearn](https://github.com/maciejkula/rustlearn) by maciejkula
- [Leaf](https://github.com/autumnai/leaf/tree/master/src) by Autumn
- [ndarray](https://github.com/bluss/rust-ndarray) by bluss

And lots of others.