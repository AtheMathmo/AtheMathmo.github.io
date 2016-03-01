---
layout: post
title: Learning in Rust
excerpt: "What are Rust and rusty-machine?"
---

What is the aim of this post:

- Why is Rust well suited for Machine Learning?
- What is rusty-machine?
- What's next?
- A brief highlight of other community efforts.

### What is Rust?

I'm not going to discuss all of the wonderful and difficult features of Rust - but will give an abstract overview.

Rust is a Systems Programming language emphasizing speed and safety. This is coupled with some nice high level code features. One of which is Traits. 


```rust
pub trait Mammal {
	pub fn speak();

	pub fn eat() {
		println!("I'm eating.");
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

I think that's enough on rust for now. As we'll see later this trait structure can help us provide contraqts for our Machine Learning models.

### What is Machine Learning?

<q>Field of study that gives computers the ability to learn without being explicitly programmed.</q> - Arthur Samuel

I think nowadays most people have a loose idea of what Machine Learning means. I wont spend too much time explaining the concept but will instead explain some terminology and give a concrete example.

In ML we have Models. These Models represent a set of hypotheses which can be used to explain a pattern which is present in some data. By giving the model some data we can train it and have it learn this pattern. The model can then be used to make predict the patterns from data it hasn't seen before.

Key take aways are:

- Model : An ML object representing a set of hypothesis.
- Train : The act of teaching a model from data (learning).
- Predict : Using the learnt model to predict the pattern of unseen data.

**(Do hypotheses here make things too complicated?)**

Let's consider a concrete example: the Logistic Regression model. This model is used for classification - we have some data points (lists of real-valued features) and a description of which class the point belongs to. For example you may consider images that may or may not contain cats. **(Expand on this, or pick a better way of describing.)**

We can train our logistic regression model using a technique known as Gradient Descent Optimization. This will allow the model to learn what the underlying parameters are - which we can use to make future predictions.

### How does Rust help with Machine Learning?

Trait structure allows us to create well structured high-level code whilst maintaining low-level performance. Hopefully we can find a way to provide developers access to fast iterative machine learning with a good view to performance and scaling - something which is not very attainable currently.

## Rusty-Machine

General purpose machine learning library. Implemented entirely in rust (which has upsides and downsides, LAPACK, BLAS).

### How do we use traits?

The trait set up allows users to easily implement a model with simple access to the necessary components. It is also very easy to modify the models, updating parameters and even swapping out components (such as the GD algorithm).

Users can easily implement their own GD algorithms, or model frameworks and plug these into the rest of the jigsaw.

### General but varied models

Using the trait models we can provide a contract for our machine learning models. Allows a huge amount of code reusability and continuity for users. (By continuity I mean that all models are used in the same way - `train` and `predict`).

Users can easily integrate their own models into the library. By using the same trait structure.

## The down sides

The library is immature. Though I think the vision is strong we're a long way off and lack a lot of key components for a ML library. Consistent data handling, visualizations and performance are all core areas that need a lot of work.

## Next steps

Improving on existing algorithms, some restructuring and separation, and addition of other useful tools.

Call for help and collaboration.
