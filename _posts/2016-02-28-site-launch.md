---
layout: post
title: "Launching this site"
date: 2016-02-28
---

Demo post marking the launch of this site!

I'll be using this blog to talk about my coding projects and other things. The first on this list will be [rusty-machine](https://github.com/AtheMathmo/rusty-machine).

I'll be describing machine learning and how rust can help.

```rust
pub trait Model<T, U> {
	pub fn train(inputs: T, targets: U);

	pub fn predict(inputs: T) -> U;
}
```