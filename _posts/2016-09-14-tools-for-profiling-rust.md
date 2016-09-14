---
layout: post
title: Tools for profiling Rust
excerpt: "Another look at profiling Rust"
comments: true
---

The first time I needed to profile a Rust application I came across [Llogiq](https://llogiq.github.io/)'s post - [Profiling Rust applications on Linux](https://llogiq.github.io/2015/07/15/profiling.html). This post was incredibly helpful to me and succeeded in getting me off the ground. After spending a little more time with Rust and needing to do a little more profiling I've discovered a few alternatives that I think make valuable additions. In this post I'll be writing about these in the hope that someone else may stumble upon it and get off the ground.

This post is only relevant for profiling on Linux - though the `cpuprofiler` introduced _may_ work on any unix system.

_Llogiq's post remains a very valuable resource and you should probably read it if you are planning on profiling._

## What is profiling?

We use profiling to measure the performance of our applications - generally in a more fine-grained manner than benchmarking alone can provide. In fact, often with profiling we can reveal slow areas of code that we may not have suspected at all.

To build a profile we monitor the application as it runs and record various information. For example, cpu cache misses or total instructions within a function. This is a complicated task but fortunately there are a number of tools available to us - some of which are discussed in [Llogiq's blog post](https://llogiq.github.io/2015/07/15/profiling.html).

In this post I'll talk about some alternatives and more recent developments that make profiling Rust applications a little easier.


## [cargo-profiler](https://github.com/pegasos1/cargo-profiler)

Cargo profiler is an awesome tool from [pegasos1](https://github.com/pegasos1) that makes much of the profiling work easier. There are clear [installation instructions](https://github.com/pegasos1/cargo-profiler#to-install) in the project README.

Essentially cargo-profiler wraps some of the profilers described in Llogiq's blog post to provide an easier interface.

```
$ cargo profiler callgrind --release

Compiling nnet-prof in release mode...

Profiling nnet-prof with callgrind...

Total Instructions...41,302,042,769

11,723,736,639 (28.4%) ???:matrixmultiply::gemm::masked_kernel
-----------------------------------------------------------------------
4,991,600,000 (12.1%) ???:_..rusty_machine..learning..nnet..BaseNeuralNet....a$C$..T....::compute_grad
-----------------------------------------------------------------------
4,646,800,000 (11.3%) ???:_......a..rulinalg..vector..Vector..T....as..core..ops..Mul..T....::mul
-----------------------------------------------------------------------
4,251,600,000 (10.3%) ???:_..rusty_machine..learning..toolkit..regularization..Regularization..T....::l2_reg_grad
-----------------------------------------------------------------------
3,212,833,861 (7.8%) ???:matrixmultiply::gemm::dgemm
-----------------------------------------------------------------------
2,257,355,716 (5.5%) e_exp.c:__ieee754_exp_sse2
-----------------------------------------------------------------------
... Some excluded ...
```

Not only does cargo-profiler make it easier to use these profiling tools - it also lets you filter and sort the results. Below I take only the top 6 results by total memory reads (Dr).

```
$ cargo profiler cachegrind --release -n 6 --sort dr

Compiling nnet-prof in release mode...

Profiling nnet-prof with cachegrind...

Total Memory Accesses...57,464,647,631	

Total L1 I-Cache Misses...63,158,267 (0%)	
Total LL I-Cache Misses...894 (0%)	
Total L1 D-Cache Misses...472,654,833 (0%)	
Total LL D-Cache Misses...44,725 (0%)	

 Ir  I1mr ILmr  Dr  D1mr DLmr  Dw  D1mw DLmw
0.28 0.09 0.05 0.27 0.00 0.00 0.30 0.07 0.01 ???:matrixmultiply::gemm::masked_kernel
-----------------------------------------------------------------------
0.12 0.38 0.20 0.14 0.19 0.00 0.14 0.30 0.00 ???:_rusty_machine..learning..nnet..BaseNeuralNeta$C$T::compute_grad
-----------------------------------------------------------------------
0.08 0.00 0.06 0.10 0.03 0.00 0.12 0.22 0.01 ???:matrixmultiply::gemm::dgemm
-----------------------------------------------------------------------
0.11 0.02 0.01 0.09 0.13 0.00 0.13 0.10 0.00 ???:_arulinalg..vector..VectorTascore..ops..MulT::mul
-----------------------------------------------------------------------
0.10 0.03 0.01 0.07 0.11 0.00 0.06 0.09 0.00 ???:_rusty_machine..learning..toolkit..regularization..RegularizationT::l2_reg_grad
-----------------------------------------------------------------------
0.04 0.08 0.17 0.06 0.13 0.01 0.05 0.00 0.09 ???:nnet_prof::main
-----------------------------------------------------------------------
```

Right now cargo-profiler supports [valgrind](http://valgrind.org/) with the _callgrind_ and _cachegrind_ plugins.

## [cpuprofiler](https://github.com/AtheMathmo/cpuprofiler)

Cpuprofiler is a new library which provides bindings to google's [gperftools cpu profiler](https://github.com/gperftools/gperftools). The profiler is a statistical sampling profiler - which means that it records the function stack information at set intervals. This technique generally has a lower overhead than instrumentation profilers like [valgrind](http://valgrind.org/).

But the real winning points for me were the powerful output formats from this profiler and the ability to profile only chosen sections of code.

### How do I use it?

There are detailed [installation instructions](https://github.com/AtheMathmo/cpuprofiler#installation) in the project README. The short of it is:

#### Install gperftools (which includes the cpu profiler).

Installation instructions are in the [gperftools](https://github.com/gperftools/gperftools) repo.

#### Link to `cpuprofiler` in your `Cargo.toml` manifest

```
[dependencies]
cpuprofiler = "0.0.2"
```

#### `start` and `stop` the profiler around the code you'd like to profile.

```rust
use cpuprofiler::PROFILER;

// Unlock the mutex and start the profiler
PROFILER.lock().unwrap().start("./my-prof.profile").expect("Couldn't start");

// Code you want to sample goes here!
neuralnet.train(&inputs, &targets);

// Unwrap the mutex and stop the profiler
PROFILER.lock().unwrap().stop().expect("Couldn't stop");
```

The gperftools library only allows one profiler to be active at any time and so we capture this by making `PROFILER` a static `Mutex<Profiler>`. 

#### Use [pprof](https://github.com/google/pprof) to evaluate the output in a large number of formats!

I've been using a legacy version of pprof which is bundled with gperftools. It's worked well for me so far.

We can view the output as text:

```
Total: 855 samples
     207  24.2%  24.2%      207  24.2% matrixmultiply::gemm::masked_kernel::hfdb4f50027c4d91c
     156  18.2%  42.5%      853  99.8% _$LT$rusty_machine..learning..optim..grad_desc..StochasticGD....
      79   9.2%  51.7%       79   9.2% _$LT$$RF$$u27$a$u20$rulinalg..vector....
    ... Some excluded ...
```

Or we can use graphviz to get an interactive graph. You should be able to pan and zoom on the svg below.

<object data="{{ site.url }}/assets/nnet-profile.svg" type="image/svg+xml" style="height: 480px; width: 100%; border: 2px black; border-radius: 15px; margin: 10px;">
	<img src="{{ site.url }}/assets/pprof-gz.jpg" alt="pprof graph example">
</object>

And pprof provides a bunch of other options that I haven't tried yet - like profile comparisons. It's also probable that the outputs from
the newer version of pprof is even nicer!

## Summary

I've briefly discussed two tools which weren't covered (didn't exist) in the previously [linked post](https://llogiq.github.io/). I believe they start to provide a more
compelling story for profiling Rust - though of course there is still a way to go.

I have only tested the above on linux - though supposedly `cpuprofiler` will work on any unix system.

In the future I'd love to try and get a simple statistical sampling profiler working in Rust (using [libbacktrace](https://github.com/rust-lang/rust/tree/master/src/libbacktrace)). I'm certainly no expert and if anyone has any pointers on doing this I'd love to hear them.

Providing support within cargo-profiler for cpuprofiler would be a cool addition (I think!).
