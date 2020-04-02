---
# Documentation: https://sourcethemes.com/academic/docs/managing-content/

title: "Introducing Mimicry"
subtitle: "Towards the reproducibility of GAN research."
summary: "A PyTorch Library for reproducible GAN research."
authors: []
tags: []
categories: []
date: 2020-04-02T03:10:45+01:00
lastmod: 2020-04-02T03:10:45+01:00
featured: true
draft: false

# Featured image
# To use, add an image named `featured.jpg/png` to your page's folder.
# Focal points: Smart, Center, TopLeft, Top, TopRight, Left, Right, BottomLeft, Bottom, BottomRight.
image:
  caption: ""
  focal_point: ""
  preview_only: false

# Projects (optional).
#   Associate this post with one or more of your projects.
#   Simply enter your project's folder or file name without extension.
#   E.g. `projects = ["internal-project"]` references `content/project/deep-learning/index.md`.
#   Otherwise, set `projects = []`.
projects: []
---
*GitHub page: [Mimicry](https://github.com/kwotsin/mimicry).*


## Overview

During the course of my research studying GANs, I have found it quite difficult to compare against the myriad of different GANs in the literature, which can be implemented differently and using different frameworks. Ideally, it is hoped that despite these differences, one could still achieve scores **close** to the reported values in the respective papers. Yet, in reality, GANs are remarkably sensitive to small differences in implementation -- take for example, when updating the discriminator multiple times per generator step, giving the discriminator the same batch of data five times instead of giving it a new batch each time, would produce noticeable differences in performance. 

Even when one nails down the implementation, there is yet another issue: currently, GAN papers often make use of [Fréchet Inception Distance (FID)](https://arxiv.org/abs/1706.08500) to quantitatively compare results, which seems objective. However, FID is noted to produce [highly biased estimates](https://arxiv.org/abs/1801.01401), which when one uses a small sample size (number of real and generated images) for computing FID, the true FID value would be severely underestimated. This is a problem, because when comparing to prior works that use smaller sample sizes, one could spuriously obtain large improvements from simply using a larger sample size. A previous large scale study done by [Kurach et al.](https://arxiv.org/abs/1807.04720) similarly noted how FID scores can be mismatched this way.

Thus, some time back, I decided to re-implement all models I'm going to compare against for my research, in order to ensure comparisons can be done the most fairly. To do this, I had to re-implement the original models and make sure they perform very close to the original scores, so I can be sure of their correctness. The resulting data became very useful for my research on InfoMax-GAN (my prior work), and I thought it might be a good idea to formalize the code base a bit more and release it as an open source work, in case people might find it useful.

The resulting work is now released as [Mimicry](https://github.com/kwotsin/mimicry), which is now available on GitHub and PyPI. Specifically, the library aims to provide the following:

- A set of [baselines scores](https://github.com/kwotsin/mimicry#baselines) for popular GANs that one could easily compare against, with transparent training hyperparameters and evaluation methods illustrated. For instance, when computing FID, the exact sample size is given. Currently, there are over 70 data points produced from 6 different GANs, 7 datasets and 3 metrics. It is hoped this could reduce the need for researchers to cross-cite many different papers for closest results.

- Standardized implementations of popular GANs, which I aim to produce scores as closely as possible. To do so, I included a section on [Reproducibility](https://github.com/kwotsin/mimicry#reproducibility) in the project page.

- A framework for researchers to focus on implementing GANs, rather than rewriting boilerplate code for training/evaluation. Currently, several popular metrics like FID, Kernel Inception Distance (KID) and Inception Score (IS) are supported. For *backwards compatibility* of the scores, I adopted the original TensorFlow implementations of these metrics, so scores produced can be compared with those in existing literature.

*Trivia: The logo is designed to look as if there are two distributions trying to be matched, which follows the idea of a generator trying to model the true data distribution of some data.*

## Challenges
While I endeavored to produce as many data points/results as possible, a natural constraint is the amount of resources (e.g. GPUs) and time. This project required a lot more software engineering knowledge than I had expected, and my lack of experience in this area meant that the work took a lot more time -- for instance, it took some time to think of how I should modularize many different components in the code. While I've done some research in this area, what I think is possibly the "easiest" way of using the library might differ from yours -- so expect some rough edges if you're using the library!

## Extensions

There's definitely a lot more I'm hoping to do with the library, particularly on expanding the scope of the problem sets and including more commonly used GANs. Currently, I've only focused on image synthesis, although it's conceivable that this could be expanded to other GAN tasks like image to image translation.

Finally, please feel free to contact me if you'd like to learn more about the work!