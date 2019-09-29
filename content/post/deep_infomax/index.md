---
# Documentation: https://sourcethemes.com/academic/docs/managing-content/

title: "Deep InfoMax"
subtitle: "`[ICLR 2019]` Learning deep representations by mutual information estimation and maximization by Hjelm et al."
summary: "`[ICLR 2019]` Learning deep representations by mutual information estimation and maximization by Hjelm et al."
authors: []
tags: []
categories: [paper-review]
date: 2019-09-22T02:36:11+08:00
lastmod: 2019-09-22T02:36:11+08:00
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
## Overview
[Deep InfoMax](https://arxiv.org/abs/1808.06670) is a principled framework for performing mutual information maximization between data and its encoded features. This framework has achieved significant improvements in unsupervised visual representation learning, as evaluated by various metrics for measuring mutual information (MI), independence of representations, and perhaps most practically, the performance on a downstream task like image classification.

In short, the main idea of Deep InfoMax revolves around the following:

- **Maximize global MI**, the MI between an input data and the resulting global feature vector produced by the encoder.
- **Maximize local MI**, the MI between the local and global features produced by the encoder.
- **Enforce a statistical constraint** to prevent a trivial solution to the MI maximization objective.

This post will attempt to summarise the key ideas in this paper.

## Preliminaries
Before we begin, we briefly revisit certain information theoretic concepts. Mutual information between some random variables $X$ and $Y$ can be formally defined as the Kullback-Leibler (KL) divergence between the joint density of X and Y, $p(x,y)$, and the product of their marginal densities $p(x)$ and $p(y)$ respectively:

$$
\mathcal{I}(X; Y)
= D_{KL} (p(x,y) \mid\mid p(x)p(y))
= \mathbb{E}\_{p(x,y)} \left[ \log \frac{p(x,y)}{p(x)p(y)} \right]
$$

Intuitively, mutual information measures how much we can learn about the random variable X through observing random variable Y. 

In unsupervised representation learning, given some input data $X$, we aim to learn a *good feature representation* $E(X)$, using some encoder function $E$. In information theoretic terms, we aim to maximize the MI between $X$ and $E(X)$. This ensuing goal is called the **InfoMax objective**, as first coined by Linsker in 1988, and can be formally defined as:

$$\max_{E \in \mathcal{E}} I(X; E(X))$$

for some $E$ from function class $\mathcal{E}$. Intuitively, a good feature representation $E(X)$ should have *high mutual information* with the input data $X$, because observing this representation tells us a lot about the input $X$. If we represent $E$ as a neural network parameterised by weights $\psi$, we can consequently replace $E$ with $E\_{\psi}$. In this case, $E\_{\psi}(X)$ should ideally capture high-level information about $X$. This is often beneficial since $E_\psi(X)$ often lies in a lower dimensional latent space, and manipulating this learnt feature representation allows us to more easily build sophisticated models at a fraction of the computational cost. For instance, simply performing a nearest neighbour search of similar feature representations should retrieve similar looking images as well (as has been done frequently in literature).

In practice, computing the MI directly is not possible when the distributions are intractable. Thus, it is common to use an MI estimator and maximize on a lower bound of the MI estimator instead.

## Model
{{< figure src="images/main_model.png" 
title="Base encoder model of Deep InfoMax."
numbered="true"
lightbox="true" >}}

As the paper introduces quite a lot of new ideas, I find it easier to understand if we start with and keep in mind the *intended model* that we want to create: a **base encoder** that captures general representations of input images (see Figure 1). This is not very different from a typical feature extractor that is often pre-trained in a fully-supervised way. However, in Deep InfoMax, the base encoder is trained in an unsupervised manner through various MI maximization tasks, as will be explained later.


Concretely, suppose we represent our encoder $E_\psi$ as a neural network with parameters $\psi$, and our empirical data distribution is represented by $\mathbb{P}$, our training of the base encoder should aim to:

- Obtain the optimal $\psi$ that maximizes $\mathcal{I} (X; E_\psi(X))$.
- Avoid trivial solutions to the MI maximization objective: the marginal distribution of the encoded features $\mathbb{U}\_{\psi, \mathbb{P}}$ must be close to some statistical prior $\mathbb{V}$ to achieve desirable properties such as independence.

To achieve these goals, the tasks aim to manipulate the local features $C\_\psi(x)$ and global features $E\_\psi(x)$ for some input $x$, where $C_\psi$ is the local feature encoder part of the same encoder architecture as $E\_\psi$. For clarity, we can define:

$$E\_\psi = f\_\psi \circ C\_\psi $$

where $f\_\psi$ represents some intermediate layer(s) connecting the local and global feature encoders. Furthermore, note that $C\_\psi(x)$ is represented by an $M \times M$ feature map, and the global features $E_\psi(x)$ as a feature vector $Y$.



## Maximizing Mutual Information
This section will explain the main components of the Deep InfoMax network, but the global MI section will be used a stepping stone to explain the various methods of MI estimation and maximization.

### Global MI

{{< figure src="images/global_MI.png" 
title="Task to maximize global MI."
numbered="true"
lightbox="true" >}}

The first task is to maximize global MI, which is done by *training a classifier* that can distinguish a positive sample coming from some joint distribution $\mathbb{J}$ and negative samples from the product of marginals $\mathbb{M}$. Doing so, we maximize on the Donsker-Varadhan (DV) representation of KL-divergence, which is a lower bound of the true mutual information. That is, we have 
$
\mathcal{I}(X;Y) 
:=
\mathcal{D}\_{KL}(\mathbb{J} \mid\mid \mathbb{M})
\geq
\widehat{\mathcal{I}}\_{\omega}^{(\text{DV})}(X;Y) 
$.
This is required since we are unable to access the intractable, true joint and marginal distributions. Formally, for some random variables $X$ and $Y$, we would aim to maximize the MI of the form:

$$
\widehat{\mathcal{I}}\_{\omega}^{(\text{DV})}(X;Y) 
:= 
\mathbb{E}\_\mathbb{J}[T\_{\omega}(x, y)] - \log \mathbb{E}\_{\mathbb{M}}[e^{T\_{\omega}(x,y)}] $$

where we have a discriminator function $T_{\omega} : \mathcal{X} \times \mathcal{Y} \to \mathbb{R}$ modelled by a neural network with parameters $\omega$ to produce a scalar score given some inputs. However, in Deep InfoMax, the discriminator function is also composed of the encoder and the classifier, so it should be parameterized with $\psi$ as well. We can thus define the discriminator function in Deep InfoMax as:

$$T\_{\psi, \omega} = D\_{\omega} \circ g \circ (C\_\psi, E\_\psi)$$

where $g$ represents a function that converts the features produced by $C\_\psi$ and $E\_\psi$ into outputs which the classifier $D\_{\omega}$ can use to produce a scalar score.

Intuitively, this discriminator function will assign a high scalar score to a positive sample, and a low scalar score to a negative sample. This positive sample refers to a pair of `(local_feat, global_feat)` that comes from the *same image*, and a negative sample is a pair `(local_feat_other, global_feat)`, where we pair an image's global features with local features from *another image*. In practice, the features from other images in the *same batch* is used to build up the set of negative samples, for efficiency.

In total, we have now defined the encoder that produces us the local/global features, a classifier to evaluate the quality of these features, and a discriminator that is composed of both these architectures. We can thus define the Global MI maximization goal as:

$$
(\hat{\omega}, \hat{\psi})\_G = \arg\\!\max\_{\omega, \psi} \widehat{\mathcal{I}}\_{\omega}(X; E_\psi(X))
$$

where $G$ denotes global. The goal aims to obtain a set of optimal weights $\omega$ for the classifier and $\psi$ for the encoder, such that the MI between data $X$ and its encoded features $E_\psi(X)$ can be maximized.



###### Alternative Estimators
Apart from using KL divergence, Deep InfoMax further explores two other MI estimators: the **Jensen-Shannon Divergence (JSD) MI estimator**, and the **Information Noise-Contrastive Estimation (InfoNCE) MI estimator**, as first proposed by Oord et al. in the [Contrastive Predictive Coding](https://arxiv.org/abs/1807.03748) (CPC) paper, the latter of which has been used to obtain really good performances across recent works on MI maximization for representation learning. 

The JSD MI estimator takes the following form:
$$
\widehat{\mathcal{I}}^{(\text{JSD})}(X; E\_{\psi}(X))
:=
\mathbb{E}\_{\mathbb{P}} 
[-\text{sp}(-T\_{\psi, \omega}(x, E\_{\psi}(x)))] - 
\mathbb{E}\_{\mathbb{P} \times \tilde{\mathbb{P}}}
[\text{sp} (T\_{\psi, \omega}(x', E\_{\psi}(x)))]
$$

where $\text{sp}$ refers to the softplus function $\text{sp}(x) = \log (1 + e^x)$, and $x'$ is sampled from $\tilde{\mathbb{P}} = \mathbb{P}$. In principle, the JSD-based estimator works similarly as the DV-based estimator, since both maximizes the expected log-ratio of the joint distribution over the product of marginals. However, in practice, the authors have found the JSD MI estimator produces results that are more stable.

Furthermore, the InfoNCE MI estimator takes the form of the following:
$$
\widehat{\mathcal{I}}^{(\text{infoNCE})} (X; E\_{\psi}(X))
:=
\mathbb{E}\_{\mathbb{P}} \left[
T\_{\psi, \omega}(x, E\_{\psi}(x))
- \mathbb{E}\_{\tilde{\mathbb{P}}} \left[
\log \sum\_{x'} e^{T\_{\psi, \omega}(x', E\_\psi(x))}
\right]
\right]
$$

Intuitively, the probability produced for a positive sample should be the highest, in *contrast* to the probabilities given to all other negative samples. This probability is computed by taking the softmax across all the logits/scores produced for each sample. The InfoNCE loss could then be obtained by computing the negative log (softmax) probability of the positive sample across all samples. Note that the infoNCE loss is but a *lower bound* of the MI estimator, for which the proof can be found in the CPC paper. 

###### Architecture
Till now, we have talked about having a classifier produce a score, but how is this actually done in practice? As explained in the Appendix of the paper, this can be done by simply flattening the local feature map $C\_\psi(x)$, concatenating it with the global features $E\_\psi(x)$, and then passing the resulting feature vector through the following network:

{{< figure src="images/global_MI_dis.png" 
title="Global DIM discriminator network"
numbered="true"
lightbox="true" >}}

As expected, the final output is a scalar score.

<!-- -------------- -->
### Local MI
{{< figure src="images/local_MI.png" 
title="Task to maximize local MI."
numbered="true"
lightbox="true" >}}

Apart from maximizing global MI, the framework also maximizes local MI -- the mutual information between the global features and local patch features. Here, the global features remain the same as before, but now rather than pairing the *entire* local feature map with one global feature vector and considering it as one positive/negative sample, we now pair a local patch feature vector obtained from the local feature map with the global feature vector instead. For example, a $1 \times 1$ spatial size vector obtained from the $M\times M$ feature map is paired with the global feature vector instead. The MI maximization goal can now be defined as:

$$
(\hat{\omega}, \hat{\psi})\_L = \arg\\!\max\_{\omega, \psi}
\frac{1}{M^2}\sum^{M^2}\_{i=1}
\widehat{\mathcal{I}}\_{\omega, \psi}(C\_{\psi}^{(i)}(X); E_\psi(X))
$$

where $L$ similarly denotes local. Doing so, we are able to further encode the representations of data that are shared across *all patches*, rather than encoding representations from just one "huge patch" that is the original image. This allows the encoder to avoid encoding representations from just certain patches in an image that solves the MI maximization objective more easily, since the representations have to be common amongst all patches (and thus high MI with every patch). Unsurprisingly, this objective has been shown in the paper to perform the best in downstream tasks, as the representations are less sensitive to certain noisy patches.

In fact, this objective has several commonalities with other highly competitive works like CPC. In CPC, the local patch features are to be predicted based on a masked context feature, that is, the context does not reveal anything about the local patch feature to predict -- after all, this is an autoregressive model. The ensuing context features -- which can be similarly viewed as the "global" feature, would thus need common representations across all local features it is supposed to predict. The paper further makes some detailed comparison with CPC and the reader is highly encouraged to have a read.

##### Architecture
{{< figure src="images/local_MI_A.png" 
title="Concat-and-convolve Local DIM network."
numbered="true"
lightbox="true"
style="width: 50px;" >}}

{{< figure src="images/local_MI_B.png" 
title="Encoder-and-dot Local DIM network."
numbered="true"
lightbox="true" >}}

To convert the paired samples into a scalar score, there are two methods explored in the paper: the Concat-and-convolve and the Encoder-and-dot designs, the latter of which produces the best performance in terms of maximizing MI.

In the **Concat-and-convolve** design (see Figure 5), one simply concatenates the global feature at each grid location in the local feature map, and perform a $1 \times 1$ convolution (via a discriminator network) to produce an $M \times M$ map of scalar scores, which has the same spatial size as the local feature map. Note that each score in the map corresponds to a score for a local patch feature vector-global feature vector pair.

On the other hand, the **Encoder-and-dot** design (see Figure 6) is a lot more interesting, since it requires one to *project* the local/global features to a higher **Reproducing Kernel Hilbert Space (RKHS)**, or in other words, a higher dimensional latent space. If this latent space is high dimensional enough, one would be able to capture properties of the data through a linear classifier, which would not have been possible at a lower dimensional latent space. Figure 7 gives an example I thought was straightforward to understand.

To do this projection, one must be able to *non-linearly* project it to a higher dimensional space: that is, one simple linear layer would not do, after all this would just amount to a linear scaling of the features! To introduce this non-linearity, we can simply project the features via a `Linear-ReLU-Linear` shallow MLP network, but Deep InfoMax framework further augments this MLP network with a linear shortcut. While Figure 6 shows a separate FC and convolutional networks for the global and local features respectively, in practice they perform the same role and have the same depth and width (e.g. 1 hidden layer, 2048 units per layer). The difference simply comes from the fact that we are dealing with a feature map for the local features, and not a vector anymore.

In fact, the concept of RKHS has made me rethink *a lot* about designing neural network architectures: if one has to relate features or variables in some way, simply projecting the features to a higher dimensional space and computing their dot products could capture similarity between these features in a more meaningful way. In fact, representations of either text, image, audio, video data can all be similarly captured through *feature vectors*, and now we have an extra tool manipulate these features for greater flexibility in modelling.

{{< figure src="images/rkhs.png" 
title="Example of XOR points in 2D (left) versus 3D (right). When in 2D, no linear classifier can separate the red from the blue points, but this is possible when one projects the XOR points to 3D, thus showing the benefits of projecting data to a higher dimensional feature space. [Credits to Arthur Gretton.](http://www.gatsby.ucl.ac.uk/~gretton/coursefiles/lecture4_introToRKHS.pdf)"
numbered="true"
lightbox="true" >}}

### Statistical Constraint

We finally introduce the statistical constraint that is one of the goals to simultaneously train the model on, apart from maximizing local/global MI. The constraint requires the encoded feature distribution (in this case, the global features) to be close to some statistical prior distribution, using the classic minimax loss found in GANs to measure the divergence $\mathcal{D}(\mathbb{V} \mid\mid \mathbb{U}\_{\psi, \mathbb{P}})$.

$$
(\hat{\omega}, \hat{\psi})\_P = 
\arg\\!\min\_{\psi} \arg\\!\max\_{\phi}
\widehat{\mathcal{D}}\_{\phi}(\mathbb{V} \mid\mid \mathbb{U}\_{\psi, \mathbb{P}}) = \mathbb{E}\_{\mathbb{V}}[\log D\_{\phi}(y)] +
\mathbb{E}\_{\mathbb{P}}[\log (1 - D\_{\phi}(E\_{\psi}(x)))]
$$

By inducing this constraint, the feature distribution will not come to be too close to the empirical data distribution -- which could still maximize the MI objectives, but do not actually maximize MI with the true data distribution. Interestingly, in practice, the authors have found that using a uniform distribution as the prior produces the best performance.

##### Architecture

{{< figure src="images/prior_constraint.png" 
title="Prior constraint network to match the distribution of encoded features to a statistical prior."
numbered="true"
lightbox="true" >}}

As seen in Figure 8, one simply produces a "fake" sample from sampling the prior distribution a feature vector of the same size as $Y$. A shallow MLP network then acts as a discriminator to product a scalar logit, which can then produce the probability of real/fake (e.g. via a sigmoid function).

### Final Objective

In total, we will now have 3 different objectives to train our base model on:

$$
\arg\max\_{\omega\_1, \omega\_2, \psi}(
A
+
B)
+
\arg\min\_{\psi}\arg\max\_{\phi}
C
$$

$$
A = \alpha \widehat{\mathcal{I}}\_{\omega\_1, \psi}(X; E\_{\psi}(X))
$$

$$
B = \frac{\beta}{M^2}\sum^{M^2}\_{i=1}
\widehat{\mathcal{I}}\_{\omega\_2, \psi}(X^{(i)}; E\_{\psi}(X))
$$

$$
C = 
\gamma \widehat{\mathcal{D}}\_{\phi}(\mathbb{V} \mid\mid \mathbb{U}\_{\psi, \mathbb{P}})
$$

where $A$, $B$, $C$, represents the global, local MI objectives and the prior matching constraint respectively; $\omega\_1$, $\omega\_2$ represents discriminator parameters for the global and local objectives; and $\alpha$, $\beta$, $\gamma$ represents hyperparameters to tune the importance of each objective.

In practice, the authors have found that setting $\alpha = 0$, $\beta=1$ and $\gamma=0.1$ achieves the best downstream task performance. For instance, on image classification, simply performing a linear evaluation on the learnt representations - that is, not finetuning the pre-trained encoder and only training an additional linear layer - achieves excellent accuracy on various image classification datasets (e.g. CIFAR10, STL-10). This is interesting since it highlights the importance of the Local DIM objective as explained earlier, and that the model can perform very well for downstream tasks even without the global MI objective (setting $\alpha = 0$). However, it should be noted that using the global DIM objective will improve the quantity of MI as evaluated by various metrics in the paper.

<!-- $$
\arg\max\_{\omega\_1, \omega\_2, \psi}(
\alpha \widehat{\mathcal{I}}\_{\omega\_1, \psi}(X; E\_{\psi}(X))\\
+
\frac{\beta}{M^2}\sum^{M^2}\_{i=1}
\widehat{\mathcal{I}}\_{\omega\_2, \psi}(X^{(i)}; E\_{\psi}(X))
+
\arg\min\_{\psi}\arg\max\_{\phi}
\gamma \widehat{D}\_{\phi}(\mathbb{V} \mid\mid \mathbb{U}\_{\psi, \mathbb{P}})
)
$$ -->

## Conclusion
This post merely aims to present the main idea of Deep InfoMax, but there is a lot more material in the original paper, which the reader is highly encouraged to read. Some further thoughts of unsupervised pre-training: it is interesting to approach the subject from the point of view of *initialization*. In such pre-training, one essentially obtains a set of hopefully near-optimal weights that could be used for a variety of downstream tasks, and in fact, if one was (really) lucky, we can even random initialize the weights without any pre-training! 

Overall, this paper was highly inspiring - it was the first paper that introduced me a principled (and rigorous) way of performing mutual information maximization, and the ideas proposed has deeply influenced the way I think about my current research on GANs.

*Finally, please feel free to send me a note if you have found any inaccuracies in the post/feel there are some things that could be better worded/explained, or simply would like to have a chat -- I would be more than happy to hear from you!*


## References

1. [Hjelm, R. Devon, et al. "Learning deep representations by mutual information estimation and maximization." arXiv preprint arXiv:1808.06670 (2018).](https://arxiv.org/abs/1808.06670)

2. [Oord, Aaron van den, Yazhe Li, and Oriol Vinyals. "Representation learning with contrastive predictive coding." arXiv preprint arXiv:1807.03748 (2018).](https://arxiv.org/abs/1807.03748)

3. [Belghazi, Mohamed Ishmael, et al. "Mine: mutual information neural estimation." arXiv preprint arXiv:1801.04062 (2018).](https://arxiv.org/abs/1801.04062)

4. [Tschannen, Michael, et al. "On Mutual Information Maximization for Representation Learning." arXiv preprint arXiv:1907.13625 (2019).](https://arxiv.org/abs/1907.13625)

5. [Linsker, Ralph. "Self-organization in a perceptual network." Computer 21.3 (1988): 105-117.](https://ieeexplore.ieee.org/abstract/document/36/)

6. [Bachman, Philip, R. Devon Hjelm, and William Buchwalter. "Learning representations by maximizing mutual information across views." arXiv preprint arXiv:1906.00910 (2019).](https://arxiv.org/abs/1906.00910)
