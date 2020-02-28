---
# Documentation: https://sourcethemes.com/academic/docs/managing-content/

title: "Implementing Self-supervised GAN with Mimicry"
subtitle: ""
summary: ""
authors: []
tags: []
categories: []
date: 2020-02-21T21:59:04Z
lastmod: 2020-02-21T21:59:04Z
featured: false
draft: true

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
## Introduction
In this tutorial, we will learn how to implement a customised GAN with Mimicry [link] and evaluate it using multple metrics. As an example, we demonstrate the implementation of the Self-supervised GAN (SSGAN) [cite][link] and train/test it on the CIFAR-10 dataset. SSGAN is of interest since at the time of this writing, it is the current state-of-the-art unconditional GAN, and since it does not follow the typical GAN training pattern of using just 1 objective, implementing it could demonstrate how a customised GAN can be implemented.

## Self-supervised GAN
{{< figure src="images/ssgan.png" 
title="Overview of SSGAN. The original GAN task for the discriminator is kept the same, with the discriminator seeing only upright image to solve this task. However, the discriminator now has an additional task of classifying 1 of 4 rotations of incoming real/fake images."
numbered="true"
lightbox="true" >}}

<!-- [![SSGAN_pic](https://i.imgur.com/EgJAbHMl.png "SSGAN_pic")](https://i.imgur.com/EgJAbHMl.png "SSGAN_pic")
 -->
The key idea of SSGAN lies in creating a new task for both the discriminator to solve: the GAN and self-supervised task, where the latter is as simple as predicting the rotations of images rotated in 1 of 4 directions randomly. We note that while this self-supervised task falls on the side of the discriminator, the resulting losses from real and fake images goes to the discriminator and generator respectively, as we shall see later.

Why is predicting rotations a self-supervised objective? The idea is that there are no explicit labels required from humans, and simply performing a transformation on the data is sufficient to obtain a label (or "pseudo-label") that describes the data. For example, given an image, we can rotate it by 0 degrees, call it class 0, or rotate it by 90 degrees and call it class 1. We can do this for 4 different rotations, so that the set of degrees of rotation $\mathcal{R} = \\{0^\circ, 90^\circ, 180^\circ, 270^\circ\\}$ gives classes $0$ to $3$ respectively (note the zero-indexing). This is unlike in datasets like ImageNet, where humans are required to explicitly label if an image represents a cat or a dog. Here, our classes are arbitrarily designed, but individually correspond to one of 4 rotations (e.g. class 3 would describe a 270 degrees rotated image). Formally, we can represent the losses for the generator (G) and discriminator (D) as such:

$$L_G = -V(G, D) - \alpha \mathbb{E}\_{x\sim P_G} \mathbb{E}\_{r \sim \mathcal{R}}[\log Q_D(R=r \vert x^r)]$$
$$L_D = V(G, D) - \beta \mathbb{E}\_{x\sim P_{\text data}} \mathbb{E}\_{r \sim \mathcal{R}}[\log Q_D(R=r \vert x^r)]$$

where V represents the GAN value function, $P_G$ represents the generated image distribution, $x^r$ represents a rotated image, and $Q_D$ represents the classification predictive distribution of a given rotated image, which we can obtain using an auxiliary, fully-connected (FC) layer.

Understanding the above 2 equations from the paper is *important*, since it explicitly tells us that only the rotation loss from the fake images is applied for the generator, and correspondingly, only the rotation loss from real images is applied for the discriminator. Intuitively, this makes sense since we do not want the discriminator to be penalised for fake images that not look realistic, and the generator should strive to produce images that look natural enough to be rotated in a way that is easy for the auxiliary classifier to solve. 

While this method of predicting rotations is simple, it has seen excellent performance in representation learning (Gidaris 2018 [link]). The insight from SSGAN is that one could use this self-supervised objective to alleviate a key problem in GANs: Catastrophic Forgetting [cite] of the discriminator. The paper presents further insights into this, for which the reader is highly encouraged to read.

However, can we simplify the equations into a form that is easier to implement? Indeed, we can further express it in the following form for some optimal discriminator $\hat{D}$ and optimal generator $\hat {G}$.

$$L_G = L_{\text{GAN}}(G, \hat{D}) + \alpha L_{SS}$$
$$L_D = L_{\text{GAN}}(\hat{G}, D) + \beta L_{SS}$$

where $L_{\text{GAN}}$ is simply the hinge loss for GANs, and $L_{SS}$ is the self-supervised loss for the rotation, which we can implement as a standard cross entropy loss. In Mimicry, the hinge loss is already implemented, so we only need to implement the self-supervised loss.



## Implementing Models
TODO: Avoid using base class here for simplicity. Point to conclusion for a demonstration of how to implement a base class.

In order to **maximize code reuse**, we will implement **Base SSGAN classes** that defines the key functionalities of SSGAN. Doing so, we can easily extend this base class for multiple resolutions (e.g. 32x32, 64x64), since there are particular well-defined GAN structures used in research that have particularly shown to work for standard resolutions, which we can adopt. 

In Mimicry, we have a `BaseGenerator` or `BaseDiscriminator` class that is a children of a `BaseModel` class (which contains basic model functions like checkpointing), but contains functions specific for building an unconditional generator/discriminator. Taking the generator as an example, the order of inheritance will look like so:
`BaseModel` -> `BaseGenerator` -> `SSGANBaseGenerator` -> `ResNetGenerator`.

### Discriminator
TODO: Maybe add diagram of auxiliary network.

We first create a `SSGANBaseDiscriminator` class inheriting from `BaseDiscriminator` class,, and this consists of two main steps:
1. Initialise key variables for SSGAN: we have 4 self-supervised classes due to the 4 rotations, and for simplicity, we set our SS task loss scaling (beta) to be 1.0
2. The GAN loss used for SSGAN is the hinge loss, so we initialise it as so in our parent `__init__` function.
```
class SSGANBaseDiscriminator(gan.BaseDiscriminator):
    """
    Base discriminator for SSGAN.
    """
    def __init__(self, num_classes=4, ss_loss_scale=1.0, loss_type='hinge', *args, **kwargs):
        super().__init__(
            loss_type=loss_type,
            *args, **kwargs)
        self.num_classes = num_classes
        self.ss_loss_scale = ss_loss_scale
```
We'll implement a forward function that assumes we have two outputs to be returned: the logits for the input images `x`, and the corresponding output logit classes for the SS rotation label.
As this is a base class, we simply return a `NotImplementedError` for now.
```
def forward(self, x):
	"""
	Feedforward function returning (a) output logits for GAN task, (b) output logits for SS task.
	"""
    raise NotImplementedError("Forward function to be inherited and implemented.")
```
Now, we need to define several functions to rotate our images. In SSGAN, the image batch is split into 4 quarters, and each quarter is rotated with a unique direction. For example, for a batch size of 64, the images 1-16 are 0 degrees rotated, images 17-32 are 90 degrees rotated, and so on. We define a function that can easily rotate a given image:

```
def _rot_tensor(self, image, deg):
    """
    Rotation for pytorch tensors using rotation matrix. Takes in a tensor of (C, H, W shape).
    """
    if deg == 90:
        return image.transpose(1, 2).flip(1)

    elif deg == 180:
        return image.flip(1).flip(2)

    elif deg == 270:
        return image.transpose(1, 2).flip(2)

    elif deg == 0:
        return image

    else:
        raise NotImplementedError("Function only supports 90,180,270,0 degree rotation.")
```
For efficiency, we implemented the rotation function for tensors rather than converting them into numpy to use the default numpy functions for rotation, as this would mean the tensors might be sent from GPU to CPU and back again. Next, we simply implement a batchwise rotation function:
```
def rotate_batch(self, images):
    """
    Rotate a quarter batch of images in each of 4 directions.
    """
    N, C, H, W = images.shape

    # Give the first 16 images label 0, next 16 label 1 etc.
    choices = [(i, i * 4 // N) for i in range(N)]

    # Collect rotated images and their labels
    ret = []
    ret_labels = []
    degrees = [0, 90, 180, 270]
    for i in range(N):
        idx, rot_label = choices[i]

        # Rotate images
        image = self._rot_tensor(images[idx], deg=degrees[rot_label])  # (C, H, W) shape
        image = torch.unsqueeze(image, 0)  # (1, C, H, W) shape

        # Get labels accordingly
        label = torch.from_numpy(np.array(rot_label))  # Zero dimension
        label = torch.unsqueeze(label, 0)

        ret.append(image)
        ret_labels.append(label)

    # Concatenate images and labels to (N, C, H, W) and (N, ) shape respectively.
    ret = torch.cat(ret, dim=0)
    ret_labels = torch.cat(ret_labels, dim=0).to(ret.device)

    return ret, ret_labels
```        
To obtain the SS loss, we build yet another function for the discriminator to obtain some losses given s
```
def compute_SS_loss(self, images, scale):
    """
    Function to compute SS loss.
    """
    # Rotate images and produce labels here.
    images_rot, class_labels = self.rotate_batch(
        images=images)

    # Compute SS loss
    _, output_classes = self.forward(images_rot)

    err_SS = F.cross_entropy(
        input=output_classes,
        target=class_labels)

    # Scale SS loss
    err_SS = scale * err_SS

    return err_SS, class_labels


```

### Generator

We first create a class inheriting `BaseGenerator`, and input the GAN loss attributes we know beforehand, which is the hinge loss. We define it here, so in the event our GAN loss changes, it can apply to all variants of this model.
```
class SSGANBaseGenerator(gan.BaseGenerator):
    def __init__(self, SS_loss_scale_G=1.0, loss_type='hinge', *args, **kwargs):
        super().__init__(
            loss_type=loss_type, *args, **kwargs)
        self.SS_loss_scale_G = SS_loss_scale_G

```
As a safety step, we can raise an error when trying to create base model for feedforwarding (optional).
```
    def forward(self, x):
        raise NotImplementedError("Forward function to be inherited and implemented.")

```

Now, what's left is just defining the `train_step` function for the generator. 
```
    def train_step(
        self,
        real_batch,
        netD,
        optG,
        log_data,
        device,
        global_step=None):
        """
        Takes one train step for the generator.
        """
        # First zero the gradients before training
        self.zero_grad()

        # Get only batch size from real batch
        batch_size = real_batch[0].shape[0]

        # Produce fake images and logits
        noise = torch.randn((batch_size, self.nz), device=device)
        fake_images = self.forward(noise)
        output, _ = netD(fake_images)

        # Compute GAN loss, upright images only.
        errG = self.compute_gan_loss(
            output=output,
            device=device)

        # Compute SS loss, rotates the images.
        errG_SS, _ = netD.compute_SS_loss(
            images=fake_images,
            scale=self.SS_loss_scale_G)

        # Backprop and update gradients
        errG_total = errG + errG_SS
        errG_total.backward()
        optG.step()

        # Log statistics
        log_data.add_metric('errG', errG, group='loss')
        log_data.add_metric('errG_SS', errG_SS, group='loss_SS')

        return log_data

```        



## Dataset


## Final Code
The overall code required to build this customised GAN can be found here [link].

## Conclusion

Self-supervision is an extremely interesting topic, and I particularly enjoy the series of lectures given by Alexei Efros (see here [link]), who gives an 