# Flow Matching 

A from-scratch PyTorch R&D project exploring **Optimal Transport Flow Matching (OT-CFM)**, the current state-of-the-art framework for generative modeling.

## Mathematics

The goal of this project is to implement the equations described in the paper Flow Matching for Generative Modeling. Here, I will try to summarize the main mathematical equations in order to fully understand the problem and the implementation. For more details, please refer to the main papers (see References section).

Let $p_t: [0, 1] \times \mathbb{R}^d \to \mathbb{R}_+^*$ be a time-dependent probability density path, and $v_t: [0, 1] \times \mathbb{R}^d \to \mathbb{R}^d$ be a time-dependent vector field. The vector field $v_t$ generates a probability path $p_t$ via a flow $\phi_t: [0, 1] \times \mathbb{R}^d \to \mathbb{R}^d$, which is defined as the solution to the following ODE:

$$\frac{\partial}{\partial t}\phi_t(x) = v_t(\phi_t(x)) \quad \text{with} \quad \phi_0(x) = x$$

To parameterize this system for generative modeling, the vector field is defined using a neural network $v_t(x; \theta)$ with parameters $\theta$, which we optimize to track a target vector field $u_t$ that generates a path connecting noise $p_0$ to data $p_1 \approx p_{\text{data}}$. For such an optimization, the following loss function can be used to train a neural network:

$$\mathcal{L}_{\text{FM}}(\theta) = \mathbb{E}_{t \sim U[0,1], \, x \sim p_t(x)} \left[ \| v_t(x; \theta) - u_t(x) \|^2 \right]$$

However, the target vector field $u_t(x)$ and $p_t(x)$ are not known for the entire population, making this loss function useless in practice. The paper introduces a new loss function, using conditional probability:

$$\mathcal{L}_{\text{CFM}}(\theta) = \mathbb{E}_{t \sim U[0,1], \, x_1 \sim p_{\text{data}}, \, x \sim p_t(x|x_1)} \left[ \| v_t(x; \theta) - u_t(x|x_1) \|^2 \right]$$

Unlike $\mathcal{L}_{\text{FM}}$, which requires knowing the global, aggregate vector field $u_t(x)$, $\mathcal{L}_{\text{CFM}}$ regresses against simpler, closed-form conditional vector fields $u_t(x|x_1)$ operating on local data paths. The paper proves that $\nabla_\theta \mathcal{L}_{\text{FM}}(\theta) = \nabla_\theta \mathcal{L}_{\text{CFM}}(\theta)$, meaning that minimizing the localized conditional loss yields the exact same optimal parameters as minimizing the intractable global loss.

For Gaussian conditional probability paths defined as $p_t(x|x_1) = \mathcal{N}(x; \mu_t(x_1), \sigma_t(x_1)^2 I)$, the paper derives a specific, closed-form expression for the target conditional vector field, given by:

$$u_t(x | x_1) = \frac{\dot{\sigma}_t(x_1)}{\sigma_t(x_1)} \Big( x - \mu_t(x_1) \Big) + \dot{\mu}_t(x_1)$$

Where $\dot{\mu}_t(x_1)$ and $\dot{\sigma}_t(x_1)$ are the time derivatives $\frac{\partial}{\partial t}\mu_t(x_1)$ and $\frac{\partial}{\partial t}\sigma_t(x_1)$. This explicit formula allows us to compute the target velocity vector for any sample $x$ at time $t$ instantly during training without running an ODE solver or backpropagating through time.

In this repository, the selected probability path is defined by a simple linear interpolation between the source noise $x_0 \sim \mathcal{N}(0, I)$ and the target data $x_1$:

$$x_t = (1 - t)x_0 + t x_1$$

This corresponds to a Gaussian conditional probability path where:
*   $\mu_t(x_1) = t x_1 \implies \dot{\mu}_t(x_1) = x_1$
*   $\sigma_t(x_1) = 1 - t \implies \dot{\sigma}_t(x_1) = -1$

Plugging these exact derivatives into the closed-form conditional vector field equation yields:

$$u_t(x | x_1) = \frac{-1}{1 - t} \Big( x - t x_1 \Big) + x_1$$

Since $x = (1 - t)x_0 + t x_1$, $x$ can be subsituted in the equation:

$$u_t(x | x_1) = \frac{-1}{1 - t} \Big( (1 - t)x_0 \Big) + x_1 = x_1 - x_0$$

## Project Architecture

* `configs/`: Configuration files for hyperparameters and model settings.
* `scripts/`: Executable entry points for training, sampling, and evaluation.
* `notebooks/`: Notebooks for studying.
* `src/`: Main source code package.
  * `data/`: Handles all data-related modules, dataset loading (MNIST), and preprocessing.
  * `models/`: Contains the neural network architectures, forward passes, and custom layers (e.g., Time Embeddings).
  * `flow/`: Manages the core algorithmic logic for Flow Matching, including interpolation paths and target vector computation.
* `tests/`: Unit tests verifying mathematical correctness, tensor broadcasting, and module integrity.

## Tests

Basic unit testing can be launched with pytest:

```bash
pytest tests\
```

## References

```bibtex
@inproceedings{lipman2023flow,
  title={Flow Matching for Generative Modeling},
  author={Lipman, Yaron and Chen, Ricky T. Q. and Ben-Hamu, Heli and Nicklas, Maximilian and Le, Matt Le and Le, Matt and Grover, Aditya},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2023},
  url={[https://arxiv.org/abs/2210.02747](https://arxiv.org/abs/2210.02747)}
}
@article{tong2023improving,
  title={Improving and Generalizing Flow-Based Generative Models with Minibatch Optimal Transport},
  author={Tong, Alexander and Malkin, Nikolay and Huguet, Guillaume and Zhang, Yanrui and Rector-Brooks, Jarrid and Fatras, Kilian and Wolf, Guy and Bengio, Yoshua},
  journal={Transactions on Machine Learning Research},
  issn={2835-8856},
  year={2023},
  url={https://arxiv.org/abs/2302.00482}
}
```
