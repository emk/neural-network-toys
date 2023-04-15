//! Various optimizers for the neural network.

use std::{fmt::Display, str::FromStr};

use anyhow::{anyhow, Result};
use ndarray::Array1;
use serde::Serialize;
use serde_json::json;

use crate::{layers::LayerStateMut, network::Network};

/// Available optimizers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OptimizerType {
    GradientDescent,
    AdamW,
}

impl Display for OptimizerType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            OptimizerType::GradientDescent => write!(f, "gd"),
            OptimizerType::AdamW => write!(f, "adamw"),
        }
    }
}

impl FromStr for OptimizerType {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "gd" => Ok(OptimizerType::GradientDescent),
            "adamw" => Ok(OptimizerType::AdamW),
            _ => Err(anyhow!("Unknown optimizer type: {}", s)),
        }
    }
}

impl Serialize for OptimizerType {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        serializer.serialize_str(&self.to_string())
    }
}

/// Metadata describing an optimizer, for inclusion in our history.
#[derive(Debug, Clone, Serialize)]
pub struct OptimizerMetadata {
    pub optimizer_type: OptimizerType,
    pub learning_rate: f32,
    #[serde(flatten)]
    pub extra: serde_json::Value,
}

/// Interface to optimizers.
pub trait Optimizer {
    /// Metadata about the optimizer.
    fn metadata(&self) -> OptimizerMetadata;

    /// Optimize a network.
    fn optimize(&mut self, network: &mut Network) -> Result<()>;
}

/// Our most basic optimizer, which can be used for any of stochastic gradient
/// descent, mini-batch gradient descent, or batch gradient descent.
pub struct GradientDescentOptimizer {
    learning_rate: f32,
}

impl GradientDescentOptimizer {
    /// Create a new gradient descent optimizer.
    pub fn new(learning_rate: f32) -> Self {
        GradientDescentOptimizer { learning_rate }
    }
}

impl Optimizer for GradientDescentOptimizer {
    fn metadata(&self) -> OptimizerMetadata {
        OptimizerMetadata {
            optimizer_type: OptimizerType::GradientDescent,
            learning_rate: self.learning_rate,
            extra: json!({}),
        }
    }

    fn optimize(&mut self, network: &mut Network) -> Result<()> {
        for layer_state in network.network_state_mut() {
            let LayerStateMut { mut params, grad } = layer_state;
            params.assign(&(&params.view() - &(&grad * self.learning_rate)));
        }
        Ok(())
    }
}

/// The state that Adam tracks about each layer.
///
/// This is the point-wise momentum and variance for each parameter, calculated
/// as an exponential moving average. These are the "biased" values, because
/// they start with a value of 0.0. The variance, in particular, only updates
/// away from 0.0 _very_ slowly.
struct AdamWLayerState {
    /// Point-wise momentum.
    m: Array1<f32>,
    /// Point-wise variance.
    v: Array1<f32>,
}

/// Build an `AdamWOptimizer`, using reasonable defaults.
pub struct AdamWOptimizerBuilder {
    learning_rate: f32,
    beta1: f32,
    beta2: f32,
    epsilon: f32,
    lambda: f32,
}

#[allow(dead_code)]
impl AdamWOptimizerBuilder {
    /// Create a new `AdamWOptimizerBuilder`.
    pub fn new() -> Self {
        AdamWOptimizerBuilder {
            learning_rate: 0.001,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
            lambda: 0.01,
        }
    }

    /// Set the learning rate.
    pub fn learning_rate(mut self, learning_rate: f32) -> Self {
        self.learning_rate = learning_rate;
        self
    }

    /// Set the beta1 parameter.
    pub fn beta1(mut self, beta1: f32) -> Self {
        self.beta1 = beta1;
        self
    }

    /// Set the beta2 parameter.
    pub fn beta2(mut self, beta2: f32) -> Self {
        self.beta2 = beta2;
        self
    }

    /// Set the epsilon parameter.
    pub fn epsilon(mut self, epsilon: f32) -> Self {
        self.epsilon = epsilon;
        self
    }

    /// Set the lambda parameter.
    pub fn lambda(mut self, lambda: f32) -> Self {
        self.lambda = lambda;
        self
    }

    /// Build the optimizer.
    pub fn build(self, network: &mut Network) -> AdamWOptimizer {
        AdamWOptimizer::new(
            network,
            self.learning_rate,
            self.beta1,
            self.beta2,
            self.epsilon,
            self.lambda,
        )
    }
}

/// An optimizer that implements the AdamW algorithm.
///
/// The AdamW algorithm is described in the paper ["Decoupled Weight Decay
/// Regularization][adamw] by Loshchilov and Hutter. It is a variant of the
/// Adam algorithm that decouples the weight decay from momentum and variance
/// calculations.
///
/// [adamw]: https://arxiv.org/abs/1711.05101
pub struct AdamWOptimizer {
    /// The current iteration number. We store this as a `i32` because that's
    /// all the `powi` function can handle. We should probably be careful not to
    /// overflow this if we use a small batch size and a lot of epochs.
    t: i32,

    /// The learning rate of the optimizer. A typical value for this is 0.001.
    learning_rate: f32,

    /// The exponential decay rate for the momentum. A typical value for this is 0.9,
    /// which corresponds very roughly to 10-20 iterations.
    beta1: f32,

    /// The exponential decay rate for the variance. A typical value for this is 0.999,
    /// which corresponds very roughly to 1000-2000 iterations.
    beta2: f32,

    /// A small constant to avoid division by zero. A typical value for this is 1e-8.
    epsilon: f32,

    /// How much should we decay our parameters by? A typical value for this is 0.01.
    lambda: f32,

    /// The state that Adam tracks about each layer.
    layer_states: Vec<AdamWLayerState>,
}

impl AdamWOptimizer {
    /// Create a new AdamW optimizer.
    pub fn new(
        network: &mut Network,
        learning_rate: f32,
        beta1: f32,
        beta2: f32,
        epsilon: f32,
        lambda: f32,
    ) -> Self {
        // Initialize the layer states to zero. We need the network to do this,
        // because we need to know the number and shape of the parameters in
        // each layer.
        //
        // We use a mutable `Network` here because there is no `network_state`
        // method that returns an immutable view into the network state. We
        // could add one, but that means more code.
        let mut layer_states = Vec::new();
        for layer_state in network.network_state_mut() {
            layer_states.push(AdamWLayerState {
                m: Array1::zeros(layer_state.params.raw_dim()),
                v: Array1::zeros(layer_state.params.raw_dim()),
            });
        }

        AdamWOptimizer {
            t: 0,
            learning_rate,
            beta1,
            beta2,
            epsilon,
            lambda,
            layer_states,
        }
    }
}

impl Optimizer for AdamWOptimizer {
    fn metadata(&self) -> OptimizerMetadata {
        OptimizerMetadata {
            optimizer_type: OptimizerType::AdamW,
            learning_rate: self.learning_rate,
            extra: json!({
                "beta1": self.beta1,
                "beta2": self.beta2,
                "epsilon": self.epsilon,
                "lambda": self.lambda,
            }),
        }
    }

    fn optimize(&mut self, network: &mut Network) -> Result<()> {
        // Keep track of the iteration number. If we overflow, we'll just
        // stick with the largest value that we can represent, which will
        // cause the bias correction terms to be arbitrarily close to 1.0.
        self.t = self.t.saturating_add(1);

        // The bias correction terms are used to correct the initial values of
        // the momentum and variance, which are both initialized to 0.0. There's
        // a really clever derivation of this. And it's a precise correction!
        // But the derivation is a bit long and complicated.
        let bias_correction_1 = 1.0 - self.beta1.powi(self.t);
        let bias_correction_2 = 1.0 - self.beta2.powi(self.t);

        // Iterate over the layers in the network.
        for (layer_state, layer_state_adam) in network
            .network_state_mut()
            .into_iter()
            .zip(self.layer_states.iter_mut())
        {
            let LayerStateMut { mut params, grad } = layer_state;
            let AdamWLayerState { m, v } = layer_state_adam;

            // Update the momentum.
            m.assign(&(&m.view() * self.beta1 + &grad.view() * (1.0 - self.beta1)));

            // Update the variance.
            v.assign(
                &(&v.view() * self.beta2
                    + &(&grad.view() * &grad.view()) * (1.0 - self.beta2)),
            );

            // Correct for the initialization bias.
            let m_hat = &m.view() / bias_correction_1;
            let v_hat = &v.view() / bias_correction_2;

            // Update the parameters.
            params.assign(
                &(&params.view()
                    - self.learning_rate
                        * &(&m_hat / (v_hat.mapv(f32::sqrt) + self.epsilon))
                    - &params.view() * self.lambda),
            );
        }
        Ok(())
    }
}
