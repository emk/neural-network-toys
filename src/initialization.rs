/// Random initialization functions for layer weights.
use ndarray::Array2;
use ndarray_rand::{
    rand_distr::{Normal, Uniform},
    RandomExt,
};

/// Different ways to initialize weights in a layer.
pub enum InitializationType {
    /// See `[xavier]`.
    Xavier,
    /// See `[he]`.
    He,
}

impl InitializationType {
    /// Initialize a weight matrix for a layer.
    pub fn weights(&self, input_size: usize, output_size: usize) -> Array2<f32> {
        match self {
            InitializationType::Xavier => xavier(input_size, output_size),
            InitializationType::He => he(input_size, output_size),
        }
    }
}

/// [Xavier][] initialization is a good default for initializing weights in a
/// layer, with the [exception][] of layers using ReLU or SELU activations.
///
/// [Xavier]: http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf
/// [exception]: https://stats.stackexchange.com/a/393012
pub fn xavier(input_size: usize, output_size: usize) -> Array2<f32> {
    let weights = Array2::random((input_size, output_size), Uniform::new(-1.0, 1.0));
    weights / (input_size as f32).sqrt()
}

/// [He][] initialization is a good default for initializing weights in a layer
/// with ReLU or SELU activations.
///
/// [He]: https://arxiv.org/abs/1502.01852
pub fn he(input_size: usize, output_size: usize) -> Array2<f32> {
    Array2::random(
        (input_size, output_size),
        Normal::new(0.0, (2.0 / input_size as f32).sqrt())
            .expect("invalid normal distribution"),
    )
}
