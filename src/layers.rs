// TASK: We are converting this file from Stochastic Gradient Descent to
// mini-batch gradient descent. To do this, we need to take many APIs
// that previously took an `Array1` or an `ArrayView1` and change them
// to take an `ArrayView2` or an `Array2`.  Our arrays will be stored
// `(examples, features)` instead of `(features,)`.
//
// We will work through the file from the top down, converting each API
// as we go.

use std::fmt::Debug;

use ndarray::{s, Array1, Array2, ArrayView2, ArrayViewMut1, ArrayViewMut2, Axis};
use ndarray_rand::rand::seq::SliceRandom;
use serde::Serialize;
use serde_json::{json, Value};

use crate::initialization::InitializationType;

/// The axis along which we store examples in an `Array2`.
///
/// This is the out set of brackets in `[[1, 2, 3], [4, 5, 6]]`.
const ARRAY2_EXAMPLES_AXIS: Axis = Axis(0);

/// The axis along which we store features in an `Array2`.
///
/// This is the inner set of brackets in `[[1, 2, 3], [4, 5, 6]]`.
const ARRAY2_FEATURES_AXIS: Axis = Axis(1);

/// Layer metadata. This is used to log information about our layers
/// as part of our training history.
#[derive(Debug, Clone, Serialize)]
pub struct LayerMetadata {
    /// The type of this layer.
    #[serde(rename = "type")]
    pub layer_type: String,

    /// The number of inputs to this layer.
    pub inputs: usize,

    /// The number of outputs from this layer.
    pub outputs: usize,

    /// The number of parameters in this layer.
    pub parameters: usize,

    /// Any extra metadata we want to store.
    #[serde(flatten)]
    extra: Value,
}

/// Mutable view into part of a layer's state, including both the parameters
/// and the gradient we computed using backpropagation. This allows the
/// optimizer to update the parameters without needing to know anything
/// about the layer's internals.
pub enum LayerStateMut<'a> {
    Array1 {
        params: ArrayViewMut1<'a, f32>,
        grad: ArrayViewMut1<'a, f32>,
    },
    Array2 {
        params: ArrayViewMut2<'a, f32>,
        grad: ArrayViewMut2<'a, f32>,
    },
}

/// A layer in our neural network.
///
/// This is designed to support mini-batch gradient descent, so all
/// functions take and return arrays with shape `(examples, features)`.
pub trait Layer: Debug + Send + Sync + 'static {
    /// Return a boxed clone of this layer.
    fn boxed_clone(&self) -> Box<dyn Layer>;

    /// The type of this layer.
    fn layer_type(&self) -> &'static str;

    /// Metadata for this layer. This is used to log information about our
    /// layers as part of our training history.
    fn metadata(&self, inputs: usize) -> LayerMetadata {
        // By default, assume layers have the same number of inputs and outputs.
        LayerMetadata {
            layer_type: self.layer_type().to_owned(),
            inputs,
            outputs: inputs,
            parameters: 0,
            extra: json!({}),
        }
    }

    /// Perform the foward pass through this layer, returning the output.
    fn forward(&self, input: &ArrayView2<f32>) -> Array2<f32>;

    /// An appropriate loss function for this layer. By default, this is mean
    /// squared error, but it can be overridden for particular layers.
    ///
    /// Our return value is a vector of length `examples`, where each element
    /// is the loss for that example.
    fn loss(&self, output: &ArrayView2<f32>, target: &ArrayView2<f32>) -> Array1<f32> {
        // Mean squared error.
        //
        // For each output, we want to compute:
        //
        // loss = 1/n * (output - target)^2
        let diff = output - target;

        // We want to compute the mean of each row, so we need to sum each row.
        diff.sum_axis(ARRAY2_FEATURES_AXIS)
            / (output.len_of(ARRAY2_FEATURES_AXIS) as f32)
    }

    /// The derivative of the loss function with respect to the output.
    fn dloss_doutput(
        &self,
        output: &ArrayView2<f32>,
        target: &ArrayView2<f32>,
    ) -> Array2<f32> {
        // loss          = 1/n * (output - target)^2
        // ∂loss/∂output = 2/n * (output - target)
        (2.0 / output.len_of(ARRAY2_FEATURES_AXIS) as f32) * (output - target)
    }

    /// Prepare this layer for a training step. This is normally a no-op, but
    /// dropout layers will use this to decide which neurons to drop.
    fn start_training_step(&mut self) {}

    /// Reset this layer to normal after a training step. This is normally a
    /// no-op, but dropout layers will use this to restore the weights of
    /// dropped neurons.
    fn end_training_step(&mut self) {}

    /// Perform the backward pass through this layer, returning `dloss_dinput`
    /// as input for the previous layer.
    ///
    /// We pass in ∂loss/∂output, and we return and ∂loss/∂input. We also compute
    /// ∂loss/∂biases and ∂loss/∂weights (when applicable) and store them.
    fn backward(
        &mut self,
        input: &ArrayView2<f32>,
        dloss_doutput: &ArrayView2<f32>,
    ) -> Array2<f32>;

    /// Provide mutable access to the parameters and gradients of this layer.
    fn layer_state_mut<'a>(&'a mut self) -> Vec<LayerStateMut<'a>> {
        vec![]
    }

    /// Update the weights and biases of this layer, and return `dloss_dinput`.
    fn update_parameters(&mut self, _learning_rate: f32) {}
}

/// Fully-connected feed-forward layer without an activation function.
#[derive(Debug, Clone)]
pub struct FullyConnectedLayer {
    /// Weight matrix.
    ///
    /// Shape: (input_size, output_size)
    weights: Array2<f32>,

    /// Biases for each neuron.
    ///
    /// Shape: (output_size,)
    biases: Array1<f32>,

    /// ∂loss/∂biases averaged over the batch.
    ///
    /// Shape: (output_size,)
    dloss_dbiases: Array1<f32>,

    /// ∂loss/∂weights averaged over the batch.
    ///
    /// Shape: (input_size, output_size).
    dloss_dweights: Array2<f32>,
}

impl FullyConnectedLayer {
    /// Contruct using random weights.
    pub fn new(
        initialization_type: InitializationType,
        input_size: usize,
        output_size: usize,
    ) -> Self {
        Self {
            weights: initialization_type.weights(input_size, output_size),
            biases: Array1::zeros(output_size),
            dloss_dbiases: Array1::zeros(output_size),
            dloss_dweights: Array2::zeros((input_size, output_size)),
        }
    }
}

impl Layer for FullyConnectedLayer {
    fn boxed_clone(&self) -> Box<dyn Layer> {
        Box::new(self.clone())
    }

    fn layer_type(&self) -> &'static str {
        "fully_connected"
    }

    fn metadata(&self, inputs: usize) -> LayerMetadata {
        //assert_eq!(inputs, self.weights.len_of(Axis(0)));
        LayerMetadata {
            layer_type: self.layer_type().to_owned(),
            inputs,
            outputs: self.weights.len_of(Axis(1)),
            parameters: self.weights.len() + self.biases.len(),
            extra: json!({}),
        }
    }

    fn forward(&self, input: &ArrayView2<f32>) -> Array2<f32> {
        input.dot(&self.weights) + &self.biases
    }

    fn backward(
        &mut self,
        input: &ArrayView2<f32>,
        dloss_doutput: &ArrayView2<f32>,
    ) -> Array2<f32> {
        // ∂loss/∂biases = ∂loss/∂output * ∂output/∂biases
        //               = dloss_doutput * 1
        self.dloss_dbiases = dloss_doutput
            .mean_axis(ARRAY2_EXAMPLES_AXIS)
            .expect("batch size > 0");

        // ∂loss/∂weights = ∂loss/∂output * ∂output/∂weights
        //                = ∂loss_doutput * input
        //
        // We need to do this manually, because the `outer` method is not
        // supported by ndarray.
        //
        // Let's work through this using Einstein tensor notation. This notation
        // has an implicit sum over repeated indices. For example:
        //
        //   a_{i,j} = b_{i,k} * c_{k,j}
        //
        // means that we sum over the repeated index `k`:
        //
        //   a_{i,j} = sum_{k} b_{i,k} * c_{k,j}
        //
        // We will use the following letters to represent our indices:
        //
        //   b: batch
        //   i: input
        //   o: output
        //
        // We will write indices in the order they appear in the array, so
        // we have:
        //
        // input_{b,i}
        // dloss_doutput_{b,o}
        //
        // And we want to compute:
        //
        // dloss_dweights_{i,o} = input_{b,i} * dloss_doutput_{b,o} / batch_size
        self.dloss_dweights = input.t().dot(dloss_doutput)
            / dloss_doutput.len_of(ARRAY2_EXAMPLES_AXIS) as f32;

        // ∂loss/∂input = ∂loss/∂output * ∂output/∂input
        //              = dloss_doutput * weights
        dloss_doutput.dot(&self.weights.t())
    }

    fn layer_state_mut<'a>(&'a mut self) -> Vec<LayerStateMut<'a>> {
        vec![
            LayerStateMut::Array1 {
                params: self.biases.view_mut(),
                grad: self.dloss_dbiases.view_mut(),
            },
            LayerStateMut::Array2 {
                params: self.weights.view_mut(),
                grad: self.dloss_dweights.view_mut(),
            },
        ]
    }

    fn update_parameters(&mut self, learning_rate: f32) {
        self.weights = &self.weights - learning_rate * &self.dloss_dweights;
        self.biases = &self.biases - learning_rate * &self.dloss_dbiases;
    }
}

/// A layer using the tanh activation function. This is a good default for
/// hidden layers.
#[derive(Debug, Clone)]
pub struct TanhLayer {}

impl TanhLayer {
    /// Initialize `TanhLayer`.
    pub fn new() -> Self {
        Self {}
    }
}

impl Layer for TanhLayer {
    fn boxed_clone(&self) -> Box<dyn Layer> {
        Box::new(self.clone())
    }

    fn layer_type(&self) -> &'static str {
        "tanh"
    }

    fn forward(&self, input: &ArrayView2<f32>) -> Array2<f32> {
        input.mapv(|x| x.tanh())
    }

    fn backward(
        &mut self,
        input: &ArrayView2<f32>,
        dloss_doutput: &ArrayView2<f32>,
    ) -> Array2<f32> {
        // ∂loss/∂input = ∂loss/∂output * ∂output/∂input
        //              = dloss_doutput * (1 - tanh^2(x))
        dloss_doutput * &(1.0 - input.mapv(|x| x.tanh().powi(2)))
    }
}

/// A layer using the ReLU activation function, which is simple to compute
/// and used in many state of the art neural networks.
#[derive(Debug, Clone)]
pub struct LeakyReluLayer {
    /// The slope of the leaky ReLU function at negative values.
    leak: f32,
}

impl LeakyReluLayer {
    /// Initialize `ReluLayer`.
    pub fn new(leak: f32) -> Self {
        Self { leak }
    }
}

impl Layer for LeakyReluLayer {
    fn boxed_clone(&self) -> Box<dyn Layer> {
        Box::new(self.clone())
    }

    fn layer_type(&self) -> &'static str {
        "leaky_relu"
    }

    fn metadata(&self, inputs: usize) -> LayerMetadata {
        LayerMetadata {
            layer_type: self.layer_type().to_owned(),
            inputs,
            outputs: inputs,
            parameters: 0,
            extra: json!({ "leak": self.leak }),
        }
    }

    fn forward(&self, input: &ArrayView2<f32>) -> Array2<f32> {
        input.mapv(|x| if x > 0.0 { x } else { x * self.leak })
    }

    fn backward(
        &mut self,
        input: &ArrayView2<f32>,
        dloss_doutput: &ArrayView2<f32>,
    ) -> Array2<f32> {
        // ∂loss/∂input = ∂loss/∂output * ∂output/∂input
        //              = dloss_doutput * (1 if x >= 0 else leak)
        //
        // The derivative of the ReLU function is technically undefined at x=0,
        // so we just use the derivative at x=0+ε.
        dloss_doutput * &input.mapv(|x| if x >= 0.0 { 1.0 } else { self.leak })
    }
}

/// A layer using the softmax activation function and categorical cross-entropy.
/// This is a good default for an output layer that chooses between multiple
/// discrete output values.
#[derive(Debug, Clone)]
pub struct SoftmaxLayer {}

impl SoftmaxLayer {
    /// Initialize `SoftmaxLayer`.
    pub fn new() -> Self {
        Self {}
    }
}

impl Layer for SoftmaxLayer {
    fn boxed_clone(&self) -> Box<dyn Layer> {
        Box::new(self.clone())
    }

    fn layer_type(&self) -> &'static str {
        "softmax"
    }

    fn forward(&self, inputs: &ArrayView2<f32>) -> Array2<f32> {
        // Softmax is exp(x) / sum(exp(x)) for each example in inputs.
        let mut outputs = inputs.mapv(|x| x.exp());
        let reciprocal_sums = outputs.sum_axis(ARRAY2_FEATURES_AXIS).mapv(|x| 1.0 / x);

        // We need to calculate:
        //
        //  outputs_{b,o} = outputs_{b,o} * reciprocal_sums_{b}
        for (mut output, reciprocal_sum) in
            outputs.outer_iter_mut().zip(reciprocal_sums.iter())
        {
            output *= *reciprocal_sum;
        }
        outputs
    }

    fn backward(
        &mut self,
        _input: &ArrayView2<f32>,
        dloss_doutput: &ArrayView2<f32>,
    ) -> Array2<f32> {
        // This seems suspiciously convenient, but see
        // https://towardsdatascience.com/derivative-of-the-softmax-function-and-the-categorical-cross-entropy-loss-ffceefc081d1
        dloss_doutput.to_owned()
    }

    fn loss(&self, output: &ArrayView2<f32>, target: &ArrayView2<f32>) -> Array1<f32> {
        // Categorical cross-entropy.
        (target * output.mapv(|x| x.ln())).sum_axis(ARRAY2_FEATURES_AXIS) * -1.0
    }

    fn dloss_doutput(
        &self,
        output: &ArrayView2<f32>,
        target: &ArrayView2<f32>,
    ) -> Array2<f32> {
        // See
        // https://towardsdatascience.com/derivative-of-the-softmax-function-and-the-categorical-cross-entropy-loss-ffceefc081d1
        output - target
    }
}

/// A layer that implments [dropout][] by randomly setting some of its inputs to
/// zero, to help prevent overfitting. This is only used during training.
///
/// [dropout]: https://www.cs.toronto.edu/~hinton/absps/JMLRdropout.pdf
#[derive(Debug, Clone)]
pub struct DropoutLayer {
    /// The probability of keeping an input.
    keep_probability: f32,

    /// The mask used to drop inputs. Kept neurons have a value of 1.0, and
    /// dropped neurons have a value of 0.0.
    mask: Array1<f32>,
}

impl DropoutLayer {
    /// Create a new `DropoutLayer` with the given keep probability.
    pub fn new(width: usize, keep_probability: f32) -> Self {
        Self {
            keep_probability,
            mask: Array1::ones(width),
        }
    }
}

impl Layer for DropoutLayer {
    fn boxed_clone(&self) -> Box<dyn Layer> {
        Box::new(self.clone())
    }

    fn layer_type(&self) -> &'static str {
        "dropout"
    }

    fn metadata(&self, inputs: usize) -> LayerMetadata {
        LayerMetadata {
            layer_type: self.layer_type().to_owned(),
            inputs,
            outputs: inputs,
            parameters: 0,
            extra: json!({ "keep_probability": self.keep_probability }),
        }
    }

    fn forward(&self, input: &ArrayView2<f32>) -> Array2<f32> {
        input * &self.mask
    }

    fn backward(
        &mut self,
        _input: &ArrayView2<f32>,
        dloss_doutput: &ArrayView2<f32>,
    ) -> Array2<f32> {
        // See https://deepnotes.io/dropout for a discussion of gradients and
        // dropout. Since we're scaling the output by the keep probability,
        // I guess we need to scale the gradient by the same amount?
        dloss_doutput * &self.mask
    }

    /// Randomize our mask for training.
    fn start_training_step(&mut self) {
        // We need to scale the mask, so that the expected value of the output
        // is the same as if we hadn't used dropout.
        let scaled = 1.0 / self.keep_probability;

        // We do this by setting exactly `keep_probability * width` values to 1.0,
        // and the rest to 0.0. Then we shuffle the array.
        let keep_count =
            (self.keep_probability * self.mask.len() as f32).round() as usize;
        self.mask.slice_mut(s![0..keep_count]).fill(scaled);
        self.mask.slice_mut(s![keep_count..]).fill(0.0);
        self.mask
            .as_slice_mut()
            .unwrap()
            .shuffle(&mut rand::thread_rng());
    }

    /// Reset our mask after a training step, for testing and inference.
    fn end_training_step(&mut self) {
        self.mask.fill(1.0);
    }
}

/// Activation functions we support.
#[derive(Debug, Clone, Copy)]
pub enum ActivationFunction {
    Tanh,
    LeakyReLU(f32),
    Softmax,
}

impl ActivationFunction {
    /// Construct a layer that applies this activation function.
    pub fn layer(&self) -> Box<dyn Layer> {
        match self {
            ActivationFunction::Tanh => Box::new(TanhLayer::new()),
            ActivationFunction::LeakyReLU(leak) => {
                Box::new(LeakyReluLayer::new(*leak))
            }
            ActivationFunction::Softmax => Box::new(SoftmaxLayer::new()),
        }
    }

    /// How should we initialize the weights for the layer feeding into this
    /// activation function?
    pub fn input_weight_inititialization_type(&self) -> InitializationType {
        match self {
            ActivationFunction::Tanh => InitializationType::Xavier,
            ActivationFunction::LeakyReLU(_) => InitializationType::He,
            ActivationFunction::Softmax => InitializationType::Xavier,
        }
    }
}

#[cfg(test)]
mod tests {
    use approx::{assert_relative_eq, relative_eq};
    use log::warn;
    use ndarray::{array, Array};
    use ndarray_rand::{rand_distr::Uniform, RandomExt};

    use super::*;

    /// Backwards compatibility for tests that were written before we had
    /// separate layers for fully connected layers and activation layers.
    #[derive(Clone, Debug)]
    struct FullyConnectedWithActivation<L: Layer + Clone> {
        fully_connected: FullyConnectedLayer,
        activation_layer: L,
    }

    impl<L: Layer + Clone> FullyConnectedWithActivation<L> {
        fn new(fully_connected: FullyConnectedLayer, activation_layer: L) -> Self {
            Self {
                fully_connected,
                activation_layer,
            }
        }

        fn new_simple(activation_layer: L) -> Self {
            let mut fully_connected =
                FullyConnectedLayer::new(InitializationType::Xavier, 1, 1);
            fully_connected.weights = array![[1.0]];
            fully_connected.biases = array![0.0];
            Self {
                fully_connected,
                activation_layer,
            }
        }
    }

    impl<L: Layer + Clone> Layer for FullyConnectedWithActivation<L> {
        fn boxed_clone(&self) -> Box<dyn Layer> {
            Box::new(self.clone())
        }

        fn layer_type(&self) -> &'static str {
            "test_layer"
        }

        fn forward(&self, input: &ArrayView2<f32>) -> Array2<f32> {
            self.activation_layer
                .forward(&self.fully_connected.forward(input).view())
        }

        fn backward(
            &mut self,
            input: &ArrayView2<f32>,
            dloss_doutput: &ArrayView2<f32>,
        ) -> Array2<f32> {
            self.fully_connected.backward(
                input,
                &self.activation_layer.backward(input, dloss_doutput).view(),
            )
        }

        fn loss(
            &self,
            output: &ArrayView2<f32>,
            target: &ArrayView2<f32>,
        ) -> Array1<f32> {
            self.activation_layer.loss(output, target)
        }

        fn dloss_doutput(
            &self,
            output: &ArrayView2<f32>,
            target: &ArrayView2<f32>,
        ) -> Array2<f32> {
            self.activation_layer.dloss_doutput(output, target)
        }

        fn update_parameters(&mut self, learning_rate: f32) {
            self.fully_connected.update_parameters(learning_rate);
        }
    }

    #[test]
    fn test_tanh_layer_single_node() {
        let mut layer = FullyConnectedWithActivation::new_simple(TanhLayer::new());

        let input = array![[1.0]];
        let output = layer.forward(&input.view());
        assert_eq!(output, array![[0.7615941559557649]]);

        let target = array![[0.0]];
        let dloss_doutput = layer.dloss_doutput(&output.view(), &target.view());
        // ∂loss/∂output = (2.0 / n) * (output - target)
        //               = (2.0 / 1) * (0.7615941559557649 - 0.0)
        assert_relative_eq!(dloss_doutput, array![[1.52318831191]], epsilon = 1e-10);

        let dloss_dinput = layer.backward(&input.view(), &dloss_doutput.view());
        layer.update_parameters(0.1);

        // ∂loss/∂preactivation = ∂loss/∂output * ∂output/∂preactivation
        //                      = 1.52318831191 * (1 - tanh^2(output))
        //                      = 1.52318831191 * (1 - 0.7615941559557649^2)
        //
        // biases -= learning_rate * ∂loss/∂preactivation
        // biases[0] = 0.0 - 0.1 * (1.52318831191 * (1 - 0.7615941559557649^2))
        assert_relative_eq!(
            layer.fully_connected.biases,
            array![-0.06397000084],
            epsilon = 1e-10
        );

        // weights -= learning_rate * ∂loss/∂preactivation * input
        // weights[0] = 1.0 - 0.1 * (1.52318831191 * (1 - 0.7615941559557649^2)) * 1.0
        assert_relative_eq!(
            layer.fully_connected.weights,
            array![[0.93602999915]],
            epsilon = 1e-10
        );

        // ∂loss/∂input = ∂loss/∂preactivation * ∂preactivation/∂input
        //              = (1.52318831191 * (1 - 0.7615941559557649^2)) * 1.0
        assert_relative_eq!(dloss_dinput, array![[0.63970000844]], epsilon = 1e-10);
    }

    #[test]
    fn test_softmax_layer_single_node() {
        let mut layer = FullyConnectedWithActivation::new_simple(SoftmaxLayer::new());

        let input = array![[1.0]];
        let output = layer.forward(&input.view());
        assert_eq!(output, array![[1.0]]);

        let target = array![[0.0]];
        let dloss_doutput = layer.dloss_doutput(&output.view(), &target.view());
        let dloss_dinput = layer.backward(&input.view(), &dloss_doutput.view());
        layer.update_parameters(0.1);
        assert_eq!(dloss_dinput, array![[1.0]]);
        assert_eq!(layer.fully_connected.weights, array![[0.9]]);
        assert_eq!(layer.fully_connected.biases, array![-0.1]);
    }

    #[test]
    fn test_leaky_relu_layer() {
        let mut layer = LeakyReluLayer::new(0.01);
        let input = array![[1.0, 0.0, -1.0], [2.0, 0.0, -2.0]];
        let output = layer.forward(&input.view());
        assert_eq!(output, array![[1.0, 0.0, -0.01], [2.0, 0.0, -0.02]]);

        let dloss_doutput = array![[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]];
        let dloss_dinput = layer.backward(&input.view(), &dloss_doutput.view());
        assert_eq!(dloss_dinput, array![[1.0, 1.0, 0.01], [2.0, 2.0, 0.02]]);
    }

    #[test]
    fn test_softmax_layer() {
        let mut fully_connected =
            FullyConnectedLayer::new(InitializationType::Xavier, 3, 2);
        fully_connected.weights = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
        fully_connected.biases = array![1.0, 0.0];
        let mut layer =
            FullyConnectedWithActivation::new(fully_connected, SoftmaxLayer::new());

        let input = array![[1.0, 2.0, 3.0]];
        // 1*1 + 2*3 + 3*5 + 1 = 23
        // 1*2 + 2*4 + 3*6 + 0 = 28
        //
        // e^23 / (e^23 + e^28) = 0.00669285092
        // e^28 / (e^23 + e^28) = 0.99330714907
        let output = layer.forward(&input.view());
        assert_relative_eq!(
            output,
            array![[0.00669285092, 0.99330714907]],
            epsilon = 1e-6
        );

        let target = array![[1.0, 0.0]];
        let dloss_doutput = layer.dloss_doutput(&output.view(), &target.view());
        let dloss_dinput = layer.backward(&input.view(), &dloss_doutput.view());
        layer.update_parameters(0.1);

        // biases -= (output - target) * leaning_rate
        // biases[0] = 1 - (0.00669285092 - 1) * 0.1 = 1.09933071491
        // biases[1] = 0 - (0.99330714907 - 0) * 0.1 = -0.0993307149
        assert_relative_eq!(
            layer.fully_connected.biases,
            array![1.09933071491, -0.0993307149],
            epsilon = 1e-6
        );

        // weights -= (output - target) * input * learning_rate
        // weights[0, 0] = 1 - (0.00669285092 - 1) * 1 * 0.1 = 1.09933071491
        // weights[0, 1] = 2 - (0.99330714907 - 0) * 1 * 0.1 = 1.90066928509
        // weights[1, 0] = 3 - (0.00669285092 - 1) * 2 * 0.1 = 3.19866142982
        // weights[1, 1] = 4 - (0.99330714907 - 0) * 2 * 0.1 = 3.80133857019
        // weights[2, 0] = 5 - (0.00669285092 - 1) * 3 * 0.1 = 5.29799214472
        // weights[2, 1] = 6 - (0.99330714907 - 0) * 3 * 0.1 = 5.70200785528
        assert_relative_eq!(
            layer.fully_connected.weights,
            array![
                [1.09933071491, 1.90066928509],
                [3.19866142982, 3.80133857019],
                [5.29799214472, 5.70200785528],
            ],
            epsilon = 1e-6
        );

        // dloss_dinput = (output - target) * weights
        // dloss_dinput[0] = (0.00669285092 - 1) * 1 + (0.99330714907 - 0) * 2 = 0.99330714906
        // dloss_dinput[1] = (0.00669285092 - 1) * 3 + (0.99330714907 - 0) * 4 = 0.99330714904
        // dloss_dinput[2] = (0.00669285092 - 1) * 5 + (0.99330714907 - 0) * 6 = 0.99330714902
        assert_relative_eq!(
            dloss_dinput,
            array![[0.99330714906, 0.99330714904, 0.99330714902]],
            epsilon = 1e-6
        );
    }

    fn test_layer_gradient_numerically<L, M>(make_random_layer: M)
    where
        L: Layer + Clone,
        M: Fn(usize, usize) -> FullyConnectedWithActivation<L>,
    {
        // We will generate a variety of random layers, compute the gradient,
        // and then numerically estimate the gradient and compare the two. If
        // more than 1% of the gradients are off, we will fail the test.
        let iterations = 100;
        let perturbation = 1e-6;
        let epsilon = 1e-5;
        let input_size = 10;
        let output_size = 10;

        let mut tested = 0;
        let mut failures = 0;
        for _ in 0..iterations {
            let mut layer = make_random_layer(input_size, output_size);
            let input = Array::random((1, input_size), Uniform::new(-1.0, 1.0));
            let target = Array::random((1, output_size), Uniform::new(0.0, 1.0));

            let output = layer.forward(&input.view());
            let loss = layer.loss(&output.view(), &target.view());

            let dloss_doutput = layer.dloss_doutput(&output.view(), &target.view());
            let dloss_dinput = layer.backward(&input.view(), &dloss_doutput.view());

            // We will now numerically estimate the gradient of the loss
            // function with respect to the weights. We will do this by
            // computing the loss function for the weight matrix with each
            // element perturbed by a small amount.
            for i in 0..input.len() {
                for j in 0..output.len() {
                    let mut new_layer = layer.clone();
                    new_layer.fully_connected.weights[[i, j]] += perturbation;
                    let new_output = new_layer.forward(&input.view());
                    let new_loss = new_layer.loss(&new_output.view(), &target.view());

                    tested += 1;
                    if !relative_eq!(
                        new_loss,
                        &loss
                            + perturbation
                                * new_layer.fully_connected.dloss_dweights[[i, j]],
                        epsilon = epsilon,
                    ) {
                        warn!(
                            "Updating weight from {} to {} updated loss from {} to {}",
                            layer.fully_connected.weights[[i, j]],
                            new_layer.fully_connected.weights[[i, j]],
                            loss,
                            new_loss,
                        );
                        failures += 1;
                    }
                }
            }

            // We will now numerically estimate the gradient of the loss
            // function with respect to the biases. We will do this by
            // computing the loss function for the bias vector with each
            // element perturbed by a small amount.
            for j in 0..output.len() {
                let mut new_layer = layer.clone();
                new_layer.fully_connected.biases[j] += perturbation;
                let new_output = new_layer.forward(&input.view());
                let new_loss = new_layer.loss(&new_output.view(), &target.view());

                tested += 1;
                if !relative_eq!(
                    new_loss,
                    &loss + perturbation * new_layer.fully_connected.dloss_dbiases[j],
                    epsilon = epsilon,
                ) {
                    warn!(
                        "Updating bias from {} to {} updated loss from {} to {}",
                        layer.fully_connected.biases[j],
                        new_layer.fully_connected.biases[j],
                        loss,
                        new_loss,
                    );
                    failures += 1;
                }
            }

            // Finally, we will numerically estimate the gradient of the loss
            // function with respect to the input. We will do this by
            // computing the loss function for the input vector with each
            // element perturbed by a small amount.
            for i in 0..input.len() {
                let mut new_input = input.clone();
                new_input[[0, i]] += perturbation;
                let new_output = layer.forward(&new_input.view());
                let new_loss = layer.loss(&new_output.view(), &target.view());

                tested += 1;
                if !relative_eq!(
                    new_loss,
                    &loss + perturbation * dloss_dinput[[0, i]],
                    epsilon = epsilon,
                ) {
                    warn!(
                        "Updating input from {} to {} updated loss from {} to {}",
                        input[[0, i]],
                        new_input[[0, i]],
                        loss,
                        new_loss,
                    );
                    failures += 1;
                }
            }

            // Check our total number of failures.
            if failures as f32 > tested as f32 * 0.01 {
                panic!("Too many failures: {}/{}", failures, tested);
            }
        }
    }

    #[test]
    fn test_tanh_layer_gradient_numerically() {
        test_layer_gradient_numerically(|input_size, output_size| {
            FullyConnectedWithActivation::new(
                FullyConnectedLayer::new(
                    InitializationType::Xavier,
                    input_size,
                    output_size,
                ),
                TanhLayer::new(),
            )
        });
    }

    #[test]
    fn test_relu_layer_gradient_numerically() {
        test_layer_gradient_numerically(|input_size, output_size| {
            FullyConnectedWithActivation::new(
                FullyConnectedLayer::new(
                    InitializationType::He,
                    input_size,
                    output_size,
                ),
                LeakyReluLayer::new(0.01),
            )
        });
    }

    #[test]
    fn test_softmax_gradient_numerically() {
        test_layer_gradient_numerically(|input_size, output_size| {
            FullyConnectedWithActivation::new(
                FullyConnectedLayer::new(
                    InitializationType::Xavier,
                    input_size,
                    output_size,
                ),
                SoftmaxLayer::new(),
            )
        });
    }
}
