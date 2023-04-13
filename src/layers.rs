use std::fmt::Debug;

use ndarray::{s, Array1, Array2, ArrayView1};
use ndarray_rand::rand::seq::SliceRandom;

use crate::initialization::InitializationType;

/// A layer in our neural network.
pub trait Layer: Debug + Send + Sync + 'static {
    /// Return a boxed clone of this layer.
    fn boxed_clone(&self) -> Box<dyn Layer>;

    /// Perform the foward pass through this layer, returning the output.
    fn forward(&self, input: &ArrayView1<f32>) -> Array1<f32>;

    /// An appropriate loss function for this layer. By default, this is mean
    /// squared error, but it can be overridden for particular layers.
    fn loss(&self, output: &ArrayView1<f32>, target: &ArrayView1<f32>) -> f32 {
        // Mean squared error.
        let diff = output - target;
        diff.dot(&diff) / output.len() as f32
    }

    /// The derivative of the loss function with respect to the output.
    fn dloss_doutput(
        &self,
        output: &ArrayView1<f32>,
        target: &ArrayView1<f32>,
    ) -> Array1<f32> {
        // loss          = 1/n * (output - target)^2
        // ∂loss/∂output = 2/n * (output - target)
        (2.0 / output.len() as f32) * (output - target)
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
        input: &ArrayView1<f32>,
        dloss_doutput: &ArrayView1<f32>,
    ) -> Array1<f32>;

    /// Update the weights and biases of this layer, and return `dloss_dinput`.
    fn update(&mut self, _learning_rate: f32) {}
}

/// Fully-connected feed-forward layer without an activation function.
#[derive(Debug, Clone)]
struct FullyConnectedLayer {
    weights: Array2<f32>,
    biases: Array1<f32>,

    dloss_dbiases: Array1<f32>,
    dloss_dweights: Array2<f32>,
}

impl FullyConnectedLayer {
    /// Contruct using random weights.
    fn new(
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

    fn forward(&self, input: &ArrayView1<f32>) -> Array1<f32> {
        input.dot(&self.weights) + &self.biases
    }

    fn backward(
        &mut self,
        input: &ArrayView1<f32>,
        dloss_doutput: &ArrayView1<f32>,
    ) -> Array1<f32> {
        // ∂loss/∂biases = ∂loss/∂output * ∂output/∂biases
        //               = dloss_doutput * 1
        self.dloss_dbiases = dloss_doutput.to_owned();

        // ∂loss/∂weights = ∂loss/∂output * ∂output/∂weights
        //                = ∂loss_doutput * input
        //
        // We need to do this manually, because the `outer` method is not
        // supported by ndarray.
        self.dloss_dweights = Array2::zeros((input.len(), dloss_doutput.len()));
        for (i, x) in input.iter().enumerate() {
            for (j, y) in dloss_doutput.iter().enumerate() {
                self.dloss_dweights[[i, j]] = x * y;
            }
        }

        // ∂loss/∂input = ∂loss/∂output * ∂output/∂input
        //              = dloss_doutput * weights
        dloss_doutput.dot(&self.weights.t())
    }

    fn update(&mut self, learning_rate: f32) {
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

    fn forward(&self, input: &ArrayView1<f32>) -> Array1<f32> {
        input.mapv(|x| x.tanh())
    }

    fn backward(
        &mut self,
        input: &ArrayView1<f32>,
        dloss_doutput: &ArrayView1<f32>,
    ) -> Array1<f32> {
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

    fn forward(&self, input: &ArrayView1<f32>) -> Array1<f32> {
        input.mapv(|x| if x > 0.0 { x } else { x * self.leak })
    }

    fn backward(
        &mut self,
        _input: &ArrayView1<f32>,
        dloss_doutput: &ArrayView1<f32>,
    ) -> Array1<f32> {
        // ∂loss/∂input = ∂loss/∂output * ∂output/∂input
        //              = dloss_doutput * (1 if x > 0 else leak)
        dloss_doutput.mapv(|x| if x > 0.0 { x } else { x * self.leak })
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

    fn forward(&self, input: &ArrayView1<f32>) -> Array1<f32> {
        // Softmax is exp(x) / sum(exp(x)).
        let output = input.mapv(|x| x.exp());
        let sum = output.sum();
        output / sum
    }

    fn backward(
        &mut self,
        _input: &ArrayView1<f32>,
        dloss_doutput: &ArrayView1<f32>,
    ) -> Array1<f32> {
        // This seems suspiciously convenient, but see
        // https://towardsdatascience.com/derivative-of-the-softmax-function-and-the-categorical-cross-entropy-loss-ffceefc081d1
        dloss_doutput.to_owned()
    }

    fn loss(&self, output: &ArrayView1<f32>, target: &ArrayView1<f32>) -> f32 {
        // Categorical cross-entropy.
        let mut loss = 0.0;
        for (output, target) in output.iter().zip(target.iter()) {
            loss -= target * output.ln();
        }
        loss
    }

    fn dloss_doutput(
        &self,
        output: &ArrayView1<f32>,
        target: &ArrayView1<f32>,
    ) -> Array1<f32> {
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

    fn forward(&self, input: &ArrayView1<f32>) -> Array1<f32> {
        input * &self.mask
    }

    fn backward(
        &mut self,
        _input: &ArrayView1<f32>,
        dloss_doutput: &ArrayView1<f32>,
    ) -> Array1<f32> {
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
pub enum ActivationFunction {
    Tanh,
    LeakyReLU { leak: f32 },
    Softmax,
}

impl ActivationFunction {
    /// Construct a layer that applies this activation function.
    pub fn layer(&self) -> Box<dyn Layer> {
        match self {
            ActivationFunction::Tanh => Box::new(TanhLayer::new()),
            ActivationFunction::LeakyReLU { leak } => {
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
            ActivationFunction::LeakyReLU { .. } => InitializationType::He,
            ActivationFunction::Softmax => InitializationType::Xavier,
        }
    }
}

/// A neural network.
#[derive(Debug)]
pub struct Network {
    layers: Vec<Box<dyn Layer>>,
}

impl Network {
    /// Create a new network.
    pub fn new() -> Self {
        Self { layers: vec![] }
    }

    /// Add a layer to the network.
    pub fn add_layer<L>(&mut self, layer: L)
    where
        L: Layer + 'static,
    {
        self.layers.push(Box::new(layer));
    }

    /// Add a fully connected layer, with the given number of inputs and outputs
    /// and the given activation function.
    pub fn add_fully_connected_layer(
        &mut self,
        input_width: usize,
        output_width: usize,
        activation_function: ActivationFunction,
    ) {
        let input_weight_initialization_type =
            activation_function.input_weight_inititialization_type();
        let fully_connected = FullyConnectedLayer::new(
            input_weight_initialization_type,
            input_width,
            output_width,
        );
        self.add_layer(fully_connected);
        self.layers.push(activation_function.layer());
    }

    /// Add a dropout layer, with the given keep probability. This is only used
    /// during training.
    pub fn add_dropout_layer(&mut self, width: usize, keep_probability: f32) {
        let dropout = DropoutLayer::new(width, keep_probability);
        self.add_layer(dropout);
    }

    /// Get the last layer of the network.
    fn last_layer(&self) -> &dyn Layer {
        self.layers.last().unwrap().as_ref()
    }

    /// Perform a forward pass through the network, returning the output.
    pub fn forward(&self, input: &ArrayView1<f32>) -> Array1<f32> {
        let mut output = input.to_owned();
        for layer in &self.layers {
            output = layer.forward(&output.view());
        }
        output
    }

    /// Compute the loss of the network's final layer.
    pub fn loss(&self, output: &ArrayView1<f32>, target: &ArrayView1<f32>) -> f32 {
        self.last_layer().loss(output, target)
    }

    /// Given an input and a target output, update the network's weights and
    /// biases, and return the loss.
    pub fn update(
        &mut self,
        input: &ArrayView1<f32>,
        target: &ArrayView1<f32>,
        learning_rate: f32,
    ) {
        // Start training on all layers.
        for layer in &mut self.layers {
            layer.start_training_step();
        }

        // Forward pass.
        let mut inputs = vec![input.to_owned()];
        for layer in &self.layers {
            inputs.push(layer.forward(&inputs.last().unwrap().view()));
        }
        let output = inputs.pop().unwrap();
        assert_eq!(inputs.len(), self.layers.len());

        // Backward pass.
        let mut dloss_doutput =
            self.last_layer().dloss_doutput(&output.view(), &target);
        for (layer, output) in self.layers.iter_mut().zip(inputs.into_iter()).rev() {
            dloss_doutput = layer.backward(&output.view(), &dloss_doutput.view());
            layer.update(learning_rate);
        }

        // End training on all layers.
        for layer in &mut self.layers {
            layer.end_training_step();
        }
    }
}

impl Clone for Network {
    fn clone(&self) -> Self {
        Self {
            layers: self.layers.iter().map(|l| l.boxed_clone()).collect(),
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
            Self {
                fully_connected: FullyConnectedLayer {
                    weights: array![[1.0]],
                    biases: array![0.0],
                    dloss_dbiases: array![0.0],
                    dloss_dweights: array![[0.0]],
                },
                activation_layer,
            }
        }
    }

    impl<L: Layer + Clone> Layer for FullyConnectedWithActivation<L> {
        fn boxed_clone(&self) -> Box<dyn Layer> {
            Box::new(self.clone())
        }

        fn forward(&self, input: &ArrayView1<f32>) -> Array1<f32> {
            self.activation_layer
                .forward(&self.fully_connected.forward(input).view())
        }

        fn backward(
            &mut self,
            input: &ArrayView1<f32>,
            dloss_doutput: &ArrayView1<f32>,
        ) -> Array1<f32> {
            self.fully_connected.backward(
                input,
                &self.activation_layer.backward(input, dloss_doutput).view(),
            )
        }

        fn loss(&self, output: &ArrayView1<f32>, target: &ArrayView1<f32>) -> f32 {
            self.activation_layer.loss(output, target)
        }

        fn dloss_doutput(
            &self,
            output: &ArrayView1<f32>,
            target: &ArrayView1<f32>,
        ) -> Array1<f32> {
            self.activation_layer.dloss_doutput(output, target)
        }

        fn update(&mut self, learning_rate: f32) {
            self.fully_connected.update(learning_rate);
        }
    }

    #[test]
    fn test_tanh_layer_single_node() {
        let mut layer = FullyConnectedWithActivation::new_simple(TanhLayer::new());

        let input = array![1.0];
        let output = layer.forward(&input.view());
        assert_eq!(output, array![0.7615941559557649]);

        let target = array![0.0];
        let dloss_doutput = layer.dloss_doutput(&output.view(), &target.view());
        // ∂loss/∂output = (2.0 / n) * (output - target)
        //               = (2.0 / 1) * (0.7615941559557649 - 0.0)
        assert_relative_eq!(dloss_doutput, array![1.52318831191], epsilon = 1e-10);

        let dloss_dinput = layer.backward(&input.view(), &dloss_doutput.view());
        layer.update(0.1);

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
        assert_relative_eq!(dloss_dinput, array![0.63970000844], epsilon = 1e-10);
    }

    #[test]
    fn test_softmax_layer_single_node() {
        let mut layer = FullyConnectedWithActivation::new_simple(SoftmaxLayer::new());

        let input = array![1.0];
        let output = layer.forward(&input.view());
        assert_eq!(output, array![1.0]);

        let target = array![0.0];
        let dloss_doutput = layer.dloss_doutput(&output.view(), &target.view());
        let dloss_dinput = layer.backward(&input.view(), &dloss_doutput.view());
        layer.update(0.1);
        assert_eq!(dloss_dinput, array![1.0]);
        assert_eq!(layer.fully_connected.weights, array![[0.9]]);
        assert_eq!(layer.fully_connected.biases, array![-0.1]);
    }

    #[test]
    fn test_softmax_layer() {
        let mut layer = FullyConnectedWithActivation::new(
            FullyConnectedLayer {
                weights: array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
                biases: array![1.0, 0.0],
                dloss_dbiases: array![0.0, 0.0],
                dloss_dweights: array![[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
            },
            SoftmaxLayer::new(),
        );

        let input = array![1.0, 2.0, 3.0];
        // 1*1 + 2*3 + 3*5 + 1 = 23
        // 1*2 + 2*4 + 3*6 + 0 = 28
        //
        // e^23 / (e^23 + e^28) = 0.00669285092
        // e^28 / (e^23 + e^28) = 0.99330714907
        let output = layer.forward(&input.view());
        assert_relative_eq!(
            output,
            array![0.00669285092, 0.99330714907],
            epsilon = 1e-10
        );

        let target = array![1.0, 0.0];
        let dloss_doutput = layer.dloss_doutput(&output.view(), &target.view());
        let dloss_dinput = layer.backward(&input.view(), &dloss_doutput.view());
        layer.update(0.1);

        // biases -= (output - target) * leaning_rate
        // biases[0] = 1 - (0.00669285092 - 1) * 0.1 = 1.09933071491
        // biases[1] = 0 - (0.99330714907 - 0) * 0.1 = -0.0993307149
        assert_relative_eq!(
            layer.fully_connected.biases,
            array![1.09933071491, -0.0993307149],
            epsilon = 1e-10
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
            epsilon = 1e-10
        );

        // dloss_dinput = (output - target) * weights
        // dloss_dinput[0] = (0.00669285092 - 1) * 1 + (0.99330714907 - 0) * 2 = 0.99330714906
        // dloss_dinput[1] = (0.00669285092 - 1) * 3 + (0.99330714907 - 0) * 4 = 0.99330714904
        // dloss_dinput[2] = (0.00669285092 - 1) * 5 + (0.99330714907 - 0) * 6 = 0.99330714902
        assert_relative_eq!(
            dloss_dinput,
            array![0.99330714906, 0.99330714904, 0.99330714902],
            epsilon = 1e-10
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
            let input = Array::random(input_size, Uniform::new(-1.0, 1.0));
            let target = Array::random(output_size, Uniform::new(0.0, 1.0));

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
                        loss + perturbation
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
                    loss + perturbation * new_layer.fully_connected.dloss_dbiases[j],
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
                new_input[i] += perturbation;
                let new_output = layer.forward(&new_input.view());
                let new_loss = layer.loss(&new_output.view(), &target.view());

                tested += 1;
                if !relative_eq!(
                    new_loss,
                    loss + perturbation * dloss_dinput[i],
                    epsilon = epsilon,
                ) {
                    warn!(
                        "Updating input from {} to {} updated loss from {} to {}",
                        input[i], new_input[i], loss, new_loss,
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
