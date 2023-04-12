use std::fmt::Debug;

use ndarray::{s, Array1, Array2, ArrayView1};
use ndarray_rand::{
    rand::seq::SliceRandom,
    rand_distr::{Normal, Uniform},
    RandomExt,
};

/// [Xavier][] initialization is a good default for initializing weights in a
/// layer, with the [exception][] of layers using ReLU or SELU activations.
///
/// [Xavier]: http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf
/// [exception]: https://stats.stackexchange.com/a/393012
fn xavier(input_size: usize, output_size: usize) -> Array2<f32> {
    let weights = Array2::random((input_size, output_size), Uniform::new(-1.0, 1.0));
    weights / (input_size as f32).sqrt()
}

/// [He][] initialization is a good default for initializing weights in a layer
/// with ReLU or SELU activations.
///
/// [He]: https://arxiv.org/abs/1502.01852
fn he(input_size: usize, output_size: usize) -> Array2<f32> {
    Array2::random(
        (input_size, output_size),
        Normal::new(0.0, (2.0 / input_size as f32).sqrt())
            .expect("invalid normal distribution"),
    )
}

/// Gradient associated with a fully-connected layer.
pub struct FullyConnectedGradient {
    dloss_dbiases: Array1<f32>,
    dloss_dweights: Array2<f32>,
    dloss_dinput: Array1<f32>,
}

/// Fully-connected feed-forward layer without an activation function.
#[derive(Debug, Clone)]
struct FullyConnected {
    weights: Array2<f32>,
    biases: Array1<f32>,
}

impl FullyConnected {
    /// Contruct using random Xavier weights.
    fn xavier(input_size: usize, output_size: usize) -> Self {
        let weights = xavier(input_size, output_size);
        let biases = Array1::zeros(output_size);
        Self { weights, biases }
    }

    /// Contruct using random He weights.
    fn he(input_size: usize, output_size: usize) -> Self {
        let weights = he(input_size, output_size);
        let biases = Array1::zeros(output_size);
        Self { weights, biases }
    }

    /// Compute the output of this layer.
    fn forward(&self, input: &ArrayView1<f32>) -> Array1<f32> {
        input.dot(&self.weights) + &self.biases
    }

    /// Given  ∂loss/∂output, compute ∂loss/∂biases, ∂loss/∂weights, and ∂loss/∂input.
    fn gradient(
        &self,
        input: &ArrayView1<f32>,
        dloss_doutput: &ArrayView1<f32>,
    ) -> FullyConnectedGradient {
        // ∂loss/∂biases = ∂loss/∂output * ∂output/∂biases
        //               = dloss_doutput * 1
        let dloss_dbiases = dloss_doutput.to_owned();

        // ∂loss/∂weights = ∂loss/∂output * ∂output/∂weights
        //                = ∂loss_doutput * input
        //
        // We need to do this manually, because the `outer` method is not
        // supported by ndarray.
        let mut dloss_dweights = Array2::zeros((input.len(), dloss_doutput.len()));
        for (i, x) in input.iter().enumerate() {
            for (j, y) in dloss_doutput.iter().enumerate() {
                dloss_dweights[[i, j]] = x * y;
            }
        }

        // ∂loss/∂input = ∂loss/∂output * ∂output/∂input
        //              = dloss_doutput * weights
        let dloss_dinput = dloss_doutput.dot(&self.weights.t());

        FullyConnectedGradient {
            dloss_dbiases,
            dloss_dweights,
            dloss_dinput,
        }
    }

    /// Update the weights and biases of this layer.
    ///
    /// We subtract the gradient because we are minimizing the loss, and going
    /// "downhill" (which is why we call this "gradient descent")!
    fn update(&mut self, gradient: &FullyConnectedGradient, learning_rate: f32) {
        self.biases = &self.biases - learning_rate * &gradient.dloss_dbiases;
        self.weights = &self.weights - learning_rate * &gradient.dloss_dweights;
    }
}

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

    /// Update the weights and biases of this layer, and return `dloss_dinput`.
    fn update(
        &mut self,
        input: &ArrayView1<f32>,
        dloss_doutput: &ArrayView1<f32>,
        learning_rate: f32,
    ) -> Array1<f32>;
}

/// `Layer` methods that require knowing the gradient type, or that are
/// are not object-safe.
pub trait LayerStatic: Layer + Clone + Sized {
    /// The gradient type associated with this layer.
    type Gradient;

    /// Compute the gradient of this layer.
    fn gradient(
        &self,
        input: &ArrayView1<f32>,
        dloss_doutput: &ArrayView1<f32>,
    ) -> Self::Gradient;
}

/// A layer that has weights and biases. Useful for things like numerical gradient checking.
trait LayerWeightsAndBiases: LayerStatic {
    fn weights(&self) -> &Array2<f32>;
    fn biases(&self) -> &Array1<f32>;
    fn weights_mut(&mut self) -> &mut Array2<f32>;
    fn biases_mut(&mut self) -> &mut Array1<f32>;
}

/// A layer using the tanh activation function. This is a good default for
/// hidden layers.
#[derive(Debug, Clone)]
pub struct TanhLayer {
    fully_connected: FullyConnected,
}

impl TanhLayer {
    /// Initialize `TanhLayer` using random weights.
    pub fn new(input_size: usize, output_size: usize) -> Self {
        let fully_connected = FullyConnected::xavier(input_size, output_size);
        Self { fully_connected }
    }
}

impl Layer for TanhLayer {
    fn boxed_clone(&self) -> Box<dyn Layer> {
        Box::new(self.clone())
    }

    fn forward(&self, input: &ArrayView1<f32>) -> Array1<f32> {
        self.fully_connected.forward(input).mapv(|x| x.tanh())
    }

    fn update(
        &mut self,
        input: &ArrayView1<f32>,
        dloss_doutput: &ArrayView1<f32>,
        learning_rate: f32,
    ) -> Array1<f32> {
        let gradient = self.gradient(input, dloss_doutput);
        self.fully_connected.update(&gradient, learning_rate);
        gradient.dloss_dinput
    }
}

impl LayerStatic for TanhLayer {
    type Gradient = FullyConnectedGradient;

    fn gradient(
        &self,
        input: &ArrayView1<f32>,
        dloss_doutput: &ArrayView1<f32>,
    ) -> Self::Gradient {
        // TODO: Store `fully_connected.forward(input)` in `forward` so we don't
        // have to compute it again.

        // ∂loss/∂preactivation = ∂loss/∂output * ∂output/∂preactivation
        //                      = dloss_doutput * (1 - tanh^2(output))
        let preactivation = self.fully_connected.forward(input);
        let tanh_squared = preactivation.mapv(|x| x.tanh()).mapv(|x| x * x);
        let dloss_dpreativation = dloss_doutput * (1.0 - tanh_squared);

        // Now we can compute our gradient.
        self.fully_connected
            .gradient(input, &dloss_dpreativation.view())
    }
}

impl LayerWeightsAndBiases for TanhLayer {
    fn weights(&self) -> &Array2<f32> {
        &self.fully_connected.weights
    }

    fn biases(&self) -> &Array1<f32> {
        &self.fully_connected.biases
    }

    fn weights_mut(&mut self) -> &mut Array2<f32> {
        &mut self.fully_connected.weights
    }

    fn biases_mut(&mut self) -> &mut Array1<f32> {
        &mut self.fully_connected.biases
    }
}

/// A layer using the ReLU activation function, which is simple to compute
/// and used in many state of the art neural networks.
#[derive(Debug, Clone)]
pub struct LeakyReluLayer {
    fully_connected: FullyConnected,
    /// The slope of the leaky ReLU function at negative values.
    leak: f32,
}

impl LeakyReluLayer {
    /// Initialize `ReluLayer` using random weights.
    pub fn new(input_size: usize, output_size: usize, leak: f32) -> Self {
        let fully_connected = FullyConnected::he(input_size, output_size);
        Self {
            fully_connected,
            leak,
        }
    }
}

impl Layer for LeakyReluLayer {
    fn boxed_clone(&self) -> Box<dyn Layer> {
        Box::new(self.clone())
    }

    fn forward(&self, input: &ArrayView1<f32>) -> Array1<f32> {
        self.fully_connected.forward(input).mapv(|x| {
            if x > 0.0 {
                x
            } else {
                x * self.leak
            }
        })
    }

    fn update(
        &mut self,
        input: &ArrayView1<f32>,
        dloss_doutput: &ArrayView1<f32>,
        learning_rate: f32,
    ) -> Array1<f32> {
        let gradient = self.gradient(input, dloss_doutput);
        self.fully_connected.update(&gradient, learning_rate);
        gradient.dloss_dinput
    }
}

impl LayerStatic for LeakyReluLayer {
    type Gradient = FullyConnectedGradient;

    fn gradient(
        &self,
        input: &ArrayView1<f32>,
        dloss_doutput: &ArrayView1<f32>,
    ) -> Self::Gradient {
        // TODO: Store `fully_connected.forward(input)` in `forward` so we don't
        // have to compute it again.

        // ∂loss/∂preactivation = ∂loss/∂output * ∂output/∂preactivation
        //                      = dloss_doutput * (1 if preactivation > 0 else 0)
        let preactivation = self.fully_connected.forward(input);
        let dloss_dpreativation = dloss_doutput
            * (preactivation.mapv(|x| if x > 0.0 { 1.0 } else { self.leak }));

        // Now we can compute our gradient.
        self.fully_connected
            .gradient(input, &dloss_dpreativation.view())
    }
}

impl LayerWeightsAndBiases for LeakyReluLayer {
    fn weights(&self) -> &Array2<f32> {
        &self.fully_connected.weights
    }

    fn biases(&self) -> &Array1<f32> {
        &self.fully_connected.biases
    }

    fn weights_mut(&mut self) -> &mut Array2<f32> {
        &mut self.fully_connected.weights
    }

    fn biases_mut(&mut self) -> &mut Array1<f32> {
        &mut self.fully_connected.biases
    }
}

/// A layer using the softmax activation function and categorical cross-entropy.
/// This is a good default for an output layer that chooses between multiple
/// discrete output values.
#[derive(Debug, Clone)]
pub struct SoftmaxLayer {
    fully_connected: FullyConnected,
}

impl SoftmaxLayer {
    /// Initialize `SoftmaxLayer` using random weights.
    pub fn new(input_size: usize, output_size: usize) -> Self {
        let fully_connected = FullyConnected::xavier(input_size, output_size);
        Self { fully_connected }
    }
}

impl Layer for SoftmaxLayer {
    fn boxed_clone(&self) -> Box<dyn Layer> {
        Box::new(self.clone())
    }

    fn forward(&self, input: &ArrayView1<f32>) -> Array1<f32> {
        let output = self.fully_connected.forward(input);

        // Softmax is exp(x) / sum(exp(x)).
        let output = output.mapv(|x| x.exp());
        let sum = output.sum();
        output / sum
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

    fn update(
        &mut self,
        input: &ArrayView1<f32>,
        dloss_doutput: &ArrayView1<f32>,
        learning_rate: f32,
    ) -> Array1<f32> {
        let gradient = self.gradient(input, dloss_doutput);
        self.fully_connected.update(&gradient, learning_rate);
        gradient.dloss_dinput
    }
}

impl LayerStatic for SoftmaxLayer {
    type Gradient = FullyConnectedGradient;

    fn gradient(
        &self,
        input: &ArrayView1<f32>,
        dloss_doutput: &ArrayView1<f32>,
    ) -> Self::Gradient {
        // This seems suspiciously convenient, but see
        // https://towardsdatascience.com/derivative-of-the-softmax-function-and-the-categorical-cross-entropy-loss-ffceefc081d1
        self.fully_connected.gradient(input, dloss_doutput)
    }
}

impl LayerWeightsAndBiases for SoftmaxLayer {
    fn weights(&self) -> &Array2<f32> {
        &self.fully_connected.weights
    }

    fn biases(&self) -> &Array1<f32> {
        &self.fully_connected.biases
    }

    fn weights_mut(&mut self) -> &mut Array2<f32> {
        &mut self.fully_connected.weights
    }

    fn biases_mut(&mut self) -> &mut Array1<f32> {
        &mut self.fully_connected.biases
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

    fn update(
        &mut self,
        _input: &ArrayView1<f32>,
        dloss_doutput: &ArrayView1<f32>,
        _learning_rate: f32,
    ) -> Array1<f32> {
        // See https://deepnotes.io/dropout for a discussion of gradients and
        // dropout. Since we're scaling the output by the keep probability,
        // I guess we need to scale the gradient by the same amount?
        dloss_doutput * &self.mask
    }
}

/// A neural network.
#[derive(Debug)]
pub struct Network {
    layers: Vec<Box<dyn Layer>>,
}

impl Network {
    /// Create a new network, specifying the first layer.
    pub fn new<L>(layer: L) -> Self
    where
        L: Layer + 'static,
    {
        Self {
            layers: vec![Box::new(layer)],
        }
    }

    /// Add a layer to the network.
    pub fn add_layer<L>(&mut self, layer: L)
    where
        L: Layer + 'static,
    {
        self.layers.push(Box::new(layer));
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
        let mut outputs = vec![input.to_owned()];
        for layer in &self.layers {
            let output = layer.forward(&outputs.last().unwrap().view());
            outputs.push(output);
        }
        let output = outputs.last().unwrap();

        // Backward pass.
        let mut dloss_doutput =
            self.last_layer().dloss_doutput(&output.view(), &target);
        for (layer, output) in self.layers.iter_mut().zip(outputs.into_iter()).rev() {
            dloss_doutput =
                layer.update(&output.view(), &dloss_doutput.view(), learning_rate);
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

    use super::*;

    #[test]
    fn test_tanh_layer_single_node() {
        let mut layer = TanhLayer {
            fully_connected: FullyConnected {
                weights: array![[1.0]],
                biases: array![0.0],
            },
        };

        let input = array![1.0];
        let output = layer.forward(&input.view());
        assert_eq!(output, array![0.7615941559557649]);

        let target = array![0.0];
        let dloss_doutput = layer.dloss_doutput(&output.view(), &target.view());
        // ∂loss/∂output = (2.0 / n) * (output - target)
        //               = (2.0 / 1) * (0.7615941559557649 - 0.0)
        assert_relative_eq!(dloss_doutput, array![1.52318831191], epsilon = 1e-10);

        let dloss_dinput = layer.update(&input.view(), &dloss_doutput.view(), 0.1);

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
        let mut layer = SoftmaxLayer {
            fully_connected: FullyConnected {
                weights: array![[1.0]],
                biases: array![0.0],
            },
        };

        let input = array![1.0];
        let output = layer.forward(&input.view());
        assert_eq!(output, array![1.0]);

        let target = array![0.0];
        let dloss_doutput = layer.dloss_doutput(&output.view(), &target.view());
        let dloss_dinput = layer.update(&input.view(), &dloss_doutput.view(), 0.1);
        assert_eq!(dloss_dinput, array![1.0]);
        assert_eq!(layer.fully_connected.weights, array![[0.9]]);
        assert_eq!(layer.fully_connected.biases, array![-0.1]);
    }

    #[test]
    fn test_softmax_layer() {
        let mut layer = SoftmaxLayer {
            fully_connected: FullyConnected {
                weights: array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
                biases: array![1.0, 0.0],
            },
        };

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
        let dloss_dinput = layer.update(&input.view(), &dloss_doutput.view(), 0.1);

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
        L: LayerStatic<Gradient = FullyConnectedGradient> + LayerWeightsAndBiases,
        M: Fn(usize, usize) -> L,
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
            let layer = make_random_layer(input_size, output_size);
            let input = Array::random(input_size, Uniform::new(-1.0, 1.0));
            let target = Array::random(output_size, Uniform::new(0.0, 1.0));

            let output = layer.forward(&input.view());
            let loss = layer.loss(&output.view(), &target.view());

            let dloss_doutput = layer.dloss_doutput(&output.view(), &target.view());
            let gradient = layer.gradient(&input.view(), &dloss_doutput.view());

            // We will now numerically estimate the gradient of the loss
            // function with respect to the weights. We will do this by
            // computing the loss function for the weight matrix with each
            // element perturbed by a small amount.
            for i in 0..input.len() {
                for j in 0..output.len() {
                    let mut new_layer = layer.clone();
                    new_layer.weights_mut()[[i, j]] += perturbation;
                    let new_output = new_layer.forward(&input.view());
                    let new_loss = new_layer.loss(&new_output.view(), &target.view());

                    tested += 1;
                    if !relative_eq!(
                        new_loss,
                        loss + perturbation * gradient.dloss_dweights[[i, j]],
                        epsilon = epsilon,
                    ) {
                        warn!(
                            "Updating weight from {} to {} updated loss from {} to {}",
                            layer.weights()[[i, j]],
                            new_layer.weights()[[i, j]],
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
                new_layer.biases_mut()[j] += perturbation;
                let new_output = new_layer.forward(&input.view());
                let new_loss = new_layer.loss(&new_output.view(), &target.view());

                tested += 1;
                if !relative_eq!(
                    new_loss,
                    loss + perturbation * gradient.dloss_dbiases[j],
                    epsilon = epsilon,
                ) {
                    warn!(
                        "Updating bias from {} to {} updated loss from {} to {}",
                        layer.biases()[j],
                        new_layer.biases()[j],
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
                    loss + perturbation * gradient.dloss_dinput[i],
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
            TanhLayer::new(input_size, output_size)
        });
    }

    #[test]
    fn test_relu_layer_gradient_numerically() {
        test_layer_gradient_numerically(|input_size, output_size| {
            LeakyReluLayer::new(input_size, output_size, 0.01)
        });
    }

    #[test]
    fn test_softmax_gradient_numerically() {
        test_layer_gradient_numerically(|input_size, output_size| {
            SoftmaxLayer::new(input_size, output_size)
        });
    }
}
