use std::fmt::Debug;

use ndarray::{Array1, Array2};
use ndarray_rand::{rand_distr::Uniform, RandomExt};

/// [Xavier][] initialization is a good default for initializing weights in a
/// layer, with the [exception][] of layers using ReLU or SELU activations.
///
/// [Xavier]: http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf
/// [exception]: https://stats.stackexchange.com/a/393012
fn xavier(input_size: usize, output_size: usize) -> Array2<f64> {
    let weights = Array2::random((input_size, output_size), Uniform::new(-1.0, 1.0));
    weights / (input_size as f64).sqrt()
}

/// Gradient associated with a fully-connected layer.
pub struct FullyConnectedGradient {
    dloss_dbiases: Array1<f64>,
    dloss_dweights: Array2<f64>,
    dloss_dinput: Array1<f64>,
}

/// Fully-connected feed-forward layer without an activation function.
#[derive(Debug, Clone)]
struct FullyConnected {
    weights: Array2<f64>,
    biases: Array1<f64>,
}

impl FullyConnected {
    /// Contruct using random Xavier weights.
    fn xavier(input_size: usize, output_size: usize) -> Self {
        let weights = xavier(input_size, output_size);
        let biases = Array1::zeros(output_size);
        Self { weights, biases }
    }

    /// Compute the output of this layer.
    fn forward(&self, input: &Array1<f64>) -> Array1<f64> {
        input.dot(&self.weights) + &self.biases
    }

    /// Given  ∂loss/∂output, compute ∂loss/∂biases, ∂loss/∂weights, and ∂loss/∂input.
    fn gradient(&self, input: &Array1<f64>, dloss_doutput: &Array1<f64>) -> FullyConnectedGradient {
        // ∂loss/∂biases = ∂loss/∂output * ∂output/∂biases
        //               = dloss_doutput * 1
        let dloss_dbiases = dloss_doutput.clone();

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
            dloss_dbiases: dloss_dbiases,
            dloss_dweights: dloss_dweights,
            dloss_dinput: dloss_dinput,
        }
    }

    /// Update the weights and biases of this layer.
    ///
    /// We subtract the gradient because we are minimizing the loss, and going
    /// "downhill" (which is why we call this "gradient descent")!
    fn update(&mut self, gradient: &FullyConnectedGradient, learning_rate: f64) {
        self.biases = &self.biases - learning_rate * &gradient.dloss_dbiases;
        self.weights = &self.weights - learning_rate * &gradient.dloss_dweights;
    }
}

/// A layer in our neural network.
pub trait Layer: Debug + Send + Sync + 'static {
    /// Perform the foward pass through this layer, returning the output.
    fn forward(&self, input: &Array1<f64>) -> Array1<f64>;

    /// An appropriate loss function for this layer.
    fn loss(&self, output: &Array1<f64>, target: &Array1<f64>) -> f64;

    /// The derivative of the loss function with respect to the output.
    fn dloss_doutput(&self, output: &Array1<f64>, target: &Array1<f64>) -> Array1<f64>;

    /// Update the weights and biases of this layer, and return `dloss_dinput`.
    fn update(
        &mut self,
        input: &Array1<f64>,
        dloss_doutput: &Array1<f64>,
        learning_rate: f64,
    ) -> Array1<f64>;
}

/// `Layer` methods that require knowing the gradient type, or that are
/// are not object-safe.
pub trait LayerStatic: Layer + Clone + Sized {
    /// The gradient type associated with this layer.
    type Gradient;

    /// Compute the gradient of this layer.
    fn gradient(&self, input: &Array1<f64>, dloss_doutput: &Array1<f64>) -> Self::Gradient;
}

/// A layer that has weights and biases. Useful for things like numerical gradient checking.
trait LayerWeightsAndBiases: LayerStatic {
    fn weights(&self) -> &Array2<f64>;
    fn biases(&self) -> &Array1<f64>;
    fn weights_mut(&mut self) -> &mut Array2<f64>;
    fn biases_mut(&mut self) -> &mut Array1<f64>;
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
    fn forward(&self, input: &Array1<f64>) -> Array1<f64> {
        self.fully_connected.forward(input).mapv(|x| x.tanh())
    }

    fn loss(&self, output: &Array1<f64>, target: &Array1<f64>) -> f64 {
        // Mean squared error.
        let diff = output - target;
        diff.dot(&diff) / output.len() as f64
    }

    fn dloss_doutput(&self, output: &Array1<f64>, target: &Array1<f64>) -> Array1<f64> {
        // loss          = 1/n * (output - target)^2
        // ∂loss/∂output = 2/n * (output - target)
        (2.0 / output.len() as f64) * (output - target)
    }

    fn update(
        &mut self,
        input: &Array1<f64>,
        dloss_doutput: &Array1<f64>,
        learning_rate: f64,
    ) -> Array1<f64> {
        let gradient = self.gradient(input, dloss_doutput);
        self.fully_connected.update(&gradient, learning_rate);
        gradient.dloss_dinput
    }
}

impl LayerStatic for TanhLayer {
    type Gradient = FullyConnectedGradient;

    fn gradient(&self, input: &Array1<f64>, dloss_doutput: &Array1<f64>) -> Self::Gradient {
        // ∂loss/∂preactivation = ∂loss/∂output * ∂output/∂preactivation
        //                      = dloss_doutput * (1 - tanh^2(output))
        let preactivation = self.fully_connected.forward(input);
        let tanh_squared = preactivation.mapv(|x| x.tanh()).mapv(|x| x * x);
        let dloss_dpreativation = dloss_doutput * (1.0 - tanh_squared);

        // Now we can compute our gradient.
        self.fully_connected.gradient(input, &dloss_dpreativation)
    }
}

impl LayerWeightsAndBiases for TanhLayer {
    fn weights(&self) -> &Array2<f64> {
        &self.fully_connected.weights
    }

    fn biases(&self) -> &Array1<f64> {
        &self.fully_connected.biases
    }

    fn weights_mut(&mut self) -> &mut Array2<f64> {
        &mut self.fully_connected.weights
    }

    fn biases_mut(&mut self) -> &mut Array1<f64> {
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
    fn forward(&self, input: &Array1<f64>) -> Array1<f64> {
        let output = self.fully_connected.forward(input);

        // Softmax is exp(x) / sum(exp(x)).
        let output = output.mapv(|x| x.exp());
        let sum = output.sum();
        output / sum
    }

    fn loss(&self, output: &Array1<f64>, target: &Array1<f64>) -> f64 {
        // Categorical cross-entropy.
        let mut loss = 0.0;
        for (output, target) in output.iter().zip(target.iter()) {
            loss -= target * output.ln();
        }
        loss
    }

    fn dloss_doutput(&self, output: &Array1<f64>, target: &Array1<f64>) -> Array1<f64> {
        // See
        // https://towardsdatascience.com/derivative-of-the-softmax-function-and-the-categorical-cross-entropy-loss-ffceefc081d1
        output - target
    }

    fn update(
        &mut self,
        input: &Array1<f64>,
        dloss_doutput: &Array1<f64>,
        learning_rate: f64,
    ) -> Array1<f64> {
        let gradient = self.gradient(input, dloss_doutput);
        self.fully_connected.update(&gradient, learning_rate);
        gradient.dloss_dinput
    }
}

impl LayerStatic for SoftmaxLayer {
    type Gradient = FullyConnectedGradient;

    fn gradient(&self, input: &Array1<f64>, dloss_doutput: &Array1<f64>) -> Self::Gradient {
        // This seems suspiciously convenient, but see
        // https://towardsdatascience.com/derivative-of-the-softmax-function-and-the-categorical-cross-entropy-loss-ffceefc081d1
        self.fully_connected.gradient(input, dloss_doutput)
    }
}

impl LayerWeightsAndBiases for SoftmaxLayer {
    fn weights(&self) -> &Array2<f64> {
        &self.fully_connected.weights
    }

    fn biases(&self) -> &Array1<f64> {
        &self.fully_connected.biases
    }

    fn weights_mut(&mut self) -> &mut Array2<f64> {
        &mut self.fully_connected.weights
    }

    fn biases_mut(&mut self) -> &mut Array1<f64> {
        &mut self.fully_connected.biases
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
    pub fn forward(&self, input: &Array1<f64>) -> Array1<f64> {
        let mut output = input.clone();
        for layer in &self.layers {
            output = layer.forward(&output);
        }
        output
    }

    /// Compute the loss of the network's final layer.
    pub fn loss(&self, output: &Array1<f64>, target: &Array1<f64>) -> f64 {
        self.last_layer().loss(output, target)
    }

    /// Given an input and a target output, update the network's weights and
    /// biases, and return the loss.
    pub fn update(&mut self, input: &Array1<f64>, target: &Array1<f64>, learning_rate: f64) -> f64 {
        // Forward pass.
        let mut output = input.clone();
        let mut outputs = vec![output.clone()];
        for layer in &self.layers {
            output = layer.forward(&output);
            outputs.push(output.clone());
        }

        // Backward pass.
        let mut dloss_doutput = self.last_layer().dloss_doutput(&output, &target);
        for (layer, output) in self.layers.iter_mut().zip(outputs.into_iter()).rev() {
            dloss_doutput = layer.update(&output, &dloss_doutput, learning_rate);
        }

        self.loss(&output, &target)
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
        let output = layer.forward(&input);
        assert_eq!(output, array![0.7615941559557649]);

        let target = array![0.0];
        let dloss_doutput = layer.dloss_doutput(&output, &target);
        // ∂loss/∂output = (2.0 / n) * (output - target)
        //               = (2.0 / 1) * (0.7615941559557649 - 0.0)
        assert_relative_eq!(dloss_doutput, array![1.52318831191], epsilon = 1e-10);

        let dloss_dinput = layer.update(&input, &dloss_doutput, 0.1);

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
        let output = layer.forward(&input);
        assert_eq!(output, array![1.0]);

        let target = array![0.0];
        let dloss_doutput = layer.dloss_doutput(&output, &target);
        let dloss_dinput = layer.update(&input, &dloss_doutput, 0.1);
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
        let output = layer.forward(&input);
        assert_relative_eq!(
            output,
            array![0.00669285092, 0.99330714907],
            epsilon = 1e-10
        );

        let target = array![1.0, 0.0];
        let dloss_doutput = layer.dloss_doutput(&output, &target);
        let dloss_dinput = layer.update(&input, &dloss_doutput, 0.1);

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
        let delta = 1e-6;
        let input_size = 3;
        let output_size = 2;

        let mut tested = 0;
        let mut failures = 0;
        for _ in 0..iterations {
            let layer = make_random_layer(input_size, output_size);
            let input = Array::random(input_size, Uniform::new(-1.0, 1.0));
            let target = Array::random(output_size, Uniform::new(0.0, 1.0));

            let output = layer.forward(&input);
            let loss = layer.loss(&output, &target);

            let dloss_doutput = layer.dloss_doutput(&output, &target);
            let gradient = layer.gradient(&input, &dloss_doutput);

            // We will now numerically estimate the gradient of the loss
            // function with respect to the weights. We will do this by
            // computing the loss function for the weight matrix with each
            // element perturbed by a small amount.
            for i in 0..input.len() {
                for j in 0..output.len() {
                    let mut new_layer = layer.clone();
                    new_layer.weights_mut()[[i, j]] += delta;
                    let new_output = new_layer.forward(&input);
                    let new_loss = new_layer.loss(&new_output, &target);

                    tested += 1;
                    if !relative_eq!(
                        new_loss,
                        loss + delta * gradient.dloss_dweights[[i, j]],
                        epsilon = 1e-6,
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
                new_layer.biases_mut()[j] += delta;
                let new_output = new_layer.forward(&input);
                let new_loss = new_layer.loss(&new_output, &target);

                tested += 1;
                if !relative_eq!(
                    new_loss,
                    loss + delta * gradient.dloss_dbiases[j],
                    epsilon = 1e-6,
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
                new_input[i] += delta;
                let new_output = layer.forward(&new_input);
                let new_loss = layer.loss(&new_output, &target);

                tested += 1;
                if !relative_eq!(
                    new_loss,
                    loss + delta * gradient.dloss_dinput[i],
                    epsilon = 1e-6,
                ) {
                    warn!(
                        "Updating input from {} to {} updated loss from {} to {}",
                        input[i], new_input[i], loss, new_loss,
                    );
                    failures += 1;
                }
            }

            // Check our total number of failures.
            if failures > tested / 100 {
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
    fn test_softmax_gradient_numerically() {
        test_layer_gradient_numerically(|input_size, output_size| {
            SoftmaxLayer::new(input_size, output_size)
        });
    }
}
