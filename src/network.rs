use ndarray::{Array1, Array2, ArrayView2};
use serde::Serialize;

use crate::layers::{
    ActivationFunction, DropoutLayer, FullyConnectedLayer, Layer, LayerMetadata,
    LayerStateMut,
};

#[derive(Debug, Clone, Serialize)]
pub struct NetworkMetadata {
    pub input_width: usize,
    pub output_width: usize,
    pub layers: Vec<LayerMetadata>,
}

impl NetworkMetadata {
    /// The total number of parameters in the network.
    pub fn parameters(&self) -> usize {
        self.layers.iter().map(|layer| layer.parameters).sum()
    }
}

/// A neural network.
#[derive(Debug)]
pub struct Network {
    input_width: usize,
    current_output_width: usize,
    layers: Vec<Box<dyn Layer>>,
}

impl Network {
    /// Create a new network with the specified number of inputs.
    pub fn new(input_width: usize) -> Self {
        Self {
            input_width,
            current_output_width: input_width,
            layers: vec![],
        }
    }

    /// Metadata about the network.
    pub fn metadata(&self) -> NetworkMetadata {
        let mut layers = vec![];
        let mut input_width = self.input_width;

        for layer in &self.layers {
            let metadata = layer.metadata(input_width);
            input_width = metadata.outputs;
            layers.push(metadata);
        }

        NetworkMetadata {
            input_width: self.input_width,
            output_width: self.current_output_width,
            layers,
        }
    }

    /// Add a fully connected layer, with the given number of inputs and outputs
    /// and the given activation function.
    pub fn add_fully_connected_layer(
        &mut self,
        output_width: usize,
        activation_function: ActivationFunction,
    ) {
        let input_weight_initialization_type =
            activation_function.input_weight_inititialization_type();
        let fully_connected = FullyConnectedLayer::new(
            input_weight_initialization_type,
            self.current_output_width,
            output_width,
        );
        self.current_output_width = output_width;
        self.layers.push(Box::new(fully_connected));
        self.layers.push(activation_function.layer());
    }

    /// Add a dropout layer, with the given keep probability. This is only used
    /// during training.
    pub fn add_dropout_layer(&mut self, keep_probability: f32) {
        let dropout = DropoutLayer::new(self.current_output_width, keep_probability);
        self.layers.push(Box::new(dropout));
    }

    /// Get the last layer of the network.
    fn last_layer(&self) -> &dyn Layer {
        self.layers.last().unwrap().as_ref()
    }

    /// Perform a forward pass through the network, returning the output.
    pub fn forward(&self, input: &ArrayView2<f32>) -> Array2<f32> {
        let mut output = input.to_owned();
        for layer in &self.layers {
            output = layer.forward(&output.view());
        }
        output
    }

    /// Compute the loss of the network's final layer.
    pub fn loss(
        &self,
        output: &ArrayView2<f32>,
        target: &ArrayView2<f32>,
    ) -> Array1<f32> {
        self.last_layer().loss(output, target)
    }

    /// Given an input and a target output, update the network's weights and
    /// biases, and return the loss.
    pub fn compute_gradients(
        &mut self,
        inputs: &ArrayView2<f32>,
        targets: &ArrayView2<f32>,
    ) {
        // Start training on all layers.
        for layer in &mut self.layers {
            layer.start_training_step();
        }

        // Forward pass.
        let mut inputs = vec![inputs.to_owned()];
        for layer in &self.layers {
            inputs.push(layer.forward(&inputs.last().unwrap().view()));
        }
        let output = inputs.pop().unwrap();
        assert_eq!(inputs.len(), self.layers.len());

        // Backward pass.
        let mut dloss_doutput =
            self.last_layer().dloss_doutput(&output.view(), &targets);
        for (layer, output) in self.layers.iter_mut().zip(inputs.into_iter()).rev() {
            dloss_doutput = layer.backward(&output.view(), &dloss_doutput.view());
        }

        // End training on all layers.
        for layer in &mut self.layers {
            layer.end_training_step();
        }
    }

    /// The state of all the layers in our network, for use by an optimizer.
    pub fn network_state_mut<'a>(&'a mut self) -> Vec<LayerStateMut<'a>> {
        self.layers
            .iter_mut()
            .map(|l| l.layer_state_mut())
            .flatten()
            .collect()
    }
}

impl Clone for Network {
    fn clone(&self) -> Self {
        Self {
            input_width: self.input_width,
            current_output_width: self.current_output_width,
            layers: self.layers.iter().map(|l| l.boxed_clone()).collect(),
        }
    }
}
