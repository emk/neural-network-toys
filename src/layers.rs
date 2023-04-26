//

use std::{cell::RefCell, fmt::Debug};

use anyhow::{anyhow, Result};
use ndarray::{
    s, Array1, Array2, Array4, ArrayView2, ArrayView4, ArrayViewMut1, Axis,
};
use ndarray_rand::rand::seq::SliceRandom;
use serde::Serialize;
use serde_json::{json, Value};

use crate::{
    im2col::{Im2ColConv, Im2ColDLossDImg},
    initialization::InitializationType,
    reshape::TryToShapeMut,
};

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

    /// A short summary of this layer, if it's interesting enough to
    /// include in an abbreviated summary of the network.
    pub short_summary: String,

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
///
/// These views may have been reshaped from the original layer state
/// in order to flatten them into consistent `ArrayViewMut1` values.
pub struct LayerStateMut<'a> {
    pub params: ArrayViewMut1<'a, f32>,
    pub grad: ArrayViewMut1<'a, f32>,
}

/// A layer in our neural network.
///
/// This is designed to support mini-batch gradient descent, so all
/// functions take and return arrays with shape `(examples, features)`.
pub trait Layer: Debug + 'static {
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
            short_summary: self.layer_type().to_owned(),
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
        let diff = (output - target).mapv(|x| x * x);
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
            short_summary: format!("⤨ {}", self.weights.len_of(Axis(1))),
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
        self.dloss_dbiases = dloss_doutput.sum_axis(ARRAY2_EXAMPLES_AXIS);

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
        // dloss_dweights_{i,o} = input_{b,i} * dloss_doutput_{b,o}
        self.dloss_dweights = input.t().dot(dloss_doutput);

        // ∂loss/∂input = ∂loss/∂output * ∂output/∂input
        //              = dloss_doutput * weights
        dloss_doutput.dot(&self.weights.t())
    }

    fn layer_state_mut<'a>(&'a mut self) -> Vec<LayerStateMut<'a>> {
        vec![
            LayerStateMut {
                params: self.biases.view_mut(),
                grad: self.dloss_dbiases.view_mut(),
            },
            LayerStateMut {
                params: self
                    .weights
                    .try_to_shape_mut(self.weights.len())
                    .expect("failed to reshape weights"),
                grad: self
                    .dloss_dweights
                    .try_to_shape_mut(self.dloss_dweights.len())
                    .expect("failed to reshape dloss_dweights"),
            },
        ]
    }

    fn update_parameters(&mut self, learning_rate: f32) {
        self.weights = &self.weights - learning_rate * &self.dloss_dweights;
        self.biases = &self.biases - learning_rate * &self.dloss_dbiases;
    }
}

/// A convolutional layer with the specified number of filters.
///
/// Conceptually:
///
/// - The input is a 4D array of shape `(batch_size, channels_in, height, width)`.
/// - The output is a 4D array of shape `(batch_size, channels_out, height, width)`.
///
/// In our input layer, our channels might represent RGB values, so we have
/// `[red, green, blue]` for each pixel. In our output layer, our channels
/// represent the output of a learned convolutional filter. They might represent
/// `[edges, corners, blobs]` for each pixel (but the actual output channels
/// will be learned by backpropagation). If we go one layer deeper, our input
/// channels might represent `[edges, corners, blobs]` for each pixel, and our
/// output channels might represent increasingly abstract features, like `[eyes,
/// noses, mouths]`.
#[derive(Debug, Clone)]
pub struct ConvLayer {
    /// The height of the images in our input.
    height: usize,

    /// The width of the images in our input.
    width: usize,

    /// The number of input channels.
    channels_in: usize,

    /// The number of output channels.
    channels_out: usize,

    /// The width and height of the kernel.
    kernel_size: usize,

    /// The amount of padding to place around the input.
    padding: usize,

    /// Our input, padded with zeros so that we can apply the kernel to the
    /// edges of the image without needing to do bounds checking.
    ///
    /// We use a `RefCell` here because the mutability of this field is an
    /// optimization, not part of our public API. We could instead make
    /// `forward` take a `&mut self` and avoid the `RefCell`.
    ///
    /// Shape: (batch_size, channels_in, height + 2 * padding, width + 2 *
    /// padding)
    padded_input: RefCell<Option<Array4<f32>>>,

    /// The weights of the filters.
    ///
    /// Shape: (channels_out, channels_in, kernel_size, kernel_size)
    weights: Array4<f32>,

    /// The biases for each output channel.
    ///
    /// Shape: (channels_out,)
    biases: Array1<f32>,

    /// ∂loss/∂weights averaged over the batch.
    ///
    /// Shape: (channels_out, channels_in, kernel_size, kernel_size)
    dloss_dweights: Array4<f32>,

    /// ∂loss/∂biases averaged over the batch.
    ///
    /// Shape: (chusizeannels_out,)
    dloss_dbiases: Array1<f32>,
}

impl ConvLayer {
    /// Initialize `ConvLayer`.
    pub fn new(
        initialization_type: InitializationType,
        height: usize,
        width: usize,
        channels_in: usize,
        channels_out: usize,
        kernel_size: usize,
    ) -> Result<Self> {
        if kernel_size > 0 && kernel_size % 2 == 0 {
            return Err(anyhow!("kernel_size must be odd"));
        }

        let weights = initialization_type
            .weights(channels_out, channels_in * kernel_size * kernel_size);
        let dloss_dweights =
            Array4::zeros((channels_out, channels_in, kernel_size, kernel_size));
        Ok(Self {
            height,
            width,
            channels_in,
            channels_out,
            kernel_size,
            padding: (kernel_size - 1) / 2,
            padded_input: RefCell::new(None),
            weights: weights
                .into_shape((channels_out, channels_in, kernel_size, kernel_size))
                .expect("failed to reshape weights"),
            biases: Array1::zeros(channels_out),
            dloss_dweights,
            dloss_dbiases: Array1::zeros(channels_out),
        })
    }

    /// Copy `input` into `self.padded_input`, making sure that it's the right
    /// shape and that the padding is set to 0. We try to minimize
    /// reallocations.
    fn set_padded_input(&self, input: &ArrayView4<f32>) {
        let padded_shape = [
            input.len_of(Axis(0)),
            input.len_of(Axis(1)),
            input.len_of(Axis(2)) + 2 * self.padding,
            input.len_of(Axis(3)) + 2 * self.padding,
        ];
        if self.padded_input.borrow().is_none()
            || self.padded_input.borrow().as_ref().unwrap().shape() != &padded_shape
        {
            *self.padded_input.borrow_mut() = Some(Array4::zeros(padded_shape));
        }
        self.padded_input
            .borrow_mut()
            .as_mut()
            .expect("failed to get padded_input")
            .slice_mut(s![
                ..,
                ..,
                self.padding..self.padding + input.len_of(Axis(2)),
                self.padding..self.padding + input.len_of(Axis(3)),
            ])
            .assign(&input);
    }
}

impl Layer for ConvLayer {
    fn boxed_clone(&self) -> Box<dyn Layer> {
        Box::new(self.clone())
    }

    fn layer_type(&self) -> &'static str {
        "conv"
    }

    fn metadata(&self, inputs: usize) -> LayerMetadata {
        //assert_eq!(inputs, self.weights.len_of(Axis(0)));
        LayerMetadata {
            layer_type: self.layer_type().to_owned(),
            short_summary: format!(
                "⧆ {}→ {}×{}²",
                self.channels_in, self.channels_out, self.kernel_size,
            ),
            inputs,
            outputs: self.channels_out * self.height * self.width,
            parameters: self.weights.len() + self.biases.len(),
            extra: json!({
                "height": self.height,
                "kernel_size": self.kernel_size,
                "channels_in": self.channels_in,
                "channels_out": self.channels_out,
                "width": self.width,
            }),
        }
    }

    fn forward(&self, input: &ArrayView2<f32>) -> Array2<f32> {
        // Number of examples we're processing.
        let example_count = input.len_of(ARRAY2_EXAMPLES_AXIS);

        // We'll use our `Iml2ColConv` to perform an efficient convolution.
        let mut iml2col = Im2ColConv::new(
            (self.channels_in, self.height, self.width),
            (
                self.channels_out,
                self.channels_in,
                self.kernel_size,
                self.kernel_size,
            ),
        );

        // Reshape biases so we can add them to the output easily.
        let biases = self
            .biases
            .to_shape((self.channels_out, 1, 1))
            .expect("reshape biases");
        let biases = biases
            .broadcast((self.channels_out, self.height, self.width))
            .expect("broadcast biases");

        // Convolve each example.
        //
        // TODO: This has high memory overhead. We might want to convolve
        // several examples at once, but not all of them.
        let mut result = Array2::zeros((
            example_count,
            self.channels_out * self.height * self.width,
        ));
        for i in 0..example_count {
            let img = input
                .slice(s![i, ..])
                .into_shape((self.channels_in, self.height, self.width))
                .expect("failed to reshape input");
            let mut out = result
                .slice_mut(s![i, ..])
                .into_shape((self.channels_out, self.height, self.width))
                .expect("failed to reshape output");
            iml2col.conv2d(img, self.weights.view(), &mut out);
            out += &biases;
        }
        result
    }

    fn backward(
        &mut self,
        input: &ArrayView2<f32>,
        dloss_doutput: &ArrayView2<f32>,
    ) -> Array2<f32> {
        // Number of examples we're processing.
        let example_count = input.len_of(ARRAY2_EXAMPLES_AXIS);

        // We need to reshape the input to a 4D array.
        let input = input
            .into_shape((example_count, self.channels_in, self.height, self.width))
            .expect("failed to reshape input");
        self.set_padded_input(&input.view());
        let padded_input = self.padded_input.borrow();
        let padded_input = padded_input.as_ref().expect("padded_input not set");

        // We need to reshape the output gradient to a 4D array.
        let dloss_doutput = dloss_doutput
            .into_shape((example_count, self.channels_out, self.height, self.width))
            .expect("failed to reshape output gradient");

        // We need to compute:
        // - ∂loss/∂weights
        // - ∂loss/∂biases
        // - ∂loss/∂input

        // ∂loss/∂weights = ∂loss/∂output * ∂output/∂weights
        //                = ∂loss/∂output * input
        self.dloss_dweights = Array4::zeros(self.weights.dim());
        for example in 0..example_count {
            for channel_out in 0..self.channels_out {
                for row in 0..self.height {
                    for col in 0..self.width {
                        for channel_in in 0..self.channels_in {
                            for kernel_row in 0..self.kernel_size {
                                for kernel_col in 0..self.kernel_size {
                                    let input_row = row + kernel_row;
                                    let input_col = col + kernel_col;
                                    self.dloss_dweights[[
                                        channel_out,
                                        channel_in,
                                        kernel_row,
                                        kernel_col,
                                    ]] += padded_input
                                        [[example, channel_in, input_row, input_col]]
                                        * dloss_doutput
                                            [[example, channel_out, row, col]];
                                }
                            }
                        }
                    }
                }
            }
        }

        // ∂loss/∂biases = ∂loss/∂output * ∂output/∂biases
        //               = ∂loss/∂output * 1
        self.dloss_dbiases = Array1::zeros(self.biases.dim());
        for example in 0..example_count {
            for channel_out in 0..self.channels_out {
                for row in 0..self.height {
                    for col in 0..self.width {
                        self.dloss_dbiases[[channel_out]] +=
                            dloss_doutput[[example, channel_out, row, col]];
                    }
                }
            }
        }

        // ∂loss/∂input = ∂loss/∂output * ∂output/∂input
        //              = ∂loss/∂output * weights
        //
        // We have an efficient implementation of this in `Im2ColDLossDImg`.
        let mut iml2col = Im2ColDLossDImg::new(
            (self.channels_in, self.height, self.width),
            (
                self.channels_out,
                self.channels_in,
                self.kernel_size,
                self.kernel_size,
            ),
        );
        let mut result = Array2::zeros((
            example_count,
            self.channels_in * self.height * self.width,
        ));
        for i in 0..example_count {
            let mut dloss_dinput = result
                .slice_mut(s![i, ..])
                .into_shape((self.channels_in, self.height, self.width))
                .expect("failed to reshape output");
            let dloss_doutput = dloss_doutput.slice(s![i, .., .., ..]);
            iml2col.dloss_dimg(
                dloss_doutput.view(),
                self.weights.view(),
                &mut dloss_dinput,
            );
        }
        result
    }
}

/// A pooling layer.
#[derive(Debug, Clone)]
pub struct PoolLayer {
    /// The height of the images in our input.
    height: usize,

    /// The width of the images in our input.
    width: usize,

    /// The height of the images in our output.
    out_height: usize,

    /// The width of the images in our output.
    out_width: usize,

    /// The number of input channels.
    channels: usize,

    /// The size of the pooling kernel.
    kernel_size: usize,

    /// The stride of the pooling kernel.
    stride: usize,

    /// Our input, padded along the right and bottom edges.
    padded_input: RefCell<Option<Array4<f32>>>,
}

impl PoolLayer {
    /// Initialize `PoolLayer`.
    pub fn new(
        height: usize,
        width: usize,
        channels: usize,
        kernel_size: usize,
        stride: usize,
    ) -> Self {
        let out_height = (height + stride - 1) / stride;
        let out_width = (width + stride - 1) / stride;

        Self {
            height,
            width,
            out_height,
            out_width,
            channels,
            kernel_size,
            stride,
            padded_input: RefCell::new(None),
        }
    }

    /// Output height.
    pub fn out_height(&self) -> usize {
        self.out_height
    }

    /// Output width.
    pub fn out_width(&self) -> usize {
        self.out_width
    }

    /// Number of total (flattened) outputs produced by this layer.
    pub fn outputs(&self) -> usize {
        self.channels * self.out_height * self.out_width
    }

    /// Set the padded input.
    fn set_padded_input(&self, input: &ArrayView4<f32>) {
        // Use stride and kernel size to compute the padding we need to add to
        // the right and bottom edges of the input.
        let in_height = (self.out_height - 1) * self.stride + self.kernel_size;
        let in_width = (self.out_width - 1) * self.stride + self.kernel_size;
        assert!(
            in_height >= self.height && in_height < self.height + self.kernel_size
        );
        assert!(in_width >= self.width && in_width < self.width + self.kernel_size);

        // Compute our padded shape.
        let padded_shape = [
            input.len_of(Axis(0)),
            input.len_of(Axis(1)),
            in_height,
            in_width,
        ];

        // Make sure self.padded_input is the right shape.
        if self.padded_input.borrow().is_none()
            || self.padded_input.borrow().as_ref().unwrap().shape() != padded_shape
        {
            // Fill with negative infinity.
            *self.padded_input.borrow_mut() =
                Some(Array4::from_elem(padded_shape, f32::NEG_INFINITY));
        }

        // Copy the input into the padded input.
        let mut padded_input = self.padded_input.borrow_mut();
        padded_input
            .as_mut()
            .expect("padded_input should be Some")
            .slice_mut(s![.., .., ..self.height, ..self.width])
            .assign(input);
    }
}

impl Layer for PoolLayer {
    fn boxed_clone(&self) -> Box<dyn Layer> {
        Box::new(self.clone())
    }

    fn layer_type(&self) -> &'static str {
        "pool"
    }

    fn metadata(&self, inputs: usize) -> LayerMetadata {
        LayerMetadata {
            layer_type: self.layer_type().to_owned(),
            short_summary: format!("pool/{}", self.stride),
            inputs,
            outputs: self.channels * self.out_height * self.out_width,
            parameters: 0,
            extra: json!({
                "height": self.height,
                "width": self.width,
                "channels": self.channels,
                "kernel_size": self.kernel_size,
                "stride": self.stride,
            }),
        }
    }

    fn forward(&self, input: &ArrayView2<f32>) -> Array2<f32> {
        // Reshape the input into a 4D array.
        let input = input
            .into_shape((
                input.len_of(ARRAY2_EXAMPLES_AXIS),
                self.channels,
                self.height,
                self.width,
            ))
            .expect("failed to reshape input");
        self.set_padded_input(&input);
        let padded_input = self.padded_input.borrow();
        let padded_input = padded_input.as_ref().expect("padded_input should be Some");

        // Compute the output.
        let mut output = Array2::zeros((
            input.len_of(ARRAY2_EXAMPLES_AXIS),
            self.channels * self.out_height * self.out_width,
        ));
        for example in 0..input.len_of(ARRAY2_EXAMPLES_AXIS) {
            for channel in 0..self.channels {
                for row in 0..self.out_height {
                    for col in 0..self.out_width {
                        let mut max = f32::NEG_INFINITY;
                        for kernel_row in 0..self.kernel_size {
                            for kernel_col in 0..self.kernel_size {
                                let input_row = row * self.stride + kernel_row;
                                let input_col = col * self.stride + kernel_col;
                                let value = padded_input
                                    [[example, channel, input_row, input_col]];
                                if value > max {
                                    max = value;
                                }
                            }
                        }
                        output[[
                            example,
                            channel * self.out_height * self.out_width
                                + row * self.out_width
                                + col,
                        ]] = max;
                    }
                }
            }
        }
        output
    }

    fn backward(
        &mut self,
        input: &ArrayView2<f32>,
        dloss_doutput: &ArrayView2<f32>,
    ) -> Array2<f32> {
        // Reshape the input into a 4D array.
        let input = input
            .into_shape((
                input.len_of(ARRAY2_EXAMPLES_AXIS),
                self.channels,
                self.height,
                self.width,
            ))
            .expect("failed to reshape input");
        self.set_padded_input(&input);
        let padded_input = self.padded_input.borrow();
        let padded_input = padded_input.as_ref().expect("padded_input should be Some");

        // Compute the output.
        let mut dloss_dinput = Array4::zeros((
            input.len_of(ARRAY2_EXAMPLES_AXIS),
            self.channels,
            self.height,
            self.width,
        ));
        for example in 0..input.len_of(ARRAY2_EXAMPLES_AXIS) {
            for channel in 0..self.channels {
                for row in 0..self.out_height {
                    for col in 0..self.out_width {
                        let mut max = f32::NEG_INFINITY;
                        let mut max_row = 0;
                        let mut max_col = 0;
                        for kernel_row in 0..self.kernel_size {
                            for kernel_col in 0..self.kernel_size {
                                let input_row = row * self.stride + kernel_row;
                                let input_col = col * self.stride + kernel_col;
                                let value = padded_input
                                    [[example, channel, input_row, input_col]];
                                if value > max {
                                    max = value;
                                    max_row = input_row;
                                    max_col = input_col;
                                }
                            }
                        }
                        dloss_dinput[[example, channel, max_row, max_col]] +=
                            dloss_doutput[[
                                example,
                                channel * self.out_height * self.out_width
                                    + row * self.out_width
                                    + col,
                            ]];
                    }
                }
            }
        }

        dloss_dinput
            .into_shape((
                input.len_of(ARRAY2_EXAMPLES_AXIS),
                self.channels * self.height * self.width,
            ))
            .expect("failed to reshape dloss_dinput")
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
            short_summary: "lrelu".to_owned(),
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
            short_summary: "💧".to_owned(),
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
    use approx::assert_relative_eq;
    use serde::Deserialize;

    use super::*;
    use crate::{
        im2col::tests::load_conv2d_fixture,
        test_utils::{deserialize_array1, deserialize_array2, deserialize_array4},
    };

    #[derive(Debug, Deserialize)]
    struct ActivationGradients {
        #[serde(deserialize_with = "deserialize_array2")]
        inputs: Array2<f32>,
    }

    #[derive(Debug, Deserialize)]
    struct ActivationFixture {
        #[serde(deserialize_with = "deserialize_array2")]
        inputs: Array2<f32>,
        #[serde(deserialize_with = "deserialize_array2")]
        outputs: Array2<f32>,
        #[serde(deserialize_with = "deserialize_array2")]
        targets: Array2<f32>,
        gradients: ActivationGradients,
    }

    fn assert_activation_layer_matches_fixture(layer: &mut dyn Layer, json: &str) {
        let fixture: ActivationFixture = serde_json::from_str(json).unwrap();
        let outputs = layer.forward(&fixture.inputs.view());
        assert_relative_eq!(outputs, fixture.outputs);

        let dloss_doutputs =
            layer.dloss_doutput(&outputs.view(), &fixture.targets.view());
        let dloss_dinputs =
            layer.backward(&fixture.inputs.view(), &dloss_doutputs.view());
        assert_relative_eq!(dloss_dinputs.view(), fixture.gradients.inputs.view());
    }

    #[test]
    fn leaky_relu_matches_fixture() {
        let json: &str = include_str!("../fixtures/layers/leaky_relu.json");
        let mut layer = LeakyReluLayer::new(0.01);
        assert_activation_layer_matches_fixture(&mut layer, json);
    }

    #[test]
    fn tanh_matches_fixture() {
        let json: &str = include_str!("../fixtures/layers/tanh.json");
        let mut layer = TanhLayer::new();
        assert_activation_layer_matches_fixture(&mut layer, json);
    }

    #[test]
    fn softmax_matches_fixture() {
        let json: &str = include_str!("../fixtures/layers/softmax.json");
        let mut layer = SoftmaxLayer::new();
        assert_activation_layer_matches_fixture(&mut layer, json);
    }

    #[test]
    fn fully_connected_matches_fixture() {
        #[derive(Debug, Deserialize)]
        struct FullyConnectedGradients {
            #[serde(deserialize_with = "deserialize_array2")]
            inputs: Array2<f32>,
            #[serde(deserialize_with = "deserialize_array2")]
            weights: Array2<f32>,
            #[serde(deserialize_with = "deserialize_array1")]
            bias: Array1<f32>,
        }

        #[derive(Debug, Deserialize)]
        struct FullyConnectedFixture {
            #[serde(deserialize_with = "deserialize_array2")]
            inputs: Array2<f32>,
            #[serde(deserialize_with = "deserialize_array2")]
            weights: Array2<f32>,
            #[serde(deserialize_with = "deserialize_array1")]
            bias: Array1<f32>,
            #[serde(deserialize_with = "deserialize_array2")]
            outputs: Array2<f32>,
            #[serde(deserialize_with = "deserialize_array2")]
            targets: Array2<f32>,
            gradients: FullyConnectedGradients,
        }

        let json: &str = include_str!("../fixtures/layers/fully_connected.json");
        let fixture: FullyConnectedFixture = serde_json::from_str(json).unwrap();
        let mut layer = FullyConnectedLayer::new(
            InitializationType::Xavier,
            fixture.inputs.ncols(),
            fixture.outputs.ncols(),
        );
        layer.weights = fixture.weights.clone();
        layer.biases = fixture.bias.clone();

        let outputs = layer.forward(&fixture.inputs.view());
        assert_relative_eq!(outputs, fixture.outputs);

        let dloss_doutputs =
            layer.dloss_doutput(&outputs.view(), &fixture.targets.view());
        let dloss_dinputs =
            layer.backward(&fixture.inputs.view(), &dloss_doutputs.view());
        assert_relative_eq!(dloss_dinputs.view(), fixture.gradients.inputs.view());
        assert_relative_eq!(layer.dloss_dbiases.view(), fixture.gradients.bias.view());
        assert_relative_eq!(
            layer.dloss_dweights.view(),
            fixture.gradients.weights.view()
        );
    }

    #[test]
    fn conv_matches_fixture() {
        let fixture = load_conv2d_fixture().unwrap();

        let height = fixture.inputs.shape()[2];
        let width = fixture.inputs.shape()[3];
        let channels_in = fixture.inputs.shape()[1];
        let channels_out = fixture.filters.shape()[0];
        let kernel_size = fixture.filters.shape()[2];
        let mut layer = ConvLayer::new(
            InitializationType::Xavier,
            height,
            width,
            channels_in,
            channels_out,
            kernel_size,
        )
        .expect("Failed to create ConvLayer");

        layer.weights = fixture.filters.clone();
        layer.biases = fixture.biases.clone();

        let inputs = fixture
            .inputs
            .clone()
            .into_shape((fixture.inputs.len_of(Axis(0)), channels_in * height * width))
            .expect("Failed to reshape inputs");
        let outputs = layer.forward(&inputs.view());
        let expected_outputs = fixture
            .outputs
            .clone()
            .into_shape((
                fixture.outputs.len_of(Axis(0)),
                channels_out * height * width,
            ))
            .expect("Failed to reshape outputs");
        assert_relative_eq!(outputs, &expected_outputs, max_relative = 1e-5);

        let targets = fixture
            .targets
            .clone()
            .into_shape((
                fixture.targets.len_of(Axis(0)),
                channels_out * height * width,
            ))
            .expect("Failed to reshape targets");
        let dloss_doutputs = layer.dloss_doutput(&outputs.view(), &targets.view());
        assert_relative_eq!(
            dloss_doutputs.view(),
            fixture
                .gradients
                .outputs
                .to_shape((
                    fixture.gradients.outputs.len_of(Axis(0)),
                    channels_out * height * width
                ))
                .unwrap(),
            max_relative = 1e-5
        );
        let dloss_dinputs = layer.backward(&inputs.view(), &dloss_doutputs.view());
        let expected_dloss_dinputs = fixture
            .gradients
            .inputs
            .clone()
            .into_shape((
                fixture.gradients.inputs.len_of(Axis(0)),
                channels_in * height * width,
            ))
            .expect("Failed to reshape dloss_dinputs");
        assert_relative_eq!(
            layer.dloss_dbiases.view(),
            fixture.gradients.biases.view(),
            max_relative = 1e-5
        );
        assert_relative_eq!(
            layer.dloss_dweights.view(),
            fixture.gradients.filters.view(),
            max_relative = 1e-4
        );
        assert_relative_eq!(
            dloss_dinputs.view(),
            expected_dloss_dinputs.view(),
            max_relative = 1e-5
        );
    }

    #[test]
    fn pool_matches_fixture() {
        // {
        //     "kernel_size": kernel_size,
        //     "stride": stride,
        //     "inputs": inputs.tolist(),
        //     "outputs": outputs.tolist(),
        //     "targets": targets.tolist(),
        //     "gradients": {
        //         "inputs": inputs.grad.tolist(),
        //     },
        // }

        #[derive(Debug, Deserialize)]
        struct PoolGradients {
            #[serde(deserialize_with = "deserialize_array4")]
            inputs: Array4<f32>,
        }

        #[derive(Debug, Deserialize)]
        struct PoolFixture {
            kernel_size: usize,
            stride: usize,
            #[serde(deserialize_with = "deserialize_array4")]
            inputs: Array4<f32>,
            #[serde(deserialize_with = "deserialize_array4")]
            outputs: Array4<f32>,
            #[serde(deserialize_with = "deserialize_array4")]
            targets: Array4<f32>,
            gradients: PoolGradients,
        }

        let json: &str = include_str!("../fixtures/layers/pool2d.json");
        let fixture: PoolFixture = serde_json::from_str(json).unwrap();
        let height = fixture.inputs.shape()[2];
        let width = fixture.inputs.shape()[3];
        let channels = fixture.inputs.shape()[1];
        let mut layer = PoolLayer::new(
            height,
            width,
            channels,
            fixture.kernel_size,
            fixture.stride,
        );

        let inputs = fixture
            .inputs
            .clone()
            .into_shape((fixture.inputs.len_of(Axis(0)), channels * height * width))
            .expect("Failed to reshape inputs");
        let outputs = layer.forward(&inputs.view());
        let out_height = fixture.outputs.shape()[2];
        let out_width = fixture.outputs.shape()[3];
        let expected_outputs = fixture
            .outputs
            .clone()
            .into_shape((
                fixture.outputs.len_of(Axis(0)),
                channels * out_height * out_width,
            ))
            .expect("Failed to reshape outputs");
        assert_relative_eq!(outputs, &expected_outputs);

        let targets = fixture
            .targets
            .clone()
            .into_shape((
                fixture.targets.len_of(Axis(0)),
                channels * out_height * out_width,
            ))
            .expect("Failed to reshape targets");
        let dloss_doutputs = layer.dloss_doutput(&outputs.view(), &targets.view());
        let dloss_dinputs = layer.backward(&inputs.view(), &dloss_doutputs.view());
        let expected_dloss_dinputs = fixture
            .gradients
            .inputs
            .clone()
            .into_shape((
                fixture.gradients.inputs.len_of(Axis(0)),
                channels * height * width,
            ))
            .expect("Failed to reshape dloss_dinputs");
        assert_relative_eq!(dloss_dinputs.view(), expected_dloss_dinputs.view());
    }
}
