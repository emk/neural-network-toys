//! "Unrolling" images into matrices for use in convolutional neural networks.
//!
//! Img shape: (C_in, H_i, W_i)
//!   Patches: ∀ h, w extract a "strip" like (C_in * H_f * W_f)
//!   Collected: (C_out * H_o * W_o, C_in * H_f * W_f)
//! Filter shape: (C_out, C_in * H_f * W_f) (can use reshape)
//! Out shape: (C_out, H_o * W_o)
use ndarray::{s, Array2, Array3, Array4, ArrayView3, ArrayView4, ArrayViewMut3};

/// Interface for performing convolutions using the fast "im2col" technique.
///
/// We pre-allocate storage for all our intermediate matrices, to avoid
/// allocating memory in hot loops.
pub struct Im2ColConv {
    /// Number of output channels ("filters") we generate.
    out_channels: usize,

    /// Number of input channels.
    in_channels: usize,

    /// Image height.
    height: usize,

    /// Image width.
    width: usize,

    /// Kernel size.
    kernel_size: usize,

    /// Padding.
    padding: usize,

    /// A copy of our input array, padded with zeros to we don't need to
    /// bounds-check inside loops.
    ///
    /// Shape: `(channels, height + 2*padding, width + 2*padding)`
    padded_input: Array3<f32>,

    /// Our input array, reorganized and copied so that each input needed for a
    /// given kernel position is adjacent.
    ///
    /// Shape: `(height * width, in_channels * kernel_size * kernel_size)`
    patches: Array2<f32>,
}

impl Im2ColConv {
    /// Create a new `Img2Col, and pre-allocate all the storage we'll need to
    /// perform a convolution.
    pub fn new(
        img_shape: (usize, usize, usize),
        kernel_shape: (usize, usize, usize, usize),
    ) -> Self {
        let (in_channels, height, width) = img_shape;
        let (out_channels, in_channels_2, kernel_size, kernel_size_2) = kernel_shape;
        assert_eq!(kernel_size % 2, 1, "kernel size must be odd");
        assert_eq!(in_channels, in_channels_2);
        assert_eq!(kernel_size, kernel_size_2);
        let padding = kernel_size / 2;
        let padded_input =
            Array3::zeros((in_channels, height + 2 * padding, width + 2 * padding));
        let patches =
            Array2::zeros((height * width, in_channels * kernel_size * kernel_size));
        Self {
            out_channels,
            in_channels,
            height,
            width,
            kernel_size,
            padding,
            padded_input,
            patches,
        }
    }

    /// Convolve `img` with `kernel`.
    pub fn conv2d(
        &mut self,
        img: ArrayView3<f32>,
        kernel: ArrayView4<f32>,
        out: &mut ArrayViewMut3<f32>,
    ) {
        debug_assert_eq!(img.dim(), (self.in_channels, self.height, self.width));
        debug_assert_eq!(
            kernel.dim(),
            (
                self.out_channels,
                self.in_channels,
                self.kernel_size,
                self.kernel_size
            ),
        );

        // Pad our input image.
        self.padded_input
            .slice_mut(s![
                ..,
                self.padding..self.padding + self.height,
                self.padding..self.padding + self.width
            ])
            .assign(&img);

        // Extract patches from the padded input image.
        let c_in_stride = self.kernel_size * self.kernel_size;
        for h in 0..self.height {
            for w in 0..self.width {
                // TODO: Is it actually faster to do element-wise copies?
                let patch = self.padded_input.slice(s![
                    ..,
                    h..h + self.kernel_size,
                    w..w + self.kernel_size
                ]);
                for c_in in 0..self.in_channels {
                    for kh in 0..self.kernel_size {
                        let row = patch.slice(s![c_in, kh, ..]);
                        let start = c_in_stride * c_in + self.kernel_size * kh;
                        self.patches
                            .slice_mut(s![
                                h * self.width + w,
                                start..start + self.kernel_size
                            ])
                            .assign(&row);
                    }
                }
            }
        }

        // Reshape the kernel into a matrix.
        let kernel = kernel
            .into_shape((
                self.out_channels,
                self.in_channels * self.kernel_size * self.kernel_size,
            ))
            .expect("reshape kernel failed");

        // Do the multiplication.
        //
        // TODO: Can we do this without allocating?
        out.assign(
            &kernel
                .dot(&self.patches.t())
                .into_shape((self.out_channels, self.height, self.width))
                .expect("reshape output failed"),
        );
    }
}

/// Interface for calculating image gradients using the im2col algorithm.
pub struct Im2ColDLossDImg {
    im2col: Im2ColConv,
}

impl Im2ColDLossDImg {
    /// Create a new `Im2ColDLossDImg`, and pre-allocate all the storage we'll
    /// need to to calculate the derivative of the loss with respect to the
    /// input image.
    pub fn new(
        img_shape: (usize, usize, usize),
        kernel_shape: (usize, usize, usize, usize),
    ) -> Self {
        let (in_channels, height, width) = img_shape;
        let (out_channels, _, kernel_size, _) = kernel_shape;

        // We can just call `new_for_conv2d` with the right shapes.
        Self {
            im2col: Im2ColConv::new(
                (out_channels, height, width),
                (in_channels, out_channels, kernel_size, kernel_size),
            ),
        }
    }

    /// Given the derivative of the loss with respect to the result of a
    /// convolution, compute the derivative of the loss with respect to the
    /// input image.
    pub fn dloss_dimg(
        &mut self,
        dloss_dresult: ArrayView3<f32>,
        kernel: ArrayView4<f32>,
        out_dloss_dimg: &mut ArrayViewMut3<f32>,
    ) {
        // Convolve the derivative of the loss with respect to the result of the
        // convolution with the flipped kernel.
        let flipped_kernel = flip_kernel(kernel);
        self.im2col
            .conv2d(dloss_dresult, flipped_kernel.view(), out_dloss_dimg);
    }
}

/// Rotate a kernel by 180 degrees, and swap input and output channels.
fn flip_kernel(kernel: ArrayView4<f32>) -> Array4<f32> {
    let (out_channels, in_channels, kernel_size, kernel_size_2) = kernel.dim();
    assert_eq!(kernel_size, kernel_size_2);
    let mut flipped_kernel =
        Array4::zeros((in_channels, out_channels, kernel_size, kernel_size));
    for c_out in 0..out_channels {
        for c_in in 0..in_channels {
            for kh in 0..kernel_size {
                for kw in 0..kernel_size {
                    flipped_kernel[[c_in, c_out, kh, kw]] = kernel
                        [[c_out, c_in, kernel_size - kh - 1, kernel_size - kw - 1]];
                }
            }
        }
    }
    flipped_kernel
}

#[cfg(test)]
pub mod tests {
    use std::ops::AddAssign;

    use anyhow::Result;
    use approx::assert_relative_eq;
    use ndarray::{array, Array1, Array4};
    use serde::Deserialize;

    use super::*;
    use crate::test_utils::{deserialize_array1, deserialize_array4};

    /// A straightforward convolution implementation using loops, that we
    /// can compare our im2col implementation against.
    fn slow_conv2d(img: ArrayView3<f32>, kernel: ArrayView4<f32>) -> Array3<f32> {
        let (in_channels, height, width) = img.dim();
        let (out_channels, in_channels_2, kernel_size, kernel_size_2) = kernel.dim();
        assert_eq!(kernel_size % 2, 1, "kernel size must be odd");
        assert_eq!(in_channels, in_channels_2);
        assert_eq!(kernel_size, kernel_size_2);
        let padding = kernel_size / 2;

        // Pad our input image.
        let mut padded_input =
            Array3::zeros((in_channels, height + 2 * padding, width + 2 * padding));
        padded_input
            .slice_mut(s![.., padding..padding + height, padding..padding + width])
            .assign(&img);

        // Convolve.
        let mut result = Array3::zeros((out_channels, height, width));
        for out_channel in 0..out_channels {
            for in_channel in 0..in_channels {
                for h in 0..height {
                    for w in 0..width {
                        let mut sum = 0.;
                        for kh in 0..kernel_size {
                            for kw in 0..kernel_size {
                                let img_val =
                                    padded_input[[in_channel, h + kh, w + kw]];
                                let kernel_val =
                                    kernel[[out_channel, in_channel, kh, kw]];
                                sum += img_val * kernel_val;
                            }
                        }
                        result[[out_channel, h, w]] += sum;
                    }
                }
            }
        }
        result
    }

    /// A straightforward gradient implementation for ∂loss/∂img using loops,
    /// that we can compare our im2col implementation against.
    ///
    /// Written by ChatGPT 4.
    fn slow_dloss_dimg(
        dloss_dresult: ArrayView3<f32>,
        kernel: ArrayView4<f32>,
    ) -> Array3<f32> {
        let (out_channels, height, width) = dloss_dresult.dim();
        let (out_channels_2, in_channels, kernel_size, kernel_size_2) = kernel.dim();
        assert_eq!(kernel_size % 2, 1, "kernel size must be odd");
        assert_eq!(out_channels, out_channels_2);
        assert_eq!(kernel_size, kernel_size_2);
        let padding = kernel_size / 2;

        // Pad our dloss_dimg.
        let mut padded_dloss_dimg =
            Array3::zeros((out_channels, height + 2 * padding, width + 2 * padding));
        padded_dloss_dimg
            .slice_mut(s![.., padding..padding + height, padding..padding + width])
            .assign(&dloss_dresult);

        // Compute the gradient with respect to the input image.
        let mut dloss_dimg = Array3::zeros((in_channels, height, width));
        for in_channel in 0..in_channels {
            for out_channel in 0..out_channels {
                for h in 0..height {
                    for w in 0..width {
                        let mut sum = 0.;
                        for kh in 0..kernel_size {
                            for kw in 0..kernel_size {
                                let dloss_val =
                                    padded_dloss_dimg[[out_channel, h + kh, w + kw]];
                                let kernel_val = kernel[[
                                    out_channel,
                                    in_channel,
                                    kernel_size - 1 - kh,
                                    kernel_size - 1 - kw,
                                ]];
                                sum += dloss_val * kernel_val;
                            }
                        }
                        dloss_dimg[[in_channel, h, w]] += sum;
                    }
                }
            }
        }
        dloss_dimg
    }

    /// A straightforward gradient implementation for ∂loss/∂kernel using loops,
    /// that we can compare our im2col implementation against.
    ///
    /// Originally written by ChatGPT 4.
    fn slow_dloss_dkernel(
        dloss_dresult: ArrayView3<f32>,
        img: ArrayView3<f32>,
        kernel_shape: (usize, usize, usize, usize),
    ) -> Array4<f32> {
        let (out_channels, in_channels, kernel_size, kernel_size_2) = kernel_shape;
        assert_eq!(kernel_size, kernel_size_2);
        assert_eq!(kernel_size % 2, 1, "kernel size must be odd");
        let (out_channels_2, height, width) = dloss_dresult.dim();
        let (in_channels_2, height_2, width_2) = img.dim();
        assert_eq!(in_channels, in_channels_2);
        assert_eq!(out_channels, out_channels_2);
        assert_eq!(height, height_2);
        assert_eq!(width, width_2);

        let padding = kernel_size / 2;

        // Pad our input image.
        let mut padded_input =
            Array3::zeros((in_channels, height + 2 * padding, width + 2 * padding));
        padded_input
            .slice_mut(s![.., padding..padding + height, padding..padding + width])
            .assign(&img);

        let mut dkernel_dresult = Array4::zeros(kernel_shape);
        for out_channel in 0..out_channels {
            for in_channel in 0..in_channels {
                for h in 0..height {
                    for w in 0..width {
                        let dloss_dresult_val = dloss_dresult[[out_channel, h, w]];
                        for kh in 0..kernel_size {
                            for kw in 0..kernel_size {
                                let img_val =
                                    padded_input[[in_channel, h + kh, w + kw]];
                                dkernel_dresult[[out_channel, in_channel, kh, kw]] +=
                                    dloss_dresult_val * img_val;
                            }
                        }
                    }
                }
            }
        }
        dkernel_dresult
    }

    /// Gradients for conv2d generated by PyTorch.
    #[derive(Debug, Deserialize)]
    pub struct ConvGradients {
        #[serde(deserialize_with = "deserialize_array4")]
        pub outputs: Array4<f32>,
        #[serde(deserialize_with = "deserialize_array4")]
        pub inputs: Array4<f32>,
        #[serde(deserialize_with = "deserialize_array4")]
        pub filters: Array4<f32>,
        #[serde(deserialize_with = "deserialize_array1")]
        pub biases: Array1<f32>,
    }

    /// A fixture for conv2d generated by PyTorch.
    #[derive(Debug, Deserialize)]
    pub struct ConvFixture {
        #[serde(deserialize_with = "deserialize_array4")]
        pub inputs: Array4<f32>,
        #[serde(deserialize_with = "deserialize_array4")]
        pub filters: Array4<f32>,
        #[serde(deserialize_with = "deserialize_array1")]
        pub biases: Array1<f32>,
        #[serde(deserialize_with = "deserialize_array4")]
        pub outputs: Array4<f32>,
        #[serde(deserialize_with = "deserialize_array4")]
        pub targets: Array4<f32>,
        pub gradients: ConvGradients,
    }

    /// Load our conv2d fixture.
    pub fn load_conv2d_fixture() -> Result<ConvFixture> {
        let json: &str = include_str!("../fixtures/layers/conv2d.json");
        Ok(serde_json::from_str(json)?)
    }

    /// "Boostrap" our confidence in the system by comparing our "slow" versions
    /// to the numbers we computed using PyTorch.
    #[test]
    fn slow_functions_match_fixtures() {
        let fixture = load_conv2d_fixture().unwrap();

        // Make sure that slow_conv2d matches the numbers we computed using
        // PyTorch. This is bit tricky, because we also need to add in the bias
        // used by the fixture.
        let result = slow_conv2d(
            fixture.inputs.slice(s![0, .., .., ..]),
            fixture.filters.view(),
        );
        let mut with_biases = result.clone();
        for (out_channel, bias) in fixture.biases.iter().enumerate() {
            with_biases
                .slice_mut(s![out_channel, .., ..])
                .add_assign(*bias);
        }
        assert_relative_eq!(
            with_biases,
            fixture.outputs.slice(s![0, .., .., ..]),
            epsilon = 1e-6
        );

        // Make sure that slow_dloss_dimg matches the numbers we computed using
        // PyTorch.
        let dloss_dresult = fixture.gradients.outputs.slice(s![0, .., .., ..]);
        let dloss_dimg = slow_dloss_dimg(dloss_dresult.view(), fixture.filters.view());
        assert_relative_eq!(
            dloss_dimg,
            fixture.gradients.inputs.slice(s![0, .., .., ..]),
            epsilon = 1e-6
        );

        // Make sure that slow_dloss_dkernel matches the numbers we computed
        // using PyTorch.
        let dloss_dkernel_0 = slow_dloss_dkernel(
            dloss_dresult.view(),
            fixture.inputs.slice(s![0, .., .., ..]),
            fixture.filters.dim(),
        );
        let dloss_dkernel_1 = slow_dloss_dkernel(
            fixture.gradients.outputs.slice(s![1, .., .., ..]).view(),
            fixture.inputs.slice(s![1, .., .., ..]),
            fixture.filters.dim(),
        );
        assert_relative_eq!(
            dloss_dkernel_0 + dloss_dkernel_1,
            fixture.gradients.filters.view(),
            epsilon = 1e-6
        );
    }

    #[rustfmt::skip]
    #[test]
    fn im2col_conv2d_handles_simple_case() {
        let img = array![[
            [0., 0., 0.],
            [0., 1., 0.],
            [0., 5., 0.],
        ]];
        let kernel_0 = array![[
            [0., 0., 0.],
            [0., 1., 0.],
            [0., 0., 0.],
        ]];
        let kenrel_1 = array![[
            [0., 0., 0.],
            [0., 2., 0.],
            [0., 0., 3.],
        ]];
        let mut kernel = Array4::zeros((2, 1, 3, 3));
        kernel.slice_mut(s![0, .., .., ..]).assign(&kernel_0);
        kernel.slice_mut(s![1, .., .., ..]).assign(&kenrel_1);
        let expected = array![
            // TODO: Check this for correctness.
            [
                [0., 0., 0.],
                [0., 1., 0.],
                [0., 5., 0.],
            ],
            [
                [3., 0., 0.],
                [15., 2., 0.],
                [0., 10., 0.],
            ],
        ];
        let slow_result = slow_conv2d(img.view(), kernel.view());
        assert_eq!(slow_result, expected);
        let mut im2col = Im2ColConv::new(img.dim(), kernel.dim());
        let mut result = Array3::zeros(expected.dim());
        im2col.conv2d(img.view(), kernel.view(), &mut result.view_mut());
        assert_eq!(result, expected);
    }

    #[test]
    fn im2col_conv2d_matches_slow_conv2d_on_fixture_data() {
        // Check against our reference implementation.
        let fixture = load_conv2d_fixture().unwrap();
        let img = fixture.inputs.slice(s![0, .., .., ..]);
        let kernel = fixture.filters.view();
        let slow_result = slow_conv2d(img, kernel);
        let mut im2col = Im2ColConv::new(img.dim(), kernel.dim());
        let mut result = Array3::zeros(slow_result.dim());
        im2col.conv2d(img.view(), kernel.view(), &mut result.view_mut());
        assert_relative_eq!(result, slow_result, epsilon = 1e-6);
    }

    #[test]
    fn im2col_dloss_dimg_matches_slow_dloss_dimg_on_fixture_data() {
        // Check against our reference implementation.
        let fixture = load_conv2d_fixture().unwrap();
        let img = fixture.inputs.slice(s![0, .., .., ..]);
        let dloss_dresult = fixture.gradients.outputs.slice(s![0, .., .., ..]);
        let kernel = fixture.filters.view();
        let slow_result = slow_dloss_dimg(dloss_dresult, kernel);
        let mut im2col = Im2ColDLossDImg::new(img.dim(), kernel.dim());
        let mut result = Array3::zeros(slow_result.dim());
        im2col.dloss_dimg(dloss_dresult.view(), kernel.view(), &mut result.view_mut());
        assert_relative_eq!(result, slow_result, epsilon = 1e-6);
    }
}
