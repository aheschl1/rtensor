use crate::{backend::{Backend, BackendMatMul}, core::{primitives::TensorBase, shape_to_stride, tensor::{AsView, AsViewMut}, value::WeightValue, MetaTensor, MetaTensorView, Shape, TensorView, TensorViewMut}, ops::linalg::{Conv, ConvConfig2D}};


impl<K, A, T, B> Conv<K, T, B> for A
where
    K: AsView<T, B>,
    A: AsView<T, B>,
    T: WeightValue,
    B: BackendMatMul<T>,
{
    type Output = TensorBase<T, B>;

    fn conv2d(&self, kernel: &K, config: &ConvConfig2D) -> Result<Self::Output, crate::core::tensor::TensorError> {
        let kernel_view = kernel.view();
        let mut self_view = self.view();
        // check dims. expecting [B, C, H, W] for input and [K, C, KH, KW] for kernel C must match
        // if the shape of the input is [C, H, W] expand to [1, C, H, W]
        // if the shape of the input is [H, W] expand to [1, 1, H, W]
        // the kernel has no flexibility, it must be [K, C, KH, KW]
        while self_view.rank() < 4 {
            self_view.unsqueeze_inplace();
        }

        if self_view.rank() != 4 {
            return Err(crate::core::tensor::TensorError::InvalidShape(format!(
                "Input must be no more than a 4D tensor [B, C, H, W], got rank {}",
                self_view.rank()
            )));
        }

        // validate channel match
        if self_view.shape()[1] != kernel_view.shape()[1] {
            return Err(crate::core::tensor::TensorError::InvalidShape(format!(
                "Input channels ({}) do not match kernel channels ({})",
                self_view.shape()[1],
                kernel_view.shape()[1]
            )));
        }

        // validate kernel size
        if kernel_view.rank() != 4 {
            return Err(crate::core::tensor::TensorError::InvalidShape(format!(
                "Kernel must be 4D tensor [K, C, KH, KW], got rank {}",
                kernel_view.rank()
            )));
        }

        let out_shape = compute_output_convolution_shape_2d(
            &self_view.shape(),
            &kernel_view.shape(),
            &config.stride,
            &config.padding,
        );

        let mut out_buffer = self_view.backend.alloc::<T>(out_shape.size())?;
        let out_backend = self_view.backend.clone();
        // out_backend.conv_2d

        // out_backend.apply_conv_2d(
        //     (&self_view.buf, self_view.meta()),
        //     (&kernel_view.buf, kernel_view.meta()), 
        //     &mut out_buffer, 
        //     config
        // )?;

        // temp_conv_slowpath()
        let out_strides = shape_to_stride(&out_shape);
        let out_meta = MetaTensor::new(out_shape, out_strides, 0);
        let mut out_tensor = TensorBase::<T, B>::from_parts(out_backend, out_buffer, out_meta);
        temp_conv_fallback(
            self_view,
            kernel_view,
            config,
            &mut out_tensor.view_mut(),
        );
        Ok(out_tensor)
    }
}

#[inline]
fn temp_conv_fallback<T: WeightValue, B: BackendMatMul<T>>(
    input: TensorView<T, B>,
    kernel: TensorView<T, B>,
    config: &ConvConfig2D,
    output: &mut TensorViewMut<T, B>,
) {
    // start by making a padded input
    let padding = config.padding;
    // let _storage = None;

    let input_view = if padding.0 > 0 || padding.1 > 0 {
        input
    } else {
        input
    };

}

#[inline]
fn compute_output_convolution_shape_2d(
    input_shape: &Shape,
    kernel_shape: &Shape,
    stride: &(usize, usize),
    padding: &(usize, usize),
) -> Shape {
    if input_shape.len() != 4 || kernel_shape.len() != 4 {
        panic!("Input and kernel must be 4D tensors");
    }
    let batch_size = input_shape[0];
    let out_channels = kernel_shape[0];
    // input_shape: [B, C, H, W]
    // kernel_shape: [K, C, KH, KW]
    let in_height = input_shape[2];
    let in_width = input_shape[3];
    let kernel_height = kernel_shape[2];
    let kernel_width = kernel_shape[3];

    let out_height = (in_height + 2*padding.0 - (kernel_height - 1) - 1) / stride.0 + 1;
    let out_width = (in_width + 2*padding.1 - (kernel_width - 1) - 1) / stride.1 + 1;

    Shape::from((batch_size, out_channels, out_height, out_width))
}