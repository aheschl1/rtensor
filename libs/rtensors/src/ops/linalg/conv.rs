use crate::{backend::{BackendMatMul}, core::{primitives::TensorBase, shape_to_stride, tensor::{AsTensor, AsView, AsViewMut, TensorAccess, TensorAccessMut}, value::WeightValue, MetaTensor, MetaTensorView, Shape, TensorView, TensorViewMut}, ops::linalg::{Conv, MatMul}};

#[derive(Clone, Copy, Debug)]
pub enum PaddingType {
    Zeros
}

#[derive(Clone, Copy, Debug)]
pub struct ConvConfig2D {
    pub stride: ConvStrides2D,
    pub padding: Padding2D,
    pub padding_type: PaddingType,
}

impl ConvConfig2D {
    pub fn new(stride: impl Into<ConvStrides2D>, padding: impl Into<Padding2D>, padding_type: PaddingType) -> Self {
        ConvConfig2D {
            stride: stride.into(),
            padding: padding.into(),
            padding_type,
        }
    }

    pub fn stride(mut self, stride: impl Into<ConvStrides2D>) -> Self {
        self.stride = stride.into();
        self
    }

    pub fn padding(mut self, padding: impl Into<Padding2D>) -> Self {
        self.padding = padding.into();
        self
    }

    pub fn padding_type(mut self, padding_type: PaddingType) -> Self {
        self.padding_type = padding_type;
        self
    }
}

impl Default for ConvConfig2D {
    fn default() -> Self {
        ConvConfig2D {
            stride: ConvStrides2D(1, 1),
            padding: Padding2D(0, 0),
            padding_type: PaddingType::Zeros,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Padding2D(usize, usize);

impl From<usize> for Padding2D {
    fn from(value: usize) -> Self {
        Padding2D(value, value)
    }
}

impl From<(usize, usize)> for Padding2D {
    fn from(value: (usize, usize)) -> Self {
        Padding2D(value.0, value.1)
    }
}

#[derive(Clone, Copy, Debug)]
pub struct ConvStrides2D(usize, usize);

impl From<usize> for ConvStrides2D {
    fn from(value: usize) -> Self {
        ConvStrides2D(value, value)
    }
}

impl From<(usize, usize)> for ConvStrides2D {
    fn from(value: (usize, usize)) -> Self {
        ConvStrides2D(value.0, value.1)
    }
}

#[derive(Clone, Copy, Debug)]
pub struct ConvConfig3D {
    pub stride: ConvStrides3D,
    pub padding: Padding3D,
    pub padding_type: PaddingType,
}

impl ConvConfig3D {
    pub fn new(stride: impl Into<ConvStrides3D>, padding: impl Into<Padding3D>, padding_type: PaddingType) -> Self {
        ConvConfig3D {
            stride: stride.into(),
            padding: padding.into(),
            padding_type,
        }
    }

    pub fn stride(mut self, stride: impl Into<ConvStrides3D>) -> Self {
        self.stride = stride.into();
        self
    }

    pub fn padding(mut self, padding: impl Into<Padding3D>) -> Self {
        self.padding = padding.into();
        self
    }

    pub fn padding_type(mut self, padding_type: PaddingType) -> Self {
        self.padding_type = padding_type;
        self
    }
}

impl Default for ConvConfig3D {
    fn default() -> Self {
        ConvConfig3D {
            stride: ConvStrides3D(1, 1, 1),
            padding: Padding3D(0, 0, 0),
            padding_type: PaddingType::Zeros,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Padding3D(usize, usize, usize);

impl From<usize> for Padding3D {
    fn from(value: usize) -> Self {
        Padding3D(value, value, value)
    }
}

impl From<(usize, usize, usize)> for Padding3D {
    fn from(value: (usize, usize, usize)) -> Self {
        Padding3D(value.0, value.1, value.2)
    }
}

#[derive(Clone, Copy, Debug)]
pub struct ConvStrides3D(usize, usize, usize);

impl From<usize> for ConvStrides3D {
    fn from(value: usize) -> Self {
        ConvStrides3D(value, value, value)
    }
}

impl From<(usize, usize, usize)> for ConvStrides3D {
    fn from(value: (usize, usize, usize)) -> Self {
        ConvStrides3D(value.0, value.1, value.2)
    }
}

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

        if config.stride.0 == 0 || config.stride.1 == 0 {
            return Err(crate::core::tensor::TensorError::UnsupportedOperation(
                "Stride values must be greater than 0".to_string(),
            ));
        }

        let out_shape = compute_output_convolution_shape_2d(
            &self_view.shape(),
            &kernel_view.shape(),
            &config.stride,
            &config.padding,
        );

        let out_buffer = self_view.backend.alloc::<T>(out_shape.size())?;
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
        )?;
        Ok(out_tensor)
    }
}

#[inline]
fn temp_conv_fallback<T: WeightValue, B: BackendMatMul<T>>(
    input: TensorView<T, B>,
    kernel: TensorView<T, B>,
    config: &ConvConfig2D,
    output: &mut TensorViewMut<T, B>,
) -> Result<(), crate::core::tensor::TensorError> {
    // start by making a padded input
    let padding = config.padding;
    let mut _storage = None;

    let input_view = if padding.0 > 0 || padding.1 > 0 {
        _storage = Some(input.pad(
            (0, 0, padding.0, padding.1), 
            &config.padding_type
        )?);
        _storage.as_ref().unwrap().view()
    } else {
        input
    };

    // kernel is shape [K, C, KH, KW] which is [in channels, out channels, kernel height, kernel width]
    let kernel_width = kernel.shape()[3];
    let kernel_height = kernel.shape()[2];

    // now we have a padded input. perform convolution using a naive sliding window approach
    for coord in input_view.iter_coords() {
        // validate edge
        if coord[2] < kernel_height / 2 || coord[2] >= input_view.shape()[2] - kernel_height / 2 {
            continue;
        }
        if coord[3] < kernel_width / 2 || coord[3] >= input_view.shape()[3] - kernel_width / 2 {
            continue;
        }
        // input is shape [B, C, H, W]
        let input_slice = input_view.slice(2, coord[2] - kernel_height / 2 ..= coord[2] + kernel_height / 2)?;
        let input_slice = input_slice.slice(3, coord[3] - kernel_width / 2 ..= coord[3] + kernel_width / 2)?;
        let input_slice = input_slice.slice(0, coord[0])?; // batch slice
        // shape is now [C, KH, KW]
        // now iterate through each output channel
        for oc in 0..kernel.shape()[0] {
            let output_coord = (
                coord[0], // batch
                oc,             // out channel will be iterated over
                ((coord[2] as isize - (kernel_height / 2) as isize) / config.stride.0 as isize) as usize,
                ((coord[3] as isize - (kernel_width / 2) as isize) / config.stride.1 as isize) as usize,
            );
            let kernel_oc = kernel.slice(0, oc)?;
            // shape is now [C, KH, KW]
            // flatten both and do dot product
            let in_shape = input_slice.size();
            let input_flat = input_slice.reshape((in_shape,))?;
            let kernel_flat = kernel_oc.reshape((in_shape,))?;
            let dot = input_flat.dot(&kernel_flat)?;
            output.set(
                output_coord.clone(), 
                dot.item()?
            ).expect(format!("Failed to set output value at coord {:?}", output_coord).as_str());
        }
    }

    Ok(())

}

#[inline]
fn compute_output_convolution_shape_2d(
    input_shape: &Shape,
    kernel_shape: &Shape,
    stride: &ConvStrides2D,
    padding: &Padding2D,
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

#[cfg(test)]
mod tests {
    use crate::{core::Tensor, ops::linalg::{conv::ConvConfig2D, Conv}};

    #[test]
    fn conv_playground() {
        let tensor = Tensor::<f32>::from_buf(vec![
            1., 2., 3., 4., 5.,
            6., 7., 8., 9., 10.,
            11.,12.,13.,14.,15.,
            16.,17.,18.,19.,20.,
            21.,22.,23.,24.,25.,

            1., 2., 3., 4., 5.,
            6., 7., 8., 9., 10.,
            11.,12.,13.,14.,15.,
            16.,17.,18.,19.,20.,
            21.,22.,23.,24.,25.,

            1., 2., 3., 4., 5.,
            6., 7., 8., 9., 10.,
            11.,12.,13.,14.,15.,
            16.,17.,18.,19.,20.,
            21.,22.,23.,24.,25.,

            1., 2., 3., 4., 5.,
            6., 7., 8., 9., 10.,
            11.,12.,13.,14.,15.,
            16.,17.,18.,19.,20.,
            21.,22.,23.,24.,25.,
        ], (2, 2, 5, 5)).unwrap(); // b, c, h, w

        let kernel = Tensor::<f32>::from_buf(vec![
            1., 0., -1.,
            1., 0., -1.,
            1., 0., -1.,

            1., 0., -1.,
            1., 0., -1.,
            1., 0., -1.,

            1., 0., -1.,
            1., 0., -1.,
            1., 0., -1.,

            1., 0., -1.,
            1., 0., -1.,
            1., 0., -1.,
        ], (2, 2, 3, 3)).unwrap(); // k, c, kh, kw

        let config = ConvConfig2D::default().padding(1);
        let out = tensor.conv2d(&kernel, &config).unwrap();

        println!("Convolution Output: {:?}", out);
    }
}