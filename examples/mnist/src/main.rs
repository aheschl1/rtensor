use std::{collections::binary_heap::Iter, path::PathBuf};

use rand::seq::SliceRandom;
use rtensors::{backend::{Backend, cpu::Cpu, cuda::Cuda}, core::{MetaTensorView, Tensor, primitives::CudaTensor, tensor::{AsView, RandomTensor, TensorAccess, WithGrad}, value::WeightValue}, grad::{self, optim::{Optim, SGD}, primitives::GradTensor}, ops::{broadcast::l1::mean_l1_loss, linalg::MatMul, unary::UnaryGradOp}};

struct Layer<T: WeightValue, B: Backend> {
    pub weight: GradTensor<T, B>,
    pub bias: GradTensor<T, B>,
}

struct DenseModel<T: WeightValue, B: Backend> {
    pub layers: Vec<Layer<T, B>>,
}

impl DenseModel<f32, Cuda> {
    fn new(input_size: usize, hidden_size: usize, output_size: usize, num_layers: usize) -> Self {
        let mut layers = Vec::new();
        for i in 0..num_layers {
            let in_size = if i == 0 { input_size } else { hidden_size };
            let out_size = if i == num_layers - 1 { output_size } else { hidden_size };
            let weight = CudaTensor::<f32>::uniform((in_size, out_size))
                .expect("Failed to create uniform tensor").param();
            let bias = CudaTensor::<f32>::zeros((1, out_size)).param();
            layers.push(Layer { weight, bias });
        }
        Self { layers }
    }

    fn forward(&self, mut x: GradTensor<f32, Cuda>) -> GradTensor<f32, Cuda> {
        for (i, layer) in self.layers.iter().enumerate() {
            x = x.matmul(&layer.weight).unwrap() + &layer.bias;
            if i != self.layers.len() - 1 {
                x = x.relu();
            }
        }
        x.sigmoid()
    }

    fn register(&self, optim: &mut SGD<f32, Cuda>) {
        for layer in &self.layers {
            optim.register_parameter(&layer.weight).unwrap();
            optim.register_parameter(&layer.bias).unwrap();
        }
    }
}

struct MnistDataset {
    images: Vec<GradTensor<f32, Cuda>>,
    targets: Vec<GradTensor<f32, Cuda>>
}

struct MnistIter<'a> {
    images: &'a [GradTensor<f32, Cuda>],
    targets: &'a [GradTensor<f32, Cuda>],
    ordering: Vec<usize>,
    idx: usize,
}

impl<'a> MnistIter<'a> {
    fn new(dataset: &'a MnistDataset) -> Self {
        let ordering: Vec<usize> = (0..dataset.images.len()).collect();
        let mut s = Self{
            images: &dataset.images,
            targets: &dataset.targets,
            ordering,
            idx: 0,
        };
        s.reset();
        s
    }
    fn reset(&mut self) {
        self.idx = 0;
        // reshuffle ordering
        let mut rng = rand::rng();
        self.ordering.shuffle(&mut rng);
    }
}

impl<'a> Iterator for MnistIter<'a> {
    type Item = (&'a GradTensor<f32, Cuda>, &'a GradTensor<f32, Cuda>);

    fn next(&mut self) -> Option<Self::Item> {
        if self.idx >= self.ordering.len() {
            return None;
        }
        let sample_idx = self.ordering[self.idx];
        self.idx += 1;
        Some((&self.images[sample_idx], &self.targets[sample_idx]))
    }
}

impl MnistDataset {
    fn load(path: PathBuf) -> Self {
        let mut images: Vec<GradTensor<f32, Cuda>> = Vec::new();
        let mut targets: Vec<GradTensor<f32, Cuda>> = Vec::new();
        // path points at a folder which holds folders: 0, 1, 2, 3, ..., 9 and in each is a bunch of images
        let subfolders = std::fs::read_dir(path).unwrap();
        for entry in subfolders {
            let entry = entry.unwrap();
            let label: u8 = entry.file_name().to_str().unwrap().parse().unwrap();
            let image_files = std::fs::read_dir(entry.path()).unwrap();
            for image_entry in image_files.into_iter().take(100) {
                let image_entry = image_entry.unwrap();
                let image = image::open(image_entry.path()).unwrap();
                let tensor: Tensor::<u8> = image.into();
                let size = tensor.size();
                let img = tensor.into_dtype::<f32>()
                    .expect("Failed to convert tensor dtype")
                    .view_as((1, size))
                    .expect("Failed to flatten") / 255.0;
                images.push(img.cuda().unwrap().grad());
                let mut target: Vec<f32> = vec![0.; 10];
                target[label as usize] = 1.;
                targets.push(CudaTensor::from_buf(target, (1, 10)).expect("Failed to create target tensor").grad());
            }
        }
        // Placeholder implementation
        Self {
            images,
            targets,
        }
    }
}

fn main () {
    grad::with::<f32, Cuda>(|ctx| {
        let train_dset = MnistDataset::load(PathBuf::from("../data/mnist/training"));
        println!("Loaded {} training samples", train_dset.images.len());
        let size = train_dset.images[0].borrow().size();
        let model = DenseModel::new(size, 15, 10, 10);
        let mut optim = SGD::<f32, Cuda>::new(0.01);
        model.register(&mut optim);

        // no batching yet so accumulate loss over multiple samples
        let mut nsamples = 0;
        let virtual_batch = 32;
        let epochs = 100;
        for _epoch in 0..epochs {
            let iterator = MnistIter::new(&train_dset);
            let mut total_loss = 0.0;
            let mut loss_samples = 0;
            for (x, y) in iterator.into_iter() {
                let input = x.grad();
                let target = y.grad();
    
                let out = model.forward(input);
                let loss = mean_l1_loss(&out, &target);
                total_loss += loss.borrow().item().expect("Failed to get loss item");
                loss_samples += 1;
                ctx.backwards(&loss).expect("Backwards failed");
                nsamples += 1;
                if nsamples >= virtual_batch {
                    optim.step().expect("Optimizer step failed");
                    nsamples = 0;
                }
            }
            println!("Epoch {}: Average Loss = {}", _epoch + 1, total_loss / loss_samples as f32);
        }
    });
}