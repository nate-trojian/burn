use burn_core as burn;

use burn::config::Config;
use burn::tensor::{Tensor, backend::AutodiffBackend};
use burn::tensor::{backend::Backend, ops::Device};
use burn::{module::AutodiffModule, record::Record};

use super::SimpleOptimizer;
use super::adaptor::OptimizerAdaptor;
use crate::grad_clipping::GradientClippingConfig;

#[cfg(not(feature = "std"))]
#[allow(unused_imports)]
use num_traits::Float as _;

/// [`Muon`] configuration.
#[derive(Config, Debug)]
pub struct MuonConfig {
    /// Learning rate.
    #[config(default = 0.02)]
    lr: f32,
    /// Weight decay config.
    #[config(default = 1e-4)]
    weight_decay: f32,
    /// Momentum.
    #[config(default = 0.95)]
    momentum: f32,
    /// [Gradient Clipping](GradientClippingConfig) config.
    grad_clipping: Option<GradientClippingConfig>,
}

/// Muon state.
#[derive(Record, Clone, new)]
pub struct MuonState<B: Backend, const D: usize> {
    momentum_buffer: Tensor<B, D>,
}

/// Newton-Shulz approximation implementation types.
#[derive(Clone)]
pub enum NewtonShulz {
    /// Stable Newton-Shulz implementation by @YouJiacheng.
    /// from <https://github.com/modula-systems/modula/blob/aed70cddf2d3ab74fa218b1377840d1fd795cfcf/modula/atom.py#L6C1-L31C13>
    Stable,
    /// Performant Newton-Shulz implementation.
    /// from <https://docs.modula.systems/algorithms/newton-schulz/#a-cursed-quintic-iteration>
    Speed,
}

impl NewtonShulz {
    /// Newton-Shulz implementations.
    pub fn execute<B: Backend>(&self, x: &Tensor<B, 2>) -> Tensor<B, 2> {
        match self {
            NewtonShulz::Stable => {
                let abc_list: [f32; 18] = [
                    3955f32 / 1024f32,
                    -8306f32 / 1024f32,
                    5008f32 / 1024f32,
                    3735f32 / 1024f32,
                    -6681f32 / 1024f32,
                    3463f32 / 1024f32,
                    3799f32 / 1024f32,
                    -6499f32 / 1024f32,
                    3211f32 / 1024f32,
                    4019f32 / 1024f32,
                    -6385f32 / 1024f32,
                    2906f32 / 1024f32,
                    2677f32 / 1024f32,
                    -3029f32 / 1024f32,
                    1162f32 / 1024f32,
                    2172f32 / 1024f32,
                    -1833f32 / 1024f32,
                    682f32 / 1024f32,
                ];
                let mut x = x.clone();
                for i in 0..5 {
                    let a = x.clone().matmul(x.clone().transpose());
                    let m = abc_list[3 * i + 1] * a.clone()
                        + abc_list[3 * i + 2] * a.clone().matmul(a.clone());
                    x = abc_list[3 * i] * x.clone() + m.matmul(x.clone());
                }
                x
            }
            NewtonShulz::Speed => {
                let mut x = x.clone();
                let (a, b, c) = (3.4445, -4.7750, 2.0315);
                for _ in 0..5 {
                    let xtx = x.clone().matmul(x.clone().transpose());
                    let m: Tensor<_, 2> = b * xtx.clone() + c * xtx.clone().matmul(xtx.clone());
                    x = a * x.clone() + m.matmul(x.clone());
                }
                x
            }
        }
    }
}

/// Muon optimizer as described in
/// [Muon: An optimizer for hidden layers in neural networks](https://kellerjordan.github.io/posts/muon/)
#[derive(Clone)]
pub struct Muon {
    weight_decay: f32,
    momentum: f32,
    newtown_shulz: NewtonShulz,
}

impl Muon {
    fn newtonshulz5<B: Backend>(&self, grad: Tensor<B, 2>) -> Tensor<B, 2> {
        let mut x = grad.clone();
        let dims = x.dims();
        let transpose = dims[1] > dims[0];
        if transpose {
            x = x.transpose();
        }
        // Frobenius norm
        let norm = x.clone().powi_scalar(2).sum().sqrt();
        x = x.div_scalar(norm.add_scalar(1e-7).into_scalar());
        x = self.newtown_shulz.execute(&x);
        if transpose { x.transpose() } else { x }
    }
}

// TODO - Reimplement Muon as an optimizer
// Add in a way to specify are head or embed layers and filter out scalar layers (D == 1)
// Specify a second optimizer to use for these other layers
impl<B: Backend> SimpleOptimizer<B> for Muon {
    type State<const D: usize> = MuonState<B, D>;

    fn step<const D: usize>(
        &self,
        lr: crate::LearningRate,
        tensor: Tensor<B, D>,
        grad: Tensor<B, D>,
        state: Option<Self::State<D>>,
    ) -> (Tensor<B, D>, Option<Self::State<D>>) {
        if D != 2 {
            panic!("Newton-Schulz iteration requires 2D tensors, got {}D", D);
        }

        let tensor_updated = tensor.clone() - tensor.mul_scalar(lr).mul_scalar(self.weight_decay);
        let prev_momentum_buffer = if let Some(state) = state {
            state.momentum_buffer.clone()
        } else {
            grad.zeros_like()
        };
        // From Muon official implementation: https://github.com/KellerJordan/Muon/blob/master/muon.py#L35-L36
        // They do two back-to-back inplace lerps
        // momentum_buffer + (1 - momentum) * (grad - momentum_buffer) = (momentum_buffer * momentum) + grad - (grad * momentum)
        // grad + momentum * (momentum_buffer * momentum - grad * momentum) = grad + momentum^2 * (momentum_buffer - grad)
        // grad * (1 - momentum^2) + momentum_buffer * momentum^2
        let momentum_squared = self.momentum.powi(2);
        let momentum_buffer = grad.clone().mul_scalar(1f32 - momentum_squared)
            + prev_momentum_buffer.mul_scalar(momentum_squared);
        let state = MuonState {
            momentum_buffer: momentum_buffer.clone(),
        };
        let buffer_dims = momentum_buffer.dims();
        let shaped_buffer = momentum_buffer.reshape([buffer_dims[0] as i32, -1]);
        let update = self.newtonshulz5(shaped_buffer);
        let reshaped_update = update.reshape::<D, [usize; D]>(buffer_dims);

        (tensor_updated - reshaped_update.mul_scalar(lr), Some(state))
    }

    fn to_device<const D: usize>(mut state: Self::State<D>, device: &Device<B>) -> Self::State<D> {
        state.momentum_buffer = state.momentum_buffer.to_device(device);
        state
    }
}

impl MuonConfig {
    /// Initialize a Muon optimizer.
    pub fn init<B: AutodiffBackend, M: AutodiffModule<B>>(&self) -> OptimizerAdaptor<Muon, M, B> {
        let m = Muon {
            weight_decay: self.weight_decay,
            momentum: self.momentum,
            newtown_shulz: NewtonShulz::Speed,
        };
        let mut optimizer = OptimizerAdaptor::from(m);
        if let Some(config) = &self.grad_clipping {
            optimizer = optimizer.with_grad_clipping(config.init());
        }
        optimizer
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{GradientsParams, Optimizer};
    use crate::{LearningRate, TestAutodiffBackend};
    use burn::tensor::ops::FloatElem;
    use burn::tensor::{Distribution, Tensor, Tolerance};
    use burn_nn::{Linear, LinearConfig};

    const LEARNING_RATE: LearningRate = 0.01;

    #[test]
    fn test_muon_optimizer_save_load_state() {
        let device = Default::default();
        // Muon cannot be used for scalar parameters, like bias
        let linear = LinearConfig::new(6, 6).with_bias(false).init(&device);
        let mut optimizer: _ = MuonConfig::new().init();
        let x = Tensor::<TestAutodiffBackend, 2>::random([2, 6], Distribution::Default, &device);
        let grads = linear.forward(x).backward();
        let params = GradientsParams::from_grads(grads, &linear);
        let _ = optimizer.step(LEARNING_RATE, linear, params);

        #[cfg(feature = "std")]
        {
            use burn::record::{BinFileRecorder, FullPrecisionSettings, Recorder};

            BinFileRecorder::<FullPrecisionSettings>::default()
                .record(
                    optimizer.to_record(),
                    std::env::temp_dir().as_path().join("test_optim_muon"),
                )
                .unwrap();
        }
        #[cfg(not(feature = "std"))]
        {
            use crate::record::{BinBytesRecorder, FullPrecisionSettings, Recorder};

            let result = BinBytesRecorder::<FullPrecisionSettings>::default()
                .record(optimizer.to_record(), ())
                .unwrap();
            assert!(!result.is_empty());
        }

        let state_optim_before = optimizer.to_record();
        let state_optim_before_copy = optimizer.to_record();
        let loaded_optimizer =
            MuonConfig::new().init::<TestAutodiffBackend, Linear<TestAutodiffBackend>>();
        let loaded_optimizer = loaded_optimizer.load_record(state_optim_before_copy);
        let state_optim_after = loaded_optimizer.to_record();

        assert_eq!(state_optim_before.len(), state_optim_after.len());
    }

    #[test]
    fn test_newton_shulz_all_zeros() {
        let device: Device<TestAutodiffBackend> = Default::default();
        let grad = Tensor::<TestAutodiffBackend, 2>::zeros([3, 3], &device);
        let expected = Tensor::<TestAutodiffBackend, 2>::zeros([3, 3], &device);
        let grad_norm = grad.clone().powi_scalar(2).sum().sqrt();
        let grad = grad.div_scalar(grad_norm.add_scalar(1e-7).into_scalar());
        let s = NewtonShulz::Speed.execute(&grad);

        let actual = s.to_data();
        let expected = expected.to_data();
        let tolerance = Tolerance::absolute(1e-4);
        actual.assert_approx_eq::<FloatElem<TestAutodiffBackend>>(&expected, tolerance);
    }

    #[test]
    fn test_newton_shulz_all_ones() {
        let device: Device<TestAutodiffBackend> = Default::default();
        let grad = Tensor::<TestAutodiffBackend, 2>::ones([3, 3], &device);
        let expected = Tensor::<TestAutodiffBackend, 2>::from_floats(
            [
                [0.2321, 0.2321, 0.2321],
                [0.2321, 0.2321, 0.2321],
                [0.2321, 0.2321, 0.2321],
            ],
            &device,
        );
        let grad_norm = grad.clone().powi_scalar(2).sum().sqrt();
        let grad = grad.div_scalar(grad_norm.add_scalar(1e-7).into_scalar());
        let s = NewtonShulz::Speed.execute(&grad);

        let actual = s.to_data();
        let expected = expected.to_data();
        let tolerance = Tolerance::absolute(1e-4);
        actual.assert_approx_eq::<FloatElem<TestAutodiffBackend>>(&expected, tolerance);
    }
}
