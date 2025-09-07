use super::SimpleOptimizer;
use super::adaptor::OptimizerAdaptor;
use crate::grad_clipping::GradientClippingConfig;
use crate::record::Record;
use crate::tensor::{
    Tensor,
    backend::{AutodiffBackend, Backend},
    ops::Device,
};
use crate::{self as burn};
use crate::{config::Config, module::AutodiffModule};

/// Muon configuration.
#[derive(Config)]
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

/// Muon optimizer as described in
/// [Muon: An optimizer for hidden layers in neural networks](https://kellerjordan.github.io/posts/muon/)
#[derive(Clone)]
pub struct Muon {
    weight_decay: f32,
    momentum: f32,
}

impl Muon {
    /// Six step Newton-Shulz method by @YouJiacheng
    /// from https://github.com/modula-systems/modula/blob/aed70cddf2d3ab74fa218b1377840d1fd795cfcf/modula/atom.py#L6C1-L31C13
    const ABC_LIST: [f32; 18] = [
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

    fn newtonshulz5<B: Backend>(&self, grad: Tensor<B, 2>) -> Tensor<B, 2> {
        let mut x = grad.clone();
        let dims: [usize; 2] = x.shape().dims();
        let transpose = dims[dims.len() - 2] > dims[dims.len() - 1];
        if transpose {
            x = x.transpose();
        }
        let norm = x.clone().abs().powi_scalar(2).sum().sqrt();
        x = x.div_scalar(norm.into_scalar());
        for i in 0..6 {
            let a = x.clone().transpose().matmul(x.clone());
            let eye = Tensor::<B, 2>::eye(dims[0], &x.device());
            x = Muon::ABC_LIST[i] * eye
                + Muon::ABC_LIST[i + 1] * a.clone()
                + Muon::ABC_LIST[i + 2] * a.clone().matmul(a.clone());
        }

        if transpose { x.transpose() } else { x }
    }
}

impl<B: Backend> SimpleOptimizer<B> for Muon {
    type State<const D: usize> = MuonState<B, D>;

    fn step<const D: usize>(
        &self,
        lr: crate::LearningRate,
        tensor: Tensor<B, D>,
        grad: Tensor<B, D>,
        state: Option<Self::State<D>>,
    ) -> (Tensor<B, D>, Option<Self::State<D>>) {
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
        let buffer_dims = momentum_buffer.shape().dims();
        let shaped_buffer = momentum_buffer.reshape([2, -1]);
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
        };
        let mut optimizer = OptimizerAdaptor::from(m);
        if let Some(config) = &self.grad_clipping {
            optimizer = optimizer.with_grad_clipping(config.init());
        }
        optimizer
    }
}

mod tests {}
