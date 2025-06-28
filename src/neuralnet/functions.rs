//! A module that has the [`ActivationFunctions`] and [`LossFunctions`].

/// An enum holding various different activation functions used in machine learning and other applications.
#[allow(dead_code)]
#[derive(Default)]
pub enum ActivationFunctions {
    #[default]
    /// Default value of [`ActivationFunctions`]. Used to assert a valid function configuration.
    None,
    /// The Rectified Linear Unit (ramp) function, useful in machine learning.
    ReLU,
    /// The Sigmoid function (logistic function) is a smooth approximation of the rectifier, the Heaviside step function.
    /// The Sigmoid function had prevalent use in machine learning, slowly superseded by [`ActivationFunctions::ReLU`].
    Sigmoid,
    /// The Sigmoid Linear Unit (sometimes called swish) function is a less computationally expensive aproximation
    /// of the GELU and Mish functions
    SiLU,
    /// The hyperbolic tangent function.
    Tanh,
    /// A similar function to the [`ActivationFunctions::Tanh`] function, having slightly smaller slope.
    SoftSign,
    /// A smooth approximation of the [`ActivationFunctions::ReLU`] function.
    SoftPlus,
    /// The gaussian activation function.
    Gaussian,
}

/// Useful type alias for the activation functions used throughout the neural network code.
pub type ActivationFunction = fn(x: &f64) -> f64;

impl ActivationFunctions {
    /// Returns the [`ActivationFunction`] of the chosen enum variant.
    pub fn function(&self) -> ActivationFunction {
        use ActivationFunctions::*;

        match self {
            None => unreachable!(),
            ReLU => |x| x.max(0.0),
            Sigmoid => |x| 1.0 / (1.0 + (-x).exp()),
            SiLU => |x| x / (1.0 + (-x).exp()),
            Tanh => |x| x.tanh(),
            SoftSign => |x| x / (1.0 + x.abs()),
            SoftPlus => |x| (1.0 + x.exp()).ln(),
            Gaussian => |x| (-(x * x)).exp(),
        }
    }

    /// Returns the derivative [`ActivationFunction`] of the chosen enum variant.
    pub fn derivative(&self) -> ActivationFunction {
        use ActivationFunctions::*;

        match self {
            None => unreachable!(),
            ReLU => |_| 1.0,
            Sigmoid => |x| {
                let exp = x.exp();
                exp * (exp + 1.0).powi(-2)
            },
            SiLU => |x| {
                let exp = x.exp();
                exp * (x + exp + 1.0).powi(-2)
            },
            Tanh => |x| x.cosh().powi(-2),
            SoftSign => |x| (1.0 + x.abs()).powi(-2),
            SoftPlus => |x| {
                let exp = x.exp();
                exp * (1.0 + exp).powi(-1)
            },
            Gaussian => |x| -2.0 * (-(x * x)).exp() * x,
        }
    }
}

/// An enum having loss (sometimes called cost) functions.
#[allow(dead_code)]
#[derive(Default)]
pub enum LossFunctions {
    #[default]
    /// Default value of [`LossFunctions`]. Used to assert a valid function configuration.
    None,
    /// The squared error loss (SEL) function. Evaluated as `(x - y)^2`. Its derivative being `2*(x - y)`.
    SquaredDifference,
    /// (Unimplemented)
    CrossEntropy,
}

/// Useful type alias for the loss functions used throughout the neural network code.
pub type LossFunction = fn(x: &f64, y: &f64) -> f64;

impl LossFunctions {
    /// Returns the [`LossFunction`] of the chosen enum variant.
    pub fn function(&self) -> LossFunction {
        use LossFunctions::*;

        match self {
            None => unreachable!(),
            SquaredDifference => |x, y| (x - y).powi(2),
            CrossEntropy => unimplemented!(),
        }
    }

    /// Returns the derivative [`LossFunction`] of the chosen enum variant.
    pub fn derivative(&self) -> LossFunction {
        use LossFunctions::*;

        match self {
            None => unreachable!(),
            SquaredDifference => |x, y| 2.0 * (x - y),
            CrossEntropy => unimplemented!(),
        }
    }
}

/// Returns the sum of applying the `loss` function over the `result` and `expected` slices.
pub fn total_loss(loss: LossFunction, result: &[f64], expected: &[f64]) -> f64 {
    assert_eq!(result.len(), expected.len(), "expected equal length slices");

    result
        .iter()
        .zip(expected)
        .map(|(got, want)| loss(got, want))
        .sum()
}
