//! The module that has all of the necessary code for working with [`NeuralNetwork`].

use super::functions::*;

/// A neural network layer representation as a vector of weights, biases,
/// a size, and calculated values.
#[derive(Default)]
struct Layer {
    /// The weights, stored as a flattened `Vec<f64>`, where each weight lies
    /// at `[j * layer_size + k]` (`j` being the current layer,
    /// and `k` being the layer before it).
    weights: Vec<f64>,
    /// The biases, having a random value for each `j`th neuron.
    biases: Vec<f64>,
    /// The values, a `Vec<f64>` of unactivated (weighted input) values
    /// set after a [`Layer::calculate`] pass.
    values: Vec<f64>,
    /// The size of the layer (neuron count).
    size: usize,
}

impl Layer {
    /// Constructs a default [`Layer`] with the specified `size` argument.
    fn new(size: usize) -> Self {
        Self {
            size,
            ..Default::default()
        }
    }

    /// Sets the initial [`Layer::weights`] and [`Layer::biases`]
    /// values to a random range in `[-10; 10]`.
    fn initialize(mut self, neuron_count: usize) -> Self {
        fn random_values(count: usize) -> Vec<f64> {
            (0..count)
                .map(|_| rand::random_range(-10.0..=10.0))
                .collect()
        }

        self.weights = random_values(neuron_count * self.size);
        self.biases = random_values(self.size);

        self
    }

    /// Runs a calculation pass using the provided `inputs` data and `activation` function,
    /// performing a dot product multiplication and returning a collected `Vec<f64>`
    /// of activated weighted inputs.
    ///
    /// In order to provide backpropagation support, each [`Layer::calculate()`] call on a [`Layer`]
    /// keeps track of the unactivated (weighted input) values in [`Layer::values`].
    fn calculate(&mut self, inputs: &[f64], activation: &ActivationFunction) -> Vec<f64> {
        self.values = vec![0.0; self.size];

        for (i, value) in self.values.iter_mut().enumerate() {
            let mut new_value = self.biases[i];

            for (j, input) in inputs.iter().enumerate() {
                new_value += self.weights[i * self.size + j] * input;
            }

            *value = new_value;
        }

        self.values.iter().map(|x| activation(x)).collect()
    }
}

/// The [`NeuralNetworkBuilder`] struct holds all of the necessary configuration
/// settings that need to be set, before constructing a new [`NeuralNetwork`] using
/// the [`NeuralNetworkBuilder::build`] method.
#[derive(Default)]
pub struct NeuralNetworkBuilder {
    input_size: usize,
    hidden_layers: Vec<usize>,
    output_size: usize,
    normalize_inputs: Option<f64>,
    activations: Option<(ActivationFunctions, ActivationFunctions)>,
    loss: Option<LossFunctions>,
    batch_size: usize,
    learning_rate: f64,
}

impl NeuralNetworkBuilder {
    /// Sets the [`NeuralNetwork::input_size`].
    pub fn with_input_size(mut self, size: usize) -> Self {
        assert_eq!(self.input_size, 0, "input size already set");
        assert_ne!(size, 0, "input size cannot be 0");

        self.input_size = size;
        self
    }

    /// Adds a new hidden layer to the [`NeuralNetwork::hidden_layers`]
    /// with the specified `size`.
    pub fn with_hidden_layer_of_size(mut self, size: usize) -> Self {
        assert_ne!(size, 0, "hidden layer size cannot be 0");

        self.hidden_layers.push(size);
        self
    }

    /// Creates a new layer of specified `size` and
    /// sets it to the [`NeuralNetwork::output_layer`].
    pub fn with_output_size(mut self, size: usize) -> Self {
        assert_eq!(self.output_size, 0, "output size already set");
        assert_ne!(size, 0, "output size cannot be 0");

        self.output_size = size;
        self
    }

    /// Sets the [`NeuralNetwork::normalize_inputs`] settings.
    pub fn normalize_inputs(mut self, max: u32) -> Self {
        assert!(self.normalize_inputs.is_none(), "normalization already set");

        self.normalize_inputs = Some(max as f64);
        self
    }

    /// Sets the [`NeuralNetwork::activations`] functions.
    pub fn with_activations(
        mut self,
        neuron: ActivationFunctions,
        output: ActivationFunctions,
    ) -> Self {
        assert!(
            self.activations.is_none(),
            "activation functions already set"
        );

        self.activations = Some((neuron, output));
        self
    }

    /// Sets the [`NeuralNetwork::loss`] function.
    pub fn with_loss(mut self, loss: LossFunctions) -> Self {
        assert!(self.loss.is_none(), "loss function already set");

        self.loss = Some(loss);
        self
    }

    /// Sets the [`NeuralNetwork::batch_size`].
    pub fn with_batch_size(mut self, size: usize) -> Self {
        assert_eq!(self.batch_size, 0, "batch size already set");
        assert_ne!(size, 0, "batch size cannot be 0");

        self.batch_size = size;
        self
    }

    /// Sets the [`NeuralNetwork::learning_rate`].
    pub fn with_learning_rate(mut self, rate: f64) -> Self {
        assert_eq!(self.learning_rate, 0.0, "learning rate already set");
        assert_ne!(rate, 0.0, "learning rate cannot be 0");

        self.learning_rate = rate;
        self
    }

    /// Creates a new [`NeuralNetwork`] with all of the configured settings.
    pub fn build(self) -> NeuralNetwork {
        assert_ne!(self.input_size, 0, "input size has to be set");
        assert_ne!(self.output_size, 0, "output size has to be set");
        assert_ne!(self.batch_size, 0, "batch size has to be set");
        assert_ne!(self.learning_rate, 0.0, "learning rate has to be set");

        // Keep track of the previous layer size to know how big
        // the next layer weights vector should be
        let mut last_size = self.input_size;
        let hidden_layers = self
            .hidden_layers
            .iter()
            .map(|&size| {
                let layer = Layer::new(size).initialize(last_size);
                last_size = size;
                layer
            })
            .collect();
        let normalize_inputs = (
            self.normalize_inputs.is_some(),
            self.normalize_inputs.unwrap_or(0.0),
        );
        let Some((neuron_activation, output_activation)) = self.activations else {
            panic!("activation functions have to be set")
        };
        let Some(loss) = self.loss else {
            panic!("loss function has to be set")
        };

        NeuralNetwork {
            input_size: self.input_size,
            hidden_layers,
            output_layer: Layer::new(self.output_size).initialize(last_size),
            normalize_inputs,
            activations: (neuron_activation, output_activation),
            loss: loss,
            batch_size: self.batch_size,
            learning_rate: self.learning_rate,
        }
    }
}

/// A multilayer perceptron feedforward neural network, used to train, test and evaluate
/// various machine learning tasks using it.
pub struct NeuralNetwork {
    input_size: usize,
    hidden_layers: Vec<Layer>,
    output_layer: Layer,
    normalize_inputs: (bool, f64),
    activations: (ActivationFunctions, ActivationFunctions),
    loss: LossFunctions,
    pub batch_size: usize,
    learning_rate: f64,
}

impl NeuralNetwork {
    /// Returns a default [`NeuralNetworkBuilder`].
    pub fn builder() -> NeuralNetworkBuilder {
        NeuralNetworkBuilder::default()
    }

    /// Does a single (forward) propagation pass using the provided `inputs`.
    ///
    /// `inputs` is of type `&[T]` allowing the use of any type that
    /// implements the `Copy + Into<f64>` traits.
    ///
    /// Returns a `Vec<f64>` that is guaranteed to be of [`NeuralNetwork::output_layer`] size.
    pub fn propagate<T: Copy + Into<f64>>(&mut self, inputs: &[T]) -> Vec<f64> {
        assert_eq!(
            inputs.len(),
            self.input_size,
            "expected input of size {}",
            self.input_size
        );

        // Keeping track of some used values
        let (normalize, max) = self.normalize_inputs;
        let (neuron_activation, output_activation) =
            (self.activations.0.function(), self.activations.1.function());

        // Depending on the configuration, it might convert and normalize or
        // just convert inputs of T into f64
        let mut values: Vec<f64> = inputs
            .iter()
            .map(|&x| if normalize { x.into() / max } else { x.into() })
            .collect();

        // Perform a single calculation pass on the (initially) inputs
        // and overwrite the values with the received output for each
        // hidden layer, "propagating" forward
        for layer in &mut self.hidden_layers {
            values = layer.calculate(&values, &neuron_activation);
        }

        // Perform the last calculation pass on the last evaluated
        // hidden layer activations (or inputs if none), and return them
        self.output_layer.calculate(&values, &output_activation)
    }

    pub fn backpropagate(&mut self, result: &[f64], expected: &[f64]) {
        let output_activation_derivative = self.activations.1.derivative();
        let neuron_activation = self.activations.0.function();

        // Start by evaluating the output layer
        let mut values = vec![0.0; self.output_layer.size];

        // for index in 0..self.output_layer.size {
        //     // values[index] = output_activation_derivative(self.output_layer.values[index])
        //     //     * loss_derivative(&result[index], &expected[index]);
        // }

        // Propagate backwards ("backpropagate")
        // for index in (0..self.hidden_layers.len()).rev() {
        //     for index in 0.. {}
        // }
    }
}
