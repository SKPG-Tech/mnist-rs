mod dataset;
mod printer;
use printer::*;
mod neuralnet;
use neuralnet::*;

fn main() {
    let training_data = dataset::read_file(
        "dataset/train-images.idx3-ubyte",
        "dataset/train-labels.idx1-ubyte",
    )
    .expect("failed to read dataset");

    print_entry(&training_data[0]);

    let mut neuralnet = NeuralNetwork::builder()
        .with_input_size(28 * 28)
        .with_hidden_layer_of_size(32)
        .with_hidden_layer_of_size(16)
        .with_output_size(10)
        .with_activations(ActivationFunctions::Sigmoid, ActivationFunctions::Sigmoid)
        .with_batch_size(300)
        .with_learning_rate(0.25)
        .normalize_inputs(255)
        .build();

    for chunk in training_data.chunks_exact(neuralnet.batch_size) {
        for entry in chunk {
            let result = neuralnet.propagate(entry.image.as_flattened());
            let mut expected = [0.0f64; 10];
            expected[entry.label as usize] = 1.0;
            neuralnet.backpropagate(&result, &expected);
        }
    }
}
