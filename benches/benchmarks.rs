#[path = "../src/dataset.rs"]
mod dataset;
#[path = "../src/neuralnet/mod.rs"]
mod neuralnet;
use neuralnet::*;

use criterion::*;

fn benchmarks(c: &mut Criterion) {
    let training_data = dataset::read_file(
        "dataset/train-images.idx3-ubyte",
        "dataset/train-labels.idx1-ubyte",
    )
    .expect("failed to read dataset");

    let mut neuralnet = NeuralNetwork::builder()
        .with_input_size(28 * 28)
        .with_hidden_layer_of_size(32)
        .with_hidden_layer_of_size(16)
        .with_output_size(10)
        .with_activations(ActivationFunctions::ReLU, ActivationFunctions::ReLU)
        .with_loss(LossFunctions::SquaredDifference)
        .with_batch_size(300)
        .with_learning_rate(0.25)
        .normalize_inputs(255)
        .build();

    let mut results: Vec<Vec<f64>> = Vec::new();
    results.resize(training_data.len(), vec![0.0; 10]);

    c.bench_function("propagation", |b| {
        b.iter(|| {
            let mut i = 0;
            for chunk in training_data.chunks_exact(neuralnet.batch_size) {
                for entry in chunk {
                    results[i].append(&mut neuralnet.propagate(entry.image.as_flattened()));
                    i += 1;
                }
            }
        })
    });

    c.bench_function("backpropagation", |b| {
        b.iter(|| {
            let mut i = 0;
            for chunk in training_data.chunks_exact(neuralnet.batch_size) {
                for entry in chunk {
                    let mut expected = [0.0f64; 10];
                    expected[entry.label as usize] = 1.0;
                    neuralnet.backpropagate(&results[i], &expected);
                    i += 1;
                }
            }
        })
    });
}

criterion_group!(
    name = benches;
    config = Criterion::default().sample_size(10);
    targets = benchmarks);
criterion_main!(benches);
