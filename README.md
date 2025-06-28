# mnist-rs

A Rust implementation of a [multilayer perceptron](https://en.wikipedia.org/wiki/Multilayer_perceptron) feedforward neural network, which is trained and tested on the [MNIST](https://en.wikipedia.org/wiki/MNIST_database) database of handwritten digits.

## Dataset

This specific neural network implementation is meant for the [MNIST](https://en.wikipedia.org/wiki/MNIST_database) database. It should be possible to use any other drop-in replacement dataset ([Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist), [QMNIST](https://github.com/facebookresearch/qmnist), etc.) that has the same specific data format.

Later I might add functionality to use any other dataset, such as [CIFAR-10 & CIFAR-100](https://www.cs.toronto.edu/~kriz/cifar.html), and so on.

## Usage

### Prerequisites

1. Download your desired MNIST-like dataset.
2. Place the downloaded database in a folder at the root of the repository.
3. Make sure to change the paths to the dataset files in `main.rs` like so:

```rs
let training_data = dataset::read_file(
    "[your folder]/[images file]",
    "[your folder]/[labels file]",
)
.expect("failed to read dataset");
```

Once you have the steps covered, running should be as simple as:

```bash
cargo run
```

### Note

The code is not meant to be used as a package, it includes a basic `main.rs` file in the `src` as a means to directly test the local dataset, or your own drop-in replacement.
