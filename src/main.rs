mod dataset;
mod printer;
use printer::*;

fn main() {
    let training_data = dataset::read_file(
        "dataset/train-images.idx3-ubyte",
        "dataset/train-labels.idx1-ubyte",
    )
    .expect("failed to read dataset");

    print_entry(&training_data[0]);
}
