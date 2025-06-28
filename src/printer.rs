use crate::dataset::Entry;

pub fn print_entry(entry: &Entry) {
    println!("Entry label: {}", entry.label);
    println!("Entry image:");

    entry.image.iter().for_each(|y| {
        y.iter().for_each(|x| print!("{:^4}", x));
        println!()
    });
}
