use byteorder::{BigEndian, ReadBytesExt};
use std::fs::File;
use std::io::{self, Read};

#[derive(Debug)]
pub struct Entry {
    pub image: [[u8; 28]; 28],
    pub label: u8,
}

pub fn read_file(data_filename: &str, label_filename: &str) -> io::Result<Vec<Entry>> {
    let mut data_file = File::open(data_filename)?;
    let mut label_file = File::open(label_filename)?;

    assert!(
        data_file.read_i32::<BigEndian>()? == 2051,
        "expected the magic number of label header info to be 2051"
    );
    assert!(
        label_file.read_i32::<BigEndian>()? == 2049,
        "expected the magic number of label header info to be 2049"
    );

    let size = data_file.read_i32::<BigEndian>()?;
    label_file.read_i32::<BigEndian>()?;

    // Ignore the row and column counts (28 and 28)
    data_file.read_i32::<BigEndian>()?;
    data_file.read_i32::<BigEndian>()?;

    let mut entries: Vec<Entry> = Vec::new();
    entries.reserve(size as usize);

    for _ in 0..size {
        let mut entry = Entry {
            image: [[0; 28]; 28],
            label: label_file.read_u8()?,
        };
        data_file.read_exact(entry.image.as_flattened_mut())?;
        entries.push(entry);
    }

    Ok(entries)
}
