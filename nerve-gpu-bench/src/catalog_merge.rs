use std::error::Error;
use std::fs;
use std::path::{Path, PathBuf};

use nerve_runtime::VulkanPlacementCalibrationCatalog;

use crate::output::write_atomic;

pub fn merge_catalog_files(inputs: &[PathBuf], output: &Path) -> Result<(), Box<dyn Error>> {
    let mut merged = VulkanPlacementCalibrationCatalog::default();
    for input in inputs {
        let payload = fs::read(input)?;
        let catalog = VulkanPlacementCalibrationCatalog::from_json_slice(&payload)?;
        merged.merge(&catalog)?;
    }
    let payload = merged.to_json_bytes()?;
    write_atomic(output, &payload)?;
    println!(
        "merged exact placement catalogs: inputs={}, references={}, observations={}, output={}",
        inputs.len(),
        merged.reference_count(),
        merged.observation_count(),
        output.display(),
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn temporary_test_directory(name: &str) -> PathBuf {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!(
            "nerve-gpu-bench-{name}-{}-{nonce}",
            std::process::id()
        ))
    }

    #[test]
    fn file_merge_publishes_a_valid_exact_catalog() {
        let directory = temporary_test_directory("catalog-merge");
        fs::create_dir_all(&directory).unwrap();
        let first = directory.join("first.json");
        let second = directory.join("second.json");
        let output = directory.join("merged.json");
        let empty = VulkanPlacementCalibrationCatalog::default()
            .to_json_bytes()
            .unwrap();
        fs::write(&first, &empty).unwrap();
        fs::write(&second, &empty).unwrap();

        merge_catalog_files(&[first, second], &output).unwrap();

        let merged =
            VulkanPlacementCalibrationCatalog::from_json_slice(&fs::read(&output).unwrap())
                .unwrap();
        assert_eq!(merged.reference_count(), 0);
        assert_eq!(merged.observation_count(), 0);
        fs::remove_dir_all(directory).unwrap();
    }

    #[test]
    fn failed_file_merge_preserves_the_previously_published_output() {
        let directory = temporary_test_directory("catalog-merge-failure");
        fs::create_dir_all(&directory).unwrap();
        let valid = directory.join("valid.json");
        let invalid = directory.join("invalid.json");
        let output = directory.join("merged.json");
        fs::write(
            &valid,
            VulkanPlacementCalibrationCatalog::default()
                .to_json_bytes()
                .unwrap(),
        )
        .unwrap();
        fs::write(&invalid, b"not a catalog").unwrap();
        fs::write(&output, b"previously accepted").unwrap();

        assert!(merge_catalog_files(&[valid, invalid], &output).is_err());
        assert_eq!(fs::read(&output).unwrap(), b"previously accepted");
        fs::remove_dir_all(directory).unwrap();
    }
}
