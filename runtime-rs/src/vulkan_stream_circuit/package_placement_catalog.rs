pub const VULKAN_PACKAGE_PLACEMENT_CALIBRATION_CATALOG_PATH: &str =
    "optimization/placement-calibration-catalog.json";

pub fn load_vulkan_package_placement_calibration_catalog(
    package_root: impl AsRef<Path>,
) -> Result<Option<VulkanPlacementCalibrationCatalog>, VulkanPlacementCalibrationCatalogError> {
    let path = package_root
        .as_ref()
        .join(VULKAN_PACKAGE_PLACEMENT_CALIBRATION_CATALOG_PATH);
    let payload = match fs::read(&path) {
        Ok(payload) => payload,
        Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(None),
        Err(error) => {
            return Err(VulkanPlacementCalibrationCatalogError(format!(
                "failed to read package placement calibration catalog {path:?}: {error}",
            )));
        }
    };
    VulkanPlacementCalibrationCatalog::from_json_slice(&payload)
        .map(Some)
        .map_err(|error| {
            VulkanPlacementCalibrationCatalogError(format!(
                "package placement calibration catalog {path:?} is invalid: {error}",
            ))
        })
}

#[cfg(test)]
mod package_placement_catalog_tests {
    use super::*;

    fn temporary_root(name: &str) -> PathBuf {
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!(
            "nerve-package-placement-{name}-{}-{nonce}",
            std::process::id(),
        ))
    }

    #[test]
    fn missing_package_placement_catalog_is_an_explicit_empty_evidence_set() {
        let root = temporary_root("missing");
        fs::create_dir_all(&root).unwrap();

        assert_eq!(
            load_vulkan_package_placement_calibration_catalog(&root).unwrap(),
            None
        );
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn package_placement_catalog_rejects_corrupt_evidence() {
        let root = temporary_root("corrupt");
        let path = root.join(VULKAN_PACKAGE_PLACEMENT_CALIBRATION_CATALOG_PATH);
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        fs::write(&path, b"not a placement catalog").unwrap();

        let error = load_vulkan_package_placement_calibration_catalog(&root).unwrap_err();
        assert!(error.to_string().contains("is invalid"));
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn package_placement_catalog_loads_canonical_evidence() {
        let root = temporary_root("valid");
        let path = root.join(VULKAN_PACKAGE_PLACEMENT_CALIBRATION_CATALOG_PATH);
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        fs::write(
            &path,
            VulkanPlacementCalibrationCatalog::default()
                .to_json_bytes()
                .unwrap(),
        )
        .unwrap();

        let catalog = load_vulkan_package_placement_calibration_catalog(&root)
            .unwrap()
            .unwrap();
        assert_eq!(catalog.observation_count(), 0);
        fs::remove_dir_all(root).unwrap();
    }
}
