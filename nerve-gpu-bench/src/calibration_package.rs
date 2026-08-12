use std::collections::BTreeMap;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};

use nerve_runtime::{
    HardwareProcessProfile, ResourceResidencyPolicy, RuntimeExecutionEnvelope,
    RuntimeInclusiveRange, VulkanResidentModelPackageManifest, VulkanResidentRuntimeModel,
    vulkan_runtime_model_with_component_placement,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CalibrationRuntimeConfig {
    pub context_size: Option<usize>,
    pub speculative_draft_tokens: Option<usize>,
    pub residency_policy: ResourceResidencyPolicy,
}

impl Default for CalibrationRuntimeConfig {
    fn default() -> Self {
        Self {
            context_size: None,
            speculative_draft_tokens: None,
            residency_policy: ResourceResidencyPolicy::Eager,
        }
    }
}

pub struct CalibrationPackage {
    source_path: PathBuf,
    manifest_dir: PathBuf,
    runtime_model: VulkanResidentRuntimeModel,
}

impl CalibrationPackage {
    pub fn load(source_path: &Path) -> Result<Self, io::Error> {
        let manifest =
            VulkanResidentModelPackageManifest::from_json_file(source_path).map_err(|error| {
                io::Error::new(
                    error.kind(),
                    format!(
                        "failed to load compiled package {}: {error}",
                        source_path.display()
                    ),
                )
            })?;
        let runtime_model = manifest
            .mount_runtime_graph_controls(None, &BTreeMap::new(), &[], None)
            .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error.to_string()))?;
        Ok(Self {
            source_path: source_path.to_path_buf(),
            manifest_dir: source_path
                .parent()
                .unwrap_or_else(|| Path::new("."))
                .to_path_buf(),
            runtime_model,
        })
    }

    pub fn source_path(&self) -> &Path {
        &self.source_path
    }

    pub fn manifest_dir(&self) -> &Path {
        &self.manifest_dir
    }

    pub fn runtime_model_for_owner(
        &self,
        owner_physical_device_id: &str,
        hardware_profile: &HardwareProcessProfile,
        config: CalibrationRuntimeConfig,
    ) -> Result<VulkanResidentRuntimeModel, io::Error> {
        if owner_physical_device_id.is_empty()
            || hardware_profile.hardware_identity.stable_device_id != owner_physical_device_id
        {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "calibration representation selection requires the exact owner hardware profile",
            ));
        }
        let placement = self
            .runtime_model
            .runtime_graph
            .instances
            .iter()
            .map(|instance| {
                (
                    instance.instance_id.clone(),
                    owner_physical_device_id.to_string(),
                )
            })
            .collect::<BTreeMap<_, _>>();
        let placed = vulkan_runtime_model_with_component_placement(
            &self.runtime_model,
            owner_physical_device_id,
            &placement,
        )
        .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error.to_string()))?;
        let execution = self.execution_envelope(config)?;
        placed
            .select_and_apply_runtime_implementations(
                &self.manifest_dir,
                &BTreeMap::from([(
                    owner_physical_device_id.to_string(),
                    hardware_profile.clone(),
                )]),
                execution,
            )
            .map(|(runtime_model, _)| runtime_model)
            .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error.to_string()))
    }

    pub fn execution_envelope(
        &self,
        config: CalibrationRuntimeConfig,
    ) -> Result<RuntimeExecutionEnvelope, io::Error> {
        let context_size = config
            .context_size
            .unwrap_or(self.runtime_model.package.max_context_activations);
        if context_size == 0 || context_size > self.runtime_model.package.max_context_activations {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!(
                    "calibration context size {context_size} is outside package range 1..={}",
                    self.runtime_model.package.max_context_activations,
                ),
            ));
        }
        let speculative_draft_tokens = match config.speculative_draft_tokens {
            Some(tokens) => tokens,
            None => self
                .runtime_model
                .package
                .recommended_speculative_draft_tokens()
                .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))?
                .unwrap_or(0),
        };
        Ok(RuntimeExecutionEnvelope {
            phases: vec!["decode".to_string(), "prefill".to_string()],
            activation_batch: RuntimeInclusiveRange {
                minimum: 1,
                maximum: context_size,
            },
            context_activations: RuntimeInclusiveRange {
                minimum: 0,
                maximum: context_size,
            },
            state_activations: RuntimeInclusiveRange {
                minimum: 0,
                maximum: context_size,
            },
            speculative_draft_tokens,
            residency_policy: config.residency_policy.as_runtime_name().replace('-', "_"),
        })
    }

    pub fn reject_output_collision(&self, output: &Path) -> Result<(), io::Error> {
        reject_output_collision(&self.source_path, output)
    }
}

fn reject_output_collision(source: &Path, output: &Path) -> Result<(), io::Error> {
    if output.exists() && fs::canonicalize(source)? == fs::canonicalize(output)? {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "calibration output must not replace the compiled package manifest",
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny_package_manifest() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../runtime-rs/test-fixtures/tiny_model/vulkan_resident_package.json")
    }

    fn temporary_test_directory(name: &str) -> PathBuf {
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!(
            "nerve-gpu-bench-{name}-{}-{nonce}",
            std::process::id()
        ))
    }

    #[test]
    fn package_catalog_cannot_overwrite_its_source_manifest() {
        let directory = temporary_test_directory("source-collision");
        fs::create_dir_all(&directory).unwrap();
        let package = directory.join("package.json");
        fs::write(&package, b"source").unwrap();
        let alias = directory.join(".").join("package.json");

        let error = reject_output_collision(&package, &alias).unwrap_err();
        assert_eq!(error.kind(), io::ErrorKind::InvalidInput);
        assert!(error.to_string().contains("must not replace"));
        fs::remove_dir_all(directory).unwrap();
    }

    #[test]
    fn calibration_execution_envelope_is_exactly_runtime_selectable() {
        let package = CalibrationPackage::load(&tiny_package_manifest()).unwrap();
        let explicit = package
            .execution_envelope(CalibrationRuntimeConfig {
                context_size: Some(8),
                speculative_draft_tokens: Some(0),
                residency_policy: ResourceResidencyPolicy::DemandPaged,
            })
            .unwrap();

        assert_eq!(explicit.phases, ["decode", "prefill"]);
        assert_eq!(explicit.activation_batch.minimum, 1);
        assert_eq!(explicit.activation_batch.maximum, 8);
        assert_eq!(explicit.context_activations.maximum, 8);
        assert_eq!(explicit.state_activations.maximum, 8);
        assert_eq!(explicit.speculative_draft_tokens, 0);
        assert_eq!(explicit.residency_policy, "demand_paged");
    }

    #[test]
    fn calibration_rejects_context_geometry_outside_the_package_contract() {
        let package = CalibrationPackage::load(&tiny_package_manifest()).unwrap();
        for context_size in [0, package.runtime_model.package.max_context_activations + 1] {
            let error = package
                .execution_envelope(CalibrationRuntimeConfig {
                    context_size: Some(context_size),
                    ..CalibrationRuntimeConfig::default()
                })
                .unwrap_err();
            assert_eq!(error.kind(), io::ErrorKind::InvalidInput);
            assert!(error.to_string().contains("outside package range"));
        }
    }

    #[test]
    fn calibration_places_before_selecting_the_runtime_representation() {
        let package = CalibrationPackage::load(&tiny_package_manifest()).unwrap();
        let profile = nerve_runtime::discover_cpu_hardware_profile().unwrap();
        let owner = profile.hardware_identity.stable_device_id.clone();
        let selected = package
            .runtime_model_for_owner(
                &owner,
                &profile,
                CalibrationRuntimeConfig {
                    context_size: Some(8),
                    speculative_draft_tokens: Some(0),
                    residency_policy: ResourceResidencyPolicy::Eager,
                },
            )
            .unwrap();

        assert!(
            selected
                .runtime_graph
                .instances
                .iter()
                .all(|instance| instance.device_id == owner)
        );
        let report = selected
            .implementation_selection
            .expect("calibration must preserve the exact selection report");
        assert_eq!(report.execution.context_activations.maximum, 8);
        assert_eq!(report.execution.residency_policy, "eager");
        assert_eq!(
            report
                .selected
                .iter()
                .map(|selection| selection.instance_ids.len())
                .sum::<usize>()
                + report.exact_instance_ids.len(),
            selected
                .runtime_graph
                .instances
                .iter()
                .filter(|instance| {
                    selected
                        .package
                        .circuit_graph
                        .components
                        .iter()
                        .find(|component| component.component_id == instance.source_component_id)
                        .is_some_and(|component| {
                            component.runtime_role.is_runtime_implementation_target()
                        })
                })
                .count()
        );
    }
}
