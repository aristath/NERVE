#[derive(Clone, Debug, PartialEq)]
pub struct VulkanPhysicalKernelArtifactManifest {
    pub artifacts: Vec<VulkanPhysicalKernelArtifact>,
}

impl VulkanPhysicalKernelArtifactManifest {
    pub fn new(artifacts: Vec<VulkanPhysicalKernelArtifact>) -> Self {
        Self { artifacts }
    }

    pub fn artifact(&self, artifact_id: &str) -> Option<&VulkanPhysicalKernelArtifact> {
        self.artifacts
            .iter()
            .find(|artifact| artifact.artifact_id == artifact_id)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct VulkanPhysicalKernelArtifact {
    pub artifact_id: String,
    pub op: String,
    pub path: String,
    pub entry_point: String,
    pub local_size_x: u32,
    pub workgroup_count_x: u32,
    pub descriptor_signature: Vec<VulkanKernelDescriptorSlotSignature>,
    pub push_constants: Vec<VulkanKernelScalarBinding>,
    pub stream_control_binding: Option<u32>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct VulkanLoadedPhysicalKernelArtifact {
    pub artifact: VulkanPhysicalKernelArtifact,
    pub resolved_path: PathBuf,
    pub words: Vec<u32>,
}

pub(crate) fn physical_execution_artifact_id(
    contract_id: &str,
    artifact_index: usize,
) -> String {
    format!("physical:{contract_id}:artifact:{artifact_index}")
}
