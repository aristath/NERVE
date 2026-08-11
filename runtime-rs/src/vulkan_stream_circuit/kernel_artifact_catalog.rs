#[derive(Clone, Debug, PartialEq)]
pub struct VulkanLoadedKernelArtifactCatalog {
    pub reusable_artifacts: Vec<VulkanLoadedReusableKernelArtifact>,
    pub physical_artifacts: Vec<VulkanLoadedPhysicalKernelArtifact>,
    pub reusable_word_count: usize,
    pub physical_word_count: usize,
}

impl VulkanLoadedKernelArtifactCatalog {
    pub fn from_manifest(
        manifest: &VulkanReusableKernelArtifactManifest,
        artifact_root: impl AsRef<Path>,
    ) -> io::Result<Self> {
        let artifact_root = artifact_root.as_ref();
        let mut reusable_artifacts = Vec::with_capacity(manifest.artifacts.len());
        let mut reusable_word_count = 0usize;

        for artifact in &manifest.artifacts {
            let resolved_path = resolve_kernel_artifact_path(artifact_root, &artifact.path);
            let words = read_spirv_words(&resolved_path)?;
            reusable_word_count = reusable_word_count
                .checked_add(words.len())
                .ok_or_else(|| {
                    io::Error::new(
                        io::ErrorKind::InvalidData,
                        "loaded reusable kernel word count overflowed",
                    )
                })?;
            reusable_artifacts.push(VulkanLoadedReusableKernelArtifact {
                artifact: artifact.clone(),
                resolved_path,
                words,
            });
        }

        Ok(Self {
            reusable_artifacts,
            physical_artifacts: Vec::new(),
            reusable_word_count,
            physical_word_count: 0,
        })
    }

    pub fn reusable_artifact(
        &self,
        family_id: &str,
    ) -> Option<&VulkanLoadedReusableKernelArtifact> {
        self.reusable_artifacts
            .iter()
            .find(|artifact| artifact.artifact.family_id == family_id)
    }

    pub fn reusable_family_ids(&self) -> Vec<&str> {
        self.reusable_artifacts
            .iter()
            .map(|artifact| artifact.artifact.family_id.as_str())
            .collect()
    }

    pub fn physical_artifact(
        &self,
        artifact_id: &str,
    ) -> Option<&VulkanLoadedPhysicalKernelArtifact> {
        self.physical_artifacts
            .iter()
            .find(|artifact| artifact.artifact.artifact_id == artifact_id)
    }

    pub fn reusable_manifest(&self) -> VulkanReusableKernelArtifactManifest {
        VulkanReusableKernelArtifactManifest::new(
            self.reusable_artifacts
                .iter()
                .map(|loaded| loaded.artifact.clone())
                .collect(),
        )
    }

    pub fn total_word_count(&self) -> usize {
        self.reusable_word_count + self.physical_word_count
    }
}

fn resolve_kernel_artifact_path(artifact_root: &Path, path: &str) -> PathBuf {
    let path = Path::new(path);
    if path.is_absolute() {
        path.to_path_buf()
    } else {
        artifact_root.join(path)
    }
}
