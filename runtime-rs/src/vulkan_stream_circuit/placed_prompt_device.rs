impl VulkanResidentInProcessPlacedStreamProcessorDevice {
    pub fn mounted(&self) -> &VulkanMountedPlacedStreamCircuit {
        &self.mounted
    }

    pub fn loaded_manifest(&self) -> &VulkanLoadedKernelArtifactCatalog {
        self.package_slice.loaded_manifest()
    }
}
