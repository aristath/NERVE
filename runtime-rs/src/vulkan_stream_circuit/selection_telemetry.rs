#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct VulkanSelectionTelemetrySnapshot {
    pub domains: Vec<VulkanSelectionTelemetryDomainSnapshot>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VulkanSelectionTelemetryDomainSnapshot {
    pub execution_scope: String,
    pub component_id: String,
    pub node_id: String,
    pub domain_id: String,
    pub resource_count: usize,
    pub selection_counts: Vec<u64>,
}

impl VulkanSelectionTelemetrySnapshot {
    pub fn delta_since(
        &self,
        previous: &Self,
    ) -> Result<Self, VulkanResidentInProcessPlacedRuntimeError> {
        if self.domains.len() != previous.domains.len() {
            return Err(selection_telemetry_error(format!(
                "selection telemetry domain count changed from {} to {}",
                previous.domains.len(),
                self.domains.len()
            )));
        }
        let domains = self
            .domains
            .iter()
            .zip(&previous.domains)
            .map(|(current, previous)| {
                if current.identity() != previous.identity()
                    || current.selection_counts.len() != previous.selection_counts.len()
                {
                    return Err(selection_telemetry_error(format!(
                        "selection telemetry domain changed from {:?} to {:?}",
                        previous.identity(),
                        current.identity()
                    )));
                }
                let selection_counts = current
                    .selection_counts
                    .iter()
                    .zip(&previous.selection_counts)
                    .enumerate()
                    .map(|(resource_id, (current_count, previous_count))| {
                        current_count.checked_sub(*previous_count).ok_or_else(|| {
                            selection_telemetry_error(format!(
                                "selection telemetry counter {} regressed from {} to {} in {}.{}.{}",
                                resource_id,
                                previous_count,
                                current_count,
                                current.component_id,
                                current.node_id,
                                current.domain_id,
                            ))
                        })
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                Ok(VulkanSelectionTelemetryDomainSnapshot {
                    execution_scope: current.execution_scope.clone(),
                    component_id: current.component_id.clone(),
                    node_id: current.node_id.clone(),
                    domain_id: current.domain_id.clone(),
                    resource_count: current.resource_count,
                    selection_counts,
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self { domains })
    }

    pub fn report(&self) -> RuntimeSelectionCoverageReport {
        let domains = self
            .domains
            .iter()
            .map(VulkanSelectionTelemetryDomainSnapshot::report)
            .collect::<Vec<_>>();
        RuntimeSelectionCoverageReport {
            domain_count: domains.len(),
            addressable_resource_count: domains
                .iter()
                .map(|domain| domain.resource_count)
                .sum(),
            selected_resource_count: domains
                .iter()
                .map(|domain| domain.selected_resource_count)
                .sum(),
            selection_count: domains
                .iter()
                .map(|domain| domain.selection_count)
                .sum(),
            domains,
        }
    }
}

impl VulkanSelectionTelemetryDomainSnapshot {
    fn identity(&self) -> (&str, &str, &str, &str, usize) {
        (
            &self.execution_scope,
            &self.component_id,
            &self.node_id,
            &self.domain_id,
            self.resource_count,
        )
    }

    fn report(&self) -> RuntimeSelectionDomainCoverageReport {
        let selected_resources = self
            .selection_counts
            .iter()
            .copied()
            .enumerate()
            .filter(|(_, count)| *count > 0)
            .map(|(resource_id, selection_count)| RuntimeSelectedResourceCountReport {
                resource_id,
                selection_count,
            })
            .collect::<Vec<_>>();
        RuntimeSelectionDomainCoverageReport {
            execution_scope: self.execution_scope.clone(),
            component_id: self.component_id.clone(),
            node_id: self.node_id.clone(),
            domain_id: self.domain_id.clone(),
            resource_count: self.resource_count,
            selected_resource_count: selected_resources.len(),
            selection_count: selected_resources
                .iter()
                .map(|resource| resource.selection_count)
                .sum(),
            selected_resources,
        }
    }
}

impl VulkanResidentInProcessPlacedStreamProcessor {
    fn selection_telemetry_snapshot(
        &self,
        devices: &BTreeMap<String, Rc<VulkanComputeDevice>>,
    ) -> Result<VulkanSelectionTelemetrySnapshot, VulkanResidentInProcessPlacedRuntimeError> {
        let mut domains = Vec::new();
        for slice in &self.device_slices {
            let device = devices.get(&slice.device_id).ok_or_else(|| {
                VulkanResidentInProcessPlacedRuntimeError::MissingBoundDevice {
                    device_id: slice.device_id.clone(),
                }
            })?;
            append_mounted_selection_telemetry(
                "target",
                device,
                &slice.mounted,
                &mut domains,
            )?;
        }
        for decoder in &self.speculative_decoders {
            let device = devices.get(&decoder.device_id).ok_or_else(|| {
                VulkanResidentInProcessPlacedRuntimeError::MissingBoundDevice {
                    device_id: decoder.device_id.clone(),
                }
            })?;
            append_mounted_selection_telemetry(
                &format!("draft:{}", decoder.id),
                device,
                &decoder.mounted,
                &mut domains,
            )?;
        }
        domains.sort_by(|left, right| left.identity().cmp(&right.identity()));
        Ok(VulkanSelectionTelemetrySnapshot { domains })
    }
}

fn append_mounted_selection_telemetry(
    execution_scope: &str,
    device: &VulkanComputeDevice,
    mounted: &VulkanMountedPlacedStreamCircuit,
    domains: &mut Vec<VulkanSelectionTelemetryDomainSnapshot>,
) -> Result<(), VulkanResidentInProcessPlacedRuntimeError> {
    if mounted.buffers.selection_telemetry_buffers.is_empty() {
        return Ok(());
    }
    let ranges = mounted
        .buffers
        .selection_telemetry_buffers
        .iter()
        .map(|telemetry| {
            VulkanResidentBufferReadRange::new(&telemetry.buffer, 0, telemetry.byte_capacity)
        })
        .collect::<Result<Vec<_>, _>>()
        .map_err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop)?;
    let readback = device
        .read_resident_buffer_ranges(&ranges)
        .map_err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop)?;
    for (index, telemetry) in mounted
        .buffers
        .selection_telemetry_buffers
        .iter()
        .enumerate()
    {
        let bytes = readback
            .range_bytes(index)
            .map_err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop)?;
        let selection_counts = bytes
            .chunks_exact(size_of::<u32>())
            .map(|bytes| {
                u64::from(u32::from_le_bytes(
                    bytes
                        .try_into()
                        .expect("u32-sized telemetry chunks are exact"),
                ))
            })
            .collect::<Vec<_>>();
        if selection_counts.len() != telemetry.resource_count {
            return Err(selection_telemetry_error(format!(
                "{}.{}.{} telemetry contains {} counters, expected {}",
                telemetry.component_id,
                telemetry.node_id,
                telemetry.domain_id,
                selection_counts.len(),
                telemetry.resource_count
            )));
        }
        domains.push(VulkanSelectionTelemetryDomainSnapshot {
            execution_scope: execution_scope.to_string(),
            component_id: telemetry.component_id.clone(),
            node_id: telemetry.node_id.clone(),
            domain_id: telemetry.domain_id.clone(),
            resource_count: telemetry.resource_count,
            selection_counts,
        });
    }
    Ok(())
}

impl VulkanResidentInProcessPlacedPromptStream {
    pub fn selection_telemetry_snapshot(
        &self,
    ) -> Result<VulkanSelectionTelemetrySnapshot, VulkanResidentInProcessPlacedRuntimeError> {
        if !self.is_idle() || self.pending_scheduler_activation.is_some() {
            return Err(placed_scheduler_divergence(
                "cannot snapshot selection telemetry while placed prompt work is pending",
            ));
        }
        self.processor.selection_telemetry_snapshot(&self.devices)
    }
}

fn selection_telemetry_error(
    message: impl Into<String>,
) -> VulkanResidentInProcessPlacedRuntimeError {
    VulkanResidentInProcessPlacedRuntimeError::BackendLoop(VulkanError(message.into()))
}

#[cfg(test)]
mod selection_telemetry_tests {
    use super::*;

    fn snapshot(counts: &[u64]) -> VulkanSelectionTelemetrySnapshot {
        VulkanSelectionTelemetrySnapshot {
            domains: vec![VulkanSelectionTelemetryDomainSnapshot {
                execution_scope: "target".to_string(),
                component_id: "component_0".to_string(),
                node_id: "selector".to_string(),
                domain_id: "resources".to_string(),
                resource_count: counts.len(),
                selection_counts: counts.to_vec(),
            }],
        }
    }

    #[test]
    fn reports_exact_selected_resource_counts() {
        let report = snapshot(&[0, 3, 0, 7]).report();

        assert_eq!(report.domain_count, 1);
        assert_eq!(report.addressable_resource_count, 4);
        assert_eq!(report.selected_resource_count, 2);
        assert_eq!(report.selection_count, 10);
        assert_eq!(
            report.domains[0].selected_resources,
            vec![
                RuntimeSelectedResourceCountReport {
                    resource_id: 1,
                    selection_count: 3,
                },
                RuntimeSelectedResourceCountReport {
                    resource_id: 3,
                    selection_count: 7,
                },
            ],
        );
    }

    #[test]
    fn turn_delta_rejects_counter_regression() {
        let previous = snapshot(&[0, 3, 1]);
        let current = snapshot(&[2, 5, 1]);
        assert_eq!(
            current.delta_since(&previous).unwrap(),
            snapshot(&[2, 2, 0]),
        );

        assert!(snapshot(&[0, 2, 1]).delta_since(&previous).is_err());
    }
}
