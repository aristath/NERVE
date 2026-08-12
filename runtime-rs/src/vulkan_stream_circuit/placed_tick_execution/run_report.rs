#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VulkanMountedPlacedResidentStreamTickRun {
    pub tick_run: VulkanMountedPlacedStreamTickRun,
    pub execution_graph_run: Option<VulkanMountedPlacedResidentExecutionGraphRun>,
}

impl VulkanMountedPlacedResidentStreamTickRun {
    pub fn execution_graph_dispatch_count(&self) -> usize {
        self.execution_graph_run
            .as_ref()
            .map(VulkanMountedPlacedResidentExecutionGraphRun::dispatch_count)
            .unwrap_or(0)
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum VulkanMountedPlacedStreamTickStage {
    ReceiveEdge {
        stage_index: usize,
        edge_index: usize,
        endpoint_id: String,
        buffer_index: usize,
        byte_capacity: usize,
        remote_device_id: String,
        remote_component_id: String,
    },
    Dispatch {
        stage_index: usize,
        dispatch: VulkanMountedPlacedStreamTickDispatch,
    },
    PublishEdge {
        stage_index: usize,
        edge_index: usize,
        endpoint_id: String,
        buffer_index: usize,
        byte_capacity: usize,
        remote_device_id: String,
        remote_component_id: String,
    },
}

impl VulkanMountedPlacedStreamTickStage {
    pub fn stage_index(&self) -> usize {
        match self {
            Self::ReceiveEdge { stage_index, .. }
            | Self::Dispatch { stage_index, .. }
            | Self::PublishEdge { stage_index, .. } => *stage_index,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VulkanMountedPlacedStreamTickDispatch {
    pub dispatch_index: usize,
    pub kernel_id: String,
    pub component_id: String,
    pub node_id: String,
    pub op: String,
    pub descriptor_count: usize,
    pub resident_descriptor_count: usize,
    pub reads: Vec<VulkanMountedPlacedStreamTickIo>,
    pub writes: Vec<VulkanMountedPlacedStreamTickIo>,
}

impl VulkanMountedPlacedStreamTickDispatch {
    fn from_bound_dispatch(dispatch: &VulkanMountedPlacedBoundDispatch) -> Self {
        let mut resident_descriptor_count = 0usize;
        let mut reads = Vec::new();
        let mut writes = Vec::new();

        for descriptor in &dispatch.descriptors {
            match &descriptor.target {
                VulkanMountedPlacedBoundDescriptorTarget::Resident { target } => {
                    resident_descriptor_count += 1;
                    if let VulkanBoundDescriptorTarget::ActivationSlot {
                        component_id,
                        signal_id,
                        slot,
                        ..
                    } = target
                    {
                        let io = VulkanMountedPlacedStreamTickIo::ActivationSlot {
                            component_id: component_id.clone(),
                            signal_id: signal_id.clone(),
                            slot: *slot,
                        };
                        match &descriptor.usage {
                            VulkanKernelDescriptorUsage::InputSignal => reads.push(io),
                            VulkanKernelDescriptorUsage::OutputSignal => writes.push(io),
                            _ => {}
                        }
                    }
                }
                VulkanMountedPlacedBoundDescriptorTarget::ModelInput { signal_id } => {
                    reads.push(VulkanMountedPlacedStreamTickIo::ModelSignal {
                        signal_id: signal_id.clone(),
                    });
                }
                VulkanMountedPlacedBoundDescriptorTarget::ModelOutput { signal_id } => {
                    writes.push(VulkanMountedPlacedStreamTickIo::ModelSignal {
                        signal_id: signal_id.clone(),
                    });
                }
                VulkanMountedPlacedBoundDescriptorTarget::LocalEdgeInputBuffer { edge } => {
                    reads.push(VulkanMountedPlacedStreamTickIo::LocalEdgeBuffer {
                        edge_index: edge.edge.edge_index,
                        buffer_index: edge.buffer_index,
                        byte_capacity: edge.byte_capacity,
                    });
                }
                VulkanMountedPlacedBoundDescriptorTarget::IncomingEdgeBuffer { endpoint } => {
                    reads.push(VulkanMountedPlacedStreamTickIo::IncomingEdgeBuffer {
                        edge_index: endpoint.endpoint.edge_index,
                        buffer_index: endpoint.buffer_index,
                        byte_capacity: endpoint.byte_capacity,
                    });
                }
                VulkanMountedPlacedBoundDescriptorTarget::ProducedPortBuffer { port } => {
                    writes.extend(port.local_edges.iter().map(|edge| {
                        VulkanMountedPlacedStreamTickIo::LocalEdgeBuffer {
                            edge_index: edge.edge.edge_index,
                            buffer_index: edge.buffer_index,
                            byte_capacity: edge.byte_capacity,
                        }
                    }));
                    writes.extend(port.outgoing_endpoints.iter().map(|endpoint| {
                        VulkanMountedPlacedStreamTickIo::OutgoingEdgeBuffer {
                            edge_index: endpoint.endpoint.edge_index,
                            buffer_index: endpoint.buffer_index,
                            byte_capacity: endpoint.byte_capacity,
                        }
                    }));
                }
            }
        }

        Self {
            dispatch_index: dispatch.dispatch_index,
            kernel_id: dispatch.kernel_id.clone(),
            component_id: dispatch.component_id.clone(),
            node_id: dispatch.node_id.clone(),
            op: dispatch.op.clone(),
            descriptor_count: dispatch.descriptors.len(),
            resident_descriptor_count,
            reads,
            writes,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum VulkanMountedPlacedStreamTickIo {
    ActivationSlot {
        component_id: String,
        signal_id: String,
        slot: usize,
    },
    ModelSignal {
        signal_id: String,
    },
    LocalEdgeBuffer {
        edge_index: usize,
        buffer_index: usize,
        byte_capacity: usize,
    },
    IncomingEdgeBuffer {
        edge_index: usize,
        buffer_index: usize,
        byte_capacity: usize,
    },
    OutgoingEdgeBuffer {
        edge_index: usize,
        buffer_index: usize,
        byte_capacity: usize,
    },
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VulkanMountedPlacedStreamTickRun {
    pub backend_id: String,
    pub device_id: String,
    pub stream_tick: u64,
    pub stages: Vec<VulkanMountedPlacedStreamTickStageRun>,
    pub planned_stage_count: usize,
    pub attempted_stage_count: usize,
    pub completed_stage_count: usize,
    pub pending_stage_count: usize,
    pub status: VulkanMountedPlacedStreamTickRunStatus,
    pub can_execute: bool,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VulkanMountedPlacedStreamTickStageRun {
    pub stage_index: usize,
    pub stage: VulkanMountedPlacedStreamTickStage,
    pub status: VulkanMountedPlacedStreamTickStageStatus,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum VulkanMountedPlacedStreamTickRunStatus {
    Completed,
    Blocked {
        stage_index: usize,
        reason: VulkanMountedPlacedStreamTickBlockReason,
    },
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum VulkanMountedPlacedStreamTickStageStatus {
    Pending,
    Completed,
    Blocked {
        reason: VulkanMountedPlacedStreamTickBlockReason,
    },
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum VulkanMountedPlacedStreamTickBlockReason {
    EdgeReceiveTransportUnavailable,
    KernelDispatchUnavailable,
    EdgePublishTransportUnavailable,
    DistributedDispatchPending { dispatch_index: usize },
}
