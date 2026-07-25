#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum VulkanPlacedEdgeDirection {
    Incoming,
    Outgoing,
}

pub struct VulkanPlacedEdgeIoBuffers {
    pub plan: VulkanPlacedEdgeIoPlan,
    pub local_buffers: Vec<VulkanPlacedLocalEdgeBufferAllocation>,
    pub incoming_buffers: Vec<VulkanPlacedEdgeBufferAllocation>,
    pub outgoing_buffers: Vec<VulkanPlacedEdgeBufferAllocation>,
    pub total_byte_capacity: usize,
}

impl VulkanPlacedEdgeIoBuffers {
    pub fn local_buffer(
        &self,
        edge_index: usize,
    ) -> Option<(usize, &VulkanPlacedLocalEdgeBufferAllocation)> {
        self.local_buffers
            .iter()
            .enumerate()
            .find(|(_, buffer)| buffer.edge.edge_index == edge_index)
    }

    pub fn local_edge_buffer(
        &self,
        edge_index: usize,
    ) -> Option<&VulkanPlacedLocalEdgeBufferAllocation> {
        self.local_buffers
            .iter()
            .find(|buffer| buffer.edge.edge_index == edge_index)
    }

    pub fn buffer(
        &self,
        direction: VulkanPlacedEdgeDirection,
        edge_index: usize,
    ) -> Option<(usize, &VulkanPlacedEdgeBufferAllocation)> {
        match direction {
            VulkanPlacedEdgeDirection::Incoming => self
                .incoming_buffers
                .iter()
                .enumerate()
                .find(|(_, buffer)| buffer.endpoint.edge_index == edge_index),
            VulkanPlacedEdgeDirection::Outgoing => self
                .outgoing_buffers
                .iter()
                .enumerate()
                .find(|(_, buffer)| buffer.endpoint.edge_index == edge_index),
        }
    }

    pub fn incoming_buffer(
        &self,
        edge_index: usize,
    ) -> Option<&VulkanPlacedEdgeBufferAllocation> {
        self.incoming_buffers
            .iter()
            .find(|buffer| buffer.endpoint.edge_index == edge_index)
    }

    pub fn outgoing_buffer(
        &self,
        edge_index: usize,
    ) -> Option<&VulkanPlacedEdgeBufferAllocation> {
        self.outgoing_buffers
            .iter()
            .find(|buffer| buffer.endpoint.edge_index == edge_index)
    }
}

pub struct VulkanPlacedLocalEdgeBufferAllocation {
    pub edge: VulkanPlacedLocalEdge,
    pub byte_capacity: usize,
    pub buffer: Arc<VulkanResidentBuffer>,
}

pub struct VulkanPlacedEdgeBufferAllocation {
    pub endpoint: VulkanPlacedEdgeEndpoint,
    pub byte_capacity: usize,
    pub buffer: Arc<VulkanResidentBuffer>,
}

pub struct VulkanPlacedEdgeEndpointBufferOverride {
    pub direction: VulkanPlacedEdgeDirection,
    pub edge_index: usize,
    pub buffer: Arc<VulkanResidentBuffer>,
}

pub struct VulkanPlacedLocalEdgeBufferOverride {
    pub edge_index: usize,
    pub buffer: Arc<VulkanResidentBuffer>,
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct VulkanPlacedEdgePacketKey {
    pub edge_index: usize,
    pub from_device_id: String,
    pub to_device_id: String,
}

impl VulkanPlacedEdgePacketKey {
    pub fn from_outgoing_endpoint(endpoint: &VulkanPlacedEdgeEndpoint) -> Self {
        Self {
            edge_index: endpoint.edge_index,
            from_device_id: endpoint.local_device_id.clone(),
            to_device_id: endpoint.remote_device_id.clone(),
        }
    }

    pub fn from_incoming_endpoint(endpoint: &VulkanPlacedEdgeEndpoint) -> Self {
        Self {
            edge_index: endpoint.edge_index,
            from_device_id: endpoint.remote_device_id.clone(),
            to_device_id: endpoint.local_device_id.clone(),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VulkanPlacedEdgePacket {
    pub key: VulkanPlacedEdgePacketKey,
    pub signal: String,
    pub source_component_id: String,
    pub destination_component_id: String,
    pub byte_count: usize,
    pub bytes: Vec<u8>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VulkanPlacedEdgeTransportReceipt {
    pub key: VulkanPlacedEdgePacketKey,
    pub signal: String,
    pub byte_count: usize,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct VulkanPlacedEdgeTransportReceiveBatch {
    pub received: Vec<VulkanPlacedEdgeTransportReceipt>,
    pub missing_packets: Vec<VulkanPlacedEdgePacketKey>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum VulkanPlacedEdgeTransferRoute {
    SameDeviceAlias,
    DeviceLocalCopy,
    DeviceLocalStaging,
    ExternalDeviceLocal,
    SharedHost,
    HostStaging,
}

impl VulkanPlacedEdgeTransferRoute {
    pub fn supports_queue_overlap(self) -> bool {
        matches!(
            self,
            Self::DeviceLocalStaging | Self::ExternalDeviceLocal | Self::SharedHost
        )
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct VulkanPlacedEdgeTransportEdgeStats {
    pub key: VulkanPlacedEdgePacketKey,
    pub signal: String,
    pub route: VulkanPlacedEdgeTransferRoute,
    pub byte_capacity: usize,
    pub publish_count: usize,
    pub receive_count: usize,
    pub transferred_byte_count: usize,
    pub queue_signal_count: usize,
    pub queue_wait_count: usize,
    pub host_wait_count: usize,
    pub queue_overlap_eligible: bool,
    pub overlap_submission_count: usize,
}

impl VulkanPlacedEdgeTransportEdgeStats {
    fn reset_tick_counts(&mut self) {
        self.publish_count = 0;
        self.receive_count = 0;
        self.transferred_byte_count = 0;
        self.queue_signal_count = 0;
        self.queue_wait_count = 0;
        self.host_wait_count = 0;
        self.overlap_submission_count = 0;
    }

    fn accumulate(&mut self, tick: &Self) {
        debug_assert_eq!(self.key, tick.key);
        debug_assert_eq!(self.route, tick.route);
        debug_assert_eq!(
            self.queue_overlap_eligible,
            tick.queue_overlap_eligible
        );
        self.publish_count = self.publish_count.saturating_add(tick.publish_count);
        self.receive_count = self.receive_count.saturating_add(tick.receive_count);
        self.transferred_byte_count = self
            .transferred_byte_count
            .saturating_add(tick.transferred_byte_count);
        self.queue_signal_count = self
            .queue_signal_count
            .saturating_add(tick.queue_signal_count);
        self.queue_wait_count = self
            .queue_wait_count
            .saturating_add(tick.queue_wait_count);
        self.host_wait_count = self.host_wait_count.saturating_add(tick.host_wait_count);
        self.overlap_submission_count = self
            .overlap_submission_count
            .saturating_add(tick.overlap_submission_count);
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct VulkanPlacedEdgeTransportStats {
    pub pending_packet_count: usize,
    pub pending_byte_count: usize,
    pub pending_direct_edge_count: usize,
    pub pending_direct_byte_count: usize,
    pub published_packet_count: usize,
    pub published_byte_count: usize,
    pub received_packet_count: usize,
    pub received_byte_count: usize,
    pub direct_copy_count: usize,
    pub direct_copy_byte_count: usize,
    pub direct_receive_count: usize,
    pub direct_receive_byte_count: usize,
    pub edges: Vec<VulkanPlacedEdgeTransportEdgeStats>,
}

impl VulkanPlacedEdgeTransportStats {
    fn accumulate(&mut self, tick: &Self) {
        self.pending_packet_count = tick.pending_packet_count;
        self.pending_byte_count = tick.pending_byte_count;
        self.pending_direct_edge_count = tick.pending_direct_edge_count;
        self.pending_direct_byte_count = tick.pending_direct_byte_count;
        self.published_packet_count = self
            .published_packet_count
            .saturating_add(tick.published_packet_count);
        self.published_byte_count = self
            .published_byte_count
            .saturating_add(tick.published_byte_count);
        self.received_packet_count = self
            .received_packet_count
            .saturating_add(tick.received_packet_count);
        self.received_byte_count = self
            .received_byte_count
            .saturating_add(tick.received_byte_count);
        self.direct_copy_count = self
            .direct_copy_count
            .saturating_add(tick.direct_copy_count);
        self.direct_copy_byte_count = self
            .direct_copy_byte_count
            .saturating_add(tick.direct_copy_byte_count);
        self.direct_receive_count = self
            .direct_receive_count
            .saturating_add(tick.direct_receive_count);
        self.direct_receive_byte_count = self
            .direct_receive_byte_count
            .saturating_add(tick.direct_receive_byte_count);
        for tick_edge in &tick.edges {
            if let Some(edge) = self.edges.iter_mut().find(|edge| {
                edge.key == tick_edge.key
                    && edge.route == tick_edge.route
                    && edge.signal == tick_edge.signal
            }) {
                edge.accumulate(tick_edge);
            } else {
                self.edges.push(tick_edge.clone());
            }
        }
        self.edges.sort_by(|left, right| {
            left.key
                .cmp(&right.key)
                .then_with(|| left.route.cmp(&right.route))
                .then_with(|| left.signal.cmp(&right.signal))
        });
    }
}
