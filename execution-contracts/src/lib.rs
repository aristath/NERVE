use std::collections::{BTreeMap, BTreeSet};
use std::fmt;

use serde::{Deserialize, Serialize};

pub const PHYSICAL_EXECUTION_CONTRACT_SCHEMA: &str = "nerve.physical_execution_contract.v4";

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ContractError(pub String);

impl fmt::Display for ContractError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl std::error::Error for ContractError {}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ExecutionPhase {
    Decode,
    Prefill,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ExecutionShape {
    SingleLane,
    MultiLane,
    SingleAndMultiLane,
}

impl ExecutionShape {
    pub fn supports(self, requested: Self) -> bool {
        self == requested || self == Self::SingleAndMultiLane
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ExecutionStrategy {
    SingleDevice,
    TensorParallel,
    ExpertParallel,
    TensorParallelExpert,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ExecutionForm {
    Local,
    ReplicatedInputPartitionedOutput,
    PartitionedInputPartialOutput,
    WholeExpertOwnership,
}

impl ExecutionStrategy {
    pub fn is_distributed(self) -> bool {
        self != Self::SingleDevice
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ParameterPartitionKind {
    Contiguous,
    BlockCyclic,
    ExpertRange,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WorkgroupXMapping {
    Proportional,
    Repeated,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PartitionOrigin {
    LocalZero,
    PushConstantU32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum InputDistribution {
    Replicated,
    Sharded,
    Routed,
    Local,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OutputCollection {
    Local,
    Concatenated,
    Reduced,
    Routed,
    Retained,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ReductionOperation {
    SumF32,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum ReductionFinalization {
    StoreF32,
    StoreF32ToBf16,
    AddBf16ResidualToBf16 { residual_binding: u32 },
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ReductionContract {
    pub operation: ReductionOperation,
    pub dimension_name: String,
    pub finalization: ReductionFinalization,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ResourceKind {
    PersistentParameter,
    Transient,
    KvState,
    RecurrentState,
    LazyResource,
    Control,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ResidencyRequirement {
    Permanent,
    Transaction,
    Demand,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ResourceAccess {
    Read,
    Write,
    ReadWrite,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EquivalenceKind {
    BitExact,
    AbsoluteRelativeTolerance,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ArtifactIdentity {
    pub path: String,
    pub sha256: String,
    pub entry_point: String,
    #[serde(default)]
    pub role: ArtifactRole,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ArtifactRole {
    Preparation,
    #[default]
    Primary,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PhysicalFormats {
    pub storage: String,
    pub compute: String,
    pub accumulation: String,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExecutionGeometry {
    pub shape_class: String,
    pub dimensions: BTreeMap<String, u64>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub dynamic_dimensions: Vec<String>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ParameterPartition {
    pub binding: u32,
    pub resource: String,
    pub dimension: u32,
    pub kind: ParameterPartitionKind,
    pub alignment_elements: u64,
    pub logical_elements_per_index: u64,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SelectedResourcePartition {
    pub selection_signal: String,
    pub address_table_binding: u32,
    pub parameter_slots_binding: u32,
    pub kind: ParameterPartitionKind,
    pub resource_count: u64,
    pub parameters_per_resource: u64,
    pub alignment_elements: u64,
    pub parameter_partitions: Vec<SelectedResourceParameterPartition>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SelectedResourceParameterPartition {
    pub parameter_slot: u64,
    pub dimension: u32,
    pub kind: ParameterPartitionKind,
    pub alignment_elements: u64,
    pub logical_elements_per_index: u64,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PartitionExtent {
    pub dimension_name: String,
    pub elements: u64,
    pub alignment_elements: u64,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PartitionLaunch {
    pub workgroup_x: WorkgroupXMapping,
    pub origin: PartitionOrigin,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub origin_push_constant: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub count_push_constant: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct InputContract {
    pub binding: u32,
    pub distribution: InputDistribution,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dimension: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub alignment_elements: Option<u64>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OutputContract {
    pub binding: u32,
    pub collection: OutputCollection,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dimension: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub alignment_elements: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reduction: Option<ReductionContract>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LocalIntermediateContract {
    pub signal: String,
    pub producer_binding: u32,
    pub consumer_binding: u32,
    pub format: String,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ResourceRequirement {
    pub resource: String,
    pub kind: ResourceKind,
    pub residency: ResidencyRequirement,
    pub access: ResourceAccess,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub binding: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub atomic_group: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EquivalenceRequirement {
    pub output: EquivalenceKind,
    pub state: EquivalenceKind,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub absolute_tolerance: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub relative_tolerance: Option<f64>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PhysicalExecutionContract {
    pub schema: String,
    pub contract_id: String,
    pub operation_family: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub region_family: Option<String>,
    pub member_node_ids: Vec<String>,
    pub artifacts: Vec<ArtifactIdentity>,
    pub implementation_digest: String,
    pub phases: Vec<ExecutionPhase>,
    pub execution_shape: ExecutionShape,
    pub formats: PhysicalFormats,
    pub geometry: ExecutionGeometry,
    pub strategy: ExecutionStrategy,
    pub execution_form: ExecutionForm,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub partition_extent: Option<PartitionExtent>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub partition_launch: Option<PartitionLaunch>,
    pub parameter_partitions: Vec<ParameterPartition>,
    pub selected_resource_partitions: Vec<SelectedResourcePartition>,
    pub inputs: Vec<InputContract>,
    pub outputs: Vec<OutputContract>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub local_intermediates: Vec<LocalIntermediateContract>,
    pub resources: Vec<ResourceRequirement>,
    pub equivalence: EquivalenceRequirement,
}

impl PhysicalExecutionContract {
    pub fn validate(&self) -> Result<(), ContractError> {
        if self.schema != PHYSICAL_EXECUTION_CONTRACT_SCHEMA {
            return invalid(format!(
                "unsupported physical execution contract schema {:?}",
                self.schema
            ));
        }
        require_digest(&self.contract_id, "contract_id")?;
        require_digest(&self.implementation_digest, "implementation_digest")?;
        for (path, value) in [
            ("operation_family", self.operation_family.as_str()),
            ("formats.storage", self.formats.storage.as_str()),
            ("formats.compute", self.formats.compute.as_str()),
            ("formats.accumulation", self.formats.accumulation.as_str()),
            ("geometry.shape_class", self.geometry.shape_class.as_str()),
        ] {
            require_non_empty(value, path)?;
        }
        if self.artifacts.is_empty() {
            return invalid("artifacts must not be empty");
        }
        let mut artifact_paths = BTreeSet::new();
        for artifact in &self.artifacts {
            require_non_empty(&artifact.path, "artifacts.path")?;
            require_digest(&artifact.sha256, "artifacts.sha256")?;
            require_non_empty(&artifact.entry_point, "artifacts.entry_point")?;
            if !artifact_paths.insert(artifact.path.as_str()) {
                return invalid("artifact paths must be unique within an implementation");
            }
        }
        if self.strategy.is_distributed()
            && self
                .artifacts
                .iter()
                .filter(|artifact| artifact.role == ArtifactRole::Primary)
                .count()
                != 1
        {
            return invalid("distributed contracts require exactly one primary artifact");
        }
        require_non_empty_unique(&self.member_node_ids, "member_node_ids")?;
        require_unique_strings(
            &self.geometry.dynamic_dimensions,
            "dynamic_dimensions",
            true,
        )?;
        if self.phases.is_empty()
            || self.phases.iter().copied().collect::<BTreeSet<_>>().len() != self.phases.len()
        {
            return invalid("phases must be non-empty and unique");
        }
        if self.geometry.dimensions.is_empty()
            || self
                .geometry
                .dimensions
                .iter()
                .any(|(name, value)| name.trim().is_empty() || *value == 0)
        {
            return invalid("geometry dimensions must have names and positive values");
        }
        if self
            .geometry
            .dynamic_dimensions
            .iter()
            .any(|name| !self.geometry.dimensions.contains_key(name))
        {
            return invalid("dynamic dimensions must name declared geometry dimensions");
        }
        if let Some(extent) = &self.partition_extent {
            require_non_empty(&extent.dimension_name, "partition_extent.dimension_name")?;
            if extent.elements == 0
                || extent.alignment_elements == 0
                || extent.elements % extent.alignment_elements != 0
            {
                return invalid("partition extent must be positive and divisible by its alignment");
            }
            if self.geometry.dimensions.get(&extent.dimension_name) != Some(&extent.elements) {
                return invalid("partition extent must match its declared geometry dimension");
            }
        }
        if let Some(launch) = &self.partition_launch {
            match (
                launch.origin,
                launch.workgroup_x,
                launch.origin_push_constant.as_deref(),
                launch.count_push_constant.as_deref(),
            ) {
                (PartitionOrigin::LocalZero, _, None, None) => {}
                (
                    PartitionOrigin::PushConstantU32,
                    WorkgroupXMapping::Proportional,
                    Some(name),
                    None,
                ) => {
                    require_non_empty(name, "partition_launch.origin_push_constant")?;
                }
                (
                    PartitionOrigin::PushConstantU32,
                    WorkgroupXMapping::Repeated,
                    Some(origin),
                    Some(count),
                ) => {
                    require_non_empty(origin, "partition_launch.origin_push_constant")?;
                    require_non_empty(count, "partition_launch.count_push_constant")?;
                    if origin == count {
                        return invalid(
                            "partition launch origin and count push constants must differ",
                        );
                    }
                }
                _ => {
                    return invalid(
                        "partition launch push constants must exactly match its origin and workgroup mapping",
                    );
                }
            }
        }
        validate_resources(&self.resources)?;
        validate_bindings(self)?;
        validate_strategy(self)?;
        validate_equivalence(&self.equivalence)?;
        Ok(())
    }
}

fn validate_bindings(contract: &PhysicalExecutionContract) -> Result<(), ContractError> {
    let mut parameter_bindings = BTreeSet::new();
    for partition in &contract.parameter_partitions {
        require_non_empty(&partition.resource, "parameter_partitions.resource")?;
        if partition.alignment_elements == 0
            || partition.logical_elements_per_index == 0
            || !parameter_bindings.insert(partition.binding)
        {
            return invalid("parameter partitions require unique bindings and positive alignment");
        }
        let matching_resources = contract
            .resources
            .iter()
            .filter(|resource| resource.resource == partition.resource)
            .collect::<Vec<_>>();
        let [resource] = matching_resources.as_slice() else {
            return invalid(
                "each parameter partition must name exactly one declared parameter resource",
            );
        };
        if !matches!(
            resource.kind,
            ResourceKind::PersistentParameter | ResourceKind::LazyResource
        ) || resource
            .binding
            .is_some_and(|binding| binding != partition.binding)
        {
            return invalid(
                "parameter partition resources must be parameters at the declared binding",
            );
        }
        if let Some(extent) = &contract.partition_extent {
            let Some(logical_alignment) = partition
                .alignment_elements
                .checked_mul(partition.logical_elements_per_index)
            else {
                return invalid("parameter partition logical alignment overflowed");
            };
            if extent.elements % partition.logical_elements_per_index != 0
                || extent.alignment_elements % logical_alignment != 0
            {
                return invalid(
                    "parameter partitions must divide the logical extent and its alignment",
                );
            }
        }
    }
    let mut selected_bindings = BTreeSet::new();
    let mut selected_signals = BTreeSet::new();
    for partition in &contract.selected_resource_partitions {
        require_non_empty(
            &partition.selection_signal,
            "selected_resource_partitions.selection_signal",
        )?;
        if partition.kind != ParameterPartitionKind::ExpertRange {
            return invalid("selected resource partitions currently require expert ranges");
        }
        if partition.address_table_binding == partition.parameter_slots_binding
            || !selected_bindings.insert(partition.address_table_binding)
            || !selected_bindings.insert(partition.parameter_slots_binding)
            || !selected_signals.insert(partition.selection_signal.as_str())
        {
            return invalid("selected resource partitions require unique signals and bindings");
        }
        if partition.resource_count == 0
            || partition.parameters_per_resource == 0
            || partition.alignment_elements == 0
        {
            return invalid("selected resource partition dimensions must be positive");
        }
        let Some(extent) = &contract.partition_extent else {
            return invalid("selected resource partitions require a partition extent");
        };
        if extent.alignment_elements % partition.alignment_elements != 0 {
            return invalid("selected resource partitions must match the logical partition extent");
        }
        let mut parameter_slots = BTreeSet::new();
        for parameter in &partition.parameter_partitions {
            if parameter.parameter_slot >= partition.parameters_per_resource
                || !parameter_slots.insert(parameter.parameter_slot)
                || parameter.kind == ParameterPartitionKind::ExpertRange
                || parameter.alignment_elements == 0
                || parameter.logical_elements_per_index == 0
            {
                return invalid(
                    "selected resource parameter partitions must name unique valid parameter slots",
                );
            }
            let Some(logical_alignment) = parameter
                .alignment_elements
                .checked_mul(parameter.logical_elements_per_index)
            else {
                return invalid("selected resource parameter partition alignment overflowed");
            };
            if extent.elements % parameter.logical_elements_per_index != 0
                || extent.alignment_elements % logical_alignment != 0
            {
                return invalid(
                    "selected resource parameter partitions must divide the logical extent and its alignment",
                );
            }
        }
        if partition
            .parameter_partitions
            .windows(2)
            .any(|pair| pair[0].parameter_slot >= pair[1].parameter_slot)
        {
            return invalid(
                "selected resource parameter partitions must be ordered by parameter slot",
            );
        }
    }
    if !contract.selected_resource_partitions.is_empty()
        && !contract.resources.iter().any(|resource| {
            resource.kind == ResourceKind::LazyResource
                && resource.residency == ResidencyRequirement::Demand
        })
    {
        return invalid("selected resource partitions require demand-resident lazy resources");
    }
    let mut input_bindings = BTreeSet::new();
    for input in &contract.inputs {
        if !input_bindings.insert(input.binding) {
            return invalid("input bindings must be unique");
        }
        let needs_partition = matches!(
            input.distribution,
            InputDistribution::Sharded | InputDistribution::Routed
        );
        if needs_partition != input.dimension.is_some()
            || needs_partition != input.alignment_elements.is_some()
            || input.alignment_elements == Some(0)
        {
            return invalid(
                "sharded and routed inputs require a dimension and positive alignment; other inputs forbid them",
            );
        }
    }
    let mut output_bindings = BTreeSet::new();
    for output in &contract.outputs {
        if !output_bindings.insert(output.binding) {
            return invalid("output bindings must be unique");
        }
        let needs_partition = matches!(
            output.collection,
            OutputCollection::Concatenated | OutputCollection::Routed
        );
        if needs_partition != output.dimension.is_some()
            || needs_partition != output.alignment_elements.is_some()
            || output.alignment_elements == Some(0)
        {
            return invalid(
                "concatenated and routed outputs require a dimension and positive alignment; other outputs forbid them",
            );
        }
        if (output.collection == OutputCollection::Reduced) != output.reduction.is_some() {
            return invalid("only reduced outputs require a reduction operation");
        }
        if let Some(reduction) = &output.reduction {
            require_non_empty(
                &reduction.dimension_name,
                "outputs.reduction.dimension_name",
            )?;
            if !contract
                .geometry
                .dimensions
                .contains_key(&reduction.dimension_name)
            {
                return invalid("reduced output dimension must name a declared geometry dimension");
            }
            if reduction.operation == ReductionOperation::SumF32
                && contract.formats.accumulation != "f32"
            {
                return invalid("sum_f32 reduction requires f32 accumulation");
            }
            if matches!(
                reduction.finalization,
                ReductionFinalization::StoreF32ToBf16
                    | ReductionFinalization::AddBf16ResidualToBf16 { .. }
            ) && !contract.geometry.dimensions[&reduction.dimension_name].is_multiple_of(2)
            {
                return invalid("BF16 finalization requires an even element count");
            }
            if let ReductionFinalization::AddBf16ResidualToBf16 { residual_binding } =
                &reduction.finalization
            {
                let elements = contract.geometry.dimensions[&reduction.dimension_name];
                debug_assert!(elements.is_multiple_of(2));
                let residual = contract
                    .inputs
                    .iter()
                    .find(|input| input.binding == *residual_binding)
                    .ok_or_else(|| {
                        ContractError(
                            "BF16 residual finalization binding must name a contract input"
                                .to_string(),
                        )
                    })?;
                if residual.distribution != InputDistribution::Replicated {
                    return invalid("BF16 residual finalization input must be replicated");
                }
            }
        }
    }
    if contract.inputs.is_empty() || contract.outputs.is_empty() {
        return invalid("execution contracts require at least one input and output");
    }
    Ok(())
}

fn validate_strategy(contract: &PhysicalExecutionContract) -> Result<(), ContractError> {
    if !contract.strategy.is_distributed() {
        if contract.execution_form != ExecutionForm::Local
            || contract.partition_extent.is_some()
            || contract.partition_launch.is_some()
            || !contract.parameter_partitions.is_empty()
            || !contract.selected_resource_partitions.is_empty()
            || contract.inputs.iter().any(|input| {
                !matches!(
                    input.distribution,
                    InputDistribution::Local | InputDistribution::Replicated
                )
            })
            || contract.outputs.iter().any(|output| {
                !matches!(
                    output.collection,
                    OutputCollection::Local | OutputCollection::Retained
                )
            })
            || !contract.local_intermediates.is_empty()
        {
            return invalid("single-device contracts cannot declare distributed flow");
        }
        return Ok(());
    }
    if contract.execution_form == ExecutionForm::Local {
        return invalid("distributed contracts require a distributed execution form");
    }
    let has_reduced_output = contract
        .outputs
        .iter()
        .any(|output| output.collection == OutputCollection::Reduced);
    if (contract.execution_form == ExecutionForm::PartitionedInputPartialOutput)
        != has_reduced_output
    {
        return invalid(
            "partitioned-input partial-output execution requires a reduced output and reduced outputs require that execution form",
        );
    }
    if contract.execution_form == ExecutionForm::PartitionedInputPartialOutput
        && !contract
            .inputs
            .iter()
            .any(|input| input.distribution == InputDistribution::Sharded)
    {
        return invalid("partitioned-input execution requires a sharded input");
    }
    if contract.partition_extent.is_none() {
        return invalid("distributed contracts require an explicit partition extent");
    }
    if contract.partition_launch.is_none() {
        return invalid("distributed contracts require an explicit partition launch");
    }
    let workgroup_x = contract
        .partition_launch
        .as_ref()
        .expect("distributed launch was checked above")
        .workgroup_x;
    let mapping_matches_form = match contract.execution_form {
        ExecutionForm::ReplicatedInputPartitionedOutput => {
            workgroup_x == WorkgroupXMapping::Proportional
        }
        ExecutionForm::PartitionedInputPartialOutput | ExecutionForm::WholeExpertOwnership => {
            workgroup_x == WorkgroupXMapping::Repeated
        }
        ExecutionForm::Local => unreachable!("local distributed form was rejected above"),
    };
    if !mapping_matches_form {
        return invalid("partition workgroup mapping disagrees with the execution form");
    }
    let strategy_matches_form = match contract.strategy {
        ExecutionStrategy::TensorParallel | ExecutionStrategy::TensorParallelExpert => matches!(
            contract.execution_form,
            ExecutionForm::ReplicatedInputPartitionedOutput
                | ExecutionForm::PartitionedInputPartialOutput
        ),
        ExecutionStrategy::ExpertParallel => {
            contract.execution_form == ExecutionForm::WholeExpertOwnership
        }
        ExecutionStrategy::SingleDevice => unreachable!("single-device strategy returned above"),
    };
    if !strategy_matches_form {
        return invalid("distributed execution strategy disagrees with its execution form");
    }
    match contract.strategy {
        ExecutionStrategy::TensorParallel if !contract.selected_resource_partitions.is_empty() => {
            return invalid("ordinary tensor parallelism cannot declare selected resources");
        }
        ExecutionStrategy::ExpertParallel
            if contract
                .selected_resource_partitions
                .iter()
                .any(|partition| {
                    !partition.parameter_partitions.is_empty()
                        || contract
                            .partition_extent
                            .as_ref()
                            .is_none_or(|extent| extent.elements != partition.resource_count)
                }) =>
        {
            return invalid(
                "whole-expert parallelism requires unfragmented selected resources matching the expert extent",
            );
        }
        ExecutionStrategy::TensorParallelExpert
            if contract.selected_resource_partitions.is_empty()
                || contract
                    .selected_resource_partitions
                    .iter()
                    .any(|partition| partition.parameter_partitions.is_empty()) =>
        {
            return invalid(
                "intra-resource tensor parallelism requires explicit selected parameter partitions",
            );
        }
        _ => {}
    }
    match contract.execution_form {
        ExecutionForm::ReplicatedInputPartitionedOutput
            if !contract
                .outputs
                .iter()
                .any(|output| output.collection == OutputCollection::Concatenated) =>
        {
            return invalid("partitioned-output execution requires a concatenated output");
        }
        ExecutionForm::WholeExpertOwnership
            if !(contract
                .parameter_partitions
                .iter()
                .any(|partition| partition.kind == ParameterPartitionKind::ExpertRange)
                || contract
                    .selected_resource_partitions
                    .iter()
                    .any(|partition| partition.kind == ParameterPartitionKind::ExpertRange))
                || !contract
                    .inputs
                    .iter()
                    .any(|input| input.distribution == InputDistribution::Routed)
                || !contract
                    .outputs
                    .iter()
                    .any(|output| output.collection == OutputCollection::Routed) =>
        {
            return invalid(
                "whole-expert execution requires expert-range parameters and routed input and output",
            );
        }
        _ => {}
    }
    if contract.parameter_partitions.is_empty() && contract.selected_resource_partitions.is_empty()
    {
        return invalid("distributed contracts require an explicit parameter partition");
    }
    let extent = contract
        .partition_extent
        .as_ref()
        .expect("distributed extent was checked above");
    if contract.inputs.iter().any(|input| {
        input
            .alignment_elements
            .is_some_and(|alignment| extent.alignment_elements % alignment != 0)
    }) || contract.outputs.iter().any(|output| {
        output
            .alignment_elements
            .is_some_and(|alignment| extent.alignment_elements % alignment != 0)
    }) {
        return invalid("distributed input and output alignment must divide partition alignment");
    }
    if !contract.outputs.iter().any(|output| {
        matches!(
            output.collection,
            OutputCollection::Concatenated
                | OutputCollection::Reduced
                | OutputCollection::Routed
                | OutputCollection::Retained
        )
    }) {
        return invalid("distributed contracts require an explicit output collection");
    }
    for intermediate in &contract.local_intermediates {
        require_non_empty(&intermediate.signal, "local_intermediates.signal")?;
        require_non_empty(&intermediate.format, "local_intermediates.format")?;
    }
    Ok(())
}

fn validate_resources(resources: &[ResourceRequirement]) -> Result<(), ContractError> {
    let mut identities = BTreeSet::new();
    for resource in resources {
        require_non_empty(&resource.resource, "resources.resource")?;
        if resource.atomic_group.as_deref().is_some_and(str::is_empty) {
            return invalid("resource atomic group must not be empty");
        }
        if !identities.insert((resource.resource.as_str(), resource.binding)) {
            return invalid("resource requirements must be unique by name and binding");
        }
        if resource.kind == ResourceKind::LazyResource
            && resource.residency != ResidencyRequirement::Demand
        {
            return invalid("lazy resources must be demand resident");
        }
    }
    Ok(())
}

fn validate_equivalence(equivalence: &EquivalenceRequirement) -> Result<(), ContractError> {
    let tolerant = equivalence.output == EquivalenceKind::AbsoluteRelativeTolerance
        || equivalence.state == EquivalenceKind::AbsoluteRelativeTolerance;
    match (
        tolerant,
        equivalence.absolute_tolerance,
        equivalence.relative_tolerance,
    ) {
        (false, None, None) => Ok(()),
        (true, Some(absolute), Some(relative))
            if absolute.is_finite()
                && relative.is_finite()
                && absolute >= 0.0
                && relative >= 0.0 =>
        {
            Ok(())
        }
        (false, _, _) => invalid("bit-exact equivalence forbids tolerances"),
        (true, _, _) => invalid("tolerant equivalence requires finite non-negative tolerances"),
    }
}

fn require_digest(value: &str, path: &str) -> Result<(), ContractError> {
    let Some(hex) = value.strip_prefix("sha256:") else {
        return invalid(format!("{path} must use sha256:<64 lowercase hex>"));
    };
    if hex.len() != 64
        || !hex
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return invalid(format!("{path} must use sha256:<64 lowercase hex>"));
    }
    Ok(())
}

fn require_non_empty(value: &str, path: &str) -> Result<(), ContractError> {
    if value.trim().is_empty() {
        return invalid(format!("{path} must not be empty"));
    }
    Ok(())
}

fn require_non_empty_unique(values: &[String], path: &str) -> Result<(), ContractError> {
    require_unique_strings(values, path, false)
}

fn require_unique_strings(
    values: &[String],
    path: &str,
    allow_empty: bool,
) -> Result<(), ContractError> {
    if (!allow_empty && values.is_empty())
        || values.iter().any(|value| value.trim().is_empty())
        || values.iter().collect::<BTreeSet<_>>().len() != values.len()
    {
        return invalid(format!(
            "{path} must contain unique, non-empty strings{}",
            if allow_empty { " or be empty" } else { "" }
        ));
    }
    Ok(())
}

fn invalid<T>(message: impl Into<String>) -> Result<T, ContractError> {
    Err(ContractError(message.into()))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn digest(byte: char) -> String {
        format!("sha256:{}", byte.to_string().repeat(64))
    }

    fn valid_contract() -> PhysicalExecutionContract {
        PhysicalExecutionContract {
            schema: PHYSICAL_EXECUTION_CONTRACT_SCHEMA.to_string(),
            contract_id: digest('a'),
            operation_family: "parallel_projection".to_string(),
            region_family: Some("ffn_gate_up".to_string()),
            member_node_ids: vec!["gate_up".to_string()],
            artifacts: vec![ArtifactIdentity {
                path: "shaders/gate_up.spv".to_string(),
                sha256: digest('b'),
                entry_point: "main".to_string(),
                role: ArtifactRole::Primary,
            }],
            implementation_digest: digest('c'),
            phases: vec![ExecutionPhase::Decode, ExecutionPhase::Prefill],
            execution_shape: ExecutionShape::SingleAndMultiLane,
            formats: PhysicalFormats {
                storage: "fp8_e4m3".to_string(),
                compute: "fp8_e4m3".to_string(),
                accumulation: "f32".to_string(),
            },
            geometry: ExecutionGeometry {
                shape_class: "projection_m4096_n11008".to_string(),
                dimensions: BTreeMap::from([
                    ("input_width".to_string(), 4096),
                    ("output_rows".to_string(), 11008),
                ]),
                dynamic_dimensions: Vec::new(),
            },
            strategy: ExecutionStrategy::TensorParallel,
            execution_form: ExecutionForm::ReplicatedInputPartitionedOutput,
            partition_extent: Some(PartitionExtent {
                dimension_name: "output_rows".to_string(),
                elements: 11008,
                alignment_elements: 128,
            }),
            partition_launch: Some(PartitionLaunch {
                workgroup_x: WorkgroupXMapping::Proportional,
                origin: PartitionOrigin::LocalZero,
                origin_push_constant: None,
                count_push_constant: None,
            }),
            parameter_partitions: vec![ParameterPartition {
                binding: 2,
                resource: "weight".to_string(),
                dimension: 0,
                kind: ParameterPartitionKind::Contiguous,
                alignment_elements: 128,
                logical_elements_per_index: 1,
            }],
            selected_resource_partitions: Vec::new(),
            inputs: vec![InputContract {
                binding: 0,
                distribution: InputDistribution::Replicated,
                dimension: None,
                alignment_elements: None,
            }],
            outputs: vec![OutputContract {
                binding: 1,
                collection: OutputCollection::Concatenated,
                dimension: Some(0),
                alignment_elements: Some(128),
                reduction: None,
            }],
            local_intermediates: Vec::new(),
            resources: vec![ResourceRequirement {
                resource: "weight".to_string(),
                kind: ResourceKind::PersistentParameter,
                residency: ResidencyRequirement::Permanent,
                access: ResourceAccess::Read,
                binding: Some(2),
                atomic_group: None,
            }],
            equivalence: EquivalenceRequirement {
                output: EquivalenceKind::BitExact,
                state: EquivalenceKind::BitExact,
                absolute_tolerance: None,
                relative_tolerance: None,
            },
        }
    }

    #[test]
    fn valid_distributed_contract_round_trips() {
        let contract = valid_contract();
        contract.validate().unwrap();
        let encoded = serde_json::to_string(&contract).unwrap();
        let decoded: PhysicalExecutionContract = serde_json::from_str(&encoded).unwrap();
        assert_eq!(decoded, contract);
    }

    #[test]
    fn execution_shape_support_is_explicit() {
        assert!(ExecutionShape::SingleLane.supports(ExecutionShape::SingleLane));
        assert!(!ExecutionShape::SingleLane.supports(ExecutionShape::MultiLane));
        assert!(ExecutionShape::SingleAndMultiLane.supports(ExecutionShape::SingleLane));
        assert!(ExecutionShape::SingleAndMultiLane.supports(ExecutionShape::MultiLane));
    }

    #[test]
    fn shared_compiler_fixture_deserializes_and_validates() {
        let fixture = include_str!("../fixtures/tensor_parallel_projection.json");
        let contract: PhysicalExecutionContract = serde_json::from_str(fixture).unwrap();
        contract.validate().unwrap();
    }

    #[test]
    fn distributed_contract_without_partition_fails_closed() {
        let mut contract = valid_contract();
        contract.parameter_partitions.clear();
        assert!(contract.validate().is_err());
    }

    #[test]
    fn selected_resource_partition_requires_exact_demand_residency_geometry() {
        let mut contract = valid_contract();
        contract.strategy = ExecutionStrategy::ExpertParallel;
        contract.execution_form = ExecutionForm::WholeExpertOwnership;
        contract
            .geometry
            .dimensions
            .insert("experts".to_string(), 4);
        contract.partition_extent = Some(PartitionExtent {
            dimension_name: "experts".to_string(),
            elements: 4,
            alignment_elements: 1,
        });
        contract.partition_launch = Some(PartitionLaunch {
            workgroup_x: WorkgroupXMapping::Repeated,
            origin: PartitionOrigin::PushConstantU32,
            origin_push_constant: Some("expert_start".to_string()),
            count_push_constant: Some("expert_count".to_string()),
        });
        contract.parameter_partitions.clear();
        contract.selected_resource_partitions = vec![SelectedResourcePartition {
            selection_signal: "routes".to_string(),
            address_table_binding: 3,
            parameter_slots_binding: 4,
            kind: ParameterPartitionKind::ExpertRange,
            resource_count: 4,
            parameters_per_resource: 2,
            alignment_elements: 1,
            parameter_partitions: Vec::new(),
        }];
        contract.inputs = vec![
            InputContract {
                binding: 0,
                distribution: InputDistribution::Replicated,
                dimension: None,
                alignment_elements: None,
            },
            InputContract {
                binding: 1,
                distribution: InputDistribution::Routed,
                dimension: Some(0),
                alignment_elements: Some(1),
            },
        ];
        contract.outputs = vec![OutputContract {
            binding: 2,
            collection: OutputCollection::Routed,
            dimension: Some(0),
            alignment_elements: Some(1),
            reduction: None,
        }];
        contract.resources = vec![ResourceRequirement {
            resource: "expert-bank".to_string(),
            kind: ResourceKind::LazyResource,
            residency: ResidencyRequirement::Demand,
            access: ResourceAccess::Read,
            binding: None,
            atomic_group: Some("selector-owned".to_string()),
        }];
        contract.validate().unwrap();

        contract.selected_resource_partitions[0].resource_count = 3;
        assert!(
            contract
                .validate()
                .unwrap_err()
                .to_string()
                .contains("matching the expert extent")
        );
    }

    #[test]
    fn intra_resource_tensor_parallelism_separates_selection_from_fragmentation() {
        let mut contract = valid_contract();
        contract.strategy = ExecutionStrategy::TensorParallelExpert;
        contract.execution_form = ExecutionForm::ReplicatedInputPartitionedOutput;
        contract
            .geometry
            .dimensions
            .insert("intermediate".to_string(), 128);
        contract.partition_extent = Some(PartitionExtent {
            dimension_name: "intermediate".to_string(),
            elements: 128,
            alignment_elements: 16,
        });
        contract.partition_launch = Some(PartitionLaunch {
            workgroup_x: WorkgroupXMapping::Proportional,
            origin: PartitionOrigin::LocalZero,
            origin_push_constant: None,
            count_push_constant: None,
        });
        contract.parameter_partitions.clear();
        contract.selected_resource_partitions = vec![SelectedResourcePartition {
            selection_signal: "routes".to_string(),
            address_table_binding: 2,
            parameter_slots_binding: 3,
            kind: ParameterPartitionKind::ExpertRange,
            resource_count: 4,
            parameters_per_resource: 2,
            alignment_elements: 1,
            parameter_partitions: vec![
                SelectedResourceParameterPartition {
                    parameter_slot: 0,
                    dimension: 0,
                    kind: ParameterPartitionKind::Contiguous,
                    alignment_elements: 16,
                    logical_elements_per_index: 1,
                },
                SelectedResourceParameterPartition {
                    parameter_slot: 1,
                    dimension: 0,
                    kind: ParameterPartitionKind::Contiguous,
                    alignment_elements: 16,
                    logical_elements_per_index: 1,
                },
            ],
        }];
        contract.inputs = vec![InputContract {
            binding: 0,
            distribution: InputDistribution::Replicated,
            dimension: None,
            alignment_elements: None,
        }];
        contract.outputs = vec![OutputContract {
            binding: 1,
            collection: OutputCollection::Concatenated,
            dimension: Some(0),
            alignment_elements: Some(16),
            reduction: None,
        }];
        contract.resources = vec![ResourceRequirement {
            resource: "expert-bank".to_string(),
            kind: ResourceKind::LazyResource,
            residency: ResidencyRequirement::Demand,
            access: ResourceAccess::Read,
            binding: None,
            atomic_group: Some("selector-owned".to_string()),
        }];
        contract.validate().unwrap();

        let mut missing_fragments = contract.clone();
        missing_fragments.selected_resource_partitions[0]
            .parameter_partitions
            .clear();
        assert!(
            missing_fragments
                .validate()
                .unwrap_err()
                .to_string()
                .contains("explicit selected parameter partitions")
        );

        let mut repeated_slot = contract.clone();
        repeated_slot.selected_resource_partitions[0].parameter_partitions[1].parameter_slot = 0;
        assert!(
            repeated_slot
                .validate()
                .unwrap_err()
                .to_string()
                .contains("unique valid parameter slots")
        );

        let mut relabelled_whole_expert = contract.clone();
        relabelled_whole_expert.strategy = ExecutionStrategy::ExpertParallel;
        relabelled_whole_expert.execution_form = ExecutionForm::WholeExpertOwnership;
        relabelled_whole_expert
            .geometry
            .dimensions
            .insert("experts".to_string(), 4);
        relabelled_whole_expert.partition_extent = Some(PartitionExtent {
            dimension_name: "experts".to_string(),
            elements: 4,
            alignment_elements: 1,
        });
        relabelled_whole_expert.partition_launch = Some(PartitionLaunch {
            workgroup_x: WorkgroupXMapping::Repeated,
            origin: PartitionOrigin::PushConstantU32,
            origin_push_constant: Some("expert_start".to_string()),
            count_push_constant: Some("expert_count".to_string()),
        });
        relabelled_whole_expert.inputs[0] = InputContract {
            binding: 0,
            distribution: InputDistribution::Routed,
            dimension: Some(0),
            alignment_elements: Some(1),
        };
        relabelled_whole_expert.outputs[0] = OutputContract {
            binding: 1,
            collection: OutputCollection::Routed,
            dimension: Some(0),
            alignment_elements: Some(1),
            reduction: None,
        };
        for fragment in
            &mut relabelled_whole_expert.selected_resource_partitions[0].parameter_partitions
        {
            fragment.alignment_elements = 1;
        }
        assert!(
            relabelled_whole_expert
                .validate()
                .unwrap_err()
                .to_string()
                .contains("unfragmented selected resources")
        );
    }

    #[test]
    fn parameter_partition_must_name_its_declared_physical_resource() {
        let mut contract = valid_contract();
        contract.parameter_partitions[0].resource = "missing".to_string();
        assert!(
            contract
                .validate()
                .unwrap_err()
                .to_string()
                .contains("exactly one declared parameter resource")
        );

        contract.parameter_partitions[0].resource = "weight".to_string();
        contract.resources[0].binding = Some(99);
        assert!(
            contract
                .validate()
                .unwrap_err()
                .to_string()
                .contains("declared binding")
        );
    }

    #[test]
    fn partial_output_contract_requires_typed_f32_sum_reduction() {
        let mut contract = valid_contract();
        contract.execution_form = ExecutionForm::PartitionedInputPartialOutput;
        contract.partition_extent = Some(PartitionExtent {
            dimension_name: "input_width".to_string(),
            elements: 4096,
            alignment_elements: 128,
        });
        contract.inputs[0] = InputContract {
            binding: 0,
            distribution: InputDistribution::Sharded,
            dimension: Some(0),
            alignment_elements: Some(128),
        };
        contract.outputs[0] = OutputContract {
            binding: 1,
            collection: OutputCollection::Reduced,
            dimension: None,
            alignment_elements: None,
            reduction: Some(ReductionContract {
                operation: ReductionOperation::SumF32,
                dimension_name: "output_rows".to_string(),
                finalization: ReductionFinalization::StoreF32,
            }),
        };
        contract.partition_launch = Some(PartitionLaunch {
            workgroup_x: WorkgroupXMapping::Repeated,
            origin: PartitionOrigin::PushConstantU32,
            origin_push_constant: Some("input_start".to_string()),
            count_push_constant: Some("input_count".to_string()),
        });
        contract.validate().unwrap();

        contract.outputs[0].reduction.as_mut().unwrap().finalization =
            ReductionFinalization::StoreF32ToBf16;
        contract.validate().unwrap();

        contract.inputs.push(InputContract {
            binding: 3,
            distribution: InputDistribution::Replicated,
            dimension: None,
            alignment_elements: None,
        });
        contract.outputs[0].reduction.as_mut().unwrap().finalization =
            ReductionFinalization::AddBf16ResidualToBf16 {
                residual_binding: 3,
            };
        contract.validate().unwrap();

        contract.inputs[1].distribution = InputDistribution::Sharded;
        contract.inputs[1].dimension = Some(0);
        contract.inputs[1].alignment_elements = Some(128);
        assert!(
            contract
                .validate()
                .unwrap_err()
                .to_string()
                .contains("must be replicated")
        );

        contract.inputs[1] = InputContract {
            binding: 3,
            distribution: InputDistribution::Replicated,
            dimension: None,
            alignment_elements: None,
        };
        contract.outputs[0].reduction.as_mut().unwrap().finalization =
            ReductionFinalization::AddBf16ResidualToBf16 {
                residual_binding: 99,
            };
        assert!(
            contract
                .validate()
                .unwrap_err()
                .to_string()
                .contains("must name a contract input")
        );

        contract.outputs[0].reduction.as_mut().unwrap().finalization =
            ReductionFinalization::AddBf16ResidualToBf16 {
                residual_binding: 3,
            };
        contract
            .geometry
            .dimensions
            .insert("output_rows".to_string(), 11_007);
        assert!(
            contract
                .validate()
                .unwrap_err()
                .to_string()
                .contains("even element count")
        );

        contract.inputs.pop();
        contract
            .geometry
            .dimensions
            .insert("output_rows".to_string(), 11_008);
        contract.outputs[0].reduction.as_mut().unwrap().finalization =
            ReductionFinalization::StoreF32;

        contract.inputs[0] = InputContract {
            binding: 0,
            distribution: InputDistribution::Replicated,
            dimension: None,
            alignment_elements: None,
        };
        assert!(
            contract
                .validate()
                .unwrap_err()
                .to_string()
                .contains("requires a sharded input")
        );
        contract.inputs[0] = InputContract {
            binding: 0,
            distribution: InputDistribution::Sharded,
            dimension: Some(0),
            alignment_elements: Some(128),
        };
        contract.execution_form = ExecutionForm::ReplicatedInputPartitionedOutput;
        assert!(
            contract
                .validate()
                .unwrap_err()
                .to_string()
                .contains("reduced outputs require that execution form")
        );

        contract.execution_form = ExecutionForm::PartitionedInputPartialOutput;
        contract
            .outputs
            .first_mut()
            .unwrap()
            .reduction
            .as_mut()
            .unwrap()
            .dimension_name = "missing_dimension".to_string();
        assert!(
            contract
                .validate()
                .unwrap_err()
                .to_string()
                .contains("declared geometry dimension")
        );
    }

    #[test]
    fn repeated_partition_requires_distinct_declared_range_controls() {
        let mut contract = valid_contract();
        contract.strategy = ExecutionStrategy::ExpertParallel;
        contract.execution_form = ExecutionForm::WholeExpertOwnership;
        contract.parameter_partitions[0].kind = ParameterPartitionKind::ExpertRange;
        contract.inputs[0] = InputContract {
            binding: 0,
            distribution: InputDistribution::Routed,
            dimension: Some(0),
            alignment_elements: Some(128),
        };
        contract.outputs[0] = OutputContract {
            binding: 1,
            collection: OutputCollection::Routed,
            dimension: Some(0),
            alignment_elements: Some(128),
            reduction: None,
        };
        contract.partition_launch = Some(PartitionLaunch {
            workgroup_x: WorkgroupXMapping::Repeated,
            origin: PartitionOrigin::PushConstantU32,
            origin_push_constant: Some("expert_start".to_string()),
            count_push_constant: None,
        });
        assert!(contract.validate().is_err());

        contract
            .partition_launch
            .as_mut()
            .unwrap()
            .count_push_constant = Some("expert_start".to_string());
        assert!(contract.validate().is_err());

        contract
            .partition_launch
            .as_mut()
            .unwrap()
            .count_push_constant = Some("expert_count".to_string());
        contract.validate().unwrap();
    }

    #[test]
    fn unknown_reduction_operation_fails_deserialization() {
        let mut value = serde_json::to_value(valid_contract()).unwrap();
        value["execution_form"] = serde_json::json!("partitioned_input_partial_output");
        value["outputs"][0] = serde_json::json!({
            "binding": 1,
            "collection": "reduced",
            "reduction": {
                "operation": "vendor_magic",
                "dimension_name": "output_rows",
                "finalization": {"kind": "store_f32"}
            },
        });
        assert!(serde_json::from_value::<PhysicalExecutionContract>(value).is_err());
    }

    #[test]
    fn block_scaled_partition_requires_logically_aligned_slices() {
        let mut contract = valid_contract();
        contract.parameter_partitions[0].alignment_elements = 1;
        contract.parameter_partitions[0].logical_elements_per_index = 256;
        assert!(contract.validate().is_err());
    }

    #[test]
    fn unknown_contract_fields_are_rejected() {
        let mut value = serde_json::to_value(valid_contract()).unwrap();
        value.as_object_mut().unwrap().insert(
            "guessed_distribution".to_string(),
            serde_json::Value::Bool(true),
        );
        assert!(serde_json::from_value::<PhysicalExecutionContract>(value).is_err());
    }

    #[test]
    fn lazy_resource_must_be_demand_resident() {
        let mut contract = valid_contract();
        contract.resources[0].kind = ResourceKind::LazyResource;
        assert!(contract.validate().is_err());
    }

    #[test]
    fn bit_exact_contract_rejects_hidden_tolerance() {
        let mut contract = valid_contract();
        contract.equivalence.absolute_tolerance = Some(0.0);
        contract.equivalence.relative_tolerance = Some(0.0);
        assert!(contract.validate().is_err());
    }
}
