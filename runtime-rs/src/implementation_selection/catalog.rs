use super::{
    BENCHMARK_RECORD_SCHEMA, IMPLEMENTATION_REGISTRY_SCHEMA, LoadedRuntimeImplementation,
    OPTIMIZATION_SCOPE_CATALOG_SCHEMA, OPTIMIZER_STAGE_SCHEMA, PROMOTION_DECISION_SCHEMA,
    RUNTIME_MOUNT_PLAN_SCHEMA, RuntimeImplementationCatalog, RuntimeImplementationRegistry,
    RuntimeImplementationWorkloadMetrics, RuntimeMountPlan, RuntimeOptimizationScope,
    VALIDATION_RECORD_SCHEMA, VULKAN_COMPONENT_OVERLAY_SCHEMA,
    VULKAN_STREAM_CIRCUIT_OVERLAY_ADAPTER,
};
use serde_json::{Map, Value};
use std::collections::{BTreeMap, BTreeSet};
use std::io;
use std::path::Path;

use super::artifacts::{
    array, confined_path, from_value, invalid, invalid_error, object, read_json, read_object,
    require_schema, required, strictly_sorted_unique, string_array, text, unsigned, unsigned_u64,
};

impl RuntimeImplementationCatalog {
    pub fn load(
        package_root: impl AsRef<Path>,
        stage_reference: &str,
        package_id: &str,
    ) -> io::Result<Self> {
        let package_root = package_root.as_ref().canonicalize()?;
        let stage_path = confined_path(&package_root, stage_reference, "optimizer stage")?;
        let stage = read_object(&stage_path, "optimizer stage")?;
        require_schema(&stage, OPTIMIZER_STAGE_SCHEMA, "optimizer stage")?;
        let stage_status = text(&stage, "status", "optimizer stage")?;
        if stage_status != "exact_baseline_retained" && stage_status != "optimized" {
            return invalid(format!(
                "optimizer stage has unsupported status {stage_status:?}"
            ));
        }
        let session = object(&stage, "session", "optimizer stage")?;
        if text(session, "package_id", "optimizer session")? != package_id {
            return invalid("optimizer stage belongs to a different compiled package");
        }
        let exact_baseline: super::RuntimeExactImplementation = from_value(
            required(&stage, "exact_baseline", "optimizer stage")?.clone(),
            "optimizer exact baseline",
        )?;
        if exact_baseline.mutable {
            return invalid("optimizer exact baseline must be immutable");
        }

        let registry_ref = object(&stage, "implementation_registry", "optimizer stage")?;
        let registry_path = confined_path(
            &package_root,
            text(
                registry_ref,
                "artifact_ref",
                "implementation registry reference",
            )?,
            "implementation registry",
        )?;
        let registry: RuntimeImplementationRegistry =
            read_json(&registry_path, "implementation registry")?;
        if registry.schema != IMPLEMENTATION_REGISTRY_SCHEMA
            || registry.package_id != package_id
            || registry.exact_baseline != exact_baseline
        {
            return invalid("implementation registry does not match the optimizer stage");
        }
        if registry.implementations.len()
            != unsigned(
                registry_ref,
                "implementation_count",
                "implementation registry reference",
            )?
        {
            return invalid("optimizer stage implementation count does not match its registry");
        }
        let implementation_ids = registry
            .implementations
            .iter()
            .map(|implementation| implementation.implementation_id.as_str())
            .collect::<Vec<_>>();
        if !strictly_sorted_unique(&implementation_ids) {
            return invalid("implementation registry entries must be sorted and unique");
        }

        let scope_ref = object(&stage, "scope_catalog", "optimizer stage")?;
        let scope_path = confined_path(
            &package_root,
            text(scope_ref, "artifact_ref", "scope catalog reference")?,
            "scope catalog",
        )?;
        let scopes = load_scopes(&scope_path, package_id)?;
        if scopes.len() != unsigned(scope_ref, "scope_count", "scope catalog reference")? {
            return invalid("optimizer stage scope count does not match its catalog");
        }

        let mut implementations = Vec::with_capacity(registry.implementations.len());
        for implementation in registry.implementations {
            implementations.push(load_implementation(&package_root, implementation, &scopes)?);
        }
        if stage_status == "exact_baseline_retained" && !implementations.is_empty() {
            return invalid("an exact-baseline stage cannot expose promoted implementations");
        }
        if stage_status == "optimized" && implementations.is_empty() {
            return invalid("an optimized stage must expose a promoted implementation");
        }

        Ok(Self {
            package_id: package_id.to_string(),
            package_root,
            stage_status: stage_status.to_string(),
            exact_baseline,
            scopes,
            implementations,
        })
    }
}

fn load_scopes(
    path: &Path,
    package_id: &str,
) -> io::Result<BTreeMap<String, RuntimeOptimizationScope>> {
    let catalog = read_object(path, "optimization scope catalog")?;
    require_schema(
        &catalog,
        OPTIMIZATION_SCOPE_CATALOG_SCHEMA,
        "optimization scope catalog",
    )?;
    if text(&catalog, "package_id", "optimization scope catalog")? != package_id {
        return invalid("optimization scope catalog belongs to another package");
    }
    let raw_scopes = array(&catalog, "scopes", "optimization scope catalog")?;
    let mut scopes = BTreeMap::new();
    for (index, raw_scope) in raw_scopes.iter().enumerate() {
        let label = format!("optimization scope {index}");
        let scope = raw_scope
            .as_object()
            .ok_or_else(|| invalid_error(format!("{label} must be an object")))?;
        let scope_id = text(scope, "scope_id", &label)?.to_string();
        let source_contract_digest = text(scope, "source_contract_digest", &label)?.to_string();
        let members = object(scope, "members", &label)?;
        let component_ids = string_array(members, "component_ids", &format!("{label} members"))?;
        if component_ids.is_empty() {
            return invalid(format!("{label} contains no source components"));
        }
        let runtime_scope = RuntimeOptimizationScope {
            scope_id: scope_id.clone(),
            source_contract_digest,
            component_ids,
        };
        if scopes.insert(scope_id, runtime_scope).is_some() {
            return invalid("optimization scope catalog contains duplicate identities");
        }
    }
    Ok(scopes)
}

fn load_implementation(
    package_root: &Path,
    implementation: super::RuntimeImplementation,
    scopes: &BTreeMap<String, RuntimeOptimizationScope>,
) -> io::Result<LoadedRuntimeImplementation> {
    if implementation.scope_ids.is_empty()
        || implementation.scope_ids.len() != implementation.source_contract_digests.len()
    {
        return invalid("runtime implementation has invalid semantic scope coverage");
    }
    let mut source_component_ids = BTreeSet::new();
    for (scope_id, source_digest) in implementation
        .scope_ids
        .iter()
        .zip(&implementation.source_contract_digests)
    {
        let scope = scopes.get(scope_id).ok_or_else(|| {
            invalid_error(format!(
                "runtime implementation references unknown scope {scope_id:?}"
            ))
        })?;
        if scope.source_contract_digest != *source_digest {
            return invalid(format!(
                "runtime implementation scope {scope_id:?} changed after promotion"
            ));
        }
        source_component_ids.extend(scope.component_ids.iter().cloned());
    }
    if implementation.comparison.benchmark_decision != "materially_faster"
        || implementation.comparison.validation_status != "passed"
        || implementation.comparison.workloads.iter().any(|workload| {
            workload.decision != "materially_faster"
                || !workload.paired.candidate_is_faster
                || workload.paired.speedup_ppm <= 0
        })
    {
        return invalid("runtime registry contains an unqualified implementation");
    }

    let root = confined_path(
        package_root,
        &implementation.artifact_bundle.root_ref,
        "implementation artifact bundle",
    )?;
    if !root.is_dir() {
        return invalid("implementation artifact bundle is missing");
    }
    let candidate_root = confined_path(&root, "candidate", "implementation candidate bundle")?;
    if !candidate_root.is_dir() {
        return invalid("implementation candidate bundle is missing");
    }
    let mount_plan_path = confined_path(
        package_root,
        &implementation.artifact_bundle.mount_plan_ref,
        "runtime mount plan",
    )?;
    if !mount_plan_path.starts_with(&candidate_root) {
        return invalid("runtime mount plan must stay inside its candidate bundle");
    }
    let mount_plan: RuntimeMountPlan = read_json(&mount_plan_path, "runtime mount plan")?;
    validate_mount_plan(
        &candidate_root,
        &mount_plan,
        &implementation.candidate_id,
        &source_component_ids,
    )?;
    let promotion_path = confined_path(
        package_root,
        &implementation.evidence.promotion_decision_ref,
        "promotion decision",
    )?;
    let promotion = read_object(&promotion_path, "promotion decision")?;
    require_schema(&promotion, PROMOTION_DECISION_SCHEMA, "promotion decision")?;
    if text(&promotion, "decision", "promotion decision")? != "promote"
        || text(&promotion, "implementation_id", "promotion decision")?
            != implementation.implementation_id
        || text(&promotion, "candidate_id", "promotion decision")? != implementation.candidate_id
        || required(&promotion, "runtime_predicate", "promotion decision")?
            != &serde_json::to_value(&implementation.runtime_predicate)
                .map_err(|error| invalid_error(error.to_string()))?
    {
        return invalid(
            "runtime implementation promotion decision does not match its registry entry",
        );
    }

    let validation_path = confined_path(
        package_root,
        &implementation.evidence.validation_record_ref,
        "validation record",
    )?;
    let validation = read_object(&validation_path, "validation record")?;
    require_schema(&validation, VALIDATION_RECORD_SCHEMA, "validation record")?;
    if text(&validation, "status", "validation record")? != "passed"
        || text(&validation, "candidate_id", "validation record")? != implementation.candidate_id
        || text(&validation, "validation_id", "validation record")?
            != implementation.comparison.validation_id
    {
        return invalid("runtime implementation validation evidence is not passed");
    }

    let benchmark_path = confined_path(
        package_root,
        &implementation.evidence.benchmark_record_ref,
        "benchmark record",
    )?;
    let benchmark = read_object(&benchmark_path, "benchmark record")?;
    require_schema(&benchmark, BENCHMARK_RECORD_SCHEMA, "benchmark record")?;
    if text(&benchmark, "decision", "benchmark record")? != "materially_faster"
        || text(&benchmark, "candidate_id", "benchmark record")? != implementation.candidate_id
        || text(&benchmark, "benchmark_id", "benchmark record")?
            != implementation.comparison.benchmark_id
    {
        return invalid("runtime implementation benchmark evidence is not a measured win");
    }
    let workload_metrics = load_workload_metrics(&benchmark_path, &benchmark, &implementation)?;

    for reference in [
        implementation
            .artifact_bundle
            .candidate_integrity_ref
            .as_str(),
        implementation.evidence.candidate_contract_ref.as_str(),
        implementation.evidence.construction_record_ref.as_str(),
        implementation.evidence.prebenchmark_record_ref.as_str(),
    ] {
        let path = confined_path(package_root, reference, "implementation evidence")?;
        if !path.is_file() {
            return invalid("runtime implementation evidence is incomplete");
        }
    }

    Ok(LoadedRuntimeImplementation {
        implementation,
        source_component_ids: source_component_ids.into_iter().collect(),
        workload_metrics,
        candidate_root,
        mount_plan,
    })
}

pub(super) fn validate_mount_plan(
    candidate_root: &Path,
    mount_plan: &RuntimeMountPlan,
    candidate_id: &str,
    source_component_ids: &BTreeSet<String>,
) -> io::Result<()> {
    if mount_plan.schema != RUNTIME_MOUNT_PLAN_SCHEMA
        || mount_plan.candidate_id != candidate_id
        || mount_plan.adapter_id != VULKAN_STREAM_CIRCUIT_OVERLAY_ADAPTER
        || mount_plan.regions.is_empty()
    {
        return invalid("runtime implementation mount plan is unsupported");
    }
    let region_sources = mount_plan
        .regions
        .iter()
        .map(|region| {
            region
                .component_replacements
                .iter()
                .map(|replacement| replacement.source_component_id.as_str())
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    if region_sources
        .iter()
        .any(|sources| sources.is_empty() || !strictly_sorted_unique(sources))
        || region_sources.windows(2).any(|pair| pair[0] >= pair[1])
    {
        return invalid("runtime mount-plan regions must be non-empty, sorted, and unique");
    }
    let replacement_sources = region_sources.iter().flatten().copied().collect::<Vec<_>>();
    if replacement_sources.len()
        != replacement_sources
            .iter()
            .copied()
            .collect::<BTreeSet<_>>()
            .len()
        || replacement_sources.iter().copied().collect::<BTreeSet<_>>()
            != source_component_ids
                .iter()
                .map(String::as_str)
                .collect::<BTreeSet<_>>()
    {
        return invalid("runtime mount plan must replace exactly its covered source components");
    }
    let mut references = BTreeSet::new();
    for replacement in mount_plan
        .regions
        .iter()
        .flat_map(|region| region.component_replacements.iter())
    {
        if !references.insert(replacement.overlay_ref.as_str()) {
            return invalid("runtime mount plan reuses an artifact reference");
        }
        let overlay = read_object(
            &confined_path(
                candidate_root,
                &replacement.overlay_ref,
                "runtime component overlay",
            )?,
            "runtime component overlay",
        )?;
        require_schema(
            &overlay,
            VULKAN_COMPONENT_OVERLAY_SCHEMA,
            "runtime component overlay",
        )?;
        if text(&overlay, "source_component_id", "runtime component overlay")?
            != replacement.source_component_id
            || !required(&overlay, "component", "runtime component overlay")?.is_object()
            || !required(&overlay, "execution", "runtime component overlay")?.is_object()
        {
            return invalid("runtime component overlay does not match its mount-plan replacement");
        }
    }
    if !strictly_sorted_unique(
        &mount_plan
            .tensor_index_refs
            .iter()
            .map(String::as_str)
            .collect::<Vec<_>>(),
    ) {
        return invalid("runtime mount-plan tensor indexes must be sorted and unique");
    }
    for reference in &mount_plan.tensor_index_refs {
        if !references.insert(reference) {
            return invalid("runtime mount plan reuses an artifact reference");
        }
        let fragment = read_object(
            &confined_path(candidate_root, reference, "runtime tensor-index fragment")?,
            "runtime tensor-index fragment",
        )?;
        require_schema(
            &fragment,
            "nerve.tensor_index.v1",
            "runtime tensor-index fragment",
        )?;
        object(&fragment, "tensors", "runtime tensor-index fragment")?;
    }
    Ok(())
}

fn load_workload_metrics(
    benchmark_path: &Path,
    benchmark: &Map<String, Value>,
    implementation: &super::RuntimeImplementation,
) -> io::Result<Vec<RuntimeImplementationWorkloadMetrics>> {
    let plan = read_object(
        &benchmark_path.with_file_name("plan.json"),
        "benchmark plan",
    )?;
    let planned = array(&plan, "workloads", "benchmark plan")?
        .iter()
        .map(|workload| {
            let workload = workload
                .as_object()
                .ok_or_else(|| invalid_error("benchmark workload must be an object"))?;
            Ok((
                text(workload, "workload_id", "benchmark workload")?.to_string(),
                object(workload, "regime", "benchmark workload")?.clone(),
            ))
        })
        .collect::<io::Result<BTreeMap<_, _>>>()?;
    let compared = implementation
        .comparison
        .workloads
        .iter()
        .map(|workload| (workload.workload_id.as_str(), workload))
        .collect::<BTreeMap<_, _>>();
    let mut metrics = Vec::new();
    for raw in array(benchmark, "workloads", "benchmark record")? {
        let workload = raw
            .as_object()
            .ok_or_else(|| invalid_error("benchmark workload result must be an object"))?;
        if text(workload, "decision", "benchmark workload result")? != "materially_faster" {
            return invalid("runtime implementation contains a non-winning workload");
        }
        let workload_id = text(workload, "workload_id", "benchmark workload result")?;
        let regime = planned
            .get(workload_id)
            .ok_or_else(|| invalid_error("benchmark result references an undeclared workload"))?;
        let comparison = compared
            .get(workload_id)
            .ok_or_else(|| invalid_error("registry comparison omits a benchmark workload"))?;
        let candidate = object(workload, "candidate", "benchmark workload result")?;
        let reference = object(workload, "reference", "benchmark workload result")?;
        let candidate_latency = object(candidate, "latency_ns", "candidate workload metrics")?;
        let reference_latency = object(reference, "latency_ns", "reference workload metrics")?;
        let reference_latency_ns = unsigned_u64(reference_latency, "mean", "reference latency")?;
        let candidate_latency_ns = unsigned_u64(candidate_latency, "mean", "candidate latency")?;
        if reference_latency_ns == 0
            || candidate_latency_ns >= reference_latency_ns
            || !comparison.paired.candidate_is_faster
            || comparison.paired.speedup_ppm <= 0
        {
            return invalid(
                "runtime implementation benchmark metrics do not describe a measured win",
            );
        }
        metrics.push(RuntimeImplementationWorkloadMetrics {
            workload_id: workload_id.to_string(),
            phase: text(regime, "execution_phase", "benchmark regime")?.to_string(),
            activation_batch_width: unsigned(regime, "activation_batch_width", "benchmark regime")?,
            context_activations: unsigned(regime, "context_size", "benchmark regime")?,
            state_activations: unsigned(regime, "state_size", "benchmark regime")?,
            reference_latency_ns,
            candidate_latency_ns,
            conversion_ns: unsigned_u64(candidate, "conversion_ns", "candidate workload metrics")?,
            conversion_bytes: unsigned_u64(
                candidate,
                "conversion_bytes",
                "candidate workload metrics",
            )?,
            boundary_count: unsigned_u64(
                candidate,
                "boundary_count",
                "candidate workload metrics",
            )?,
            speedup_ppm: comparison.paired.speedup_ppm,
        });
    }
    metrics.sort_by(|left, right| left.workload_id.cmp(&right.workload_id));
    if metrics.len() != compared.len() {
        return invalid("registry comparison and benchmark workload coverage differ");
    }
    Ok(metrics)
}
