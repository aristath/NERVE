use std::collections::BTreeMap;
use std::error::Error;
use std::io::{self, Read};
use std::path::PathBuf;

use nerve_runtime::{
    ResourceResidencyPolicy, VULKAN_RUNTIME_RESIDENCY_PLAN_SCHEMA,
    VulkanResidentModelPackageManifest, plan_vulkan_runtime_residency,
};
use serde::{Deserialize, Serialize};

const REQUEST_SCHEMA: &str = "nerve.runtime_residency_planner_request.v3";
const RESPONSE_SCHEMA: &str = "nerve.runtime_residency_planner_response.v3";

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ResidencyPlannerRequest {
    schema: String,
    package_manifest: PathBuf,
    cases: Vec<ResidencyPlannerCase>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ResidencyPlannerCase {
    case_id: String,
    default_device_id: String,
    component_placement: BTreeMap<String, String>,
    context_capacity_activations: usize,
    speculative_draft_tokens: usize,
    residency_policy: ResourceResidencyPolicy,
}

#[derive(Debug, Serialize)]
struct ResidencyPlannerResponse {
    schema: String,
    plans: Vec<ResidencyPlannerCaseResponse>,
}

#[derive(Debug, Serialize)]
struct ResidencyPlannerCaseResponse {
    case_id: String,
    plan: nerve_runtime::VulkanRuntimeResidencyPlan,
}

fn main() {
    if std::env::args()
        .skip(1)
        .eq(["--runtime-implementation-fingerprint"])
    {
        println!("{}", nerve_runtime::RUNTIME_IMPLEMENTATION_FINGERPRINT);
        return;
    }
    if let Err(error) = run() {
        eprintln!("nerve-residency-planner error: {error}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), Box<dyn Error>> {
    let mut input = Vec::new();
    io::stdin().read_to_end(&mut input)?;
    let request: ResidencyPlannerRequest = serde_json::from_slice(&input)?;
    if request.schema != REQUEST_SCHEMA {
        return Err(
            invalid_input(format!("unsupported request schema {:?}", request.schema)).into(),
        );
    }
    if request.cases.is_empty() {
        return Err(invalid_input("residency planning requires at least one case").into());
    }
    let package_manifest = request.package_manifest.canonicalize()?;
    if !package_manifest.is_file() {
        return Err(invalid_input("package manifest is not a regular file").into());
    }
    let package_root = package_manifest
        .parent()
        .ok_or_else(|| invalid_input("package manifest has no package root"))?;
    let manifest = VulkanResidentModelPackageManifest::from_json_file(&package_manifest)?;
    let mut plans = Vec::with_capacity(request.cases.len());
    let mut case_ids = std::collections::BTreeSet::new();
    let mut tensor_index = None;
    for case in request.cases {
        if case.case_id.is_empty() || !case_ids.insert(case.case_id.clone()) {
            return Err(
                invalid_input("residency planner case_id must be unique and nonempty").into(),
            );
        }
        if case.default_device_id.is_empty() {
            return Err(
                invalid_input("residency planner default_device_id must be nonempty").into(),
            );
        }
        let runtime_model = manifest.clone().mount_runtime_graph_controls(
            Some(&case.default_device_id),
            &case.component_placement,
            &[],
            None,
        )?;
        if tensor_index.is_none() {
            tensor_index = Some(runtime_model.load_runtime_tensor_index(package_root)?);
        }
        let plan = plan_vulkan_runtime_residency(
            package_root,
            &runtime_model,
            tensor_index
                .as_ref()
                .expect("tensor index was initialized above"),
            case.context_capacity_activations,
            case.speculative_draft_tokens,
            case.residency_policy,
        )?;
        if plan.schema != VULKAN_RUNTIME_RESIDENCY_PLAN_SCHEMA {
            return Err(
                invalid_input("runtime returned an unexpected residency plan schema").into(),
            );
        }
        plans.push(ResidencyPlannerCaseResponse {
            case_id: case.case_id,
            plan,
        });
    }
    serde_json::to_writer(
        io::stdout().lock(),
        &ResidencyPlannerResponse {
            schema: RESPONSE_SCHEMA.to_string(),
            plans,
        },
    )?;
    Ok(())
}

fn invalid_input(message: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidInput, message.into())
}
