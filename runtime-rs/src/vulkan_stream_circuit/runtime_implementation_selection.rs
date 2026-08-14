#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct VulkanRuntimeComponentOverlay {
    schema: String,
    source_component_id: String,
    component: VulkanResidentPackageComponentCircuit,
    execution: VulkanResidentComponentExecutionSpec,
    resident_derivations: Vec<VulkanRuntimeParameterResidentDerivation>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct VulkanRuntimeOutputTransducerOverlay {
    schema: String,
    source_component_id: String,
    component: VulkanResidentPackageComponentCircuit,
    output_transducer: VulkanResidentOutputTransducerPackageSpec,
    speculative_output_transducers: Vec<VulkanRuntimeDraftOutputTransducerOverlay>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct VulkanRuntimeDraftOutputTransducerOverlay {
    decoder_id: String,
    output_transducer: VulkanResidentDraftOutputTransducerPackageSpec,
}

#[derive(Clone, Debug, PartialEq)]
pub struct VulkanRuntimeHybridRepresentationApplication {
    pub runtime_model: VulkanResidentRuntimeModel,
    pub selection: crate::RuntimeImplementationSelectionReport,
    pub semantic_contract_id: String,
}

pub fn vulkan_runtime_implementation_semantic_contract_id(
    selected: &crate::RuntimeSelectedImplementation,
) -> Result<String, VulkanResidentTokenModelPackageError> {
    if selected.source_contract_digests.is_empty()
        || selected.source_contract_digests.iter().any(String::is_empty)
    {
        return Err(VulkanResidentTokenModelPackageError::new(
            "runtime implementation has no source-semantic contract",
        ));
    }
    let mut source_contract_digests = selected.source_contract_digests.clone();
    source_contract_digests.sort();
    let payload = serde_json::to_vec(&serde_json::json!({
        "schema": "nerve.runtime_source_semantic_contract.v1",
        "source_contract_digests": source_contract_digests,
    }))
    .map_err(|error| {
        VulkanResidentTokenModelPackageError::new(format!(
            "failed to encode runtime source-semantic contract: {error}",
        ))
    })?;
    Ok(format!("sha256:{:x}", Sha256::digest(payload)))
}

impl crate::RuntimeSelectionRequest {
    pub fn from_vulkan_runtime_model(
        runtime_model: &VulkanResidentRuntimeModel,
        profiles_by_logical_device: &BTreeMap<
            String,
            crate::HardwareProcessProfile,
        >,
        execution: crate::RuntimeExecutionEnvelope,
        exact_baseline_incompatible_instance_ids: BTreeSet<String>,
    ) -> Result<Self, VulkanResidentTokenModelPackageError> {
        let source_roles = runtime_model
            .package
            .circuit_graph
            .components
            .iter()
            .map(|component| {
                (
                    component.component_id.as_str(),
                    component.runtime_role,
                )
            })
            .collect::<BTreeMap<_, _>>();
        let mut instances = runtime_model
            .runtime_graph
            .instances
            .iter()
            .filter(|instance| {
                source_roles
                    .get(instance.source_component_id.as_str())
                    .is_some_and(|role| role.is_runtime_implementation_target())
            })
            .map(|instance| {
                let mut logical_device_ids = vec![
                    runtime_model
                        .placement
                        .device_for_component(&instance.instance_id)
                        .to_string(),
                ];
                if let Some(shards) = runtime_model
                    .placement
                    .component_shard_devices
                    .get(&instance.instance_id)
                {
                    logical_device_ids.extend(shards.iter().cloned());
                }
                logical_device_ids.sort();
                logical_device_ids.dedup();
                crate::RuntimeSelectionInstance {
                    instance_id: instance.instance_id.clone(),
                    source_component_id: instance
                        .source_component_id
                        .clone(),
                    logical_device_ids,
                }
            })
            .collect::<Vec<_>>();
        instances.sort_by(|left, right| {
            left.instance_id.cmp(&right.instance_id)
        });
        let included_instances = instances
            .iter()
            .map(|instance| instance.instance_id.as_str())
            .collect::<BTreeSet<_>>();
        let mut edges = runtime_model
            .runtime_graph
            .effective_edges()
            .map_err(|error| {
                VulkanResidentTokenModelPackageError::new(
                    error.to_string(),
                )
            })?
            .into_iter()
            .filter(|edge| {
                included_instances
                    .contains(edge.source.component_id.as_str())
                    && included_instances
                        .contains(edge.destination.component_id.as_str())
            })
            .map(|edge| crate::RuntimeSelectionEdge {
                source_instance_id: edge.source.component_id,
                destination_instance_id: edge.destination.component_id,
            })
            .collect::<Vec<_>>();
        edges.sort_by(|left, right| {
            (
                left.source_instance_id.as_str(),
                left.destination_instance_id.as_str(),
            )
                .cmp(&(
                    right.source_instance_id.as_str(),
                    right.destination_instance_id.as_str(),
                ))
        });
        let devices = profiles_by_logical_device
            .iter()
            .map(|(logical_device_id, profile)| {
                crate::RuntimeSelectionDevice {
                    logical_device_id: logical_device_id.clone(),
                    physical_device_id: profile
                        .hardware_identity
                        .stable_device_id
                        .clone(),
                    profile: profile.clone(),
                }
            })
            .collect();
        Ok(Self {
            execution,
            devices,
            instances,
            edges,
            exact_baseline_incompatible_instance_ids,
        })
    }
}

impl VulkanResidentRuntimeModel {
    pub fn select_runtime_implementations(
        &self,
        package_root: impl AsRef<Path>,
        profiles_by_logical_device: &BTreeMap<
            String,
            crate::HardwareProcessProfile,
        >,
        execution: crate::RuntimeExecutionEnvelope,
    ) -> io::Result<crate::RuntimeImplementationSelectionReport> {
        let package_root = package_root.as_ref();
        let catalog = self
            .package
            .implementation_catalog(package_root)?;
        let exact_baseline_incompatible_instance_ids =
            vulkan_runtime_exact_baseline_incompatible_instance_ids(
                self,
                package_root,
                profiles_by_logical_device,
            )
            .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error.to_string()))?;
        let request =
            crate::RuntimeSelectionRequest::from_vulkan_runtime_model(
                self,
                profiles_by_logical_device,
                execution,
                exact_baseline_incompatible_instance_ids,
            )
            .map_err(|error| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    error.to_string(),
                )
            })?;
        catalog.select(&request)
    }

    pub fn select_and_apply_runtime_implementations(
        self,
        package_root: impl AsRef<Path>,
        profiles_by_logical_device: &BTreeMap<
            String,
            crate::HardwareProcessProfile,
        >,
        execution: crate::RuntimeExecutionEnvelope,
    ) -> Result<
        (
            Self,
            crate::RuntimeImplementationSelectionReport,
        ),
        VulkanResidentTokenModelPackageError,
    > {
        let package_root = package_root
            .as_ref()
            .canonicalize()
            .map_err(|error| {
                VulkanResidentTokenModelPackageError::new(format!(
                    "failed to resolve runtime package root: {error}"
                ))
            })?;
        let catalog = self
            .package
            .implementation_catalog(&package_root)
            .map_err(|error| {
                VulkanResidentTokenModelPackageError::new(format!(
                    "failed to load runtime implementation catalog: {error}"
                ))
            })?;
        let exact_baseline_incompatible_instance_ids =
            vulkan_runtime_exact_baseline_incompatible_instance_ids(
                &self,
                &package_root,
                profiles_by_logical_device,
            )?;
        let request = crate::RuntimeSelectionRequest::from_vulkan_runtime_model(
            &self,
            profiles_by_logical_device,
            execution,
            exact_baseline_incompatible_instance_ids,
        )?;
        let selection = catalog.select(&request).map_err(|error| {
            VulkanResidentTokenModelPackageError::new(format!(
                "failed to select runtime implementations: {error}"
            ))
        })?;
        let mounted = self.apply_runtime_implementation_catalog_selection(
            &package_root,
            &catalog,
            selection.clone(),
        )?;
        Ok((mounted, selection))
    }

    /// Mounts the exact baseline plus every independently applicable verified
    /// implementation region for one concrete placement and execution
    /// envelope. The globally selected winning set is included as well when it
    /// is not already represented. This keeps exhaustive calibration linear in
    /// selectable regions rather than enumerating the power set of unrelated
    /// layer replacements.
    pub fn applicable_runtime_implementation_variants(
        &self,
        package_root: impl AsRef<Path>,
        profiles_by_logical_device: &BTreeMap<String, crate::HardwareProcessProfile>,
        execution: crate::RuntimeExecutionEnvelope,
    ) -> Result<Vec<Self>, VulkanResidentTokenModelPackageError> {
        let package_root = package_root.as_ref().canonicalize().map_err(|error| {
            VulkanResidentTokenModelPackageError::new(format!(
                "failed to resolve runtime package root: {error}"
            ))
        })?;
        let catalog = self
            .package
            .implementation_catalog(&package_root)
            .map_err(|error| {
                VulkanResidentTokenModelPackageError::new(format!(
                    "failed to load runtime implementation catalog: {error}"
                ))
            })?;
        // Exhaustive component calibration mounts one candidate region at a
        // time and executes only its targeted signature. Unrelated exact
        // components are deliberately not required to execute on this owner;
        // the targeted calibration path validates the mounted signature before
        // submitting it.
        let request = crate::RuntimeSelectionRequest::from_vulkan_runtime_model(
            self,
            profiles_by_logical_device,
            execution,
            BTreeSet::new(),
        )?;
        let reports = catalog
            .calibration_selections(&request)
            .map_err(|error| {
                VulkanResidentTokenModelPackageError::new(format!(
                    "failed to enumerate runtime implementation variants: {error}"
                ))
            })?;
        let mut variants = Vec::with_capacity(reports.len() + 1);
        variants.push(self.clone());
        for report in reports {
            variants.push(self.clone().apply_runtime_implementation_catalog_selection(
                &package_root,
                &catalog,
                report,
            )?);
        }
        Ok(variants)
    }

    /// Mounts every independently applicable, compiler-validated
    /// signal-processor representation. The hybrid solver consumes these as
    /// indivisible semantic applications; unrelated exact components remain
    /// present only so each mounted model is a valid self-contained package.
    pub fn hybrid_signal_representation_applications(
        &self,
        package_root: impl AsRef<Path>,
        profiles_by_logical_device: &BTreeMap<String, crate::HardwareProcessProfile>,
        execution: crate::RuntimeExecutionEnvelope,
    ) -> Result<Vec<VulkanRuntimeHybridRepresentationApplication>, VulkanResidentTokenModelPackageError>
    {
        let package_root = package_root.as_ref().canonicalize().map_err(|error| {
            VulkanResidentTokenModelPackageError::new(format!(
                "failed to resolve runtime package root: {error}",
            ))
        })?;
        let catalog = self
            .package
            .implementation_catalog(&package_root)
            .map_err(|error| {
                VulkanResidentTokenModelPackageError::new(format!(
                    "failed to load runtime implementation catalog: {error}",
                ))
            })?;
        // Each independent application is only a candidate. Exact-baseline
        // coverage is enforced after the hybrid graph chooses a complete
        // non-overlapping representation/placement route.
        let request = crate::RuntimeSelectionRequest::from_vulkan_runtime_model(
            self,
            profiles_by_logical_device,
            execution,
            BTreeSet::new(),
        )?;
        self.hybrid_signal_representation_applications_from_catalog(
            &package_root,
            &catalog,
            &request,
        )
    }

    fn hybrid_signal_representation_applications_from_catalog(
        &self,
        package_root: &Path,
        catalog: &crate::RuntimeImplementationCatalog,
        request: &crate::RuntimeSelectionRequest,
    ) -> Result<Vec<VulkanRuntimeHybridRepresentationApplication>, VulkanResidentTokenModelPackageError>
    {
        let mut independent_request = request.clone();
        independent_request
            .exact_baseline_incompatible_instance_ids
            .clear();
        let reports = catalog
            .independent_application_selections(&independent_request)
            .map_err(|error| {
                VulkanResidentTokenModelPackageError::new(format!(
                    "failed to enumerate hybrid runtime representations: {error}",
                ))
            })?;
        let roles_by_instance = self
            .circuit_graph
            .components
            .iter()
            .map(|component| (component.component_id.as_str(), component.runtime_role))
            .collect::<BTreeMap<_, _>>();
        let mut applications = Vec::new();
        for report in reports {
            let [selected] = report.selected.as_slice() else {
                return Err(VulkanResidentTokenModelPackageError::new(
                    "independent runtime representation selection did not contain exactly one application",
                ));
            };
            let roles = selected
                .instance_ids
                .iter()
                .map(|instance_id| {
                    roles_by_instance.get(instance_id.as_str()).copied().ok_or_else(|| {
                        VulkanResidentTokenModelPackageError::new(format!(
                            "runtime representation references unknown mounted instance {instance_id:?}",
                        ))
                    })
                })
                .collect::<Result<Vec<_>, _>>()?;
            if roles.iter().all(|role| role.is_signal_processor()) {
                let semantic_contract_id =
                    vulkan_runtime_implementation_semantic_contract_id(selected)?;
                let runtime_model = self.clone().apply_runtime_implementation_catalog_selection(
                    package_root,
                    catalog,
                    report.clone(),
                )?;
                applications.push(VulkanRuntimeHybridRepresentationApplication {
                    runtime_model,
                    selection: report,
                    semantic_contract_id,
                });
            } else if roles.iter().any(|role| role.is_signal_processor()) {
                return Err(VulkanResidentTokenModelPackageError::new(
                    "runtime representation application crosses the signal-processor physical-island boundary",
                ));
            }
        }
        applications.sort_by(|left, right| {
            let left = &left.selection.selected[0];
            let right = &right.selection.selected[0];
            (left.instance_ids.as_slice(), left.implementation_id.as_str()).cmp(&(
                right.instance_ids.as_slice(),
                right.implementation_id.as_str(),
            ))
        });
        Ok(applications)
    }

    fn apply_runtime_implementation_catalog_selection(
        mut self,
        package_root: &Path,
        catalog: &crate::RuntimeImplementationCatalog,
        selection: crate::RuntimeImplementationSelectionReport,
    ) -> Result<Self, VulkanResidentTokenModelPackageError> {
        if selection.package_id != self.package.package_id {
            return Err(VulkanResidentTokenModelPackageError::new(
                "runtime implementation selection belongs to another package",
            ));
        }
        let expected_totals = (
            checked_selection_total(
                selection
                    .selected
                    .iter()
                    .map(|item| item.estimated_saved_ns),
                "estimated saved time",
            )?,
            checked_selection_total(
                selection
                    .selected
                    .iter()
                    .map(|item| item.conversion_ns),
                "conversion time",
            )?,
            checked_selection_total(
                selection
                    .selected
                    .iter()
                    .map(|item| item.conversion_bytes),
                "conversion bytes",
            )?,
            checked_selection_total(
                selection
                    .selected
                    .iter()
                    .map(|item| item.boundary_count),
                "representation boundary count",
            )?,
        );
        if expected_totals
            != (
                selection.total_estimated_saved_ns,
                selection.total_conversion_ns,
                selection.total_conversion_bytes,
                selection.total_boundary_count,
            )
        {
            return Err(VulkanResidentTokenModelPackageError::new(
                "runtime implementation selection totals are inconsistent",
            ));
        }
        let implementations = catalog
            .implementations
            .iter()
            .map(|loaded| {
                (
                    loaded.implementation.implementation_id.as_str(),
                    loaded,
                )
            })
            .collect::<BTreeMap<_, _>>();
        let mut mounted_instances = BTreeSet::new();
        let mut loaded_tensor_fragments = BTreeSet::new();
        let mut mounted_resident_derivation_sources = BTreeMap::new();

        let mut applications = selection
            .selected
            .iter()
            .map(|selected| {
                let loaded = implementations
                .get(selected.implementation_id.as_str())
                .ok_or_else(|| {
                    VulkanResidentTokenModelPackageError::new(format!(
                        "selected implementation {:?} is absent from the package",
                        selected.implementation_id
                    ))
                })?;
                validate_selected_implementation(selected, loaded)?;
                Ok((selected, *loaded))
            })
            .collect::<Result<Vec<_>, VulkanResidentTokenModelPackageError>>()?;
        for left in 0..applications.len() {
            for right in (left + 1)..applications.len() {
                if !runtime_applications_are_composable(
                    applications[left].0,
                    applications[left].1,
                    applications[right].0,
                    applications[right].1,
                ) {
                    return Err(VulkanResidentTokenModelPackageError::new(
                        "runtime implementation selection contains conflicting applications",
                    ));
                }
            }
        }
        applications.sort_by(|left, right| {
            (
                runtime_mount_plan_is_region_only(&left.1.mount_plan),
                left.0.instance_ids.as_slice(),
                left.0.implementation_id.as_str(),
            )
                .cmp(&(
                    runtime_mount_plan_is_region_only(&right.1.mount_plan),
                    right.0.instance_ids.as_slice(),
                    right.0.implementation_id.as_str(),
                ))
        });

        for (selected, loaded) in applications {
            mount_runtime_candidate_application(
                &mut self,
                package_root,
                RuntimeCandidateApplication {
                    candidate_root: &loaded.candidate_root,
                    mount_plan: &loaded.mount_plan,
                    application_id: &selected.implementation_id,
                    instance_ids: &selected.instance_ids,
                },
                RuntimeCandidateMountLedger {
                    mounted_instances: &mut mounted_instances,
                    loaded_tensor_fragments: &mut loaded_tensor_fragments,
                    mounted_resident_derivation_sources:
                        &mut mounted_resident_derivation_sources,
                },
            )?;
        }
        let source_roles = self
            .package
            .circuit_graph
            .components
            .iter()
            .map(|component| {
                (
                    component.component_id.as_str(),
                    component.runtime_role,
                )
            })
            .collect::<BTreeMap<_, _>>();
        let mut expected_exact_instances = self
            .runtime_graph
            .instances
            .iter()
            .filter(|instance| {
                source_roles
                    .get(instance.source_component_id.as_str())
                    .is_some_and(|role| role.is_runtime_implementation_target())
            })
            .map(|instance| instance.instance_id.clone())
            .filter(|instance_id| !mounted_instances.contains(instance_id))
            .collect::<Vec<_>>();
        expected_exact_instances.sort();
        if selection.exact_instance_ids != expected_exact_instances {
            return Err(VulkanResidentTokenModelPackageError::new(
                "runtime implementation selection exact coverage is inconsistent",
            ));
        }
        self.tensor_index_fragments.sort_by(|left, right| {
            left.index_path.cmp(&right.index_path)
        });
        let graph = self
            .circuit_graph
            .to_resolved_lowered_execution_graph(PathBuf::from("."))?;
        validate_component_executions_against_graph(
            &self.package.package_id,
            &self.component_executions,
            &graph,
        )?;
        validate_generation_execution_contract(&self.package, &self.circuit_graph)?;
        self.implementation_selection = Some(selection);
        self.load_runtime_tensor_index(package_root)?;
        Ok(self)
    }

    pub fn apply_staged_runtime_candidate(
        self,
        package_root: impl AsRef<Path>,
        candidate: &crate::RuntimeStagedCandidate,
    ) -> Result<Self, VulkanResidentTokenModelPackageError> {
        self.apply_staged_runtime_candidate_application(
            package_root,
            candidate,
            None,
        )
    }

    pub fn apply_staged_runtime_candidate_for_target(
        self,
        package_root: impl AsRef<Path>,
        candidate: &crate::RuntimeStagedCandidate,
        target_instance_id: &str,
    ) -> Result<Self, VulkanResidentTokenModelPackageError> {
        if target_instance_id.is_empty() {
            return Err(VulkanResidentTokenModelPackageError::new(
                "targeted staged candidate application requires a runtime instance",
            ));
        }
        self.apply_staged_runtime_candidate_application(
            package_root,
            candidate,
            Some(target_instance_id),
        )
    }

    fn apply_staged_runtime_candidate_application(
        mut self,
        package_root: impl AsRef<Path>,
        candidate: &crate::RuntimeStagedCandidate,
        target_instance_id: Option<&str>,
    ) -> Result<Self, VulkanResidentTokenModelPackageError> {
        let package_root = package_root
            .as_ref()
            .canonicalize()
            .map_err(|error| {
                VulkanResidentTokenModelPackageError::new(format!(
                    "failed to resolve runtime package root: {error}"
                ))
            })?;
        let verified = crate::RuntimeStagedCandidate::load(
            &package_root,
            &candidate.candidate_root,
        )
        .map_err(|error| {
            VulkanResidentTokenModelPackageError::new(format!(
                "failed to verify staged runtime candidate at mount time: {error}",
            ))
        })?;
        if &verified != candidate {
            return Err(VulkanResidentTokenModelPackageError::new(
                "staged runtime candidate changed after it was loaded",
            ));
        }
        let mut instances = self
            .runtime_graph
            .instances
            .iter()
            .map(|instance| crate::RuntimeSelectionInstance {
                instance_id: instance.instance_id.clone(),
                source_component_id: instance
                    .source_component_id
                    .clone(),
                logical_device_ids: Vec::new(),
            })
            .collect::<Vec<_>>();
        instances.sort_by(|left, right| {
            left.instance_id.cmp(&right.instance_id)
        });
        let mut edges = self
            .runtime_graph
            .effective_edges()
            .map_err(|error| {
                VulkanResidentTokenModelPackageError::new(
                    error.to_string(),
                )
            })?
            .into_iter()
            .map(|edge| crate::RuntimeSelectionEdge {
                source_instance_id: edge.source.component_id,
                destination_instance_id: edge.destination.component_id,
            })
            .collect::<Vec<_>>();
        edges.sort_by(|left, right| {
            (
                left.source_instance_id.as_str(),
                left.destination_instance_id.as_str(),
            )
                .cmp(&(
                    right.source_instance_id.as_str(),
                    right.destination_instance_id.as_str(),
                ))
        });
        let mut applications = crate::implementation_selection::independent_region_applications(
            &candidate.mount_plan.regions,
            &instances,
            &edges,
        );
        if let Some(target) = target_instance_id {
            applications.retain(|instance_ids| {
                instance_ids.iter().any(|instance_id| instance_id == target)
            });
        }
        if applications.is_empty() {
            return Err(VulkanResidentTokenModelPackageError::new(
                format!(
                    "staged candidate {:?} has no complete matching runtime region{} for source components {:?}",
                    candidate.candidate_id,
                    target_instance_id.map_or_else(
                        String::new,
                        |target| format!(" containing target instance {target:?}"),
                    ),
                    candidate.source_component_ids,
                ),
            ));
        }
        let mut mounted_instances = BTreeSet::new();
        let mut loaded_tensor_fragments = BTreeSet::new();
        let mut mounted_resident_derivation_sources = BTreeMap::new();
        for instance_ids in &applications {
            mount_runtime_candidate_application(
                &mut self,
                &package_root,
                RuntimeCandidateApplication {
                    candidate_root: &candidate.candidate_root,
                    mount_plan: &candidate.mount_plan,
                    application_id: &candidate.candidate_id,
                    instance_ids,
                },
                RuntimeCandidateMountLedger {
                    mounted_instances: &mut mounted_instances,
                    loaded_tensor_fragments: &mut loaded_tensor_fragments,
                    mounted_resident_derivation_sources:
                        &mut mounted_resident_derivation_sources,
                },
            )?;
        }
        self.tensor_index_fragments.sort_by(|left, right| {
            left.index_path.cmp(&right.index_path)
        });
        let graph = self
            .circuit_graph
            .to_resolved_lowered_execution_graph(PathBuf::from("."))?;
        validate_component_executions_against_graph(
            &self.package.package_id,
            &self.component_executions,
            &graph,
        )?;
        validate_generation_execution_contract(
            &self.package,
            &self.circuit_graph,
        )?;
        self.load_runtime_tensor_index(&package_root)?;
        Ok(self)
    }

    pub fn load_runtime_tensor_index(
        &self,
        package_root: impl AsRef<Path>,
    ) -> Result<TensorIndex, VulkanResidentTokenModelPackageError> {
        let package_root = package_root.as_ref();
        let tensor_index_path = resolve_resident_model_package_path(
            package_root,
            &self.package.tensor_index_path,
        );
        let mut tensor_index =
            TensorIndex::from_package_json_file(&tensor_index_path).map_err(
                |error| {
                    VulkanResidentTokenModelPackageError::new(format!(
                        "failed to load tensor index {tensor_index_path:?}: {error}"
                    ))
                },
            )?;
        for fragment in &self.tensor_index_fragments {
            let loaded = TensorIndex::from_package_fragment_json_file(
                &fragment.index_path,
                &fragment.candidate_root,
            )
            .map_err(|error| {
                VulkanResidentTokenModelPackageError::new(format!(
                    "failed to load runtime tensor-index fragment {:?}: {error}",
                    fragment.index_path
                ))
            })?;
            tensor_index.merge_fragment(loaded).map_err(|error| {
                VulkanResidentTokenModelPackageError::new(format!(
                    "failed to merge runtime tensor-index fragment {:?}: {error}",
                    fragment.index_path
                ))
            })?;
        }
        Ok(tensor_index)
    }
}

struct RuntimeCandidateApplication<'a> {
    candidate_root: &'a Path,
    mount_plan: &'a crate::RuntimeMountPlan,
    application_id: &'a str,
    instance_ids: &'a [String],
}

struct RuntimeCandidateMountLedger<'a> {
    mounted_instances: &'a mut BTreeSet<String>,
    loaded_tensor_fragments: &'a mut BTreeSet<PathBuf>,
    mounted_resident_derivation_sources:
        &'a mut BTreeMap<String, Vec<VulkanRuntimeParameterResidentDerivation>>,
}

fn runtime_mount_plan_is_region_only(
    mount_plan: &crate::RuntimeMountPlan,
) -> bool {
    mount_plan
        .regions
        .iter()
        .flat_map(|region| &region.replacements)
        .all(crate::RuntimeReplacement::is_component_region)
}

fn runtime_applications_are_composable(
    left: &crate::RuntimeSelectedImplementation,
    left_loaded: &crate::LoadedRuntimeImplementation,
    right: &crate::RuntimeSelectedImplementation,
    right_loaded: &crate::LoadedRuntimeImplementation,
) -> bool {
    let shared_instance = left
        .instance_ids
        .iter()
        .any(|instance| right.instance_ids.contains(instance));
    if !shared_instance {
        return true;
    }
    let overlapping_scope = left
        .scope_ids
        .iter()
        .any(|scope| right.scope_ids.contains(scope));
    !overlapping_scope
        && (runtime_mount_plan_is_region_only(&left_loaded.mount_plan)
            || runtime_mount_plan_is_region_only(&right_loaded.mount_plan))
}

fn mount_runtime_candidate_application(
    runtime_model: &mut VulkanResidentRuntimeModel,
    package_root: &Path,
    application: RuntimeCandidateApplication<'_>,
    mut ledger: RuntimeCandidateMountLedger<'_>,
) -> Result<(), VulkanResidentTokenModelPackageError> {
    let RuntimeCandidateApplication {
        candidate_root,
        mount_plan,
        application_id,
        instance_ids,
    } = application;
    let effective_edges = runtime_model
        .runtime_graph
        .effective_edges()
        .map_err(|error| {
            VulkanResidentTokenModelPackageError::new(error.to_string())
        })?;
    let selection_instances = runtime_model
        .runtime_graph
        .instances
        .iter()
        .map(|instance| crate::RuntimeSelectionInstance {
            instance_id: instance.instance_id.clone(),
            source_component_id: instance.source_component_id.clone(),
            logical_device_ids: Vec::new(),
        })
        .collect::<Vec<_>>();
    let selection_edges = effective_edges
        .iter()
        .map(|edge| crate::RuntimeSelectionEdge {
            source_instance_id: edge.source.component_id.clone(),
            destination_instance_id: edge.destination.component_id.clone(),
        })
        .collect::<Vec<_>>();
    let mut declared_instance_ids = instance_ids.to_vec();
    declared_instance_ids.sort();
    declared_instance_ids.dedup();
    let matching_regions = mount_plan
        .regions
        .iter()
        .enumerate()
        .filter_map(|(region_index, region)| {
            let applications = crate::implementation_selection::independent_region_applications(
                std::slice::from_ref(region),
                &selection_instances,
                &selection_edges,
            );
            applications
                .into_iter()
                .any(|candidate| candidate == declared_instance_ids)
                .then_some((region_index, region))
        })
        .collect::<Vec<_>>();
    if matching_regions.len() != 1 || declared_instance_ids.len() != instance_ids.len() {
        return Err(VulkanResidentTokenModelPackageError::new(format!(
            "runtime candidate application {application_id:?} does not identify exactly one complete semantic region",
        )));
    }
    let (region_index, region) = matching_regions[0];
    mount_runtime_candidate_region_application(
        runtime_model,
        package_root,
        candidate_root,
        region,
        &format!("{application_id}:region_{region_index}"),
        &declared_instance_ids,
        &effective_edges,
        &mut ledger,
    )?;
    for reference in &mount_plan.tensor_index_refs {
        let index_path = contained_candidate_artifact(
            candidate_root,
            reference,
            "runtime tensor-index fragment",
        )?;
        if ledger.loaded_tensor_fragments.insert(index_path.clone()) {
            runtime_model.tensor_index_fragments.push(
                VulkanRuntimeTensorIndexFragment {
                    index_path,
                    candidate_root: candidate_root.to_path_buf(),
                },
            );
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn mount_runtime_candidate_region_application(
    runtime_model: &mut VulkanResidentRuntimeModel,
    package_root: &Path,
    candidate_root: &Path,
    region: &crate::RuntimeMountRegion,
    application_id: &str,
    instance_ids: &[String],
    effective_edges: &[crate::stream_circuit::StreamCircuitGraphEdge],
    ledger: &mut RuntimeCandidateMountLedger<'_>,
) -> Result<(), VulkanResidentTokenModelPackageError> {
    let runtime_instances = runtime_model
        .runtime_graph
        .instances
        .iter()
        .map(|instance| {
            (instance.instance_id.clone(), instance.clone())
        })
        .collect::<BTreeMap<_, _>>();
    let source_components = runtime_model
        .package
        .circuit_graph
        .components
        .iter()
        .map(|component| {
            (component.component_id.clone(), component.clone())
        })
        .collect::<BTreeMap<_, _>>();
    let island_instances = instance_ids
        .iter()
        .map(|instance_id| {
            let instance = runtime_instances
                .get(instance_id)
                .cloned()
                .ok_or_else(|| {
                    VulkanResidentTokenModelPackageError::new(format!(
                        "runtime candidate application {application_id:?} references unknown instance {instance_id:?}",
                    ))
                })?;
            ledger.mounted_instances.insert(instance_id.clone());
            Ok((instance_id.as_str(), instance))
        })
        .collect::<Result<BTreeMap<_, _>, _>>()?;
    let island_ids = island_instances
        .keys()
        .copied()
        .collect::<BTreeSet<_>>();
    if !runtime_candidate_island_connected(&island_ids, &effective_edges) {
        return Err(VulkanResidentTokenModelPackageError::new(format!(
            "runtime candidate application {application_id:?} spans a disconnected runtime island",
        )));
    }

    for replacement in &region.replacements {
        let source_component_id = replacement.source_component_id();
        let matching_instances = island_instances
            .values()
            .filter(|instance| {
                instance.source_component_id
                    == source_component_id
            })
            .collect::<Vec<_>>();
        if matching_instances.len() != 1 {
            return Err(VulkanResidentTokenModelPackageError::new(format!(
                "runtime candidate application {application_id:?} does not map source component {:?} exactly once in island {:?}",
                source_component_id,
                instance_ids,
            )));
        }
        let source = source_components
            .get(source_component_id)
            .cloned()
            .ok_or_else(|| {
                VulkanResidentTokenModelPackageError::new(format!(
                    "runtime candidate application {application_id:?} references unknown source component {:?}",
                    source_component_id
                ))
            })?;
        let overlay_path = contained_candidate_artifact(
            candidate_root,
            replacement.overlay_ref(),
            "runtime overlay",
        )?;
        let bytes = fs::read(&overlay_path).map_err(|error| {
            VulkanResidentTokenModelPackageError::new(format!(
                "failed to read runtime overlay {overlay_path:?}: {error}"
            ))
        })?;
        match replacement {
            crate::RuntimeReplacement::Component { .. } => {
                let mut overlay: VulkanRuntimeComponentOverlay =
                    serde_json::from_slice(&bytes).map_err(|error| {
                        VulkanResidentTokenModelPackageError::new(format!(
                            "invalid runtime component overlay {overlay_path:?}: {error}"
                        ))
                    })?;
                validate_runtime_component_overlay(
                    &overlay,
                    &source,
                    matching_instances[0].instance_id.as_str(),
                    &island_ids,
                    &effective_edges,
                    &runtime_model.runtime_graph.boundary,
                )?;
                let source_execution = runtime_model
                    .component_executions
                    .iter()
                    .find(|execution| {
                        execution.component_id
                            == matching_instances[0].instance_id
                    })
                    .cloned()
                    .ok_or_else(|| {
                        VulkanResidentTokenModelPackageError::new(format!(
                            "runtime candidate application {application_id:?} cannot find source execution for instance {:?}",
                            matching_instances[0].instance_id,
                        ))
                    })?;
                rebase_overlay_shader_paths(
                    &mut overlay.execution,
                    &source_execution,
                    package_root,
                    candidate_root,
                )?;
                if !overlay.resident_derivations.is_empty() {
                    match ledger
                        .mounted_resident_derivation_sources
                        .get(source_component_id)
                    {
                        Some(mounted) if mounted == &overlay.resident_derivations => {}
                        Some(_) => {
                            return Err(VulkanResidentTokenModelPackageError::new(format!(
                                "runtime instances of source component {source_component_id:?} selected conflicting resident representations",
                            )));
                        }
                        None => {
                            apply_runtime_component_resident_derivations(
                                runtime_model,
                                package_root,
                                source_component_id,
                                &overlay.execution,
                                &overlay.resident_derivations,
                            )?;
                            ledger.mounted_resident_derivation_sources.insert(
                                source_component_id.to_string(),
                                overlay.resident_derivations.clone(),
                            );
                        }
                    }
                }
                mount_runtime_component_overlay(
                    runtime_model,
                    matching_instances[0].instance_id.as_str(),
                    overlay,
                )?;
            }
            crate::RuntimeReplacement::ComponentRegion { .. } => {
                let mut overlay: VulkanRuntimeComponentRegionOverlay =
                    serde_json::from_slice(&bytes).map_err(|error| {
                        VulkanResidentTokenModelPackageError::new(format!(
                            "invalid runtime component-region overlay {overlay_path:?}: {error}"
                        ))
                    })?;
                validate_runtime_component_region_overlay(
                    &overlay,
                    &source,
                    runtime_model
                        .package
                        .component_executions
                        .iter()
                        .find(|execution| execution.component_id == source_component_id)
                        .ok_or_else(|| {
                            VulkanResidentTokenModelPackageError::new(format!(
                                "runtime component-region overlay cannot find source execution for {source_component_id:?}"
                            ))
                        })?,
                )?;
                rebase_component_region_shader_paths(
                    &mut overlay,
                    package_root,
                    candidate_root,
                )?;
                mount_runtime_component_region_overlay(
                    runtime_model,
                    matching_instances[0].instance_id.as_str(),
                    overlay,
                    package_root,
                )?;
            }
            crate::RuntimeReplacement::OutputTransducer { .. } => {
                let mut overlay: VulkanRuntimeOutputTransducerOverlay =
                    serde_json::from_slice(&bytes).map_err(|error| {
                        VulkanResidentTokenModelPackageError::new(format!(
                            "invalid runtime output-transducer overlay {overlay_path:?}: {error}"
                        ))
                    })?;
                validate_runtime_output_transducer_overlay(
                    runtime_model,
                    &overlay,
                    &source,
                    matching_instances[0].instance_id.as_str(),
                    &island_ids,
                    &effective_edges,
                )?;
                rebase_output_transducer_overlay_shader_paths(
                    &mut overlay,
                    &runtime_model.package,
                    package_root,
                    candidate_root,
                )?;
                mount_runtime_output_transducer_overlay(
                    runtime_model,
                    matching_instances[0].instance_id.as_str(),
                    overlay,
                )?;
            }
        }
    }
    Ok(())
}

fn runtime_candidate_island_connected(
    island_ids: &BTreeSet<&str>,
    edges: &[crate::stream_circuit::StreamCircuitGraphEdge],
) -> bool {
    let Some(first) = island_ids.first().copied() else {
        return false;
    };
    let mut reached = BTreeSet::from([first]);
    let mut pending = VecDeque::from([first]);
    while let Some(current) = pending.pop_front() {
        for edge in edges {
            let neighbor = if edge.source.component_id == current {
                Some(edge.destination.component_id.as_str())
            } else if edge.destination.component_id == current {
                Some(edge.source.component_id.as_str())
            } else {
                None
            };
            if let Some(neighbor) = neighbor
                && island_ids.contains(neighbor)
                && reached.insert(neighbor)
            {
                pending.push_back(neighbor);
            }
        }
    }
    reached.len() == island_ids.len()
}

fn checked_selection_total(
    mut values: impl Iterator<Item = u64>,
    label: &str,
) -> Result<u64, VulkanResidentTokenModelPackageError> {
    values.try_fold(0u64, |total, value| {
        total.checked_add(value).ok_or_else(|| {
            VulkanResidentTokenModelPackageError::new(format!(
                "runtime implementation selection {label} overflows"
            ))
        })
    })
}

fn validate_selected_implementation(
    selected: &crate::RuntimeSelectedImplementation,
    loaded: &crate::LoadedRuntimeImplementation,
) -> Result<(), VulkanResidentTokenModelPackageError> {
    if selected.candidate_id != loaded.implementation.candidate_id
        || selected.scope_ids != loaded.implementation.scope_ids
        || selected.source_contract_digests
            != loaded.implementation.source_contract_digests
        || selected.predicate != loaded.implementation.runtime_predicate
        || selected.mount_adapter_id != loaded.mount_plan.adapter_id
        || selected.representation != loaded.implementation.representation
        || selected.provenance != loaded.implementation.provenance
        || selected.benchmark_id
            != loaded.implementation.comparison.benchmark_id
        || selected.validation_id
            != loaded.implementation.comparison.validation_id
        || selected.validation_status != "passed"
        || selected.decision_reason != loaded.implementation.decision_reason
        || loaded.mount_plan.adapter_id
            != crate::VULKAN_STREAM_CIRCUIT_OVERLAY_ADAPTER
    {
        return Err(VulkanResidentTokenModelPackageError::new(format!(
            "selected implementation {:?} does not match its verified package contract",
            selected.implementation_id
        )));
    }
    Ok(())
}

fn contained_candidate_artifact(
    candidate_root: &Path,
    reference: &str,
    label: &str,
) -> Result<PathBuf, VulkanResidentTokenModelPackageError> {
    contained_regular_artifact(
        candidate_root,
        reference,
        label,
        "candidate bundle",
    )
}

fn contained_regular_artifact(
    root: &Path,
    reference: &str,
    label: &str,
    root_label: &str,
) -> Result<PathBuf, VulkanResidentTokenModelPackageError> {
    let relative = Path::new(reference);
    if reference.is_empty()
        || relative.is_absolute()
        || relative.components().any(|component| {
            !matches!(component, std::path::Component::Normal(_))
        })
        || relative.to_string_lossy() != reference
    {
        return Err(VulkanResidentTokenModelPackageError::new(format!(
            "{label} reference is not a canonical {root_label}-relative path"
        )));
    }
    let mut lexical = root.to_path_buf();
    for component in relative.components() {
        let std::path::Component::Normal(component) = component else {
            unreachable!("relative components were validated above");
        };
        lexical.push(component);
        let metadata = fs::symlink_metadata(&lexical).map_err(|error| {
            VulkanResidentTokenModelPackageError::new(format!(
                "{label} is missing or unreadable: {error}"
            ))
        })?;
        if metadata.file_type().is_symlink() {
            return Err(VulkanResidentTokenModelPackageError::new(
                format!("{label} crosses a symbolic link"),
            ));
        }
    }
    let path = lexical.canonicalize().map_err(|error| {
        VulkanResidentTokenModelPackageError::new(format!(
            "{label} is missing or unreadable: {error}"
        ))
    })?;
    if !path.starts_with(root) || !path.is_file() {
        return Err(VulkanResidentTokenModelPackageError::new(format!(
            "{label} escapes its {root_label}"
        )));
    }
    Ok(path)
}

fn validate_runtime_component_overlay(
    overlay: &VulkanRuntimeComponentOverlay,
    source: &VulkanResidentPackageComponentCircuit,
    runtime_instance_id: &str,
    island_instance_ids: &BTreeSet<&str>,
    effective_edges: &[crate::stream_circuit::StreamCircuitGraphEdge],
    graph_boundary: &StreamCircuitGraphBoundary,
) -> Result<(), VulkanResidentTokenModelPackageError> {
    if overlay.schema != crate::VULKAN_COMPONENT_OVERLAY_SCHEMA
        || overlay.execution.component_id != source.component_id
        || overlay.execution.operator_type != source.operator_type
    {
        return Err(VulkanResidentTokenModelPackageError::new(format!(
            "runtime component overlay for {:?} changes its logical source identity",
            source.component_id
        )));
    }
    validate_runtime_overlay_component(
        &overlay.source_component_id,
        &overlay.component,
        source,
        runtime_instance_id,
        island_instance_ids,
        effective_edges,
        graph_boundary,
    )
}

#[allow(clippy::too_many_arguments)]
fn validate_runtime_overlay_component(
    overlay_source_component_id: &str,
    overlay_component: &VulkanResidentPackageComponentCircuit,
    source: &VulkanResidentPackageComponentCircuit,
    runtime_instance_id: &str,
    island_instance_ids: &BTreeSet<&str>,
    effective_edges: &[crate::stream_circuit::StreamCircuitGraphEdge],
    graph_boundary: &StreamCircuitGraphBoundary,
) -> Result<(), VulkanResidentTokenModelPackageError> {
    if overlay_source_component_id != source.component_id
        || overlay_component.component_id != source.component_id
        || overlay_component.circuit.source.component_id
            != source.component_id
        || overlay_component.operator_type != source.operator_type
        || overlay_component.runtime_role != source.runtime_role
        || overlay_component.circuit.runtime_role != source.runtime_role
    {
        return Err(VulkanResidentTokenModelPackageError::new(format!(
            "runtime overlay for {:?} changes its logical source identity",
            source.component_id
        )));
    }
    let source_inputs = source
        .circuit
        .boundary
        .inputs
        .iter()
        .map(|port| (port.id.as_str(), port))
        .collect::<BTreeMap<_, _>>();
    let overlay_inputs = overlay_component
        .circuit
        .boundary
        .inputs
        .iter()
        .map(|port| (port.id.as_str(), port))
        .collect::<BTreeMap<_, _>>();
    let source_outputs = source
        .circuit
        .boundary
        .outputs
        .iter()
        .map(|port| (port.id.as_str(), port))
        .collect::<BTreeMap<_, _>>();
    let overlay_outputs = overlay_component
        .circuit
        .boundary
        .outputs
        .iter()
        .map(|port| (port.id.as_str(), port))
        .collect::<BTreeMap<_, _>>();
    if source_inputs.keys().ne(overlay_inputs.keys())
        || source_outputs.keys().ne(overlay_outputs.keys())
    {
        return Err(VulkanResidentTokenModelPackageError::new(format!(
            "runtime component overlay for {:?} changes its logical port identities",
            source.component_id
        )));
    }
    let external_inputs = effective_edges
        .iter()
        .filter(|edge| {
            edge.destination.component_id == runtime_instance_id
                && !island_instance_ids
                    .contains(edge.source.component_id.as_str())
        })
        .map(|edge| edge.destination.port_id.as_str())
        .chain(
            graph_boundary
                .external_inputs
                .iter()
                .filter(|port| {
                    port.endpoint.component_id == runtime_instance_id
                })
                .map(|port| port.endpoint.port_id.as_str()),
        )
        .collect::<BTreeSet<_>>();
    let external_outputs = effective_edges
        .iter()
        .filter(|edge| {
            edge.source.component_id == runtime_instance_id
                && !island_instance_ids
                    .contains(edge.destination.component_id.as_str())
        })
        .map(|edge| edge.source.port_id.as_str())
        .chain(
            graph_boundary
                .public_outputs
                .iter()
                .filter(|port| {
                    port.endpoint.component_id == runtime_instance_id
                })
                .map(|port| port.endpoint.port_id.as_str()),
        )
        .collect::<BTreeSet<_>>();
    for port_id in external_inputs {
        if source_inputs.get(port_id) != overlay_inputs.get(port_id) {
            return Err(VulkanResidentTokenModelPackageError::new(format!(
                "runtime component overlay for {:?} changes external input port {port_id:?}",
                source.component_id
            )));
        }
    }
    for port_id in external_outputs {
        if source_outputs.get(port_id) != overlay_outputs.get(port_id) {
            return Err(VulkanResidentTokenModelPackageError::new(format!(
                "runtime component overlay for {:?} changes external output port {port_id:?}",
                source.component_id
            )));
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn validate_runtime_output_transducer_overlay(
    runtime_model: &VulkanResidentRuntimeModel,
    overlay: &VulkanRuntimeOutputTransducerOverlay,
    source: &VulkanResidentPackageComponentCircuit,
    runtime_instance_id: &str,
    island_instance_ids: &BTreeSet<&str>,
    effective_edges: &[crate::stream_circuit::StreamCircuitGraphEdge],
) -> Result<(), VulkanResidentTokenModelPackageError> {
    if overlay.schema != crate::VULKAN_OUTPUT_TRANSDUCER_OVERLAY_SCHEMA
        || source.runtime_role != CircuitRuntimeRole::OutputTransducer
    {
        return Err(VulkanResidentTokenModelPackageError::new(format!(
            "runtime output-transducer overlay for {:?} has an incompatible identity",
            source.component_id
        )));
    }
    validate_runtime_overlay_component(
        &overlay.source_component_id,
        &overlay.component,
        source,
        runtime_instance_id,
        island_instance_ids,
        effective_edges,
        &runtime_model.runtime_graph.boundary,
    )?;
    validate_output_transducer_logical_contract(
        &runtime_model.package.output_transducer,
        &overlay.output_transducer,
    )?;

    let source_drafts = runtime_model
        .package
        .speculative_decoders
        .iter()
        .filter_map(|decoder| {
            decoder
                .dedicated_output_transducer()
                .map(|output| (decoder.id.as_str(), output))
        })
        .collect::<BTreeMap<_, _>>();
    let overlay_draft_ids = overlay
        .speculative_output_transducers
        .iter()
        .map(|decoder| decoder.decoder_id.as_str())
        .collect::<Vec<_>>();
    if overlay_draft_ids.windows(2).any(|pair| pair[0] >= pair[1])
        || overlay_draft_ids.iter().copied().collect::<BTreeSet<_>>()
            != source_drafts.keys().copied().collect::<BTreeSet<_>>()
    {
        return Err(VulkanResidentTokenModelPackageError::new(
            "runtime output-transducer overlay must cover each speculative decoder exactly once",
        ));
    }
    for draft in &overlay.speculative_output_transducers {
        validate_draft_output_transducer_logical_contract(
            source_drafts[draft.decoder_id.as_str()],
            &draft.output_transducer,
        )?;
    }
    Ok(())
}

fn validate_output_transducer_logical_contract(
    source: &VulkanResidentOutputTransducerPackageSpec,
    overlay: &VulkanResidentOutputTransducerPackageSpec,
) -> Result<(), VulkanResidentTokenModelPackageError> {
    let source_spec = &source.spec;
    let overlay_spec = &overlay.spec;
    if overlay_spec.transducer_id != source_spec.transducer_id
        || overlay_spec.input_signal_id != source_spec.input_signal_id
        || overlay_spec.node_ids != source_spec.node_ids
        || overlay_spec.norm_parameter_shape != source_spec.norm_parameter_shape
        || overlay_spec.input_frame_byte_capacity != source_spec.input_frame_byte_capacity
        || overlay_spec.normalized_frame_byte_capacity
            != source_spec.normalized_frame_byte_capacity
        || overlay_spec.logits_byte_capacity != source_spec.logits_byte_capacity
    {
        return Err(VulkanResidentTokenModelPackageError::new(
            "runtime output-transducer overlay changes its logical signal or tensor geometry",
        ));
    }
    Ok(())
}

fn validate_draft_output_transducer_logical_contract(
    source: &VulkanResidentDraftOutputTransducerPackageSpec,
    overlay: &VulkanResidentDraftOutputTransducerPackageSpec,
) -> Result<(), VulkanResidentTokenModelPackageError> {
    if overlay.component_id != source.component_id
        || overlay.input_signal_id != source.input_signal_id
        || overlay.hidden_signal_id != source.hidden_signal_id
        || overlay.logits_signal_id != source.logits_signal_id
        || overlay.norm_parameter_shape != source.norm_parameter_shape
        || overlay.input_frame_byte_capacity != source.input_frame_byte_capacity
        || overlay.output_hidden_byte_capacity != source.output_hidden_byte_capacity
        || overlay.logits_byte_capacity != source.logits_byte_capacity
        || overlay.vocabulary_size != source.vocabulary_size
        || overlay.hidden_size != source.hidden_size
    {
        return Err(VulkanResidentTokenModelPackageError::new(
            "runtime draft output-transducer overlay changes its logical signal or tensor geometry",
        ));
    }
    Ok(())
}

fn rebase_overlay_shader_paths(
    execution: &mut VulkanResidentComponentExecutionSpec,
    source_execution: &VulkanResidentComponentExecutionSpec,
    package_root: &Path,
    candidate_root: &Path,
) -> Result<(), VulkanResidentTokenModelPackageError> {
    for kernel in &mut execution.kernels {
        let source_kernel = source_execution
            .kernels
            .iter()
            .find(|source| source.node_id == kernel.node_id);
        kernel.shader_path = rebase_overlay_shader_path(
            &kernel.shader_path,
            source_kernel.map(|source| source.shader_path.as_str()),
            package_root,
            candidate_root,
            "runtime implementation shader",
        )?;
        for implementation in &mut kernel.batch_implementations {
            let source_implementation = source_kernel
                .map(|source| {
                    source_batch_implementation_for_overlay(
                        source,
                        implementation,
                    )
                })
                .transpose()?
                .flatten();
            for (stage_index, stage) in
                implementation.stages.iter_mut().enumerate()
            {
                let source_path = source_implementation
                    .and_then(|source| source.stages.get(stage_index))
                    .map(|source| source.shader_path.as_str());
                stage.shader_path = rebase_overlay_shader_path(
                    &stage.shader_path,
                    source_path,
                    package_root,
                    candidate_root,
                    "runtime implementation batch shader",
                )?;
            }
        }
    }
    Ok(())
}

fn source_batch_implementation_for_overlay<'a>(
    source_kernel: &'a VulkanResidentComponentKernelSpec,
    overlay: &VulkanResidentComponentBatchImplementationSpec,
) -> Result<
    Option<&'a VulkanResidentComponentBatchImplementationSpec>,
    VulkanResidentTokenModelPackageError,
> {
    if let Some(exact) = source_kernel
        .batch_implementations
        .iter()
        .find(|source| *source == overlay)
    {
        return Ok(Some(exact));
    }
    let matches = source_kernel
        .batch_implementations
        .iter()
        .filter(|source| {
            source.execution_domain == overlay.execution_domain
                && source.lane_tile_width == overlay.lane_tile_width
                && source.selection_priority == overlay.selection_priority
                && source.independent_candidate_compatible
                    == overlay.independent_candidate_compatible
                && source.causal_sequence_compatible
                    == overlay.causal_sequence_compatible
                && source.parallel_block_compatible
                    == overlay.parallel_block_compatible
                && source.device_requirements == overlay.device_requirements
                && source.stages.len() == overlay.stages.len()
                && source.stages.iter().zip(&overlay.stages).all(
                    |(source_stage, overlay_stage)| {
                        source_stage.descriptor_bindings
                            == overlay_stage.descriptor_bindings
                            && source_stage.state_snapshot_binding
                                == overlay_stage.state_snapshot_binding
                            && source_stage.state_snapshot_source_binding
                                == overlay_stage.state_snapshot_source_binding
                            && source_stage.control == overlay_stage.control
                    },
                )
        })
        .collect::<Vec<_>>();
    match matches.as_slice() {
        [] => Ok(None),
        [source] => Ok(Some(*source)),
        _ => Err(VulkanResidentTokenModelPackageError::new(format!(
            "runtime overlay batch implementation for {}.{} has ambiguous source identity; physical shader and dispatch geometry are not identity fields",
            source_kernel.node_id,
            overlay.lane_tile_width,
        ))),
    }
}

fn rebase_output_transducer_overlay_shader_paths(
    overlay: &mut VulkanRuntimeOutputTransducerOverlay,
    source: &VulkanResidentModelPackageManifest,
    package_root: &Path,
    candidate_root: &Path,
) -> Result<(), VulkanResidentTokenModelPackageError> {
    let target_paths = [
        (
            &mut overlay.output_transducer.embedding_norm_shader_path,
            source.output_transducer.embedding_norm_shader_path.as_str(),
        ),
        (
            &mut overlay.output_transducer.embedding_norm_batch_shader_path,
            source
                .output_transducer
                .embedding_norm_batch_shader_path
                .as_str(),
        ),
        (
            &mut overlay.output_transducer.projection_shader_path,
            source.output_transducer.projection_shader_path.as_str(),
        ),
        (
            &mut overlay.output_transducer.projection_batch_shader_path,
            source
                .output_transducer
                .projection_batch_shader_path
                .as_str(),
        ),
    ];
    for (overlay_path, source_path) in target_paths {
        *overlay_path = rebase_overlay_shader_path(
            overlay_path,
            Some(source_path),
            package_root,
            candidate_root,
            "runtime output-transducer shader",
        )?;
    }

    let source_drafts = source
        .speculative_decoders
        .iter()
        .filter_map(|decoder| {
            decoder
                .dedicated_output_transducer()
                .map(|output| (decoder.id.as_str(), output))
        })
        .collect::<BTreeMap<_, _>>();
    for draft in &mut overlay.speculative_output_transducers {
        let source_draft = source_drafts[draft.decoder_id.as_str()];
        draft.output_transducer.norm_shader_path = rebase_overlay_shader_path(
            &draft.output_transducer.norm_shader_path,
            Some(&source_draft.norm_shader_path),
            package_root,
            candidate_root,
            "runtime draft output-transducer norm shader",
        )?;
        draft.output_transducer.projection_shader_path = rebase_overlay_shader_path(
            &draft.output_transducer.projection_shader_path,
            Some(&source_draft.projection_shader_path),
            package_root,
            candidate_root,
            "runtime draft output-transducer projection shader",
        )?;
    }
    Ok(())
}

fn rebase_overlay_shader_path(
    overlay_reference: &str,
    source_reference: Option<&str>,
    package_root: &Path,
    candidate_root: &Path,
    label: &str,
) -> Result<String, VulkanResidentTokenModelPackageError> {
    let path = if source_reference == Some(overlay_reference) {
        contained_package_artifact(
            package_root,
            overlay_reference,
            label,
        )?
    } else {
        contained_candidate_artifact(
            candidate_root,
            overlay_reference,
            label,
        )?
    };
    Ok(path.to_string_lossy().into_owned())
}

fn contained_package_artifact(
    package_root: &Path,
    reference: &str,
    label: &str,
) -> Result<PathBuf, VulkanResidentTokenModelPackageError> {
    contained_regular_artifact(
        package_root,
        reference,
        label,
        "source package",
    )
}

fn mount_runtime_component_overlay(
    runtime_model: &mut VulkanResidentRuntimeModel,
    runtime_instance_id: &str,
    mut overlay: VulkanRuntimeComponentOverlay,
) -> Result<(), VulkanResidentTokenModelPackageError> {
    package::attach_component_kernel_runtime_contracts(
        &mut overlay.component,
        &overlay.execution,
    )?;
    overlay.component.component_id = runtime_instance_id.to_string();
    overlay.component.circuit.source.component_id =
        runtime_instance_id.to_string();
    overlay.execution.component_id = runtime_instance_id.to_string();
    let instance = runtime_model
        .runtime_graph
        .instances
        .iter()
        .find(|instance| instance.instance_id == runtime_instance_id)
        .ok_or_else(|| {
            VulkanResidentTokenModelPackageError::new(format!(
                "runtime implementation cannot find graph instance {runtime_instance_id:?}"
            ))
        })?;
    package::apply_runtime_graph_state_policy(
        &mut overlay.component,
        &runtime_model.runtime_graph,
        instance,
    );
    let component = runtime_model
        .circuit_graph
        .components
        .iter_mut()
        .find(|component| component.component_id == runtime_instance_id)
        .ok_or_else(|| {
            VulkanResidentTokenModelPackageError::new(format!(
                "runtime implementation cannot find component {runtime_instance_id:?}"
            ))
        })?;
    *component = overlay.component;
    let execution = runtime_model
        .component_executions
        .iter_mut()
        .find(|execution| execution.component_id == runtime_instance_id)
        .ok_or_else(|| {
            VulkanResidentTokenModelPackageError::new(format!(
                "runtime implementation cannot find execution for {runtime_instance_id:?}"
            ))
        })?;
    *execution = overlay.execution;
    Ok(())
}

fn mount_runtime_output_transducer_overlay(
    runtime_model: &mut VulkanResidentRuntimeModel,
    runtime_instance_id: &str,
    mut overlay: VulkanRuntimeOutputTransducerOverlay,
) -> Result<(), VulkanResidentTokenModelPackageError> {
    overlay.component.component_id = runtime_instance_id.to_string();
    overlay.component.circuit.source.component_id =
        runtime_instance_id.to_string();
    overlay.output_transducer.spec.transducer_id =
        runtime_instance_id.to_string();
    let component = runtime_model
        .circuit_graph
        .components
        .iter_mut()
        .find(|component| component.component_id == runtime_instance_id)
        .ok_or_else(|| {
            VulkanResidentTokenModelPackageError::new(format!(
                "runtime implementation cannot find output transducer {runtime_instance_id:?}"
            ))
        })?;
    *component = overlay.component;
    runtime_model.package.output_transducer = overlay.output_transducer;

    let speculative_outputs = overlay
        .speculative_output_transducers
        .into_iter()
        .map(|draft| (draft.decoder_id, draft.output_transducer))
        .collect::<BTreeMap<_, _>>();
    for decoder in &mut runtime_model.package.speculative_decoders {
        if decoder.output_transducer.is_some() {
            decoder.output_transducer = Some(
                speculative_outputs
                    .get(&decoder.id)
                    .expect("validated output overlay must cover every dedicated speculative output")
                    .clone(),
            );
        }
    }
    Ok(())
}
