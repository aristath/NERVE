fn compiled_store_process_file_descriptor_count() -> usize {
    fs::read_dir("/proc/self/fd").unwrap().count()
}

fn compiled_store_worker_thread_count() -> usize {
    fs::read_dir("/proc/self/task")
        .unwrap()
        .map(|entry| entry.unwrap().path().join("comm"))
        .filter_map(|path| fs::read_to_string(path).ok())
        .filter(|name| name.trim().starts_with("nerve-resource"))
        .count()
}

#[test]
fn compiled_resource_backing_workers_follow_wave_width_without_oversubscribing_cores() {
    assert_eq!(
        compiled_resource_backing_worker_count_for_parallelism(8, 32),
        8
    );
    assert_eq!(
        compiled_resource_backing_worker_count_for_parallelism(32, 8),
        8
    );
    assert_eq!(
        compiled_resource_backing_worker_count_for_parallelism(1, 32),
        1
    );
    assert_eq!(
        compiled_resource_backing_worker_count_for_parallelism(0, 0),
        1
    );
}

#[test]
fn compiled_resource_device_store_loads_reuses_and_retires_stable_resources() {
    let device =
        selected_test_vulkan_device().expect("selected Vulkan test device must open");
    let root =
        crate::test_support::TempDir::new("compiled_resource_device_store");
    let weight_bytes = b"abcdefghABCDEFGH";
    fs::write(root.path().join("weights.bin"), weight_bytes).unwrap();
    let mut digest_table = Vec::new();
    digest_table.extend_from_slice(&Sha256::digest(&weight_bytes[..8]));
    digest_table.extend_from_slice(&Sha256::digest(&weight_bytes[8..]));
    fs::write(root.path().join("digests.bin"), &digest_table).unwrap();

    let content_id = |byte: char| {
        format!("sha256:{}", byte.to_string().repeat(64))
    };
    let template_id = content_id('1');
    let member_seed = content_id('2');
    let selector_id = content_id('3');
    let mut contract = CompiledResourceResidencyContract {
        schema: COMPILED_RESOURCE_RESIDENCY_SCHEMA.to_string(),
        identity_algorithm: RESOURCE_IDENTITY_ALGORITHM.to_string(),
        state_machine_schema:
            RESOURCE_RESIDENCY_STATE_MACHINE_SCHEMA.to_string(),
        supported_policies: vec![
            ResourceResidencyPolicy::DemandRetained,
            ResourceResidencyPolicy::Eager,
        ],
        resources: Vec::new(),
        atomic_groups: Vec::new(),
        partition_templates: vec![CompiledPartitionTemplate {
            id: template_id.clone(),
            partition_count: 2,
            lifetime: CompiledResourceLifetime::Dynamic,
            group_identity_seed: content_id('4'),
            member_templates: vec![CompiledPartitionMemberTemplate {
                resource_identity_seed: member_seed.clone(),
                range_templates: vec![CompiledResourceRangeTemplate {
                    artifact_path: "weights.bin".to_string(),
                    base_byte_offset: 0,
                    stride_bytes: 8,
                    byte_count: 8,
                    alignment_bytes: 8,
                    integrity: CompiledResourceRangeIntegrityTemplate {
                        algorithm: "sha256_table".to_string(),
                        digest_table_path: "digests.bin".to_string(),
                        digest_table_byte_offset: 0,
                        digest_stride_bytes: 32,
                        table_sha256: Sha256::digest(&digest_table)
                            .iter()
                            .map(|byte| format!("{byte:02x}"))
                            .collect(),
                    },
                }],
                compatibility: CompiledResourceCompatibility {
                    device_api: "vulkan".to_string(),
                    storage_class: "storage_buffer".to_string(),
                    read_only: true,
                    required_features: vec![
                        "buffer_device_address".to_string(),
                    ],
                },
            }],
            dependencies: Vec::new(),
        }],
        bindings: vec![CompiledResourceBinding {
            execution_scope: "target".to_string(),
            component_id: "component".to_string(),
            node_id: "selected_compute".to_string(),
            parameter_id: "bank".to_string(),
            mapping: CompiledResourceBindingMapping::PartitionTemplateMember {
                partition_template_id: template_id.clone(),
                resource_identity_seed: member_seed,
            },
        }],
        selectors: vec![CompiledResourceSelector {
            id: selector_id.clone(),
            execution_scope: "target".to_string(),
            component_id: "component".to_string(),
            node_id: "choose".to_string(),
            domain_id: "resources".to_string(),
            resource_count: 2,
            selection_signal: "selected".to_string(),
            encoding: CompiledResourceSelectionEncoding {
                element_type: CompiledResourceSelectionElementType::U32,
                selection_count_per_activation: 2,
                index_shift: 0,
                index_mask: 0xffff,
            },
            mapping: CompiledResourceSelectorMapping::PartitionTemplate {
                partition_template_id: template_id,
            },
        }],
        checkpoints: Vec::new(),
    };
    let alias_selector_id = content_id('5');
    let mut alias_selector = contract.selectors[0].clone();
    alias_selector.id = alias_selector_id.clone();
    alias_selector.component_id = "component_repeat".to_string();
    contract.selectors.push(alias_selector);
    let unowned_selector_id = content_id('6');
    let mut unowned_selector = contract.selectors[0].clone();
    unowned_selector.id = unowned_selector_id.clone();
    unowned_selector.component_id = "other_component".to_string();
    contract.selectors.push(unowned_selector);
    let inspection = contract.inspection_report().unwrap();
    assert_eq!(
        inspection.supported_policies,
        ["demand-retained", "eager"]
    );
    assert_eq!(inspection.always_resident.unit_count, 0);
    assert_eq!(
        inspection.dynamically_addressable.unit_count,
        2
    );
    assert_eq!(
        inspection.dynamically_addressable.resource_count,
        2
    );
    assert_eq!(
        inspection.dynamically_addressable.maximum_payload_bytes,
        16
    );
    assert_eq!(inspection.scopes.len(), 1);
    assert_eq!(inspection.scopes[0].component_count, 3);
    assert_eq!(inspection.scopes[0].selector_count, 3);
    assert_eq!(inspection.scopes[0].addressable_unit_count, 2);
    let contract = Arc::new(contract);
    let layout = Arc::new(
        VulkanCompiledResourceAddressLayout::from_contract(&contract)
            .unwrap(),
    );
    let capacity_error = match VulkanCompiledResourceDeviceStore::new(
        &device,
        "amd-test",
        device.physical_device_id(),
        vec!["gpu0".to_string()],
        root.path(),
        Arc::clone(&contract),
        Arc::clone(&layout),
        BTreeSet::from([
            selector_id.clone(),
            alias_selector_id.clone(),
        ]),
        4096,
        4096,
        1024,
        1,
        128,
        64,
        layout.address_table_byte_count().unwrap(),
    ) {
        Ok(_) => panic!("physical allocation padding was not admitted"),
        Err(error) => error,
    };
    assert!(
        capacity_error
            .to_string()
            .contains("physical allocation bytes")
    );
    let store = VulkanCompiledResourceDeviceStore::new(
        &device,
        "amd-test",
        device.physical_device_id(),
        vec!["gpu0".to_string()],
        root.path(),
        Arc::clone(&contract),
        Arc::clone(&layout),
        BTreeSet::from([
            selector_id.clone(),
            alias_selector_id.clone(),
        ]),
        4096,
        8192,
        1024,
        1,
        128,
        64,
        layout.address_table_byte_count().unwrap(),
    )
    .unwrap();
    let buffers = store
        .dynamic_buffers_for_components(
            &device,
            "target",
            &BTreeSet::from(["component".to_string()]),
        )
        .unwrap();
    let owner = DeviceResourceResidencyOwnerId::new("graph").unwrap();
    store.mark_mount_complete().unwrap();
    let initial = store.residency_report().unwrap();
    assert_eq!(initial.initial_payload_bytes, 0);
    assert_eq!(initial.current_payload_bytes, 0);
    assert_eq!(initial.initial_resident_unit_count, 0);
    assert_eq!(initial.resident_unit_count, 0);
    assert_eq!(initial.addressable_unit_count, 2);
    assert_eq!(initial.always_resident_parameter_bytes, 128);
    assert_eq!(initial.runtime_working_set_device_bytes, 64);
    assert!(
        initial.initial_device_bytes
            >= 128 + 64 + initial.metadata_device_bytes
    );
    assert_eq!(initial.scopes.len(), 1);
    assert_eq!(initial.scopes[0].execution_scope, "target");
    assert_eq!(initial.scopes[0].component_count, 2);
    assert_eq!(initial.scopes[0].addressable_unit_count, 2);

    let unowned_error = store
        .load_selector_resource(
            &device,
            &unowned_selector_id,
            0,
            owner.clone(),
        )
        .unwrap_err();
    assert!(unowned_error.to_string().contains("is unknown"));

    let mut corrupt_weight_bytes = weight_bytes.to_vec();
    corrupt_weight_bytes[8] ^= 0xff;
    fs::write(
        root.path().join("weights.bin"),
        &corrupt_weight_bytes,
    )
    .unwrap();
    let corrupt_error = store
        .load_selector_resource(
            &device,
            &selector_id,
            1,
            owner.clone(),
        )
        .unwrap_err();
    assert!(
        corrupt_error
            .to_string()
            .contains("failed SHA-256"),
        "unexpected corrupt-resource error: {corrupt_error}"
    );
    fs::write(root.path().join("weights.bin"), weight_bytes).unwrap();

    store
        .load_selector_resource(
            &device,
            &selector_id,
            0,
            owner.clone(),
        )
        .unwrap();
    store
        .load_selector_resource(&device, &alias_selector_id, 0, owner)
        .unwrap();
    store
        .record_gpu_gate_misses(&selector_id, 2)
        .unwrap();
    store
        .record_gpu_gate_misses(&alias_selector_id, 3)
        .unwrap();

    let stats = store.statistics().unwrap();
    assert_eq!(stats.miss_count, 2);
    assert_eq!(stats.hit_count, 1);
    assert_eq!(stats.resident_group_count, 1);
    assert_eq!(stats.failed_group_count, 1);
    assert_eq!(stats.dynamic_resident_bytes, 8);
    assert_eq!(stats.high_water_resident_group_count, 1);
    assert_eq!(stats.high_water_dynamic_resident_bytes, 8);
    let report = store.residency_report().unwrap();
    assert_eq!(report.physical_device_id, device.physical_device_id());
    assert_eq!(report.logical_device_ids, ["gpu0"]);
    assert_eq!(report.current_payload_bytes, 8);
    assert_eq!(report.high_water_payload_bytes, 8);
    assert_eq!(report.resident_unit_count, 1);
    assert_eq!(report.high_water_resident_unit_count, 1);
    assert_eq!(report.residency_directory_hit_count, 1);
    assert_eq!(report.residency_load_required_count, 2);
    assert_eq!(report.gpu_selection_count, 0);
    assert_eq!(report.gpu_resident_hit_count, 0);
    assert_eq!(report.gpu_miss_count, 5);
    assert_eq!(report.successful_load_count, 1);
    assert_eq!(report.failed_load_count, 1);
    assert_eq!(report.failed_unit_count, 1);
    assert_eq!(report.physical_read_count, 1);
    assert_eq!(report.physical_bytes_read, 8);
    assert_eq!(report.uploaded_bytes, 8);
    assert_eq!(report.components.len(), 2);
    assert!(
        report
            .components
            .iter()
            .all(|component| component.addressable_unit_count == 2
                && component.resident_unit_count == 1)
    );
    assert_eq!(
        report
            .components
            .iter()
            .map(|component| component.gpu_miss_count)
            .collect::<Vec<_>>(),
        [2, 3]
    );
    let address_words = buffers
        .address_table()
        .read_bytes(layout.slot_count() * 32)
        .unwrap()
        .chunks_exact(4)
        .map(|word| u32::from_le_bytes(word.try_into().unwrap()))
        .collect::<Vec<_>>();
    let selected_slot = layout.selectors[0].resource_address_slots[0][0];
    assert_eq!(address_words[selected_slot * 8 + 6], 1);
    assert_ne!(
        address_words[selected_slot * 8]
            | address_words[selected_slot * 8 + 1],
        0
    );

    store.inject_teardown_failure_before_address_clear();
    let injected = store.unload().unwrap_err();
    assert!(
        injected
            .to_string()
            .contains("injected compiled resource teardown failure")
    );
    let quiescing_error = store
        .load_selector_resource(
            &device,
            &selector_id,
            0,
            DeviceResourceResidencyOwnerId::new("quiescing-owner").unwrap(),
        )
        .unwrap_err();
    assert!(quiescing_error.to_string().contains("Quiescing"));
    assert_eq!(
        store.statistics().unwrap().dynamic_resident_bytes,
        0
    );
    let retained_words = buffers
        .address_table()
        .read_bytes(layout.slot_count() * 32)
        .unwrap()
        .chunks_exact(4)
        .map(|word| u32::from_le_bytes(word.try_into().unwrap()))
        .collect::<Vec<_>>();
    assert_eq!(retained_words[selected_slot * 8 + 6], 1);

    let release = store.unload().unwrap();

    assert_eq!(release.group_count, 1);
    assert_eq!(release.byte_count, 8);
    assert_eq!(
        store.statistics().unwrap().dynamic_resident_bytes,
        0
    );
    let unloaded = store.residency_report().unwrap();
    assert_eq!(unloaded.current_payload_bytes, 0);
    assert_eq!(unloaded.resident_unit_count, 0);
    assert_eq!(unloaded.failed_unit_count, 0);
    assert_eq!(unloaded.high_water_payload_bytes, 8);
    assert_eq!(unloaded.high_water_resident_unit_count, 1);
    let retired_words = buffers
        .address_table()
        .read_bytes(layout.slot_count() * 32)
        .unwrap()
        .chunks_exact(4)
        .map(|word| u32::from_le_bytes(word.try_into().unwrap()))
        .collect::<Vec<_>>();
    for record in retired_words.chunks_exact(8) {
        assert_eq!(record[0] | record[1], 0);
        assert_eq!(record[2] | record[3], 0);
        assert_eq!(record[6], 0);
    }
    let repeated = store.unload().unwrap();
    assert_eq!(repeated, DeviceResourceResidencyRelease::default());
    let retired_error = store
        .load_selector_resource(
            &device,
            &selector_id,
            0,
            DeviceResourceResidencyOwnerId::new("retired-owner").unwrap(),
        )
        .unwrap_err();
    assert!(retired_error.to_string().contains("Unloaded"));
    drop(buffers);
    drop(store);

    let baseline_file_descriptors =
        compiled_store_process_file_descriptor_count();
    let baseline_workers = compiled_store_worker_thread_count();
    for cycle_index in 0..3 {
        let cycle_store = VulkanCompiledResourceDeviceStore::new(
            &device,
            format!("amd-test-cycle-{cycle_index}"),
            device.physical_device_id(),
            vec!["gpu0".to_string()],
            root.path(),
            Arc::clone(&contract),
            Arc::clone(&layout),
            BTreeSet::from([
                selector_id.clone(),
                alias_selector_id.clone(),
            ]),
            4096,
            8192,
            1024,
            1,
            128,
            64,
            layout.address_table_byte_count().unwrap(),
        )
        .unwrap();
        cycle_store.mark_mount_complete().unwrap();
        if cycle_index == 0 {
            cycle_store.inject_next_upload_as_device_lost();
            let device_loss = cycle_store
                .load_selector_resource(
                    &device,
                    &selector_id,
                    0,
                    DeviceResourceResidencyOwnerId::new(
                        "device-loss-owner",
                    )
                    .unwrap(),
                )
                .unwrap_err();
            assert!(device_loss.to_string().contains("ERROR_DEVICE_LOST"));
            let terminal = cycle_store
                .load_selector_resource(
                    &device,
                    &selector_id,
                    1,
                    DeviceResourceResidencyOwnerId::new(
                        "post-device-loss-owner",
                    )
                    .unwrap(),
                )
                .unwrap_err();
            assert!(terminal.to_string().contains("Failed"));
            assert!(terminal.to_string().contains("ERROR_DEVICE_LOST"));
            let cycle_release = cycle_store.unload().unwrap();
            assert_eq!(
                cycle_release,
                DeviceResourceResidencyRelease::default()
            );
            drop(cycle_store);
            assert_eq!(
                compiled_store_worker_thread_count(),
                baseline_workers
            );
            assert_eq!(
                compiled_store_process_file_descriptor_count(),
                baseline_file_descriptors
            );
            continue;
        }
        if cycle_index == 1 {
            reset_vulkan_resident_execution_counters();
            assert_eq!(
                cycle_store
                    .load_selector_resources(
                        &device,
                        &selector_id,
                        &[0, 1],
                        DeviceResourceResidencyOwnerId::new(
                            "batched-cycle",
                        )
                        .unwrap(),
                    )
                    .unwrap(),
                2
            );
            let counters = vulkan_resident_execution_counters();
            assert_eq!(counters.resident_copy_queue_submits, 2);
            assert_eq!(counters.resident_copy_waits, 2);
        } else {
            cycle_store
                .load_selector_resource(
                    &device,
                    &selector_id,
                    0,
                    DeviceResourceResidencyOwnerId::new(format!(
                        "cycle-{cycle_index}"
                    ))
                    .unwrap(),
                )
                .unwrap();
        }
        let expected_payload_bytes =
            if cycle_index == 1 { 16 } else { 8 };
        assert_eq!(
            cycle_store
                .residency_report()
                .unwrap()
                .current_payload_bytes,
            expected_payload_bytes
        );
        assert_eq!(cycle_store.backing_store.retained_payload_bytes(), 0);
        let cycle_release = cycle_store.unload().unwrap();
        assert_eq!(
            cycle_release.group_count,
            if cycle_index == 1 { 2 } else { 1 }
        );
        assert_eq!(cycle_release.byte_count, expected_payload_bytes);
        drop(cycle_store);
        assert_eq!(
            compiled_store_worker_thread_count(),
            baseline_workers
        );
        assert_eq!(
            compiled_store_process_file_descriptor_count(),
            baseline_file_descriptors
        );
    }
}
