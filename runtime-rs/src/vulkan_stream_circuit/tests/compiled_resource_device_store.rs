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
                selection_count_per_activation: 1,
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
    let contract = Arc::new(contract);
    let layout = Arc::new(
        VulkanCompiledResourceAddressLayout::from_contract(&contract)
            .unwrap(),
    );
    let capacity_error = match VulkanCompiledResourceDeviceStore::new(
        &device,
        "amd-test",
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

    let unowned_error = store
        .load_selector_resource(
            &device,
            &unowned_selector_id,
            0,
            owner.clone(),
        )
        .unwrap_err();
    assert!(unowned_error.to_string().contains("is unknown"));

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

    let stats = store.statistics().unwrap();
    assert_eq!(stats.miss_count, 1);
    assert_eq!(stats.hit_count, 1);
    assert_eq!(stats.resident_group_count, 1);
    assert_eq!(stats.dynamic_resident_bytes, 8);
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

    let release = store.unload().unwrap();

    assert_eq!(release.group_count, 1);
    assert_eq!(release.byte_count, 8);
    assert_eq!(
        store.statistics().unwrap().dynamic_resident_bytes,
        0
    );
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
}
