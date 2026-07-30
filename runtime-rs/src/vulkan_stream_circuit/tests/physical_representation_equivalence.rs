fn run_fixture_layer_00_causal_batch(
    device: Rc<VulkanComputeDevice>,
    manifest_path: &Path,
    input_frames: &[u8],
    token_ids: &[u32],
) -> (Vec<u8>, Vec<u8>) {
    const FRAME_BYTES: usize = 5_120 * std::mem::size_of::<u16>();
    const DYNAMIC_STATE_CAPACITY: usize = 64;

    let package_dir = manifest_path.parent().unwrap();
    let manifest =
        VulkanResidentModelPackageManifest::from_json_file(manifest_path).unwrap();
    let chain = [
        ("layer_00".to_string(), "layer_00".to_string()),
        ("layer_01".to_string(), "layer_01".to_string()),
    ];
    let runtime_model = manifest
        .mount_runtime_graph_controls(
            Some("gpu0"),
            &BTreeMap::new(),
            &[],
            Some(&chain),
        )
        .unwrap();
    let package_slice = Arc::new(
        VulkanResidentModelPackageDeviceSlice::from_runtime_model_for_device(
            &device,
            package_dir,
            runtime_model,
            "gpu0",
            Some(DYNAMIC_STATE_CAPACITY),
        )
        .unwrap(),
    );
    let mounted = package_slice
        .create_mounted_stream_circuit(&device)
        .unwrap();
    mounted.buffers.initialize_state_buffers(&device).unwrap();
    let reusable_manifest =
        resident_package_reusable_kernel_manifest(&mounted.placed_plan);
    let mounted_bound = mounted
        .mounted_placed_bound_dispatch_plan(&reusable_manifest)
        .unwrap();
    let tick_plan =
        VulkanMountedPlacedStreamTickPlan::from_mounted_bound_plan(&mounted_bound);
    let resident_execution_plan =
        VulkanMountedPlacedResidentStreamTickExecutionPlan::
            from_tick_plan_with_distributed_dispatch_groups(
                &device,
                &mounted,
                &mounted_bound,
                package_slice.loaded_manifest(),
                tick_plan,
                &[],
            )
            .unwrap();
    let processor_device = VulkanResidentInProcessPlacedStreamProcessorDevice {
        device_id: "gpu0".to_string(),
        hosted_component_count: package_slice.hosted_component_count,
        incoming_edge_count: package_slice.incoming_edge_count,
        outgoing_edge_count: package_slice.outgoing_edge_count,
        dispatch_count: mounted_bound.dispatches.len(),
        package_slice,
        mounted,
        mounted_bound,
        resident_execution_plan,
        demand_residency_context: None,
    };
    let devices = BTreeMap::from([("gpu0".to_string(), device)]);
    let quantum_calibrators = BTreeMap::from([(
        "gpu0".to_string(),
        Rc::new(RefCell::new(RuntimeExecutionQuantumCalibrator::default())),
    )]);
    let distributed_execution_plan = VulkanDistributedExecutionPlan {
        device_ids: Vec::new(),
        storage_buffer_offset_alignment: 256,
        dispatches: Vec::new(),
        dispatch_groups: Vec::new(),
        shared_input_byte_capacity: 0,
        shared_output_byte_capacity: 0,
        distributed_parameter_byte_count: 0,
    };
    let distributed_parameter_buffers = VulkanDistributedParameterBuffers {
        plan: VulkanDistributedParameterAllocationPlan {
            allocations: Vec::new(),
            allocation_count: 0,
            tensor_count: 0,
            total_byte_capacity: 0,
        },
        buffers: Vec::new(),
        total_byte_capacity: 0,
    };
    let placed_slices = [processor_device];
    let runner = VulkanResidentPlacedComponentBatchRunner::new(
        &devices,
        &placed_slices,
        "pairpacked-prefill-equivalence",
        &quantum_calibrators,
        16,
        VulkanComponentBatchExecutionMode::CausalSequence,
        true,
        &distributed_execution_plan,
        &distributed_parameter_buffers,
    )
    .unwrap();
    let input = runner.slices[0]
        .signal_buffer(&VulkanComponentBatchSignalKey::ModelInput(
            "input_frame".to_string(),
        ))
        .unwrap();
    assert_eq!(input.frame_byte_capacity, FRAME_BYTES);
    input.buffer.write_bytes(input_frames).unwrap();
    runner
        .run_causal_sequence(
            &devices,
            0,
            "gpu0",
            &placed_slices[0].mounted,
            token_ids,
            0,
            u32::try_from(DYNAMIC_STATE_CAPACITY).unwrap(),
        )
        .unwrap();
    assert!(runner.commit_causal_state_prefix(token_ids.len()).unwrap());
    let output = runner.slices[0]
        .signal_buffer(&VulkanComponentBatchSignalKey::ModelOutput(
            "output_frame".to_string(),
        ))
        .unwrap()
        .buffer
        .read_bytes(FRAME_BYTES * token_ids.len())
        .unwrap();
    let mut state = Vec::new();
    for allocation in &placed_slices[0].mounted.buffers.state_buffers {
        state.extend(
            allocation
                .buffer
                .read_bytes(allocation.byte_capacity)
                .unwrap(),
        );
    }
    (output, state)
}

#[test]
fn fused_pairpacked_rms_norm_preserves_complete_causal_batch() {
    const WIDTH: usize = 12;
    const FRAME_BYTES: usize = 5_120 * std::mem::size_of::<u16>();

    let (
        Some(candidate_manifest_path),
        Some(reference_manifest_path),
    ) = (
        std::env::var_os("NERVE_REPRESENTATION_CANDIDATE_PACKAGE")
            .map(PathBuf::from),
        std::env::var_os("NERVE_REPRESENTATION_REFERENCE_PACKAGE")
            .map(PathBuf::from),
    )
    else {
        eprintln!(
            "skipping fused pairpacked causal-batch equivalence: \
             representation candidate or reference package is unset"
        );
        return;
    };
    let candidate_package_dir = candidate_manifest_path.parent().unwrap();
    let tensor_index = TensorIndex::from_package_json_file(
        candidate_package_dir.join("tensors.json"),
    )
    .unwrap();
    let embedding_storage = crate::tensor_storage::TensorStorage::from_index(
        &tensor_index,
        "model.language_model.embed_tokens.weight",
    )
    .unwrap();
    let mut embedding_file = std::fs::File::open(&embedding_storage.source_file).unwrap();
    let mut header_len_bytes = [0u8; 8];
    embedding_file.read_exact(&mut header_len_bytes).unwrap();
    let tensor_data_start = 8 + u64::from_le_bytes(header_len_bytes);
    let token_ids = (1..=u32::try_from(WIDTH).unwrap()).collect::<Vec<_>>();
    let mut input_frames = Vec::with_capacity(WIDTH * FRAME_BYTES);
    for token_id in &token_ids {
        embedding_file
            .seek(SeekFrom::Start(
                tensor_data_start
                    + u64::try_from(
                        embedding_storage.data_start
                            + usize::try_from(*token_id).unwrap() * FRAME_BYTES,
                    )
                    .unwrap(),
            ))
            .unwrap();
        let mut frame = vec![0; FRAME_BYTES];
        embedding_file.read_exact(&mut frame).unwrap();
        input_frames.extend(frame);
    }

    let device = Rc::new(
        selected_test_vulkan_device().expect("selected Vulkan test device must open"),
    );
    let (reference_output, reference_state) = run_fixture_layer_00_causal_batch(
        device.clone(),
        &reference_manifest_path,
        &input_frames,
        &token_ids,
    );
    let (candidate_output, candidate_state) = run_fixture_layer_00_causal_batch(
        device,
        &candidate_manifest_path,
        &input_frames,
        &token_ids,
    );
    assert_eq!(candidate_output, reference_output);
    assert_eq!(candidate_state, reference_state);
}
