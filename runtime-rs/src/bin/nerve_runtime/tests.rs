#[cfg(test)]
mod tests {
    use std::collections::{BTreeMap, BTreeSet};
    use std::fs;
    use std::time::{SystemTime, UNIX_EPOCH};

    use chrono::{FixedOffset, TimeZone};
    use tokenizers::models::wordlevel::WordLevel;
    use tokenizers::pre_tokenizers::whitespace::Whitespace;
    use tokenizers::processors::template::TemplateProcessing;
    use tokenizers::{AddedToken, Tokenizer};

    use nerve_runtime::{
        ResourceResidencyPolicy, RuntimeChatFormatter, RuntimeChatMessage, RuntimeChatSession,
        RuntimeRecoverableChatTurnError, VulkanComputeDeviceInfo,
        VulkanResidentDistributedExecutionPhaseCounters,
        VulkanResidentHfTokenizerTextCodec, VulkanResidentTokenTextCodec,
        VulkanResidentTokenTextCodecError, assistant_content_token_ids, chat_transcript_codec,
        model_owned_assistant_turn_stop_token_id, normalize_chat_template_for_runtime,
        normalize_generated_tokens_at_protocol_boundary,
    };

    use super::{
        Args, RuntimeChatReplOutcome, RuntimeChatTurnOutcome, RuntimeSustainedDecodeReport,
        RuntimeSustainedDecodeSample, parse_allowed_physical_device_id, parse_args_from,
        parse_chat_template_variable, parse_device_binding_assignment, parse_source_chain,
        parse_vulkan_device_uuid_ref,
        rank_runtime_auto_placement_candidates_across_capability_classes,
        resolve_runtime_context_size, resolve_runtime_vulkan_physical_device_ref_in,
        resolve_speculative_draft_tokens, runtime_chat_repl_control, runtime_critical_path_lines,
        runtime_device_bindings_report, runtime_distributed_execution_phase_counter_lines,
        runtime_physical_device_bindings_in, runtime_uses_explicit_placement, submit_chat_turn,
        usage, validate_explicit_logical_device_bindings,
    };

    fn formatter(template_source: &str) -> RuntimeChatFormatter {
        RuntimeChatFormatter {
            template_source: template_source.to_string(),
            template_variables: serde_json::Map::new(),
            render_time: FixedOffset::east_opt(0)
                .unwrap()
                .with_ymd_and_hms(2026, 7, 18, 12, 0, 0)
                .unwrap(),
            compiled_codec: None,
        }
    }

    #[test]
    fn chat_repl_distinguishes_new_conversation_from_process_exit() {
        assert_eq!(
            runtime_chat_repl_control("/new"),
            Some(RuntimeChatReplOutcome::NewConversation),
        );
        assert_eq!(
            runtime_chat_repl_control("/NEW"),
            Some(RuntimeChatReplOutcome::NewConversation),
        );
        for command in ["exit", "quit", "/exit", "/quit"] {
            assert_eq!(
                runtime_chat_repl_control(command),
                Some(RuntimeChatReplOutcome::Exit),
            );
        }
        assert_eq!(runtime_chat_repl_control("new"), None);
        assert_eq!(runtime_chat_repl_control("/new topic"), None);
    }

    #[derive(Clone, Copy)]
    struct CharacterCodec;

    #[test]
    fn duplicate_default_device_is_rejected() {
        let error = parse_args_from(
            ["--device", "gpu0", "--device", "gpu1"]
                .into_iter()
                .map(str::to_string),
        )
        .unwrap_err();

        assert_eq!(error, "--device may only be supplied once");
    }

    #[test]
    fn automatic_capacity_packing_yields_only_to_explicit_placement_controls() {
        assert!(!runtime_uses_explicit_placement(&Args::default()));

        let mut allowed_inventory = Args::default();
        allowed_inventory
            .allowed_physical_device_ids
            .insert("vulkan-uuid:00000000030000000000000000000000".to_string());
        assert!(!runtime_uses_explicit_placement(&allowed_inventory));

        let mut logical = Args::default();
        logical.default_device_id = Some("chosen".to_string());
        assert!(runtime_uses_explicit_placement(&logical));

        let mut physical = Args::default();
        physical.vulkan_device_index = Some(2);
        assert!(runtime_uses_explicit_placement(&physical));

        let mut custom = Args::default();
        custom
            .node_devices
            .insert("block_0".to_string(), "gpu1".to_string());
        assert!(runtime_uses_explicit_placement(&custom));
    }

    #[test]
    fn automatic_capacity_packing_prefers_one_unreserved_gpu_over_spilling_from_default() {
        let ranked = rank_runtime_auto_placement_candidates_across_capability_classes(
            vec![
                (
                    10,
                    true,
                    0,
                    "same".to_string(),
                    nerve_runtime::VulkanRuntimePlacementCandidate {
                        device_id: "reserved-default".to_string(),
                        safe_capacity_bytes: 4,
                    },
                ),
                (
                    10,
                    false,
                    1,
                    "same".to_string(),
                    nerve_runtime::VulkanRuntimePlacementCandidate {
                        device_id: "roomy".to_string(),
                        safe_capacity_bytes: 32,
                    },
                ),
                (
                    10,
                    false,
                    2,
                    "same".to_string(),
                    nerve_runtime::VulkanRuntimePlacementCandidate {
                        device_id: "also-roomy".to_string(),
                        safe_capacity_bytes: 32,
                    },
                ),
            ],
            Some("same"),
        );

        assert_eq!(
            ranked
                .iter()
                .map(|candidate| candidate.device_id.as_str())
                .collect::<Vec<_>>(),
            ["roomy", "also-roomy", "reserved-default"],
        );
    }

    #[test]
    fn automatic_capacity_packing_retains_heterogeneous_spill_targets() {
        let ranked = rank_runtime_auto_placement_candidates_across_capability_classes(
            vec![
                (
                    10,
                    true,
                    0,
                    "primary-class".to_string(),
                    nerve_runtime::VulkanRuntimePlacementCandidate {
                        device_id: "primary-reserved".to_string(),
                        safe_capacity_bytes: 24,
                    },
                ),
                (
                    10,
                    false,
                    1,
                    "primary-class".to_string(),
                    nerve_runtime::VulkanRuntimePlacementCandidate {
                        device_id: "primary-roomy".to_string(),
                        safe_capacity_bytes: 32,
                    },
                ),
                (
                    10,
                    false,
                    2,
                    "spill-class".to_string(),
                    nerve_runtime::VulkanRuntimePlacementCandidate {
                        device_id: "heterogeneous-spill".to_string(),
                        safe_capacity_bytes: 28,
                    },
                ),
            ],
            Some("primary-class"),
        );

        assert_eq!(
            ranked
                .iter()
                .map(|candidate| candidate.device_id.as_str())
                .collect::<Vec<_>>(),
            ["primary-roomy", "primary-reserved", "heterogeneous-spill"],
        );
    }

    #[test]
    fn automatic_capacity_packing_uses_live_package_cost_within_equal_capacity() {
        let ranked = rank_runtime_auto_placement_candidates_across_capability_classes(
            vec![
                (
                    300,
                    true,
                    0,
                    "same".to_string(),
                    nerve_runtime::VulkanRuntimePlacementCandidate {
                        device_id: "slow-default".to_string(),
                        safe_capacity_bytes: 32,
                    },
                ),
                (
                    100,
                    false,
                    1,
                    "same".to_string(),
                    nerve_runtime::VulkanRuntimePlacementCandidate {
                        device_id: "fast".to_string(),
                        safe_capacity_bytes: 32,
                    },
                ),
            ],
            Some("same"),
        );

        assert_eq!(
            ranked
                .iter()
                .map(|candidate| candidate.device_id.as_str())
                .collect::<Vec<_>>(),
            ["fast", "slow-default"],
        );
    }

    #[test]
    fn residency_policy_is_an_explicit_normal_runtime_control() {
        let demand = parse_args_from(
            ["--residency-policy", "demand-retained"]
                .into_iter()
                .map(str::to_string),
        )
        .unwrap();
        assert_eq!(
            demand.resource_residency_policy,
            ResourceResidencyPolicy::DemandRetained
        );
        let paged = parse_args_from(
            ["--residency-policy", "demand-paged"]
                .into_iter()
                .map(str::to_string),
        )
        .unwrap();
        assert_eq!(
            paged.resource_residency_policy,
            ResourceResidencyPolicy::DemandPaged
        );

        let eager = parse_args_from(std::iter::empty()).unwrap();
        assert_eq!(
            eager.resource_residency_policy,
            ResourceResidencyPolicy::Eager
        );

        let error = parse_args_from(
            ["--residency-policy", "automatic"]
                .into_iter()
                .map(str::to_string),
        )
        .unwrap_err();
        assert_eq!(
            error,
            "invalid --residency-policy \"automatic\"; expected eager, demand-retained, or demand-paged"
        );
    }

    #[test]
    fn speculative_confidence_threshold_is_a_bounded_runtime_control() {
        let args = parse_args_from(
            ["--speculative-confidence-threshold", "0.625"]
                .into_iter()
                .map(str::to_string),
        )
        .unwrap();
        assert_eq!(args.speculative_confidence_threshold, 0.625);

        for invalid in ["-0.1", "1.1", "NaN", "inf"] {
            let error = parse_args_from(
                ["--speculative-confidence-threshold", invalid]
                    .into_iter()
                    .map(str::to_string),
            )
            .unwrap_err();
            assert_eq!(
                error,
                "--speculative-confidence-threshold must be finite and in [0, 1]"
            );
        }
    }

    #[test]
    fn speculative_width_distinguishes_package_default_from_explicit_disable() {
        let package_default = parse_args_from(std::iter::empty()).unwrap();
        assert_eq!(package_default.speculative_draft_tokens, None);

        let disabled = parse_args_from(
            ["--speculative-draft-tokens", "0"]
                .into_iter()
                .map(str::to_string),
        )
        .unwrap();
        assert_eq!(disabled.speculative_draft_tokens, Some(0));

        let explicit = parse_args_from(
            ["--speculative-draft-tokens", "7"]
                .into_iter()
                .map(str::to_string),
        )
        .unwrap();
        assert_eq!(explicit.speculative_draft_tokens, Some(7));
    }

    #[test]
    fn speculative_width_uses_package_recommendation_unless_explicitly_overridden() {
        assert_eq!(
            resolve_speculative_draft_tokens(None, || Ok(Some(7))).unwrap(),
            7,
        );
        assert_eq!(
            resolve_speculative_draft_tokens(None, || Ok(None)).unwrap(),
            0,
        );

        let package_was_consulted = std::cell::Cell::new(false);
        assert_eq!(
            resolve_speculative_draft_tokens(Some(0), || {
                package_was_consulted.set(true);
                Err("conflicting package recommendation".to_string())
            })
            .unwrap(),
            0,
        );
        assert!(!package_was_consulted.get());
    }

    #[test]
    fn speculative_controls_are_described_by_contract_not_model_family() {
        let usage = usage();
        let normalized = usage.to_ascii_lowercase();

        assert!(normalized.contains("compiled speculative-decoder tokens"));
        assert!(normalized.contains("package recommendation"));
        assert!(normalized.contains("pass 0 explicitly to disable"));
        assert!(normalized.contains("compiled confidence"));
        for family in ["DeepSeek", "DSpark", "Qwen", "MTP"] {
            assert!(
                !usage.contains(family),
                "runtime usage leaked model-family term {family:?}"
            );
        }
    }

    #[test]
    fn component_shard_pools_are_explicit_runtime_controls() {
        let args = parse_args_from(
            [
                "--place-node",
                "layer_07=gpu1",
                "--shard-component",
                "layer_07=gpu1,gpu2,gpu3",
            ]
            .into_iter()
            .map(str::to_string),
        )
        .unwrap();

        assert_eq!(
            args.component_shard_devices,
            BTreeMap::from([(
                "layer_07".to_string(),
                vec!["gpu1".to_string(), "gpu2".to_string(), "gpu3".to_string()],
            )])
        );
    }

    #[test]
    fn component_shard_pools_reject_ambiguous_or_repeated_devices() {
        for invalid in [
            "layer_07=gpu0",
            "layer_07=gpu0,gpu0",
            "layer_07=gpu0,",
            "=gpu0,gpu1",
        ] {
            let error = parse_args_from(
                ["--shard-component", invalid]
                    .into_iter()
                    .map(str::to_string),
            )
            .unwrap_err();

            assert!(
                error.contains("invalid component shard assignment"),
                "{invalid:?}: {error}"
            );
        }
    }

    #[test]
    fn explicit_device_bindings_must_exist_in_the_effective_graph() {
        let args = parse_args_from(
            [
                "--bind-device",
                "gpu0=vulkan-uuid:00000000070000000000000000000000",
                "--bind-device",
                "gpu1=vulkan-uuid:000000000a0000000000000000000000",
            ]
            .into_iter()
            .map(str::to_string),
        )
        .unwrap();

        let error =
            validate_explicit_logical_device_bindings(&args, &["runtime_default".to_string()])
                .unwrap_err();
        assert!(
            error
                .to_string()
                .contains("absent from the effective graph")
        );
        validate_explicit_logical_device_bindings(&args, &["gpu0".to_string(), "gpu1".to_string()])
            .unwrap();
    }

    #[test]
    fn device_capability_inspection_is_a_package_free_mode() {
        let args = parse_args_from(
            [
                "--inspect-devices",
                "--initialize-device-contexts",
                "--json",
            ]
            .into_iter()
            .map(str::to_string),
        )
        .unwrap();

        assert!(args.inspect_devices);
        assert!(args.initialize_device_contexts);
        assert!(args.json);
        assert!(args.package_manifest.is_none());
    }

    #[test]
    fn device_context_initialization_requires_device_inspection() {
        let error = parse_args_from(
            ["--initialize-device-contexts"]
                .into_iter()
                .map(str::to_string),
        )
        .unwrap_err();

        assert_eq!(
            error,
            "--initialize-device-contexts requires --inspect-devices"
        );
    }

    #[test]
    fn physical_device_allowlist_accepts_only_unique_canonical_vulkan_uuids() {
        let first = "vulkan-uuid:00000000070000000000000000000000";
        let second = "vulkan-uuid:000000000a0000000000000000000000";
        let args = parse_args_from(
            [
                "--allow-physical-device",
                first,
                "--allow-physical-device",
                second,
            ]
            .into_iter()
            .map(str::to_string),
        )
        .unwrap();

        assert_eq!(
            args.allowed_physical_device_ids,
            BTreeSet::from([first.to_string(), second.to_string()])
        );
        assert_eq!(parse_allowed_physical_device_id(first).unwrap(), first);

        let duplicate = parse_args_from(
            [
                "--allow-physical-device",
                first,
                "--allow-physical-device",
                first,
            ]
            .into_iter()
            .map(str::to_string),
        )
        .unwrap_err();
        assert!(duplicate.contains("duplicate allowed physical device"));

        for invalid in [
            "vulkan:0",
            "vulkan-uuid:000000000A0000000000000000000000",
            "vulkan-uuid:abcd",
            "cpu0",
        ] {
            assert!(
                parse_allowed_physical_device_id(invalid).is_err(),
                "accepted invalid allowlist target {invalid:?}"
            );
        }
    }

    impl VulkanResidentTokenTextCodec for CharacterCodec {
        fn encode_text(&self, text: &str) -> Result<Vec<u32>, VulkanResidentTokenTextCodecError> {
            Ok(text.chars().map(u32::from).collect())
        }

        fn decode_tokens(
            &self,
            token_ids: &[u32],
        ) -> Result<String, VulkanResidentTokenTextCodecError> {
            token_ids
                .iter()
                .map(|token_id| {
                    char::from_u32(*token_id).ok_or_else(|| {
                        VulkanResidentTokenTextCodecError::new(format!(
                            "invalid character token {token_id}"
                        ))
                    })
                })
                .collect()
        }
    }

    #[test]
    fn chat_template_tokenization_does_not_inject_post_processor_special_tokens() {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let tokenizer_dir = std::env::temp_dir().join(format!(
            "nerve-chat-tokenizer-specials-{}-{unique}",
            std::process::id()
        ));
        fs::create_dir_all(&tokenizer_dir).unwrap();

        let mut tokenizer = Tokenizer::new(WordLevel::default());
        tokenizer
            .add_special_tokens([AddedToken::from("<bos>", true)])
            .unwrap();
        tokenizer
            .add_tokens([AddedToken::from("hello", false)])
            .unwrap();
        let bos_id = tokenizer.token_to_id("<bos>").unwrap();
        let hello_id = tokenizer.token_to_id("hello").unwrap();
        tokenizer.with_pre_tokenizer(Some(Whitespace));
        tokenizer.with_post_processor(Some(
            TemplateProcessing::builder()
                .try_single("<bos> $A")
                .unwrap()
                .special_tokens(vec![("<bos>", bos_id)])
                .build()
                .unwrap(),
        ));
        tokenizer
            .save(tokenizer_dir.join("tokenizer.json"), false)
            .unwrap();

        let raw_codec = VulkanResidentHfTokenizerTextCodec::from_model_dir(&tokenizer_dir)
            .unwrap()
            .with_add_special_tokens(true);
        let chat_codec = chat_transcript_codec(&tokenizer_dir).unwrap();

        assert_eq!(
            raw_codec.encode_text("hello").unwrap(),
            vec![bos_id, hello_id]
        );
        assert_eq!(chat_codec.encode_text("hello").unwrap(), vec![hello_id]);

        fs::remove_dir_all(tokenizer_dir).unwrap();
    }

    #[test]
    fn context_defaults_to_model_capacity_and_rejects_impossible_requests() {
        assert_eq!(
            resolve_runtime_context_size(131_072, None, 65_536).unwrap(),
            131_072
        );
        assert_eq!(
            resolve_runtime_context_size(131_072, Some(8_192), 4_096).unwrap(),
            8_192
        );

        let too_small = resolve_runtime_context_size(131_072, Some(4_096), 4_097).unwrap_err();
        assert_eq!(too_small.kind(), std::io::ErrorKind::InvalidInput);
        assert!(too_small.to_string().contains("cannot hold"));

        let too_large = resolve_runtime_context_size(32_768, Some(65_536), 1).unwrap_err();
        assert_eq!(too_large.kind(), std::io::ErrorKind::InvalidInput);
        assert!(too_large.to_string().contains("exceeds the model maximum"));

        let zero_model = resolve_runtime_context_size(0, None, 0).unwrap_err();
        assert_eq!(zero_model.kind(), std::io::ErrorKind::InvalidData);
    }

    #[test]
    fn runtime_device_binding_parser_validates_syntax_without_device_discovery() {
        assert_eq!(
            parse_device_binding_assignment("gpu0 = vulkan:5").unwrap(),
            ("gpu0".to_string(), "vulkan:5".to_string())
        );
        assert_eq!(
            parse_device_binding_assignment("remote = lan:worker-a").unwrap(),
            ("remote".to_string(), "lan:worker-a".to_string())
        );
        assert_eq!(
            resolve_runtime_vulkan_physical_device_ref_in("vulkan:7", &[]).unwrap(),
            Some(7)
        );

        for invalid in [
            "gpu0=vulkan:",
            "gpu0=vulkan-latest",
            "cpu0=cpu:",
            "cpu0=cpuish",
            "gpu0=vulkan-uuid:abcd",
        ] {
            assert!(
                parse_device_binding_assignment(invalid).is_err(),
                "accepted invalid binding {invalid:?}"
            );
        }
    }

    #[test]
    fn runtime_physical_bindings_distinguish_logical_from_physical_placement() {
        let available_devices = vec![
            VulkanComputeDeviceInfo {
                physical_device_index: 2,
                physical_device_id: "vulkan-uuid:00000000000000000000000000000002".to_string(),
                device_uuid: [2; 16],
                device_name: "GPU 2".to_string(),
                pci_address: Some("0000:02:00.0".to_string()),
                device_type: "discrete_gpu".to_string(),
                vendor_id: 1,
                device_id: 2,
                api_version: 1,
                driver_version: 1,
                compute_queue_family_indices: vec![0],
                memory_heaps: Vec::new(),
                selected_by_default: true,
            },
            VulkanComputeDeviceInfo {
                physical_device_index: 3,
                physical_device_id: "vulkan-uuid:00000000000000000000000000000003".to_string(),
                device_uuid: [3; 16],
                device_name: "GPU 3".to_string(),
                pci_address: Some("0000:03:00.0".to_string()),
                device_type: "discrete_gpu".to_string(),
                vendor_id: 1,
                device_id: 3,
                api_version: 1,
                driver_version: 1,
                compute_queue_family_indices: vec![0],
                memory_heaps: Vec::new(),
                selected_by_default: false,
            },
        ];
        let logical_device_ids = vec!["device_a".to_string(), "device_b".to_string()];
        let colocated = runtime_physical_device_bindings_in(
            &Args::default(),
            &logical_device_ids,
            &available_devices,
        )
        .unwrap();
        assert_eq!(colocated.get("device_a"), Some(&2));
        assert_eq!(colocated.get("device_b"), Some(&2));
        assert_eq!(colocated.values().collect::<BTreeSet<_>>().len(), 1);

        let mut split_args = Args::default();
        split_args
            .device_bindings
            .insert("device_b".to_string(), "vulkan:3".to_string());
        let split = runtime_physical_device_bindings_in(
            &split_args,
            &logical_device_ids,
            &available_devices,
        )
        .unwrap();
        assert_eq!(split.get("device_a"), Some(&2));
        assert_eq!(split.get("device_b"), Some(&3));
        assert_eq!(split.values().collect::<BTreeSet<_>>().len(), 2);
    }

    #[test]
    fn fully_explicit_device_binding_report_does_not_request_an_unused_default_gpu() {
        let logical_device_ids = vec!["device_a".to_string(), "device_b".to_string()];
        let mut args = Args::default();
        args.device_bindings
            .insert("device_a".to_string(), "vulkan:2".to_string());
        args.device_bindings
            .insert("device_b".to_string(), "vulkan:3".to_string());

        let available_devices = vec![
            VulkanComputeDeviceInfo {
                physical_device_index: 2,
                physical_device_id: "vulkan-uuid:00000000000000000000000000000002".to_string(),
                device_uuid: [2; 16],
                device_name: "GPU 2".to_string(),
                pci_address: None,
                device_type: "discrete_gpu".to_string(),
                vendor_id: 1,
                device_id: 2,
                api_version: 1,
                driver_version: 1,
                compute_queue_family_indices: vec![0],
                memory_heaps: Vec::new(),
                selected_by_default: true,
            },
            VulkanComputeDeviceInfo {
                physical_device_index: 3,
                physical_device_id: "vulkan-uuid:00000000000000000000000000000003".to_string(),
                device_uuid: [3; 16],
                device_name: "GPU 3".to_string(),
                pci_address: None,
                device_type: "discrete_gpu".to_string(),
                vendor_id: 1,
                device_id: 3,
                api_version: 1,
                driver_version: 1,
                compute_queue_family_indices: vec![0],
                memory_heaps: Vec::new(),
                selected_by_default: false,
            },
        ];
        let report = runtime_device_bindings_report(&args, &logical_device_ids, &available_devices);

        assert_eq!(report.process_vulkan_device_index, None);
        assert_eq!(report.default_vulkan_device_index, None);
        assert_eq!(report.requested_vulkan_device_indices, vec![2, 3]);
    }

    #[test]
    fn runtime_source_chain_parser_preserves_duplicates_only_with_unique_instances() {
        assert_eq!(
            parse_source_chain("layer_0 -> repeat=layer_0 -> layer_1").unwrap(),
            vec![
                ("layer_0".to_string(), "layer_0".to_string()),
                ("repeat".to_string(), "layer_0".to_string()),
                ("layer_1".to_string(), "layer_1".to_string()),
            ]
        );
        assert!(parse_source_chain("layer_0,layer_0").is_err());
        assert!(parse_source_chain("layer_0,,layer_1").is_err());
        assert!(parse_source_chain("repeat=").is_err());
    }

    #[test]
    fn stable_vulkan_device_uuid_references_are_parsed_without_discovery() {
        assert_eq!(
            parse_vulkan_device_uuid_ref("vulkan-uuid:000000000a0000000000000000000000").unwrap(),
            Some([0, 0, 0, 0, 10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        );
        assert!(
            parse_vulkan_device_uuid_ref("vulkan-uuid:not-a-device")
                .unwrap_err()
                .contains("32 hexadecimal digits")
        );
        assert_eq!(parse_vulkan_device_uuid_ref("vulkan:3").unwrap(), None);
    }

    #[test]
    fn model_template_controls_pipe_turn_role_names() {
        let mut formatter = formatter(
            "{%- for message in messages %}{{- '<|turn>' + ('model' if message.role == 'assistant' else message.role) + '\n' + (message.content | trim) + '<turn|>\n' }}{%- endfor %}{%- if add_generation_prompt %}{{- '<|turn>model\n' }}{%- endif %}",
        );
        formatter.template_variables.insert(
            "bos_token".to_string(),
            serde_json::Value::String("<bos>".to_string()),
        );
        let messages = vec![
            RuntimeChatMessage {
                role: "user".to_string(),
                content: "Hello".to_string(),
            },
            RuntimeChatMessage {
                role: "assistant".to_string(),
                content: "Hi there".to_string(),
            },
            RuntimeChatMessage {
                role: "user".to_string(),
                content: "Remember me".to_string(),
            },
        ];

        assert_eq!(
            formatter.format_messages(&messages, true).unwrap(),
            "<|turn>user\nHello<turn|>\n<|turn>model\nHi there<turn|>\n<|turn>user\nRemember me<turn|>\n<|turn>model\n"
        );
    }

    #[test]
    fn model_template_keeps_default_reasoning_branch() {
        let formatter = formatter(
            "{%- for message in messages %}{{- '<|im_start|>' + message.role + '\n' + message.content + '<|im_end|>\n' }}{%- endfor %}{%- if add_generation_prompt %}{{- '<|im_start|>assistant\n' }}{%- if enable_thinking is defined and enable_thinking is false %}{{- '<think>\n\n</think>\n\n' }}{%- else %}{{- '<think>\n' }}{%- endif %}{%- endif %}",
        );

        assert_eq!(
            formatter
                .format_messages(
                    &[RuntimeChatMessage {
                        role: "user".to_string(),
                        content: "Solve this".to_string(),
                    }],
                    true,
                )
                .unwrap(),
            "<|im_start|>user\nSolve this<|im_end|>\n<|im_start|>assistant\n<think>\n"
        );
    }

    #[test]
    fn model_template_accepts_boolean_reasoning_control() {
        let mut formatter = formatter(
            "{%- if add_generation_prompt %}{%- if enable_thinking is false %}direct{%- else %}thinking{%- endif %}{%- endif %}",
        );
        formatter.template_variables.insert(
            "enable_thinking".to_string(),
            serde_json::Value::Bool(false),
        );

        assert_eq!(
            formatter
                .format_messages(
                    &[RuntimeChatMessage {
                        role: "user".to_string(),
                        content: "Answer directly".to_string(),
                    }],
                    true,
                )
                .unwrap(),
            "direct"
        );
    }

    #[test]
    fn chat_template_variables_require_json_values_and_jinja_names() {
        assert_eq!(
            parse_chat_template_variable("enable_thinking=false").unwrap(),
            (
                "enable_thinking".to_string(),
                serde_json::Value::Bool(false)
            )
        );
        assert_eq!(
            parse_chat_template_variable("tool_choice=\"auto\"").unwrap(),
            (
                "tool_choice".to_string(),
                serde_json::Value::String("auto".to_string())
            )
        );
        assert!(parse_chat_template_variable("enable-thinking=false").is_err());
        assert!(parse_chat_template_variable("enable_thinking=disabled").is_err());
    }

    #[test]
    fn hugging_face_generation_metadata_preserves_rendered_content_and_trimming() {
        let normalized = normalize_chat_template_for_runtime(
            "before \n{%- generation -%}\nassistant content\n{%- endgeneration -%}\n after",
        );
        let formatter = formatter(&normalized);

        assert_eq!(
            formatter.format_messages(&[], false).unwrap(),
            "beforeassistant contentafter"
        );
    }

    #[test]
    fn model_template_can_supply_a_dated_default_system_turn() {
        let formatter = formatter(
            "{%- if messages[0].role == 'system' %}{%- set loop_messages = messages[1:] %}{%- else %}{{- '<|start_of_role|>system<|end_of_role|>Current Date: ' + strftime_now('%B %d, %Y') + '.<|end_of_text|>\n' }}{%- set loop_messages = messages %}{%- endif %}{%- for message in loop_messages %}{{- '<|start_of_role|>' + message.role + '<|end_of_role|>' + message.content + '<|end_of_text|>\n' }}{%- if loop.last and add_generation_prompt %}{{- '<|start_of_role|>assistant<|end_of_role|>' }}{%- endif %}{%- endfor %}",
        );

        assert_eq!(
            formatter
                .format_messages(
                    &[RuntimeChatMessage {
                        role: "user".to_string(),
                        content: "Hello".to_string(),
                    }],
                    true,
                )
                .unwrap(),
            "<|start_of_role|>system<|end_of_role|>Current Date: July 18, 2026.<|end_of_text|>\n<|start_of_role|>user<|end_of_role|>Hello<|end_of_text|>\n<|start_of_role|>assistant<|end_of_role|>"
        );
    }

    #[test]
    fn chat_turn_transaction_commits_only_template_canonical_assistant_content() {
        let mut session = RuntimeChatSession {
            formatter: formatter(
                "{%- for message in messages -%}{%- if message.role == 'user' -%}{{- '[user]' + message.content + '!' -}}{%- else -%}{{- '[assistant]' + message.content.split('</think>')[-1] + '!' -}}{%- endif -%}{%- endfor -%}{%- if add_generation_prompt -%}{{- '[assistant]<think>' -}}{%- endif -%}",
            ),
            messages: Vec::new(),
            committed_token_ids: Vec::new(),
        };

        let first = session.prepare_user_turn("first", &CharacterCodec).unwrap();
        assert_eq!(
            CharacterCodec
                .decode_tokens(&first.user_token_delta)
                .unwrap(),
            "[user]first!"
        );
        assert_eq!(
            CharacterCodec
                .decode_tokens(&first.generation_prompt_token_delta)
                .unwrap(),
            "[assistant]<think>"
        );
        let assistant_message = serde_json::json!({
            "role": "assistant",
            "content": "private reasoning</think>answer",
        });
        let (assistant_delta, canonical) = session
            .render_assistant_commit_token_delta(
                &first,
                "first",
                &assistant_message,
                &CharacterCodec,
            )
            .unwrap();
        assert_eq!(
            CharacterCodec.decode_tokens(&assistant_delta).unwrap(),
            "[assistant]answer![user]"
        );
        assert!(
            !CharacterCodec
                .decode_tokens(&canonical)
                .unwrap()
                .contains("private reasoning")
        );
        session.commit_assistant_turn("first", &assistant_message, canonical);

        let second = session
            .prepare_user_turn("second", &CharacterCodec)
            .unwrap();
        assert_eq!(
            CharacterCodec
                .decode_tokens(&second.user_token_delta)
                .unwrap(),
            "second!"
        );
        assert_eq!(
            CharacterCodec
                .decode_tokens(&second.generation_prompt_token_delta)
                .unwrap(),
            "[assistant]<think>"
        );
    }

    #[test]
    fn canonical_chat_history_preserves_structured_assistant_messages() {
        let mut session = RuntimeChatSession {
            formatter: formatter(
                "{%- for message in messages -%}{%- if message.role == 'user' -%}{{- '[user]' + message.content + '!' -}}{%- else -%}{{- '[assistant]' + message.content -}}{%- for call in message.tool_calls -%}{{- '[call]' + call.function.name -}}{%- endfor -%}{{- '!' -}}{%- endif -%}{%- endfor -%}{%- if add_generation_prompt -%}{{- '[assistant]' -}}{%- endif -%}",
            ),
            messages: Vec::new(),
            committed_token_ids: Vec::new(),
        };
        let prepared = session
            .prepare_user_turn("find it", &CharacterCodec)
            .unwrap();
        let assistant_message = serde_json::json!({
            "role": "assistant",
            "content": "answer",
            "reasoning_content": "private reasoning",
            "tool_calls": [{
                "type": "function",
                "function": {
                    "name": "lookup",
                    "arguments": "{\"city\":\"Athens\"}",
                },
            }],
        });

        let (_, canonical) = session
            .render_assistant_commit_token_delta(
                &prepared,
                "find it",
                &assistant_message,
                &CharacterCodec,
            )
            .unwrap();
        let rendered = CharacterCodec.decode_tokens(&canonical).unwrap();
        assert!(rendered.contains("[assistant]answer[call]lookup!"));
        assert!(!rendered.contains("private reasoning"));

        session.commit_assistant_turn("find it", &assistant_message, canonical);
        assert_eq!(session.messages[1], assistant_message);
        assert!(
            !session
                .prepare_user_turn("next", &CharacterCodec)
                .unwrap()
                .user_token_delta
                .is_empty()
        );
    }

    #[test]
    fn recoverable_chat_turn_rejection_preserves_the_canonical_session() {
        fn reject(
            _: usize,
            _: &RuntimeChatSession,
            _: &str,
            _: &nerve_runtime::RuntimePreparedChatTurn,
        ) -> Result<super::RuntimeChatTurn, Box<dyn std::error::Error>> {
            Err(Box::new(RuntimeRecoverableChatTurnError::new(
                "generated assistant protocol validation failed before canonical commit",
                Box::new(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    "reserved token",
                )),
            )))
        }

        let mut session = RuntimeChatSession {
            formatter: formatter(
                "{%- for message in messages -%}{{- '[' + message.role + ']' + message.content + '!' -}}{%- endfor -%}{%- if add_generation_prompt -%}{{- '[assistant]' -}}{%- endif -%}",
            ),
            messages: Vec::new(),
            committed_token_ids: Vec::new(),
        };
        let messages_before = session.messages.clone();
        let tokens_before = session.committed_token_ids.clone();
        let mut submit = reject;

        let outcome = submit_chat_turn(
            &mut session,
            &CharacterCodec,
            &CharacterCodec,
            &mut submit,
            7,
            "try this turn",
        )
        .unwrap();

        assert_eq!(outcome, RuntimeChatTurnOutcome::Rejected);
        assert_eq!(session.messages, messages_before);
        assert_eq!(session.committed_token_ids, tokens_before);
    }

    #[test]
    fn chat_continuation_is_not_confused_by_delimiters_inside_user_content() {
        let mut session = RuntimeChatSession {
            formatter: formatter(
                "{%- for message in messages -%}{{- '[' + message.role + ']' + message.content + '<stop>' -}}{%- endfor -%}{%- if add_generation_prompt -%}{{- '[assistant]' -}}{%- endif -%}",
            ),
            messages: Vec::new(),
            committed_token_ids: Vec::new(),
        };
        let first = session.prepare_user_turn("first", &CharacterCodec).unwrap();
        let assistant_message = serde_json::json!({
            "role": "assistant",
            "content": "answer containing <stop> text",
        });
        let (_, canonical) = session
            .render_assistant_commit_token_delta(
                &first,
                "first",
                &assistant_message,
                &CharacterCodec,
            )
            .unwrap();
        session.commit_assistant_turn("first", &assistant_message, canonical);

        let user_content = "new <stop> injection";
        let prepared = session
            .prepare_user_turn(user_content, &CharacterCodec)
            .unwrap();
        assert_eq!(
            CharacterCodec
                .decode_tokens(&prepared.user_token_delta)
                .unwrap(),
            "new <stop> injection<stop>"
        );
        assert_eq!(
            CharacterCodec
                .decode_tokens(&prepared.generation_prompt_token_delta)
                .unwrap(),
            "[assistant]"
        );
    }

    #[test]
    fn assistant_transcript_excludes_trailing_turn_stop_tokens() {
        assert_eq!(assistant_content_token_ids(&[1, 2, 99], &[98, 99]), &[1, 2]);
        assert_eq!(
            assistant_content_token_ids(&[1, 2, 98, 99], &[98, 99]),
            &[1, 2]
        );
    }

    #[test]
    fn protocol_boundary_discards_every_token_completed_after_batched_termination() {
        let mut generated = vec![10, 11, 12, 99, 100, 101];

        normalize_generated_tokens_at_protocol_boundary(&mut generated, Some(3)).unwrap();

        assert_eq!(generated, vec![10, 11, 12]);
    }

    #[test]
    fn protocol_boundary_rejects_a_boundary_missing_from_the_completed_generation() {
        let mut generated = vec![10, 11, 12];

        let error = normalize_generated_tokens_at_protocol_boundary(&mut generated, Some(3))
            .expect_err("a protocol terminator must be present after retained content");

        assert!(
            error
                .to_string()
                .contains("completed generation contains only 3")
        );
        assert_eq!(generated, vec![10, 11, 12]);
    }

    #[test]
    fn model_owned_assistant_turn_delimiter_is_discovered_generically() {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let tokenizer_dir = std::env::temp_dir().join(format!(
            "nerve-chat-delimiter-{}-{unique}",
            std::process::id()
        ));
        fs::create_dir_all(&tokenizer_dir).unwrap();
        let mut tokenizer = Tokenizer::new(WordLevel::default());
        tokenizer
            .add_special_tokens([AddedToken::from("<end_of_turn>", true)])
            .unwrap();
        tokenizer
            .save(tokenizer_dir.join("tokenizer.json"), false)
            .unwrap();
        let formatter = formatter(
            "{%- for message in messages %}{{- message.content }}{%- if message.role == 'assistant' %}{{- '<end_of_turn>' }}{%- endif %}{%- endfor %}",
        );

        assert_eq!(
            model_owned_assistant_turn_stop_token_id(&tokenizer_dir, &formatter).unwrap(),
            Some(0)
        );

        fs::remove_dir_all(tokenizer_dir).unwrap();
    }

    #[test]
    fn configured_model_owned_assistant_turn_delimiter_is_discovered() {
        let Some(tokenizer_dir) = std::env::var_os("NERVE_TEST_CHAT_TOKENIZER_DIR") else {
            return;
        };
        let expected = std::env::var("NERVE_TEST_CHAT_STOP_ID")
            .expect("NERVE_TEST_CHAT_STOP_ID must accompany NERVE_TEST_CHAT_TOKENIZER_DIR")
            .parse::<u32>()
            .expect("NERVE_TEST_CHAT_STOP_ID must be a u32");
        let tokenizer_dir = std::path::PathBuf::from(tokenizer_dir);
        let formatter =
            RuntimeChatFormatter::from_tokenizer_dir(&tokenizer_dir, &BTreeMap::new()).unwrap();

        assert_eq!(
            model_owned_assistant_turn_stop_token_id(&tokenizer_dir, &formatter).unwrap(),
            Some(expected)
        );
    }

    #[test]
    fn configured_chat_template_supports_structural_multi_turn_continuation() {
        let Some(tokenizer_dir) = std::env::var_os("NERVE_TEST_CHAT_TOKENIZER_DIR") else {
            return;
        };
        let tokenizer_dir = std::path::PathBuf::from(tokenizer_dir);
        let mut session =
            RuntimeChatSession::from_tokenizer_dir(&tokenizer_dir, &BTreeMap::new()).unwrap();
        let codec = chat_transcript_codec(&tokenizer_dir).unwrap();
        let first = session
            .prepare_user_turn("Explain the result.", &codec)
            .unwrap();
        let generated = "private reasoning</think>The result is four.";
        let assistant_message = session
            .formatter
            .parse_assistant_completion(generated, true)
            .unwrap();
        let (_, canonical) = session
            .render_assistant_commit_token_delta(
                &first,
                "Explain the result.",
                &assistant_message,
                &codec,
            )
            .unwrap();
        session.commit_assistant_turn("Explain the result.", &assistant_message, canonical);

        let next = session
            .prepare_user_turn(
                "Why? Include <|im_end|> literally in this question.",
                &codec,
            )
            .unwrap();
        assert!(!next.user_token_delta.is_empty());
        let decoded = codec.decode_tokens(&next.user_token_delta).unwrap();
        assert!(decoded.contains("Why?"), "{decoded:?}");
    }

    #[test]
    fn configured_chat_template_honors_non_thinking_variable() {
        let Some(tokenizer_dir) = std::env::var_os("NERVE_TEST_CHAT_TOKENIZER_DIR") else {
            return;
        };
        let tokenizer_dir = std::path::PathBuf::from(tokenizer_dir);
        let variables = BTreeMap::from([(
            "enable_thinking".to_string(),
            serde_json::Value::Bool(false),
        )]);
        let session = RuntimeChatSession::from_tokenizer_dir(&tokenizer_dir, &variables).unwrap();
        let codec = chat_transcript_codec(&tokenizer_dir).unwrap();

        let prepared = session
            .prepare_user_turn("Answer directly.", &codec)
            .unwrap();
        let prompt_ids = [
            prepared.user_token_delta.as_slice(),
            prepared.generation_prompt_token_delta.as_slice(),
        ]
        .concat();
        let rendered = codec.decode_tokens(&prompt_ids).unwrap();

        assert!(rendered.contains("<think>\n\n</think>\n\n"), "{rendered:?}");
    }

    #[test]
    fn sustained_decode_report_exposes_early_and_late_context_windows() {
        let samples = (0..8)
            .map(|index| RuntimeSustainedDecodeSample {
                context_activation: 32_000 + index,
                transient_state_activation: 31_000 + index as u64,
                inter_token_time_ns: if index < 4 { 10_000_000 } else { 20_000_000 },
            })
            .collect::<Vec<_>>();

        let report = RuntimeSustainedDecodeReport::from_samples(&samples);

        assert_eq!(report.measured_token_count, 8);
        assert_eq!(report.windows.len(), 4);
        assert_eq!(
            report.windows.first().unwrap().context_activation_start,
            32_000
        );
        assert_eq!(
            report.windows.last().unwrap().context_activation_end,
            32_007
        );
        assert!(
            report.windows.last().unwrap().elapsed_ns > report.windows.first().unwrap().elapsed_ns
        );
    }

    #[test]
    fn critical_path_output_separates_device_timestamps_and_omits_empty_phases() {
        let report = nerve_runtime::RuntimeCriticalPathReport {
            wall_duration_ns: 10_000_000,
            host_exclusive_work_duration_ns: 9_500_000,
            host_attributed_critical_path_duration_ns: 9_500_000,
            host_unattributed_duration_ns: 500_000,
            host_parallel_overlap_duration_ns: 0,
            host_coverage_basis_points: 9_500,
            device_timestamp_duration_ns: 20_000_000,
            generated_token_count: 2,
            execution_window_count: 1,
            device_sampled_execution_window_count: 1,
            phases: vec![
                nerve_runtime::RuntimeCriticalPathPhaseReport {
                    phase: "queue_submission".to_string(),
                    host_invocation_count: 2,
                    host_exclusive_duration_ns: 1_000_000,
                    host_inclusive_duration_ns: 1_500_000,
                    host_max_inclusive_duration_ns: 900_000,
                    device_timestamp_count: 0,
                    device_duration_ns: 0,
                    device_max_duration_ns: 0,
                    host_exclusive_per_generated_token_ns: Some(500_000),
                    device_per_generated_token_ns: Some(0),
                    host_exclusive_per_execution_window_ns: Some(1_000_000),
                    device_per_execution_window_ns: Some(0),
                    device_per_sampled_execution_window_ns: Some(0),
                },
                nerve_runtime::RuntimeCriticalPathPhaseReport {
                    phase: "unused".to_string(),
                    ..Default::default()
                },
            ],
        };

        let lines = runtime_critical_path_lines(&report);

        assert!(lines.iter().any(|line| line.contains("coverage=95.00%")));
        assert!(lines.iter().any(|line| {
            line.contains("reported separately") && line.contains("device intervals may overlap")
        }));
        assert!(
            lines
                .iter()
                .any(|line| line.contains("device_detail_sampled_windows=1"))
        );
        assert!(
            lines
                .iter()
                .any(|line| line.contains("phase=queue_submission"))
        );
        assert!(!lines.iter().any(|line| line.contains("phase=unused")));
    }

    #[test]
    fn distributed_execution_output_uses_the_conversation_gate_schema() {
        let lines = runtime_distributed_execution_phase_counter_lines(
            "prefill",
            &VulkanResidentDistributedExecutionPhaseCounters {
                island_submissions: 11,
                shard_submissions: 23,
                tensor_parallel_island_submissions: 2,
                whole_expert_parallel_island_submissions: 3,
                intra_expert_tensor_parallel_island_submissions: 5,
                hybrid_island_submissions: 1,
            },
        );

        assert_eq!(
            lines,
            vec![
                "  distributed_prefill_island_submissions=11",
                "  distributed_prefill_shard_submissions=23",
                "  distributed_prefill_tensor_parallel_island_submissions=2",
                "  distributed_prefill_whole_expert_parallel_island_submissions=3",
                "  distributed_prefill_intra_expert_tensor_parallel_island_submissions=5",
                "  distributed_prefill_hybrid_island_submissions=1",
            ]
        );
    }
}
