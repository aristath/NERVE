fn main() {
    if let Err(error) = run() {
        eprintln!("nerve-runtime error: {error}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), Box<dyn Error>> {
    if std::env::args()
        .skip(1)
        .eq(["--runtime-implementation-fingerprint"])
    {
        println!("{}", nerve_runtime::RUNTIME_IMPLEMENTATION_FINGERPRINT);
        return Ok(());
    }
    if std::env::args()
        .skip(1)
        .eq(["--runtime-device-local-memory-policy"])
    {
        println!(
            "{}",
            serde_json::to_string(&nerve_runtime::vulkan_device_local_memory_policy())?
        );
        return Ok(());
    }
    if std::env::args()
        .skip(1)
        .any(|arg| arg == "--help" || arg == "-h")
    {
        print_usage();
        return Ok(());
    }

    let args = parse_args().map_err(|error| io::Error::new(io::ErrorKind::InvalidInput, error))?;
    if args.inspect_devices {
        return inspect_device_capabilities(&args);
    }
    let package_manifest = args.package_manifest.as_ref().ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::InvalidInput,
            "--package is required; run `python -m nerve --compile-model <MODEL_DIR>` first",
        )
    })?;
    let manifest_dir = package_manifest
        .parent()
        .map(Path::to_path_buf)
        .unwrap_or_else(|| PathBuf::from("."));
    let manifest = VulkanResidentModelPackageManifest::from_json_file(package_manifest)?;
    if args.inspect_runtime {
        return inspect_runtime_topology(&args, package_manifest, &manifest_dir, manifest);
    }
    if args.inspect_package {
        return inspect_package(&args, package_manifest, &manifest_dir, manifest);
    }
    if args.inspect_graph {
        return inspect_graph(&args, package_manifest, &manifest_dir, manifest);
    }
    let max_context_activations = manifest.max_context_activations;
    let tokenizer_dir = tokenizer_dir_from_manifest(&manifest_dir, &manifest)?;
    let runtime_model = runtime_model_from_manifest(&args, manifest)?;
    if args.inspect_placement {
        return inspect_placement(&args, package_manifest, &manifest_dir, runtime_model);
    }
    if let Some(device_id) = args.inspect_device_slice.as_deref() {
        return inspect_device_slice(
            &args,
            package_manifest,
            &manifest_dir,
            runtime_model,
            device_id,
        );
    }
    let codec = VulkanResidentHfTokenizerTextCodec::from_model_dir(&tokenizer_dir)?
        .with_add_special_tokens(args.add_special_tokens)
        .with_skip_special_tokens(args.skip_special_tokens);
    if args.chat {
        let capacity =
            choose_chat_runtime_context_size(max_context_activations, args.context_size)?;
        let prepared = runtime_capacity_packed_model(
            &args,
            &manifest_dir,
            runtime_model,
            capacity,
        )?;
        return run_placed_chat(
            &args,
            &manifest_dir,
            &tokenizer_dir,
            prepared.runtime_model,
            prepared.auto_placement,
            capacity,
            &codec,
            args.prompt.as_deref(),
        );
    }
    let prompt = args
        .prompt
        .as_ref()
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "--prompt is required"))?;
    let prompt_ids = codec.encode_text(prompt)?;
    let scheduled_token_activations = prompt_ids
        .len()
        .checked_add(args.max_new_tokens)
        .ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidInput,
                "prompt token count plus --max-new-tokens overflowed usize",
            )
        })?;
    let capacity = choose_runtime_context_size(
        max_context_activations,
        args.context_size,
        prompt_ids.len(),
    )?;
    let prepared = runtime_capacity_packed_model(
        &args,
        &manifest_dir,
        runtime_model,
        capacity,
    )?;
    let context = PromptRunContext {
        args: &args,
        package_manifest,
        manifest_dir: &manifest_dir,
        tokenizer_dir: &tokenizer_dir,
        prompt,
        prompt_ids: &prompt_ids,
        scheduled_token_activations,
        capacity,
        codec: &codec,
    };

    run_placed_prompt(&context, prepared.runtime_model)
}
