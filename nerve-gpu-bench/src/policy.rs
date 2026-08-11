use std::collections::BTreeSet;

use crate::model::{RunPolicy, Selection, SkippedTarget, Target};

pub fn apply_selection_policy(targets: &[Target], policy: &RunPolicy) -> Selection {
    let include_targets = policy.include_targets.iter().collect::<BTreeSet<_>>();
    let exclude_targets = policy.exclude_targets.iter().collect::<BTreeSet<_>>();
    let exclude_pci = policy.exclude_pci.iter().collect::<BTreeSet<_>>();
    let exclude_kinds = policy.exclude_kinds.iter().collect::<BTreeSet<_>>();

    let mut selected_target_ids = Vec::new();
    let mut skipped_targets = Vec::new();

    for target in targets {
        let reason = if target.kind == "unavailable" {
            Some("target_unavailable")
        } else if !include_targets.is_empty() && !include_targets.contains(&target.stable_target_id)
        {
            Some("not_in_include_set")
        } else if exclude_targets.contains(&target.stable_target_id) {
            Some("user_excluded_target")
        } else if target
            .pci_address
            .as_ref()
            .is_some_and(|pci| exclude_pci.contains(pci))
        {
            Some("user_excluded_pci")
        } else if exclude_kinds.contains(&target.kind) {
            Some("user_excluded_kind")
        } else if target.backend != "vulkan" {
            Some("no_vulkan_execution_backend")
        } else {
            None
        };

        if let Some(reason) = reason {
            skipped_targets.push(SkippedTarget {
                stable_target_id: target.stable_target_id.clone(),
                reason: reason.to_string(),
            });
        } else {
            selected_target_ids.push(target.stable_target_id.clone());
        }
    }

    let known_ids = targets
        .iter()
        .map(|target| target.stable_target_id.as_str())
        .collect::<BTreeSet<_>>();
    let known_pci = targets
        .iter()
        .filter_map(|target| target.pci_address.as_deref())
        .collect::<BTreeSet<_>>();
    let known_kinds = targets
        .iter()
        .map(|target| target.kind.as_str())
        .collect::<BTreeSet<_>>();
    let mut diagnostics = Vec::new();

    for requested in &policy.include_targets {
        if !known_ids.contains(requested.as_str()) {
            diagnostics.push(format!(
                "include target {requested:?} did not match any target"
            ));
        }
    }
    for requested in &policy.exclude_targets {
        if !known_ids.contains(requested.as_str()) {
            diagnostics.push(format!(
                "exclude target {requested:?} did not match any target"
            ));
        }
    }
    for requested in &policy.exclude_pci {
        if !known_pci.contains(requested.as_str()) {
            diagnostics.push(format!(
                "exclude PCI address {requested:?} did not match any target"
            ));
        }
    }
    for requested in &policy.exclude_kinds {
        if !known_kinds.contains(requested.as_str()) {
            diagnostics.push(format!(
                "exclude kind {requested:?} did not match any target"
            ));
        }
    }

    Selection {
        selected_target_ids,
        skipped_targets,
        diagnostics,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn target(id: &str, backend: &str, kind: &str, pci: Option<&str>) -> Target {
        Target {
            stable_target_id: id.to_string(),
            backend: backend.to_string(),
            kind: kind.to_string(),
            name: id.to_string(),
            vendor_id: None,
            vendor_name: None,
            device_id: None,
            pci_address: pci.map(str::to_string),
            physical_location: None,
            numa_node: None,
            boot_vga: None,
            pci_link: None,
            vulkan: None,
            capabilities: Vec::new(),
            format_capabilities: Vec::new(),
            diagnostics: Vec::new(),
        }
    }

    #[test]
    fn inclusion_and_exclusion_are_explicit_policy() {
        let targets = [
            target("cpu:host", "cpu", "cpu", None),
            target(
                "vulkan-uuid:11111111111111111111111111111111",
                "vulkan",
                "discrete_gpu",
                Some("0000:01:00.0"),
            ),
            target(
                "vulkan-uuid:22222222222222222222222222222222",
                "vulkan",
                "integrated_gpu",
                Some("0000:02:00.0"),
            ),
        ];
        let policy = RunPolicy {
            payload_bytes: 1024,
            samples: 1,
            benchmark_formats: vec!["f32".to_string()],
            benchmark_workloads: vec!["dense_projection".to_string()],
            include_targets: Vec::new(),
            exclude_targets: Vec::new(),
            exclude_pci: Vec::new(),
            exclude_kinds: vec!["integrated_gpu".to_string()],
            pair_measurements: true,
            max_group_size: 3,
            execute: false,
        };
        let selection = apply_selection_policy(&targets, &policy);
        assert_eq!(
            selection.selected_target_ids,
            ["vulkan-uuid:11111111111111111111111111111111"]
        );
        assert_eq!(selection.skipped_targets.len(), 2);
        assert_eq!(
            selection
                .skipped_targets
                .iter()
                .find(|target| target.stable_target_id == "cpu:host")
                .unwrap()
                .reason,
            "no_vulkan_execution_backend"
        );
        assert_eq!(
            selection
                .skipped_targets
                .iter()
                .find(|target| {
                    target.stable_target_id
                        == "vulkan-uuid:22222222222222222222222222222222"
                })
                .unwrap()
                .reason,
            "user_excluded_kind"
        );
    }

    #[test]
    fn unavailable_targets_are_discovered_but_not_selected() {
        let targets = [
            target("cpu:host", "cpu", "cpu", None),
            target("vulkan:unavailable", "vulkan", "unavailable", None),
        ];
        let policy = RunPolicy {
            payload_bytes: 1024,
            samples: 1,
            benchmark_formats: vec!["f32".to_string()],
            benchmark_workloads: vec!["dense_projection".to_string()],
            include_targets: Vec::new(),
            exclude_targets: Vec::new(),
            exclude_pci: Vec::new(),
            exclude_kinds: Vec::new(),
            pair_measurements: true,
            max_group_size: 3,
            execute: false,
        };
        let selection = apply_selection_policy(&targets, &policy);
        assert!(selection.selected_target_ids.is_empty());
        assert_eq!(selection.skipped_targets.len(), 2);
        assert_eq!(
            selection
                .skipped_targets
                .iter()
                .find(|target| target.stable_target_id == "vulkan:unavailable")
                .unwrap()
                .reason,
            "target_unavailable"
        );
    }
}
