use super::{RuntimeExecutionEnvelope, RuntimeImplementationPredicate, RuntimeSelectionDevice};
use crate::{HardwareProcessAvailability, HardwareProcessProgrammability};
use std::collections::{BTreeMap, BTreeSet};

impl RuntimeImplementationPredicate {
    pub fn validate(&self) -> Result<(), String> {
        if self.schema != super::RUNTIME_IMPLEMENTATION_PREDICATE_SCHEMA {
            return Err(format!(
                "unsupported runtime implementation predicate schema {:?}",
                self.schema
            ));
        }
        if self.predicate_id.is_empty()
            || self.hardware.capability_classes.is_empty()
            || self.hardware.device_kinds.is_empty()
            || self.hardware.apis.is_empty()
            || self.execution.phases.is_empty()
        {
            return Err("runtime implementation predicate is incomplete".to_string());
        }
        for values in [
            &self.hardware.device_kinds,
            &self.hardware.apis,
            &self.hardware.capability_classes,
            &self.hardware.required_processes,
            &self.hardware.required_features,
            &self.execution.phases,
            &self.execution.alternative_phases,
            &self.execution.source_retained_phases,
            &self.placement.required_interconnects,
        ] {
            if !sorted_unique(values) {
                return Err(
                    "runtime implementation predicate lists must be sorted and unique".to_string(),
                );
            }
        }
        let phases = self
            .execution
            .phases
            .iter()
            .map(String::as_str)
            .collect::<BTreeSet<_>>();
        let alternative_phases = self
            .execution
            .alternative_phases
            .iter()
            .map(String::as_str)
            .collect::<BTreeSet<_>>();
        let source_retained_phases = self
            .execution
            .source_retained_phases
            .iter()
            .map(String::as_str)
            .collect::<BTreeSet<_>>();
        if alternative_phases.is_empty()
            || !alternative_phases.is_disjoint(&source_retained_phases)
            || alternative_phases
                .union(&source_retained_phases)
                .copied()
                .collect::<BTreeSet<_>>()
                != phases
        {
            return Err(
                "runtime execution phases must be partitioned into alternative and source-retained phases"
                    .to_string(),
            );
        }
        let measured_device_count = self.placement.maximum_device_count;
        if self.placement.minimum_device_count == 0
            || self.placement.minimum_device_count > measured_device_count
        {
            return Err("runtime device-count range is invalid".to_string());
        }
        for range in [
            self.execution.activation_batch,
            self.execution.context_activations,
            self.execution.state_activations,
        ] {
            if range.minimum > range.maximum {
                return Err("runtime execution predicate contains an inverted range".to_string());
            }
        }
        if self.execution.activation_batch.minimum == 0 {
            return Err("runtime activation batch must be positive".to_string());
        }
        if self.execution.speculative_draft_token_counts.is_empty()
            || !self
                .execution
                .speculative_draft_token_counts
                .windows(2)
                .all(|pair| pair[0] < pair[1])
        {
            return Err(
                "runtime speculative draft-token counts must be nonempty, sorted, and unique"
                    .to_string(),
            );
        }
        match self.placement.mode.as_str() {
            "local" if measured_device_count == 1 => {}
            "distributed" if measured_device_count >= 2 => {}
            "either" => {}
            _ => {
                return Err(format!(
                    "runtime placement predicate {:?} conflicts with its device count",
                    self.placement.mode
                ));
            }
        }
        Ok(())
    }

    pub fn mismatch_reasons(
        &self,
        execution: &RuntimeExecutionEnvelope,
        devices: &[&RuntimeSelectionDevice],
    ) -> Vec<String> {
        let mut reasons = Vec::new();
        if let Err(error) = self.validate() {
            reasons.push(error);
            return reasons;
        }
        for phase in &execution.phases {
            if !self.execution.phases.contains(phase)
                && !self
                    .execution
                    .phases
                    .iter()
                    .any(|candidate| candidate == "mixed")
            {
                reasons.push(format!(
                    "execution phase {phase:?} is outside {:?}",
                    self.execution.phases
                ));
            }
        }
        if !execution
            .phases
            .iter()
            .any(|phase| self.execution.alternative_phases.contains(phase))
        {
            reasons.push("runtime request does not execute an alternative phase".to_string());
        }
        if !self
            .execution
            .speculative_draft_token_counts
            .contains(&execution.speculative_draft_tokens)
        {
            reasons.push(format!(
                "speculative draft-token count {} is outside {:?}",
                execution.speculative_draft_tokens, self.execution.speculative_draft_token_counts
            ));
        }
        for (label, predicate_range, requested_range) in [
            (
                "activation batch",
                self.execution.activation_batch,
                execution.activation_batch,
            ),
            (
                "context activations",
                self.execution.context_activations,
                execution.context_activations,
            ),
            (
                "state activations",
                self.execution.state_activations,
                execution.state_activations,
            ),
        ] {
            if predicate_range.minimum > requested_range.minimum
                || predicate_range.maximum < requested_range.maximum
            {
                reasons.push(format!(
                    "{label} envelope {}..={} is outside {}..={}",
                    requested_range.minimum,
                    requested_range.maximum,
                    predicate_range.minimum,
                    predicate_range.maximum
                ));
            }
        }

        let unique_devices = devices
            .iter()
            .map(|device| (device.physical_device_id.as_str(), device.profile.clone()))
            .collect::<BTreeMap<_, _>>();
        let device_count = unique_devices.len();
        if device_count < self.placement.minimum_device_count
            || device_count > self.placement.maximum_device_count
        {
            reasons.push(format!(
                "physical device count {device_count} is outside {}..={}",
                self.placement.minimum_device_count, self.placement.maximum_device_count
            ));
        }
        match self.placement.mode.as_str() {
            "local" if device_count != 1 => {
                reasons.push("local implementation spans more than one physical device".to_string())
            }
            "distributed" if device_count < 2 => {
                reasons.push("distributed implementation has no cross-device placement".to_string())
            }
            _ => {}
        }

        let actual_capabilities = unique_devices
            .values()
            .map(|profile| profile.capability_class.as_str())
            .collect::<BTreeSet<_>>();
        let allowed_capabilities = self
            .hardware
            .capability_classes
            .iter()
            .map(String::as_str)
            .collect::<BTreeSet<_>>();
        if !actual_capabilities.is_subset(&allowed_capabilities) {
            reasons.push(format!(
                "capability classes {actual_capabilities:?} are outside {allowed_capabilities:?}"
            ));
        }

        let actual_kinds = unique_devices
            .values()
            .map(|profile| profile.hardware_identity.device_kind.as_str())
            .collect::<BTreeSet<_>>();
        let allowed_kinds = self
            .hardware
            .device_kinds
            .iter()
            .map(String::as_str)
            .collect::<BTreeSet<_>>();
        if !actual_kinds.is_subset(&allowed_kinds) {
            reasons.push(format!(
                "device kinds {actual_kinds:?} are outside {allowed_kinds:?}"
            ));
        }
        let actual_apis = unique_devices
            .values()
            .map(|profile| profile.provenance.api.as_str())
            .collect::<BTreeSet<_>>();
        let allowed_apis = self
            .hardware
            .apis
            .iter()
            .map(String::as_str)
            .collect::<BTreeSet<_>>();
        if !actual_apis.is_subset(&allowed_apis) {
            reasons.push(format!(
                "device APIs {actual_apis:?} are outside {allowed_apis:?}"
            ));
        }

        let processes = unique_devices
            .values()
            .flat_map(|profile| profile.processes.iter())
            .filter(|process| {
                process.availability == HardwareProcessAvailability::Available
                    && process.programmability != HardwareProcessProgrammability::None
            })
            .collect::<Vec<_>>();
        let available_processes = processes
            .iter()
            .map(|process| process.name.as_str())
            .collect::<BTreeSet<_>>();
        for required in &self.hardware.required_processes {
            if !available_processes.contains(required.as_str()) {
                reasons.push(format!(
                    "required hardware process {required:?} is unavailable"
                ));
            }
        }

        let mut available_features = BTreeSet::new();
        for process in processes {
            for value in &process.operations {
                available_features.insert(value.as_str());
            }
            for value in &process.numeric_formats {
                available_features.insert(value.as_str());
            }
            for value in &process.required_extensions {
                available_features.insert(value.as_str());
            }
            for value in &process.required_features {
                available_features.insert(value.as_str());
            }
        }
        for profile in unique_devices.values() {
            available_features.extend(profile.capability_extensions.keys().map(String::as_str));
        }
        for required in &self.hardware.required_features {
            if !available_features.contains(required.as_str()) {
                reasons.push(format!(
                    "required hardware feature {required:?} is unavailable"
                ));
            }
        }

        let mut available_interconnects = BTreeSet::new();
        for profile in unique_devices.values() {
            for interconnect in &profile.interconnects {
                if interconnect.availability == HardwareProcessAvailability::Available {
                    available_interconnects.insert(interconnect.name.as_str());
                    available_interconnects.insert(interconnect.kind.as_str());
                    available_interconnects.insert(interconnect.api.as_str());
                    available_interconnects
                        .extend(interconnect.operations.iter().map(String::as_str));
                }
            }
        }
        for required in &self.placement.required_interconnects {
            if !available_interconnects.contains(required.as_str()) {
                reasons.push(format!("required interconnect {required:?} is unavailable"));
            }
        }
        reasons
    }
}

fn sorted_unique(values: &[String]) -> bool {
    values.windows(2).all(|pair| pair[0] < pair[1])
}
