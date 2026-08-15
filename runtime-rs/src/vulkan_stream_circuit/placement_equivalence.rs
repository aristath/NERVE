#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum VulkanPlacementEquivalenceKind {
    BitExact,
    AbsoluteRelativeTolerance,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum VulkanPlacementScalarFormat {
    Bf16,
    F32,
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct VulkanPlacementEquivalenceIdentity {
    pub output: VulkanPlacementEquivalenceKind,
    pub state: VulkanPlacementEquivalenceKind,
    pub absolute_tolerance_bits: Option<u64>,
    pub relative_tolerance_bits: Option<u64>,
    pub output_scalar_format: Option<VulkanPlacementScalarFormat>,
}

impl VulkanPlacementEquivalenceIdentity {
    pub fn bit_exact() -> Self {
        Self {
            output: VulkanPlacementEquivalenceKind::BitExact,
            state: VulkanPlacementEquivalenceKind::BitExact,
            absolute_tolerance_bits: None,
            relative_tolerance_bits: None,
            output_scalar_format: None,
        }
    }

    pub fn absolute_tolerance(&self) -> Option<f64> {
        self.absolute_tolerance_bits.map(f64::from_bits)
    }

    pub fn relative_tolerance(&self) -> Option<f64> {
        self.relative_tolerance_bits.map(f64::from_bits)
    }

    fn validate(&self) -> Result<(), VulkanPlacementCalibrationCatalogError> {
        let tolerant = self.output == VulkanPlacementEquivalenceKind::AbsoluteRelativeTolerance
            || self.state == VulkanPlacementEquivalenceKind::AbsoluteRelativeTolerance;
        let valid_tolerances = self
            .absolute_tolerance()
            .zip(self.relative_tolerance())
            .is_some_and(|(absolute, relative)| {
                absolute.is_finite() && relative.is_finite() && absolute >= 0.0 && relative >= 0.0
            });
        if tolerant != valid_tolerances
            || (self.output == VulkanPlacementEquivalenceKind::AbsoluteRelativeTolerance)
                != self.output_scalar_format.is_some()
        {
            return Err(VulkanPlacementCalibrationCatalogError(
                "placement equivalence identity has inconsistent tolerance or scalar-format metadata"
                    .to_string(),
            ));
        }
        if self.state == VulkanPlacementEquivalenceKind::AbsoluteRelativeTolerance {
            return Err(VulkanPlacementCalibrationCatalogError(
                "numeric state equivalence is unavailable without a typed compiled state layout"
                    .to_string(),
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct VulkanPlacementOutputSegment {
    pub binding: usize,
    pub name: String,
    pub bytes: Vec<u8>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct VulkanPlacementOutputArtifact {
    pub scalar_format: VulkanPlacementScalarFormat,
    pub segments: Vec<VulkanPlacementOutputSegment>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum VulkanPlacementOutputEquivalenceEvidence {
    BitExact,
    AbsoluteRelativeTolerance {
        compared_element_count: usize,
        maximum_absolute_error_bits: u64,
        maximum_relative_error_bits: u64,
    },
}

fn validate_output_artifact(
    artifact: &VulkanPlacementOutputArtifact,
) -> Result<(), VulkanPlacementCalibrationCatalogError> {
    let scalar_bytes = match artifact.scalar_format {
        VulkanPlacementScalarFormat::Bf16 => 2,
        VulkanPlacementScalarFormat::F32 => 4,
    };
    if artifact.segments.is_empty()
        || artifact.segments.iter().any(|segment| {
            segment.name.is_empty()
                || segment.bytes.is_empty()
                || !segment.bytes.len().is_multiple_of(scalar_bytes)
        })
        || !artifact.segments.windows(2).all(|pair| {
            (pair[0].binding, pair[0].name.as_str()) < (pair[1].binding, pair[1].name.as_str())
        })
    {
        return Err(VulkanPlacementCalibrationCatalogError(
            "placement numeric output artifact is empty, unaligned, or non-canonical".to_string(),
        ));
    }
    Ok(())
}

pub fn vulkan_placement_output_artifact_digest(
    artifact: &VulkanPlacementOutputArtifact,
) -> Result<String, VulkanPlacementCalibrationCatalogError> {
    validate_output_artifact(artifact)?;
    let mut digest = Sha256::new();
    for segment in &artifact.segments {
        digest.update(segment.binding.to_le_bytes());
        digest.update(segment.name.as_bytes());
        digest.update(&segment.bytes);
    }
    Ok(targeted_finalized_artifact_digest(
        digest.finalize().as_slice(),
    ))
}

pub fn validate_vulkan_placement_output_equivalence(
    equivalence: &VulkanPlacementEquivalenceIdentity,
    reference_digest: &str,
    reference_artifact: Option<&VulkanPlacementOutputArtifact>,
    candidate_digest: &str,
    candidate_artifact: Option<&VulkanPlacementOutputArtifact>,
) -> Result<VulkanPlacementOutputEquivalenceEvidence, VulkanPlacementCalibrationCatalogError> {
    equivalence.validate()?;
    match equivalence.output {
        VulkanPlacementEquivalenceKind::BitExact => {
            if let Some(reference) = reference_artifact {
                validate_output_artifact(reference)?;
                if vulkan_placement_output_artifact_digest(reference)? != reference_digest {
                    return Err(VulkanPlacementCalibrationCatalogError(
                        "bit-exact canonical output artifact disagrees with its digest".to_string(),
                    ));
                }
            }
            if let Some(candidate) = candidate_artifact {
                validate_output_artifact(candidate)?;
                if vulkan_placement_output_artifact_digest(candidate)? != candidate_digest {
                    return Err(VulkanPlacementCalibrationCatalogError(
                        "bit-exact candidate output artifact disagrees with its digest".to_string(),
                    ));
                }
            }
            if reference_digest != candidate_digest {
                if let (Some(reference), Some(candidate)) =
                    (reference_artifact, candidate_artifact)
                {
                    if reference.scalar_format != candidate.scalar_format
                        || reference.segments.len() != candidate.segments.len()
                    {
                        return Err(VulkanPlacementCalibrationCatalogError(
                            "bit-exact placement output has different typed shapes".to_string(),
                        ));
                    }
                    let mut compared_element_count = 0usize;
                    let mut differing_element_count = 0usize;
                    let mut maximum_absolute_error = 0.0f64;
                    let mut maximum_relative_error = 0.0f64;
                    let mut worst_difference = None;
                    for (segment_index, (reference_segment, candidate_segment)) in reference
                        .segments
                        .iter()
                        .zip(&candidate.segments)
                        .enumerate()
                    {
                        if reference_segment.binding != candidate_segment.binding
                            || reference_segment.name != candidate_segment.name
                            || reference_segment.bytes.len() != candidate_segment.bytes.len()
                        {
                            return Err(VulkanPlacementCalibrationCatalogError(
                                "bit-exact placement output has different segment shapes"
                                    .to_string(),
                            ));
                        }
                        for (element_index, (expected, actual)) in placement_output_values(
                            reference.scalar_format,
                            &reference_segment.bytes,
                        )
                        .zip(placement_output_values(
                            candidate.scalar_format,
                            &candidate_segment.bytes,
                        ))
                        .enumerate()
                        {
                            compared_element_count += 1;
                            if expected.to_bits() == actual.to_bits() {
                                continue;
                            }
                            differing_element_count += 1;
                            let absolute_error = f64::from((actual - expected).abs());
                            let relative_error = if expected == 0.0 {
                                f64::INFINITY
                            } else {
                                absolute_error / f64::from(expected.abs())
                            };
                            maximum_relative_error =
                                maximum_relative_error.max(relative_error);
                            if absolute_error > maximum_absolute_error {
                                maximum_absolute_error = absolute_error;
                                worst_difference = Some((
                                    segment_index,
                                    reference_segment.binding,
                                    reference_segment.name.clone(),
                                    element_index,
                                    expected,
                                    actual,
                                    absolute_error,
                                    relative_error,
                                ));
                            }
                        }
                    }
                    if let Some((
                        segment_index,
                        binding,
                        name,
                        element_index,
                        expected,
                        actual,
                        absolute_error,
                        relative_error,
                    )) = worst_difference
                    {
                        return Err(VulkanPlacementCalibrationCatalogError(format!(
                            "bit-exact placement output differs from its canonical reference: reference_digest={reference_digest:?}, candidate_digest={candidate_digest:?}, differing_element_count={differing_element_count}, compared_element_count={compared_element_count}, maximum_absolute_error={maximum_absolute_error}, maximum_relative_error={maximum_relative_error}, segment={segment_index}, binding={binding}, name={name:?}, element={element_index}, expected={expected}, actual={actual}, absolute_error={absolute_error}, relative_error={relative_error}",
                        )));
                    }
                }
                return Err(VulkanPlacementCalibrationCatalogError(
                    format!(
                        "bit-exact placement output differs from its canonical reference: reference_digest={reference_digest:?}, candidate_digest={candidate_digest:?}"
                    ),
                ));
            }
            Ok(VulkanPlacementOutputEquivalenceEvidence::BitExact)
        }
        VulkanPlacementEquivalenceKind::AbsoluteRelativeTolerance => {
            let reference = reference_artifact.ok_or_else(|| {
                VulkanPlacementCalibrationCatalogError(
                    "tolerant placement output has no canonical numeric artifact".to_string(),
                )
            })?;
            let candidate = candidate_artifact.ok_or_else(|| {
                VulkanPlacementCalibrationCatalogError(
                    "tolerant placement output has no candidate numeric artifact".to_string(),
                )
            })?;
            validate_output_artifact(reference)?;
            validate_output_artifact(candidate)?;
            if vulkan_placement_output_artifact_digest(reference)? != reference_digest
                || vulkan_placement_output_artifact_digest(candidate)? != candidate_digest
            {
                return Err(VulkanPlacementCalibrationCatalogError(
                    "tolerant placement output digest does not identify its numeric artifact"
                        .to_string(),
                ));
            }
            if reference.scalar_format != equivalence.output_scalar_format.unwrap()
                || candidate.scalar_format != reference.scalar_format
                || reference.segments.len() != candidate.segments.len()
            {
                return Err(VulkanPlacementCalibrationCatalogError(
                    "tolerant placement output artifact disagrees with its compiled scalar layout"
                        .to_string(),
                ));
            }
            let absolute_tolerance = equivalence.absolute_tolerance().unwrap();
            let relative_tolerance = equivalence.relative_tolerance().unwrap();
            let mut compared_element_count = 0usize;
            let mut maximum_absolute_error = 0.0f64;
            let mut maximum_relative_error = 0.0f64;
            let mut exceeded_element_count = 0usize;
            let mut worst_excess_ratio = 0.0f64;
            let mut worst_violation = None;
            for (segment_index, (reference_segment, candidate_segment)) in reference
                .segments
                .iter()
                .zip(&candidate.segments)
                .enumerate()
            {
                if reference_segment.binding != candidate_segment.binding
                    || reference_segment.name != candidate_segment.name
                    || reference_segment.bytes.len() != candidate_segment.bytes.len()
                {
                    return Err(VulkanPlacementCalibrationCatalogError(
                        "tolerant placement output segments do not have the same typed shape"
                            .to_string(),
                    ));
                }
                let reference_values =
                    placement_output_values(reference.scalar_format, &reference_segment.bytes);
                let candidate_values =
                    placement_output_values(candidate.scalar_format, &candidate_segment.bytes);
                for (element_index, (expected, actual)) in
                    reference_values.zip(candidate_values).enumerate()
                {
                    if expected.to_bits() == actual.to_bits() {
                        compared_element_count += 1;
                        continue;
                    }
                    if !expected.is_finite() || !actual.is_finite() {
                        return Err(VulkanPlacementCalibrationCatalogError(
                            format!(
                                "tolerant placement output introduced a non-finite mismatch: segment={segment_index}, binding={}, name={:?}, element={element_index}, expected={expected}, actual={actual}",
                                reference_segment.binding,
                                reference_segment.name,
                            ),
                        ));
                    }
                    let absolute_error = f64::from((actual - expected).abs());
                    let relative_error = if expected == 0.0 {
                        if absolute_error == 0.0 {
                            0.0
                        } else {
                            f64::INFINITY
                        }
                    } else {
                        absolute_error / f64::from(expected.abs())
                    };
                    maximum_absolute_error = maximum_absolute_error.max(absolute_error);
                    maximum_relative_error = maximum_relative_error.max(relative_error);
                    if absolute_error
                        > absolute_tolerance + relative_tolerance * f64::from(expected.abs())
                    {
                        let allowed_error =
                            absolute_tolerance + relative_tolerance * f64::from(expected.abs());
                        exceeded_element_count += 1;
                        let excess_ratio = absolute_error / allowed_error;
                        if excess_ratio > worst_excess_ratio {
                            worst_excess_ratio = excess_ratio;
                            worst_violation = Some((
                                segment_index,
                                reference_segment.binding,
                                reference_segment.name.clone(),
                                element_index,
                                expected,
                                actual,
                                absolute_error,
                                relative_error,
                                allowed_error,
                            ));
                        }
                    }
                    compared_element_count += 1;
                }
            }
            if compared_element_count == 0 {
                return Err(VulkanPlacementCalibrationCatalogError(
                    "tolerant placement output compared no values".to_string(),
                ));
            }
            if let Some((
                segment_index,
                binding,
                name,
                element_index,
                expected,
                actual,
                absolute_error,
                relative_error,
                allowed_error,
            )) = worst_violation
            {
                return Err(VulkanPlacementCalibrationCatalogError(format!(
                    "tolerant placement output exceeds its compiled absolute-relative bound: exceeded_element_count={exceeded_element_count}, compared_element_count={compared_element_count}, maximum_absolute_error={maximum_absolute_error}, maximum_relative_error={maximum_relative_error}, worst_excess_ratio={worst_excess_ratio}, segment={segment_index}, binding={binding}, name={name:?}, element={element_index}, expected={expected}, actual={actual}, absolute_error={absolute_error}, relative_error={relative_error}, allowed_error={allowed_error}",
                )));
            }
            Ok(
                VulkanPlacementOutputEquivalenceEvidence::AbsoluteRelativeTolerance {
                    compared_element_count,
                    maximum_absolute_error_bits: maximum_absolute_error.to_bits(),
                    maximum_relative_error_bits: maximum_relative_error.to_bits(),
                },
            )
        }
    }
}

fn placement_output_values(
    format: VulkanPlacementScalarFormat,
    bytes: &[u8],
) -> Box<dyn Iterator<Item = f32> + '_> {
    match format {
        VulkanPlacementScalarFormat::Bf16 => Box::new(bytes.chunks_exact(2).map(|bytes| {
            f32::from_bits(u32::from(u16::from_le_bytes([bytes[0], bytes[1]])) << 16)
        })),
        VulkanPlacementScalarFormat::F32 => Box::new(
            bytes
                .chunks_exact(4)
                .map(|bytes| f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]])),
        ),
    }
}

#[cfg(test)]
mod placement_equivalence_tests {
    use super::*;

    fn bf16_artifact(values: &[f32]) -> VulkanPlacementOutputArtifact {
        VulkanPlacementOutputArtifact {
            scalar_format: VulkanPlacementScalarFormat::Bf16,
            segments: vec![VulkanPlacementOutputSegment {
                binding: 1,
                name: "hidden".to_string(),
                bytes: values
                    .iter()
                    .map(|value| ((*value).to_bits() >> 16) as u16)
                    .flat_map(u16::to_le_bytes)
                    .collect(),
            }],
        }
    }

    fn tolerant() -> VulkanPlacementEquivalenceIdentity {
        VulkanPlacementEquivalenceIdentity {
            output: VulkanPlacementEquivalenceKind::AbsoluteRelativeTolerance,
            state: VulkanPlacementEquivalenceKind::BitExact,
            absolute_tolerance_bits: Some(0.01f64.to_bits()),
            relative_tolerance_bits: Some(0.01f64.to_bits()),
            output_scalar_format: Some(VulkanPlacementScalarFormat::Bf16),
        }
    }

    #[test]
    fn accepts_only_numeric_output_within_the_compiled_combined_bound() {
        let reference = bf16_artifact(&[1.0, 0.0]);
        let accepted = bf16_artifact(&[1.0078125, 0.0078125]);
        let reference_digest = vulkan_placement_output_artifact_digest(&reference).unwrap();
        let accepted_digest = vulkan_placement_output_artifact_digest(&accepted).unwrap();
        let evidence = validate_vulkan_placement_output_equivalence(
            &tolerant(),
            &reference_digest,
            Some(&reference),
            &accepted_digest,
            Some(&accepted),
        )
        .unwrap();
        assert!(matches!(
            evidence,
            VulkanPlacementOutputEquivalenceEvidence::AbsoluteRelativeTolerance {
                compared_element_count: 2,
                ..
            }
        ));

        let rejected = bf16_artifact(&[1.03125, 0.0]);
        let rejected_digest = vulkan_placement_output_artifact_digest(&rejected).unwrap();
        let error = validate_vulkan_placement_output_equivalence(
            &tolerant(),
            &reference_digest,
            Some(&reference),
            &rejected_digest,
            Some(&rejected),
        )
        .unwrap_err()
        .to_string();
        assert!(error.contains("exceeds"));
        assert!(error.contains("segment=0"));
        assert!(error.contains("binding=1"));
        assert!(error.contains("name=\"hidden\""));
        assert!(error.contains("element=0"));
        assert!(error.contains("expected=1"));
        assert!(error.contains("actual=1.03125"));
        assert!(error.contains("allowed_error=0.02"));
        assert!(error.contains("exceeded_element_count=1"));
        assert!(error.contains("compared_element_count=2"));
        assert!(
            validate_vulkan_placement_output_equivalence(
                &tolerant(),
                &reference_digest,
                Some(&reference),
                "sha256:ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff",
                Some(&accepted),
            )
            .unwrap_err()
            .to_string()
            .contains("does not identify")
        );
    }

    #[test]
    fn tolerance_failure_reports_the_worst_error_across_the_complete_output() {
        let reference = bf16_artifact(&[1.0, 2.0, 4.0]);
        let rejected = bf16_artifact(&[1.03125, 2.0625, 5.0]);
        let error = validate_vulkan_placement_output_equivalence(
            &tolerant(),
            &vulkan_placement_output_artifact_digest(&reference).unwrap(),
            Some(&reference),
            &vulkan_placement_output_artifact_digest(&rejected).unwrap(),
            Some(&rejected),
        )
        .unwrap_err()
        .to_string();

        assert!(error.contains("exceeded_element_count=3"));
        assert!(error.contains("compared_element_count=3"));
        assert!(error.contains("element=2"));
        assert!(error.contains("expected=4"));
        assert!(error.contains("actual=5"));
        assert!(error.contains("maximum_absolute_error=1"));
    }

    #[test]
    fn rejects_shape_drift_nonfinite_drift_and_untyped_bit_exact_payloads() {
        let reference = bf16_artifact(&[1.0]);
        let reference_digest = vulkan_placement_output_artifact_digest(&reference).unwrap();
        let mut wrong_shape = bf16_artifact(&[1.0]);
        wrong_shape.segments[0].name = "other".to_string();
        let wrong_shape_digest = vulkan_placement_output_artifact_digest(&wrong_shape).unwrap();
        assert!(
            validate_vulkan_placement_output_equivalence(
                &tolerant(),
                &reference_digest,
                Some(&reference),
                &wrong_shape_digest,
                Some(&wrong_shape),
            )
            .is_err()
        );

        let nonfinite = bf16_artifact(&[f32::NAN]);
        let nonfinite_digest = vulkan_placement_output_artifact_digest(&nonfinite).unwrap();
        assert!(
            validate_vulkan_placement_output_equivalence(
                &tolerant(),
                &reference_digest,
                Some(&reference),
                &nonfinite_digest,
                Some(&nonfinite),
            )
            .unwrap_err()
            .to_string()
            .contains("non-finite")
        );

        assert!(
            validate_vulkan_placement_output_equivalence(
                &VulkanPlacementEquivalenceIdentity::bit_exact(),
                "same",
                None,
                "different",
                None,
            )
            .is_err()
        );
        let reference_digest = vulkan_placement_output_artifact_digest(&reference).unwrap();
        assert!(
            validate_vulkan_placement_output_equivalence(
                &VulkanPlacementEquivalenceIdentity::bit_exact(),
                &reference_digest,
                Some(&reference),
                &reference_digest,
                Some(&reference),
            )
            .is_ok()
        );
    }

    #[test]
    fn bit_exact_failure_reports_complete_typed_difference_statistics() {
        let reference = bf16_artifact(&[1.0, 2.0, 4.0]);
        let candidate = bf16_artifact(&[1.0, 2.5, 5.0]);
        let error = validate_vulkan_placement_output_equivalence(
            &VulkanPlacementEquivalenceIdentity::bit_exact(),
            &vulkan_placement_output_artifact_digest(&reference).unwrap(),
            Some(&reference),
            &vulkan_placement_output_artifact_digest(&candidate).unwrap(),
            Some(&candidate),
        )
        .unwrap_err()
        .to_string();

        assert!(error.contains("differing_element_count=2"));
        assert!(error.contains("compared_element_count=3"));
        assert!(error.contains("maximum_absolute_error=1"));
        assert!(error.contains("element=2"));
        assert!(error.contains("expected=4"));
        assert!(error.contains("actual=5"));
    }
}
