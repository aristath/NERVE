/// An owned description of one dispatch in a larger resident command plan.
///
/// Dispatch bindings and control buffers remain mounted elsewhere. Dynamic
/// push constants live here, so a caller can assemble invocations from several
/// independently mounted graph segments before borrowing the final sequence
/// steps for one command-buffer recording.
pub struct VulkanResidentKernelSequenceInvocation<'a> {
    dispatch: &'a VulkanResidentKernelDispatch,
    push_constants: Vec<u8>,
    direct_workgroup_count: Option<[u32; 2]>,
    indirect_dispatch: Option<(&'a VulkanResidentBuffer, usize)>,
    condition: Option<(&'a VulkanResidentBuffer, usize, bool, u32)>,
    critical_path_region_index: Option<u32>,
}

impl<'a> VulkanResidentKernelSequenceInvocation<'a> {
    pub fn new(
        dispatch: &'a VulkanResidentKernelDispatch,
        push_constants: Vec<u8>,
    ) -> Self {
        Self {
            dispatch,
            push_constants,
            direct_workgroup_count: None,
            indirect_dispatch: None,
            condition: None,
            critical_path_region_index: None,
        }
    }

    pub fn new_direct_with_workgroup_count(
        dispatch: &'a VulkanResidentKernelDispatch,
        push_constants: Vec<u8>,
        workgroup_count_x: u32,
        workgroup_count_y: u32,
    ) -> Result<Self, VulkanError> {
        VulkanResidentKernelSequenceStep::new_direct_with_workgroup_count(
            dispatch,
            &push_constants,
            workgroup_count_x,
            workgroup_count_y,
        )?;
        Ok(Self {
            dispatch,
            push_constants,
            direct_workgroup_count: Some([workgroup_count_x, workgroup_count_y]),
            indirect_dispatch: None,
            condition: None,
            critical_path_region_index: None,
        })
    }

    pub fn new_indirect(
        dispatch: &'a VulkanResidentKernelDispatch,
        push_constants: Vec<u8>,
        buffer: &'a VulkanResidentBuffer,
        byte_offset: usize,
    ) -> Result<Self, VulkanError> {
        VulkanResidentKernelSequenceStep::new_indirect(
            dispatch,
            &push_constants,
            buffer,
            byte_offset,
        )?;
        Ok(Self {
            dispatch,
            push_constants,
            direct_workgroup_count: None,
            indirect_dispatch: Some((buffer, byte_offset)),
            condition: None,
            critical_path_region_index: None,
        })
    }

    pub fn with_condition(
        mut self,
        predicate: &'a VulkanResidentBuffer,
        byte_offset: usize,
        inverted: bool,
        region_id: u32,
    ) -> Result<Self, VulkanError> {
        self.borrowed_step()?.with_condition(
            predicate,
            byte_offset,
            inverted,
            region_id,
        )?;
        self.condition = Some((predicate, byte_offset, inverted, region_id));
        Ok(self)
    }

    pub fn with_critical_path_region(mut self, region_index: u32) -> Self {
        self.critical_path_region_index = Some(region_index);
        self
    }

    pub fn borrowed_step(&self) -> Result<VulkanResidentKernelSequenceStep<'_>, VulkanError> {
        let mut step = if let Some([workgroup_count_x, workgroup_count_y]) =
            self.direct_workgroup_count
        {
            VulkanResidentKernelSequenceStep::new_direct_with_workgroup_count(
                self.dispatch,
                &self.push_constants,
                workgroup_count_x,
                workgroup_count_y,
            )?
        } else if let Some((buffer, byte_offset)) = self.indirect_dispatch {
            VulkanResidentKernelSequenceStep::new_indirect(
                self.dispatch,
                &self.push_constants,
                buffer,
                byte_offset,
            )?
        } else {
            VulkanResidentKernelSequenceStep::new(self.dispatch, &self.push_constants)
        };
        if let Some((predicate, byte_offset, inverted, region_id)) = self.condition {
            step = step.with_condition(predicate, byte_offset, inverted, region_id)?;
        }
        if let Some(region_index) = self.critical_path_region_index {
            step = step.with_critical_path_region(region_index);
        }
        Ok(step)
    }
}

pub fn borrow_resident_kernel_sequence_invocations<'a>(
    invocations: &'a [VulkanResidentKernelSequenceInvocation<'a>],
) -> Result<Vec<VulkanResidentKernelSequenceStep<'a>>, VulkanError> {
    invocations
        .iter()
        .map(VulkanResidentKernelSequenceInvocation::borrowed_step)
        .collect()
}
