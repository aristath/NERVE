const VULKAN_TIERED_HOST_MINIMUM_HEADROOM_BYTES: usize = 1024 * 1024 * 1024;
const VULKAN_TIERED_HOST_TOTAL_MEMORY_HEADROOM_DIVISOR: usize = 64;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct VulkanHostMemoryCapacity {
    total_bytes: usize,
    available_bytes: usize,
}

impl VulkanHostMemoryCapacity {
    fn safe_tiered_payload_bytes(self) -> usize {
        let headroom = VULKAN_TIERED_HOST_MINIMUM_HEADROOM_BYTES
            .max(self.total_bytes / VULKAN_TIERED_HOST_TOTAL_MEMORY_HEADROOM_DIVISOR);
        self.available_bytes.saturating_sub(headroom)
    }
}

fn read_vulkan_host_memory_capacity()
-> Result<VulkanHostMemoryCapacity, VulkanResidentTokenModelPackageError> {
    let contents = fs::read_to_string("/proc/meminfo").map_err(|error| {
        VulkanResidentTokenModelPackageError::new(format!(
            "failed to inspect available host memory for tiered residency: {error}"
        ))
    })?;
    parse_vulkan_host_memory_capacity(&contents).map_err(|error| {
        VulkanResidentTokenModelPackageError::new(format!(
            "failed to inspect available host memory for tiered residency: {error}"
        ))
    })
}

fn parse_vulkan_host_memory_capacity(contents: &str) -> Result<VulkanHostMemoryCapacity, String> {
    let mut total_kib = None;
    let mut available_kib = None;
    for line in contents.lines() {
        let Some((name, value)) = line.split_once(':') else {
            continue;
        };
        if name != "MemTotal" && name != "MemAvailable" {
            continue;
        }
        let mut fields = value.split_whitespace();
        let kib = fields
            .next()
            .ok_or_else(|| format!("{name} has no value"))?
            .parse::<usize>()
            .map_err(|error| format!("{name} is not an integer: {error}"))?;
        if fields.next() != Some("kB") || fields.next().is_some() {
            return Err(format!("{name} does not use the expected kB unit"));
        }
        match name {
            "MemTotal" => total_kib = Some(kib),
            "MemAvailable" => available_kib = Some(kib),
            _ => unreachable!(),
        }
    }
    let total_bytes = total_kib
        .ok_or_else(|| "MemTotal is missing".to_string())?
        .checked_mul(1024)
        .ok_or_else(|| "MemTotal byte count overflowed".to_string())?;
    let available_bytes = available_kib
        .ok_or_else(|| "MemAvailable is missing".to_string())?
        .checked_mul(1024)
        .ok_or_else(|| "MemAvailable byte count overflowed".to_string())?;
    if total_bytes == 0 || available_bytes > total_bytes {
        return Err("host memory capacity is inconsistent".to_string());
    }
    Ok(VulkanHostMemoryCapacity {
        total_bytes,
        available_bytes,
    })
}
