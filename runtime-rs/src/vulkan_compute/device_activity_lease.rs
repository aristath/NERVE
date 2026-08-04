const AMD_PCI_VENDOR_ID: u32 = 0x1002;
const DRM_IOCTL_BASE: u32 = b'd' as u32;
const DRM_IOCTL_GET_CAP_NUMBER: u32 = 0x0c;
const DRM_CAP_DUMB_BUFFER: u64 = 0x1;
const DRM_ACTIVITY_LEASE_INTERVAL: Duration = Duration::from_secs(1);
const DRM_ACTIVITY_LEASE_START_TIMEOUT: Duration = Duration::from_secs(2);

#[repr(C)]
struct DrmGetCap {
    capability: u64,
    value: u64,
}

const fn linux_iowr_request<T>(kind: u32, number: u32) -> libc::c_ulong {
    const NR_BITS: u32 = 8;
    const TYPE_BITS: u32 = 8;
    const SIZE_BITS: u32 = 14;
    const NR_SHIFT: u32 = 0;
    const TYPE_SHIFT: u32 = NR_SHIFT + NR_BITS;
    const SIZE_SHIFT: u32 = TYPE_SHIFT + TYPE_BITS;
    const DIR_SHIFT: u32 = SIZE_SHIFT + SIZE_BITS;
    const READ_WRITE: u32 = 3;

    ((READ_WRITE << DIR_SHIFT)
        | (kind << TYPE_SHIFT)
        | (number << NR_SHIFT)
        | ((std::mem::size_of::<T>() as u32) << SIZE_SHIFT)) as libc::c_ulong
}

const DRM_IOCTL_GET_CAP: libc::c_ulong =
    linux_iowr_request::<DrmGetCap>(DRM_IOCTL_BASE, DRM_IOCTL_GET_CAP_NUMBER);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum VulkanDeviceActivityLeasePhase {
    Starting,
    Active,
    Stopping,
    Stopped,
    Failed,
}

struct VulkanDeviceActivityLeaseState {
    phase: VulkanDeviceActivityLeasePhase,
    stop_requested: bool,
    pulse_count: u64,
    failure: Option<String>,
}

struct VulkanDeviceActivityLeaseShared {
    state: Mutex<VulkanDeviceActivityLeaseState>,
    changed: std::sync::Condvar,
}

#[derive(Clone)]
struct VulkanDeviceActivityLeaseHealth {
    device_id: Arc<str>,
    shared: Arc<VulkanDeviceActivityLeaseShared>,
}

struct VulkanDeviceActivityLease {
    health: VulkanDeviceActivityLeaseHealth,
    worker: Option<std::thread::JoinHandle<()>>,
}

impl VulkanDeviceActivityLeaseHealth {
    fn inactive(device_id: impl Into<Arc<str>>) -> Self {
        Self {
            device_id: device_id.into(),
            shared: Arc::new(VulkanDeviceActivityLeaseShared {
                state: Mutex::new(VulkanDeviceActivityLeaseState {
                    phase: VulkanDeviceActivityLeasePhase::Stopped,
                    stop_requested: true,
                    pulse_count: 0,
                    failure: None,
                }),
                changed: std::sync::Condvar::new(),
            }),
        }
    }

    fn require_healthy(&self) -> Result<(), VulkanError> {
        let state = self.shared.state.lock().map_err(|_| {
            VulkanError(format!(
                "Vulkan device {:?} activity-lease state was poisoned",
                self.device_id
            ))
        })?;
        match state.phase {
            VulkanDeviceActivityLeasePhase::Active => Ok(()),
            VulkanDeviceActivityLeasePhase::Stopped if state.pulse_count == 0 => Ok(()),
            VulkanDeviceActivityLeasePhase::Failed => Err(VulkanError(format!(
                "Vulkan device {:?} activity lease failed after {} pulses: {}",
                self.device_id,
                state.pulse_count,
                state.failure.as_deref().unwrap_or("unknown failure")
            ))),
            phase => Err(VulkanError(format!(
                "Vulkan device {:?} activity lease is not usable: {phase:?}",
                self.device_id
            ))),
        }
    }

    #[cfg(test)]
    fn snapshot(&self) -> (VulkanDeviceActivityLeasePhase, u64, Option<String>) {
        let state = self.shared.state.lock().unwrap();
        (state.phase, state.pulse_count, state.failure.clone())
    }
}

impl VulkanDeviceActivityLease {
    fn start_linux_drm(
        device_id: impl Into<Arc<str>>,
        render_major: u32,
        render_minor: u32,
    ) -> Result<Self, VulkanError> {
        use std::fs::OpenOptions;
        use std::os::fd::AsRawFd as _;

        let device_id = device_id.into();
        let render_node = std::path::PathBuf::from(format!(
            "/dev/char/{render_major}:{render_minor}"
        ));
        let render_file = OpenOptions::new()
            .read(true)
            .write(true)
            .open(&render_node)
            .map_err(|error| {
                VulkanError(format!(
                    "failed to open DRM render node {} for Vulkan device {:?}: {error}",
                    render_node.display(),
                    device_id
                ))
            })?;
        Self::start_with_pulse(
            device_id,
            DRM_ACTIVITY_LEASE_INTERVAL,
            move || {
                let raw_fd = render_file.as_raw_fd();
                let mut query = DrmGetCap {
                    capability: DRM_CAP_DUMB_BUFFER,
                    value: 0,
                };
                let result = unsafe { libc::ioctl(raw_fd, DRM_IOCTL_GET_CAP, &mut query) };
                if result == 0 {
                    Ok(())
                } else {
                    Err(std::io::Error::last_os_error())
                }
            },
        )
    }

    fn start_with_pulse<F>(
        device_id: Arc<str>,
        interval: Duration,
        mut pulse: F,
    ) -> Result<Self, VulkanError>
    where
        F: FnMut() -> std::io::Result<()> + Send + 'static,
    {
        if interval.is_zero() {
            return Err(VulkanError(
                "Vulkan device activity-lease interval must not be zero".to_string(),
            ));
        }
        let shared = Arc::new(VulkanDeviceActivityLeaseShared {
            state: Mutex::new(VulkanDeviceActivityLeaseState {
                phase: VulkanDeviceActivityLeasePhase::Starting,
                stop_requested: false,
                pulse_count: 0,
                failure: None,
            }),
            changed: std::sync::Condvar::new(),
        });
        let health = VulkanDeviceActivityLeaseHealth {
            device_id: Arc::clone(&device_id),
            shared: Arc::clone(&shared),
        };
        let thread_name = format!(
            "nerve-gpu-lease-{}",
            device_id
                .chars()
                .filter(|character| character.is_ascii_hexdigit())
                .take(12)
                .collect::<String>()
        );
        let worker = std::thread::Builder::new()
            .name(thread_name)
            .spawn(move || {
                loop {
                    match pulse() {
                        Ok(()) => {
                            let Ok(mut state) = shared.state.lock() else {
                                return;
                            };
                            state.pulse_count = state.pulse_count.saturating_add(1);
                            state.phase = VulkanDeviceActivityLeasePhase::Active;
                            shared.changed.notify_all();
                            let Ok((mut state, _)) = shared
                                .changed
                                .wait_timeout_while(state, interval, |state| {
                                    !state.stop_requested
                                })
                            else {
                                return;
                            };
                            if state.stop_requested {
                                state.phase = VulkanDeviceActivityLeasePhase::Stopped;
                                shared.changed.notify_all();
                                return;
                            }
                        }
                        Err(error) => {
                            if let Ok(mut state) = shared.state.lock() {
                                state.phase = VulkanDeviceActivityLeasePhase::Failed;
                                state.failure = Some(error.to_string());
                                shared.changed.notify_all();
                            }
                            return;
                        }
                    }
                }
            })
            .map_err(|error| {
                VulkanError(format!(
                    "failed to start Vulkan device {:?} activity lease: {error}",
                    device_id
                ))
            })?;
        let lease = Self {
            health,
            worker: Some(worker),
        };
        lease.wait_until_active()?;
        Ok(lease)
    }

    fn wait_until_active(&self) -> Result<(), VulkanError> {
        let state = self.health.shared.state.lock().map_err(|_| {
            VulkanError(format!(
                "Vulkan device {:?} activity-lease state was poisoned",
                self.health.device_id
            ))
        })?;
        let (state, timeout) = self
            .health
            .shared
            .changed
            .wait_timeout_while(state, DRM_ACTIVITY_LEASE_START_TIMEOUT, |state| {
                state.phase == VulkanDeviceActivityLeasePhase::Starting
            })
            .map_err(|_| {
                VulkanError(format!(
                    "Vulkan device {:?} activity-lease state was poisoned",
                    self.health.device_id
                ))
            })?;
        if timeout.timed_out() && state.phase == VulkanDeviceActivityLeasePhase::Starting {
            return Err(VulkanError(format!(
                "Vulkan device {:?} activity lease did not start within {} ms",
                self.health.device_id,
                DRM_ACTIVITY_LEASE_START_TIMEOUT.as_millis()
            )));
        }
        drop(state);
        self.health.require_healthy()
    }

    fn health(&self) -> VulkanDeviceActivityLeaseHealth {
        self.health.clone()
    }

    fn stop(&mut self) -> Result<(), VulkanError> {
        if let Ok(mut state) = self.health.shared.state.lock() {
            state.stop_requested = true;
            if state.phase == VulkanDeviceActivityLeasePhase::Active {
                state.phase = VulkanDeviceActivityLeasePhase::Stopping;
            }
            self.health.shared.changed.notify_all();
        }
        if let Some(worker) = self.worker.take()
            && worker.join().is_err()
        {
            return Err(VulkanError(format!(
                "Vulkan device {:?} activity-lease worker panicked",
                self.health.device_id
            )));
        }
        let state = self.health.shared.state.lock().map_err(|_| {
            VulkanError(format!(
                "Vulkan device {:?} activity-lease state was poisoned",
                self.health.device_id
            ))
        })?;
        if state.phase == VulkanDeviceActivityLeasePhase::Failed {
            return Err(VulkanError(format!(
                "Vulkan device {:?} activity lease failed after {} pulses: {}",
                self.health.device_id,
                state.pulse_count,
                state.failure.as_deref().unwrap_or("unknown failure")
            )));
        }
        if state.phase != VulkanDeviceActivityLeasePhase::Stopped {
            return Err(VulkanError(format!(
                "Vulkan device {:?} activity lease stopped in invalid phase {:?}",
                self.health.device_id, state.phase
            )));
        }
        Ok(())
    }
}

impl Drop for VulkanDeviceActivityLease {
    fn drop(&mut self) {
        let _ = self.stop();
    }
}

fn physical_device_drm_render_node(
    instance: &ash::Instance,
    physical_device: vk::PhysicalDevice,
) -> Result<(u32, u32), VulkanError> {
    if !physical_device_supports_extension(
        instance,
        physical_device,
        ash::ext::physical_device_drm::NAME,
    )? {
        return Err(VulkanError(
            "AMD Vulkan device does not expose VK_EXT_physical_device_drm".to_string(),
        ));
    }
    let mut drm = vk::PhysicalDeviceDrmPropertiesEXT::default();
    let mut properties = vk::PhysicalDeviceProperties2::default().push_next(&mut drm);
    unsafe { instance.get_physical_device_properties2(physical_device, &mut properties) };
    if drm.has_render == 0 || drm.render_major < 0 || drm.render_minor < 0 {
        return Err(VulkanError(
            "AMD Vulkan device does not expose a DRM render node".to_string(),
        ));
    }
    Ok((
        u32::try_from(drm.render_major)
            .map_err(|_| VulkanError("DRM render major exceeds u32".to_string()))?,
        u32::try_from(drm.render_minor)
            .map_err(|_| VulkanError("DRM render minor exceeds u32".to_string()))?,
    ))
}

#[cfg(test)]
mod device_activity_lease_tests {
    use super::*;

    #[test]
    fn linux_drm_get_cap_request_matches_the_kernel_abi() {
        assert_eq!(std::mem::size_of::<DrmGetCap>(), 16);
        assert_eq!(DRM_IOCTL_GET_CAP, 0xc010_640c);
    }

    #[test]
    fn activity_lease_pulses_until_explicitly_stopped() {
        let pulses = Arc::new(AtomicU64::new(0));
        let worker_pulses = Arc::clone(&pulses);
        let mut lease = VulkanDeviceActivityLease::start_with_pulse(
            Arc::<str>::from("fixture-device"),
            Duration::from_millis(2),
            move || {
                worker_pulses.fetch_add(1, Ordering::SeqCst);
                Ok(())
            },
        )
        .unwrap();
        let deadline = Instant::now() + Duration::from_millis(100);
        while pulses.load(Ordering::SeqCst) < 2 && Instant::now() < deadline {
            std::thread::yield_now();
        }

        lease.health().require_healthy().unwrap();
        lease.stop().unwrap();

        let (phase, pulse_count, failure) = lease.health().snapshot();
        assert_eq!(phase, VulkanDeviceActivityLeasePhase::Stopped);
        assert!(pulse_count >= 2);
        assert_eq!(failure, None);
    }

    #[test]
    fn activity_lease_surfaces_a_pulse_failure() {
        let calls = Arc::new(AtomicU64::new(0));
        let worker_calls = Arc::clone(&calls);
        let mut lease = VulkanDeviceActivityLease::start_with_pulse(
            Arc::<str>::from("fixture-device"),
            Duration::from_millis(1),
            move || {
                if worker_calls.fetch_add(1, Ordering::SeqCst) == 0 {
                    Ok(())
                } else {
                    Err(std::io::Error::other("pulse rejected"))
                }
            },
        )
        .unwrap();
        let deadline = Instant::now() + Duration::from_millis(100);
        let error = loop {
            if let Err(error) = lease.health().require_healthy() {
                break error;
            }
            assert!(
                Instant::now() < deadline,
                "activity lease did not publish its injected pulse failure"
            );
            std::thread::yield_now();
        };

        assert!(calls.load(Ordering::SeqCst) >= 2);
        assert!(error.to_string().contains("pulse rejected"));
        assert!(lease.stop().unwrap_err().to_string().contains("pulse rejected"));
    }

    #[test]
    fn selected_amd_device_remains_submit_ready_across_runtime_pm_window() {
        let Some(raw_device_index) = std::env::var("NERVE_TEST_VULKAN_DEVICE_INDEX").ok() else {
            eprintln!("skipping AMD activity-lease integration test: explicit Vulkan device index unset");
            return;
        };
        let device_index = raw_device_index
            .parse::<usize>()
            .expect("NERVE_TEST_VULKAN_DEVICE_INDEX must be an integer");
        let device = VulkanComputeDevice::new_for_physical_device_index(device_index)
            .expect("selected AMD Vulkan device must open with an activity lease");
        let deadline = Instant::now() + Duration::from_secs(8);
        loop {
            let (phase, pulse_count, failure) = device.activity_lease_health.snapshot();
            assert_eq!(failure, None);
            assert_eq!(phase, VulkanDeviceActivityLeasePhase::Active);
            if pulse_count >= 7 {
                break;
            }
            assert!(
                Instant::now() < deadline,
                "AMD activity lease produced only {pulse_count} pulses across the runtime-PM window"
            );
            std::thread::sleep(Duration::from_millis(10));
        }

        let source = device.create_resident_buffer(4).unwrap();
        let destination = device.create_resident_buffer(4).unwrap();
        source.write_bytes(&[17, 34, 51, 68]).unwrap();
        destination.write_bytes(&[0; 4]).unwrap();
        device
            .copy_resident_buffer_bytes(&source, &destination, 4)
            .unwrap();
        assert_eq!(destination.read_bytes(4).unwrap(), [17, 34, 51, 68]);
        device.quiesce().unwrap();
    }
}
