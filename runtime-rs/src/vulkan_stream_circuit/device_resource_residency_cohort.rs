#[derive(Clone)]
pub struct DeviceResourceResidencyCohortMember<P: DeviceResidentResourcePayload> {
    manager: DeviceResourceResidencyManager<P>,
    group_ids: BTreeSet<String>,
}

impl<P: DeviceResidentResourcePayload> DeviceResourceResidencyCohortMember<P> {
    pub fn new(
        manager: DeviceResourceResidencyManager<P>,
        group_ids: BTreeSet<String>,
    ) -> Result<Self, DeviceResourceResidencyError> {
        if group_ids.is_empty() || group_ids.iter().any(|group_id| group_id.trim().is_empty()) {
            return Err(DeviceResourceResidencyError::invalid_descriptor(
                "distributed residency cohort member has no valid groups",
            ));
        }
        Ok(Self { manager, group_ids })
    }

    pub fn device_id(&self) -> &str {
        self.manager.device_id()
    }

    pub fn group_ids(&self) -> &BTreeSet<String> {
        &self.group_ids
    }
}

pub struct DeviceResourceResidencyCohort<P: DeviceResidentResourcePayload> {
    id: String,
    members: Vec<DeviceResourceResidencyCohortMember<P>>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DeviceResourceResidencyCohortState {
    Absent,
    Resident,
    Transitioning,
    Partial,
    Failed,
}

pub struct DeviceResourceResidencyCohortEviction<P: DeviceResidentResourcePayload> {
    evictions: Vec<DeviceResourceResidencyEviction<P>>,
    release: DeviceResourceResidencyRelease,
}

impl<P: DeviceResidentResourcePayload> DeviceResourceResidencyCohortEviction<P> {
    pub fn release(&self) -> DeviceResourceResidencyRelease {
        self.release
    }

    pub fn member_eviction_count(&self) -> usize {
        self.evictions.len()
    }
}

struct DeviceResourceResidencyCohortLoad<P: DeviceResidentResourcePayload> {
    permit: DeviceResourceLoadPermit<P>,
    group: Option<DeviceResidentResourceGroup<P>>,
}

impl<P: DeviceResidentResourcePayload> DeviceResourceResidencyCohort<P> {
    pub fn new(
        id: impl Into<String>,
        mut members: Vec<DeviceResourceResidencyCohortMember<P>>,
    ) -> Result<Self, DeviceResourceResidencyError> {
        let id = id.into();
        if id.trim().is_empty() || members.len() < 2 {
            return Err(DeviceResourceResidencyError::invalid_descriptor(
                "distributed residency cohort needs an identity and at least two device members",
            ));
        }
        members.sort_by(|left, right| {
            (left.device_id(), Arc::as_ptr(&left.manager.inner) as usize).cmp(&(
                right.device_id(),
                Arc::as_ptr(&right.manager.inner) as usize,
            ))
        });
        if members.windows(2).any(|pair| {
            pair[0].device_id() >= pair[1].device_id()
                || Arc::ptr_eq(&pair[0].manager.inner, &pair[1].manager.inner)
        }) {
            return Err(DeviceResourceResidencyError::invalid_descriptor(
                "distributed residency cohort must name each physical device manager exactly once",
            ));
        }
        Ok(Self { id, members })
    }

    pub fn id(&self) -> &str {
        &self.id
    }

    pub fn members(&self) -> &[DeviceResourceResidencyCohortMember<P>] {
        &self.members
    }

    pub fn state(&self) -> Result<DeviceResourceResidencyCohortState, DeviceResourceResidencyError> {
        let states = self.lock_member_states()?;
        let mut resident = 0usize;
        let mut absent = 0usize;
        let mut transitioning = 0usize;
        let mut failed = 0usize;
        let mut expected = 0usize;
        for (member, state) in self.members.iter().zip(states.iter()) {
            for group_id in &member.group_ids {
                expected += 1;
                match state.entries.get(group_id) {
                    Some(DeviceResourceResidencyEntry::Resident { .. }) => resident += 1,
                    Some(DeviceResourceResidencyEntry::Loading { .. }) => transitioning += 1,
                    Some(DeviceResourceResidencyEntry::Failed { .. }) => failed += 1,
                    None => absent += 1,
                }
            }
        }
        let state = if failed > 0 {
            DeviceResourceResidencyCohortState::Failed
        } else if resident == expected {
            DeviceResourceResidencyCohortState::Resident
        } else if absent == expected {
            DeviceResourceResidencyCohortState::Absent
        } else if transitioning > 0 && resident == 0 {
            DeviceResourceResidencyCohortState::Transitioning
        } else {
            DeviceResourceResidencyCohortState::Partial
        };
        Ok(state)
    }

    pub fn publish_loads(
        &self,
        loads: Vec<(
            DeviceResourceLoadPermit<P>,
            DeviceResidentResourceGroup<P>,
        )>,
    ) -> Result<(), DeviceResourceResidencyError> {
        let expected = self.expected_physical_groups();
        let mut staged = loads
            .into_iter()
            .map(|(permit, group)| DeviceResourceResidencyCohortLoad {
                permit,
                group: Some(group),
            })
            .collect::<Vec<_>>();
        let mut actual = BTreeSet::new();
        for load in &staged {
            let manager = load.permit.manager.upgrade().ok_or_else(|| {
                DeviceResourceResidencyError::new(
                    DeviceResourceResidencyErrorKind::Stopped,
                    "distributed residency cohort load outlived a device manager",
                )
            })?;
            let key = (Arc::as_ptr(&manager) as usize, load.permit.descriptor.id.clone());
            if !actual.insert(key) {
                return Err(DeviceResourceResidencyError::invalid_publication(
                    "distributed residency cohort repeats a physical load",
                ));
            }
            if load
                .group
                .as_ref()
                .is_none_or(|group| group.descriptor != load.permit.descriptor)
            {
                return Err(DeviceResourceResidencyError::invalid_publication(
                    "distributed residency cohort load changed an immutable descriptor",
                ));
            }
        }
        if actual != expected {
            return Err(DeviceResourceResidencyError::invalid_publication(
                "distributed residency cohort publication must contain every physical member exactly once",
            ));
        }

        let mut states = self.lock_member_states()?;
        let member_indices = self
            .members
            .iter()
            .enumerate()
            .map(|(index, member)| (Arc::as_ptr(&member.manager.inner) as usize, index))
            .collect::<BTreeMap<_, _>>();
        let mut additions_by_member = vec![0usize; self.members.len()];
        let mut loads_by_member = vec![0u64; self.members.len()];
        for load in &staged {
            let manager = load.permit.manager.upgrade().ok_or_else(|| {
                DeviceResourceResidencyError::new(
                    DeviceResourceResidencyErrorKind::Stopped,
                    "distributed residency cohort load outlived a device manager",
                )
            })?;
            let member_index = *member_indices
                .get(&(Arc::as_ptr(&manager) as usize))
                .expect("the exact physical load set was validated");
            let state = &states[member_index];
            match state.entries.get(&load.permit.descriptor.id) {
                Some(DeviceResourceResidencyEntry::Loading {
                    descriptor,
                    operation_id,
                    ..
                }) if *descriptor == load.permit.descriptor
                    && *operation_id == load.permit.operation_id => {}
                _ => {
                    return Err(DeviceResourceResidencyError::new(
                        DeviceResourceResidencyErrorKind::StaleOperation,
                        "distributed residency cohort contains a stale physical load",
                    ));
                }
            }
            additions_by_member[member_index] = additions_by_member[member_index]
                .checked_add(load.permit.descriptor.byte_count)
                .ok_or_else(|| {
                    DeviceResourceResidencyError::new(
                        DeviceResourceResidencyErrorKind::Capacity,
                        "distributed residency cohort byte count overflowed",
                    )
                })?;
            loads_by_member[member_index] += 1;
        }
        for (member_index, (member, state)) in
            self.members.iter().zip(states.iter()).enumerate()
        {
            let published_bytes = state
                .dynamic_resident_bytes
                .checked_add(additions_by_member[member_index])
                .ok_or_else(|| {
                    DeviceResourceResidencyError::new(
                        DeviceResourceResidencyErrorKind::Capacity,
                        "distributed residency cohort resident bytes overflowed",
                    )
                })?;
            if state.reserved_loading_bytes < additions_by_member[member_index]
                || member
                    .manager
                    .inner
                    .always_resident_bytes
                    .checked_add(published_bytes)
                    .is_none_or(|bytes| bytes > member.manager.inner.capacity_bytes)
                || state
                    .next_access_epoch
                    .checked_add(loads_by_member[member_index])
                    .is_none()
            {
                return Err(DeviceResourceResidencyError::new(
                    DeviceResourceResidencyErrorKind::Capacity,
                    "distributed residency cohort cannot commit its reserved physical loads",
                ));
            }
        }

        let mut outcomes = Vec::with_capacity(staged.len());
        for load in &mut staged {
            let manager = load
                .permit
                .manager
                .upgrade()
                .expect("validated physical manager remains held by the cohort");
            let member_index = member_indices[&(Arc::as_ptr(&manager) as usize)];
            let state = &mut states[member_index];
            state.next_access_epoch += 1;
            let access_epoch = state.next_access_epoch;
            let entry = state
                .entries
                .remove(&load.permit.descriptor.id)
                .expect("cohort load was prevalidated while every manager was locked");
            let DeviceResourceResidencyEntry::Loading {
                owners, outcome, ..
            } = entry
            else {
                unreachable!("cohort load state changed while every manager was locked")
            };
            let group = Arc::new(
                load.group
                    .take()
                    .expect("cohort publication consumes each staged group once"),
            );
            state.reserved_loading_bytes -= load.permit.descriptor.byte_count;
            state.dynamic_resident_bytes += load.permit.descriptor.byte_count;
            state.high_water_dynamic_resident_bytes = state
                .high_water_dynamic_resident_bytes
                .max(state.dynamic_resident_bytes);
            state.successful_load_count = state.successful_load_count.saturating_add(1);
            if !state
                .loaded_group_ids
                .insert(load.permit.descriptor.id.clone())
            {
                state.reload_count = state.reload_count.saturating_add(1);
            }
            state.entries.insert(
                load.permit.descriptor.id.clone(),
                DeviceResourceResidencyEntry::Resident {
                    descriptor: load.permit.descriptor.clone(),
                    owners,
                    active_leases: BTreeMap::new(),
                    last_access_epoch: access_epoch,
                    group,
                },
            );
            load.permit.finished = true;
            outcomes.push(outcome);
        }
        for state in states.iter_mut() {
            let resident_group_count = state
                .entries
                .values()
                .filter(|entry| matches!(entry, DeviceResourceResidencyEntry::Resident { .. }))
                .count();
            state.high_water_resident_group_count =
                state.high_water_resident_group_count.max(resident_group_count);
        }
        drop(states);
        for outcome in outcomes {
            outcome.finish(Ok(()));
        }
        Ok(())
    }

    pub fn evict_inactive(
        &self,
    ) -> Result<DeviceResourceResidencyCohortEviction<P>, DeviceResourceResidencyError> {
        let mut states = self.lock_member_states()?;
        for (member, state) in self.members.iter().zip(states.iter()) {
            for group_id in &member.group_ids {
                match state.entries.get(group_id) {
                    Some(DeviceResourceResidencyEntry::Resident {
                        active_leases, ..
                    }) if active_leases.is_empty() => {}
                    Some(DeviceResourceResidencyEntry::Resident { .. }) => {
                        return Err(DeviceResourceResidencyError::new(
                            DeviceResourceResidencyErrorKind::InUse,
                            format!(
                                "cannot evict distributed residency cohort {:?} while group {group_id:?} on {:?} has an active lease",
                                self.id,
                                member.device_id(),
                            ),
                        ));
                    }
                    Some(_) => {
                        return Err(DeviceResourceResidencyError::new(
                            DeviceResourceResidencyErrorKind::InUse,
                            format!(
                                "cannot evict distributed residency cohort {:?} while group {group_id:?} on {:?} is transitioning",
                                self.id,
                                member.device_id(),
                            ),
                        ));
                    }
                    None => {
                        return Err(DeviceResourceResidencyError::new(
                            DeviceResourceResidencyErrorKind::StaleOperation,
                            format!(
                                "cannot evict partial distributed residency cohort {:?}: group {group_id:?} is absent on {:?}",
                                self.id,
                                member.device_id(),
                            ),
                        ));
                    }
                }
            }
        }

        let mut evictions = Vec::with_capacity(self.members.len());
        let mut release = DeviceResourceResidencyRelease::default();
        for (member, state) in self.members.iter().zip(states.iter_mut()) {
            let mut retired_groups = Vec::with_capacity(member.group_ids.len());
            let mut member_release = DeviceResourceResidencyRelease::default();
            for group_id in &member.group_ids {
                let DeviceResourceResidencyEntry::Resident {
                    descriptor, group, ..
                } = state
                    .entries
                    .remove(group_id)
                    .expect("cohort eviction was prevalidated")
                else {
                    unreachable!("cohort eviction state changed while every manager was locked")
                };
                state.dynamic_resident_bytes -= descriptor.byte_count;
                member_release.group_count += 1;
                member_release.byte_count += descriptor.byte_count;
                retired_groups.push(group);
            }
            state.eviction_count = state.eviction_count.saturating_add(1);
            state.evicted_group_count = state.evicted_group_count.saturating_add(
                u64::try_from(member_release.group_count).unwrap_or(u64::MAX),
            );
            state.evicted_byte_count = state.evicted_byte_count.saturating_add(
                u64::try_from(member_release.byte_count).unwrap_or(u64::MAX),
            );
            release.group_count += member_release.group_count;
            release.byte_count += member_release.byte_count;
            evictions.push(DeviceResourceResidencyEviction {
                release: member_release,
                retired_groups,
            });
        }
        Ok(DeviceResourceResidencyCohortEviction { evictions, release })
    }

    fn expected_physical_groups(&self) -> BTreeSet<(usize, String)> {
        self.members
            .iter()
            .flat_map(|member| {
                let manager = Arc::as_ptr(&member.manager.inner) as usize;
                member
                    .group_ids
                    .iter()
                    .cloned()
                    .map(move |group_id| (manager, group_id))
            })
            .collect()
    }

    fn lock_member_states(
        &self,
    ) -> Result<
        Vec<std::sync::MutexGuard<'_, DeviceResourceResidencyState<P>>>,
        DeviceResourceResidencyError,
    > {
        self.members
            .iter()
            .map(|member| {
                member.manager.inner.state.lock().map_err(|_| {
                    DeviceResourceResidencyError::new(
                        DeviceResourceResidencyErrorKind::Stopped,
                        "distributed residency cohort member manager was poisoned",
                    )
                })
            })
            .collect()
    }
}
