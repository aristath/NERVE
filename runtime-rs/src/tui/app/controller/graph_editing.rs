impl App {
    fn duplicate_selected(&mut self) {
        let Some(selected_id) = self.selected_instance_id.clone() else {
            return;
        };
        let Some(editor) = &mut self.editor else {
            return;
        };
        match editor.duplicate_layer_instance_after(&selected_id) {
            Ok(duplicate_id) => {
                self.sync_graph_from_editor(Some(duplicate_id));
                self.status = "Graph draft updated · not mounted".to_string();
            }
            Err(error) => self.status = error.to_string(),
        }
    }

    fn remove_selected(&mut self) {
        let Some(selected_id) = self.selected_instance_id.clone() else {
            return;
        };
        if self.last_valid_sequence.len() <= 1 {
            self.status = "An execution graph must contain at least one node".to_string();
            return;
        }
        let selected_index = self.selected_index().unwrap_or(0);
        let Some(editor) = &mut self.editor else {
            return;
        };
        match editor.remove_layer_instance(&selected_id) {
            Ok(()) => {
                let replacement = editor
                    .layer_instances()
                    .get(selected_index.saturating_sub(1))
                    .map(|instance| instance.instance_id.clone());
                self.sync_graph_from_editor(replacement);
                self.status = "Graph draft updated · not mounted".to_string();
            }
            Err(error) => self.status = error.to_string(),
        }
    }

    fn move_selected(&mut self, delta: i32) {
        let Some(index) = self.selected_index() else {
            return;
        };
        let target = index.saturating_add_signed(delta as isize);
        if target >= self.last_valid_sequence.len() || target == index {
            return;
        }
        let mut ordered_ids = self
            .instances()
            .into_iter()
            .map(|instance| instance.instance_id)
            .collect::<Vec<_>>();
        ordered_ids.swap(index, target);
        let Some(editor) = &mut self.editor else {
            return;
        };
        match editor.reorder_layer_instances(&ordered_ids) {
            Ok(()) => {
                let selected_id = self.selected_instance_id.clone();
                self.sync_graph_from_editor(selected_id);
                self.status = "Graph draft updated · not mounted".to_string();
            }
            Err(error) => self.status = error.to_string(),
        }
    }

    fn ensure_selection_exists(&mut self) {
        let instances = self.instances();
        if self.selected_instance_id.as_ref().is_none_or(|selected| {
            !instances
                .iter()
                .any(|instance| &instance.instance_id == selected)
        }) {
            self.selected_instance_id = instances
                .first()
                .map(|instance| instance.instance_id.clone());
        }
    }

    fn selected_index(&self) -> Option<usize> {
        let selected = self.selected_instance_id.as_ref()?;
        self.instances()
            .iter()
            .position(|instance| &instance.instance_id == selected)
    }

    fn instance_count(&self) -> usize {
        self.instances().len()
    }

    pub(crate) fn instances(&self) -> Vec<RuntimeEditorInstance> {
        self.editor
            .as_ref()
            .map(RuntimeModelEditor::layer_instances)
            .unwrap_or_default()
    }

    fn sync_graph_from_editor(&mut self, selected_instance_id: Option<String>) {
        let Some(editor) = &self.editor else {
            return;
        };
        self.last_valid_sequence = editor.layer_sequence();
        self.sequence
            .set(format_layer_sequence(&self.last_valid_sequence));
        self.sequence_error = None;
        self.selected_instance_id = selected_instance_id;
        self.ensure_selection_exists();
    }
}
