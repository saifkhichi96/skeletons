from __future__ import annotations

import time

import numpy as np
import torch
from PySide6 import QtCore, QtGui, QtWidgets

from ..compat import load_ik_api
from ..models import SceneNode, TargetMotionSpec
from ..theme import SCENE_COLORS, SECONDARY_TEXT_STYLE
from ..widgets import FloatSlider, _FlowLayout


class IKControls:
    """IK controls sharing the workspace's model, pose, viewport, and analysis."""

    MAX_TARGETS = 8

    def _init_ik(self) -> None:
        self.ik_api = load_ik_api()
        self.ik_running = False
        self._ik_sequence_results: list[dict] | None = None
        self._manual_ik_targets: list[TargetMotionSpec] = []
        self._ik_confidence = np.empty(0)
        self._ik_trail_count = 0
        self.ik_solver = None
        self.ik_iter = 0
        self.ik_total_steps = 0
        self.ik_root_step = 0.35
        self.ik_delay_ms = 33
        self.ik_max_iter = 120
        self.target_specs: list[TargetMotionSpec] = []
        self._ik_target_serial = 0
        self._placement_target: str | None = None
        self._ik_target_time = time.perf_counter()
        self._ik_fk_output = None
        self._ik_timer = QtCore.QTimer(self)
        self._ik_timer.setInterval(self.ik_delay_ms)
        self._ik_timer.timeout.connect(self._ik_tick)

    def _build_ik_tab(self) -> None:
        configure = QtWidgets.QGroupBox("Configure")
        form = QtWidgets.QFormLayout(configure)
        form.setRowWrapPolicy(QtWidgets.QFormLayout.RowWrapPolicy.WrapLongRows)
        self.ik_solver_combo = QtWidgets.QComboBox()
        for label, key in (("CCD", "ccd"), ("DLS", "dls"), ("Gradient", "gradient")):
            self.ik_solver_combo.addItem(label, key)
        self.ik_solver_combo.currentIndexChanged.connect(self._invalidate_ik_solver)
        form.addRow("Solver", self.ik_solver_combo)
        self.ik_root_step_slider = FloatSlider(
            "Root follow",
            minimum=0.0,
            maximum=1.0,
            factor=100,
            decimals=2,
            compact=True,
        )
        self.ik_root_step_slider.set_value(self.ik_root_step)
        self.ik_root_step_slider.value_changed.connect(self._set_ik_root_step)
        form.addRow(self.ik_root_step_slider)
        self.ik_delay_slider = FloatSlider(
            "Tick delay",
            minimum=10,
            maximum=500,
            factor=1,
            decimals=0,
            suffix=" ms",
            compact=True,
        )
        self.ik_delay_slider.set_value(self.ik_delay_ms)
        self.ik_delay_slider.value_changed.connect(self._set_ik_delay)
        form.addRow(self.ik_delay_slider)
        self.ik_iterations_spin = QtWidgets.QSpinBox()
        self.ik_iterations_spin.setRange(1, 5000)
        self.ik_iterations_spin.setValue(self.ik_max_iter)
        self.ik_iterations_spin.setToolTip(
            "Iterations per dataset frame; counter wraps for manual targets."
        )
        self.ik_iterations_spin.valueChanged.connect(self._set_ik_iterations)
        form.addRow("Iterations", self.ik_iterations_spin)
        self.ik_layout.addWidget(configure)

        targets = QtWidgets.QGroupBox("Targets")
        target_layout = QtWidgets.QVBoxLayout(targets)
        target_layout.setSpacing(6)
        self.ik_target_source = QtWidgets.QComboBox()
        self.ik_target_source.addItem("Manual targets", "manual")
        self.ik_target_source.addItem("Dataset end effectors", "dataset")
        self.ik_target_source.currentIndexChanged.connect(self._on_ik_source_changed)
        target_layout.addWidget(self.ik_target_source)
        self.ik_dataset_note = QtWidgets.QLabel(
            "Run solves the selected range, warm-starting each frame. Step solves "
            "one iteration on the current frame. Targets use valid 3D leaf joints."
        )
        self.ik_dataset_note.setWordWrap(True)
        target_layout.addWidget(self.ik_dataset_note)
        self.ik_target_combo = QtWidgets.QComboBox()
        self.ik_target_combo.currentIndexChanged.connect(self._edit_ik_target)
        target_layout.addWidget(self.ik_target_combo)
        row = _FlowLayout()
        self.ik_add_target_button = QtWidgets.QPushButton("Add Target")
        self.ik_add_target_button.clicked.connect(self._add_ik_target)
        self.ik_edit_target_button = QtWidgets.QPushButton("Edit Target")
        self.ik_edit_target_button.clicked.connect(self._edit_ik_target)
        self.ik_remove_target_button = QtWidgets.QPushButton("Remove")
        self.ik_remove_target_button.clicked.connect(self._remove_ik_target)
        for button in (
            self.ik_add_target_button,
            self.ik_edit_target_button,
            self.ik_remove_target_button,
        ):
            row.addWidget(button)
        target_layout.addLayout(row)
        self.ik_target_editor = QtWidgets.QWidget()
        editor = QtWidgets.QVBoxLayout(self.ik_target_editor)
        editor.setContentsMargins(0, 0, 0, 0)
        self.ik_joint_combo = QtWidgets.QComboBox()
        self.ik_joint_combo.currentTextChanged.connect(self._on_ik_target_changed)
        editor.addWidget(QtWidgets.QLabel("Bound joint"))
        editor.addWidget(self.ik_joint_combo)
        self.ik_moving_check = QtWidgets.QCheckBox("Moving target")
        self.ik_moving_check.toggled.connect(self._on_ik_target_changed)
        editor.addWidget(self.ik_moving_check)
        grid = QtWidgets.QGridLayout()
        self.ik_position_spins = []
        for axis, name in enumerate(("X", "Y", "Z")):
            spin = QtWidgets.QDoubleSpinBox()
            spin.setRange(-1000, 1000)
            spin.setDecimals(3)
            spin.setSingleStep(0.01)
            spin.setMinimumWidth(0)
            spin.setSizePolicy(
                QtWidgets.QSizePolicy.Policy.Ignored, QtWidgets.QSizePolicy.Policy.Fixed
            )
            spin.valueChanged.connect(self._on_ik_target_changed)
            grid.addWidget(QtWidgets.QLabel(name), 0, axis)
            grid.addWidget(spin, 1, axis)
            self.ik_position_spins.append(spin)
        editor.addLayout(grid)
        self.ik_place_button = QtWidgets.QPushButton("Place in scene")
        self.ik_place_button.setCheckable(True)
        self.ik_place_button.setToolTip(
            "Set Y, then click in the scene to place X/Z. Esc cancels."
        )
        self.ik_place_button.toggled.connect(self._begin_target_placement)
        center = QtWidgets.QPushButton("To joint")
        center.clicked.connect(self._center_ik_targets)
        randomize = QtWidgets.QPushButton("Randomize motion")
        randomize.clicked.connect(self._randomize_ik_targets)
        row = _FlowLayout()
        for button in (self.ik_place_button, center, randomize):
            row.addWidget(button)
        editor.addLayout(row)
        target_layout.addWidget(self.ik_target_editor)
        self.ik_target_editor.hide()
        self.ik_targets_group = targets
        self.ik_layout.addWidget(targets)

        solve = QtWidgets.QGroupBox("Solve")
        layout = QtWidgets.QVBoxLayout(solve)
        row = _FlowLayout()
        self.ik_run_button = QtWidgets.QPushButton("Run IK")
        self.ik_run_button.setProperty("prominent", True)
        self.ik_run_button.clicked.connect(self._run_ik)
        self.ik_stop_button = QtWidgets.QPushButton("Stop")
        self.ik_stop_button.clicked.connect(self._stop_ik)
        self.ik_step_button = QtWidgets.QPushButton("Step")
        self.ik_step_button.clicked.connect(self._step_ik_once)
        self.ik_reset_button = QtWidgets.QPushButton("Reset Pose")
        self.ik_reset_button.clicked.connect(self._reset_ik_pose)
        for button in (
            self.ik_run_button,
            self.ik_stop_button,
            self.ik_step_button,
            self.ik_reset_button,
        ):
            row.addWidget(button)
        self.ik_export_button = QtWidgets.QPushButton("Export Sequence…")
        self.ik_export_button.clicked.connect(self._save_fit_result_to_file)
        row.addWidget(self.ik_export_button)
        layout.addLayout(row)
        self.ik_status_label = QtWidgets.QLabel("Idle.")
        self.ik_status_label.setStyleSheet(SECONDARY_TEXT_STYLE)
        self.ik_status_label.setWordWrap(True)
        layout.addWidget(self.ik_status_label)
        self.ik_layout.addWidget(solve)

    def _build_display_controls(self) -> None:
        group = QtWidgets.QGroupBox("Display")
        layout = QtWidgets.QVBoxLayout(group)
        row = _FlowLayout()
        for attr, label, checked in (
            ("show_rig_check", "Rig", True),
            ("show_dataset_check", "Dataset", True),
            ("show_markers_check", "Markers", True),
            ("show_links_check", "Link COMs", True),
            ("show_contacts_check", "Contacts", True),
            ("show_targets_check", "IK targets", True),
            ("show_trails_check", "Target trails", True),
        ):
            check = QtWidgets.QCheckBox(label)
            check.setChecked(checked)
            check.toggled.connect(self._refresh_view)
            setattr(self, attr, check)
            row.addWidget(check)
        layout.addLayout(row)
        self.floor_check = QtWidgets.QCheckBox("Show floor")
        self.floor_check.setChecked(True)
        self.floor_check.toggled.connect(self.viewport.set_floor_visible)
        self.fit_camera_button = QtWidgets.QPushButton("Fit Camera")
        self.fit_camera_button.clicked.connect(self._fit_camera)
        row = _FlowLayout()
        row.addWidget(self.floor_check)
        row.addWidget(self.fit_camera_button)
        layout.addLayout(row)
        self.display_layout.addWidget(group)

    def _invalidate_ik_solver(self, *_args) -> None:
        self.ik_solver = None

    def _set_ik_root_step(self, value: float) -> None:
        self.ik_root_step = value

    def _set_ik_delay(self, value: float) -> None:
        self.ik_delay_ms = int(value)
        self._ik_timer.setInterval(self.ik_delay_ms)

    def _set_ik_iterations(self, value: int) -> None:
        self.ik_max_iter = value

    def _ik_tab_active(self) -> bool:
        return self.left_tabs.currentIndex() == 4

    def _on_ik_tab_changed(self, *_args) -> None:
        active = self._ik_tab_active()
        for check in (
            self.show_markers_check,
            self.show_links_check,
            self.show_contacts_check,
            self.show_targets_check,
            self.show_trails_check,
        ):
            check.setEnabled(active)
        if not active:
            self._cancel_target_placement()
        if self.model is not None:
            self._refresh_view()

    def _dataset_ik_mode(self) -> bool:
        return self.ik_target_source.currentData() == "dataset"

    def _on_ik_source_changed(self, *_args) -> None:
        self._stop_ik()
        self._cancel_target_placement()
        if self._dataset_ik_mode():
            self._manual_ik_targets = self.target_specs
            self.target_specs = []
            self._preset_preview = False
            self.animation_timer.stop()
            self.sequence_timer.stop()
            blocker = QtCore.QSignalBlocker(self.playback_source_combo)
            self.playback_source_combo.setCurrentIndex(1)
            del blocker
            self._refresh_dataset_ik_targets()
        else:
            self.target_specs = self._manual_ik_targets
            self._sync_ik_targets()
        self.ik_solver = None
        self._update_playback_controls()
        self._refresh_view()

    def _refresh_dataset_ik_targets(self) -> None:
        if not self._dataset_ik_mode():
            return
        if self.loaded_dataset is None or self.current_sample is None:
            blocker = QtCore.QSignalBlocker(self.ik_target_source)
            self.ik_target_source.setCurrentIndex(0)
            del blocker
            self.target_specs = self._manual_ik_targets
            self._sync_ik_targets()
            return
        joints = np.asarray(self.target_joints_3d)
        weights = np.ones(self.model.NUM_JOINTS)
        confidence = self.current_sample.get("confidences")
        if confidence is not None:
            values = confidence.detach().cpu().numpy().squeeze()
            if values.ndim == 2 and values.shape[0] == self.model.NUM_JOINTS:
                values = values.mean(axis=-1)
            try:
                weights = np.broadcast_to(values, weights.shape).copy()
            except ValueError:
                weights = np.zeros_like(weights)
        parents = set(self.model.parents)
        targets = []
        valid_weights = []
        for index, name in enumerate(self.model.joint_names):
            if index in parents or index == self.model.root_index:
                continue
            if (
                not np.isfinite(joints[index]).all()
                or not np.isfinite(weights[index])
                or weights[index] <= 0
            ):
                continue
            targets.append(
                TargetMotionSpec(
                    name=f"Dataset: {name}",
                    frame=name,
                    base=joints[index].copy(),
                    color=SCENE_COLORS["target_a"],
                    phase=0,
                    speed=0,
                    height=0,
                    x_amp=0,
                    y_amp=0,
                    drift=np.zeros(3),
                    frozen=True,
                )
            )
            valid_weights.append(weights[index])
        self.target_specs = targets
        self._ik_confidence = np.asarray(valid_weights)
        self.ik_solver = None
        self._sync_ik_targets()

    def _complete_ik_frame(self, error: float) -> None:
        results = self._ik_sequence_results
        if results is None:
            return
        results.append(
            {
                "full_pose": self.full_pose.detach().clone(),
                "transl": self.translation.detach().clone(),
                "scales": (self.scales * self.global_bone_scale).detach().clone(),
                "joints": self._current_joints().copy(),
                "error": error,
            }
        )
        if self.current_sample_index < self.range_end_index:
            self.ik_iter = 0
            self._apply_sample_index(self.current_sample_index + 1)
            return
        payload = {
            key: torch.stack([frame[key] for frame in results])
            for key in ("full_pose", "transl", "scales")
        }
        errors = [frame["error"] for frame in results if np.isfinite(frame["error"])]
        payload.update(
            joints=np.stack([frame["joints"] for frame in results]),
            losses={"end_effector_error": float(np.mean(errors))} if errors else {},
            iterations=self.ik_total_steps,
            mode="ik",
            frame_count=len(results),
        )
        self._stop_ik()
        self._on_fit_finished(payload)
        self.ik_status_label.setText(f"Completed IK across {len(results)} frames.")
        self._update_fit_controls()

    def _reset_ik_for_rig(self) -> None:
        self._stop_ik()
        self._cancel_target_placement()
        self.ik_solver = None
        self.ik_iter = 0
        self._ik_fk_output = None
        self._ik_target_time = time.perf_counter()
        self.target_specs = []
        self._manual_ik_targets = []
        blocker = QtCore.QSignalBlocker(self.ik_target_source)
        self.ik_target_source.setCurrentIndex(0)
        del blocker
        self._sync_ik_targets()
        self.ik_status_label.setText("Add a target to start.")
        self._ik_timer.start()

    def _selected_ik_target(self) -> TargetMotionSpec | None:
        name = self.ik_target_combo.currentData()
        return next((spec for spec in self.target_specs if spec.name == name), None)

    def _sync_ik_targets(self, selected: str | None = None) -> None:
        selected = selected or self.ik_target_combo.currentData()
        blocker = QtCore.QSignalBlocker(self.ik_target_combo)
        self.ik_target_combo.clear()
        for spec in self.target_specs:
            self.ik_target_combo.addItem(f"{spec.name} · {spec.frame}", spec.name)
        index = self.ik_target_combo.findData(selected)
        self.ik_target_combo.setCurrentIndex(index if index >= 0 else 0)
        del blocker
        self._edit_ik_target()
        self._update_ik_controls()

    def _edit_ik_target(self, *_args) -> None:
        self._cancel_target_placement()
        spec = self._selected_ik_target()
        self.ik_target_editor.setVisible(
            spec is not None and not self._dataset_ik_mode()
        )
        if spec is None or self.model is None or self._dataset_ik_mode():
            return
        used = {target.frame for target in self.target_specs if target is not spec}
        blocker = QtCore.QSignalBlocker(self.ik_joint_combo)
        self.ik_joint_combo.clear()
        self.ik_joint_combo.addItems(
            [name for name in self.model.joint_names if name not in used]
        )
        self.ik_joint_combo.setCurrentText(spec.frame)
        del blocker
        blocker = QtCore.QSignalBlocker(self.ik_moving_check)
        self.ik_moving_check.setChecked(not spec.frozen)
        del blocker
        for spin, value in zip(self.ik_position_spins, spec.base):
            blocker = QtCore.QSignalBlocker(spin)
            spin.setValue(float(value))
            del blocker

    def _add_ik_target(self, *_args) -> None:
        if (
            self.model is None
            or self.fit_thread is not None
            or len(self.target_specs) >= self.MAX_TARGETS
        ):
            return
        used = {spec.frame for spec in self.target_specs}
        available = [name for name in self.model.joint_names if name not in used]
        if not available:
            return
        preferred = self.pose_joint_combo.currentText()
        frame, accepted = QtWidgets.QInputDialog.getItem(
            self,
            "Add Target",
            "Joint",
            available,
            available.index(preferred) if preferred in available else 0,
            False,
        )
        if not accepted:
            return
        self._stop_ik()
        self._ik_target_serial += 1
        extent = max(float(np.ptp(self._current_joints(), axis=0).max()), 0.05)
        color = QtGui.QColor.fromHsvF(
            ((self._ik_target_serial - 1) * 0.618) % 1, 0.8, 0.9
        )
        spec = TargetMotionSpec(
            name=f"Target {self._ik_target_serial}",
            frame=frame,
            base=self._current_joints()[self.model.joint_names.index(frame)].copy(),
            color=color.getRgbF(),
            phase=0.0,
            speed=0.5,
            height=extent * 0.1,
            x_amp=extent * 0.05,
            y_amp=extent * 0.05,
            drift=np.zeros(3),
            frozen=True,
        )
        self.target_specs.append(spec)
        self.ik_solver = None
        self._sync_ik_targets(spec.name)
        self.ik_status_label.setText("Place targets, then run IK.")
        self._refresh_view()

    def _remove_ik_target(self, *_args) -> None:
        spec = self._selected_ik_target()
        if spec is None:
            return
        self._stop_ik()
        self._cancel_target_placement()
        self.target_specs = [
            target for target in self.target_specs if target is not spec
        ]
        self.ik_solver = None
        self._sync_ik_targets()
        self.ik_status_label.setText(
            "Targets updated." if self.target_specs else "Add a target to start."
        )
        self._refresh_view()

    def _on_ik_target_changed(self, *_args) -> None:
        spec = self._selected_ik_target()
        if spec is None:
            return
        frame = self.ik_joint_combo.currentText()
        if any(
            other is not spec and other.frame == frame for other in self.target_specs
        ):
            return
        spec.frame = frame
        spec.frozen = not self.ik_moving_check.isChecked()
        spec.base = np.array([spin.value() for spin in self.ik_position_spins])
        blocker = QtCore.QSignalBlocker(self.ik_target_combo)
        self.ik_target_combo.setItemText(
            self.ik_target_combo.currentIndex(), f"{spec.name} · {spec.frame}"
        )
        del blocker
        self.ik_solver = None
        if self._placement_target is not None:
            self.viewport.set_placement_height(float(spec.base[1]))
        self._refresh_view()

    def _begin_target_placement(self, checked: bool) -> None:
        spec = self._selected_ik_target()
        if not checked or spec is None or not self._ik_tab_active():
            self._cancel_target_placement()
            return
        self._placement_target = spec.name
        self.viewport.set_placement_height(float(spec.base[1]))
        self.ik_status_label.setText(
            f"Click scene at Y={spec.base[1]:.3f} to place {spec.name}. Esc cancels."
        )

    def _cancel_target_placement(self, *_args) -> None:
        self._placement_target = None
        self.viewport.set_placement_height(None)
        blocker = QtCore.QSignalBlocker(self.ik_place_button)
        self.ik_place_button.setChecked(False)
        del blocker

    def _place_ik_target(self, position: object) -> None:
        spec = next(
            (
                target
                for target in self.target_specs
                if target.name == self._placement_target
            ),
            None,
        )
        self._cancel_target_placement()
        if spec is None:
            return
        spec.base = np.asarray(position, dtype=float)
        self.ik_solver = None
        self._sync_ik_targets(spec.name)
        self.ik_status_label.setText(f"Placed {spec.name}.")
        self._refresh_view()

    def _randomize_ik_targets(self, *_args) -> None:
        spec = self._selected_ik_target()
        if spec is None:
            return
        rng = np.random.default_rng()
        extent = max(float(np.ptp(self._current_joints(), axis=0).max()), 0.05)
        spec.phase = float(rng.uniform(0, 1))
        spec.speed = float(rng.uniform(0.44, 0.66))
        spec.height = extent * float(rng.uniform(0.05, 0.15))
        spec.x_amp = extent * float(rng.uniform(0.02, 0.10))
        spec.y_amp = extent * float(rng.uniform(0.02, 0.10))
        spec.drift = extent * rng.uniform(-0.01, 0.01, 3)
        spec.drift[1] = 0.0
        spec.frozen = False
        self._sync_ik_targets(spec.name)
        self._refresh_view()

    def _center_ik_targets(self, *_args) -> None:
        spec = self._selected_ik_target()
        if spec is None:
            return
        spec.base = self._current_joints()[
            self.model.joint_names.index(spec.frame)
        ].copy()
        self._sync_ik_targets(spec.name)
        self._refresh_view()

    def _ik_pose_state(self) -> tuple[torch.Tensor, torch.Tensor]:
        pose, translation = self._current_pose_state()
        q = self.model.zero_pose()
        for index, name in enumerate(self.model.joint_names):
            if index == self.model.root_index or self.model.spec.joint_dof(index) == 0:
                continue
            section = self.model.dof_slice(name)
            if self.model.spec.joint_dof(index) == 1:
                axis = (self.model.spec.joint_axes or (None,) * self.model.NUM_JOINTS)[
                    index
                ] or (1.0, 0.0, 0.0)
                axis = q.new_tensor(axis)
                q[:, section] = (pose[index] * axis / axis.norm()).sum()
            else:
                q[:, section] = pose[index]
        root = self.model.identity_root()
        root[:, :3, :3] = self.ik_api.axis_angle_to_matrix(pose[self.model.root_index])
        root[:, :3, 3] = translation
        # Public IK solvers read body scaling from the shared model state.
        self.model.scales = (self.scales * self.global_bone_scale).unsqueeze(0)
        return q, root

    def _take_ik_control(self) -> None:
        pose, translation = self._current_pose_state()
        self.full_pose, self.translation = pose.detach(), translation.detach()
        self.animation_timer.stop()
        self.sequence_timer.stop()
        self._preset_preview = False
        self.preview_override_joints = None
        self._update_playback_controls()

    def _run_ik(self, *_args) -> None:
        if self.model is None or self.fit_thread is not None:
            return
        if self._dataset_ik_mode():
            if self.loaded_dataset is None or self.current_sample is None:
                return
        elif not self.target_specs:
            return
        self._take_ik_control()
        self.ik_running = True
        self.ik_iter = 0
        self.ik_total_steps = 0
        self.metric_plot.set_series("IK frame error", [], [])
        self.ik_solver = None
        if self._dataset_ik_mode():
            self._ik_sequence_results = []
            self.sequence_fit_payload = None
            blocker = QtCore.QSignalBlocker(self.playback_source_combo)
            self.playback_source_combo.setCurrentIndex(1)
            del blocker
            self._apply_sample_index(self.range_start_index)
        self._update_fit_controls()
        self._log(
            "Started dataset sequence IK." if self._dataset_ik_mode() else "Started IK."
        )

    def _stop_ik(self, *_args) -> None:
        was_running = self.ik_running
        self.ik_running = False
        self._ik_sequence_results = None
        if hasattr(self, "ik_run_button"):
            if was_running:
                self.ik_status_label.setText("Stopped. Current pose retained.")
            self._update_fit_controls()

    def _step_ik_once(self, *_args) -> None:
        if self.model is None or self.fit_thread is not None or not self.target_specs:
            return
        self._stop_ik()
        self._take_ik_control()
        self._solve_ik_step()

    def _reset_ik_pose(self, *_args) -> None:
        self._stop_ik()
        self._take_ik_control()
        self.full_pose.zero_()
        self.translation.zero_()
        self.ik_iter = 0
        self.ik_total_steps = 0
        self.metric_plot.set_series("IK frame error", [], [])
        self.ik_solver = None
        self.ik_status_label.setText("Idle.")
        self._sync_pose_sliders()
        self._sync_translation_sliders()
        self._refresh_view()

    def _solve_ik_step(self) -> None:
        if not self.target_specs:
            if self._ik_sequence_results is not None:
                self._log(
                    f"Frame {self.current_sample_index}: no valid end effectors; retained preceding pose."
                )
                self._complete_ik_frame(float("nan"))
            return
        q, root = self._ik_pose_state()
        frames = [spec.frame for spec in self.target_specs]
        if self.ik_solver is None:
            solver = {
                "ccd": self.ik_api.ccd_ik_cls,
                "dls": self.ik_api.dls_ik_cls,
                "gradient": self.ik_api.gradient_ik_cls,
            }[self.ik_solver_combo.currentData()]
            self.ik_solver = solver(self.model, frames, max_iter=1)
        targets = q.new_tensor(self._ik_target_positions()).unsqueeze(0)
        weights = q.new_tensor(
            self._ik_confidence if self._dataset_ik_mode() else np.ones(len(frames))
        ).unsqueeze(0)
        q = self.ik_solver.solve(q, targets, root=root, confidence=weights).q
        for index, name in enumerate(self.model.joint_names):
            if index == self.model.root_index or self.model.spec.joint_dof(index) == 0:
                continue
            values = q[0, self.model.dof_slice(name)]
            if values.numel() == 1:
                axis = (self.model.spec.joint_axes or (None,) * self.model.NUM_JOINTS)[
                    index
                ] or (1.0, 0.0, 0.0)
                axis = q.new_tensor(axis)
                values = values * axis / axis.norm()
            self.full_pose[index] = values.detach()
        self._clamp_full_pose_in_place()
        q, root = self._ik_pose_state()
        current = self.model.frame_positions(q, root, frames)
        self.translation += (
            self.ik_root_step
            * ((targets - current) * weights.unsqueeze(-1)).sum(dim=1)[0].detach()
            / weights.sum().clamp_min(1e-8)
        )
        q, root = self._ik_pose_state()
        error = float(
            (targets - self.model.frame_positions(q, root, frames))
            .norm(dim=-1)
            .mean()
            .detach()
        )
        self.ik_iter += 1
        self.ik_total_steps += 1
        self.metric_plot.add_point("IK frame error", float(self.ik_total_steps), error)
        frame_note = (
            f"Frame {self.current_sample_index - self.range_start_index + 1}/{self._selected_range_length()} · "
            if self._dataset_ik_mode()
            else ""
        )
        self.ik_status_label.setText(
            f"{frame_note}Iteration {self.ik_iter}/{self.ik_max_iter} · error {error:.4f}"
        )
        self._sync_pose_sliders()
        self._sync_translation_sliders()
        self._refresh_view()
        if self.ik_iter >= self.ik_max_iter:
            if self._ik_sequence_results is not None:
                self._complete_ik_frame(error)
            else:
                self.ik_iter = 0

    def _update_ik_controls(self) -> None:
        ready = self.model is not None and self.fit_thread is None
        dataset_mode = self._dataset_ik_mode()
        dataset_ready = (
            self.loaded_dataset is not None and self.current_sample is not None
        )
        sequence_running = self._ik_sequence_results is not None
        self.ik_target_source.model().item(1).setEnabled(dataset_ready)
        self.ik_target_source.setEnabled(ready and not self.ik_running)
        self.ik_dataset_note.setVisible(dataset_mode)
        self.ik_target_editor.setVisible(not dataset_mode and bool(self.target_specs))
        for widget in (
            self.ik_solver_combo,
            self.ik_iterations_spin,
            self.ik_root_step_slider,
        ):
            widget.setEnabled(ready and not sequence_running)
        self.ik_export_button.setEnabled(
            ready and not self.ik_running and self.sequence_fit_payload is not None
        )
        self.ik_run_button.setEnabled(
            ready
            and (bool(self.target_specs) or dataset_mode and dataset_ready)
            and not self.ik_running
        )
        self.ik_step_button.setEnabled(
            ready and bool(self.target_specs) and not self.ik_running
        )
        self.ik_reset_button.setEnabled(ready)
        self.ik_stop_button.setEnabled(self.ik_running)
        self.ik_targets_group.setEnabled(ready)
        used = {spec.frame for spec in self.target_specs}
        self.ik_add_target_button.setEnabled(
            ready
            and not dataset_mode
            and len(used)
            < min(self.MAX_TARGETS, self.model.NUM_JOINTS if self.model else 0)
        )
        self.ik_edit_target_button.setEnabled(
            ready and not dataset_mode and bool(self.target_specs)
        )
        self.ik_remove_target_button.setEnabled(
            ready and not dataset_mode and bool(self.target_specs)
        )
        if not ready:
            self._cancel_target_placement()

    def _ik_target_positions(self) -> np.ndarray:
        t = time.perf_counter() - self._ik_target_time
        return np.array([spec.position(t) for spec in self.target_specs]).reshape(-1, 3)

    def _ik_tick(self) -> None:
        if self.model is None or self.fit_thread is not None or not self.isVisible():
            return
        if self.ik_running:
            self._solve_ik_step()
        elif (
            self._ik_tab_active()
            and self.show_targets_check.isChecked()
            and any(not spec.frozen for spec in self.target_specs)
        ):
            self._refresh_view()

    def _refresh_ik_layers(self) -> None:
        q, root = self._ik_pose_state()
        with torch.no_grad():
            self._ik_fk_output = self.model.forward_kinematics(q, root)
        fk = self._ik_fk_output
        active = self._ik_tab_active()
        for kind, check, positions, color in (
            (
                "markers",
                self.show_markers_check,
                fk.marker_positions[0],
                SCENE_COLORS["marker"],
            ),
            (
                "contacts",
                self.show_contacts_check,
                fk.contact_positions[0],
                SCENE_COLORS["contact"],
            ),
        ):
            points = positions.detach().cpu().numpy()
            self.viewport.set_scatter_layer(
                kind,
                points if active and check.isChecked() and len(points) else None,
                size=0.06,
                color=color,
            )
        coms = self._ik_link_positions()
        self.viewport.set_scatter_layer(
            "links",
            coms
            if active and self.show_links_check.isChecked() and len(coms)
            else None,
            size=0.07,
            color=SCENE_COLORS["link"],
        )
        targets = self._ik_target_positions()
        self.viewport.set_scatter_layer(
            "targets",
            targets
            if active and self.show_targets_check.isChecked() and len(targets)
            else None,
            size=0.15,
            colors=np.array([spec.color for spec in self.target_specs]),
        )
        t = time.perf_counter() - self._ik_target_time
        for index, spec in enumerate(self.target_specs):
            trail = None
            if (
                active
                and not spec.frozen
                and self.show_targets_check.isChecked()
                and self.show_trails_check.isChecked()
            ):
                trail = np.array(
                    [
                        spec.position(float(value))
                        for value in np.linspace(max(0.0, t - 1.25), t, 40)
                    ]
                )
            self.viewport.set_line_layer(
                f"trail_{index}", trail, color=spec.color, width=2.0
            )

        for index in range(len(self.target_specs), self._ik_trail_count):
            self.viewport.set_line_layer(
                f"trail_{index}", None, color=SCENE_COLORS["target_a"], width=2.0
            )

        self._ik_trail_count = len(self.target_specs)

    def _ik_link_positions(self) -> np.ndarray:
        fk = self._ik_fk_output
        positions = []
        for index, link in enumerate(self.model.spec.links):
            local = fk.scales[0, index] * fk.joints.new_tensor(link.com)
            point = fk.joints[0, index] + fk.global_rotations[0, index] @ local
            positions.append(point.detach().cpu().numpy())
        return np.array(positions).reshape(-1, 3)

    def _ik_scene_nodes(self) -> list[SceneNode]:
        fk = self._ik_fk_output
        if fk is None:
            return []
        nodes = []
        for kind, frames, positions in (
            (
                "marker",
                self.model.spec.markers,
                fk.marker_positions[0].detach().cpu().numpy(),
            ),
            (
                "contact",
                self.model.spec.contacts,
                fk.contact_positions[0].detach().cpu().numpy(),
            ),
            ("link", self.model.spec.links, self._ik_link_positions()),
        ):
            for index, (frame, position) in enumerate(zip(frames, positions)):
                properties = {"world_position": position, **vars(frame)}
                nodes.append(
                    SceneNode(
                        group=kind.title() + "s",
                        label=frame.name,
                        kind=kind,
                        payload=index,
                        properties=properties,
                    )
                )
        for index, (spec, position) in enumerate(
            zip(self.target_specs, self._ik_target_positions())
        ):
            nodes.append(
                SceneNode(
                    group="Targets",
                    label=spec.name,
                    kind="target",
                    payload=spec.name,
                    properties={
                        "frame": spec.frame,
                        "base": spec.base,
                        "position": position,
                        "frozen": spec.frozen,
                        "speed": spec.speed,
                        "height": spec.height,
                    },
                )
            )
        return nodes
