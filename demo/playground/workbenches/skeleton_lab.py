from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PySide6 import QtCore, QtWidgets

from ..compat import CompatibilityImportError, load_skeleton_api
from ..models import FitHistoryPoint, SceneNode
from ..theme import SECONDARY_TEXT_STYLE
from ..viewport import SceneViewport
from ..widgets import (
    EventLogWidget,
    FloatSlider,
    InspectorPanel,
    MetricPlotWidget,
    ProjectionPreview,
    SceneExplorer,
    _FlowLayout,
    _scroll_panel,
)
from .ik_lab import IKControls


class SkeletonWorkbench(IKControls, QtWidgets.QWidget):
    status_message = QtCore.Signal(str, int)

    ANIMATION_DT = 1.0 / 60.0

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("PlaygroundRoot")
        self._api_error: str | None = None
        try:
            self.api = load_skeleton_api()
        except CompatibilityImportError as exc:  # pragma: no cover - runtime dependency
            self.api = None
            self._api_error = str(exc)
            self._build_unavailable_ui(str(exc))
            return

        self.model = None
        self.current_skeleton_name: str | None = None
        self.available_animations: list[Any] = []
        self.animation_time = 0.0
        self.animation_speed = 1.0
        self.full_pose = torch.zeros(1, 3)
        self.translation = torch.zeros(3)
        self.scales = torch.ones(1, 3)
        self.global_bone_scale = 1.0
        self.rom_limits: dict[str, list[Any]] = {}
        self.scale_joint_indices: list[int] = []
        self.loaded_dataset = None
        self.dataset_path: Path | None = None
        self.range_start_index = 0
        self.range_end_index = 0
        self.current_sample_index = 0
        self.range_sequence: dict[str, torch.Tensor] | None = None
        self.sequence_fit_payload: dict[str, object] | None = None
        self.current_sample: dict[str, torch.Tensor] | None = None
        self.target_joints_3d: np.ndarray | None = None
        self.target_joints_2d: np.ndarray | None = None
        self.current_camera = None
        self.priors_path: Path | None = None
        self.preview_override_joints: np.ndarray | None = None
        self.fit_thread: QtCore.QThread | None = None
        self.fit_worker = None
        self.fit_history: list[FitHistoryPoint] = []
        self.selected_node: SceneNode | None = None
        self._preset_preview = True
        self._init_ik()

        self.animation_timer = QtCore.QTimer(self)
        self.animation_timer.setInterval(int(round(self.ANIMATION_DT * 1000)))
        self.animation_timer.timeout.connect(self._advance_animation)
        self.sequence_timer = QtCore.QTimer(self)
        self.sequence_timer.setInterval(120)
        self.sequence_timer.timeout.connect(self._advance_sequence_frame)

        self._build_ui()
        default_name = (
            "human36m"
            if "human36m" in self.api.supported_skeletons
            else self.api.supported_skeletons[0]
        )
        self._load_skeleton_by_name(default_name)

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------
    def _build_unavailable_ui(self, message: str) -> None:
        layout = QtWidgets.QVBoxLayout(self)
        label = QtWidgets.QLabel(
            "The general skeleton workspace could not import the skeletons package.\n\n"
            + message
        )
        label.setWordWrap(True)
        layout.addWidget(label)
        layout.addStretch(1)

    def _build_ui(self) -> None:
        main_layout = QtWidgets.QHBoxLayout(self)
        main_layout.setContentsMargins(8, 8, 8, 8)
        main_layout.setSpacing(10)
        main_splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        main_layout.addWidget(main_splitter)

        self.left_tabs = QtWidgets.QTabWidget()
        self.left_tabs.setMinimumWidth(300)
        self.left_tabs.setMaximumWidth(460)
        for key, label in (
            ("rig", "Scene"),
            ("data", "Data"),
            ("move", "Move"),
            ("fit", "Fit"),
            ("ik", "IK"),
        ):
            tab = QtWidgets.QWidget()
            layout = QtWidgets.QVBoxLayout(tab)
            layout.setContentsMargins(6, 6, 6, 6)
            layout.setSpacing(8)
            setattr(self, key + "_tab", tab)
            setattr(self, key + "_layout", layout)
            self.left_tabs.addTab(_scroll_panel(tab), label)
        main_splitter.addWidget(self.left_tabs)

        self.viewport = SceneViewport()
        self.scene_explorer = SceneExplorer()
        self.scene_explorer.setMinimumHeight(160)
        self.scene_explorer.node_selected.connect(self._on_scene_node_selected)
        self.inspector = InspectorPanel()
        self.metric_plot = MetricPlotWidget("Fit / IK Metrics")
        self.event_log = EventLogWidget()
        self._build_pose_tab()
        self.rig_layout.addWidget(self.scene_explorer, 1)
        self.rig_layout.addWidget(self.inspector, 2)
        self._build_fitting_tab()
        self._build_motion_tab()
        self._build_ik_tab()
        for key in ("data", "move", "fit", "ik"):
            getattr(self, key + "_layout").addStretch(1)

        center_splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical)
        self.projection_preview = ProjectionPreview()
        center_splitter.addWidget(self.viewport)
        center_splitter.addWidget(self.projection_preview)
        center_splitter.setStretchFactor(0, 3)
        center_splitter.setStretchFactor(1, 1)
        main_splitter.addWidget(center_splitter)

        right_panel = QtWidgets.QWidget()
        right_panel.setMinimumWidth(240)
        right_panel.setMaximumWidth(380)
        right_layout = QtWidgets.QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)
        display_panel = QtWidgets.QWidget()
        self.display_layout = QtWidgets.QVBoxLayout(display_panel)
        self.display_layout.setContentsMargins(6, 6, 6, 6)
        self._build_display_controls()
        right_layout.addWidget(display_panel)
        metrics_tabs = QtWidgets.QTabWidget()
        metrics_tabs.addTab(self.metric_plot, "Metrics")
        metrics_tabs.addTab(self.event_log, "Logs")
        right_layout.addWidget(metrics_tabs, 1)
        main_splitter.addWidget(right_panel)
        main_splitter.setStretchFactor(0, 0)
        main_splitter.setStretchFactor(1, 1)
        main_splitter.setStretchFactor(2, 0)
        main_splitter.setSizes([360, 900, 300])
        self.left_tabs.currentChanged.connect(self._on_ik_tab_changed)
        self.viewport.point_placed.connect(self._place_ik_target)
        self.viewport.placement_cancelled.connect(self._cancel_target_placement)
        self._on_ik_tab_changed()

    def _build_pose_tab(self) -> None:
        layout = self.move_layout

        skeleton_group = QtWidgets.QGroupBox("Skeleton")
        skeleton_layout = QtWidgets.QVBoxLayout(skeleton_group)
        self.skeleton_combo = QtWidgets.QComboBox()
        self.skeleton_combo.addItems(sorted(self.api.supported_skeletons))
        self.skeleton_combo.currentTextChanged.connect(self._load_skeleton_by_name)
        self.skeleton_info_label = QtWidgets.QLabel()
        self.skeleton_info_label.setStyleSheet(SECONDARY_TEXT_STYLE)
        self.skeleton_info_label.setWordWrap(True)
        skeleton_layout.addWidget(self.skeleton_combo)
        skeleton_layout.addWidget(self.skeleton_info_label)
        self.rig_layout.addWidget(skeleton_group)

        pose_group = QtWidgets.QGroupBox("Joint Pose")
        pose_group.setProperty("compact", True)
        pose_layout = QtWidgets.QVBoxLayout(pose_group)
        self.pose_joint_combo = QtWidgets.QComboBox()
        self.pose_joint_combo.currentIndexChanged.connect(self._sync_pose_sliders)
        self.pose_joint_combo.currentIndexChanged.connect(self._sync_rom_controls)
        self.pose_joint_combo.currentIndexChanged.connect(self._refresh_view)
        pose_layout.addWidget(self.pose_joint_combo)
        self.pose_sliders: list[FloatSlider] = []
        for axis_name in self.api.axis_names:
            slider = FloatSlider(
                axis_name,
                minimum=-180.0,
                maximum=180.0,
                factor=10,
                decimals=1,
                suffix="°",
                compact=True,
            )
            slider.value_changed.connect(
                self._make_pose_callback(len(self.pose_sliders))
            )
            pose_layout.addWidget(slider)
            self.pose_sliders.append(slider)
        pose_button_row = _FlowLayout()
        self.zero_joint_button = QtWidgets.QPushButton("Zero Joint")
        self.zero_joint_button.clicked.connect(self._zero_selected_joint)
        self.export_state_button = QtWidgets.QPushButton("Export State…")
        self.export_state_button.clicked.connect(self._save_pose_state)
        self.import_state_button = QtWidgets.QPushButton("Load State…")
        self.import_state_button.clicked.connect(self._load_pose_state)
        pose_button_row.addWidget(self.zero_joint_button)
        pose_button_row.addWidget(self.export_state_button)
        pose_button_row.addWidget(self.import_state_button)
        pose_layout.addLayout(pose_button_row)
        layout.addWidget(pose_group)

        rom_group = QtWidgets.QGroupBox("ROM Limits")
        rom_group.setProperty("compact", True)
        rom_layout = QtWidgets.QVBoxLayout(rom_group)
        grid = QtWidgets.QGridLayout()
        for column, title in enumerate(("Axis", "Min", "Max")):
            grid.addWidget(QtWidgets.QLabel(title), 0, column)
        self.rom_enable_checks: list[QtWidgets.QCheckBox] = []
        self.rom_min_spins: list[QtWidgets.QDoubleSpinBox] = []
        self.rom_max_spins: list[QtWidgets.QDoubleSpinBox] = []
        for axis, axis_name in enumerate(self.api.axis_names):
            enable = QtWidgets.QCheckBox(axis_name)
            enable.toggled.connect(self._on_rom_limit_changed)
            grid.addWidget(enable, axis + 1, 0)
            self.rom_enable_checks.append(enable)
            min_spin = QtWidgets.QDoubleSpinBox()
            min_spin.setRange(-180.0, 180.0)
            min_spin.setDecimals(1)
            min_spin.setSingleStep(1.0)
            min_spin.setSuffix("°")
            min_spin.setMinimumWidth(0)
            min_spin.setSizePolicy(
                QtWidgets.QSizePolicy.Policy.Ignored, QtWidgets.QSizePolicy.Policy.Fixed
            )
            min_spin.valueChanged.connect(self._on_rom_limit_changed)
            grid.addWidget(min_spin, axis + 1, 1)
            self.rom_min_spins.append(min_spin)
            max_spin = QtWidgets.QDoubleSpinBox()
            max_spin.setRange(-180.0, 180.0)
            max_spin.setDecimals(1)
            max_spin.setSingleStep(1.0)
            max_spin.setSuffix("°")
            max_spin.setMinimumWidth(0)
            max_spin.setSizePolicy(
                QtWidgets.QSizePolicy.Policy.Ignored, QtWidgets.QSizePolicy.Policy.Fixed
            )
            max_spin.valueChanged.connect(self._on_rom_limit_changed)
            grid.addWidget(max_spin, axis + 1, 2)
            self.rom_max_spins.append(max_spin)
        rom_layout.addLayout(grid)
        rom_button_row = _FlowLayout()
        self.reset_joint_rom_button = QtWidgets.QPushButton("Reset Joint ROM")
        self.reset_joint_rom_button.clicked.connect(self._reset_selected_joint_rom)
        self.clear_all_rom_button = QtWidgets.QPushButton("Clear All ROM")
        self.clear_all_rom_button.clicked.connect(self._clear_all_rom)
        self.load_rom_button = QtWidgets.QPushButton("Load ROM…")
        self.load_rom_button.clicked.connect(self._load_rom_limits_from_file)
        self.save_rom_button = QtWidgets.QPushButton("Save ROM…")
        self.save_rom_button.clicked.connect(self._save_rom_limits_to_file)
        rom_button_row.addWidget(self.reset_joint_rom_button)
        rom_button_row.addWidget(self.clear_all_rom_button)
        rom_button_row.addWidget(self.load_rom_button)
        rom_button_row.addWidget(self.save_rom_button)
        rom_layout.addLayout(rom_button_row)
        layout.addWidget(rom_group)

        translation_group = QtWidgets.QGroupBox("Translation")
        translation_group.setProperty("compact", True)
        translation_layout = QtWidgets.QVBoxLayout(translation_group)
        self.translation_sliders: list[FloatSlider] = []
        for axis_name in self.api.axis_names:
            slider = FloatSlider(
                axis_name,
                minimum=-2.0,
                maximum=2.0,
                factor=100,
                decimals=2,
                suffix=" m",
                compact=True,
            )
            slider.value_changed.connect(
                self._make_translation_callback(len(self.translation_sliders))
            )
            translation_layout.addWidget(slider)
            self.translation_sliders.append(slider)
        layout.addWidget(translation_group)

        scale_group = QtWidgets.QGroupBox("Body Scale")
        scale_group.setProperty("compact", True)
        scale_layout = QtWidgets.QVBoxLayout(scale_group)
        self.global_scale_slider = FloatSlider(
            "Global scale",
            minimum=0.25,
            maximum=2.50,
            factor=100,
            decimals=2,
            suffix="x",
            compact=True,
        )
        self.global_scale_slider.value_changed.connect(self._on_global_scale_changed)
        self.scale_joint_combo = self.pose_joint_combo
        self.scale_joint_combo.currentIndexChanged.connect(self._sync_scale_sliders)
        scale_layout.addWidget(self.global_scale_slider)
        self.scale_sliders: list[FloatSlider] = []
        for axis_name in self.api.axis_names:
            slider = FloatSlider(
                axis_name,
                minimum=0.25,
                maximum=2.50,
                factor=100,
                decimals=2,
                suffix="x",
                compact=True,
            )
            slider.value_changed.connect(
                self._make_scale_callback(len(self.scale_sliders))
            )
            scale_layout.addWidget(slider)
            self.scale_sliders.append(slider)
        scale_button_row = _FlowLayout()
        self.reset_scale_button = QtWidgets.QPushButton("Reset Body")
        self.reset_scale_button.clicked.connect(self._reset_selected_bone)
        self.reset_all_button = QtWidgets.QPushButton("Reset All")
        self.reset_all_button.clicked.connect(self._reset_all)
        scale_button_row.addWidget(self.reset_scale_button)
        scale_button_row.addWidget(self.reset_all_button)
        scale_layout.addLayout(scale_button_row)
        layout.addWidget(scale_group)

    def _build_motion_tab(self) -> None:
        group = QtWidgets.QGroupBox("Motion Preset")
        layout = QtWidgets.QVBoxLayout(group)
        self.animation_combo = QtWidgets.QComboBox()
        self.animation_combo.currentIndexChanged.connect(self._on_animation_changed)
        self.animation_combo.setToolTip(
            "ROM Wander follows the configured joint limits."
        )
        layout.addWidget(self.animation_combo)
        self.data_layout.insertWidget(1, group)

        playback = QtWidgets.QGroupBox("Playback")
        playback.setProperty("compact", True)
        layout = QtWidgets.QVBoxLayout(playback)
        self.playback_source_combo = QtWidgets.QComboBox()
        self.playback_source_combo.addItem("Motion preset", "preset")
        self.playback_source_combo.addItem("Dataset / fitted sequence", "dataset")
        self.playback_source_combo.currentIndexChanged.connect(
            self._on_playback_source_changed
        )
        layout.addWidget(self.playback_source_combo)
        row = _FlowLayout()
        self.animation_toggle_button = QtWidgets.QPushButton("Play")
        self.animation_toggle_button.setProperty("prominent", True)
        self.animation_toggle_button.clicked.connect(self._toggle_playback)
        self.sequence_play_button = self.animation_toggle_button
        self.animation_restart_button = QtWidgets.QPushButton("Restart")
        self.animation_restart_button.clicked.connect(self._restart_playback)
        row.addWidget(self.animation_toggle_button)
        row.addWidget(self.animation_restart_button)
        layout.addLayout(row)
        self.animation_speed_slider = FloatSlider(
            "Speed",
            minimum=0.25,
            maximum=2.5,
            factor=100,
            decimals=2,
            suffix="x",
            compact=True,
        )
        self.animation_speed_slider.value_changed.connect(
            self._on_animation_speed_changed
        )
        self.animation_phase_slider = FloatSlider(
            "Phase", minimum=0.0, maximum=1.0, factor=1000, decimals=3, compact=True
        )
        self.animation_phase_slider.value_changed.connect(
            self._on_animation_phase_changed
        )
        layout.addWidget(self.animation_speed_slider)
        layout.addWidget(self.animation_phase_slider)
        self.sequence_frame_label = QtWidgets.QLabel("No active range.")
        self.sequence_frame_label.setStyleSheet(SECONDARY_TEXT_STYLE)
        self.sequence_frame_label.setWordWrap(True)
        layout.addWidget(self.sequence_frame_label)
        self.sample_spin = QtWidgets.QSpinBox()
        self.sample_spin.setPrefix("Frame ")
        self.sample_spin.valueChanged.connect(self._on_sample_index_changed)
        layout.addWidget(self.sample_spin)
        self.sequence_frame_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.sequence_frame_slider.valueChanged.connect(
            self._on_sequence_frame_slider_changed
        )
        layout.addWidget(self.sequence_frame_slider)
        self.move_layout.insertWidget(0, playback)

    def _update_playback_controls(self) -> None:
        preset_source = self.playback_source_combo.currentData() == "preset"
        ready = (
            self._active_animation() is not None
            if preset_source
            else self.loaded_dataset is not None and len(self.loaded_dataset) > 0
        )
        idle = self.fit_thread is None and not self.ik_running
        playing = (
            self.animation_timer.isActive()
            if preset_source
            else self.sequence_timer.isActive()
        )
        self.animation_toggle_button.setEnabled(
            ready and idle and (preset_source or self._selected_range_length() > 1)
        )
        self.animation_toggle_button.setText("Pause" if playing else "Play")
        self.animation_restart_button.setEnabled(ready and idle)
        self.animation_speed_slider.setEnabled(ready and idle)
        self.animation_phase_slider.setVisible(preset_source)
        self.animation_phase_slider.setEnabled(ready and idle)
        for widget in (
            self.sample_spin,
            self.sequence_frame_slider,
            self.sequence_frame_label,
        ):
            widget.setVisible(not preset_source)
        self.sample_spin.setEnabled(ready and idle)
        self.sequence_frame_slider.setEnabled(ready and idle)
        self.animation_combo.setEnabled(self.fit_thread is None)
        self.playback_source_combo.setEnabled(idle)

    def _on_playback_source_changed(self, *_args) -> None:
        self._stop_ik()
        self.animation_timer.stop()
        self.sequence_timer.stop()
        self._preset_preview = self.playback_source_combo.currentData() == "preset"
        if not self._preset_preview and self.loaded_dataset is not None:
            self._apply_sample_index(self.current_sample_index)
        self._update_playback_controls()
        if self.model is not None:
            self._refresh_view()

    def _toggle_playback(self, *_args) -> None:
        self._stop_ik()
        if self.playback_source_combo.currentData() == "preset":
            self._preset_preview = True
            self._toggle_animation()
        else:
            self._toggle_sequence_playback()
        self._update_playback_controls()

    def _restart_playback(self, *_args) -> None:
        if self.playback_source_combo.currentData() == "preset":
            self._preset_preview = True
            self._restart_animation()
        elif self.loaded_dataset is not None:
            self._apply_sample_index(self.range_start_index)

    def _build_fitting_tab(self) -> None:
        layout = self.fit_layout

        dataset_group = QtWidgets.QGroupBox("Load Dataset")
        dataset_layout = QtWidgets.QVBoxLayout(dataset_group)
        row = _FlowLayout()
        self.dataset_path_edit = QtWidgets.QLineEdit()
        self.dataset_path_edit.setReadOnly(True)
        self.dataset_path_edit.setPlaceholderText("No dataset loaded")
        self.load_dataset_button = QtWidgets.QPushButton("Dataset…")
        self.load_dataset_button.clicked.connect(self._load_dataset_from_file)
        self.clear_dataset_button = QtWidgets.QPushButton("Clear")
        self.clear_dataset_button.clicked.connect(self._clear_dataset)
        row.addWidget(self.dataset_path_edit, 1)
        row.addWidget(self.load_dataset_button)
        row.addWidget(self.clear_dataset_button)
        dataset_layout.addLayout(row)

        synthetic_group = QtWidgets.QGroupBox("Generate Synthetic Data")
        synthetic_layout = QtWidgets.QVBoxLayout(synthetic_group)
        controls = self.api.synthetic_fitting_controls_cls()
        grid = QtWidgets.QFormLayout()
        grid.setRowWrapPolicy(QtWidgets.QFormLayout.RowWrapPolicy.WrapLongRows)
        self.synthetic_frames_spin = QtWidgets.QSpinBox()
        self.synthetic_frames_spin.setRange(1, 4096)
        self.synthetic_frames_spin.setValue(int(controls.num_frames))
        self.synthetic_pose_std_spin = QtWidgets.QDoubleSpinBox()
        self.synthetic_pose_std_spin.setRange(0.0, 1.5)
        self.synthetic_pose_std_spin.setDecimals(3)
        self.synthetic_pose_std_spin.setSingleStep(0.01)
        self.synthetic_pose_std_spin.setValue(float(controls.pose_std))
        self.synthetic_noise_spin = QtWidgets.QDoubleSpinBox()
        self.synthetic_noise_spin.setRange(0.0, 100.0)
        self.synthetic_noise_spin.setDecimals(1)
        self.synthetic_noise_spin.setSingleStep(1.0)
        self.synthetic_noise_spin.setSuffix(" px")
        self.synthetic_noise_spin.setValue(float(controls.noise_std))
        self.synthetic_dropout_spin = QtWidgets.QDoubleSpinBox()
        self.synthetic_dropout_spin.setRange(0.0, 0.95)
        self.synthetic_dropout_spin.setDecimals(2)
        self.synthetic_dropout_spin.setSingleStep(0.05)
        self.synthetic_dropout_spin.setValue(float(controls.confidence_dropout))
        self.synthetic_seed_spin = QtWidgets.QSpinBox()
        self.synthetic_seed_spin.setRange(-1, 2_147_483_647)
        self.synthetic_seed_spin.setSpecialValueText("Random")
        self.synthetic_seed_spin.setValue(
            -1 if getattr(controls, "seed", None) is None else int(controls.seed)
        )
        self.generate_synthetic_button = QtWidgets.QPushButton("Generate Dataset")
        self.generate_synthetic_button.clicked.connect(self._generate_synthetic_dataset)
        grid.addRow("Frames", self.synthetic_frames_spin)
        grid.addRow("Pose σ", self.synthetic_pose_std_spin)
        grid.addRow("Noise", self.synthetic_noise_spin)
        grid.addRow("Dropout", self.synthetic_dropout_spin)
        grid.addRow("Seed", self.synthetic_seed_spin)
        grid.addRow(self.generate_synthetic_button)
        synthetic_layout.addLayout(grid)

        range_row = _FlowLayout()
        self.range_start_spin = QtWidgets.QSpinBox()
        self.range_start_spin.setEnabled(False)
        self.range_start_spin.valueChanged.connect(self._on_range_start_changed)
        self.range_end_spin = QtWidgets.QSpinBox()
        self.range_end_spin.setEnabled(False)
        self.range_end_spin.valueChanged.connect(self._on_range_end_changed)
        range_row.addWidget(QtWidgets.QLabel("Range"))
        range_row.addWidget(self.range_start_spin)
        range_row.addWidget(QtWidgets.QLabel("to"))
        range_row.addWidget(self.range_end_spin)
        dataset_layout.addLayout(range_row)

        self.dataset_info_label = QtWidgets.QLabel("No fitting dataset loaded.")
        self.dataset_info_label.setStyleSheet(SECONDARY_TEXT_STYLE)
        self.dataset_info_label.setWordWrap(True)
        dataset_layout.addWidget(self.dataset_info_label)
        self.data_layout.addWidget(dataset_group)
        self.data_layout.addWidget(synthetic_group)

        fit_group = QtWidgets.QGroupBox("Fit")
        fit_layout = QtWidgets.QVBoxLayout(fit_group)
        priors_group = QtWidgets.QGroupBox("Priors")
        priors_layout = QtWidgets.QVBoxLayout(priors_group)
        priors_row = _FlowLayout()
        self.priors_path_edit = QtWidgets.QLineEdit()
        self.priors_path_edit.setReadOnly(True)
        self.priors_path_edit.setPlaceholderText("Optional priors checkpoint")
        self.load_priors_button = QtWidgets.QPushButton("Priors…")
        self.load_priors_button.clicked.connect(self._select_priors_checkpoint)
        self.clear_priors_button = QtWidgets.QPushButton("Clear")
        self.clear_priors_button.clicked.connect(self._clear_priors_checkpoint)
        priors_row.addWidget(self.priors_path_edit, 1)
        priors_row.addWidget(self.load_priors_button)
        priors_row.addWidget(self.clear_priors_button)
        priors_layout.addLayout(priors_row)
        layout.addWidget(priors_group)
        mode_row = _FlowLayout()
        self.fit_mode_combo = QtWidgets.QComboBox()
        self.fit_mode_combo.addItem("3D joints", userData="3d")
        self.fit_mode_combo.addItem("2D reprojection", userData="2d")
        self.fit_mode_combo.currentIndexChanged.connect(self._update_fit_controls)
        self.fit_optimize_scale_check = QtWidgets.QCheckBox("Optimize body scales")
        self.fit_optimize_scale_check.setChecked(True)
        mode_row.addWidget(QtWidgets.QLabel("Mode"))
        mode_row.addWidget(self.fit_mode_combo)
        mode_row.addWidget(self.fit_optimize_scale_check)
        fit_layout.addLayout(mode_row)
        self.fit_pose_prior_check = QtWidgets.QCheckBox("Use pose prior latent")
        self.fit_pose_prior_check.setChecked(True)
        self.fit_init_from_ik_check = QtWidgets.QCheckBox("Initialize from 3D IK")
        self.fit_init_from_ik_check.setChecked(True)
        fit_layout.addWidget(self.fit_pose_prior_check)
        fit_layout.addWidget(self.fit_init_from_ik_check)
        hparams_row = _FlowLayout()
        self.fit_iters_spin = QtWidgets.QSpinBox()
        self.fit_iters_spin.setRange(10, 4000)
        self.fit_iters_spin.setValue(220)
        self.fit_lr_spin = QtWidgets.QDoubleSpinBox()
        self.fit_lr_spin.setRange(0.0001, 0.1000)
        self.fit_lr_spin.setDecimals(4)
        self.fit_lr_spin.setSingleStep(0.0010)
        self.fit_lr_spin.setValue(0.0100)
        hparams_row.addWidget(QtWidgets.QLabel("Iters"))
        hparams_row.addWidget(self.fit_iters_spin)
        hparams_row.addWidget(QtWidgets.QLabel("LR"))
        hparams_row.addWidget(self.fit_lr_spin)
        fit_layout.addLayout(hparams_row)
        self.fit_progress_bar = QtWidgets.QProgressBar()
        self.fit_progress_bar.setRange(0, 1)
        self.fit_progress_bar.setValue(0)
        self.fit_status_label = QtWidgets.QLabel("Idle.")
        self.fit_status_label.setStyleSheet(SECONDARY_TEXT_STYLE)
        self.fit_status_label.setWordWrap(True)
        fit_layout.addWidget(self.fit_progress_bar)
        fit_layout.addWidget(self.fit_status_label)
        fit_button_row = _FlowLayout()
        self.run_fit_button = QtWidgets.QPushButton("Run Range Fit")
        self.run_fit_button.setProperty("prominent", True)
        self.run_fit_button.clicked.connect(self._start_fit)
        self.cancel_fit_button = QtWidgets.QPushButton("Cancel")
        self.cancel_fit_button.clicked.connect(self._cancel_fit)
        self.save_fit_button = QtWidgets.QPushButton("Save Fit…")
        self.save_fit_button.clicked.connect(self._save_fit_result_to_file)
        fit_button_row.addWidget(self.run_fit_button)
        fit_button_row.addWidget(self.cancel_fit_button)
        fit_button_row.addWidget(self.save_fit_button)
        fit_layout.addLayout(fit_button_row)
        layout.addWidget(fit_group)

    # ------------------------------------------------------------------
    # Logging / status helpers
    # ------------------------------------------------------------------
    def _log(self, message: str, timeout_ms: int = 4000) -> None:
        self.event_log.append_event(message)
        self.status_message.emit(message, timeout_ms)

    def _set_fit_idle(self, message: str = "Idle.") -> None:
        self.fit_progress_bar.setRange(0, 1)
        self.fit_progress_bar.setValue(0)
        self.fit_status_label.setText(message)
        self._update_fit_controls()

    # ------------------------------------------------------------------
    # Model / selection helpers
    # ------------------------------------------------------------------
    def _load_skeleton_by_name(self, skeleton_name: str) -> None:
        if self.api is None:
            return
        self._stop_ik()
        self.sequence_timer.stop()
        self.current_skeleton_name = skeleton_name
        blocker = QtCore.QSignalBlocker(self.skeleton_combo)
        self.skeleton_combo.setCurrentText(skeleton_name)
        del blocker
        self._preset_preview = True
        self.model = self.api.create(
            skeleton_name,
            create_global_orient=False,
            create_body_pose=False,
            create_scales=False,
            create_transl=False,
        )
        dtype = self.model.rest_offsets.dtype
        self.full_pose = torch.zeros(self.model.NUM_JOINTS, 3, dtype=dtype)
        self.translation = torch.zeros(3, dtype=dtype)
        self.scales = torch.ones(self.model.NUM_JOINTS, 3, dtype=dtype)
        self.global_bone_scale = 1.0
        self.rom_limits = self.api.default_rom_limits(self.model.joint_names)
        self.scale_joint_indices = list(range(self.model.NUM_JOINTS))
        self.animation_time = 0.0
        self.animation_timer.stop()
        self.preview_override_joints = None
        self.target_joints_3d = None
        self.target_joints_2d = None
        self.current_camera = None
        self.current_sample = None
        self.range_sequence = None
        self.sequence_fit_payload = None
        self.dataset_path = None
        self.loaded_dataset = None
        self.priors_path = None
        self.priors_path_edit.clear()
        self.viewport.set_topology(tuple(self.model.parents))
        self.projection_preview.set_topology(tuple(self.model.parents))
        self._populate_joint_selectors()
        self._populate_animation_selectors()
        self._sync_pose_sliders()
        self._sync_rom_controls()
        self._sync_translation_sliders()
        self._sync_scale_sliders()
        self._sync_global_scale_slider()
        self._sync_animation_phase_slider()
        self._update_animation_controls()
        self._clear_dataset_controls()
        self._set_fit_idle()
        self.skeleton_info_label.setText(
            f"Spec: {self.model.spec.name}\nRoot joint: {self.model.joint_names[self.model.root_index]}\nJoints: {self.model.NUM_JOINTS}"
        )
        self._reset_ik_for_rig()
        self._refresh_view(fit_camera=True)
        self._refresh_scene_explorer()
        self._log(f"Loaded skeleton: {self.model.spec.name}")

    def _populate_joint_selectors(self) -> None:
        pose_joint = self.model.root_index
        self.pose_joint_combo.blockSignals(True)
        self.pose_joint_combo.clear()
        self.pose_joint_combo.addItems(self.model.joint_names)
        self.pose_joint_combo.setCurrentIndex(pose_joint)
        self.pose_joint_combo.blockSignals(False)

    def _populate_animation_selectors(self) -> None:
        current_key = (
            self._active_animation().key
            if self._active_animation() is not None
            else "none"
        )
        self.available_animations = list(
            self.api.available_animation_presets(self.model)
        )
        self.animation_combo.blockSignals(True)
        self.animation_combo.clear()
        self.animation_combo.addItem("None", userData="none")
        for preset in self.available_animations:
            self.animation_combo.addItem(preset.label, userData=preset.key)
        restore_index = 0
        for index in range(self.animation_combo.count()):
            if self.animation_combo.itemData(index) == current_key:
                restore_index = index
                break
        self.animation_combo.setCurrentIndex(restore_index)
        self.animation_combo.blockSignals(False)
        self.animation_speed_slider.set_value(self.animation_speed, emit=False)

    def _refresh_scene_explorer(self) -> None:
        if self.model is None:
            return
        joints = self._display_joints()
        nodes: list[SceneNode] = []
        for joint_index, joint_name in enumerate(self.model.joint_names):
            parent_index = int(self.model.parents[joint_index])
            parent_name = (
                self.model.joint_names[parent_index] if parent_index >= 0 else "<root>"
            )
            nodes.append(
                SceneNode(
                    group="Joints",
                    label=joint_name,
                    kind="joint",
                    payload=joint_index,
                    description=f"Parent: {parent_name}",
                    properties={
                        "index": joint_index,
                        "parent": parent_name,
                        "position": np.asarray(joints[joint_index], dtype=float),
                    },
                )
            )
        if self.loaded_dataset is not None:
            nodes.append(
                SceneNode(
                    group="Dataset",
                    label="Current frame",
                    kind="sample",
                    payload=self.current_sample_index,
                    properties={
                        "frame_index": self.current_sample_index,
                        "range": f"{self.range_start_index}-{self.range_end_index}",
                        "has_2d": self.target_joints_2d is not None,
                        "has_3d": self.target_joints_3d is not None,
                    },
                )
            )
            if self.current_camera is not None:
                nodes.append(
                    SceneNode(
                        group="Dataset",
                        label="Perspective camera",
                        kind="camera",
                        payload="camera",
                        properties={
                            "fx": float(self.current_camera.fx),
                            "fy": float(self.current_camera.fy),
                            "cx": float(self.current_camera.cx),
                            "cy": float(self.current_camera.cy),
                        },
                    )
                )
        nodes.extend(self._ik_scene_nodes())
        self.scene_explorer.set_nodes(nodes)
        current = self.scene_explorer.currentItem()
        node = current.data(0, QtCore.Qt.ItemDataRole.UserRole) if current else None
        self.selected_node = node if isinstance(node, SceneNode) else None
        if self.selected_node is not None:
            if self.selected_node.kind == "joint":
                self._update_joint_inspector(int(self.selected_node.payload))
            else:
                self.inspector.set_content(
                    title=self.selected_node.label,
                    subtitle=self.selected_node.kind,
                    properties=self.selected_node.properties,
                    notes=self.selected_node.description,
                )

    def _on_scene_node_selected(self, node: object) -> None:
        self.selected_node = node if isinstance(node, SceneNode) else None
        if self.selected_node is None:
            self.inspector.clear()
            return
        if self.selected_node.kind == "joint":
            joint_index = int(self.selected_node.payload)
            self.pose_joint_combo.blockSignals(True)
            self.pose_joint_combo.setCurrentIndex(joint_index)
            self.pose_joint_combo.blockSignals(False)
            self._sync_pose_sliders()
            self._sync_rom_controls()
            self._sync_scale_sliders()
            self._update_joint_inspector(joint_index)
            self._refresh_view()
            return
        self.inspector.set_content(
            title=self.selected_node.label,
            subtitle=self.selected_node.kind,
            properties=self.selected_node.properties,
            notes=self.selected_node.description,
        )

    def _update_joint_inspector(self, joint_index: int) -> None:
        joints = self._display_joints()
        joint_name = self.model.joint_names[joint_index]
        parent_index = int(self.model.parents[joint_index])
        parent_name = (
            self.model.joint_names[parent_index] if parent_index >= 0 else "<root>"
        )
        limits = self.rom_limits.get(joint_name, [])
        limit_text = ", ".join(
            f"{axis}: [{limit.minimum_deg:.1f}, {limit.maximum_deg:.1f}]"
            if limit.enabled
            else f"{axis}: free"
            for axis, limit in zip(self.api.axis_names, limits)
        )
        self.inspector.set_content(
            title=joint_name,
            subtitle=f"Joint {joint_index}",
            properties={
                "parent": parent_name,
                "joint_type": self.model.spec.joint_type(joint_index),
                "targets": ", ".join(
                    spec.name
                    for spec in self.target_specs
                    if spec.frame in {joint_name, "joint:" + joint_name}
                )
                or "—",
                "world_position": np.asarray(joints[joint_index], dtype=float),
                "pose_deg": np.degrees(
                    self.full_pose[joint_index].detach().cpu().numpy()
                ),
                "scale_xyz": self.scales[joint_index].detach().cpu().numpy(),
                "global_scale": self.global_bone_scale,
                "rom": limit_text,
            },
        )

    # ------------------------------------------------------------------
    # Pose / animation state
    # ------------------------------------------------------------------
    def _rom_limits_for_joint(self, joint_index: int) -> list[Any]:
        return self.rom_limits[self.model.joint_names[joint_index]]

    def _clamp_angle_to_rom_limit(
        self, joint_index: int, axis: int, value_rad: float
    ) -> float:
        limit = self._rom_limits_for_joint(joint_index)[axis]
        if not getattr(limit, "enabled", False):
            return value_rad
        minimum = math.radians(float(limit.minimum_deg))
        maximum = math.radians(float(limit.maximum_deg))
        return min(max(value_rad, minimum), maximum)

    def _clamp_pose_to_rom_limits(self, pose: torch.Tensor) -> torch.Tensor:
        clamped = pose.clone()
        for joint_index, joint_name in enumerate(self.model.joint_names):
            limits = self.rom_limits.get(joint_name)
            if limits is None:
                continue
            for axis, limit in enumerate(limits):
                if not getattr(limit, "enabled", False):
                    continue
                clamped[joint_index, axis] = self._clamp_angle_to_rom_limit(
                    joint_index, axis, float(clamped[joint_index, axis])
                )
        return clamped

    def _clamp_full_pose_in_place(self) -> None:
        self.full_pose = self._clamp_pose_to_rom_limits(self.full_pose)

    def _make_pose_callback(self, axis: int):
        def callback(value: float) -> None:
            self._stop_ik()
            joint_index = self.pose_joint_combo.currentIndex()
            if joint_index < 0:
                return
            self.full_pose[joint_index, axis] = self._clamp_angle_to_rom_limit(
                joint_index, axis, math.radians(value)
            )
            self._sync_pose_sliders()
            self._refresh_view()

        return callback

    def _make_translation_callback(self, axis: int):
        def callback(value: float) -> None:
            self._stop_ik()
            self.translation[axis] = value
            self._refresh_view()

        return callback

    def _make_scale_callback(self, axis: int):
        def callback(value: float) -> None:
            self._stop_ik()
            joint_index = self._selected_scale_joint()
            self.scales[joint_index, axis] = value
            self._refresh_view()

        return callback

    def _selected_scale_joint(self) -> int:
        selection = self.scale_joint_combo.currentIndex()
        if selection < 0:
            return self.model.root_index
        return self.scale_joint_indices[selection]

    def _active_animation(self) -> Any | None:
        key = self.animation_combo.currentData()
        if key in (None, "none"):
            return None
        for preset in self.available_animations:
            if preset.key == key:
                return preset
        return None

    def _sync_pose_sliders(self, *_args) -> None:
        joint_index = self.pose_joint_combo.currentIndex()
        if joint_index < 0:
            return
        self._update_pose_slider_ranges()
        values_deg = [math.degrees(float(v)) for v in self.full_pose[joint_index]]
        for axis, slider in enumerate(self.pose_sliders):
            slider.set_value(values_deg[axis], emit=False)
        self._update_joint_inspector(joint_index)

    def _sync_translation_sliders(self) -> None:
        for axis, slider in enumerate(self.translation_sliders):
            slider.set_value(float(self.translation[axis]), emit=False)

    def _sync_scale_sliders(self, *_args) -> None:
        selection = self.scale_joint_combo.currentIndex()
        if selection < 0 or not self.scale_joint_indices:
            return
        joint_index = self.scale_joint_indices[selection]
        for axis, slider in enumerate(self.scale_sliders):
            slider.set_value(float(self.scales[joint_index, axis]), emit=False)

    def _sync_global_scale_slider(self) -> None:
        self.global_scale_slider.set_value(self.global_bone_scale, emit=False)

    def _sync_animation_phase_slider(self) -> None:
        preset = self._active_animation()
        phase = 0.0
        if preset is not None and float(preset.period) > 0.0:
            phase = (self.animation_time / float(preset.period)) % 1.0
        self.animation_phase_slider.set_value(phase, emit=False)

    def _update_pose_slider_ranges(self) -> None:
        joint_index = self.pose_joint_combo.currentIndex()
        if joint_index < 0:
            return
        self._clamp_full_pose_in_place()
        limits = self._rom_limits_for_joint(joint_index)
        for axis, slider in enumerate(self.pose_sliders):
            limit = limits[axis]
            if getattr(limit, "enabled", False):
                slider.set_range(float(limit.minimum_deg), float(limit.maximum_deg))
            else:
                slider.set_range(-180.0, 180.0)

    def _sync_rom_controls(self, *_args) -> None:
        joint_index = self.pose_joint_combo.currentIndex()
        if joint_index < 0:
            return
        self._update_pose_slider_ranges()
        limits = self._rom_limits_for_joint(joint_index)
        for axis, limit in enumerate(limits):
            widgets = (
                self.rom_enable_checks[axis],
                self.rom_min_spins[axis],
                self.rom_max_spins[axis],
            )
            for widget in widgets:
                blocked = widget.blockSignals(True)
                if isinstance(widget, QtWidgets.QCheckBox):
                    widget.setChecked(bool(limit.enabled))
                elif widget is self.rom_min_spins[axis]:
                    widget.setValue(float(limit.minimum_deg))
                else:
                    widget.setValue(float(limit.maximum_deg))
                widget.blockSignals(blocked)
            self.rom_min_spins[axis].setEnabled(bool(limit.enabled))
            self.rom_max_spins[axis].setEnabled(bool(limit.enabled))

    def _update_animation_controls(self) -> None:
        self._update_playback_controls()

    def _on_animation_changed(self, *_args) -> None:
        self._stop_ik()
        self.sequence_timer.stop()
        self._preset_preview = True
        blocker = QtCore.QSignalBlocker(self.playback_source_combo)
        self.playback_source_combo.setCurrentIndex(0)
        del blocker
        self.animation_time = 0.0
        if self._active_animation() is None:
            self.animation_timer.stop()
        self._sync_animation_phase_slider()
        self._update_animation_controls()
        self._refresh_view()

    def _toggle_animation(self, *_args) -> None:
        if self._active_animation() is None:
            return
        if self.animation_timer.isActive():
            self.animation_timer.stop()
        else:
            if self.sequence_timer.isActive():
                self.sequence_timer.stop()
            self.animation_timer.start()
        self._update_animation_controls()

    def _restart_animation(self, *_args) -> None:
        self.animation_time = 0.0
        self._sync_animation_phase_slider()
        self._refresh_view()

    def _on_animation_speed_changed(self, value: float) -> None:
        self.animation_speed = value

        self.sequence_timer.setInterval(max(1, int(round(120 / value))))

    def _on_animation_phase_changed(self, value: float) -> None:
        preset = self._active_animation()
        if preset is None:
            return
        self.animation_time = float(value) * float(preset.period)
        self._refresh_view()

    def _advance_animation(self) -> None:
        preset = self._active_animation()
        if preset is None:
            self.animation_timer.stop()
            self._update_animation_controls()
            return
        self.animation_time = (
            self.animation_time + self.ANIMATION_DT * self.animation_speed
        ) % float(preset.period)
        self._sync_animation_phase_slider()
        self._refresh_view()

    def _on_global_scale_changed(self, value: float) -> None:
        self._stop_ik()
        self.global_bone_scale = value
        self._refresh_view()

    def _zero_selected_joint(self, *_args) -> None:
        self._stop_ik()
        joint_index = self.pose_joint_combo.currentIndex()
        if joint_index < 0:
            return
        self.full_pose[joint_index].zero_()
        self._clamp_full_pose_in_place()
        self._sync_pose_sliders()
        self._refresh_view()

    def _reset_selected_bone(self, *_args) -> None:
        self._stop_ik()
        joint_index = self._selected_scale_joint()
        self.scales[joint_index].fill_(1.0)
        self._sync_scale_sliders()
        self._refresh_view()

    def _reset_all(self, *_args) -> None:
        self._stop_ik()
        self.full_pose.zero_()
        self.translation.zero_()
        self.scales.fill_(1.0)
        self.global_bone_scale = 1.0
        self.animation_time = 0.0
        self.sequence_timer.stop()
        self.preview_override_joints = None
        self.sequence_fit_payload = None
        self._sync_pose_sliders()
        self._sync_translation_sliders()
        self._sync_scale_sliders()
        self._sync_global_scale_slider()
        self._sync_animation_phase_slider()
        self._refresh_view(fit_camera=True)
        self._log("Reset skeleton state.")

    # ------------------------------------------------------------------
    # ROM / state I/O
    # ------------------------------------------------------------------
    def _serialize_rom_limits(self) -> dict[str, object]:
        return self.api.serialize_rom_limits(
            self.rom_limits, skeleton=self.model.spec.name
        )

    def _apply_rom_payload(self, payload: dict[str, object]) -> None:
        self.rom_limits = self.api.apply_rom_payload(
            payload,
            skeleton=self.model.spec.name,
            joint_names=self.model.joint_names,
        )
        self._clamp_full_pose_in_place()
        self._sync_rom_controls()
        self._sync_pose_sliders()
        self._refresh_view()

    def _default_rom_path(self) -> Path:
        return Path.cwd() / "rom_limits" / f"{self.model.spec.name}.json"

    def _on_rom_limit_changed(self, *_args) -> None:
        joint_index = self.pose_joint_combo.currentIndex()
        if joint_index < 0:
            return
        joint_name = self.model.joint_names[joint_index]
        for axis in range(3):
            enabled = self.rom_enable_checks[axis].isChecked()
            minimum_deg = float(self.rom_min_spins[axis].value())
            maximum_deg = float(self.rom_max_spins[axis].value())
            if minimum_deg > maximum_deg:
                if self.sender() is self.rom_min_spins[axis]:
                    maximum_deg = minimum_deg
                else:
                    minimum_deg = maximum_deg
            self.rom_limits[joint_name][axis] = self.api.axis_rom_limit_cls(
                enabled=enabled,
                minimum_deg=minimum_deg,
                maximum_deg=maximum_deg,
            )
            self.rom_min_spins[axis].setEnabled(enabled)
            self.rom_max_spins[axis].setEnabled(enabled)
        self._clamp_full_pose_in_place()
        self._sync_rom_controls()
        self._sync_pose_sliders()
        self._refresh_view()

    def _reset_selected_joint_rom(self, *_args) -> None:
        joint_index = self.pose_joint_combo.currentIndex()
        if joint_index < 0:
            return
        self.rom_limits[self.model.joint_names[joint_index]] = [
            self.api.axis_rom_limit_cls() for _ in range(3)
        ]
        self._sync_rom_controls()
        self._sync_pose_sliders()
        self._refresh_view()
        self._log(f"Cleared ROM limits for {self.model.joint_names[joint_index]}.")

    def _clear_all_rom(self, *_args) -> None:
        self.rom_limits = self.api.default_rom_limits(self.model.joint_names)
        self._sync_rom_controls()
        self._sync_pose_sliders()
        self._refresh_view()
        self._log("Cleared all ROM limits.")

    def _save_rom_limits_to_file(self, *_args) -> None:
        default_path = self._default_rom_path()
        default_path.parent.mkdir(parents=True, exist_ok=True)
        file_path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save ROM Limits", str(default_path), "JSON Files (*.json)"
        )
        if not file_path:
            return
        path = Path(file_path)
        if path.suffix.lower() != ".json":
            path = path.with_suffix(".json")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self._serialize_rom_limits(), indent=2), encoding="utf-8"
        )
        self._log(f"Saved ROM limits to {path.name}.")

    def _load_rom_limits_from_file(self, *_args) -> None:
        default_path = self._default_rom_path()
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Load ROM Limits", str(default_path), "JSON Files (*.json)"
        )
        if not file_path:
            return
        try:
            payload = json.loads(Path(file_path).read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                raise ValueError("ROM file must contain a JSON object.")
            self._apply_rom_payload(payload)
            self._log(f"Loaded ROM limits from {Path(file_path).name}.")
        except Exception as exc:  # pragma: no cover - GUI path
            QtWidgets.QMessageBox.warning(self, "Load ROM Limits", str(exc))

    def _save_pose_state(self, *_args) -> None:
        if self.model is None:
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Export Playground State",
            str(Path.cwd() / f"{self.model.spec.name}_state.json"),
            "JSON Files (*.json)",
        )
        if not path:
            return
        payload = {
            "skeleton": self.model.spec.name,
            "full_pose": self.full_pose.detach().cpu().tolist(),
            "translation": self.translation.detach().cpu().tolist(),
            "scales": self.scales.detach().cpu().tolist(),
            "global_bone_scale": self.global_bone_scale,
            "rom_limits": self._serialize_rom_limits(),
        }
        Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")
        self._log(f"Exported state to {Path(path).name}.")

    def _load_pose_state(self, *_args) -> None:
        self._stop_ik()
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Load Playground State", str(Path.cwd()), "JSON Files (*.json)"
        )
        if not path:
            return
        try:
            payload = json.loads(Path(path).read_text(encoding="utf-8"))
            if payload.get("skeleton") and payload["skeleton"] != self.model.spec.name:
                self._load_skeleton_by_name(str(payload["skeleton"]))
            self.full_pose = torch.tensor(
                payload["full_pose"], dtype=self.model.rest_offsets.dtype
            )
            self.translation = torch.tensor(
                payload["translation"], dtype=self.model.rest_offsets.dtype
            )
            self.scales = torch.tensor(
                payload["scales"], dtype=self.model.rest_offsets.dtype
            )
            self.global_bone_scale = float(payload.get("global_bone_scale", 1.0))
            if isinstance(payload.get("rom_limits"), dict):
                self._apply_rom_payload(payload["rom_limits"])
            self._sync_pose_sliders()
            self._sync_translation_sliders()
            self._sync_scale_sliders()
            self._sync_global_scale_slider()
            self._refresh_view()
            self._log(f"Loaded playground state from {Path(path).name}.")
        except Exception as exc:  # pragma: no cover - GUI path
            QtWidgets.QMessageBox.warning(self, "Load Playground State", str(exc))

    # ------------------------------------------------------------------
    # Dataset / fitting workflow
    # ------------------------------------------------------------------
    def _clear_dataset_controls(self) -> None:
        self._stop_ik()
        self.sequence_timer.stop()
        self.loaded_dataset = None
        self.dataset_path = None
        self.range_start_index = 0
        self.range_end_index = 0
        self.current_sample_index = 0
        self.range_sequence = None
        self.sequence_fit_payload = None
        self.current_sample = None
        self.target_joints_3d = None
        self.target_joints_2d = None
        self.current_camera = None
        self.dataset_path_edit.clear()
        for spin in (self.range_start_spin, self.range_end_spin, self.sample_spin):
            spin.blockSignals(True)
            spin.setRange(0, 0)
            spin.setValue(0)
            spin.setEnabled(False)
            spin.blockSignals(False)
        self.sequence_frame_slider.blockSignals(True)
        self.sequence_frame_slider.setRange(0, 0)
        self.sequence_frame_slider.setValue(0)
        self.sequence_frame_slider.setEnabled(False)
        self.sequence_frame_slider.blockSignals(False)
        self.dataset_info_label.setText("No fitting dataset loaded.")
        self.sequence_frame_label.setText("No active range.")
        self.projection_preview.clear()
        self.preview_override_joints = None
        self.metric_plot.clear_series()
        self.fit_history.clear()
        self._refresh_dataset_ik_targets()
        self._update_fit_controls()

    def _load_dataset_from_file(self, *_args) -> None:
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Load Fitting Dataset", str(Path.cwd()), "NumPy Archives (*.npz)"
        )
        if not file_path:
            return
        try:
            dataset = self.api.frame_dataset_cls.from_npz(
                file_path, expected_num_joints=self.model.NUM_JOINTS
            )
            self._set_loaded_dataset(
                dataset, source_label=str(file_path), path=Path(file_path)
            )
            self._log(f"Loaded fitting dataset {Path(file_path).name}.")
        except Exception as exc:  # pragma: no cover - GUI path
            QtWidgets.QMessageBox.warning(self, "Load Dataset", str(exc))

    def _generate_synthetic_dataset(self, *_args) -> None:
        seed = self.synthetic_seed_spin.value()
        controls = self.api.synthetic_fitting_controls_cls(
            num_frames=self.synthetic_frames_spin.value(),
            pose_std=self.synthetic_pose_std_spin.value(),
            noise_std=self.synthetic_noise_spin.value(),
            confidence_dropout=self.synthetic_dropout_spin.value(),
            seed=None if seed < 0 else seed,
        )
        try:
            generated = self.api.create_synthetic_fitting_dataset(
                self.model, controls, device=self.model.rest_offsets.device
            )
            generated.frame_dataset.metadata["seed"] = controls.seed
            source_label = f"Synthetic {controls.num_frames} frames | pose={controls.pose_std:.3f}, noise={controls.noise_std:.1f}px"
            self._set_loaded_dataset(
                generated.frame_dataset, source_label=source_label, path=None
            )
            self._log(
                f"Generated {len(generated.frame_dataset)} synthetic fitting frames."
            )
        except Exception as exc:  # pragma: no cover - GUI path
            QtWidgets.QMessageBox.warning(self, "Generate Dataset", str(exc))

    def _set_loaded_dataset(
        self, dataset, *, source_label: str, path: Path | None
    ) -> None:
        self._stop_ik()
        self.animation_timer.stop()
        self._preset_preview = False
        blocker = QtCore.QSignalBlocker(self.playback_source_combo)
        self.playback_source_combo.setCurrentIndex(1)
        del blocker
        self.sequence_timer.stop()
        self.loaded_dataset = dataset
        self.dataset_path = path
        self.range_start_index = 0
        self.range_end_index = max(len(dataset) - 1, 0)
        self.current_sample_index = 0
        self.current_sample = None
        self.range_sequence = self._build_range_sequence()
        self.sequence_fit_payload = None
        self.preview_override_joints = None
        self.dataset_path_edit.setText(source_label)
        for spin in (self.range_start_spin, self.range_end_spin):
            spin.blockSignals(True)
            spin.setRange(0, max(len(dataset) - 1, 0))
            spin.setEnabled(len(dataset) > 0)
            spin.blockSignals(False)
        self.range_start_spin.setValue(self.range_start_index)
        self.range_end_spin.setValue(self.range_end_index)
        self.sample_spin.blockSignals(True)
        self.sample_spin.setRange(self.range_start_index, self.range_end_index)
        self.sample_spin.setEnabled(len(dataset) > 0)
        self.sample_spin.setValue(self.current_sample_index)
        self.sample_spin.blockSignals(False)
        self.sequence_frame_slider.blockSignals(True)
        self.sequence_frame_slider.setRange(
            self.range_start_index, self.range_end_index
        )
        self.sequence_frame_slider.setEnabled(len(dataset) > 0)
        self.sequence_frame_slider.setValue(self.current_sample_index)
        self.sequence_frame_slider.blockSignals(False)
        if len(dataset) > 0:
            self._apply_sample_index(0)
        else:
            self.target_joints_3d = None
            self.target_joints_2d = None
            self._refresh_dataset_ik_targets()
            self._update_sequence_frame_controls()
            self._update_fit_controls()
            self._refresh_view()

    def _build_range_sequence(self) -> dict[str, torch.Tensor] | None:
        if self.loaded_dataset is None or len(self.loaded_dataset) == 0:
            return None
        samples = [
            self.loaded_dataset[index]
            for index in range(self.range_start_index, self.range_end_index + 1)
        ]
        if not samples:
            return None
        sequence: dict[str, torch.Tensor] = {}
        for key in samples[0]:
            sequence[key] = torch.stack([sample[key] for sample in samples], dim=0)
        return sequence

    def _sample_camera(self):
        if self.current_sample is None or "joints_2d" not in self.current_sample:
            return None
        return self.api.perspective_camera_cls(
            fx=self.current_sample.get("fx", torch.tensor(1000.0)).reshape(()),
            fy=self.current_sample.get("fy", torch.tensor(1000.0)).reshape(()),
            cx=self.current_sample.get("cx", torch.tensor(512.0)).reshape(()),
            cy=self.current_sample.get("cy", torch.tensor(512.0)).reshape(()),
        )

    def _apply_sample_index(self, index: int) -> None:
        if self.loaded_dataset is None or len(self.loaded_dataset) == 0:
            return
        index = int(np.clip(index, self.range_start_index, self.range_end_index))
        self.current_sample_index = index
        sample = self.loaded_dataset[index]
        self.current_sample = {
            key: value.detach().cpu().clone()
            if isinstance(value, torch.Tensor)
            else value
            for key, value in sample.items()
        }
        self.target_joints_3d = sample["joints_3d"].detach().cpu().numpy()
        self.target_joints_2d = (
            sample["joints_2d"].detach().cpu().numpy()
            if "joints_2d" in sample
            else None
        )
        self.current_camera = self._sample_camera()
        if self.sequence_fit_payload is not None:
            relative_index = self.current_sample_index - self.range_start_index
            self.full_pose = self.sequence_fit_payload["full_pose"][
                relative_index
            ].clone()
            self.translation = self.sequence_fit_payload["transl"][
                relative_index
            ].clone()
            self.scales = self.sequence_fit_payload["scales"][relative_index].clone()
            self._sync_pose_sliders()
            self._sync_translation_sliders()
            self._sync_scale_sliders()
        has_2d = "joints_2d" in sample
        camera_note = (
            "camera defaults"
            if has_2d and self.current_camera is not None
            else "no camera"
        )
        self.dataset_info_label.setText(
            f"Frames: {len(self.loaded_dataset)} | Range: {self.range_start_index}-{self.range_end_index}\n"
            f"Current frame: {index} | 3D joints: yes | 2D joints: {'yes' if has_2d else 'no'} | {camera_note}"
        )
        self._update_sequence_frame_controls()
        self._refresh_dataset_ik_targets()
        self._refresh_view(fit_camera=self._ik_sequence_results is None)
        self._update_fit_controls()
        self._refresh_scene_explorer()

    def _clear_dataset(self, *_args) -> None:
        self._clear_dataset_controls()
        self._refresh_view()
        self._refresh_scene_explorer()
        self._log("Cleared fitting dataset.")

    def _on_range_start_changed(self, value: int) -> None:
        if self.loaded_dataset is None:
            return
        value = int(np.clip(value, 0, max(len(self.loaded_dataset) - 1, 0)))
        if value > self.range_end_index:
            self.range_end_index = value
            self.range_end_spin.blockSignals(True)
            self.range_end_spin.setValue(value)
            self.range_end_spin.blockSignals(False)
        self.range_start_index = value
        self.range_sequence = self._build_range_sequence()
        self.sequence_fit_payload = None
        if self.current_sample_index < self.range_start_index:
            self.current_sample_index = self.range_start_index
        self._apply_sample_index(self.current_sample_index)

    def _on_range_end_changed(self, value: int) -> None:
        if self.loaded_dataset is None:
            return
        value = int(np.clip(value, 0, max(len(self.loaded_dataset) - 1, 0)))
        if value < self.range_start_index:
            self.range_start_index = value
            self.range_start_spin.blockSignals(True)
            self.range_start_spin.setValue(value)
            self.range_start_spin.blockSignals(False)
        self.range_end_index = value
        self.range_sequence = self._build_range_sequence()
        self.sequence_fit_payload = None
        if self.current_sample_index > self.range_end_index:
            self.current_sample_index = self.range_end_index
        self._apply_sample_index(self.current_sample_index)

    def _on_sample_index_changed(self, value: int) -> None:
        self._apply_sample_index(value)

    def _on_sequence_frame_slider_changed(self, value: int) -> None:
        self._apply_sample_index(value)

    def _selected_range_length(self) -> int:
        return max(self.range_end_index - self.range_start_index + 1, 0)

    def _update_sequence_frame_controls(self) -> None:
        dataset_ready = self.loaded_dataset is not None and len(self.loaded_dataset) > 0
        if not dataset_ready:
            self.sequence_frame_label.setText("No active range.")
            return
        self.sample_spin.blockSignals(True)
        self.sample_spin.setRange(self.range_start_index, self.range_end_index)
        self.sample_spin.setValue(self.current_sample_index)
        self.sample_spin.blockSignals(False)
        self.sequence_frame_slider.blockSignals(True)
        self.sequence_frame_slider.setRange(
            self.range_start_index, self.range_end_index
        )
        self.sequence_frame_slider.setValue(self.current_sample_index)
        self.sequence_frame_slider.blockSignals(False)
        frame_offset = self.current_sample_index - self.range_start_index + 1
        range_length = self._selected_range_length()
        fit_state = (
            "fitted sequence"
            if self.sequence_fit_payload is not None
            else "target range"
        )
        self.sequence_frame_label.setText(
            f"Frame {frame_offset}/{range_length} in selected range ({self.current_sample_index} absolute, {fit_state})."
        )

    def _toggle_sequence_playback(self, *_args) -> None:
        if self.loaded_dataset is None or self._selected_range_length() <= 1:
            return
        if self.sequence_timer.isActive():
            self.sequence_timer.stop()
        else:
            if self.animation_timer.isActive():
                self.animation_timer.stop()
                self._update_animation_controls()
            self.sequence_timer.start()
        self._update_fit_controls()

    def _advance_sequence_frame(self) -> None:
        if self.loaded_dataset is None or self._selected_range_length() <= 1:
            self.sequence_timer.stop()
            self._update_fit_controls()
            return
        next_index = self.current_sample_index + 1
        if next_index > self.range_end_index:
            next_index = self.range_start_index
        self._apply_sample_index(next_index)

    def _select_priors_checkpoint(self, *_args) -> None:
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Load Priors Checkpoint",
            str(Path.cwd()),
            "PyTorch Checkpoints (*.pt *.pth *.ckpt);;All Files (*)",
        )
        if not file_path:
            return
        try:
            self.api.load_fitting_prior_checkpoint(
                Path(file_path), skeleton=self.model.spec.name
            )
            self.priors_path = Path(file_path)
            self.priors_path_edit.setText(str(self.priors_path))
            self._log(f"Loaded priors checkpoint {self.priors_path.name}.")
            self._update_fit_controls()
        except Exception as exc:  # pragma: no cover - GUI path
            QtWidgets.QMessageBox.warning(self, "Load Priors", str(exc))

    def _clear_priors_checkpoint(self, *_args) -> None:
        self.priors_path = None
        self.priors_path_edit.clear()
        self._update_fit_controls()

    def _update_fit_controls(self, *_args) -> None:
        dataset_ready = (
            self.loaded_dataset is not None and self.current_sample is not None
        )
        sample_has_2d = dataset_ready and "joints_2d" in self.current_sample
        fitting_running = (
            self.fit_thread is not None or self._ik_sequence_results is not None
        )
        fit_mode = self.fit_mode_combo.currentData()
        if fit_mode == "2d" and not sample_has_2d:
            self.fit_mode_combo.blockSignals(True)
            self.fit_mode_combo.setCurrentIndex(0)
            self.fit_mode_combo.blockSignals(False)
            fit_mode = "3d"
        self.fit_mode_combo.setEnabled(dataset_ready and not fitting_running)
        self.fit_optimize_scale_check.setEnabled(not fitting_running)
        self.fit_pose_prior_check.setEnabled(not fitting_running)
        self.fit_init_from_ik_check.setEnabled(fit_mode == "3d" and not fitting_running)
        self.fit_iters_spin.setEnabled(not fitting_running)
        self.fit_lr_spin.setEnabled(not fitting_running)
        self.run_fit_button.setEnabled(dataset_ready and not fitting_running)
        self.cancel_fit_button.setEnabled(self.fit_thread is not None)
        self.save_fit_button.setEnabled(
            self.sequence_fit_payload is not None and not fitting_running
        )
        self.skeleton_combo.setEnabled(not fitting_running)
        self.import_state_button.setEnabled(not fitting_running)
        self.load_dataset_button.setEnabled(not fitting_running)
        self.clear_dataset_button.setEnabled(dataset_ready and not fitting_running)
        self.load_priors_button.setEnabled(not fitting_running)
        self.clear_priors_button.setEnabled(
            self.priors_path is not None and not fitting_running
        )
        for widget in (
            self.synthetic_frames_spin,
            self.synthetic_pose_std_spin,
            self.synthetic_noise_spin,
            self.synthetic_dropout_spin,
            self.synthetic_seed_spin,
            self.generate_synthetic_button,
        ):
            widget.setEnabled(not fitting_running)
        for widget in (
            self.range_start_spin,
            self.range_end_spin,
            self.sample_spin,
            self.sequence_frame_slider,
        ):
            widget.setEnabled(dataset_ready and not fitting_running)
        self.sequence_play_button.setEnabled(
            dataset_ready and self._selected_range_length() > 1 and not fitting_running
        )
        self.sequence_play_button.setText(
            "Pause Range" if self.sequence_timer.isActive() else "Play Range"
        )

        self._update_ik_controls()
        self._update_playback_controls()

    def _start_fit(self, *_args) -> None:
        if (
            self.current_sample is None
            or self.current_skeleton_name is None
            or self.range_sequence is None
        ):
            return
        if self.fit_thread is not None:
            return
        sequence = {
            key: value.detach().cpu().clone()
            for key, value in self.range_sequence.items()
        }
        fit_mode = self.fit_mode_combo.currentData()
        if fit_mode == "2d" and "joints_2d" not in sequence:
            QtWidgets.QMessageBox.warning(
                self, "Run Fit", "The selected range does not contain joints_2d."
            )
            return
        self._stop_ik()
        self.animation_timer.stop()
        self._preset_preview = False
        self.sequence_timer.stop()
        self.sequence_fit_payload = None
        blocker = QtCore.QSignalBlocker(self.playback_source_combo)
        self.playback_source_combo.setCurrentIndex(1)
        del blocker
        self.preview_override_joints = None
        self.fit_history.clear()
        self.metric_plot.clear_series()
        progress_interval = max(1, min(10, self.fit_iters_spin.value() // 25 or 1))
        self.fit_thread = QtCore.QThread(self)
        self.fit_worker = self.api.fitting_worker_cls(
            skeleton_name=self.current_skeleton_name,
            sequence=sequence,
            priors_path=self.priors_path,
            mode=fit_mode,
            num_iters=self.fit_iters_spin.value(),
            lr=self.fit_lr_spin.value(),
            optimize_scales=self.fit_optimize_scale_check.isChecked(),
            use_pose_prior_latent=self.fit_pose_prior_check.isChecked(),
            init_from_ik=self.fit_init_from_ik_check.isChecked(),
            progress_interval=progress_interval,
        )
        self.fit_worker.moveToThread(self.fit_thread)
        self.fit_thread.started.connect(self.fit_worker.run)
        self.fit_worker.progress.connect(self._on_fit_progress)
        self.fit_worker.finished.connect(self._on_fit_finished)
        self.fit_worker.failed.connect(self._on_fit_failed)
        self.fit_worker.canceled.connect(self._on_fit_canceled)
        self.fit_worker.finished.connect(self.fit_thread.quit)
        self.fit_worker.failed.connect(self.fit_thread.quit)
        self.fit_worker.canceled.connect(self.fit_thread.quit)
        self.fit_worker.finished.connect(self.fit_worker.deleteLater)
        self.fit_worker.failed.connect(self.fit_worker.deleteLater)
        self.fit_worker.canceled.connect(self.fit_worker.deleteLater)
        self.fit_thread.finished.connect(self._cleanup_fit_worker)
        self.fit_thread.finished.connect(self.fit_thread.deleteLater)
        self.fit_progress_bar.setRange(
            0, self.fit_iters_spin.value() * self._selected_range_length()
        )
        self.fit_progress_bar.setValue(0)
        self.fit_status_label.setText(
            f"Running {fit_mode.upper()} fit over frames {self.range_start_index}-{self.range_end_index}..."
        )
        self._update_fit_controls()
        self.fit_thread.start()
        self._log("Started fitting worker.")

    def _cancel_fit(self, *_args) -> None:
        if self.fit_worker is not None:
            self.fit_worker.cancel()
            self.fit_status_label.setText("Cancel requested...")

    def _on_fit_progress(self, payload: object) -> None:
        if not isinstance(payload, dict):
            return
        joints_np = np.asarray(payload["joints"], dtype=float)
        loss_dict = dict(payload.get("losses", {}))
        total_loss = float(sum(loss_dict.values())) if loss_dict else 0.0
        frame_index = int(payload["frame_index"])
        absolute_index = self.range_start_index + frame_index
        self.preview_override_joints = joints_np
        self.fit_progress_bar.setRange(0, int(payload["total_iters"]))
        self.fit_progress_bar.setValue(int(payload["total_iter"]))
        self.current_sample_index = absolute_index
        if self.loaded_dataset is not None and 0 <= absolute_index < len(
            self.loaded_dataset
        ):
            sample = self.loaded_dataset[absolute_index]
            self.current_sample = {
                key: value.detach().cpu().clone()
                if isinstance(value, torch.Tensor)
                else value
                for key, value in sample.items()
            }
            self.target_joints_3d = sample["joints_3d"].detach().cpu().numpy()
            self.target_joints_2d = (
                sample["joints_2d"].detach().cpu().numpy()
                if "joints_2d" in sample
                else None
            )
            self.current_camera = self._sample_camera()
        self.fit_history.append(
            FitHistoryPoint(step=int(payload["total_iter"]), value=total_loss)
        )
        self.metric_plot.add_point("loss", float(payload["total_iter"]), total_loss)
        self._update_sequence_frame_controls()
        self.sample_spin.blockSignals(True)
        self.sample_spin.setValue(absolute_index)
        self.sample_spin.blockSignals(False)
        self.sequence_frame_slider.blockSignals(True)
        self.sequence_frame_slider.setValue(absolute_index)
        self.sequence_frame_slider.blockSignals(False)
        self.fit_status_label.setText(
            f"Frame {frame_index + 1}/{int(payload['frame_count'])}, iter {int(payload['iter'])}/{int(payload['iters_per_frame'])}: {self._format_losses(loss_dict)}"
        )
        self._refresh_view()

    def _on_fit_finished(self, payload: object) -> None:
        if not isinstance(payload, dict):
            self._on_fit_failed("Unexpected fitting payload.")
            return
        self.sequence_fit_payload = {
            "full_pose": payload["full_pose"].clone(),
            "transl": payload["transl"].clone(),
            "scales": payload["scales"].clone(),
            "joints": np.asarray(payload["joints"], dtype=float).copy(),
            "losses": dict(payload["losses"]),
            "iterations": int(payload["iterations"]),
            "mode": str(payload["mode"]),
            "frame_count": int(payload["frame_count"]),
        }
        self.preview_override_joints = None
        self.current_sample_index = self.range_start_index
        self.full_pose = self.sequence_fit_payload["full_pose"][0].clone()
        self.translation = self.sequence_fit_payload["transl"][0].clone()
        self.scales = self.sequence_fit_payload["scales"][0].clone()
        self.global_bone_scale = 1.0
        self._sync_pose_sliders()
        self._sync_translation_sliders()
        self._sync_scale_sliders()
        self._sync_global_scale_slider()
        self._update_sequence_frame_controls()
        self.fit_progress_bar.setRange(0, max(int(payload["iterations"]), 1))
        self.fit_progress_bar.setValue(int(payload["iterations"]))
        operation = "IK solve" if payload["mode"] == "ik" else "fit"
        self.fit_status_label.setText(
            f"Completed {operation} across {int(payload['frame_count'])} frames in {int(payload['iterations'])} steps: {self._format_losses(payload['losses'])}"
        )
        self._apply_sample_index(self.range_start_index)
        self._log(f"Completed {operation} and applied solved sequence.")

    def _save_fit_result_to_file(self, *_args) -> None:
        if self.sequence_fit_payload is None:
            return
        default_path = (
            Path.cwd()
            / f"{self.current_skeleton_name}_fit_{self.range_start_index}_{self.range_end_index}.npz"
        )
        file_path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save Fitted Sequence", str(default_path), "NumPy Archives (*.npz)"
        )
        if not file_path:
            return
        path = Path(file_path)
        if path.suffix.lower() != ".npz":
            path = path.with_suffix(".npz")
        try:
            frame_indices = np.arange(
                self.range_start_index, self.range_end_index + 1, dtype=np.int64
            )
            export_payload = self.api.build_fit_export_payload(
                self.sequence_fit_payload,
                frame_indices=frame_indices,
                range_sequence=self.range_sequence,
                metadata={
                    "skeleton": self.model.spec.name,
                    "source": str(self.dataset_path)
                    if self.dataset_path is not None
                    else self.dataset_path_edit.text(),
                },
            )
            np.savez_compressed(path, **export_payload)
            self._log(f"Saved fitted sequence to {path.name}.")
        except Exception as exc:  # pragma: no cover - GUI path
            QtWidgets.QMessageBox.warning(self, "Save Fit", str(exc))

    def _on_fit_failed(self, message: str) -> None:
        self.preview_override_joints = None
        self.fit_status_label.setText(f"Fit failed: {message}")
        self._refresh_view()
        self._log(f"Fitting failed: {message}")
        QtWidgets.QMessageBox.warning(self, "Fitting Failed", message)

    def _on_fit_canceled(self) -> None:
        self.preview_override_joints = None
        self.fit_status_label.setText("Fit canceled.")
        self._refresh_view()
        self._log("Canceled fitting run.")

    def _cleanup_fit_worker(self) -> None:
        self.fit_thread = None
        self.fit_worker = None
        self._update_fit_controls()

    @staticmethod
    def _format_losses(losses: dict[str, float]) -> str:
        if not losses:
            return "No loss terms yet."
        return ", ".join(f"{name}={value:.4f}" for name, value in losses.items())

    # ------------------------------------------------------------------
    # Rendering helpers
    # ------------------------------------------------------------------
    def _current_pose_state(self) -> tuple[torch.Tensor, torch.Tensor]:
        pose = self.full_pose.clone()
        translation = self.translation.clone()
        preset = (
            self._active_animation()
            if self._preset_preview
            and self.playback_source_combo.currentData() == "preset"
            else None
        )
        if preset is None:
            return self._clamp_pose_to_rom_limits(pose), translation
        pose_delta, translation_delta = preset.generator(self, self.animation_time)
        pose = self._clamp_pose_to_rom_limits(pose + pose_delta)
        translation = translation + translation_delta
        return pose, translation

    def _current_joints(self) -> np.ndarray:
        pose, translation = self._current_pose_state()
        with torch.no_grad():
            output = self.model(
                full_pose=pose,
                scales=self.scales * self.global_bone_scale,
                transl=translation,
            )
        return output.joints.detach().cpu().numpy()

    def _display_joints(self) -> np.ndarray:
        if self.preview_override_joints is not None:
            return self.preview_override_joints
        return self._current_joints()

    def _project_joints(self, joints: np.ndarray | None) -> np.ndarray | None:
        if joints is None or self.current_camera is None:
            return None
        with torch.no_grad():
            projected = self.current_camera.project(
                torch.from_numpy(np.asarray(joints, dtype=np.float32)).unsqueeze(0)
            )
        return projected.squeeze(0).detach().cpu().numpy()

    def _update_projection_preview(self, joints: np.ndarray | None) -> None:
        self.projection_preview.set_target_points(self.target_joints_2d)
        self.projection_preview.set_predicted_points(self._project_joints(joints))

    def _fit_camera(self, *_args) -> None:
        joints = self._display_joints()
        if self.target_joints_3d is not None:
            joints = np.concatenate([joints, self.target_joints_3d], axis=0)
        if (
            self._ik_tab_active()
            and self.target_specs
            and self.show_targets_check.isChecked()
        ):
            joints = np.concatenate([joints, self._ik_target_positions()], axis=0)
        self.viewport.fit_camera_to_points(joints)

    def _refresh_view(self, *_args, fit_camera: bool = False) -> None:
        if self.model is None:
            return
        joints = self._display_joints()
        selected_joint = self.pose_joint_combo.currentIndex()
        self.viewport.update_skeleton(
            joints, selected_joint=selected_joint if selected_joint >= 0 else None
        )
        if not self.show_rig_check.isChecked():
            self.viewport.primary_layer.hide()
        self.viewport.set_target_overlay(
            self.target_joints_3d if self.show_dataset_check.isChecked() else None
        )
        self._refresh_ik_layers()
        self._update_projection_preview(joints)
        if fit_camera:
            self._fit_camera()
        self._refresh_scene_explorer()
