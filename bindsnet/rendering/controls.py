# language=rst
"""
The control surface for :class:`~bindsnet.rendering.app.Application`: a separate,
GL-free Qt window. A video-player-style transport row (Reset / Play-Pause / Step+1 /
Step+5 / Run-N) drives the Application's run-state machine (``running`` /
``step_budget`` / ``reset``), and a collapsible, bordered Parameters panel to the side
holds text inputs for (eventually) rebuilding the network model.

The plots ALWAYS render on vispy's GLFW backend, which renders straight to the window
and steps the sim from a tight poll-loop -- that's what keeps the simulation fast.
vispy's Qt backend instead renders through an offscreen ``QOpenGLWidget`` that Qt
composites every frame and dispatches each sim step via a ``QTimer`` through the Qt
event loop -- both much slower. So the controls deliberately live in their own widget
window with no GL surface: it's pumped at ~60 Hz from a vispy timer while GLFW drives
the plots at full speed, and the cheap ``QLabel.setText`` time readout stays out of the
render path (an in-canvas readout would re-lay-out glyph geometry every sim step).
"""


class QtControlPanel:
  # language=rst
  """Real Qt buttons + an N field + a collapsible parameters panel in their own
  GL-free window. PySide6 is imported lazily so importing the renderer doesn't
  require Qt until it's used.

  :param application: the owning :class:`Application`.
  :param parameters: optional ``{label: value}`` map rendered as editable text rows
      in the Parameters panel. Purely cosmetic for now -- the fields are locked once
      the sim advances and unlocked again on reset, but applying them is future work.
  """

  needs_pump = True   # a vispy timer must call pump() so Qt processes input events

  def __init__(self, application, parameters: dict | None = None):
    from PySide6 import QtWidgets, QtGui

    self.app = application
    self.parameters = parameters or {}
    # With a model builder, the panel grows an "Apply & Reload" button. Field type
    # inference keys off the original default values.
    self.can_reload = bool(getattr(application, "can_reload", False))
    self._param_fields = {}   # kwarg name -> QLineEdit
    self.status_label = None  # build/validation feedback (only when can_reload)
    # Reuse an existing QApplication, else create one. No GL surface -> off the fast path.
    self.qapp = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    # Closing the control window stops the run too.
    class _ControlWindow(QtWidgets.QWidget):
      def __init__(self, on_close):
        super().__init__()
        self._on_close = on_close
      def closeEvent(self, event):
        self._on_close()
        event.accept()

    self.win = _ControlWindow(lambda: application.canvas.close())
    self.win.setWindowTitle("BindsNET Controls")

    # Top-level: transport controls on the left, collapsible parameters on the right.
    root = QtWidgets.QHBoxLayout(self.win)
    root.setContentsMargins(8, 6, 8, 6)
    root.setSpacing(10)

    # Transport controls left, collapsible parameters right.
    root.addLayout(self._build_transport(QtWidgets, QtGui))
    root.addWidget(self._build_parameters(QtWidgets))

    self.win.adjustSize()
    self.win.show()

  #### Transport row (video-player style) ####
  def _icon(self, QtWidgets, std_pixmap):
    # Themed icon from the active Qt style (no asset files needed).
    return self.win.style().standardIcon(std_pixmap)

  def _icon_button(self, QtWidgets, std_pixmap, tooltip, slot):
    from PySide6 import QtCore
    btn = QtWidgets.QPushButton()
    btn.setIcon(self._icon(QtWidgets, std_pixmap))
    btn.setIconSize(QtCore.QSize(22, 22))
    btn.setFixedSize(40, 36)
    btn.setToolTip(tooltip)
    btn.clicked.connect(slot)
    return btn

  def _build_transport(self, QtWidgets, QtGui):
    SP = QtWidgets.QStyle.StandardPixmap
    box = QtWidgets.QVBoxLayout()
    box.setSpacing(6)

    row = QtWidgets.QHBoxLayout()
    row.setSpacing(4)

    # Reset: clear state + history, rewind to t=0.
    self.reset_btn = self._icon_button(
      QtWidgets, SP.SP_BrowserReload, "Reset simulation", self._on_reset_clicked)
    # Play / Pause: toggle continuous run; icon swaps in set_playing().
    self.play_btn = self._icon_button(
      QtWidgets, SP.SP_MediaPlay, "Play / Pause", self._on_play_clicked)
    # Step +1 / +5: queue discrete steps, consumed even while paused.
    self.step1_btn = self._icon_button(
      QtWidgets, SP.SP_MediaSeekForward, "Step 1", lambda: self._step_n(1))
    self.step5_btn = self._icon_button(
      QtWidgets, SP.SP_MediaSkipForward, "Step 5", lambda: self._step_n(5))

    for b in (self.reset_btn, self.play_btn, self.step1_btn, self.step5_btn):
      row.addWidget(b)

    # Separator, then the Run-N field + button.
    sep = QtWidgets.QFrame()
    sep.setFrameShape(QtWidgets.QFrame.Shape.VLine)
    sep.setFrameShadow(QtWidgets.QFrame.Shadow.Sunken)
    row.addWidget(sep)

    row.addWidget(QtWidgets.QLabel("N:"))
    self.n_input = QtWidgets.QLineEdit("100")
    self.n_input.setValidator(QtGui.QIntValidator(1, 1_000_000_000, self.n_input))
    self.n_input.setMaximumWidth(80)
    self.n_input.returnPressed.connect(self._run_n)   # Enter in the field = Run N
    row.addWidget(self.n_input)

    self.run_btn = QtWidgets.QPushButton("Run N")
    self.run_btn.clicked.connect(self._run_n)
    row.addWidget(self.run_btn)

    row.addStretch(1)
    box.addLayout(row)

    # Speed row: editable steps/sec cap ("max"/"inf" = unlimited) + live rate readout.
    speed_row = QtWidgets.QHBoxLayout()
    speed_row.setSpacing(4)
    speed_row.addWidget(QtWidgets.QLabel("Max steps/s:"))
    self.rate_input = QtWidgets.QLineEdit(self._format_rate(self.app.max_steps_per_second))
    self.rate_input.setMaximumWidth(80)
    self.rate_input.setToolTip("Steps per second cap. Use 'max' (or 'inf') for unlimited.")
    self.rate_input.returnPressed.connect(self._apply_rate)
    self.rate_input.editingFinished.connect(self._apply_rate)
    speed_row.addWidget(self.rate_input)
    speed_row.addStretch(1)
    box.addLayout(speed_row)

    self.time_label = QtWidgets.QLabel("t = 0")
    box.addWidget(self.time_label)
    self.rate_label = QtWidgets.QLabel("running at 0 steps/s")
    box.addWidget(self.rate_label)
    box.addStretch(1)
    return box

  @staticmethod
  def _format_rate(value):
    return "max" if value == float("inf") else f"{int(value)}"

  def _apply_rate(self):
    # Push to the Application; on a bad value, restore the field to the app's value.
    try:
      self.app.set_max_steps_per_second(self.rate_input.text())
    except ValueError:
      pass
    self.rate_input.setText(self._format_rate(self.app.max_steps_per_second))

  def set_steps_per_second(self, sps):
    self.rate_label.setText(f"running at {sps:.0f} steps/s")

  #### Collapsible parameters panel ####
  def _build_parameters(self, QtWidgets):
    from PySide6 import QtCore
    # Toggle button above a bordered group box of editable rows; toggling hides the box
    # and shrinks the window.
    container = QtWidgets.QWidget()
    col = QtWidgets.QVBoxLayout(container)
    col.setContentsMargins(0, 0, 0, 0)
    col.setSpacing(4)

    self.params_toggle = QtWidgets.QToolButton()
    self.params_toggle.setText("Parameters")
    self.params_toggle.setCheckable(True)
    self.params_toggle.setChecked(True)
    self.params_toggle.setToolButtonStyle(
      QtCore.Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
    self.params_toggle.setArrowType(QtCore.Qt.ArrowType.DownArrow)
    self.params_toggle.toggled.connect(self._toggle_params)
    col.addWidget(self.params_toggle, alignment=QtCore.Qt.AlignmentFlag.AlignLeft)

    self.params_box = QtWidgets.QGroupBox("Model parameters")
    self.params_box.setToolTip(
      "Edit, then click 'Apply & Reload' to rebuild the model."
      if self.can_reload else
      "Editable only before the simulation starts (or after Reset).")
    form = QtWidgets.QFormLayout(self.params_box)
    form.setContentsMargins(10, 10, 10, 10)
    form.setVerticalSpacing(6)

    # Supplied params (or illustrative rows). Key = builder kwarg; label = prettified key.
    items = self.parameters.items() if self.parameters else [
      ("excitatory_size", ""), ("inhibitory_size", "")]
    for name, value in items:
      field = QtWidgets.QLineEdit(str(value))
      field.setMaximumWidth(120)
      self._param_fields[name] = field
      form.addRow(QtWidgets.QLabel(f"{self._pretty(name)}:"), field)

    col.addWidget(self.params_box)

    # "Apply & Reload": read the fields, rebuild the network in place (builder only).
    if self.can_reload:
      self.reload_btn = QtWidgets.QPushButton("Apply & Reload")
      self.reload_btn.setToolTip(
        "Rebuild the model from these parameters and reset the simulation to t=0.")
      self.reload_btn.clicked.connect(self._on_reload_clicked)
      col.addWidget(self.reload_btn)
      self.status_label = QtWidgets.QLabel("")
      self.status_label.setWordWrap(True)
      col.addWidget(self.status_label)

    col.addStretch(1)
    return container

  @staticmethod
  def _pretty(name: str) -> str:
    # kwarg name -> display label, e.g. "in_size" -> "In size".
    return str(name).replace("_", " ").strip().capitalize()

  def _toggle_params(self, shown):
    from PySide6 import QtCore
    self.params_box.setVisible(shown)
    self.params_toggle.setArrowType(
      QtCore.Qt.ArrowType.DownArrow if shown else QtCore.Qt.ArrowType.RightArrow)
    self.win.adjustSize()   # reclaim the hidden box's space

  def set_params_locked(self, locked):
    # Cosmetic fields (no builder) lock once the sim advances, unlock on reset. With a
    # builder they stay editable -- skip locking.
    if self.can_reload:
      return
    for field in self._param_fields.values():
      field.setReadOnly(locked)
      field.setEnabled(not locked)

  #### Live model reload ####
  @staticmethod
  def _coerce_like(default, text):
    # Coerce field text to the default's type (int sizes, float connectivities, not
    # strings). Raises ValueError on bad input (caught by Application.reload_model).
    text = str(text).strip()
    if isinstance(default, bool):
      return text.lower() in ("1", "true", "yes", "on")
    if isinstance(default, int):    # bool already handled above
      return int(round(float(text)))
    if isinstance(default, float):
      return float(text)
    return text                     # str / unknown: pass through verbatim

  def get_parameter_values(self) -> dict:
    # {kwarg: value}, each coerced to its default's type, ready to splat into the builder.
    return {name: self._coerce_like(self.parameters.get(name), field.text())
            for name, field in self._param_fields.items()}

  def set_parameter_values(self, values: dict):
    # Write coerced values back into the fields after a reload.
    for name, value in values.items():
      if name in self._param_fields:
        self._param_fields[name].setText(str(value))

  def show_status(self, message: str, error: bool = False):
    if self.status_label is not None:
      self.status_label.setStyleSheet("color: #d33;" if error else "color: #3a3;")
      self.status_label.setText(message)

  def _on_reload_clicked(self):
    self.app.reload_model()

  #### Button handlers ####
  def _on_play_clicked(self):
    self.set_params_locked(True)
    self.app.toggle_play()

  def _step_n(self, n):
    self.set_params_locked(True)
    self.app.run_n(n)

  def _run_n(self):
    text = self.n_input.text()
    if text:
      self.set_params_locked(True)
      self.app.run_n(int(text))

  def _on_reset_clicked(self):
    self.app.reset()

  #### Callbacks from the Application ####
  def pump(self):
    self.qapp.processEvents()

  def set_time(self, t, runtime):
    self.time_label.setText(f"t = {t} / {runtime}")

  def set_playing(self, playing):
    from PySide6 import QtWidgets
    SP = QtWidgets.QStyle.StandardPixmap
    self.play_btn.setIcon(self._icon(
      QtWidgets, SP.SP_MediaPause if playing else SP.SP_MediaPlay))

  def on_reset(self):
    # Re-arm the controls: idle (Play icon), buttons live, time zeroed, params editable.
    self.set_playing(False)
    for b in (self.reset_btn, self.play_btn, self.step1_btn, self.step5_btn,
              self.run_btn):
      b.setEnabled(True)
    self.set_params_locked(False)
    self.set_time(0, self.app.runtime)

  def on_finish(self):
    self.set_playing(False)
    # Reset stays enabled so a finished run can be rewound and replayed.
    for b in (self.play_btn, self.step1_btn, self.step5_btn, self.run_btn):
      b.setEnabled(False)

  def shutdown(self):
    self.win.close()
