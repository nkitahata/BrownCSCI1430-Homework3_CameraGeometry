"""
Shared Dear PyGui UI utilities for CSCI 1430 computer vision demos.
Contains callback factories, UI component builders, and viewport setup.
"""

import contextlib
import os
import dearpygui.dearpygui as dpg


# =============================================================================
# Font Loading
# =============================================================================

# Resolved font handles — set by load_fonts(), used by bind_mono_font()
_default_font = None
_mono_font = None

# DejaVu Sans / Mono are bundled in demos/fonts/ (Bitstream Vera license,
# freely redistributable — see demos/fonts/LICENSE_DEJAVU).
_DEMOS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_FONTS_DIR = os.path.join(_DEMOS_DIR, "fonts")

_SANS_CANDIDATES = [
    os.path.join(_FONTS_DIR, "DejaVuSans.ttf"),
    # Fallbacks: venv copy, then system fonts
    os.path.join(_DEMOS_DIR, "..", ".venv", "Lib", "site-packages",
                 "matplotlib", "mpl-data", "fonts", "ttf", "DejaVuSans.ttf"),
    "C:/Windows/Fonts/segoeui.ttf",
    "C:/Windows/Fonts/arial.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
]

_MONO_CANDIDATES = [
    os.path.join(_FONTS_DIR, "DejaVuSansMono.ttf"),
    os.path.join(_DEMOS_DIR, "..", ".venv", "Lib", "site-packages",
                 "matplotlib", "mpl-data", "fonts", "ttf", "DejaVuSansMono.ttf"),
    "C:/Windows/Fonts/consola.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
    "/System/Library/Fonts/Menlo.ttc",
]


# Font is rasterized at the maximum UI scale (3×) so that
# set_global_font_scale always DOWN-scales (or is 1:1).  Downscaling a
# high-res atlas is far crisper than upscaling a low-res one.
_BASE_FONT_SIZE = 14
_MAX_UI_SCALE = 3.0
_ATLAS_FONT_SIZE = int(_BASE_FONT_SIZE * _MAX_UI_SCALE + 0.5)   # 42 px


def load_fonts(size=None):
    """Load proportional + monospace fonts with Unicode glyph support.

    The font is rasterized at a large size (_ATLAS_FONT_SIZE = 42 px) so that
    set_global_font_scale always scales *down* — giving much crisper text
    than the old approach of upscaling a 14 px raster.

    Must be called after dpg.create_context() and before building any windows.
    Binds the proportional font as the global default.

    Args:
        size: Ignored (kept for call-site compatibility).

    Returns:
        (default_font, mono_font) — DPG font handles.
        Either may be None if no suitable font was found.
    """
    global _default_font, _mono_font

    for fp in _SANS_CANDIDATES:
        if not os.path.exists(fp):
            continue
        with dpg.font_registry():
            with dpg.font(fp, _ATLAS_FONT_SIZE) as _default_font:
                dpg.add_font_range_hint(dpg.mvFontRangeHint_Default)
                dpg.add_font_range(0x1D00, 0x1D7F)   # Phonetic Extensions (ᵀ)
                dpg.add_font_range(0x2000, 0x206F)   # General Punctuation (—)
                dpg.add_font_range(0x00D7, 0x00D7)   # × multiplication sign
                dpg.add_font_range(0x0370, 0x03FF)   # Greek (σ λ)
                dpg.add_font_range(0x2070, 0x209F)   # Super/Subscripts (₀₁₂ₙ₋)
                dpg.add_font_range(0x2190, 0x21FF)   # Arrows (↑ ↓ ← →)
                dpg.add_font_range(0x2200, 0x22FF)   # Math Operators (√)
                dpg.add_font_range(0x2500, 0x257F)   # Box Drawing (─ │ ┌ etc.)
            for mfp in _MONO_CANDIDATES:
                if os.path.exists(mfp):
                    with dpg.font(mfp, _ATLAS_FONT_SIZE) as _mono_font:
                        dpg.add_font_range_hint(dpg.mvFontRangeHint_Default)
                        dpg.add_font_range(0x1D00, 0x1D7F)   # Phonetic Extensions (ᵀ)
                        dpg.add_font_range(0x2000, 0x206F)   # General Punctuation (—)
                        dpg.add_font_range(0x00D7, 0x00D7)   # × multiplication sign
                        dpg.add_font_range(0x0370, 0x03FF)   # Greek (σ λ)
                        dpg.add_font_range(0x2070, 0x209F)   # Super/Subscripts (₀₁₂ₙ₋)
                        dpg.add_font_range(0x2190, 0x21FF)   # Arrows (→ ↑ ↓ ←)
                        dpg.add_font_range(0x2200, 0x22FF)   # Math Operators (√)
                        dpg.add_font_range(0x2500, 0x257F)   # Box Drawing (─ │)
                    break
        dpg.bind_font(_default_font)
        print(f"[fonts] Loaded at {_ATLAS_FONT_SIZE}px (atlas): "
              f"{os.path.basename(fp)}"
              + (f"  mono: {os.path.basename(mfp)}" if _mono_font else ""))
        break
    else:
        print("[fonts] No Unicode font found; using DPG default")

    return _default_font, _mono_font


def _ui_scale_to_gfs(ui_scale):
    """Convert a user-facing UI scale value to a global-font-scale value.

    Because the atlas is rasterized at _MAX_UI_SCALE (3×), the GFS is always
    ≤ 1.0, meaning we only ever *downscale* — which preserves glyph quality.
    """
    return float(ui_scale) / _MAX_UI_SCALE


def bind_mono_font(*tags):
    """Bind the monospace font to the given DPG item tags.

    No-op if load_fonts() hasn't been called or no mono font was found.

    Args:
        *tags: One or more DPG item tags (strings) to bind
    """
    if _mono_font is None:
        return
    for tag in tags:
        dpg.bind_item_font(tag, _mono_font)


# =============================================================================
# Callback Factories
# =============================================================================

UI_SCALES = ["1.0", "1.25", "1.5", "1.75", "2.0", "2.5", "3.0"]


def make_ui_scale_callback():
    """Create UI scale callback (for combo dropdown).

    Returns:
        Callback function that updates global font scale
    """
    def callback(sender, value):
        dpg.set_global_font_scale(_ui_scale_to_gfs(value))
    return callback


def make_state_updater(state, attr):
    """Factory for simple state update callbacks.

    Args:
        state: State object to update
        attr: Attribute name to update on state

    Returns:
        Callback function that sets state.attr = value
    """
    def callback(sender, value):
        setattr(state, attr, value)
    return callback


def make_reset_callback(state, attr, slider_tag, default_value):
    """Factory for reset button callbacks.

    Args:
        state: State object to update
        attr: Attribute name to reset
        slider_tag: Tag of slider widget to update
        default_value: Value to reset to

    Returns:
        Callback function that resets state and slider
    """
    def callback():
        setattr(state, attr, default_value)
        dpg.set_value(slider_tag, default_value)
    return callback


def make_reset_all_callback(defaults, state, extra_reset=None):
    """Create a callback that resets all state attrs to defaults and syncs DPG widgets.

    For each key in defaults:
      1. setattr(state, key, value)
      2. If a DPG item with tag "{key}_slider", "{key}_checkbox", or "{key}_combo"
         exists, set its value

    Args:
        defaults: Dict of {attr_name: default_value}
        state: State object
        extra_reset: Optional callable for demo-specific reset logic
                     (e.g. resetting non-DEFAULTS state, custom widget tags)
    """
    def callback():
        for attr, val in defaults.items():
            if hasattr(state, attr):
                setattr(state, attr, val)
            for suffix in ("_slider", "_checkbox", "_combo"):
                tag = f"{attr}{suffix}"
                if dpg.does_item_exist(tag):
                    dpg.set_value(tag, val)
                    break
        if extra_reset:
            extra_reset()
    return callback


# =============================================================================
# UI Component Builders
# =============================================================================

_panel_themes = {}  # cache: RGB tuple → DPG theme id

# Poll-based collapse registry: {outer_id: (hdr_id, expanded_h, prev_open)}
_panel_registry = {}
_COLLAPSED_H = 48   # header bar + outer window padding (px at default font scale)
_PANEL_PAD = 20      # extra px added to caller's height for header/padding overhead


def poll_collapsible_panels():
    """Poll collapsible panel headers and resize their containers.

    Call once per frame in the render loop.  DPG/ImGui does not re-layout
    parent containers when a nested collapsing_header toggles, so we check
    each header's open/closed state every frame and adjust the outer
    child_window height manually.
    """
    for outer_id, (hdr_id, expanded_h, prev_open) in _panel_registry.items():
        try:
            is_open = dpg.get_value(hdr_id)
        except Exception:
            continue
        if is_open != prev_open:
            _panel_registry[outer_id] = (hdr_id, expanded_h, is_open)
            dpg.configure_item(
                outer_id, height=expanded_h if is_open else _COLLAPSED_H)


def _get_header_theme(color):
    """Get or create a DPG theme for a collapsing_header with colored text."""
    key = tuple(color)
    if key not in _panel_themes:
        with dpg.theme() as t:
            with dpg.theme_component(dpg.mvCollapsingHeader):
                dpg.add_theme_color(dpg.mvThemeCol_Text, color,
                                    category=dpg.mvThemeCat_Core)
        _panel_themes[key] = t
    return _panel_themes[key]


@contextlib.contextmanager
def control_panel(label, width=0, height=0, color=None,
                  default_open=True, tag=None, border=True):
    """Collapsible control panel with colored title.

    Combines two DPG widgets:
      outer child_window (size + border + no_scrollbar) →
        collapsing_header (collapse/expand + themed text color)

    When height > 0 the panel is registered for poll-based collapse:
    poll_collapsible_panels() checks each header's open/closed state
    every frame and resizes the outer child_window accordingly.

    Args:
        label:        Panel title text on the collapsing header.
        width:        Outer width in pixels. 0 = auto-fill.
        height:       Outer height in pixels. 0 = auto-fit to content.
        color:        RGB tuple for header text, e.g. (150, 200, 255).
                      None = default theme text color.
        default_open: Whether the panel starts expanded.
        tag:          Optional DPG tag for the outer child_window
                      (useful for show/hide via configure_item).
        border:       Visible border on outer container.

    Yields:
        The collapsing_header DPG id.
    """
    actual_h = height + _PANEL_PAD if height > 0 else 0
    outer_kw = {"width": width, "border": border, "no_scrollbar": True}
    if actual_h > 0:
        outer_kw["height"] = actual_h
    if tag is not None:
        outer_kw["tag"] = tag

    with dpg.child_window(**outer_kw) as outer_id:
        with dpg.collapsing_header(label=label,
                                   default_open=default_open) as hdr:
            if color is not None:
                dpg.bind_item_theme(hdr, _get_header_theme(color))
            yield hdr
        # Register for poll-based collapse/expand
        if actual_h > 0:
            _panel_registry[outer_id] = (hdr, actual_h, default_open)


def add_global_controls(defaults, state, cat_mode_callback=None,
                        pause_callback=None, reset_extra=None,
                        guide=None, guide_title="Guide"):
    """Add standard global controls row.

    Layout: [Reset All] [UI Scale] [Cat Mode] [Pause] [(no webcam)] [Guide ?]

    Args:
        defaults: Dictionary containing default values (must have "ui_scale")
        state: State object with use_camera and cat_mode attributes
        cat_mode_callback: Callback for cat mode checkbox (None to hide)
        pause_callback: Callback for pause checkbox (None to hide).
            Enabled only when a camera is present.
        reset_extra: Optional callable for demo-specific reset logic beyond DEFAULTS
        guide: Optional list of {"title": ..., "body": ...} guide steps
        guide_title: Title for the guide modal window
    """
    with dpg.group(horizontal=True):
        dpg.add_button(
            label="Reset All",
            callback=make_reset_all_callback(defaults, state, reset_extra)
        )
        dpg.add_spacer(width=10)

        dpg.add_combo(
            label="UI Scale",
            items=UI_SCALES,
            default_value=str(defaults.get("ui_scale", 1.5)),
            callback=make_ui_scale_callback(),
            width=80
        )
        dpg.add_spacer(width=20)

        if cat_mode_callback is not None:
            dpg.add_checkbox(
                label="Cat Mode",
                default_value=getattr(state, "cat_mode", False),
                callback=cat_mode_callback,
                tag="cat_mode_checkbox",
                enabled=getattr(state, "use_camera", True)
            )

        if pause_callback is not None:
            dpg.add_checkbox(
                label="Pause",
                default_value=False,
                callback=pause_callback,
                tag="pause_checkbox",
                enabled=getattr(state, "use_camera", True)
            )

        if (cat_mode_callback is not None or pause_callback is not None):
            if not getattr(state, "use_camera", True):
                dpg.add_text("(no webcam)", color=(255, 100, 100))

        if guide:
            dpg.add_spacer(width=20)
            add_guide_button(guide, guide_title)


def create_guide_window(guide_steps, title="Guide"):
    """Create a hidden modal window with scrollable guide content.

    Uses DPG's native title bar so the built-in close button works.

    Args:
        guide_steps: List of {"title": ..., "body": ...} dicts
        title: Window title text

    Returns:
        The DPG window tag (string)
    """
    tag = dpg.generate_uuid()
    with dpg.window(modal=True, show=False, tag=tag, label=title,
                    width=600, height=500, pos=[100, 50]):
        with dpg.child_window(border=False):
            for i, step in enumerate(guide_steps):
                if step.get("title"):
                    dpg.add_text(step["title"], color=(255, 220, 120))
                if step.get("body"):
                    dpg.add_text(step["body"], wrap=550, color=(200, 200, 220))
                if i < len(guide_steps) - 1:
                    dpg.add_spacer(height=5)
                    dpg.add_separator()
                    dpg.add_spacer(height=5)
    return tag


def add_guide_button(guide_steps, title="Guide"):
    """Add a "?" button that opens a guide modal.

    Must be called within a DPG layout context (e.g. inside a group).

    Args:
        guide_steps: List of {"title": ..., "body": ...} dicts
        title: Title for the guide modal window
    """
    if not guide_steps:
        return
    win_tag = create_guide_window(guide_steps, title)
    dpg.add_button(
        label="?", width=25,
        callback=lambda: dpg.configure_item(win_tag, show=True))


def add_parameter_row(label, tag, default, min_val, max_val, callback,
                      reset_callback, slider_type="float", format_str=None,
                      **kwargs):
    """Add a parameter row with label, slider, and reset button.

    Must be called within a dpg.table context (created by create_parameter_table).

    Args:
        label: Text label for the parameter
        tag: Unique tag for the slider widget
        default: Default value for the slider
        min_val: Minimum slider value
        max_val: Maximum slider value
        callback: Callback function when slider changes
        reset_callback: Callback function for reset button
        slider_type: "float" or "int"
        format_str: Optional format string for float sliders (e.g., "%.2f")
        **kwargs: Silently absorbs legacy keyword args (e.g. width).
    """
    with dpg.table_row():
        dpg.add_text(label)
        if slider_type == "float":
            fmt = {"format": format_str} if format_str else {}
            dpg.add_slider_float(
                tag=tag,
                default_value=default,
                min_value=min_val,
                max_value=max_val,
                callback=callback,
                width=-1,
                **fmt
            )
        else:
            dpg.add_slider_int(
                tag=tag,
                default_value=default,
                min_value=min_val,
                max_value=max_val,
                callback=callback,
                width=-1,
            )
        dpg.add_button(label="R", callback=reset_callback, width=25)


def add_parameter_spacer_row():
    """Add an empty spacer row in a parameter table.

    Must be called within a dpg.table context.
    Useful for filling out 2-column layouts.
    """
    with dpg.table_row():
        dpg.add_spacer()
        dpg.add_spacer()
        dpg.add_spacer()


@contextlib.contextmanager
def create_parameter_table():
    """Create a 3-column parameter table (label | slider | reset).

    The label column auto-fits to its text content.
    The slider column stretches to fill remaining width.
    The reset column is fixed at 30 px.
    """
    with dpg.table(
        header_row=False,
        borders_innerV=False,
        borders_outerV=False,
        borders_innerH=False,
        borders_outerH=False,
        policy=dpg.mvTable_SizingFixedFit,
    ):
        dpg.add_table_column()                                             # label (auto-fit)
        dpg.add_table_column(width_stretch=True)                           # slider (stretches)
        dpg.add_table_column(width_fixed=True, init_width_or_weight=30)   # reset
        yield


def setup_viewport(title, width, height, main_window_tag, resize_callback, ui_scale=1.5):
    """Setup viewport with resize handler.

    Args:
        title: Viewport window title
        width: Initial viewport width
        height: Initial viewport height
        main_window_tag: Tag of the main window to set as primary
        resize_callback: Callback function for viewport resize events
        ui_scale: Initial UI scale factor
    """
    with dpg.item_handler_registry(tag="viewport_handler"):
        dpg.add_item_resize_handler(callback=resize_callback)

    dpg.create_viewport(title=title, width=width, height=height)
    dpg.setup_dearpygui()
    dpg.show_viewport()
    dpg.set_primary_window(main_window_tag, True)
    dpg.bind_item_handler_registry(main_window_tag, "viewport_handler")
    dpg.set_global_font_scale(_ui_scale_to_gfs(ui_scale))


def create_texture(width, height, tag):
    """Create a blank RGBA texture.

    Must be called within a dpg.texture_registry context.

    Args:
        width: Texture width in pixels
        height: Texture height in pixels
        tag: Unique tag for the texture
    """
    blank_data = [0.0] * (width * height * 4)
    dpg.add_raw_texture(
        width, height, blank_data,
        format=dpg.mvFormat_Float_rgba,
        tag=tag
    )


def add_status_section():
    """Add a status text section with separators.

    Returns the tag "status_text" for updating.
    """
    dpg.add_separator()
    dpg.add_text("", tag="status_text")
    dpg.add_separator()


def add_image_pair(label1, texture1, image_tag1, label2, texture2, image_tag2):
    """Add a horizontal pair of labeled images.

    Args:
        label1: Label for first image
        texture1: Texture tag for first image
        image_tag1: Tag for first image widget
        label2: Label for second image
        texture2: Texture tag for second image
        image_tag2: Tag for second image widget
    """
    with dpg.group(horizontal=True):
        with dpg.group():
            dpg.add_text(label1)
            dpg.add_image(texture1, tag=image_tag1)
        dpg.add_spacer(width=10)
        with dpg.group():
            dpg.add_text(label2)
            dpg.add_image(texture2, tag=image_tag2)


def create_dual_parameter_table():
    """Create a 7-column table for dual-column parameter layouts.

    Layout: Label1 | Slider1 | Reset1 | Spacer | Label2 | Slider2 | Reset2

    Returns:
        dpg.table context manager
    """
    return dpg.table(
        header_row=False,
        borders_innerV=False,
        borders_outerV=False,
        borders_innerH=False,
        borders_outerH=False,
        policy=dpg.mvTable_SizingFixedFit
    )


def add_dual_table_columns(slider_width=100, reset_width=30, spacer_width=20):
    """Add columns for a dual-parameter table.

    Must be called immediately after creating the table with create_dual_parameter_table().
    Label columns auto-fit to text content; slider/reset columns are fixed-width.

    Args:
        slider_width: Width of slider columns
        reset_width: Width of reset button columns
        spacer_width: Width of spacer column between the two parameter sets
    """
    dpg.add_table_column()  # label 1 (auto-fit)
    dpg.add_table_column(width_fixed=True, init_width_or_weight=slider_width)
    dpg.add_table_column(width_fixed=True, init_width_or_weight=reset_width)
    dpg.add_table_column(width_fixed=True, init_width_or_weight=spacer_width)
    dpg.add_table_column()  # label 2 (auto-fit)
    dpg.add_table_column(width_fixed=True, init_width_or_weight=slider_width)
    dpg.add_table_column(width_fixed=True, init_width_or_weight=reset_width)
