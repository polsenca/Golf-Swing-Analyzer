"""
Golf Swing Analyzer — Streamlit front-end
Run locally:  streamlit run app.py
Deploy:       push to GitHub → connect Streamlit Community Cloud
"""

import hashlib
import json
import tempfile
import threading
import time
from pathlib import Path

try:
    from streamlit.runtime.scriptrunner import add_script_run_ctx
except ImportError:
    add_script_run_ctx = None

import altair as alt
import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# Page config (must be first Streamlit call)
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Golf Swing Analyzer",
    page_icon="⛳",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Import analyzer (same directory)
# ---------------------------------------------------------------------------
import sys
sys.path.insert(0, str(Path(__file__).parent))
from golf_swing_analyzer import analyze, ensure_model  # noqa: E402

# ---------------------------------------------------------------------------
# Session-state defaults
# ---------------------------------------------------------------------------
_DEFAULTS = {
    "analysis_state": "idle",   # idle | running | done | error
    "report": None,
    "progress": (0, 1, ""),
    "error": None,
    "annotated_video_bytes": None,
    "last_run_key": None,
    "tmp_video_path": None,
}
for _k, _v in _DEFAULTS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
PHASE_ORDER = [
    "ADDRESS", "TAKEAWAY", "BACKSWING", "TOP",
    "DOWNSWING", "IMPACT", "FOLLOW_THROUGH", "FINISH",
]

METRIC_LABELS = {
    "spine_angle":       "Spine tilt (°)",
    "shoulder_turn":     "Shoulder turn (°)",
    "hip_rotation":      "Hip rotation (°)",
    "lead_knee_flex":    "Lead knee (°)",
    "trail_knee_flex":   "Trail knee (°)",
    "lead_elbow_angle":  "Lead elbow (°)",
    "head_drift_x":      "Head drift X (%)",
    "head_drift_y":      "Head drift Y (%)",
}

# Target ranges for reference lines in charts
METRIC_TARGETS = {
    "spine_angle":      (25, 45),
    "shoulder_turn":    (20, 65),
    "lead_knee_flex":   (140, 165),
    "trail_knee_flex":  (140, 165),
    "lead_elbow_angle": (155, 185),
    "head_drift_x":     (-6, 6),
    "head_drift_y":     (-6, 6),
}

SEV_ICON = {"error": "🔴", "warning": "🟡", "tip": "🔵", "ok": "🟢"}
SEV_COLOR = {"error": "error", "warning": "warning", "tip": "info", "ok": "success"}


def _fmt(val, key):
    if val is None:
        return "—"
    if "drift" in key:
        return f"{val * 100:.1f}%"
    return f"{val:.1f}°"


def _run_analysis(tmp_path, output_path, handed, model, skip):
    """Runs in a background thread."""
    try:
        def cb(cur, tot, phase):
            st.session_state.progress = (cur, tot, phase)

        report = analyze(
            video_path=tmp_path,
            output_path=output_path,
            report_path=None,
            handed=handed,
            model_variant=model,
            skip=skip,
            progress_callback=cb,
        )
        if output_path:
            with open(output_path, "rb") as f:
                st.session_state.annotated_video_bytes = f.read()

        st.session_state.report = report
        st.session_state.analysis_state = "done"

    except Exception as exc:
        st.session_state.error = str(exc)
        st.session_state.analysis_state = "error"

    finally:
        # Clean up temp input file
        try:
            Path(tmp_path).unlink(missing_ok=True)
        except Exception:
            pass
        if output_path:
            try:
                Path(output_path).unlink(missing_ok=True)
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Sidebar — upload + settings
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("⛳ Golf Swing Analyzer")
    st.caption("Upload a face-on swing video and get instant feedback on your mechanics.")

    uploaded = st.file_uploader(
        "Video file",
        type=["mp4", "mov", "avi", "m4v", "MOV", "MP4"],
        help="Best results with a face-on view, full body visible, 30–60 fps.",
    )

    st.divider()

    col_a, col_b = st.columns(2)
    with col_a:
        handed = st.radio("Handedness", ["Right", "Left"], index=0).lower()
    with col_b:
        model = st.selectbox(
            "Model accuracy",
            ["lite", "full", "heavy"],
            index=1,
            help="lite = fastest · heavy = most accurate (slower first run — downloads model)",
        )

    skip = st.select_slider(
        "Speed vs. accuracy",
        options=[0, 1, 2, 3],
        value=0,
        format_func=lambda x: ["All frames", "2× faster", "3× faster", "4× faster"][x],
    )

    save_video = st.checkbox(
        "Save annotated video",
        value=True,
        help="Adds ~50% to processing time. Produces a downloadable video with skeleton overlay.",
    )

    st.divider()

    # Compute a fingerprint of current settings so we know if they changed
    run_key = (
        hashlib.md5(
            f"{getattr(uploaded, 'name', '')}{getattr(uploaded, 'size', 0)}"
            f"{handed}{model}{skip}{save_video}".encode()
        ).hexdigest()
        if uploaded
        else None
    )

    settings_changed = run_key != st.session_state.last_run_key
    can_analyze = (
        uploaded is not None
        and st.session_state.analysis_state in ("idle", "error")
        and settings_changed
    )

    analyze_btn = st.button(
        "Analyze Swing",
        disabled=not can_analyze,
        use_container_width=True,
        type="primary",
    )

    if st.session_state.analysis_state == "done":
        if st.button("Clear / Re-analyze", use_container_width=True):
            for k, v in _DEFAULTS.items():
                st.session_state[k] = v
            st.rerun()

# ---------------------------------------------------------------------------
# Kick off background analysis
# ---------------------------------------------------------------------------
if analyze_btn and uploaded and can_analyze:
    # Write upload to temp file
    suffix = Path(uploaded.name).suffix or ".mp4"
    tmp_in = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    tmp_in.write(uploaded.getbuffer())
    tmp_in.close()

    out_path = None
    if save_video:
        tmp_out = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
        tmp_out.close()
        out_path = tmp_out.name

    st.session_state.tmp_video_path = tmp_in.name
    st.session_state.analysis_state = "running"
    st.session_state.progress = (0, 1, "Starting…")
    st.session_state.report = None
    st.session_state.annotated_video_bytes = None
    st.session_state.error = None
    st.session_state.last_run_key = run_key

    t = threading.Thread(
        target=_run_analysis,
        args=(tmp_in.name, out_path, handed, model, skip),
        daemon=True,
    )
    if add_script_run_ctx:
        add_script_run_ctx(t)
    t.start()
    st.rerun()

# ---------------------------------------------------------------------------
# Progress display
# ---------------------------------------------------------------------------
if st.session_state.analysis_state == "running":
    cur, tot, phase = st.session_state.progress
    pct = cur / tot if tot > 0 else 0
    label = f"Analyzing… **{phase.replace('_', ' ')}**  ({cur} / {tot} frames)"
    st.progress(pct, text=label)
    st.caption("This typically takes 30–90 seconds depending on video length and model choice.")
    time.sleep(0.5)
    st.rerun()

# ---------------------------------------------------------------------------
# Error state
# ---------------------------------------------------------------------------
elif st.session_state.analysis_state == "error":
    st.error(f"Analysis failed: {st.session_state.error}")

# ---------------------------------------------------------------------------
# Landing screen
# ---------------------------------------------------------------------------
elif st.session_state.analysis_state == "idle":
    st.markdown(
        """
        ## How it works
        1. **Upload** a swing video using the sidebar (face-on angle gives best results)
        2. **Choose** your handedness and model accuracy
        3. Hit **Analyze Swing** — processing takes ~30–90 seconds
        4. Review your **feedback**, **per-phase metrics**, and download the **annotated video**

        ---
        ### What gets measured
        | Metric | Phases checked |
        |---|---|
        | Spine tilt | Address, Impact |
        | Shoulder turn | Top of backswing |
        | Hip rotation | Impact |
        | Knee flex | Address |
        | Lead arm extension | Impact |
        | Head stability | Backswing → Follow-through |

        > **Camera tip:** Film face-on at hip height with your full body in frame.
        """,
        unsafe_allow_html=False,
    )

# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------
elif st.session_state.analysis_state == "done" and st.session_state.report:
    report = st.session_state.report
    summary = report["summary"]
    phase_avgs = report.get("phase_averages", {})

    # ── Top-line stats ──────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Duration", f"{report['duration_seconds']:.1f}s")
    c2.metric("Frame rate", f"{report['fps']:.0f} fps")
    c3.metric("Frames", str(report["total_frames"]))
    n_issues = len(summary["error"]) + len(summary["warning"])
    c4.metric("Issues found", str(n_issues), delta=None)

    st.divider()

    # ── Tabs ────────────────────────────────────────────────────────────────
    tab_feedback, tab_phases, tab_video, tab_raw = st.tabs(
        ["💬 Feedback", "📊 Phase Metrics", "🎬 Annotated Video", "🔧 Raw Data"]
    )

    # ── Tab 1: Feedback ──────────────────────────────────────────────────────
    with tab_feedback:
        col_left, col_right = st.columns([3, 2])

        with col_left:
            st.subheader("Swing Feedback")
            all_fb = report["feedback"]
            if not all_fb:
                st.success("No issues detected — great swing!")
            else:
                for sev in ("error", "warning", "tip", "ok"):
                    for item in summary[sev]:
                        icon = SEV_ICON[sev]
                        key_label = item["key"].replace("_", " ").title()
                        val_str = f"  `{item['value']}`" if item.get("value") is not None else ""
                        msg_fn = getattr(st, SEV_COLOR[sev])
                        msg_fn(f"{icon} **{key_label}**{val_str}  \n{item['message']}")

        with col_right:
            st.subheader("Score Card")
            sc1, sc2 = st.columns(2)
            sc1.metric("Errors",   len(summary["error"]))
            sc2.metric("Warnings", len(summary["warning"]))
            sc3, sc4 = st.columns(2)
            sc3.metric("Tips",     len(summary["tip"]))
            sc4.metric("OK",       len(summary["ok"]))

            st.divider()
            st.download_button(
                label="Download JSON report",
                data=json.dumps(report, indent=2),
                file_name="swing_report.json",
                mime="application/json",
                use_container_width=True,
            )

    # ── Tab 2: Phase Metrics ─────────────────────────────────────────────────
    with tab_phases:
        col_tbl, col_chart = st.columns([3, 2])

        with col_tbl:
            st.subheader("Per-Phase Averages")

            rows = []
            for ph in PHASE_ORDER:
                avgs = phase_avgs.get(ph)
                if not avgs:
                    continue
                row = {"Phase": ph.replace("_", " "), "Frames": avgs.get("frame_count", 0)}
                for k, lbl in METRIC_LABELS.items():
                    v = avgs.get(k)
                    row[lbl] = _fmt(v, k)
                rows.append(row)

            if rows:
                df = pd.DataFrame(rows).set_index("Phase")
                st.dataframe(df, use_container_width=True)

        with col_chart:
            st.subheader("Metric by Phase")
            metric_choice = st.selectbox(
                "Choose metric",
                list(METRIC_LABELS.keys()),
                format_func=lambda k: METRIC_LABELS[k],
            )

            chart_rows = []
            for ph in PHASE_ORDER:
                avgs = phase_avgs.get(ph)
                if not avgs:
                    continue
                v = avgs.get(metric_choice)
                if v is None:
                    continue
                if "drift" in metric_choice:
                    v = v * 100  # convert to %
                chart_rows.append({"Phase": ph.replace("_", " "), "Value": v})

            if chart_rows:
                cdf = pd.DataFrame(chart_rows)
                base = alt.Chart(cdf).mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4).encode(
                    x=alt.X("Phase:N", sort=[r["Phase"] for r in chart_rows],
                            axis=alt.Axis(labelAngle=-35)),
                    y=alt.Y("Value:Q", title=METRIC_LABELS[metric_choice]),
                    color=alt.condition(
                        alt.datum.Value > 0,
                        alt.value("#2e7d32"),
                        alt.value("#c62828"),
                    ),
                    tooltip=["Phase", alt.Tooltip("Value:Q", format=".1f")],
                )

                # Add target range band if available
                layers = [base]
                if metric_choice in METRIC_TARGETS:
                    lo, hi = METRIC_TARGETS[metric_choice]
                    band_df = pd.DataFrame({"lo": [lo], "hi": [hi]})
                    band = (
                        alt.Chart(band_df)
                        .mark_rect(opacity=0.12, color="#1565c0")
                        .encode(y="lo:Q", y2="hi:Q")
                    )
                    lo_line = (
                        alt.Chart(band_df)
                        .mark_rule(color="#1565c0", strokeDash=[4, 4], opacity=0.6)
                        .encode(y="lo:Q")
                    )
                    hi_line = (
                        alt.Chart(band_df)
                        .mark_rule(color="#1565c0", strokeDash=[4, 4], opacity=0.6)
                        .encode(y="hi:Q")
                    )
                    layers += [band, lo_line, hi_line]

                st.altair_chart(alt.layer(*layers).properties(height=320), use_container_width=True)
                if metric_choice in METRIC_TARGETS:
                    lo, hi = METRIC_TARGETS[metric_choice]
                    unit = "%" if "drift" in metric_choice else "°"
                    st.caption(f"Blue band = target range: {lo}–{hi}{unit}")
            else:
                st.info("No data available for this metric.")

    # ── Tab 3: Annotated Video ───────────────────────────────────────────────
    with tab_video:
        video_bytes = st.session_state.annotated_video_bytes
        if video_bytes:
            st.subheader("Annotated Swing")
            st.caption(
                "The skeleton overlay and phase/metric labels are burned into every frame. "
                "Scrub to any position to inspect a specific moment."
            )
            st.video(video_bytes)
            st.download_button(
                label="Download annotated video",
                data=video_bytes,
                file_name="swing_annotated.mp4",
                mime="video/mp4",
                use_container_width=True,
            )
        else:
            st.info(
                "Enable **Save annotated video** in the sidebar before running analysis "
                "to generate the skeleton-overlay video here."
            )

    # ── Tab 4: Raw Data ──────────────────────────────────────────────────────
    with tab_raw:
        st.subheader("Full JSON Report")
        st.json(report)
