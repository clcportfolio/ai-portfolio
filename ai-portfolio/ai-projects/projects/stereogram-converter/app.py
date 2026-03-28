"""
app.py — Streamlit UI for stereogram-converter.

Upload a depth map image → adjust parameters → download a stereogram.
No LLM. No API keys required.

Run with:
    streamlit run app.py
"""

import io

import streamlit as st

import pipeline

# ── Page config ────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Stereogram Converter",
    page_icon="👁",
    layout="wide",
)

# ── Sidebar ────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("👁 Stereogram Converter")
    st.markdown(
        "Upload a **depth map** — a greyscale image where bright pixels are "
        "close to the viewer and dark pixels are far away — and this tool "
        "converts it into a **stereogram** (Magic Eye image)."
    )

    st.divider()

    st.markdown("**How to view**")
    st.markdown(
        "Hold the image at arm's length. Relax your focus as if looking "
        "through the screen at something far behind it. The 3D shape should "
        "snap into view within a few seconds."
    )

    st.divider()

    st.markdown("**Tech stack**")
    st.markdown(
        "- [Pillow](https://pillow.readthedocs.io) — image I/O\n"
        "- [NumPy](https://numpy.org) — SIRTS algorithm\n"
        "- [Streamlit](https://streamlit.io) — UI\n"
        "- No LLM / no API keys required"
    )

    st.divider()
    st.markdown("[GitHub Repo](https://github.com/your-repo)  ·  Part of the AI Portfolio")

# ── Main layout ────────────────────────────────────────────────────────────────

st.title("Depth Map → Stereogram")
st.caption("Upload a greyscale depth map and get a Magic Eye image back.")

col_upload, col_params = st.columns([3, 2], gap="large")

with col_upload:
    st.subheader("1. Upload images")

    depth_file = st.file_uploader(
        "Depth map (required)",
        type=["png", "jpg", "jpeg", "bmp", "webp", "tiff", "tif", "avif"],
        help="Greyscale image: white = near, black = far. Colour images are converted to luminance.",
    )

    texture_file = st.file_uploader(
        "Texture tile (optional)",
        type=["png", "jpg", "jpeg", "bmp", "webp", "tiff", "tif", "avif"],
        help="Pattern used to fill the stereogram. Leave blank to use random colour noise.",
    )

with col_params:
    st.subheader("2. Adjust parameters")

    auto_eye_sep = st.checkbox("Auto eye separation (image width ÷ 8)", value=True)
    eye_separation = None
    if not auto_eye_sep:
        eye_separation = st.slider(
            "Eye separation (px)",
            min_value=40, max_value=300, value=100, step=5,
            help="Horizontal distance between the two virtual eyes. Wider = easier to view.",
        )

    depth_factor = st.slider(
        "Depth factor",
        min_value=0.10, max_value=0.80, value=0.33, step=0.05,
        format="%.2f",
        help="Controls how pronounced the 3D effect is. 0.33 is a safe starting point.",
    )

st.divider()

# ── Run ────────────────────────────────────────────────────────────────────────

run_disabled = depth_file is None
st.subheader("3. Generate")

if run_disabled:
    st.info("Upload a depth map above to enable the generator.")

if st.button("Generate Stereogram", type="primary", disabled=run_disabled):
    user_input = {
        "depth_map_bytes": depth_file.getvalue(),
        "depth_map_name":  depth_file.name,
        "depth_factor":    depth_factor,
    }
    if eye_separation is not None:
        user_input["eye_separation"] = eye_separation
    if texture_file is not None:
        user_input["texture_bytes"] = texture_file.getvalue()
        user_input["texture_name"]  = texture_file.name

    with st.spinner("Rendering stereogram..."):
        try:
            result = pipeline.run(user_input)
        except ValueError as e:
            st.error(f"Input validation failed: {e}")
            st.stop()

    # ── Results ────────────────────────────────────────────────────────────────

    if not result.get("output_bytes"):
        st.error("Rendering failed. See details below.")
    else:
        st.success("Done!")

        out_col, info_col = st.columns([3, 2], gap="large")

        with out_col:
            st.subheader("Result")
            st.image(result["output_bytes"], caption="Stereogram — relax your focus to see in 3D", use_container_width=True)

            st.download_button(
                label="⬇ Download stereogram",
                data=result["output_bytes"],
                file_name=result.get("output_name", "stereogram.png"),
                mime="image/png",
            )

        with info_col:
            st.subheader("Parameters used")
            st.metric("Width",          f"{result.get('width', '?')} px")
            st.metric("Height",         f"{result.get('height', '?')} px")
            st.metric("Eye separation", f"{result.get('eye_separation', '?')} px")
            st.metric("Depth factor",   f"{result.get('depth_factor', '?'):.2f}")

            texture_label = texture_file.name if texture_file else "random noise"
            st.markdown(f"**Texture:** {texture_label}")

        # Show depth map alongside for comparison
        with st.expander("Input depth map", expanded=False):
            st.image(depth_file.getvalue(), caption=f"Depth map: {depth_file.name}", use_container_width=True)

    # ── Warnings ───────────────────────────────────────────────────────────────
    if result.get("errors"):
        with st.expander("Pipeline warnings", expanded=True):
            for msg in result["errors"]:
                st.warning(msg)
