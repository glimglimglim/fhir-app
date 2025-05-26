"""
Streamlit app to extract FHIR R4 JSON from medical PDFs or images using GPT‑4o.

Install dependencies:
    pip install streamlit openai pillow pdf2image

`pdf2image` requires the Poppler utilities (https://poppler.freedesktop.org) on your PATH.

Run the app **from a terminal** with:

    streamlit run streamlit_app.py

Running it via plain `python streamlit_app.py` will not spin‑up the Streamlit
server and will show *ScriptRunContext* warnings.
"""

from __future__ import annotations

import base64
import io
import json
import mimetypes
import os
import tempfile
from pathlib import Path
from typing import List

import openai
import streamlit as st
from PIL import Image

try:
    from pdf2image import convert_from_path
except ImportError:
    convert_from_path = None  # We handle the missing dependency later.

###############################################################################
# Configuration
###############################################################################

SYSTEM_PROMPT: str = (
    "You are a medical data extractor. "
    "Return **only valid FHIR R4 JSON** that captures every clinical datum from the input. "
    "If uncertain about a value, omit it or leave the field empty—do not invent data."
)

###############################################################################
# Utility functions
###############################################################################

def _file_to_images(path: Path) -> List[Image.Image]:
    """Convert a PDF (all pages) or a single image file into a list of PIL Images."""
    mime, _ = mimetypes.guess_type(path)

    if mime == "application/pdf":
        if convert_from_path is None:
            raise RuntimeError(
                "`pdf2image` isn't installed, or Poppler is missing. "
                "Install with `pip install pdf2image` and add Poppler utilities to PATH."
            )
        # 300 DPI strikes a good balance between OCR accuracy and file size.
        return convert_from_path(path, dpi=300)

    if mime and mime.startswith("image/"):
        return [Image.open(path)]

    raise ValueError(f"Unsupported file type: {path}")


def _pil_to_base64(img: Image.Image) -> str:
    """Convert a PIL Image to a base‑64‑encoded PNG string (without the prefix)."""
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def file_to_base64_chunks(path: Path) -> List[str]:
    """Return one base‑64 PNG string *per* page / image in the file."""
    return [_pil_to_base64(im.convert("RGB")) for im in _file_to_images(path)]


def gpt4o_fhir_from_images(b64_images: List[str]) -> dict:
    """Call GPT‑4o‑mini with the images and get FHIR JSON back as a Python dict."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{b64}",
                        "detail": "high",
                    },
                }
                for b64 in b64_images
            ],
        },
    ]

    resp = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=0.0,
        response_format={"type": "json_object"},
        stream=False,
        max_tokens=4096,
    )

    return json.loads(resp.choices[0].message.content)

###############################################################################
# Streamlit UI
###############################################################################

st.set_page_config(page_title="FHIR Extractor", layout="centered")
st.title("📄 ➜ 🩺  FHIR Extractor for Medical Documents")

st.markdown(
    "Upload a medical PDF or image and receive structured **FHIR R4 JSON** "
    "No data is stored server‑side."
)

with st.sidebar:
    st.header("🔑 OpenAI API Key")
    # Gracefully attempt to read a default key from secrets.toml *or* env‑var.
    try:
        default_key = st.secrets["OPENAI_API_KEY"]
    except Exception:  # No secrets file or key missing.
        default_key = os.getenv("OPENAI_API_KEY", "")

    api_key_input = st.text_input(
        "Enter your OpenAI API key",
        value=default_key,
        type="password",
        placeholder="sk‑...",
    )

    if api_key_input:
        openai.api_key = api_key_input
    elif os.getenv("OPENAI_API_KEY"):
        openai.api_key = os.getenv("OPENAI_API_KEY")

    st.markdown(
        "---\n⚠️ **Privacy reminder:** ensure you are authorised to process any patient data you upload."
    )

uploaded_file = st.file_uploader(
    "Choose a PDF or image",
    type=["pdf", "png", "jpg", "jpeg", "tiff", "tif"],
)

# A small stateful flag so the preview checkbox only shows after a successful run.
show_images = st.session_state.get("show_images", False)

if uploaded_file and (openai.api_key or api_key_input):
    if st.button("🚀 Extract FHIR JSON", type="primary"):
        suffix = Path(uploaded_file.name).suffix
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded_file.getbuffer())
            tmp_path = Path(tmp.name)

        with st.spinner("Converting file …"):
            try:
                b64_chunks = file_to_base64_chunks(tmp_path)
                images = _file_to_images(tmp_path)
            except Exception as e:
                st.error(f"Error processing file: {e}")
                st.stop()

        with st.spinner(f"Sending {len(b64_chunks)} page(s)/image(s) to GPT‑4o‑mini …"):
            try:
                fhir_json = gpt4o_fhir_from_images(b64_chunks)
            except Exception as e:
                st.error(f"OpenAI API error: {e}")
                st.stop()

        st.success("FHIR extraction complete!")

        # Side-by-side layout
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("🖼️ Converted Image(s)")
            for idx, img in enumerate(images, start=1):
                st.image(img, caption=f"Page {idx}", use_container_width=True)

        with col2:
            st.subheader("🧾 FHIR R4 JSON")
            st.json(fhir_json, expanded=True)
            st.download_button(
                label="💾 Download JSON",
                data=json.dumps(fhir_json, indent=2),
                file_name=f"{Path(uploaded_file.name).stem}_fhir.json",
                mime="application/json",
            )