"""
Streamlit app to extract FHIR R4 JSON from medical PDFs **or images** with GPT‑4o.

Key change: PDFs are passed **directly as base‑64‐encoded data URLs**, so GPT‑4o‑mini parses the raw PDF without Vision.

This revision also hardens secret handling: if no secrets.toml is present, the app now *silently falls back* to an environment variable or manual input, instead of crashing with `StreamlitSecretNotFoundError`.

Dependencies:
    pip install streamlit openai pillow

Run from a terminal with:
    streamlit run streamlit_app.py
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

def _pil_to_base64(img: Image.Image) -> str:
    """Convert a PIL Image to a base‑64‑encoded PNG string (without the prefix)."""
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def build_content_for_file(path: Path) -> List[dict]:
    """Build the `content` payload for the Chat API message.

    * Images → base‑64 PNG data URLs.
    * PDFs  → base‑64 **PDF** data URLs.
    """
    mime, _ = mimetypes.guess_type(path)

    if mime and mime.startswith("image/"):
        img = Image.open(path).convert("RGB")
        b64 = _pil_to_base64(img)
        return [
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{b64}",
                    "detail": "high",
                },
            }
        ]

    if mime == "application/pdf":
        with open(path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        return [
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:application/pdf;base64,{b64}",
                    "detail": "high",
                },
            }
        ]

    raise ValueError(f"Unsupported file type: {path}")


def gpt4o_fhir_from_file(path: Path) -> dict:
    """Send the uploaded document/image to GPT‑4o‑mini and return FHIR JSON."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_content_for_file(path)},
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
st.title("📄 ➜ 🩺  FHIR Extractor for Medical Documents")

st.markdown(
    "Upload a medical PDF or image and receive structured **FHIR R4 JSON**.\n"
    "No data is stored server‑side."
)

with st.sidebar:
    st.header("🔑 OpenAI API Key")

    # 1️⃣ Try secrets.toml first (won't crash if missing).
    try:
        api_key = st.secrets["OPENAI_API_KEY"]
    except Exception:
        api_key = None

    # 2️⃣ Fall back to an env‑var for local dev.
    if not api_key:
        api_key = os.getenv("OPENAI_API_KEY")

    # 3️⃣ If still missing, prompt the user.
    if not api_key:
        api_key = st.text_input("Enter OpenAI API Key", type="password")

    if api_key:
        openai.api_key = api_key
        st.success("API key loaded.")
    else:
        st.error(
            "API key not found. Add it to `.streamlit/secrets.toml`, set the "
            "`OPENAI_API_KEY` environment variable, or paste it above."
        )
        st.stop()

uploaded_file = st.file_uploader(
    "Choose a PDF or image",
    type=["pdf", "png", "jpg", "jpeg", "tiff", "tif"],
)

if uploaded_file and openai.api_key:
    if st.button("🚀 Extract FHIR JSON", type="primary"):
        suffix = Path(uploaded_file.name).suffix
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded_file.getbuffer())
            tmp_path = Path(tmp.name)

        with st.spinner("Sending file to GPT‑4o …"):
            try:
                fhir_json = gpt4o_fhir_from_file(tmp_path)
            except Exception as e:
                st.error(f"OpenAI API error: {e}")
                st.stop()

        st.success("FHIR extraction complete!")

        mime, _ = mimetypes.guess_type(tmp_path)
        if mime and mime.startswith("image/"):
            st.subheader("🖼️ Image preview")
            st.image(tmp_path, use_container_width=True)
        elif mime == "application/pdf":
            st.info("PDF uploaded and sent as base‑64. Preview not available.")
        else:
            st.info("File uploaded but preview not available.")

        st.subheader("🧾 FHIR R4 JSON")
        st.json(fhir_json, expanded=True)
        st.download_button(
            label="💾 Download JSON",
            data=json.dumps(fhir_json, indent=2),
            file_name=f"{Path(uploaded_file.name).stem}_fhir.json",
            mime="application/json",
        )
