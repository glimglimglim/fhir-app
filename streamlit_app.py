"""
Streamlit app to extract FHIR R4 JSON from medical PDFs **or images** with GPT‑4o.

🛠 **Fix:** The OpenAI vision endpoint (even in GPT‑4o‑mini) only accepts `gif`, `jpeg/jpg`, `png`, or `webp`.  Raw PDFs over `data:application/pdf;base64,…` trigger a **400 – invalid file format**.  

We therefore render each PDF page to PNG in‑memory (via **PyMuPDF**) and send those images instead.  Users who upload images continue to be sent as‑is.

Dependencies:
    pip install streamlit openai pillow pymupdf

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

import fitz  # PyMuPDF
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

# Max pages to render from each PDF to limit cost/payload
MAX_PDF_PAGES = 5

###############################################################################
# Utility functions
###############################################################################

def _pil_to_base64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def _pdf_to_png_base64_list(path: Path, max_pages: int = MAX_PDF_PAGES) -> List[str]:
    """Return list of base‑64 PNG strings (no prefix) for first *max_pages*."""
    pngs: List[str] = []
    doc = fitz.open(path)
    zoom = 2  # 2× scale for clarity
    mat = fitz.Matrix(zoom, zoom)
    for p in range(min(len(doc), max_pages)):
        pix = doc.load_page(p).get_pixmap(matrix=mat, alpha=False)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        pngs.append(_pil_to_base64(img))
    doc.close()
    return pngs


def build_content_for_file(path: Path) -> List[dict]:
    """Build the `content` elements for Chat API message."""
    mime, _ = mimetypes.guess_type(path)

    # Images — send directly
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

    # PDFs — render pages → PNG base64
    if mime == "application/pdf":
        b64_list = _pdf_to_png_base64_list(path)
        if not b64_list:
            raise ValueError("PDF contained no pages.")
        contents: List[dict] = []
        for b64 in b64_list:
            contents.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{b64}",
                        "detail": "high",
                    },
                }
            )
        return contents

    raise ValueError(f"Unsupported file type: {path}")


def gpt4o_fhir_from_file(path: Path) -> dict:
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

    try:
        api_key = st.secrets["OPENAI_API_KEY"]
    except Exception:
        api_key = None

    if not api_key:
        api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        api_key = st.text_input("Enter OpenAI API Key", type="password")

    if api_key:
        openai.api_key = api_key
        st.success("API key loaded.")
    else:
        st.error("OpenAI API key not found. Provide one to continue.")
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
            try:
                first_page = _pdf_to_png_base64_list(tmp_path, max_pages=1)[0]
                st.subheader("📄 PDF preview – page 1")
                st.image(f"data:image/png;base64,{first_page}")
            except Exception:
                st.info("PDF preview unavailable.")
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
