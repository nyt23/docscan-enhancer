"""
Document Scanner & Enhancer — Streamlit + OpenCV
Pipeline: grayscale → Gaussian blur → optional sharpen → adaptive threshold
"""

from __future__ import annotations

import cv2
import numpy as np
import streamlit as st


def _odd_kernel(n: int) -> int:
    n = int(n)
    if n < 1:
        return 1
    return n if n % 2 == 1 else n + 1


def _odd_block(n: int) -> int:
    n = _odd_kernel(n)
    return max(3, min(n, 99))


def process_document(
    bgr: np.ndarray,
    blur_kernel: int,
    block_size: int,
    thresh_c: float,
    sharpen: float,
) -> np.ndarray:
    """Return single-channel scan-like image (uint8, 0–255)."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    k = _odd_kernel(blur_kernel)
    blurred = cv2.GaussianBlur(gray, (k, k), 0)

    if sharpen > 0:
        sigma = 1.0 + sharpen * 2.0
        base = blurred.astype(np.float32)
        smooth = cv2.GaussianBlur(base, (0, 0), sigmaX=sigma, sigmaY=sigma)
        amount = min(sharpen * 1.2, 2.5)
        blurred = np.clip(base * (1.0 + amount) - smooth * amount, 0, 255).astype(
            np.uint8
        )

    bs = _odd_block(block_size)
    c = float(thresh_c)
    out = cv2.adaptiveThreshold(
        blurred,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        bs,
        c,
    )
    return out


def decode_upload(file_bytes: bytes) -> np.ndarray | None:
    arr = np.frombuffer(file_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return img


def main() -> None:
    st.set_page_config(
        page_title="Document Scanner & Enhancer",
        page_icon="📄",
        layout="wide",
    )

    st.title("Document Scanner & Enhancer")
    st.caption(
        "Upload a document photo and tune blur, threshold, and sharpening for a "
        "clean, high-contrast scan-like result."
    )

    with st.sidebar:
        st.header("Processing")
        blur_kernel = st.slider(
            "Blur intensity (kernel size)",
            min_value=1,
            max_value=21,
            value=5,
            step=2,
            help="Larger values smooth more noise but can soften text.",
        )
        thresh_c = st.slider(
            "Threshold sensitivity",
            min_value=-10.0,
            max_value=30.0,
            value=8.0,
            step=0.5,
            help="Controls black/white separation (OpenCV adaptive threshold C). "
            "Higher values often lighten the page; lower values darken ink.",
        )
        with st.expander("Optional: sharpen & window"):
            sharpen = st.slider(
                "Sharpen strength",
                min_value=0.0,
                max_value=2.0,
                value=0.35,
                step=0.05,
                help="Unsharp mask before thresholding. Set to 0 to disable.",
            )
            block_size = st.slider(
                "Adaptive window size",
                min_value=3,
                max_value=51,
                value=15,
                step=2,
                help="Larger neighborhoods adapt more slowly to shadows and glare.",
            )

    uploaded = st.file_uploader(
        "Upload an image",
        type=["png", "jpg", "jpeg", "webp", "bmp", "tiff", "tif"],
    )

    if uploaded is None:
        st.info("Choose an image to see the original and processed preview.")
        st.caption(
            "Transforms: grayscale → Gaussian blur → optional unsharp mask → "
            "adaptive Gaussian threshold."
        )
        return

    data = uploaded.getvalue()
    bgr = decode_upload(data)
    if bgr is None:
        st.error("Could not read this file as an image.")
        return

    processed = process_document(
        bgr,
        blur_kernel=blur_kernel,
        block_size=block_size,
        thresh_c=thresh_c,
        sharpen=sharpen,
    )

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Original")
        st.image(
            cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB),
            use_container_width=True,
        )
    with col2:
        st.subheader("Scanned preview")
        st.image(processed, use_container_width=True, clamp=True)

    st.divider()
    st.caption(
        "Transforms: grayscale → Gaussian blur → optional unsharp mask → "
        "adaptive Gaussian threshold."
    )


if __name__ == "__main__":
    main()
