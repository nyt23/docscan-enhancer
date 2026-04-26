# Document Scanner & Enhancer

A small web app that turns phone photos of documents (notes, receipts, forms) into high-contrast, scan-like images that are easier to read and share. Upload an image, adjust sliders, and see the result update immediately.

## Tech stack

- Python 3
- [Streamlit](https://streamlit.io/) — UI and file upload
- [OpenCV](https://opencv.org/) (`opencv-python-headless`) — image processing
- NumPy — arrays and buffer decoding

## Features

- Upload common image formats (PNG, JPEG, WebP, BMP, TIFF, …)
- Side-by-side **original** and **scanned preview**
- **Blur intensity** — Gaussian kernel size for noise smoothing
- **Threshold sensitivity** — controls local black/white separation (adaptive threshold `C`)
- Optional **sharpen** and **adaptive window size** (sidebar expander)
- Processing runs on every slider change (full script rerun, typical Streamlit behavior)

## Image pipeline

1. Convert to **grayscale**
2. **Gaussian blur** (kernel size from the blur slider)
3. Optional **unsharp mask** on the grayscale (before thresholding)
4. **Adaptive Gaussian threshold** — produces the binary “scanned” look

## Setup

```bash
cd docscan-enhancer
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Run

```bash
streamlit run app.py
```

The app opens in your browser (Streamlit prints the local URL in the terminal).

## Project layout

| Path | Description |
|------|-------------|
| `app.py` | Streamlit entrypoint and OpenCV pipeline |
| `requirements.txt` | Python dependencies |

## Tips

- Noisy or grainy photos: try a slightly **larger blur** first.
- Text too faint or background too dark: adjust **threshold sensitivity**; small changes often matter.
- Uneven lighting or glare: open **Optional: sharpen & window** and try a different **adaptive window size**.

## License

Use and modify for your own projects or coursework as needed.
