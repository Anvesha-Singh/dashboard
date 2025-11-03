# Analog & Power Electronics VQA

A lightweight, resource-efficient Visual Question Answering (VQA) web interface for analog and power-electronics circuits. The application is built with Streamlit and provides two primary workflows:

- Image analysis (circuit/PCB/component inspection) using a SmolVLM-style vision-language model (mocked by default in this repo).
- SPICE simulation analysis (upload an LTspice `.asc` file to run simulated waveform analysis and plots).

This repository contains a demo-ready Streamlit app (`vqa_dashboard.py`) that uses mock model/simulation code for offline/demo usage and is structured for easy production integration with an actual SmolVLM model and a SPICE engine.

## Key features

- Upload circuit images (PNG/JPG/JPEG) and ask natural-language questions about components, defects, waveforms, voltages, currents, and topology.
- Upload LTspice `.asc` files and request simulation-focused questions (transient, AC sweep, DC operating point). The app returns a textual analysis and waveform plots.
- Dark themed, responsive UI using Streamlit with custom CSS styling.
- Session caching for the model so the model (or mock) is loaded once per server session.

## Files of interest

- `vqa_dashboard.py` — Main Streamlit application (single-file app). Contains all UI, mock model loading and inference functions, simulation mock, and plotting code.
- `requirements.txt` — Python dependencies used by the project (verify and update when enabling real model code).
- `index.html`, `src/`, `package.json`, and other frontend files are present if you are integrating or building a companion frontend; currently the Streamlit app is the main runnable artifact.

## How it works (high level)

1. On startup the app calls `load_smolvlm_model()` which currently returns a mocked model + processor + tokenizer after a short simulated delay. In production this should load the SmolVLM model, processor and tokenizer (see the commented code in `vqa_dashboard.py`).
2. The UI offers two columns: Image Analysis and Simulation Analysis. The user uploads either an image or an LTspice `.asc` file and then types a natural language question.
3. For image analysis, `run_smolvlm_inference()` is called. In this demo it returns intelligent mock replies based on keywords in the question (component identification, waveform analysis, defects, electrical analysis, or a general summary). In production, replace the mock block with actual model inference code using the model/processor/tokenizer.
4. For simulation analysis, `run_simulation_analysis()` currently simulates a SPICE run (delay) and returns realistic textual analysis plus 4 waveform plots (transient output voltage, supply current, power dissipation and frequency spectrum). In production you can integrate PyLTspice or call an LTspice command-line to generate real waveforms.

## Example user flows

- Image VQA:
  1. Upload a circuit PCB photo.
  2. Click "Analyze Image".
  3. Enter a question like: "What components are in this circuit?" or "Is there any visible defect?"
  4. Click "Get Answer" — the app displays the uploaded image and the model's analysis.

- Simulation VQA:
  1. Upload an LTspice `.asc` file.
  2. Click "Run Simulation".
  3. Ask a question like: "Show transient response" or "What's the settling time?"
  4. Click "Get Answer" — the app shows a textual simulation summary and waveform plots.

## Running locally (WSL / Linux / macOS)

Recommended: run inside the provided Python environment (venv/conda). The project contains `requirements.txt`.

1. Open WSL terminal in the project directory (project root where `vqa_dashboard.py` lives).

2. Create and activate a virtual environment (example using python -m venv):

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

3. Run the Streamlit app:

```bash
streamlit run vqa_dashboard.py
```

4. The Streamlit server typically opens http://localhost:8501. If running remotely or inside WSL, open the forwarded URL in your browser.

Notes for Windows with WSL: run the same commands inside the WSL shell. If you want to run in native Windows Python, adapt the venv activation commands accordingly.

## Dependencies

- The demo relies on the Python standard scientific stack (numpy, matplotlib, pillow) and Streamlit. See `requirements.txt` shipped in the repository. If you enable the real SmolVLM model you will need additional packages such as `transformers`, `torch`/`accelerate`, and model-specific dependencies.

Minimum (demo) dependencies you should ensure are installed:

- streamlit
- pillow
- numpy
- matplotlib

If enabling real model inference, add:

- transformers
- torch (preferably with CUDA if using GPU)
- accelerate (optional)

## Production integration notes (enabling SmolVLM)

The `vqa_dashboard.py` file includes commented production-ready code inside `load_smolvlm_model()` and `run_smolvlm_inference()` showing how to load a SmolVLM model with Hugging Face transformers. Steps to enable:

1. Install and pin `transformers`, `torch` (matching your CUDA), and any `trust_remote_code` dependencies.
2. Uncomment and set the `model_name` to the SmolVLM checkpoint you plan to use (e.g. a Hugging Face repo name). Use the smallest variant for low-resource deployment.
3. Consider using `torch_dtype=torch.float16` and `device_map='auto'` for memory savings on GPU.
4. For CPU-only deployments, use float32 and smaller batch sizes.
5. Update `requirements.txt` and CI config accordingly.

Important: model weights may be large. For edge or low-memory deployments prefer quantized or distilled variants and validate inference latency.

## Simulation engine integration

The demo simulates SPICE behavior. For accurate simulation:

1. Integrate PyLTspice or call LTspice on the host if available.
2. Parse `.asc` and run the requested analysis (transient/AC/DC) programmatically.
3. Capture waveform data and feed it into the plotting code (the current plotting code in `run_simulation_analysis()` can be reused for display).

## Developer notes / architecture

- Session state: Streamlit `st.session_state` tracks `analysis_mode`, `uploaded_file`, `question`, and `show_result` so users can interact without losing state.
- Caching: `@st.cache_resource` is used for model loading so the heavy model loads once per server session.
- Mock behavior: The repo intentionally uses mocked model responses and simulated SPICE to keep the demo lightweight.

Edge cases to consider when enabling production behavior:

- Large model initialization causing timeouts — prefer background model loading with a progress indicator.
- Uploaded file sizes — enforce maximum size limits and provide user feedback.
- Security — do not execute untrusted code embedded in `.asc` files. Sanitize all inputs.

## Example prompts the app understands (demo/mocks)

- "What components are in this circuit?"
- "Identify any defects visible in the PCB photo."
- "Analyze the transient response and give settling time."
- "What is the supply current and efficiency?"

These map to keyword branches in the mock inference logic and will return structured responses.

## Testing and verification

- Quick manual test: upload a sample PCB image and ask component-related question; upload a simple `.asc` file and ask for transient analysis. The mock responses and plot generation will validate UI and plumbing.
- Automated tests: consider adding unit tests for parsing `.asc` files and for the plotting functions (e.g., ensure `run_simulation_analysis()` returns a buffer and a non-empty answer string).

## Next steps / suggestions

- Add an `examples/` folder with sample images and `.asc` files for quicker demos.
- Add a small integration test that runs the Streamlit app (or a subset of functions) to validate key endpoints.
- Add model configuration via environment variables or a config file (model path/name, device, quantization flags).

## License & attribution

Include a license file for your project (e.g., MIT) if you plan to publish. If using pre-trained models or third-party code, follow their license terms and include attribution.

---