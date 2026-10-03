# LLM Drift Detector

A lightweight framework using **FastAPI**, **Streamlit**, and **Sentence-Transformers** (`all-MiniLM-L6-v2`) to capture and visualize drift in live LLM production queries relative to a baseline reference dataset.

## Project Setup and Running
1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the Backend API**:
   ```bash
   uvicorn app:app --reload
   ```
   The backend is configured to work both when executed from the project root and when imported as a package.

3. **Run the Streamlit Dashboard**:
   ```bash
   python -m streamlit run dashboard.py
   ```
   If the `streamlit` command is not recognized in your environment, use the module form above. If your API is running on a non-default URL or port, set `LLM_DRIFT_API_URL` before launching the dashboard.

4. **Package Import Notes**:
   ```python
   from llm_drift_detector.detector import LLMDriftDetector
   ```
   This project includes a reusable package entry point to avoid import errors when the app is run outside a single-file script setup.
