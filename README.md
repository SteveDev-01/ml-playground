# Machine Learning Playground

An end‑to‑end ML workflow on the Titanic dataset, demonstrating data loading, EDA, model training, evaluation, serving, containerization, and CI.

## 🚀 Features

- **Exploratory Data Analysis**  
  Interactive Jupyter notebook with data profiling and visualizations.

- **Modular Codebase**  
  Python modules under `src/` for data, features, models, training, evaluation, and API.

- **Hyperparameter Tuning**  
  Grid search over model parameters (configurable via `config/train_config.yaml`).

- **Model Serving**  
  FastAPI endpoint at `/predict` for live inference.

- **Containerization**  
  Dockerfile to build and run the app in a container.

- **CI/CD**  
  GitHub Actions workflow for linting (flake8) and testing (pytest) on every push.

## 📦 Installation

1. **Clone the repo**  
   ```bash
   git clone https://github.com/<your-username>/ml-playground.git
   cd ml-playground
   ```

2. **Create & activate a virtual environment**  
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies**  
   ```bash
   pip install -r requirements.txt
   ```

## 📊 Exploratory Data Analysis

Open the Jupyter notebook:
```bash
jupyter notebook notebooks/exploratory_data_analysis.ipynb
```

## 🤖 Training & Evaluation

- **Training:**  
  ```bash
  python src/train.py --config config/train_config.yaml
  ```
- **Evaluation:**  
  ```bash
  python src/evaluate.py --model-path models/best_model.pkl
  ```

## 🚀 Serving the Model

Start FastAPI server:
```bash
uvicorn src.api:app --reload
```
- **Endpoint:** `POST /predict` with JSON payload of features.

## 🐳 Docker

Build and run:
```bash
docker build -t ml-playground .
docker run -p 8000:8000 ml-playground
```

## ✅ CI/CD

GitHub Actions workflow (`.github/workflows/ci.yaml`) auto‑runs flake8 and pytest on every push.

---

## 🤝 Contributing

Feel free to open issues or submit PRs to improve the pipeline, add new models, or enhance deployment.

## 📄 License

This project is MIT‑licensed. See [LICENSE](LICENSE) for details.
