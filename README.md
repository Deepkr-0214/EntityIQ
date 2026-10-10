# EntityIQ: Business Entity Resolution Platform

EntityIQ identifies business records from different data sources that refer to the same real-world organization. It is designed for records with inconsistent business names, partial or differently formatted addresses, spelling variations, legal suffixes, and selected non-Latin scripts.

The project combines a trained machine-learning matching model with a REST API, a Streamlit dashboard, a documentation retrieval assistant (RAG), and a tool-coordinating agent. These components have different jobs: the LightGBM model scores possible business matches; RAG answers questions from project documents; and the agent coordinates approved platform tools.

## Where entity resolution is useful

Organizations often receive business information from CRM systems, supplier lists, directories, or other internal datasets. The same business can appear under several spellings or addresses. Entity resolution can help with:

- Finding and consolidating duplicate customer or supplier records.
- Connecting records across internal data sources for analytics and reporting.
- Improving data quality by surfacing likely duplicates and inconsistencies.
- Prioritizing records for human review in compliance or risk workflows.

This repository is configured around a three-source business matching challenge. Source 1 is the reference set, and the system looks for zero, one, or multiple matches in Sources 2 and 3. Treat model matches as predictions whose quality should be evaluated on the data and workflow where they will be used.

## How the matching pipeline works

1. **Normalize records:** standardize text and prepare names and addresses for comparison, including supported transliteration and legal suffix handling.
2. **Generate candidates:** use blocking indexes over name and address clues to narrow the possible pairs. This avoids comparing every record against every other record.
3. **Build comparison features:** calculate name and address similarity signals, including token overlap and character-level similarity.
4. **Score candidate pairs:** load the trained LightGBM classifier to estimate whether each pair is a match.
5. **Apply decision thresholds:** use score and ranking logic to produce the final match list while controlling false positives.
6. **Explain and serve results:** expose matching, explanations, data checks, and related capabilities through the API and dashboard.

## Role of the training data

The labeled training dataset supplies examples of business-record pairs and whether each pair is a true match. The training code computes comparison features for those examples and fits a classifier to learn patterns associated with matches and non-matches. For example, the model can learn how name similarity and address agreement together relate to a match.

At inference time, the pipeline applies the same feature process to new candidate pairs and asks the trained model to score them. The repository includes serialized artifacts in `business_entity_resolution/models/`, including `lightgbm.pkl` and model metadata, so the application can load an already trained model. Training data is not itself the prediction engine: the trained model artifact contains the learned parameters used to score new pairs.

Training and matching code is under `business_entity_resolution/src/modeling/` and `business_entity_resolution/src/pipeline/`. The precise data files required to retrain depend on the training script and the challenge dataset available to you.

## Tech stack

| Area | Technologies | Role |
| --- | --- | --- |
| Language and data | Python, Pandas, NumPy | Application code and tabular/numerical data processing |
| Machine learning | scikit-learn, LightGBM | Feature processing, model training, and business-pair classification |
| Text matching | RapidFuzz, Python-Levenshtein, Unidecode | String similarity and text normalization support |
| API | FastAPI, Uvicorn, Pydantic | REST endpoints, HTTP server, and request validation |
| User interface | Streamlit | Interactive web dashboard |
| RAG index | scikit-learn TF-IDF and project code | Search project documents and retrieve relevant passages |
| Testing | pytest, httpx | Automated unit/integration and API tests |
| Deployment | Docker, Docker Compose | Containerized deployment configuration |

## RAG and agentic AI

### RAG knowledge assistant

The RAG component indexes project documentation and reports, retrieves relevant passages for a question, and returns an answer grounded in those sources with citations. In this project its index is TF-IDF based. It is a knowledge assistant for project material; it does not train or replace the business matching model and does not determine whether two businesses are the same.

### Agent coordinator

The agent component provides a coordinator over a set of typed platform tools. The documented tools cover tasks such as schema inspection, data validation, quality reporting, record matching, pair explanation, document queries, and evaluation metrics. The coordinator selects or invokes these capabilities; it is not the classifier trained on the matching dataset.

## Run locally

Open a terminal in the project directory and run:

```powershell
cd business_entity_resolution
python -m pip install -r requirements.txt
```

Start the REST API in one terminal:

```powershell
python -m uvicorn src.api.main:app --host 127.0.0.1 --port 8000
```

Start the dashboard in another terminal:

```powershell
python -m streamlit run src/app/streamlit_app.py --server.address 127.0.0.1 --server.port 8501
```

Then open:

- Dashboard: <http://localhost:8501>
- API documentation: <http://localhost:8000/docs>
- API health endpoint: <http://localhost:8000/health>

To run the test suite from `business_entity_resolution/`:

```powershell
python -m pytest tests/ -v
```

Docker configuration is provided in `business_entity_resolution/docker-compose.yml`.

## Project layout

```text
business_entity_resolution/
├── models/             # Serialized matching models and RAG index
├── reports/            # Dataset and model evaluation reports
├── src/
│   ├── agent/          # Tool coordinator and validated tools
│   ├── api/            # FastAPI application
│   ├── app/            # Streamlit dashboard
│   ├── blocking/       # Candidate generation
│   ├── explainability/ # Pair-level model explanations
│   ├── features/       # Pair comparison features
│   ├── modeling/       # Training and thresholds
│   ├── pipeline/       # Matching pipeline
│   ├── preprocessing/  # Name/address normalization
│   └── rag/            # Document indexing and retrieval QA
├── tests/              # Automated tests
├── requirements.txt
└── docker-compose.yml
```

## Reported evaluation

The project status and reports describe evaluation results for the supplied challenge setup, including a reported macro F0.5 of 0.8798. These are project-reported benchmark figures, not a guarantee of performance on new organizations or datasets. Re-evaluate thresholds and model quality before applying the system to a different data population.

## Further documentation

- Detailed platform and challenge notes: [`business_entity_resolution/README.md`](business_entity_resolution/README.md)
- Current project status: [`PROJECT_STATUS.md`](PROJECT_STATUS.md)
- Challenge data and submission rules: [`student_resource/README.md`](student_resource/README.md)
