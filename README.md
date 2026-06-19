# Smart VET - FastAPI + Render + Streamlit Cloud

Projeto para inferencia de sintomas e imagens de pets usando modelos no Hugging Face.

## 1. Estrutura

- `api/` -> API em FastAPI
- `streamlit_app/` -> Interface web em Streamlit
- `render.yaml` -> Blueprint para deploy no Render via Dockerfile
- `.env.example` -> Variaveis de ambiente de referencia

## 2. Ambiente virtual (Windows PowerShell)

```powershell
cd "c:\Projects\POS TECH\Smart_VET"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

## 3. Rodar a API localmente

```powershell
cd api
pip install -r requirements.txt
copy ..\.env.example .env
# Edite o .env com seus modelos do Hugging Face
uvicorn app.main:app --reload
```

Documentacao Swagger: `http://localhost:8000/docs`

### Variaveis obrigatorias para inferencia

- `HF_TOKEN` (se o modelo for privado)
- `HF_SYMPTOMS_MODEL_ID`
- `HF_IMAGE_MODEL_ID`
- `HF_MODEL_REPO` (repositorio com artefato tabular `.pkl`)
- `HF_MODEL_FILENAME` (nome do arquivo `.pkl`)

### Variaveis de triagem clinica (opcionais)

- `TRIAGE_POSITIVE_LABELS` (labels que representam risco, separadas por virgula)
- `TRIAGE_HIGH_THRESHOLD` (limiar para risco alto)
- `TRIAGE_MEDIUM_THRESHOLD` (limiar para risco medio)

## 4. Docker local da API

```powershell
cd api
docker build -t smart-vet-api .
docker run --rm -p 8000:8000 --env-file ..\.env smart-vet-api
```

## 5. Deploy da API no Render (Dockerfile)

1. Suba o projeto para um repositorio GitHub.
2. No Render, selecione `New +` -> `Blueprint`.
3. Conecte o repositorio com o arquivo `render.yaml` na raiz.
4. Configure as env vars no Render:
   - `HF_TOKEN`
   - `HF_SYMPTOMS_MODEL_ID`
   - `HF_IMAGE_MODEL_ID`
  - `HF_MODEL_REPO`
  - `HF_MODEL_FILENAME` (ex.: `best_model.pkl`)
  - `TRIAGE_POSITIVE_LABELS`
  - `TRIAGE_HIGH_THRESHOLD`
  - `TRIAGE_MEDIUM_THRESHOLD`
   - `ALLOWED_ORIGINS` (inclua a URL do Streamlit Cloud)
5. Aguarde o deploy e valide `GET /health`.

## 6. Rodar Streamlit localmente

```powershell
cd streamlit_app
pip install -r requirements.txt
$env:FASTAPI_URL="http://localhost:8000"
streamlit run app.py
```

## 7. Publicar no Streamlit Cloud

1. No Streamlit Cloud, clique em `New app`.
2. Selecione o repositorio e a entrypoint `streamlit_app/app.py`.
3. Em `Advanced settings` -> `Secrets`, adicione:

```toml
FASTAPI_URL = "https://SEU-SERVICO-RENDER.onrender.com"
```

4. Deploy e teste com sintomas + imagem.

## 8. Endpoints principais da API

- `GET /health`
- `POST /predict/symptoms` (JSON com campo `symptoms`)
- `POST /predict/image` (multipart com campo `file`)
- `POST /predict` (multipart com `symptoms` e/ou `image`)
- `POST /predict/condition` (JSON com `animal_name` + `symptoms1..symptoms5`)

As respostas de predicao retornam tambem:

- `top_label`
- `top_score`
- `triage_level` (`alto`, `medio`, `baixo`, `indeterminado`)
- `recommendation`

## 9. Exemplo de chamada cURL

```bash
curl -X POST "http://localhost:8000/predict" \
  -F "symptoms=apatia e falta de apetite" \
  -F "image=@pet.jpg"
```

## 10. Exemplo para modelo tabular (estilo baseline)

```bash
curl -X POST "http://localhost:8000/predict/condition" \
  -H "Content-Type: application/json" \
  -d '{
    "animal_name": "dog",
    "symptoms1": "fever",
    "symptoms2": "diarrhea",
    "symptoms3": "vomiting",
    "symptoms4": "dehydration",
    "symptoms5": "pains"
  }'
```
