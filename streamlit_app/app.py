import os
from io import BytesIO

import requests
import streamlit as st
from PIL import Image

FASTAPI_URL = os.getenv("FASTAPI_URL", "http://localhost:8000")


def diagnosis_hints(detail: str) -> list[str]:
    normalized = (detail or "").lower()
    hints: list[str] = []

    if "hf_model_repo" in normalized:
        hints.append("Defina HF_MODEL_REPO no .env com o repositorio Hugging Face do modelo tabular.")
    if "hf_model_filename" in normalized:
        hints.append("Defina HF_MODEL_FILENAME no .env com o nome do arquivo .pkl (ex.: best_model.pkl).")
    if "hf_symptoms_model_id" in normalized:
        hints.append("Defina HF_SYMPTOMS_MODEL_ID no .env com o modelo de sintomas.")
    if "hf_image_model_id" in normalized:
        hints.append("Defina HF_IMAGE_MODEL_ID no .env com o modelo de imagem.")
    if "401" in normalized or "unauthorized" in normalized or "token" in normalized:
        hints.append("Revise HF_TOKEN no .env. Se o modelo for privado, o token precisa ter permissao de leitura.")
    if "404" in normalized or "not found" in normalized:
        hints.append("Confirme se o repositorio/arquivo do modelo existe e se os nomes estao corretos no .env.")
    if "timed out" in normalized or "timeout" in normalized:
        hints.append("Aumente timeout e valide conectividade de rede com Hugging Face/Render.")
    if "module" in normalized and "app" in normalized and "no module named" in normalized:
        hints.append("Suba a API a partir da pasta api ou use --app-dir api no comando uvicorn.")

    if not hints:
        hints.append("Verifique logs da API no terminal para identificar o erro raiz e validar variaveis do .env.")

    return hints


def render_condition_result(result: dict) -> None:
    prediction_binary = result.get("prediction_binary")
    prediction_label = result.get("prediction_label", "N/A")
    probability_yes = result.get("probability_yes")

    is_high_risk = prediction_binary == 1
    status_text = "Risco Alto" if is_high_risk else "Risco Baixo"
    status_color = "red" if is_high_risk else "green"

    st.markdown("### Resultado Clinico")
    st.markdown(f"**Status:** :{status_color}[{status_text}]")

    col_a, col_b = st.columns(2)
    with col_a:
        st.metric("Classe prevista", prediction_label)
    with col_b:
        probability_text = "N/A" if probability_yes is None else f"{float(probability_yes) * 100:.2f}%"
        st.metric("Probabilidade de perigo", probability_text)

    recommendation = (
        "Encaminhar para avaliacao veterinaria imediata."
        if is_high_risk
        else "Monitorar e manter acompanhamento veterinario."
    )
    st.info(recommendation)

    with st.expander("Ver resposta bruta da API"):
        st.json(result)


def run_api_diagnostics() -> None:
    st.markdown("### Diagnostico da API")

    try:
        health_response = requests.get(f"{FASTAPI_URL}/health", timeout=15)
        health_response.raise_for_status()
        health_payload = health_response.json()
        st.success("Conexao com a API OK")
        st.write(
            {
                "health_status": health_payload.get("status"),
                "api_env": health_payload.get("env"),
                "fastapi_url": FASTAPI_URL,
            }
        )
    except requests.RequestException as exc:
        st.error(f"Falha no endpoint /health: {exc}")
        st.info("Confirme se a API esta rodando e se a FASTAPI_URL esta correta.")
        return

    checks = [
        (
            "POST /predict/symptoms",
            f"{FASTAPI_URL}/predict/symptoms",
            {"symptoms": "fever and loss of appetite"},
        ),
        (
            "POST /predict/condition",
            f"{FASTAPI_URL}/predict/condition",
            {
                "animal_name": "dog",
                "symptoms1": "fever",
                "symptoms2": "diarrhea",
                "symptoms3": "vomiting",
                "symptoms4": "dehydration",
                "symptoms5": "pains",
            },
        ),
    ]

    for label, url, payload in checks:
        try:
            response = requests.post(url, json=payload, timeout=20)
            if response.status_code < 400:
                st.success(f"{label}: OK ({response.status_code})")
            else:
                detail = (
                    response.json().get("detail")
                    if response.headers.get("content-type", "").startswith("application/json")
                    else response.text
                )
                st.warning(f"{label}: erro {response.status_code} - {detail}")
                for hint in diagnosis_hints(str(detail)):
                    st.caption(f"Sugestao: {hint}")
        except requests.RequestException as exc:
            st.error(f"{label}: falha de conexao - {exc}")
            for hint in diagnosis_hints(str(exc)):
                st.caption(f"Sugestao: {hint}")


def render_quick_fix_commands() -> None:
    st.markdown("### Comandos de correcao rapida")
    st.caption("Copie e execute no PowerShell conforme o ambiente local.")

    st.markdown("Iniciar API (forma recomendada)")
    st.code(
        'Set-Location "C:\\Projects\\POS TECH\\Smart_VET\\api"\n'
        '..\\.venv\\Scripts\\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000',
        language="powershell",
    )

    st.markdown("Iniciar API a partir da raiz (alternativa)")
    st.code(
        'Set-Location "C:\\Projects\\POS TECH\\Smart_VET"\n'
        '.\\.venv\\Scripts\\python.exe -m uvicorn app.main:app --app-dir api --host 127.0.0.1 --port 8000',
        language="powershell",
    )

    st.markdown("Iniciar Streamlit")
    st.code(
        'Set-Location "C:\\Projects\\POS TECH\\Smart_VET"\n'
        '.\\.venv\\Scripts\\python.exe -m streamlit run .\\streamlit_app\\app.py',
        language="powershell",
    )

    st.markdown("Exemplo minimo de .env")
    st.code(
        "HF_TOKEN=your_huggingface_token\n"
        "HF_SYMPTOMS_MODEL_ID=your-org/your-symptoms-model\n"
        "HF_IMAGE_MODEL_ID=your-org/your-image-model\n"
        "HF_MODEL_REPO=guicon/techchallenge-animal-condition-model\n"
        "HF_MODEL_FILENAME=best_model.pkl\n"
        "API_TITLE=Tech Challenge Animal Condition Inference API\n"
        "API_DESCRIPTION=API de inferencia que integra modelos Hugging Face para sintomas, imagem e modelo tabular de condicao animal.\n"
        f"ALLOWED_ORIGINS=http://localhost:8501,{FASTAPI_URL}",
        language="ini",
    )

st.set_page_config(page_title="Smart VET", page_icon="🐾", layout="wide")

st.title("Smart VET - Triagem Inteligente")
st.caption("Use os modos abaixo para inferencia multimodal ou tabular via FastAPI + Hugging Face")

with st.expander("Diagnostico rapido de conexao", expanded=False):
    if st.button("Testar conexao com a API"):
        run_api_diagnostics()
    render_quick_fix_commands()

tab_multimodal, tab_condition = st.tabs(["Multimodal (texto + imagem)", "Condicao (modelo tabular)"])

with tab_multimodal:
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Sintomas")
        symptoms_text = st.text_area(
            "Descreva os sintomas do pet",
            height=180,
            placeholder="Ex.: vomito, falta de apetite, apatia ha 2 dias...",
            key="mm_symptoms",
        )

    with col2:
        st.subheader("Imagem")
        uploaded_file = st.file_uploader("Envie uma imagem", type=["png", "jpg", "jpeg"], key="mm_image")
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Imagem enviada", use_container_width=True)

    if st.button("Executar inferencia multimodal", type="primary"):
        if not symptoms_text and uploaded_file is None:
            st.warning("Informe sintomas, imagem, ou ambos.")
        else:
            with st.spinner("Consultando API multimodal..."):
                files = {}
                data = {}

                if symptoms_text:
                    data["symptoms"] = symptoms_text

                if uploaded_file is not None:
                    image_bytes = uploaded_file.getvalue()
                    files["image"] = (
                        uploaded_file.name,
                        BytesIO(image_bytes),
                        uploaded_file.type or "application/octet-stream",
                    )

                try:
                    response = requests.post(
                        f"{FASTAPI_URL}/predict",
                        data=data,
                        files=files if files else None,
                        timeout=60,
                    )
                    response.raise_for_status()
                    result = response.json()

                    st.success("Inferencia multimodal concluida")

                    if result.get("symptoms_result"):
                        st.markdown("### Resultado de Sintomas")
                        st.json(result["symptoms_result"])

                    if result.get("image_result"):
                        st.markdown("### Resultado de Imagem")
                        st.json(result["image_result"])

                except requests.RequestException as exc:
                    st.error(f"Falha ao consultar a API: {exc}")
                    st.info(f"Verifique FASTAPI_URL: {FASTAPI_URL}")

with tab_condition:
    st.subheader("Inferencia de Condicao (5 sintomas)")

    col_animal, col_s1 = st.columns(2)
    with col_animal:
        animal_name = st.text_input("Animal", value="dog", key="cond_animal")
    with col_s1:
        symptoms1 = st.text_input("Sintoma 1", value="fever", key="cond_s1")

    col_s2, col_s3 = st.columns(2)
    with col_s2:
        symptoms2 = st.text_input("Sintoma 2", value="diarrhea", key="cond_s2")
    with col_s3:
        symptoms3 = st.text_input("Sintoma 3", value="vomiting", key="cond_s3")

    col_s4, col_s5 = st.columns(2)
    with col_s4:
        symptoms4 = st.text_input("Sintoma 4", value="dehydration", key="cond_s4")
    with col_s5:
        symptoms5 = st.text_input("Sintoma 5", value="pains", key="cond_s5")

    if st.button("Executar inferencia de condicao", type="secondary"):
        payload = {
            "animal_name": animal_name,
            "symptoms1": symptoms1,
            "symptoms2": symptoms2,
            "symptoms3": symptoms3,
            "symptoms4": symptoms4,
            "symptoms5": symptoms5,
        }

        try:
            response = requests.post(
                f"{FASTAPI_URL}/predict/condition",
                json=payload,
                timeout=60,
            )
            response.raise_for_status()
            result = response.json()

            st.success("Inferencia de condicao concluida")
            render_condition_result(result)

        except requests.RequestException as exc:
            st.error(f"Falha ao consultar a API: {exc}")
            st.info(f"Verifique FASTAPI_URL: {FASTAPI_URL}")

st.divider()
st.caption(f"Endpoint atual da API: {FASTAPI_URL}")
