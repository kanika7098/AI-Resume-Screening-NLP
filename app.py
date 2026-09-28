import re, os, json, joblib
import pandas as pd
import streamlit as st

MODEL_PATH = "models/resume_classifier.joblib"

st.set_page_config(
    page_title="AI Resume Screening System",
    page_icon="📄",
    layout="wide"
)

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

model = load_model()

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+|https\S+", " ", text)
    text = re.sub(r"\S+@\S+", " ", text)
    text = re.sub(r"\b\d{10,}\b", " ", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()

def predict(text, top_n=3):
    cleaned = clean_text(text)
    prediction = model.predict([cleaned])[0]
    scores = model.decision_function([cleaned])[0]
    classes = model.named_steps["classifier"].classes_
    ranking = scores.argsort()[::-1][:top_n]
    out = pd.DataFrame({
        "Job Category": classes[ranking],
        "Model Score": scores[ranking]
    })
    # Relative ranking score, not a calibrated probability.
    lo, hi = out["Model Score"].min(), out["Model Score"].max()
    out["Relative Match"] = 100 if hi == lo else (out["Model Score"]-lo)/(hi-lo)*100
    return prediction, out

def extract_uploaded(file):
    name = file.name.lower()
    if name.endswith(".txt"):
        return file.read().decode("utf-8", errors="ignore")
    if name.endswith(".pdf"):
        try:
            from pypdf import PdfReader
            reader = PdfReader(file)
            return "\n".join(page.extract_text() or "" for page in reader.pages)
        except Exception as e:
            st.error(f"Could not read PDF: {e}")
            return ""
    if name.endswith(".docx"):
        try:
            from docx import Document
            doc = Document(file)
            return "\n".join(p.text for p in doc.paragraphs)
        except Exception as e:
            st.error(f"Could not read DOCX: {e}")
            return ""
    st.error("Unsupported file. Please upload TXT, PDF, or DOCX.")
    return ""

st.markdown("""
<style>
.main-title {font-size: 42px; font-weight: 800; margin-bottom: 5px;}
.subtitle {font-size: 18px; opacity: .75; margin-bottom: 25px;}
.card {padding: 20px; border-radius: 15px; border: 1px solid rgba(128,128,128,.25); margin-bottom: 15px;}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">📄 AI Resume Screening System</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">NLP-based resume classification and job-category matching</div>', unsafe_allow_html=True)

with st.sidebar:
    st.header("⚙️ Controls")
    top_n = st.slider("Top categories to display", 1, 5, 3)
    st.info("Educational project: predictions should support, not replace, human recruitment decisions.")

left, right = st.columns([1, 1])

with left:
    st.subheader("1. Upload Resume")
    uploaded = st.file_uploader("Upload TXT, PDF or DOCX", type=["txt","pdf","docx"])
    if uploaded:
        uploaded_text = extract_uploaded(uploaded)
        if uploaded_text:
            st.success(f"Loaded: {uploaded.name}")
            st.session_state["resume_text"] = uploaded_text

with right:
    st.subheader("2. Paste Resume")
    pasted = st.text_area("Paste resume content here", height=250,
                          placeholder="Paste the candidate's resume text...")
    if pasted.strip():
        st.session_state["resume_text"] = pasted

resume_text = st.session_state.get("resume_text", "")

st.divider()

if st.button("🔍 Analyze Resume", type="primary", use_container_width=True):
    if len(resume_text.strip()) < 30:
        st.warning("Please upload or paste a sufficiently detailed resume.")
    else:
        prediction, results = predict(resume_text, top_n)
        st.subheader("📊 Analysis Result")
        st.success(f"Predicted Job Category: **{prediction}**")

        c1, c2, c3 = st.columns(3)
        c1.metric("Characters", f"{len(resume_text):,}")
        c2.metric("Words", f"{len(resume_text.split()):,}")
        c3.metric("Top Match", prediction)

        st.subheader("🏆 Top Matching Categories")
        display = results.copy()
        display["Relative Match"] = display["Relative Match"].round(1).astype(str) + "%"
        st.dataframe(display, use_container_width=True, hide_index=True)

        st.subheader("📝 Extracted Resume Preview")
        st.text_area("Resume text", resume_text[:5000], height=220, disabled=True)

st.caption("Built with Python, Pandas, scikit-learn, TF-IDF, Linear SVM and Streamlit.")
