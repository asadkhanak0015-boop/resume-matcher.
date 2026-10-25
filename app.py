import re
from collections import Counter

import plotly.graph_objects as go
import streamlit as st
from docx import Document
from pypdf import PdfReader
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Resume vs Job Matcher", page_icon="📄", layout="wide")

# ---------------------------------------------------------------- settings
FILLER = {
    "experience", "work", "working", "team", "teams", "ability", "strong",
    "skills", "skill", "required", "preferred", "responsibilities", "role",
    "job", "candidate", "company", "years", "year", "will", "must", "good",
    "excellent", "knowledge", "understanding", "using", "use", "including",
    "etc", "new", "like", "looking", "join", "within", "across", "based",
    "plus", "least", "well", "also", "highly", "need", "needs", "build",
    "building", "help", "ensure", "support", "opportunity", "position",
}
STOP = set(ENGLISH_STOP_WORDS) | FILLER

SKILLS = [
    "python", "java", "javascript", "typescript", "c++", "c#", "sql", "mysql",
    "postgresql", "mongodb", "r", "html", "css", "react", "node.js", "django",
    "flask", "fastapi", "streamlit", "pandas", "numpy", "scikit-learn",
    "tensorflow", "pytorch", "keras", "machine learning", "deep learning",
    "nlp", "computer vision", "data analysis", "data visualization",
    "data science", "power bi", "tableau", "excel", "git", "github", "docker",
    "kubernetes", "aws", "azure", "gcp", "linux", "rest api", "api",
    "statistics", "etl", "spark", "hadoop", "agile", "scrum", "communication",
    "leadership", "problem solving", "teamwork", "project management",
]


# ---------------------------------------------------------------- helpers
def read_file(uploaded) -> str:
    name = uploaded.name.lower()
    if name.endswith(".pdf"):
        reader = PdfReader(uploaded)
        return "\n".join((page.extract_text() or "") for page in reader.pages)
    if name.endswith(".docx"):
        doc = Document(uploaded)
        return "\n".join(p.text for p in doc.paragraphs)
    return uploaded.read().decode("utf-8", errors="ignore")


def clean(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9+#.\s-]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def tokens_of(text: str):
    toks = [t.strip(".-") for t in re.findall(r"[a-z][a-z0-9+#.-]*", text)]
    return [t for t in toks if len(t) > 2 and t not in STOP]


def top_keywords(text: str, n: int = 30):
    counts = Counter(tokens_of(text))
    return [w for w, _ in counts.most_common(n)]


def find_skills(text: str):
    found = set()
    for s in SKILLS:
        pattern = r"(?<![a-z0-9])" + re.escape(s) + r"(?![a-z0-9])"
        if re.search(pattern, text):
            found.add(s)
    return found


def analyze(resume_raw: str, jd_raw: str):
    resume, jd = clean(resume_raw), clean(jd_raw)

    # 1) text similarity (TF-IDF + cosine)
    vec = TfidfVectorizer(ngram_range=(1, 2), stop_words="english")
    matrix = vec.fit_transform([resume, jd])
    cosine = float(cosine_similarity(matrix[0], matrix[1])[0][0])

    # 2) keyword coverage
    jd_keywords = top_keywords(jd, 30)
    resume_tokens = set(tokens_of(resume))
    matched_kw = [k for k in jd_keywords if k in resume_tokens]
    missing_kw = [k for k in jd_keywords if k not in resume_tokens]
    coverage = len(matched_kw) / len(jd_keywords) if jd_keywords else 0

    # 3) skills
    jd_skills = find_skills(jd)
    resume_skills = find_skills(resume)
    matched_sk = sorted(jd_skills & resume_skills)
    missing_sk = sorted(jd_skills - resume_skills)
    skill_cov = len(matched_sk) / len(jd_skills) if jd_skills else None

    # final score: blend of the three signals
    if skill_cov is None:
        score = 0.4 * min(cosine * 2, 1) + 0.6 * coverage
    else:
        score = 0.25 * min(cosine * 2, 1) + 0.35 * coverage + 0.40 * skill_cov

    return {
        "score": round(score * 100),
        "cosine": cosine,
        "coverage": coverage,
        "skill_cov": skill_cov,
        "matched_kw": matched_kw,
        "missing_kw": missing_kw,
        "matched_sk": matched_sk,
        "missing_sk": missing_sk,
    }


def gauge(score: int):
    color = "#e74c3c" if score < 40 else "#f39c12" if score < 70 else "#2ecc71"
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=score,
            number={"suffix": "%"},
            gauge={"axis": {"range": [0, 100]}, "bar": {"color": color}},
        )
    )
    fig.update_layout(height=280, margin=dict(t=20, b=0, l=20, r=20))
    return fig


def report_text(r) -> str:
    lines = [
        "RESUME vs JOB DESCRIPTION REPORT",
        "=" * 36,
        f"Overall match score: {r['score']}%",
        "",
        "Matched skills: " + (", ".join(r["matched_sk"]) or "none"),
        "Missing skills: " + (", ".join(r["missing_sk"]) or "none"),
        "",
        "Matched keywords: " + (", ".join(r["matched_kw"]) or "none"),
        "Missing keywords: " + (", ".join(r["missing_kw"]) or "none"),
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------- UI
st.title("📄 Resume vs Job Description Matcher")
st.caption("Upload your resume, paste a job description, and see how well they match.")

with st.sidebar:
    st.header("How the score works")
    st.markdown(
        "- **Text similarity** (TF-IDF + cosine)\n"
        "- **Keyword coverage** of the job description\n"
        "- **Skill match** against a built-in skills list\n\n"
        "No data is stored. Everything is processed in memory."
    )

left, right = st.columns(2)
with left:
    st.subheader("1. Your resume")
    resume_file = st.file_uploader("Upload PDF, DOCX or TXT", type=["pdf", "docx", "txt"])
with right:
    st.subheader("2. Job description")
    jd_text = st.text_area("Paste the job description here", height=200)

if st.button("Analyze match", type="primary"):
    if not resume_file or not jd_text.strip():
        st.warning("Please upload a resume and paste a job description first.")
    else:
        resume_text = read_file(resume_file)
        if len(resume_text.strip()) < 50:
            st.error(
                "Could not read enough text from the resume. "
                "If it is a scanned PDF, try a text-based PDF or DOCX."
            )
        else:
            r = analyze(resume_text, jd_text)

            st.divider()
            c1, c2 = st.columns([1, 1])
            with c1:
                st.plotly_chart(gauge(r["score"]), use_container_width=True)
            with c2:
                st.metric("Text similarity", f"{r['cosine'] * 100:.0f}%")
                st.metric("Keyword coverage", f"{r['coverage'] * 100:.0f}%")
                if r["skill_cov"] is not None:
                    st.metric("Skill match", f"{r['skill_cov'] * 100:.0f}%")
                if r["score"] >= 70:
                    st.success("Strong match. Small tweaks could make it even better.")
                elif r["score"] >= 40:
                    st.warning("Decent match. Add the missing skills and keywords below.")
                else:
                    st.error("Weak match. Tailor your resume to this job.")

            a, b = st.columns(2)
            with a:
                st.subheader("✅ Matched")
                st.write("**Skills:** " + (", ".join(r["matched_sk"]) or "none found"))
                st.write("**Keywords:** " + (", ".join(r["matched_kw"]) or "none found"))
            with b:
                st.subheader("❌ Missing")
                st.write("**Skills:** " + (", ".join(r["missing_sk"]) or "none, great!"))
                st.write("**Keywords:** " + (", ".join(r["missing_kw"]) or "none, great!"))

            if r["missing_sk"] or r["missing_kw"]:
                st.subheader("💡 Suggestions")
                if r["missing_sk"]:
                    st.write(
                        "Add these skills to your resume if you genuinely have them: "
                        + ", ".join(r["missing_sk"][:8])
                    )
                if r["missing_kw"]:
                    st.write(
                        "Work these job-specific terms into your experience or summary: "
                        + ", ".join(r["missing_kw"][:10])
                    )

            st.download_button(
                "Download report (.txt)",
                report_text(r),
                file_name="match_report.txt",
            )
