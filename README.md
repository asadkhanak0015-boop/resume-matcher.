# Resume vs Job Description Matcher

A Streamlit web app that compares a resume against a job description and shows a match score, matched and missing skills and keywords, and suggestions for improvement.

**Live demo:** _paste your Streamlit link here_

## Features
- Upload a resume as PDF, DOCX or TXT
- Paste any job description
- Match score from 0 to 100 with a gauge chart
- Matched and missing skills and keywords
- Improvement suggestions
- Downloadable text report

## How it works
The score blends three signals:
1. **Text similarity**: TF-IDF vectors compared with cosine similarity (scikit-learn)
2. **Keyword coverage**: share of the job description's top keywords found in the resume
3. **Skill match**: detection of common technical and soft skills in both documents

## Tech stack
Python, Streamlit, scikit-learn, Plotly, pypdf, python-docx

## Run locally
```bash
git clone https://github.com/YOUR-USERNAME/resume-matcher.git
cd resume-matcher
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS / Linux
pip install -r requirements.txt
streamlit run app.py
```

## Limitations
- Scanned (image-only) PDFs are not supported, since there is no OCR
- The skills list is built in and can be extended in `app.py`

## Author
Your Name
