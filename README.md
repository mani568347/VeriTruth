# VeriTruth

VeriTruth is an AI-powered news verification web application. Users submit a news
claim as **text**, a **URL**, or a **screenshot/image**, and VeriTruth returns a
**REAL / FAKE verdict** produced by a fine-tuned BERT classifier, supported by
transparent analysis signals and a human-readable AI explanation.

## Features

- **Text verification** — paste a news claim or article directly
- **URL verification** — article text is extracted from the link before analysis
- **Image / screenshot verification** — text is recovered from screenshots with Tesseract OCR
- **BERT-based REAL / FAKE classification** — the fine-tuned transformer is the main classifier
- **Analysis Signals** — source authority, similarity, recency, headline alignment, bias/tone and author signals computed from the submitted content
- **Article-specific AI explanation** — Groq generates a readable explanation of *this* article; it never changes the BERT verdict
- **Feedback** — users mark an analysis correct/incorrect with an optional reason
- **Ask VeriTruth** — a follow-up AI assistant (chat panel) that answers questions about the current analysis in context
- **Authentication** — email sign up / sign in, Google login, logout, and email-based password reset
- **History & reports** — saved analyses per user, plus an admin dashboard

## Tech Stack

| Layer | Technology |
|---|---|
| Backend | Python, Django 5.2 |
| Database | MySQL |
| ML model | PyTorch + Hugging Face Transformers (fine-tuned BERT) |
| OCR | Tesseract OCR (via pytesseract) |
| AI text | Groq API (explanations + assistant only — never classification) |
| Auth | Django auth + django-allauth (Google) + django-axes (lockout) |
| Frontend | HTML, CSS, JavaScript (vanilla) |

## System Flow

**Text**
```
Text → BERT → Result → Signals → AI Explanation
```

**URL**
```
URL → Article Extraction → BERT → Result → Signals → AI Explanation
```

**Image**
```
Image → Tesseract OCR → Extracted Text → BERT → Result → Signals → AI Explanation
```

- **Feedback:** every analysis can be rated correct/incorrect; ratings are stored
  for review in the admin dashboard. Feedback never retrains the model on its own.
- **Ask VeriTruth:** follow-up questions are sent — together with the current
  analysis context (verdict, confidence, signals, explanation) — to Groq, which
  answers in a scrollable chat panel without altering the original prediction.

BERT is always the decision-maker. Supporting signals contribute to the
composite trust index, and Groq only explains and answers questions.

## Prerequisites

- Python 3.10+
- MySQL 8.x running locally
- [Tesseract OCR](https://github.com/UB-Mannheim/tesseract/wiki) installed
  (Windows: add its folder to `PATH` so `tesseract --version` works)
- A Groq API key (free at console.groq.com)
- Optional: Google OAuth credentials for social login

## Setup

### 1. Clone the repository
```bash
git clone <your-repo-url>
cd fake_news
```

### 2. Create and activate a virtual environment

Windows:
```cmd
python -m venv venv
venv\Scripts\activate
```

macOS / Linux:
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Install Tesseract
Download and install from the link above (skip if already installed).

### 5. Create the MySQL database
```sql
CREATE DATABASE veritruth_db CHARACTER SET utf8mb4;
```

### 6. Configure environment
```bash
copy .env.example .env      # Windows
cp .env.example .env        # macOS/Linux
```
Fill in:
- `SECRET_KEY` — generate one: `python -c "from django.core.management.utils import get_random_secret_key; print(get_random_secret_key())"`
- `DB_NAME`, `DB_USER`, `DB_PASSWORD`, `DB_HOST`, `DB_PORT`
- `GROQ_API_KEY`
- `EMAIL_HOST_USER`, `EMAIL_HOST_PASSWORD` (a Gmail **app password** for reset emails)

Google login: register the OAuth client in Google Cloud Console and add it in
Django admin → *Social Applications* (site: `example.com` for local dev), or set
`GOOGLE_CLIENT_ID` / `GOOGLE_CLIENT_SECRET` in `.env`.

### 7. Model setup

Place the fine-tuned BERT checkpoint in:
```
ml/bert_fake_news/best_checkpoint/
├── config.json
├── model.safetensors
├── tokenizer.json
└── tokenizer_config.json
```
The checkpoint is **not stored in Git** (the weights file is far larger than
GitHub's file limit). Copy the `best_checkpoint` folder from an existing
installation, or transfer it with an external drive / cloud storage / Git LFS.

**Not committed on purpose:** `.env`, model weights, training corpora under
`detector/data/`, `media/` uploads, and local database dumps.

### 8. Run migrations
```bash
python manage.py migrate
```

### 9. Create a superuser (optional, for the admin dashboard)
```bash
python manage.py createsuperuser
```

### 10. Run the server
```bash
python manage.py runserver 127.0.0.1:8000
```
Open http://127.0.0.1:8000

## Project Structure

```
fake_news/
├── manage.py
├── fake_news/          # Django project (settings, urls, wsgi)
├── detector/           # Main app: views, models, auth, OCR, Groq integration
│   └── migrations/
├── ml/                 # BERT inference service + model checkpoints
├── templates/          # HTML templates
├── static/             # CSS / JS / images
├── media/              # User uploads (runtime, not committed)
└── requirements.txt
```

## Security Notes

- All credentials (database, Groq, SMTP, OAuth, `SECRET_KEY`) are loaded from
  `.env` — never hard-coded and never committed.
- Login attempts are rate-limited by django-axes (5 failures → temporary lockout).
- Analysis routes require authentication; logged-out visitors see a compact
  sign-in prompt before submitting.

## Disclaimer

VeriTruth is a research/prototype tool. Its verdicts are probabilistic model
predictions, not ground truth. Always verify important claims through multiple
reliable sources.
