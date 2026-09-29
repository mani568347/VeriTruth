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

The BERT weights are **not stored in Git** (the weights file is far larger than
GitHub's 100 MB limit). See **BERT Model Setup** below for how to place or fetch
the checkpoint.

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

## BERT Model Setup

`model.safetensors` is ~437 MB, which exceeds GitHub's 100 MB per-file limit, so
the weights are excluded from Git (`*.safetensors` is in `.gitignore`). No Git LFS
is required: the source lives on GitHub and the model is obtained separately from
the Hugging Face Hub.

Target layout — the app expects exactly these files:
```
ml/bert_fake_news/best_checkpoint/
├── config.json
├── model.safetensors
├── tokenizer.json
└── tokenizer_config.json
```

1. **Get a checkpoint.** Either train locally
   (`python ml/bert_training.py`, which writes
   `ml/bert_fake_news/best_checkpoint/`) or reuse the folder from an existing
   installation — an external drive or cloud storage works too.
2. **Create a Hugging Face model repository** at https://huggingface.co/new,
   e.g. `YOUR_HUGGINGFACE_USERNAME/veritruth-bert-fake-news` (public, so no token
   is needed to download it).
3. **Upload the four checkpoint files above** into the repository root — via the
   web "Add files" UI or:
   ```bash
   hf upload YOUR_HUGGINGFACE_USERNAME/veritruth-bert-fake-news ml/bert_fake_news/best_checkpoint
   ```
4. **Point the project at it.** In `.env`:
   ```
   BERT_MODEL_ID=YOUR_HUGGINGFACE_USERNAME/veritruth-bert-fake-news
   ```
5. **On another machine**, after cloning and installing requirements, fetch the
   model into the checkpoint directory:
   ```bash
   python scripts/download_model.py
   ```
   The script is idempotent: if the checkpoint is already complete it reports
   "Model already available" and downloads nothing.
6. **Loading order.** `ml/bert_service.py` uses the local checkpoint when all
   required files exist; otherwise it loads `BERT_MODEL_ID` from the Hub, which
   downloads once and reuses the local cache afterwards. The model is never
   downloaded per request.
7. **If neither source is available**, analysis stops with a clear message —
   *"VeriTruth BERT model is not available. Please complete the model setup
   described in the README."* No other model is substituted and no placeholder
   verdict is produced.
8. **Private repository (optional).** Set `HF_TOKEN` in the environment;
   `huggingface_hub` picks it up automatically. Never commit a token.

> **Note:** the `veritruth-bert-fake-news` repository is a placeholder name in this
> README and in `.env.example`. The repository has not been created yet — replace
> `YOUR_HUGGINGFACE_USERNAME` once you publish it.

## Project Structure

```
fake_news/
├── manage.py
├── fake_news/          # Django project (settings, urls, wsgi)
├── detector/           # Main app: views, models, auth, OCR, Groq integration
│   └── migrations/
├── ml/                 # BERT inference service + model checkpoints
├── scripts/            # Setup helpers (download_model.py)
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
- Password-reset links only reach accounts whose stored `email` is real and
  deliverable. Django deliberately shows the same "reset link sent" success page
  for an unknown or mistyped address (so the form cannot be used to enumerate
  accounts), so that page is not proof of delivery — confirm against the mailbox
  itself.

## Disclaimer

VeriTruth is a research/prototype tool. Its verdicts are probabilistic model
predictions, not ground truth. Always verify important claims through multiple
reliable sources.
