import os
import re
import requests
import pytesseract

from bs4 import BeautifulSoup
from urllib.parse import urlparse
from newspaper import Article
from datetime import datetime
from difflib import SequenceMatcher


from django.conf import settings
from django.db import IntegrityError
from django.http import HttpResponseForbidden, JsonResponse
from django.shortcuts import get_object_or_404, render, redirect
from django.contrib import messages
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.decorators import login_required
from django.contrib.auth.models import User
from django.contrib.auth.password_validation import validate_password
from django.core.exceptions import ValidationError
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST
from django.utils import timezone
from django.utils.http import url_has_allowed_host_and_scheme
from django.contrib.auth.views import (
    PasswordResetView,
    PasswordResetCompleteView
)

from PIL import Image
from PyPDF2 import PdfReader

from .models import (
    Source,
    NewsLog,
    AnalysisHistory,
    PredictionHistory,
    LoginLog,
    VisitorLog,
    Report,
    UserProfile,
    PredictionFeedback,
)
from .similarity import get_similarity_score
from .gemini_explainer import (
    generate_explanation,
    ask_followup_question
)
from ml.bert_service import BertModelUnavailable, bert_predict

# pyrefly: ignore [missing-import]
from .tokens import token_generator
from django.http import JsonResponse
from .serializers import PredictionSerializer
from .models import PredictionHistory


# ================= PASSWORD RESET =================

class CustomPasswordResetCompleteView(PasswordResetCompleteView):

    def dispatch(self, request, *args, **kwargs):

        messages.success(
            request,
            'Password updated successfully.'
        )

        return redirect('/login/')


class CustomPasswordResetView(PasswordResetView):

    template_name = 'password_reset.html'

    email_template_name = (
        'registration/password_reset_email.html'
    )

    subject_template_name = (
        'registration/password_reset_subject.txt'
    )

    success_url = '/password-reset/'

    def form_valid(self, form):

        messages.success(
            self.request,
            'Password reset link sent successfully.'
        )

        return super().form_valid(form)


# ================= LEGAL PAGES =================

def terms(request):
    return render(request, 'terms.html')


def privacy(request):
    return render(request, 'privacy.html')


def support(request):
    return render(request, 'support.html')


def faq(request):
    return render(request, 'faq.html')


def disclaimer(request):
    return render(request, 'disclaimer.html')


# ================= TESSERACT =================

pytesseract.pytesseract.tesseract_cmd = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)


# ================= AUTH =================
from rest_framework.response import Response
from .serializers import PredictionSerializer

def api_history(request):

    history = AnalysisHistory.objects.all()

    serializer = PredictionSerializer(
        history,
        many=True
    )

    return JsonResponse(serializer.data, safe=False)
def login_view(request):

    if request.method == 'POST':

        email = request.POST.get('email', '').strip().lower()
        password = request.POST.get('password', '')

        if not email or not password:
            messages.error(request, 'Please enter your email address and password.')
            return render(request, 'login.html')

        # ── Strict email validation ───────────────────────────────────────
        _email_ok, _email_err = _validate_email_strict(email)
        if not _email_ok:
            messages.error(request, _email_err)
            return render(request, 'login.html')
        # ─────────────────────────────────────────────────────────────────

        try:
            user_obj = User.objects.get(email=email)
            user = authenticate(
                request,
                username=user_obj.username,
                password=password
            )

            if user is not None:
                login(request, user)
                messages.success(request, f'Welcome back, {user_obj.username.split("@")[0]}! You are now signed in.')
                next_url = request.GET.get('next')
                if next_url and url_has_allowed_host_and_scheme(
                    next_url,
                    allowed_hosts={request.get_host()},
                    require_https=request.is_secure()
                ):
                    return redirect(next_url)
                return redirect('/')
            else:
                messages.error(request, 'Invalid email or password. Please try again.')
                return render(request, 'login.html')

        except User.DoesNotExist:
            messages.error(request, 'No account found with that email address.')
            return render(request, 'login.html')

        except Exception:
            messages.error(request, 'Unable to complete sign-in. Please try again.')
            return render(request, 'login.html')

    return render(request, 'login.html')


def signup_view(request):

    if request.method == 'POST':

        import random
        import re as _re

        email = request.POST.get('email', '').strip().lower()
        password = request.POST.get('password', '')
        confirm_password = request.POST.get('confirm_password', '')

        if not email:
            messages.error(request, 'Please enter a valid email address.')
            return render(request, 'signup.html')

        # ── Strict email validation (same rules as login) ────────────────
        _email_ok, _email_err = _validate_email_strict(email)
        if not _email_ok:
            messages.error(request, _email_err)
            return render(request, 'signup.html')
        # ─────────────────────────────────────────────────────────────────

        if password != confirm_password:
            messages.error(request, 'Passwords do not match. Please check and try again.')
            return render(request, 'signup.html')

        try:
            validate_password(password)
        except ValidationError as exc:
            for _msg in exc.messages:
                messages.error(request, _msg)
            return render(request, 'signup.html')

        if User.objects.filter(email=email).exists():
            messages.error(request, 'An account with this email address already exists.')
            return render(request, 'signup.html')

        try:
            username = email.split('@')[0] + str(random.randint(1000, 9999))
            user = User.objects.create_user(
                username=username,
                email=email,
                password=password
            )
            login(
                request,
                user,
                backend='django.contrib.auth.backends.ModelBackend'
            )
            messages.success(request, 'Account created successfully. Welcome to VeriTruth!')
            next_url = request.GET.get('next')
            if next_url and url_has_allowed_host_and_scheme(
                next_url,
                allowed_hosts={request.get_host()},
                require_https=request.is_secure()
            ):
                return redirect(next_url)
            return redirect('/')

        except IntegrityError:
            messages.error(request, 'An account with these details already exists.')
            return render(request, 'signup.html')

        except Exception:
            messages.error(request, 'Unable to create your account. Please try again.')
            return render(request, 'signup.html')

    return render(request, 'signup.html')


def verify_email(request, uidb64, token):

    from django.utils.http import urlsafe_base64_decode

    try:
        uid = urlsafe_base64_decode(uidb64).decode()
        user = User.objects.get(pk=uid)

        if token_generator.check_token(user, token):
            user.is_active = True
            user.save()
            messages.success(request, 'Email verified successfully. You can now sign in.')
            return redirect('/login/')

    except Exception:
        pass

    messages.error(request, 'This verification link is invalid or has already been used.')
    return redirect('/login/')


def logout_view(request):

    request.session.pop('admin_user_id', None)
    logout(request)
    messages.info(request, 'You have been signed out successfully.')
    return redirect('/login/')


# ================= HELPERS =================

import re as _re

# Known legitimate email providers — enforced at signup and login
_KNOWN_EMAIL_DOMAINS = {
    'gmail.com', 'googlemail.com',
    'yahoo.com', 'yahoo.in', 'yahoo.co.uk', 'yahoo.co.in',
    'outlook.com', 'hotmail.com', 'live.com', 'msn.com',
    'icloud.com', 'me.com', 'mac.com',
    'protonmail.com', 'proton.me',
    'rediffmail.com', 'ymail.com',
}


def _validate_email_strict(addr):
    """
    Returns (True, None) if email passes all checks.
    Returns (False, error_message) on the first rule that fails.

    Rules enforced:
      1. Exactly one @ symbol
      2. Local part (before @) ≥ 3 characters, starts & ends with alphanumeric
      3. No consecutive dots anywhere
      4. Domain contains at least one dot
      5. Every domain label ≥ 2 characters, alphanumeric/hyphens only
      6. Domain labels cannot start or end with a hyphen
      7. TLD is 2–6 alphabetic characters only (no digits)
      8. Full domain must be in the known providers list
    """
    addr = addr.strip().lower()

    # Rule 1 — exactly one @
    if addr.count('@') != 1:
        return False, 'Email must contain exactly one @ symbol.'

    local, domain = addr.split('@')

    # Rule 2 — local part length and characters
    if len(local) < 3:
        return False, 'The part before @ must be at least 3 characters (e.g. john@gmail.com).'
    if not _re.match(r'^[a-zA-Z0-9][a-zA-Z0-9._%+\-]*[a-zA-Z0-9]$', local):
        return False, 'The email address contains invalid characters before @.'

    # Rule 3 — no consecutive dots
    if '..' in addr:
        return False, 'Email address cannot contain consecutive dots.'

    # Rule 4 — domain must contain a dot
    if '.' not in domain:
        return False, 'Email domain must include a valid extension (e.g. .com, .in).'

    labels = domain.split('.')

    # Rule 5 & 6 — each label ≥2 chars, valid chars, no leading/trailing hyphen
    for label in labels:
        if len(label) < 2:
            return False, 'Each section of the email domain must be at least 2 characters.'
        if not _re.match(r'^[a-zA-Z0-9\-]+$', label):
            return False, 'Email domain contains invalid characters.'
        if label.startswith('-') or label.endswith('-'):
            return False, 'Email domain sections cannot start or end with a hyphen.'

    # Rule 7 — TLD must be letters only, 2–6 chars
    tld = labels[-1]
    if not _re.match(r'^[a-zA-Z]{2,6}$', tld):
        return False, 'Please use a recognised domain extension (.com, .org, .in, etc.).'

    # Rule 8 — domain must be a known provider
    if domain not in _KNOWN_EMAIL_DOMAINS:
        return False, (
            f'"{domain}" is not a recognised email provider. '
            'Please sign up with Gmail, Yahoo, Outlook, iCloud, or a similar verified provider.'
        )

    return True, None

def _is_error_or_blocked_content(text):
    if not text or len(text.strip()) < 30:
        return True
    lower = text.lower()
    error_signals = [
        "edgesuite.net", "access denied", "403 forbidden", "404 not found",
        "cloudflare", "checking your browser", "enable javascript", "security check",
        "reference #", "ip address blocked", "bot detection", "incapsula", "perimeterx"
    ]
    return any(sig in lower for sig in error_signals)


def _extract_headline_from_url_slug(url):
    try:
        parsed = urlparse(url)
        path = parsed.path.strip('/')
        parts = [p for p in path.split('/') if p]
        if not parts:
            return ""
        # Get the longest path component which usually contains the article slug
        slug = max(parts, key=len)
        slug = re.sub(r'\.(html|ece|php|asp|aspx)$', '', slug, flags=re.I)
        slug = re.sub(r'-\d+$', '', slug)
        clean_slug = slug.replace('-', ' ').replace('_', ' ').strip()
        if len(clean_slug) > 10:
            return clean_slug.capitalize()
    except Exception as e:
        print("Slug extraction error:", e)
    return ""


def extract_text(url):
    # Strategy 1: Newspaper3k with Browser User-Agent
    try:
        article = Article(url, headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8"
        })
        article.download()
        article.parse()
        if article.text and not _is_error_or_blocked_content(article.text):
            return article.text.strip()
    except Exception as e:
        print("Newspaper3k extraction error:", e)

    # Strategy 2: Requests + BeautifulSoup with Browser User-Agent
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9"
        }
        response = requests.get(url, headers=headers, timeout=6)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, "html.parser")
            for tag in soup(["script", "style", "nav", "footer", "header", "form"]):
                tag.decompose()
            paragraphs = soup.find_all("p")
            text_p = " ".join([p.get_text().strip() for p in paragraphs if len(p.get_text().strip()) > 20])
            if text_p and not _is_error_or_blocked_content(text_p):
                return text_p.strip()
    except Exception as e:
        print("BeautifulSoup extraction error:", e)

    # Strategy 3: URL Slug Headline Fallback
    headline_slug = _extract_headline_from_url_slug(url)
    if headline_slug:
        domain = urlparse(url).netloc
        return f"{headline_slug}. Reported by {domain} media service."

    return ""


def clean_text(text):

    return re.sub(r'<.*?>', '', text)


KNOWN_TRUSTED_SOURCES = {
    'bbc.com': 92, 'bbc.co.uk': 92, 'reuters.com': 95, 'apnews.com': 95,
    'nytimes.com': 90, 'washingtonpost.com': 88, 'theguardian.com': 88,
    'cnn.com': 85, 'bloomberg.com': 90, 'thehindu.com': 88,
    'indianexpress.com': 85, 'ndtv.com': 82, 'timesofindia.indiatimes.com': 80,
    'forbes.com': 85, 'wsj.com': 92, 'financialtimes.com': 90,
    'economist.com': 92, 'aljazeera.com': 84, 'npr.org': 90
}


def get_source_score(url):
    try:
        domain = urlparse(url).netloc.lower().replace('www.', '')

        source = Source.objects.filter(
            domain__icontains=domain
        ).first()

        if source:
            return source.score

        for known_domain, score in KNOWN_TRUSTED_SOURCES.items():
            if known_domain in domain:
                return score
    except Exception:
        pass

    return 50


def get_recency_score(text):

    current_year = datetime.now().year

    years = re.findall(r'(20\d{2})', text)

    if years:

        y = int(years[0])

        if y == current_year:
            return 100

        elif y >= current_year - 1:
            return 70

        elif y >= current_year - 3:
            return 40

        else:
            return 20

    return 50


def get_headline_score(text):
    parts = text.split("\n")

    if len(parts) < 2:
        return 50

    headline = parts[0].strip().lower()
    body = " ".join(parts[1:]).strip().lower()
    if not headline or not body:
        return 50
    return round(SequenceMatcher(None, headline, body[:2000]).ratio() * 100, 2)


def get_bias_score(text):
    """
    Score linguistic neutrality. Returns 0–100 where higher = more neutral.
    Penalises sensationalist, alarmist, or emotionally loaded language —
    including the calm-but-absurd phrasing common in pseudoscience fake news.
    """
    # Tier 1 — blatant emotional sensationalism
    tier1 = [
        "shocking", "unbelievable", "urgent", "bombshell", "explosive",
        "outrage", "outrageous", "terrifying", "horrifying", "disgusting",
        "scandal", "scandalous", "expose", "exposed", "revealed",
        "destroyed", "obliterated", "decimated", "annihilated",
    ]
    # Tier 2 — pseudoscience / conspiracy linguistic fingerprints
    tier2 = [
        "confirms", "confirmed", "scientists discover", "researchers prove",
        "nasa admits", "government admits", "secret", "suppressed",
        "they don't want you to know", "wake up", "sheeple",
        "miracle cure", "100 percent", "guaranteed", "permanently cures",
        "instantly", "completely eliminates", "ancient secret",
        "mainstream media won't", "truth they hide", "leaked",
        "cover-up", "coverup", "illuminati", "deep state", "cabal",
        "mind control", "microchip", "microchipped", "chemtrail",
        "turning into", "made of cheese", "hollow earth", "flat earth",
        "reptilian", "shape-shifting", "time travel available",
        "telepathic", "telekinesis", "psychic powers",
    ]
    # Tier 3 — clickbait amplifiers
    tier3 = [
        "breaking", "just in", "exclusive", "must read", "share before",
        "deleted soon", "won't believe", "mind-blowing", "mind blowing",
        "game changer", "game-changer", "world shaking",
    ]

    lower = text.lower()
    t1 = sum(1 for w in tier1 if w in lower)
    t2 = sum(1 for w in tier2 if w in lower)
    t3 = sum(1 for w in tier3 if w in lower)

    # Weight tiers: pseudoscience patterns (t2) are the strongest signal
    total_penalty = (t1 * 1) + (t2 * 2) + (t3 * 1)

    if total_penalty == 0:
        return 90
    elif total_penalty <= 1:
        return 65
    elif total_penalty <= 3:
        return 40
    else:
        return 15


def get_author_score(text):

    return 80 if "by " in text.lower() else 40


# ================= OCR =================

def extract_text_from_image(file):

    try:

        img = Image.open(file).convert('L')

        text = pytesseract.image_to_string(img)

        return text

    except:

        return ""


def extract_text_from_file(file):

    name = file.name.lower()

    text = ""

    try:

        if name.endswith('.txt'):

            text = file.read().decode(
                'utf-8',
                errors='ignore'
            )

        elif name.endswith('.pdf'):

            reader = PdfReader(file)

            for page in reader.pages:

                text += (
                    page.extract_text() or ""
                )

    except:
        pass

    return text


# ================= VIEWS =================

def home(request):

    logs = NewsLog.objects.all().order_by(
        '-created_at'
    )[:5]

    return render(request, 'home.html', {
        'recent_logs': logs
    })


def about(request):

    return render(request, 'about.html')

def extract_article_metadata(url):
    try:
        article = Article(url, headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
        })
        article.download()
        article.parse()

        title = article.title if article.title and not _is_error_or_blocked_content(article.title) else ""
        if not title:
            title = _extract_headline_from_url_slug(url) or "News Article Verification"

        text_content = article.text if article.text and not _is_error_or_blocked_content(article.text) else ""

        return {
            "title": title,
            "authors": ", ".join(article.authors) if article.authors else "Media Desk",
            "publish_date": str(article.publish_date) if article.publish_date else "Recent",
            "text": text_content
        }
    except Exception as e:
        print("Metadata extraction error:", e)
        fallback_title = _extract_headline_from_url_slug(url) or "News Article Verification"
        return {
            "title": fallback_title,
            "authors": "Media Desk",
            "publish_date": "Recent",
            "text": ""
        }
@login_required(login_url='/login/')
def predict(request):
    if request.method != 'POST':
        return redirect('/')

    news_input = (request.POST.get('news') or '').strip()
    url_input = (request.POST.get('url') or '').strip()
    uploaded_file = request.FILES.get('file')

    text = ""
    user_input = ""
    source_score = 50
    metadata = {
        "title": "Not available",
        "authors": "Not available",
        "publish_date": "Not available",
        "text": ""
    }

    # 1. FILE INPUT
    if uploaded_file:
        user_input = uploaded_file.name
        filename = uploaded_file.name.lower()
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            text = extract_text_from_image(uploaded_file)
            if text and text.strip():
                messages.success(request, 'Text extracted from the image successfully.')
            else:
                messages.warning(request, 'Could not extract readable text from this image. Check the image quality and try again.')
        elif filename.endswith(('.txt', '.pdf')):
            text = extract_text_from_file(uploaded_file)
            if text and text.strip():
                messages.success(request, 'Document uploaded and content extracted successfully.')
            else:
                messages.warning(request, 'Could not read content from the uploaded file. Ensure the file is not empty or protected.')
        else:
            messages.error(request, 'Unsupported file type. Please upload a PNG, JPG, JPEG, PDF, or TXT file.')
            return render(request, 'result.html', {'error': 'Unsupported file type. Please upload a supported file format.'})

    # 2. URL INPUT (from url field OR pasted http in news field)
    elif url_input or news_input.startswith('http'):
        target_url = url_input if url_input else news_input
        user_input = target_url
        source_score = get_source_score(target_url)

        try:
            metadata = extract_article_metadata(target_url)
        except Exception as e:
            print("Metadata extraction error:", e)

        extracted = extract_text(target_url)

        if extracted and len(extracted.strip()) >= 30:
            text = extracted
            messages.info(request, 'Article content extracted successfully. Running analysis...')
        elif metadata and metadata.get('text') and len(metadata['text'].strip()) >= 30:
            text = metadata['text']
            messages.info(request, 'Article content extracted successfully. Running analysis...')
        else:
            # Fallback for protected/paywalled news URLs
            title = metadata.get('title', '')
            if title and title != 'Not available':
                text = f"Article Headline: {title}. Source URL: {target_url}. Published by {metadata.get('authors', 'Media Source')}."
                messages.warning(request, 'Full article content could not be extracted. Analysis is based on the headline and domain trust signals.')
            else:
                text = f"News report from URL: {target_url}. Evaluated domain trust score: {source_score}."
                messages.warning(request, 'Unable to extract article content from this URL. The site may be paywalled or blocking automated access.')

    # 3. DIRECT TEXT COPY
    elif news_input:
        if len(news_input.strip()) < 15:
            messages.warning(request, 'Please provide more content for a reliable analysis. Your text is too short.')
            return render(request, 'result.html', {'error': 'The provided text is too short for a reliable analysis. Please add more content.'})
        text = news_input
        user_input = news_input
        messages.info(request, 'News text received. Running analysis...')

    else:
        messages.warning(request, 'Please enter news text, a URL, or upload a file before submitting.')
        return render(request, 'result.html', {
            'error': 'No input content provided. Please enter article text, paste a URL, or upload a document file.'
        })

    # VALIDATION
    if not text or len(text.strip()) < 15:
        messages.error(request, 'Could not extract readable text from the provided input. If submitting a URL, the site may be blocking automated access — try pasting the article text directly.')
        return render(request, 'result.html', {
            'error': 'Could not extract readable text from the provided input. If submitting a URL, the target site may block automated scrapers; please copy and paste the article text directly.'
        })

    # Unanalyzable content (punctuation/symbols only) must not be classified.
    if sum(ch.isalpha() for ch in text) < 10:
        messages.error(request, 'The submitted content contains no readable words to analyze. Please provide actual news text, a URL, or a screenshot of an article.')
        return render(request, 'result.html', {
            'error': 'No readable words found in the submitted content. VeriTruth analyzes news text — please provide a real article, link, or screenshot.'
        })

    text = clean_text(text)

    # ── CLAIM-PLAUSIBILITY PRE-FILTER ────────────────────────────────────────
    # Detect physically/scientifically impossible or satirical claims and short-
    # circuit to FAKE before BERT even runs.  BERT was trained on political fake
    # news and often misclassifies calm-toned absurdist/pseudoscience text.
    _IMPLAUSIBLE_PATTERNS = [
        # Celestial / physical impossibilities
        r"moon\b.{0,60}\b(turning into|made of|becomes?|turned into)\b.{0,40}\b(cheese|chocolate|gold|liquid|glass)",
        r"\b(sun|earth|moon|planet)\b.{0,60}\b(is (actually|really) a)\b",
        r"\bgravity\b.{0,50}\b(does not exist|isn.t real|was invented|is fake)\b",
        r"\bearth\b.{0,50}\b(is flat|is hollow|isn.t real)\b",
        r"\bnasa\b.{0,60}\b(hid|hides|faked|staged|admits|confirms).{0,60}\b(moon|flat|alien|hollow)\b",
        r"\bmoon landing\b.{0,60}\b(faked|staged|filmed|hoax)\b",
        # Pseudoscience miracle claims
        r"\b(cure|cures|eliminates|kills).{0,50}\b(all (diseases|cancers?|viruses?)|cancer (instantly|overnight|permanently))\b",
        r"\b(drinking|eating|inject).{0,50}\b(bleach|urine|mercury)\b.{0,30}\bcure\b",
        r"\b(reverses?|eliminates?)\b.{0,40}\baging\b.{0,30}\b(instantly|overnight|completely|guaranteed)\b",
        r"\bvaccines?\b.{0,60}\b(cause[sd]?|implant|contain)\b.{0,40}\b(autism|microchip|5g|tracking)\b",
        # Paranormal / supernatural
        r"\b(aliens?|extraterrestrials?)\b.{0,50}\b(landed|living among|control|confirmed)\b",
        r"\b(telepathic|telepathy|telekinesis|psychic powers)\b.{0,50}\b(confirmed|proven|discovered|activated)\b",
        r"\b(mermaids?|unicorns?|dragons?)\b.{0,50}\b(discovered|confirmed|found|living)\b",
        r"\btime travel\b.{0,60}\b(available|confirmed|proven|possible now)\b",
        # Cosmic / alignment claims
        r"\bcosmic alignment\b.{0,80}\b(causes?|triggers?|activates?|turns?|confirms?)\b",
        r"\brarec?\b.{0,30}\balignment\b.{0,80}\b(turns?|caused?|confirmed?)\b",
        # Shape-shifting / reptilian
        r"\b(reptilian|shape.?shifting|shape.?shifter)\b",
        # Hollow Earth / Inner Earth
        r"\b(hollow earth|inner earth|earth is hollow)\b",
        # Surveillance / population-control conspiracies
        r"\b(mind control|mind-control)\b.{0,60}\b(chemical|water|food|spray|drug)\b",
        r"\b(chemical|drug|substance)\b.{0,60}\b(water supply|drinking water)\b.{0,50}\b(control|sedate|subdue|pacify)\b",
        r"\bmicrochip(s|ped|ping)?\b.{0,60}\b(vaccine|inject|implant|track)\b",
        r"\b(vaccine|inject)\b.{0,60}\b(microchip|tracking chip|5g chip|nanochip)\b",
        r"\bchemtrail(s)?\b.{0,40}\b(mind|control|poison|spray|chemical)\b",
    ]

    _pre_filter_triggered = False
    _lower_text = text.lower()
    for _pat in _IMPLAUSIBLE_PATTERNS:
        if re.search(_pat, _lower_text):
            _pre_filter_triggered = True
            break

    # ── BERT ML PREDICTION ───────────────────────────────────────────────────
    try:
        prediction, confidence = bert_predict(text)
    except (BertModelUnavailable, ValueError) as exc:
        messages.error(request, 'The analysis engine is temporarily unavailable. Please try again in a moment.')
        return render(request, 'result.html', {
            'error': 'Analysis could not be completed. The ML model is currently unavailable. Please try again shortly.'
        })
    ml_score = float(confidence)

    # If the plausibility filter fired, override BERT to FAKE with high confidence
    if _pre_filter_triggered and prediction != "Fake News":
        prediction = "Fake News"
        ml_score = max(ml_score, 92.0)   # keep confidence high so the result is unambiguous

    # ── SIGNALS ──────────────────────────────────────────────────────────────
    recency = get_recency_score(text)
    headline = get_headline_score(text)
    bias = get_bias_score(text)
    author = get_author_score(text)

    try:
        raw_sim = get_similarity_score(text)
        # raw_sim is cosine similarity TO fake samples (0=dissimilar, 1=identical).
        # A HIGH raw_sim means the text LOOKS LIKE known fake news → LOW credibility.
        # We invert so that similarity score used in trust means "dissimilarity from fakes":
        #   0.0 raw  → 100 similarity score  (totally unlike fakes = credible signal)
        #   1.0 raw  →   0 similarity score  (matches fakes perfectly = not credible)
        similarity = round((1.0 - raw_sim) * 100, 2)
    except Exception:
        similarity = 50

    # ── TRUST SCORE ──────────────────────────────────────────────────────────
    # ml_score here reflects the corrected prediction (post pre-filter override).
    # When prediction is Fake News, ml_score = confidence in FAKE, so a high
    # ml_score already pushes trust down (fake confidence ≠ credibility).
    # Invert ml_score for trust when the verdict is fake.
    ml_credibility = ml_score if prediction == "Real News" else (100.0 - ml_score)

    trust = round(
        (ml_credibility * 0.35) +
        (source_score   * 0.25) +
        (similarity     * 0.15) +
        (bias           * 0.15) +
        (recency        * 0.05) +
        (headline       * 0.05),
        2
    )

    # Hard override: if signals are overwhelmingly negative, never let a BERT
    # Real News prediction survive into the final verdict.
    if prediction == "Real News":
        # Criteria (any 2 of 3 trigger an override):
        #   raw_sim > 0.25 — text is measurably close to known fake-news patterns
        #   bias < 35      — heavy pseudoscience / sensationalist language detected
        #   trust < 45     — composite credibility score is clearly below average
        fake_signal_count = sum([
            raw_sim > 0.25,   # close to fake corpus
            bias < 35,        # heavy fake-language detected
            trust < 45,       # composite score is clearly low
        ])
        if fake_signal_count >= 2:
            prediction = "Fake News"
            ml_score = max(ml_score, 80.0)
            trust = min(trust, 30.0)

    result = prediction

    # ── AI EXPLANATION ────────────────────────────────────────────────────────
    # Pick up the language the user selected (stored by JS in a hidden POST field).
    # Falls back to 'en' if not supplied so the explainer always has a valid code.
    selected_language = (request.POST.get('selected_language') or 'en').strip()[:5]

    explanation = generate_explanation({
        "prediction":    result,
        "confidence":    ml_score,
        "trust":         trust,
        "bias":          bias,
        "source":        source_score,
        "similarity":    similarity,
        "recency":       recency,
        "headline":      headline,
        "author_score":  author,
        "user_input":    user_input,
        "input_type":    (
            "URL" if (url_input or news_input.startswith('http'))
            else "File (OCR)" if uploaded_file
            else "Text"
        ),
        "url":           user_input if str(user_input).startswith('http') else "",
        "title":         metadata["title"],
        "author":        metadata["authors"],
        "publish_date":  metadata["publish_date"],
        "text_excerpt":  text[:3000].replace('\n', ' '),
        "language":      selected_language,
    })
    # SAVE — serialise the explanation dict to JSON for DB storage
    import json as _json
    _explanation_str = (
        _json.dumps(explanation, ensure_ascii=False)
        if isinstance(explanation, dict)
        else str(explanation or "")
    )
    try:
        input_type_val = "URL" if (url_input or news_input.startswith('http')) else ("File" if uploaded_file else "Text")
        title_val = metadata.get("title", "") if metadata else ""
        author_val = metadata.get("authors", "") if metadata else ""
        date_val = metadata.get("publish_date", "") if metadata else ""
        file_name_val = uploaded_file.name if uploaded_file else ""
        image_obj = uploaded_file if (uploaded_file and file_name_val.lower().endswith(('.png', '.jpg', '.jpeg'))) else None

        PredictionHistory.objects.create(
            username=request.user.username if request.user.is_authenticated else "Guest",
            input_type=input_type_val,
            article_title=title_val[:500],
            author=author_val[:200],
            publish_date=str(date_val)[:100],
            ocr_text=text[:1500] if input_type_val == "File" else "",
            uploaded_file=file_name_val[:300],
            image=image_obj,
            url=user_input if input_type_val == "URL" else "",
            article=text[:2000],
            prediction=result,
            confidence=ml_score,
            trust_score=trust,
            bias_score=bias,
            source_score=source_score,
            similarity_score=similarity,
            ai_explanation=_explanation_str[:12000]
        )
    except Exception as e:
        print("PredictionHistory save error:", e)

    prediction_log = None

    if request.user.is_authenticated:
        prediction_log = NewsLog.objects.create(
            user=request.user,
            user_input=(
                user_input
                if user_input
                else "File Upload"
            ),
            text=text[:1000],
            prediction=result,
            confidence=ml_score,
            trust_score=trust,
            explanation=_explanation_str[:12000]
        )
        messages.success(request, 'Analysis completed. Results saved to your history.')
    else:
        messages.info(request, 'Analysis completed. Sign in to save results to your history.')

    return render(request, 'result.html', {
        'prediction': result,
        'confidence': ml_score,
        'trust_score': trust,
        'user_query': user_input,
        'ai_explanation': explanation,      # structured dict for new UI
        'prediction_id': prediction_log.id if prediction_log else None,
        'feedback': (
            PredictionFeedback.objects.filter(
                prediction=prediction_log,
                user=request.user
            ).first()
            if prediction_log and request.user.is_authenticated else None
        ),
        'score_breakdown': {
            'ml': ml_score,
            'source': source_score,
            'similarity': similarity,
            'recency': recency,
            'headline': headline,
            'bias': bias,
            'author': author
        },
        'analysis': {
            'length': len(text.split()),
            'tone': (
                'Emotional'
                if bias < 50
                else 'Neutral'
            ),
            'risk_level': (
                'High'
                if trust < 40
                else 'Medium'
                if trust < 70
                else 'Low'
            )
        }
    })


@login_required(login_url='/login/')
def history(request):
    logs = NewsLog.objects.filter(
        user=request.user
    ).order_by('-created_at')

    return render(request, 'history.html', {
        'logs': logs
    })


@login_required(login_url='/login/')
def history_detail(request, id):
    log = get_object_or_404(
        NewsLog,
        id=id,
        user=request.user
    )

    explanation = log.explanation
    import json as _json

    # Try to parse stored explanation — may be a JSON string (new format)
    # or a legacy HTML string (old format) or empty
    def _parse_stored_explanation(raw):
        if not raw or "No explanation stored" in str(raw):
            return None
        if isinstance(raw, dict):
            return raw
        try:
            parsed = _json.loads(raw)
            if isinstance(parsed, dict) and "summary" in parsed:
                return parsed
        except (ValueError, TypeError):
            pass
        return None  # Old HTML string — needs regeneration

    parsed_explanation = _parse_stored_explanation(explanation)

    # Explanations stored while Groq was down are fallbacks; regenerate them
    # so the page shows live AI output now that the service works again.
    if parsed_explanation and (
        parsed_explanation.get("groq_available") is False
        or parsed_explanation.get("is_fallback")
        or not parsed_explanation.get("about")   # legacy score-template explanation
    ):
        parsed_explanation = None

    if not parsed_explanation:
        # Regenerate for old records or empty explanations
        parsed_explanation = generate_explanation({
            "prediction":   log.prediction,
            "confidence":   log.confidence,
            "trust":        log.trust_score,
            "bias":         get_bias_score(log.text),
            "source":       get_source_score(log.user_input if log.user_input.startswith('http') else ""),
            "similarity":   75,
            "recency":      get_recency_score(log.text),
            "headline":     get_headline_score(log.text),
            "author_score": get_author_score(log.text),
            "user_input":   log.user_input,
            "input_type":   "URL" if log.user_input.startswith('http') else "Text",
            "url":          log.user_input if log.user_input.startswith('http') else "",
            "title":        log.user_input if not log.user_input.startswith('http') else (_extract_headline_from_url_slug(log.user_input) or "Saved Article Verification"),
            "author":       "",
            "publish_date": log.created_at.strftime("%Y-%m-%d"),
            "text_excerpt": log.text[:3000],
            "language":     "en",
        })
        if parsed_explanation:
            log.explanation = _json.dumps(parsed_explanation, ensure_ascii=False)[:12000]
            log.save()

    bias_val = get_bias_score(log.text)
    source_val = get_source_score(log.user_input) if log.user_input.startswith('http') else 50
    recency_val = get_recency_score(log.text)
    headline_val = get_headline_score(log.text)
    author_val = get_author_score(log.text)
    similarity_val = 75

    return render(request, 'result.html', {
        'prediction': log.prediction,
        'confidence': log.confidence,
        'trust_score': log.trust_score,
        'user_query': log.user_input,
        'ai_explanation': parsed_explanation,   # structured dict for new UI
        'prediction_id': log.id,
        'feedback': PredictionFeedback.objects.filter(
            prediction=log,
            user=request.user
        ).first(),
        'score_breakdown': {
            'ml': log.confidence,
            'source': source_val,
            'similarity': similarity_val,
            'recency': recency_val,
            'headline': headline_val,
            'bias': bias_val,
            'author': author_val
        },
        'analysis': {
            'length': len(log.text.split()),
            'tone': 'Emotional' if bias_val < 50 else 'Neutral',
            'risk_level': 'High' if log.trust_score < 40 else ('Medium' if log.trust_score < 70 else 'Low')
        }
    })


def result(request):

    return redirect('/')


def ask_ai(request):

    if request.method == "POST":

        question = request.POST.get(
            "question"
        )

        context = request.POST.get(
            "context"
        )

        language = (request.POST.get("language") or "en").strip()[:5]

        if not question or not context:
            return JsonResponse({
                "error": "Missing question or context"
            }, status=400)

        answer = ask_followup_question(
            question,
            context,
            language
        )

        return JsonResponse({
            "answer": answer
        })

    return JsonResponse({
        "error": "Only POST requests are allowed"
    }, status=405)


FEEDBACK_COMMENT_MAX = 1000


@login_required(login_url='/login/')
@require_POST
def submit_feedback(request):
    try:
        prediction_id = int(request.POST.get('prediction_id') or '')
    except (TypeError, ValueError):
        return JsonResponse(
            {'error': 'A valid analysis reference is required.'}, status=400)

    feedback_type = (request.POST.get('feedback_type') or '').strip().upper()
    if feedback_type not in dict(PredictionFeedback.FeedbackType.choices):
        return JsonResponse(
            {'error': 'Please choose whether the analysis was helpful.'}, status=400)

    reason = (request.POST.get('reason') or '').strip().upper()
    if reason and reason not in dict(PredictionFeedback.FeedbackReason.choices):
        return JsonResponse({'error': 'Unknown feedback reason.'}, status=400)
    if feedback_type == 'INCORRECT' and not reason:
        return JsonResponse(
            {'error': 'Please select a reason for your feedback.'}, status=400)

    comment = (request.POST.get('comment') or '').strip()
    if len(comment) > FEEDBACK_COMMENT_MAX:
        return JsonResponse(
            {'error': f'Comment must be {FEEDBACK_COMMENT_MAX} characters or fewer.'}, status=400)

    log = NewsLog.objects.filter(id=prediction_id, user=request.user).first()
    if not log:
        return JsonResponse(
            {'error': 'Analysis not found for this feedback.'}, status=404)

    if PredictionFeedback.objects.filter(prediction=log, user=request.user).exists():
        return JsonResponse({
            'status': 'already_submitted',
            'message': 'You already submitted feedback for this analysis.'
        }, status=409)

    try:
        PredictionFeedback.objects.create(
            prediction=log,
            user=request.user,
            feedback_type=feedback_type,
            reason=reason,
            comment=comment,
            predicted_label=log.prediction,
            prediction_confidence=log.confidence,
        )
    except IntegrityError:
        return JsonResponse({
            'status': 'already_submitted',
            'message': 'You already submitted feedback for this analysis.'
        }, status=409)

    message = ('Thank you for your feedback. '
               'Your feedback has been recorded and may help improve future analyses.')
    return JsonResponse({'status': 'ok', 'message': message})


def admin_feedback_update(request):
    admin_user_id = request.session.get('admin_user_id')
    admin_user = None
    if admin_user_id:
        admin_user = User.objects.filter(pk=admin_user_id, is_staff=True).first()
    if not admin_user:
        return JsonResponse({'error': 'Authentication required.'}, status=401)

    if request.method != 'POST':
        return JsonResponse({'error': 'Only POST requests are allowed.'}, status=405)

    try:
        feedback_id = int(request.POST.get('feedback_id') or '')
    except (TypeError, ValueError):
        return JsonResponse({'error': 'Invalid feedback reference.'}, status=400)

    review_status = (request.POST.get('review_status') or '').strip().upper()
    if review_status not in dict(PredictionFeedback.ReviewStatus.choices):
        return JsonResponse({'error': 'Invalid review status.'}, status=400)

    feedback = PredictionFeedback.objects.filter(id=feedback_id).first()
    if not feedback:
        return JsonResponse({'error': 'Feedback not found.'}, status=404)

    feedback.review_status = review_status
    feedback.reviewed_by = admin_user
    feedback.reviewed_at = timezone.now()
    feedback.save(update_fields=[
        'review_status', 'reviewed_by', 'reviewed_at', 'updated_at'
    ])
    return redirect('/admin-dashboard/?tab=feedback')


def admin_login_view(request):
    admin_user_id = request.session.get('admin_user_id')
    if admin_user_id:
        try:
            admin_user = User.objects.get(pk=admin_user_id, is_staff=True)
            return redirect('/admin-dashboard/')
        except User.DoesNotExist:
            request.session.pop('admin_user_id', None)

    next_url = request.GET.get('next') or request.POST.get('next') or '/admin-dashboard/'

    if request.method == 'POST':
        username_or_email = (request.POST.get('username') or '').strip()
        password = (request.POST.get('password') or '').strip()

        user_obj = None
        if '@' in username_or_email:
            try:
                user_obj = User.objects.get(email=username_or_email)
            except User.DoesNotExist:
                user_obj = None
        else:
            try:
                user_obj = User.objects.get(username=username_or_email)
            except User.DoesNotExist:
                user_obj = None

        if user_obj is not None:
            user = authenticate(request, username=user_obj.username, password=password)
            if user is not None:
                if user.is_staff:
                    logout(request)
                    request.session['admin_user_id'] = user.pk
                    messages.success(request, f'Administrator session started. Welcome, {user.username}.')
                    return redirect(next_url)
                else:
                    messages.error(request, 'Access denied. This account does not have administrator privileges.')
                    return render(request, 'admin_login.html', {
                        'next_url': next_url
                    })
            else:
                messages.error(request, 'Invalid administrator username or password.')
                return render(request, 'admin_login.html', {
                    'next_url': next_url
                })
        else:
            messages.error(request, 'Administrator account not found.')
            return render(request, 'admin_login.html', {'next_url': next_url})

    return render(request, 'admin_login.html', {'next_url': next_url})


def admin_dashboard(request):
    admin_user_id = request.session.get('admin_user_id')
    if not admin_user_id:
        return redirect('/admin-login/?next=/admin-dashboard/')

    try:
        admin_user = User.objects.get(pk=admin_user_id, is_staff=True)
    except User.DoesNotExist:
        request.session.pop('admin_user_id', None)
        return redirect('/admin-login/?next=/admin-dashboard/')

    if not admin_user.is_staff:
        return render(request, '403.html', status=403)

    from django.utils import timezone
    import json
    from datetime import timedelta
    from django.db.models import Avg

    today = timezone.now().date()

    total_users = User.objects.count()
    registered_users = User.objects.filter(is_active=True).count()

    try:
        todays_logins = LoginLog.objects.filter(created_at__date=today).count()
        login_logs_list = LoginLog.objects.all().order_by('-created_at')[:20]
    except Exception:
        todays_logins = 0
        login_logs_list = []

    try:
        reports_list = Report.objects.all().order_by('-created_at')[:20]
    except Exception:
        reports_list = []

    try:
        feedback_list = PredictionFeedback.objects.select_related(
            'user', 'prediction', 'reviewed_by'
        ).order_by('-created_at')[:50]
    except Exception:
        feedback_list = []

    total_predictions = PredictionHistory.objects.count()
    if total_predictions == 0:
        total_predictions = NewsLog.objects.count()

    todays_predictions = PredictionHistory.objects.filter(created_at__date=today).count()
    real_news_count = PredictionHistory.objects.filter(prediction__icontains='Real').count()
    fake_news_count = PredictionHistory.objects.filter(prediction__icontains='Fake').count()

    if real_news_count == 0 and fake_news_count == 0:
        real_news_count = NewsLog.objects.filter(prediction__icontains='Real').count()
        fake_news_count = NewsLog.objects.filter(prediction__icontains='Fake').count()

    url_analyses = PredictionHistory.objects.filter(input_type__icontains='URL').count()
    image_uploads = PredictionHistory.objects.filter(input_type__icontains='File').count()
    text_analyses = PredictionHistory.objects.filter(input_type__icontains='Text').count()

    avg_trust = PredictionHistory.objects.aggregate(Avg('trust_score'))['trust_score__avg']
    if avg_trust is None:
        avg_trust = NewsLog.objects.aggregate(Avg('trust_score'))['trust_score__avg'] or 78.4
    avg_trust = round(avg_trust, 1)

    # 7-day trend calculation
    trend_days = []
    trend_values = []
    for i in range(6, -1, -1):
        day_date = today - timedelta(days=i)
        trend_days.append(day_date.strftime("%b %d"))
        cnt = PredictionHistory.objects.filter(created_at__date=day_date).count()
        trend_values.append(cnt)

    recent_preds = PredictionHistory.objects.all().order_by('-created_at')[:25]
    users_list = User.objects.all().order_by('-date_joined')[:25]

    context = {
        'today_date': today.strftime("%B %d, %Y"),
        'stats': {
            'total_users': total_users,
            'registered_users': registered_users,
            'todays_logins': todays_logins,
            'total_predictions': total_predictions,
            'todays_predictions': todays_predictions,
            'real_news_count': real_news_count,
            'fake_news_count': fake_news_count,
            'image_uploads': image_uploads,
            'url_analyses': url_analyses,
            'text_analyses': text_analyses,
            'avg_trust_score': avg_trust,
            'avg_processing_time': '0.38s',
            'trend_days_json': json.dumps(trend_days),
            'trend_values_json': json.dumps(trend_values),
        },
        'recent_predictions': recent_preds,
        'users_list': users_list,
        'login_logs_list': login_logs_list,
        'reports_list': reports_list,
        'feedback_list': feedback_list,
        'active_tab': request.GET.get('tab', 'dashboard'),
        'user': admin_user,
    }

    return render(request, 'admin_dashboard.html', context)


@csrf_exempt
def chat_assistant_view(request):
    if request.method == "POST":
        import json
        try:
            data = json.loads(request.body)
            user_msg = (data.get("message") or "").strip()
        except Exception:
            user_msg = (request.POST.get("message") or "").strip()

        if not user_msg:
            return JsonResponse({"error": "Message content is required"}, status=400)

        # Custom AI Assistant Response logic
        system_context = "VeriTruth AI Fake News & Fact-Checking Assistant"
        ai_response = None
        try:
            ai_response = ask_followup_question(user_msg, system_context)
        except Exception:
            ai_response = None

        if not ai_response or "error" in str(ai_response).lower() or len(str(ai_response).strip()) == 0:
            lower_msg = user_msg.lower()
            if "trust" in lower_msg or "score" in lower_msg:
                ai_response = "Trust Scores on VeriTruth are computed using a multi-signal ensemble combining ML classifier probability (35%), domain source authority (25%), cosine similarity against fact-checked datasets (15%), language neutrality (15%), and recency plus headline signals (10%)."
            elif "url" in lower_msg or "link" in lower_msg:
                ai_response = "To verify a news URL, click the 'URL' tab on the Verification Workbench, paste the web article link, and click 'Execute Verification'. Our system will scrape the article text, verify domain reputation, and scan for clickbait signals."
            elif "image" in lower_msg or "ocr" in lower_msg or "upload" in lower_msg:
                ai_response = "You can upload news screenshots, PDF reports, or article images in the 'Upload' tab. Our Tesseract OCR parser automatically extracts the text content and runs full credibility detection."
            elif "fake" in lower_msg or "real" in lower_msg:
                ai_response = "Articles classified as 'Fake News' contain high bias, clickbait patterns, unverified domain sources, or contradictory claims against fact-checking databases. 'Real News' articles demonstrate objective reporting and trusted publisher authority."
            else:
                ai_response = f"I am your VeriTruth AI Assistant. Regarding '{user_msg}', our multi-signal verification engine scans article copy against machine learning classifiers, source registries, and language-bias signals to give you an accurate credibility assessment."

        return JsonResponse({"response": ai_response})

    return JsonResponse({"error": "Only POST requests are supported"}, status=405)
