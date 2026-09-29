"""
VeriTruth AI Explanation Engine
Uses Groq to generate structured, conversational explanations of prediction results.
Returns a parsed dict — NOT raw HTML — so the template can render sections cleanly.
"""

import os
import re
import json
import time
from dotenv import load_dotenv
from groq import Groq, APIConnectionError, APIStatusError, APITimeoutError

# ── Environment ───────────────────────────────────────────────────────────────
env_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env')
load_dotenv(dotenv_path=env_path)

api_key    = os.getenv("GROQ_API_KEY")
# NOTE: llama3-8b-8192 was decommissioned by Groq (400 model_decommissioned).
groq_model = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
client     = None

if api_key:
    try:
        client = Groq(api_key=api_key)
    except Exception as exc:
        print(f"[VeriTruth] Warning: Could not initialise Groq client: {exc}")

# ── Language map ─────────────────────────────────────────────────────────────
_LANG_NAMES = {
    "en": "English",
    "hi": "Hindi",
    "te": "Telugu",
    "ta": "Tamil",
    "bn": "Bengali",
    "mr": "Marathi",
    "es": "Spanish",
    "fr": "French",
    "de": "German",
}


def _lang_name(code: str) -> str:
    return _LANG_NAMES.get((code or "en").lower()[:2], "English")


# ── Fallback builder (no Groq) ────────────────────────────────────────────────
def _build_fallback(data: dict) -> dict:
    """
    Used only when Groq is unavailable or returns unusable output.
    Without the model we cannot summarise the article, so `about` stays empty
    and the template shows an "explanation unavailable" notice instead of
    inventing article content. Signal values are reported as-is.
    """
    verdict     = data.get("prediction", "Unknown")
    confidence  = round(float(data.get("confidence", 0) or 0), 1)
    trust       = round(float(data.get("trust", 50) or 50), 1)

    why_result = (
        f"The BERT classifier selected {verdict} with {confidence}% confidence, "
        f"and the supporting signals produced a Trust Index of {trust}%. "
        + (
            "The available signals are listed below; the narrative explanation could not be generated."
        )
    )

    verify = [
        "Check the original publisher and confirm its reputation.",
        "Search for the same story in at least two independent outlets.",
        "Verify names, dates and figures against official sources.",
    ]

    return {
        "about":          "",
        "why_result":     why_result,
        "key_claims":     [],
        "what_to_verify": verify,
        "is_fallback":    True,
        "groq_available": False,
    }


# ── JSON response parser ──────────────────────────────────────────────────────
def _parse_groq_response(text: str) -> dict | None:
    """
    Tries to extract a JSON object from the Groq response.
    Handles markdown fences (```json ... ```), leading/trailing prose,
    and unbalanced fragments. Returns None if parsing fails.
    """
    if not text:
        return None

    text = text.strip()

    # 1) Content inside a markdown code fence, if present
    fence = re.search(r'```(?:json)?\s*([\s\S]*?)```', text, flags=re.IGNORECASE)
    candidates = []
    if fence:
        candidates.append(fence.group(1).strip())

    # 2) Whole text with any stray fences stripped
    cleaned = re.sub(r'```(?:json)?', '', text).strip()
    candidates.append(cleaned)

    # 3) First '{' … last '}' slice (handles prose before/after the JSON)
    start, end = text.find('{'), text.rfind('}')
    if start != -1 and end > start:
        candidates.append(text[start:end + 1])

    for cand in candidates:
        try:
            obj = json.loads(cand)
            if isinstance(obj, dict) and ("about" in obj or "summary" in obj):
                return obj
        except json.JSONDecodeError:
            continue

    return None


# ── Groq call with retry on transient failures ────────────────────────────────
_RETRYABLE_STATUS = {408, 409, 429, 500, 502, 503, 504}


def _groq_chat(messages: list, max_tokens: int, timeout: int, retries: int = 2):
    """Single Groq chat call; retries rate-limits / server errors / timeouts."""
    last_exc = None
    for attempt in range(retries + 1):
        try:
            return client.chat.completions.create(
                model=groq_model,
                messages=messages,
                temperature=0.3,
                max_tokens=max_tokens,
                timeout=timeout,
            )
        except APIStatusError as exc:
            last_exc = exc
            if exc.status_code not in _RETRYABLE_STATUS:
                raise
        except (APITimeoutError, APIConnectionError) as exc:
            last_exc = exc
        if attempt < retries:
            time.sleep(1.5 * (attempt + 1))
    raise last_exc


# ── Main explanation generator ────────────────────────────────────────────────
def generate_explanation(data: dict) -> dict:
    """
    Calls Groq with the submitted article plus the existing analysis values and
    returns an article-specific explanation:
    {
        "about":          str,   # what the submitted article actually says
        "why_result":     str,   # why the existing analysis gave this result
        "key_claims":     list,  # up to 3 claims taken from the article
        "what_to_verify": list,  # article-specific verification steps
        "is_fallback":    bool,
        "groq_available": bool,
    }
    Falls back to _build_fallback() if Groq is unavailable or returns bad output.
    Never raises — always returns a valid dict.
    """
    if not api_key or not client:
        result = _build_fallback(data)
        result["groq_available"] = False
        return result

    verdict     = data.get("prediction", "Unknown")
    confidence  = round(float(data.get("confidence", 0) or 0), 1)
    trust       = round(float(data.get("trust", 50) or 50), 1)
    bias        = round(float(data.get("bias", 50) or 50), 1)
    source      = round(float(data.get("source", 50) or 50), 1)
    similarity  = round(float(data.get("similarity", 50) or 50), 1)
    recency     = round(float(data.get("recency", 50) or 50), 1)
    headline    = round(float(data.get("headline", 50) or 50), 1)
    author_sig  = data.get("author_score", "Not available")
    title       = data.get("title", "") or "Not available"
    author      = data.get("author", "") or "Not available"
    pub_date    = data.get("publish_date", "") or "Not available"
    url         = data.get("url", "") or ""
    input_type  = data.get("input_type", "") or "Text"
    excerpt     = (data.get("text_excerpt") or data.get("article_text") or "")[:3000].strip()
    language    = _lang_name(data.get("language", "en"))

    if not excerpt:
        excerpt = "Not available — the submitted content contained no extractable text."

    prompt = f"""You are explaining an EXISTING news-analysis result to the user.
Read the submitted article carefully before generating the explanation.

Your explanation must be specific to THIS submitted article. Do not simply
repeat the numerical scores, and do not reuse a generic template.

=== SUBMITTED ARTICLE ===
Input type     : {input_type}
Title          : {title}
Author         : {author}
Published      : {pub_date}
URL / source   : {url or "Not available"}
Article text   :
{excerpt}

=== EXISTING ANALYSIS RESULT (already computed — do not recompute) ===
BERT prediction : {verdict}
BERT confidence : {confidence}%   (confidence in the selected class, NOT a probability of truth)
Source Authority: {source}%
Similarity      : {similarity}%   (higher = further from known fake-news language patterns)
Headline Align. : {headline}%
Language / Bias : {bias}%
Recency         : {recency}%
Author signal   : {author_sig}
Trust Index     : {trust}%        (combined assessment of the supporting signals)

=== WHAT TO PRODUCE ===
Respond ONLY with a single valid JSON object in this exact shape — no markdown
fences, no preamble, no trailing text:

{{
  "about": "2-4 sentences in plain language: what this article actually says.
            Base it strictly on the submitted text above.",
  "why_result": "3-5 sentences: why the existing analysis led to '{verdict}'.
                 Discuss the article's own content and wording (unsupported
                 assertions, missing attribution, certainty of tone, missing
                 dates/sources, sensational phrasing, or the opposite), then
                 connect that to the signals listed above.",
  "key_claims": [
    "The article claims that ... (1-3 entries, only claims present in the text)"
  ],
  "what_to_verify": [
    "A verification step specific to this article's subject (2-4 entries)"
  ]
}}

=== STRICT RULES ===
- BERT has already decided the classification. NEVER change, override or
  contradict it, and never produce your own verdict.
- Never invent people, organisations, dates, statistics, studies, quotes, URLs,
  government announcements or evidence. Only use what is in the submitted text
  and the analysis values above.
- Attribute claims to the article: "The article claims…", "The submitted text
  states…". Never present them as independently verified facts.
- Do not call a score a probability of truth or falsity. BERT confidence means
  the classifier's certainty about its selected class; the Trust Index is a
  combined assessment of the supporting signals.
- If the article is too short or vague, say that plainly: "Not enough
  information is available in the submitted article to determine this."
  Return an empty "key_claims" array rather than inventing claims.
- Verification steps must relate to what THIS article discusses (a government
  decision → the relevant official announcement; a study → the original
  publication; a recent event → the reported date against independent
  coverage). Never name a specific source you cannot see in the data.
- For OCR/image input, discuss only the extracted text — never visual details.
- For URL input, do not treat the URL itself as proof of reliability.
- Write natural, varied prose. Do not start every sentence the same way, and do
  not use the phrase "the article is flagged as".

=== LANGUAGE ===
Write the ENTIRE response — every value in the JSON — in {language}.
Technical terms such as BERT, Trust Index and Confidence may remain in English
within the {language} text. Do not translate only the headings.
"""

    try:
        response = _groq_chat(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are the VeriTruth AI explanation writer. You explain an "
                        "existing news-analysis result for the specific article the user "
                        "submitted. You always respond with a single valid JSON object — "
                        "no markdown fences, no preamble, no trailing text. You never "
                        "fabricate evidence and you never change the model's prediction."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            max_tokens=1400,
            timeout=40,
        )

        raw = response.choices[0].message.content or ""
        parsed = _parse_groq_response(raw)

        if parsed:
            parsed.setdefault("about", "")
            parsed.setdefault("why_result", "")
            parsed.setdefault("key_claims", [])
            parsed.setdefault("what_to_verify", [])

            for key in ("about", "why_result"):
                if not isinstance(parsed[key], str):
                    parsed[key] = str(parsed[key] or "")
                parsed[key] = parsed[key].strip()

            for key in ("key_claims", "what_to_verify"):
                if isinstance(parsed[key], str):
                    parsed[key] = [parsed[key]] if parsed[key].strip() else []
                if not isinstance(parsed[key], list):
                    parsed[key] = []
                parsed[key] = [str(item).strip() for item in parsed[key] if str(item).strip()]

            parsed["key_claims"]     = parsed["key_claims"][:3]
            parsed["what_to_verify"] = parsed["what_to_verify"][:4]

            parsed["is_fallback"]    = False
            parsed["groq_available"] = True
            return parsed

        # JSON parsing failed — use fallback with groq_available=True so
        # the UI doesn't show a "service unavailable" warning
        fallback = _build_fallback(data)
        fallback["is_fallback"]    = True
        fallback["groq_available"] = True   # Groq worked, just bad JSON
        return fallback

    except Exception as exc:
        # Never log the API key — only exception type and message.
        print(f"[VeriTruth] Groq explanation error ({type(exc).__name__}, model={groq_model}): {exc}")
        fallback = _build_fallback(data)
        fallback["groq_available"] = False
        return fallback


# ── Follow-up question ────────────────────────────────────────────────────────
_UNAVAILABLE_MSG = (
    "VeriTruth couldn't answer this follow-up right now. "
    "Your original analysis is still available above."
)


def ask_followup_question(question: str, context: str, language: str = "en") -> str:
    """
    Answers a follow-up question about an EXISTING analysis.
    Explanation/investigation only — it must never re-classify or override
    the BERT prediction, and it must not fabricate evidence.
    `language` is an optional code (e.g. "en", "te"); the answer is produced
    in that language. Existing two-argument callers keep working.
    """
    if not api_key or not client:
        return _UNAVAILABLE_MSG

    lang = _lang_name(language)

    prompt = f"""You are the VeriTruth AI Investigative Assistant.
A news-credibility analysis has ALREADY been completed. Your only job is to
explain and investigate that existing result — you are NOT re-running detection.

=== EXISTING ANALYSIS CONTEXT ===
{context}

=== USER'S FOLLOW-UP QUESTION ===
{question}

=== STRICT RULES ===
- The classification (FAKE / REAL) is already decided by the BERT model. NEVER
  change, override, or contradict it. If the user doubts the verdict, explain
  why the system reached it — do not reclassify.
- Answer ONLY from the context above. Do NOT invent facts, sources, quotes,
  statistics, or evidence that are not present.
- Do NOT claim you searched the web or accessed external databases. VeriTruth
  did not perform a live web search for this answer.
- If the context lacks enough detail to answer, say so plainly and suggest what
  the user should verify independently.
- Use hedged, careful language ("suggests", "indicates", "likely"). Never claim
  absolute certainty.

=== FORMAT ===
- Keep it concise and readable — a short paragraph, plus bullet points when
  listing signals or verification steps.
- You may use simple markdown: **bold**, "- " bullets, and "1. " numbered lists.
- Do NOT use headings larger than a short bolded lead-in. No tables, no code.
- Avoid long walls of text.

=== LANGUAGE ===
Write the ENTIRE answer in {lang}. Technical terms (BERT, VeriTruth, trust
score) may stay in English within the {lang} text.
"""

    try:
        response = _groq_chat(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are the VeriTruth AI Investigative Assistant. "
                        "You explain an existing fake-news analysis without "
                        "changing its verdict, and you never fabricate evidence."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            max_tokens=500,
            timeout=25,
        )
        return (response.choices[0].message.content or "").strip() or _UNAVAILABLE_MSG

    except Exception as exc:
        # Never log the API key — only exception type and model.
        print(f"[VeriTruth] Groq follow-up error ({type(exc).__name__}, model={groq_model}): {exc}")
        return _UNAVAILABLE_MSG
