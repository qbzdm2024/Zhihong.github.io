"""
Automatic full-text downloader for systematic review records.

Tries sources in this order, accepting whatever format is available first:
  1. Unpaywall       — open-access PDF or HTML landing page
  2. Semantic Scholar — openAccessPdf or externalIds → PMC
  3. OpenAlex        — pdf_url or oa_url (HTML/landing)
  4. Europe PMC      — PMC full-text HTML (very reliable for biomedical)
  5. PubMed Central  — HTML full-text directly via PMC ID

Saves the result as:
  - <record_id>.pdf   if a real PDF was obtained
  - <record_id>.txt   if HTML full-text was extracted

Pipeline text extraction reads both formats via _find_fulltext_file().
Records with no accessible full text are marked FULL_TEXT_NEEDED.
"""
import logging
import re
import time
from html.parser import HTMLParser
from pathlib import Path
from typing import Optional, Tuple

import requests

logger = logging.getLogger(__name__)

_RATE_LIMIT_SECS = 0.4

# Polite-pool email for Unpaywall / OpenAlex
CONTACT_EMAIL = "systematic.review@research.org"

_HEADERS = {
    "User-Agent": (
        "SystematicReviewBot/1.0 "
        "(academic systematic review; "
        f"mailto:{CONTACT_EMAIL})"
    ),
    "Accept": "application/json",
}


# ─────────────────────────────────────────────────────────
# HTML → text extractor (no extra deps)
# ─────────────────────────────────────────────────────────

class _TextExtractor(HTMLParser):
    """Strip HTML tags and return clean text."""
    def __init__(self):
        super().__init__()
        self._parts = []
        self._skip = False

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style", "nav", "header", "footer"):
            self._skip = True

    def handle_endtag(self, tag):
        if tag in ("script", "style", "nav", "header", "footer"):
            self._skip = False
        if tag in ("p", "div", "section", "h1", "h2", "h3", "h4", "li", "br"):
            self._parts.append("\n")

    def handle_data(self, data):
        if not self._skip:
            self._parts.append(data)

    def get_text(self) -> str:
        raw = "".join(self._parts)
        # Collapse runs of whitespace/newlines
        text = re.sub(r"\n{3,}", "\n\n", raw)
        text = re.sub(r" {2,}", " ", text)
        return text.strip()


def _html_to_text(html: str) -> str:
    p = _TextExtractor()
    try:
        p.feed(html)
        return p.get_text()
    except Exception:
        # Fallback: crude tag strip
        return re.sub(r"<[^>]+>", " ", html)


# ─────────────────────────────────────────────────────────
# Helpers: fetch with retries
# ─────────────────────────────────────────────────────────

def _get(url: str, accept: str = "application/json", timeout: int = 20) -> Optional[requests.Response]:
    try:
        headers = {**_HEADERS, "Accept": accept}
        r = requests.get(url, headers=headers, timeout=timeout)
        if r.status_code == 200:
            return r
    except Exception as e:
        logger.debug(f"GET {url[:60]} → {e}")
    return None


# ─────────────────────────────────────────────────────────
# SOURCE 1: Unpaywall
# ─────────────────────────────────────────────────────────

def _unpaywall_urls(doi: str) -> Tuple[Optional[str], Optional[str]]:
    """Return (pdf_url, html_url) from Unpaywall, either may be None."""
    if not doi:
        return None, None
    r = _get(f"https://api.unpaywall.org/v2/{doi}?email={CONTACT_EMAIL}")
    if not r:
        return None, None
    data = r.json()
    best = data.get("best_oa_location") or {}
    pdf = best.get("url_for_pdf")
    html = best.get("url_for_landing_page") if not pdf else None
    # Also check all OA locations for a PDF
    if not pdf:
        for loc in (data.get("oa_locations") or []):
            if loc.get("url_for_pdf"):
                pdf = loc["url_for_pdf"]
                break
    return pdf, html


# ─────────────────────────────────────────────────────────
# SOURCE 2: Semantic Scholar
# ─────────────────────────────────────────────────────────

def _semantic_scholar_info(doi: str, title: str) -> Tuple[Optional[str], Optional[str]]:
    """Return (pdf_url, pmcid) from Semantic Scholar."""
    pdf_url = pmcid = None
    try:
        if doi:
            url = (f"https://api.semanticscholar.org/graph/v1/paper/DOI:{doi}"
                   f"?fields=openAccessPdf,externalIds")
        elif title:
            url = (f"https://api.semanticscholar.org/graph/v1/paper/search"
                   f"?query={requests.utils.quote(title)}&fields=openAccessPdf,externalIds&limit=1")
        else:
            return None, None
        r = _get(url)
        if r:
            data = r.json()
            if "data" in data:
                data = data["data"][0] if data["data"] else {}
            oa = data.get("openAccessPdf") or {}
            pdf_url = oa.get("url")
            pmcid = (data.get("externalIds") or {}).get("PubMedCentral")
    except Exception as e:
        logger.debug(f"SemanticScholar error: {e}")
    return pdf_url, pmcid


# ─────────────────────────────────────────────────────────
# SOURCE 3: OpenAlex
# ─────────────────────────────────────────────────────────

def _openalex_urls(doi: str) -> Tuple[Optional[str], Optional[str]]:
    """Return (pdf_url, oa_html_url) from OpenAlex."""
    if not doi:
        return None, None
    r = _get(f"https://api.openalex.org/works/doi:{doi}?mailto={CONTACT_EMAIL}")
    if not r:
        return None, None
    data = r.json()
    pdf_url = html_url = None
    for loc in (data.get("locations") or []):
        if loc.get("pdf_url") and not pdf_url:
            pdf_url = loc["pdf_url"]
        if loc.get("landing_page_url") and not html_url:
            html_url = loc["landing_page_url"]
    if not pdf_url:
        oa = data.get("open_access") or {}
        oa_url = oa.get("oa_url", "")
        if oa_url.endswith(".pdf"):
            pdf_url = oa_url
        elif oa_url:
            html_url = html_url or oa_url
    return pdf_url, html_url


# ─────────────────────────────────────────────────────────
# SOURCE 4 & 5: Europe PMC / PubMed Central HTML
# ─────────────────────────────────────────────────────────

def _pmc_html_text(pmcid: str) -> Optional[str]:
    """Fetch Europe PMC HTML full-text and convert to plain text."""
    if not pmcid:
        return None
    # Europe PMC provides XML full text for free
    xml_url = (f"https://www.ebi.ac.uk/europepmc/webservices/rest/{pmcid}/fullTextXML")
    r = _get(xml_url, accept="application/xml")
    if r and len(r.text) > 500:
        # Strip XML tags → readable text
        text = re.sub(r"<[^>]+>", " ", r.text)
        text = re.sub(r"\s{2,}", " ", text).strip()
        return text if len(text) > 200 else None

    # Fallback: NCBI PMC HTML
    html_url = f"https://www.ncbi.nlm.nih.gov/pmc/articles/{pmcid}/"
    r2 = _get(html_url, accept="text/html")
    if r2 and len(r2.text) > 1000:
        return _html_to_text(r2.text) or None
    return None


def _europepmc_find_pmcid(doi: str, title: str) -> Optional[str]:
    """Search Europe PMC to find PMC ID for a record."""
    query = doi if doi else title
    if not query:
        return None
    url = (f"https://www.ebi.ac.uk/europepmc/webservices/rest/search"
           f"?query={requests.utils.quote(query)}"
           f"&format=json&resulttype=core&pageSize=1")
    r = _get(url)
    if r:
        results = r.json().get("resultList", {}).get("result", [])
        if results:
            rec = results[0]
            if rec.get("isOpenAccess") == "Y" or rec.get("pmcid"):
                return rec.get("pmcid")
    return None


# ─────────────────────────────────────────────────────────
# PDF download + verification
# ─────────────────────────────────────────────────────────

def _download_pdf(pdf_url: str, save_path: Path) -> bool:
    """Download and verify a PDF. Returns True if a valid PDF was saved."""
    try:
        headers = {**_HEADERS, "Accept": "application/pdf,*/*"}
        r = requests.get(pdf_url, headers=headers, timeout=45, stream=True)
        if r.status_code != 200:
            return False
        chunks, total = [], 0
        for chunk in r.iter_content(chunk_size=16384):
            chunks.append(chunk)
            total += len(chunk)
            if total > 25_000_000:
                logger.debug(f"PDF too large at {pdf_url[:60]}")
                return False
        content = b"".join(chunks)
        if not content.startswith(b"%PDF"):
            logger.debug(f"Not a PDF (magic check failed): {pdf_url[:60]}")
            return False
        save_path.write_bytes(content)
        return True
    except Exception as e:
        logger.debug(f"PDF download failed {pdf_url[:60]}: {e}")
        return False


# ─────────────────────────────────────────────────────────
# HTML / text landing-page extraction
# ─────────────────────────────────────────────────────────

def _fetch_html_text(url: str, min_chars: int = 1000) -> Optional[str]:
    """Fetch a URL as HTML and return its plain-text body, or None."""
    try:
        r = requests.get(url, headers={**_HEADERS, "Accept": "text/html,*/*"},
                         timeout=30)
        if r.status_code != 200:
            return None
        ct = r.headers.get("content-type", "")
        # Don't mistake a PDF response for HTML
        if "pdf" in ct:
            return None
        text = _html_to_text(r.text)
        return text if len(text) >= min_chars else None
    except Exception as e:
        logger.debug(f"HTML fetch failed {url[:60]}: {e}")
        return None


# ─────────────────────────────────────────────────────────
# PUBLIC API
# ─────────────────────────────────────────────────────────

def try_download_fulltext(
    doi: str,
    title: str,
    record_id: str,
    pdf_dir: Path,
) -> Tuple[bool, str]:
    """
    Try to obtain the full text for one record by any available means.

    Strategy:
      1. Unpaywall PDF
      2. Semantic Scholar PDF
      3. OpenAlex PDF
      4. PMC HTML (via Semantic Scholar PMCID or Europe PMC search)
      5. Europe PMC XML/HTML
      6. Unpaywall HTML landing page (last resort)

    Saves result as <record_id>.pdf  (real PDF)
                 or <record_id>.txt  (plain text extracted from HTML/XML)

    Returns:
      (True,  source_name)   if full text was obtained
      (False, "not_found")   if nothing worked
    """
    doi = (doi or "").strip()
    title = (title or "").strip()

    pdf_path = pdf_dir / f"{record_id}.pdf"
    txt_path = pdf_dir / f"{record_id}.txt"

    # Already have it
    if pdf_path.exists() and pdf_path.stat().st_size > 500:
        return True, "cached_pdf"
    if txt_path.exists() and txt_path.stat().st_size > 200:
        return True, "cached_txt"

    # ── Step 1: Gather URLs from APIs (one round-trip per service) ──────────
    time.sleep(_RATE_LIMIT_SECS)
    uw_pdf, uw_html = _unpaywall_urls(doi)

    time.sleep(_RATE_LIMIT_SECS)
    ss_pdf, ss_pmcid = _semantic_scholar_info(doi, title)

    time.sleep(_RATE_LIMIT_SECS)
    oa_pdf, oa_html = _openalex_urls(doi)

    # ── Step 2: Try PDFs first (highest quality) ────────────────────────────
    for source, pdf_url in [
        ("Unpaywall", uw_pdf),
        ("SemanticScholar", ss_pdf),
        ("OpenAlex", oa_pdf),
    ]:
        if pdf_url:
            logger.info(f"[{source}] PDF → {pdf_url[:70]}")
            time.sleep(_RATE_LIMIT_SECS)
            if _download_pdf(pdf_url, pdf_path):
                return True, f"{source}_pdf"

    # ── Step 3: PMC HTML (Europe PMC or NCBI) ───────────────────────────────
    pmcid = ss_pmcid
    if not pmcid:
        time.sleep(_RATE_LIMIT_SECS)
        pmcid = _europepmc_find_pmcid(doi, title)

    if pmcid:
        logger.info(f"[PMC] HTML → {pmcid}")
        time.sleep(_RATE_LIMIT_SECS)
        text = _pmc_html_text(pmcid)
        if text and len(text) >= 1000:
            txt_path.write_text(text, encoding="utf-8")
            return True, "PMC_html"

    # ── Step 4: Unpaywall or OpenAlex HTML landing page ─────────────────────
    for source, html_url in [("Unpaywall_html", uw_html), ("OpenAlex_html", oa_html)]:
        if html_url:
            logger.info(f"[{source}] HTML → {html_url[:70]}")
            time.sleep(_RATE_LIMIT_SECS)
            text = _fetch_html_text(html_url, min_chars=1500)
            if text:
                txt_path.write_text(text, encoding="utf-8")
                return True, source

    return False, "not_found"
