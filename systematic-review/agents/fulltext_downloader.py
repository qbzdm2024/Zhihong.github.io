"""
Automatic full-text downloader for systematic review records.

Tries sources in this order, accepting whatever format is available first:
  1.  Unpaywall        — open-access PDF or HTML landing page
  2.  Semantic Scholar — openAccessPdf or externalIds → PMC
  3.  OpenAlex         — pdf_url or oa_url (HTML/landing)
  4.  Europe PMC       — PMC full-text HTML (very reliable for biomedical)
  5.  PubMed Central   — HTML full-text directly via PMC ID
  6.  arXiv            — PDF via arXiv API (exact + keyword title search)
  7.  CORE             — open-access aggregator PDF/text
  8.  Direct URL       — record's own URL field
  9.  ACL Anthology    — all ACL/EMNLP/NAACL/COLING NLP papers (free PDF)
 10.  OpenReview       — ICLR/NeurIPS/ICML papers (free PDF/HTML)
 11.  Zenodo           — open-access repository
 12.  bioRxiv/medRxiv  — biomedical preprints
 13.  Google Scholar   — real Chromium browser via Playwright (DOI first, then title)
 13b. Google Scholar   — requests scrape fallback (when Playwright unavailable)
 14.  BASE             — Bielefeld Academic Search Engine (300M+ OA docs)
 15.  CrossRef         — DOI landing page resolution
 16.  PubMed           — title search → PMCID → full text

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
    """Fetch Europe PMC XML full-text and convert to structured plain text.

    Extracts only the <body> element from JATS/NLM XML so that article
    metadata (journal IDs, author lists, PMC status fields) is excluded.
    Section headings (<title>) and paragraphs (<p>) are converted to
    blank-line-separated text so the heading detector in phase1_extractor
    can reliably find Methods / Results headings.
    """
    if not pmcid:
        return None

    # Europe PMC provides JATS XML full text for free
    xml_url = f"https://www.ebi.ac.uk/europepmc/webservices/rest/{pmcid}/fullTextXML"
    r = _get(xml_url, accept="application/xml")
    if r and len(r.text) > 500:
        xml = r.text

        # ── 1. Keep only the <body> block using XML parser ──────────────────
        # Try lxml / ElementTree first (more reliable than regex on large docs)
        xml_body: Optional[str] = None
        try:
            import xml.etree.ElementTree as ET
            # Strip the DOCTYPE declaration if present (breaks stdlib ET)
            xml_no_doctype = re.sub(r"<!DOCTYPE[^>]*>", "", xml, flags=re.DOTALL)
            root = ET.fromstring(xml_no_doctype)
            body_el = root.find(".//{http://dtd.nlm.nih.gov/publishing/2.3}body") \
                      or root.find("body") \
                      or root.find(".//body")
            if body_el is not None:
                xml_body = ET.tostring(body_el, encoding="unicode")
        except Exception:
            pass

        # Fallback: regex extraction of <body>…</body> block
        if xml_body is None:
            body_m = re.search(r"<body\b[^>]*>(.*?)</body>", xml, re.DOTALL | re.IGNORECASE)
            if body_m:
                xml_body = body_m.group(1)

        # If we still have no body, refuse to save the metadata-only content
        if not xml_body:
            logger.warning("[PMC] No <body> found in XML for %s — skipping", pmcid)
            xml_body = None
            # fall through to NCBI HTML fallback below

        if xml_body:
            # ── 2. Convert structural tags → blank lines before stripping ──
            xml_body = re.sub(r"<title\b[^>]*>", "\n\n", xml_body)
            xml_body = re.sub(r"</title>", "\n", xml_body)
            xml_body = re.sub(r"</?p\b[^>]*>", "\n\n", xml_body)
            xml_body = re.sub(r"</?sec\b[^>]*>", "\n\n", xml_body)

            # ── 3. Strip remaining tags and normalise whitespace ────────────
            text = re.sub(r"<[^>]+>", " ", xml_body)
            text = re.sub(r"[ \t]{2,}", " ", text)
            text = re.sub(r"\n{3,}", "\n\n", text).strip()

            # ── 4. Sanity-check: reject if it still looks like JATS metadata
            sample = text[:400]
            looks_like_metadata = (
                bool(re.search(r"PMC\d{5,}", sample))
                or "pmc-status-" in sample
                or "&amp;" in sample
                or "Writing – original draft" in sample
                or "Writing - original draft" in sample
            )
            if not looks_like_metadata and len(text) > 200:
                return text
            logger.warning("[PMC] Extracted text still looks like metadata for %s — trying HTML fallback", pmcid)

    # Fallback: NCBI PMC HTML
    html_url = f"https://www.ncbi.nlm.nih.gov/pmc/articles/{pmcid}/"
    r2 = _get(html_url, accept="text/html")
    if r2 and len(r2.text) > 1000:
        return _html_to_text(r2.text) or None
    return None


def _europepmc_find_pmcid(doi: str, title: str) -> Optional[str]:
    """Search Europe PMC to find PMC ID for a record.
    Returns PMCID whenever found — PMC full-text is always free regardless of
    the isOpenAccess flag, which is sometimes missing on valid PMC articles.
    """
    query = doi if doi else title
    if not query:
        return None
    url = (f"https://www.ebi.ac.uk/europepmc/webservices/rest/search"
           f"?query={requests.utils.quote(query)}"
           f"&format=json&resulttype=core&pageSize=3")
    r = _get(url)
    if r:
        results = r.json().get("resultList", {}).get("result", [])
        for rec in results:
            pmcid = rec.get("pmcid")
            if pmcid:
                return pmcid
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
# SOURCE 6: arXiv (critical for CS/AI papers)
# ─────────────────────────────────────────────────────────

def _arxiv_pdf_url(doi: str, title: str) -> Optional[str]:
    """Search arXiv by DOI or title; return PDF URL if found.

    Two passes:
      1. Exact quoted title search  (ti:"...") — precise but misses subtitle variants
      2. Keyword title search       (ti:word1+word2+...) — catches renamed preprints
    Match threshold: 40% word overlap (down from 60%) to handle published titles
    that differ slightly from their arXiv preprint version.
    """
    import xml.etree.ElementTree as ET

    # Direct arXiv DOI (e.g. 10.48550/arXiv.2301.00001)
    if doi and "arxiv" in doi.lower():
        arxiv_id = doi.split("/")[-1].replace("arXiv.", "").replace("arxiv.", "")
        if arxiv_id:
            return f"https://arxiv.org/pdf/{arxiv_id}"

    if not title:
        return None

    def _parse_entries(xml_text: str) -> Optional[str]:
        try:
            root = ET.fromstring(xml_text)
            ns = {"atom": "http://www.w3.org/2005/Atom"}
            entries = root.findall("atom:entry", ns)
            for entry in entries:
                entry_title = (entry.findtext("atom:title", "", ns) or "").lower().strip()
                query_title = title.lower().strip()
                t_words = set(query_title.split())
                e_words = set(entry_title.split())
                # 40% overlap — tolerates subtitle/punctuation differences
                if t_words and len(t_words & e_words) / len(t_words) >= 0.40:
                    for link in entry.findall("atom:link", ns):
                        if link.get("type") == "application/pdf" or link.get("title") == "pdf":
                            return link.get("href")
                    entry_id = entry.findtext("atom:id", "", ns)
                    if entry_id:
                        arxiv_id = entry_id.split("/abs/")[-1]
                        return f"https://arxiv.org/pdf/{arxiv_id}"
        except Exception as e:
            logger.debug(f"arXiv parse error: {e}")
        return None

    # Pass 1: exact quoted title
    query1 = requests.utils.quote(f'ti:"{title}"')
    url1 = f"http://export.arxiv.org/api/query?search_query={query1}&max_results=5&sortBy=relevance"
    r1 = _get(url1, accept="application/atom+xml", timeout=20)
    if r1:
        result = _parse_entries(r1.text)
        if result:
            return result

    # Pass 2: keyword search — important words only (skip stop-words, ≥4 chars)
    stop = {"with", "from", "that", "this", "using", "based", "deep", "neural",
            "learning", "approach", "study", "analysis", "towards", "method"}
    keywords = [w for w in re.sub(r"[^\w\s]", " ", title.lower()).split()
                if len(w) >= 4 and w not in stop][:8]
    if keywords:
        kw_query = requests.utils.quote("ti:" + "+".join(keywords))
        url2 = (f"http://export.arxiv.org/api/query?search_query={kw_query}"
                f"&max_results=5&sortBy=relevance")
        time.sleep(_RATE_LIMIT_SECS)
        r2 = _get(url2, accept="application/atom+xml", timeout=20)
        if r2:
            result = _parse_entries(r2.text)
            if result:
                return result

    return None


# ─────────────────────────────────────────────────────────
# SOURCE 7: CORE open-access aggregator
# ─────────────────────────────────────────────────────────

def _core_download_url(doi: str, title: str) -> Optional[str]:
    """Search CORE for an open-access download URL."""
    query = doi if doi else title
    if not query:
        return None
    # CORE v3 search
    url = (f"https://api.core.ac.uk/v3/search/works"
           f"?q={requests.utils.quote(query)}&limit=3&fields=downloadUrl,doi,title")
    r = _get(url)
    if not r:
        return None
    try:
        results = r.json().get("results") or []
        for item in results:
            dl = item.get("downloadUrl")
            if dl:
                # Verify it's the right paper if we searched by title
                if not doi:
                    item_title = (item.get("title") or "").lower()
                    if not any(w in item_title for w in title.lower().split()[:4]):
                        continue
                return dl
    except Exception as e:
        logger.debug(f"CORE error: {e}")
    return None


# ─────────────────────────────────────────────────────────
# SOURCE 9: ACL Anthology (NLP/CL papers — all freely available)
# ─────────────────────────────────────────────────────────

def _acl_anthology_pdf(doi: str, title: str) -> Optional[str]:
    """Find a paper in the ACL Anthology and return its PDF URL."""
    # ACL DOIs look like 10.18653/v1/P19-1001 or 10.18653/v1/2020.acl-main.1
    if doi and "18653" in doi:
        # Direct PDF: https://aclanthology.org/P19-1001.pdf
        paper_id = doi.split("/")[-1]
        return f"https://aclanthology.org/{paper_id}.pdf"

    if not title:
        return None
    # ACL Anthology search API
    q = requests.utils.quote(title)
    r = _get(f"https://aclanthology.org/search/?q={q}&limit=5", accept="text/html")
    if not r:
        return None
    # Extract first result link from the HTML
    matches = re.findall(r'href="/([\w\d.]+?)/"[^>]*>\s*<strong', r.text)
    if not matches:
        matches = re.findall(r'href="/(20\d\d\.\S+?|[A-Z]\d\d-\d+?)\.(?:html|pdf)"', r.text)
    for paper_id in matches[:3]:
        candidate = f"https://aclanthology.org/{paper_id}.pdf"
        # Lightweight check: verify title in the anthology page
        page = _get(f"https://aclanthology.org/{paper_id}/", accept="text/html")
        if page:
            page_title = re.search(r'<title>([^<]+)</title>', page.text)
            if page_title:
                pt = page_title.group(1).lower()
                tw = set(title.lower().split())
                if len(tw & set(pt.split())) / max(len(tw), 1) >= 0.6:
                    return candidate
    return None


# ─────────────────────────────────────────────────────────
# SOURCE 10: OpenReview (ICLR, NeurIPS, ICML, etc.)
# ─────────────────────────────────────────────────────────

def _openreview_pdf(title: str) -> Optional[str]:
    """Search OpenReview for a paper and return its PDF URL."""
    if not title:
        return None
    q = requests.utils.quote(title)
    r = _get(f"https://api2.openreview.net/notes/search?term={q}&limit=5&source=forum&sort=cdate:desc")
    if not r:
        return None
    try:
        notes = r.json().get("notes") or []
        for note in notes:
            note_title = (note.get("content", {}).get("title", {}).get("value") or
                          note.get("content", {}).get("title") or "")
            if isinstance(note_title, dict):
                note_title = note_title.get("value", "")
            t_words = set(title.lower().split())
            n_words = set(str(note_title).lower().split())
            if t_words and len(t_words & n_words) / len(t_words) >= 0.65:
                forum_id = note.get("forum") or note.get("id")
                if forum_id:
                    return f"https://openreview.net/pdf?id={forum_id}"
    except Exception as e:
        logger.debug(f"OpenReview error: {e}")
    return None


# ─────────────────────────────────────────────────────────
# SOURCE 11: Zenodo
# ─────────────────────────────────────────────────────────

def _zenodo_pdf(doi: str, title: str) -> Optional[str]:
    """Search Zenodo for open-access record; return PDF URL if available."""
    query = doi if doi else title
    if not query:
        return None
    q = requests.utils.quote(query)
    r = _get(f"https://zenodo.org/api/records?q={q}&size=3&sort=bestmatch")
    if not r:
        return None
    try:
        hits = r.json().get("hits", {}).get("hits", [])
        for hit in hits:
            # Check title match if querying by title
            if not doi:
                hit_title = (hit.get("metadata", {}).get("title") or "").lower()
                t_words = set(title.lower().split())
                if not t_words or len(t_words & set(hit_title.split())) / len(t_words) < 0.6:
                    continue
            for f in hit.get("files", []):
                if f.get("type") == "pdf" or f.get("key", "").endswith(".pdf"):
                    return f.get("links", {}).get("self") or f.get("links", {}).get("download")
    except Exception as e:
        logger.debug(f"Zenodo error: {e}")
    return None


# ─────────────────────────────────────────────────────────
# SOURCE 12: bioRxiv / medRxiv preprints
# ─────────────────────────────────────────────────────────

def _biorxiv_pdf(doi: str, title: str) -> Optional[str]:
    """Check bioRxiv/medRxiv for a preprint version; return PDF URL."""
    # If DOI is already a biorxiv/medrxiv DOI (10.1101/...)
    if doi and doi.startswith("10.1101/"):
        return f"https://www.biorxiv.org/content/{doi}v1.full.pdf"

    if not title:
        return None
    # bioRxiv search API
    q = requests.utils.quote(title)
    r = _get(f"https://api.biorxiv.org/details/biorxiv/{q}/0/5/json", timeout=15)
    if r:
        try:
            collection = r.json().get("collection") or []
            for item in collection:
                item_title = (item.get("title") or "").lower()
                t_words = set(title.lower().split())
                if t_words and len(t_words & set(item_title.split())) / len(t_words) >= 0.7:
                    biorxiv_doi = item.get("doi")
                    if biorxiv_doi:
                        return f"https://www.biorxiv.org/content/{biorxiv_doi}v1.full.pdf"
        except Exception as e:
            logger.debug(f"bioRxiv error: {e}")
    return None


# ─────────────────────────────────────────────────────────
# SOURCE 13: Google Scholar — best-effort PDF link scrape
# ─────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────
# SOURCE 13: Google Scholar via Playwright (real browser)
# ─────────────────────────────────────────────────────────

def _playwright_scholar_pdf(doi: str, title: str, pdf_path: Path) -> bool:
    """Use a real Chromium browser (Playwright) to find and download a PDF
    from Google Scholar.

    Why this works when requests-based scraping fails:
    - Playwright executes JavaScript, so Scholar renders its full results page
    - The browser looks identical to a real user visit (headers, TLS, timing)
    - Google Scholar [PDF] links are found in the rendered DOM, not the raw HTML
    - The PDF is then downloaded via the browser, inheriting any session cookies

    Search order: DOI first (exact match), then quoted title as fallback.
    Returns True and saves the PDF to pdf_path if successful.
    Falls back silently if Playwright is not installed.
    """
    try:
        from playwright.sync_api import sync_playwright, TimeoutError as PWTimeout
    except ImportError:
        logger.debug("Playwright not installed — skipping browser-based Scholar fetch")
        return False

    def _search_and_grab(query: str, browser) -> Optional[str]:
        """Open Scholar search page and return first real PDF URL found."""
        page = browser.new_page()
        try:
            q = requests.utils.quote(query)
            url = f"https://scholar.google.com/scholar?q={q}&hl=en&as_sdt=0%2C5"
            page.goto(url, wait_until="domcontentloaded", timeout=25000)
            page.wait_for_timeout(2000)  # let JS finish rendering

            # Collect all href values from the rendered page
            hrefs = page.eval_on_selector_all(
                "a[href]", "els => els.map(e => e.getAttribute('href'))"
            )
            for href in hrefs:
                if not href or "google.com" in href or "scholar.google" in href:
                    continue
                href_lower = href.lower()
                if href.endswith(".pdf") or "/pdf/" in href_lower or "full.pdf" in href_lower:
                    return href
        except PWTimeout:
            logger.debug(f"Playwright Scholar timeout for: {query[:60]}")
        except Exception as e:
            logger.debug(f"Playwright Scholar page error: {e}")
        finally:
            page.close()
        return None

    def _browser_download_pdf(pdf_url: str, browser) -> bool:
        """Download a PDF URL using the browser (handles redirects/auth)."""
        page = browser.new_page()
        try:
            # Use the browser's built-in download capability
            with page.expect_download(timeout=30000) as dl_info:
                page.goto(pdf_url, timeout=30000)
            download = dl_info.value
            download.save_as(str(pdf_path))
            # Verify it's a real PDF
            if pdf_path.exists() and pdf_path.stat().st_size > 1000:
                content = pdf_path.read_bytes()[:5]
                if content.startswith(b"%PDF"):
                    return True
                pdf_path.unlink(missing_ok=True)
        except Exception:
            # Page loaded without triggering a download → try reading content directly
            try:
                resp = page.request.get(pdf_url)
                if resp.status == 200:
                    body = resp.body()
                    if body[:4] == b"%PDF":
                        pdf_path.write_bytes(body)
                        return True
            except Exception as e2:
                logger.debug(f"Playwright PDF fetch error: {e2}")
        finally:
            page.close()
        return False

    try:
        with sync_playwright() as pw:
            browser = pw.chromium.launch(
                headless=True,
                args=["--no-sandbox", "--disable-dev-shm-usage"],
            )
            try:
                # Pass 1: DOI search (most precise)
                pdf_url = None
                if doi:
                    pdf_url = _search_and_grab(doi, browser)

                # Pass 2: quoted title fallback
                if not pdf_url and title:
                    pdf_url = _search_and_grab(f'"{title}"', browser)

                if pdf_url:
                    logger.info(f"[Playwright/Scholar] found PDF → {pdf_url[:70]}")
                    # First try direct download via requests (faster)
                    if _download_pdf(pdf_url, pdf_path):
                        return True
                    # Fallback: use the browser to download (handles JS redirects)
                    return _browser_download_pdf(pdf_url, browser)
            finally:
                browser.close()
    except Exception as e:
        logger.debug(f"Playwright launch error: {e}")

    return False


def _google_scholar_pdf(doi: str, title: str) -> Optional[str]:
    """Scrape Google Scholar search results for a direct PDF link.

    Search order:
      1. By DOI  — most precise; Scholar resolves DOIs to the exact paper and
                   shows a [PDF] badge whenever a free copy is available
                   (publisher OA, institutional repo, ResearchGate, etc.)
      2. By title — fallback when DOI is absent

    Google sometimes returns 429/captcha; the function fails silently then.
    """
    if not doi and not title:
        return None

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        ),
        "Accept-Language": "en-US,en;q=0.9",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    }

    def _scrape(query_str: str) -> Optional[str]:
        q = requests.utils.quote(query_str)
        url = f"https://scholar.google.com/scholar?q={q}&hl=en&as_sdt=0%2C5"
        try:
            r = requests.get(url, headers=headers, timeout=20)
            if r.status_code != 200:
                logger.debug(f"Google Scholar returned {r.status_code} for: {query_str[:60]}")
                return None
            html = r.text
            # Primary: links ending in .pdf
            pdf_links = re.findall(r'href="(https?://[^"]+\.pdf[^"]*)"', html, re.IGNORECASE)
            if not pdf_links:
                # Secondary: URLs containing /pdf/ path segments
                pdf_links = re.findall(
                    r'href="(https?://[^"]*(?:/pdf/|/PDF/|full\.pdf|fulltext\.pdf)[^"]*)"',
                    html, re.IGNORECASE,
                )
            for link in pdf_links[:5]:
                if "google.com" not in link and "scholar.google" not in link:
                    return link
        except Exception as e:
            logger.debug(f"Google Scholar scrape error: {e}")
        return None

    # Pass 1: DOI search — highest precision
    if doi:
        result = _scrape(doi)
        if result:
            return result

    # Pass 2: quoted title search — fallback
    if title:
        time.sleep(_RATE_LIMIT_SECS)
        result = _scrape(f'"{title}"')
        if result:
            return result

    return None


# ─────────────────────────────────────────────────────────
# SOURCE 14: BASE — Bielefeld Academic Search Engine
# ─────────────────────────────────────────────────────────

def _base_search_pdf(title: str) -> Optional[str]:
    """Search BASE (Bielefeld Academic Search Engine) for an open-access PDF.

    BASE aggregates >300 million documents from open-access repositories
    worldwide.  It has a public JSON API with no API key required.
    """
    if not title:
        return None
    # Use the first 8 significant words to keep the query focused
    words = [w for w in re.sub(r"[^\w\s]", " ", title).split() if len(w) >= 3][:8]
    if not words:
        return None
    q = requests.utils.quote(" ".join(words))
    url = (
        "https://api.base-search.net/cgi-bin/BaseHttpSearchInterface.fcgi"
        f"?func=PerformSearch&query=dctitle:{q}"
        "&hits=5&offset=0&format=json&boost=oa"
    )
    r = _get(url, timeout=25)
    if not r:
        return None
    try:
        docs = r.json().get("response", {}).get("docs") or []
        for doc in docs:
            # Verify title similarity
            doc_title = " ".join(doc.get("dctitle") or []).lower() if isinstance(
                doc.get("dctitle"), list) else (doc.get("dctitle") or "").lower()
            t_words = set(title.lower().split())
            d_words = set(doc_title.split())
            if not t_words or len(t_words & d_words) / len(t_words) < 0.45:
                continue
            # dclinktype == "1" means open-access full text
            links = doc.get("dclink") or []
            if isinstance(links, str):
                links = [links]
            for link in links:
                if link and link.startswith("http"):
                    return link
    except Exception as e:
        logger.debug(f"BASE search error: {e}")
    return None


# ─────────────────────────────────────────────────────────
# SOURCE (old 13): CrossRef DOI resolution → landing page
# ─────────────────────────────────────────────────────────

def _crossref_landing_url(doi: str) -> Optional[str]:
    """Use CrossRef to get the canonical landing page URL for a DOI."""
    if not doi:
        return None
    r = _get(f"https://api.crossref.org/works/{requests.utils.quote(doi)}"
             f"?mailto={CONTACT_EMAIL}")
    if not r:
        return None
    try:
        msg = r.json().get("message", {})
        # Direct URL or resource link
        url = msg.get("URL") or msg.get("resource", {}).get("primary", {}).get("URL")
        if url and url != f"https://doi.org/{doi}":
            return url
        # links field may contain open-access PDF/HTML
        for link in msg.get("link", []):
            content_type = link.get("content-type", "")
            if "pdf" in content_type:
                return link.get("URL")
        for link in msg.get("link", []):
            if link.get("URL"):
                return link["URL"]
    except Exception as e:
        logger.debug(f"CrossRef error: {e}")
    return None


# ─────────────────────────────────────────────────────────
# SOURCE 14: PubMed title search → PMCID
# ─────────────────────────────────────────────────────────

def _pubmed_pmcid(doi: str, title: str) -> Optional[str]:
    """Search PubMed by DOI or title and return PMCID if open-access available."""
    if doi:
        query = f"{doi}[DOI]"
    elif title:
        # Use the first 10 words to reduce false matches
        short = " ".join(title.split()[:10])
        query = f"{short}[Title]"
    else:
        return None
    q = requests.utils.quote(query)
    # eSearch to get PMID
    r = _get(f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
             f"?db=pubmed&term={q}&retmax=3&retmode=json&tool=SysReviewBot&email={CONTACT_EMAIL}")
    if not r:
        return None
    try:
        ids = r.json().get("esearchresult", {}).get("idlist", [])
        if not ids:
            return None
        pmid = ids[0]
        # eLink: check if this PMID has a PMC full-text link
        rl = _get(f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/elink.fcgi"
                  f"?dbfrom=pubmed&db=pmc&id={pmid}&retmode=json"
                  f"&tool=SysReviewBot&email={CONTACT_EMAIL}")
        if rl:
            linksets = rl.json().get("linksets", [])
            for ls in linksets:
                for ld in ls.get("linksetdbs", []):
                    if ld.get("dbto") == "pmc":
                        pmc_ids = ld.get("links", [])
                        if pmc_ids:
                            return f"PMC{pmc_ids[0]}"
    except Exception as e:
        logger.debug(f"PubMed error: {e}")
    return None


# ─────────────────────────────────────────────────────────
# PUBLIC API
# ─────────────────────────────────────────────────────────

def try_download_fulltext(
    doi: str,
    title: str,
    record_id: str,
    pdf_dir: Path,
    record_url: str = "",
) -> Tuple[bool, str]:
    """
    Try to obtain the full text for one record by any available means.

    Strategy:
      1.  Unpaywall PDF
      2.  Semantic Scholar PDF
      3.  OpenAlex PDF
      4.  PMC HTML (via Semantic Scholar PMCID or Europe PMC search)
      5.  Europe PMC XML/HTML
      6.  Unpaywall / OpenAlex HTML landing page
      7.  arXiv PDF (exact + keyword title search)
      8.  CORE open-access aggregator
      9.  Direct record URL
      10. ACL Anthology PDF (all NLP/CL papers)
      11. OpenReview PDF (ICLR, NeurIPS, ICML)
      12. Zenodo open-access repository
      13. bioRxiv / medRxiv preprints
      14. Google Scholar [PDF] badge scrape (best-effort)
      15. BASE search engine (300M+ open-access docs)
      16. CrossRef DOI → landing page / PDF link
      17. PubMed title search → PMCID → PMC full text

    Saves result as <record_id>.pdf  (real PDF)
                 or <record_id>.txt  (plain text extracted from HTML/XML)

    Returns:
      (True,  source_name)   if full text was obtained
      (False, "not_found")   if nothing worked
    """
    doi = (doi or "").strip()
    title = (title or "").strip()
    record_url = (record_url or "").strip()

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
            text = _fetch_html_text(html_url, min_chars=1000)
            if text:
                txt_path.write_text(text, encoding="utf-8")
                return True, source

    # ── Step 5: arXiv (title search — highly effective for CS/AI papers) ────
    time.sleep(_RATE_LIMIT_SECS)
    arxiv_pdf = _arxiv_pdf_url(doi, title)
    if arxiv_pdf:
        logger.info(f"[arXiv] PDF → {arxiv_pdf[:70]}")
        time.sleep(_RATE_LIMIT_SECS)
        if _download_pdf(arxiv_pdf, pdf_path):
            return True, "arXiv_pdf"
        # arXiv also serves HTML — try the abs page text
        abs_url = arxiv_pdf.replace("pdf", "abs", 1)
        text = _fetch_html_text(abs_url, min_chars=1000)
        if text:
            txt_path.write_text(text, encoding="utf-8")
            return True, "arXiv_html"

    # ── Step 6: CORE open-access aggregator ─────────────────────────────────
    time.sleep(_RATE_LIMIT_SECS)
    core_url = _core_download_url(doi, title)
    if core_url:
        logger.info(f"[CORE] → {core_url[:70]}")
        time.sleep(_RATE_LIMIT_SECS)
        if core_url.endswith(".pdf") or "pdf" in core_url.lower():
            if _download_pdf(core_url, pdf_path):
                return True, "CORE_pdf"
        text = _fetch_html_text(core_url, min_chars=1000)
        if text:
            txt_path.write_text(text, encoding="utf-8")
            return True, "CORE_html"

    # ── Step 7: Direct URL from record ──────────────────────────────────────
    if record_url and record_url.startswith("http"):
        logger.info(f"[DirectURL] → {record_url[:70]}")
        time.sleep(_RATE_LIMIT_SECS)
        if record_url.endswith(".pdf"):
            if _download_pdf(record_url, pdf_path):
                return True, "direct_pdf"
        text = _fetch_html_text(record_url, min_chars=1000)
        if text:
            txt_path.write_text(text, encoding="utf-8")
            return True, "direct_html"

    # ── Step 8: ACL Anthology (all NLP/CL conference papers free) ───────────
    time.sleep(_RATE_LIMIT_SECS)
    acl_pdf = _acl_anthology_pdf(doi, title)
    if acl_pdf:
        logger.info(f"[ACL] PDF → {acl_pdf[:70]}")
        time.sleep(_RATE_LIMIT_SECS)
        if _download_pdf(acl_pdf, pdf_path):
            return True, "ACL_pdf"

    # ── Step 9: OpenReview (ICLR, NeurIPS, ICML, ICLR workshops) ────────────
    time.sleep(_RATE_LIMIT_SECS)
    or_pdf = _openreview_pdf(title)
    if or_pdf:
        logger.info(f"[OpenReview] PDF → {or_pdf[:70]}")
        time.sleep(_RATE_LIMIT_SECS)
        if _download_pdf(or_pdf, pdf_path):
            return True, "OpenReview_pdf"

    # ── Step 10: Zenodo open-access repository ───────────────────────────────
    time.sleep(_RATE_LIMIT_SECS)
    zen_pdf = _zenodo_pdf(doi, title)
    if zen_pdf:
        logger.info(f"[Zenodo] PDF → {zen_pdf[:70]}")
        time.sleep(_RATE_LIMIT_SECS)
        if _download_pdf(zen_pdf, pdf_path):
            return True, "Zenodo_pdf"

    # ── Step 11: bioRxiv / medRxiv preprint ─────────────────────────────────
    time.sleep(_RATE_LIMIT_SECS)
    biorxiv_pdf = _biorxiv_pdf(doi, title)
    if biorxiv_pdf:
        logger.info(f"[bioRxiv] PDF → {biorxiv_pdf[:70]}")
        time.sleep(_RATE_LIMIT_SECS)
        if _download_pdf(biorxiv_pdf, pdf_path):
            return True, "bioRxiv_pdf"

    # ── Step 12: Google Scholar via real Chromium browser (Playwright) ──────
    # Playwright renders JS, bypasses bot detection, and finds [PDF] links
    # that the plain requests-based scraper cannot see.
    if _playwright_scholar_pdf(doi, title, pdf_path):
        return True, "GoogleScholar_playwright"

    # ── Step 12b: Google Scholar — requests-based scrape fallback ────────────
    # Used when Playwright is not installed or returns no result.
    time.sleep(_RATE_LIMIT_SECS)
    gs_pdf = _google_scholar_pdf(doi, title)
    if gs_pdf:
        logger.info(f"[GoogleScholar] PDF → {gs_pdf[:70]}")
        time.sleep(_RATE_LIMIT_SECS)
        if _download_pdf(gs_pdf, pdf_path):
            return True, "GoogleScholar_pdf"
        text = _fetch_html_text(gs_pdf, min_chars=1000)
        if text:
            txt_path.write_text(text, encoding="utf-8")
            return True, "GoogleScholar_html"

    # ── Step 12b: BASE (Bielefeld Academic Search Engine) ─────────────────────
    time.sleep(_RATE_LIMIT_SECS)
    base_url = _base_search_pdf(title)
    if base_url:
        logger.info(f"[BASE] → {base_url[:70]}")
        time.sleep(_RATE_LIMIT_SECS)
        if base_url.lower().endswith(".pdf") or "pdf" in base_url.lower():
            if _download_pdf(base_url, pdf_path):
                return True, "BASE_pdf"
        text = _fetch_html_text(base_url, min_chars=1000)
        if text:
            txt_path.write_text(text, encoding="utf-8")
            return True, "BASE_html"

    # ── Step 13: CrossRef → landing page HTML ────────────────────────────────
    time.sleep(_RATE_LIMIT_SECS)
    cr_url = _crossref_landing_url(doi)
    if cr_url:
        logger.info(f"[CrossRef] → {cr_url[:70]}")
        time.sleep(_RATE_LIMIT_SECS)
        if cr_url.lower().endswith(".pdf"):
            if _download_pdf(cr_url, pdf_path):
                return True, "CrossRef_pdf"
        text = _fetch_html_text(cr_url, min_chars=1000)
        if text:
            txt_path.write_text(text, encoding="utf-8")
            return True, "CrossRef_html"

    # ── Step 13: PubMed title search → new PMCID not found above ─────────────
    if not pmcid:  # only if Steps 3-4 didn't find one
        time.sleep(_RATE_LIMIT_SECS)
        pmcid2 = _pubmed_pmcid(doi, title)
        if pmcid2:
            logger.info(f"[PubMed] PMCID → {pmcid2}")
            time.sleep(_RATE_LIMIT_SECS)
            text = _pmc_html_text(pmcid2)
            if text and len(text) >= 1000:
                txt_path.write_text(text, encoding="utf-8")
                return True, "PubMed_PMC"

    return False, "not_found"
