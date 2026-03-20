"""
Automatic full-text downloader for systematic review records.

Tries, in order:
  1. Unpaywall API  (open-access PDF by DOI)
  2. Semantic Scholar API (openAccessPdf field)
  3. OpenAlex API (best open-access location)
  4. Europe PMC  (PubMed Central full-text)

Records that cannot be auto-fetched are marked FULL_TEXT_NEEDED so the
researcher can download and upload them manually.
"""
import logging
import time
from pathlib import Path
from typing import Optional, Tuple

import requests

logger = logging.getLogger(__name__)

# Respect rate limits – pause between requests per source
_RATE_LIMIT_SECS = 0.5

# Configurable contact email for Unpaywall & OpenAlex (polite pool)
CONTACT_EMAIL = "systematic.review@research.org"

REQUEST_HEADERS = {
    "User-Agent": (
        "SystematicReviewBot/1.0 "
        "(automated full-text retrieval for academic research; "
        f"mailto:{CONTACT_EMAIL})"
    ),
    "Accept": "application/json",
}


# ─────────────────────────────────────────────────────────
# SOURCE 1: Unpaywall
# ─────────────────────────────────────────────────────────

def _unpaywall_pdf_url(doi: str) -> Optional[str]:
    """Return best open-access PDF URL from Unpaywall, or None."""
    if not doi:
        return None
    try:
        url = f"https://api.unpaywall.org/v2/{doi}?email={CONTACT_EMAIL}"
        r = requests.get(url, headers=REQUEST_HEADERS, timeout=15)
        if r.status_code == 200:
            data = r.json()
            best = data.get("best_oa_location") or {}
            return best.get("url_for_pdf") or best.get("url_for_landing_page")
    except Exception as e:
        logger.debug(f"Unpaywall error ({doi}): {e}")
    return None


# ─────────────────────────────────────────────────────────
# SOURCE 2: Semantic Scholar
# ─────────────────────────────────────────────────────────

def _semantic_scholar_pdf_url(doi: str, title: str) -> Optional[str]:
    """Return open-access PDF URL from Semantic Scholar, or None."""
    try:
        if doi:
            url = f"https://api.semanticscholar.org/graph/v1/paper/DOI:{doi}?fields=openAccessPdf,externalIds"
        elif title:
            # Fallback: search by title
            url = f"https://api.semanticscholar.org/graph/v1/paper/search?query={requests.utils.quote(title)}&fields=openAccessPdf&limit=1"
        else:
            return None

        r = requests.get(url, headers=REQUEST_HEADERS, timeout=15)
        if r.status_code == 200:
            data = r.json()
            # Handle search result (list) vs direct lookup (dict)
            if "data" in data:
                data = data["data"][0] if data["data"] else {}
            oa = data.get("openAccessPdf") or {}
            return oa.get("url")
    except Exception as e:
        logger.debug(f"Semantic Scholar error ({doi}): {e}")
    return None


# ─────────────────────────────────────────────────────────
# SOURCE 3: OpenAlex
# ─────────────────────────────────────────────────────────

def _openalex_pdf_url(doi: str) -> Optional[str]:
    """Return best open-access PDF URL from OpenAlex, or None."""
    if not doi:
        return None
    try:
        url = f"https://api.openalex.org/works/doi:{doi}?mailto={CONTACT_EMAIL}"
        r = requests.get(url, headers=REQUEST_HEADERS, timeout=15)
        if r.status_code == 200:
            data = r.json()
            # Check primary OA URL
            oa = data.get("open_access", {})
            oa_url = oa.get("oa_url")
            if oa_url:
                return oa_url
            # Check per-location PDF URLs
            for loc in data.get("locations", []):
                if loc.get("pdf_url"):
                    return loc["pdf_url"]
    except Exception as e:
        logger.debug(f"OpenAlex error ({doi}): {e}")
    return None


# ─────────────────────────────────────────────────────────
# SOURCE 4: Europe PMC
# ─────────────────────────────────────────────────────────

def _europepmc_pdf_url(doi: str, title: str) -> Optional[str]:
    """Return Europe PMC full-text XML/PDF URL if available, or None."""
    try:
        query = doi if doi else title
        url = (
            f"https://www.ebi.ac.uk/europepmc/webservices/rest/search"
            f"?query={requests.utils.quote(query)}&format=json&resulttype=core&pageSize=1"
        )
        r = requests.get(url, headers=REQUEST_HEADERS, timeout=15)
        if r.status_code == 200:
            results = r.json().get("resultList", {}).get("result", [])
            if results:
                result = results[0]
                pmcid = result.get("pmcid")
                if pmcid and result.get("isOpenAccess") == "Y":
                    return f"https://europepmc.org/backend/ptpmcrender.fcgi?accid={pmcid}&blobtype=pdf"
    except Exception as e:
        logger.debug(f"EuropePMC error ({doi}): {e}")
    return None


# ─────────────────────────────────────────────────────────
# DOWNLOAD
# ─────────────────────────────────────────────────────────

def _download_and_verify(pdf_url: str, save_path: Path) -> bool:
    """Download URL to save_path. Returns True only if a valid PDF was saved."""
    try:
        headers = {**REQUEST_HEADERS, "Accept": "application/pdf,*/*"}
        r = requests.get(pdf_url, headers=headers, timeout=45, stream=True)
        if r.status_code != 200:
            return False

        # Stream to a temp buffer to check PDF magic bytes
        chunks = []
        total = 0
        for chunk in r.iter_content(chunk_size=16384):
            chunks.append(chunk)
            total += len(chunk)
            if total > 20_000_000:  # 20 MB cap
                logger.warning(f"PDF too large (>20 MB), skipping: {pdf_url}")
                return False

        content = b"".join(chunks)
        # Must start with PDF magic bytes or be HTML fallback we can use
        if not content.startswith(b"%PDF"):
            # Some repositories return HTML landing pages — reject
            logger.debug(f"Not a PDF (no magic bytes): {pdf_url}")
            return False

        save_path.write_bytes(content)
        logger.info(f"Downloaded {len(content):,} bytes → {save_path.name}")
        return True

    except Exception as e:
        logger.debug(f"Download failed ({pdf_url}): {e}")
        return False


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
    Try to automatically obtain the full text for one record.

    Strategy (in order):
      Unpaywall → Semantic Scholar → OpenAlex → Europe PMC

    Returns:
      (True, source_name)  if a valid PDF was saved
      (False, "not_found") if no open-access copy could be located
    """
    save_path = pdf_dir / f"{record_id}.pdf"
    if save_path.exists() and save_path.stat().st_size > 1000:
        logger.debug(f"PDF already exists: {save_path.name}")
        return True, "cached"

    doi = (doi or "").strip()
    title = (title or "").strip()

    sources = [
        ("Unpaywall",       lambda: _unpaywall_pdf_url(doi)),
        ("SemanticScholar", lambda: _semantic_scholar_pdf_url(doi, title)),
        ("OpenAlex",        lambda: _openalex_pdf_url(doi)),
        ("EuropePMC",       lambda: _europepmc_pdf_url(doi, title)),
    ]

    for source_name, get_url in sources:
        time.sleep(_RATE_LIMIT_SECS)
        try:
            pdf_url = get_url()
            if not pdf_url:
                continue
            logger.info(f"[{source_name}] {record_id[:8]} → {pdf_url[:80]}")
            if _download_and_verify(pdf_url, save_path):
                return True, source_name
        except Exception as e:
            logger.warning(f"[{source_name}] error for {record_id[:8]}: {e}")

    return False, "not_found"
