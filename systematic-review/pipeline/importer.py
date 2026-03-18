"""
Import module: loads search results from multiple formats into RawRecord objects.
Supported formats: RIS, CSV, BibTeX, PubMed XML, JSON.
"""
import json
import csv
import re
import uuid
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime

try:
    import rispy
    HAS_RISPY = True
except ImportError:
    HAS_RISPY = False

try:
    import bibtexparser
    HAS_BIBTEX = True
except ImportError:
    HAS_BIBTEX = False

try:
    import xml.etree.ElementTree as ET
    HAS_XML = True
except ImportError:
    HAS_XML = False

from .models import RawRecord


# RIS tag mapping to standard fields
RIS_FIELD_MAP = {
    "TI": "title",
    "T1": "title",
    "AU": "authors",
    "A1": "authors",
    "PY": "year",
    "Y1": "year",
    "JO": "journal_venue",
    "JF": "journal_venue",
    "T2": "journal_venue",
    "AB": "abstract",
    "N2": "abstract",
    "DO": "doi",
    "UR": "url",
    "KW": "keywords",
    "DB": "source_db",
}


def import_ris(filepath: str, source_db: str = "Unknown") -> List[RawRecord]:
    """Import from RIS format."""
    if not HAS_RISPY:
        raise ImportError("rispy not installed. Run: pip install rispy")

    records = []
    with open(filepath, "r", encoding="utf-8") as f:
        entries = rispy.load(f)

    for entry in entries:
        title = entry.get("title") or entry.get("primary_title") or ""
        if not title:
            continue

        authors_list = entry.get("authors") or entry.get("first_authors") or []
        authors = "; ".join(authors_list) if isinstance(authors_list, list) else str(authors_list)

        year_raw = entry.get("year") or entry.get("publication_year")
        try:
            year = int(str(year_raw)[:4]) if year_raw else None
        except (ValueError, TypeError):
            year = None

        abstract = entry.get("abstract") or ""
        keywords_list = entry.get("keywords") or []
        keywords = "; ".join(keywords_list) if isinstance(keywords_list, list) else str(keywords_list)

        record = RawRecord(
            record_id=str(uuid.uuid4()),
            source_db=entry.get("name_of_database") or source_db,
            title=title.strip(),
            authors=authors,
            year=year,
            journal_venue=entry.get("journal_name") or entry.get("secondary_title") or "",
            doi=entry.get("doi") or "",
            abstract=abstract,
            keywords=keywords,
            url=entry.get("url") or "",
            raw_data=dict(entry),
        )
        records.append(record)

    return records


def import_csv(filepath: str, source_db: str = "Unknown",
               column_map: Optional[Dict[str, str]] = None) -> List[RawRecord]:
    """
    Import from CSV format.
    column_map: maps CSV column names to RawRecord fields.
    Default assumes standard Scopus/PubMed CSV column names.
    """
    default_map = {
        # Scopus defaults
        "Title": "title",
        "Authors": "authors",
        "Year": "year",
        "Source title": "journal_venue",
        "DOI": "doi",
        "Abstract": "abstract",
        "Author Keywords": "keywords",
        "Link": "url",
        # PubMed defaults
        "Article Title": "title",
        "Publication Year": "year",
        "Journal/Book": "journal_venue",
        "Author Names": "authors",
    }
    col_map = column_map or default_map

    records = []
    with open(filepath, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            mapped = {col_map[k]: v for k, v in row.items() if k in col_map and v}

            title = mapped.get("title", "").strip()
            if not title:
                continue

            year_raw = mapped.get("year")
            try:
                year = int(str(year_raw)[:4]) if year_raw else None
            except (ValueError, TypeError):
                year = None

            record = RawRecord(
                record_id=str(uuid.uuid4()),
                source_db=source_db,
                title=title,
                authors=mapped.get("authors"),
                year=year,
                journal_venue=mapped.get("journal_venue"),
                doi=mapped.get("doi"),
                abstract=mapped.get("abstract"),
                keywords=mapped.get("keywords"),
                url=mapped.get("url"),
                raw_data=dict(row),
            )
            records.append(record)

    return records


def import_bibtex(filepath: str, source_db: str = "Unknown") -> List[RawRecord]:
    """Import from BibTeX format."""
    if not HAS_BIBTEX:
        raise ImportError("bibtexparser not installed. Run: pip install bibtexparser")

    with open(filepath, "r", encoding="utf-8") as f:
        bib_db = bibtexparser.load(f)

    records = []
    for entry in bib_db.entries:
        title = entry.get("title", "").strip("{}")
        if not title:
            continue

        year_raw = entry.get("year")
        try:
            year = int(year_raw) if year_raw else None
        except (ValueError, TypeError):
            year = None

        record = RawRecord(
            record_id=str(uuid.uuid4()),
            source_db=source_db,
            title=title,
            authors=entry.get("author"),
            year=year,
            journal_venue=entry.get("journal") or entry.get("booktitle"),
            doi=entry.get("doi"),
            abstract=entry.get("abstract"),
            keywords=entry.get("keywords"),
            url=entry.get("url"),
            raw_data=entry,
        )
        records.append(record)

    return records


def import_json(filepath: str, source_db: str = "Unknown") -> List[RawRecord]:
    """Import from JSON array format (custom or exported)."""
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        data = [data]

    records = []
    for item in data:
        title = item.get("title", "").strip()
        if not title:
            continue

        year_raw = item.get("year") or item.get("publication_year")
        try:
            year = int(str(year_raw)[:4]) if year_raw else None
        except (ValueError, TypeError):
            year = None

        record = RawRecord(
            record_id=item.get("record_id", str(uuid.uuid4())),
            source_db=item.get("source_db", source_db),
            title=title,
            authors=item.get("authors"),
            year=year,
            journal_venue=item.get("journal_venue") or item.get("journal"),
            doi=item.get("doi"),
            abstract=item.get("abstract"),
            keywords=item.get("keywords"),
            url=item.get("url"),
            raw_data=item,
        )
        records.append(record)

    return records


def import_pubmed_xml(filepath: str) -> List[RawRecord]:
    """Import from PubMed XML export format."""
    tree = ET.parse(filepath)
    root = tree.getroot()
    records = []

    for article in root.findall(".//PubmedArticle"):
        medline = article.find("MedlineCitation")
        if medline is None:
            continue

        art = medline.find("Article")
        if art is None:
            continue

        title_el = art.find("ArticleTitle")
        title = "".join(title_el.itertext()).strip() if title_el is not None else ""
        if not title:
            continue

        abstract_el = art.find(".//AbstractText")
        abstract = "".join(abstract_el.itertext()).strip() if abstract_el is not None else ""

        author_list = art.find("AuthorList")
        authors = []
        if author_list is not None:
            for author in author_list.findall("Author"):
                last = author.findtext("LastName") or ""
                fore = author.findtext("ForeName") or ""
                authors.append(f"{last}, {fore}".strip(", "))
        authors_str = "; ".join(authors)

        journal_el = art.find("Journal")
        journal_name = ""
        year = None
        if journal_el is not None:
            journal_name = journal_el.findtext("Title") or journal_el.findtext("ISOAbbreviation") or ""
            pub_date = journal_el.find(".//PubDate")
            if pub_date is not None:
                year_str = pub_date.findtext("Year")
                try:
                    year = int(year_str) if year_str else None
                except ValueError:
                    year = None

        pmid = medline.findtext("PMID")
        doi = ""
        for id_el in article.findall(".//ArticleId"):
            if id_el.get("IdType") == "doi":
                doi = id_el.text or ""

        keywords = []
        for kw in medline.findall(".//Keyword"):
            if kw.text:
                keywords.append(kw.text.strip())

        record = RawRecord(
            record_id=str(uuid.uuid4()),
            source_db="PubMed",
            title=title,
            authors=authors_str,
            year=year,
            journal_venue=journal_name,
            doi=doi,
            abstract=abstract,
            keywords="; ".join(keywords),
            url=f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if pmid else "",
            raw_data={"pmid": pmid},
        )
        records.append(record)

    return records


def load_all_from_directory(directory: str) -> List[RawRecord]:
    """
    Auto-detect and import all supported files from a directory.
    Returns combined list of RawRecord objects.
    """
    path = Path(directory)
    all_records: List[RawRecord] = []
    stats = {}

    handlers = {
        ".ris": lambda p: import_ris(str(p), source_db=p.stem),
        ".csv": lambda p: import_csv(str(p), source_db=p.stem),
        ".bib": lambda p: import_bibtex(str(p), source_db=p.stem),
        ".json": lambda p: import_json(str(p), source_db=p.stem),
        ".xml": lambda p: import_pubmed_xml(str(p)),
    }

    for file in sorted(path.iterdir()):
        ext = file.suffix.lower()
        if ext in handlers:
            try:
                records = handlers[ext](file)
                all_records.extend(records)
                stats[file.name] = len(records)
                print(f"  Imported {len(records)} records from {file.name}")
            except Exception as e:
                print(f"  ERROR importing {file.name}: {e}")
                stats[file.name] = f"ERROR: {e}"

    print(f"\nTotal imported: {len(all_records)} records from {len(stats)} files")
    return all_records


def save_records(records: List[RawRecord], output_path: str):
    """Save records as JSON lines file."""
    with open(output_path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(record.model_dump_json() + "\n")


def load_records(filepath: str) -> List[RawRecord]:
    """Load records from JSON lines file."""
    records = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(RawRecord.model_validate_json(line))
    return records
