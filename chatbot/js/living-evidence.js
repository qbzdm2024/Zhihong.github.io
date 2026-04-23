/**
 * Living Evidence Engine
 * Provides real-time PubMed evidence retrieval as a fallback when the local
 * knowledge base has low relevance (score below TFIDF_THRESHOLD).
 *
 * Pipeline:
 *   1. Check 24-hr localStorage cache
 *   2. Extract search terms via LLM
 *   3. Search PubMed E-utilities (esearch → efetch)
 *   4. LLM screening for HF self-management relevance
 *   5. LLM synthesis into plain-language answer with PMID citations
 *   6. Cache result
 *
 * No API key required for PubMed (rate-limited to 3 req/sec without NCBI key).
 */

const LE_PUBMED_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils";
const LE_CACHE_PREFIX = "pubmed_cache_";
const LE_CACHE_TTL_MS = 24 * 60 * 60 * 1000; // 24 hours
const LE_MAX_RESULTS = 5;
const LE_RATE_LIMIT_NO_KEY = 340;   // ms — ~3 req/sec
const LE_RATE_LIMIT_WITH_KEY = 110; // ms — ~10 req/sec (NCBI key)
const LE_FETCH_TIMEOUT_MS = 10000;

class LivingEvidenceEngine {
  /**
   * @param {Object} options
   * @param {ChatEngine} options.chatEngine - reference to the ChatEngine instance
   * @param {string|null} options.ncbiApiKey - optional NCBI E-utilities API key
   */
  constructor({ chatEngine, ncbiApiKey = null }) {
    this.chatEngine = chatEngine;
    this.ncbiApiKey = ncbiApiKey;
    this._lastCallTime = 0;
    this.purgeExpiredCache();
  }

  // ── Public API ──────────────────────────────────────────────────────────────

  /**
   * Query living evidence for a user question.
   * @param {string} userQuestion
   * @returns {Promise<{response:string|null, pmids:string[], usedLivingEvidence:boolean, fromCache:boolean}>}
   */
  async query(userQuestion) {
    try {
      // 1. Check cache
      const cacheKey = this._cacheKey(userQuestion);
      const cached = this._readCache(cacheKey);
      if (cached) {
        console.log("[LivingEvidence] Cache hit");
        return { ...cached, fromCache: true };
      }

      // 2. Extract search terms via LLM
      const terms = await this._extractSearchTerms(userQuestion);
      if (!terms || terms.length === 0) {
        return { response: null, pmids: [], usedLivingEvidence: false, fromCache: false };
      }

      // 3. Search PubMed
      await this._throttle();
      const pmids = await this._esearch(terms);
      if (!pmids || pmids.length === 0) {
        return { response: null, pmids: [], usedLivingEvidence: false, fromCache: false };
      }

      // 4. Fetch abstracts
      await this._throttle();
      const xml = await this._efetchAbstracts(pmids.slice(0, LE_MAX_RESULTS));
      const abstracts = this._parseAbstractsXml(xml);
      if (abstracts.length === 0) {
        return { response: null, pmids: [], usedLivingEvidence: false, fromCache: false };
      }

      // 5. LLM screening
      const relevantPmids = await this._screenAbstracts(abstracts, userQuestion);
      if (relevantPmids.length === 0) {
        return { response: null, pmids: [], usedLivingEvidence: false, fromCache: false };
      }
      const relevantAbstracts = abstracts.filter((a) => relevantPmids.includes(a.pmid));

      // 6. LLM synthesis
      const response = await this._synthesizeEvidence(relevantAbstracts, userQuestion);
      if (!response) {
        return { response: null, pmids: [], usedLivingEvidence: false, fromCache: false };
      }

      const result = { response, pmids: relevantPmids, usedLivingEvidence: true, fromCache: false };

      // 7. Write cache
      this._writeCache(cacheKey, { response, pmids: relevantPmids, usedLivingEvidence: true });

      return result;

    } catch (err) {
      console.error("[LivingEvidence] Error:", err.message);
      return { response: null, pmids: [], usedLivingEvidence: false, fromCache: false };
    }
  }

  // ── Search Term Extraction ──────────────────────────────────────────────────

  async _extractSearchTerms(question) {
    const prompt = `You are a PubMed search assistant. Extract 3-5 PubMed/MeSH search terms for this heart failure patient question. Return ONLY a valid JSON array of strings, no explanation, no markdown.

Question: "${question}"

Example output: ["heart failure self-management","sodium restriction","fluid overload"]`;

    try {
      const resp = await fetch("https://api.openai.com/v1/chat/completions", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "Authorization": `Bearer ${this.chatEngine.apiKey}`
        },
        body: JSON.stringify({
          model: this.chatEngine.model,
          max_tokens: 100,
          temperature: 0,
          messages: [{ role: "user", content: prompt }]
        })
      });
      if (!resp.ok) return null;
      const data = await resp.json();
      const text = data.choices[0]?.message?.content?.trim() || "";
      // Parse JSON array from response
      const match = text.match(/\[[\s\S]*\]/);
      if (!match) return null;
      const terms = JSON.parse(match[0]);
      return Array.isArray(terms) ? terms.filter((t) => typeof t === "string" && t.length > 0) : null;
    } catch (err) {
      console.warn("[LivingEvidence] Search term extraction failed:", err.message);
      return null;
    }
  }

  // ── PubMed E-utilities ──────────────────────────────────────────────────────

  async _esearch(terms) {
    const baseQuery = terms.join(" AND ");
    // Scope to HF self-management to reduce noise
    const scopedQuery = `(${baseQuery}) AND (heart failure[MeSH] OR "heart failure") AND (self-management OR "patient education" OR "home care" OR "self care" OR guideline)`;

    const url = new URL(`${LE_PUBMED_BASE}/esearch.fcgi`);
    url.searchParams.set("db", "pubmed");
    url.searchParams.set("term", scopedQuery);
    url.searchParams.set("retmax", "10");
    url.searchParams.set("sort", "relevance");
    url.searchParams.set("retmode", "json");
    if (this.ncbiApiKey) url.searchParams.set("api_key", this.ncbiApiKey);

    const resp = await this._fetchWithTimeout(url.toString());
    if (!resp.ok) throw new Error(`PubMed esearch HTTP ${resp.status}`);
    const json = await resp.json();
    return json.esearchresult?.idlist ?? [];
  }

  async _efetchAbstracts(pmids) {
    const url = new URL(`${LE_PUBMED_BASE}/efetch.fcgi`);
    url.searchParams.set("db", "pubmed");
    url.searchParams.set("id", pmids.join(","));
    url.searchParams.set("rettype", "abstract");
    url.searchParams.set("retmode", "xml");
    if (this.ncbiApiKey) url.searchParams.set("api_key", this.ncbiApiKey);

    const resp = await this._fetchWithTimeout(url.toString());
    if (!resp.ok) throw new Error(`PubMed efetch HTTP ${resp.status}`);
    return await resp.text();
  }

  _parseAbstractsXml(xml) {
    try {
      const parser = new DOMParser();
      const doc = parser.parseFromString(xml, "application/xml");
      const articles = doc.querySelectorAll("PubmedArticle");
      return Array.from(articles).map((article) => {
        const pmid = article.querySelector("PMID")?.textContent?.trim() ?? "";
        const title = article.querySelector("ArticleTitle")?.textContent?.trim() ?? "";
        const abstractTexts = Array.from(article.querySelectorAll("AbstractText"))
          .map((el) => el.textContent?.trim())
          .filter(Boolean);
        const abstract = abstractTexts.join(" ");
        const year = article.querySelector("PubDate Year")?.textContent?.trim()
          ?? article.querySelector("PubDate MedlineDate")?.textContent?.slice(0, 4)
          ?? "";
        const journal = article.querySelector("ISOAbbreviation")?.textContent?.trim()
          ?? article.querySelector("Title")?.textContent?.trim()
          ?? "";
        return { pmid, title, abstract, year, journal };
      }).filter((a) => a.pmid && a.abstract.length > 50);
    } catch (err) {
      console.warn("[LivingEvidence] XML parse error:", err.message);
      return [];
    }
  }

  // ── LLM Screening ───────────────────────────────────────────────────────────

  async _screenAbstracts(abstracts, userQuestion) {
    if (abstracts.length === 0) return [];

    const abstractList = abstracts.map((a, i) =>
      `[${i + 1}] PMID:${a.pmid} | ${a.year}\n"${a.title}"\n${a.abstract.slice(0, 400)}...`
    ).join("\n\n");

    const prompt = `You are a clinical librarian screening PubMed abstracts for relevance to a heart failure patient's question.

Patient question: "${userQuestion}"

Abstracts:
${abstractList}

Return ONLY a JSON array of relevant PMID strings (e.g., ["12345678","98765432"]). Include only abstracts that provide useful, directly applicable information to answer this patient's heart failure question. Return [] if none are relevant.`;

    try {
      const resp = await fetch("https://api.openai.com/v1/chat/completions", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "Authorization": `Bearer ${this.chatEngine.apiKey}`
        },
        body: JSON.stringify({
          model: this.chatEngine.model,
          max_tokens: 100,
          temperature: 0,
          messages: [{ role: "user", content: prompt }]
        })
      });
      if (!resp.ok) return [];
      const data = await resp.json();
      const text = data.choices[0]?.message?.content?.trim() || "";
      const match = text.match(/\[[\s\S]*\]/);
      if (!match) return [];
      const pmids = JSON.parse(match[0]);
      return Array.isArray(pmids) ? pmids.filter((p) => typeof p === "string") : [];
    } catch (err) {
      console.warn("[LivingEvidence] Screening failed:", err.message);
      return [];
    }
  }

  // ── LLM Synthesis ───────────────────────────────────────────────────────────

  async _synthesizeEvidence(abstracts, userQuestion) {
    if (abstracts.length === 0) return null;

    const abstractList = abstracts.map((a) =>
      `PMID:${a.pmid} (${a.year}, ${a.journal})\nTitle: "${a.title}"\nAbstract: ${a.abstract}`
    ).join("\n\n---\n\n");

    const prompt = `You are a heart failure patient educator. Using only the research abstracts below, write a clear, evidence-based answer to the patient's question.

Requirements:
- Write in plain language suitable for a patient with 8th-grade health literacy
- Keep the answer concise (150-250 words)
- Cite each source inline using the format [PMID:XXXXXXXX]
- Never recommend stopping prescribed medications
- Never diagnose; recommend consulting the healthcare team for individual decisions
- Always end with "⚗ This information is based on recent research (PubMed literature search). Please discuss with your healthcare team."

Patient question: "${userQuestion}"

Research abstracts:
${abstractList}`;

    try {
      const resp = await fetch("https://api.openai.com/v1/chat/completions", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "Authorization": `Bearer ${this.chatEngine.apiKey}`
        },
        body: JSON.stringify({
          model: this.chatEngine.model,
          max_tokens: 500,
          temperature: 0.3,
          messages: [{ role: "user", content: prompt }]
        })
      });
      if (!resp.ok) return null;
      const data = await resp.json();
      return data.choices[0]?.message?.content?.trim() || null;
    } catch (err) {
      console.warn("[LivingEvidence] Synthesis failed:", err.message);
      return null;
    }
  }

  // ── Caching ─────────────────────────────────────────────────────────────────

  _cacheKey(question) {
    // Simple deterministic key from first 120 chars of question (URL-safe base64)
    const normalized = question.toLowerCase().replace(/\s+/g, " ").trim().slice(0, 120);
    try {
      return LE_CACHE_PREFIX + btoa(unescape(encodeURIComponent(normalized)));
    } catch {
      return LE_CACHE_PREFIX + normalized.replace(/[^a-z0-9]/g, "_");
    }
  }

  _readCache(key) {
    try {
      const raw = localStorage.getItem(key);
      if (!raw) return null;
      const { timestamp, data } = JSON.parse(raw);
      if (Date.now() - timestamp > LE_CACHE_TTL_MS) {
        localStorage.removeItem(key);
        return null;
      }
      return data;
    } catch {
      return null;
    }
  }

  _writeCache(key, data) {
    try {
      localStorage.setItem(key, JSON.stringify({ timestamp: Date.now(), data }));
    } catch (e) {
      // localStorage quota exceeded — purge old entries and retry once
      this.purgeExpiredCache();
      try { localStorage.setItem(key, JSON.stringify({ timestamp: Date.now(), data })); } catch { /* ignore */ }
    }
  }

  purgeExpiredCache() {
    try {
      const now = Date.now();
      Object.keys(localStorage)
        .filter((k) => k.startsWith(LE_CACHE_PREFIX))
        .forEach((k) => {
          try {
            const { timestamp } = JSON.parse(localStorage.getItem(k));
            if (now - timestamp > LE_CACHE_TTL_MS) localStorage.removeItem(k);
          } catch { localStorage.removeItem(k); }
        });
    } catch { /* ignore */ }
  }

  // ── Rate Limiting & Fetch ────────────────────────────────────────────────────

  async _throttle() {
    const minInterval = this.ncbiApiKey ? LE_RATE_LIMIT_WITH_KEY : LE_RATE_LIMIT_NO_KEY;
    const now = Date.now();
    const elapsed = now - this._lastCallTime;
    if (elapsed < minInterval) {
      await new Promise((r) => setTimeout(r, minInterval - elapsed));
    }
    this._lastCallTime = Date.now();
  }

  async _fetchWithTimeout(url) {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), LE_FETCH_TIMEOUT_MS);
    try {
      return await fetch(url, { signal: controller.signal });
    } finally {
      clearTimeout(timer);
    }
  }
}

window.LivingEvidenceEngine = LivingEvidenceEngine;
