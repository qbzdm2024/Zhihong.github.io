/**
 * Knowledge Base Manager
 * Handles loading, storage, retrieval, and management of HF education content.
 * Supports dynamic addition/removal of knowledge sources.
 */

class KnowledgeBase {
  constructor() {
    this.chunks = [];
    this.sources = [];
    this.storageKey = "hf_chatbot_kb";
    this.customSourcesKey = "hf_chatbot_custom_sources";
  }

  /**
   * Initialize the knowledge base by loading built-in data
   * and any user-added custom sources from localStorage.
   */
  async initialize() {
    // Load the built-in knowledge JSON
    try {
      const resp = await fetch("./data/hf-knowledge.json");
      const data = await resp.json();
      this.chunks = data.chunks || [];
      this.sources = data.sources || [];
    } catch (e) {
      console.error("Failed to load built-in knowledge:", e);
      this.chunks = [];
      this.sources = [];
    }

    // Load any persisted custom chunks from localStorage
    try {
      const saved = localStorage.getItem(this.storageKey);
      if (saved) {
        const customData = JSON.parse(saved);
        if (customData.chunks) {
          // Merge custom chunks (avoid duplicates by id)
          const existingIds = new Set(this.chunks.map((c) => c.id));
          customData.chunks.forEach((c) => {
            if (!existingIds.has(c.id)) {
              this.chunks.push(c);
              existingIds.add(c.id);
            }
          });
        }
        if (customData.sources) {
          const existingSourceIds = new Set(this.sources.map((s) => s.id));
          customData.sources.forEach((s) => {
            if (!existingSourceIds.has(s.id)) {
              this.sources.push(s);
              existingSourceIds.add(s.id);
            }
          });
        }
      }
    } catch (e) {
      console.warn("Could not load custom knowledge from storage:", e);
    }

    // Apply enabled/disabled overrides saved by user
    try {
      const overrides = localStorage.getItem("hf_source_overrides");
      if (overrides) {
        const parsed = JSON.parse(overrides);
        this.sources.forEach((s) => {
          if (parsed[s.id] !== undefined) {
            s.enabled = parsed[s.id];
          }
        });
      }
    } catch (e) { /* ignore */ }

    return this;
  }

  /** Persist custom chunks and sources to localStorage */
  _persist() {
    const builtInIds = new Set();
    // We consider chunks with sourceIds in the built-in sources as built-in
    const builtInSourceIds = new Set([
      "cornell_homecare", "cornell_traffictool", "aha_hf_warning",
      "aha_hf_living", "acc_aha_2022", "tls_rct_2024"
    ]);
    const customChunks = this.chunks.filter(
      (c) => !builtInSourceIds.has(c.sourceId)
    );
    const customSources = this.sources.filter(
      (s) => !builtInSourceIds.has(s.id)
    );
    localStorage.setItem(
      this.storageKey,
      JSON.stringify({ chunks: customChunks, sources: customSources })
    );

    // Save enabled/disabled overrides
    const overrides = {};
    this.sources.forEach((s) => { overrides[s.id] = s.enabled; });
    localStorage.setItem("hf_source_overrides", JSON.stringify(overrides));
  }

  // ─── Source Management ───────────────────────────────────────────────────

  getSources() {
    return this.sources;
  }

  toggleSource(sourceId, enabled) {
    const source = this.sources.find((s) => s.id === sourceId);
    if (source) {
      source.enabled = enabled;
      this._persist();
    }
  }

  removeCustomSource(sourceId) {
    const builtInIds = new Set([
      "cornell_homecare", "cornell_traffictool", "aha_hf_warning",
      "aha_hf_living", "acc_aha_2022", "tls_rct_2024"
    ]);
    if (builtInIds.has(sourceId)) {
      throw new Error("Cannot remove built-in sources. You can disable them instead.");
    }
    this.sources = this.sources.filter((s) => s.id !== sourceId);
    this.chunks = this.chunks.filter((c) => c.sourceId !== sourceId);
    this._persist();
  }

  /**
   * Add a custom knowledge source with provided text content.
   * @param {Object} sourceInfo - { name, url, type, description }
   * @param {string} content - Full text content from the source
   */
  addCustomSource(sourceInfo, content) {
    const id = "custom_" + Date.now();
    const source = {
      id,
      name: sourceInfo.name,
      url: sourceInfo.url || "",
      type: sourceInfo.type || "custom",
      description: sourceInfo.description || "",
      enabled: true,
      addedAt: new Date().toISOString()
    };
    this.sources.push(source);

    // Split content into chunks (by paragraph or ~500 chars)
    const paragraphs = content
      .split(/\n{2,}/)
      .map((p) => p.trim())
      .filter((p) => p.length > 50);

    paragraphs.forEach((para, i) => {
      this.chunks.push({
        id: `${id}_chunk_${i}`,
        sourceId: id,
        title: sourceInfo.name + (paragraphs.length > 1 ? ` (Part ${i + 1})` : ""),
        category: sourceInfo.category || "general",
        keywords: this._extractKeywords(para),
        content: para
      });
    });

    this._persist();
    return source;
  }

  // ─── Retrieval ───────────────────────────────────────────────────────────

  /**
   * Retrieve the most relevant knowledge chunks for a query.
   * Uses TF-IDF-style keyword matching.
   * @param {string} query
   * @param {number} topK - Number of top results
   * @returns {Array} ranked chunks with scores
   */
  retrieve(query, topK = 5) {
    const enabledSourceIds = new Set(
      this.sources.filter((s) => s.enabled).map((s) => s.id)
    );
    const queryTokens = this._tokenize(query);

    const scored = this.chunks
      .filter((chunk) => enabledSourceIds.has(chunk.sourceId))
      .map((chunk) => {
        const score = this._score(queryTokens, chunk);
        return { chunk, score };
      })
      .filter((item) => item.score > 0)
      .sort((a, b) => b.score - a.score)
      .slice(0, topK);

    return scored.map((item) => ({
      ...item.chunk,
      score: item.score,
      source: this.sources.find((s) => s.id === item.chunk.sourceId)
    }));
  }

  /**
   * Score a chunk against query tokens using keyword + content matching
   */
  _score(queryTokens, chunk) {
    let score = 0;
    const contentTokens = this._tokenize(chunk.content + " " + chunk.title);
    const keywordTokens = (chunk.keywords || []).flatMap((k) =>
      this._tokenize(k)
    );

    queryTokens.forEach((qt) => {
      // Exact keyword match = 3 points
      if (keywordTokens.includes(qt)) score += 3;
      // Title match = 2 points
      if (this._tokenize(chunk.title).includes(qt)) score += 2;
      // Content match = 1 point
      if (contentTokens.includes(qt)) score += 1;
      // Partial match in content = 0.5 points
      if (contentTokens.some((ct) => ct.includes(qt) && ct !== qt)) score += 0.5;
    });

    // Boost triage-related chunks for symptom queries
    const triageWords = ["pain", "breathe", "breath", "swell", "weight", "dizzy", "faint", "chest", "confusion", "emergency"];
    if (queryTokens.some((qt) => triageWords.includes(qt)) && chunk.category === "triage") {
      score *= 1.5;
    }

    return score;
  }

  _tokenize(text) {
    return text
      .toLowerCase()
      .replace(/[^a-z0-9\s]/g, " ")
      .split(/\s+/)
      .filter((w) => w.length > 2 && !this._isStopWord(w));
  }

  _isStopWord(word) {
    const stopWords = new Set([
      "the", "and", "for", "are", "but", "not", "you", "all", "can",
      "has", "her", "was", "one", "our", "out", "day", "get", "use",
      "may", "its", "that", "this", "with", "your", "from", "they",
      "will", "have", "more", "been", "also", "when", "what", "who"
    ]);
    return stopWords.has(word);
  }

  _extractKeywords(text) {
    const tokens = this._tokenize(text);
    const freq = {};
    tokens.forEach((t) => { freq[t] = (freq[t] || 0) + 1; });
    return Object.entries(freq)
      .sort((a, b) => b[1] - a[1])
      .slice(0, 10)
      .map(([word]) => word);
  }

  /**
   * Build a context string for the LLM from retrieved chunks
   * @param {Array} chunks - Retrieved chunks
   * @returns {string} formatted context with citations
   */
  buildContext(chunks) {
    if (!chunks || chunks.length === 0) {
      return "No specific knowledge base entries retrieved.";
    }
    return chunks
      .map((chunk, i) => {
        const sourceInfo = chunk.source
          ? `[Source ${i + 1}: ${chunk.source.name}${chunk.source.url ? " — " + chunk.source.url : ""}]`
          : `[Source ${i + 1}: Unknown source]`;
        return `${sourceInfo}\n${chunk.title}\n${chunk.content}`;
      })
      .join("\n\n---\n\n");
  }

  /** Get stats about the knowledge base */
  getStats() {
    const enabledSources = this.sources.filter((s) => s.enabled).length;
    const enabledSourceIds = new Set(
      this.sources.filter((s) => s.enabled).map((s) => s.id)
    );
    const enabledChunks = this.chunks.filter((c) =>
      enabledSourceIds.has(c.sourceId)
    ).length;
    const categories = [...new Set(this.chunks.map((c) => c.category))];
    return {
      totalSources: this.sources.length,
      enabledSources,
      totalChunks: this.chunks.length,
      enabledChunks,
      categories
    };
  }
}

window.KnowledgeBase = KnowledgeBase;
