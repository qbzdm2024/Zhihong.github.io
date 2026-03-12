/**
 * Heart Failure Chatbot - Main Application Controller
 */

class App {
  constructor() {
    this.kb = new KnowledgeBase();
    this.triageEngine = new TriageEngine();
    this.chatEngine = null;
    this.isLoading = false;
    this.lastTriageResults = null;
    this.triageMode = "rule"; // "rule" | "ai-rule" | "ai-independently"
    // Triage only fires when ≥1 of the 7 HF triage symptoms is present in the message.

    // Traffic-light follow-up state (rule / ai-rule modes)
    this.triageFollowUpState = null;
    // Structure: { originalText: string, detectedSymptoms: string[], answered: boolean }

    // AI-independent mode follow-up state (separate from traffic-light follow-ups)
    this.aiIndependentFollowUpState = null;
    // Structure: { originalText: string, answered: boolean, roundCount: number }
  }

  async initialize() {
    await this.kb.initialize();
    this.chatEngine = new ChatEngine(this.kb, this.triageEngine);
    this.chatEngine.loadApiKey();

    this._bindEvents();
    this._renderSources();
    this._updateKBStats();
    this._updateConnectionStatus();
    this._renderWelcomeMessage();
  }

  // ─── Helpers ─────────────────────────────────────────────────────────────

  /** Build per-category extracted answers from combined patient text */
  _extractAllAnswers(text, detectedSymptoms) {
    const answers = {};
    for (const cat of detectedSymptoms) {
      answers[cat] = this.triageEngine.extractAnswers(text, cat);
    }
    return answers;
  }

  // ─── UI Helpers ──────────────────────────────────────────────────────────

  _showToast(message, type = "info") {
    const container = document.getElementById("toast-container");
    const toast = document.createElement("div");
    const icons = { success: "✓", error: "✕", warning: "⚠", info: "ℹ" };
    toast.className = `toast ${type}`;
    toast.innerHTML = `<span>${icons[type] || "ℹ"}</span> ${message}`;
    container.appendChild(toast);
    setTimeout(() => toast.remove(), 3500);
  }

  _setLoading(loading, text = "Thinking...") {
    this.isLoading = loading;
    const overlay = document.getElementById("loading-overlay");
    const loadingText = document.getElementById("loading-text");
    if (loadingText) loadingText.textContent = text;
    overlay.style.display = loading ? "flex" : "none";
    document.getElementById("send-btn").disabled = loading;
  }

  _getCurrentTime() {
    return new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
  }

  _updateConnectionStatus() {
    const dot = document.getElementById("connection-dot");
    const label = document.getElementById("connection-label");
    if (this.chatEngine.apiKey) {
      dot.className = "status-dot connected";
      label.textContent = "API Connected";
    } else {
      dot.className = "status-dot disconnected";
      label.textContent = "API Key Required";
    }
  }

  _updateKBStats() {
    const stats = this.kb.getStats();
    document.getElementById("kb-sources-count").textContent = stats.enabledSources + "/" + stats.totalSources;
    document.getElementById("kb-chunks-count").textContent = stats.enabledChunks;
  }

  // ─── Chat Messages ───────────────────────────────────────────────────────

  _renderWelcomeMessage() {
    const container = document.getElementById("chat-messages");
    const welcome = document.createElement("div");
    welcome.className = "welcome-message";
    welcome.innerHTML = `
      <h3>💙 Heart Failure Self-Management Assistant</h3>
      <p>I'm here to help you understand and manage heart failure. I can answer questions about symptoms, diet, exercise, medications, and daily monitoring.</p>
      <div class="welcome-features">
        <div class="welcome-feature">🔍 <strong>RAG-powered:</strong> Answers cite evidence from AHA guidelines, clinical tools, and research</div>
        <div class="welcome-feature">🚦 <strong>Dual triage:</strong> Rule-based (traffic light) + AI reasoning for symptom assessment</div>
        <div class="welcome-feature">📚 <strong>Dynamic knowledge base:</strong> Add your own sources in the sidebar</div>
      </div>
      <p style="margin-top:10px; font-size:13px; color:#718096;">
        ⚠️ <strong>Important:</strong> This tool is for education only. Always contact your healthcare team for medical decisions.
        In an emergency, call <strong>911</strong>.
      </p>
    `;
    container.appendChild(welcome);
  }

  // Returns the created message element so callers can append triage cards
  _addMessage(role, content, sources = []) {
    const container = document.getElementById("chat-messages");
    const msgDiv = document.createElement("div");
    msgDiv.className = `message ${role}`;

    const avatarDiv = document.createElement("div");
    avatarDiv.className = "message-avatar";
    avatarDiv.textContent = role === "user" ? "👤" : "💙";

    const bodyDiv = document.createElement("div");
    bodyDiv.className = "message-body";

    const bubble = document.createElement("div");
    bubble.className = "message-bubble";
    if (role === "assistant") {
      bubble.innerHTML = this._renderMarkdown(content);
    } else {
      bubble.textContent = content;
    }
    bodyDiv.appendChild(bubble);

    // Source chips
    if (sources && sources.length > 0) {
      const sourcesDiv = document.createElement("div");
      sourcesDiv.className = "message-sources";
      sources.forEach((src) => {
        const chip = document.createElement("a");
        chip.className = "source-chip";
        chip.href = src.url || "#";
        chip.target = "_blank";
        chip.rel = "noopener noreferrer";
        chip.title = src.name;
        chip.textContent = "📚 " + src.name;
        sourcesDiv.appendChild(chip);
      });
      bodyDiv.appendChild(sourcesDiv);
    }

    const timeDiv = document.createElement("div");
    timeDiv.className = "message-time";
    timeDiv.textContent = this._getCurrentTime();
    bodyDiv.appendChild(timeDiv);

    msgDiv.appendChild(avatarDiv);
    msgDiv.appendChild(bodyDiv);
    container.appendChild(msgDiv);
    container.scrollTop = container.scrollHeight;
    return msgDiv; // return so callers can attach inline triage
  }

  /**
   * Append a side-by-side inline triage card directly inside a chat message.
   * @param {HTMLElement} messageEl - the message div returned by _addMessage
   * @param {Object|null} ruleResult
   * @param {Object|null} aiResult - traffic-light AI result
   * @param {string|null} aiError - error message if AI triage failed
   * @param {Object|null} aiIndResult - AI-independent result (separate mode)
   */
  _appendInlineTriage(messageEl, ruleResult, aiResult, aiError = null, aiIndResult = null) {
    if (!ruleResult && !aiResult && !aiError && !aiIndResult) return;
    const body = messageEl.querySelector(".message-body");
    const bubble = messageEl.querySelector(".message-bubble");

    const wrapper = document.createElement("div");
    wrapper.className = "inline-triage";

    if (ruleResult)   wrapper.appendChild(this._buildInlineCard("rule", "📏 Rule-Based (Traffic Light)", ruleResult));
    if (aiResult)     wrapper.appendChild(this._buildInlineCard("ai",   "🤖 AI Reasoning (Traffic Light)", aiResult));
    if (aiIndResult)  wrapper.appendChild(this._buildInlineCard("ai",   "🧠 AI Independent Assessment", aiIndResult));
    if (!aiResult && !aiIndResult && aiError) {
      const errCard = document.createElement("div");
      errCard.className = "inline-triage-card ai";
      errCard.innerHTML = `<div class="inline-triage-header ai">🤖 AI Reasoning</div>
        <div class="inline-triage-body" style="color:#718096; font-size:12px; padding:8px;">
          AI triage unavailable: ${aiError.slice(0, 120)}. Rule-based result above is still valid.
        </div>`;
      wrapper.appendChild(errCard);
    }

    // Insert after the bubble, before sources/time
    bubble.insertAdjacentElement("afterend", wrapper);

    // Comparison note if both present
    if (ruleResult && aiResult) {
      const note = document.createElement("div");
      note.className = "inline-triage-note";
      if (ruleResult.zone !== aiResult.zone) {
        const zoneOrder = { GREEN: 0, YELLOW: 1, RED: 2 };
        const safer = (zoneOrder[ruleResult.zone] ?? -1) >= (zoneOrder[aiResult.zone] ?? -1)
          ? ruleResult : aiResult;
        note.className = "inline-triage-note disagree";
        note.innerHTML =
          `⚠️ <strong>The two systems disagree:</strong><br>` +
          `• Rule-based: <strong>${ruleResult.zone}</strong>${ruleResult.urgency && ruleResult.urgency !== ruleResult.zone ? " — " + ruleResult.urgency : ""}<br>` +
          `• AI Assessment: <strong>${aiResult.zone}</strong>${aiResult.urgency && aiResult.urgency !== aiResult.zone ? " — " + aiResult.urgency : ""}<br>` +
          `<span style="margin-top:4px;display:inline-block;">Following the more cautious recommendation: ` +
          `<strong>${safer.zone}</strong>. ${safer.action}</span>`;
      } else {
        note.className = "inline-triage-note agree";
        note.innerHTML = `✓ <strong>Both systems agree: ${ruleResult.zone}</strong> — ${ruleResult.action}`;
      }
      wrapper.insertAdjacentElement("afterend", note);
    }

    const container = document.getElementById("chat-messages");
    container.scrollTop = container.scrollHeight;
  }

  _buildInlineCard(type, label, result) {
    const zoneIcons = { GREEN: "🟢", YELLOW: "🟡", RED: "🔴", UNKNOWN: "⚪" };
    const card = document.createElement("div");
    card.className = `inline-triage-card ${type}`;

    const header = document.createElement("div");
    header.className = `inline-triage-header ${type}`;
    header.textContent = label;

    const body = document.createElement("div");
    body.className = "inline-triage-body";

    // Zone badge + urgency sub-label
    const badge = document.createElement("div");
    badge.className = `zone-badge ${result.zone}`;
    badge.style.marginBottom = "5px";
    const urgencyLabel = result.urgency && result.urgency !== result.zone
      ? ` — ${result.urgency}`
      : "";
    badge.textContent = `${zoneIcons[result.zone] || "⚪"} ${result.zone}${urgencyLabel}`;
    body.appendChild(badge);

    // Category-level flags (rule-based) or key symptoms (AI)
    const flags = result.flags || result.keySymptoms || [];
    if (flags.length > 0) {
      const ul = document.createElement("ul");
      ul.className = "inline-triage-flags";
      flags.slice(0, 4).forEach((f) => {
        const li = document.createElement("li");
        li.textContent = f;
        ul.appendChild(li);
      });
      body.appendChild(ul);
    }

    // Recommended action
    if (result.action) {
      const act = document.createElement("div");
      act.className = `inline-triage-action ${result.zone}`;
      act.textContent = result.action;
      body.appendChild(act);
    }

    // HF education review note (rule-based only)
    if (result.reviewNote) {
      const note = document.createElement("div");
      note.style.cssText = "margin-top:5px; font-size:12px; color:#4a5568; font-style:italic;";
      note.textContent = result.reviewNote;
      body.appendChild(note);
    }

    card.appendChild(header);
    card.appendChild(body);
    return card;
  }

  _addTypingIndicator() {
    const container = document.getElementById("chat-messages");
    const msgDiv = document.createElement("div");
    msgDiv.className = "message assistant";
    msgDiv.id = "typing-indicator";
    const avatarDiv = document.createElement("div");
    avatarDiv.className = "message-avatar";
    avatarDiv.textContent = "💙";
    const bubble = document.createElement("div");
    bubble.className = "message-bubble";
    bubble.innerHTML = `<div class="typing-indicator"><div class="typing-dot"></div><div class="typing-dot"></div><div class="typing-dot"></div></div>`;
    msgDiv.appendChild(avatarDiv);
    msgDiv.appendChild(bubble);
    container.appendChild(msgDiv);
    container.scrollTop = container.scrollHeight;
  }

  _removeTypingIndicator() {
    const indicator = document.getElementById("typing-indicator");
    if (indicator) indicator.remove();
  }

  _addErrorMessage(errorText) {
    const container = document.getElementById("chat-messages");
    const msgDiv = document.createElement("div");
    msgDiv.className = "message assistant";
    const avatarDiv = document.createElement("div");
    avatarDiv.className = "message-avatar";
    avatarDiv.textContent = "⚠️";
    const bubble = document.createElement("div");
    bubble.className = "message-bubble error-message";
    bubble.textContent = "Error: " + errorText;
    msgDiv.appendChild(avatarDiv);
    msgDiv.appendChild(bubble);
    container.appendChild(msgDiv);
    container.scrollTop = container.scrollHeight;
  }

  // Simple markdown renderer
  _renderMarkdown(text) {
    return text
      .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;")
      .replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")
      .replace(/\*(.+?)\*/g, "<em>$1</em>")
      .replace(/`(.+?)`/g, "<code>$1</code>")
      .replace(/^## (.+)$/gm, "<h2>$1</h2>")
      .replace(/^### (.+)$/gm, "<h3>$1</h3>")
      .replace(/\[([^\]]+)\]\(([^)]+)\)/g, '<a href="$2" target="_blank" rel="noopener">$1</a>')
      .replace(/^[*-] (.+)$/gm, "<li>$1</li>")
      .replace(/(<li>.*<\/li>)/s, "<ul>$1</ul>")
      .replace(/\n\n/g, "</p><p>")
      .replace(/^(?!<[hul])(.+)$/gm, (_, p) => p.startsWith("<") ? _ : `<p>${p}</p>`)
      .replace(/<p><\/p>/g, "")
      .replace(/<p>(<[hul])/g, "$1")
      .replace(/(<\/[hul][^>]*>)<\/p>/g, "$1");
  }

  // ─── Triage Panel ────────────────────────────────────────────────────────

  _renderTriageResults(ruleResult, aiResult) {
    this.lastTriageResults = { ruleResult, aiResult };
    const body = document.getElementById("triage-panel-body");
    body.innerHTML = "";

    const comparisonDiv = document.createElement("div");
    comparisonDiv.className = "triage-comparison";

    // Rule-based card
    if (ruleResult) {
      const card = this._buildTriageCard("rule", "📏 Rule-Based (Traffic Light)", ruleResult);
      comparisonDiv.appendChild(card);
    }

    // AI card
    if (aiResult) {
      const label = aiResult.method === "ai_independent"
        ? "🧠 AI Independent Assessment"
        : "🤖 AI Rule-Based Reasoning";
      const card = this._buildTriageCard("ai", label, aiResult);
      comparisonDiv.appendChild(card);
    }

    body.appendChild(comparisonDiv);

    // Comparison summary
    if (ruleResult && aiResult) {
      const comparison = this.triageEngine.compareTriageResults(ruleResult, aiResult);
      const summary = document.createElement("div");
      summary.className = `triage-comparison-summary ${comparison.agree ? "agree" : "disagree"}`;

      if (comparison.agree) {
        summary.innerHTML = `<strong>✓ Both systems agree:</strong> ${ruleResult.zone} ZONE — ${comparison.recommendedAction}`;
      } else {
        summary.innerHTML = `
          <strong>⚠️ Systems disagree:</strong> Rule-based says <strong>${comparison.ruleBasedZone}</strong>, AI says <strong>${comparison.aiZone}</strong><br>
          <span style="color:#d69e2e">${comparison.discrepancyNote}</span><br>
          <strong>Recommended action (conservative):</strong> ${comparison.recommendedAction}
        `;
      }
      body.appendChild(summary);
    }
  }

  _buildTriageCard(type, headerLabel, result) {
    const card = document.createElement("div");
    card.className = "triage-card";

    const header = document.createElement("div");
    header.className = `triage-card-header ${type}`;
    header.textContent = headerLabel;

    const body = document.createElement("div");
    body.className = "triage-card-body";

    // Zone badge
    const badge = document.createElement("div");
    badge.className = `zone-badge ${result.zone}`;
    const zoneIcons = { GREEN: "🟢", YELLOW: "🟡", RED: "🔴", UNKNOWN: "⚪" };
    badge.innerHTML = `${zoneIcons[result.zone] || "⚪"} ${result.zone}`;
    body.appendChild(badge);

    // Flags or key symptoms
    const flagsSource = result.flags || result.keySymptoms || [];
    if (flagsSource.length > 0) {
      const flagsDiv = document.createElement("div");
      flagsDiv.className = "triage-flags";
      flagsSource.slice(0, 4).forEach((flag) => {
        const d = document.createElement("div");
        d.className = "triage-flag";
        d.textContent = flag;
        flagsDiv.appendChild(d);
      });
      body.appendChild(flagsDiv);
    }

    // Action
    if (result.action) {
      const action = document.createElement("div");
      action.className = `triage-action ${result.zone}`;
      action.textContent = result.action;
      body.appendChild(action);
    }

    // For AI results, show reasoning toggle
    if (type === "ai" && (result.reasoning || result.evidenceBasis)) {
      const toggleBtn = document.createElement("button");
      toggleBtn.className = "collapsible-btn";
      toggleBtn.textContent = "Show clinical reasoning ▼";

      const reasoningDiv = document.createElement("div");
      reasoningDiv.className = "collapsible-content hidden";
      reasoningDiv.textContent = result.reasoning || result.evidenceBasis;

      toggleBtn.addEventListener("click", () => {
        const isHidden = reasoningDiv.classList.contains("hidden");
        reasoningDiv.classList.toggle("hidden");
        toggleBtn.textContent = isHidden ? "Hide reasoning ▲" : "Show clinical reasoning ▼";
      });

      body.appendChild(toggleBtn);
      body.appendChild(reasoningDiv);
    }

    // Sources
    if (result.sources && result.sources.length > 0) {
      const srcDiv = document.createElement("div");
      srcDiv.style.cssText = "margin-top:6px; display:flex; flex-wrap:wrap; gap:3px;";
      result.sources.forEach((src) => {
        const chip = document.createElement("a");
        chip.href = src.url || "#";
        chip.target = "_blank";
        chip.className = "source-chip";
        chip.title = src.name;
        chip.textContent = "📚 " + src.name.split(" ").slice(0, 4).join(" ");
        srcDiv.appendChild(chip);
      });
      body.appendChild(srcDiv);
    }

    card.appendChild(header);
    card.appendChild(body);
    return card;
  }

  _clearTriagePanel() {
    document.getElementById("triage-panel-body").innerHTML =
      '<div class="triage-empty">Describe your symptoms in the chat to see triage assessment.</div>';
    this.lastTriageResults = null;
  }

  // ─── Knowledge Source Management ─────────────────────────────────────────

  _renderSources() {
    const list = document.getElementById("source-list");
    list.innerHTML = "";
    const sources = this.kb.getSources();

    sources.forEach((source) => {
      const li = document.createElement("li");
      li.className = `source-item ${source.enabled ? "" : "disabled"}`;
      li.dataset.sourceId = source.id;

      const checkbox = document.createElement("input");
      checkbox.type = "checkbox";
      checkbox.className = "source-toggle";
      checkbox.checked = source.enabled;
      checkbox.title = "Enable/disable this source";
      checkbox.addEventListener("change", (e) => {
        this.kb.toggleSource(source.id, e.target.checked);
        li.classList.toggle("disabled", !e.target.checked);
        this._updateKBStats();
      });

      const info = document.createElement("div");
      info.className = "source-info";

      const name = document.createElement("div");
      name.className = "source-name";
      name.textContent = source.name;
      name.title = source.name;

      const meta = document.createElement("div");
      meta.className = "source-meta";

      const typeBadge = document.createElement("span");
      typeBadge.className = `source-type-badge ${source.type || ""}`;
      typeBadge.textContent = (source.type || "custom").replace("_", " ");
      meta.appendChild(typeBadge);

      info.appendChild(name);
      info.appendChild(meta);

      li.appendChild(checkbox);
      li.appendChild(info);

      // Remove button for custom sources
      const builtInIds = new Set([
        "cornell_homecare", "cornell_traffictool", "aha_hf_warning",
        "aha_hf_living", "acc_aha_2022", "tls_rct_2024"
      ]);
      if (!builtInIds.has(source.id)) {
        const removeBtn = document.createElement("button");
        removeBtn.className = "source-remove";
        removeBtn.textContent = "×";
        removeBtn.title = "Remove this source";
        removeBtn.addEventListener("click", () => {
          if (confirm(`Remove source "${source.name}"?`)) {
            try {
              this.kb.removeCustomSource(source.id);
              li.remove();
              this._updateKBStats();
              this._showToast("Source removed", "success");
            } catch (err) {
              this._showToast(err.message, "error");
            }
          }
        });
        li.appendChild(removeBtn);
      }

      list.appendChild(li);
    });
  }

  // ─── Event Binding ───────────────────────────────────────────────────────

  _bindEvents() {
    // Send message
    document.getElementById("send-btn").addEventListener("click", () => this._handleSend());
    document.getElementById("chat-input").addEventListener("keydown", (e) => {
      if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        this._handleSend();
      }
    });
    // Auto-resize textarea
    document.getElementById("chat-input").addEventListener("input", (e) => {
      e.target.style.height = "auto";
      e.target.style.height = Math.min(e.target.scrollHeight, 120) + "px";
    });

    // Quick action buttons
    document.querySelectorAll(".quick-btn").forEach((btn) => {
      btn.addEventListener("click", () => {
        document.getElementById("chat-input").value = btn.dataset.message;
        this._handleSend();
      });
    });

    // API key input — fire on blur AND on paste so user doesn't have to
    // click away after typing; also migrate any key saved by the old Claude version
    const apiInput = document.getElementById("api-key-input");
    const saveKey = (value) => {
      const key = value.trim();
      if (!key) return;
      this.chatEngine.setApiKey(key);
      this._updateConnectionStatus();
      this._showToast("API key saved", "success");
    };
    apiInput.addEventListener("blur",  (e) => saveKey(e.target.value));
    apiInput.addEventListener("paste", (e) => {
      // paste gives the value before paste; read after a tick
      setTimeout(() => saveKey(apiInput.value), 0);
    });
    // Also support pressing Enter inside the field
    apiInput.addEventListener("keydown", (e) => {
      if (e.key === "Enter") { e.preventDefault(); saveKey(e.target.value); apiInput.blur(); }
    });

    // Model select
    document.getElementById("model-select").addEventListener("change", (e) => {
      this.chatEngine.setModel(e.target.value);
    });

    // Add source button
    document.getElementById("add-source-btn").addEventListener("click", () => {
      document.getElementById("add-source-modal").classList.remove("hidden");
    });

    // Modal close
    document.getElementById("modal-close-btn").addEventListener("click", () => {
      document.getElementById("add-source-modal").classList.add("hidden");
    });
    document.getElementById("modal-cancel-btn").addEventListener("click", () => {
      document.getElementById("add-source-modal").classList.add("hidden");
    });
    document.getElementById("add-source-modal").addEventListener("click", (e) => {
      if (e.target === e.currentTarget)
        document.getElementById("add-source-modal").classList.add("hidden");
    });

    // Confirm add source
    document.getElementById("modal-confirm-btn").addEventListener("click", () => this._handleAddSource());

    // Run triage button
    document.getElementById("run-triage-btn").addEventListener("click", () => this._handleRunTriage());

    // Clear triage
    document.getElementById("clear-triage-btn").addEventListener("click", () => this._clearTriagePanel());

    // Clear chat
    document.getElementById("clear-chat-btn").addEventListener("click", () => {
      document.getElementById("chat-messages").innerHTML = "";
      this.chatEngine.clearHistory();
      this._renderWelcomeMessage();
      this._clearTriagePanel();
      this._showToast("Conversation cleared", "info");
    });

    // Triage mode tabs
    document.querySelectorAll(".triage-tab").forEach((tab) => {
      tab.addEventListener("click", () => {
        document.querySelectorAll(".triage-tab").forEach((t) => t.classList.remove("active"));
        tab.classList.add("active");
        this.triageMode = tab.dataset.mode;
      });
    });

    // Pre-fill API key if saved
    if (this.chatEngine.apiKey) {
      document.getElementById("api-key-input").value = this.chatEngine.apiKey;
    }
  }

  // ─── Main Send Handler ───────────────────────────────────────────────────

  async _handleSend() {
    const input = document.getElementById("chat-input");
    const message = input.value.trim();
    if (!message || this.isLoading) return;

    input.value = "";
    input.style.height = "auto";

    this._addMessage("user", message);

    if (!this.chatEngine.apiKey) {
      this._addMessage("assistant",
        "⚙️ Please enter your **OpenAI API key** in the Settings panel on the left to start chatting. " +
        "You can get an API key from [platform.openai.com/api-keys](https://platform.openai.com/api-keys).", []);
      return;
    }

    this._setLoading(true, "Retrieving knowledge & generating response...");
    this._addTypingIndicator();

    try {
      const intent = this.chatEngine.detectIntent(message);

      // ── Case A: Patient is answering pending follow-up questions ──────────
      if (this.triageFollowUpState && !this.triageFollowUpState.answered) {
        this.triageFollowUpState.answered = true;
        const combinedText = this.triageFollowUpState.originalText +
          "\nAdditional information: " + message;

        // Re-detect from combined text — follow-up may introduce new categories
        let detectedSymptoms = this.triageFollowUpState.detectedSymptoms;
        try {
          const llmDetection = await this.triageEngine.detectSymptomsWithLLM(
            combinedText, this.chatEngine.apiKey, this.chatEngine.model
          );
          for (const c of llmDetection.categories) {
            if (!detectedSymptoms.includes(c)) detectedSymptoms = [...detectedSymptoms, c];
          }
        } catch (e) {
          console.warn("LLM re-detection failed in Case A:", e.message);
        }

        // LLM extraction from combined text for the most accurate answers
        let extractedAnswers = {};
        try {
          extractedAnswers = await this.triageEngine.extractAnswersWithLLM(
            combinedText, detectedSymptoms, this.chatEngine.apiKey, this.chatEngine.model
          );
        } catch (e) {
          console.warn("LLM answer extraction failed in Case A, using regex:", e.message);
          extractedAnswers = this._extractAllAnswers(combinedText, detectedSymptoms);
        }

        // Run triage directly — no educational response before showing the triage result
        this._removeTypingIndicator();
        const msgEl = this._addMessage("assistant", "Symptom assessment:", []);
        await this._runAutoTriage(combinedText, detectedSymptoms, extractedAnswers, msgEl);
        this.triageFollowUpState = null;
        return;
      }

      // ── Case A': Patient answering AI-independent follow-up questions ──────
      if (this.aiIndependentFollowUpState && !this.aiIndependentFollowUpState.answered) {
        this.aiIndependentFollowUpState.answered = true;
        const currentRound = this.aiIndependentFollowUpState.roundCount || 1;
        const combinedText = this.aiIndependentFollowUpState.originalText +
          "\nPatient's answers to follow-up questions: " + message;

        // After 3 rounds, force triage — no more follow-up questions allowed
        const promptText = currentRound >= 3
          ? combinedText + "\n[Please provide your final triage assessment now — do not ask for any more follow-up questions.]"
          : combinedText;

        this._setLoading(true, "Running AI independent triage...");
        let aiIndResult;
        try {
          aiIndResult = await this.chatEngine.runAIIndependentTriage(promptText);
        } catch (err) {
          this._removeTypingIndicator();
          this._setLoading(false);
          const msgEl = this._addMessage("assistant", "Symptom assessment:", []);
          this._appendInlineTriage(msgEl, null, null, err.message, null);
          this.aiIndependentFollowUpState = null;
          return;
        }
        this._removeTypingIndicator();
        this._setLoading(false);

        if (!aiIndResult.isFollowUp || currentRound >= 3) {
          // Show triage result (or error if AI still returned follow-up despite force instruction)
          const msgEl = this._addMessage("assistant", "Symptom assessment:", []);
          if (!aiIndResult.isFollowUp) {
            this._appendInlineTriage(msgEl, null, null, null, aiIndResult);
          } else {
            this._appendInlineTriage(msgEl, null, null, "Could not complete assessment with available information.", null);
          }
          this.aiIndependentFollowUpState = null;
        } else {
          // Ask another round of follow-up questions (still within limit)
          this.aiIndependentFollowUpState = { originalText: combinedText, answered: false, roundCount: currentRound + 1 };
          const qText = (aiIndResult.acknowledgment ? aiIndResult.acknowledgment + "\n\n" : "") +
            "A few more questions:\n\n" +
            aiIndResult.questions.map((q, i) => `**${i + 1}.** ${q}`).join("\n\n");
          this._addMessage("assistant", qText, []);
        }
        return;
      }

      // ── Detect which of the 7 categories are present and whether it's personal ──
      // LLM handles natural phrasing + distinguishes personal reports from education questions.
      // Fall back to regex if LLM fails.
      let detectedSymptoms, isPersonalReport;
      try {
        const llmDetection = await this.triageEngine.detectSymptomsWithLLM(
          message, this.chatEngine.apiKey, this.chatEngine.model
        );
        detectedSymptoms  = llmDetection.categories;
        isPersonalReport  = llmDetection.isPersonalReport;
      } catch (e) {
        console.warn("LLM symptom detection failed, using regex fallback:", e.message);
        detectedSymptoms = this.triageEngine.detectSymptoms(message);
        isPersonalReport = this.triageEngine.isPersonalSymptomReport(message);
      }
      // ── AI-independently mode: bypass traffic light entirely ──────────────
      // AI decides its own follow-up questions (max 3 rounds) and triage zone.
      if (this.triageMode === "ai-independently" && isPersonalReport) {
        this._setLoading(true, "Running AI independent triage...");
        let aiIndResult;
        try {
          aiIndResult = await this.chatEngine.runAIIndependentTriage(message);
        } catch (err) {
          this._removeTypingIndicator();
          this._setLoading(false);
          const msgEl = this._addMessage("assistant", "Symptom assessment:", []);
          this._appendInlineTriage(msgEl, null, null, err.message, null);
          return;
        }
        this._removeTypingIndicator();
        this._setLoading(false);

        if (aiIndResult.isFollowUp) {
          // Save state (round 1); triage runs in Case A' after patient answers
          this.aiIndependentFollowUpState = { originalText: message, answered: false, roundCount: 1 };
          const qText = (aiIndResult.acknowledgment ? aiIndResult.acknowledgment + "\n\n" : "") +
            "To assess your symptoms, please answer:\n\n" +
            aiIndResult.questions.map((q, i) => `**${i + 1}.** ${q}`).join("\n\n");
          this._addMessage("assistant", qText, []);
        } else {
          const msgEl = this._addMessage("assistant", "Symptom assessment:", []);
          this._appendInlineTriage(msgEl, null, null, null, aiIndResult);
        }
        return;
      }

      // Triage fires only when the patient is reporting personal symptoms.
      // Pure education questions ("What causes leg swelling?") remain in Case D.
      const shouldTriage = detectedSymptoms.length > 0 && isPersonalReport && this.triageMode !== "none";

      // ── Case B/C: Personal symptom report with triage categories detected ─────
      if (shouldTriage) {
        // FOLLOW-UP GATING uses regex extraction — conservative (only explicit statements).
        // The LLM tends to infer values ("legs are swollen" → isNewOrWorse:"yes", legs:"both")
        // which would skip necessary follow-up questions. Regex returns null for anything
        // not explicitly stated, ensuring all required questions are always asked.
        const regexAnswers   = this._extractAllAnswers(message, detectedSymptoms);
        const neededFollowUps = this.triageEngine.getNeededFollowUps(message, detectedSymptoms, regexAnswers);

        if (neededFollowUps.length > 0) {
          // Case B: missing info — ask follow-up questions directly; triage runs in Case A after reply
          this.triageFollowUpState = {
            originalText:     message,
            detectedSymptoms: detectedSymptoms,
            answered:         false
          };
          this._removeTypingIndicator();
          const qText = "To assess your symptoms, I need a few more details:\n\n" +
            neededFollowUps.map((q, i) => `**${i + 1}.** ${q.question}`).join("\n\n");
          this._addMessage("assistant", qText, []);
          return;
        }

        // Case C: All info explicitly present — run triage directly, no educational response.
        let extractedAnswers = regexAnswers;
        try {
          extractedAnswers = await this.triageEngine.extractAnswersWithLLM(
            message, detectedSymptoms, this.chatEngine.apiKey, this.chatEngine.model
          );
        } catch (e) {
          console.warn("LLM answer extraction failed, using regex answers:", e.message);
        }
        this._removeTypingIndicator();
        const msgEl = this._addMessage("assistant", "Symptom assessment:", []);
        await this._runAutoTriage(message, detectedSymptoms, extractedAnswers, msgEl);
        return;
      }

      // ── Case D: No triage-relevant symptoms — normal education response ────
      // If the query contains cardiac symptoms (fast heartbeat, palpitations, etc.)
      // that fall outside the 7 traffic light categories and no HF context was
      // established, flag it so the LLM can ask a clarifying question.
      const cardiacNonTriage = this.triageEngine.detectCardiacNonTriageQuery(message);
      const result = await this.chatEngine.chat(message, {
        triageMode: false,
        cardiacNonTriage
      });
      this._removeTypingIndicator();
      this._addMessage("assistant", result.content, result.sources);

    } catch (err) {
      this._removeTypingIndicator();
      this._addErrorMessage(err.message);
      this._showToast("Error: " + err.message, "error");
    } finally {
      this._setLoading(false);
    }
  }

  /**
   * Run rule-based and/or AI triage and append the inline triage card.
   * @param {string} symptomText - full patient text (original + follow-ups)
   * @param {string[]} detectedSymptoms - categories detected
   * @param {Object} extractedAnswers - per-category answer objects
   * @param {HTMLElement|null} messageEl - chat bubble to attach inline card
   */
  async _runAutoTriage(symptomText, detectedSymptoms, extractedAnswers, messageEl = null) {
    let ruleResult = null;
    let aiResult   = null;
    let aiError    = null;

    // Rule-based: exact traffic light logic — synchronous, never fails
    if (this.triageMode !== "ai-rule") {
      ruleResult = this.triageEngine.ruleBasedTriage(extractedAnswers, detectedSymptoms);
    }

    // AI rule-based: traffic light follow-up questions + AI zone decision
    if (this.triageMode === "ai-rule") {
      this._setLoading(true, "Running AI triage assessment...");
      try {
        aiResult = await this.chatEngine.runAITriage(symptomText, detectedSymptoms);
      } catch (err) {
        console.error("AI triage error:", err);
        aiError = err.message;
      } finally {
        this._setLoading(false);
      }
    }

    if (ruleResult || aiResult || aiError) {
      if (messageEl) this._appendInlineTriage(messageEl, ruleResult, aiResult, aiError);
    }
  }

  async _handleRunTriage() {
    const symptomText = (document.getElementById("triage-symptom-text")?.value || "").trim();
    const formData    = this._readTriageForm();
    const hasForm     = this._hasAnyFormInput(formData);

    if (!hasForm && !symptomText) {
      this._showToast("Please fill in symptom values or describe symptoms", "warning");
      return;
    }

    this._setLoading(true, "Running triage assessment...");
    try {
      let ruleResult = null;
      let aiResult   = null;

      // When free text is provided, use the same LLM detection + extraction pipeline
      // as the chat flow so rule-only and both modes give consistent results.
      let textDetectedSymptoms = [];
      let textExtractedAnswers = {};

      if (symptomText && this.chatEngine.apiKey) {
        try {
          const llmDetection = await this.triageEngine.detectSymptomsWithLLM(
            symptomText, this.chatEngine.apiKey, this.chatEngine.model
          );
          textDetectedSymptoms = llmDetection.categories;
        } catch (e) {
          textDetectedSymptoms = this.triageEngine.detectSymptoms(symptomText);
        }
        if (textDetectedSymptoms.length > 0) {
          try {
            textExtractedAnswers = await this.triageEngine.extractAnswersWithLLM(
              symptomText, textDetectedSymptoms, this.chatEngine.apiKey, this.chatEngine.model
            );
          } catch (e) {
            textExtractedAnswers = this._extractAllAnswers(symptomText, textDetectedSymptoms);
          }
        }
      } else if (symptomText) {
        // No API key — regex fallback
        textDetectedSymptoms = this.triageEngine.detectSymptoms(symptomText);
        textExtractedAnswers = this._extractAllAnswers(symptomText, textDetectedSymptoms);
      }

      if (this.triageMode !== "ai-rule" && this.triageMode !== "ai-independently") {
        if (symptomText && textDetectedSymptoms.length > 0) {
          // Text input takes priority — same pipeline as chat
          ruleResult = this.triageEngine.ruleBasedTriage(textExtractedAnswers, textDetectedSymptoms);
        } else {
          // Fall back to form checkboxes
          const { detectedSymptoms, extractedAnswers } = this._formDataToCategories(formData);
          if (detectedSymptoms.length > 0) {
            ruleResult = this.triageEngine.ruleBasedTriage(extractedAnswers, detectedSymptoms);
          }
        }
      }

      const triageText = symptomText || this._formToText(formData);

      if (this.triageMode === "ai-independently") {
        if (!this.chatEngine.apiKey) {
          this._showToast("API key required for AI independent triage", "warning");
        } else {
          const aiIndResult = await this.chatEngine.runAIIndependentTriage(triageText);
          if (aiIndResult.isFollowUp) {
            // In sidebar mode, show follow-up questions as an info message in the panel
            const body = document.getElementById("triage-panel-body");
            body.innerHTML = "";
            const infoDiv = document.createElement("div");
            infoDiv.style.cssText = "padding:10px; font-size:13px; color:#4a5568; line-height:1.7;";
            infoDiv.innerHTML = `<strong>🧠 AI needs more information:</strong><br><br>` +
              (aiIndResult.acknowledgment ? `<em>${aiIndResult.acknowledgment}</em><br><br>` : "") +
              aiIndResult.questions.map((q, i) => `${i + 1}. ${q}`).join("<br>") +
              `<br><br><em>Please enter your answers in the symptom text box above and click Run Triage again.</em>`;
            body.appendChild(infoDiv);
          } else {
            this._renderTriageResults(null, aiIndResult);
          }
        }
      } else if (this.triageMode === "ai-rule" && this.chatEngine.apiKey) {
        const detectedForAI = textDetectedSymptoms.length > 0
          ? textDetectedSymptoms
          : Object.keys(this._formDataToCategories(formData).extractedAnswers);
        aiResult = await this.chatEngine.runAITriage(triageText, detectedForAI);
      } else if (this.triageMode === "ai-rule" && !this.chatEngine.apiKey) {
        this._showToast("API key required for AI rule-based triage", "warning");
      }

      if (this.triageMode !== "ai-independently") {
        if (ruleResult || aiResult) {
          this._renderTriageResults(ruleResult, aiResult);
        } else {
          this._showToast("No triage-relevant symptoms detected", "info");
        }
      }
    } catch (err) {
      this._showToast("Triage error: " + err.message, "error");
    } finally {
      this._setLoading(false);
    }
  }

  /**
   * Convert sidebar form checkbox values to per-category answers for the new triage engine.
   */
  _formDataToCategories(form) {
    const detectedSymptoms = [];
    const extractedAnswers = {};

    // SOB
    if (form.sobScale > 0) {
      detectedSymptoms.push("sob");
      extractedAnswers.sob = {
        isNew:         null,
        coSymptoms:    [
          ...(form.chestPain        ? ["chest_pain:severe"] : []),
          ...(form.swellingNewWorse ? ["leg_swelling"]      : []),
          ...(form.newCough         ? ["cough"]             : [])
        ],
        isWorseUsual:  null,
        changedLastDay:null
      };
    }

    // Chest discomfort — checkbox maps to the "chest_pain:severe" co-symptom in SOB,
    // but also represents the chestDiscomfort category itself.
    if (form.chestPain) {
      detectedSymptoms.push("chestDiscomfort");
      extractedAnswers.chestDiscomfort = {
        isNew:      null,
        coSymptoms: ["shooting_pain"],  // treat form "chest pain" as cardiac → RED path
        isAtRest:   null
      };
    }

    // Fatigue
    if (form.unusualFatigue) {
      detectedSymptoms.push("fatigue");
      extractedAnswers.fatigue = {
        coSymptoms:      [
          ...(form.sobScale > 0     ? ["sob"]          : []),
          ...(form.swellingNewWorse ? ["leg_swelling"] : []),
          ...(form.newCough         ? ["cough"]        : [])
        ],
        isWorseUsual:    null,
        canDoActivities: null
      };
    }

    // Weight change
    if (form.weightGain1Day >= 2 || form.weightGain1Week >= 3) {
      detectedSymptoms.push("weightChange");
      extractedAnswers.weightChange = {
        changeType:  form.weightGain1Day >= 2 ? "day_gain" : "week_gain",
        isExpected:  "no",
        coSymptoms:  [
          ...(form.sobScale > 0     ? ["sob"]          : []),
          ...(form.newCough         ? ["cough"]        : []),
          ...(form.swellingNewWorse ? ["leg_swelling"] : [])
        ]
      };
    }

    // Confusion
    if (form.severeConfusion) {
      detectedSymptoms.push("confusion");
      extractedAnswers.confusion = {
        isNew:      "yes",
        coSymptoms: [
          ...(form.faintingOrLoss ? ["unconscious"]   : []),
          ...(form.strokeSigns    ? ["slurred_speech", "face_asym", "weakness"] : [])
        ]
      };
    }

    // Leg swelling
    if (form.swellingNewWorse) {
      detectedSymptoms.push("legSwelling");
      extractedAnswers.legSwelling = {
        isNewOrWorse: "yes",
        legs:         null,
        tookDiuretic: null
      };
    }

    // Lightheadedness
    if (form.dizzinessStanding) {
      detectedSymptoms.push("lightheaded");
      extractedAnswers.lightheaded = {
        isNew:         null,
        isWorseUsual:  null,
        changedLastDay: null
      };
    }

    // Standalone RED flags (fainting, pink cough, stroke) that don't fit neatly into
    // a single category — surface them via confusion/sob pathways
    if (form.faintingOrLoss && !form.severeConfusion) {
      if (!detectedSymptoms.includes("confusion")) {
        detectedSymptoms.push("confusion");
        extractedAnswers.confusion = { isNew: "yes", coSymptoms: ["unconscious"] };
      } else {
        extractedAnswers.confusion.coSymptoms.push("unconscious");
      }
    }
    if (form.strokeSigns && !detectedSymptoms.includes("confusion")) {
      detectedSymptoms.push("confusion");
      extractedAnswers.confusion = { isNew: "yes", coSymptoms: ["slurred_speech", "face_asym", "weakness"] };
    }

    return { detectedSymptoms, extractedAnswers };
  }

  /** Build a readable symptom text from form values (used for AI triage text input) */
  _formToText(form) {
    const parts = [];
    if (form.sobScale > 0)      parts.push(`shortness of breath (severity ${form.sobScale}/10)`);
    if (form.chestPain)         parts.push("chest pain or pressure");
    if (form.faintingOrLoss)    parts.push("fainting or loss of consciousness");
    if (form.pinkFoamyCough)    parts.push("coughing up pink or foamy mucus");
    if (form.severeConfusion)   parts.push("sudden confusion or altered mental status");
    if (form.strokeSigns)       parts.push("stroke signs (facial droop, arm weakness, speech difficulty)");
    if (form.swellingNewWorse)  parts.push("new or worsening leg/ankle swelling");
    if (form.ortho)             parts.push("orthopnea (needs extra pillows, wakes up breathless)");
    if (form.dizzinessStanding) parts.push("dizziness when standing");
    if (form.irregularHeartbeat)parts.push("irregular heartbeat or palpitations");
    if (form.rapidHeartRate)    parts.push("rapid heart rate (>100 bpm at rest)");
    if (form.unusualFatigue)    parts.push("unusual fatigue limiting usual activities");
    if (form.newCough)          parts.push("new dry cough");
    if (form.decreasedUrine)    parts.push("decreased urine output");
    if (form.weightGain1Day > 0) parts.push(`weight gain of ${form.weightGain1Day} lbs today`);
    if (form.weightGain1Week > 0) parts.push(`weight gain of ${form.weightGain1Week} lbs this week`);
    return "Patient-reported symptoms: " + (parts.join("; ") || "none specified");
  }

  _readTriageForm() {
    const getVal   = (id) => { const el = document.getElementById(id); return el ? el.value   : null; };
    const getCheck = (id) => { const el = document.getElementById(id); return el ? el.checked : false; };
    const getNum   = (id) => { const v = getVal(id); return v !== null && v !== "" ? parseFloat(v) : 0; };

    return {
      weightGain1Day:    getNum("t-weight-day"),
      weightGain1Week:   getNum("t-weight-week"),
      sobScale:          getNum("t-sob"),
      chestPain:         getCheck("t-chest-pain"),
      faintingOrLoss:    getCheck("t-fainting"),
      pinkFoamyCough:    getCheck("t-pink-cough"),
      severeConfusion:   getCheck("t-confusion"),
      strokeSigns:       getCheck("t-stroke"),
      swellingNewWorse:  getCheck("t-swelling"),
      ortho:             getCheck("t-ortho"),
      dizzinessStanding: getCheck("t-dizzy"),
      irregularHeartbeat:getCheck("t-irregular"),
      rapidHeartRate:    getCheck("t-rapid-hr"),
      unusualFatigue:    getCheck("t-fatigue"),
      newCough:          getCheck("t-cough"),
      decreasedUrine:    getCheck("t-urine"),
    };
  }

  _hasAnyFormInput(data) {
    return Object.values(data).some((v) => v === true || (typeof v === "number" && v > 0));
  }

  async _handleAddSource() {
    const name = document.getElementById("new-source-name").value.trim();
    const url = document.getElementById("new-source-url").value.trim();
    const content = document.getElementById("new-source-content").value.trim();
    const type = document.getElementById("new-source-type").value;

    if (!name || !content) {
      this._showToast("Source name and content are required", "warning");
      return;
    }

    try {
      this.kb.addCustomSource({ name, url, type, description: name }, content);
      this._renderSources();
      this._updateKBStats();
      document.getElementById("add-source-modal").classList.add("hidden");

      // Clear form
      ["new-source-name", "new-source-url", "new-source-content"].forEach((id) => {
        document.getElementById(id).value = "";
      });

      this._showToast(`Source "${name}" added successfully`, "success");
    } catch (err) {
      this._showToast("Error adding source: " + err.message, "error");
    }
  }
}

// ─── Bootstrap ───────────────────────────────────────────────────────────────
document.addEventListener("DOMContentLoaded", async () => {
  window.app = new App();
  await window.app.initialize();
});
