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
    this.triageMode = "both"; // "rule", "ai", or "both"
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
    return msgDiv;
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
      const card = this._buildTriageCard("ai", "🤖 AI Reasoning", aiResult);
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

    // API key input
    document.getElementById("api-key-input").addEventListener("change", (e) => {
      this.chatEngine.setApiKey(e.target.value);
      this._updateConnectionStatus();
      this._showToast("API key saved", "success");
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
      const result = await this.chatEngine.chat(message, { triageMode: intent.isTriage });
      this._removeTypingIndicator();
      this._addMessage("assistant", result.content, result.sources);

      // Run triage if symptoms detected
      if (intent.isTriage && this.triageMode !== "none") {
        await this._runAutoTriage(message);
      }
    } catch (err) {
      this._removeTypingIndicator();
      this._addErrorMessage(err.message);
      this._showToast("Error: " + err.message, "error");
    } finally {
      this._setLoading(false);
    }
  }

  async _runAutoTriage(symptomText) {
    try {
      let ruleResult = null;
      let aiResult = null;
      const structuredData = this._readTriageForm();

      // Always run rule-based if we have structured data
      if (structuredData && this._hasAnyStructuredInput(structuredData)) {
        ruleResult = this.triageEngine.ruleBasedTriage(structuredData);
      }

      // Run AI triage if mode allows
      if (this.triageMode === "ai" || this.triageMode === "both") {
        this._setLoading(true, "Running AI triage assessment...");
        aiResult = await this.chatEngine.runAITriage(
          symptomText,
          (this.triageMode === "both" && ruleResult) ? structuredData : null
        );
      } else if (this.triageMode === "rule" && !ruleResult) {
        // Fallback rule triage without structured data
        ruleResult = this.triageEngine.ruleBasedTriage({});
      }

      if (ruleResult || aiResult) {
        this._renderTriageResults(ruleResult, aiResult);
        // Scroll triage panel into view
        document.getElementById("triage-panel").scrollIntoView({ behavior: "smooth", block: "nearest" });
      }
    } catch (err) {
      console.error("Triage error:", err);
    } finally {
      this._setLoading(false);
    }
  }

  async _handleRunTriage() {
    const structuredData = this._readTriageForm();
    const symptomText = document.getElementById("triage-symptom-text")?.value ||
      "Patient-reported symptoms from structured form";

    if (!this._hasAnyStructuredInput(structuredData) && !symptomText.trim()) {
      this._showToast("Please fill in symptom values or describe symptoms", "warning");
      return;
    }

    this._setLoading(true, "Running triage assessment...");
    try {
      let ruleResult = null;
      let aiResult = null;

      if (this.triageMode !== "ai") {
        ruleResult = this.triageEngine.ruleBasedTriage(structuredData);
      }

      if ((this.triageMode === "ai" || this.triageMode === "both") && this.chatEngine.apiKey) {
        aiResult = await this.chatEngine.runAITriage(symptomText, structuredData);
      } else if (this.triageMode === "ai" && !this.chatEngine.apiKey) {
        this._showToast("API key required for AI triage", "warning");
        ruleResult = this.triageEngine.ruleBasedTriage(structuredData);
      }

      this._renderTriageResults(ruleResult, aiResult);
    } catch (err) {
      this._showToast("Triage error: " + err.message, "error");
    } finally {
      this._setLoading(false);
    }
  }

  _readTriageForm() {
    const getVal = (id) => {
      const el = document.getElementById(id);
      return el ? el.value : null;
    };
    const getCheck = (id) => {
      const el = document.getElementById(id);
      return el ? el.checked : false;
    };
    const getNum = (id) => {
      const val = getVal(id);
      return val !== null && val !== "" ? parseFloat(val) : 0;
    };

    return {
      weightGain1Day: getNum("t-weight-day"),
      weightGain1Week: getNum("t-weight-week"),
      sobScale: getNum("t-sob"),
      chestPain: getCheck("t-chest-pain"),
      faintingOrLoss: getCheck("t-fainting"),
      pinkFoamyCough: getCheck("t-pink-cough"),
      severeConfusion: getCheck("t-confusion"),
      strokeSigns: getCheck("t-stroke"),
      swellingNewWorse: getCheck("t-swelling"),
      ortho: getCheck("t-ortho"),
      dizzinessStanding: getCheck("t-dizzy"),
      irregularHeartbeat: getCheck("t-irregular"),
      rapidHeartRate: getCheck("t-rapid-hr"),
      unusualFatigue: getCheck("t-fatigue"),
      newCough: getCheck("t-cough"),
      decreasedUrine: getCheck("t-urine"),
    };
  }

  _hasAnyStructuredInput(data) {
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
