/**
 * Chat Engine
 * Handles LLM interaction via OpenAI API.
 * Implements RAG: retrieves knowledge chunks, builds context, generates response.
 */

class ChatEngine {
  constructor(knowledgeBase, triageEngine) {
    this.kb = knowledgeBase;
    this.triage = triageEngine;
    this.apiKey = "";
    this.model = "gpt-4o-mini"; // Default: fast and cost-efficient
    this.conversationHistory = [];
    this.maxHistoryTurns = 10;
  }

  setApiKey(key) {
    this.apiKey = key.trim();
    localStorage.setItem("hf_chatbot_openai_key", this.apiKey);
  }

  loadApiKey() {
    const saved = localStorage.getItem("hf_chatbot_openai_key");
    if (saved) { this.apiKey = saved; return this.apiKey; }
    // Migrate key saved by earlier Claude/Anthropic version of the chatbot
    const legacy = localStorage.getItem("hf_chatbot_api_key");
    if (legacy && legacy.startsWith("sk-") && !legacy.startsWith("sk-ant-")) {
      // Looks like an OpenAI key that got stored under the old name
      this.apiKey = legacy;
      localStorage.setItem("hf_chatbot_openai_key", legacy);
      localStorage.removeItem("hf_chatbot_api_key");
    }
    return this.apiKey;
  }

  setModel(model) {
    this.model = model;
  }

  clearHistory() {
    this.conversationHistory = [];
  }

  // ─── Intent Detection ────────────────────────────────────────────────────

  /**
   * Detect general message intent (off-topic, question, greeting).
   * Triage detection is now handled exclusively by TriageEngine.detectSymptoms()
   * and TriageEngine.isPersonalSymptomReport() to keep logic in one place.
   */
  detectIntent(message) {
    const isOffTopic =
      /\b(tax(es)?|invest(ment)?|financ|insur|legal|lawsuit|weather|sport|politic|election|recipe|mechanic)\b/i.test(message) &&
      !/heart|cardiac|chest|breath|swell|sodium|fluid|medication|edema/i.test(message);

    return {
      isOffTopic,
      isQuestion: /[?]/.test(message) || /^(what|how|when|why|can|should|is|are)\b/i.test(message),
      isGreeting: /^(hi|hello|hey|good\s*(morning|afternoon|evening))/i.test(message)
    };
  }

  // ─── System Prompt ───────────────────────────────────────────────────────

  /**
   * Build system prompt for the LLM.
   * @param {string} context - retrieved knowledge chunks
   * @param {boolean} isTriageMode - true when patient described current symptoms
   * @param {Array<{key,question,category}>|null} followUpQuestions - structured follow-ups to ask
   */
  buildSystemPrompt(context, isTriageMode = false, followUpQuestions = null) {
    const basePrompt = `You are a compassionate, knowledgeable heart failure self-management assistant.
Your purpose is to educate heart failure patients and caregivers about heart failure management only.

## Scope Boundary
You ONLY answer questions related to heart failure and its management (symptoms, diet, exercise, medications, monitoring, triage).
If a question is clearly outside this scope, respond with:
"I'm here to support you with heart failure-related questions. For [topic], you may want to consult [appropriate resource]."

## Citation Rules — IMPORTANT
- Use INLINE citations throughout your answer: write "According to [Source Name](URL), ..."
- End your response with a "## References" section listing every source cited.
- If the knowledge base does not clearly support an answer, say so and direct the patient to their care team.

## Core Principles
1. Cite sources INLINE.
2. Be compassionate but clear. Use plain language.
3. Safety first — when in doubt, recommend contacting the healthcare team.
4. Never replace medical advice.

## Knowledge Base Context
${context}

## Response Format
- Weave inline citations naturally into sentences
- Use headers and bullets for clarity
- End with a ## References section`;

    // ── Follow-up mode: ask clarifying questions before triaging ─────────────
    if (followUpQuestions && followUpQuestions.length > 0) {
      // Group questions by category for clarity
      const byCategory = {};
      followUpQuestions.forEach(q => {
        if (!byCategory[q.category]) byCategory[q.category] = [];
        byCategory[q.category].push(q);
      });

      const questionList = Object.entries(byCategory).map(([cat, qs]) => {
        const catName = {
          sob: "Shortness of Breath", chestDiscomfort: "Chest Discomfort",
          fatigue: "Fatigue", weightChange: "Weight Change",
          confusion: "Confusion", legSwelling: "Leg/Ankle Swelling",
          lightheaded: "Dizziness/Lightheadedness"
        }[cat] || cat;
        return `**${catName}:**\n` + qs.map((q, i) => `${i + 1}. ${q.question}`).join("\n");
      }).join("\n\n");

      return basePrompt + `

## YOUR TASK: Gather Information Before Triage
The patient has described symptoms that need clarification before a triage assessment can be completed.

1. Briefly acknowledge what they described with empathy (1–2 sentences).
2. Ask ONLY the following questions, grouped by symptom — number them clearly:

${questionList}

IMPORTANT:
- Do NOT provide a triage zone or zone recommendation yet.
- Do NOT speculate on severity before having the answers.
- A triage assessment will be shown after the patient responds.`;
    }

    // ── Triage mode: acknowledge symptoms and let the triage card do the work ──
    if (isTriageMode) {
      return basePrompt + `

## THIS MESSAGE CONTAINS CURRENT SYMPTOM DESCRIPTIONS
The patient is describing symptoms they are experiencing right now.
- Acknowledge the specific symptoms with empathy.
- Provide relevant education with inline citations.
- Note that a triage assessment is displayed below your response.
- Do NOT repeat zone classifications in the text — the triage card shows them.`;
    }

    // ── Education mode: general Q&A, no triage zone ──────────────────────────
    return basePrompt + `

## EDUCATION MODE
The patient is asking a general question, not reporting current symptoms.
Do NOT provide a triage zone. Provide evidence-based educational information.`;
  }

  // ─── Main Chat Function ──────────────────────────────────────────────────

  /**
   * Send a message and get a response with RAG context.
   * @param {string} userMessage
   * @param {Object} options - { triageMode, followUpQuestions }
   * @returns {Object} { content, sources }
   */
  async chat(userMessage, options = {}) {
    if (!this.apiKey) {
      throw new Error("API key not set. Please enter your OpenAI API key in the settings panel.");
    }

    const retrievedChunks = this.kb.retrieve(userMessage, 6);
    const context = this.kb.buildContext(retrievedChunks);

    const systemPrompt = this.buildSystemPrompt(
      context,
      options.triageMode || false,
      options.followUpQuestions || null
    );

    const messages = [
      { role: "system", content: systemPrompt },
      ...this.conversationHistory.slice(-this.maxHistoryTurns * 2),
      { role: "user", content: userMessage }
    ];

    const response = await fetch("https://api.openai.com/v1/chat/completions", {
      method: "POST",
      headers: { "Content-Type": "application/json", "Authorization": `Bearer ${this.apiKey}` },
      body: JSON.stringify({ model: this.model, max_tokens: 1500, messages })
    });

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      throw new Error(errorData.error?.message || `API error: ${response.status} ${response.statusText}`);
    }

    const data = await response.json();
    const assistantMessage = data.choices[0]?.message?.content || "";

    this.conversationHistory.push({ role: "user", content: userMessage });
    this.conversationHistory.push({ role: "assistant", content: assistantMessage });

    const usedSources = [];
    const seenSourceIds = new Set();
    retrievedChunks.forEach((chunk) => {
      if (chunk.source && !seenSourceIds.has(chunk.source.id)) {
        usedSources.push(chunk.source);
        seenSourceIds.add(chunk.source.id);
      }
    });

    return { content: assistantMessage, sources: usedSources, retrievedChunks, usage: data.usage };
  }

  /**
   * Run AI-powered triage — fully independent of rule-based result.
   * @param {string} symptomDescription - free-text symptoms
   * @param {string[]} detectedCategories - hint for the AI about which categories were detected
   * @returns {Object} AI triage result
   */
  async runAITriage(symptomDescription, detectedCategories = []) {
    if (!this.apiKey) throw new Error("API key not set.");

    const triagePrompt = this.triage.buildAITriagePrompt(symptomDescription, detectedCategories);

    const response = await fetch("https://api.openai.com/v1/chat/completions", {
      method: "POST",
      headers: { "Content-Type": "application/json", "Authorization": `Bearer ${this.apiKey}` },
      body: JSON.stringify({
        model: this.model,
        max_tokens: 1000,
        messages: [{ role: "user", content: triagePrompt }]
      })
    });

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      throw new Error(errorData.error?.message || `API error: ${response.status}`);
    }

    const data = await response.json();
    const aiResponse = data.choices[0]?.message?.content || "";
    return this.triage.parseAITriageResponse(aiResponse, symptomDescription);
  }
}

window.ChatEngine = ChatEngine;
