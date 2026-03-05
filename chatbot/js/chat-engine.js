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
    if (saved) this.apiKey = saved;
    return this.apiKey;
  }

  setModel(model) {
    this.model = model;
  }

  clearHistory() {
    this.conversationHistory = [];
  }

  // ─── Intent Detection ────────────────────────────────────────────────────

  detectIntent(message) {
    // Off-topic: clearly non-HF subjects with no cardiac context
    const isOffTopic =
      /\b(tax(es)?|invest(ment)?|financ|insur|legal|lawsuit|weather|sport|politic|election|recipe|mechanic)\b/i.test(message) &&
      !/heart|cardiac|chest|breath|swell|sodium|fluid|medication|edema/i.test(message);

    // First-person present-tense language signals the patient is describing
    // their own current state (not asking a general education question)
    const firstPerson = /\b(i'?m|i am|i have|i'?ve|i feel|i notice[d]?|i'?ve been|i started|i'?ve had|i gained|i woke)\b/i.test(message);

    // Specific traffic-light symptom patterns (per Cornell tool criteria)
    const trafficLightPatterns = [
      /chest\s*(pain|discomfort|pressure|tightness)/i,
      /short(ness)?\s*of\s*breath|breathless|can'?t\s*breath|dyspnea/i,
      /swell(ing|en|ed)|edema|puffy\s*(leg|ankle|feet|foot)/i,
      /weight\s*gain|gained\s*\d+\s*(lb|pound|kg)/i,
      /faint(ing)?|pass(ed)?\s*out|collaps|los(t|ing)\s*consciousness/i,
      /racing\s*heart|heart\s*rac|palpitation|irregular\s*(heart|beat|pulse)/i,
      /dizzy|lightheaded/i,
      /confus(ed|ion)|disoriented/i,
      /pink\s*mucus|foamy\s*cough/i,
      /wak(e|ing)\s*up\s*(breathless|short\s*of\s*breath)|extra\s*pillow/i,
      /less\s*urin|not\s*urinat|decreased\s*urin/i,
    ];

    const hasTrafficLightSymptom = trafficLightPatterns.some((p) => p.test(message));

    // Only triage when patient is personally reporting current symptoms
    const isTriage = firstPerson && hasTrafficLightSymptom;

    return {
      isTriage,
      isOffTopic,
      isQuestion: /[?]/.test(message) || /^(what|how|when|why|can|should|is|are)\b/i.test(message),
      isGreeting: /^(hi|hello|hey|good\s*(morning|afternoon|evening))/i.test(message)
    };
  }

  // ─── System Prompt ───────────────────────────────────────────────────────

  buildSystemPrompt(context, isTriageMode = false) {
    const basePrompt = `You are a compassionate, knowledgeable heart failure self-management assistant.
Your purpose is to educate heart failure patients and caregivers about heart failure management only.

## Scope Boundary
You ONLY answer questions related to heart failure and its management (symptoms, diet, exercise, medications, monitoring, triage).
If a question is clearly outside this scope (e.g., taxes, weather, legal matters, unrelated cooking), respond with:
"I'm here to support you with heart failure-related questions. For [topic], you may want to consult [appropriate resource]."

## Citation Rules — IMPORTANT
- Use INLINE citations throughout your answer: write "According to [Source Name](URL), ..." or "...as noted by [Source Name](URL)."
- Do NOT save all citations for the end. Weave them naturally into sentences.
- End your response with a "## References" section listing every source cited.
- If the retrieved knowledge base context does not clearly support an answer, say:
  "I'm not finding enough reliable information in my current resources to answer that clearly. I can offer general guidance, but your care team is the best source for your specific situation."

## Triage Rules
- ONLY perform triage if the patient is personally describing CURRENT symptoms they are experiencing right now.
- Do NOT triage general/educational questions like "what are symptoms of HF?" or "what does chest pain mean?"
- If symptom details are insufficient for a proper assessment, ask specific follow-up questions BEFORE concluding. Example: "To better assess your situation, could you tell me: (1) How severe is your shortness of breath on a scale of 0–10? (2) Have you noticed any weight gain in the past day or week? (3) Is this happening at rest or only with activity?"
- When triage IS appropriate, clearly state the zone (🟢 GREEN / 🟡 YELLOW / 🔴 RED) and cite the Traffic Light Tool or guidelines as your basis.

## Core Principles
1. Cite sources INLINE using "According to [Name](URL), ..."
2. Be compassionate but clear. Use plain language.
3. Safety first — when in doubt, recommend contacting the healthcare team.
4. Never replace medical advice. Your guidance supplements the care team.
5. Be specific and actionable.

## Knowledge Base Context
The following has been retrieved for this query — use it as your primary evidence source:
---
${context}
---

## Response Format
- Weave inline citations naturally into sentences
- Use headers and bullets for clarity
- End with a ## References section
- End with an encouraging note when appropriate`;

    if (isTriageMode) {
      return basePrompt + `

## THIS MESSAGE CONTAINS CURRENT SYMPTOM DESCRIPTIONS
Acknowledge the specific symptoms mentioned, provide relevant education with inline citations, then note that a triage assessment is displayed below your response. Do not repeat the full triage logic in the text — it is shown in the triage panel.`;
    }

    return basePrompt;
  }

  // ─── Main Chat Function ──────────────────────────────────────────────────

  /**
   * Send a message and get a response with RAG context.
   * @param {string} userMessage
   * @param {Object} options - { triageMode: boolean }
   * @returns {Object} { content, sources, triageResult }
   */
  async chat(userMessage, options = {}) {
    if (!this.apiKey) {
      throw new Error("API key not set. Please enter your OpenAI API key in the settings panel.");
    }

    // Retrieve relevant knowledge
    const retrievedChunks = this.kb.retrieve(userMessage, 6);
    const context = this.kb.buildContext(retrievedChunks);
    const intent = this.detectIntent(userMessage);
    const isTriageMode = options.triageMode || intent.isTriage;

    // Build messages for OpenAI API (system message goes in messages array)
    const systemPrompt = this.buildSystemPrompt(context, isTriageMode);
    const messages = [
      { role: "system", content: systemPrompt },
      ...this.conversationHistory.slice(-this.maxHistoryTurns * 2),
      { role: "user", content: userMessage }
    ];

    // Call OpenAI API
    const response = await fetch("https://api.openai.com/v1/chat/completions", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "Authorization": `Bearer ${this.apiKey}`
      },
      body: JSON.stringify({
        model: this.model,
        max_tokens: 1500,
        messages
      })
    });

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      throw new Error(
        errorData.error?.message ||
        `API error: ${response.status} ${response.statusText}`
      );
    }

    const data = await response.json();
    const assistantMessage = data.choices[0]?.message?.content || "";

    // Update conversation history
    this.conversationHistory.push({ role: "user", content: userMessage });
    this.conversationHistory.push({ role: "assistant", content: assistantMessage });

    // Collect unique sources from retrieved chunks
    const usedSources = [];
    const seenSourceIds = new Set();
    retrievedChunks.forEach((chunk) => {
      if (chunk.source && !seenSourceIds.has(chunk.source.id)) {
        usedSources.push(chunk.source);
        seenSourceIds.add(chunk.source.id);
      }
    });

    return {
      content: assistantMessage,
      sources: usedSources,
      retrievedChunks,
      isTriageMode,
      usage: data.usage
    };
  }

  /**
   * Run AI-powered triage on symptom description
   * @param {string} symptomDescription
   * @param {Object|null} structuredSymptoms
   * @returns {Object} AI triage result
   */
  async runAITriage(symptomDescription, structuredSymptoms = null) {
    if (!this.apiKey) {
      throw new Error("API key not set.");
    }

    const triagePrompt = this.triage.buildAITriagePrompt(symptomDescription, structuredSymptoms);

    const response = await fetch("https://api.openai.com/v1/chat/completions", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "Authorization": `Bearer ${this.apiKey}`
      },
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
