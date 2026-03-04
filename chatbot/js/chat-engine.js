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
    const lower = message.toLowerCase();
    const triageKeywords = [
      "symptom", "feeling", "feel", "breathe", "breathing", "breath",
      "chest", "pain", "swelling", "swollen", "weight", "gained", "dizzy",
      "dizziness", "faint", "confused", "palpitation", "heartbeat",
      "emergency", "should i go", "call 911", "hospital", "urgent",
      "worse", "worsening", "triage", "dangerous", "serious", "severe",
      "cough", "tired", "fatigue", "ankle", "edema", "fluid"
    ];
    const isTriage = triageKeywords.some((kw) => lower.includes(kw));
    return {
      isTriage,
      isQuestion: message.includes("?") || lower.startsWith("what") ||
        lower.startsWith("how") || lower.startsWith("when") ||
        lower.startsWith("why") || lower.startsWith("can") || lower.startsWith("should"),
      isGreeting: /^(hi|hello|hey|good morning|good afternoon|good evening)/i.test(message)
    };
  }

  // ─── System Prompt ───────────────────────────────────────────────────────

  buildSystemPrompt(context, isTriageMode = false) {
    const basePrompt = `You are a compassionate, knowledgeable heart failure self-management assistant.
Your purpose is to educate heart failure patients and caregivers about:
- Heart failure basics, symptoms, and disease management
- Diet (sodium and fluid restrictions), exercise, and medications
- Daily monitoring practices (weight, blood pressure)
- When to contact the care team vs. when to call 911

## Core Principles
1. **Always cite your sources**: When providing information, reference specific sources (name and URL if available) from the knowledge base context provided.
2. **Be compassionate but clear**: Use plain language. Be warm and supportive.
3. **Safety first**: When in doubt, recommend contacting the healthcare team.
4. **Never replace medical advice**: Remind users that your guidance supplements, not replaces, their care team.
5. **Be specific**: Give actionable, specific information rather than vague advice.

## Citation Format
When citing information, use this format inline: (Source: [Source Name](URL))
At the end of your response, include a "## References" section listing all sources cited.

## Knowledge Base Context
The following information has been retrieved from the knowledge base to help answer this query:

---
${context}
---

## Response Guidelines
- Keep responses clear and well-organized with headers where helpful
- For symptom questions, always mention that the patient should contact their care team
- For emergency symptoms (chest pain, severe breathlessness, fainting), always emphasize calling 911
- Use bullet points for lists of symptoms, actions, or tips
- End with an encouraging, supportive note when appropriate`;

    if (isTriageMode) {
      return basePrompt + `

## TRIAGE MODE
This message appears to contain symptom descriptions. After providing educational information:
1. Acknowledge the symptoms the patient mentioned
2. Note that the chatbot is running a triage assessment
3. Remind them that triage results are shown below the chat`;
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
