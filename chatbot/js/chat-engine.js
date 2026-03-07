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

  detectIntent(message) {
    // Off-topic: clearly non-HF subjects with no cardiac context
    const isOffTopic =
      /\b(tax(es)?|invest(ment)?|financ|insur|legal|lawsuit|weather|sport|politic|election|recipe|mechanic)\b/i.test(message) &&
      !/heart|cardiac|chest|breath|swell|sodium|fluid|medication|edema/i.test(message);

    // First-person present-tense language signals the patient is describing
    // their own current state (not asking a general education question).
    // Fix #3: include "my/mine" possessive ownership so "My usual leg swelling is
    // getting worse" is correctly treated as a personal symptom report.
    const firstPerson = /\b(i'?m|i am|i have|i'?ve|i feel|i notice[d]?|i'?ve been|i started|i'?ve had|i gained|i woke|my|mine)\b/i.test(message);

    // Specific traffic-light symptom patterns (per Cornell tool criteria).
    // Fix #3: added "getting worse" + body-part patterns so worsening of known
    // symptoms (e.g. "my usual leg swelling is getting worse") correctly triggers triage.
    // Added fatigue pattern — it is listed on the traffic light symptom selector.
    const trafficLightPatterns = [
      /chest\s*(pain|discomfort|pressure|tightness)/i,
      /short(ness)?\s*of\s*breath|breathless|can'?t\s*breath|dyspnea/i,
      /swell(ing|en|ed)|edema|puffy\s*(leg|ankle|feet|foot)/i,
      // catches "my usual leg swelling is getting worse"
      /(?:leg|ankle|foot|feet|swell)\w*\s+(?:is\s+)?(?:getting\s+)?wors(?:e|ening)/i,
      /weight\s*(gain|is\s+up|going\s+up)|gained\s*\d+\s*(lb|pound|kg)/i,
      /faint(ing)?|pass(ed)?\s*out|collaps|los(t|ing)\s*consciousness/i,
      /racing\s*heart|heart\s*rac|palpitation|irregular\s*(heart|beat|pulse)/i,
      /dizzy|lightheaded/i,
      /confus(ed|ion)|disoriented/i,
      /fatigue|tired.*limit|exhaust.*activit/i,
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

  /**
   * @param {string} context - retrieved knowledge chunks
   * @param {boolean} isTriageMode - patient is reporting current symptoms
   * @param {Array<{key:string,question:string}>|null} followUpQuestions
   *   - if set, AI should ask ONLY these questions (do not triage yet)
   * @param {string[]} alreadyKnownInfo - facts patient already mentioned (don't re-ask)
   */
  buildSystemPrompt(context, isTriageMode = false, followUpQuestions = null, alreadyKnownInfo = []) {
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
- IMPORTANT: Do NOT include any triage zone badge (🟢/🟡/🔴), zone name (GREEN/YELLOW/RED), or urgency recommendation in your text. A separate triage assessment card is automatically generated and shown alongside your response — stating the zone in your text would duplicate it. Simply acknowledge the symptoms empathetically and provide relevant education.

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

    // ── Follow-up mode (Fix #4 & #5): AI asks specific missing questions ──────
    // Do NOT provide a triage zone yet — just gather the needed information.
    if (followUpQuestions && followUpQuestions.length > 0) {
      const questionList = followUpQuestions
        .map((q, i) => `${i + 1}. ${q.question}`)
        .join("\n");
      const knownSection = alreadyKnownInfo.length > 0
        ? `\n\n**Already known — do NOT ask again:**\n${alreadyKnownInfo.map(k => `- ${k}`).join("\n")}`
        : "";
      return basePrompt + `

## FOLLOW-UP QUESTIONS NEEDED (Fix #4 / #5)
The patient has described symptoms that need clarification before a full triage can be completed.
Your task in this response is to:
1. Briefly acknowledge what the patient described with empathy.
2. Ask ONLY the following specific follow-up questions — number them clearly:
${questionList}${knownSection}

IMPORTANT: Do NOT provide a triage zone or final recommendation yet.
Do NOT re-ask anything listed under "Already known" above.
After the patient answers, a full triage assessment will be shown.`;
    }

    // ── Triage mode: patient answered follow-ups / enough info already present ─
    if (isTriageMode) {
      const knownSection = alreadyKnownInfo.length > 0
        ? `\n\nAlready known from patient's description:\n${alreadyKnownInfo.map(k => `- ${k}`).join("\n")}\nDo NOT ask about these again.`
        : "";
      return basePrompt + `

## THIS MESSAGE CONTAINS CURRENT SYMPTOM DESCRIPTIONS${knownSection}
Acknowledge the patient's symptoms with empathy and provide relevant education with inline citations.
Do NOT include any triage zone (🟢/🟡/🔴 GREEN/YELLOW/RED), urgency level, or "call 911 / contact care team" recommendation in your text — a separate triage assessment card is shown automatically below your response. Avoid duplicating it.`;
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

    // Build messages for OpenAI API (system message goes in messages array).
    // Pass follow-up question context when provided (Fixes #4 & #5).
    const systemPrompt = this.buildSystemPrompt(
      context,
      isTriageMode,
      options.followUpQuestions || null,
      options.alreadyKnownInfo || []
    );
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
  async runAITriage(symptomDescription) {
    if (!this.apiKey) {
      throw new Error("API key not set.");
    }

    // AI receives only the free-text description — no rule-based result injected.
    // This keeps AI and rule-based systems fully independent (Fix #1).
    const triagePrompt = this.triage.buildAITriagePrompt(symptomDescription);

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
