/**
 * Heart Failure Triage Engine
 * Implements two triage modes:
 * 1. Rule-based: Cornell Traffic Light Tool (exact logic from reference implementation)
 * 2. AI-assisted: LLM reasoning (independent clinical assessment — not bound by rule logic)
 *
 * Zone list (matches reference): ["green", "yellow_consider", "yellow_call", "red"]
 *
 * 7 symptom categories covered by the traffic light tool:
 *   sob | chestDiscomfort | fatigue | weightChange | confusion | legSwelling | lightheaded
 */

/**
 * Build OpenAI API request body with correct parameters per model generation.
 * - GPT-4.x: max_tokens + temperature
 * - GPT-5.4 (non-reasoning): max_completion_tokens + temperature
 * - Reasoning models (o-series, gpt-5, gpt-5.6-*, gpt-6*): max_completion_tokens only.
 *   They reject any temperature other than the default, and hidden reasoning counts
 *   against the output limit, so the limit is raised to leave room for the answer.
 * @param {string} model
 * @param {number} maxTokens
 * @param {Array} messages
 * @param {number|null} temperature  null = omit; ignored for reasoning models
 */
function modelAcceptsTemperature(model) {
  return /^gpt-4/i.test(model) || /^gpt-5\.4/i.test(model);
}

function buildOpenAIBody(model, maxTokens, messages, temperature = 0.7) {
  const isGpt4      = /^gpt-4/i.test(model);
  const isReasoning = !modelAcceptsTemperature(model);
  const body = { model, messages };

  if (isGpt4) {
    body.max_tokens = maxTokens;
  } else {
    body.max_completion_tokens = isReasoning ? Math.max(maxTokens, 8000) : maxTokens;
  }

  if (!isReasoning && temperature !== null) {
    body.temperature = temperature;
  }

  return body;
}
window.buildOpenAIBody = buildOpenAIBody;

class TriageEngine {
  constructor() {
    this.triageHistory = [];
    this.ZONE_LIST = ["green", "yellow_consider", "yellow_call", "red"];

    // Human-readable labels per category
    this.CATEGORY_LABELS = {
      sob:            "Shortness of breath",
      chestDiscomfort:"Chest discomfort",
      fatigue:        "Fatigue",
      weightChange:   "Weight change",
      confusion:      "Confusion / mental fogginess",
      legSwelling:    "Leg or ankle swelling",
      lightheaded:    "Dizziness / lightheadedness"
    };
  }

  // ─── Symptom Detection ────────────────────────────────────────────────────

  /**
   * Detect which of the 7 traffic light symptom categories are present in text.
   * Returns an array of category keys.
   * @param {string} text
   * @returns {string[]}
   */
  detectSymptoms(text) {
    const detected = [];
    const t = text;

    // ── Shortness of breath ──────────────────────────────────────────────────
    if (/short(ness)?\s*of\s*breath|breathless|can'?t\s*(catch\s*my\s*)?breath|dyspnea|hard\s*to\s*breath|difficulty\s*breath|trouble\s*breath|out\s*of\s*breath|winded|gasping|struggling\s*to\s*breath|labored\s*breath|can'?t\s*breathe|hard\s*to\s*breathe/i.test(t))
      detected.push("sob");

    // ── Chest discomfort ─────────────────────────────────────────────────────
    if (/chest\s*(discomfort|ache|pain|pressure|tightness|heaviness|hurt)|tight(ness)?\s*(in|around)?\s*(my\s*)?chest|chest\s*feel(s|ing)?\s*(tight|heavy|weird|strange)|squeezing\s*(in|around|on)?\s*(my\s*)?chest|pressure\s*(in|on|around)\s*(my\s*)?chest/i.test(t))
      detected.push("chestDiscomfort");

    // ── Fatigue — broad synonyms ─────────────────────────────────────────────
    // Covers "very tired", "too tired", "feeling weak", "no strength", etc.
    if (/\b(fatigue|fatigued|tired|tiredness|exhausted|exhaustion|no\s*energy|worn\s*out|wiped\s*out|weak(?:ness)?|lethargic|lethargy|sluggish|drained|run\s*down|burned?\s*out|no\s*strength|can'?t\s*keep\s*up|feel(ing)?\s*(so\s*)?(weak|tired|exhausted|run\s*down))\b/i.test(t))
      detected.push("fatigue");

    // ── Weight change ─────────────────────────────────────────────────────────
    // Covers "gained some weight", "put on weight", "heavier", "scale went up"
    if (/weight\s*(gain|loss|change|up|down|increase|decrease|going\s*(up|down)|went\s*up|is\s*up)|gained\s*(?:some|a\s*(?:bit|few|lot|couple)|about|\d+)?\s*(?:weight|pounds?|lbs?|kg)|put\s*on\s*(?:some|a\s*(?:bit|few))?\s*(?:weight|pounds?)|lost\s*(?:some|\d+)\s*(?:weight|pounds?|lbs?|kg)|scale\s*(?:is|went|has\s*gone|shows?)\s*(?:up|higher|more)|heavier\s*than\s*(usual|before|yesterday|normal)/i.test(t))
      detected.push("weightChange");

    // ── Confusion / mental fogginess ─────────────────────────────────────────
    if (/confus(ed|ion)|disoriented|not\s*think(ing)?\s*clearly|mental\s*status|foggy|brain\s*fog|muddled|can'?t\s*think|trouble\s*think|hard\s*to\s*think|forgetful|memory\s*(loss|problem|issue)|mind\s*blank|unclear\s*thinking/i.test(t))
      detected.push("confusion");

    // ── Leg / ankle / foot swelling ───────────────────────────────────────────
    // Covers "puffier", "look swollen", "my ankles look bigger", etc.
    if (/swell(ing|en|ed)|edema|puffy\s*(leg|ankle|feet|foot)|puffier|(?:leg|ankle|foot|feet|lower\s*leg)\s*(?:look|feel|are|is)?\s*(?:puff|swoll|bigger|larger|blow|heavier|worse)|swollen\s*(ankle|leg|feet|foot)|bloated.*(leg|ankle|feet)/i.test(t))
      detected.push("legSwelling");

    // ── Lightheadedness / dizziness / falls ──────────────────────────────────
    if (/dizzy|dizziness|lightheaded|light-headed|vertigo|feel\s*(faint|like\s*(?:i'?m\s*)?(?:faint|pass(?:ing)?\s*out|going\s*to\s*faint))|nearly\s*faint|almost\s*faint|unsteady|balance\s*(problem|issue|off)|fell\s*down|fall(?:ing)?\s*(down|over)|los(?:t|ing)\s*(?:my\s*)?balance/i.test(t))
      detected.push("lightheaded");

    return detected;
  }

  /**
   * LLM-based symptom detection: send the patient message to the LLM and ask it to
   * classify which of the 7 traffic light categories are present and whether it's a
   * personal symptom report. Far more robust than regex for colloquial language.
   *
   * @param {string} text - patient message
   * @param {string} apiKey - OpenAI API key
   * @param {string} model - model ID (defaults to gpt-4o-mini for speed)
   * @returns {Promise<{categories: string[], isPersonalReport: boolean}>}
   */
  async detectSymptomsWithLLM(text, apiKey, model = "gpt-4o-mini") {
    const prompt = `You are a clinical symptom classifier for a heart failure triage tool.
Given a patient message, identify which of these 7 symptom categories are PERSONALLY reported and whether the message is a personal symptom report.

Categories (use these exact keys):
- sob: Shortness of breath, breathlessness, difficulty breathing, can't catch breath, winded, gasping
- chestDiscomfort: Chest pain, pressure, tightness, heaviness, chest ache, squeezing in chest
- fatigue: Fatigue, exhaustion, tiredness, weakness, no energy, worn out, wiped out, drained
- weightChange: Weight GAIN only (not loss) — scale went up, gained weight/pounds/lbs, heavier than usual, put on weight
- confusion: Confusion, disorientation, foggy thinking, can't think clearly, memory problems
- legSwelling: Leg, ankle, or foot swelling, puffiness, edema, ankles look bigger/puffy
- lightheaded: Dizziness, lightheadedness, nearly fainted, unsteady, fell, balance problems

Patient message:
"""
${text}
"""

Return ONLY valid JSON — no explanation, no markdown:
{"categories": ["sob", "fatigue"], "denied": ["legSwelling"], "isPersonalReport": true}

Rules:
- denied: categories the patient explicitly says they do NOT have ("no leg swelling", "I have no shortness of breath", "no chest pain"). A denied category must NEVER appear in categories — the traffic light tool only assesses a symptom the patient actually has. Use [] when nothing is denied.
- categories: symptoms the patient is PERSONALLY REPORTING — include whether acute, chronic, stable, or recurring. A 2-week-old stable ankle swelling still counts as legSwelling. Only return [] if it is a general/education question with no personal symptom report.
- isPersonalReport:
  - true → patient describing their own symptoms (current, recent, or ongoing): "I gained 5 lbs", "my legs are swollen", "I've been having ankle swelling for 2 weeks", "I feel tired lately"
  - true even with a question attached: "I gained 5 lbs this week — is this serious?", "my ankles are swollen, should I call?"
  - false → general education question: "What causes weight gain in HF?", "How is leg swelling treated?", "What are HF symptoms?"
  - false → hypothetical: "If someone has leg swelling, what does it mean?"
- Only include a category if the patient is REPORTING that symptom personally, not merely asking about it
- Include weightChange ONLY for weight GAIN — weight loss alone does NOT trigger it`;

    const response = await fetch("https://api.openai.com/v1/chat/completions", {
      method: "POST",
      headers: { "Content-Type": "application/json", "Authorization": `Bearer ${apiKey}` },
      body: JSON.stringify(buildOpenAIBody(model, 160, [{ role: "user", content: prompt }], 0))
    });

    if (!response.ok) {
      const err = await response.json().catch(() => ({}));
      throw new Error(err.error?.message || `API error: ${response.status}`);
    }

    const data = await response.json();
    const raw = data.choices[0]?.message?.content || "{}";
    const jsonMatch = raw.match(/\{[\s\S]*\}/);
    if (jsonMatch) {
      const parsed = JSON.parse(jsonMatch[0]);
      const denied = Array.isArray(parsed.denied) ? parsed.denied.filter(c => this.CATEGORY_LABELS[c]) : [];
      return {
        // A symptom the patient denies is never assessed (the tool only runs on a symptom that is present)
        categories:      Array.isArray(parsed.categories) ? parsed.categories.filter(c => this.CATEGORY_LABELS[c] && !denied.includes(c)) : [],
        denied,
        isPersonalReport: parsed.isPersonalReport === true
      };
    }
    return { categories: [], denied: [], isPersonalReport: false };
  }

  /**
   * LLM-based answer extraction: given patient text and detected categories, ask the LLM
   * to extract structured answers for each category's traffic-light questions.
   * Returns null for fields not mentioned so follow-up questions are triggered correctly.
   *
   * @param {string} text - patient text (original + any follow-up answers)
   * @param {string[]} categories - detected category keys
   * @param {string} apiKey
   * @param {string} model
   * @returns {Promise<Object>} { sob: {...}, weightChange: {...}, ... }
   */
  async extractAnswersWithLLM(text, categories, apiKey, model = "gpt-4o-mini") {
    // Per-category field specs matching the traffic light tool questions exactly
    const catSpecs = {
      sob: {
        desc: "Shortness of Breath",
        fields: `- isNew: "yes" (new symptom, never had before) | "no" (recurring/ongoing — a symptom the patient already had that is getting worse is NOT new) | null
- coSymptoms: array — include each item from ["chest_pain:severe","chest_pain:not_severe","fever","cough","leg_swelling","mucus","wheeze"] that is clearly present. Use "chest_pain:severe" for clearly severe/crushing chest pain; "chest_pain:not_severe" for mild/moderate chest pain.
- worseAtRest: "yes" (short of breath at rest, e.g. while just sitting or lying down, or it gets worse at rest) | "no" (only short of breath with activity; not at rest) | null (not stated)
- worseWithActivity: "yes" (worse when walking around, climbing stairs, exercising) | "no" | null`
      },
      chestDiscomfort: {
        desc: "Chest Discomfort",
        fields: `- isNew: "yes" | "no" | null
- coSymptoms: array from ["shooting_pain","sob","heart_racing","sweating","nausea","regurg"] present
  (shooting_pain = the tool's option "Shooting pain down arm or pain on left side of chest": pain that shoots, goes into, spreads or radiates to an arm, or pain on the left side of the chest)
- isAtRest: "rest" (occurs at rest/sitting/lying) | "activity" (only with exertion) | null`
      },
      fatigue: {
        desc: "Fatigue / Unusual Tiredness",
        fields: `- coSymptoms: array from ["fever","sob","weight_change","leg_swelling","cough","lightheaded"] present
- isWorseUsual: "yes" | "no" | null
- canDoActivities: "yes" (still managing normal daily activities) | "no" (fatigue preventing them) | null`
      },
      weightChange: {
        desc: "Weight Gain (ONLY weight GAIN counts — weight loss is not a triage trigger)",
        fields: `- changeType: "day_gain" (patient reports gaining ≥2 lbs since yesterday or in one day) | "week_gain" (patient reports gaining >5 lbs in past week) | "other" (weight gain present but below those amounts, or amount/timeframe not clear) | null (no weight gain described)
- tookDiuretic: "yes" (patient says they ARE taking water pills / diuretics as prescribed) | "no" (patient says they are NOT taking them) | "unsure" | null (not mentioned)
- coSymptoms: array from ["sob","cough","leg_swelling","nausea","bowel_changes"] present`
      },
      confusion: {
        desc: "Confusion / Mental Status Change",
        fields: `- isNew: "yes" | "no" | null
- coSymptoms: array from ["unconscious","slurred_speech","face_asym","weakness","fever","urination","lightheaded"] present`
      },
      legSwelling: {
        desc: "Leg / Ankle Swelling",
        fields: `- isNew: "yes" (the leg swelling is a new symptom) | "no" (has had it before / ongoing) | null (not stated)
- isWorse: "yes" (worse than previous days, e.g. goes up higher now) | "no" (same as before / not worse) | null (not stated)
- takingWaterPills: "yes" (patient says they HAVE been taking their regular water pills / diuretics, e.g. Lasix) | "no" (patient says they have NOT been taking them) | null (not mentioned). This is about their regular water pills, NOT about an extra dose.
- legs: "one" (patient explicitly says one leg/ankle, e.g. "my right ankle", "one leg") | "both" (patient explicitly says both legs/ankles, bilateral, or uses the plural "ankles"/"legs"/"feet") | null (not stated — do NOT infer from "ankle" singular)
- dietChange: "yes" (recent diet change such as eating more salt) | "no" | null (not mentioned)`
      },
      lightheaded: {
        desc: "Dizziness / Lightheadedness / Falls",
        fields: `- isNew: "yes" | "no" | null (is this a new symptom the patient has not had before?)
- fellToday: "yes" | "no" | "unsure" | null (did the patient fall today?)
- bloodPressure: "yes" (patient says their blood pressure is HIGH or LOW) | "no" (patient says their blood pressure is normal) | "unsure" | null (blood pressure not mentioned)`
      }
    };

    const emptyStructures = {
      sob:            { isNew: null, coSymptoms: [], worseAtRest: null, worseWithActivity: null },
      chestDiscomfort:{ isNew: null, coSymptoms: [], isAtRest: null },
      fatigue:        { coSymptoms: [], isWorseUsual: null, canDoActivities: null },
      weightChange:   { changeType: null, tookDiuretic: null, coSymptoms: [] },
      confusion:      { isNew: null, coSymptoms: [] },
      legSwelling:    { isNew: null, isWorse: null, takingWaterPills: null, legs: null, dietChange: null },
      lightheaded:    { isNew: null, fellToday: null, bloodPressure: null }
    };

    const relevantCats = categories.filter(c => catSpecs[c]);
    const fieldDocs = relevantCats
      .map(c => `### ${catSpecs[c].desc} (key: "${c}")\n${catSpecs[c].fields}`)
      .join("\n\n");
    const skeleton = relevantCats.reduce((acc, c) => {
      acc[c] = emptyStructures[c] || {};
      return acc;
    }, {});

    const prompt = `Extract clinical information from this patient message for heart failure symptom assessment.

Patient message:
"""
${text}
"""

For each category below, extract the specified fields. Return null for anything not clearly stated or strongly implied — do NOT infer or guess.

${fieldDocs}

Return ONLY this JSON (fill in actual values or keep null):
${JSON.stringify(skeleton, null, 2)}`;

    const response = await fetch("https://api.openai.com/v1/chat/completions", {
      method: "POST",
      headers: { "Content-Type": "application/json", "Authorization": `Bearer ${apiKey}` },
      body: JSON.stringify(buildOpenAIBody(model, 600, [{ role: "user", content: prompt }], 0))
    });

    if (!response.ok) {
      const err = await response.json().catch(() => ({}));
      throw new Error(err.error?.message || `API error: ${response.status}`);
    }

    const data = await response.json();
    const raw = data.choices[0]?.message?.content || "{}";
    const jsonMatch = raw.match(/\{[\s\S]*\}/);
    if (jsonMatch) {
      try {
        const parsed = JSON.parse(jsonMatch[0]);
        // Safety net for the tool's RED option "Shooting pain down arm or pain on left side of chest":
        // if the patient's own words describe it, record it even when the LLM left it out.
        if (parsed.chestDiscomfort && this._describesArmOrLeftChestPain(text)) {
          const co = Array.isArray(parsed.chestDiscomfort.coSymptoms) ? parsed.chestDiscomfort.coSymptoms : [];
          if (!co.includes("shooting_pain")) co.push("shooting_pain");
          parsed.chestDiscomfort.coSymptoms = co;
        }
        return parsed;
      } catch (e) {
        console.warn("Failed to parse LLM extraction response:", e);
      }
    }
    return {};
  }

  /**
   * True when the text describes the tool's chest option "Shooting pain down arm or pain on left
   * side of chest" and does not deny it ("no pain in my arm").
   */
  _describesArmOrLeftChestPain(text) {
    const arm  = /(pain|ache|aching|pressure|tightness|discomfort)[^.?!]{0,60}?(goes|going|spread(s|ing)?|radiat(es|ing)|shoot(s|ing)?|travel(s|l?ing)?|mov(es|ing)|run(s|ning)?)\s*(down|in|into|to|through|up)\s*(to\s*)?(my|the)?\s*(left|right)?\s*(arm|shoulder)|shooting\s*pain\s*(down|in|into)\s*(my|the)?\s*(left|right)?\s*arm|pain\s*(down|in)\s*(my|the)\s*(left|right)?\s*arm/i;
    const left = /(pain|ache|pressure|tightness)\s*(is\s*)?(on|in)\s*(the|my)\s*left\s*side\s*of\s*(the|my)\s*chest|left[-\s]*sided\s*chest\s*(pain|pressure|tightness)/i;
    const denied = /\b(no|not|without|don'?t\s*have|doesn'?t|isn'?t|never)\b[^.?!]{0,40}(arm|left\s*side)/i;
    return (arm.test(text) || left.test(text)) && !denied.test(text);
  }

  /**
   * Detect cardiac-related queries that aren't in the 7 traffic light categories
   * (e.g. fast heartbeat, palpitations without chest discomfort or SOB context).
   * Used to ask a clarifying HF-diagnosis question in education mode.
   * @param {string} text
   * @returns {boolean}
   */
  detectCardiacNonTriageQuery(text) {
    const cardiacPatterns =
      /heart\s*(beat(ing)?|racing|pounding|flutter|skip|pound|thump)|rapid\s*(heart|pulse|beat)|palpitation|tachycardia|arrhythmia|afib|irregular\s*(heart|beat|rhythm|pulse)|my\s*heart\s*(was|is|has\s*been)\s*(beat|racing|pound|flutter|going\s*fast|skipping)/i;
    const hfAlreadyMentioned =
      /heart\s*failure|hf\b|cardiomyopathy|ejection\s*fraction|cardiologist|home\s*care.*heart|heart.*home\s*care/i;
    return cardiacPatterns.test(text) && !hfAlreadyMentioned.test(text);
  }

  /**
   * Returns true when the text reads as a first-person, present-tense symptom report
   * (as opposed to a general education question).
   * @param {string} text
   * @returns {boolean}
   */
  isPersonalSymptomReport(text) {
    return /\b(i'?m|i am|i have|i'?ve|i feel|i notice[d]?|i'?ve been|i started|i'?ve had|i gained|i lost|i woke|my|mine)\b/i.test(text);
  }

  // ─── Answer Extraction ────────────────────────────────────────────────────

  /**
   * Extract structured answers for one symptom category from free text.
   * @param {string} text - all patient text (original + any follow-up answers)
   * @param {string} category
   * @returns {Object} - category-specific answer object; null = not yet known
   */
  extractAnswers(text, category) {
    switch (category) {
      case "sob":            return this._extractSobAnswers(text);
      case "chestDiscomfort":return this._extractChestAnswers(text);
      case "fatigue":        return this._extractFatigueAnswers(text);
      case "weightChange":   return this._extractWeightAnswers(text);
      case "confusion":      return this._extractConfusionAnswers(text);
      case "legSwelling":    return this._extractLegSwellingAnswers(text);
      case "lightheaded":    return this._extractLightheadedAnswers(text);
      default:               return {};
    }
  }

  // ── Per-category extractors ──────────────────────────────────────────────

  _extractSobAnswers(text) {
    // answer_list: [isNew, coSymptoms_str, worseAtRest, worseWithActivity]  (the tool's q0–q3)
    const a = { isNew: null, coSymptoms: [], worseAtRest: null, worseWithActivity: null };

    // [0] new?
    if (/\b(new\b|first\s*time|never.*before|just\s*started|started\s*recently|recently\s*start)\b/i.test(text))
      a.isNew = "yes";
    else if (/\b(not\s*new|existing|had.*before|usual|chronic|same\s*as\s*before|always\s*had|happen\s*(regularly|before))\b/i.test(text))
      a.isNew = "no";

    // [1] co-symptoms (SOB-specific: chest_pain:severe, chest_pain:not_severe, fever, cough, leg_swelling, mucus, wheeze)
    if (/chest\s*(pain|pressure|tightness|heaviness).*(severe|very\s*bad|intense|crushing|terrible|10\s*out\s*of)/i.test(text) ||
        /(severe|very\s*bad|intense|crushing|terrible).*chest\s*(pain|pressure|tightness|heaviness)/i.test(text))
      a.coSymptoms.push("chest_pain:severe");
    else if (/chest\s*(pain|pressure|tightness|heaviness)/i.test(text))
      a.coSymptoms.push("chest_pain:not_severe");

    // Fever: catch both the word "fever" AND numeric temperature (e.g. "temperature of 101.8°F", "38.5°C", "temp 100.4")
    if (/\bfever\b|\btemp(erature)?\s*(of\s*)?(?:1(?:0[0-9]|[1-9]\d)(\.\d+)?)\s*°?[Ff]|\btemp(erature)?\s*(of\s*)?(?:3[89]|4[0-1])(\.\d+)?\s*°?[Cc]/i.test(text))
      a.coSymptoms.push("fever");
    if (/\bcough(ing)?\b/i.test(text))        a.coSymptoms.push("cough");
    if (/swell(ing|en)?|swollen|edema/i.test(text)) a.coSymptoms.push("leg_swelling");
    if (/mucus|phlegm|sputum/i.test(text))    a.coSymptoms.push("mucus");
    if (/wheez(ing|e)/i.test(text))           a.coSymptoms.push("wheeze");

    // [2] worse at rest?  (tool: "Is shortness of breath worse at rest, e.g. while the patient is just sitting?")
    if (/not\s*(short\s*of\s*breath\s*)?at\s*rest|only\s*(when|with|during|on)\s*(i\s*)?(walk|climb|exert|activ|mov|exercis)|fine\s*(at|when)\s*(rest|sitting)/i.test(text))
      a.worseAtRest = "no";
    else if (/\bat\s*rest\b|while\s*(just\s*)?(sitting|resting|lying)|even\s*(when\s*)?(sitting|resting|lying)|when\s*(i\s*)?(sit|lie|lay)\b|lying\s*(down|flat)/i.test(text))
      a.worseAtRest = "yes";

    // [3] worse with activity?  (tool: "Is shortness of breath worse with activity, e.g. walking around, exercising?")
    if (/(walk|climb|stairs|exert|exercis|activit|moving\s*around)/i.test(text))
      a.worseWithActivity = "yes";

    return a;
  }

  _extractChestAnswers(text) {
    // answer_list: [isNew, coSymptoms_str, isAtRest]
    const a = { isNew: null, coSymptoms: [], isAtRest: null };

    // [0] new?
    if (/\b(new\b|first\s*time|never.*before|just\s*started)\b/i.test(text))
      a.isNew = "yes";
    else if (/\b(not\s*new|existing|had.*before|usual|chronic|same\s*as\s*before)\b/i.test(text))
      a.isNew = "no";

    // [1] co-symptoms (shooting_pain is RED; sob, heart_racing, sweating, nausea, regurg → YELLOW logic)
    if (/shooting\s*(pain|sensation).*arm|arm.*shooting|pain.*shoot(ing)?\s*down|shoot(s|ing)?\s*down\s*(my|the)?\s*arm/i.test(text))
      a.coSymptoms.push("shooting_pain");
    else if (this._describesArmOrLeftChestPain(text))
      a.coSymptoms.push("shooting_pain");
    if (/short(ness)?\s*of\s*breath|breathless/i.test(text))
      a.coSymptoms.push("sob");
    if (/racing.*heart|heart.*rac|palpitation|heart\s*flutter/i.test(text))
      a.coSymptoms.push("heart_racing");
    if (/\bsweat(ing)?\b/i.test(text))
      a.coSymptoms.push("sweating");
    if (/\b(nausea|nauseat|vomit)\b/i.test(text))
      a.coSymptoms.push("nausea");
    if (/regurgitat|heartburn|indigestion|acid\s*reflux|sour\s*taste/i.test(text))
      a.coSymptoms.push("regurg");

    // [2] rest vs activity
    if (/\bat\s*rest\b|resting|just\s*sitting|while\s*sitting|lying\s*(down|flat)|not\s*(being\s*)?active/i.test(text))
      a.isAtRest = "rest";
    else if (/during\s*activit|while\s*(being\s*)?active|walking|exercis|moving|going\s*up|climbing/i.test(text))
      a.isAtRest = "activity";

    return a;
  }

  _extractFatigueAnswers(text) {
    // answer_list: [coSymptoms_str, isWorseUsual, canDoActivities]
    const a = { coSymptoms: [], isWorseUsual: null, canDoActivities: null };

    // [0] co-symptoms
    if (/\bfever\b/i.test(text))                                         a.coSymptoms.push("fever");
    if (/short(ness)?\s*of\s*breath|breathless/i.test(text))             a.coSymptoms.push("sob");
    if (/weight\s*(gain|loss|change)|gained|lost.*lb|lb.*gain/i.test(text)) a.coSymptoms.push("weight_change");
    if (/swell(ing|en)?|edema/i.test(text))                              a.coSymptoms.push("leg_swelling");
    if (/\bcough(ing)?\b/i.test(text))                                   a.coSymptoms.push("cough");
    if (/dizzy|lightheaded/i.test(text))                                  a.coSymptoms.push("lightheaded");

    // [1] worse than usual?
    if (/worse\s*(than\s*usual|than\s*normal)|worsening|getting\s*worse|more\s*tired|more\s*fatigue/i.test(text))
      a.isWorseUsual = "yes";
    else if (/same\s*as\s*usual|not\s*worse|no\s*change/i.test(text))
      a.isWorseUsual = "no";

    // [2] able to do usual activities?
    if (/can'?t\s*(do|perform|manage)|unable\s*to\s*do|limiting|stopped|can\s*no\s*longer|not\s*able/i.test(text))
      a.canDoActivities = "no";
    else if (/still\s*(able|can|doing|managing)|no\s*problem\s*doing|able\s*to\s*do/i.test(text))
      a.canDoActivities = "yes";

    return a;
  }

  _extractWeightAnswers(text) {
    // answer_list: [changeType, tookDiuretic, coSymptoms_str]
    // NOTE: Only weight GAIN is one of the 7 triage symptoms. Weight loss alone does not trigger triage.
    const a = { changeType: null, tookDiuretic: null, coSymptoms: [] };

    // [0] changeType — weight GAIN patterns only
    const isGain = /weight\s*(gain|up|increase|going\s*up|went\s*up|is\s*up)|gained|put\s*on\s*(weight|pound)|heavier|scale\s*(is|went|shows?)\s*(up|higher|more)|increas\w*\s+[\d.]+\s*(?:lb|pound|kg)|went\s*up\s+[\d.]+|up\s+[\d.]+\s*(?:lb|pound|kg)/i.test(text);

    if (isGain) {
      const isWeek = /week|7\s*day/i.test(text);
      const isDay  = /\bevery\s*day\b|today|one\s*day|24\s*hour|overnight|yesterday|since\s*yesterday|per\s*day|each\s*day/i.test(text);
      const lbMatch = text.match(/(\d+(?:\.\d+)?)\s*(?:pound|lb)/i);
      const kgMatch = text.match(/(\d+(?:\.\d+)?)\s*kg/i);
      const amount  = lbMatch ? parseFloat(lbMatch[1]) : (kgMatch ? parseFloat(kgMatch[1]) * 2.2 : null);

      if (isDay  && amount !== null && amount >= 2) a.changeType = "day_gain";
      else if (isWeek && amount !== null && amount >= 5) a.changeType = "week_gain";
      else if (isDay)  a.changeType = "day_gain";   // timeframe known, amount not clear
      else if (isWeek) a.changeType = "week_gain";  // timeframe known, amount not clear
      else             a.changeType = "other";       // vague gain
    }

    // [1] tookDiuretic — "Has the patient been taking their water pills (diuretic pills)?"
    // Check "not taking" BEFORE "taking" to avoid false positives
    if (/not\s*tak\w*.*(?:water\s*pill|diuretic|furosemide|lasix|torsemide|bumex)|didn'?t\s*take.*(?:water\s*pill|diuretic)|haven'?t\s*taken|stopped.*(?:water\s*pill|diuretic)/i.test(text))
      a.tookDiuretic = "no";
    else if (/tak\w*.*(?:water\s*pill|diuretic|furosemide|lasix|torsemide|bumex)|on\s*(?:a\s*)?(?:diuretic|water\s*pill)|my\s*(?:water\s*pill|diuretic)/i.test(text))
      a.tookDiuretic = "yes";

    // [2] co-symptoms for weight change: SOB, cough, leg swelling, nausea, bowel changes
    if (/short(ness)?\s*of\s*breath|breathless/i.test(text))       a.coSymptoms.push("sob");
    if (/\bcough(ing)?\b/i.test(text))                              a.coSymptoms.push("cough");
    if (/swell(ing|en)?|swollen|edema/i.test(text))                  a.coSymptoms.push("leg_swelling");
    if (/\b(nausea|nauseat|vomit|sick\s*to\s*my\s*stomach)\b/i.test(text)) a.coSymptoms.push("nausea");
    if (/bowel|diarrhea|constipat|stool|loose.*stool|changes.*bowel/i.test(text)) a.coSymptoms.push("bowel_changes");

    return a;
  }

  _extractConfusionAnswers(text) {
    // answer_list: [isNew, "", coSymptoms_str]  (index 1 unused in reference logic)
    const a = { isNew: null, coSymptoms: [] };

    // [0] new?
    if (/\b(new\b|first\s*time|never.*before|just\s*started|suddenly|sudden)\b/i.test(text))
      a.isNew = "yes";
    else if (/\b(not\s*new|existing|always|chronic|usual|same\s*as\s*before)\b/i.test(text))
      a.isNew = "no";

    // [2] co-symptoms
    if (/unconscious|passed?\s*out|blackout|faint/i.test(text))              a.coSymptoms.push("unconscious");
    if (/slurred?\s*speech|trouble\s*speak|speech\s*(difficult|problem)/i.test(text)) a.coSymptoms.push("slurred_speech");
    if (/face.*droop|drooping.*face|facial\s*(droop|asymm)/i.test(text))    a.coSymptoms.push("face_asym");
    if (/\b(weakness|arm.*weak|leg.*weak|one\s*side.*weak)\b/i.test(text))  a.coSymptoms.push("weakness");
    if (/\bfever\b/i.test(text))                                             a.coSymptoms.push("fever");
    if (/urinat|urinary|bathroom\s*frequent/i.test(text))                   a.coSymptoms.push("urination");
    if (/dizzy|lightheaded/i.test(text))                                     a.coSymptoms.push("lightheaded");

    return a;
  }

  _extractLegSwellingAnswers(text) {
    // answer_list: [isNew, isWorse, takingWaterPills, legs, dietChange]  (the tool's q0–q4)
    const a = { isNew: null, isWorse: null, takingWaterPills: null, legs: null, dietChange: null };

    // [0] new symptom?
    if (/\bnew\b|first\s*time|never.*before|just\s*started|recently\s*started/i.test(text))
      a.isNew = "yes";
    else if (/\b(not\s*new|had.*before|usual|chronic|always\s*have|for\s*(weeks|months|years))\b/i.test(text))
      a.isNew = "no";

    // [1] worse than previous days?
    // Note: "swollen" and "swell" are different strings — both must be covered.
    if (/same\s*as\s*usual|not\s*(gotten\s*|getting\s*)?(any\s*)?worse|no\s*change|hasn'?t\s*changed|has\s*not\s*changed|not\s*changed|unchanged/i.test(text))
      a.isWorse = "no";
    else if (/worse|worsening|more\s*(swollen|swell\w*)|goes?\s*up\s*higher|higher\s*(up|now)/i.test(text))
      a.isWorse = "yes";

    // [2] taking their water pills?  (tool: "Has the patient been taking their water pills (i.e. lasix)?")
    if (/(not|haven'?t|didn'?t|stopped|missed|skipp?ed|ran\s*out).{0,25}(water\s*pills?|diuretics?|lasix|furosemide|torsemide|bumex)/i.test(text))
      a.takingWaterPills = "no";
    else if (/(taking|take|took).{0,25}(water\s*pills?|diuretics?|lasix|furosemide|torsemide|bumex)/i.test(text))
      a.takingWaterPills = "yes";

    // [3] one or both?
    if (/both\s*(leg|ankle|feet|foot)|both\s*side|bilateral/i.test(text))
      a.legs = "both";
    else if (/one\s*(leg|ankle|foot)|right\s*(leg|ankle|foot)|left\s*(leg|ankle|foot)|single/i.test(text))
      a.legs = "one";

    // [4] recent diet change (e.g. more salt)?
    if (/more\s*salt|salty|ate\s*out|fast\s*food|diet\s*(has\s*)?chang|changed\s*my\s*diet/i.test(text))
      a.dietChange = "yes";

    return a;
  }

  _extractLightheadedAnswers(text) {
    // answer_list: [isNew, fellToday, bloodPressure]
    const a = { isNew: null, fellToday: null, bloodPressure: null };

    // [0] new symptom?
    if (/\b(new\b|first\s*time|never.*before|just\s*started)\b/i.test(text))
      a.isNew = "yes";
    else if (/\b(not\s*new|existing|always|chronic|usual|same\s*as\s*before)\b/i.test(text))
      a.isNew = "no";

    // [1] fell today?
    if (/\b(fell|fall(?:en|ing)?|tripp?ed|stumbled|lost.*balance|dropped.*floor)\b/i.test(text))
      a.fellToday = "yes";
    else if (/no.*fall|didn.?t\s*fall|haven.?t\s*fall|did\s*not\s*fall/i.test(text))
      a.fellToday = "no";

    // [2] high or low blood pressure?
    if (/\b(high|low|elevated|abnormal)\s*blood\s*pressure\b|\bhypertension\b|\bhypotension\b|bp\s*(is|was|running)?\s*(high|low)/i.test(text))
      a.bloodPressure = "yes";
    else if (/normal.*blood.*pressure|bp.*normal|blood.*pressure.*normal|no.*blood.*pressure.*problem/i.test(text))
      a.bloodPressure = "no";

    return a;
  }

  // ─── Follow-up Question Generation ───────────────────────────────────────

  /**
   * Determine which follow-up questions are still needed before triage can run.
   * Skips questions already answered in the combined patient text.
   * @param {string} text - all patient text so far
   * @param {string[]} detectedSymptoms - category keys to check
   * @returns {Array<{category, key, question}>}
   */
  /**
   * @param {string} text - patient text (for regex fallback extraction)
   * @param {string[]} detectedSymptoms
   * @param {Object} preExtractedAnswers - answers already extracted by LLM (preferred); {} if not available
   */
  getNeededFollowUps(text, detectedSymptoms, preExtractedAnswers = {}) {
    const questions = [];
    for (const category of detectedSymptoms) {
      // Use LLM-extracted answers when available; fall back to regex
      const answers = preExtractedAnswers[category] || this.extractAnswers(text, category);
      const missing = this._getMissingQuestionsForCategory(category, answers);
      missing.forEach(q => questions.push({ ...q, category }));
    }
    return questions;
  }

  _getMissingQuestionsForCategory(category, answers) {
    const q = [];
    switch (category) {

      case "sob":
        if (answers.isNew === null)
          q.push({ key: "sob_new", question: "Is your shortness of breath a **new** symptom for you, or something you have had before?" });
        if (answers.coSymptoms.length === 0)
          q.push({ key: "sob_co", question: "Are any of the following happening alongside your shortness of breath? (say all that apply)\n• Chest pain or pressure (and if so — is it severe?)\n• Fever\n• Cough\n• Leg or ankle swelling\n• Mucus or phlegm\n• Wheezing\n• None of these" });
        // Tool q2 (required): "Is shortness of breath worse at rest (e.g. while the patient is just sitting)?"
        // Tool q3 (worse with activity) is optional and only sets the review flag, so it is not asked.
        if (answers.worseAtRest === null || answers.worseAtRest === undefined)
          q.push({ key: "sob_rest", question: "Is your shortness of breath **worse at rest** (for example, while you are just sitting)?" });
        break;

      case "chestDiscomfort":
        if (answers.isNew === null)
          q.push({ key: "chest_new", question: "Is this chest discomfort **new** for you, or something you have felt before?" });
        if (answers.coSymptoms.length === 0)
          q.push({ key: "chest_co", question: "Are any of the following happening alongside your chest discomfort? (say all that apply)\n• Shooting pain down your arm, or pain on the left side of your chest\n• Shortness of breath\n• Heart racing or palpitations\n• Sweating\n• Nausea or vomiting\n• Regurgitation or heartburn\n• None of these" });
        if (answers.isAtRest === null)
          q.push({ key: "chest_rest", question: "Is the chest discomfort happening **at rest** (sitting, lying down), or only during physical activity (walking, exercising)?" });
        break;

      case "fatigue":
        if (answers.coSymptoms.length === 0)
          q.push({ key: "fatigue_co", question: "Are any of the following happening alongside your fatigue? (say all that apply)\n• Fever\n• Shortness of breath\n• Weight gain or loss\n• Leg or ankle swelling\n• Cough\n• Dizziness or lightheadedness\n• None of these" });
        if (answers.isWorseUsual === null)
          q.push({ key: "fatigue_worse", question: "Is your fatigue **worse than your usual level**?" });
        if (answers.canDoActivities === null)
          q.push({ key: "fatigue_activities", question: "Are you **still able to do your usual daily activities** (walking around the house, self-care), or is the fatigue preventing you?" });
        break;

      case "weightChange":
        if (answers.changeType === null)
          q.push({ key: "weight_type", question: "How much weight have you gained?\n• **2 or more pounds since yesterday** (gained in one day)\n• **More than 5 pounds in the past week**\n• Other amount or unsure" });
        if (answers.tookDiuretic === null)
          q.push({ key: "weight_diuretic", question: "Have you been taking your **water pills (diuretic pills)** as prescribed?\n• Yes\n• No\n• Unsure" });
        if (answers.coSymptoms.length === 0)
          q.push({ key: "weight_co", question: "Do you have any of these additional symptoms? (say all that apply)\n• Shortness of breath\n• Cough\n• Leg or ankle swelling\n• Nausea or vomiting\n• Changes in bowel movements\n• None of these" });
        break;

      case "confusion":
        if (answers.isNew === null)
          q.push({ key: "confusion_new", question: "Is this confusion or mental fogginess **new** for you, or have you experienced it before?" });
        if (answers.coSymptoms.length === 0)
          q.push({ key: "confusion_co", question: "Are any of the following also happening? (say all that apply)\n• Loss of consciousness or passing out\n• Slurred speech or trouble speaking\n• Facial drooping on one side\n• Arm or leg weakness\n• Fever\n• Changes in urination\n• Dizziness or lightheadedness\n• None of these" });
        break;

      case "legSwelling":
        // The tool's leg-swelling questions q0–q3, in the tool's wording (q4, diet change, only sets the review flag).
        if (answers.isNew === null || answers.isNew === undefined)
          q.push({ key: "swell_new", question: "Is the leg swelling a **new** symptom for you?" });
        if (answers.isWorse === null || answers.isWorse === undefined)
          q.push({ key: "swell_worse", question: "Is the leg swelling **worse than previous days** (for example, it goes up higher now)?" });
        if (answers.takingWaterPills === null || answers.takingWaterPills === undefined)
          q.push({ key: "swell_pills", question: "Have you been taking your **water pills** (for example, Lasix)?" });
        if (answers.legs === null || answers.legs === undefined)
          q.push({ key: "swell_legs", question: "Is the swelling in **one leg** or **both legs**?" });
        break;

      case "lightheaded":
        if (answers.isNew === null)
          q.push({ key: "lh_new", question: "Is this lightheadedness (or feeling faint) a **new symptom** for you?\n• Yes\n• No\n• Unsure" });
        if (answers.fellToday === null)
          q.push({ key: "lh_fell", question: "Did you **fall today**?\n• Yes\n• No\n• Unsure" });
        if (answers.bloodPressure === null)
          q.push({ key: "lh_bp", question: "Do you have **high or low blood pressure**?\n• Yes\n• No\n• Unsure" });
        break;
    }
    return q;
  }

  // ─── Traffic Light Logic (exact reference implementation) ─────────────────

  /**
   * Helper: count how many items in validList appear in string_to_search.
   * Returns true when count >= threshold.
   */
  _checkStringForMultiple(threshold, stringToSearch, validList) {
    let count = 0;
    for (const item of validList) {
      if (stringToSearch.includes(item)) {
        count++;
        if (count >= threshold) return true;
      }
    }
    return false;
  }

  // NOTE: The reference implementation has a JS parsing quirk in getResultChestDiscomfort
  // where `["sob","heart_racing"] || checkStringForMultiple(2,...)` evaluates to
  // `["sob","heart_racing"]` (truthy short-circuit), making the 5-item check dead code.
  // We replicate the *effective* logic exactly.

  /** Shortness of breath — traffic light logic */
  getResultSob(answerList) {
    let zone = this.ZONE_LIST[0];
    let reviewFlag = false;

    // RED — exit early
    if (answerList[1].includes("chest_pain:severe"))
      return { zone: this.ZONE_LIST[3], reviewFlag: true };

    // yellow_consider (no hf review)
    if (answerList[1].includes("fever") ||
        this._checkStringForMultiple(2, answerList[1], ["cough", "leg_swelling", "mucus", "wheeze"]))
      zone = this.ZONE_LIST[1];

    // yellow_consider + hf review
    if (answerList[0] === "yes" ||
        answerList[1].includes("chest_pain:not_severe") ||
        answerList[2] === "yes") {
      zone = this.ZONE_LIST[1];
      reviewFlag = true;
    }

    // hf review only
    if (answerList[3] === "yes" ||
        this._checkStringForMultiple(1, answerList[1], ["cough", "leg_swelling"]))
      reviewFlag = true;

    return { zone, reviewFlag };
  }

  /** Chest discomfort — traffic light logic (replicates reference exactly) */
  getResultChestDiscomfort(answerList) {
    let zone = this.ZONE_LIST[0];
    const reviewFlag = false;

    // RED — exit early
    if (answerList[1].includes("shooting_pain"))
      return { zone: this.ZONE_LIST[3], reviewFlag: true };

    // yellow_consider
    // Note: third condition checks for sob OR heart_racing
    // (reference code's array-literal || quirk makes only these two active)
    if (answerList[0] === "yes" ||
        answerList[2] === "rest" ||
        this._checkStringForMultiple(1, answerList[1], ["sob", "heart_racing"]))
      zone = this.ZONE_LIST[1];

    return { zone, reviewFlag };
  }

  /** Fatigue — traffic light logic */
  getResultFatigue(answerList) {
    let zone = this.ZONE_LIST[0];
    let reviewFlag = false;

    // yellow_consider
    if (answerList[0].includes("fever") ||
        this._checkStringForMultiple(2, answerList[0], ["sob", "weight_change", "leg_swelling", "cough", "lightheaded"]) ||
        answerList[1] === "yes")
      zone = this.ZONE_LIST[1];

    // hf review
    if (this._checkStringForMultiple(1, answerList[0], ["weight_change", "leg_swelling", "cough", "lightheaded"]) ||
        answerList[2] === "no")
      reviewFlag = true;

    return { zone, reviewFlag };
  }

  /** Weight gain — traffic light logic
   *  answerList[0] = changeType: "day_gain" | "week_gain" | "other" | null
   *  answerList[1] = tookDiuretic: "yes" | "no" | "unsure" | null
   *  answerList[2] = coSymptoms comma-joined: may include sob, cough, leg_swelling, nausea, bowel_changes
   */
  getResultWeightChange(answerList) {
    let zone = this.ZONE_LIST[0];
    let reviewFlag = false;

    // The tool lists nausea and bowel changes as options, but its rule only counts these three.
    const CO_SYMS = ["sob", "cough", "leg_swelling"];

    // yellow_consider: patient NOT taking diuretics (medication non-compliance)
    if (answerList[1] === "no")
      zone = this.ZONE_LIST[1];

    // yellow_consider + hf review: ≥1 co-symptom AND (not on diuretics OR vague gain amount)
    if (this._checkStringForMultiple(1, answerList[2], CO_SYMS) &&
        (answerList[1] === "no" || answerList[0] === "other")) {
      zone = this.ZONE_LIST[1];
      reviewFlag = true;
    }

    // yellow_call: measured day or week gain — regardless of diuretic status
    if (answerList[0] === "day_gain" || answerList[0] === "week_gain")
      zone = this.ZONE_LIST[2];

    // yellow_call + hf review: ≥2 co-symptoms AND (not on diuretics OR vague gain amount)
    if (this._checkStringForMultiple(2, answerList[2], CO_SYMS) &&
        (answerList[1] === "no" || answerList[0] === "other")) {
      zone = this.ZONE_LIST[2];
      reviewFlag = true;
    }

    return { zone, reviewFlag };
  }

  /** Confusion — traffic light logic */
  getResultConfusion(answerList) {
    let zone = this.ZONE_LIST[0];
    let reviewFlag = false;

    // RED — exit early
    if (answerList[2].includes("unconscious"))
      return { zone: this.ZONE_LIST[3], reviewFlag: true };

    // yellow_consider
    if (answerList[0] === "yes" ||
        this._checkStringForMultiple(2, answerList[2], ["slurred_speech", "face_asym", "weakness", "fever", "urination", "lightheaded"]))
      zone = this.ZONE_LIST[1];

    // hf review
    if (answerList[2].includes("lightheaded"))
      reviewFlag = true;

    return { zone, reviewFlag };
  }

  /** Leg swelling — traffic light logic */
  getResultLegSwelling(answerList) {
    let zone = this.ZONE_LIST[0];
    let reviewFlag = false;

    // yellow_consider (no hf review)
    if (answerList[2] === "yes" || answerList[3] === "one")
      zone = this.ZONE_LIST[1];

    // hf review
    if (answerList[3] === "both" || answerList[4] === "yes")
      reviewFlag = true;

    return { zone, reviewFlag };
  }

  /** Lightheadedness — traffic light logic */
  getResultLightheaded(answerList) {
    let zone = this.ZONE_LIST[0];
    const reviewFlag = false;

    // yellow_consider
    if (answerList[0] === "yes" || answerList[1] === "yes" || answerList[2] === "yes")
      zone = this.ZONE_LIST[1];

    return { zone, reviewFlag };
  }

  // ─── Rule-Based Triage (main entry point) ────────────────────────────────

  /**
   * Run the traffic light tool rule-based triage for all detected categories.
   * Takes the most severe zone across all categories.
   * @param {Object} answersByCategory - { sob: {isNew, coSymptoms, ...}, ... }
   * @param {string[]} detectedSymptoms - which categories to evaluate
   * @returns {Object} triage result
   */
  ruleBasedTriage(answersByCategory, detectedSymptoms) {
    const zoneOrder = { green: 0, yellow_consider: 1, yellow_call: 2, red: 3 };
    let highestZone = "green";
    let combinedReviewFlag = false;
    const flags = [];
    const categoryResults = {};

    for (const cat of detectedSymptoms) {
      const ans = answersByCategory[cat] || {};
      let result = null;
      let answerList;

      switch (cat) {
        case "sob":
          answerList = [
            ans.isNew || "no",
            (ans.coSymptoms || []).join(","),
            ans.worseAtRest || "no",        // tool q2: worse at rest?
            ans.worseWithActivity || "no"   // tool q3: worse with activity? (review flag only)
          ];
          result = this.getResultSob(answerList);
          break;

        case "chestDiscomfort":
          answerList = [
            ans.isNew || "no",
            (ans.coSymptoms || []).join(","),
            ans.isAtRest || "activity"
          ];
          result = this.getResultChestDiscomfort(answerList);
          break;

        case "fatigue":
          answerList = [
            (ans.coSymptoms || []).join(","),
            ans.isWorseUsual || "no",
            ans.canDoActivities !== null ? ans.canDoActivities : "yes"
          ];
          result = this.getResultFatigue(answerList);
          break;

        case "weightChange":
          answerList = [
            ans.changeType || "other",
            // tookDiuretic: "yes"/"no"/"unsure"/null — only "no" triggers yellow_consider
            ans.tookDiuretic !== null ? ans.tookDiuretic : "unsure",
            (ans.coSymptoms || []).join(",")
          ];
          result = this.getResultWeightChange(answerList);
          break;

        case "confusion":
          answerList = [
            ans.isNew || "no",
            "",  // index 1 unused in reference logic
            (ans.coSymptoms || []).join(",")
          ];
          result = this.getResultConfusion(answerList);
          break;

        case "legSwelling":
          // Same order as the tool's five questions. An unanswered question stays blank, exactly as
          // in the tool, so it cannot trigger a zone. The tool's rule reads only q2–q4.
          answerList = [
            ans.isNew || "",             // q0 new symptom? (not used by the tool's rule)
            ans.isWorse || "",           // q1 worse than previous days? (not used by the tool's rule)
            ans.takingWaterPills || "",  // q2 taking water pills?
            ans.legs || "",              // q3 one leg or both?
            ans.dietChange || ""         // q4 recent diet change?
          ];
          result = this.getResultLegSwelling(answerList);
          break;

        case "lightheaded":
          answerList = [
            ans.isNew     !== null ? ans.isNew     : "no",
            ans.fellToday !== null ? ans.fellToday : "no",
            ans.bloodPressure !== null ? ans.bloodPressure : "no"
          ];
          result = this.getResultLightheaded(answerList);
          break;
      }

      if (result) {
        categoryResults[cat] = result;
        if (result.reviewFlag) combinedReviewFlag = true;
        if (zoneOrder[result.zone] > zoneOrder[highestZone])
          highestZone = result.zone;

        const zoneLabel = result.zone.replace("_", " ").toUpperCase();
        flags.push(`${this.CATEGORY_LABELS[cat]} → ${zoneLabel}`);
      }
    }

    // ── Map raw zone to display values ──────────────────────────────────────
    const zoneDisplay = {
      green:          { zone: "GREEN",  color: "#38a169", urgency: "STABLE",
        action: "Continue your usual medication, diet, fluid restrictions, and daily monitoring. Keep your next scheduled appointment." },
      yellow_consider:{ zone: "YELLOW", color: "#d69e2e", urgency: "CONSIDER CALLING",
        action: "Consider contacting your care team today to discuss your symptoms. If you cannot reach them and symptoms worsen, go to the emergency room." },
      yellow_call:    { zone: "YELLOW", color: "#d69e2e", urgency: "CALL TODAY",
        action: "Call your care team (nurse or doctor) today — do not wait until your next appointment. If you cannot reach them and symptoms worsen, go to the emergency room." },
      red:            { zone: "RED",    color: "#e53e3e", urgency: "EMERGENCY",
        action: "Call 911 immediately or have someone drive you to the emergency room. Do NOT drive yourself. This is a medical emergency." }
    };

    const display = zoneDisplay[highestZone];
    const reviewNote = combinedReviewFlag
      ? "📚 Your care team may want to review your heart failure self-management plan with you."
      : null;

    const triageResult = {
      method:       "rule_based",
      zone:         display.zone,
      zoneRaw:      highestZone,
      color:        display.color,
      urgency:      display.urgency,
      action:       display.action,
      flags,
      reviewFlag:   combinedReviewFlag,
      reviewNote,
      categoryResults,
      timestamp:    new Date().toISOString(),
      sources: [
        { name: "Cornell Homecare Traffic Light Tool", url: "http://cornellhomecare.com/traffictool" },
        { name: "Traffic Light Self-Monitoring RCT 2024", url: "https://pubmed.ncbi.nlm.nih.gov/38368847/" }
      ]
    };

    this.triageHistory.push(triageResult);
    return triageResult;
  }

  // ─── AI Triage Prompt ─────────────────────────────────────────────────────

  /**
   * Build a system prompt for the AI triage.
   * The AI performs its own independent clinical reasoning — it does NOT receive
   * the rule-based result. The AI uses the traffic light categories as a framework
   * for follow-up questions but makes its own zone decision.
   * @param {string} freeTextSymptoms
   * @param {string[]} detectedCategories - hints for the AI
   * @returns {string}
   */
  buildAITriagePrompt(freeTextSymptoms, detectedCategories = []) {
    const catHints = detectedCategories.length > 0
      ? `\nRelevant symptom categories detected: ${detectedCategories.map(c => this.CATEGORY_LABELS[c] || c).join(", ")}`
      : "";

    return `You are a clinical decision support assistant for heart failure patient triage.
Your task is to assess the patient's symptoms and provide a triage zone recommendation.

CRITICAL RULES:
1. You MUST always assign a zone — GREEN, YELLOW_CONSIDER, YELLOW_CALL, or RED. Never return "UNKNOWN" or refuse to classify.
2. You are a support tool, not a replacement for medical advice.

## Patient-Reported Symptoms${catHints}
${freeTextSymptoms}

## Zone Definitions
- 🟢 GREEN          : Symptoms are stable, consistent with patient's baseline. No acute concern. Continue usual care.
- 🟡 YELLOW_CONSIDER: Symptoms may indicate early or mild change. Patient should monitor closely and CONSIDER calling their care team if symptoms persist or worsen; a same-day call is not urgent.
- 🟡 YELLOW_CALL    : Symptoms indicate likely HF worsening. Patient MUST call their care team TODAY (not tomorrow, not if-it-gets-worse).
- 🔴 RED            : Acute decompensation or life-threatening emergency. Call 911 / go to ER immediately.

## Instructions
Perform an independent clinical triage using 2022 AHA/ACC/HFSA Heart Failure Guidelines:
1. Assign zone (GREEN/YELLOW_CONSIDER/YELLOW_CALL/RED) based on the symptom constellation
2. State your clinical reasoning — consider fluid overload, hemodynamic instability, ischemia, arrhythmia, and functional status
3. List the key symptoms driving your assessment
4. Give immediate action instructions

Format as JSON with fields: zone (GREEN/YELLOW_CONSIDER/YELLOW_CALL/RED), urgency, reasoning, keySymptoms (array of strings), immediateActions (array of strings), evidenceBasis, disclaimer`;
  }

  /**
   * Parse and normalize the AI triage JSON response.
   */
  parseAITriageResponse(aiResponse, originalSymptoms) {
    try {
      const jsonMatch = aiResponse.match(/\{[\s\S]*\}/);
      if (jsonMatch) {
        const parsed = JSON.parse(jsonMatch[0]);
        const zoneColors = { GREEN: "#38a169", YELLOW: "#d69e2e", RED: "#e53e3e" };
        const rawZone = (parsed.zone || "").toUpperCase().replace(/-/g,"_");
        // Normalise 4-level → 3-level display zone; preserve urgency sub-label
        const displayMap = {
          GREEN:          { zone: "GREEN",  urgency: "STABLE" },
          YELLOW_CONSIDER:{ zone: "YELLOW", urgency: "CONSIDER CALLING" },
          YELLOW_CALL:    { zone: "YELLOW", urgency: "CALL TODAY" },
          RED:            { zone: "RED",    urgency: "EMERGENCY" }
        };
        const norm = displayMap[rawZone] || { zone: "YELLOW", urgency: parsed.urgency || "YELLOW" };
        return {
          method:       "ai_reasoning",
          zone:         norm.zone,
          color:        zoneColors[norm.zone] || "#718096",
          urgency:      parsed.urgency || norm.urgency,
          reasoning:    parsed.reasoning || "",
          keySymptoms:  parsed.keySymptoms || [],
          action: Array.isArray(parsed.immediateActions)
            ? parsed.immediateActions.join("; ")
            : (parsed.immediateActions || ""),
          evidenceBasis: parsed.evidenceBasis || "",
          disclaimer:   parsed.disclaimer || "This AI assessment is for educational purposes only and does not replace medical advice.",
          rawResponse:  aiResponse,
          timestamp:    new Date().toISOString(),
          sources: [{ name: "2022 AHA/ACC/HFSA Heart Failure Guidelines", url: "https://www.ahajournals.org/doi/10.1161/CIR.0000000000001063" }]
        };
      }
    } catch (e) { /* fall through to text parsing */ }

    // Fallback: extract zone from text
    const u = aiResponse.toUpperCase();
    let zone = "UNKNOWN";
    if (u.includes("RED ZONE") || u.includes("🔴") || u.includes("EMERGENCY")) zone = "RED";
    else if (u.includes("YELLOW ZONE") || u.includes("🟡") || u.includes("URGENT"))    zone = "YELLOW";
    else if (u.includes("GREEN ZONE") || u.includes("🟢") || u.includes("STABLE"))     zone = "GREEN";

    const zoneColors = { GREEN: "#38a169", YELLOW: "#d69e2e", RED: "#e53e3e", UNKNOWN: "#718096" };
    return {
      method: "ai_reasoning", zone, color: zoneColors[zone], urgency: zone,
      reasoning: aiResponse, keySymptoms: [], action: "",
      disclaimer: "This AI assessment is for educational purposes only and does not replace medical advice.",
      rawResponse: aiResponse, timestamp: new Date().toISOString()
    };
  }

  // ─── AI-Independent Triage ────────────────────────────────────────────────

  /**
   * Build a prompt for the AI-independent triage mode.
   * The AI uses its own clinical framework — NOT the traffic light categories —
   * to decide whether follow-up questions are needed, and to assign a zone.
   * @param {string} symptomText - patient's full symptom description (may include follow-up answers)
   * @returns {string}
   */
  /**
   * @param {string} symptomText
   * @param {number} roundNumber  1 = first call (may ask follow-ups); 2+ = must triage now
   */
  buildAIIndependentTriagePrompt(symptomText, roundNumber = 1) {
    const mustDecideNow = roundNumber >= 2;
    const forceNote = mustDecideNow
      ? `\n\n⚠️ OVERRIDE: You have already asked one round of clarifying questions. You MUST now provide a final triage assessment. Use action "triage". Do NOT use action "follow_up" again — make your best clinical judgment with the information available.`
      : "";

    const step1 = mustDecideNow
      ? `STEP 1 — You must triage now:
- You have already gathered additional information. Proceed directly to triage using action "triage".`
      : `STEP 1 — Decide if you need more information:
- If the description is too vague to assess severity confidently → ask 1–3 focused clinical follow-up questions
- If you have sufficient detail to assess severity → proceed directly to triage`;

    const followUpOption = mustDecideNow
      ? ""
      : `Option A — Need more information:
{
  "action": "follow_up",
  "acknowledgment": "Brief empathetic acknowledgment of what they shared (1-2 sentences)",
  "questions": ["Question 1?", "Question 2?", "Question 3?"]
}

`;

    const followUpRule = mustDecideNow
      ? `- You MUST use action "triage" — do NOT use "follow_up"`
      : `- Follow-up questions: ask at most 3, focused on what would change your zone assignment`;

    return `You are a clinical expert specializing in heart failure management. This chatbot is designed exclusively for patients with confirmed heart failure — do NOT ask about HF diagnosis, history, or whether the patient has heart failure. Assume they do.${forceNote}

Your task: provide a triage assessment based on the patient's description.

Patient's description:
"${symptomText}"

${step1}

STEP 2 — Respond ONLY with one of these two JSON formats:

${followUpOption}Option B — Ready to triage:
{
  "action": "triage",
  "zone": "GREEN" | "YELLOW_CONSIDER" | "YELLOW_CALL" | "RED",
  "urgency": "short urgency label (e.g. Stable, Consider calling, Call today, Emergency — call 911)",
  "reasoning": "2-4 sentence clinical reasoning based on heart failure pathophysiology",
  "keySymptoms": ["symptom 1", "symptom 2"],
  "immediateActions": ["Action 1", "Action 2"],
  "disclaimer": "This AI assessment is for educational purposes only and does not replace medical advice."
}

RULES:
- Output ONLY valid JSON — no text outside the JSON object
- Zone: GREEN = stable, YELLOW_CONSIDER = monitor and consider calling, YELLOW_CALL = must call today, RED = emergency (call 911 / go to ER now)
- Never use UNKNOWN or plain YELLOW — always commit to GREEN, YELLOW_CONSIDER, YELLOW_CALL, or RED
${followUpRule}
- Triage reasoning: use heart failure clinical knowledge (fluid overload, decompensation, hemodynamic instability, arrhythmia)`;
  }

  /**
   * Parse the AI-independent triage JSON response.
   * @param {string} aiResponse
   * @returns {Object} - either { isFollowUp: true, acknowledgment, questions }
   *                     or a standard triage result object
   */
  parseAIIndependentTriageResponse(aiResponse) {
    try {
      const jsonMatch = aiResponse.match(/\{[\s\S]*\}/);
      if (jsonMatch) {
        const parsed = JSON.parse(jsonMatch[0]);

        if (parsed.action === "follow_up") {
          return {
            isFollowUp:      true,
            acknowledgment:  parsed.acknowledgment || "",
            questions:       Array.isArray(parsed.questions) ? parsed.questions : []
          };
        }

        if (parsed.action === "triage") {
          const zoneColors = { GREEN: "#38a169", YELLOW: "#d69e2e", RED: "#e53e3e" };
          const rawZone = (parsed.zone || "").toUpperCase().replace(/-/g,"_");
          const displayMap = {
            GREEN:          { zone: "GREEN",  urgency: "STABLE" },
            YELLOW_CONSIDER:{ zone: "YELLOW", urgency: "CONSIDER CALLING" },
            YELLOW_CALL:    { zone: "YELLOW", urgency: "CALL TODAY" },
            RED:            { zone: "RED",    urgency: "EMERGENCY" }
          };
          const norm = displayMap[rawZone] || { zone: "YELLOW", urgency: parsed.urgency || "YELLOW" };
          return {
            isFollowUp:   false,
            method:       "ai_independent",
            zone:         norm.zone,
            color:        zoneColors[norm.zone] || "#718096",
            urgency:      parsed.urgency || norm.urgency,
            reasoning:    parsed.reasoning || "",
            keySymptoms:  parsed.keySymptoms || [],
            action:       Array.isArray(parsed.immediateActions)
              ? parsed.immediateActions.join("; ")
              : (parsed.immediateActions || ""),
            disclaimer:   parsed.disclaimer || "This AI assessment is for educational purposes only and does not replace medical advice.",
            rawResponse:  aiResponse,
            timestamp:    new Date().toISOString(),
            sources: [{ name: "2022 AHA/ACC/HFSA Heart Failure Guidelines", url: "https://www.ahajournals.org/doi/10.1161/CIR.0000000000001063" }]
          };
        }
      }
    } catch (e) { /* fall through */ }

    // Fallback: extract zone from text
    const u = aiResponse.toUpperCase();
    let zone = "YELLOW"; // display zone (3-level)
    let urgency = "CONSIDER CALLING";
    if (u.includes("RED ZONE") || u.includes("🔴") || u.includes("EMERGENCY")) { zone = "RED"; urgency = "EMERGENCY"; }
    else if (u.includes("GREEN ZONE") || u.includes("🟢") || u.includes("STABLE")) { zone = "GREEN"; urgency = "STABLE"; }
    else if (u.includes("YELLOW_CALL") || u.includes("CALL TODAY")) { urgency = "CALL TODAY"; }

    const zoneColors = { GREEN: "#38a169", YELLOW: "#d69e2e", RED: "#e53e3e" };
    return {
      isFollowUp:  false,
      method:      "ai_independent",
      zone, color: zoneColors[zone], urgency,
      reasoning:   aiResponse, keySymptoms: [], action: "",
      disclaimer:  "This AI assessment is for educational purposes only and does not replace medical advice.",
      rawResponse: aiResponse, timestamp: new Date().toISOString()
    };
  }

  /**
   * Generate a comparison summary between rule-based and AI triage results.
   */
  compareTriageResults(ruleResult, aiResult) {
    const agree = ruleResult.zone === aiResult.zone;
    const zoneOrder = { GREEN: 0, YELLOW: 1, RED: 2, UNKNOWN: -1 };
    const ruleLevel = zoneOrder[ruleResult.zone] ?? -1;
    const aiLevel   = zoneOrder[aiResult.zone]  ?? -1;

    let discrepancyNote = "";
    if (!agree) {
      if (aiLevel > ruleLevel)
        discrepancyNote = "The AI assessment classified symptoms as more severe. Consider contacting your care team.";
      else if (aiLevel < ruleLevel)
        discrepancyNote = "The rule-based system classified symptoms as more severe. When in doubt, follow the more conservative recommendation.";
    }

    const recommended = (aiLevel >= ruleLevel) ? aiResult : ruleResult;
    return {
      agree, discrepancyNote,
      ruleBasedZone:     ruleResult.zone,
      aiZone:            aiResult.zone,
      recommendedZone:   recommended.zone,
      recommendedAction: recommended.action,
      conservativeNote: !agree ? "⚠️ The two systems disagreed. We recommend following the more urgent recommendation." : ""
    };
  }
}

window.TriageEngine = TriageEngine;
