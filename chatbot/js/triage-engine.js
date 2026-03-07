/**
 * Heart Failure Triage Engine
 * Implements two triage modes:
 * 1. Rule-based: Cornell Traffic Light Tool logic (deterministic)
 * 2. AI-assisted: LLM reasoning for comparative analysis
 */

class TriageEngine {
  constructor() {
    this.triageHistory = [];
  }

  // ─── Rule-Based Triage (Cornell Traffic Light System) ────────────────────

  /**
   * Perform rule-based triage based on structured symptom data.
   * Returns a triage result with zone, reasoning, and recommendations.
   * @param {Object} symptoms - Structured symptom inputs
   * @returns {Object} triageResult
   */
  ruleBasedTriage(symptoms) {
    const {
      weightGain1Day = 0,          // lbs gained in last 24 hours
      weightGain1Week = 0,         // lbs gained over last week
      sobScale = 0,                // shortness of breath 0-10
      chestPain = false,           // clear cardiac chest pain/pressure/tightness → RED
      faintingOrLoss = false,      // fainting / loss of consciousness
      pinkFoamyCough = false,      // coughing pink/foamy mucus
      severeConfusion = false,     // sudden confusion / change in mental status
      strokeSigns = false,         // facial droop, arm weakness, speech difficulty
      swellingNewWorse = false,    // new or worsening leg/ankle swelling
      ortho = false,               // waking at night breathless / needs extra pillows
      dizzinessStanding = false,   // dizziness when standing
      irregularHeartbeat = false,  // palpitations / irregular heartbeat
      rapidHeartRate = false,      // heart rate > 100 at rest
      unusualFatigue = false,      // fatigue limiting usual activities
      newCough = false,            // new dry hacking cough
      decreasedUrine = false,      // decreased urine output
      heartRateBpm = null,         // actual heart rate if known
      // ── Chest discomfort follow-up decision tree (traffic light tool) ───
      chestDiscomfortOnly = false,
      chestIsNew = null,
      chestCoArmPain = false, chestCoSOB = false, chestCoRacingHeart = false,
      chestCoSweating = false, chestCoNausea = false,
      chestCoRegurgitation = false, chestCoNone = false,
      chestAtRest = false, chestDuringActivity = false,
    } = symptoms;

    const redFlags = [];
    const yellowFlags = [];

    // ─── RED ZONE checks ────────────────────────────────────────────────────
    if (chestPain)
      redFlags.push("Chest pain or pressure — possible cardiac emergency");
    if (faintingOrLoss)
      redFlags.push("Fainting or loss of consciousness");
    if (pinkFoamyCough)
      redFlags.push("Coughing up pink or foamy mucus — sign of pulmonary edema");
    if (severeConfusion)
      redFlags.push("Sudden confusion or altered mental status");
    if (strokeSigns)
      redFlags.push("Signs of stroke (facial droop, arm weakness, speech difficulty)");
    if (sobScale >= 7)
      redFlags.push(`Severe shortness of breath at rest (SOB scale: ${sobScale}/10)`);
    if (weightGain1Day >= 4)
      redFlags.push(`Rapid weight gain: ${weightGain1Day} lbs in 24 hours (threshold: ≥4 lbs)`);
    if (weightGain1Week > 5 && sobScale >= 4)
      redFlags.push(`Weight gain ${weightGain1Week} lbs over 1 week with worsening symptoms`);
    if (heartRateBpm !== null && heartRateBpm > 120 && sobScale >= 4)
      redFlags.push(`Very high heart rate (${heartRateBpm} bpm) with breathing symptoms`);

    // ── Chest discomfort: cardiac co-symptoms present → RED ─────────────────
    // Traffic light tool: arm pain / SOB / racing heart / sweating / nausea
    // alongside chest discomfort = possible cardiac emergency regardless of new/existing.
    if (chestDiscomfortOnly) {
      const cardiacCoList = [
        chestCoArmPain        && "shooting arm pain",
        chestCoSOB            && "shortness of breath",
        chestCoRacingHeart    && "racing heart",
        chestCoSweating       && "sweating",
        chestCoNausea         && "nausea/vomiting",
      ].filter(Boolean);
      if (cardiacCoList.length > 0)
        redFlags.push(`Chest discomfort with cardiac co-symptoms: ${cardiacCoList.join(", ")}`);
    }

    // ─── YELLOW ZONE checks ─────────────────────────────────────────────────
    if (redFlags.length === 0) {
      if (weightGain1Day >= 2 && weightGain1Day < 4)
        yellowFlags.push(`Weight gain: ${weightGain1Day} lbs in 24 hours (threshold: 2–3 lbs)`);
      if (weightGain1Week >= 3 && weightGain1Week <= 5)
        yellowFlags.push(`Weight gain: ${weightGain1Week} lbs over 1 week`);
      if (sobScale >= 4 && sobScale < 7)
        yellowFlags.push(`Moderate shortness of breath with activity (SOB scale: ${sobScale}/10)`);
      if (swellingNewWorse)
        yellowFlags.push("New or worsening leg/ankle/foot swelling");
      if (ortho)
        yellowFlags.push("Waking at night breathless or needing extra pillows to sleep");
      if (dizzinessStanding)
        yellowFlags.push("Dizziness or lightheadedness when standing");
      if (irregularHeartbeat)
        yellowFlags.push("New irregular heartbeat or palpitations");
      if (rapidHeartRate)
        yellowFlags.push("Resting heart rate elevated (>100 bpm)");
      if (unusualFatigue)
        yellowFlags.push("Unusual fatigue limiting daily activities");
      if (newCough)
        yellowFlags.push("New dry or persistent cough");
      if (decreasedUrine)
        yellowFlags.push("Decreased urine output");

      // ── Chest discomfort: no cardiac co-symptoms → apply new/rest logic ──
      // Traffic light tool decision tree:
      //   Regurgitation ONLY (no cardiac co-symptoms) → GI cause → GREEN (no flag)
      //   New symptom, no clear cause           → YELLOW
      //   Existing + at rest                    → YELLOW
      //   Existing + only during activity       → GREEN (no flag)
      //   Unknown if new (unanswered)            → YELLOW (safety default)
      if (chestDiscomfortOnly) {
        const hasCardiacCo = chestCoArmPain || chestCoSOB || chestCoRacingHeart || chestCoSweating || chestCoNausea;
        const hasGIOnly = chestCoRegurgitation && !hasCardiacCo;
        if (!hasCardiacCo && !hasGIOnly) {
          if (chestIsNew === true)
            yellowFlags.push("New chest discomfort (no cardiac co-symptoms) — contact your care team");
          else if (chestAtRest)
            yellowFlags.push("Chest discomfort occurring at rest — contact your care team");
          else if (chestIsNew === null)
            // Unknown whether new: default to YELLOW for safety
            yellowFlags.push("Chest discomfort — unable to confirm if new; contact care team to be safe");
          // chestIsNew === false + activity-only + no cardiac co-symptoms → GREEN (stable)
        }
        // hasGIOnly → no flag → GREEN
      }
    }

    // ─── Determine Zone ─────────────────────────────────────────────────────
    let zone, color, action, urgency;
    if (redFlags.length > 0) {
      zone = "RED";
      color = "#e53e3e";
      urgency = "EMERGENCY";
      action = "Call 911 immediately or have someone drive you to the emergency room. Do NOT drive yourself. This is a medical emergency.";
    } else if (yellowFlags.length > 0) {
      zone = "YELLOW";
      color = "#d69e2e";
      urgency = "URGENT";
      action = "Contact your care team (nurse or doctor) within the next few hours — do not wait until your next scheduled appointment. If you cannot reach your care team and symptoms worsen, go to the emergency room.";
    } else {
      zone = "GREEN";
      color = "#38a169";
      urgency = "STABLE";
      action = "Continue your usual medication, diet, fluid restrictions, and daily monitoring. Keep your next scheduled follow-up appointment.";
    }

    const result = {
      method: "rule_based",
      zone,
      color,
      urgency,
      action,
      flags: redFlags.length > 0 ? redFlags : yellowFlags,
      inputSymptoms: symptoms,
      timestamp: new Date().toISOString(),
      sources: [
        {
          name: "Cornell Homecare Traffic Light Tool",
          url: "http://cornellhomecare.com/traffictool"
        },
        {
          name: "Traffic Light Self-Monitoring RCT 2024",
          url: "https://pubmed.ncbi.nlm.nih.gov/38368847/"
        }
      ]
    };

    this.triageHistory.push(result);
    return result;
  }

  /**
   * Build a structured triage prompt for the AI to assess.
   * The AI uses its OWN independent clinical reasoning — it does NOT see
   * the rule-based result so the two systems remain truly independent.
   * @param {string} freeTextSymptoms - User's description of symptoms
   * @returns {string} system prompt for AI triage
   */
  buildAITriagePrompt(freeTextSymptoms) {
    return `You are a clinical decision support assistant for heart failure patient triage.
Your task is to assess the patient's symptoms and provide a triage recommendation using your own clinical reasoning.

IMPORTANT: Always advise the patient to consult their healthcare team. You are a support tool, not a replacement for medical advice.

## Patient-Reported Symptoms
${freeTextSymptoms}

## Instructions
Perform an independent clinical triage assessment using evidence-based reasoning (2022 AHA/ACC/HFSA Heart Failure Guidelines):

1. **Zone Assignment**: Assign one of three zones:
   - 🟢 GREEN: Stable — continue usual care
   - 🟡 YELLOW: Concerning — contact care team within hours
   - 🔴 RED: Emergency — call 911 immediately

2. **Clinical Reasoning**: Explain WHY you assigned this zone. Consider:
   - Hemodynamic instability signs
   - Fluid overload indicators
   - Ischemic/arrhythmic features (distinguish cardiac chest pain from GI symptoms like regurgitation)
   - Functional status change
   - Risk stratification factors

3. **Key Symptoms of Concern**: List the specific symptoms driving your assessment

4. **Immediate Actions**: Give specific, actionable instructions

Format your response as JSON with fields: zone, urgency, reasoning, keySymptoms (array), immediateActions, evidenceBasis, disclaimer`;
  }

  /**
   * Parse and normalize AI triage response
   * @param {string} aiResponse - Raw LLM response
   * @param {string} originalSymptoms - Original symptom text
   * @returns {Object} Normalized triage result
   */
  parseAITriageResponse(aiResponse, originalSymptoms) {
    try {
      // Try to extract JSON from the response
      const jsonMatch = aiResponse.match(/\{[\s\S]*\}/);
      if (jsonMatch) {
        const parsed = JSON.parse(jsonMatch[0]);
        const zoneColors = { GREEN: "#38a169", YELLOW: "#d69e2e", RED: "#e53e3e" };
        return {
          method: "ai_reasoning",
          zone: parsed.zone || "UNKNOWN",
          color: zoneColors[parsed.zone] || "#718096",
          urgency: parsed.urgency || parsed.zone,
          reasoning: parsed.reasoning || "",
          keySymptoms: parsed.keySymptoms || [],
          action: Array.isArray(parsed.immediateActions)
            ? parsed.immediateActions.join("; ")
            : (parsed.immediateActions || ""),
          evidenceBasis: parsed.evidenceBasis || "",
          disclaimer: parsed.disclaimer || "This AI assessment is for educational purposes only and does not replace medical advice.",
          rawResponse: aiResponse,
          timestamp: new Date().toISOString(),
          sources: [
            {
              name: "2022 AHA/ACC/HFSA Heart Failure Guidelines",
              url: "https://www.ahajournals.org/doi/10.1161/CIR.0000000000001063"
            }
          ]
        };
      }
    } catch (e) {
      // Fall back to text parsing
    }

    // Fallback: extract zone from text
    const upperResponse = aiResponse.toUpperCase();
    let zone = "UNKNOWN";
    if (upperResponse.includes("RED ZONE") || upperResponse.includes("🔴") || upperResponse.includes("EMERGENCY"))
      zone = "RED";
    else if (upperResponse.includes("YELLOW ZONE") || upperResponse.includes("🟡") || upperResponse.includes("URGENT"))
      zone = "YELLOW";
    else if (upperResponse.includes("GREEN ZONE") || upperResponse.includes("🟢") || upperResponse.includes("STABLE"))
      zone = "GREEN";

    const zoneColors = { GREEN: "#38a169", YELLOW: "#d69e2e", RED: "#e53e3e", UNKNOWN: "#718096" };

    return {
      method: "ai_reasoning",
      zone,
      color: zoneColors[zone],
      urgency: zone,
      reasoning: aiResponse,
      keySymptoms: [],
      action: "",
      disclaimer: "This AI assessment is for educational purposes only and does not replace medical advice.",
      rawResponse: aiResponse,
      timestamp: new Date().toISOString()
    };
  }

  /**
   * Generate a side-by-side comparison between rule-based and AI triage
   * @param {Object} ruleResult - Rule-based triage result
   * @param {Object} aiResult - AI triage result
   * @returns {Object} comparison object
   */
  compareTriageResults(ruleResult, aiResult) {
    const agree = ruleResult.zone === aiResult.zone;
    const ruleZoneOrder = { GREEN: 0, YELLOW: 1, RED: 2, UNKNOWN: -1 };
    const ruleLevel = ruleZoneOrder[ruleResult.zone] ?? -1;
    const aiLevel = ruleZoneOrder[aiResult.zone] ?? -1;

    let discrepancyNote = "";
    if (!agree) {
      if (aiLevel > ruleLevel) {
        discrepancyNote = "The AI assessment classified symptoms as more severe than the rule-based system. Consider contacting your care team.";
      } else if (aiLevel < ruleLevel) {
        discrepancyNote = "The rule-based system classified symptoms as more severe. When in doubt, always follow the more conservative (higher urgency) recommendation.";
      }
    }

    // Conservatively recommend the higher urgency result
    const recommended = (aiLevel >= ruleLevel) ? aiResult : ruleResult;

    return {
      agree,
      discrepancyNote,
      ruleBasedZone: ruleResult.zone,
      aiZone: aiResult.zone,
      recommendedZone: recommended.zone,
      recommendedAction: recommended.action,
      conservativeNote: !agree ? "⚠️ The two systems disagreed. We recommend following the more urgent recommendation." : ""
    };
  }

  /**
   * Extract structured symptom flags from free-text description.
   * Mirrors the 14 fields used by ruleBasedTriage so text in chat
   * can drive the rule-based engine even when the form is empty.
   * @param {string} text
   * @returns {Object} partial symptoms object
   */
  extractSymptomsFromText(text) {
    const symptoms = {
      weightGain1Day: 0, weightGain1Week: 0, sobScale: 0,
      chestPain: false, faintingOrLoss: false, pinkFoamyCough: false,
      severeConfusion: false, strokeSigns: false, swellingNewWorse: false,
      ortho: false, dizzinessStanding: false, irregularHeartbeat: false,
      rapidHeartRate: false, unusualFatigue: false, newCough: false,
      decreasedUrine: false, heartRateBpm: null,
      // ── Chest discomfort decision tree fields (traffic light tool) ────────
      chestDiscomfortOnly: false, // "discomfort/ache" without clear cardiac language
      chestIsNew: null,           // null=unknown, true=new, false=existing
      chestCoArmPain: false,      // concurrent shooting arm/left-side pain
      chestCoSOB: false,          // concurrent shortness of breath
      chestCoRacingHeart: false,  // concurrent racing heart / palpitations
      chestCoSweating: false,     // concurrent sweating
      chestCoNausea: false,       // concurrent nausea or vomiting
      chestCoRegurgitation: false,// concurrent regurgitation / heartburn (GI indicator)
      chestCoNone: false,         // patient explicitly said "none of these"
      chestAtRest: false,         // occurring at rest / while sitting
      chestDuringActivity: false, // occurring during activity / walking
    };

    // ── Chest pain classification ─────────────────────────────────────────
    // Clear cardiac language (pain/pressure/tightness/heaviness/hurt) → RED directly.
    // "Chest discomfort/ache" without GI context → needs decision tree (Fix #2).
    const hasCardiacChest = /chest\s*(pain|pressure|tightness|heaviness|hurt)/i.test(text);
    const hasChestDiscomfort = /chest\s*(discomfort|ache)/i.test(text);
    const hasGIContext = /regurgitat|heartburn|indigestion|acid\s*reflux|gerd|stomach\s*acid|sour\s*taste|burp/i.test(text);

    if (hasCardiacChest) {
      symptoms.chestPain = true;           // → RED, no further questions needed
    } else if (hasChestDiscomfort && !hasGIContext) {
      symptoms.chestDiscomfortOnly = true; // → follow-up decision tree
    }
    // If hasChestDiscomfort + hasGIContext and no cardiac keywords → GI cause → no flag → GREEN

    // ── Extract follow-up answer details (from combined text after patient replies) ──
    if (hasCardiacChest || hasChestDiscomfort) {
      // Q1: new symptom?
      if (/\b(new\s*symptom|first\s*time|never\s*(had|experienced)\s*(this|it)\s*before|started\s*recently|yes[,.]?\s*(it\s*is|this\s*is)\s*new)\b/i.test(text))
        symptoms.chestIsNew = true;
      if (/\b(not\s*new|existing|had\s*(this|it)\s*before|always\s*had|usual\s*symptom|been\s*having|chronic|same\s*as\s*(before|usual)|no[,.]?\s*(it[''s]*\s*)?(is\s*)?not\s*new)\b/i.test(text))
        symptoms.chestIsNew = false;

      // Q2: co-occurring symptoms
      if (/shooting.*arm|arm.*pain|pain.*left\s*side.*chest|left\s*side.*pain/i.test(text))
        symptoms.chestCoArmPain = true;
      if (/short(ness)?\s*of\s*breath|breathless/i.test(text))
        symptoms.chestCoSOB = true;
      if (/racing.*heart|heart.*rac|palpitation|irregular.*heart/i.test(text))
        symptoms.chestCoRacingHeart = true;
      if (/\bsweat(ing)?\b/i.test(text))
        symptoms.chestCoSweating = true;
      if (/\b(nausea|nauseat|vomit)\b/i.test(text))
        symptoms.chestCoNausea = true;
      if (/regurgitat|heartburn|indigestion|acid\s*reflux/i.test(text))
        symptoms.chestCoRegurgitation = true;
      if (/none\s*of\s*(these|the\s*above|them)|no\s*(other|concurrent|additional)\s*symptoms?/i.test(text))
        symptoms.chestCoNone = true;

      // Q3: rest vs. activity
      if (/\bat\s*rest\b|resting|just\s*sitting|while\s*sitting|not\s*(being\s*)?active|lying\s*(down|flat)/i.test(text))
        symptoms.chestAtRest = true;
      if (/\b(while\s*being\s*active|during\s*activit|walking|exercis|moving|going\s*up|climbing)/i.test(text))
        symptoms.chestDuringActivity = true;
    }

    if (/faint(ing|ed)?|pass(ed)?\s*out|los(t|ing)\s*consciousness|blackout|collaps/i.test(text))
      symptoms.faintingOrLoss = true;

    if (/pink\s*mucus|foamy\s*(cough|mucus)|cough(ing)?\s*(up\s*)?(blood|pink|foam)/i.test(text))
      symptoms.pinkFoamyCough = true;

    if (/confus(ed|ion)|disoriented|not\s*think(ing)?\s*clearly|mental\s*status/i.test(text))
      symptoms.severeConfusion = true;

    if (/stroke|facial\s*droop|arm\s*weak|speech\s*(difficult|problem|slur)|slurred/i.test(text))
      symptoms.strokeSigns = true;

    if (/swell(ing|en|ed)|edema|puffy\s*(legs?|ankles?|feet|foot)/i.test(text))
      symptoms.swellingNewWorse = true;

    if (/(extra|more)\s*pillow|lying\s*(flat|down)|sleep(ing)?\s*(sitting|upright|elevated)|wak(e|ing|ed)\s*(up\s*)?(breathless|short\s*of\s*breath|gasping)/i.test(text))
      symptoms.ortho = true;

    if (/dizzy|dizziness|lightheaded|light-headed/i.test(text))
      symptoms.dizzinessStanding = true;

    if (/palpitation|irregular\s*(heart|beat|pulse|rhythm)|flutter(ing)?|skipping\s*beat/i.test(text))
      symptoms.irregularHeartbeat = true;

    if (/racing\s*(heart|pulse)|heart\s*rac(ing|e)|rapid\s*(heart|pulse|beat)|tachycardia/i.test(text))
      symptoms.rapidHeartRate = true;

    if (/fatigue|tired|exhausted|weak(ness)?|no\s*energy/i.test(text))
      symptoms.unusualFatigue = true;

    if (/cough(ing)?/i.test(text) && !symptoms.pinkFoamyCough)
      symptoms.newCough = true;

    if (/less\s*(urin|pee)|not\s*urinat|decreased\s*urin|dark\s*urin|hardly\s*pee/i.test(text))
      symptoms.decreasedUrine = true;

    // SOB — infer severity from context words
    if (/short(ness)?\s*of\s*breath|breathless|can'?t\s*breath|difficulty\s*breath(ing)?|dyspnea|hard\s*to\s*breath/i.test(text)) {
      if (/at\s*rest|just\s*sitting|resting|severe|can'?t\s*breath|struggling|gasping/i.test(text))
        symptoms.sobScale = 8;
      else if (/moderate|walking|mild\s*activity/i.test(text))
        symptoms.sobScale = 5;
      else
        symptoms.sobScale = 4;
    }

    // Weight gain — extract numeric value
    const weightPatterns = [
      /gain(ed)?\s+(\d+\.?\d*)\s*(pound|lb)/i,
      /(\d+\.?\d*)\s*(pound|lb)\s*(heavier|gain)/i,
      /weight\s*(gain|gained|up)\s*(\d+\.?\d*)\s*(pound|lb)/i,
      /up\s+(\d+\.?\d*)\s*(pound|lb)/i
    ];
    for (const pattern of weightPatterns) {
      const m = text.match(pattern);
      if (m) {
        const amount = parseFloat(m.find((v, i) => i > 0 && /^\d/.test(v)));
        if (!isNaN(amount)) {
          if (/week|7\s*day/i.test(text)) symptoms.weightGain1Week = amount;
          else symptoms.weightGain1Day = amount;
        }
        break;
      }
    }

    // Explicit heart rate BPM
    const hrMatch = text.match(/(\d{2,3})\s*(bpm|beats?\s*per\s*minute)/i);
    if (hrMatch) symptoms.heartRateBpm = parseInt(hrMatch[1]);

    return symptoms;
  }

  /**
   * Determine which follow-up questions are still needed before running rule-based triage.
   * Questions are skipped when the patient already provided that information.
   * Fix #4 & #5: ask follow-ups per traffic light logic, skip already-known info.
   * @param {Object} extractedSymptoms - from extractSymptomsFromText
   * @param {string} text - original patient text
   * @returns {Array<{key: string, question: string}>}
   */
  getNeededFollowUps(extractedSymptoms, text) {
    const questions = [];

    // ── Leg swelling ──────────────────────────────────────────────────────
    if (extractedSymptoms.swellingNewWorse) {
      // Traffic light tool asks: did you take an extra water pill?
      if (!/water\s*pill|diuretic|furosemide|torsemide|lasix|bumex|extra\s*pill/i.test(text)) {
        questions.push({
          key: "waterPill",
          question: "Have you already tried taking an extra water pill (diuretic) as directed by your doctor? If yes, has the swelling improved, stayed the same, or gotten worse?"
        });
      }
    }

    // ── Shortness of breath ────────────────────────────────────────────────
    if (/short(ness)?\s*of\s*breath|breathless|dyspnea|can'?t\s*breath/i.test(text)) {
      // Ask severity only if not already rated
      const alreadyRated = /\b([0-9]|10)\s*(out\s*of|\/)\s*10|\brate[sd]?\b.*\b[0-9]\b|\b[0-9]\b.*\bscale/i.test(text);
      if (!alreadyRated && extractedSymptoms.sobScale < 4) {
        questions.push({
          key: "sobSeverity",
          question: "On a scale of 0–10, how severe is your shortness of breath? (0 = none, 10 = worst ever)"
        });
      }
      // Ask activity context only if NOT already mentioned (Fix #5)
      const activityAlreadyMentioned = /at\s*rest|just\s*sitting|resting|walking|during\s*walk|with\s*activit|exert|getting\s*worse\s*when\s*(walk|mov|climb|stand)/i.test(text);
      if (!activityAlreadyMentioned) {
        questions.push({
          key: "sobActivity",
          question: "Does the breathlessness happen only during activity, or also when you are at rest?"
        });
      }
    }

    // ── Weight gain ────────────────────────────────────────────────────────
    if (/weight\s*(gain|up|increase)|gained.*(?:pound|lb|kg)/i.test(text)) {
      if (extractedSymptoms.weightGain1Day === 0 && extractedSymptoms.weightGain1Week === 0) {
        questions.push({
          key: "weightAmount",
          question: "How many pounds (or kg) have you gained, and over what period — since yesterday, or over the past week?"
        });
      }
    }

    // ── Chest discomfort decision tree (traffic light tool — 3 questions) ──
    // Only applies to "discomfort/ache" — clear cardiac pain goes straight to RED.
    if (extractedSymptoms.chestDiscomfortOnly) {
      // Q1 — Is it a new symptom? (skip if already stated)
      const newAlreadyStated = /\b(new\s*symptom|first\s*time|never.*before|not\s*new|existing|had.*before|usual|chronic|same\s*as\s*before)\b/i.test(text);
      if (!newAlreadyStated && extractedSymptoms.chestIsNew === null) {
        questions.push({
          key: "chestNew",
          question: "Is this chest discomfort a new symptom for you, or is it something you have experienced before?"
        });
      }

      // Q2 — Co-occurring symptoms? (skip if any already mentioned)
      const coAlreadyMentioned =
        extractedSymptoms.chestCoArmPain || extractedSymptoms.chestCoSOB ||
        extractedSymptoms.chestCoRacingHeart || extractedSymptoms.chestCoSweating ||
        extractedSymptoms.chestCoNausea || extractedSymptoms.chestCoRegurgitation ||
        extractedSymptoms.chestCoNone;
      if (!coAlreadyMentioned) {
        questions.push({
          key: "chestCoSymptoms",
          question: "Are any of the following also happening alongside the chest discomfort? Please let me know which apply:\n" +
            "• Shooting pain down your arm, or pain on the left side of your chest\n" +
            "• Shortness of breath\n" +
            "• Heart racing\n" +
            "• Sweating\n" +
            "• Nausea or vomiting\n" +
            "• Regurgitation or heartburn\n" +
            "• None of these"
        });
      }

      // Q3 — At rest or during activity? (skip if already stated)
      const activityAlreadyStated = /\bat\s*rest\b|resting|sitting|walking|exercis|during\s*activit|while\s*being\s*active/i.test(text);
      if (!activityAlreadyStated && !extractedSymptoms.chestAtRest && !extractedSymptoms.chestDuringActivity) {
        questions.push({
          key: "chestActivity",
          question: "Is the chest discomfort happening while you are physically active (e.g., walking, exercising), or does it also occur when you are at rest (e.g., just sitting or lying down)?"
        });
      }
    }

    return questions;
  }

  /**
   * Summarise information already provided by the patient so follow-up
   * questions can skip topics that have been addressed (Fix #5).
   * @param {string} text
   * @returns {string[]} plain-language list of known facts
   */
  extractAlreadyKnownInfo(text) {
    const known = [];

    if (/short(ness)?\s*of\s*breath|breathless/i.test(text)) {
      if (/at\s*rest|just\s*sitting|resting|not\s*mov/i.test(text))
        known.push("shortness of breath at rest");
      if (/walk|activit|exert|mov|climb|going up/i.test(text))
        known.push("shortness of breath worsens with activity or walking");
      const scaleMatch = text.match(/\b(\d{1,2})\s*(?:out\s*of|\/)\s*10/i);
      if (scaleMatch) known.push(`shortness of breath severity rated ${scaleMatch[1]}/10`);
      if (/worse|worsening|getting worse/i.test(text))
        known.push("symptoms are getting worse");
    }

    if (/swell(ing)?|edema/i.test(text)) {
      if (/usual|my usual|always|chronic/i.test(text))
        known.push("leg swelling is a known baseline symptom");
      if (/worse|worsening|getting worse/i.test(text))
        known.push("swelling is getting worse than usual");
      if (/water\s*pill|diuretic|furosemide|lasix/i.test(text))
        known.push("patient mentioned water pill / diuretic use");
    }

    if (/weight/i.test(text)) {
      const lbsMatch = text.match(/(\d+(?:\.\d+)?)\s*(?:pound|lb)/i);
      if (lbsMatch) known.push(`weight gain of ${lbsMatch[1]} lbs mentioned`);
      if (/today|one\s*day|24\s*hour|overnight/i.test(text))
        known.push("weight gain was within the last day");
      if (/week|7\s*day/i.test(text))
        known.push("weight gain was over the past week");
    }

    if (/chest/i.test(text)) {
      if (/regurgitat|heartburn|indigestion|acid/i.test(text))
        known.push("chest discomfort is accompanied by GI symptoms (regurgitation/heartburn) — likely non-cardiac");
      if (/new\s*symptom|first\s*time|never.*before/i.test(text))
        known.push("chest discomfort is a NEW symptom");
      if (/not\s*new|existing|had.*before|usual|chronic/i.test(text))
        known.push("chest discomfort is an existing/known symptom (not new)");
      if (/shooting.*arm|arm.*pain|left\s*side.*chest/i.test(text))
        known.push("concurrent shooting arm pain mentioned");
      if (/racing.*heart|heart.*rac|palpitation/i.test(text))
        known.push("concurrent racing heart mentioned");
      if (/\bsweat(ing)?\b/i.test(text))
        known.push("concurrent sweating mentioned");
      if (/nausea|vomit/i.test(text))
        known.push("concurrent nausea/vomiting mentioned");
      if (/\bat\s*rest\b|resting|just\s*sitting/i.test(text))
        known.push("chest discomfort occurs at rest");
      if (/walking|exercis|during\s*activit|while\s*being\s*active/i.test(text))
        known.push("chest discomfort occurs during activity");
      if (/none\s*of\s*(these|the\s*above)|no\s*(other|concurrent)\s*symptoms/i.test(text))
        known.push("patient confirmed no concurrent co-symptoms");
    }

    return known;
  }

  /**
   * Merge form-input symptoms with text-extracted symptoms.
   * Boolean flags use OR; numeric values take the maximum.
   * Form values take priority when the user explicitly entered them.
   */
  mergeSymptoms(formData, textData) {
    const merged = { ...textData };
    Object.keys(formData).forEach((key) => {
      const fv = formData[key];
      const tv = textData[key];
      if (typeof fv === "boolean") {
        merged[key] = fv || (tv === true);
      } else if (typeof fv === "number") {
        merged[key] = Math.max(fv || 0, tv || 0);
      } else if (fv !== null && fv !== undefined) {
        merged[key] = fv;
      }
    });
    return merged;
  }
}

// Export for use in app
window.TriageEngine = TriageEngine;
