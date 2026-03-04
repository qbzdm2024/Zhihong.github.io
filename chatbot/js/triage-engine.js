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
      chestPain = false,           // chest pain or pressure
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
   * @param {string} freeTextSymptoms - User's description of symptoms
   * @param {Object|null} structuredSymptoms - Optional structured data
   * @returns {string} system prompt for AI triage
   */
  buildAITriagePrompt(freeTextSymptoms, structuredSymptoms = null) {
    let structuredSection = "";
    if (structuredSymptoms) {
      const ruleResult = this.ruleBasedTriage(structuredSymptoms);
      structuredSection = `
## Rule-Based Triage (for comparison)
The deterministic traffic light system classified this as: **${ruleResult.zone} ZONE**
Triggered flags: ${ruleResult.flags.length > 0 ? ruleResult.flags.join("; ") : "None"}
`;
    }

    return `You are a clinical decision support assistant for heart failure patient triage.
Your task is to assess the patient's symptoms and provide a triage recommendation using clinical reasoning.

This is for EDUCATIONAL and COMPARATIVE purposes. Your triage result will be compared against a deterministic rule-based traffic light system.

IMPORTANT: Always advise the patient to consult their healthcare team. You are a support tool, not a replacement for medical advice.

${structuredSection}

## Patient-Reported Symptoms
${freeTextSymptoms}

## Instructions
Perform a clinical triage assessment:

1. **Zone Assignment**: Assign one of three zones:
   - 🟢 GREEN: Stable — continue usual care
   - 🟡 YELLOW: Concerning — contact care team within hours
   - 🔴 RED: Emergency — call 911 immediately

2. **Clinical Reasoning**: Explain WHY you assigned this zone using evidence-based reasoning. Consider:
   - Hemodynamic instability signs
   - Fluid overload indicators
   - Ischemic/arrhythmic features
   - Functional status change
   - Risk stratification factors

3. **Key Symptoms of Concern**: List the specific symptoms driving your assessment

4. **Immediate Actions**: Give specific, actionable instructions

5. **Comparison Note**: If rule-based data is provided, note any agreement or discrepancy and explain the clinical basis for any difference

Base your reasoning on the 2022 AHA/ACC/HFSA Heart Failure Guidelines and standard clinical practice.

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
}

// Export for use in app
window.TriageEngine = TriageEngine;
