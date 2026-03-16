"""
Refined Omaha System prompt templates — paste into your notebook and pass to test_turn().

Usage:
    from refined_prompts import MY_SS_PROMPT, MY_INT_PROMPT
    tst.test_turn(
        turn="...",
        ss_prompt_template=MY_SS_PROMPT,
        int_prompt_template=MY_INT_PROMPT,
    )

Placeholders that are filled in automatically by build_ss_prompt / build_intervention_prompt:
    {query}    — the conversation turn being classified
    {options}  — the top-K retrieved Omaha options (numbered list)
    {context}  — surrounding turns (intervention prompt only)
"""

# ─────────────────────────────────────────────────────────────────────────────
# SIGNS / SYMPTOMS PROMPT
# ─────────────────────────────────────────────────────────────────────────────
MY_SS_PROMPT = """
Identify the health problem from the conversation turn below and map it to the MOST RELEVANT Omaha System classification.

━━━ INSTRUCTIONS ━━━

1. Only classify a patient health issue that belongs to one of these four domains:
   - Environmental Domain  (e.g., safety, housing, sanitation)
   - Psychological Domain  (e.g., sadness, anxiety, confusion)
   - Physiological Domain  (e.g., pain, abnormal vitals, wound, breathing)
   - Health-related Behaviors Domain  (e.g., medication use, nutrition, substance use)

2. Return None if:
   - The turn is a nurse/clinician performing a clinical action (checking BP, assessing wound,
     reviewing meds) — those are INTERVENTIONS, not patient signs/symptoms.
   - The turn is a greeting, filler, acknowledgment, question, or casual conversation.
   - The turn describes a normal reading or normal behavior with no abnormality.
   - No option from the list closely matches.

3. SKIN WOUND RULES — choose precisely:
   - Blister, burn blister, traumatic wound, open wound, laceration, abrasion, or patient
     confirming a wound exists / how long they've had it  →  Skin | lesion/pressure ulcer
   - Wound drainage, leaking fluid  →  Skin | drainage
   - Redness or warmth around a wound  →  Skin | inflammation
   - A wound that is not closing or healing as expected  →  Skin | delayed incisional healing
   - Patient expresses general skin concern without the above specifics  →  Skin | other

4. RESPIRATION RULES:
   - Wheezing, abnormal lung sounds heard on auscultation  →  Respiration | abnormal breath sounds
   - Patient cannot breathe without assistance / breathes irregularly  →  Respiration | unable to breathe independently
   - Patient reports trouble breathing / dyspnea at rest or exertion  →  Respiration | abnormal breath sounds
     (use "unable to breathe independently" ONLY when mechanical support is implied)

5. CIRCULATION RULES:
   - Blood pressure reading that is high or low  →  Circulation | abnormal blood pressure reading
   - History of hypertension / cardiac condition without current abnormal reading  →  Circulation | other

6. MEDICATION / HEALTH BEHAVIOR RULES:
   - Patient storing or mixing medications improperly  →  Medication regimen | inadequate system for taking medication
   - Patient waiting for / lacking a needed medication  →  Medication regimen | other
   - Patient does not monitor their own glucose / has no glucometer  →  (classify under the relevant problem if an option matches; otherwise None)

7. Select the SINGLE most relevant classification. Use EXACT wording from the options — do not rephrase.

8. DO NOT overinfer. If the turn lacks enough information, return None.

━━━ QUERY ━━━

{query}

━━━ OPTIONS ━━━

{options}

━━━ RESPONSE FORMAT ━━━

If a match is found (ONE classification only):
Domain: [Exact match]
Problem: [Exact match]
Signs/Symptoms: [Exact match]

If no match:
None

━━━ EXAMPLES ━━━

Example 1 — abnormal vital sign reading:
Query: "144/82. Okay. A little up. A little bit, but not bad."
Response:
Domain: Physiological Domain
Problem: Circulation
Signs/Symptoms: abnormal blood pressure reading

Example 2 — wound confirmed by patient (duration):
Query: "Oh, let me see. For five days. Five days."
Response:
Domain: Physiological Domain
Problem: Skin
Signs/Symptoms: lesion/pressure ulcer

Example 3 — traumatic wound mechanism:
Query: "So traumatic wound."
Response:
Domain: Physiological Domain
Problem: Skin
Signs/Symptoms: lesion/pressure ulcer

Example 4 — wound drainage:
Query: "Leaking some, yeah, leaky."
Response:
Domain: Physiological Domain
Problem: Skin
Signs/Symptoms: drainage

Example 5 — wound redness:
Query: "So you see? It is red."
Response:
Domain: Physiological Domain
Problem: Skin
Signs/Symptoms: inflammation

Example 6 — abnormal breath sounds on auscultation:
Query: "Clear. Very good. A little wheezing. A little asthma wheezing. A little bit."
Response:
Domain: Physiological Domain
Problem: Respiration
Signs/Symptoms: abnormal breath sounds

Example 7 — breathing difficulty at night:
Query: "I can not breathe at night."
Response:
Domain: Physiological Domain
Problem: Respiration
Signs/Symptoms: abnormal breath sounds

Example 8 — nurse performing a clinical action (no SS classification):
Query: "I'm going to check your blood pressure, heart, and lungs."
Response:
None

Example 9 — normal greeting / casual talk:
Query: "Beautiful. You saved my life."
Response:
None

Example 10 — improper medication storage:
Query: "I can't figure out myself. That's taking the same thing."
Response:
Domain: Health-related Behaviors Domain
Problem: Medication regimen
Signs/Symptoms: inadequate system for taking medication

Example 11 — patient expresses pain:
Query: "Don't hurt me too much."
Response:
Domain: Physiological Domain
Problem: Pain
Signs/Symptoms: expresses discomfort/pain

Example 12 — normal blood pressure reading (do NOT classify):
Query: "98.2. No fever. Very good."
Response:
None
"""


# ─────────────────────────────────────────────────────────────────────────────
# INTERVENTION PROMPT
# ─────────────────────────────────────────────────────────────────────────────
MY_INT_PROMPT = """
You are a clinical coding assistant mapping conversation turns to Omaha System interventions.

Context (reference only):
{context}

Current turn:
{query}

━━━ TASK ━━━

1. Determine if the CURRENT TURN contains a clinician action: teaching/guidance, treatment/procedure,
   case management, or surveillance.
2. If yes, identify up to 3 Omaha System interventions expressed in this turn.
3. If no clinician action is present, return NONE.

━━━ RULES ━━━

- Only classify actions in the CURRENT TURN.
- Patient speech, greetings, filler, acknowledgments → NONE.
- Maximum 3 interventions. Do NOT repeat the same Category | Target pair.
- Use EXACT Omaha System terminology from the options list below.

━━━ CATEGORY DEFINITIONS ━━━

SURVEILLANCE — monitoring, measuring, assessing, reviewing:
  checking BP / pulse / O2 / temperature    → Surveillance | signs/symptoms-physical
  assessing heart or lungs                  → Surveillance | signs/symptoms-physical
  checking / photographing wound            → Surveillance | dressing change/wound care
  reviewing / checking medications          → Surveillance | medication administration
  asking about glucose monitoring           → Surveillance | signs/symptoms-physical

TREATMENTS AND PROCEDURES — hands-on clinical care:
  cleaning wound, applying saline/gauze     → Treatments and Procedures | dressing change/wound care
  administering medication                  → Treatments and Procedures | medication administration
  taking wound photograph                   → Treatments and Procedures | dressing change/wound care

TEACHING, GUIDANCE, AND COUNSELING — explaining, instructing, advising:
  explaining wound care steps               → Teaching, Guidance, and Counseling | dressing change/wound care
  warning about infection / redness         → Teaching, Guidance, and Counseling | signs/symptoms-physical
                                              AND Teaching, Guidance, and Counseling | dressing change/wound care
  medication instructions / teaching        → Teaching, Guidance, and Counseling | medication administration
  advising about personal hygiene / shower  → Teaching, Guidance, and Counseling | personal hygiene
  advising about safety (ice, fall risk)    → Teaching, Guidance, and Counseling | safety

CASE MANAGEMENT — coordination, referrals, contacting providers:
  calling / contacting doctor               → Case Management | medical/dental care
  arranging antibiotic prescription         → Case Management | medication coordination/ordering
  providing phone number / contact info     → Case Management | other
  coordinating follow-up visits             → Case Management | continuity of care

━━━ MULTI-ACTION RULE ━━━
Use multiple interventions ONLY when the turn clearly contains multiple distinct actions.
A turn mentioning both "check blood pressure" and "check wound" gets TWO interventions.
Do NOT list the same Category | Target more than once.

━━━ AVAILABLE OPTIONS ━━━

{options}

━━━ OUTPUT FORMAT ━━━

1. Category: [exact] | Target: [exact]
2. Category: [exact] | Target: [exact]
3. Category: [exact] | Target: [exact]

or

NONE

━━━ EXAMPLES ━━━

Query: "I'm going to check your blood pressure, heart, and lungs. And then I'll check the wound."
1. Category: Surveillance | Target: signs/symptoms-physical
2. Category: Surveillance | Target: dressing change/wound care

Query: "98.2. No fever. Very good."
1. Category: Surveillance | Target: signs/symptoms-physical

Query: "So redness should not get bigger. If redness gets bigger, call me."
1. Category: Teaching, Guidance, and Counseling | Target: signs/symptoms-physical
2. Category: Teaching, Guidance, and Counseling | Target: dressing change/wound care

Query: "I'll use one gauze to clean, and then put a wet one on top, and bandage it up."
1. Category: Treatments and Procedures | Target: dressing change/wound care

Query: "I'll call your doctor and we'll get the antibiotics."
1. Category: Case Management | Target: medication coordination/ordering
2. Category: Case Management | Target: medical/dental care

Query: "I'm going to write you my phone number and my name. Any problems, call me or call the office."
1. Category: Case Management | Target: other

Query: "When you shower, take the bandage off, wash with soap and water, then bandage it back up."
1. Category: Teaching, Guidance, and Counseling | Target: dressing change/wound care
2. Category: Teaching, Guidance, and Counseling | Target: personal hygiene

Query: "Now I'm going to check your medications."
1. Category: Surveillance | Target: medication administration

Query: "Oh, let me see. For five days."
NONE

Query: "Okay."
NONE
"""
