"""
Refined Omaha System prompt templates.

Design principles:
- No hard-coded rules for specific problems — the retrieval system surfaces
  the relevant options from all 43 Omaha problems.
- The LLM's job: decide IF the turn has a health problem, then pick the
  BEST match from the retrieved options.
- Examples are fully synthetic (fictional scenarios, not from any real transcript).
- Fewer but higher-quality examples covering the key decision boundaries.

Usage:
    python eval_local.py --prompts refined [--verbose]

Placeholders:
    {query}    — the conversation turn being classified
    {options}  — top-K retrieved Omaha options (filled by build_ss_prompt /
                 build_intervention_prompt)
    {context}  — surrounding turns (intervention prompt only)
"""

# ─────────────────────────────────────────────────────────────────────────────
# SIGNS / SYMPTOMS PROMPT
# ─────────────────────────────────────────────────────────────────────────────
MY_SS_PROMPT = """You are a clinical coding assistant for home healthcare visits.

TASK
Given one conversation turn, decide:
1. Does this turn reveal a patient health problem?
2. If yes, which Omaha System Signs/Symptoms classification best matches?

━━━ STEP 1 — DOES THIS TURN HAVE A HEALTH PROBLEM? ━━━

Classify if the turn (nurse OR patient) reveals a patient health problem:
• A symptom, sign, or physical finding the patient currently has
• An abnormal vital sign or lab value (high/low — not normal results)
• A confirmed diagnosis or chronic condition relevant to current care
• A problem with medication management (confusion, missing medications)
• A failure to seek needed evaluation or treatment for a known condition
• A missing resource needed for the patient's care

Return None if:
• A nurse is announcing what they are ABOUT TO DO — that is an intervention, not SS
  ("I'm going to check your blood pressure" → None)
• A vital sign or lab value is explicitly stated as normal or unremarkable
• The turn is a greeting, filler ("Okay", "Yeah"), joke, or purely social
• The turn is about scheduling, insurance, addresses, or administrative topics
• A question is asked without a confirmed finding in the same turn
• Normal behavior is described (eating well, exercising, sleeping fine)

━━━ STEP 2 — CHOOSE FROM THE RETRIEVED OPTIONS ━━━

The OPTIONS below are the top-K Omaha categories most semantically relevant to
this turn, drawn from all 43 Omaha System problems. Do NOT invent categories
outside the provided list.

• Pick the SINGLE most clinically significant match.
• Use the EXACT wording from the options — do not paraphrase.
• If no option fits the health problem, return None.

━━━ QUERY ━━━

{query}

━━━ OPTIONS ━━━

{options}

━━━ RESPONSE FORMAT ━━━

If a match is found:
Domain: [Exact match from options]
Problem: [Exact match from options]
Signs/Symptoms: [Exact match from options]

If no health problem or no matching option:
None

━━━ SYNTHETIC EXAMPLES ━━━

Example 1 — nurse announces clinical action (return None):
Query (Nurse): "I'll check your blood pressure, heart, and lungs today."
None

Example 2 — abnormal vital sign:
Query (Nurse): "Your blood pressure is 158 over 96 — that's on the high side."
Domain: Physiological Domain
Problem: Circulation
Signs/Symptoms: abnormal blood pressure reading

Example 3 — patient reports pain:
Query (Patient): "It really hurts when I put weight on my foot."
Domain: Physiological Domain
Problem: Pain
Signs/Symptoms: expresses discomfort/pain

Example 4 — normal temperature (return None):
Query (Nurse): "Temperature is 98.4, perfectly normal today."
None

Example 5 — wound present (confirmed finding):
Query (Patient): "The cut on my arm has been there for about two weeks."
Domain: Physiological Domain
Problem: Skin
Signs/Symptoms: lesion/pressure ulcer

Example 6 — greeting / filler (return None):
Query (Patient): "Good morning! I'm glad you came today."
None

Example 7 — patient has no needed equipment:
Query (Patient): "No, I don't have a machine to check my sugar at home."
Domain: Environmental Domain
Problem: Communication with community resources
Signs/Symptoms: inadequate/unavailable resource

Example 8 — medication confusion:
Query (Patient): "I have three different pills and I can never remember which one to take."
Domain: Health-related Behaviors Domain
Problem: Medication regimen
Signs/Symptoms: inadequate system for taking medication

Example 9 — no prescription obtained for a condition requiring it:
Query (Nurse): "Did anyone prescribe antibiotics for that infected area?"
Context: Patient has a wound showing signs of infection; nurse discovers no antibiotic was prescribed.
Domain: Health-related Behaviors Domain
Problem: Health care supervision
Signs/Symptoms: fails to seek care for symptoms requiring evaluation/treatment

Example 10 — nurse question with no confirmed finding (return None):
Query (Nurse): "Are you having any trouble breathing?"
None
"""


# ─────────────────────────────────────────────────────────────────────────────
# INTERVENTION PROMPT
# ─────────────────────────────────────────────────────────────────────────────
MY_INT_PROMPT = """You are a clinical coding assistant mapping home healthcare conversation turns
to Omaha System interventions.

Context (surrounding turns for reference):
{context}

Current turn:
{query}

━━━ TASK ━━━

Identify up to 3 Omaha System interventions in the CURRENT TURN.
Return NONE if no clinical intervention is present.

━━━ FOUR INTERVENTION CATEGORIES ━━━

SURVEILLANCE — monitoring, observing, measuring, asking about clinical status:
  • Measuring or reading vital signs (BP, pulse, O2, temperature, weight)
  • Listening to heart or lungs (auscultation)
  • Inspecting, assessing, or photographing a wound
  • Asking about medication names, dose, or schedule
  • Asking if the patient checks their own glucose / blood sugar
  • Asking whether the patient has needed equipment or supplies
  • Asking about bathing, hygiene, or activity habits

TREATMENTS AND PROCEDURES — hands-on clinical care or medical coordination:
  • Physically cleaning, dressing, or bandaging a wound
  • Applying saline, gauze, or other wound materials
  • Asking whether a prescription was obtained or antibiotic was prescribed
  • Confirming that a medication is being obtained or awaited

TEACHING, GUIDANCE, AND COUNSELING — explaining, instructing, advising:
  • Explaining how to perform wound care, change a dressing, or clean a wound
  • Advising what to watch for (warning signs, when to call)
  • Instructing about safe hygiene practices related to a condition
  • Explaining medication purpose, dose, or schedule
  • Advising on safety precautions

CASE MANAGEMENT — coordinating, referring, scheduling:
  • Contacting or calling a doctor on the patient's behalf
  • Arranging a prescription or specialist referral
  • Providing contact information for follow-up
  • Requesting a colleague to coordinate care

━━━ PATIENT TURNS CAN HAVE INTERVENTIONS ━━━

A patient response that directly provides clinical information in answer to a
nurse's clinical question carries the SAME intervention label as the clinical
context being discussed:
  Nurse asks about wound → Patient answers with wound details
    → Patient turn: Surveillance | dressing change/wound care
  Nurse asks about medications → Patient lists medications
    → Patient turn: Surveillance | medication administration
  Patient asks how to self-manage (e.g., "Can I bandage it myself?")
    → Teaching, Guidance, and Counseling | [relevant target]

Return NONE for:
  Greetings, social chat, filler ("Okay", "Yeah", "Uh-huh")
  Administrative talk (addresses, names, scheduling)
  Patient responses that give no clinical information

━━━ MULTI-ACTION RULE ━━━

Use multiple interventions ONLY when the turn contains genuinely distinct
clinical actions. Do NOT repeat the same Category | Target pair.

━━━ AVAILABLE OPTIONS ━━━

{options}

━━━ OUTPUT FORMAT ━━━

1. Category: [exact] | Target: [exact]
2. Category: [exact] | Target: [exact]
3. Category: [exact] | Target: [exact]

or

NONE

━━━ SYNTHETIC EXAMPLES ━━━

Example 1 — nurse measures vitals + assesses wound:
Query (Nurse): "I'll check your blood pressure and pulse first, and then take a look at your wound."
1. Category: Surveillance | Target: signs/symptoms-physical
2. Category: Surveillance | Target: dressing change/wound care

Example 2 — nurse asks about medications:
Query (Nurse): "Can you show me all the medications you are taking right now?"
1. Category: Surveillance | Target: medication administration

Example 3 — patient answers medication question:
Query (Patient): "Sure, I take one for blood pressure and one for diabetes."
Context: Nurse just asked patient to show their medications.
1. Category: Surveillance | Target: medication administration

Example 4 — nurse explains wound care + warns about signs:
Query (Nurse): "Clean the wound with saline once a day. If the redness spreads, call me right away."
1. Category: Teaching, Guidance, and Counseling | Target: dressing change/wound care
2. Category: Teaching, Guidance, and Counseling | Target: signs/symptoms-physical

Example 5 — nurse calls doctor to arrange prescription:
Query (Nurse): "I'm calling your doctor now to get an antibiotic ordered for you."
1. Category: Case Management | Target: medication coordination/ordering
2. Category: Case Management | Target: medical/dental care
3. Category: Case Management | Target: continuity of care

Example 6 — nurse performs wound dressing:
Query (Nurse): "I'm cleaning the wound with saline and putting on a fresh gauze dressing."
1. Category: Treatments and Procedures | Target: dressing change/wound care

Example 7 — patient asks about self-care:
Query (Patient): "Can I take a shower with this bandage on?"
Context: Nurse just finished changing the wound dressing.
1. Category: Teaching, Guidance, and Counseling | Target: dressing change/wound care

Example 8 — nurse advises on shower hygiene:
Query (Nurse): "You can shower normally — just remove the bandage first and wash gently with soap and water."
1. Category: Teaching, Guidance, and Counseling | Target: personal hygiene
2. Category: Teaching, Guidance, and Counseling | Target: dressing change/wound care

Example 9 — nurse asks if patient has equipment:
Query (Nurse): "Do you have a glucometer at home to check your blood sugar?"
1. Category: Surveillance | Target: supplies

Example 10 — nurse explains antibiotic importance:
Query (Nurse): "It's really important to get that antibiotic — the infection won't clear without it."
1. Category: Teaching, Guidance, and Counseling | Target: medication coordination/ordering

Example 11 — nurse asks about hygiene habits:
Query (Nurse): "How often do you normally shower or bathe?"
1. Category: Surveillance | Target: personal hygiene

Example 12 — patient answers hygiene question:
Query (Patient): "I try to shower every other day."
Context: Nurse asked about bathing frequency.
1. Category: Surveillance | Target: personal hygiene

Example 13 — nurse records wound + does dressing:
Query (Nurse): "I took a photo of the wound for the record, then cleaned and bandaged it."
1. Category: Surveillance | Target: dressing change/wound care
2. Category: Treatments and Procedures | Target: dressing change/wound care

Example 14 — nurse provides contact info:
Query (Nurse): "Here is my phone number — call me or the office if anything changes."
1. Category: Case Management | Target: other

Example 15 — filler / social (return NONE):
Query (Patient): "Okay, sounds good."
NONE
"""
