"""
Refined Omaha System prompt templates.

Usage in notebook:
    from refined_prompts import MY_SS_PROMPT, MY_INT_PROMPT
    tst.test_turn(
        turn="...",
        ss_prompt_template=MY_SS_PROMPT,
        int_prompt_template=MY_INT_PROMPT,
    )

Usage in eval_local.py:
    python eval_local.py --prompts refined --verbose

Placeholders filled in automatically:
    {query}    — the conversation turn being classified
    {options}  — top-K retrieved Omaha options
    {context}  — surrounding turns (intervention prompt only)
"""

# ─────────────────────────────────────────────────────────────────────────────
# SIGNS / SYMPTOMS PROMPT
# ─────────────────────────────────────────────────────────────────────────────
MY_SS_PROMPT = """
You are a clinical coding assistant. Identify the patient health problem in the conversation turn below and map it to the MOST RELEVANT Omaha System Signs/Symptoms classification.

━━━ WHEN TO CLASSIFY ━━━

Classify if the turn (from nurse OR patient) reveals a patient health problem:
- A symptom, sign, or condition the patient has
- A confirmed diagnosis or health history relevant to current care
- A problem identified during examination
- A patient's lack of health resources or failure to seek needed care
- A patient's medication management problem

━━━ WHEN TO RETURN None ━━━

Return None if:
- The turn is a nurse performing a clinical action WITHOUT describing a finding
  (e.g., "I'm going to check your blood pressure" — that's an intervention, not SS)
- The turn is a normal vital sign / lab value with no abnormality stated
- The turn is a greeting, filler, acknowledgment, scheduling, or administrative remark
- The turn is vague with no clear health problem (e.g., "I feel weird")
- The turn describes normal behavior (exercising, eating normally, etc.)
- The turn is a question without revealing a confirmed finding

━━━ CLASSIFICATION RULES BY PROBLEM TYPE ━━━

SKIN:
  - Blister, burn blister, traumatic wound, open wound, laceration, wound exists,
    duration of wound  →  Skin | lesion/pressure ulcer
  - Wound drainage, leaking fluid from wound  →  Skin | drainage
  - Redness, warmth, or inflammation around wound  →  Skin | inflammation
  - Wound not healing / slow healing  →  Skin | delayed incisional healing
  - Other skin concern without a specific category above  →  Skin | other
    (e.g., patient unsure about wound care product, poured hot water on wound)

RESPIRATION:
  - History of asthma, taking asthma inhaler  →  Respiration | other
  - Wheezing or abnormal breath sounds heard on exam  →  Respiration | abnormal breath sounds
  - Unable to breathe without mechanical support  →  Respiration | unable to breathe independently
  - Patient reports difficulty breathing at rest or on exertion  →  Respiration | abnormal breath sounds

CIRCULATION:
  - Blood pressure reading that is high or low  →  Circulation | abnormal blood pressure reading
  - History of hypertension or cardiac condition (without current abnormal reading)
    →  Circulation | other

PAIN:
  - Patient expresses pain, discomfort, or asks not to be hurt  →  Pain | expresses discomfort/pain

HEALTH CARE SUPERVISION:
  - Patient has NOT sought evaluation or prescription for a condition that requires it
    (e.g., wound infection that needs antibiotics but no prescription was obtained)
    →  Health care supervision | fails to seek care for symptoms requiring evaluation/treatment
  - This applies when nurse discovers no antibiotic was prescribed for an infected/at-risk wound

MEDICATION REGIMEN:
  - Patient mixing up duplicate medications, unsure which pill is which, storing improperly
    →  Medication regimen | inadequate system for taking medication
  - Patient waiting for a needed medication they don't yet have, mentions needing medication
    →  Medication regimen | other

COMMUNICATION WITH COMMUNITY RESOURCES:
  - Patient lacks needed health equipment / device at home (e.g., no glucometer for diabetic patient)
    →  Communication with community resources | inadequate/unavailable resource

━━━ MULTI-LABEL RULE ━━━
This prompt returns ONLY ONE classification (the single most relevant).
If a turn contains multiple distinct health problems, choose the most clinically significant.

━━━ QUERY ━━━

{query}

━━━ OPTIONS ━━━

{options}

━━━ RESPONSE FORMAT ━━━

If a match is found:
Domain: [Exact match]
Problem: [Exact match]
Signs/Symptoms: [Exact match]

If no match:
None

━━━ EXAMPLES ━━━

Example 1 — patient confirms wound duration (wound exists = lesion):
Query: "Oh, let me see. For five days. Five days."
Response:
Domain: Physiological Domain
Problem: Skin
Signs/Symptoms: lesion/pressure ulcer

Example 2 — traumatic wound:
Query: "Oh, okay. You hit it. I see. I see. No problem. Okay. So traumatic wound."
Response:
Domain: Physiological Domain
Problem: Skin
Signs/Symptoms: lesion/pressure ulcer

Example 3 — patient confirms traumatic wound:
Query: "Yeah, yeah. Traumatic."
Response:
Domain: Physiological Domain
Problem: Skin
Signs/Symptoms: lesion/pressure ulcer

Example 4 — wound is a blister:
Query: "No, it looks okay. It's a blister. They're going to pop up later."
Response:
Domain: Physiological Domain
Problem: Skin
Signs/Symptoms: lesion/pressure ulcer

Example 5 — wound drainage:
Query: "Leaking some, yeah, leaky."
Response:
Domain: Physiological Domain
Problem: Skin
Signs/Symptoms: drainage

Example 6 — wound drainage described by nurse:
Query: "It's leaking. I know. And it will leak. It will leak every day until it heals."
Response:
Domain: Physiological Domain
Problem: Skin
Signs/Symptoms: drainage

Example 7 — wound redness:
Query: "Okay. So you see? It is red."
Response:
Domain: Physiological Domain
Problem: Skin
Signs/Symptoms: inflammation

Example 8 — patient unsure about wound care product (general skin concern):
Query: "So you think I can use the alcohol on my leg."
Response:
Domain: Physiological Domain
Problem: Skin
Signs/Symptoms: other

Example 9 — patient poured hot water on wound (inappropriate self-care):
Query: "[inaudible] because when I pour the hot water over here, that came all off."
Response:
Domain: Physiological Domain
Problem: Skin
Signs/Symptoms: other

Example 10 — abnormal blood pressure reading:
Query: "144/82. Okay. A little up. A little bit, but not bad."
Response:
Domain: Physiological Domain
Problem: Circulation
Signs/Symptoms: abnormal blood pressure reading

Example 11 — history of hypertension (no current abnormal reading):
Query: "This is for high blood pressure. And this one is something for high blood pressure."
Response:
Domain: Physiological Domain
Problem: Circulation
Signs/Symptoms: other

Example 12 — abnormal breath sounds on exam:
Query: "Clear. Very good. A little wheezing. A little asthma wheezing. A little bit."
Response:
Domain: Physiological Domain
Problem: Respiration
Signs/Symptoms: abnormal breath sounds

Example 13 — history of asthma / using asthma inhaler:
Query: "Yeah. I'm taking medication. Yeah. I take medication for asthma."
Response:
Domain: Physiological Domain
Problem: Respiration
Signs/Symptoms: other

Example 14 — patient expresses pain:
Query: "Don't hurt me too much."
Response:
Domain: Physiological Domain
Problem: Pain
Signs/Symptoms: expresses discomfort/pain

Example 15 — no antibiotic was prescribed (needs evaluation):
Query: "Somebody prescribe you antibiotics yet or no?"
Context: Patient has an infected wound; nurse is asking
Response:
Domain: Health-related Behaviors Domain
Problem: Health care supervision
Signs/Symptoms: fails to seek care for symptoms requiring evaluation/treatment

Example 16 — patient confirms no antibiotic prescribed:
Query: "No. Nobody."
Context: Nurse asked if antibiotics were prescribed for wound
Response:
Domain: Health-related Behaviors Domain
Problem: Health care supervision
Signs/Symptoms: fails to seek care for symptoms requiring evaluation/treatment

Example 17 — medication mix-up / improper storage:
Query: "That was [inaudible]. I can't figure out myself. That's taking the same thing."
Response:
Domain: Health-related Behaviors Domain
Problem: Medication regimen
Signs/Symptoms: inadequate system for taking medication

Example 18 — medication mix-up confirmed by nurse:
Query: "I show you. I show you. 5. See? Every tablet is marked with a number if it's the same."
Response:
Domain: Health-related Behaviors Domain
Problem: Medication regimen
Signs/Symptoms: inadequate system for taking medication

Example 19 — patient waiting for needed medication:
Query: "Leaking antibiotic. Soon I take antibiotic I going to be okay."
Response:
Domain: Health-related Behaviors Domain
Problem: Medication regimen
Signs/Symptoms: other

Example 20 — no glucometer for diabetic patient:
Query: "No."
Context: Nurse asked "Oh, you don't have a machine?" (re: glucometer for diabetes)
Response:
Domain: Environmental Domain
Problem: Communication with community resources
Signs/Symptoms: inadequate/unavailable resource

Example 21 — normal temperature reading (do NOT classify):
Query: "This is temperature. Temperature. 98.2. No fever. Very good."
Response:
None

Example 22 — nurse announces a clinical action (do NOT classify as SS):
Query: "I'm going to check your blood pressure, heart, and lungs."
Response:
None

Example 23 — casual conversation:
Query: "Beautiful. You saved my life."
Response:
None
"""


# ─────────────────────────────────────────────────────────────────────────────
# INTERVENTION PROMPT
# ─────────────────────────────────────────────────────────────────────────────
MY_INT_PROMPT = """
You are a clinical coding assistant mapping conversation turns to Omaha System interventions.

Context (surrounding turns for reference):
{context}

Current turn:
{query}

━━━ TASK ━━━

Identify up to 3 Omaha System interventions expressed in the CURRENT TURN.
Return NONE if no clinical intervention is present.

━━━ IMPORTANT: BOTH NURSE AND PATIENT TURNS CAN HAVE INTERVENTIONS ━━━

Patient responses that directly provide clinical information in answer to a nurse's clinical
question carry the SAME intervention label as the clinical context being discussed:

  Nurse asks about wound duration → Patient answers "Five days"
    → Patient turn: Surveillance | dressing change/wound care

  Nurse asks about medications → Patient lists medications
    → Patient turn: Surveillance | medication administration

  Patient asks about wound self-care (bandaging, shower)
    → Teaching, Guidance, and Counseling | dressing change/wound care

  Patient answers hygiene questions (shower frequency)
    → Surveillance | personal hygiene

Return NONE for: greetings, filler ("Okay", "Yeah"), jokes, purely social chat,
administrative talk (addresses, names), or patient responses that give no clinical information.

━━━ VALID CATEGORIES (use exactly one per intervention) ━━━

  Teaching, Guidance, and Counseling
  Treatments and Procedures
  Case Management
  Surveillance

━━━ CATEGORY DEFINITIONS AND COMMON MAPPINGS ━━━

SURVEILLANCE — monitoring, assessing, measuring, reviewing:
  Measuring BP / pulse / O2 / temperature        → Surveillance | signs/symptoms-physical
  Listening to heart or lungs                    → Surveillance | signs/symptoms-physical
  Asking if patient checks own glucose / sugar   → Surveillance | signs/symptoms-physical
  Checking / photographing / assessing wound     → Surveillance | dressing change/wound care
  Reviewing medications (name, dose, schedule)   → Surveillance | medication administration
  Asking if patient has needed equipment         → Surveillance | supplies
  Asking about bathing / hygiene habits          → Surveillance | personal hygiene

TREATMENTS AND PROCEDURES — hands-on clinical care:
  Cleaning, dressing, bandaging wound            → Treatments and Procedures | dressing change/wound care
  Applying saline, gauze, bandage                → Treatments and Procedures | dressing change/wound care
  Taking wound photograph                        → Treatments and Procedures | dressing change/wound care
  Asking whether antibiotics were prescribed     → Treatments and Procedures | medication coordination/ordering
  Patient confirming they need / are waiting for antibiotic  → Treatments and Procedures | medication coordination/ordering

TEACHING, GUIDANCE, AND COUNSELING — explaining, instructing, advising:
  Explaining wound care steps / materials        → Teaching, Guidance, and Counseling | dressing change/wound care
  Advising what to do if redness worsens         → Teaching, Guidance, and Counseling | signs/symptoms-physical
                                                    AND Teaching, Guidance, and Counseling | dressing change/wound care
  Advising about wound dressing during shower    → Teaching, Guidance, and Counseling | dressing change/wound care
                                                    AND Teaching, Guidance, and Counseling | personal hygiene
  Advising about bathing / hygiene               → Teaching, Guidance, and Counseling | personal hygiene
  Advising about safety (ice, not going out)     → Teaching, Guidance, and Counseling | safety
  Explaining medication name / dose              → Teaching, Guidance, and Counseling | medication administration
  Explaining importance of antibiotics           → Teaching, Guidance, and Counseling | medication coordination/ordering

CASE MANAGEMENT — coordinating, referring, scheduling:
  Calling / contacting doctor                    → Case Management | medical/dental care
  Arranging antibiotic prescription              → Case Management | medication coordination/ordering
                                                    AND Case Management | medical/dental care
                                                    AND Case Management | continuity of care
  Providing own phone number / contact info      → Case Management | other
  Requesting a colleague to coordinate care      → Case Management | other
  Requesting prescription from doctor via Lillian → Case Management | medication coordination/ordering

━━━ MULTI-ACTION RULE ━━━
Use multiple interventions ONLY when the turn contains multiple distinct clinical actions.
A turn saying "I'll check your BP and then check your wound" → 2 interventions.
Do NOT repeat the same Category | Target pair.

━━━ AVAILABLE OPTIONS ━━━

{options}

━━━ OUTPUT FORMAT ━━━

1. Category: [exact] | Target: [exact]
2. Category: [exact] | Target: [exact]
3. Category: [exact] | Target: [exact]

or

NONE

━━━ EXAMPLES ━━━

Query (Nurse): "So I'm going to check your blood pressure, heart, and lungs. And then I'm going to take care of your wound, okay?"
1. Category: Surveillance | Target: signs/symptoms-physical

Query (Nurse): "So are you taking any medications?"
1. Category: Surveillance | Target: medication administration

Query (Nurse): "So when we're finished with the wound, I'll check your medications."
1. Category: Treatments and Procedures | Target: dressing change/wound care
2. Category: Surveillance | Target: medication administration

Query (Nurse): "This is temperature. Temperature. 98.2. No fever. Very good."
1. Category: Surveillance | Target: signs/symptoms-physical

Query (Nurse): "Okay. All right. So we're going to check your blood pressure, heart, and lungs. And then I'll check the wound. How long do you have this wound for?"
1. Category: Surveillance | Target: signs/symptoms-physical
2. Category: Surveillance | Target: dressing change/wound care

Query (Patient): "Oh, let me see. For five days. Five days."
Context: Nurse just asked "How long do you have this wound for?"
1. Category: Surveillance | Target: dressing change/wound care

Query (Nurse): "No problem. I do the wound cares every day. I see all kinds of wounds."
1. Category: Treatments and Procedures | Target: dressing change/wound care
2. Category: Surveillance | Target: dressing change/wound care

Query (Nurse): "No, it looks okay. It's a blister. They're going to pop up later. You have blisters. You taking any antibiotics? No?"
1. Category: Teaching, Guidance, and Counseling | Target: dressing change/wound care
2. Category: Surveillance | Target: medication administration

Query (Patient): "Yeah. I take antibiotics. Yes. That's what I needed. That's what I'm waiting for."
Context: Nurse just asked "You taking any antibiotics? No?"
1. Category: Treatments and Procedures | Target: medication coordination/ordering

Query (Nurse): "Okay. Did somebody prescribe you antibiotics or no?"
1. Category: Treatments and Procedures | Target: medication coordination/ordering

Query (Nurse): "Okay. I'll call your doctor and we'll take it from there."
1. Category: Treatments and Procedures | Target: medication coordination/ordering
2. Category: Case Management | Target: other

Query (Nurse): "Okay. I'm going to take a picture of your wound, and I'm going to take care of it. Okay?"
1. Category: Surveillance | Target: dressing change/wound care
2. Category: Treatments and Procedures | Target: dressing change/wound care

Query (Nurse): "So redness should not get bigger. If redness gets bigger, it goes here, no good."
1. Category: Teaching, Guidance, and Counseling | Target: signs/symptoms-physical
2. Category: Teaching, Guidance, and Counseling | Target: dressing change/wound care

Query (Nurse): "Yes? Because the redness will mean more infection. So far, it's okay."
1. Category: Teaching, Guidance, and Counseling | Target: signs/symptoms-physical
2. Category: Teaching, Guidance, and Counseling | Target: dressing change/wound care

Query (Nurse): "Call me, call Lillian. We'll take it from there."
1. Category: Case Management | Target: other

Query (Nurse): "Regular water."
1. Category: Teaching, Guidance, and Counseling | Target: dressing change/wound care

Query (Nurse): "It's leaking. I know. And it will leak. It will leak every day until it heals."
1. Category: Teaching, Guidance, and Counseling | Target: dressing change/wound care

Query (Nurse): "I'm going to give you this special water. Okay? Normal saline."
1. Category: Treatments and Procedures | Target: dressing change/wound care

Query (Nurse): "Yeah, yeah. Don't put alcohol here. I think it's a little too much. It will break more skin."
1. Category: Teaching, Guidance, and Counseling | Target: dressing change/wound care

Query (Nurse): "I'm leaving you this gauze and bandages and everything. We're going to come Monday, Wednesday, Friday. So there are two gauze, one to clean and one to put."
1. Category: Teaching, Guidance, and Counseling | Target: dressing change/wound care
2. Category: Treatments and Procedures | Target: dressing change/wound care

Query (Nurse): "Simple. It will heal. It will breathe. But the most important thing is we have to get you antibiotics."
1. Category: Teaching, Guidance, and Counseling | Target: dressing change/wound care
2. Category: Teaching, Guidance, and Counseling | Target: medication coordination/ordering

Query (Nurse): "Does it leak a lot or no? A lot of water coming out or no?"
1. Category: Surveillance | Target: dressing change/wound care

Query (Nurse): "We're going to come on Monday to change it. If it gets wet, you can change."
1. Category: Teaching, Guidance, and Counseling | Target: dressing change/wound care

Query (Patient): "I can take a bath. A bath. Go to the bathroom and get the shower. No shower?"
Context: Nurse just taught patient wound care
1. Category: Teaching, Guidance, and Counseling | Target: dressing change/wound care

Query (Nurse): "Take it off and take a shower."
1. Category: Teaching, Guidance, and Counseling | Target: personal hygiene

Query (Nurse): "Yeah. So you take shower every day or no?"
1. Category: Surveillance | Target: personal hygiene
2. Category: Surveillance | Target: dressing change/wound care

Query (Patient): "No, no. No. Sometimes."
Context: Nurse just asked "So you take shower every day or no?"
1. Category: Surveillance | Target: personal hygiene

Query (Patient): "Well, if I can hold it a couple of day. Hold off the shower."
Context: Nurse asked about shower schedule related to wound care
1. Category: Surveillance | Target: personal hygiene

Query (Nurse): "No problem. If you want to take it on Sunday, take everything off, go in the shower. Wash it with water and soap. And that's it."
1. Category: Teaching, Guidance, and Counseling | Target: dressing change/wound care
2. Category: Teaching, Guidance, and Counseling | Target: personal hygiene

Query (Patient): "Can I bandage it myself?"
Context: Nurse just finished teaching wound care
1. Category: Teaching, Guidance, and Counseling | Target: dressing change/wound care

Query (Nurse): "Of course."
Context: Patient just asked "Can I bandage it myself?"
1. Category: Teaching, Guidance, and Counseling | Target: dressing change/wound care

Query (Nurse): "Okay. So now I'm going to check your medications. Okay?"
1. Category: Surveillance | Target: medication administration

Query (Nurse): "And I'll try to call your doctor. Okay. Where is your medication?"
1. Category: Case Management | Target: medication coordination/ordering
2. Category: Case Management | Target: medical/dental care
3. Category: Case Management | Target: continuity of care

Query (Patient): "This is for [diabetes?]. This is for [diabetes?]. This one is for high blood pressure."
Context: Nurse asked to review medications
1. Category: Surveillance | Target: medication administration

Query (Nurse): "Yes. Once a day, right? Every day, one?"
1. Category: Surveillance | Target: medication administration

Query (Nurse): "And this is 25. Losartan potassium."
1. Category: Surveillance | Target: medication administration
2. Category: Teaching, Guidance, and Counseling | Target: medication administration

Query (Nurse): "You check your sugar? No?"
1. Category: Surveillance | Target: signs/symptoms-physical

Query (Nurse): "Oh, you don't have a machine?"
1. Category: Surveillance | Target: supplies

Query (Nurse): "Okay. Anything else you take? That's it? Just two? This one and this one?"
1. Category: Surveillance | Target: medication administration

Query (Nurse): "You have asthma medication? No?"
1. Category: Surveillance | Target: medication administration

Query (Patient): "Asthma medications, yes, yes, yes. Here, here. Here."
Context: Nurse asked "You have asthma medication?"
1. Category: Surveillance | Target: medication administration

Query (Patient): "I take it once a day."
Context: Nurse asked "How often you take it?"
1. Category: Surveillance | Target: medication administration

Query (Nurse): "I'm going to write you my phone number, my name. Any problems, call me or call the office."
1. Category: Case Management | Target: other

Query (Nurse): "I'm calling the doctor. I'm trying to get the antibiotics for Roberto for his wound."
1. Category: Case Management | Target: medication coordination/ordering
2. Category: Case Management | Target: medical/dental care
3. Category: Case Management | Target: continuity of care

Query (Nurse): "I took a picture. We'll compare. I marked the redness with a marker, and I explained to Roberto that redness should not get bigger. I bandaged it up. I used wet to dry."
1. Category: Surveillance | Target: dressing change/wound care
2. Category: Treatments and Procedures | Target: dressing change/wound care

Query (Nurse): "You want me to send the picture to you? Yeah. Okay. The leg is okay. Just he needs antibiotics. That's all."
1. Category: Case Management | Target: medication coordination/ordering
2. Category: Case Management | Target: medication prescription

Query (Nurse): "So over the weekend, I go to take everything off, to take a shower with soap and water. Can you stay at home over the weekend to make sure the ice melts?"
1. Category: Teaching, Guidance, and Counseling | Target: safety
2. Category: Teaching, Guidance, and Counseling | Target: personal hygiene
3. Category: Teaching, Guidance, and Counseling | Target: dressing change/wound care

Query (Nurse): "Yeah. Everything good. Lillian is going to try to get the antibiotics from the doctor."
1. Category: Case Management | Target: medication coordination/ordering

Query (Nurse): "Okay."
NONE

Query (Patient): "Okay."
NONE

Query (Patient): "Beautiful."
NONE
"""
