# Second Mind — Project Overview

## One-liner
An ambient memory assistant for people with dementia: a room camera watches for where things are and what's happening, and a wearable mic/speaker (M5StickC Plus2) lets the patient ask natural questions and get a calm spoken answer — pulled from what the camera has observed, from personal/medical info the caregiver set up in advance, or from the patient's scheduled routines and appointments. The assistant also proactively reminds the patient about medications, appointments, daily routines, and any custom reminders the caregiver has configured — and checks its own observations before confirming events actually occurred.

## Problem
People with dementia frequently forget where they placed everyday objects (glasses, keys, medication), forget scheduled events (doctor appointments, meals, exercise), lose track of daily routines, and struggle to recall basic facts about their own life (what day it is, who a family member is, whether they've already taken a dose). They may also fail to recognize familiar faces or forget where they are in the context of the day. This causes distress and repeated, anxious questioning. Caregivers can't be present 24/7.

## Solution
Multiple data sources feed one voice assistant:

1. **Room memory** — a camera continuously (low-frequency polling, not real-time video) observes the room and writes short structured facts like "reading glasses placed on kitchen counter, 3:42pm" or "medication bottle picked up, 8:05am" into a memory log. No video is ever stored — only these text observations.
2. **Patient profile** — the caregiver (not the patient) fills in a one-time setup: patient's basic info, relevant medical history/conditions, current medications **with schedule times**, family members with names and relationships, appointments, daily routines, and any custom reminders.
3. **Schedule & appointment store** — structured upcoming events (doctor visits, meal times, exercise blocks, social calls) that feed both reactive answers and the proactive reminder loop.

The patient interacts entirely by voice through the M5StickC Plus2 wearable. They press a button to ask a question ("where are my glasses?", "what day is it?", "who is Sarah?", "do I have a doctor's appointment today?", "did I take my morning pills?"), and get a short, warm, spoken answer routed to the right source. The assistant also speaks up on its own, unprompted, across a range of reminder types — not just medication.

## Core user flows

- **"Where is X?"** → Memory retrieval agent searches the room memory log → answers from the most recent matching observation.
- **"What day/time is it? What should I do today?"** → Schedule & General QA agent answers from the schedule store and patient profile.
- **"Who is [family member]? What's their number?"** → General QA agent answers from the patient profile (name, relationship, and — if relevant to the ask — contact info).
- **"Have I taken my medication? What do I take?"** → Routed to the **Memory retrieval agent**, not General QA. General QA only knows the *schedule* (from the profile); it has no way to know whether the dose actually happened. The Memory retrieval agent pulls the medication schedule from the patient profile, then checks the room memory log for a matching "medication taken/bottle picked up" observation inside the relevant time window.
- **"Do I have any appointments today / this week?"** → Schedule agent queries the appointment store and answers with a warm summary of the day's or week's events.
- **"What is my routine for this morning?"** → Schedule agent retrieves the patient's daily routine blocks (e.g. "After breakfast you usually do your stretches, then read the paper") from the profile and answers contextually based on the current time of day.
- **"Who is this person?"** *(optional, stretch)* → People recognition agent cross-references a detected face or a voiced name against the family/contacts list in the profile.
- **"Where am I? What room is this?"** → Context/location agent infers the patient's likely location from recent camera observations (e.g. "You're in the kitchen — the kettle and fruit bowl are visible") and from the time of day.
- **"I need help" / "Call my daughter"** → Emergency agent surfaces the relevant contact number immediately (does not auto-dial unless trivial to wire up) and logs a help-request event.
- **Anything ambiguous or unrecognized** → General QA agent gives a gentle, honest "I'm not sure" rather than guessing.
- **Follow-up questions** → "Did I move them after lunch?" following "where are my glasses?" — the assistant resolves "them" to "glasses" from the prior turn, and filters the memory log to events after the reference time (lunch), rather than treating it as a fresh, context-free question.

## Proactive care loop
The assistant doesn't only wait to be asked. A background reminder loop monitors the full schedule against the memory log and clock, firing proactive spoken nudges across multiple reminder categories:

- **Medication reminders:** At each scheduled dose time (+ a configurable grace window, e.g. 15–20 min), the system checks whether a matching "medication taken" observation has been logged. If not, it proactively pushes a short, calm spoken reminder ("It's time for your morning medication — it's on the kitchen counter"). If the dose still isn't observed after a second grace window, this is logged as a missed-dose event and nudging stops for that dose — the assistant does **not** nag repeatedly or shame the patient, one gentle nudge plus one follow-up nudge only.
- **Appointment reminders:** A configurable lead time before each appointment (e.g. 30 min and 10 min), the system pushes a reminder ("Your doctor's appointment is in 30 minutes — do you need help getting ready?").
- **Daily routine reminders:** At the scheduled start time of each routine block (e.g. breakfast, morning walk, medication, afternoon rest), the system announces the next step if the patient hasn't yet asked about it.
- **Custom caregiver reminders:** Free-text, time-based reminders the caregiver can add ("Remind him to drink a glass of water at 3pm" or "Remind her that Sarah is visiting at 5pm").
- **Emergency / help reminders:** If the patient hasn't interacted with the device for an unusually long quiet period (configurable threshold), the system can send a gentle check-in prompt.

All proactive reminders reuse the same confidence machinery as reactive answers: a reminder is only suppressed (treated as "already handled") when the matching observation is high-confidence; a low-confidence/ambiguous observation still triggers the reminder, erring toward reminding rather than silently assuming compliance.

## Confidence-aware answers, with visible reasoning
Every answer the assistant gives is paired with an internal confidence level (high / medium / low) **and a short reasoning trace explaining why**, based on how directly the memory log, profile, or schedule supports it:
- **High** — an exact, recent, unambiguous match (e.g. one matching object/event logged in the last hour, or a medication-taken observation inside the scheduled window, or a calendar event confirmed within seconds).
- **Medium** — a match exists but is old, or there were multiple plausible matches and the most recent one was picked.
- **Low** — no direct match; the answer is inferred or the assistant genuinely doesn't know.

Every agent response internally returns `{answer_text, confidence, reasoning}` — the reasoning is a one-line explanation of *why* that confidence was assigned. This reasoning is:
- Logged and shown in the debug JSON / demo view so judges can see the assistant isn't guessing.
- Used to decide hedging language in the *spoken* answer — low-confidence answers are phrased with visible hedging ("I think they might be on the counter, but I'm not fully sure") rather than stated as fact.
- Used by the proactive reminder loop to decide whether to suppress or fire a reminder.

This matters both for demo credibility and for real dementia-care safety — a confidently wrong answer is worse than an honest "I don't know," and a caregiver reviewing logs later should be able to see *why* the assistant answered the way it did.

## Multi-turn context awareness
The assistant keeps a short rolling **session context** of the current conversation (not just the room memory log) so it can resolve pronouns, follow-ups, and chained time references within a session:
- Tracks: last subject/entity ("glasses"), last resolved location, last time-reference used ("after lunch" → an actual timestamp), and last intent (MEMORY/PROFILE/SCHEDULE/GENERAL/EMERGENCY).
- "Where are my glasses" → "did I move them after lunch" → "and before that?" all resolve against the same subject ("glasses") without the patient having to restate it, and each follow-up narrows or shifts the time window relative to the last resolved reference rather than starting over.
- Context is scoped to a session (e.g. resets after ~10 minutes of inactivity or a clearly new subject) so unrelated later questions don't get incorrectly anchored to an old topic.
- The confidence/reasoning trace above also applies here: if the assistant can't confidently resolve what "them" refers to, it says so rather than guessing at the wrong object.

## Data the caregiver provides (setup, once, before demo)
- **Patient:** name, age, relevant conditions, current medications (name + **schedule with explicit times**, no dosage-adjustment logic — just recall, never medical advice).
- **Family members:** name, relationship to patient, phone number, photo (optional, for stretch people-recognition).
- **Appointments:** date, time, description, location (e.g. "Dr. Singh, Thursday 10am, City Clinic").
- **Daily routines:** named blocks with start times (e.g. "Morning routine: 7:30am — wake up, breakfast, medications, 30-min walk").
- **Custom reminders:** free-text messages with trigger times (e.g. "At 3pm — remind to drink water").
- This is stored locally (not on the patient's device) and is only read by the appropriate agents — never by the perception/memory pipeline itself.

## What this project deliberately does NOT do
- Does not store or transmit video — only derived text observations.
- Does not give medical advice, dosing guidance, or diagnosis — it recalls stated facts and observed events only, and always defers to "ask your caregiver or doctor" for anything beyond simple recall/confirmation.
- Does not attempt emotion detection, fall detection, or health monitoring beyond the medication-observation check described above (out of scope — mention as future work only).
- Does not require cloud connectivity for the core demo — model runs locally via Ollama.
- Does not nag or repeat reminders indefinitely — capped at one nudge + one follow-up nudge per scheduled event.
- Does not auto-dial contacts or send messages without explicit human confirmation.

## MVP scope (must work for demo)
1. Camera observes 2–3 objects placed in view, including a medication bottle; perception agent logs them.
2. Patient presses button on M5StickC Plus2, asks "where is my [object]?", gets correct spoken/displayed answer.
3. Patient asks one general/profile question ("who is my daughter?"), gets correct answer.
4. Patient asks one everyday question ("what day is it? do I have any appointments today?"), gets a sensible answer from the schedule store.
5. Patient asks "have I taken my medication?" and gets an answer grounded in an actual observed event (or an honest "not yet"), not just a recitation of the schedule.
6. At least one live demo of the proactive reminder firing unprompted — either a medication reminder or an appointment reminder — when a scheduled window passes without a matching event.

## Stretch scope (only if ahead of schedule)
- **Caregiver-alert agent:** detects repeated questions (patient asking the same thing 3+ times) as a distress signal and logs an alert.
- **People recognition:** when the patient asks "who is this person?", cross-reference the spoken name or (stretch) a face detection signal against the family contacts list.
- **Location/context reminder:** at a configurable time or trigger (e.g. patient has been idle in the kitchen for 20+ minutes), gently remind them what they should be doing next.
- **Simple caregiver dashboard** (even just a terminal or a static HTML page) showing today's memory log, missed-dose events, and any repeated-question alerts.
- **Emergency flow:** patient says "I need help" or "call my daughter" → surfaces the contact number immediately, logs a help-request event.

## Success criteria for the demo
- End-to-end voice round trip works live, twice, without a rebuild.
- At least one memory-retrieval answer, one profile-based answer, and one schedule/appointment answer all work correctly on stage.
- The medication check-in answers from an actual observed event, and at least one proactive reminder (medication or appointment) fires live.
- Judges understand the "no video stored, only meaning stored" privacy angle — this is the pitch's strongest differentiator, say it explicitly.

## Tech stack summary
See `architecture.md` §9 for the full, versioned tech stack and §8 for how Gemma is prompted/tuned for each agent role. See `architecture.md` §10 for the multi-agent system design. See `roadmap.md` for the parallel two-developer build plan.
