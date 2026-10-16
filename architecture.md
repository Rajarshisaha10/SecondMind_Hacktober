# Second Mind — Architecture

Companion to `project.md`. This is the technical design: components, data flow, schemas, APIs, agent prompting/training strategy, tech stack, and the M5StickC firmware behavior. Written so an engineer (or Claude Code) can start implementing directly from this file.

---

## 1. System components

| Component | Runs on | Responsibility |
|---|---|---|
| Perception agent | Laptop | Polls webcam, describes scene via Gemma vision, writes structured facts to memory store |
| Memory store | Laptop (SQLite/JSON) | Append-only log of object/location/time/event observations, including medication-related events |
| Patient profile store | Laptop (SQLite/JSON) | Caregiver-entered static facts: patient info, medications **with schedule times**, family contacts, appointments, daily routines, custom reminders |
| Schedule store | Laptop (SQLite/JSON) | Structured upcoming events: appointments, routine blocks, custom caregiver reminders — queried by the Schedule agent and Reminder agent |
| Session context store | Laptop (in-memory / SQLite) | Rolling short-term conversation context (last subject, last time reference, last intent) for multi-turn resolution |
| Orchestrator agent | Laptop (FastAPI) | Receives transcribed question, resolves it against session context, classifies intent, routes to the right agent |
| Memory retrieval agent | Laptop | Answers "where is X" questions **and** medication-taken questions (hybrid: profile schedule + memory log evidence) |
| Schedule agent | Laptop | Answers appointment, routine, and "what should I do today" questions from the schedule store |
| General QA agent | Laptop | Answers everyday questions and profile-only questions (family, date/time, medication *schedule* facts with no evidence lookup) |
| Reminder agent | Laptop (background loop) | Periodically compares all scheduled events (medication doses, appointments, routines, custom reminders) against the clock and memory log; queues proactive spoken reminders |
| Emergency agent | Laptop | Handles "I need help" / "call X" intents — surfaces contact info immediately and logs a help-request event |
| STT | Laptop (faster-whisper) | Converts incoming audio to text |
| TTS | Laptop (pyttsx3/gTTS) | Converts agent's text answer (reactive or proactive) to audio |
| M5StickC Plus2 firmware | Device | Button-triggered wake + record, sends audio to laptop, plays back response, polls for proactive reminders while idle |

---

## 2. High-level data flow

```
[Camera] --frame every 5-10s--> [Perception agent] --JSON facts--> [Memory store]

[M5StickC button press] --record audio--> [POST /ask (audio)] --> [STT]
   --> transcript text --> [Orchestrator agent] --reads/writes--> [Session context]
        --classify--> "MEMORY"      --> [Memory retrieval agent] --reads--> [Memory store]
        --classify--> "MEDICATION"  --> [Memory retrieval agent] --reads--> [Memory store] + [Patient profile store]
        --classify--> "SCHEDULE"    --> [Schedule agent]         --reads--> [Schedule store]
        --classify--> "PROFILE"     --> [General QA agent]       --reads--> [Patient profile store]
        --classify--> "GENERAL"     --> [General QA agent]       (no lookup, direct answer)
        --classify--> "EMERGENCY"   --> [Emergency agent]        --reads--> [Patient profile store]
   --> {answer_text, confidence, reasoning} --> [TTS] --> audio response --> [M5StickC speaker]

[Reminder agent, every N min] --reads--> [Patient profile store] + [Schedule store] (all scheduled events)
   --compares against clock + [Memory store] (observed events)
   --if window passed with no match--> queues reminder --> [GET /reminder/check, polled by device] --> [TTS] --> [M5StickC speaker]
```

Three independent loops:
- **Passive loop** (perception → memory store) runs continuously, no user interaction.
- **Active loop** (button press → answer) is request/response, triggered by the patient.
- **Proactive loop** (reminder agent → queued push, device polls and plays) runs on a timer, not triggered by the patient at all.

---

## 3. Backend API (FastAPI on laptop)

### `POST /ask`
Triggered by the M5StickC on button release.
- Request: multipart audio file (WAV, 16kHz mono recommended) or raw bytes with a known sample rate header.
- Server pipeline: STT → orchestrator (with session context) → agent → TTS.
- Response: audio file (WAV/MP3) the M5Stick plays directly. Also return the transcript + answer + confidence + reasoning as headers or a small JSON sidecar for debugging/demo logging.

```json
// Debug response shape (if returning JSON instead of raw audio for testing)
{
  "transcript": "have i taken my medication",
  "intent": "MEDICATION",
  "answer_text": "Yes — I saw you pick up your morning pills around 8:10, a little after your 8am dose time.",
  "confidence": "high",
  "reasoning": "medication_taken event matched at 08:10, 10 min after scheduled 08:00 dose (grace window 20 min)",
  "audio_url": "/audio/answer_1234.wav"
}
```

### `GET /health`
Simple liveness check for demo-day debugging — confirms Ollama, memory store, profile store, schedule store, and the reminder loop thread are all reachable/running. Build this early; it saves you during the actual demo.

### `GET /reminder/check`
Polled by the M5StickC periodically (e.g. every 20–30s) while idle/not actively recording.
- Response: `{ "pending": false }` if nothing queued, or `{ "pending": true, "audio_url": "/audio/reminder_5678.wav", "message": "...", "reminder_type": "medication|appointment|routine|custom|emergency" }` if a reminder is waiting.
- Device plays it and then calls `POST /reminder/ack` (or the server clears the queue once served) so the same reminder isn't replayed on the next poll.
- Kept as a separate lightweight endpoint (not folded into `/ask`) so it's cheap enough to poll frequently without taxing STT/TTS.

### `POST /reminder/ack`
Device calls this after successfully playing a queued reminder, so the server can clear it and start the "second nudge" grace window instead of re-queuing immediately.

### Internal-only (not device-facing, used by perception loop and reminder loop)
Perception agent writes directly to the memory store, and the reminder agent reads profile + schedule + memory stores directly — neither needs to route through the device-facing API layer. Keep both as background threads/processes in the same FastAPI app for simplicity.

---

## 4. Data schemas

### Memory store (one row per observation)
```json
{
  "id": "uuid",
  "timestamp": "2026-07-29T15:42:00",
  "object": "reading glasses",
  "location": "kitchen counter",
  "action": "placed",
  "event_type": "object_observation",
  "raw_description": "A pair of glasses was set down near the fruit bowl on the counter."
}
```
Medication events use the same table with a distinguishing `event_type`, so the Memory retrieval agent and the Reminder agent can query on that field directly without keyword-guessing:
```json
{
  "id": "uuid",
  "timestamp": "2026-07-29T08:10:00",
  "object": "medication bottle",
  "location": "kitchen counter",
  "action": "picked up",
  "event_type": "medication_taken",
  "raw_description": "A pill bottle was picked up and appears to have been opened."
}
```
Keep `raw_description` for fallback/debugging even though `object`/`location`/`action`/`event_type` are what retrieval queries against.

### Patient profile store (set up once, by caregiver, before the demo)
```json
{
  "patient": {
    "name": "string",
    "age": 0,
    "conditions": ["string"],
    "medications": [
      {
        "name": "string",
        "schedule_times": ["08:00", "20:00"],
        "grace_window_minutes": 20
      }
    ]
  },
  "family": [
    { "name": "string", "relationship": "string", "phone": "string" }
  ]
}
```

### Schedule store (caregiver-entered, read by Schedule agent and Reminder agent)
```json
{
  "appointments": [
    {
      "id": "uuid",
      "title": "Dr. Singh — annual checkup",
      "datetime": "2026-07-31T10:00:00",
      "location": "City Clinic, 2nd floor",
      "reminder_lead_minutes": [30, 10]
    }
  ],
  "routines": [
    {
      "id": "uuid",
      "name": "Morning routine",
      "days": ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"],
      "start_time": "07:30",
      "steps": ["Wake up", "Breakfast", "Take medications", "30-minute walk"]
    }
  ],
  "custom_reminders": [
    {
      "id": "uuid",
      "message": "Drink a glass of water",
      "trigger_time": "15:00",
      "days": ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"],
      "repeat": true
    }
  ]
}
```

### Session context (per active conversation, short-lived)
```json
{
  "session_id": "uuid",
  "last_subject": "reading glasses",
  "last_resolved_time_ref": "2026-07-29T13:00:00",
  "last_intent": "MEMORY",
  "updated_at": "2026-07-29T13:05:00"
}
```
In-memory dict keyed by a fixed device/session id is enough for the demo (single patient, single device) — no need for real session management. Expire/reset after ~10 minutes of inactivity.

### Reminder queue (in-memory, tiny)
```json
{
  "pending": true,
  "reminder_type": "medication|appointment|routine|custom",
  "label": "string",
  "scheduled_for": "08:00",
  "message": "string",
  "audio_url": "string"
}
```
One pending slot is enough for the demo; the reminder agent overwrites/clears it rather than stacking multiple reminders. For production, this would be a priority queue, but the single-slot model is sufficient for a live demonstration.

---

## 5. Agent design

### Perception agent (prompt sketch)
System prompt instructs Gemma to look at the frame and return **only** JSON, e.g.:
> "You are watching a room to help someone with memory loss. List any object that was just placed, moved, or is newly visible, with its location in the room, and an `event_type` of either `object_observation` or `medication_taken` (use `medication_taken` only if a pill bottle, pill organizer, or medication packaging was picked up, opened, or clearly interacted with). Respond only with a JSON array of {object, location, action, event_type}. If nothing notable changed, respond with an empty array."

Run this on a diffed/changed frame if possible (simple frame-diff check before calling the model) to avoid spamming the memory store with duplicate "nothing changed" entries and to save inference time.

### Session context resolver (part of orchestrator)
Before classification, the orchestrator does a lightweight resolution pass using the session context:
> "Given the prior conversation subject '{last_subject}' and last resolved time reference '{last_resolved_time_ref}', rewrite this follow-up question into a fully explicit, standalone question. If it isn't a follow-up, return it unchanged. Question: '{transcript}'"

The rewritten, standalone question is what actually gets classified and routed — this keeps every downstream agent prompt simple (it never has to reason about pronouns itself).

### Orchestrator agent (prompt sketch)
One cheap classification call, run on the *resolved* question:
> "Classify the following question into exactly one category: MEMORY (asking where an object is, or about a past observed event that isn't medication), MEDICATION (asking whether/when medication was taken, or general medication status), SCHEDULE (asking about appointments, today's plan, daily routines, or upcoming events), PROFILE (asking about family or personal facts, or the medication *schedule* only, with no 'did I take it' component), EMERGENCY (patient says 'I need help', 'call X', or similar distress phrase), or GENERAL (everyday question like date, time, or small talk). Respond with only the category word.\nQuestion: {transcript}"

MEDICATION always routes to the Memory retrieval agent (not General QA) because only the memory log can confirm whether something actually happened, and a wrong "yes you took it" from schedule alone is a safety risk. EMERGENCY always routes to the Emergency agent immediately, bypassing all other routing logic.

### Memory retrieval agent (prompt sketch)
For ordinary object queries:
> "Given this log of recent observations: {relevant entries}. Answer the question '{transcript}' in one short, warm sentence. If nothing matches, say gently that you don't know yet. Also output a confidence level (high/medium/low) and a one-line reasoning string explaining that confidence."

For MEDICATION intent specifically, the agent is given both sources:
> "The patient's medication schedule is: {medication name + schedule_times + grace_window_minutes from profile}. Recent `medication_taken` events from the observation log are: {matching entries, if any}. Determine whether the most recent relevant scheduled dose has a matching observed event within its grace window. Answer '{transcript}' in one short, warm, non-alarming sentence — confirm if there's a match, say gently that you haven't seen it yet if not, and never imply a missed dose is dangerous. Output confidence (high/medium/low) and a one-line reasoning string stating exactly which schedule time and which (if any) observation you matched."

Retrieval itself (both cases) is a simple keyword/field match against `object`/`event_type` before even calling the model — only call Gemma to phrase the final sentence naturally and to produce the reasoning string; the confidence level itself is computed deterministically in code (see below), not left to the model to invent.

**Confidence is computed in code, not by the LLM**, to keep it trustworthy:
- HIGH: exactly one matching event inside the expected time window (< 1hr old for objects; inside grace window for medication).
- MEDIUM: a match exists but is stale, or multiple candidates existed and the most recent was chosen.
- LOW: no matching event found at all.
The model only phrases the natural-language reasoning string and the spoken answer around that pre-computed level — it does not get to override it.

### Schedule agent (prompt sketch)
Answers appointment, routine, and "what should I do" questions from the schedule store:
> "You are a calm, patient voice assistant for someone with memory difficulties. The current date and time are {datetime}. The patient's upcoming appointments are: {appointment list}. Their daily routines are: {routine list}. Their custom reminders for today are: {custom reminder list}. Answer the question '{transcript}' in two or three short, warm sentences. Prioritize events that are happening today or soon. Never fabricate events not in the list."

Confidence for schedule answers is always HIGH when the event is directly in the store, MEDIUM when inferred from a routine (e.g. "you usually do X around now"), and LOW if no relevant event is found.

### General QA agent (prompt sketch)
> "You are a calm, patient voice assistant for someone with memory difficulties. Use short, simple, reassuring sentences. Never give medical advice beyond recalling stated facts — if asked what medication is on file or when it's scheduled, state only what's on file; if asked whether it's been *taken*, do not answer — that question is handled elsewhere. Patient profile: {profile JSON, if the question needs it}. Question: {transcript}"

Only pass the profile JSON into the prompt when the orchestrator classified the question as PROFILE or GENERAL, to keep prompts short and avoid the model leaking family contact info into unrelated answers. General QA never receives the memory store.

### Emergency agent (prompt sketch)
Handles EMERGENCY-classified intents immediately, before any other lookup:
> "The patient has said something that may be a distress call or request for help. The relevant contacts are: {family list with phone numbers}. Respond in one very short, calm sentence identifying the requested contact and their number. Always end with 'I'm here with you.' Do not suggest calling emergency services unless no family contact is relevant."

The emergency agent also writes a `help_request` event to the memory store (for caregiver log visibility) and sets a flag so the proactive reminder agent can note unusual silence patterns around the time of the request.

### Reminder agent (background loop)
Not triggered by a question — runs on a timer (e.g. every 5 minutes):
1. **Medication:** For each medication in the profile, compute the most recent scheduled `schedule_times` entry that has already passed today. Query the memory store for a `medication_taken` event inside `[scheduled_time, scheduled_time + grace_window_minutes]`. If none found and no reminder has already fired for that dose, generate a reminder using a fixed template and queue it.
2. **Appointments:** For each appointment in the schedule store, check whether the current time is within any of the configured `reminder_lead_minutes` windows. If so and no reminder has fired yet for this appointment + lead-time slot, queue an appointment reminder.
3. **Routines:** At the start time of each daily routine block, if the patient has not already asked about the routine in the current session, queue a gentle routine prompt.
4. **Custom reminders:** At the trigger time of each custom reminder, if not already fired today (or if `repeat: true` and the window is fresh), queue the caregiver's custom message.
5. All queued reminders use a fixed template (no LLM call needed — keep it deterministic and fast). Synthesize the audio via TTS ahead of time.
6. Mark each fired reminder so it doesn't repeat until the next valid window. Never queue more than one reminder of the same type per window per scheduled event — one gentle nudge plus one follow-up nudge only.

---

## 6. M5StickC Plus2 firmware behavior

**Interaction model: hold-to-talk on Button A, plus idle polling for reminders.**
- Idle state: device is low-power, screen shows a simple "press and hold to ask" icon. While idle (not recording, not mid-request), it polls `GET /reminder/check` every 20–30s.
- **Button A press (down)**: wake device, show a "listening" icon, start recording from the mic. Pauses reminder polling until the request/response cycle completes.
- **Button A release (up)**: stop recording, show a "thinking" icon, send the recorded WAV to the laptop's `/ask` endpoint over Wi-Fi (HTTP POST).
- **On response received**: show a "speaking" icon, play the returned audio through the speaker.
- **On reminder poll returning `pending: true`**: show a "reminder" icon, play the reminder audio, then call `POST /reminder/ack`.
- **On error/timeout** (no Wi-Fi, server unreachable): show a clear "sorry, try again" icon/tone rather than failing silently — important for demo resilience and for real dementia-friendly UX (silence reads as the device being broken).
- Button B (optional stretch): could double as a fixed "help" button — one press sends a canned "I need help" event straight to the orchestrator, skipping STT.

**Why hold-to-talk over a wake word:** far more reliable for a 6-hour build, avoids false triggers, and is easier for a patient to understand physically ("press while you talk") than a voice trigger.

**Firmware stack:** Arduino (M5StickCPlus2 library) or UIFlow, whichever the team is faster in. Keep the Wi-Fi credentials and server IP hardcoded for the demo — do not build a captive-portal config flow, it's not worth the time.

---

## 7. Suggested repo structure

```
second-mind/
  backend/
    main.py              # FastAPI app: /ask, /health, /reminder/check, /reminder/ack, perception + reminder loop startup
    perception.py        # camera polling + Gemma vision calls
    agents/
      orchestrator.py     # session-context resolution + intent classification
      memory_agent.py     # object queries + medication-taken hybrid check
      schedule_agent.py   # appointment, routine, and "what today" queries
      qa_agent.py         # general and profile questions
      reminder_agent.py   # background timer loop (all reminder types)
      emergency_agent.py  # help/contact-surfacing intent handler
    stores/
      memory_store.py      # SQLite/JSON read-write for observations (object + medication events)
      profile_store.py     # SQLite/JSON read-write for patient profile (incl. schedule_times)
      schedule_store.py    # SQLite/JSON read-write for appointments, routines, custom reminders
      session_store.py     # in-memory session context
    stt.py                 # faster-whisper wrapper
    tts.py                 # pyttsx3/gTTS wrapper
  firmware/
    second_mind_stick/    # Arduino/UIFlow project for M5StickC Plus2 (ask flow + reminder polling)
  profile_setup.json        # caregiver-entered patient profile (gitignored/demo-only)
  schedule_setup.json       # caregiver-entered appointments/routines/custom reminders (gitignored/demo-only)
  project.md
  architecture.md
  roadmap.md
  context.md
```

---

## 8. Model usage & prompting strategy ("training" Gemma)

Given the 6-hour build window, there is no fine-tuning of Gemma's weights — "training" here means **prompt/system-message engineering plus deterministic scaffolding around the model**, which is both faster to build and safer for a dementia-care use case (a fine-tuned model's behavior is harder to audit live on stage than a fixed prompt + code-computed confidence).

- **Local serving:** Ollama running `gemma3:4b` locally. The 4B variant is already multimodal (vision + text), so a single `ollama pull gemma3:4b` covers both the perception agent (camera frames) and all text agents (orchestrator, memory, QA, schedule, emergency). No second model pull is required.
- **Per-role system prompts via Ollama Modelfiles:** each agent (perception, orchestrator, memory retrieval, schedule, medication check, general QA, emergency) gets its own `Modelfile` baking in its system prompt and a fixed low temperature (0–0.2 for classification/extraction, slightly higher for the final spoken phrasing) so behavior is consistent across demo runs.
- **Few-shot examples, not fine-tuning:** each Modelfile/system prompt includes 3–5 worked examples (a real transcript → correct JSON or correct category) directly in the prompt. This is the actual "training" happening — in-context, not weight updates — and is enough for a narrow, well-scoped task like intent classification or structured JSON extraction.
- **Structured output enforced by prompt + validation, not by the model alone:** every JSON-returning call (perception, classification) is validated in code after the call; on a malformed response, retry once with a stricter reminder appended to the prompt, then fall back to a safe default (e.g. `event_type: object_observation`, `intent: GENERAL`) rather than crashing the demo.
- **Confidence is never model-generated** (see §5) — it's computed deterministically from timestamps/matches in code. The model only explains it in natural language. This avoids the model "hallucinating" a confidence level that doesn't match the actual evidence.
- **Stretch, only if far ahead of schedule:** a very small LoRA fine-tune of the orchestrator's classification prompt on a hand-written set of ~50 example questions, if the team wants to demonstrate real fine-tuning rather than just prompting — explicitly optional, cut first if time is short (it adds negligible accuracy over good few-shot prompting at this scope).

---

## 9. Tech stack (proper/full list)

| Layer | Choice | Notes |
|---|---|---|
| LLM runtime | Ollama (local) | No cloud dependency for the core demo |
| Model | `gemma3:4b` via Ollama | Single pull; the 4B variant is multimodal — covers perception (vision) and all text agents |
| Backend framework | FastAPI (Python 3.11+) | `/ask`, `/health`, `/reminder/check`, `/reminder/ack`; background threads for perception + reminder loops |
| STT | faster-whisper (local, `small`/`base` model for latency) | Audio in → transcript |
| TTS | pyttsx3 (offline) with gTTS as a fallback if network is available at the venue | Text answers/reminders → audio |
| Camera capture | OpenCV | Webcam polling + simple frame-diff |
| Memory store | SQLite (preferred) or flat JSON | Append-only observation log, queried by `object`/`event_type`/`timestamp` |
| Profile store | SQLite or flat JSON | Caregiver-entered, includes `schedule_times`/`grace_window_minutes` |
| Schedule store | SQLite or flat JSON | Appointments, routines, custom reminders — queried by Schedule agent and Reminder agent |
| Session context store | In-process Python dict (or SQLite table) | Single-session, short TTL, no need for Redis at this scale |
| Wearable | M5StickC Plus2 | Mic, speaker, buttons, Wi-Fi |
| Firmware | Arduino (M5StickCPlus2 library) or UIFlow | Hold-to-talk + idle reminder polling |
| Networking | Local Wi-Fi, HTTP (no HTTPS needed for a local-network demo) | Server IP hardcoded for the demo |

---

## 10. Multi-agent system design

This section explains the architectural decision to use multiple specialized agents rather than a single general-purpose LLM call, and describes how the agents coordinate.

### Why multiple agents?

A single monolithic prompt would need to handle object retrieval, medication verification, appointment lookup, schedule reasoning, profile facts, and emergency triage simultaneously. This creates compounding failure modes: the model must attend to many data sources at once, confidence computation becomes ambiguous, and a single prompt failure kills all functionality. Specialized agents solve this by:

- **Separation of concerns** — each agent has one clear job, one data source, and one failure mode. Debugging a bad schedule answer doesn't require reasoning about medication logic.
- **Latency control** — the orchestrator classification call is cheap (a small Gemma text model). Heavy retrieval and phrasing only happen in the agent that actually needs it. The reminder agent runs completely off the request path.
- **Deterministic scaffolding** — confidence is computed in code, not by a monolithic LLM. The orchestrator resolves intent before any expensive call is made.
- **Independent extensibility** — a new reminder type (e.g. hydration check) or a new agent (e.g. face recognition) can be added by implementing one new module and registering one new intent category in the orchestrator. No other agent changes.

### How Gemma powers the agents

All agents share a single local Ollama instance running one model: **`gemma3:4b`**. The 4B variant is already multimodal — it handles both image input (for the Perception agent) and text tasks (for every other agent) from a single pull. No separate vision or text variant is needed.

Each agent is assigned its own **Ollama Modelfile** with a baked-in system prompt and fixed temperature. This means the same model weights serve every role — the specialization is entirely in the prompt context, not in separate model instances. This keeps resource usage low (one Ollama process, one model loaded) while providing the full benefit of role-specific behavior.

### Responsibilities of each agent

| Agent | Gemma call? | Primary data source | Output |
|---|---|---|---|
| Perception | Yes (vision) | Webcam frame | JSON observation array written to memory store |
| Orchestrator | Yes (text, classification) | Transcript + session context | Intent category string (MEMORY / MEDICATION / SCHEDULE / PROFILE / EMERGENCY / GENERAL) |
| Memory retrieval | Yes (text, phrasing) | Memory store + (for MEDICATION) profile store | `{answer_text, confidence, reasoning}` |
| Schedule | Yes (text, phrasing) | Schedule store | `{answer_text, confidence, reasoning}` |
| General QA | Yes (text, phrasing) | Profile store or no store | `{answer_text, confidence, reasoning}` |
| Emergency | Yes (text, phrasing) | Profile store (contacts) | `{answer_text, confidence, reasoning}` + writes `help_request` event |
| Reminder | No (deterministic templates) | Profile store + Schedule store + memory store | Queued reminder audio via TTS |

### Communication between agents

Agents do **not** call each other directly. All coordination goes through shared stores and the orchestrator:

```
Orchestrator → (intent string) → routes call to appropriate agent function
Agent         → (reads) → appropriate store(s)
Agent         → (returns) → {answer_text, confidence, reasoning} to orchestrator
Orchestrator  → (passes answer) → TTS → device
Reminder agent → (writes pending slot) → Reminder queue → device polls → TTS → device
Perception agent → (writes rows) → Memory store → read by Memory retrieval agent and Reminder agent
```

There are no synchronous inter-agent calls. The orchestrator's routing function is a simple dispatch table — it calls the appropriate agent function directly in process (not over a network), receives its response struct, and passes it to TTS. This keeps the stack simple and avoids network-latency overhead between agents that are all running on the same laptop.

### Orchestration flow (step by step)

1. **Device sends audio** to `POST /ask`.
2. **STT** converts audio to transcript text.
3. **Session context resolver** (inside orchestrator) rewrites the transcript into a fully explicit, standalone question if it contains pronouns or time references resolved from the session.
4. **Orchestrator classification call** (`gemma3:4b`, low temperature) returns one of: MEMORY / MEDICATION / SCHEDULE / PROFILE / EMERGENCY / GENERAL.
5. **Intent dispatch** — orchestrator calls the matching agent function in-process:
   - MEMORY → `memory_agent.answer_object_query(transcript, memory_store)`
   - MEDICATION → `memory_agent.answer_medication_query(transcript, memory_store, profile_store)`
   - SCHEDULE → `schedule_agent.answer(transcript, schedule_store, current_datetime)`
   - PROFILE → `qa_agent.answer(transcript, profile_store)`
   - EMERGENCY → `emergency_agent.answer(transcript, profile_store, memory_store)`
   - GENERAL → `qa_agent.answer(transcript, None)`
6. **Agent returns** `{answer_text, confidence, reasoning}`.
7. **Orchestrator updates** session context with the new subject, intent, and time reference extracted from this turn.
8. **TTS** synthesizes `answer_text` and the response is returned to the device.

The reminder agent runs on a completely independent timer thread, writing to the reminder queue without touching the orchestrator. The device polls the queue separately via `GET /reminder/check`.

### Adding a new agent

To add a new capability (e.g. a hydration-check agent, a face-recognition agent, or a caregiver-alert agent):

1. **Create** `backend/agents/new_agent.py` with a function matching the signature `answer(transcript, *stores) -> AgentResponse`.
2. **Add** a new intent category to the orchestrator's classification prompt (one new word and one worked example in the few-shot list).
3. **Register** the new category in the orchestrator's dispatch table.
4. **Add** any new store the agent needs to `backend/stores/` and to the `GET /health` check.
5. **Optionally** create a dedicated Modelfile for the new agent if its system prompt differs substantially from existing agents.

No other agent is modified. The reminder agent can optionally be extended to watch for new event types by adding a new check inside its timer loop.

---

## 11. Offline/failure fallbacks (for demo-day resilience)

- If Ollama vision calls are too slow for live polling, pre-seed the memory store with 2–3 staged observations before the demo (including at least one `medication_taken` event) and let live perception be a bonus, not the critical path.
- If M5StickC Wi-Fi is flaky at the venue, have a laptop-mic fallback path (`/ask` accepts direct upload/local mic input too) so the core agent logic can still be demoed. The reminder loop can also fall back to logging to console/laptop TTS instead of the device.
- If STT is unreliable with ambient noise, pre-test the exact demo phrases against the model beforehand and keep them short and simple, including the medication check-in phrase.
- If the proactive reminder timing is awkward to trigger live on demand, shrink `grace_window_minutes` and set a schedule time a minute or two into the demo so it fires naturally on stage, rather than faking it.
- If the schedule store is empty at demo time, the Schedule agent falls back gracefully: it tells the patient what time it is and acknowledges it has no appointments on file, rather than crashing.
