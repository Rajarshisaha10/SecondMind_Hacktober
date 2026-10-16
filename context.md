# Second Mind — Context

A single-page reference for anyone joining the project (or re-opening it after a break). Read this first, then `project.md` for product details, `architecture.md` for technical design, and `roadmap.md` for the build plan.

---

## Project overview

**Second Mind** is an ambient memory assistant for people with dementia. A room camera passively observes the environment, and a wearable device (M5StickC Plus2) lets the patient ask natural spoken questions and receive calm spoken answers. The assistant also proactively reminds the patient about medications, appointments, daily routines, and custom caregiver reminders — without the patient having to ask.

**Core privacy principle:** No video is ever stored. The camera's output is a stream of short structured text observations ("reading glasses placed on kitchen counter, 3:42pm"). Only these derived facts persist. This is the project's strongest pitch differentiator.

**Target user:** A patient with dementia living at home, supported by a caregiver who sets up the system once in advance.

---

## Architecture

The system is built as a **multi-agent pipeline** running entirely on a laptop. All inference is local via Ollama. No cloud connectivity is required for the demo.

### Components

| Agent / Component | Role |
|---|---|
| Perception agent | Polls webcam, calls Gemma vision, writes text observations to memory store |
| Orchestrator agent | Receives transcript, resolves session context, classifies intent, dispatches to the right agent |
| Memory retrieval agent | Handles "where is X" and medication-taken queries (hybrid: memory log + profile) |
| Schedule agent | Answers appointment, routine, and "what should I do today" questions |
| General QA agent | Answers family/profile questions and general everyday questions |
| Emergency agent | Handles "I need help" / "call X" — surfaces contacts, logs help-request event |
| Reminder agent | Background timer loop — fires medication, appointment, routine, and custom reminders proactively |
| STT | faster-whisper: audio → transcript |
| TTS | pyttsx3 (offline): text → spoken audio |
| M5StickC Plus2 | Wearable device: hold-to-talk button, mic, speaker, Wi-Fi, idle reminder polling |

### Three independent loops

1. **Passive loop** — Perception agent continuously writes observations to memory store. No user interaction.
2. **Active loop** — Patient presses button → audio → STT → orchestrator → agent → TTS → spoken answer on device.
3. **Proactive loop** — Reminder agent checks schedule and memory store on a timer; queues reminders; device polls and plays them.

### Intent categories (orchestrator output)

`MEMORY` → `MEDICATION` → `SCHEDULE` → `PROFILE` → `EMERGENCY` → `GENERAL`

Each maps to a specific agent (see `architecture.md` §10 for full dispatch table and flow).

### Confidence model

Every agent response returns `{answer_text, confidence, reasoning}`. Confidence (`high` / `medium` / `low`) is **computed deterministically in code**, not by the LLM. The LLM only phrases the reasoning string and the spoken answer. This prevents hallucinated confidence levels and makes the system auditable.

---

## Technologies

| Layer | Technology | Why |
|---|---|---|
| LLM inference | Ollama (local) | No cloud dependency, easy model swapping |
| Model | `gemma3:4b` via Ollama | Single pull; already multimodal — covers perception (vision) and all text agents |
| Backend | FastAPI (Python 3.11+) | Simple async HTTP, background threads, easy to test |
| STT | faster-whisper (`base`/`small`) | Local, low latency |
| TTS | pyttsx3 (+ gTTS fallback) | Fully offline; gTTS as backup if network available |
| Camera | OpenCV | Webcam capture + frame-diff |
| Storage | SQLite (preferred) or flat JSON | Simple, zero-dependency, queryable |
| Wearable | M5StickC Plus2 | Mic, speaker, button, Wi-Fi in one compact device |
| Firmware | Arduino (M5StickCPlus2 library) or UIFlow | Hold-to-talk + reminder polling loop |
| Transport | Local Wi-Fi, plain HTTP | Demo-safe; server IP hardcoded |

---

## Directory structure

```
second-mind/
  backend/
    main.py                  # FastAPI app — mounts all routes, starts background threads
    perception.py            # Webcam loop + Gemma vision calls → memory store
    agents/
      orchestrator.py        # Session-context resolver + intent classifier + dispatcher
      memory_agent.py        # Object queries + medication-taken hybrid check
      schedule_agent.py      # Appointment, routine, and daily-plan queries
      qa_agent.py            # Profile and general everyday questions
      emergency_agent.py     # Help / contact-surfacing intents
      reminder_agent.py      # Background timer: all proactive reminder types
    stores/
      memory_store.py        # Read/write for observation log (object + medication events)
      profile_store.py       # Read-only for caregiver-entered patient profile
      schedule_store.py      # Read-only for appointments, routines, custom reminders
      session_store.py       # In-memory per-session context (expires after ~10 min)
    stt.py                   # faster-whisper wrapper: bytes → transcript string
    tts.py                   # pyttsx3 wrapper: text string → WAV bytes / audio file
  firmware/
    second_mind_stick/       # Arduino/UIFlow project for M5StickC Plus2
  profile_setup.json         # Caregiver-entered patient profile (gitignored)
  schedule_setup.json        # Caregiver-entered schedule data (gitignored)
  project.md                 # Product overview, user flows, scope
  architecture.md            # Technical design: components, schemas, APIs, agent prompts
  roadmap.md                 # Two-track build plan, milestones, integration checkpoints
  context.md                 # This file — project summary for new contributors
```

Track A (Developer A) owns `backend/agents/`, `backend/stores/`, `backend/main.py`, `backend/stt.py`, `backend/tts.py`.
Track B (Developer B) owns `backend/perception.py`, `firmware/`.
Both may touch config files (`profile_setup.json`, `schedule_setup.json`, `README.md`) — coordinate verbally before editing.

---

## Coding conventions

- **Python 3.11+** throughout the backend.
- **All agent functions return a typed dict** matching `AgentResponse = {answer_text: str, confidence: str, reasoning: str}`. Never return raw strings from an agent.
- **Confidence is computed in code**, not by the LLM. Use the deterministic rules in `architecture.md` §5. The LLM only generates `reasoning` (a one-line explanation string) and `answer_text`.
- **Store reads are synchronous and direct** — agents read from stores in-process; no inter-agent HTTP calls.
- **No hardcoded delays** — use the `grace_window_minutes` field from the profile/schedule store, not magic numbers in agent code.
- **JSON validation after every LLM call** — if the model returns malformed JSON, retry once with a stricter prompt suffix, then fall back to a safe default. Never let a malformed LLM response crash the server.
- **One Ollama Modelfile per agent role** — system prompt and temperature baked in. Temperature: `0.0–0.2` for classification/extraction; `0.5–0.7` for spoken-answer phrasing.
- **Gitignore `profile_setup.json` and `schedule_setup.json`** — these contain patient data and must not be committed. Provide `profile_setup.example.json` and `schedule_setup.example.json` with anonymized placeholder data.
- **Test text pipelines before wiring audio** — use `curl` or pytest scripts to confirm all six intent routes work correctly before connecting STT/TTS or the M5StickC.

---

## Assumptions

- **Single patient, single device** — session management is a fixed in-memory dict keyed by a constant device ID. No multi-user session handling required.
- **Single room** — the camera observes one room. Location references in observations are relative to that room (e.g. "kitchen counter", "dining table"). No floor-plan or multi-room awareness.
- **Caregiver sets up the system before the demo** — `profile_setup.json` and `schedule_setup.json` are hand-edited (or filled via a minimal form if time allows). No in-session patient-driven configuration.
- **Demo network is a known local Wi-Fi** — server IP is hardcoded. No mDNS, no DHCP discovery, no captive portal.
- **No dosage adjustment or medical advice** — the assistant only recalls what is on file and what it has observed. It always defers to "ask your caregiver or doctor" for anything beyond simple fact recall.
- **No video stored** — the perception agent only calls Gemma vision on frames and stores the resulting text observations. No JPEG or video file is persisted.
- **Ollama is running locally on Laptop A** — both agents and perception share a single Ollama instance. Laptop B's perception loop can POST observations to Laptop A's server, or run a second Ollama instance locally if latency is a concern.

---

## Important design decisions

### Multi-agent over monolithic prompt
A single prompt handling all intents would create compounding failure modes and ambiguous confidence computation. Specialized agents give one clear failure mode per capability, allow independent testing, and make the system extensible — adding a new capability means adding one new agent file and one new intent category in the orchestrator.

### Confidence computed in code, not by the LLM
LLMs can hallucinate confidence levels that do not reflect the actual evidence. Confidence is computed deterministically from timestamps and store matches. The LLM is only asked to explain that confidence in natural language and phrase the spoken answer — it cannot override the computed level.

### MEDICATION always routes to Memory retrieval agent, never to General QA
General QA only knows the medication *schedule* (from the profile). It has no access to the memory log and cannot verify whether a dose was actually taken. Routing "have I taken my medication?" to General QA would risk a false "yes" based on the schedule alone — a safety risk in dementia care.

### EMERGENCY intent dispatches before any other logic
Emergency intents (`"I need help"`, `"call my daughter"`) skip the normal confidence/retrieval pipeline and go directly to the Emergency agent. Speed matters more than reasoning depth here.

### Hold-to-talk over wake word
A wake word adds a false-trigger failure mode and significant build time. Hold-to-talk is physically intuitive for the target patient population ("press while you talk"), reliable, and implementable in hours.

### One reminder slot, not a queue
For the demo, the reminder agent maintains a single pending slot rather than a priority queue. This is intentional — if a dose window fires while an appointment reminder is pending, the most recent takes priority. A production system would use a proper queue; the single-slot model is explicitly a demo simplification.

### Proactive reminders fire at most twice per event
One gentle nudge + one follow-up nudge per scheduled event (medication dose, appointment, routine block, custom reminder). The assistant never nags. After two nudges with no confirmation, the event is logged as missed and nudging stops.

### No cloud dependency for the core demo
All inference (Gemma via Ollama), STT (faster-whisper), TTS (pyttsx3), and storage (SQLite) run locally. This is both a privacy feature and a demo-resilience feature (venue Wi-Fi may not have internet access).

---

## Current roadmap summary

The build is structured as two parallel tracks across 6 hours, with four integration checkpoints. See `roadmap.md` for full task lists, ownership, milestones, and the merge strategy.

| Checkpoint | ~When | Event |
|---|---|---|
| CP1 | 0:20 | Schema agreement + environment confirmed on both laptops |
| CP2 | 1:30 | Agents route text-only test calls; device produces WAV |
| CP3 | 3:30 | First full spoken round trip over Wi-Fi |
| CP4 | 5:00 | Demo dry run — full script passes twice |

**MVP must-haves (never cut):**
- Object location query answered from memory log.
- Medication-taken query answered from memory log + profile (not schedule alone).
- Appointment/schedule query answered from schedule store.
- Family profile query answered from profile store.
- Proactive reminder fires unprompted (at least one type: medication or appointment).
- Orchestrator correctly routes all six intent categories.

---

## Future work

Items explicitly out of scope for the current build, noted here to inform future development:

- **Caregiver dashboard** — web or terminal UI showing today's memory log, missed-dose events, repeated-question alerts, and appointment status.
- **Caregiver alert agent** — detects when the patient asks the same question 3+ times in a session (a distress signal) and logs/notifies the caregiver.
- **People recognition** — cross-reference a spoken name or (stretch) a face detection signal against the family contacts list in the profile.
- **Multi-room awareness** — extend the perception agent to tag observations with room context if multiple cameras are available.
- **Emotion detection / fall detection** — future health-monitoring capabilities; currently out of scope and not attempted.
- **Auto-dialing** — the Emergency agent surfaces contact numbers but does not auto-dial. A future implementation could integrate with a SIP client or mobile notification.
- **Fine-tuning** — a small LoRA fine-tune of the orchestrator's classification model on domain-specific examples. Currently implemented via few-shot prompting only.
- **Mobile caregiver app** — a companion app for the caregiver to update the profile, schedule, and custom reminders without hand-editing JSON.
- **Multi-patient support** — the current session model assumes a single patient and a single device. Multi-patient support would require proper session management and device authentication.
- **Cloud backup / sync** — privacy-preserving encrypted backup of the observation log and schedule store for caregivers managing remote patients.
