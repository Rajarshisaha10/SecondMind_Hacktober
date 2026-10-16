# Second Mind — Development Roadmap

Companion to `project.md` and `architecture.md`. Two parallel development tracks so each developer can work independently on a separate laptop with minimal merge conflicts. Tracks integrate at defined checkpoints. A cut list is provided if either track falls behind.

---

## Parallel track ownership

| Track | Owner | Laptop | Primary concern |
|---|---|---|---|
| **Track A — Backend, agents & stores** | Developer A | Laptop A | FastAPI server, all agents, all stores, STT/TTS wrappers, reminder loop |
| **Track B — Firmware, hardware & audio pipeline** | Developer B | Laptop B | M5StickC firmware, Wi-Fi transport, audio record/playback, camera perception loop |

Both developers clone the same repo. Each track operates on separate directories (`backend/` vs `firmware/` and `perception.py`) that do not overlap, minimizing merge conflicts. Integration happens at four explicit checkpoints (marked below) where both developers merge to `main` and sync.

---

## Shared interfaces (agree before coding — ~0:20)

These are the only contracts both tracks must honour. Agree on them at the kickoff checkpoint and do not change them unilaterally.

### 1. `POST /ask` request/response
- Request: multipart WAV, 16kHz mono.
- Response: WAV audio file. Debug headers: `X-Transcript`, `X-Intent`, `X-Confidence`, `X-Reasoning`.

### 2. `GET /reminder/check` response shape
```json
{ "pending": false }
// or
{
  "pending": true,
  "audio_url": "/audio/reminder_5678.wav",
  "message": "It's time for your morning medication.",
  "reminder_type": "medication|appointment|routine|custom"
}
```

### 3. `POST /reminder/ack` — no body, 200 OK on success.

### 4. `GET /health` — 200 JSON with `{ "ollama": true, "memory_store": true, "schedule_store": true, "reminder_loop": true }`.

### 5. Server address — hardcoded for demo: `http://192.168.x.x:8000`. Set once and document in `README.md`.

### 6. Audio format — WAV, 16kHz mono for uploads; WAV or MP3 acceptable for playback responses.

---

## Integration checkpoints

| # | When | What both developers do | Done when |
|---|---|---|---|
| **CP1** | ~0:20 | Agree schemas + interfaces. Track A: FastAPI `/health` is green. Track B: dev environment confirmed, M5StickC plugged in, Ollama test call succeeds. | `/health` returns 200; Ollama responds to a test prompt on both laptops. |
| **CP2** | ~1:30 | Merge to `main`. Track A: orchestrator routes text-only test calls correctly. Track B: pressing M5StickC button produces a saved WAV on Laptop B. | Text test calls route correctly; WAV file produced and audible. |
| **CP3** | ~3:30 | Merge to `main`. First full end-to-end round trip. Track A connects STT+TTS to `/ask`; Track B sends real device audio to Track A's server. | A real spoken question produces a spoken answer back on the device. |
| **CP4** | ~5:00 | Merge to `main`. Demo dry run. Both tracks run the complete demo script together. Fix what breaks. | Demo script completes twice in a row without a restart. |

---

## Milestones

### Milestone 1 — Setup & schemas (0:00 – 0:30) · both tracks
- [ ] Pull and test `gemma3:4b` via Ollama on Laptop A (`ollama pull gemma3:4b`). The 4B variant is multimodal — one pull covers both vision (perception agent) and text (all other agents). Verify with a text classification call and a vision call before writing any agent code.
- [ ] Scaffold FastAPI backend with `/health` endpoint (Track A).
- [ ] Agree on all data schemas (`architecture.md` §4) and shared interface shapes (above) — write them down before either track writes code.
- [ ] Caregiver `profile_setup.json` drafted with placeholder demo data (patient name, 1–2 conditions, 1–2 medications, 2 family members with phone numbers).
- [ ] `schedule_setup.json` drafted with 1 appointment (30 min from demo time), 1 daily routine, and 1 custom reminder.
- [ ] **CP1 sync.**

---

### Milestone 2A — Stores & agent skeletons (0:30 – 1:30) · Track A

- [ ] `memory_store.py` — SQLite with `object`, `location`, `action`, `event_type`, `timestamp` columns. Seed with 2–3 demo observations.
- [ ] `profile_store.py` — reads `profile_setup.json`. Returns patient, medications (with schedule_times), family contacts.
- [ ] `schedule_store.py` — reads `schedule_setup.json`. Returns appointments, routines, custom reminders.
- [ ] `session_store.py` — in-memory dict, expires after 10 min inactivity.
- [ ] `orchestrator.py` — classification prompt wired; returns MEMORY / MEDICATION / SCHEDULE / PROFILE / EMERGENCY / GENERAL for hardcoded test inputs.
- [ ] `memory_agent.py` stub — keyword match on memory store, no Gemma phrasing yet.
- [ ] `qa_agent.py` stub — returns family member data from profile.
- [ ] `schedule_agent.py` stub — returns appointment list from schedule store.

**Done when:** `python -m pytest tests/test_routing.py` passes for all six intent categories (text in, correct agent stub called).

---

### Milestone 2B — Camera + perception loop (0:30 – 1:30) · Track B

> Starts after CP1 confirms the memory store schema is agreed.

- [ ] `perception.py` — OpenCV webcam loop, frame every 5–10s, simple frame-diff to skip unchanged frames.
- [ ] Gemma vision call — sends JPEG frame to Ollama multimodal, parses JSON array response.
- [ ] Writes valid rows to `memory_store` on Laptop A (use the agreed schema exactly).
- [ ] M5StickC dev environment confirmed (Arduino IDE or UIFlow, M5StickCPlus2 library installed).

**Done when:** placing an object in front of the webcam on Laptop B produces a correct row in the memory store (on Laptop A, or locally if stores not yet shared) within ~15s.

---

### **CP2 merge** (~1:30) — merge both branches to `main`

---

### Milestone 3A — Agents solidified (1:30 – 2:30) · Track A

- [ ] `memory_agent.py` — full Gemma phrasing call, confidence computed in code, returns `{answer_text, confidence, reasoning}`.
- [ ] `memory_agent.py` medication hybrid — pulls schedule from profile store, checks memory store for `medication_taken` events in grace window.
- [ ] `schedule_agent.py` — full Gemma phrasing call for appointment + routine answers.
- [ ] `qa_agent.py` — profile and general QA answers with appropriate hedging.
- [ ] `emergency_agent.py` — surfaces contact from profile, writes `help_request` to memory store.
- [ ] `reminder_agent.py` — background timer, handles medication + appointment + routine + custom reminder types. Queues to reminder slot.
- [ ] Session context resolver in `orchestrator.py` — rewrites pronouns/follow-ups to standalone questions.
- [ ] `stt.py` — faster-whisper wrapper, accepts WAV bytes, returns transcript string.
- [ ] `tts.py` — pyttsx3 wrapper, accepts text string, returns WAV bytes or writes audio file.

**Done when:** `curl`/test-script questions for all six intent types return correct `{answer_text, confidence, reasoning}` without touching hardware. Reminder fires on a test timer event.

---

### Milestone 3B — M5StickC firmware (2:30 – 3:30) · Track B

- [ ] Button A hold-to-talk: press starts recording, release stops and POSTs WAV to `/ask`.
- [ ] Receives audio response and plays it on speaker.
- [ ] Idle reminder polling: `GET /reminder/check` every 20–30s; if `pending: true`, plays audio and calls `POST /reminder/ack`.
- [ ] On-screen state icons: listening / thinking / speaking / reminder / error.
- [ ] Error/timeout handling: shows "sorry, try again" icon and tone on Wi-Fi failure.
- [ ] (Stretch) Button B: sends canned "I need help" POST to `/ask`, skipping STT.

**Done when:** pressing the button on the device produces a saved, audible WAV file on Laptop A, and a pre-synthesized reminder audio plays on the device when the poll returns `pending: true`.

---

### **CP3 merge** (~3:30) — merge both branches to `main`

### Milestone 4 — Full round trip (3:30 – 4:30) · both tracks together

- [ ] Wire `stt.py` + `tts.py` into `main.py`'s `/ask` handler — device audio in → transcript → agent → spoken audio out.
- [ ] Confirm round-trip latency is acceptable live (target < 5s end-to-end for a short question).
- [ ] Reminder agent fires a medication reminder live and device plays it without the patient pressing anything.
- [ ] Test appointment reminder firing.
- [ ] Confirm session context: ask "where are my glasses" then "did I move them after lunch" — resolves correctly.

**Done when:** press button, speak a question, hear a correct spoken answer through the device speaker. All six intent types confirmed live.

---

### Milestone 5 — Integration hardening (4:30 – 5:00) · both tracks

- [ ] Run the full demo scenario 3+ times back to back; fix stale connections, store state, audio buffering issues.
- [ ] Tune agent system prompts for tone: short, warm, simple sentences. Especially QA and emergency agents.
- [ ] Confirm `/health` endpoint shows all systems green.
- [ ] If ahead: caregiver-alert stub (repeated-question counter), or minimal caregiver HTML dashboard.

**Done when:** the exact demo script works 3 times in a row without a restart.

---

### **CP4 merge** (~5:00)

---

### Milestone 6 — Demo prep (5:00 – 6:00) · both tracks

- [ ] Write down the exact demo script: which objects get placed, exact phrasing of each question, expected answers for each of the six intent categories.
- [ ] Set appointment reminder 10–15 min from demo start time so it fires naturally on stage.
- [ ] Shrink `grace_window_minutes` if needed so the medication reminder fires comfortably on stage.
- [ ] Prepare the one-line pitch: *"No video is ever stored — only what the AI understood about the room, so the patient's privacy is protected while their memory is supported."*
- [ ] Charge the M5StickC; confirm Wi-Fi at the venue; test laptop-mic fallback path (see `architecture.md` §11).
- [ ] Two people independently run the full demo successfully.

---

## Merge strategy

- Each track works on a dedicated branch: `track-a` (Developer A) and `track-b` (Developer B).
- **No force-pushes to `main`**. All merges go through a pull request, even during the hackathon, so conflicts are visible.
- **Non-overlapping directories** keep merge conflicts rare:
  - Track A owns: `backend/agents/`, `backend/stores/`, `backend/main.py`, `backend/stt.py`, `backend/tts.py`
  - Track B owns: `backend/perception.py`, `firmware/`
  - Both may touch: `profile_setup.json`, `schedule_setup.json`, `README.md` — coordinate verbally before editing these shared files.
- At each numbered checkpoint, both developers commit their current state, open a PR into `main`, resolve any conflicts together (should be minimal given the directory split), and both pull the merged `main` before continuing.
- If a conflict does occur in a shared file, the rule is: Track A's version of backend logic wins; Track B's version of firmware/perception logic wins; for config files, merge manually and confirm both developers see the same schema.

---

## Cut list (in order, if either track falls behind)

1. **Caregiver-alert / repeated-question detection** — cut first, pure stretch.
2. **Caregiver dashboard UI** — cut second, not needed for the core pitch.
3. **People recognition / face lookup** — cut, depends on camera + profile integration that may not be ready.
4. **M5StickC audio playback** — fall back to the laptop speaking the answer aloud; device shows text or a status icon.
5. **M5StickC audio recording** — fall back to a laptop mic / push-to-talk key for the live demo.
6. **Live camera perception** — fall back to pre-seeded memory store entries, staged before the demo, with perception code shown as "this is what generated these, running live in the background."
7. **Appointment + routine + custom reminders** — if the reminder agent is running out of time, cut to medication-only reminders, which are the most demo-impactful.

**Never cut:** the orchestrator routing between memory / medication / schedule / profile / general — this is the core "multi-agent" story for judges, and it's the cheapest part of the system to keep working (it's a single classification call with a dispatch table).
