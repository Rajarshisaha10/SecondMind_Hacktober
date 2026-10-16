# Second Mind — Ambient Memory Assistant for Dementia Care

[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-1.0.0-009688.svg)](https://fastapi.tiangolo.com/)
[![LLM](https://img.shields.io/badge/LLM-Gemma%203%204B%20%28Ollama%29-orange.svg)](https://ollama.com/)
[![Hardware](https://img.shields.io/badge/Hardware-M5StickC%20Plus2-red.svg)](https://docs.m5stack.com/en/core/m5stickc_plus2)
[![Database](https://img.shields.io/badge/Storage-SQLite-003B57.svg)](https://www.sqlite.org/)
[![Privacy First](https://img.shields.io/badge/Privacy-Zero%20Video%20Storage-success.svg)](#-core-privacy-principle)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **One-Liner:** An ambient memory assistant for individuals with dementia and mild cognitive impairment. A room camera passively observes environment facts without storing video, and a wearable device (M5StickC Plus2) enables natural spoken questions with calm audio answers while proactively delivering non-intrusive medication, appointment, and routine reminders.

---

## 📋 Table of Contents
- [✨ Key Features & Vision](#-key-features--vision)
- [🔒 Core Privacy Principle](#-core-privacy-principle)
- [🏗️ System Architecture & Three Independent Loops](#️-system-architecture--three-independent-loops)
- [🧠 Multi-Agent Engine & Confidence System](#-multi-agent-engine--confidence-system)
- [🖥️ Caregiver Web Setup & Control Dashboard](#️-caregiver-web-setup--control-dashboard)
- [⌚ Wearable Hardware & Firmware (M5StickC Plus2)](#-wearable-hardware--firmware-m5stickc-plus2)
- [💾 Data Schemas & Database Architecture](#-data-schemas--database-architecture)
- [📡 Complete REST API Reference](#-complete-rest-api-reference)
- [📁 Repository Structure](#-repository-structure)
- [🚀 Installation & Quickstart Guide](#-installation--quickstart-guide)
- [🧪 Testing & Verification](#-testing--verification)
- [🎬 Hackathon Demo Scenario Script](#-hackathon-demo-scenario-script)
- [🛡️ Safety, Ethics & Privacy Guidelines](#️-safety-ethics--privacy-guidelines)
- [📄 License](#-license)

---

## ✨ Key Features & Vision

Living with dementia or Mild Cognitive Impairment (MCI) creates daily frustration around misplaced items, missed medication, lost routines, and anxiety. **Second Mind** acts as an ambient cognitive safety net that preserves patient dignity and independence while reducing caregiver stress.

- **Passive Camera Perception**: Rooms are passively observed using OpenCV. Visual scenes are analyzed in memory by `gemma3:4b` to track objects, locations, and medication events.
- **Wearable Natural Speech Interface**: The patient uses a lightweight M5StickC Plus2 wrist/pendant device with a simple "Hold-to-Talk" button.
- **Context-Aware Conversational Resolution**: Automatically resolves pronouns (*"them"*, *"it"*) and time references (*"this morning"*, *"after lunch"*) across continuous turns.
- **Deterministic Python Confidence Engine**: Computes exact `HIGH`, `MEDIUM`, or `LOW` confidence in Python code based on observation freshness and timestamp validation—preventing LLM hallucination.
- **Proactive Medication & Routine Reminders**: Background loop evaluates scheduled events against passive observation logs to gently nudge the patient only when a dose or routine was actually missed.
- **Emergency Instant-Triage**: Recognizes distress phrases (*"I need help"*, *"call my daughter"*) instantly surfacing emergency contacts and logging safety events.
- **Caregiver Control Center**: Web-based dashboard for setup, real-time memory log inspection, schedule management, and system diagnostics.

---

## 🔒 Core Privacy Principle

```
 ┌────────────────┐      ┌─────────────────────────┐      ┌────────────────────────┐
 │  Webcam Stream │ ───► │ In-Memory Frame Buffer  │ ───► │  OpenCV Frame Diffing  │
 └────────────────┘      └─────────────────────────┘      └───────────┬────────────┘
                                                                      │ (Frame Changed)
                                                                      ▼
 ┌────────────────┐      ┌─────────────────────────┐      ┌────────────────────────┐
 │   SQLite DB    │ ◄─── │  Structured Text Log    │ ◄─── │ Local Gemma 3 Vision   │
 │(second_mind.db)│      │ ("glasses on counter")  │      │ (In-memory analysis)   │
 └────────────────┘      └─────────────────────────┘      └────────────────────────┘
                                                              (Raw Frame Discarded)
```

**No video or images are ever recorded or stored.** 

1. Camera frames pass through an in-memory buffer analyzed by local OpenCV frame-diffing.
2. Only modified frames trigger local vision extraction via `gemma3:4b`.
3. Scene understanding converts visual frames into short, structured text facts (*e.g., "reading glasses placed on kitchen counter, 09:15"*).
4. Raw frames are **immediately discarded from memory**. Only text rows persist in the local `second_mind.db` SQLite database.

---

## 🏗️ System Architecture & Three Independent Loops

Second Mind runs entirely locally on laptop/edge hardware powered by **`gemma3:4b` via Ollama**.

```
                           ┌────────────────────────┐
                           │   Room Camera (OpenCV) │
                           └───────────┬────────────┘
                                       │ (Frame diff / poll)
                                       ▼
                           ┌────────────────────────┐
                           │   Perception Agent     │ (Track B - Passive Loop)
                           └───────────┬────────────┘
                                       │ (Write text observations)
                                       ▼
┌────────────────────────┐ ┌────────────────────────┐ ┌────────────────────────┐
│  Patient Profile Store │ │      Memory Store      │ │     Schedule Store     │
│ (profile_setup.json)   │ │   (second_mind.db)     │ │ (schedule_setup.json)  │
└───────────┬────────────┘ └───────────┬────────────┘ └───────────┬────────────┘
            │                          │                          │
            └───────────────────┐      │      ┌───────────────────┘
                                ▼      ▼      ▼
                           ┌────────────────────────┐
                           │   Multi-Agent Engine   │ (Track A - Edge Runtime)
                           │  - Orchestrator Agent  │
                           │  - Memory Agent        │
                           │  - Schedule Agent      │
                           │  - QA Agent            │
                           │  - Emergency Agent     │
                           │  - Reminder Agent      │
                           └───────────┬────────────┘
                                       │
            ┌──────────────────────────┴──────────────────────────┐
            ▼                                                     ▼
┌───────────────────────┐                             ┌───────────────────────┐
│ Active Question Loop  │                             │  Proactive Care Loop  │
│ (Button Press -> Ask) │                             │ (Idle Poll / Reminder)│
└───────────┬───────────┘                             └───────────┬───────────┘
            │                                                     │
            └──────────────────────────┬──────────────────────────┘
                                       ▼
                           ┌────────────────────────┐
                           │  M5StickC Plus2        │ (Track B - Wearable Device)
                           │  - Speaker / Mic       │
                           │  - Hold-to-Talk Button │
                           └────────────────────────┘
```

### The 3 System Loops

| Loop Name | Frequency | Trigger | Flow |
|---|---|---|---|
| **1. Passive Observation Loop** | Continuous (5–10s diff) | Room Motion / Camera | Webcam frame → Frame diff → `gemma3:4b` vision → `MemoryStore` (`second_mind.db`) |
| **2. Active Question Loop** | On Demand | Wearable Button Press | M5Stick Mic → Wi-Fi audio → STT → Orchestrator → Specialized Agent → Confidence Engine → TTS → M5Stick Speaker |
| **3. Proactive Care Loop** | Periodic (Every 5 min) | Background Timer Thread | `ReminderAgent` evaluates schedule vs memory log → Queues audio nudge → M5Stick idle polls `/reminder/check` → Plays audio → `POST /reminder/ack` |

---

## 🧠 Multi-Agent Engine & Confidence System

Rather than relying on a single monolithic LLM prompt, Second Mind utilizes **7 specialized autonomous agents** coordinated by a central Orchestrator.

```
                               ┌──────────────────────┐
                               │   Incoming Question  │
                               └──────────┬───────────┘
                                          │
                                          ▼
                               ┌──────────────────────┐
                               │  Session Context     │ (Resolves pronouns & time references)
                               │  Resolver            │
                               └──────────┬───────────┘
                                          │
                                          ▼
                               ┌──────────────────────┐
                               │  Orchestrator Agent  │ (Classifies 6 intent categories)
                               └──────────┬───────────┘
                                          │
         ┌──────────────┬─────────────────┼─────────────────┬──────────────┐
         ▼              ▼                 ▼                 ▼              ▼
   ┌──────────┐   ┌───────────┐    ┌─────────────┐   ┌───────────┐   ┌───────────┐
   │ MEMORY   │   │MEDICATION │    │  SCHEDULE   │   │  PROFILE  │   │ EMERGENCY │
   └────┬─────┘   └─────┬─────┘    └──────┬──────┘   └─────┬─────┘   └─────┬─────┘
        │               │                 │                │               │
        ▼               ▼                 ▼                ▼               ▼
 ┌──────────────┐ ┌──────────────┐ ┌─────────────┐ ┌───────────┐ ┌───────────┐
 │ MemoryAgent  │ │ MemoryAgent  │ │ScheduleAgent│ │  QAAgent  │ │ Emergency │
 │ (Obs log)    │ │(Log+Profile) │ │ (Schedule)  │ │ (Profile) │ │   Agent   │
 └──────┬───────┘ └──────┬───────┘ └──────┬──────┘ └─────┬─────┘ └─────┬─────┘
        │                │                │              │             │
        └────────────────┴────────┬───────┴──────────────┴─────────────┘
                                  ▼
                     ┌──────────────────────────┐
                     │   Python Confidence      │ (Calculates HIGH / MEDIUM / LOW
                     │   & Validation Engine    │  deterministically in code)
                     └────────────┬─────────────┘
                                  ▼
                     ┌──────────────────────────┐
                     │   TTS Speech Synthesis   │ (Generates WAV audio response)
                     └──────────────────────────┘
```

### Agent Roles & Descriptions

1. **`PerceptionAgent`**: Monitors room webcam feed, runs frame-diffing, extracts structured object & medication events via local Gemma Vision, and writes rows to `MemoryStore`.
2. **`OrchestratorAgent`**: Rewrites follow-up queries using rolling `SessionStore` context (resolving pronouns like *"it"* or *"them"*), then classifies intent into one of 6 categories: `MEMORY`, `MEDICATION`, `SCHEDULE`, `PROFILE`, `EMERGENCY`, or `GENERAL`.
3. **`MemoryAgent`**: Resolves object location queries (*"Where are my glasses?"*) and hybrid medication verification (*"Did I take my morning pills?"*) by cross-referencing `MemoryStore` observations against caregiver profile schedule times.
4. **`ScheduleAgent`**: Answers questions about appointments, routine daily steps, and custom caregiver reminders from `ScheduleStore`.
5. **`QAAgent`**: Answers profile-based family/medical facts and general everyday conversational questions with calm, reassuring phrasing.
6. **`EmergencyAgent`**: Fast-triages distress statements (*"I need help"*, *"call my son"*), immediately surfaces emergency phone numbers, logs a `help_request` event, and provides calming voice output.
7. **`ReminderAgent`**: Autonomous background timer thread that checks upcoming schedule events against observed memory logs to queue proactive spoken reminders.

### Deterministic Python Confidence Engine

To ensure safety in dementia care, **LLMs are never allowed to invent confidence scores**. Confidence is calculated deterministically in Python code:

- 🟢 **HIGH**: Exact match within expected time window (< 1 hour for objects; inside configured grace window for medication).
- 🟡 **MEDIUM**: Match exists but observation is stale, or multiple candidate locations exist.
- 🔴 **LOW**: No matching observation found in memory store.

---

## 🖥️ Caregiver Web Setup & Control Dashboard

Served locally at **`http://localhost:8000/dashboard`** (or `http://localhost:8000/`), the dashboard provides caregivers with complete administrative control.

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│ SECOND MIND — Caregiver Setup & Ambient Intelligence Control Center   [SYSTEM ONLINE]  │
├────────────────────────────────────────────────────────────────────────────────────────┤
│ [👤 Patient Profile]   [📅 Schedule & Routines]   [👁️ Live Memory Log]   [⚡ System Health] │
├────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                        │
│ ┌───────────────────────────────┐ ┌──────────────────────────────────────────────────┐ │
│ │ Patient Metadata              │ │ Medication Schedules                             │ │
│ │ Name: Arthur Pendelton        │ │ - Donepezil (5mg) @ 08:00 (Grace: 20 min)       │ │
│ │ Age: 76                       │ │ - Memantine (10mg) @ 20:00 (Grace: 30 min)      │ │
│ │ Conditions: MCI, Hypertension │ └──────────────────────────────────────────────────┘ │
│ └───────────────────────────────┘                                                      │
│ ┌────────────────────────────────────────────────────────────────────────────────────┐ │
│ │ Emergency Family Contacts                                                          │ │
│ │ - Likhita (Daughter) - 555-0199 [Primary]                                          │ │
│ └────────────────────────────────────────────────────────────────────────────────────┘ │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

### Dashboard Tabs
- **Patient Profile**: Edit patient name, age, medical conditions, medication dosing schedules, and family emergency contact numbers.
- **Schedule & Routines**: Manage upcoming appointments, structured daily routines (morning/evening steps), and custom caregiver reminders.
- **Live Memory Log**: Monitor real-time room observations captured by the perception loop with filtering by event type (`object_observation`, `medication_taken`, `help_request`).
- **System Diagnostics**: Real-time health status monitoring for Ollama connection, SQLite database, Whisper STT, pyttsx3 TTS, and active background threads.

---

## ⌚ Wearable Hardware & Firmware (M5StickC Plus2)

The patient interface is built on the **M5StickC Plus2** ESP32 wearable development board.

```
                         +------------------------+
                         |     M5StickC Plus2     |
                         |                        |
                         |   +----------------+   |
                         |   |  LCD Display   |   |
                         |   |                |   |
                         |   |   [LISTENING]  |   |
                         |   +----------------+   |
                         |                        |
                         |   (o) Mic   ( ) Speaker|
                         |                        |
                         |   [ Button A: HOLD ]   |
                         +-----------+------------+
                                     |
                                     ▼ (Wi-Fi HTTP)
                         +------------------------+
                         |   FastAPI Backend      |
                         +------------------------+
```

### Hardware Specs & Pinout
- **MCU**: ESP32-PICO-D4 (Wi-Fi & Bluetooth)
- **Screen**: 1.14 inch Color LCD (135x240)
- **Microphone**: SPM1423 PDM Microphone
- **Speaker**: Built-in I2S Buzzer / Speaker
- **Battery**: 200mAh Lithium Polymer
- **Buttons**: Button A (Main Hold-to-Talk front button), Button B (Side function button)

### Firmware Interaction Flow (`firmware/second_mind_stick/second_mind_stick.ino`)
1. **Idle State**: Screen displays *"Hold Button A to Ask"*. Polling loop queries `GET /reminder/check` every 25 seconds.
2. **Button Down**: Suspends polling, initiates microphone recording, screen displays 🔴 *"Listening..."*.
3. **Button Up**: Stops recording, posts recorded WAV audio to `POST /ask`, screen displays 🟡 *"Thinking..."*.
4. **Response Received**: Plays returned WAV answer over speaker, screen displays 🟢 *"Speaking..."*.
5. **Reminder Triggered**: On `GET /reminder/check` returning `pending: true`, downloads audio URL, plays reminder, and issues `POST /reminder/ack`.

---

## 💾 Data Schemas & Database Architecture

### SQLite Memory Store (`second_mind.db`)
Stores structured room observations logged by the passive perception loop.

```sql
CREATE TABLE IF NOT EXISTS observations (
    id TEXT PRIMARY KEY,
    timestamp TEXT NOT NULL,
    object TEXT NOT NULL,
    location TEXT NOT NULL,
    action TEXT NOT NULL,
    event_type TEXT NOT NULL,
    raw_description TEXT
);

CREATE INDEX idx_obs_event_type ON observations(event_type);
CREATE INDEX idx_obs_timestamp ON observations(timestamp);
```

**Observation Schema Example:**
```json
{
  "id": "c62a84b1-9b1e-4c7b-a19f-0987654321fe",
  "timestamp": "2026-07-30T09:15:00Z",
  "object": "reading glasses",
  "location": "kitchen counter",
  "action": "placed",
  "event_type": "object_observation",
  "raw_description": "A pair of reading glasses was set down near the fruit bowl."
}
```

### Caregiver Setup Files
- **`profile_setup.json`**: Patient details, medical facts, medication schedule times, and emergency contacts.
- **`schedule_setup.json`**: Caregiver-defined appointments, daily routine sequences, and custom recurring reminders.

---

## 📡 Complete REST API Reference

### 1. `GET /health`
Liveness check verifying subsystem readiness.
```json
{
  "status": "healthy",
  "ollama": "connected",
  "database": "connected",
  "whisper": "ready",
  "tts": "ready",
  "agents": 6,
  "uptime_seconds": 128.4
}
```

### 2. `POST /ask`
Primary endpoint for active questions (accepts multipart WAV audio file or JSON transcript body).
- **Query Params**: `return_json=true` (optional, for debugging)
- **Request Body (JSON option)**:
  ```json
  {
    "transcript": "Where are my reading glasses?",
    "device_id": "demo_device_01"
  }
  ```
- **Response Headers**:
  - `X-Transcript`: `Where are my reading glasses?`
  - `X-Intent`: `MEMORY`
  - `X-Confidence`: `high`
  - `X-Reasoning`: `Matched single recent observation on kitchen counter.`
  - `X-Answer-Text`: `Your reading glasses are on the kitchen counter.`

### 3. `GET /reminder/check`
Polled by wearable device every 20-30s while idle.
```json
{
  "pending": true,
  "audio_url": "/audio/reminder_1722345600.wav",
  "message": "It is time for your morning medication.",
  "reminder_type": "medication"
}
```

### 4. `POST /reminder/ack`
Acknowledges and clears currently pending reminder slot.
```json
{
  "status": "acknowledged",
  "cleared": true
}
```

### 5. `POST /observations`
Perception integration endpoint for Track B vision pipeline.
```json
[
  {
    "object": "medication bottle",
    "location": "small table next to bed",
    "action": "picked up and opened",
    "event_type": "medication_taken",
    "raw_description": "Pill bottle opened."
  }
]
```

### 6. Caregiver Setup REST APIs
- `GET /api/profile` | `POST /api/profile`: Retrieve or update patient profile data.
- `GET /api/schedule` | `POST /api/schedule`: Retrieve or update appointment and routine schedules.
- `GET /api/observations`: Fetch recent memory observations for dashboard UI.

---

## 📁 Repository Structure

```
SecondMind/
├── backend/
│   ├── main.py                  # FastAPI application entrypoint & routing
│   ├── config.py                # Pydantic environment configurations
│   ├── constants.py             # System enums (Intent, EventType, Confidence)
│   ├── llm_service.py           # Ollama client wrapper & low-level LLM calls
│   ├── logger.py                # Structured logging & request tracing
│   ├── ollama_client.py         # HTTP client communicating with Ollama API
│   ├── perception.py            # OpenCV webcam loop & Gemma Vision integration
│   ├── prompt_manager.py        # System prompt templates & few-shot examples
│   ├── schemas.py               # Pydantic response models & validation
│   ├── stt.py                   # Speech-to-Text wrapper (faster-whisper)
│   ├── tts.py                   # Text-to-Speech wrapper (pyttsx3)
│   ├── agents/
│   │   ├── orchestrator.py      # Session resolver & 6-category intent classifier
│   │   ├── memory_agent.py      # Object location & medication hybrid check
│   │   ├── schedule_agent.py    # Appointment, routine, & custom schedule queries
│   │   ├── qa_agent.py          # Profile facts & general conversational QA
│   │   ├── emergency_agent.py   # Emergency contact triage & help event logging
│   │   └── reminder_agent.py    # Background timer loop for proactive reminders
│   ├── services/
│   │   ├── ask_service.py       # Orchestrates STT -> Agent -> Confidence -> TTS
│   │   ├── health_service.py    # Subsystem readiness diagnostics
│   │   └── reminder_service.py # Lifecycle manager for background reminder thread
│   ├── stores/
│   │   ├── memory_store.py      # SQLite repository for room observations
│   │   ├── profile_store.py     # Caregiver patient profile storage repository
│   │   ├── schedule_store.py    # Caregiver appointment & routine schedule repository
│   │   └── session_store.py     # Thread-safe rolling conversation context store
│   └── tests/                   # Automated pytest suite (10 test modules)
│       ├── test_ask_endpoint.py
│       ├── test_confidence.py
│       ├── test_dashboard_api.py
│       ├── test_demo_scenario.py
│       ├── test_health.py
│       ├── test_reminder_agent.py
│       ├── test_reminder_endpoints.py
│       ├── test_routing.py
│       ├── test_session_context.py
│       └── test_stores.py
├── firmware/
│   └── second_mind_stick/
│       └── second_mind_stick.ino # M5StickC Plus2 Arduino ESP32 firmware
├── frontend/
│   ├── index.html               # Caregiver Setup Dashboard web app
│   ├── style.css                # Custom CSS styling (Neobrutalism design theme)
│   └── app.js                   # Dashboard logic & API synchronization
├── audio_cache/                 # Generated WAV audio cache directory
├── second_mind.db               # SQLite database file
├── profile_setup.example.json   # Template patient profile setup file
├── profile_setup.json           # Active caregiver patient profile setup
├── schedule_setup.example.json  # Template schedule setup file
├── schedule_setup.json          # Active caregiver schedule setup
├── architecture.md              # Technical design specification
├── project.md                   # Project requirements & vision document
└── README.md                    # Detailed documentation readme
```

---

## 🚀 Installation & Quickstart Guide

### 1. Prerequisites
- **Python 3.11+** installed on host machine.
- **Ollama** installed and running locally ([https://ollama.com](https://ollama.com)).
- Pull the multimodal **`gemma3:4b`** model:
  ```bash
  ollama pull gemma3:4b
  ```

### 2. Environment Setup & Dependencies
Clone the repository and install required Python packages:

```bash
git clone https://github.com/YourRepo/SecondMind.git
cd SecondMind

# Create virtual environment
python -m venv venv
# Activate virtual environment (Windows)
.\venv\Scripts\activate
# Activate virtual environment (Linux/macOS)
source venv/bin/activate

# Install dependencies from requirements.txt
pip install -r requirements.txt
```

### 3. Initialize Setup Configuration Files
Copy default template files to create local configuration settings:

```bash
cp profile_setup.example.json profile_setup.json
cp schedule_setup.example.json schedule_setup.json
```

### 4. Running the System

#### A. Launch the Core Backend & Caregiver Dashboard
Start the FastAPI application with Uvicorn:

```bash
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```

- **Caregiver Dashboard**: Open browser to `http://localhost:8000/dashboard` (or `http://localhost:8000/`)
- **Swagger Interactive API Docs**: Open browser to `http://localhost:8000/docs`
- **System Health Diagnostics**: `http://localhost:8000/health`

#### B. (Optional) Launch the Passive Camera Perception Loop
In a **second terminal**, run the webcam vision loop powered by Gemma 3:

```bash
python backend/perception.py
```

### 5. Flash M5StickC Plus2 Wearable Firmware
1. Open `firmware/second_mind_stick/second_mind_stick.ino` in Arduino IDE.
2. Install required libraries via Arduino Library Manager:
   - `M5StickCPlus2` (by M5Stack)
   - `ArduinoJson` (v6.x or v7.x supported)
3. Copy `firmware/second_mind_stick/config.h.example` to `config.h` (or customize directly):
   ```cpp
   const char* WIFI_SSID     = "YOUR_WIFI_SSID";
   const char* WIFI_PASSWORD = "YOUR_WIFI_PASSWORD";
   const char* SERVER_IP     = "192.168.1.100";  // Laptop IP running backend
   const int   SERVER_PORT   = 8000;
   ```
4. Select Board: **M5StickC-Plus2**, Port, and click **Upload**.

### 6. Environment Configuration Variables
You can optionally configure Second Mind settings via a `.env` file or environment variables:

| Variable | Default | Purpose |
|---|---|---|
| `SECOND_MIND_OLLAMA_MODEL` | `gemma3:4b` | Gemma model tag in Ollama |
| `SECOND_MIND_OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama HTTP endpoint |
| `SECOND_MIND_SESSION_TTL_MINUTES` | `10` | Conversation memory retention |
| `SECOND_MIND_STT_MODEL_SIZE` | `base` | faster-whisper model size |

---

## 🧪 Testing & Verification

Second Mind includes a comprehensive automated unit test suite covering stores, routing, confidence calculations, session resolution, reminder loops, and full demo scripts.

Run the test suite using `pytest`:

```bash
python -m pytest backend/tests/ -v
```

---

## 🎬 Hackathon Demo Scenario Script

Follow this step-by-step sequence for a complete live demonstration:

1. **Passive Observation Log**: Camera logs `reading glasses` placed on `kitchen counter` at 09:15, and `medication bottle` picked up at 08:10.
2. **Object Query**: Patient asks *"Where are my reading glasses?"* → `MEMORY` agent answers *"Your reading glasses are on the kitchen counter."*
3. **Follow-up Query**: Patient asks *"Did I move them after lunch?"* → Session resolver rewrites `"them"` to `"reading glasses"` → answers contextually.
4. **Schedule Query**: Patient asks *"Do I have any doctor appointments today?"* → `SCHEDULE` agent answers *"Dr. Singh checkup tomorrow at 10:00 AM."*
5. **Profile Query**: Patient asks *"Who is Sarah?"* → `PROFILE` agent answers *"Sarah is your daughter."*
6. **Medication Verification**: Patient asks *"Have I taken my morning medication?"* → `MEDICATION` agent checks schedule against `second_mind.db` → confirms dose taken at 08:00 AM.
7. **Proactive Reminder**: Background thread evaluates schedule against memory log → device idle polls `/reminder/check` → plays audio reminder → issues `/reminder/ack`.
8. **Emergency Trigger**: Patient says *"I need help, call my daughter!"* → `EMERGENCY` agent surfaces contact `555-0199`, logs `help_request` event, and provides calming voice output.

---

## 🛡️ Safety, Ethics & Privacy Guidelines

- **Privacy by Design**: Zero cloud dependencies. Video frames remain strictly ephemeral in local system RAM.
- **Caregiver Authority**: Medical schedules and emergency contact numbers are controlled exclusively by designated caregivers through the secure local Web Dashboard.
- **Dementia-Friendly Phrasing Guidelines**: All system prompt templates enforce warm, simple, short, non-alarming, and non-judgmental spoken responses.

---

## 📄 License

This project is licensed under the **MIT License**.
