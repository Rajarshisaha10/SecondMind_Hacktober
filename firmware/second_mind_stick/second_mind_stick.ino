/*
 * second_mind_stick.ino  —  Second Mind M5StickC Plus2 firmware (Track B)
 *
 * An ambient memory assistant for dementia patients.  The patient wears or
 * keeps nearby an M5StickC Plus2 that lets them ask the backend questions
 * (Button A) or request emergency help (Button B).  The firmware polls for
 * proactive reminders while idle, and recovers from Wi‑Fi disconnection.
 *
 * ── Buttons ─────────────────────────────────────────────────────────────────
 *   Button A  (hold to talk) — records while held, sends to POST /ask,
 *                               plays back the spoken answer.
 *   Button B  (emergency)    — sends a silent dummy WAV to POST /ask with
 *                               header X-Emergency: true (emergency agent).
 *
 * ── Backend endpoints ───────────────────────────────────────────────────────
 *   GET   /health            – health check (ollama, memory_store, etc.)
 *   POST  /ask               – audio Q&A (multipart/form-data, field "audio")
 *   GET   /reminder/check    – polls for pending reminders
 *   POST  /reminder/ack      – acknowledges a played reminder
 *
 * ── Required Arduino libraries ──────────────────────────────────────────────
 *   M5StickCPlus2            – board / LCD / mic / speaker / power
 *   WiFi                     – ESP32 Wi-Fi
 *   HTTPClient               – HTTP(S) client
 *   ArduinoJson 6.x          – JSON parsing for /reminder/check responses
 *
 * ── Behaviour summary ───────────────────────────────────────────────────────
 *   Boot   →  Connect Wi‑Fi  →  Health check  →  idle screen
 *   Idle   →  polls /reminder/check every 25 s
 *          →  if 5 min without any HTTP 200, runs Wi‑Fi watchdog
 *   Btn A  →  startRecording() while held  →  POST /ask  →  playAudio()
 *   Btn B  →  sendEmergencyToServer() (zeroed 0.5 s WAV + X-Emergency)
 *          →  /ask response played normally
 */

#include <M5StickCPlus2.h>
#include <WiFi.h>
#include <HTTPClient.h>
#include <ArduinoJson.h>

// ---------------------------------------------------------------------------
// Colour constants (fallback definitions if the library does not provide them)
// ---------------------------------------------------------------------------
#ifndef ORANGE
#define ORANGE 0xFD20
#endif
#ifndef NAVY
#define NAVY 0x000F
#endif
#ifndef DARKGREY
#define DARKGREY 0x4208
#endif

// ---------------------------------------------------------------------------
// Configuration — loaded from config.h or defaults below
// ---------------------------------------------------------------------------
#if __has_include("config.h")
#include "config.h"
#else
const char* WIFI_SSID     = "YOUR_WIFI_SSID";
const char* WIFI_PASSWORD = "YOUR_WIFI_PASSWORD";
const char* SERVER_IP     = "192.168.1.100";
const int   SERVER_PORT   = 8000;
#endif

const unsigned long WIFI_TIMEOUT_MS        = 20000;
const unsigned long LOOP_DELAY_MS          = 100;
const unsigned long REMINDER_POLL_INTERVAL_MS = 25000;
const unsigned long WATCHDOG_TIMEOUT_MS    = 300000;   // 5 minutes
const unsigned long WATCHDOG_RETRY_MS      = 30000;    // retry every 30 s

// Derived URL (assembled in setup())
String SERVER_BASE;

// ---------------------------------------------------------------------------
// Audio constants — 16 kHz mono 16-bit PCM
// ---------------------------------------------------------------------------
#define SAMPLE_RATE          16000
#define BITS_PER_SAMPLE      16
#define NUM_CHANNELS         1
#define BYTES_PER_SAMPLE     2
#define MAX_RECORDING_SEC    8
#define MAX_RECORDING_BYTES  (SAMPLE_RATE * BYTES_PER_SAMPLE * MAX_RECORDING_SEC)
#define RECORD_CHUNK_SAMPLES 256
#define EMERGENCY_SILENCE_BYTES  (SAMPLE_RATE / 2 * BYTES_PER_SAMPLE)  // 0.5 s
static_assert(sizeof(int16_t) == BYTES_PER_SAMPLE,
              "The WAV payload must use 16-bit signed PCM samples");

// The M5StickC Plus2 has an onboard digital PDM microphone. M5StickCPlus2
// owns the I2S setup, PDM decimation, DMA, clock divider, and conversion to
// signed little-endian 16-bit PCM. Do not add a second I2S driver or treat
// samples from M5.Mic.record() as 24/32-bit words.
//
// Enable this only for a diagnostic capture. It writes one framed raw PCM
// stream to USB serial before WAV framing or HTTP transmission. Capture at
// 921600 baud and remove the BEGIN/END marker lines from the saved file.
static constexpr bool ENABLE_RAW_PCM_SERIAL_DUMP = false;
static constexpr uint32_t SERIAL_DEBUG_BAUD = 921600;

// ---------------------------------------------------------------------------
// Global state
// ---------------------------------------------------------------------------
int16_t*      pcmBuffer        = nullptr;
uint32_t      recordingSamples = 0;
volatile bool isRecording      = false;
unsigned long lastPollTime     = 0;
unsigned long lastSuccessfulInteraction = 0;
bool          wasOffline       = false;

// ---------------------------------------------------------------------------
// Forward declarations
// ---------------------------------------------------------------------------
void playAudio(const uint8_t* data, size_t length);
void showIdleScreen();
void printCaptureDiagnostics(const int16_t* samples, uint32_t sampleCount);
void dumpRawPcmToSerial(const int16_t* samples, uint32_t sampleCount);

// ---------------------------------------------------------------------------
// Draw the idle screen (dark blue background, title, instruction, battery)
// ---------------------------------------------------------------------------
void showIdleScreen() {
  M5.Lcd.fillScreen(NAVY);

  M5.Lcd.setTextSize(2);
  M5.Lcd.setTextColor(WHITE);
  String title = "Second Mind";
  int16_t x = (M5.Lcd.width() - M5.Lcd.textWidth(title)) / 2;
  if (x < 0) x = 0;
  M5.Lcd.setCursor(x, 30);
  M5.Lcd.println(title);

  M5.Lcd.setTextSize(1);
  String inst = "Hold A to ask";
  x = (M5.Lcd.width() - M5.Lcd.textWidth(inst)) / 2;
  if (x < 0) x = 0;
  M5.Lcd.setCursor(x, 65);
  M5.Lcd.println(inst);

  int bat = M5.Power.getBatteryLevel();
  if (bat >= 0) {
    M5.Lcd.setTextColor(DARKGREY);
    String batStr = String(bat) + "%";
    x = (M5.Lcd.width() - M5.Lcd.textWidth(batStr)) / 2;
    if (x < 0) x = 0;
    M5.Lcd.setCursor(x, 90);
    M5.Lcd.println(batStr);
  }
}

// ---------------------------------------------------------------------------
// Helper — clear the screen, show a centred message, draw battery corner
// ---------------------------------------------------------------------------
void showStatus(String msg, uint16_t color) {
  M5.Lcd.fillScreen(BLACK);
  M5.Lcd.setTextColor(color);
  M5.Lcd.setTextSize(2);

  int16_t textW = M5.Lcd.textWidth(msg);
  int16_t x = (M5.Lcd.width() - textW) / 2;
  if (x < 0) x = 0;

  int16_t y = (M5.Lcd.height() - 20) / 2;
  if (y < 0) y = 0;

  M5.Lcd.setCursor(x, y);
  M5.Lcd.println(msg);

  int bat = M5.Power.getBatteryLevel();
  if (bat >= 0) {
    M5.Lcd.setTextSize(1);
    M5.Lcd.setTextColor(DARKGREY);
    char buf[8];
    snprintf(buf, sizeof(buf), "%d%%", bat);
    M5.Lcd.setCursor(M5.Lcd.width() - M5.Lcd.textWidth(buf) - 2, 2);
    M5.Lcd.print(buf);
  }
}

// ---------------------------------------------------------------------------
// Build a 44-byte WAV header for 16 kHz mono 16-bit PCM
// ---------------------------------------------------------------------------
void buildWavHeader(uint8_t* hdr, uint32_t dataBytes) {
  uint32_t fileSize  = 36 + dataBytes;
  uint16_t blockAlign = NUM_CHANNELS * BYTES_PER_SAMPLE;
  uint32_t byteRate  = SAMPLE_RATE * blockAlign;

  hdr[0]='R'; hdr[1]='I'; hdr[2]='F'; hdr[3]='F';
  hdr[4]=fileSize&0xFF; hdr[5]=(fileSize>>8)&0xFF;
  hdr[6]=(fileSize>>16)&0xFF; hdr[7]=(fileSize>>24)&0xFF;
  hdr[8]='W'; hdr[9]='A'; hdr[10]='V'; hdr[11]='E';

  hdr[12]='f'; hdr[13]='m'; hdr[14]='t'; hdr[15]=' ';
  hdr[16]=16; hdr[17]=0; hdr[18]=0; hdr[19]=0;
  hdr[20]=1; hdr[21]=0;
  hdr[22]=NUM_CHANNELS; hdr[23]=0;
  hdr[24]=SAMPLE_RATE&0xFF;        hdr[25]=(SAMPLE_RATE>>8)&0xFF;
  hdr[26]=(SAMPLE_RATE>>16)&0xFF;  hdr[27]=(SAMPLE_RATE>>24)&0xFF;
  hdr[28]=byteRate&0xFF;           hdr[29]=(byteRate>>8)&0xFF;
  hdr[30]=(byteRate>>16)&0xFF;     hdr[31]=(byteRate>>24)&0xFF;
  hdr[32]=blockAlign&0xFF; hdr[33]=(blockAlign>>8)&0xFF;
  hdr[34]=BITS_PER_SAMPLE;  hdr[35]=0;

  hdr[36]='d'; hdr[37]='a'; hdr[38]='t'; hdr[39]='a';
  hdr[40]=dataBytes&0xFF;       hdr[41]=(dataBytes>>8)&0xFF;
  hdr[42]=(dataBytes>>16)&0xFF; hdr[43]=(dataBytes>>24)&0xFF;
}

// ---------------------------------------------------------------------------
// Button A handler — record from built-in mic via M5.Mic
// ---------------------------------------------------------------------------
void startRecording() {
  showStatus("Listening...", GREEN);
  recordingSamples = 0;
  isRecording = true;

  // The board shares its audio/I2S hardware between speaker and microphone.
  // The official M5Stack microphone example ends the speaker before capture;
  // otherwise the microphone can be silent or return corrupted samples.
  M5.Speaker.end();
  if (!M5.Mic.begin()) {
    isRecording = false;
    M5.Speaker.begin();
    showStatus("Mic init failed", RED);
    Serial.println("Mic: M5.Mic.begin() failed");
    M5.Speaker.tone(500, 200);
    delay(1500);
    showIdleScreen();
    return;
  }

  const uint32_t maxSamples = MAX_RECORDING_SEC * SAMPLE_RATE;

  while (isRecording && recordingSamples < maxSamples) {
    uint32_t toRead = RECORD_CHUNK_SAMPLES;
    if (recordingSamples + toRead > maxSamples) {
      toRead = maxSamples - recordingSamples;
    }

    // record() takes a signed int16_t PCM destination and a sample count.
    // Passing a byte offset here was unsafe and could misalign/corrupt audio.
    bool ok = M5.Mic.record(pcmBuffer + recordingSamples,
                            toRead, SAMPLE_RATE);
    if (ok) {
      recordingSamples += toRead;
    } else {
      // Never advertise or transmit samples that the driver did not capture.
      Serial.println("Mic: record() did not return a complete PCM chunk");
    }

    M5.update();
    if (M5.BtnA.wasReleased()) {
      isRecording = false;
    }
  }

  M5.Mic.end();
  isRecording = false;
  M5.Speaker.begin();

  if (recordingSamples < SAMPLE_RATE / 100) {  // less than 10 ms
    showStatus("Too short", RED);
    M5.Speaker.tone(500, 200);
    delay(1500);
    showIdleScreen();
    return;
  }

  printCaptureDiagnostics(pcmBuffer, recordingSamples);
  if (ENABLE_RAW_PCM_SERIAL_DUMP) {
    dumpRawPcmToSerial(pcmBuffer, recordingSamples);
  }

  showStatus("Thinking...", YELLOW);
  sendAudioToServer();
}

// Print objective capture measurements before WAV framing. These identify a
// disconnected mic (near-zero RMS), clipping, and incorrect gain without
// altering the original PCM samples.
void printCaptureDiagnostics(const int16_t* samples, uint32_t sampleCount) {
  int16_t minSample = INT16_MAX;
  int16_t maxSample = INT16_MIN;
  uint32_t clipped = 0;
  int64_t sumSquares = 0;

  for (uint32_t i = 0; i < sampleCount; ++i) {
    const int16_t sample = samples[i];
    if (sample < minSample) minSample = sample;
    if (sample > maxSample) maxSample = sample;
    if (sample <= -32752 || sample >= 32751) ++clipped;
    const int32_t widened = sample;
    sumSquares += (int64_t)widened * widened;
  }

  const float rms = sqrtf((float)sumSquares / (float)sampleCount);
  const float durationSeconds = (float)sampleCount / (float)SAMPLE_RATE;
  const uint32_t pcmBytes = sampleCount * sizeof(int16_t);
  const float clippingPercent = 100.0f * (float)clipped / (float)sampleCount;

  Serial.printf(
      "Mic capture: rate=%u Hz, channels=%u, bits=%u, samples=%lu, bytes=%lu, "
      "duration=%.3f s, min=%d, max=%d, rms=%.1f, clipped=%lu (%.3f%%)\n",
      SAMPLE_RATE, NUM_CHANNELS, BITS_PER_SAMPLE,
      (unsigned long)sampleCount, (unsigned long)pcmBytes, durationSeconds,
      minSample, maxSample, rms, (unsigned long)clipped, clippingPercent);
}

// This is signed little-endian 16-bit mono PCM exactly as returned by
// M5.Mic.record(). A host-side capture of this stream isolates microphone
// corruption from WAV framing and Wi-Fi/HTTP transport.
void dumpRawPcmToSerial(const int16_t* samples, uint32_t sampleCount) {
  const uint32_t byteCount = sampleCount * sizeof(int16_t);
  Serial.printf("RAW_PCM_BEGIN rate=%u channels=%u bits=%u bytes=%lu\n",
                SAMPLE_RATE, NUM_CHANNELS, BITS_PER_SAMPLE,
                (unsigned long)byteCount);
  Serial.flush();
  Serial.write(reinterpret_cast<const uint8_t*>(samples), byteCount);
  Serial.flush();
  Serial.printf("\nRAW_PCM_END samples=%lu duration=%.3f\n",
                (unsigned long)sampleCount,
                (float)sampleCount / (float)SAMPLE_RATE);
}

// ---------------------------------------------------------------------------
// Core HTTP POST to /ask — builds multipart body from PCM data, sends it,
// plays the response WAV.  Returns true on success.
// ---------------------------------------------------------------------------
bool doPostAudio(const uint8_t* pcmData, uint32_t dataBytes, bool emergency) {
  uint8_t wavHdr[44];
  buildWavHeader(wavHdr, dataBytes);

  const char* boundary = "----SecondMindBoundary7MA4YWxk";
  String part1 = "--";
  part1 += boundary;
  part1 += "\r\nContent-Disposition: form-data; name=\"audio\"; filename=\"recording.wav\"\r\n";
  part1 += "Content-Type: audio/wav\r\n\r\n";

  String part2 = "\r\n--";
  part2 += boundary;
  part2 += "--\r\n";

  size_t totalSize = part1.length() + 44 + dataBytes + part2.length();
  uint8_t* body = (uint8_t*)malloc(totalSize);
  if (!body) {
    showStatus("Out of memory", RED);
    M5.Speaker.tone(500, 200);
    delay(1000);
    return false;
  }

  size_t off = 0;
  memcpy(body + off, part1.c_str(), part1.length());  off += part1.length();
  memcpy(body + off, wavHdr, 44);                     off += 44;
  memcpy(body + off, pcmData, dataBytes);              off += dataBytes;
  memcpy(body + off, part2.c_str(), part2.length());

  HTTPClient http;
  http.begin(SERVER_BASE + "/ask");
  http.addHeader("Content-Type", "multipart/form-data; boundary=" + String(boundary));
  if (emergency) {
    http.addHeader("X-Emergency", "true");
  }
  http.setTimeout(15000);

  int httpCode = http.POST(body, totalSize);
  free(body);

  if (httpCode == 200) {
    lastSuccessfulInteraction = millis();

    WiFiClient* stream = http.getStreamPtr();
    size_t respSize = http.getSize();

    if (respSize > 44 && respSize < 500000) {
      uint8_t* respData = (uint8_t*)malloc(respSize);
      if (respData) {
        size_t read = 0;
        while (read < respSize) {
          size_t n = stream->readBytes(respData + read, respSize - read);
          if (n == 0) break;
          read += n;
        }
        if (read == respSize) {
          http.end();
          playAudio(respData, respSize);
          free(respData);
          return true;
        }
        free(respData);
      }
    }
    showStatus("Bad response", RED);
  } else {
    showStatus("Try again", RED);
  }

  M5.Speaker.tone(500, 200);
  http.end();
  delay(1000);
  return false;
}

// ---------------------------------------------------------------------------
// POST /ask — normal (non-emergency) path, uses the recorded PCM buffer
// ---------------------------------------------------------------------------
void sendAudioToServer() {
  uint32_t dataBytes = recordingSamples * BYTES_PER_SAMPLE;
  doPostAudio(reinterpret_cast<const uint8_t*>(pcmBuffer), dataBytes, false);
  showIdleScreen();
}

// ---------------------------------------------------------------------------
// Button B emergency — send silent 0.5 s WAV with X-Emergency: true
// ---------------------------------------------------------------------------
void sendEmergencyToServer() {
  showStatus("Sending help...", RED);

  uint8_t* emgBuf = (uint8_t*)malloc(EMERGENCY_SILENCE_BYTES);
  if (!emgBuf) {
    showStatus("Out of memory", RED);
    M5.Speaker.tone(500, 200);
    delay(1500);
    showIdleScreen();
    return;
  }
  memset(emgBuf, 0, EMERGENCY_SILENCE_BYTES);

  doPostAudio(emgBuf, EMERGENCY_SILENCE_BYTES, true);
  free(emgBuf);

  showIdleScreen();
}

// ---------------------------------------------------------------------------
// Parse WAV response and play via M5.Speaker
// ---------------------------------------------------------------------------
void playAudio(const uint8_t* data, size_t length) {
  showStatus("Speaking...", BLUE);

  if (length < 44) return;

  uint32_t wavSampleRate = (uint32_t)data[24] | ((uint32_t)data[25] << 8) |
                           ((uint32_t)data[26] << 16) | ((uint32_t)data[27] << 24);
  if (wavSampleRate == 0) wavSampleRate = SAMPLE_RATE;

  size_t off = 12;
  uint32_t dataSize = 0;
  while (off + 8 <= length) {
    uint32_t cs = (uint32_t)data[off+4] | ((uint32_t)data[off+5] << 8) |
                  ((uint32_t)data[off+6] << 16) | ((uint32_t)data[off+7] << 24);
    if (data[off]=='d' && data[off+1]=='a' && data[off+2]=='t' && data[off+3]=='a') {
      dataSize = cs;
      off += 8;
      break;
    }
    off += 8 + cs;
  }
  if (dataSize == 0 || off + dataSize > length) {
    Serial.printf("Audio: bad WAV (dataSize=%u, length=%u)\n", dataSize, length);
    return;
  }

  M5.Speaker.begin();
  M5.Speaker.setVolume(200);
  M5.Speaker.playRaw(data + off, dataSize, wavSampleRate);
  while (M5.Speaker.isPlaying()) {
    M5.update();
    delay(10);
  }

  delay(500);
  showIdleScreen();
}

// ---------------------------------------------------------------------------
// Poll /reminder/check — called from loop() while idle
// ---------------------------------------------------------------------------
void checkForReminder() {
  HTTPClient http;
  http.begin(SERVER_BASE + "/reminder/check");
  http.setTimeout(5000);

  int httpCode = http.GET();
  if (httpCode != 200) {
    http.end();
    return;
  }

  lastSuccessfulInteraction = millis();

  String body = http.getString();
  http.end();

#if defined(ARDUINOJSON_VERSION_MAJOR) && ARDUINOJSON_VERSION_MAJOR >= 7
  JsonDocument doc;
#else
  StaticJsonDocument<512> doc;
#endif
  DeserializationError error = deserializeJson(doc, body);
  if (error) {
    Serial.println("Reminder: JSON parse failed");
    return;
  }

  bool pending = doc["pending"] | false;
  if (!pending) {
    return;
  }

  const char* audioUrl = doc["audio_url"];
  const char* message  = doc["message"] | "";

  showStatus("Reminder!", ORANGE);

  if (audioUrl != nullptr && strlen(audioUrl) > 0) {
    HTTPClient audioHttp;
    audioHttp.begin(SERVER_BASE + String(audioUrl));
    audioHttp.setTimeout(10000);
    int audioCode = audioHttp.GET();

    if (audioCode == 200) {
      lastSuccessfulInteraction = millis();

      size_t respSize = audioHttp.getSize();
      if (respSize > 44 && respSize < 500000) {
        uint8_t* respData = (uint8_t*)malloc(respSize);
        if (respData) {
          WiFiClient* stream = audioHttp.getStreamPtr();
          size_t read = 0;
          while (read < respSize) {
            size_t n = stream->readBytes(respData + read, respSize - read);
            if (n == 0) break;
            read += n;
          }
          if (read == respSize) {
            audioHttp.end();
            playAudio(respData, respSize);
            free(respData);

            HTTPClient ackHttp;
            ackHttp.begin(SERVER_BASE + "/reminder/ack");
            ackHttp.setTimeout(5000);
            int ackCode = ackHttp.POST("");
            if (ackCode == 200) {
              lastSuccessfulInteraction = millis();
            }
            Serial.printf("Reminder: ack result %d\n", ackCode);
            ackHttp.end();

            showIdleScreen();
            return;
          }
          free(respData);
        }
      }
    }
    audioHttp.end();
  }

  if (message != nullptr && strlen(message) > 0) {
    showStatus(message, ORANGE);
    delay(4000);
  }

  HTTPClient ackHttp;
  ackHttp.begin(SERVER_BASE + "/reminder/ack");
  ackHttp.setTimeout(5000);
  int ackCode = ackHttp.POST("");
  if (ackCode == 200) {
    lastSuccessfulInteraction = millis();
  }
  Serial.printf("Reminder: ack (fallback) result %d\n", ackCode);
  ackHttp.end();

  showIdleScreen();
}

// ---------------------------------------------------------------------------
// Wi‑Fi watchdog  —  called when no successful HTTP interaction for 5+ min
// ---------------------------------------------------------------------------
void runWatchdog() {
  if (!wasOffline) {
    showStatus("Check connection", RED);
    delay(3000);
    M5.update();
  }

  WiFi.disconnect(true);
  WiFi.mode(WIFI_STA);
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);

  unsigned long reconnectStart = millis();
  while (WiFi.status() != WL_CONNECTED && millis() - reconnectStart < 10000) {
    delay(500);
    M5.update();
  }

  if (WiFi.status() == WL_CONNECTED) {
    showStatus("Reconnected", GREEN);
    delay(2000);
    lastSuccessfulInteraction = millis();
    wasOffline = false;
  } else {
    if (!wasOffline) {
      showStatus("Offline", RED);
      delay(3000);
      wasOffline = true;
    }
    lastSuccessfulInteraction = millis() - WATCHDOG_TIMEOUT_MS + WATCHDOG_RETRY_MS;
  }

  showIdleScreen();
}

// ---------------------------------------------------------------------------
// Setup — runs once at power-on / reset
// ---------------------------------------------------------------------------
void setup() {
  M5.begin();
  M5.Lcd.setRotation(1);
  M5.Lcd.fillScreen(BLACK);
  M5.Lcd.setTextSize(2);

  SERVER_BASE = "http://" + String(SERVER_IP) + ":" + String(SERVER_PORT);

  Serial.begin(SERIAL_DEBUG_BAUD);

  pcmBuffer = (int16_t*)malloc(MAX_RECORDING_BYTES);
  if (!pcmBuffer) {
    showStatus("Out of memory", RED);
    Serial.println("FATAL: PCM buffer allocation failed");
    while (true) {
      M5.update();
      delay(LOOP_DELAY_MS);
    }
  }

  showStatus("Connecting...", WHITE);
  Serial.printf("WiFi: connecting to %s ...\n", WIFI_SSID);

  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);

  unsigned long start = millis();
  while (WiFi.status() != WL_CONNECTED) {
    if (millis() - start >= WIFI_TIMEOUT_MS) {
      showStatus("WiFi FAILED", RED);
      Serial.println("WiFi: timed out.");
      while (true) {
        M5.update();
        delay(LOOP_DELAY_MS);
      }
    }
    M5.Lcd.print(".");
    delay(500);
    Serial.print(".");
  }
  Serial.println();

  String ipStr = "IP: " + WiFi.localIP().toString();
  showStatus(ipStr, GREEN);
  Serial.printf("WiFi: connected, IP = %s\n", WiFi.localIP().toString().c_str());
  delay(2000);

  showStatus("Checking server...", YELLOW);

  HTTPClient http;
  String healthUrl = SERVER_BASE + "/health";
  http.begin(healthUrl);
  http.setTimeout(5000);

  Serial.printf("Health: GET %s\n", healthUrl.c_str());
  int httpCode = http.GET();

  if (httpCode == 200) {
    lastSuccessfulInteraction = millis();

    String body = http.getString();
    if (body.indexOf("healthy") >= 0 || body.indexOf("connected") >= 0 || body.indexOf("ollama") >= 0) {
      showStatus("Ready", GREEN);
      Serial.println("Health: passed (ollama=true found).");
    } else {
      showStatus("No backend agent", RED);
      Serial.printf("Health: missing ollama=true in body: %s\n", body.c_str());
      http.end();
      while (true) {
        M5.update();
        delay(LOOP_DELAY_MS);
      }
    }
  } else {
    showStatus("Server unreachable", RED);
    Serial.printf("Health: HTTP %d\n", httpCode);
    http.end();
    while (true) {
      M5.update();
      delay(LOOP_DELAY_MS);
    }
  }

  http.end();

  lastPollTime = millis();
  wasOffline = false;
  showIdleScreen();
  delay(1000);
}

// ---------------------------------------------------------------------------
// Loop — handle button A (record), button B (emergency), reminder polling,
//         and Wi‑Fi watchdog
// ---------------------------------------------------------------------------
void loop() {
  M5.update();

  if (M5.BtnA.wasPressed() && !isRecording) {
    startRecording();
  }

  if (M5.BtnB.wasPressed() && !isRecording) {
    sendEmergencyToServer();
  }

  if (!isRecording && !M5.Speaker.isPlaying()) {
    unsigned long now = millis();

    if (now - lastPollTime >= REMINDER_POLL_INTERVAL_MS) {
      lastPollTime = now;
      checkForReminder();
      now = millis();   // refresh after checkForReminder() may have updated lastSuccessfulInteraction
    }

    if (now - lastSuccessfulInteraction >= WATCHDOG_TIMEOUT_MS) {
      runWatchdog();
    }
  }

  delay(LOOP_DELAY_MS);
}
