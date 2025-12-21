# AgentCast V2 - Implementation Walkthrough

This guide details how to run and verify the AgentCast V2 backend using Postman.

## 1. Prerequisites
Ensure your server is running. Open a terminal and run:
```bash
make run
```
You should see: `Uvicorn running on http://127.0.0.1:8000`.

## 2. API Endpoint Details
-   **URL**: `http://127.0.0.1:8000/generate-podcast`
-   **Method**: `POST`
-   **Headers**: `Content-Type: application/json`

## 3. Postman Test Scenarios

### Scenario A: Standard English Podcast
*Tests the core Host-Guest loop in the default language.*

**Body (JSON):**
```json
{
    "topic": "The Future of Artificial General Intelligence",
    "tone": "Casual",
    "language": "English"
}
```
**Expected Result:**
-   **Status**: 200 OK
-   **Response**: A script with `Host:` and `Guest:` lines in English.
-   **Server Log**: You should see 3 turns of conversation (Host -> Guest -> Judge).

---

### Scenario B: Multilingual Support (Hindi)
*Tests the Translation Agent and Code-Mixing capabilities.*

**Body (JSON):**
```json
{
    "topic": "Black Holes and Event Horizons",
    "tone": "Formal",
    "language": "Hindi"
}
```
**Expected Result:**
-   **Response**: A script in Hindi script (Devanagari).
-   **Key Check**: Technical terms like "Black Hole", "Event Horizon", and "Gravity" should remain in English (Code-Mixing).

---

### Scenario C: Regional Language (Kannada)
*Tests support for Dravidian languages.*

**Body (JSON):**
```json
{
    "topic": "Sustainable Farming Practices",
    "tone": "Educational",
    "language": "Kannada"
}
```
**Expected Result:**
-   **Response**: A script in Kannada.
-   **Key Check**: Look for preservation of context. The idioms should be adapted (e.g., "Mother Earth" -> "Bhoomi Tayi").

---

### Scenario D: High-Stakes Debate
*Tests the "Tone" parameter's influence on the Host Agent.*

**Body (JSON):**
```json
{
    "topic": "Is Remote Work Good for Society?",
    "tone": "Debate",
    "language": "English"
}
```
**Expected Result:**
-   **Response**: The Host should ask more challenging, slightly aggressive questions.
-   **DCS Judge**: Watch the server logs for the Judge's `next_action`. You might see `steer_back` or `clarify` if the Guest gets too defensive.

## 4. Troubleshooting

| Issue | Cause | Fix |
| :--- | :--- | :--- |
| **403 Permission Denied** | Google API Key is invalid or leaked. | Generate a new key in Google AI Studio and update [.env](file:///e:/AgentCast-V2/.env). |
| **429 Resource Exhausted** | Free Tier Quota (20 RPM/RPD) exceeded. | Wait for the quota to reset (daily) or switch to a paid "Pay-as-you-go" key. |
| **500 Validation Error** | Translator returned a list instead of text. | Fixed in latest build; retry the request. |

## 5. Console Output
While Postman handles the request, watch your terminal running `make run`. You will see the "thought process" of the AI:
```text
--- HOST NODE ---
🎙️ HOST: Welcome to AgentCast! Today...

--- GUEST NODE ---
👨‍🔬 GUEST: It's great to be here...

--- JUDGE NODE ---
⚖️ JUDGE: Coherence=9/10 | Action=standard_follow_up
```
