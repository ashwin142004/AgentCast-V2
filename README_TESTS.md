# AgentCast V2 - Implementation Walkthrough

This guide details how to run and verify the AgentCast V2 backend using Postman.

## 1. Prerequisites
Ensure your server is running. Open a terminal and run:
```bash
make run
```
You should see: `Uvicorn running on http://127.0.0.1:8000`.

## 2. API Endpoints

### 2.1 Generate Podcast
-   **URL**: `http://127.0.0.1:8000/generate-podcast`
-   **Method**: `POST`
-   **Headers**: `Content-Type: application/json`
-   **Payload**:
    ```json
    {
        "topic": "string",
        "tone": "Casual" 
    }
    ```
    *Note: `tone` can be "Casual", "Formal", or "Debate".*

### 2.2 Translate
-   **URL**: `http://127.0.0.1:8000/translate`
-   **Method**: `POST`
-   **Headers**: `Content-Type: application/json`
-   **Payload (Option A - Script)**:
    ```json
    {
        "script": [
            {"Host": "Hello", "Guest": "Hi"}
        ],
        "target_language": "Hindi"
    }
    ```
-   **Payload (Option B - Single Text)**:
    ```json
    {
        "text": "Hello World",
        "target_language": "Kannada"
    }
    ```

## 3. Postman Test Scenarios

### Scenario A: Standard English Podcast
*Tests the core Host-Guest loop generation.*

**Endpoint**: `/generate-podcast`
**Body (JSON):**
```json
{
    "topic": "The Future of Artificial General Intelligence",
    "tone": "Casual"
}
```
**Expected Result:**
-   **Status**: 200 OK
-   **Response**: A JSON object containing a `script` array.
    ```json
    {
        "status": "completed",
        "script": [
            {"Host": "...", "Guest": "..."}
        ],
        "language": "English"
    }
    ```
-   **Server Log**: You should see 3 turns of conversation (Host -> Guest -> Judge).

---

### Scenario B: Translate Script (Hindi)
*Tests the detailed script translation mechanism.*

**Endpoint**: `/translate`
**Body (JSON):**
```json
{
    "script": [
        {"Host": "Welcome to the show.", "Guest": "Thanks for having me."}
    ],
    "target_language": "Hindi"
}
```
**Expected Result:**
-   **Response**:
    ```json
    {
        "translated_script": [
            {"Host": "...", "Guest": "..."}
        ],
        "language": "Hindi"
    }
    ```
-   **Key Check**: The values should be in Devanagari script.

---

### Scenario C: Translate Text (Kannada)
*Tests the single text fallback translation.*

**Endpoint**: `/translate`
**Body (JSON):**
```json
{
    "text": "The universe is vast and mysterious.",
    "target_language": "Kannada"
}
```
**Expected Result:**
-   **Response**:
    ```json
    {
        "translated_text": "...",
        "language": "Kannada"
    }
    ```

---

### Scenario D: High-Stakes Debate
*Tests the "Tone" parameter's influence on the Host Agent.*

**Endpoint**: `/generate-podcast`
**Body (JSON):**
```json
{
    "topic": "Is Remote Work Good for Society?",
    "tone": "Debate"
}
```
**Expected Result:**
-   **Response**: The generated script's Host dialogue should be challenging and aggressive.
-   **DCS Judge**: Watch the server logs for `next_action`.

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
