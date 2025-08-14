# Gemini 2.5 Flash Proxy for Janitor.ai (OpenAI-compatible)
# To run: python -m venv venv && source venv/bin/activate && pip install fastapi uvicorn httpx
# Deployable on Render.com


from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
import httpx
import os
import json
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI()

# Enable CORS for all origins (adjust origins as needed for more security)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# Set your Gemini API key as an environment variable on Render

GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent"
GEMINI_API_STREAM_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:streamGenerateContent"



@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    body = await request.json()
    messages = body.get("messages", [])
    prompt = "\n".join([m.get("content", "") for m in messages if m.get("role") in ("user", "system")])

    # Extract Gemini API key from request (Authorization header, x-api-key header, body, or query)
    api_key = None
    auth_header = request.headers.get("authorization")
    if auth_header and auth_header.lower().startswith("bearer "):
        api_key = auth_header.split(" ", 1)[1]
    elif request.headers.get("x-api-key"):
        api_key = request.headers.get("x-api-key")
    elif body.get("api_key"):
        api_key = body.get("api_key")
    else:
        # Try to get from query params
        query_params = dict(request.query_params)
        api_key = query_params.get("api_key")
    if not api_key:
        # Fallback to environment variable
        api_key = os.getenv("GEMINI_API_KEY", "")
    if not api_key:
        return JSONResponse(status_code=401, content={"error": "Gemini API key required. Provide it in Authorization header (Bearer YOUR_KEY), x-api-key header, api_key in JSON body, or as api_key query param."})

    # Gemini API expects a different format; adapt as needed
    gemini_payload = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}]
    }

    headers = {"Content-Type": "application/json"}
    params = {"key": api_key}

    # Streaming support
    if body.get("stream"):
        async def event_generator():
            buffer = "" # Initialize a buffer to accumulate lines
            stream_url = GEMINI_API_STREAM_URL
            try:
                async with httpx.AsyncClient(timeout=60) as client:
                    async with client.stream("POST", stream_url, params=params, headers=headers, json=gemini_payload) as resp:
                        async for line in resp.aiter_lines():
                            if not line.strip():
                                continue
                            print(f"Received raw line from Gemini: {line}") # Debugging line
                            buffer += line.strip() # Append line to buffer, remove leading/trailing whitespace

                            if not buffer: # Skip if buffer is empty after stripping
                                continue

                            try:
                                # Try to parse the buffer as a complete JSON object
                                data = json.loads(buffer)
                                print(f"Parsed JSON from Gemini: {data}") # Debugging line
                                # Extract text and finish_reason from Gemini chunk
                                text = ""
                                finish_reason = None
                                if "candidates" in data and data["candidates"]:
                                    candidate = data["candidates"][0]
                                    if "content" in candidate and "parts" in candidate["content"]:
                                        for part in candidate["content"]["parts"]:
                                            if "text" in part:
                                                text += part["text"]
                                    if "finishReason" in candidate:
                                        finish_reason = candidate["finishReason"]

                                if text or finish_reason:
                                    chunk = {
                                        "id": "chatcmpl-proxy-stream",
                                        "object": "chat.completion.chunk",
                                        "created": 0,
                                        "model": "gemini-2.5-flash-proxy",
                                        "choices": [
                                            {
                                                "index": 0,
                                                "delta": {"content": text},
                                                "finish_reason": finish_reason
                                            }
                                        ]
                                    }
                                    json_chunk = json.dumps(chunk)
                                    print(f"Yielding chunk to JanitorAI: {json_chunk}") # Debugging line
                                    yield f"data: {json_chunk}\n\n"
                                buffer = "" # Clear buffer on successful parse
                            except json.JSONDecodeError as e:
                                # If parsing fails, it means we don't have a complete JSON object yet.
                                # Continue accumulating in the buffer.
                                print(f"JSON Decode Error (buffering): {e} for current buffer: {buffer[:100]}...") # Debugging line
                                continue
                            except Exception as e:
                                print(f"Error processing Gemini stream data: {e}") # Debugging line
                                continue
                # End of stream
                yield "data: [DONE]\n\n"
            except Exception as e:
                print(f"Top-level streaming error: {e}") # Debugging line
                yield f"data: {{\"error\": \"{str(e)}\"}}\n\n"

        return StreamingResponse(event_generator(), media_type="text/event-stream")

    # Non-streaming (regular) response
    async with httpx.AsyncClient() as client:
        try:
            resp = await client.post(GEMINI_API_URL, params=params, headers=headers, json=gemini_payload, timeout=30)
            resp.raise_for_status()
            gemini_data = resp.json()
            # Extract the response text (adjust as per Gemini API response)
            reply = gemini_data["candidates"][0]["content"]["parts"][0]["text"]
        except Exception as e:
            return JSONResponse(status_code=500, content={"error": str(e)})

    # Return in OpenAI-compatible format
    return {
        "id": "chatcmpl-proxy",
        "object": "chat.completion",
        "created": 0,
        "model": "gemini-2.5-flash-proxy",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": reply},
                "finish_reason": "stop"
            }
        ],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    }

# For local dev: uvicorn main:app --host 0.0.0.0 --port 8000
