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

    # Define your detailed custom prompt as the prefill command
    prefill_message = {
        "role": "system",
        "content": """I am Paige, your AO3-level roleplay partner. I write in third-person POV for {{char}} and any NPCs. You control {{user}}’s actions, thoughts, and words. I control {{char}} and NPCs unless we collaborate otherwise. I’ll keep scenes immersive, descriptive, and natural, with realistic consequences.
Tone & Style:
    Explicit, graphic, and unfiltered in all NSFW scenes.
    Dialogue stays true to character. I may use crude, mature language (fuck, slut, cock, etc.).
    Actions in asterisks, thoughts in backticks, spoken lines in quotes.
    Posts 50–150+ words, split into paragraphs, always ending with room for {{user}} to act.
NSFW / Smut Rules:
I will create the most intense AO3-style smut between {{char}} and {{user}}, using descriptive, explicit, and graphic details:
    Describe pussy, dick, asshole, mouth; color, wetness, hardness, arousal, nipples, orgasms, squirting, cum, pre-cum.
    Use crude slang for body parts and actions (“cock”, “cunt”, “tits”, “fuck”, “cum”, “slut”, etc.).
    Include raw physical actions (plap plap plap), messy details, fluids, sounds (slurp, glorp, schlorp, glug).
    Write {{char}}’s moans and reactions explicitly (“ahhh~♥”, “mmmphh”, “fuckkk!”, etc.).
    Sex is a conversation and dance — no instant orgasms unless story-appropriate.
    Reactions depend on arousal, attraction, and context; consider age and experience.
    Make anatomy realistic and physical sensations believable.
Other Scene Types:
    Romance: Build chemistry with touches, eye contact, confessions, “one bed” trope, pinning, slow burn or fast passion. Make kisses cinematic and earned.
    Combat: Gritty, graphic fights with wounds, blood, gore, cursing, and desperation. Use shorter sentences for pacing, mix action and dialogue.
    Emotional/Psychological: Explore trauma, fears, desires, coping mechanisms, breakdowns, healing, regression, or villain arcs if it fits the character.
Character Development:
{{char}} evolves based on in-story events and interactions with {{user}}. Growth or regression must have emotional and narrative cause.

(OOC: When {{user}} gives an OOC command, pause RP and follow it.)
"""
    }

    # Insert the prefill message at the beginning of the messages list
    messages.insert(0, prefill_message)

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
            buffer = ""
            stream_url = GEMINI_API_STREAM_URL
            try:
                async with httpx.AsyncClient(timeout=60) as client:
                    async with client.stream("POST", stream_url, params=params, headers=headers, json=gemini_payload) as resp:
                        async for line in resp.aiter_lines():
                            if not line.strip():
                                continue
                            print(f"Received raw line from Gemini: {line}") # Debugging line
                            buffer += line.strip()

                            # Attempt to find and parse complete JSON objects from the buffer
                            # This loop will continue as long as there are potential JSON objects in the buffer
                            while True:
                                open_brace_count = 0
                                close_brace_count = 0
                                json_start_index = -1
                                json_end_index = -1

                                # Iterate through the buffer to find a balanced JSON object
                                for i, char in enumerate(buffer):
                                    if char == '{':
                                        if json_start_index == -1: # Mark the start of the first object
                                            json_start_index = i
                                        open_brace_count += 1
                                    elif char == '}':
                                        close_brace_count += 1

                                    # If we found an opening brace and the counts are balanced, we have a complete object
                                    if json_start_index != -1 and open_brace_count > 0 and open_brace_count == close_brace_count:
                                        json_end_index = i
                                        break # Found a complete JSON object

                                if json_start_index != -1 and json_end_index != -1:
                                    # Extract the potential JSON string
                                    json_str = buffer[json_start_index : json_end_index + 1]
                                    try:
                                        data = json.loads(json_str)
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

                                        # Remove the successfully parsed JSON object and any leading/trailing non-JSON chars
                                        # up to the end of the parsed object from the buffer.
                                        buffer = buffer[json_end_index + 1:].strip()
                                        print(f"Remaining buffer: {buffer[:100]}...") # Debugging line
                                        # Continue the while loop to check for more objects in the remaining buffer
                                        continue
                                    except json.JSONDecodeError as e:
                                        # If the extracted json_str is not valid JSON, it means our brace counting was off
                                        # or the data is truly malformed. Break from this inner loop to get more lines.
                                        print(f"JSON Decode Error (inner parse): {e} for json_str: {json_str[:100]}...") # Debugging line
                                        break
                                    except Exception as e:
                                        print(f"Error processing Gemini stream data (inner): {e}") # Debugging line
                                        break # Break from inner loop if unexpected error
                                else:
                                    # No complete JSON object found in the current buffer, need more data
                                    break # Break from inner while loop, get more lines from the stream

                # After the stream ends, check if there's any remaining data in the buffer
                if buffer.strip():
                    print(f"Stream ended with remaining buffer: {buffer[:100]}...")
                    # At this point, any remaining buffer is likely an incomplete fragment or malformed data.
                    # We can try a final parse if it looks like a list, but generally, it's best to discard.
                    # For robustness, we'll try to parse it as a list if it starts with '['
                    if buffer.strip().startswith('['):
                        try:
                            final_data = json.loads(buffer.strip())
                            if isinstance(final_data, list):
                                for item in final_data:
                                    text = ""
                                    finish_reason = None
                                    if "candidates" in item and item["candidates"]:
                                        candidate = item["candidates"][0]
                                        if "content" in candidate and "parts" in candidate["content"]:
                                            for part in candidate["content"]["parts"]:
                                                if "text" in part:
                                                    text += part["text"]
                                        if "finishReason" in candidate:
                                            finish_reason = candidate["finishReason"]
                                    if text or finish_reason:
                                        chunk = {
                                            "id": "chatcmpl-proxy-stream-final",
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
                                        print(f"Yielding final chunk to JanitorAI: {json_chunk}")
                                        yield f"data: {json_chunk}\n\n"
                        except json.JSONDecodeError as e:
                            print(f"Final buffer JSON Decode Error (list parse): {e} for buffer: {buffer[:100]}...")
                        except Exception as e:
                            print(f"Error processing final buffer (list parse): {e}")
                    else:
                        print(f"Discarding remaining buffer (not a list or complete object): {buffer[:100]}...")

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
