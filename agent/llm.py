import os
import time
import itertools
import threading
from pathlib import Path

# ==========================
# Load .env manually
# ==========================

_env_path = Path(__file__).parent.parent / ".env"

if _env_path.exists():
    for line in _env_path.read_text().splitlines():
        line = line.strip()

        if (
            line
            and not line.startswith("#")
            and "=" in line
        ):
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())

# ==========================
# Imports
# ==========================

from google import genai
from config import TEMPERATURE

# ==========================
# Gemini API Keys
# ==========================

API_KEYS = [
    os.environ.get("GEMINI_API_KEY_1", ""),
    os.environ.get("GEMINI_API_KEY_2", ""),
    os.environ.get("GEMINI_API_KEY_3", ""),
    os.environ.get("GEMINI_API_KEY_4", ""),
    os.environ.get("GEMINI_API_KEY_5", ""),
    os.environ.get("GEMINI_API_KEY_6", ""),
    os.environ.get("GEMINI_API_KEY_7", ""),
]

# Remove empty entries
API_KEYS = [k for k in API_KEYS if k]

if not API_KEYS:
    raise ValueError(
        "No Gemini API keys found. "
        "Add GEMINI_API_KEY_1 ... GEMINI_API_KEY_7 to .env"
    )

# Thread-safe key rotation
_key_cycle = itertools.cycle(API_KEYS)
_key_lock = threading.Lock()


def get_next_key():
    with _key_lock:
        return next(_key_cycle)


def create_client(api_key):
    return genai.Client(api_key=api_key)


print(
    f"[LLM] Google AI Studio enabled "
    f"({len(API_KEYS)} API keys loaded)"
)

# ==========================
# Main LLM Call
# ==========================


def call_llm(
    system: str,
    user: str,
    model: str,
    max_tokens: int = 1024,
    temperature: float = TEMPERATURE,
    retries: int = 20,
):

    prompt = f"""
SYSTEM:
{system}

USER:
{user}
"""

    last_error = ""

    current_key = get_next_key()

    for attempt in range(retries):

        try:

            client = create_client(current_key)

            response = client.models.generate_content(
                model=model,
                contents=prompt,
            )

            return response.text or ""

        except Exception as e:

            last_error = str(e)

            print(
                f"[LLM] Gemini error attempt {attempt + 1}: "
                f"{last_error[:150]}"
            )

            # ==========================
            # Quota exhausted -> switch key
            # ==========================

            if (
                "429" in last_error
                or "RESOURCE_EXHAUSTED" in last_error
                or "quota" in last_error.lower()
            ):

                current_key = get_next_key()

                print(
                    "[LLM] Quota exhausted. "
                    "Switching to next Gemini API key..."
                )

                time.sleep(2)
                continue

            # ==========================
            # Other errors -> retry
            # ==========================

            wait = min(5 * (attempt + 1), 60)

            print(
                f"[LLM] Retrying in {wait}s..."
            )

            time.sleep(wait)

    return f"[LLM ERROR: {last_error[:200]}]"