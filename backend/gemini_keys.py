"""
Rotates across multiple Gemini API keys (e.g. free-tier keys from separate
Google accounts) so the pipeline keeps working once one key's quota runs out,
instead of failing the whole run. Keys are tried in order; a quota/rate-limit
error on the current key silently advances to the next one and retries the
same request - no change needed in extract.py or score.py beyond using
RotatingGeminiClient instead of configuring genai directly.
"""

import os


def _load_keys():
    """Reads GEMINI_API_KEY_1, GEMINI_API_KEY_2, GEMINI_API_KEY_3, ... in order,
    stopping at the first gap. Falls back to a single GEMINI_API_KEY if no
    numbered keys are set, so existing .env files still work unchanged."""
    keys = []
    i = 1
    while True:
        key = os.environ.get(f"GEMINI_API_KEY_{i}")
        if not key:
            break
        keys.append(key)
        i += 1
    if not keys and os.environ.get("GEMINI_API_KEY"):
        keys.append(os.environ["GEMINI_API_KEY"])
    return keys


def _is_quota_error(exc: Exception) -> bool:
    """Best-effort detection of a quota/billing/rate-limit error (the case we
    want to rotate keys for) vs. a genuine bug (which should still raise).
    Gemini's SDK normally surfaces quota errors as ResourceExhausted (HTTP 429);
    the text-based fallback below covers other SDK versions/error shapes."""
    try:
        from google.api_core.exceptions import ResourceExhausted, PermissionDenied
        if isinstance(exc, (ResourceExhausted, PermissionDenied)):
            return True
    except ImportError:
        pass
    msg = str(exc).lower()
    return any(s in msg for s in
               ["quota", "429", "rate limit", "resource_exhausted", "exceeded", "billing"])


class RotatingGeminiClient:
    """
    Drop-in wrapper around a Gemini model that transparently rotates across
    multiple API keys when the active one hits its quota.

        client = RotatingGeminiClient(model_name="gemini-2.5-flash")
        response = client.generate_content([prompt, image])   # or just prompt

    Raises RuntimeError immediately if no GEMINI_API_KEY_* / GEMINI_API_KEY is
    configured at all (caller can catch this to fall back to another engine).
    Raises whatever the underlying SDK raised only once every configured key
    has been tried and failed with a quota-type error.
    """

    def __init__(self, model_name: str = "gemini-2.5-flash"):
        self.keys = _load_keys()
        if not self.keys:
            raise RuntimeError(
                "No Gemini API keys found. Set GEMINI_API_KEY_1 (and optionally "
                "_2, _3, ...) or GEMINI_API_KEY in your .env file."
            )
        self.model_name = model_name
        self.index = 0
        self._model = None
        self._configure_current()

    def _configure_current(self):
        import google.generativeai as genai
        genai.configure(api_key=self.keys[self.index])
        self._model = genai.GenerativeModel(self.model_name)

    def _rotate(self) -> bool:
        if self.index + 1 >= len(self.keys):
            return False
        self.index += 1
        print(f"[gemini] Key {self.index}/{len(self.keys)} hit its quota - rotating to next key.")
        self._configure_current()
        return True

    def generate_content(self, *args, **kwargs):
        while True:
            try:
                return self._model.generate_content(*args, **kwargs)
            except Exception as e:
                if _is_quota_error(e) and self._rotate():
                    continue
                raise