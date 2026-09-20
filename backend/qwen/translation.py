"""
The one place a query or an answer changes language.

Phase 3 of the multilingual work order puts the seam here and nowhere else: a non-English query is
translated to English *before* MC2 sees it, so MC2, MC3 and every specialist engine reason over the
English string exactly as they always have. The transformation is recorded in the job trace, so a
reviewer can see the exact string the pipeline actually reasoned over.

Translation uses the Qwen3 instance already in the pipeline. No translation model, service or
dependency is added — which is also why a deployment without Qwen3 cannot serve a non-English
query at all, and says so rather than guessing (see TranslationUnavailable).
"""

import json
from typing import Any, Dict, Optional

# The five interface languages. "en" needs no translation and is handled by the caller.
SUPPORTED_QUERY_LANGUAGES = ("en", "hi", "kn", "te", "ta")

LANGUAGE_NAMES = {
    "en": "English",
    "hi": "Hindi",
    "kn": "Kannada",
    "te": "Telugu",
    "ta": "Tamil",
}


class TranslationUnavailable(RuntimeError):
    """
    Raised when a non-English query arrives and no language model can translate it.

    This is not a soft failure: the deterministic Tool Registry planner matches English keywords,
    so running it on a Kannada string would produce a confident answer to a question the pipeline
    never understood. The job is refused instead.
    """


def is_supported(language: Optional[str]) -> bool:
    return language in SUPPORTED_QUERY_LANGUAGES


def _generate_json(engine: Any, prompt: str, max_new_tokens: int) -> Dict[str, Any]:
    """
    Runs one generation and parses the JSON object out of it.

    Decoding is greedy (temperature 0.0), which is what measurement supports. A small temperature
    plus a repetition penalty was tried and reverted: on the same three real strings it made the
    model ignore the output schema for the query (it answered with a meta-object describing the
    task) and return empty translations for both caveats, while 0.0 translated all three
    correctly. The repetition loop that prompted the experiment came from batching several
    passages into one prompt, not from the temperature — see translate_texts_from_english.
    """
    result = engine.generate(prompt, temperature=0.0, max_new_tokens=max_new_tokens)
    # Both engines in qwen/ report success as "ok" (see inference.py and ollama_inference.py).
    if not isinstance(result, dict) or result.get("status") != "ok":
        raise TranslationUnavailable(
            f"the language model did not return a translation: {(result or {}).get('error', 'unknown error')}"
        )
    text = (result.get("response") or "").strip()
    if not text:
        raise TranslationUnavailable("the language model returned an empty translation")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Some backends wrap the object in prose even when asked not to.
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end <= start:
            raise TranslationUnavailable("the language model's translation was not valid JSON")
        try:
            return json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            raise TranslationUnavailable("the language model's translation was not valid JSON")


def translate_query_to_english(engine: Any, query: str, source_language: str) -> Dict[str, str]:
    """
    Translates a user's query into English for the pipeline.

    Returns the trace record the work order requires, with both strings kept so the transformation
    is auditable:
        {"from": "kn", "to": "en", "engine": "...", "original": "...", "translated": "..."}
    """
    if engine is None:
        raise TranslationUnavailable("no language model is loaded")
    if not is_supported(source_language) or source_language == "en":
        raise TranslationUnavailable(f"unsupported source language: {source_language!r}")

    source_name = LANGUAGE_NAMES[source_language]
    prompt = (
        f"You are a translation engine for a satellite-imagery analysis system.\n"
        f"Translate the user's question from {source_name} into English.\n"
        "Rules:\n"
        "- Translate meaning, not word for word. The result must read as a natural English question.\n"
        "- Keep technical terms, place names, sensor names and numbers unchanged (for example SAR, "
        "Sentinel-1, NDVI, GeoTIFF).\n"
        "- Do not answer the question. Do not add commentary.\n"
        'Output ONLY this JSON object: {"translation": "the English question"}\n\n'
        f"{source_name} question: {query}"
    )
    payload = _generate_json(engine, prompt, max_new_tokens=256)
    translated = str(payload.get("translation") or "").strip()
    if not translated:
        raise TranslationUnavailable("the language model returned an empty translation")

    return {
        "from": source_language,
        "to": "en",
        "engine": getattr(engine, "model_name", None) or engine.__class__.__name__,
        "original": query,
        "translated": translated,
    }


def translate_text_from_english(engine: Any, text: str, target_language: str) -> str:
    """
    Translates finished answer prose into the user's language.

    Used only on the answer and its caveats, after they have been synthesised in English from
    verified evidence. Returns the English text unchanged if translation is not possible, so a
    failure here degrades the wording rather than losing the answer.
    """
    if engine is None or not text or not text.strip():
        return text
    if not is_supported(target_language) or target_language == "en":
        return text

    target_name = LANGUAGE_NAMES[target_language]
    prompt = (
        f"You are a translation engine for a satellite-imagery analysis system.\n"
        f"Translate the following English text into {target_name}.\n"
        "Rules:\n"
        "- Translate meaning, not word for word.\n"
        "- Keep these unchanged, in Latin script: model and tool names, stage codes such as MC1 to "
        "MC8, status words such as ABSTAIN, file formats, CRS strings, units and all numbers.\n"
        "- Do not add, remove or soften any statement. This text reports what a system did and did "
        "not establish; its hedges and limitations must survive translation.\n"
        f'Output ONLY this JSON object: {{"translation": "the {target_name} text"}}\n\n'
        f"English text: {text}"
    )
    try:
        payload = _generate_json(engine, prompt, max_new_tokens=1024)
    except TranslationUnavailable:
        return text
    translated = str(payload.get("translation") or "").strip()
    return translated or text


def translate_texts_from_english(engine: Any, texts: list, target_language: str) -> list:
    """
    Translates several passages, one generation each.

    A batched "translate this numbered list" prompt was tried first and abandoned: Qwen3-4B does
    not attend to the individual items. At temperature 0.0 it looped a single phrase until it ran
    out of tokens and the JSON was left unterminated; with a temperature and a repetition penalty
    it returned the *same* invented sentence for every passage — which passes a count check and
    would have shipped nonsense as the caveats of a real answer. One passage per call is the path
    that demonstrably works, and it degrades per passage: anything the model cannot translate
    comes back in English rather than as something plausible and wrong.

    Caveats are what keep an answer honest, so this trades latency for not fabricating them.
    """
    if engine is None or not texts:
        return texts
    if not is_supported(target_language) or target_language == "en":
        return texts
    return [translate_text_from_english(engine, text, target_language) for text in texts]
