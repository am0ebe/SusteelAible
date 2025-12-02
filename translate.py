from transformers import pipeline

lang_detector = pipeline("text-classification", model="papluca/xlm-roberta-base-language-detection")

result = lang_detector("Das ist ein Nachhaltigkeitsbericht.")
print(result)  # [{'label': 'de', 'score': 0.999...}]



#######################

import langid
from transformers import pipeline

# Translation models
translators = {
    "de": pipeline("translation", model="Helsinki-NLP/opus-mt-de-en"), # speed 5 accu 4
    "es": pipeline("translation", model="Helsinki-NLP/opus-mt-es-en"),
    "it": pipeline("translation", model="Helsinki-NLP/opus-mt-it-en"),
    "fr": pipeline("translation", model="Helsinki-NLP/opus-mt-fr-en"),
}

def detect_language(text):
    lang, score = langid.classify(text)
    return lang

def translate_chunk(lang, chunk):
    if lang not in translators:
        return chunk  # fallback: no translation
    return translators[lang](chunk)[0]["translation_text"]

# --- Example usage ---
lang = detect_language(first_page_text)

translated_chunks = [translate_chunk(lang, c) for c in pdf_chunks]


