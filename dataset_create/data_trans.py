import json
from deep_translator import GoogleTranslator

def translate_fhir_json(json_data, source_lang="en", target_lang="vi"):
    translator = GoogleTranslator(source=source_lang, target=target_lang)

    def translate_field(value):
        if isinstance(value, str):
            return translator.translate(value)
        elif isinstance(value, list):
            return [translate_field(v) for v in value]
        elif isinstance(value, dict):
            return {k: translate_field(v) for k, v in value.items()}
        else:
            return value

    return translate_field(json_data)

# Load your JSON
with open(r"D:\Work_LacViet\LacViet_AI\fhir-rag-cpg\dataset_create\cpg_output\transient-ischemic-attack--tia--and-minor-stroke\bundle.json", "r", encoding="utf-8") as f:
        data = json.load(f)

# Translate from English to Vietnamese
translated_data = translate_fhir_json(data, source_lang="en", target_lang="vi")

# Save the translated version
with open("guideline_bundle_vi.json", "w", encoding="utf-8") as f:
    json.dump(translated_data, f, ensure_ascii=False, indent=2)
