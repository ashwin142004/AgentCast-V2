import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from indicnlp.transliterate.unicode_transliterate import UnicodeIndicTransliterator

class Translator:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._init()
        return cls._instance

    def _init(self):
        print("Loading AI Model... (Downloads ~4GB first time)")

        self.model_name = "ai4bharat/indictrans2-en-indic-1B"
        
        # Setup Device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Running on: {self.device}")

        # Load Tokenizer & Model
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True
        )

        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            self.model_name,
            trust_remote_code=True
        ).to(self.device)
        
        self.model.eval()

        self.src_lang = "eng_Latn"

        # Map: User Friendly Name -> (FLORES Code, ISO Code for Script)
        self.lang_map = {
            "Hindi":     ("hin_Deva", "hi"),
            "Kannada":   ("kan_Knda", "kn"),
            "Tamil":     ("tam_Taml", "ta"),
            "Telugu":    ("tel_Telu", "te"),
            "Malayalam": ("mal_Mlym", "ml"),
            "Marathi":   ("mar_Deva", "mr"),
            "Bengali":   ("ben_Beng", "bn"),
            "Gujarati":  ("guj_Gujr", "gu")
        }

    def translate(self, text, target_language):
        if not text or not text.strip():
            return ""
            
        if target_language.lower() == "english":
            return text

        # 1. Resolve Language Codes
        target_info = self.lang_map.get(target_language)
        
        # Case-insensitive fallback
        if not target_info:
            for k, v in self.lang_map.items():
                if k.lower() == target_language.lower():
                    target_info = v
                    break
        
        if not target_info:
            return f"Error: Unsupported language '{target_language}'"

        flores_code, iso_code = target_info

        # 2. Format Input
        input_text = f"{self.src_lang} {flores_code} {text}"

        # 3. Tokenize
        inputs = self.tokenizer(
            [input_text],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=256
        ).to(self.device)

        # 4. Generate (With use_cache=False to prevent crash)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=256,
                num_beams=5,
                use_cache=False 
            )

        # 5. Decode (This output will be in HINDI script usually)
        decoded_text = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
        decoded_text = decoded_text.strip()

        # 6. Post-Process: Convert Script (Hindi -> Target)
        # If the target is NOT Hindi/Marathi, we likely need to convert the script.
        # The model often outputs Dravidian languages in Devanagari.
        if iso_code != "hi" and iso_code != "mr":
            try:
                # Transliterate from Hindi (Devanagari) to Target Script
                final_text = UnicodeIndicTransliterator.transliterate(decoded_text, "hi", iso_code)
                return final_text
            except Exception as e:
                print(f"Transliteration Error: {e}")
                return decoded_text # Fallback to Devanagari if conversion fails
        
        return decoded_text

# Singleton
_translator = None

def get_translator():
    global _translator
    if _translator is None:
        _translator = Translator()
    return _translator

def translate_script(text, target_language):
    return get_translator().translate(text, target_language)