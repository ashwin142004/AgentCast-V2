import edge_tts
import os
import uuid
from typing import List, Dict
from app.schemas import DialogueTurn

# Directory to save audio files
AUDIO_DIR = "static/audio"
os.makedirs(AUDIO_DIR, exist_ok=True)

class TTSHandler:
    def __init__(self):
        # Mapping Language -> {Host: Voice, Guest: Voice}
        # Using Microsoft Edge TTS Voices
        self.voice_map = {
            "English": {
                "Host": "en-US-GuyNeural",     # Male
                "Guest": "en-US-JennyNeural"   # Female
            },
            "Hindi": {
                "Host": "hi-IN-MadhurNeural",  # Male
                "Guest": "hi-IN-SwaraNeural"   # Female
            },
            "Kannada": {
                "Host": "kn-IN-GaganNeural",   # Male
                "Guest": "kn-IN-SapnaNeural"   # Female
            },
            "Tamil": {
                "Host": "ta-IN-ValluvarNeural",# Male
                "Guest": "ta-IN-PallaviNeural" # Female
            },
            "Telugu": {
                "Host": "te-IN-MohanNeural",   # Male
                "Guest": "te-IN-ShrutiNeural"  # Female
            },
            "Malayalam": {
                "Host": "ml-IN-MidhunNeural",  # Male
                "Guest": "ml-IN-SobhanaNeural" # Female
            },
            "Marathi": {
                "Host": "mr-IN-ManoharNeural", # Male
                "Guest": "mr-IN-AarohiNeural"  # Female
            },
            "Bengali": {
                "Host": "bn-IN-BashkarNeural", # Male
                "Guest": "bn-IN-TanishaaNeural"# Female
            },
            "Gujarati": {
                "Host": "gu-IN-NiranjanNeural",# Male
                "Guest": "gu-IN-DhwaniNeural"  # Female
            }
        }
        
        # Fallback voices
        self.default_host = "en-US-GuyNeural"
        self.default_guest = "en-US-JennyNeural"

    async def generate_audio(self, text: str, voice: str) -> str:
        """
        Generates audio for the given text and voice.
        Returns the path to the generated file.
        """
        if not text:
            return ""
            
        communicate = edge_tts.Communicate(text, voice)
        filename = f"{uuid.uuid4()}.mp3"
        filepath = os.path.join(AUDIO_DIR, filename)
        
        await communicate.save(filepath)
        return filepath

    async def process_script(self, script: List[DialogueTurn], language: str) -> str:
        """
        Processes a full dialogue script, generates audio for each turn, 
        and combines them into a single audio file.
        Returns the path to the combined audio file.
        """
        lang_voices = self.voice_map.get(language, self.voice_map["English"])
        
        host_voice = lang_voices.get("Host", self.default_host)
        guest_voice = lang_voices.get("Guest", self.default_guest)
        
        temp_files = []
        
        print(f"Generating TTS for {len(script)} turns in {language}...")
        
        try:
            # 1. Generate individual clips
            for turn in script:
                # Host
                host_path = await self.generate_audio(turn.Host, host_voice)
                temp_files.append(host_path)
                
                # Guest
                guest_path = await self.generate_audio(turn.Guest, guest_voice)
                temp_files.append(guest_path)
            
            # 2. Combine Clips
            # We try pydub first; if ffmpeg is missing, we fallback to binary append for MP3
            combined_filename = f"podcast_{uuid.uuid4()}.mp3"
            combined_filepath = os.path.join(AUDIO_DIR, combined_filename)
            
            try:
                from pydub import AudioSegment
                combined_audio = AudioSegment.empty()
                for tf in temp_files:
                    try:
                        segment = AudioSegment.from_mp3(tf)
                        combined_audio += segment
                        # Optional: Add small silence?
                        # combined_audio += AudioSegment.silent(duration=300) 
                    except Exception as e:
                        print(f"Pydub read error for {tf}: {e}. Trying binary append fallback.")
                        raise ImportError("Pydub failed, switching to binary append")
                
                combined_audio.export(combined_filepath, format="mp3")
                print(f"Combined audio saved (using pydub): {combined_filepath}")
                
            except (ImportError, FileNotFoundError, Exception) as e:
                print(f"ffmpeg/pydub issue ({e}). Falling back to binary concatenation.")
                with open(combined_filepath, 'wb') as outfile:
                    for tf in temp_files:
                        with open(tf, 'rb') as infile:
                            outfile.write(infile.read())
                print(f"Combined audio saved (binary append): {combined_filepath}")

            return combined_filepath
            
        finally:
            # Cleanup temp files
            for tf in temp_files:
                if os.path.exists(tf):
                    try:
                        os.remove(tf)
                    except:
                        pass

# Global instance
tts_handler = TTSHandler()
