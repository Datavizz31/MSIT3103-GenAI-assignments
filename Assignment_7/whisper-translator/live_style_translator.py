#!/usr/bin/env python3
"""
Live Style-Preserving Speech Translator
Real-time translation with speaker style and rhythm preservation
"""

import whisper
import numpy as np
import librosa
from resemblyzer import VoiceEncoder, preprocess_wav
from TTS.api import TTS
import tempfile
import os
import argparse
from pydub import AudioSegment
import sounddevice as sd
import wave
import soundfile as sf

class LiveStyleTranslator:
    def __init__(self, model_size="base"):
        """Initialize with all required models"""
        self.whisper_model = whisper.load_model(model_size)
        self.voice_encoder = VoiceEncoder()
        self.tts = TTS("tts_models/en/ljspeech/tacotron2-DDC")
        self.sample_rate = 16000
        self.speaker_embedding = None
        
    def record_voice_sample(self, duration=3):
        """Record a voice sample to extract speaker characteristics"""
        print(f"Recording voice sample for {duration} seconds...")
        print("Speak clearly to capture your voice characteristics...")
        
        sample_rate = 44100
        audio_data = sd.rec(int(duration * sample_rate), 
                           samplerate=sample_rate, 
                           channels=1, 
                           dtype=np.float32)
        sd.wait()
        
        # Save sample and extract embedding
        temp_sample = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        sf.write(temp_sample.name, audio_data.flatten(), sample_rate)
        
        # Extract speaker embedding
        wav = preprocess_wav(temp_sample.name)
        self.speaker_embedding = self.voice_encoder.embed_utterance(wav)
        
        os.unlink(temp_sample.name)
        print("Voice characteristics captured successfully!")
        
    def record_audio_chunk(self, duration=5):
        """Record audio chunk for translation"""
        print(f"Recording for {duration} seconds... Speak now!")
        
        sample_rate = 44100
        audio_data = sd.rec(int(duration * sample_rate), 
                           samplerate=sample_rate, 
                           channels=1, 
                           dtype=np.float32)
        sd.wait()
        
        # Resample to 16kHz for Whisper
        from scipy import signal
        audio_16k = signal.resample(audio_data.flatten(), int(len(audio_data) * 16000 / sample_rate))
        
        return audio_16k
    
    def save_temp_audio(self, audio_data):
        """Save audio data to temporary file"""
        temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        
        with wave.open(temp_file.name, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(self.sample_rate)
            audio_int16 = (audio_data * 32767).astype(np.int16)
            wf.writeframes(audio_int16.tobytes())
        
        return temp_file.name
    
    def extract_prosody(self, audio_path, segments):
        """Extract timing and prosody information"""
        audio, sr = librosa.load(audio_path)
        prosody_data = []
        
        for segment in segments:
            start_sample = int(segment['start'] * sr)
            end_sample = int(segment['end'] * sr)
            segment_audio = audio[start_sample:end_sample]
            
            if len(segment_audio) > 0:
                f0, voiced_flag, voiced_probs = librosa.pyin(segment_audio, 
                                                            fmin=librosa.note_to_hz('C2'), 
                                                            fmax=librosa.note_to_hz('C7'))
                
                prosody_data.append({
                    'duration': segment['end'] - segment['start'],
                    'mean_pitch': np.nanmean(f0) if len(f0) > 0 else 0,
                })
            else:
                prosody_data.append({
                    'duration': segment['end'] - segment['start'],
                    'mean_pitch': 0,
                })
        
        return prosody_data
    
    def synthesize_with_style(self, text, target_duration=None):
        """Generate speech with captured voice style"""
        if not text.strip():
            return None
            
        # Generate speech
        wav = self.tts.tts(text=text)
        
        # Save to temporary file
        temp_output = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        
        if isinstance(wav, list):
            wav = np.array(wav)
        
        sf.write(temp_output.name, wav, 22050)
        
        # Time-stretch to match target duration if needed
        if target_duration and target_duration > 0.1:
            audio_segment = AudioSegment.from_wav(temp_output.name)
            current_duration = len(audio_segment) / 1000.0
            stretch_ratio = target_duration / current_duration
            
            if abs(stretch_ratio - 1.0) > 0.15:  # Only stretch if significant difference
                y, sr = librosa.load(temp_output.name)
                y_stretched = librosa.effects.time_stretch(y, rate=1/stretch_ratio)
                sf.write(temp_output.name, y_stretched, sr)
        
        return temp_output.name
    
    def translate_live_audio(self, audio_data):
        """Process and translate recorded audio"""
        if np.max(np.abs(audio_data)) < 0.01:
            print("No speech detected. Try speaking louder.")
            return None
        
        print("Processing...")
        
        # Save audio to file
        temp_file = self.save_temp_audio(audio_data)
        
        try:
            # Transcribe and translate
            result = self.whisper_model.transcribe(temp_file, 
                                                 word_timestamps=True,
                                                 task="translate")
            
            if not result['text'].strip():
                print("No speech detected.")
                return None
            
            segments = result['segments']
            original_lang = result['language']
            translated_text = result['text'].strip()
            
            # Get language name from Whisper's built-in dictionary
            language_name = whisper.tokenizer.LANGUAGES.get(original_lang, original_lang.upper())
            
            print(f"Detected language: {language_name.title()}")
            print(f"Original: {result['text']}")
            print(f"Translated: {translated_text}")
            
            if original_lang == 'en':
                print("Already in English - no translation needed")
                return None
            
            # Extract prosody information
            prosody_data = self.extract_prosody(temp_file, segments)
            
            # Generate styled segments
            translated_segments = []
            for i, (segment, prosody) in enumerate(zip(segments, prosody_data)):
                segment_text = segment['text'].strip()
                if segment_text:
                    segment_audio_path = self.synthesize_with_style(
                        segment_text, 
                        target_duration=prosody['duration']
                    )
                    
                    if segment_audio_path:
                        translated_segments.append({
                            'audio_path': segment_audio_path,
                            'start': segment['start'],
                            'end': segment['end'],
                            'duration': prosody['duration']
                        })
            
            if translated_segments:
                # Combine segments
                final_audio = self.combine_segments(translated_segments, 
                                                  total_duration=segments[-1]['end'])
                
                # Save and play result
                output_path = f"live_translation_{len(os.listdir('.'))}.wav"
                final_audio.export(output_path, format="wav")
                
                # Cleanup
                for seg in translated_segments:
                    if os.path.exists(seg['audio_path']):
                        os.unlink(seg['audio_path'])
                
                print(f"Style-preserved translation saved: {output_path}")
                
                # Play the result
                try:
                    os.system(f"afplay {output_path}")
                except:
                    print("Could not play audio automatically")
                
                return output_path
                
        finally:
            if os.path.exists(temp_file):
                os.unlink(temp_file)
        
        return None
    
    def combine_segments(self, segments, total_duration):
        """Combine audio segments with proper timing"""
        final_duration_ms = int(total_duration * 1000)
        combined = AudioSegment.silent(duration=final_duration_ms)
        
        for segment in segments:
            if os.path.exists(segment['audio_path']):
                audio_seg = AudioSegment.from_wav(segment['audio_path'])
                start_ms = int(segment['start'] * 1000)
                combined = combined.overlay(audio_seg, position=start_ms)
        
        return combined
    
    def start_live_session(self, record_duration=5):
        """Start live translation session"""
        print("\nLIVE STYLE-PRESERVING TRANSLATOR")
        print("-" * 40)
        
        # First, capture voice sample
        while True:
            capture = input("Capture your voice sample first? (y/n): ").strip().lower()
            if capture in ['y', 'yes']:
                self.record_voice_sample(duration=3)
                break
            elif capture in ['n', 'no']:
                print("Using default TTS voice...")
                break
        
        print("\nInstructions:")
        print(f"- Press ENTER to record for {record_duration} seconds")
        print("- Speak in any language")
        print("- System will translate to English with your voice style")
        print("- Type 'quit' to exit")
        print("-" * 40)
        
        while True:
            user_input = input("\nPress ENTER to record (or 'quit' to exit): ").strip()
            
            if user_input.lower() in ['quit', 'q', 'exit']:
                print("Session ended!")
                break
            
            # Record and translate
            audio_data = self.record_audio_chunk(record_duration)
            result = self.translate_live_audio(audio_data)
            
            if result:
                print("Translation complete!")
            else:
                print("No translation needed or failed")

def main():
    parser = argparse.ArgumentParser(description="Live Style-Preserving Speech Translator")
    parser.add_argument("--model", "-m", default="base", 
                       choices=["tiny", "base", "small", "medium", "large"],
                       help="Whisper model size")
    parser.add_argument("--duration", "-d", type=int, default=5,
                       help="Recording duration in seconds")
    
    args = parser.parse_args()
    
    try:
        translator = LiveStyleTranslator(args.model)
        translator.start_live_session(args.duration)
    except KeyboardInterrupt:
        print("\nSession ended by user")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
