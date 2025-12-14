import numpy as np
from scipy.io.wavfile import write
import os
import subprocess
import shutil


def midi_to_freq(midi_number):
    """Converts MIDI note number to frequency in Hz."""
    return 440.0 * (2.0 ** ((midi_number - 69) / 12.0))

def generate_sine_wave(frequency, duration, sample_rate=44100, amplitude=0.5):
    """Generates a sine wave for a given frequency and duration."""
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    wave = amplitude * np.sin(2 * np.pi * frequency * t)
    return wave

def save_melody_as_mp3(melody, filename="output.mp3", bpm=90):
    """
    Synthesizes a melody and saves it as an MP3 file.
    Falls back to WAV if MP3 conversion fails (e.g. missing ffmpeg).
    """
    sample_rate = 44100
    audio_data = []
    
    beat_duration = 60.0 / bpm # Seconds per beat
    
    print(f"Generating audio for melody with {len(melody.notes)} notes...")

    for note in melody.notes:
        freq = midi_to_freq(note.pitch)
        duration_sec = note.duration * beat_duration
        
        # Generate wave
        wave = generate_sine_wave(freq, duration_sec, sample_rate)
        
        # Apply simple envelope (fade in/out) to avoid clicks
        envelope_len = int(0.01 * sample_rate) # 10ms
        if len(wave) > 2 * envelope_len:
            envelope = np.ones_like(wave)
            envelope[:envelope_len] = np.linspace(0, 1, envelope_len)
            envelope[-envelope_len:] = np.linspace(1, 0, envelope_len)
            wave = wave * envelope
            
        audio_data.append(wave)
        
    if not audio_data:
        print("No notes to generate.")
        return
        
    combined_wave = np.concatenate(audio_data)
    
    # Normalize to 16-bit PCM
    max_val = np.max(np.abs(combined_wave))
    if max_val > 0:
        combined_wave = combined_wave * 32767 / max_val
    combined_wave = combined_wave.astype(np.int16)
    
    # Save as temporary WAV
    wav_filename = filename.replace(".mp3", ".wav")
    if wav_filename == filename:
        wav_filename = filename + ".wav"
        
    write(wav_filename, sample_rate, combined_wave)
    print(f"Saved temporary WAV file: {wav_filename}")
    
    # Convert to MP3
    try:
        # Use ffmpeg to convert WAV to MP3
        subprocess.run(
            ["ffmpeg", "-y", "-i", wav_filename, "-c:a", "libmp3lame", "-b:a", "320k", filename],
            check=True
        )
        print(f"Successfully converted and saved melody to {filename}")
        # Clean up wav
        if os.path.exists(wav_filename):
            os.remove(wav_filename)
            print(f"Removed temporary WAV file.")
    except Exception as e:
        print(f"Could not convert to MP3 (ffmpeg might be missing or not in PATH).")
        print(f"The melody is available as a WAV file: {wav_filename}")
        print(f"Error details: {e}")
