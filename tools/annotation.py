# Copyright (c) 2025 SparkAudio
#               2025 Xinsheng Wang (w.xinshawn@gmail.com)

from typing import Dict

from tools.utils.audio import load_audio
from tools.utils.audio import remove_silence_on_both_ends
from tools.audio_signal.pitch import get_pitch_mean_and_std
from tools.text.text2syllable import text2syllables


def Annotation(
    audio_path: str, sample_rate: int, text: str,
) -> Dict[any, any]:

    # get speech signal related info
    audio = load_audio(audio_path, sampling_rate=sample_rate, volume_normalize=True)
    duration = len(audio) / sample_rate
    audio = remove_silence_on_both_ends(audio, sample_rate)
    speech_duration = len(audio) / sample_rate
    pitch, pitch_std = get_pitch_mean_and_std(audio, sample_rate)
    
    # get speaking rate
    syllable_info = text2syllables(text)
    normalized_text = syllable_info['normalized_text']
    syllable_num = syllable_info['syllable_num']
    syllables = syllable_info['syllables']
    speed = syllable_num / speech_duration

    return {
        "pitch": pitch,
        "pitch_std": pitch_std,
        "speed": speed,
        "duration": duration,
        "speech_duration": speech_duration,
        "syllable_num": syllable_num,
        "text": text,
        "normalized_text": normalized_text,
        "syllables": syllables,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--audio_path', type=str, default='/aifs4su/xinshengwang/data/speech/17_Librispeech_SLR12/LibriSpeech/test/wavs/908-31957-0025.wav')
    parser.add_argument('--text', type=str, default='I LOVE THEE WITH A LOVE I SEEMED TO LOSE WITH MY LOST SAINTS I LOVE THEE WITH THE BREATH SMILES TEARS OF ALL MY LIFE AND IF GOD CHOOSE I SHALL BUT LOVE THEE BETTER AFTER DEATH.')
    args = parser.parse_args()
    meta = Annotation(args.audio_path, 16000, args.text)
    print(meta)
