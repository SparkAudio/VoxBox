# Copyright (c) 2025 SparkAudio
#               2025 Xinsheng Wang (w.xinshawn@gmail.com)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Caculate pitch with pyworld
"""
import time
import numpy as np
import pyworld
import librosa
import scipy.signal
from typing import Tuple


def caculate_pitch_with_pyworld(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    hop_length = int(sample_rate * 0.005)
    win_length = hop_length * 6
    frame_period = hop_length / sample_rate * 1000
    audio = audio.astype(np.float64)
    time1 = time.time()
    f01, t1 = pyworld.dio(audio, sample_rate, frame_period=frame_period)
    time2 = time.time()
    f01 = pyworld.stonemask(audio, f01, t1, sample_rate)
    time3 = time.time()
    f02, t2 = pyworld.harvest(audio, sample_rate, frame_period=frame_period)
    time4 = time.time()
    zcr = librosa.feature.zero_crossing_rate(
        audio, frame_length=win_length, hop_length=hop_length
    )
    zcr = scipy.signal.savgol_filter(zcr[0], 11, 5, mode="nearest")
    time5 = time.time()
    print("dio", time2 - time1)
    print("sto", time3 - time2)
    print("har", time4 - time3)
    print("zcr", time5 - time4)
    dio_v = (f01 > 0) * 1
    zcr_v = (zcr < 0.06) * 1
    voice = ((dio_v + zcr_v) > 0) * 1
    pitch = f02 * voice
    pitch[pitch < 0] = 0
    pitch[pitch > 2400] = 0
    pitch[pitch == np.nan] = 0
    return pitch


def caculate_pitch_with_pyworld_simple(
    audio: np.ndarray, sample_rate: int
) -> np.ndarray:
    hop_length = int(sample_rate * 0.005)
    win_length = hop_length * 6
    frame_period = hop_length / sample_rate * 1000
    audio = audio.astype(np.float64)
    f01, t1 = pyworld.dio(audio, sample_rate, frame_period=frame_period)
    pitch = pyworld.stonemask(audio, f01, t1, sample_rate)

    pitch[pitch < 50] = 0
    pitch[pitch > 2400] = 0
    pitch[pitch == np.nan] = 0
    return pitch


def normalize_pitch(pitch: np.ndarray) -> np.ndarray:
    pitch = pitch[pitch > 0]
    max_pitch = np.max(pitch)
    min_pitch = np.min(pitch)
    if max_pitch == min_pitch or max_pitch == 0:
        return pitch
    norm_pitch = (pitch - min_pitch) / (max_pitch - min_pitch)
    return norm_pitch


def get_pitch_mean_and_std(audio: np.ndarray, sample_rate: int) -> Tuple[float, float]:
    pitch = caculate_pitch_with_pyworld_simple(audio, sample_rate)
    pitch = pitch[pitch > 0]
    if len(pitch) == 0:
        return 0, 0
    mean = np.mean(pitch)
    norm_pitch = normalize_pitch(pitch)
    norm_std = np.std(norm_pitch)
    return mean, norm_std


if __name__ == "__main__":
    audio, sample_rate = librosa.load(
        "example/audios/m3ed/m3ed_Angry_0000000014.wav", sr=16000
    )

    start = time.time()
    pitch_mean, norm_std = get_pitch_mean_and_std(audio, sample_rate)
    end = time.time()

    print("time", end - start)
    print(pitch_mean, norm_std)
