import librosa
import numpy as np


class Augmentation:
    AUG_NOISE = 'noise'
    AUG_SHIFT = 'shift'
    AUG_PITCH = 'pitch'
    AUG_STRETCH = 'stretch'

    augmentation_styles = [AUG_NOISE, AUG_SHIFT, AUG_PITCH, AUG_STRETCH]

    aug_combinations = [
        [AUG_NOISE],
        [AUG_SHIFT],
        [AUG_PITCH],
        [AUG_STRETCH],

        [AUG_NOISE, AUG_SHIFT],
        [AUG_NOISE, AUG_PITCH],
        [AUG_NOISE, AUG_STRETCH],
        [AUG_SHIFT, AUG_PITCH],
        [AUG_SHIFT, AUG_STRETCH],
        [AUG_PITCH, AUG_STRETCH],

        [AUG_NOISE, AUG_SHIFT, AUG_PITCH],
        [AUG_NOISE, AUG_SHIFT, AUG_STRETCH],
        [AUG_NOISE, AUG_PITCH, AUG_STRETCH],
        [AUG_PITCH, AUG_STRETCH, AUG_SHIFT],

        [AUG_NOISE, AUG_SHIFT, AUG_PITCH, AUG_STRETCH]
    ]

    @staticmethod
    def add_noise(data):
        """
        Add White Noise
        :param data:
        :return:
        """
        noise_amp = 0.005 * np.random.uniform() * np.amax(data)
        data = data.astype('float64') + noise_amp * np.random.normal(size=data.shape[0])
        return data

    @staticmethod
    def add_shift(data):
        """
        Add Random Shifting
        :param data:
        :return:
        """
        s_range = int(np.random.uniform(low=-5, high=5) * 500)
        return np.roll(data, s_range)

    @staticmethod
    def add_pitch(data, sample_rate):
        """
        Add Random Pitch
        :param data:
        :param sample_rate:
        :return:
        """
        bins_per_octave = 12
        pitch_pm = 3
        pitch_change = pitch_pm * 2 * (np.random.uniform(-1, 1))
        data = librosa.effects.pitch_shift(data.astype('float64'),
                                           sample_rate, n_steps=pitch_change,
                                           bins_per_octave=bins_per_octave
                                           )
        return data

    @staticmethod
    def add_stretch(data, min_stretch=0.5, max_stretch=2):
        """
        Add positive or negative stretch
        :param max_stretch:
        :param min_stretch:
        :param data:
        :return:
        """
        return librosa.effects.time_stretch(data, np.random.uniform(min_stretch, max_stretch))
