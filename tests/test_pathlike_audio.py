import numpy as np
import torch

import whisper.audio as audio


def test_log_mel_spectrogram_accepts_pathlike(monkeypatch, tmp_path):
    fake_audio_path = tmp_path / "fake.flac"

    called = {"ok": False}

    def fake_run(cmd, capture_output, check):
        called["ok"] = True
        # Produce 1 second of silence at the expected sample rate.
        samples = np.zeros(audio.SAMPLE_RATE, dtype=np.int16)

        class Result:
            stdout = samples.tobytes()

        return Result()

    monkeypatch.setattr(audio, "run", fake_run)

    mel = audio.log_mel_spectrogram(fake_audio_path)

    assert called["ok"] is True
    assert isinstance(mel, torch.Tensor)
    assert mel.shape[0] == 80


def test_load_audio_accepts_pathlike_and_fspaths(monkeypatch, tmp_path):
    fake_audio_path = tmp_path / "fake.flac"

    def fake_run(cmd, capture_output, check):
        # The input path should be normalized to a string via os.fspath.
        input_index = cmd.index("-i") + 1
        assert isinstance(cmd[input_index], str)

        samples = np.zeros(audio.SAMPLE_RATE, dtype=np.int16)

        class Result:
            stdout = samples.tobytes()

        return Result()

    monkeypatch.setattr(audio, "run", fake_run)

    wav = audio.load_audio(fake_audio_path)
    assert isinstance(wav, np.ndarray)
    assert wav.dtype == np.float32
