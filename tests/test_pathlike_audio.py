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
