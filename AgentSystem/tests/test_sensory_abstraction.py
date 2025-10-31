"""Tests for hardware abstraction in the sensory input module."""

import importlib.util
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

MODULE_PATH = Path(__file__).resolve().parents[1] / "modules" / "sensory_input.py"
spec = importlib.util.spec_from_file_location("sensory_input_module", MODULE_PATH)
sensory_input = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(sensory_input)

AudioCaptureBackend = sensory_input.AudioCaptureBackend
AudioProcessor = sensory_input.AudioProcessor
SensoryInputModule = sensory_input.SensoryInputModule
VideoCaptureBackend = sensory_input.VideoCaptureBackend
VideoProcessor = sensory_input.VideoProcessor


class DummyAudioBackend(AudioCaptureBackend):
    """Synthetic audio backend that emits a few deterministic chunks."""

    def __init__(self, iterations: int = 60) -> None:
        self.iterations = iterations
        self._running = False
        self.sample_rate = 16000

    @property
    def is_available(self) -> bool:
        return True

    def start(self, sample_rate: int, chunk_size: int, device_index: Optional[int] = None) -> bool:
        self.sample_rate = sample_rate
        self._running = True
        self._remaining = self.iterations
        self._chunk = (b"\x00\x01" * (chunk_size // 2)) or b"\x00\x01"
        return True

    def read_chunk(self, chunk_size: int) -> bytes:
        if not self._running:
            return b""
        if self._remaining <= 0:
            time.sleep(0.01)
            return b""
        self._remaining -= 1
        return self._chunk

    def stop(self) -> None:
        self._running = False

    def list_devices(self) -> List[Dict[str, Any]]:
        return [
            {
                "index": 0,
                "name": "dummy-audio",
                "channels": 1,
                "sample_rate": self.sample_rate,
            }
        ]


class DummyVideoBackend(VideoCaptureBackend):
    """Synthetic video backend that emits dictionary frames."""

    def __init__(self, frames: Optional[List[Dict[str, Any]]] = None) -> None:
        self.frames = frames or [{"frame": 1}, {"frame": 2}]
        self._running = False
        self._cursor = 0

    @property
    def is_available(self) -> bool:
        return True

    def start(self, camera_index: int, width: int, height: int, fps: int) -> bool:
        self._running = True
        self._cursor = 0
        return True

    def read_frame(self) -> Optional[Any]:
        if not self._running or self._cursor >= len(self.frames):
            time.sleep(0.01)
            return None
        frame = self.frames[self._cursor]
        self._cursor += 1
        return frame

    def stop(self) -> None:
        self._running = False

    def list_cameras(self) -> List[Dict[str, Any]]:
        return [{"index": 0, "name": "dummy-video"}]


def _drain_events(fetcher, timeout: float = 0.5) -> List[Dict[str, Any]]:
    events: List[Dict[str, Any]] = []
    start = time.time()
    while time.time() - start < timeout:
        event = fetcher()
        if event:
            events.append(event)
        else:
            time.sleep(0.01)
    return events


def test_audio_processor_with_dummy_backend() -> None:
    backend = DummyAudioBackend()
    processor = AudioProcessor(capture_backend=backend)
    assert processor.is_available

    assert processor.start_recording()
    time.sleep(0.2)
    processor.stop_recording()
    time.sleep(0.05)

    events = _drain_events(lambda: processor.get_next_audio_event(timeout=0.01))
    assert events, "Expected at least one audio event from dummy backend"
    assert all("timestamp" in event for event in events)


def test_video_processor_with_dummy_backend() -> None:
    backend = DummyVideoBackend()
    processor = VideoProcessor(capture_backend=backend)
    assert processor.capture_backend is backend

    assert processor.start_capture()
    time.sleep(0.1)
    processor.stop_capture()

    events = _drain_events(lambda: processor.get_next_video_event(timeout=0.01))
    assert events, "Expected at least one video event from dummy backend"
    assert all(event.get("type") == "video_frame" for event in events)


def test_sensory_module_accepts_synthetic_backends() -> None:
    module = SensoryInputModule()
    module.configure_audio_backend(DummyAudioBackend())
    module.configure_video_backend(DummyVideoBackend())

    audio_info = module.check_audio_availability()
    video_info = module.check_video_availability()

    assert audio_info["available"]
    assert video_info["available"]

    module.register_event_callback(lambda event: None)
    module.start_event_processing()
    module.add_test_event({"type": "synthetic"})
    time.sleep(0.1)
    module.stop_event_processing()

    latest = module.get_latest_sensory_events()
    assert latest["count"] >= 1
    module.shutdown()


def test_multimodal_context_generation() -> None:
    module = SensoryInputModule()
    module.configure_audio_backend(DummyAudioBackend(iterations=5))
    module.configure_video_backend(DummyVideoBackend(frames=[{"frame": 1}]))

    module._latest_audio_event = {"type": "speech", "raw_data": b"hello"}
    module._latest_video_event = {"type": "video_frame", "objects": ["pool"], "frame_shape": (64, 64, 3)}
    module._try_fuse_events()

    context = module.get_multimodal_context()
    assert context["success"]
    assert context["fused"]["embedding"], "Expected fused embedding values"
    module.shutdown()
