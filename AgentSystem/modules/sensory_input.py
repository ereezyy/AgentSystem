"""
Sensory Input Module
-------------------
Processes audio, video, and other sensory inputs for agent awareness
"""

import os
import time
import threading
import queue
import json
import base64
from collections import deque
from typing import Dict, List, Any, Optional, Callable, Union, Iterable
from datetime import datetime

# Local imports
from AgentSystem.utils.logger import get_logger

# Get module logger
logger = get_logger("modules.sensory_input")

try:
    import cv2  # type: ignore
    import numpy as np  # type: ignore
    import pyaudio  # type: ignore
    import speech_recognition as sr  # type: ignore
    from PIL import Image  # type: ignore
    SENSORY_IMPORTS_AVAILABLE = True
except ImportError:
    cv2 = None  # type: ignore
    np = None  # type: ignore
    pyaudio = None  # type: ignore
    sr = None  # type: ignore
    Image = None  # type: ignore
    logger.warning(
        "Sensory input dependencies not available. Install with: pip install opencv-python numpy pyaudio SpeechRecognition pillow"
    )
    SENSORY_IMPORTS_AVAILABLE = False


class AudioCaptureBackend:
    """Interface for audio capture implementations."""

    @property
    def is_available(self) -> bool:
        return False

    def start(self, sample_rate: int, chunk_size: int, device_index: Optional[int] = None) -> bool:
        """Start the capture stream."""
        return False

    def read_chunk(self, chunk_size: int) -> bytes:
        """Read a chunk of audio data."""
        return b""

    def stop(self) -> None:
        """Stop the capture stream."""

    def list_devices(self) -> List[Dict[str, Any]]:
        """Return available capture devices."""
        return []

    def shutdown(self) -> None:
        """Release backend resources."""
        self.stop()


class VideoCaptureBackend:
    """Interface for video capture implementations."""

    @property
    def is_available(self) -> bool:
        return False

    def start(self, camera_index: int, width: int, height: int, fps: int) -> bool:
        """Open the capture stream."""
        return False

    def read_frame(self) -> Optional[Any]:
        """Read a frame from the stream."""
        return None

    def stop(self) -> None:
        """Stop the capture stream."""

    def list_cameras(self) -> List[Dict[str, Any]]:
        """Return available camera descriptions."""
        return []


class SyntheticAudioBackend(AudioCaptureBackend):
    """Feed audio data from a predefined iterable for simulations."""

    def __init__(self, chunks: Optional[Iterable[bytes]] = None) -> None:
        self._chunks = deque(chunks or [])
        self._active = False

    @property
    def is_available(self) -> bool:
        return True

    def start(self, sample_rate: int, chunk_size: int, device_index: Optional[int] = None) -> bool:
        self._active = True
        return True

    def read_chunk(self, chunk_size: int) -> bytes:
        if not self._active or not self._chunks:
            return b""
        return self._chunks.popleft()

    def stop(self) -> None:
        self._active = False


class SyntheticVideoBackend(VideoCaptureBackend):
    """Provide synthetic frames for testing or simulation feeds."""

    def __init__(self, frames: Optional[Iterable[Any]] = None) -> None:
        self._frames = deque(frames or [])
        self._active = False

    @property
    def is_available(self) -> bool:
        return True

    def start(self, camera_index: int, width: int, height: int, fps: int) -> bool:
        self._active = True
        return True

    def read_frame(self) -> Optional[Any]:  # type: ignore[override]
        if not self._active or not self._frames:
            return None
        return self._frames.popleft()

    def stop(self) -> None:
        self._active = False


class MultimodalFusionEngine:
    """Fuse vision, audio, and optional text into a shared embedding."""

    def __init__(self) -> None:
        self._history: deque = deque(maxlen=100)

    def fuse(
        self,
        audio_event: Optional[Dict[str, Any]] = None,
        video_event: Optional[Dict[str, Any]] = None,
        text: Optional[str] = None,
    ) -> Dict[str, Any]:
        features: Dict[str, Any] = {"timestamp": time.time()}
        if audio_event:
            raw = audio_event.get("raw_data")
            features["audio_energy"] = len(raw) if isinstance(raw, (bytes, bytearray)) else 0
            features["audio_label"] = audio_event.get("type")
        if video_event:
            features["visual_objects"] = video_event.get("objects", [])
            features["frame_shape"] = video_event.get("frame_shape")
        if text:
            features["text"] = text
        features["embedding"] = self._build_signature(features)
        self._history.append(features)
        return features

    def _build_signature(self, features: Dict[str, Any]) -> List[float]:
        signature = [0.0, 0.0, 0.0]
        if "audio_energy" in features:
            signature[0] = min(1.0, features["audio_energy"] / 10000.0)
        if features.get("visual_objects"):
            signature[1] = min(1.0, len(features["visual_objects"]))
        if features.get("text"):
            signature[2] = min(1.0, len(str(features["text"])) / 200.0)
        return signature

    def recent_history(self) -> List[Dict[str, Any]]:
        return list(self._history)


class CrossModalReasoner:
    """Perform lightweight reasoning across fused sensory channels."""

    def __init__(self, fusion_engine: MultimodalFusionEngine) -> None:
        self.fusion_engine = fusion_engine

    def infer_context(self, fused_event: Dict[str, Any]) -> Dict[str, Any]:
        audio_energy = fused_event.get("audio_energy", 0)
        objects = fused_event.get("visual_objects", [])
        context = "unknown"
        if audio_energy and objects:
            if "water" in objects or "pool" in objects:
                context = "water_scene" if audio_energy > 0 else "still_water"
            elif "person" in objects:
                context = "conversation" if audio_energy > 0 else "observation"
        elif objects:
            context = "visual_only"
        elif audio_energy:
            context = "audio_only"
        return {"context": context, "confidence": 0.6 if context != "unknown" else 0.2}


class PyAudioCaptureBackend(AudioCaptureBackend):
    """Hardware audio capture implementation using PyAudio."""

    def __init__(self) -> None:
        self._audio = pyaudio.PyAudio() if SENSORY_IMPORTS_AVAILABLE and pyaudio else None
        self._stream = None

    @property
    def is_available(self) -> bool:
        return bool(self._audio)

    def start(self, sample_rate: int, chunk_size: int, device_index: Optional[int] = None) -> bool:
        if not self._audio:
            return False
        self._stream = self._audio.open(
            format=pyaudio.paInt16,  # type: ignore[attr-defined]
            channels=1,
            rate=sample_rate,
            input=True,
            frames_per_buffer=chunk_size,
            input_device_index=device_index,
        )
        return True

    def read_chunk(self, chunk_size: int) -> bytes:
        if not self._stream:
            return b""
        return self._stream.read(chunk_size, exception_on_overflow=False)

    def stop(self) -> None:
        if self._stream:
            try:
                self._stream.stop_stream()
                self._stream.close()
            finally:
                self._stream = None

    def list_devices(self) -> List[Dict[str, Any]]:
        devices: List[Dict[str, Any]] = []
        if not self._audio:
            return devices

        for i in range(self._audio.get_device_count()):
            device_info = self._audio.get_device_info_by_index(i)
            if device_info.get('maxInputChannels', 0) > 0:
                devices.append({
                    'index': i,
                    'name': device_info.get('name'),
                    'channels': device_info.get('maxInputChannels'),
                    'sample_rate': int(device_info.get('defaultSampleRate', 0)),
                })
        return devices

    def shutdown(self) -> None:
        self.stop()
        if self._audio:
            self._audio.terminate()
            self._audio = None


class OpenCVCaptureBackend(VideoCaptureBackend):
    """Hardware video capture implementation using OpenCV."""

    def __init__(self) -> None:
        self._cap = None

    @property
    def is_available(self) -> bool:
        return cv2 is not None

    def start(self, camera_index: int, width: int, height: int, fps: int) -> bool:
        if cv2 is None:
            return False

        self._cap = cv2.VideoCapture(camera_index)
        if not self._cap or not self._cap.isOpened():
            self._cap = None
            return False

        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        if fps > 0:
            self._cap.set(cv2.CAP_PROP_FPS, fps)
        return True

    def read_frame(self) -> Optional[Any]:
        if not self._cap:
            return None
        ret, frame = self._cap.read()
        if not ret:
            return None
        return frame

    def stop(self) -> None:
        if self._cap:
            self._cap.release()
            self._cap = None

    def list_cameras(self) -> List[Dict[str, Any]]:
        cameras: List[Dict[str, Any]] = []
        if cv2 is None:
            return cameras

        for i in range(10):
            cap = cv2.VideoCapture(i)
            try:
                if cap.isOpened():
                    cameras.append({
                        'index': i,
                        'name': f"Camera {i}",
                        'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                        'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                        'fps': int(cap.get(cv2.CAP_PROP_FPS)),
                    })
            finally:
                cap.release()
        return cameras


class AudioProcessor:
    """Processes audio input from microphone"""

    def __init__(
        self,
        sample_rate: int = 16000,
        chunk_size: int = 1024,
        capture_backend: Optional[AudioCaptureBackend] = None,
    ):
        """
        Initialize the audio processor
        
        Args:
            sample_rate: Audio sample rate in Hz
            chunk_size: Number of audio frames per buffer
        """
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.audio_queue = queue.Queue()
        self.is_recording = False
        self.recording_thread = None

        # Speech recognition
        self.recognizer = sr.Recognizer() if SENSORY_IMPORTS_AVAILABLE and sr else None

        self.capture_backend: Optional[AudioCaptureBackend] = capture_backend
        if self.capture_backend is None and SENSORY_IMPORTS_AVAILABLE and pyaudio:
            self.capture_backend = PyAudioCaptureBackend()

    @property
    def is_available(self) -> bool:
        return bool(self.capture_backend and self.capture_backend.is_available)

    @property
    def devices(self) -> List[Dict[str, Any]]:
        if not self.capture_backend:
            return []
        return self.capture_backend.list_devices()

    def use_backend(self, backend: Optional[AudioCaptureBackend]) -> None:
        """Swap the capture backend implementation."""
        if self.capture_backend and self.capture_backend is not backend:
            self.capture_backend.shutdown()
        self.capture_backend = backend
        
    def _get_audio_devices(self) -> List[Dict[str, Any]]:
        """Get available audio input devices"""
        devices = []
        if not SENSORY_IMPORTS_AVAILABLE:
            return devices
            
        for i in range(self.audio.get_device_count()):
            device_info = self.audio.get_device_info_by_index(i)
            if device_info.get('maxInputChannels') > 0:
                devices.append({
                    'index': i,
                    'name': device_info.get('name'),
                    'channels': device_info.get('maxInputChannels'),
                    'sample_rate': int(device_info.get('defaultSampleRate'))
                })
        return devices
    
    def start_recording(self, device_index: Optional[int] = None) -> bool:
        """
        Start recording audio
        
        Args:
            device_index: Index of audio device to use (default: system default)
            
        Returns:
            Success flag
        """
        if not self.is_available:
            logger.error("No audio capture backend configured")
            return False

        if self.is_recording:
            logger.warning("Already recording")
            return False

        try:
            self.is_recording = True
            self.recording_thread = threading.Thread(
                target=self._record_audio_thread,
                args=(device_index,),
                daemon=True
            )
            self.recording_thread.start()
            logger.info("Started audio recording")
            return True
        except Exception as e:
            logger.error(f"Error starting audio recording: {e}")
            self.is_recording = False
            return False
    
    def stop_recording(self) -> bool:
        """
        Stop recording audio
        
        Returns:
            Success flag
        """
        if not self.is_recording:
            logger.warning("Not recording")
            return False
        
        self.is_recording = False
        if self.recording_thread:
            self.recording_thread.join(timeout=2.0)
            self.recording_thread = None

        if self.capture_backend:
            self.capture_backend.stop()

        logger.info("Stopped audio recording")
        return True
    
    def _record_audio_thread(self, device_index: Optional[int] = None) -> None:
        """
        Thread function for continuous audio recording
        
        Args:
            device_index: Index of audio device to use
        """
        if not self.capture_backend:
            logger.error("No audio capture backend available")
            self.is_recording = False
            return

        if not self.capture_backend.start(self.sample_rate, self.chunk_size, device_index=device_index):
            logger.error("Failed to start audio backend stream")
            self.is_recording = False
            return

        # Buffer to accumulate audio chunks
        audio_buffer: List[bytes] = []
        buffer_duration_sec = 0.0
        target_duration_sec = 3  # Process in 3-second chunks

        try:
            while self.is_recording:
                try:
                    data = self.capture_backend.read_chunk(self.chunk_size)
                    if not data:
                        time.sleep(0.01)
                        continue

                    audio_buffer.append(data)
                    buffer_duration_sec += self.chunk_size / self.sample_rate

                    if buffer_duration_sec >= target_duration_sec:
                        threading.Thread(
                            target=self._process_audio_chunk,
                            args=(b''.join(audio_buffer),),
                            daemon=True,
                        ).start()

                        audio_buffer = []
                        buffer_duration_sec = 0.0

                except Exception as e:
                    logger.error(f"Error recording audio: {e}")
                    time.sleep(0.1)
        finally:
            self.capture_backend.stop()
    
    def _process_audio_chunk(self, audio_data: bytes) -> None:
        """
        Process an audio chunk for speech recognition

        Args:
            audio_data: Raw audio data
        """
        try:
            if not self.recognizer or sr is None:
                self._queue_raw_audio_event(audio_data)
                return

            # Convert audio data to AudioData for speech recognition
            audio = sr.AudioData(audio_data, self.sample_rate, 2)  # 2 bytes per sample (16-bit)

            # Try to recognize speech
            try:
                text = self.recognizer.recognize_google(audio)
                if text:
                    logger.debug(f"Recognized speech: {text}")

                    # Add to queue
                    self.audio_queue.put({
                        'type': 'speech',
                        'text': text,
                        'timestamp': datetime.now().isoformat(),
                        'confidence': 0.8  # Estimated confidence
                    })
            except sr.UnknownValueError:
                # No speech detected, add audio features instead
                self._extract_audio_features(audio_data)
            except Exception as e:
                logger.error(f"Speech recognition error: {e}")
                self._queue_raw_audio_event(audio_data)

        except Exception as e:
            logger.error(f"Error processing audio chunk: {e}")
            self._queue_raw_audio_event(audio_data)

    def _extract_audio_features(self, audio_data: bytes) -> None:
        """
        Extract features from audio data when speech isn't detected

        Args:
            audio_data: Raw audio data
        """
        try:
            if np is None:
                self._queue_raw_audio_event(audio_data)
                return

            # Convert to numpy array for processing
            audio_np = np.frombuffer(audio_data, dtype=np.int16)

            # Calculate basic audio features
            if len(audio_np) > 0:
                rms = np.sqrt(np.mean(np.square(audio_np.astype(np.float32))))
                peak = np.max(np.abs(audio_np))
                zero_crossings = np.sum(np.diff(np.signbit(audio_np)))
                
                # Only queue if there's significant audio (not silence)
                if rms > 500:  # Arbitrary threshold, adjust as needed
                    self.audio_queue.put({
                        'type': 'audio_features',
                        'timestamp': datetime.now().isoformat(),
                        'features': {
                            'rms': float(rms),
                            'peak': int(peak),
                            'zero_crossings': int(zero_crossings)
                        }
                    })
        except Exception as e:
            logger.error(f"Error extracting audio features: {e}")
            self._queue_raw_audio_event(audio_data)

    def _queue_raw_audio_event(self, audio_data: bytes) -> None:
        """Add a raw audio event to the queue for downstream processing."""
        snippet = base64.b64encode(audio_data[: min(len(audio_data), self.chunk_size * 2)]).decode('utf-8') if audio_data else ""
        self.audio_queue.put({
            'type': 'audio_chunk',
            'timestamp': datetime.now().isoformat(),
            'sample_rate': self.sample_rate,
            'preview': snippet,
        })
    
    def get_next_audio_event(self, timeout: Optional[float] = 0.1) -> Optional[Dict[str, Any]]:
        """
        Get the next audio event from the queue
        
        Args:
            timeout: Timeout in seconds (None to block indefinitely)
            
        Returns:
            Audio event or None if queue is empty
        """
        try:
            return self.audio_queue.get(block=timeout is not None, timeout=timeout)
        except queue.Empty:
            return None

    def shutdown(self) -> None:
        """Release backend resources."""
        if self.capture_backend:
            self.capture_backend.shutdown()


class VideoProcessor:
    """Processes video input from webcam or other sources"""

    def __init__(
        self,
        width: int = 640,
        height: int = 480,
        fps: int = 5,
        capture_backend: Optional[VideoCaptureBackend] = None,
    ):
        """
        Initialize the video processor
        
        Args:
            width: Frame width
            height: Frame height
            fps: Target frames per second for processing
        """
        self.width = width
        self.height = height
        self.fps = fps
        self.frame_interval = 1.0 / fps
        
        self.video_queue = queue.Queue(maxsize=10)  # Limit queue size
        self.is_capturing = False
        self.capture_thread = None
        
        self.capture_backend: Optional[VideoCaptureBackend] = capture_backend
        if self.capture_backend is None and SENSORY_IMPORTS_AVAILABLE and cv2 is not None:
            self.capture_backend = OpenCVCaptureBackend()

        # Initialize face detection if OpenCV is available
        self.face_cascade = None
        if SENSORY_IMPORTS_AVAILABLE and cv2 is not None:
            try:
                # Load the pre-trained face cascade classifier
                self.face_cascade = cv2.CascadeClassifier(
                    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
                )
            except Exception as e:
                logger.error(f"Error initializing face detection: {e}")
    
    def list_cameras(self) -> List[Dict[str, Any]]:
        """
        List available camera devices
        
        Returns:
            List of camera information dictionaries
        """
        if not self.capture_backend:
            return []

        return self.capture_backend.list_cameras()
    
    def start_capture(self, camera_index: int = 0) -> bool:
        """
        Start capturing video
        
        Args:
            camera_index: Index of camera to use
            
        Returns:
            Success flag
        """
        if not self.capture_backend or not self.capture_backend.is_available:
            logger.error("No video capture backend configured")
            return False

        if self.is_capturing:
            logger.warning("Already capturing video")
            return False

        try:
            # Initialize camera
            if not self.capture_backend.start(camera_index, self.width, self.height, self.fps):
                logger.error(f"Failed to open camera {camera_index}")
                return False

            # Start capture thread
            self.is_capturing = True
            self.capture_thread = threading.Thread(
                target=self._capture_video_thread,
                daemon=True
            )
            self.capture_thread.start()
            
            logger.info(f"Started video capture from camera {camera_index}")
            return True
            
        except Exception as e:
            logger.error(f"Error starting video capture: {e}")
            if self.capture_backend:
                self.capture_backend.stop()
            self.is_capturing = False
            return False
    
    def stop_capture(self) -> bool:
        """
        Stop capturing video
        
        Returns:
            Success flag
        """
        if not self.is_capturing:
            logger.warning("Not capturing video")
            return False
        
        self.is_capturing = False
        if self.capture_thread:
            self.capture_thread.join(timeout=2.0)
            self.capture_thread = None
        
        if self.capture_backend:
            self.capture_backend.stop()
            
        logger.info("Stopped video capture")
        return True
    
    def _capture_video_thread(self) -> None:
        """Thread function for continuous video capture"""
        last_frame_time = 0
        
        while self.is_capturing:
            try:
                # Maintain target frame rate
                current_time = time.time()
                if current_time - last_frame_time < self.frame_interval:
                    time.sleep(0.001)  # Short sleep to prevent CPU spin
                    continue

                # Capture frame
                if not self.capture_backend:
                    logger.error("No video capture backend available during capture")
                    break

                frame = self.capture_backend.read_frame()
                if frame is None:
                    logger.error("Failed to capture frame")
                    time.sleep(0.1)  # Prevent tight loop on errors
                    continue
                
                # Process the frame in a separate thread
                threading.Thread(
                    target=self._process_frame,
                    args=(frame.copy() if hasattr(frame, 'copy') else frame,),  # Copy to prevent race conditions
                    daemon=True
                ).start()
                
                last_frame_time = time.time()
                
            except Exception as e:
                logger.error(f"Error in video capture: {e}")
                time.sleep(0.1)  # Prevent tight loop on errors

        if self.capture_backend:
            self.capture_backend.stop()

    def _process_frame(self, frame: Any) -> None:
        """
        Process a captured video frame

        Args:
            frame: Video frame to process
        """
        try:
            event: Dict[str, Any]
            if np is not None and cv2 is not None and hasattr(frame, "shape"):
                faces = []
                if self.face_cascade is not None:
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    detected_faces = self.face_cascade.detectMultiScale(
                        gray,
                        scaleFactor=1.1,
                        minNeighbors=5,
                        minSize=(30, 30)
                    )

                    for (x, y, w, h) in detected_faces:
                        faces.append({
                            'x': int(x),
                            'y': int(y),
                            'width': int(w),
                            'height': int(h)
                        })

                brightness = float(np.mean(frame))
                if len(getattr(frame, "shape", [])) >= 3 and frame.shape[2] == 3:
                    color_means = [
                        float(np.mean(frame[:, :, 0])),
                        float(np.mean(frame[:, :, 1])),
                        float(np.mean(frame[:, :, 2]))
                    ]
                else:
                    color_means = [brightness]

                thumbnail = cv2.resize(frame, (160, 120))
                _, jpeg_data = cv2.imencode('.jpg', thumbnail, [cv2.IMWRITE_JPEG_QUALITY, 70])
                thumbnail_b64 = base64.b64encode(jpeg_data).decode('utf-8')
                resolution = {
                    'width': int(frame.shape[1]),
                    'height': int(frame.shape[0])
                }
            else:
                faces = []
                color_means = []
                brightness = 0.0
                thumbnail_b64 = None
                resolution = {}

            event = {
                'type': 'video_frame',
                'timestamp': datetime.now().isoformat(),
                'features': {
                    'brightness': brightness,
                    'color_means': color_means,
                    'faces': faces,
                    'resolution': resolution
                },
                'thumbnail': thumbnail_b64
            }

            # Add to queue (non-blocking to prevent slowdowns)
            try:
                self.video_queue.put(event, block=False)
            except queue.Full:
                # If queue is full, remove oldest item and add new one
                try:
                    self.video_queue.get_nowait()
                    self.video_queue.put(event, block=False)
                except:
                    pass
                
        except Exception as e:
            logger.error(f"Error processing video frame: {e}")
    
    def get_next_video_event(self, timeout: Optional[float] = 0.1) -> Optional[Dict[str, Any]]:
        """
        Get the next video event from the queue

        Args:
            timeout: Timeout in seconds (None to block indefinitely)
            
        Returns:
            Video event or None if queue is empty
        """
        try:
            return self.video_queue.get(block=timeout is not None, timeout=timeout)
        except queue.Empty:
            return None

    def use_backend(self, backend: Optional[VideoCaptureBackend]) -> None:
        """Swap the capture backend implementation."""
        if self.capture_backend and self.capture_backend is not backend:
            self.capture_backend.stop()
        self.capture_backend = backend

    def shutdown(self) -> None:
        """Release backend resources."""
        if self.capture_backend:
            self.capture_backend.stop()


class SensoryInputModule:
    """Module for processing sensory inputs (audio, video, etc.)"""
    
    def __init__(self):
        """Initialize the sensory input module"""
        self.audio_processor = AudioProcessor()
        self.video_processor = VideoProcessor()
        
        # Callbacks for processing events
        self.event_callbacks = []
        
        # Event processing thread
        self.processing_thread = None
        self.is_processing = False

        # Event buffer for batch processing
        self.event_buffer = []
        self.buffer_lock = threading.Lock()

        # Multimodal grounding helpers
        self.fusion_engine = MultimodalFusionEngine()
        self.cross_modal_reasoner = CrossModalReasoner(self.fusion_engine)
        self._latest_audio_event: Optional[Dict[str, Any]] = None
        self._latest_video_event: Optional[Dict[str, Any]] = None
        self._fused_events: deque = deque(maxlen=50)
        
        logger.info(
            "Initialized SensoryInputModule (audio backend available: %s, video backend available: %s)",
            self.audio_processor.is_available,
            bool(self.video_processor.capture_backend and self.video_processor.capture_backend.is_available),
        )
    
    def get_tools(self) -> Dict[str, Any]:
        """Get tools provided by this module"""
        return {
            "start_audio_recording": {
                "description": "Start recording audio from microphone",
                "function": self.start_audio_recording,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "device_index": {
                            "type": "integer",
                            "description": "Index of audio device to use (optional)"
                        }
                    }
                }
            },
            "stop_audio_recording": {
                "description": "Stop recording audio",
                "function": self.stop_audio_recording,
                "parameters": {
                    "type": "object",
                    "properties": {}
                }
            },
            "list_audio_devices": {
                "description": "List available audio input devices",
                "function": self.list_audio_devices,
                "parameters": {
                    "type": "object",
                    "properties": {}
                }
            },
            "start_video_capture": {
                "description": "Start capturing video from webcam",
                "function": self.start_video_capture,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "camera_index": {
                            "type": "integer",
                            "description": "Index of camera to use",
                            "default": 0
                        }
                    }
                }
            },
            "stop_video_capture": {
                "description": "Stop capturing video",
                "function": self.stop_video_capture,
                "parameters": {
                    "type": "object",
                    "properties": {}
                }
            },
            "list_cameras": {
                "description": "List available camera devices",
                "function": self.list_cameras,
                "parameters": {
                    "type": "object",
                    "properties": {}
                }
            },
            "get_latest_sensory_events": {
                "description": "Get the latest sensory events (audio, video, etc.)",
                "function": self.get_latest_sensory_events,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "max_events": {
                            "type": "integer",
                            "description": "Maximum number of events to return",
                            "default": 10
                        }
                    }
                }
            },
            "start_event_processing": {
                "description": "Start continuous sensory event processing",
                "function": self.start_event_processing,
                "parameters": {
                    "type": "object",
                    "properties": {}
                }
            },
            "stop_event_processing": {
                "description": "Stop continuous sensory event processing",
                "function": self.stop_event_processing,
                "parameters": {
                    "type": "object",
                    "properties": {}
                }
            },
            "get_multimodal_context": {
                "description": "Fuse recent audio/video into a unified context",
                "function": self.get_multimodal_context,
                "parameters": {
                    "type": "object",
                    "properties": {}
                }
            }
        }
    
    def start_audio_recording(self, device_index: Optional[int] = None) -> Dict[str, Any]:
        """
        Start recording audio from microphone
        
        Args:
            device_index: Index of audio device to use (optional)
            
        Returns:
            Dictionary with result information
        """
        if not self.audio_processor.is_available:
            return {
                "success": False,
                "error": "Audio capture backend not available"
            }

        success = self.audio_processor.start_recording(device_index)
        return {
            "success": success,
            "message": "Audio recording started" if success else "Failed to start audio recording"
        }
    
    def stop_audio_recording(self) -> Dict[str, Any]:
        """
        Stop recording audio
        
        Returns:
            Dictionary with result information
        """
        if not self.audio_processor.is_available:
            return {
                "success": False,
                "error": "Audio capture backend not available"
            }

        success = self.audio_processor.stop_recording()
        return {
            "success": success,
            "message": "Audio recording stopped" if success else "Not recording audio"
        }
    
    def list_audio_devices(self) -> Dict[str, Any]:
        """
        List available audio input devices
        
        Returns:
            Dictionary with audio device information
        """
        if not self.audio_processor.is_available:
            return {
                "success": False,
                "error": "Audio capture backend not available"
            }

        devices = self.audio_processor.devices
        return {
            "success": True,
            "devices": devices,
            "count": len(devices)
        }
    
    def start_video_capture(self, camera_index: int = 0) -> Dict[str, Any]:
        """
        Start capturing video from webcam
        
        Args:
            camera_index: Index of camera to use
            
        Returns:
            Dictionary with result information
        """
        backend = self.video_processor.capture_backend
        if not backend or not backend.is_available:
            return {
                "success": False,
                "error": "Video capture backend not available"
            }

        success = self.video_processor.start_capture(camera_index)
        return {
            "success": success,
            "message": "Video capture started" if success else "Failed to start video capture"
        }
    
    def stop_video_capture(self) -> Dict[str, Any]:
        """
        Stop capturing video
        
        Returns:
            Dictionary with result information
        """
        backend = self.video_processor.capture_backend
        if not backend or not backend.is_available:
            return {
                "success": False,
                "error": "Video capture backend not available"
            }

        success = self.video_processor.stop_capture()
        return {
            "success": success,
            "message": "Video capture stopped" if success else "Not capturing video"
        }
    
    def list_cameras(self) -> Dict[str, Any]:
        """
        List available camera devices

        Returns:
            Dictionary with camera information
        """
        backend = self.video_processor.capture_backend
        if not backend:
            return {
                "success": False,
                "error": "Video capture backend not available"
            }

        cameras = self.video_processor.list_cameras()
        return {
            "success": True,
            "cameras": cameras,
            "count": len(cameras)
        }

    def configure_audio_backend(self, backend: Optional[AudioCaptureBackend]) -> Dict[str, Any]:
        """Attach a new audio backend implementation."""
        self.audio_processor.use_backend(backend)
        return {
            "success": True,
            "available": self.audio_processor.is_available
        }

    def configure_video_backend(self, backend: Optional[VideoCaptureBackend]) -> Dict[str, Any]:
        """Attach a new video backend implementation."""
        self.video_processor.use_backend(backend)
        return {
            "success": True,
            "available": bool(self.video_processor.capture_backend and self.video_processor.capture_backend.is_available)
        }

    def check_audio_availability(self) -> Dict[str, Any]:
        """Return audio backend availability information."""
        devices = self.audio_processor.devices if self.audio_processor.is_available else []
        return {
            "available": self.audio_processor.is_available,
            "device_count": len(devices),
            "devices": devices
        }

    def check_video_availability(self) -> Dict[str, Any]:
        """Return video backend availability information."""
        backend = self.video_processor.capture_backend
        cameras = backend.list_cameras() if backend else []
        return {
            "available": bool(backend and backend.is_available),
            "camera_count": len(cameras),
            "cameras": cameras
        }

    def add_test_event(self, event: Dict[str, Any]) -> None:
        """Inject a synthetic event for testing or simulation."""
        event.setdefault("timestamp", datetime.now().isoformat())
        with self.buffer_lock:
            self.event_buffer.append(event)
            if len(self.event_buffer) > 100:
                self.event_buffer = self.event_buffer[-100:]

        for callback in list(self.event_callbacks):
            try:
                callback(event)
            except Exception as exc:
                logger.error(f"Error in injected event callback: {exc}")
    
    def register_event_callback(self, callback: Callable[[Dict[str, Any]], None]) -> None:
        """
        Register a callback function for processing sensory events
        
        Args:
            callback: Function to call with each event
        """
        self.event_callbacks.append(callback)
        logger.debug(f"Registered event callback (total: {len(self.event_callbacks)})")
    
    def unregister_event_callback(self, callback: Callable[[Dict[str, Any]], None]) -> bool:
        """
        Unregister a previously registered callback function
        
        Args:
            callback: Function to unregister
            
        Returns:
            Success flag
        """
        if callback in self.event_callbacks:
            self.event_callbacks.remove(callback)
            logger.debug(f"Unregistered event callback (total: {len(self.event_callbacks)})")
            return True
        return False
    
    def start_event_processing(self) -> Dict[str, Any]:
        """
        Start continuous sensory event processing
        
        Returns:
            Dictionary with result information
        """
        if self.is_processing:
            return {
                "success": False,
                "message": "Event processing already running"
            }

        self.is_processing = True
        self.processing_thread = threading.Thread(
            target=self._event_processing_thread,
            daemon=True
        )
        self.processing_thread.start()
        
        logger.info("Started sensory event processing")
        return {
            "success": True,
            "message": "Sensory event processing started"
        }
    
    def stop_event_processing(self) -> Dict[str, Any]:
        """
        Stop continuous sensory event processing
        
        Returns:
            Dictionary with result information
        """
        if not self.is_processing:
            return {
                "success": False,
                "message": "Event processing not running"
            }
        
        self.is_processing = False
        if self.processing_thread:
            self.processing_thread.join(timeout=2.0)
            self.processing_thread = None
        
        logger.info("Stopped sensory event processing")
        return {
            "success": True,
            "message": "Sensory event processing stopped"
        }
    
    def _event_processing_thread(self) -> None:
        """Thread function for continuous event processing"""
        while self.is_processing:
            try:
                # Get audio events
                if self.audio_processor:
                    audio_event = self.audio_processor.get_next_audio_event(timeout=0.01)
                    if audio_event:
                        # Process event through callbacks
                        for callback in self.event_callbacks:
                            try:
                                callback(audio_event)
                            except Exception as e:
                                logger.error(f"Error in audio event callback: {e}")
                        
                        # Add to buffer
                        with self.buffer_lock:
                            self.event_buffer.append(audio_event)
                            # Limit buffer size
                            if len(self.event_buffer) > 100:
                                self.event_buffer = self.event_buffer[-100:]
                        self._latest_audio_event = audio_event
                        self._try_fuse_events()

                # Get video events
                if self.video_processor:
                    video_event = self.video_processor.get_next_video_event(timeout=0.01)
                    if video_event:
                        # Process event through callbacks
                        for callback in self.event_callbacks:
                            try:
                                callback(video_event)
                            except Exception as e:
                                logger.error(f"Error in video event callback: {e}")
                        
                        # Add to buffer
                        with self.buffer_lock:
                            self.event_buffer.append(video_event)
                            # Limit buffer size
                            if len(self.event_buffer) > 100:
                                self.event_buffer = self.event_buffer[-100:]
                        self._latest_video_event = video_event
                        self._try_fuse_events()
                
                # Short sleep to prevent tight loop
                time.sleep(0.01)
                
            except Exception as e:
                logger.error(f"Error in event processing: {e}")
                time.sleep(0.1)  # Longer sleep on errors
    
    def get_latest_sensory_events(self, max_events: int = 10) -> Dict[str, Any]:
        """
        Get the latest sensory events (audio, video, etc.)

        Args:
            max_events: Maximum number of events to return

        Returns:
            Dictionary with sensory events
        """
        with self.buffer_lock:
            events = self.event_buffer[-max_events:] if self.event_buffer else []

        return {
            "success": True,
            "events": events,
            "count": len(events)
        }

    def get_multimodal_context(self) -> Dict[str, Any]:
        """Return the latest fused sensory context and reasoning."""
        if not self._fused_events:
            return {"success": False, "error": "Insufficient data for fusion"}
        fused_event = self._fused_events[-1]
        reasoning = self.cross_modal_reasoner.infer_context(fused_event)
        return {"success": True, "fused": fused_event, "reasoning": reasoning}

    def _try_fuse_events(self) -> None:
        if self._latest_audio_event is None and self._latest_video_event is None:
            return
        fused = self.fusion_engine.fuse(
            audio_event=self._latest_audio_event,
            video_event=self._latest_video_event,
        )
        self._fused_events.append(fused)

    def shutdown(self) -> None:
        """Release resources used by the sensory processors."""
        if self.audio_processor:
            self.audio_processor.shutdown()
        if self.video_processor:
            self.video_processor.shutdown()
        self._fused_events.clear()
