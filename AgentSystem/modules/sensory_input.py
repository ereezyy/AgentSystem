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
from typing import Dict, List, Any, Optional, Callable, Union
from datetime import datetime

# Local imports
from AgentSystem.utils.logger import get_logger

# Get module logger
logger = get_logger("modules.sensory_input")

try:
    import cv2
    import numpy as np
    import pyaudio
    import speech_recognition as sr
    from PIL import Image
    SENSORY_IMPORTS_AVAILABLE = True
except ImportError:
    logger.warning("Sensory input dependencies not available. Install with: pip install opencv-python numpy pyaudio SpeechRecognition pillow")
    SENSORY_IMPORTS_AVAILABLE = False


class AudioProcessor:
    """Processes audio input from microphone"""
    
    def __init__(self, sample_rate: int = 16000, chunk_size: int = 1024):
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
        self.recognizer = sr.Recognizer() if SENSORY_IMPORTS_AVAILABLE else None
        
        # Audio device info
        self.audio = pyaudio.PyAudio() if SENSORY_IMPORTS_AVAILABLE else None
        self.devices = self._get_audio_devices() if SENSORY_IMPORTS_AVAILABLE else []
        
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
        if not SENSORY_IMPORTS_AVAILABLE:
            logger.error("Audio processing dependencies not available")
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
        
        logger.info("Stopped audio recording")
        return True
    
    def _record_audio_thread(self, device_index: Optional[int] = None) -> None:
        """
        Thread function for continuous audio recording
        
        Args:
            device_index: Index of audio device to use
        """
        stream = self.audio.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=self.chunk_size,
            input_device_index=device_index
        )
        
        # Buffer to accumulate audio chunks
        audio_buffer = []
        buffer_duration_sec = 0
        target_duration_sec = 3  # Process in 3-second chunks
        
        while self.is_recording:
            try:
                # Read audio chunk
                data = stream.read(self.chunk_size, exception_on_overflow=False)
                audio_buffer.append(data)
                
                # Calculate buffer duration
                buffer_duration_sec += self.chunk_size / self.sample_rate
                
                # If buffer reaches target duration, process it
                if buffer_duration_sec >= target_duration_sec:
                    # Process buffer (in a separate thread to avoid blocking)
                    threading.Thread(
                        target=self._process_audio_chunk,
                        args=(b''.join(audio_buffer),),
                        daemon=True
                    ).start()
                    
                    # Clear buffer
                    audio_buffer = []
                    buffer_duration_sec = 0
                    
            except Exception as e:
                logger.error(f"Error recording audio: {e}")
                time.sleep(0.1)  # Prevent tight loop on errors
        
        # Clean up
        stream.stop_stream()
        stream.close()
    
    def _process_audio_chunk(self, audio_data: bytes) -> None:
        """
        Process an audio chunk for speech recognition
        
        Args:
            audio_data: Raw audio data
        """
        try:
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
                
        except Exception as e:
            logger.error(f"Error processing audio chunk: {e}")
    
    def _extract_audio_features(self, audio_data: bytes) -> None:
        """
        Extract features from audio data when speech isn't detected
        
        Args:
            audio_data: Raw audio data
        """
        try:
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


class VideoProcessor:
    """Processes video input from webcam or other sources"""
    
    def __init__(self, width: int = 640, height: int = 480, fps: int = 5):
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
        
        # OpenCV capture object
        self.cap = None
        
        # Initialize face detection if OpenCV is available
        self.face_cascade = None
        if SENSORY_IMPORTS_AVAILABLE:
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
        if not SENSORY_IMPORTS_AVAILABLE:
            return []
            
        cameras = []
        for i in range(10):  # Try up to 10 camera indices
            try:
                cap = cv2.VideoCapture(i)
                if cap.isOpened():
                    cameras.append({
                        'index': i,
                        'name': f"Camera {i}",
                        'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                        'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                        'fps': int(cap.get(cv2.CAP_PROP_FPS))
                    })
                    cap.release()
            except Exception:
                pass
        return cameras
    
    def start_capture(self, camera_index: int = 0) -> bool:
        """
        Start capturing video
        
        Args:
            camera_index: Index of camera to use
            
        Returns:
            Success flag
        """
        if not SENSORY_IMPORTS_AVAILABLE:
            logger.error("Video processing dependencies not available")
            return False
            
        if self.is_capturing:
            logger.warning("Already capturing video")
            return False
        
        try:
            # Initialize camera
            self.cap = cv2.VideoCapture(camera_index)
            if not self.cap.isOpened():
                logger.error(f"Failed to open camera {camera_index}")
                return False
                
            # Set resolution
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            
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
            if self.cap:
                self.cap.release()
                self.cap = None
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
        
        if self.cap:
            self.cap.release()
            self.cap = None
            
        logger.info("Stopped video capture")
        return True
    
    def _capture_video_thread(self) -> None:
        """Thread function for continuous video capture"""
        last_frame_time = 0
        
        while self.is_capturing and self.cap and self.cap.isOpened():
            try:
                # Maintain target frame rate
                current_time = time.time()
                if current_time - last_frame_time < self.frame_interval:
                    time.sleep(0.001)  # Short sleep to prevent CPU spin
                    continue
                
                # Capture frame
                ret, frame = self.cap.read()
                if not ret:
                    logger.error("Failed to capture frame")
                    time.sleep(0.1)  # Prevent tight loop on errors
                    continue
                
                # Process the frame in a separate thread
                threading.Thread(
                    target=self._process_frame,
                    args=(frame.copy(),),  # Copy to prevent race conditions
                    daemon=True
                ).start()
                
                last_frame_time = time.time()
                
            except Exception as e:
                logger.error(f"Error in video capture: {e}")
                time.sleep(0.1)  # Prevent tight loop on errors
    
    def _process_frame(self, frame: Any) -> None:
        """
        Process a captured video frame
        
        Args:
            frame: Video frame to process
        """
        try:
            # Detect faces
            faces = []
            if self.face_cascade is not None:
                # Convert to grayscale for face detection
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                # Detect faces
                detected_faces = self.face_cascade.detectMultiScale(
                    gray, 
                    scaleFactor=1.1, 
                    minNeighbors=5, 
                    minSize=(30, 30)
                )
                
                # Process detected faces
                for (x, y, w, h) in detected_faces:
                    faces.append({
                        'x': int(x),
                        'y': int(y),
                        'width': int(w),
                        'height': int(h)
                    })
            
            # Extract basic image features
            # Calculate average brightness
            brightness = np.mean(frame)
            
            # Calculate color distribution
            if frame.shape[2] == 3:  # Check if frame has 3 color channels
                color_means = [
                    float(np.mean(frame[:, :, 0])),  # Blue
                    float(np.mean(frame[:, :, 1])),  # Green
                    float(np.mean(frame[:, :, 2]))   # Red
                ]
            else:
                color_means = [float(brightness)]
                
            # Create a thumbnail for visualizing
            thumbnail = cv2.resize(frame, (160, 120))
            _, jpeg_data = cv2.imencode('.jpg', thumbnail, [cv2.IMWRITE_JPEG_QUALITY, 70])
            thumbnail_b64 = base64.b64encode(jpeg_data).decode('utf-8')
            
            # Create event object
            event = {
                'type': 'video_frame',
                'timestamp': datetime.now().isoformat(),
                'features': {
                    'brightness': float(brightness),
                    'color_means': color_means,
                    'faces': faces,
                    'resolution': {
                        'width': frame.shape[1],
                        'height': frame.shape[0]
                    }
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


class SensoryInputModule:
    """Module for processing sensory inputs (audio, video, etc.)"""
    
    def __init__(self):
        """Initialize the sensory input module"""
        self.audio_processor = AudioProcessor() if SENSORY_IMPORTS_AVAILABLE else None
        self.video_processor = VideoProcessor() if SENSORY_IMPORTS_AVAILABLE else None
        
        # Callbacks for processing events
        self.event_callbacks = []
        
        # Event processing thread
        self.processing_thread = None
        self.is_processing = False
        
        # Event buffer for batch processing
        self.event_buffer = []
        self.buffer_lock = threading.Lock()
        
        logger.info(f"Initialized SensoryInputModule (dependencies available: {SENSORY_IMPORTS_AVAILABLE})")
    
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
        if not SENSORY_IMPORTS_AVAILABLE or not self.audio_processor:
            return {
                "success": False,
                "error": "Audio processing dependencies not available"
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
        if not SENSORY_IMPORTS_AVAILABLE or not self.audio_processor:
            return {
                "success": False,
                "error": "Audio processing dependencies not available"
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
        if not SENSORY_IMPORTS_AVAILABLE or not self.audio_processor:
            return {
                "success": False,
                "error": "Audio processing dependencies not available"
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
        if not SENSORY_IMPORTS_AVAILABLE or not self.video_processor:
            return {
                "success": False,
                "error": "Video processing dependencies not available"
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
        if not SENSORY_IMPORTS_AVAILABLE or not self.video_processor:
            return {
                "success": False,
                "error": "Video processing dependencies not available"
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
        if not SENSORY_IMPORTS_AVAILABLE or not self.video_processor:
            return {
                "success": False,
                "error": "Video processing dependencies not available"
            }
        
        cameras = self.video_processor.list_cameras()
        return {
            "success": True,
            "cameras": cameras,
            "count": len(cameras)
        }
    
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
        
        if not SENSORY_IMPORTS_AVAILABLE:
            return {
                "success": False,
                "error": "Sensory processing dependencies not available"
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
