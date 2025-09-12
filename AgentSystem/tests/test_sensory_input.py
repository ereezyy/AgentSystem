"""
Test Sensory Input Module
------------------------
Tests the functionality of the sensory input module including
audio recording, video capture, and event processing.
"""

import os
import sys
import time
import unittest
import logging
import threading
from pathlib import Path

# Add parent directory to path to allow imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the module to test
from AgentSystem.modules.sensory_input import SensoryInputModule
from AgentSystem.utils.logger import configure_logging

# Configure logging
configure_logging(level="INFO")
logger = logging.getLogger("tests.test_sensory_input")


class TestSensoryInputModule(unittest.TestCase):
    """Test the SensoryInputModule class"""
    
    def setUp(self):
        """Set up the test environment"""
        self.module = SensoryInputModule()
        self.events = []
        self.event_lock = threading.Lock()
    
    def tearDown(self):
        """Clean up after tests"""
        # Stop any ongoing processing
        self.module.stop_audio_recording()
        self.module.stop_video_capture()
        self.module.stop_event_processing()
    
    def event_callback(self, event):
        """Callback for sensory events"""
        with self.event_lock:
            self.events.append(event)
    
    def test_audio_recording_availability(self):
        """Test if audio recording is available"""
        result = self.module.check_audio_availability()
        logger.info(f"Audio availability: {result}")
        
        # This is not a hard fail, just informational
        if not result.get("available", False):
            self.skipTest("Audio recording not available on this system")
    
    def test_video_capture_availability(self):
        """Test if video capture is available"""
        result = self.module.check_video_availability()
        logger.info(f"Video availability: {result}")
        
        # This is not a hard fail, just informational
        if not result.get("available", False):
            self.skipTest("Video capture not available on this system")
    
    def test_start_stop_audio_recording(self):
        """Test starting and stopping audio recording"""
        # Check if audio is available
        if not self.module.check_audio_availability().get("available", False):
            self.skipTest("Audio recording not available")
        
        # Start recording
        result = self.module.start_audio_recording()
        self.assertTrue(result.get("success", False), "Failed to start audio recording")
        
        # Wait a moment
        time.sleep(1)
        
        # Stop recording
        result = self.module.stop_audio_recording()
        self.assertTrue(result.get("success", False), "Failed to stop audio recording")
    
    def test_start_stop_video_capture(self):
        """Test starting and stopping video capture"""
        # Check if video is available
        if not self.module.check_video_availability().get("available", False):
            self.skipTest("Video capture not available")
        
        # Start capture
        result = self.module.start_video_capture()
        self.assertTrue(result.get("success", False), "Failed to start video capture")
        
        # Wait a moment
        time.sleep(1)
        
        # Stop capture
        result = self.module.stop_video_capture()
        self.assertTrue(result.get("success", False), "Failed to stop video capture")
    
    def test_event_processing(self):
        """Test event processing"""
        # Register callback
        self.module.register_event_callback(self.event_callback)
        
        # Start event processing
        result = self.module.start_event_processing()
        self.assertTrue(result.get("success", False), "Failed to start event processing")
        
        # Generate a test event
        self.module.add_test_event({"type": "test", "data": "test_data"})
        
        # Wait for event processing
        time.sleep(1)
        
        # Stop event processing
        result = self.module.stop_event_processing()
        self.assertTrue(result.get("success", False), "Failed to stop event processing")
        
        # Check if event was processed
        with self.event_lock:
            test_events = [e for e in self.events if e.get("type") == "test"]
            self.assertGreaterEqual(len(test_events), 1, "Test event not processed")
    
    def test_image_processing(self):
        """Test image processing capabilities"""
        # Check if the module has the process_image method
        if not hasattr(self.module, "process_image"):
            self.skipTest("Image processing not implemented")
        
        # Create a test image if possible
        try:
            import numpy as np
            # Create a simple test image (gray square)
            test_image = np.ones((100, 100, 3), dtype=np.uint8) * 128
        except ImportError:
            self.skipTest("NumPy not available for image testing")
        
        # Process the test image
        result = self.module.process_image(test_image)
        
        # Just verify we get a result, don't validate specific features
        # as they depend on the specific implementation
        self.assertIsNotNone(result, "Image processing returned None")
    
    def test_audio_processing(self):
        """Test audio processing capabilities"""
        # Check if the module has the process_audio method
        if not hasattr(self.module, "process_audio"):
            self.skipTest("Audio processing not implemented")
        
        # Create a test audio sample if possible
        try:
            import numpy as np
            # Create a simple test audio sample (sine wave)
            sample_rate = 16000
            duration = 1  # seconds
            t = np.linspace(0, duration, int(sample_rate * duration), False)
            test_audio = np.sin(2 * np.pi * 440 * t).astype(np.float32)
        except ImportError:
            self.skipTest("NumPy not available for audio testing")
        
        # Process the test audio
        result = self.module.process_audio(test_audio, sample_rate)
        
        # Just verify we get a result, don't validate specific features
        self.assertIsNotNone(result, "Audio processing returned None")


def run_quick_test():
    """Run a quick functional test of the module"""
    print("\n=== Quick Functional Test of Sensory Input Module ===\n")
    
    # Create the module
    module = SensoryInputModule()
    
    # Check availabilities
    print("Checking device availability...")
    audio_avail = module.check_audio_availability()
    video_avail = module.check_video_availability()
    
    print(f"Audio devices available: {audio_avail.get('available', False)}")
    if audio_avail.get('available', False):
        print(f"  Device: {audio_avail.get('device_name', 'Unknown')}")
    
    print(f"Video devices available: {video_avail.get('available', False)}")
    if video_avail.get('available', False):
        print(f"  Device: {video_avail.get('device_name', 'Unknown')}")
    
    # Define a simple event callback
    def print_event(event):
        event_type = event.get("type", "unknown")
        if event_type == "audio_level":
            # Don't print these, too noisy
            return
        print(f"Event received: {event_type}")
        if event_type == "speech":
            print(f"  Speech: {event.get('text', '')}")
        elif event_type == "video_frame":
            features = event.get("features", {})
            if "faces" in features:
                print(f"  Faces detected: {len(features['faces'])}")
    
    # Register the callback
    module.register_event_callback(print_event)
    
    # Start event processing
    print("\nStarting event processing...")
    module.start_event_processing()
    
    # Try to start audio recording if available
    if audio_avail.get('available', False):
        print("Starting audio recording...")
        result = module.start_audio_recording()
        if result.get("success", False):
            print("Audio recording started successfully")
        else:
            print(f"Failed to start audio recording: {result.get('error', 'Unknown error')}")
    
    # Try to start video capture if available
    if video_avail.get('available', False):
        print("Starting video capture...")
        result = module.start_video_capture()
        if result.get("success", False):
            print("Video capture started successfully")
        else:
            print(f"Failed to start video capture: {result.get('error', 'Unknown error')}")
    
    # Run for a few seconds
    print("\nRunning for 5 seconds...\n")
    try:
        time.sleep(5)
    except KeyboardInterrupt:
        print("Test interrupted by user")
    
    # Stop everything
    print("\nStopping sensory input...")
    module.stop_audio_recording()
    module.stop_video_capture()
    module.stop_event_processing()
    
    print("\n=== Test Completed ===\n")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--quick":
        # Run quick functional test
        run_quick_test()
    else:
        # Run unit tests
        unittest.main()
