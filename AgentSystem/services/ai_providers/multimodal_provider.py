"""
Multi-Modal AI Provider
Handles vision, audio, video, and code AI capabilities
"""

import os
import io
import base64
import requests
from typing import Dict, List, Any, Optional, Union
from PIL import Image
import openai
from openai import OpenAI
import anthropic
from ..ai import ai_service
from ...utils.logger import get_logger

logger = get_logger(__name__)

class MultiModalProvider:
    """Multi-modal AI provider for vision, audio, video, and code tasks"""

    def __init__(self):
        self.openai_client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        self.anthropic_client = anthropic.Anthropic(api_key=os.getenv('ANTHROPIC_API_KEY'))
        self.elevenlabs_api_key = os.getenv('ELEVENLABS_API_KEY')
        self.runway_api_key = os.getenv('RUNWAY_API_KEY')

    # VISION AI CAPABILITIES

    async def analyze_image(self, image_data: Union[str, bytes], prompt: str = "Describe this image in detail") -> Dict[str, Any]:
        """
        Analyze image using GPT-4V or Claude 3.5 Sonnet with vision
        """
        try:
            # Convert image to base64 if needed
            if isinstance(image_data, bytes):
                image_base64 = base64.b64encode(image_data).decode('utf-8')
            else:
                image_base64 = image_data

            # Try GPT-4V first
            response = self.openai_client.chat.completions.create(
                model="gpt-4-vision-preview",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{image_base64}"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=1000
            )

            return {
                "success": True,
                "provider": "gpt-4v",
                "analysis": response.choices[0].message.content,
                "confidence": 0.95
            }

        except Exception as e:
            logger.error(f"Vision analysis failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "provider": "gpt-4v"
            }

    async def generate_image(self, prompt: str, size: str = "1024x1024", quality: str = "standard") -> Dict[str, Any]:
        """
        Generate images using DALL-E 3
        """
        try:
            response = self.openai_client.images.generate(
                model="dall-e-3",
                prompt=prompt,
                size=size,
                quality=quality,
                n=1
            )

            return {
                "success": True,
                "provider": "dall-e-3",
                "image_url": response.data[0].url,
                "revised_prompt": response.data[0].revised_prompt
            }

        except Exception as e:
            logger.error(f"Image generation failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "provider": "dall-e-3"
            }

    # AUDIO AI CAPABILITIES

    async def transcribe_audio(self, audio_file_path: str) -> Dict[str, Any]:
        """
        Transcribe audio using Whisper
        """
        try:
            with open(audio_file_path, "rb") as audio_file:
                transcript = self.openai_client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file
                )

            return {
                "success": True,
                "provider": "whisper",
                "transcript": transcript.text
            }

        except Exception as e:
            logger.error(f"Audio transcription failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "provider": "whisper"
            }

    async def text_to_speech(self, text: str, voice: str = "alloy") -> Dict[str, Any]:
        """
        Convert text to speech using OpenAI TTS
        """
        try:
            response = self.openai_client.audio.speech.create(
                model="tts-1",
                voice=voice,
                input=text
            )

            # Save audio to file
            audio_filename = f"output_{hash(text)}.mp3"
            audio_path = f"/tmp/{audio_filename}"

            with open(audio_path, "wb") as f:
                f.write(response.content)

            return {
                "success": True,
                "provider": "openai-tts",
                "audio_file": audio_path,
                "voice": voice
            }

        except Exception as e:
            logger.error(f"Text-to-speech failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "provider": "openai-tts"
            }

    async def advanced_tts_elevenlabs(self, text: str, voice_id: str = "EXAVITQu4vr4xnSDxMaL") -> Dict[str, Any]:
        """
        High-quality text-to-speech using ElevenLabs
        """
        if not self.elevenlabs_api_key:
            return {"success": False, "error": "ElevenLabs API key not configured"}

        try:
            url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"

            headers = {
                "Accept": "audio/mpeg",
                "Content-Type": "application/json",
                "xi-api-key": self.elevenlabs_api_key
            }

            data = {
                "text": text,
                "model_id": "eleven_monolingual_v1",
                "voice_settings": {
                    "stability": 0.5,
                    "similarity_boost": 0.5
                }
            }

            response = requests.post(url, json=data, headers=headers)

            if response.status_code == 200:
                audio_filename = f"elevenlabs_{hash(text)}.mp3"
                audio_path = f"/tmp/{audio_filename}"

                with open(audio_path, "wb") as f:
                    f.write(response.content)

                return {
                    "success": True,
                    "provider": "elevenlabs",
                    "audio_file": audio_path,
                    "voice_id": voice_id
                }
            else:
                return {
                    "success": False,
                    "error": f"ElevenLabs API error: {response.status_code}",
                    "provider": "elevenlabs"
                }

        except Exception as e:
            logger.error(f"ElevenLabs TTS failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "provider": "elevenlabs"
            }

    # CODE AI CAPABILITIES

    async def generate_code(self, prompt: str, language: str = "python") -> Dict[str, Any]:
        """
        Generate code using specialized code models
        """
        try:
            # Use GPT-4 with code-specific prompting
            system_prompt = f"""You are an expert {language} programmer. Generate high-quality, production-ready code based on the user's requirements. Include proper error handling, documentation, and follow best practices."""

            response = self.openai_client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1  # Lower temperature for more deterministic code
            )

            code = response.choices[0].message.content

            return {
                "success": True,
                "provider": "gpt-4-code",
                "code": code,
                "language": language
            }

        except Exception as e:
            logger.error(f"Code generation failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "provider": "gpt-4-code"
            }

    async def review_code(self, code: str, language: str = "python") -> Dict[str, Any]:
        """
        Review and analyze code for issues, improvements, and security
        """
        try:
            prompt = f"""
            Review this {language} code for:
            1. Security vulnerabilities
            2. Performance issues
            3. Best practices violations
            4. Potential bugs
            5. Suggestions for improvement

            Code:
            ```{language}
            {code}
            ```

            Provide detailed analysis and specific recommendations.
            """

            response = self.openai_client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1
            )

            return {
                "success": True,
                "provider": "gpt-4-review",
                "review": response.choices[0].message.content,
                "language": language
            }

        except Exception as e:
            logger.error(f"Code review failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "provider": "gpt-4-review"
            }

    # VIDEO AI CAPABILITIES (Placeholder for RunwayML integration)

    async def generate_video(self, prompt: str, duration: int = 4) -> Dict[str, Any]:
        """
        Generate video using RunwayML (requires API integration)
        """
        if not self.runway_api_key:
            return {
                "success": False,
                "error": "RunwayML API key not configured. Video generation requires RunwayML subscription.",
                "provider": "runway"
            }

        # Placeholder for RunwayML API integration
        return {
            "success": False,
            "error": "RunwayML integration coming soon",
            "provider": "runway"
        }

    async def analyze_video(self, video_path: str, prompt: str = "Analyze this video") -> Dict[str, Any]:
        """
        Analyze video content (placeholder for future implementation)
        """
        return {
            "success": False,
            "error": "Video analysis coming soon",
            "provider": "placeholder"
        }

# Global instance
multimodal_provider = MultiModalProvider()
