"""
Streaming AI Response Service
Provides real-time token streaming, progress indicators, and interrupt/resume capabilities
"""

import asyncio
import json
import time
import uuid
from typing import Dict, List, Any, Optional, AsyncGenerator, Callable
from dataclasses import dataclass, asdict
from enum import Enum
import openai
from openai import OpenAI
import anthropic
from ..utils.logger import get_logger

logger = get_logger(__name__)

class StreamStatus(Enum):
    """Stream status states"""
    PENDING = "pending"
    STREAMING = "streaming"
    PAUSED = "paused"
    COMPLETED = "completed"
    ERROR = "error"
    INTERRUPTED = "interrupted"

@dataclass
class StreamChunk:
    """Individual chunk of streamed content"""
    id: str
    content: str
    chunk_type: str  # "text", "function_call", "metadata"
    timestamp: float
    token_count: int
    is_final: bool = False

@dataclass
class StreamProgress:
    """Progress information for streaming"""
    stream_id: str
    status: StreamStatus
    total_tokens: int
    processed_tokens: int
    estimated_completion: float
    start_time: float
    current_time: float
    provider: str
    model: str

class StreamingSession:
    """Manages a single streaming session"""

    def __init__(self, stream_id: str, provider: str, model: str):
        self.stream_id = stream_id
        self.provider = provider
        self.model = model
        self.status = StreamStatus.PENDING
        self.chunks: List[StreamChunk] = []
        self.start_time = time.time()
        self.pause_time = None
        self.total_pause_duration = 0
        self.callbacks: List[Callable] = []
        self.interrupt_requested = False

    def add_chunk(self, content: str, chunk_type: str = "text", is_final: bool = False):
        """Add a new chunk to the stream"""
        chunk = StreamChunk(
            id=str(uuid.uuid4()),
            content=content,
            chunk_type=chunk_type,
            timestamp=time.time(),
            token_count=len(content.split()),
            is_final=is_final
        )
        self.chunks.append(chunk)

        # Notify callbacks
        for callback in self.callbacks:
            try:
                callback(chunk)
            except Exception as e:
                logger.error(f"Callback error: {e}")

    def pause(self):
        """Pause the streaming session"""
        if self.status == StreamStatus.STREAMING:
            self.status = StreamStatus.PAUSED
            self.pause_time = time.time()

    def resume(self):
        """Resume the streaming session"""
        if self.status == StreamStatus.PAUSED and self.pause_time:
            self.total_pause_duration += time.time() - self.pause_time
            self.pause_time = None
            self.status = StreamStatus.STREAMING

    def interrupt(self):
        """Interrupt the streaming session"""
        self.interrupt_requested = True
        self.status = StreamStatus.INTERRUPTED

    def get_progress(self) -> StreamProgress:
        """Get current progress information"""
        current_time = time.time()
        processed_tokens = sum(chunk.token_count for chunk in self.chunks)

        # Estimate completion based on current rate
        elapsed_time = current_time - self.start_time - self.total_pause_duration
        if elapsed_time > 0 and processed_tokens > 0:
            tokens_per_second = processed_tokens / elapsed_time
            # Rough estimate assuming 1000 tokens total (can be adjusted)
            estimated_total = max(1000, processed_tokens * 1.5)
            remaining_tokens = estimated_total - processed_tokens
            estimated_completion = remaining_tokens / tokens_per_second if tokens_per_second > 0 else 0
        else:
            estimated_completion = 0

        return StreamProgress(
            stream_id=self.stream_id,
            status=self.status,
            total_tokens=1000,  # Placeholder
            processed_tokens=processed_tokens,
            estimated_completion=estimated_completion,
            start_time=self.start_time,
            current_time=current_time,
            provider=self.provider,
            model=self.model
        )

class StreamingService:
    """Main streaming service for AI responses"""

    def __init__(self):
        self.active_sessions: Dict[str, StreamingSession] = {}
        self.openai_client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        self.anthropic_client = anthropic.Anthropic(api_key=os.getenv('ANTHROPIC_API_KEY'))

    async def start_openai_stream(self, messages: List[Dict], model: str = "gpt-4", **kwargs) -> str:
        """Start streaming from OpenAI"""
        stream_id = str(uuid.uuid4())
        session = StreamingSession(stream_id, "openai", model)
        self.active_sessions[stream_id] = session

        # Start streaming in background
        asyncio.create_task(self._openai_stream_worker(session, messages, model, **kwargs))

        return stream_id

    async def _openai_stream_worker(self, session: StreamingSession, messages: List[Dict], model: str, **kwargs):
        """Worker for OpenAI streaming"""
        try:
            session.status = StreamStatus.STREAMING

            response = self.openai_client.chat.completions.create(
                model=model,
                messages=messages,
                stream=True,
                **kwargs
            )

            for chunk in response:
                if session.interrupt_requested:
                    break

                # Handle pause
                while session.status == StreamStatus.PAUSED:
                    await asyncio.sleep(0.1)
                    if session.interrupt_requested:
                        break

                if session.interrupt_requested:
                    break

                # Process chunk
                if chunk.choices and chunk.choices[0].delta.content:
                    content = chunk.choices[0].delta.content
                    session.add_chunk(content, "text")

                # Small delay to prevent overwhelming
                await asyncio.sleep(0.01)

            if not session.interrupt_requested:
                session.status = StreamStatus.COMPLETED
                session.add_chunk("", "text", is_final=True)

        except Exception as e:
            logger.error(f"OpenAI streaming error: {e}")
            session.status = StreamStatus.ERROR
            session.add_chunk(f"Error: {str(e)}", "error", is_final=True)

    async def start_anthropic_stream(self, messages: List[Dict], model: str = "claude-3-sonnet-20240229", **kwargs) -> str:
        """Start streaming from Anthropic"""
        stream_id = str(uuid.uuid4())
        session = StreamingSession(stream_id, "anthropic", model)
        self.active_sessions[stream_id] = session

        # Start streaming in background
        asyncio.create_task(self._anthropic_stream_worker(session, messages, model, **kwargs))

        return stream_id

    async def _anthropic_stream_worker(self, session: StreamingSession, messages: List[Dict], model: str, **kwargs):
        """Worker for Anthropic streaming"""
        try:
            session.status = StreamStatus.STREAMING

            # Convert messages format for Anthropic
            system_message = ""
            user_messages = []

            for msg in messages:
                if msg["role"] == "system":
                    system_message = msg["content"]
                else:
                    user_messages.append(msg)

            with self.anthropic_client.messages.stream(
                model=model,
                max_tokens=kwargs.get("max_tokens", 1000),
                system=system_message,
                messages=user_messages
            ) as stream:
                for text in stream.text_stream:
                    if session.interrupt_requested:
                        break

                    # Handle pause
                    while session.status == StreamStatus.PAUSED:
                        await asyncio.sleep(0.1)
                        if session.interrupt_requested:
                            break

                    if session.interrupt_requested:
                        break

                    session.add_chunk(text, "text")
                    await asyncio.sleep(0.01)

            if not session.interrupt_requested:
                session.status = StreamStatus.COMPLETED
                session.add_chunk("", "text", is_final=True)

        except Exception as e:
            logger.error(f"Anthropic streaming error: {e}")
            session.status = StreamStatus.ERROR
            session.add_chunk(f"Error: {str(e)}", "error", is_final=True)

    async def get_stream_chunks(self, stream_id: str, since_chunk_id: Optional[str] = None) -> List[StreamChunk]:
        """Get chunks from a stream, optionally since a specific chunk"""
        if stream_id not in self.active_sessions:
            return []

        session = self.active_sessions[stream_id]
        chunks = session.chunks

        if since_chunk_id:
            # Find the index of the since_chunk_id
            start_index = 0
            for i, chunk in enumerate(chunks):
                if chunk.id == since_chunk_id:
                    start_index = i + 1
                    break
            chunks = chunks[start_index:]

        return chunks

    async def get_stream_progress(self, stream_id: str) -> Optional[StreamProgress]:
        """Get progress information for a stream"""
        if stream_id not in self.active_sessions:
            return None

        return self.active_sessions[stream_id].get_progress()

    async def pause_stream(self, stream_id: str) -> bool:
        """Pause a streaming session"""
        if stream_id not in self.active_sessions:
            return False

        self.active_sessions[stream_id].pause()
        return True

    async def resume_stream(self, stream_id: str) -> bool:
        """Resume a streaming session"""
        if stream_id not in self.active_sessions:
            return False

        self.active_sessions[stream_id].resume()
        return True

    async def interrupt_stream(self, stream_id: str) -> bool:
        """Interrupt a streaming session"""
        if stream_id not in self.active_sessions:
            return False

        self.active_sessions[stream_id].interrupt()
        return True

    async def get_full_response(self, stream_id: str) -> Optional[str]:
        """Get the full response from a completed stream"""
        if stream_id not in self.active_sessions:
            return None

        session = self.active_sessions[stream_id]
        text_chunks = [chunk.content for chunk in session.chunks if chunk.chunk_type == "text"]
        return "".join(text_chunks)

    async def cleanup_session(self, stream_id: str):
        """Clean up a completed streaming session"""
        if stream_id in self.active_sessions:
            del self.active_sessions[stream_id]

    async def get_active_streams(self) -> Dict[str, Dict[str, Any]]:
        """Get information about all active streams"""
        streams = {}
        for stream_id, session in self.active_sessions.items():
            progress = session.get_progress()
            streams[stream_id] = {
                "progress": asdict(progress),
                "chunk_count": len(session.chunks),
                "provider": session.provider,
                "model": session.model
            }
        return streams

    def add_stream_callback(self, stream_id: str, callback: Callable[[StreamChunk], None]):
        """Add a callback for stream events"""
        if stream_id in self.active_sessions:
            self.active_sessions[stream_id].callbacks.append(callback)

# Global streaming service instance
streaming_service = StreamingService()

# WebSocket handler for real-time streaming (to be integrated with main.py)
class StreamingWebSocketHandler:
    """WebSocket handler for real-time streaming"""

    def __init__(self, websocket):
        self.websocket = websocket
        self.active_streams = set()

    async def handle_message(self, message: Dict[str, Any]):
        """Handle incoming WebSocket message"""
        action = message.get("action")

        if action == "start_stream":
            await self._handle_start_stream(message)
        elif action == "pause_stream":
            await self._handle_pause_stream(message)
        elif action == "resume_stream":
            await self._handle_resume_stream(message)
        elif action == "interrupt_stream":
            await self._handle_interrupt_stream(message)
        elif action == "get_progress":
            await self._handle_get_progress(message)

    async def _handle_start_stream(self, message: Dict[str, Any]):
        """Handle start stream request"""
        provider = message.get("provider", "openai")
        model = message.get("model", "gpt-4")
        messages = message.get("messages", [])

        if provider == "openai":
            stream_id = await streaming_service.start_openai_stream(messages, model)
        elif provider == "anthropic":
            stream_id = await streaming_service.start_anthropic_stream(messages, model)
        else:
            await self.websocket.send(json.dumps({
                "type": "error",
                "message": f"Unsupported provider: {provider}"
            }))
            return

        self.active_streams.add(stream_id)

        # Set up callback for real-time updates
        def chunk_callback(chunk: StreamChunk):
            asyncio.create_task(self.websocket.send(json.dumps({
                "type": "chunk",
                "stream_id": stream_id,
                "chunk": asdict(chunk)
            })))

        streaming_service.add_stream_callback(stream_id, chunk_callback)

        await self.websocket.send(json.dumps({
            "type": "stream_started",
            "stream_id": stream_id,
            "provider": provider,
            "model": model
        }))

    async def _handle_pause_stream(self, message: Dict[str, Any]):
        """Handle pause stream request"""
        stream_id = message.get("stream_id")
        success = await streaming_service.pause_stream(stream_id)

        await self.websocket.send(json.dumps({
            "type": "stream_paused",
            "stream_id": stream_id,
            "success": success
        }))

    async def _handle_resume_stream(self, message: Dict[str, Any]):
        """Handle resume stream request"""
        stream_id = message.get("stream_id")
        success = await streaming_service.resume_stream(stream_id)

        await self.websocket.send(json.dumps({
            "type": "stream_resumed",
            "stream_id": stream_id,
            "success": success
        }))

    async def _handle_interrupt_stream(self, message: Dict[str, Any]):
        """Handle interrupt stream request"""
        stream_id = message.get("stream_id")
        success = await streaming_service.interrupt_stream(stream_id)

        await self.websocket.send(json.dumps({
            "type": "stream_interrupted",
            "stream_id": stream_id,
            "success": success
        }))

    async def _handle_get_progress(self, message: Dict[str, Any]):
        """Handle get progress request"""
        stream_id = message.get("stream_id")
        progress = await streaming_service.get_stream_progress(stream_id)

        await self.websocket.send(json.dumps({
            "type": "progress_update",
            "stream_id": stream_id,
            "progress": asdict(progress) if progress else None
        }))
