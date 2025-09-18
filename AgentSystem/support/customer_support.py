"""
Customer Success and Support Automation for AgentSystem
Provides AI-driven chatbots, automated ticketing, and onboarding support
"""

import os
import logging
import asyncio
from typing import Dict, List, Any, Optional
from datetime import datetime
import aioredis
import asyncpg
from AgentSystem.services.ai import ai_service

logger = logging.getLogger(__name__)

class CustomerSupport:
    """Manages customer success and support automation for AgentSystem"""

    def __init__(self, db_pool: asyncpg.Pool, redis_client: aioredis.Redis):
        """Initialize the customer support system"""
        self.db_pool = db_pool
        self.redis = redis_client
        self._running = False
        self._chatbot_task = None
        self._ticket_task = None
        self.chatbot_interval = 60  # Check for new chats every minute
        self.ticket_interval = 300  # Check for ticket updates every 5 minutes

        logger.info("Customer Support System initialized")

    async def start(self):
        """Start the customer support system with background tasks"""
        self._running = True
        self._chatbot_task = asyncio.create_task(self._chatbot_loop())
        self._ticket_task = asyncio.create_task(self._ticket_monitor_loop())
        logger.info("Customer Support System started")

    async def stop(self):
        """Stop the customer support system and complete any pending tasks"""
        self._running = False
        if self._chatbot_task:
            self._chatbot_task.cancel()
            try:
                await self._chatbot_task
            except asyncio.CancelledError:
                pass
        if self._ticket_task:
            self._ticket_task.cancel()
            try:
                await self._ticket_task
            except asyncio.CancelledError:
                pass

        logger.info("Customer Support System stopped")

    async def handle_chat_request(self, tenant_id: str, user_id: str, message: str) -> Dict[str, Any]:
        """Handle a user chat request with AI chatbot response"""
        logger.info(f"Handling chat request for tenant {tenant_id}, user {user_id}")

        # Store chat message
        async with self.db_pool.acquire() as conn:
            chat_id = await conn.fetchval("""
                INSERT INTO tenant_management.chat_messages (tenant_id, user_id, message, source)
                VALUES ($1, $2, $3, 'user')
                RETURNING id
            """, tenant_id, user_id, message)

        # Generate AI response
        ai_response = await self._generate_chat_response(tenant_id, user_id, message)

        # Store AI response
        async with self.db_pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO tenant_management.chat_messages (tenant_id, user_id, message, source, parent_id)
                VALUES ($1, $2, $3, 'bot', $4)
            """, tenant_id, user_id, ai_response['text'], chat_id)

        logger.info(f"Generated AI response for chat request {chat_id}")
        return {
            "chat_id": chat_id,
            "response": ai_response['text'],
            "timestamp": datetime.now().isoformat()
        }

    async def create_support_ticket(self, tenant_id: str, user_id: str, issue: str, priority: str = "medium") -> Dict[str, Any]:
        """Create a support ticket for a user issue"""
        logger.info(f"Creating support ticket for tenant {tenant_id}, user {user_id}")

        async with self.db_pool.acquire() as conn:
            ticket_id = await conn.fetchval("""
                INSERT INTO tenant_management.support_tickets (tenant_id, user_id, issue, priority, status)
                VALUES ($1, $2, $3, $4, 'open')
                RETURNING id
            """, tenant_id, user_id, issue, priority)

        # Generate initial automated response
        initial_response = await self._generate_ticket_response(tenant_id, user_id, issue, priority)

        async with self.db_pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO tenant_management.ticket_responses (ticket_id, response, source)
                VALUES ($1, $2, 'bot')
            """, ticket_id, initial_response['text'])

        logger.info(f"Created support ticket {ticket_id} with initial response")
        return {
            "ticket_id": ticket_id,
            "status": "open",
            "initial_response": initial_response['text'],
            "created_at": datetime.now().isoformat()
        }

    async def get_onboarding_tutorial(self, tenant_id: str, user_id: str, topic: str = "getting_started") -> Dict[str, Any]:
        """Generate or retrieve an onboarding tutorial for a specific topic"""
        logger.info(f"Retrieving onboarding tutorial for tenant {tenant_id}, user {user_id}, topic {topic}")

        # Check if tutorial exists in cache
        cache_key = f"tutorial:{tenant_id}:{topic}"
        cached_tutorial = await self.redis.get(cache_key)
        if cached_tutorial:
            return {
                "topic": topic,
                "content": cached_tutorial.decode(),
                "cached": True
            }

        # Generate tutorial content if not cached
        tutorial_content = await self._generate_tutorial_content(tenant_id, topic)

        # Cache the tutorial for future use
        await self.redis.setex(cache_key, 86400 * 7, tutorial_content['text'])  # Cache for 7 days

        # Log tutorial access
        async with self.db_pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO tenant_management.tutorial_access (tenant_id, user_id, topic)
                VALUES ($1, $2, $3)
            """, tenant_id, user_id, topic)

        logger.info(f"Generated onboarding tutorial for topic {topic}")
        return {
            "topic": topic,
            "content": tutorial_content['text'],
            "cached": False
        }

    async def _chatbot_loop(self):
        """Background task to monitor and respond to chat requests"""
        while self._running:
            try:
                # Check for new chat messages without bot responses
                async with self.db_pool.acquire() as conn:
                    unanswered_chats = await conn.fetch("""
                        SELECT cm.id, cm.tenant_id, cm.user_id, cm.message
                        FROM tenant_management.chat_messages cm
                        WHERE cm.source = 'user'
                        AND NOT EXISTS (
                            SELECT 1
                            FROM tenant_management.chat_messages cm2
                            WHERE cm2.parent_id = cm.id AND cm2.source = 'bot'
                        )
                        AND cm.created_at > NOW() - INTERVAL '1 hour'
                        LIMIT 10
                    """)

                    for chat in unanswered_chats:
                        await self.handle_chat_request(chat['tenant_id'], chat['user_id'], chat['message'])

                await asyncio.sleep(self.chatbot_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in chatbot loop: {e}")
                await asyncio.sleep(300)  # Wait before retrying

    async def _ticket_monitor_loop(self):
        """Background task to monitor and update support tickets"""
        while self._running:
            try:
                # Check for open tickets needing follow-up
                async with self.db_pool.acquire() as conn:
                    open_tickets = await conn.fetch("""
                        SELECT st.id, st.tenant_id, st.user_id, st.issue, st.priority
                        FROM tenant_management.support_tickets st
                        WHERE st.status = 'open'
                        AND (st.last_updated < NOW() - INTERVAL '4 hours' OR st.last_updated IS NULL)
                        LIMIT 10
                    """)

                    for ticket in open_tickets:
                        follow_up = await self._generate_ticket_follow_up(
                            ticket['tenant_id'], ticket['user_id'], ticket['issue'], ticket['priority']
                        )

                        await conn.execute("""
                            INSERT INTO tenant_management.ticket_responses (ticket_id, response, source)
                            VALUES ($1, $2, 'bot')
                        """, ticket['id'], follow_up['text'])

                        await conn.execute("""
                            UPDATE tenant_management.support_tickets
                            SET last_updated = NOW()
                            WHERE id = $1
                        """, ticket['id'])

                        logger.info(f"Generated follow-up for ticket {ticket['id']}")

                await asyncio.sleep(self.ticket_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in ticket monitor loop: {e}")
                await asyncio.sleep(300)  # Wait before retrying

    async def _generate_chat_response(self, tenant_id: str, user_id: str, message: str) -> Dict[str, Any]:
        """Generate an AI response for a chat message"""
        prompt = f"You are a customer support chatbot for AgentSystem, a SaaS platform with AI capabilities. Respond helpfully to the user's message: '{message}'"
        try:
            if ai_service:
                return await ai_service.generate_text(prompt, max_tokens=500)
            else:
                return {"text": "I'm here to help! However, AI services are currently unavailable. Please try again later or contact support directly."}
        except Exception as e:
            logger.error(f"Error generating chat response: {e}")
            return {"text": "Sorry, I encountered an error. Please try again or contact support for assistance."}

    async def _generate_ticket_response(self, tenant_id: str, user_id: str, issue: str, priority: str) -> Dict[str, Any]:
        """Generate an initial AI response for a support ticket"""
        prompt = f"You are a customer support system for AgentSystem. Generate an initial response for a support ticket with issue: '{issue}', priority: '{priority}'. Acknowledge the issue, provide initial troubleshooting steps if relevant, and set expectations for resolution time based on priority."
        try:
            if ai_service:
                return await ai_service.generate_text(prompt, max_tokens=500)
            else:
                return {"text": f"Thank you for reporting this issue with priority {priority}. We've received your ticket and will respond as soon as possible. For urgent issues, please contact support directly."}
        except Exception as e:
            logger.error(f"Error generating ticket response: {e}")
            return {"text": "Thank you for your ticket. We've encountered an error processing your request, but our team has been notified and will follow up soon."}

    async def _generate_ticket_follow_up(self, tenant_id: str, user_id: str, issue: str, priority: str) -> Dict[str, Any]:
        """Generate a follow-up response for an open ticket"""
        prompt = f"You are a customer support system for AgentSystem. Generate a follow-up response for an open support ticket with issue: '{issue}', priority: '{priority}'. Check on the user's status, offer additional assistance, and reiterate expected resolution time based on priority."
        try:
            if ai_service:
                return await ai_service.generate_text(prompt, max_tokens=500)
            else:
                return {"text": f"We're following up on your open ticket with priority {priority}. Have you resolved the issue, or do you need further assistance? Our team is here to help."}
        except Exception as e:
            logger.error(f"Error generating ticket follow-up: {e}")
            return {"text": "We're checking on your open ticket. If you still need assistance, please reply, and our team will assist you promptly."}

    async def _generate_tutorial_content(self, tenant_id: str, topic: str) -> Dict[str, Any]:
        """Generate content for an onboarding tutorial"""
        prompt = f"You are a tutorial generator for AgentSystem. Create a detailed onboarding tutorial for the topic '{topic}'. Include step-by-step instructions, best practices, and tips for new users."
        try:
            if ai_service:
                return await ai_service.generate_text(prompt, max_tokens=1000)
            else:
                return {"text": f"This is a placeholder tutorial for {topic}. Detailed step-by-step guidance will be available soon. For now, please refer to the user guide or contact support for assistance."}
        except Exception as e:
            logger.error(f"Error generating tutorial content: {e}")
            return {"text": f"Sorry, there was an error generating the tutorial for {topic}. Please check back later or contact support for help with onboarding."}
