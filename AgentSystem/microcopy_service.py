import json
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
import uuid

logger = logging.getLogger(__name__)

class MicrocopyService:
    def __init__(self):
        self.variants = {}
        self.interactions = []
        self._initialize_sample_data()

    def _initialize_sample_data(self):
        """Initialize with sample data from schema"""
        sample_variants = [
            {
                "key": "task_create_button",
                "variant_name": "control",
                "context": "button",
                "type": "cta",
                "content": {"text": "Create New Task", "tooltip": "Create a new task for AI agents"},
                "is_active": True
            },
            {
                "key": "task_create_button",
                "variant_name": "variant_a",
                "context": "button",
                "type": "cta",
                "content": {"text": "+ New Task", "tooltip": "Start a new task"},
                "is_active": True
            },
            {
                "key": "task_create_button",
                "variant_name": "variant_b",
                "context": "button",
                "type": "cta",
                "content": {"text": "Start a Task", "tooltip": "Get your AI agents working on something new"},
                "is_active": True
            },
            {
                "key": "task_submit_button",
                "variant_name": "control",
                "context": "button",
                "type": "cta",
                "content": {"text": "Submit Task"},
                "is_active": True
            },
            {
                "key": "task_submit_button",
                "variant_name": "variant_a",
                "context": "button",
                "type": "cta",
                "content": {"text": "Launch Task Now", "icon": "🚀"},
                "is_active": True
            },
            {
                "key": "error_task_not_found",
                "variant_name": "control",
                "context": "error",
                "type": "error",
                "content": {"message": "Task not found", "help": None},
                "is_active": True
            },
            {
                "key": "error_task_not_found",
                "variant_name": "variant_a",
                "context": "error",
                "type": "error",
                "content": {
                    "message": "We couldn't find that task",
                    "help": "It may have been deleted or the link is incorrect. Try refreshing the page.",
                    "action": {"label": "View All Tasks", "url": "/tasks"}
                },
                "is_active": True
            },
            {
                "key": "success_task_created",
                "variant_name": "control",
                "context": "success",
                "type": "success",
                "content": {"message": "Task created successfully"},
                "is_active": True
            },
            {
                "key": "success_task_created",
                "variant_name": "variant_a",
                "context": "success",
                "type": "success",
                "content": {
                    "title": "Task Created Successfully",
                    "message": "Your AI agent will start working on it soon",
                    "action": {"label": "Track Progress", "icon": "→"}
                },
                "is_active": True
            },
            {
                "key": "form_task_name",
                "variant_name": "control",
                "context": "form",
                "type": "label",
                "content": {"label": "Task Name", "placeholder": None, "hint": None},
                "is_active": True
            },
            {
                "key": "form_task_name",
                "variant_name": "variant_a",
                "context": "form",
                "type": "label",
                "content": {
                    "label": "What should we call this task?",
                    "placeholder": "e.g., Analyze customer feedback from Q1",
                    "hint": "Keep it short and descriptive (max 100 characters)"
                },
                "is_active": True
            },
            {
                "key": "empty_tasks",
                "variant_name": "control",
                "context": "empty",
                "type": "help",
                "content": {"message": "No tasks"},
                "is_active": True
            },
            {
                "key": "empty_tasks",
                "variant_name": "variant_a",
                "context": "empty",
                "type": "help",
                "content": {
                    "icon": "📋",
                    "title": "No tasks yet",
                    "message": "Create your first task to get your AI agents working",
                    "action": {"label": "Create Your First Task", "onclick": "openTaskModal()"}
                },
                "is_active": True
            }
        ]

        for v in sample_variants:
            v_id = str(uuid.uuid4())
            v['id'] = v_id
            self.variants[v_id] = v

    def get_active_variants(self) -> List[Dict[str, Any]]:
        """Get all active variants"""
        return [v for v in self.variants.values() if v.get('is_active', True)]

    def get_variants_by_key(self, key: str) -> List[Dict[str, Any]]:
        """Get variants for a specific key"""
        return [v for v in self.variants.values() if v['key'] == key]

    def track_interaction(self, variant_id: str, interaction_data: Dict[str, Any]) -> bool:
        """Track user interaction"""
        if variant_id not in self.variants:
            return False

        interaction = {
            "id": str(uuid.uuid4()),
            "variant_id": variant_id,
            "timestamp": datetime.utcnow().isoformat(),
            **interaction_data
        }
        self.interactions.append(interaction)
        logger.info(f"Tracked interaction for variant {variant_id}: {interaction_data.get('interaction_type')}")
        return True

    def get_effectiveness_report(self, key: str) -> Dict[str, Any]:
        """Generate simple effectiveness report"""
        variants = self.get_variants_by_key(key)
        report = []

        for v in variants:
            v_interactions = [i for i in self.interactions if i['variant_id'] == v['id']]
            views = len([i for i in v_interactions if i.get('interaction_type') == 'view'])
            clicks = len([i for i in v_interactions if i.get('interaction_type') == 'click'])

            report.append({
                "variant_id": v['id'],
                "variant_name": v['variant_name'],
                "metrics": {
                    "total_views": views,
                    "total_clicks": clicks,
                    "ctr": clicks / views if views > 0 else 0
                }
            })

        return {
            "key": key,
            "variants": report
        }

microcopy_service = MicrocopyService()
