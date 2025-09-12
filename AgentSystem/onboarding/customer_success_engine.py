"""
AgentSystem Automated Onboarding and Customer Success Engine
Intelligent customer journey automation for maximum retention and growth
"""

import asyncio
import json
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import logging
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

class OnboardingStage(Enum):
    """Customer onboarding stages"""
    SIGNUP = "signup"
    EMAIL_VERIFICATION = "email_verification"
    PROFILE_SETUP = "profile_setup"
    FIRST_AGENT_CREATION = "first_agent_creation"
    FIRST_WORKFLOW = "first_workflow"
    INTEGRATION_SETUP = "integration_setup"
    FIRST_SUCCESS = "first_success"
    COMPLETED = "completed"

class CustomerHealthScore(Enum):
    """Customer health scoring"""
    EXCELLENT = "excellent"  # 90-100
    GOOD = "good"           # 70-89
    AT_RISK = "at_risk"     # 50-69
    CRITICAL = "critical"   # 0-49

@dataclass
class OnboardingStep:
    """Individual onboarding step"""
    step_id: str
    title: str
    description: str
    stage: OnboardingStage
    required: bool
    estimated_time_minutes: int
    completion_criteria: Dict[str, Any]
    help_resources: List[str]
    automation_triggers: List[str]

@dataclass
class CustomerJourney:
    """Customer journey tracking"""
    tenant_id: str
    current_stage: OnboardingStage
    completed_steps: List[str]
    health_score: float
    health_status: CustomerHealthScore
    last_activity: datetime
    engagement_score: float
    risk_factors: List[str]
    success_milestones: List[str]

class CustomerSuccessEngine:
    """Automated customer success and onboarding engine"""

    def __init__(self, db_connection_string: str):
        self.engine = create_engine(db_connection_string)
        self.Session = sessionmaker(bind=self.engine)
        self.logger = logging.getLogger(__name__)

        # Define onboarding flow
        self.onboarding_steps = self._define_onboarding_flow()

        # Success metrics thresholds
        self.success_thresholds = {
            "time_to_first_value": 24 * 60,  # 24 hours in minutes
            "activation_score": 70,
            "engagement_score": 60,
            "health_score": 75
        }

    def _define_onboarding_flow(self) -> List[OnboardingStep]:
        """Define comprehensive onboarding flow"""
        return [
            OnboardingStep(
                step_id="welcome_email",
                title="Welcome to AgentSystem",
                description="Send personalized welcome email with getting started guide",
                stage=OnboardingStage.SIGNUP,
                required=True,
                estimated_time_minutes=2,
                completion_criteria={"email_sent": True, "email_opened": True},
                help_resources=["getting_started_guide", "video_tour"],
                automation_triggers=["user_signup"]
            ),
            OnboardingStep(
                step_id="email_verification",
                title="Verify Your Email",
                description="Confirm email address to activate account",
                stage=OnboardingStage.EMAIL_VERIFICATION,
                required=True,
                estimated_time_minutes=5,
                completion_criteria={"email_verified": True},
                help_resources=["email_verification_help"],
                automation_triggers=["verification_reminder_24h", "verification_reminder_72h"]
            ),
            OnboardingStep(
                step_id="profile_setup",
                title="Complete Your Profile",
                description="Set up company information and preferences",
                stage=OnboardingStage.PROFILE_SETUP,
                required=True,
                estimated_time_minutes=10,
                completion_criteria={"company_name": True, "industry": True, "team_size": True},
                help_resources=["profile_setup_guide"],
                automation_triggers=["profile_incomplete_reminder"]
            ),
            OnboardingStep(
                step_id="first_agent_creation",
                title="Create Your First AI Agent",
                description="Build your first AI agent to see the power of automation",
                stage=OnboardingStage.FIRST_AGENT_CREATION,
                required=True,
                estimated_time_minutes=15,
                completion_criteria={"agent_created": True, "agent_configured": True},
                help_resources=["agent_creation_tutorial", "agent_templates"],
                automation_triggers=["agent_creation_reminder", "guided_agent_setup"]
            ),
            OnboardingStep(
                step_id="first_workflow",
                title="Build Your First Workflow",
                description="Create an automated workflow to streamline your processes",
                stage=OnboardingStage.FIRST_WORKFLOW,
                required=True,
                estimated_time_minutes=20,
                completion_criteria={"workflow_created": True, "workflow_executed": True},
                help_resources=["workflow_builder_guide", "workflow_templates"],
                automation_triggers=["workflow_creation_assistance"]
            ),
            OnboardingStep(
                step_id="integration_setup",
                title="Connect Your Tools",
                description="Integrate with your existing tools and platforms",
                stage=OnboardingStage.INTEGRATION_SETUP,
                required=False,
                estimated_time_minutes=25,
                completion_criteria={"integration_connected": True},
                help_resources=["integration_guides", "api_documentation"],
                automation_triggers=["integration_suggestions"]
            ),
            OnboardingStep(
                step_id="first_success",
                title="Achieve First Success",
                description="Complete your first successful automation",
                stage=OnboardingStage.FIRST_SUCCESS,
                required=True,
                estimated_time_minutes=30,
                completion_criteria={"automation_success": True, "value_realized": True},
                help_resources=["success_stories", "optimization_tips"],
                automation_triggers=["success_celebration", "expansion_opportunities"]
            )
        ]

    async def start_customer_onboarding(self, tenant_id: str, user_data: Dict[str, Any]) -> Dict[str, Any]:
        """Initialize customer onboarding journey"""

        try:
            # Create customer journey record
            journey = CustomerJourney(
                tenant_id=tenant_id,
                current_stage=OnboardingStage.SIGNUP,
                completed_steps=[],
                health_score=50.0,  # Starting score
                health_status=CustomerHealthScore.AT_RISK,
                last_activity=datetime.now(),
                engagement_score=0.0,
                risk_factors=[],
                success_milestones=[]
            )

            # Store in database
            await self._save_customer_journey(journey)

            # Trigger welcome sequence
            await self._trigger_onboarding_step("welcome_email", tenant_id, user_data)

            # Schedule follow-up automations
            await self._schedule_onboarding_automations(tenant_id)

            return {
                "success": True,
                "journey_id": tenant_id,
                "current_stage": journey.current_stage.value,
                "next_steps": await self._get_next_onboarding_steps(tenant_id),
                "estimated_completion_time": await self._calculate_estimated_completion_time(tenant_id)
            }

        except Exception as e:
            self.logger.error(f"Failed to start onboarding for {tenant_id}: {e}")
            raise

    async def update_onboarding_progress(self, tenant_id: str, step_id: str, completion_data: Dict[str, Any]) -> Dict[str, Any]:
        """Update customer onboarding progress"""

        try:
            # Get current journey
            journey = await self._get_customer_journey(tenant_id)
            if not journey:
                raise ValueError(f"No onboarding journey found for tenant {tenant_id}")

            # Mark step as completed
            if step_id not in journey.completed_steps:
                journey.completed_steps.append(step_id)

            # Update stage if necessary
            step = next((s for s in self.onboarding_steps if s.step_id == step_id), None)
            if step:
                journey.current_stage = step.stage

            # Recalculate health score
            journey.health_score = await self._calculate_health_score(tenant_id, journey)
            journey.health_status = self._get_health_status(journey.health_score)
            journey.last_activity = datetime.now()

            # Update engagement score
            journey.engagement_score = await self._calculate_engagement_score(tenant_id)

            # Check for stage advancement
            if await self._should_advance_stage(journey):
                journey.current_stage = await self._get_next_stage(journey.current_stage)

            # Save updated journey
            await self._save_customer_journey(journey)

            # Trigger next steps
            await self._trigger_next_onboarding_steps(tenant_id, journey)

            # Check for success milestones
            await self._check_success_milestones(tenant_id, journey, completion_data)

            return {
                "success": True,
                "current_stage": journey.current_stage.value,
                "progress_percentage": await self._calculate_progress_percentage(journey),
                "health_score": journey.health_score,
                "health_status": journey.health_status.value,
                "next_steps": await self._get_next_onboarding_steps(tenant_id),
                "achievements": await self._get_recent_achievements(tenant_id)
            }

        except Exception as e:
            self.logger.error(f"Failed to update onboarding progress for {tenant_id}: {e}")
            raise

    async def calculate_customer_health(self, tenant_id: str) -> Dict[str, Any]:
        """Calculate comprehensive customer health score"""

        try:
            journey = await self._get_customer_journey(tenant_id)
            if not journey:
                return {"error": "Customer journey not found"}

            # Calculate various health factors
            health_factors = {
                "onboarding_progress": await self._calculate_onboarding_progress_score(journey),
                "engagement_level": await self._calculate_engagement_score(tenant_id),
                "feature_adoption": await self._calculate_feature_adoption_score(tenant_id),
                "usage_frequency": await self._calculate_usage_frequency_score(tenant_id),
                "support_interactions": await self._calculate_support_score(tenant_id),
                "billing_health": await self._calculate_billing_health_score(tenant_id)
            }

            # Calculate weighted overall score
            weights = {
                "onboarding_progress": 0.25,
                "engagement_level": 0.20,
                "feature_adoption": 0.20,
                "usage_frequency": 0.15,
                "support_interactions": 0.10,
                "billing_health": 0.10
            }

            overall_score = sum(health_factors[factor] * weights[factor] for factor in health_factors)

            # Identify risk factors
            risk_factors = []
            if health_factors["engagement_level"] < 40:
                risk_factors.append("low_engagement")
            if health_factors["usage_frequency"] < 30:
                risk_factors.append("low_usage")
            if health_factors["support_interactions"] < 50:
                risk_factors.append("support_issues")
            if health_factors["billing_health"] < 70:
                risk_factors.append("billing_concerns")

            # Generate recommendations
            recommendations = await self._generate_health_recommendations(tenant_id, health_factors, risk_factors)

            # Update journey with new health data
            journey.health_score = overall_score
            journey.health_status = self._get_health_status(overall_score)
            journey.risk_factors = risk_factors
            await self._save_customer_journey(journey)

            return {
                "tenant_id": tenant_id,
                "overall_health_score": overall_score,
                "health_status": self._get_health_status(overall_score).value,
                "health_factors": health_factors,
                "risk_factors": risk_factors,
                "recommendations": recommendations,
                "last_calculated": datetime.now().isoformat()
            }

        except Exception as e:
            self.logger.error(f"Failed to calculate customer health for {tenant_id}: {e}")
            raise

    async def trigger_intervention(self, tenant_id: str, intervention_type: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Trigger automated customer success intervention"""

        interventions = {
            "low_engagement": self._intervention_low_engagement,
            "onboarding_stuck": self._intervention_onboarding_stuck,
            "feature_adoption": self._intervention_feature_adoption,
            "churn_risk": self._intervention_churn_risk,
            "expansion_opportunity": self._intervention_expansion_opportunity,
            "success_milestone": self._intervention_success_milestone
        }

        if intervention_type not in interventions:
            raise ValueError(f"Unknown intervention type: {intervention_type}")

        try:
            result = await interventions[intervention_type](tenant_id, context)

            # Log intervention
            await self._log_intervention(tenant_id, intervention_type, context, result)

            return result

        except Exception as e:
            self.logger.error(f"Failed to trigger intervention {intervention_type} for {tenant_id}: {e}")
            raise

    async def _calculate_health_score(self, tenant_id: str, journey: CustomerJourney) -> float:
        """Calculate customer health score based on multiple factors"""

        # Base score from onboarding progress
        progress_score = len(journey.completed_steps) / len(self.onboarding_steps) * 100

        # Engagement score (recent activity)
        days_since_activity = (datetime.now() - journey.last_activity).days
        engagement_score = max(0, 100 - (days_since_activity * 10))

        # Usage score (from analytics)
        usage_score = await self._get_usage_score(tenant_id)

        # Weighted average
        health_score = (progress_score * 0.4 + engagement_score * 0.3 + usage_score * 0.3)

        return min(100, max(0, health_score))

    async def _get_usage_score(self, tenant_id: str) -> float:
        """Get usage-based score from analytics"""

        with self.Session() as session:
            # Get recent usage data
            query = text("""
                SELECT COALESCE(SUM(api_calls), 0) as total_calls,
                       COALESCE(SUM(ai_tokens), 0) as total_tokens
                FROM usage_tracking
                WHERE tenant_id = :tenant_id
                AND timestamp >= :start_date
            """)

            start_date = datetime.now() - timedelta(days=7)
            result = session.execute(query, {
                "tenant_id": tenant_id,
                "start_date": start_date
            }).fetchone()

            total_calls = result.total_calls if result.total_calls else 0

            # Score based on usage (0-100 scale)
            if total_calls == 0:
                return 0
            elif total_calls < 100:
                return 30
            elif total_calls < 500:
                return 60
            elif total_calls < 1000:
                return 80
            else:
                return 100

    def _get_health_status(self, score: float) -> CustomerHealthScore:
        """Convert numeric health score to status"""
        if score >= 90:
            return CustomerHealthScore.EXCELLENT
        elif score >= 70:
            return CustomerHealthScore.GOOD
        elif score >= 50:
            return CustomerHealthScore.AT_RISK
        else:
            return CustomerHealthScore.CRITICAL

    async def _intervention_low_engagement(self, tenant_id: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Intervention for low engagement customers"""

        actions = [
            {
                "type": "email",
                "template": "re_engagement_series",
                "personalization": await self._get_customer_personalization(tenant_id)
            },
            {
                "type": "in_app_message",
                "content": "We noticed you haven't been active lately. Let us help you get the most out of AgentSystem!",
                "cta": "Get Personal Demo"
            },
            {
                "type": "customer_success_outreach",
                "priority": "high",
                "message": "Schedule proactive check-in call"
            }
        ]

        # Execute actions
        for action in actions:
            await self._execute_intervention_action(tenant_id, action)

        return {
            "intervention": "low_engagement",
            "actions_taken": len(actions),
            "expected_outcome": "increased_engagement",
            "follow_up_date": (datetime.now() + timedelta(days=3)).isoformat()
        }

    async def _intervention_onboarding_stuck(self, tenant_id: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Intervention for customers stuck in onboarding"""

        current_step = context.get("current_step", "unknown")

        actions = [
            {
                "type": "guided_tutorial",
                "step": current_step,
                "interactive": True
            },
            {
                "type": "email",
                "template": "onboarding_assistance",
                "step_specific": True
            },
            {
                "type": "calendar_booking",
                "purpose": "onboarding_assistance",
                "duration": 30
            }
        ]

        for action in actions:
            await self._execute_intervention_action(tenant_id, action)

        return {
            "intervention": "onboarding_stuck",
            "stuck_step": current_step,
            "actions_taken": len(actions),
            "assistance_provided": True
        }

    async def _save_customer_journey(self, journey: CustomerJourney):
        """Save customer journey to database"""

        with self.Session() as session:
            # Upsert journey data
            query = text("""
                INSERT INTO customer_journeys (
                    tenant_id, current_stage, completed_steps, health_score,
                    health_status, last_activity, engagement_score, risk_factors,
                    success_milestones, updated_at
                ) VALUES (
                    :tenant_id, :current_stage, :completed_steps, :health_score,
                    :health_status, :last_activity, :engagement_score, :risk_factors,
                    :success_milestones, CURRENT_TIMESTAMP
                ) ON CONFLICT (tenant_id) DO UPDATE SET
                    current_stage = EXCLUDED.current_stage,
                    completed_steps = EXCLUDED.completed_steps,
                    health_score = EXCLUDED.health_score,
                    health_status = EXCLUDED.health_status,
                    last_activity = EXCLUDED.last_activity,
                    engagement_score = EXCLUDED.engagement_score,
                    risk_factors = EXCLUDED.risk_factors,
                    success_milestones = EXCLUDED.success_milestones,
                    updated_at = EXCLUDED.updated_at
            """)

            session.execute(query, {
                "tenant_id": journey.tenant_id,
                "current_stage": journey.current_stage.value,
                "completed_steps": json.dumps(journey.completed_steps),
                "health_score": journey.health_score,
                "health_status": journey.health_status.value,
                "last_activity": journey.last_activity,
                "engagement_score": journey.engagement_score,
                "risk_factors": json.dumps(journey.risk_factors),
                "success_milestones": json.dumps(journey.success_milestones)
            })
            session.commit()

    async def _get_customer_journey(self, tenant_id: str) -> Optional[CustomerJourney]:
        """Get customer journey from database"""

        with self.Session() as session:
            query = text("""
                SELECT tenant_id, current_stage, completed_steps, health_score,
                       health_status, last_activity, engagement_score, risk_factors,
                       success_milestones
                FROM customer_journeys
                WHERE tenant_id = :tenant_id
            """)

            result = session.execute(query, {"tenant_id": tenant_id}).fetchone()

            if not result:
                return None

            return CustomerJourney(
                tenant_id=result.tenant_id,
                current_stage=OnboardingStage(result.current_stage),
                completed_steps=json.loads(result.completed_steps),
                health_score=result.health_score,
                health_status=CustomerHealthScore(result.health_status),
                last_activity=result.last_activity,
                engagement_score=result.engagement_score,
                risk_factors=json.loads(result.risk_factors),
                success_milestones=json.loads(result.success_milestones)
            )

class OnboardingAutomationEngine:
    """Automated onboarding workflow engine"""

    def __init__(self, customer_success_engine: CustomerSuccessEngine):
        self.cs_engine = customer_success_engine
        self.logger = logging.getLogger(__name__)

    async def run_daily_health_checks(self):
        """Run daily customer health assessments"""

        # Get all active customers
        with self.cs_engine.Session() as session:
            query = text("""
                SELECT tenant_id FROM customer_journeys
                WHERE current_stage != 'completed'
                AND last_activity >= :cutoff_date
            """)

            cutoff_date = datetime.now() - timedelta(days=30)
            results = session.execute(query, {"cutoff_date": cutoff_date}).fetchall()

        # Process each customer
        for row in results:
            tenant_id = row.tenant_id

            try:
                # Calculate health score
                health_data = await self.cs_engine.calculate_customer_health(tenant_id)

                # Trigger interventions if needed
                if health_data["health_status"] in ["critical", "at_risk"]:
                    await self._trigger_health_interventions(tenant_id, health_data)

            except Exception as e:
                self.logger.error(f"Failed health check for {tenant_id}: {e}")

    async def _trigger_health_interventions(self, tenant_id: str, health_data: Dict[str, Any]):
        """Trigger appropriate interventions based on health data"""

        risk_factors = health_data.get("risk_factors", [])

        for risk_factor in risk_factors:
            intervention_type = {
                "low_engagement": "low_engagement",
                "low_usage": "low_engagement",
                "support_issues": "customer_success_outreach",
                "billing_concerns": "billing_assistance"
            }.get(risk_factor)

            if intervention_type:
                await self.cs_engine.trigger_intervention(
                    tenant_id,
                    intervention_type,
                    {"health_data": health_data}
                )
