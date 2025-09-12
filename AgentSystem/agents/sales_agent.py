
"""
💰 AgentSystem Sales Automation Agent
AI-powered sales automation for prospect research, lead qualification, and email sequences
"""

import asyncio
import json
import re
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
from enum import Enum
import logging
import hashlib
import requests
from urllib.parse import urlparse
import pandas as pd
import numpy as np

import asyncpg
import aioredis
from fastapi import HTTPException
from pydantic import BaseModel, Field, EmailStr, HttpUrl
import openai
import anthropic

from ..core.agent_swarm import SpecializedAgent, AgentCapability
from ..usage.usage_tracker import ServiceType, track_ai_request
from ..optimization.cost_optimizer_clean import CostOptimizer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LeadStatus(str, Enum):
    NEW = "new"
    QUALIFIED = "qualified"
    CONTACTED = "contacted"
    ENGAGED = "engaged"
    OPPORTUNITY = "opportunity"
    CUSTOMER = "customer"
    LOST = "lost"
    UNRESPONSIVE = "unresponsive"

class LeadSource(str, Enum):
    WEBSITE = "website"
    SOCIAL_MEDIA = "social_media"
    EMAIL_CAMPAIGN = "email_campaign"
    REFERRAL = "referral"
    COLD_OUTREACH = "cold_outreach"
    TRADE_SHOW = "trade_show"
    WEBINAR = "webinar"
    CONTENT_MARKETING = "content_marketing"

class EmailSequenceType(str, Enum):
    COLD_OUTREACH = "cold_outreach"
    WARM_FOLLOW_UP = "warm_follow_up"
    NURTURE_SEQUENCE = "nurture_sequence"
    RE_ENGAGEMENT = "re_engagement"
    ONBOARDING = "onboarding"
    UPSELL = "upsell"

class CompanySize(str, Enum):
    STARTUP = "startup"          # 1-10 employees
    SMALL = "small"              # 11-50 employees
    MEDIUM = "medium"            # 51-200 employees
    LARGE = "large"              # 201-1000 employees
    ENTERPRISE = "enterprise"    # 1000+ employees

@dataclass
class Lead:
    """Lead/prospect data structure"""
    id: str
    email: str
    first_name: str
    last_name: str
    company: str
    title: str
    phone: Optional[str] = None
    linkedin_url: Optional[str] = None
    company_website: Optional[str] = None
    company_size: Optional[CompanySize] = None
    industry: Optional[str] = None
    lead_source: LeadSource = LeadSource.WEBSITE
    status: LeadStatus = LeadStatus.NEW
    lead_score: float = 0.0
    created_at: datetime = None
    last_contacted: Optional[datetime] = None
    notes: List[str] = None
    custom_fields: Dict[str, Any] = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.notes is None:
            self.notes = []
        if self.custom_fields is None:
            self.custom_fields = {}

@dataclass
class ProspectResearch:
    """Prospect research results"""
    lead_id: str
    company_info: Dict[str, Any]
    contact_info: Dict[str, Any]
    social_presence: Dict[str, Any]
    recent_news: List[Dict[str, Any]]
    technology_stack: List[str]
    competitors: List[str]
    pain_points: List[str]
    buying_signals: List[str]
    personalization_angles: List[str]
    research_score: float
    confidence_level: float

@dataclass
class EmailSequence:
    """Email sequence configuration"""
    id: str
    name: str
    sequence_type: EmailSequenceType
    target_audience: str
    emails: List[Dict[str, Any]]
    timing_days: List[int]
    open_rate: float = 0.0
    reply_rate: float = 0.0
    conversion_rate: float = 0.0
    active: bool = True

class SalesAutomationAgent(SpecializedAgent):
    """AI-powered sales automation agent"""

    def __init__(self, tenant_id: str, db_pool: asyncpg.Pool, redis_client: aioredis.Redis,
                 cost_optimizer: CostOptimizer):
        super().__init__(
            agent_id=f"sales_agent_{tenant_id}",
            agent_type="sales_automation",
            capabilities=[
                AgentCapability.LEAD_QUALIFICATION,
                AgentCapability.PROSPECT_RESEARCH,
                AgentCapability.EMAIL_AUTOMATION,
                AgentCapability.CRM_INTEGRATION,
                AgentCapability.SALES_ANALYTICS
            ]
        )

        self.tenant_id = tenant_id
        self.db_pool = db_pool
        self.redis = redis_client
        self.cost_optimizer = cost_optimizer

        # Research tools and APIs
        self.research_tools = {
            'linkedin_api': 'https://api.linkedin.com/v2/',
            'clearbit_api': 'https://person.clearbit.com/v1/people/email/',
            'hunter_io': 'https://api.hunter.io/v2/',
            'crunchbase': 'https://api.crunchbase.com/v4/',
            'google_news': 'https://newsapi.org/v2/everything'
        }

        # Lead scoring model weights
        self.scoring_weights = {
            'company_size': 0.25,
            'title_seniority': 0.20,
            'industry_fit': 0.15,
            'engagement_level': 0.15,
            'buying_signals': 0.15,
            'technology_fit': 0.10
        }

        # Email templates for different sequences
        self.email_templates = self._initialize_email_templates()

    def _initialize_email_templates(self) -> Dict[str, List[str]]:
        """Initialize email sequence templates"""
        return {
            EmailSequenceType.COLD_OUTREACH: [
                # Email 1: Initial outreach
                """
                Subject: Quick question about {company}'s {pain_point}

                Hi {first_name},

                I noticed {company} is {recent_news_context}. This often indicates challenges with {pain_point}.

                I've helped similar {industry} companies like {competitor_example} {specific_result}.

                Would you be open to a 15-minute conversation to explore if we could help {company} achieve similar results?

                Best regards,
                {sender_name}
                """,

                # Email 2: Follow-up with value
                """
                Subject: {first_name}, thought you'd find this interesting

                Hi {first_name},

                Following up on my previous email about {pain_point}.

                I came across this case study showing how {similar_company} {specific_achievement}.
                Given {company}'s {specific_situation}, I thought this might be relevant.

                [Case study link]

                Would love to hear your thoughts on this approach for {company}.

                Best,
                {sender_name}
                """,

                # Email 3: Social proof and final attempt
                """
                Subject: Last note - {company_benefit}

                {first_name},

                I'll keep this brief since I know you're busy.

                Three companies in {industry} have recently implemented our solution:
                - {company1}: {result1}
                - {company2}: {result2}
                - {company3}: {result3}

                If {company} could achieve even 50% of these results, would that be worth a brief conversation?

                Either way, I won't reach out again unless you express interest.

                {sender_name}
                """
            ],

            EmailSequenceType.NURTURE_SEQUENCE: [
                # Nurture sequence for leads not ready to buy
                """
                Subject: Industry insights for {company}

                Hi {first_name},

                Hope you're doing well. I wanted to share some insights from our latest {industry} report that might interest you.

                Key findings:
                - {insight1}
                - {insight2}
                - {insight3}

                [Download full report]

                Let me know if you'd like to discuss how these trends might impact {company}.

                Best,
                {sender_name}
                """
            ]
        }

    async def research_prospect(self, lead: Lead) -> ProspectResearch:
        """Conduct comprehensive prospect research"""

        try:
            # Parallel research across multiple sources
            research_tasks = [
                self._research_company_info(lead.company, lead.company_website),
                self._research_contact_info(lead.email, lead.linkedin_url),
                self._research_social_presence(lead.company, lead.first_name, lead.last_name),
                self._research_recent_news(lead.company),
                self._research_technology_stack(lead.company_website),
                self._research_competitors(lead.company, lead.industry)
            ]

            research_results = await asyncio.gather(*research_tasks, return_exceptions=True)

            # Compile research data
            company_info = research_results[0] if not isinstance(research_results[0], Exception) else {}
            contact_info = research_results[1] if not isinstance(research_results[1], Exception) else {}
            social_presence = research_results[2] if not isinstance(research_results[2], Exception) else {}
            recent_news = research_results[3] if not isinstance(research_results[3], Exception) else []
            technology_stack = research_results[4] if not isinstance(research_results[4], Exception) else []
            competitors = research_results[5] if not isinstance(research_results[5], Exception) else []

            # Analyze pain points and buying signals using AI
            analysis_prompt = f"""
            Analyze the following prospect information and identify:
            1. Potential pain points
            2. Buying signals
            3. Personalization angles for outreach

            Company: {lead.company}
            Industry: {lead.industry}
            Contact: {lead.first_name} {lead.last_name}, {lead.title}
            Company Info: {json.dumps(company_info, default=str)}
            Recent News: {json.dumps(recent_news, default=str)}
            Technology: {technology_stack}
            """

            analysis = await self._get_ai_analysis(analysis_prompt)

            # Extract structured data from analysis
            pain_points = self._extract_pain_points(analysis)
            buying_signals = self._extract_buying_signals(analysis)
            personalization_angles = self._extract_personalization_angles(analysis)

            # Calculate research score and confidence
            research_score = self._calculate_research_score(
                company_info, contact_info, social_presence, recent_news, technology_stack
            )

            research = ProspectResearch(
                lead_id=lead.id,
                company_info=company_info,
                contact_info=contact_info,
                social_presence=social_presence,
                recent_news=recent_news,
                technology_stack=technology_stack,
                competitors=competitors,
                pain_points=pain_points,
                buying_signals=buying_signals,
                personalization_angles=personalization_angles,
                research_score=research_score,
                confidence_level=min(research_score / 100.0, 1.0)
            )

            # Store research results
            await self._store_prospect_research(research)

            return research

        except Exception as e:
            logger.error(f"Prospect research failed for {lead.id}: {e}")
            raise HTTPException(status_code=500, detail=f"Prospect research failed: {str(e)}")

    async def _research_company_info(self, company: str, website: Optional[str]) -> Dict[str, Any]:
        """Research company information"""

        company_info = {
            'name': company,
            'website': website,
            'description': '',
            'size': '',
            'industry': '',
            'founded': '',
            'location': '',
            'revenue': '',
            'funding': ''
        }

        # Placeholder for actual API integrations
        # In production, would integrate with Clearbit, Crunchbase, etc.
        if website:
            try:
                # Scrape basic company info from website
                response = requests.get(website, timeout=10)
                if response.status_code == 200:
                    # Extract company description from meta tags
                    from bs4 import BeautifulSoup
                    soup = BeautifulSoup(response.content, 'html.parser')

                    meta_desc = soup.find('meta', {'name': 'description'})
                    if meta_desc:
                        company_info['description'] = meta_desc.get('content', '')[:500]

            except Exception as e:
                logger.warning(f"Failed to scrape company website {website}: {e}")

        return company_info

    async def _research_contact_info(self, email: str, linkedin_url: Optional[str]) -> Dict[str, Any]:
        """Research contact information"""

        contact_info = {
            'email': email,
            'linkedin_url': linkedin_url,
            'social_profiles': [],
            'professional_background': '',
            'recent_posts': [],
            'connections': 0,
            'influence_score': 0
        }

        # Placeholder for LinkedIn API integration
        if linkedin_url:
            # Would integrate with LinkedIn API to get professional background
            contact_info['professional_background'] = "Senior executive with experience in technology and business development"
            contact_info['connections'] = 500  # Placeholder
            contact_info['influence_score'] = 75   # Placeholder

        return contact_info

    async def _research_social_presence(self, company: str, first_name: str, last_name: str) -> Dict[str, Any]:
        """Research social media presence"""

        social_presence = {
            'company_social': {
                'linkedin': f"https://linkedin.com/company/{company.lower().replace(' ', '-')}",
                'twitter': f"https://twitter.com/{company.lower().replace(' ', '')}",
                'facebook': '',
                'instagram': ''
            },
            'personal_social': {
                'linkedin': f"https://linkedin.com/in/{first_name.lower()}-{last_name.lower()}",
                'twitter': f"https://twitter.com/{first_name.lower()}{last_name.lower()}"
            },
            'recent_activity': [],
            'engagement_level': 'medium'
        }

        return social_presence

    async def _research_recent_news(self, company: str) -> List[Dict[str, Any]]:
        """Research recent company news"""

        news_items = []

        # Placeholder for news API integration
        # In production, would use NewsAPI, Google News, etc.
        sample_news = [
            {
                'title': f'{company} announces new product launch',
                'url': 'https://example.com/news1',
                'published_date': (datetime.now() - timedelta(days=7)).isoformat(),
                'source': 'TechCrunch',
                'sentiment': 'positive'
            },
            {
                'title': f'{company} raises Series B funding',
                'url': 'https://example.com/news2',
                'published_date': (datetime.now() - timedelta(days=30)).isoformat(),
                'source': 'VentureBeat',
                'sentiment': 'positive'
            }
        ]

        news_items.extend(sample_news)

        return news_items

    async def _research_technology_stack(self, website: Optional[str]) -> List[str]:
        """Research company's technology stack"""

        tech_stack = []

        if website:
            # Placeholder for technology detection
            # In production, would use tools like Wappalyzer, BuiltWith, etc.
            sample_tech = [
                'React', 'Node.js', 'AWS', 'Salesforce', 'Google Analytics',
                'Stripe', 'Intercom', 'Slack', 'Zoom', 'Microsoft Office 365'
            ]
            tech_stack = sample_tech[:5]  # Return subset

        return tech_stack

    async def _research_competitors(self, company: str, industry: Optional[str]) -> List[str]:
        """Research company competitors"""

        competitors = []

        # Placeholder for competitive analysis
        # In production, would use Crunchbase, SimilarWeb, etc.
        if industry:
            sample_competitors = [
                f"{industry} Solutions Inc",
                f"Global {industry} Corp",
                f"{industry} Tech Leader"
            ]
            competitors = sample_competitors

        return competitors

    async def _get_ai_analysis(self, prompt: str) -> str:
        """Get AI analysis for prospect research"""

        # Optimize AI request
        optimization_result = await self.cost_optimizer.optimize_request(
            tenant_id=self.tenant_id,
            request_data={
                'prompt': prompt,
                'max_tokens': 1000,
                'temperature': 0.3,
                'task_type': 'analysis'
            }
        )

        # Call AI provider (simplified)
        if optimization_result.selected_provider == "cache":
            return "Cached analysis result"
        else:
            # Would call actual AI provider
            return f"AI analysis using {optimization_result.selected_model}:\n\nPain Points:\n- Scaling challenges\n- Technology integration issues\n- Cost optimization needs\n\nBuying Signals:\n- Recent funding\n- Job postings for technical roles\n- New product launches\n\nPersonalization:\n- Recent company news\n- Technology stack alignment\n- Industry challenges"

    def _extract_pain_points(self, analysis: str) -> List[str]:
        """Extract pain points from AI analysis"""

        # Simple extraction - in production would use NLP
        pain_points = []
        lines = analysis.split('\n')

        in_pain_points_section = False
        for line in lines:
            if 'pain points:' in line.lower():
                in_pain_points_section = True
                continue
            elif line.startswith('Buying Signals:') or line.startswith('Personalization:'):
                in_pain_points_section = False
                continue

            if in_pain_points_section and line.strip().startswith('-'):
                pain_points.append(line.strip()[1:].strip())

        return pain_points or ['Scaling challenges', 'Technology integration', 'Cost optimization']

    def _extract_buying_signals(self, analysis: str) -> List[str]:
        """Extract buying signals from AI analysis"""

        # Simple extraction - in production would use NLP
        buying_signals = []
        lines = analysis.split('\n')

        in_buying_signals_section = False
        for line in lines:
            if 'buying signals:' in line.lower():
                in_buying_signals_section = True
                continue
            elif line.startswith('Personalization:'):
                in_buying_signals_section = False
                continue

            if in_buying_signals_section and line.strip().startswith('-'):
                buying_signals.append(line.strip()[1:].strip())

        return buying_signals or ['Recent funding', 'Job postings', 'Technology investments']

    def _extract_personalization_angles(self, analysis: str) -> List[str]:
        """Extract personalization angles from AI analysis"""

        # Simple extraction
        personalization = []
        lines = analysis.split('\n')

        in_personalization_section = False
        for line in lines:
            if 'personalization:' in line.lower():
                in_personalization_section = True
                continue

            if in_personalization_section and line.strip().startswith('-'):
                personalization.append(line.strip()[1:].strip())

        return personalization or ['Recent company news', 'Technology alignment', 'Industry expertise']

    def _calculate_research_score(self, company_info: Dict, contact_info: Dict,
                                social_presence: Dict, recent_news: List, tech_stack: List) -> float:
        """Calculate research completeness score"""

        score = 0

        # Company info score (30 points)
        if company_info.get('description'):
            score += 10
        if company_info.get('size'):
            score += 10
        if company_info.get('industry'):
            score += 10

        # Contact info score (25 points)
        if contact_info.get('linkedin_url'):
            score += 15
        if contact_info.get('professional_background'):
            score += 10

        # Social presence score (20 points)
        if social_presence.get('company_social'):
            score += 10
        if social_presence.get('personal_social'):
            score += 10

        # Recent news score (15 points)
        score += min(len(recent_news) * 5, 15)

        # Technology stack score (10 points)
        score += min(len(tech_stack) * 2, 10)

        return min(score, 100)

    async def _store_prospect_research(self, research: ProspectResearch):
        """Store prospect research in database"""

        async with self.db_pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO tenant_management.prospect_research (
                    lead_id, tenant_id, company_info, contact_info, social_presence,
                    recent_news, technology_stack, competitors, pain_points,
                    buying_signals, personalization_angles, research_score,
                    confidence_level, created_at
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14)
                ON CONFLICT (lead_id, tenant_id) DO UPDATE SET
                    company_info = EXCLUDED.company_info,
                    contact_info = EXCLUDED.contact_info,
                    social_presence = EXCLUDED.social_presence,
                    recent_news = EXCLUDED.recent_news,
                    technology_stack = EXCLUDED.technology_stack,
                    competitors = EXCLUDED.competitors,
                    pain_points = EXCLUDED.pain_points,
                    buying_signals = EXCLUDED.buying_signals,
                    personalization_angles = EXCLUDED.personalization_angles,
                    research_score = EXCLUDED.research_score,
                    confidence_level = EXCLUDED.confidence_level,
                    updated_at = NOW()
            """,
                research.lead_id, self.tenant_id,
                json.dumps(research.company_info),
                json.dumps(research.contact_info),
                json.dumps(research.social_presence),
                json.dumps(research.recent_news, default=str),
                json.dumps(research.technology_stack),
                json.dumps(research.competitors),
                json.dumps(research.pain_points),
                json.dumps(research.buying_signals),
                json.dumps(research.personalization_angles),
                research.research_score,
                research.confidence_level,
                datetime.now()
            )

    async def calculate_lead_score(self, lead: Lead, research: Optional[ProspectResearch] = None) -> float:
        """Calculate lead score based on multiple factors"""

        score = 0.0

        # Company size score (25%)
        company_size_scores = {
            CompanySize.STARTUP: 60,
            CompanySize.SMALL: 70,
            CompanySize.MEDIUM: 85,
            CompanySize.LARGE: 90,
            CompanySize.ENTERPRISE: 95
        }
        if lead.company_size:
            score += company_size_scores[lead.company_size] * self.scoring_weights['company_size']

        # Title seniority score (20%)
        title_scores = self._calculate_title_score(lead.title)
        score += title_scores * self.scoring_weights['title_seniority']

        # Industry fit score (15%)
        industry_score = self._calculate_industry_score(lead.industry)
        score += industry_score * self.scoring_weights['industry_fit']

        # Engagement level score (15%)
        engagement_score = await self._calculate_engagement_score(lead)
        score += engagement_score * self.scoring_weights['engagement_level']

        # Buying signals score (15%)
        if research:
            buying_signals_score = len(research.buying_signals) * 20  # 20 points per signal
            score += min(buying_signals_score, 100) * self.scoring_weights['buying_signals']

        # Technology fit score (10%)
        if research:
            tech_fit_score = self._calculate_tech_fit_score(research.technology_stack)
            score += tech_fit_score * self.scoring_weights['technology_fit']

        return min(score, 100.0)

    def _calculate_title_score(self, title: str) -> float:
        """Calculate score based on title seniority"""

        if not title:
            return 50.0

        title_lower = title.lower()

        # Executive level
        if any(exec_title in title_lower for exec_title in ['ceo', 'cto', 'cfo', 'founder', 'president', 'vp', 'vice president']):
            return 95.0

        # Director level
        if any(dir_title in title_lower for dir_title in ['director', 'head of', 'lead']):
            return 85.0

        # Manager level
        if any(mgr_title in title_lower for mgr_title in ['manager', 'senior', 'principal']):
            return 75.0

        # Individual contributor
        return 60.0

    def _calculate_industry_score(self, industry: Optional[str]) -> float:
        """Calculate score based on industry fit"""

        if not industry:
            return 50.0

        # High-value industries for our platform
        high_value_industries = [
            'technology', 'software', 'saas', 'fintech', 'healthcare',
            'e-commerce', 'marketing', 'consulting', 'education'
        ]

        industry_lower = industry.lower()

        for high_value in high_value_industries:
            if high_value in industry_lower:
                return 90.0

        return 70.0  # Default for other industries

    async def _calculate_engagement_score(self, lead: Lead) -> float:
        """Calculate engagement score based on lead behavior"""

        # Placeholder for engagement tracking
        # In production, would track email opens, clicks, website visits, etc.

        engagement_score = 50.0  # Base score

        # Bonus for recent contact
        if lead.last_contacted:
            days_since_contact = (datetime.now() - lead.last_contacted).days
            if days_since_contact < 7:
                engagement_score += 30
            elif days_since_contact < 30:
                engagement_score += 15

        return min(engagement_score, 100.0)

    def _calculate_tech_fit_score(self, tech_stack: List[str]) -> float:
        """Calculate technology fit score"""

        if not tech_stack:
            return 50.0

        # Technologies that indicate good fit for our platform
        complementary_tech = [
            'salesforce', 'hubspot', 'marketo', 'slack', 'teams',
            'aws', 'azure', 'gcp', 'stripe', 'intercom'
        ]

        fit_count = 0
        for tech in tech_stack:
            if any(comp_tech in tech.lower() for comp_tech in complementary_tech):
                fit_count += 1

        # Score based on technology alignment
        return min(50 + (fit_count * 15), 100.0)

    async def generate_personalized_email(self, lead: Lead, research: ProspectResearch,
                                        sequence_type: EmailSequenceType, email_number: int = 1) -> str:
        """Generate personalized email content"""

        # Get email template
        template = self.email_templates[sequence_type][email_number - 1]

        # Prepare personalization variables
        personalization_vars = {
            'first_name': lead.first_name,
            'last_name': lead.last_name,
            'company': lead.company,
            'title': lead.title,
            'industry': lead.industry or 'technology',
            'pain_point': research.pain_points[0] if research.pain_points else 'operational efficiency',
            'recent_news_context': research.recent_news[0]['title'] if research.recent_news else 'expanding operations',
            'competitor_example': research.competitors[0] if research.competitors else 'industry leaders',
            'specific_result': 'increased efficiency by 40%',  # Would be dynamic
            'sender_name': 'Sales Team',  # Would be from tenant settings
            'company_benefit': f'boosting {lead.company} revenue by 25%',
            'similar_company': research.competitors[0] if research.competitors else 'similar companies',
            'specific_achievement': 'reduced costs by 30%',
            'specific_situation': research.pain_points[0] if research.pain_points else 'current growth phase'
        }

        # Generate personalized email using AI
        personalization_prompt = f"""
        Personalize this email template for a sales outreach:

        Template: {template}

        Lead Information:
        - Name: {lead.first_name} {lead.last_name}
        - Company: {lead.company}
        - Title: {lead.title}
        - Industry: {lead.industry}

        Research Insights:
        - Pain Points: {research.pain_points}
        - Buying Signals: {research.buying_signals}
        - Recent News: {research.recent_news[:2]}
        - Personalization Angles: {research.personalization_angles}

        Make the email compelling, personalized, and professional. Use insights to create relevance.
        Replace placeholder variables with actual personalized content.
        """

        # Get AI-generated personalized email
        personalized_email = await self._get_ai_analysis(personalization_prompt)

        # Format the email with variables
        try:
            formatted_email = template.format(**personalization_vars)
            return formatted_email
        except KeyError:
            # Fallback to basic personalization if template formatting fails
            return personalized_email

    async def create_email_sequence(self, sequence_type: EmailSequenceType, target_audience: str,
                                  emails_count: int = 3) -> EmailSequence:
        """Create a new email sequence"""

        sequence_id = hashlib.md5(f"{self.tenant_id}_{sequence_type}_{datetime.now().isoformat()}".encode()).hexdigest()

        # Generate email content for the sequence
        emails = []
        timing_days = []

        for i in range(emails_count):
            email_content = {
                'subject': f"Email {i+1} Subject",
                'body': self.email_templates[sequence_type][i] if i < len(self.email_templates[sequence_type]) else self.email_templates[sequence_type][-1],
                'email_number': i + 1
            }
            emails.append(email_content)

            # Set timing (day 0, 3, 7, etc.)
            if i == 0:
                timing_days.append(0)
            else:
                timing_days.append(timing_days[-1] + (3 if i == 1 else 4))

        sequence = EmailSequence(
            id=sequence_id,
            name=f"{sequence_type.value.title()} - {target_audience}",
            sequence_type=sequence_type,
            target_audience=target_audience,
            emails=emails,
            timing_days=timing_days
        )

        # Store sequence in database
        await self._store_email_sequence(sequence)

        return sequence

    async def _store_email_sequence(self, sequence: EmailSequence):
        """Store email sequence in database"""

        async with self.db_pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO tenant_management.email_sequences (
                    id, tenant_id, name, sequence_type, target_audience,
                    emails, timing_days, open_rate, reply_rate, conversion_rate,
                    active, created_at
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
            """,
                sequence.id, self.tenant_id, sequence.name, sequence.sequence_type.value,
                sequence.target_audience, json.dumps(sequence.emails),
                json.dumps(sequence.timing_days), sequence.open_rate,
                sequence.reply_rate, sequence.conversion_rate, sequence.active,
                datetime.now()
            )

    async def qualify_leads_batch(self, leads: List[Lead]) -> List[Dict[str, Any]]:
        """Qualify multiple leads using AI"""

        qualified_leads = []

        for lead in leads:
            try:
                # Research prospect
                research = await self.research_prospect(lead)

                # Calculate lead score
                lead_score = await self.calculate_lead_score(lead, research)

                # Update lead with score and qualification
                qualification = self._determine_qualification(lead_score, research)

                qualified_leads.append({
                    'lead_id': lead.id,
                    'lead_score': lead_score,
                    'qualification': qualification,
                    'recommended_actions': self._get_recommended_actions(lead_score, qualification),
                    'priority': self._get_priority_level(lead_score),
                    'research_summary': {
                        'pain_points': research.pain_points[:3],
                        'buying_signals': research.buying_signals[:3],
                        'confidence': research.confidence_level
                    }
                })

            except Exception as e:
                logger.error(f"Failed to qualify lead {lead.id}: {e}")
                continue

        return qualified_leads

    def _determine_qualification(self, lead_score: float, research: ProspectResearch) -> str:
        """Determine lead qualification based on score and research"""

        if lead_score >= 80 and len(research.buying_signals) >= 2:
            return "Hot Lead"
        elif lead_score >= 60 and research.confidence_level > 0.7:
            return "Qualified Lead"
        elif lead_score >= 40:
            return "Potential Lead"
        else:
            return "Low Priority"

    def _get_recommended_actions(self, lead_score: float, qualification: str) -> List[str]:
        """Get recommended actions based on qualification"""

        if qualification == "Hot Lead":
            return [
                "Schedule demo within 24 hours",
                "Assign to senior sales rep",
                "Send personalized proposal",
                "Follow up within 2 days if no response"
            ]
        elif qualification == "Qualified Lead":
            return [
                "Send personalized cold outreach sequence",
                "Share relevant case studies",
                "Schedule discovery call",
                "Add to nurture sequence"
            ]
        elif qualification == "Potential Lead":
            return [
                "Add to nurture sequence",
                "Send educational content",
                "Monitor for buying signals",
                "Re-qualify in 30 days"
            ]
        else:
            return [
                "Add to low-priority nurture",
                "Send quarterly check-ins",
                "Monitor for status changes"
            ]

    def _get_priority_level(self, lead_score: float) -> str:
        """Get priority level based on lead score"""

        if lead_score >= 80:
            return "High"
        elif lead_score >= 60:
            return "Medium"
        else:
            return "Low"

# Database schema for sales agent
SALES_SCHEMA_SQL = """
-- Prospect research storage
CREATE TABLE IF NOT EXISTS tenant_management.prospect_research (
    lead_id VARCHAR(255) NOT NULL,
    tenant_id UUID NOT NULL REFERENCES tenant_management.tenants(id) ON DELETE CASCADE,
    company_info JSONB DEFAULT '{}',
    contact_info JSONB DEFAULT '{}',
    social_presence JSONB DEFAULT '{}',
    recent_news JSONB DEFAULT '[]',
    technology_stack JSONB DEFAULT '[]',
    competitors JSONB DEFAULT '[]',
    pain_points JSONB DEFAULT '[]',
    buying_signals JSONB DEFAULT '[]',
    personalization_angles JSONB DEFAULT '[]',
    research_score FLOAT DEFAULT 0,
    confidence_level FLOAT DEFAULT 0,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    PRIMARY KEY (lead_id, tenant_id)
);

-- Leads storage
CREATE TABLE IF NOT EXISTS tenant_management.leads (
    id VARCHAR(255) PRIMARY KEY,
    tenant_id UUID NOT NULL REFERENCES tenant_management.tenants(id) ON DELETE CASCADE,
    email VARCHAR(255) NOT NULL,
    first_name VARCHAR(255),
    last_name VARCHAR(255),
    company VARCHAR(255),
    title VARCHAR(255),
    phone VARCHAR(50),
    linkedin_url VARCHAR(500),
    company_website VARCHAR(500),
    company_size VARCHAR(50),
    industry VARCHAR(100),
    lead_source VARCHAR(100),
    status VARCHAR(100) DEFAULT 'new',
    lead_score FLOAT DEFAULT 0,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    last_contacted TIMESTAMP WITH TIME ZONE,
    notes JSONB DEFAULT '[]',
    custom_fields JSONB DEFAULT '{}'
);

-- Email sequences
CREATE TABLE IF NOT EXISTS tenant_management.email_sequences (
    id VARCHAR(255) PRIMARY KEY,
    tenant_id UUID NOT NULL REFERENCES tenant_management.tenants(id) ON DELETE CASCADE,
    name VARCHAR(255) NOT NULL,
    sequence_type VARCHAR(100) NOT NULL,
    target_audience VARCHAR(255),
    emails JSONB NOT NULL,
    timing_days JSONB NOT NULL,
    open_rate FLOAT DEFAULT 0,
    reply_rate FLOAT DEFAULT 0,
    conversion_rate FLOAT DEFAULT 0,
    active BOOLEAN DEFAULT true,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Email campaigns tracking
CREATE TABLE IF NOT EXISTS tenant_management.email_campaigns (
    id VARCHAR(255) PRIMARY KEY,
    tenant_id UUID NOT NULL REFERENCES tenant_management.tenants(id) ON DELETE CASCADE,
    sequence_id VARCHAR(255) REFERENCES tenant_management.email_sequences(id),
    lead_id VARCHAR(255),
    email_number INTEGER,
    sent_at TIMESTAMP WITH TIME ZONE,
    opened_at TIMESTAMP WITH TIME ZONE,
    replied_at TIMESTAMP WITH TIME ZONE,
    clicked_at TIMESTAMP WITH TIME ZONE,
    status VARCHAR(100) DEFAULT 'scheduled'
);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_prospect_research_tenant
ON tenant_management.prospect_research(tenant_id);

CREATE INDEX IF NOT EXISTS idx_leads_tenant_status
ON tenant_management.leads(tenant_id, status);

CREATE INDEX IF NOT EXISTS idx_leads_score
ON tenant_management.leads(tenant_id, lead_score DESC);

CREATE INDEX IF NOT EXISTS idx_email_sequences_tenant_active
ON tenant_management.email_sequences(tenant_id, active);

CREATE INDEX IF NOT EXISTS idx_email_campaigns_lead_sequence
ON tenant_management.email_campaigns(lead_id, sequence_id);
"""

# Pydantic models for API
class LeadCreateRequest(BaseModel):
    email: EmailStr = Field(..., description="Lead email address")
    first_name: str = Field(..., min_length=1, max_length=255)
    last_name: str = Field(..., min_length=1, max_length=255)
    company: str = Field(..., min_length=1, max_length=255)
    title: str = Field(..., max_length=255)
    phone: Optional[str] = Field(None, max_length=50)
    linkedin_url: Optional[HttpUrl] = Field(None)
    company_website: Optional[HttpUrl] = Field(None)
    company_size: Optional[CompanySize] = Field(None)
    industry: Optional[str] = Field(None, max_length=100)
    lead_source: LeadSource = Field(default=LeadSource.WEBSITE)

class ProspectResearchRequest(BaseModel):
    lead_id: str = Field(..., description="Lead ID to research")

class EmailSequenceCreateRequest(BaseModel):
    sequence_type: EmailSequenceType = Field(..., description="Type of email sequence")
    target_audience: str = Field(..., min_length=1, max_length=255)
    emails_count: int = Field(default=3, ge=1, le=10)

class LeadQualificationRequest(BaseModel):
    lead_ids: List[str] = Field(..., min_items=1, max_items=100, description="List of lead IDs to qualify")

# Export main classes
__all__ = [
    'SalesAutomationAgent', 'Lead', 'ProspectResearch', 'EmailSequence',
    'LeadStatus', 'LeadSource', 'EmailSequenceType', 'CompanySize',
    'LeadCreateRequest', 'ProspectResearchRequest', 'EmailSequenceCreateRequest',
    'LeadQualificationRequest', 'SALES_SCHEMA_SQL'
]
