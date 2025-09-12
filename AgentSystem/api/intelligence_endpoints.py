"""
Competitive Intelligence API Endpoints - AgentSystem Profit Machine
Advanced competitive analysis and market intelligence system endpoints
"""

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks, Query
from fastapi.security import HTTPBearer
from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any, Union
from datetime import datetime, timedelta, date
from uuid import UUID, uuid4
import asyncio
import json
from enum import Enum

from ..auth.auth_service import verify_token, get_current_tenant
from ..database.connection import get_db_connection
from ..intelligence.competitive_monitor import (
    CompetitiveMonitor, CompetitorTier, IntelligenceType, MonitoringFrequency,
    ThreatLevel, CompetitorProfile, CompetitiveIntelligence, MarketTrend
)

# Initialize router
router = APIRouter(prefix="/api/v1/intelligence", tags=["Competitive Intelligence"])
security = HTTPBearer()

# Enums
class CompetitorTierAPI(str, Enum):
    DIRECT = "direct"
    INDIRECT = "indirect"
    SUBSTITUTE = "substitute"
    ADJACENT = "adjacent"

class IntelligenceTypeAPI(str, Enum):
    PRICING = "pricing"
    FEATURES = "features"
    MARKETING = "marketing"
    FUNDING = "funding"
    HIRING = "hiring"
    PARTNERSHIPS = "partnerships"
    PRODUCT_UPDATES = "product_updates"
    CUSTOMER_REVIEWS = "customer_reviews"

class ThreatLevelAPI(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

# Request Models
class CompetitorCreateRequest(BaseModel):
    name: str = Field(..., description="Competitor name")
    website: str = Field(..., description="Competitor website URL")
    tier: CompetitorTierAPI = Field(default=CompetitorTierAPI.DIRECT, description="Competitor tier")
    market_cap: Optional[float] = Field(None, description="Market capitalization")
    funding_raised: Optional[float] = Field(None, description="Total funding raised")
    employee_count: Optional[int] = Field(None, description="Number of employees")
    founded_year: Optional[int] = Field(None, description="Year founded")
    headquarters: Optional[str] = Field(None, description="Headquarters location")
    key_products: List[str] = Field(default_factory=list, description="Key products/services")
    target_markets: List[str] = Field(default_factory=list, description="Target markets")
    key_executives: List[Dict[str, str]] = Field(default_factory=list, description="Key executives")
    description: Optional[str] = Field(None, description="Company description")

class CompetitiveAnalysisRequest(BaseModel):
    analysis_period_days: int = Field(default=30, ge=1, le=365, description="Analysis period in days")
    competitors: Optional[List[UUID]] = Field(None, description="Specific competitors to analyze")
    intelligence_types: Optional[List[IntelligenceTypeAPI]] = Field(None, description="Types of intelligence to include")

# Response Models
class CompetitorResponse(BaseModel):
    competitor_id: UUID
    name: str
    website: str
    tier: CompetitorTierAPI
    market_cap: Optional[float]
    funding_raised: Optional[float]
    employee_count: Optional[int]
    founded_year: Optional[int]
    headquarters: Optional[str]
    key_products: List[str]
    target_markets: List[str]
    key_executives: List[Dict[str, str]]
    description: Optional[str]
    is_active: bool
    last_updated: datetime
    created_at: datetime

class CompetitiveIntelligenceResponse(BaseModel):
    intelligence_id: UUID
    competitor_id: UUID
    competitor_name: str
    intelligence_type: IntelligenceTypeAPI
    title: str
    summary: str
    details: Dict[str, Any]
    source_url: Optional[str]
    source_type: str
    confidence_score: float
    threat_level: ThreatLevelAPI
    impact_assessment: str
    recommended_actions: List[str]
    detected_date: datetime
    expiry_date: Optional[datetime]
    status: str

class CompetitiveDashboardResponse(BaseModel):
    total_competitors_monitored: int
    total_intelligence_items: int
    intelligence_last_7d: int
    critical_threats: int
    high_threats: int
    active_intelligence: int
    avg_confidence_score: float
    active_alerts: int
    latest_intelligence_date: Optional[datetime]

class CompetitiveAlertResponse(BaseModel):
    alert_id: UUID
    competitor_id: UUID
    competitor_name: str
    alert_type: str
    title: str
    message: str
    severity: ThreatLevelAPI
    status: str
    acknowledged_by: Optional[str]
    acknowledged_at: Optional[datetime]
    created_at: datetime

class ThreatSummaryResponse(BaseModel):
    threat_level: ThreatLevelAPI
    threat_count: int
    competitors_with_threats: int
    avg_confidence: float
    recent_threats: int

# Initialize competitive monitor
competitive_monitor = CompetitiveMonitor()

# Endpoints

@router.post("/competitors", response_model=CompetitorResponse)
async def add_competitor(
    request: CompetitorCreateRequest,
    background_tasks: BackgroundTasks,
    tenant_id: UUID = Depends(get_current_tenant),
    token: str = Depends(security)
):
    """Add a new competitor to monitor"""
    try:
        # Verify token
        await verify_token(token.credentials)

        # Create competitor data
        competitor_data = {
            'name': request.name,
            'website': request.website,
            'tier': request.tier.value,
            'market_cap': request.market_cap,
            'funding_raised': request.funding_raised,
            'employee_count': request.employee_count,
            'founded_year': request.founded_year,
            'headquarters': request.headquarters,
            'key_products': request.key_products,
            'target_markets': request.target_markets,
            'key_executives': request.key_executives,
            'description': request.description
        }

        # Add competitor
        competitor_id = await competitive_monitor.add_competitor(tenant_id, competitor_data)

        # Start monitoring in background
        background_tasks.add_task(
            competitive_monitor.monitor_competitor,
            tenant_id,
            competitor_id
        )

        # Get created competitor
        async with get_db_connection() as conn:
            query = """
                SELECT * FROM intelligence.competitors
                WHERE competitor_id = $1 AND tenant_id = $2
            """
            result = await conn.fetchrow(query, competitor_id, tenant_id)

            return CompetitorResponse(
                competitor_id=result['competitor_id'],
                name=result['name'],
                website=result['website'],
                tier=CompetitorTierAPI(result['tier']),
                market_cap=result['market_cap'],
                funding_raised=result['funding_raised'],
                employee_count=result['employee_count'],
                founded_year=result['founded_year'],
                headquarters=result['headquarters'],
                key_products=json.loads(result['key_products']),
                target_markets=json.loads(result['target_markets']),
                key_executives=json.loads(result['key_executives']),
                description=result['description'],
                is_active=result['is_active'],
                last_updated=result['last_updated'],
                created_at=result['created_at']
            )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to add competitor: {str(e)}")

@router.get("/competitors", response_model=List[CompetitorResponse])
async def get_competitors(
    tier: Optional[CompetitorTierAPI] = None,
    is_active: Optional[bool] = None,
    limit: int = Query(default=50, le=500),
    offset: int = Query(default=0, ge=0),
    tenant_id: UUID = Depends(get_current_tenant),
    token: str = Depends(security)
):
    """Get competitors with filtering"""
    try:
        # Verify token
        await verify_token(token.credentials)

        # Build query
        conditions = ["tenant_id = $1"]
        params = [tenant_id]
        param_count = 1

        if tier:
            param_count += 1
            conditions.append(f"tier = ${param_count}")
            params.append(tier.value)

        if is_active is not None:
            param_count += 1
            conditions.append(f"is_active = ${param_count}")
            params.append(is_active)

        async with get_db_connection() as conn:
            query = f"""
                SELECT * FROM intelligence.competitors
                WHERE {' AND '.join(conditions)}
                ORDER BY tier, name
                LIMIT ${param_count + 1} OFFSET ${param_count + 2}
            """
            params.extend([limit, offset])

            results = await conn.fetch(query, *params)

            return [
                CompetitorResponse(
                    competitor_id=row['competitor_id'],
                    name=row['name'],
                    website=row['website'],
                    tier=CompetitorTierAPI(row['tier']),
                    market_cap=row['market_cap'],
                    funding_raised=row['funding_raised'],
                    employee_count=row['employee_count'],
                    founded_year=row['founded_year'],
                    headquarters=row['headquarters'],
                    key_products=json.loads(row['key_products']),
                    target_markets=json.loads(row['target_markets']),
                    key_executives=json.loads(row['key_executives']),
                    description=row['description'],
                    is_active=row['is_active'],
                    last_updated=row['last_updated'],
                    created_at=row['created_at']
                )
                for row in results
            ]

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get competitors: {str(e)}")

@router.post("/competitors/{competitor_id}/monitor")
async def monitor_competitor(
    competitor_id: UUID,
    intelligence_types: Optional[List[IntelligenceTypeAPI]] = None,
    tenant_id: UUID = Depends(get_current_tenant),
    token: str = Depends(security)
):
    """Start monitoring a specific competitor"""
    try:
        # Verify token
        await verify_token(token.credentials)

        # Convert intelligence types
        monitor_types = None
        if intelligence_types:
            monitor_types = [IntelligenceType(t.value) for t in intelligence_types]

        # Start monitoring (simplified for now)
        intelligence_items = await competitive_monitor.monitor_competitor(
            tenant_id,
            competitor_id,
            monitor_types
        )

        return {"message": "Competitor monitoring completed", "items_found": len(intelligence_items)}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to start monitoring: {str(e)}")

@router.get("/intelligence", response_model=List[CompetitiveIntelligenceResponse])
async def get_competitive_intelligence(
    competitor_id: Optional[UUID] = None,
    intelligence_type: Optional[IntelligenceTypeAPI] = None,
    threat_level: Optional[ThreatLevelAPI] = None,
    days_back: int = Query(default=30, ge=1, le=365),
    limit: int = Query(default=100, le=1000),
    offset: int = Query(default=0, ge=0),
    tenant_id: UUID = Depends(get_current_tenant),
    token: str = Depends(security)
):
    """Get competitive intelligence with filtering"""
    try:
        # Verify token
        await verify_token(token.credentials)

        # Build query
        conditions = ["ci.tenant_id = $1", "ci.detected_date >= $2"]
        params = [tenant_id, datetime.utcnow() - timedelta(days=days_back)]
        param_count = 2

        if competitor_id:
            param_count += 1
            conditions.append(f"ci.competitor_id = ${param_count}")
            params.append(competitor_id)

        if intelligence_type:
            param_count += 1
            conditions.append(f"ci.intelligence_type = ${param_count}")
            params.append(intelligence_type.value)

        if threat_level:
            param_count += 1
            conditions.append(f"ci.threat_level = ${param_count}")
            params.append(threat_level.value)

        async with get_db_connection() as conn:
            query = f"""
                SELECT ci.*, c.name as competitor_name
                FROM intelligence.competitive_intelligence ci
                JOIN intelligence.competitors c ON ci.competitor_id = c.competitor_id
                WHERE {' AND '.join(conditions)}
                ORDER BY ci.detected_date DESC, ci.threat_level DESC
                LIMIT ${param_count + 1} OFFSET ${param_count + 2}
            """
            params.extend([limit, offset])

            results = await conn.fetch(query, *params)

            return [
                CompetitiveIntelligenceResponse(
                    intelligence_id=row['intelligence_id'],
                    competitor_id=row['competitor_id'],
                    competitor_name=row['competitor_name'],
                    intelligence_type=IntelligenceTypeAPI(row['intelligence_type']),
                    title=row['title'],
                    summary=row['summary'],
                    details=json.loads(row['details']) if row['details'] else {},
                    source_url=row['source_url'],
                    source_type=row['source_type'],
                    confidence_score=float(row['confidence_score']),
                    threat_level=ThreatLevelAPI(row['threat_level']),
                    impact_assessment=row['impact_assessment'],
                    recommended_actions=json.loads(row['recommended_actions']) if row['recommended_actions'] else [],
                    detected_date=row['detected_date'],
                    expiry_date=row['expiry_date'],
                    status=row['status']
                )
                for row in results
            ]

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get competitive intelligence: {str(e)}")

@router.post("/analysis/landscape")
async def analyze_competitive_landscape(
    request: CompetitiveAnalysisRequest,
    tenant_id: UUID = Depends(get_current_tenant),
    token: str = Depends(security)
):
    """Analyze the competitive landscape"""
    try:
        # Verify token
        await verify_token(token.credentials)

        # Run analysis
        analysis = await competitive_monitor.analyze_competitive_landscape(
            tenant_id=tenant_id,
            analysis_period_days=request.analysis_period_days
        )

        return analysis

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to analyze competitive landscape: {str(e)}")

@router.post("/reports/generate")
async def generate_competitive_report(
    report_type: str = Query(default="comprehensive", description="Report type"),
    competitors: Optional[List[UUID]] = Query(None, description="Specific competitors to include"),
    tenant_id: UUID = Depends(get_current_tenant),
    token: str = Depends(security)
):
    """Generate comprehensive competitive intelligence report"""
    try:
        # Verify token
        await verify_token(token.credentials)

        # Generate report
        report = await competitive_monitor.generate_competitive_report(
            tenant_id=tenant_id,
            report_type=report_type,
            competitors=competitors
        )

        # Store report
        async with get_db_connection() as conn:
            query = """
                INSERT INTO intelligence.competitive_reports (
                    report_id, tenant_id, report_name, report_type,
                    analysis_period_days, competitors_analyzed, intelligence_items_analyzed,
                    executive_summary, competitive_score, market_position,
                    key_findings, strategic_recommendations, threat_assessment,
                    opportunities, market_trends, report_data
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16)
                RETURNING report_id
            """
            result = await conn.fetchrow(
                query,
                report['report_id'],
                tenant_id,
                f"Competitive Analysis {datetime.utcnow().strftime('%Y-%m-%d')}",
                report_type,
                30,  # Default analysis period
                len(report.get('competitor_profiles', [])),
                0,   # intelligence_items_analyzed
                report['executive_summary'],
                50,  # competitive_score
                'unknown',  # market_position
                json.dumps([]),  # key_findings
                json.dumps(report.get('recommendations', [])),
                json.dumps({}),  # threat_assessment
                json.dumps([]),  # opportunities
                json.dumps([]),  # market_trends
                json.dumps(report)
            )

            return {"message": "Competitive report generated", "report_id": result['report_id']}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate competitive report: {str(e)}")

@router.get("/dashboard", response_model=CompetitiveDashboardResponse)
async def get_intelligence_dashboard(
    tenant_id: UUID = Depends(get_current_tenant),
    token: str = Depends(security)
):
    """Get competitive intelligence dashboard data"""
    try:
        # Verify token
        await verify_token(token.credentials)

        async with get_db_connection() as conn:
            query = """
                SELECT * FROM intelligence.intelligence_dashboard_stats
                WHERE tenant_id = $1
            """
            result = await conn.fetchrow(query, tenant_id)

            if not result:
                # Return empty dashboard
                return CompetitiveDashboardResponse(
                    total_competitors_monitored=0,
                    total_intelligence_items=0,
                    intelligence_last_7d=0,
                    critical_threats=0,
                    high_threats=0,
                    active_intelligence=0,
                    avg_confidence_score=0,
                    active_alerts=0,
                    latest_intelligence_date=None
                )

            return CompetitiveDashboardResponse(
                total_competitors_monitored=result['total_competitors_monitored'],
                total_intelligence_items=result['total_intelligence_items'],
                intelligence_last_7d=result['intelligence_last_7d'],
                critical_threats=result['critical_threats'],
                high_threats=result['high_threats'],
                active_intelligence=result['active_intelligence'],
                avg_confidence_score=float(result['avg_confidence_score'] or 0),
                active_alerts=result['active_alerts'],
                latest_intelligence_date=result['latest_intelligence_date']
            )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get intelligence dashboard: {str(e)}")

@router.get("/threats/summary", response_model=List[ThreatSummaryResponse])
async def get_threat_summary(
    tenant_id: UUID = Depends(get_current_tenant),
    token: str = Depends(security)
):
    """Get threat level summary"""
    try:
        # Verify token
        await verify_token(token.credentials)

        async with get_db_connection() as conn:
            query = """
                SELECT * FROM intelligence.threat_summary
                WHERE tenant_id = $1
                ORDER BY
                    CASE threat_level
                        WHEN 'critical' THEN 1
                        WHEN 'high' THEN 2
                        WHEN 'medium' THEN 3
                        ELSE 4
                    END
            """
            results = await conn.fetch(query, tenant_id)

            return [
                ThreatSummaryResponse(
                    threat_level=ThreatLevelAPI(row['threat_level']),
                    threat_count=row['threat_count'],
                    competitors_with_threats=row['competitors_with_threats'],
                    avg_confidence=float(row['avg_confidence']),
                    recent_threats=row['recent_threats']
                )
                for row in results
            ]

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get threat summary: {str(e)}")

@router.get("/alerts", response_model=List[CompetitiveAlertResponse])
async def get_competitive_alerts(
    competitor_id: Optional[UUID] = None,
    alert_type: Optional[str] = None,
    severity: Optional[ThreatLevelAPI] = None,
    status: Optional[str] = None,
    limit: int = Query(default=50, le=500),
    offset: int = Query(default=0, ge=0),
    tenant_id: UUID = Depends(get_current_tenant),
    token: str = Depends(security)
):
    """Get competitive alerts with filtering"""
    try:
        # Verify token
        await verify_token(token.credentials)

        # Build query
        conditions = ["ca.tenant_id = $1"]
        params = [tenant_id]
        param_count = 1

        if competitor_id:
            param_count += 1
            conditions.append(f"ca.competitor_id = ${param_count}")
            params.append(competitor_id)

        if alert_type:
            param_count += 1
            conditions.append(f"ca.alert_type = ${param_count}")
            params.append(alert_type)

        if severity:
            param_count += 1
            conditions.append(f"ca.severity = ${param_count}")
            params.append(severity.value)

        if status:
            param_count += 1
            conditions.append(f"ca.status = ${param_count}")
            params.append(status)

        async with get_db_connection() as conn:
            query = f"""
                SELECT ca.*, c.name as competitor_name
                FROM intelligence.competitor_alerts ca
                JOIN intelligence.competitors c ON ca.competitor_id = c.competitor_id
                WHERE {' AND '.join(conditions)}
                ORDER BY ca.created_at DESC
                LIMIT ${param_count + 1} OFFSET ${param_count + 2}
            """
            params.extend([limit, offset])

            results = await conn.fetch(query, *params)

            return [
                CompetitiveAlertResponse(
                    alert_id=row['alert_id'],
                    competitor_id=row['competitor_id'],
                    competitor_name=row['competitor_name'],
                    alert_type=row['alert_type'],
                    title=row['title'],
                    message=row['message'],
                    severity=ThreatLevelAPI(row['severity']),
                    status=row['status'],
                    acknowledged_by=row['acknowledged_by'],
                    acknowledged_at=row['acknowledged_at'],
                    created_at=row['created_at']
                )
                for row in results
            ]

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get competitive alerts: {str(e)}")

@router.put("/alerts/{alert_id}/acknowledge")
async def acknowledge_competitive_alert(
    alert_id: UUID,
    acknowledged_by: str = Query(..., description="Name of person acknowledging alert"),
    tenant_id: UUID = Depends(get_current_tenant),
    token: str = Depends(security)
):
    """Acknowledge a competitive alert"""
    try:
        # Verify token
        await verify_token(token.credentials)

        async with get_db_connection() as conn:
            query = """
                UPDATE intelligence.competitor_alerts
                SET status = 'acknowledged',
                    acknowledged_by = $3,
                    acknowledged_at = NOW()
                WHERE alert_id = $1 AND tenant_id = $2
            """
            result = await conn.execute(query, alert_id, tenant_id, acknowledged_by)

            if result == "UPDATE 0":
                raise HTTPException(status_code=404, detail="Alert not found")

            return {"message": "Alert acknowledged successfully"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to acknowledge alert: {str(e)}")

@router.post("/refresh-dashboard")
async def refresh_intelligence_dashboard(
    tenant_id: UUID = Depends(get_current_tenant),
    token: str = Depends(security)
):
    """Refresh intelligence dashboard materialized view"""
    try:
        # Verify token
        await verify_token(token.credentials)

        async with get_db_connection() as conn:
            await conn.execute("SELECT intelligence.refresh_intelligence_dashboard_stats()")

        return {"message": "Intelligence dashboard refreshed successfully"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to refresh dashboard: {str(e)}")

@router.post("/escalate-alerts")
async def escalate_competitive_alerts(
    tenant_id: UUID = Depends(get_current_tenant),
    token: str = Depends(security)
):
    """Manually trigger alert escalation"""
    try:
        # Verify token
        await verify_token(token.credentials)

        async with get_db_connection() as conn:
            await conn.execute("SELECT intelligence.auto_escalate_competitive_alerts()")

        return {"message": "Alert escalation completed"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to escalate alerts: {str(e)}")

# Health check endpoint
@router.get("/health")
async def intelligence_health_check():
    """Health check for competitive intelligence system"""
    try:
        async with get_db_connection() as conn:
            await conn.fetchval("SELECT 1")

        return {
            "status": "healthy",
            "service": "competitive_intelligence",
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Service unhealthy: {str(e)}")
