"""
Partner Channel and Reseller Program API Endpoints - AgentSystem Profit Machine
RESTful API for comprehensive partner ecosystem management
"""

from fastapi import APIRouter, HTTPException, Depends, Query, BackgroundTasks
from fastapi.security import HTTPBearer
from typing import List, Optional, Dict, Any
from uuid import UUID
from datetime import datetime, timedelta
import json
import logging

from ..auth.auth_middleware import verify_token, get_current_tenant
from ..partners.channel_manager import ChannelManager, create_channel_manager, PartnerType, PartnerStatus
from ..database.connection import get_db_connection
from ..models.api_models import StandardResponse, PaginatedResponse
from ..usage.usage_tracker import UsageTracker

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/api/v1/partners", tags=["Partner Channel Management"])
security = HTTPBearer()

# Initialize channel manager
channel_manager = create_channel_manager()

@router.on_event("startup")
async def startup_event():
    """Initialize channel manager on startup"""
    await channel_manager.initialize()

# Partner Management Endpoints

@router.post("/onboard", response_model=StandardResponse)
async def onboard_partner(
    partner_application: Dict[str, Any],
    background_tasks: BackgroundTasks,
    token: str = Depends(security)
):
    """Onboard a new partner"""
    try:
        # Verify token
        await verify_token(token.credentials)

        # Validate application
        required_fields = ['company_name', 'contact_name', 'contact_email', 'partner_type']
        for field in required_fields:
            if field not in partner_application:
                raise HTTPException(
                    status_code=400,
                    detail=f"Missing required field: {field}"
                )

        # Onboard partner
        partner_id = await channel_manager.onboard_partner(partner_application)

        return StandardResponse(
            success=True,
            message="Partner onboarding initiated successfully",
            data={
                "partner_id": str(partner_id),
                "company_name": partner_application['company_name'],
                "status": "pending"
            }
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to onboard partner: {e}")
        raise HTTPException(status_code=500, detail="Failed to onboard partner")

@router.post("/approve/{partner_id}", response_model=StandardResponse)
async def approve_partner(
    partner_id: UUID,
    approval_details: Dict[str, Any],
    background_tasks: BackgroundTasks,
    token: str = Depends(security)
):
    """Approve a pending partner application"""
    try:
        # Verify token
        await verify_token(token.credentials)

        # Approve partner
        result = await channel_manager.approve_partner(partner_id, approval_details)

        return StandardResponse(
            success=True,
            message="Partner approved successfully",
            data=result
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to approve partner: {e}")
        raise HTTPException(status_code=500, detail="Failed to approve partner")

@router.get("/", response_model=PaginatedResponse)
async def get_partners(
    page: int = Query(1, ge=1),
    limit: int = Query(20, ge=1, le=100),
    status: Optional[str] = None,
    partner_type: Optional[str] = None,
    token: str = Depends(security)
):
    """Get partners with filtering and pagination"""
    try:
        # Verify token
        await verify_token(token.credentials)

        offset = (page - 1) * limit

        async with get_db_connection() as conn:
            # Build query
            where_conditions = []
            params = []
            param_count = 0

            if status:
                param_count += 1
                where_conditions.append(f"status = ${param_count}")
                params.append(status)

            if partner_type:
                param_count += 1
                where_conditions.append(f"partner_type = ${param_count}")
                params.append(partner_type)

            where_clause = ""
            if where_conditions:
                where_clause = "WHERE " + " AND ".join(where_conditions)

            # Get total count
            count_query = f"""
                SELECT COUNT(*) FROM partners.partners
                {where_clause}
            """
            total_count = await conn.fetchval(count_query, *params)

            # Get partners
            query = f"""
                SELECT
                    partner_id, company_name, contact_name, contact_email,
                    partner_type, status, certification_level, territory,
                    specializations, white_label_enabled, created_date
                FROM partners.partners
                {where_clause}
                ORDER BY created_date DESC
                LIMIT ${param_count + 1} OFFSET ${param_count + 2}
            """
            params.extend([limit, offset])

            results = await conn.fetch(query, *params)

            partners = []
            for result in results:
                partner = {
                    'partner_id': str(result['partner_id']),
                    'company_name': result['company_name'],
                    'contact_name': result['contact_name'],
                    'contact_email': result['contact_email'],
                    'partner_type': result['partner_type'],
                    'status': result['status'],
                    'certification_level': result['certification_level'],
                    'territory': json.loads(result['territory']),
                    'specializations': json.loads(result['specializations']),
                    'white_label_enabled': result['white_label_enabled'],
                    'created_date': result['created_date'].isoformat()
                }
                partners.append(partner)

            return PaginatedResponse(
                success=True,
                message="Partners retrieved successfully",
                data=partners,
                pagination={
                    'page': page,
                    'limit': limit,
                    'total': total_count,
                    'pages': (total_count + limit - 1) // limit
                }
            )

    except Exception as e:
        logger.error(f"Failed to get partners: {e}")
        raise HTTPException(status_code=500, detail="Failed to get partners")

@router.get("/{partner_id}/performance", response_model=StandardResponse)
async def get_partner_performance(
    partner_id: UUID,
    period_months: int = Query(12, ge=1, le=36),
    token: str = Depends(security)
):
    """Get comprehensive partner performance metrics"""
    try:
        # Verify token
        await verify_token(token.credentials)

        # Get performance metrics
        performance = await channel_manager.get_partner_performance(partner_id, period_months)

        return StandardResponse(
            success=True,
            message="Partner performance retrieved successfully",
            data=performance
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to get partner performance: {e}")
        raise HTTPException(status_code=500, detail="Failed to get partner performance")

# Deal Management Endpoints

@router.post("/deals/register", response_model=StandardResponse)
async def register_deal(
    deal_details: Dict[str, Any],
    background_tasks: BackgroundTasks,
    token: str = Depends(security)
):
    """Register a new partner deal"""
    try:
        # Verify token
        await verify_token(token.credentials)

        # Validate deal details
        required_fields = ['partner_id', 'customer_id', 'deal_name', 'deal_value', 'expected_close_date']
        for field in required_fields:
            if field not in deal_details:
                raise HTTPException(
                    status_code=400,
                    detail=f"Missing required field: {field}"
                )

        partner_id = UUID(deal_details['partner_id'])

        # Register deal
        deal_id = await channel_manager.register_deal(partner_id, deal_details)

        return StandardResponse(
            success=True,
            message="Deal registered successfully",
            data={
                "deal_id": str(deal_id),
                "partner_id": str(partner_id),
                "deal_name": deal_details['deal_name'],
                "deal_value": deal_details['deal_value']
            }
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to register deal: {e}")
        raise HTTPException(status_code=500, detail="Failed to register deal")

@router.post("/deals/{deal_id}/close", response_model=StandardResponse)
async def close_deal(
    deal_id: UUID,
    closure_details: Dict[str, Any],
    background_tasks: BackgroundTasks,
    token: str = Depends(security)
):
    """Close a partner deal and calculate commission"""
    try:
        # Verify token
        await verify_token(token.credentials)

        # Close deal
        result = await channel_manager.close_deal(deal_id, closure_details)

        return StandardResponse(
            success=True,
            message="Deal closed successfully",
            data=result
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to close deal: {e}")
        raise HTTPException(status_code=500, detail="Failed to close deal")

@router.get("/analytics/overview", response_model=StandardResponse)
async def get_partner_analytics_overview(
    days: int = Query(90, ge=1, le=365),
    partner_type: Optional[str] = None,
    token: str = Depends(security)
):
    """Get partner channel analytics overview"""
    try:
        # Verify token
        await verify_token(token.credentials)

        since_date = datetime.utcnow() - timedelta(days=days)

        async with get_db_connection() as conn:
            # Build partner filter
            partner_filter = ""
            params = [since_date]
            param_count = 1

            if partner_type:
                param_count += 1
                partner_filter = f"AND p.partner_type = ${param_count}"
                params.append(partner_type)

            # Get partner stats
            partner_stats_query = f"""
                SELECT
                    COUNT(*) as total_partners,
                    COUNT(*) FILTER (WHERE status = 'active') as active_partners,
                    COUNT(*) FILTER (WHERE status = 'pending') as pending_partners
                FROM partners.partners p
                WHERE created_date >= $1 {partner_filter}
            """
            partner_stats = await conn.fetchrow(partner_stats_query, *params)

            # Get deal stats
            deal_stats_query = f"""
                SELECT
                    COUNT(*) as total_deals,
                    COUNT(*) FILTER (WHERE deal_stage = 'closed_won') as closed_deals,
                    COALESCE(SUM(deal_value) FILTER (WHERE deal_stage = 'closed_won'), 0) as total_revenue,
                    COALESCE(SUM(commission_amount) FILTER (WHERE deal_stage = 'closed_won'), 0) as total_commission
                FROM partners.partner_deals pd
                JOIN partners.partners p ON pd.partner_id = p.partner_id
                WHERE pd.created_date >= $1 {partner_filter}
            """
            deal_stats = await conn.fetchrow(deal_stats_query, *params)

            analytics = {
                'period_days': days,
                'partner_type_filter': partner_type,
                'partner_metrics': {
                    'total_partners': partner_stats['total_partners'],
                    'active_partners': partner_stats['active_partners'],
                    'pending_partners': partner_stats['pending_partners']
                },
                'deal_metrics': {
                    'total_deals': deal_stats['total_deals'],
                    'closed_deals': deal_stats['closed_deals'],
                    'win_rate': round(((deal_stats['closed_deals'] or 0) / max(deal_stats['total_deals'] or 1, 1)) * 100, 2),
                    'total_revenue': float(deal_stats['total_revenue'] or 0),
                    'total_commission': float(deal_stats['total_commission'] or 0)
                }
            }

            return StandardResponse(
                success=True,
                message="Partner analytics overview retrieved successfully",
                data=analytics
            )

    except Exception as e:
        logger.error(f"Failed to get partner analytics: {e}")
        raise HTTPException(status_code=500, detail="Failed to get partner analytics")

# Export router
__all__ = ["router"]
