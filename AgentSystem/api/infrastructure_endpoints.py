
"""
Multi-Region Deployment Infrastructure API Endpoints - AgentSystem Profit Machine
RESTful API for global deployment management with auto-scaling, failover, and compliance
"""

from fastapi import APIRouter, HTTPException, Depends, Query, BackgroundTasks
from fastapi.security import HTTPBearer
from typing import List, Optional, Dict, Any
from uuid import UUID
from datetime import datetime, timedelta
import json
import logging

from ..auth.auth_middleware import verify_token, get_current_tenant
from ..infrastructure.deployment_manager import DeploymentManager, create_deployment_manager, DeploymentType
from ..database.connection import get_db_connection
from ..models.api_models import StandardResponse, PaginatedResponse
from ..usage.usage_tracker import UsageTracker

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/api/v1/infrastructure", tags=["Infrastructure Management"])
security = HTTPBearer()

# Initialize deployment manager
deployment_manager = create_deployment_manager()

@router.on_event("startup")
async def startup_event():
    """Initialize deployment manager on startup"""
    await deployment_manager.initialize()

# Region Management Endpoints

@router.get("/regions", response_model=StandardResponse)
async def get_regions(
    status: Optional[str] = None,
    cloud_provider: Optional[str] = None,
    token: str = Depends(security)
):
    """Get available regions"""
    try:
        # Verify token
        await verify_token(token.credentials)

        async with get_db_connection() as conn:
            # Build query
            where_conditions = []
            params = []

            if status:
                where_conditions.append("status = $1")
                params.append(status)

            if cloud_provider:
                param_num = len(params) + 1
                where_conditions.append(f"cloud_provider = ${param_num}")
                params.append(cloud_provider)

            where_clause = ""
            if where_conditions:
                where_clause = "WHERE " + " AND ".join(where_conditions)

            query = f"""
                SELECT
                    region_id, region_name, region_code, continent, country, city,
                    latitude, longitude, cloud_provider, status, is_primary,
                    data_residency_compliant, compliance_certifications, timezone,
                    created_date, last_health_check
                FROM infrastructure.regions
                {where_clause}
                ORDER BY is_primary DESC, region_name
            """

            results = await conn.fetch(query, *params)

            regions = []
            for result in results:
                region = {
                    'region_id': result['region_id'],
                    'region_name': result['region_name'],
                    'region_code': result['region_code'],
                    'continent': result['continent'],
                    'country': result['country'],
                    'city': result['city'],
                    'latitude': float(result['latitude']),
                    'longitude': float(result['longitude']),
                    'cloud_provider': result['cloud_provider'],
                    'status': result['status'],
                    'is_primary': result['is_primary'],
                    'data_residency_compliant': result['data_residency_compliant'],
                    'compliance_certifications': json.loads(result['compliance_certifications']),
                    'timezone': result['timezone'],
                    'created_date': result['created_date'].isoformat(),
                    'last_health_check': result['last_health_check'].isoformat()
                }
                regions.append(region)

            return StandardResponse(
                success=True,
                message="Regions retrieved successfully",
                data=regions
            )

    except Exception as e:
        logger.error(f"Failed to get regions: {e}")
        raise HTTPException(status_code=500, detail="Failed to get regions")

@router.get("/regions/{region_id}/health", response_model=StandardResponse)
async def get_region_health(
    region_id: str,
    hours: int = Query(24, ge=1, le=168),
    token: str = Depends(security)
):
    """Get health status for specific region"""
    try:
        # Verify token
        await verify_token(token.credentials)

        async with get_db_connection() as conn:
            # Get region info
            region_query = """
                SELECT region_name, status FROM infrastructure.regions
                WHERE region_id = $1
            """
            region_result = await conn.fetchrow(region_query, region_id)

            if not region_result:
                raise HTTPException(status_code=404, detail="Region not found")

            # Get recent health metrics
            since_time = datetime.utcnow() - timedelta(hours=hours)

            metrics_query = """
                SELECT
                    target_id, timestamp, api_response_time_ms,
                    cpu_utilization, memory_utilization, status,
                    requests_per_second, error_rate, uptime_percentage
                FROM infrastructure.health_metrics
                WHERE region_id = $1 AND timestamp >= $2
                ORDER BY timestamp DESC
            """

            metrics_results = await conn.fetch(metrics_query, region_id, since_time)

            # Get deployment targets in region
            targets_query = """
                SELECT target_id, deployment_type, environment, is_active, health_status
                FROM infrastructure.deployment_targets
                WHERE region_id = $1
            """
            targets_results = await conn.fetch(targets_query, region_id)

            # Calculate health summary
            health_summary = {
                'region_id': region_id,
                'region_name': region_result['region_name'],
                'region_status': region_result['status'],
                'total_targets': len(targets_results),
                'active_targets': len([t for t in targets_results if t['is_active']]),
                'healthy_targets': len([t for t in targets_results if t['health_status'] == 'healthy']),
                'warning_targets': len([t for t in targets_results if t['health_status'] == 'warning']),
                'critical_targets': len([t for t in targets_results if t['health_status'] == 'critical']),
                'avg_response_time_ms': 0,
                'avg_cpu_utilization': 0,
                'avg_memory_utilization': 0,
                'avg_error_rate': 0,
                'uptime_percentage': 100,
                'metrics_count': len(metrics_results),
                'time_range_hours': hours
            }

            if metrics_results:
                health_summary['avg_response_time_ms'] = sum(m['api_response_time_ms'] for m in metrics_results) / len(metrics_results)
                health_summary['avg_cpu_utilization'] = sum(m['cpu_utilization'] for m in metrics_results) / len(metrics_results)
                health_summary['avg_memory_utilization'] = sum(m['memory_utilization'] for m in metrics_results) / len(metrics_results)
                health_summary['avg_error_rate'] = sum(m['error_rate'] for m in metrics_results) / len(metrics_results)
                health_summary['uptime_percentage'] = sum(m['uptime_percentage'] for m in metrics_results) / len(metrics_results)

            return StandardResponse(
                success=True,
                message="Region health retrieved successfully",
                data=health_summary
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get region health: {e}")
        raise HTTPException(status_code=500, detail="Failed to get region health")

# Deployment Management Endpoints

@router.post("/deploy", response_model=StandardResponse)
async def deploy_to_region(
    deployment_request: Dict[str, Any],
    background_tasks: BackgroundTasks,
    token: str = Depends(security)
):
    """Deploy AgentSystem to specific region"""
    try:
        # Verify token
        await verify_token(token.credentials)

        # Validate request
        required_fields = ['region_id', 'deployment_type']
        for field in required_fields:
            if field not in deployment_request:
                raise HTTPException(
                    status_code=400,
                    detail=f"Missing required field: {field}"
                )

        region_id = deployment_request['region_id']
        deployment_type = DeploymentType(deployment_request['deployment_type'])
        deployment_config = deployment_request.get('config', {})

        # Execute deployment in background
        target_id = await deployment_manager.deploy_to_region(
            region_id, deployment_type, deployment_config
        )

        return StandardResponse(
            success=True,
            message="Deployment started successfully",
            data={
                "target_id": str(target_id),
                "region_id": region_id,
                "deployment_type": deployment_type.value
            }
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to deploy to region: {e}")
        raise HTTPException(status_code=500, detail="Failed to deploy to region")

@router.get("/deployments", response_model=PaginatedResponse)
async def get_deployments(
    page: int = Query(1, ge=1),
    limit: int = Query(20, ge=1, le=100),
    region_id: Optional[str] = None,
    deployment_type: Optional[str] = None,
    status: Optional[str] = None,
    token: str = Depends(security)
):
    """Get deployment targets"""
    try:
        # Verify token
        await verify_token(token.credentials)

        offset = (page - 1) * limit

        async with get_db_connection() as conn:
            # Build query
            where_conditions = []
            params = []
            param_count = 0

            if region_id:
                param_count += 1
                where_conditions.append(f"dt.region_id = ${param_count}")
                params.append(region_id)

            if deployment_type:
                param_count += 1
                where_conditions.append(f"dt.deployment_type = ${param_count}")
                params.append(deployment_type)

            if status:
                param_count += 1
                where_conditions.append(f"dt.health_status = ${param_count}")
                params.append(status)

            where_clause = ""
            if where_conditions:
                where_clause = "WHERE " + " AND ".join(where_conditions)

            # Get total count
            count_query = f"""
                SELECT COUNT(*) FROM infrastructure.deployment_targets dt
                {where_clause}
            """
            total_count = await conn.fetchval(count_query, *params)

            # Get deployments
            query = f"""
                SELECT
                    dt.target_id, dt.region_id, dt.deployment_type, dt.environment,
                    dt.api_endpoint, dt.health_status, dt.is_active, dt.created_date,
                    dt.current_instances, dt.target_instances, dt.last_deployment,
                    r.region_name, r.country, r.cloud_provider
                FROM infrastructure.deployment_targets dt
                JOIN infrastructure.regions r ON dt.region_id = r.region_id
                {where_clause}
                ORDER BY dt.created_date DESC
                LIMIT ${param_count + 1} OFFSET ${param_count + 2}
            """
            params.extend([limit, offset])

            results = await conn.fetch(query, *params)

            deployments = []
            for result in results:
                deployment = {
                    'target_id': str(result['target_id']),
                    'region_id': result['region_id'],
                    'region_name': result['region_name'],
                    'country': result['country'],
                    'cloud_provider': result['cloud_provider'],
                    'deployment_type': result['deployment_type'],
                    'environment': result['environment'],
                    'api_endpoint': result['api_endpoint'],
                    'health_status': result['health_status'],
                    'is_active': result['is_active'],
                    'current_instances': result['current_instances'],
                    'target_instances': result['target_instances'],
                    'created_date': result['created_date'].isoformat(),
                    'last_deployment': result['last_deployment'].isoformat() if result['last_deployment'] else None
                }
                deployments.append(deployment)

            return PaginatedResponse(
                success=True,
                message="Deployments retrieved successfully",
                data=deployments,
                pagination={
                    'page': page,
                    'limit': limit,
                    'total': total_count,
                    'pages': (total_count + limit - 1) // limit
                }
            )

    except Exception as e:
        logger.error(f"Failed to get deployments: {e}")
        raise HTTPException(status_code=500, detail="Failed to get deployments")

@router.post("/deployments/{target_id}/scale", response_model=StandardResponse)
async def scale_deployment(
    target_id: UUID,
    scaling_request: Dict[str, Any],
    background_tasks: BackgroundTasks,
    token: str = Depends(security)
):
    """Scale deployment in specific region"""
    try:
        # Verify token
        await verify_token(token.credentials)

        # Execute scaling
        result = await deployment_manager.scale_deployment(target_id, scaling_request)

        return StandardResponse(
            success=True,
            message="Scaling completed successfully",
            data=result
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to scale deployment: {e}")
        raise HTTPException(status_code=500, detail="Failed to scale deployment")

# Failover Management Endpoints

@router.post("/failover", response_model=StandardResponse)
async def execute_failover(
    failover_request: Dict[str, Any],
    background_tasks: BackgroundTasks,
    token: str = Depends(security)
):
    """Execute failover from failed region to backup region"""
    try:
        # Verify token
        await verify_token(token.credentials)

        # Validate request
        if 'failed_region_id' not in failover_request:
            raise HTTPException(
                status_code=400,
                detail="Missing required field: failed_region_id"
            )

        failed_region_id = failover_request['failed_region_id']
        target_region_id = failover_request.get('target_region_id')

        # Execute failover
        result = await deployment_manager.failover_to_region(failed_region_id, target_region_id)

        return StandardResponse(
            success=True,
            message="Failover executed successfully",
            data=result
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to execute failover: {e}")
        raise HTTPException(status_code=500, detail="Failed to execute failover")

@router.get("/failover/history", response_model=PaginatedResponse)
async def get_failover_history(
    page: int = Query(1, ge=1),
    limit: int = Query(20, ge=1, le=100),
    region_id: Optional[str] = None,
    days: int = Query(30, ge=1, le=365),
    token: str = Depends(security)
):
    """Get failover event history"""
    try:
        # Verify token
        await verify_token(token.credentials)

        offset = (page - 1) * limit
        since_date = datetime.utcnow() - timedelta(days=days)

        async with get_db_connection() as conn:
            # Build query
            where_conditions = ["start_time >= $1"]
            params = [since_date]
            param_count = 1

            if region_id:
                param_count += 1
                where_conditions.append(f"(failed_region_id = ${param_count} OR target_region_id = ${param_count})")
                params.append(region_id)

            where_clause = "WHERE " + " AND ".join(where_conditions)

            # Get total count
            count_query = f"""
                SELECT COUNT(*) FROM infrastructure.failover_events
                {where_clause}
            """
            total_count = await conn.fetchval(count_query, *params)

            # Get failover events
            query = f"""
                SELECT
                    fe.event_id, fe.failed_region_id, fe.target_region_id,
                    fe.trigger_reason, fe.event_type, fe.start_time, fe.end_time,
                    fe.total_duration_ms, fe.affected_targets, fe.successful_failovers,
                    fe.failed_failovers, fe.status, fe.error_message,
                    fr.region_name as failed_region_name,
                    tr.region_name as target_region_name
                FROM infrastructure.failover_events fe
                JOIN infrastructure.regions fr ON fe.failed_region_id = fr.region_id
                JOIN infrastructure.regions tr ON fe.target_region_id = tr.region_id
                {where_clause}
                ORDER BY fe.start_time DESC
                LIMIT ${param_count + 1} OFFSET ${param_count + 2}
            """
            params.extend([limit, offset])

            results = await conn.fetch(query, *params)

            failover_events = []
            for result in results:
                event = {
                    'event_id': str(result['event_id']),
                    'failed_region_id': result['failed_region_id'],
                    'failed_region_name': result['failed_region_name'],
                    'target_region_id': result['target_region_id'],
                    'target_region_name': result['target_region_name'],
                    'trigger_reason': result['trigger_reason'],
                    'event_type': result['event_type'],
                    'start_time': result['start_time'].isoformat(),
                    'end_time': result['end_time'].isoformat() if result['end_time'] else None,
                    'total_duration_ms': result['total_duration_ms'],
                    'affected_targets': result['affected_targets'],
                    'successful_failovers': result['successful_failovers'],
                    'failed_failovers': result['failed_failovers'],
                    'status': result['status'],
                    'error_message': result['error_message']
                }
                failover_events.append(event)

            return PaginatedResponse(
                success=True,
                message="Failover history retrieved successfully",
                data=failover_events,
                pagination={
                    'page': page,
                    'limit': limit,
                    'total': total_count,
                    'pages': (total_count + limit - 1) // limit
                }
            )

    except Exception as e:
        logger.error(f"Failed to get failover history: {e}")
        raise HTTPException(status_code=500, detail="Failed to get failover history")

# Global Health and Monitoring Endpoints

@router.get("/health/global", response_model=StandardResponse)
async def get_global_health(
    token: str = Depends(security)
):
    """Get global health status across all regions"""
    try:
        # Verify token
        await verify_token(token.credentials)

        # Get global health status
        health_status = await deployment_manager.get_global_health_status()

        return StandardResponse(
            success=True,
            message="Global health status retrieved successfully",
            data=health_status
        )

    except Exception as e:
        logger.error(f"Failed to get global health: {e}")
        raise HTTPException(status_code=500, detail="Failed to get global health")

@router.post("/traffic/optimize", response_model=StandardResponse)
async def optimize_traffic_routing(
    background_tasks: BackgroundTasks,
    token: str = Depends(security)
):
    """Optimize traffic routing across regions"""
    try:
        # Verify token
        await verify_token(token.credentials)

        # Execute traffic optimization
        result = await deployment_manager.optimize_traffic_routing()

        return StandardResponse(
            success=True,
            message="Traffic routing optimized successfully",
            data=result
        )

    except Exception as e:
        logger.error(f"Failed to optimize traffic routing: {e}")
        raise HTTPException(status_code=500, detail="Failed to optimize traffic routing")

# Backup Management Endpoints

@router.post("/backup/{region_id}", response_model=StandardResponse)
async def backup_region_data(
    region_id: str,
    backup_request: Dict[str, Any],
    background_tasks: BackgroundTasks,
    token: str = Depends(security)
):
    """Backup data for specific region"""
    try:
        # Verify token
        await verify_token(token.credentials)

        backup_type = backup_request.get('backup_type', 'incremental')

        # Execute backup
        result = await deployment_manager.backup_region_data(region_id, backup_type)

        return StandardResponse(
            success=True,
            message="Region backup completed successfully",
            data=result
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to backup region data: {e}")
        raise HTTPException(status_code=500, detail="Failed to backup region data")

@router.get("/backup/status", response_model=PaginatedResponse)
async def get_backup_status(
    page: int = Query(1, ge=1),
    limit: int = Query(20, ge=1, le=100),
    region_id: Optional[str] = None,
    backup_type: Optional[str] = None,
    days: int = Query(7, ge=1, le=90),
    token: str = Depends(security)
):
    """Get backup status"""
    try:
        # Verify token
        await verify_token(token.credentials)

        offset = (page - 1) * limit
        since_date = datetime.utcnow() - timedelta(days=days)

        async with get_db_connection() as conn:
            # Build query
            where_conditions = ["created_date >= $1"]
            params = [since_date]
            param_count = 1

            if region_id:
                param_count += 1
                where_conditions.append(f"region_id = ${param_count}")
                params.append(region_id)

            if backup_type:
                param_count += 1
                where_conditions.append(f"backup_type = ${param_count}")
                params.append(backup_type)

            where_clause = "WHERE " + " AND ".join(where_conditions)

            # Get total count
            count_query = f"""
                SELECT COUNT(*) FROM infrastructure.backup_status
                {where_clause}
            """
            total_count = await conn.fetchval(count_query, *params)

            # Get backup status
            query = f"""
                SELECT
                    bs.backup_id, bs.region_id, bs.backup_type, bs.backup_size_gb,
                    bs.backup_location, bs.encryption_enabled, bs.retention_days,
                    bs.created_date, bs.completed_date, bs.status, bs.error_message,
                    r.region_name
                FROM infrastructure.backup_status bs
                JOIN infrastructure.regions r ON bs.region_id = r.region_id
                {where_clause}
                ORDER BY bs.created_date DESC
                LIMIT ${param_count + 1} OFFSET ${param_count + 2}
            """
            params.extend([limit, offset])

            results = await conn.fetch(query, *params)

            backups = []
            for result in results:
                backup = {
                    'backup_id': str(result['backup_id']),
                    'region_id': result['region_id'],
                    'region_name': result['region_name'],
                    'backup_type': result['backup_type'],
                    'backup_size_gb': float(result['backup_size_gb']),
                    'backup_location': result['backup_location'],
                    'encryption_enabled': result['encryption_enabled'],
                    'retention_days': result['retention_days'],
                    'created_date': result['created_date'].isoformat(),
                    'completed_date': result['completed_date'].isoformat() if result['completed_date'] else None,
                    'status': result['status'],
                    'error_message': result['error_message']
                }
                backups.append(backup)

            return PaginatedResponse(
                success=True,
                message="Backup status retrieved successfully",
                data=backups,
                pagination={
                    'page': page,
                    'limit': limit,
                    'total': total_count,
                    'pages': (total_count + limit - 1) // limit
                }
            )

    except Exception as e:
        logger.error(f"Failed to get backup status: {e}")
        raise HTTPException(status_code=500, detail="Failed to get backup status")

# Compliance Endpoints

@router.get("/compliance/status", response_model=StandardResponse)
async def get_compliance_status(
    token: str = Depends(security)
):
    """Get compliance status across all regions"""
    try:
        # Verify token
        await verify_token(token.credentials)

        # Get compliance status
        compliance_status = await deployment_manager.get_compliance_status()

        return StandardResponse(
            success=True,
            message="Compliance status retrieved successfully",
            data=compliance_status
        )

    except Exception as e:
        logger.error(f"Failed to get compliance status: {e}")
        raise HTTPException(status_code=500, detail="Failed to get compliance status")

@router.get("/compliance/audits", response_model=PaginatedResponse)
async def get_compliance_audits(
    page: int = Query(1, ge=1),
    limit: int = Query(20, ge=1, le=100),
    region_id: Optional[str] = None,
    framework: Optional[str] = None,
    days: int = Query(90, ge=1, le=365),
    token: str = Depends(security)
):
    """Get compliance audit history"""
    try:
        # Verify token
        await verify_token(token.credentials)

        offset = (page - 1) * limit
        since_date = datetime.utcnow() - timedelta(days=days)

        async with get_db_connection() as conn:
            # Build query
            where_conditions = ["audit_date >= $1"]
            params = [since_date]
            param_count = 1

            if region_id:
                param_count += 1
                where_conditions.append(f"region_id = ${param_count}")
                params.append(region_id)

            if framework:
                param_count += 1
                where_conditions.append(f"compliance_framework = ${param_count}")
                params.append(framework)

            where_clause = "WHERE " + " AND ".join(where_conditions)

            # Get total count
            count_query = f"""
                SELECT COUNT(*) FROM infrastructure.compliance_audits
                {where_clause}
            """
            total_count = await conn.fetchval(count_query, *params)

            # Get compliance audits
            query = f"""
                SELECT
                    ca.audit_id, ca.region_id, ca.audit_type, ca.compliance_framework,
                    ca.audit_date, ca.auditor_name, ca.audit_status, ca.compliance_score,
                    ca.next_audit_date, ca.certification_valid_until,
                    r.region_name
                FROM infrastructure.compliance_audits ca
                JOIN infrastructure.regions r ON ca.region_id = r.region_id
                {where_clause}
                ORDER BY ca.audit_date DESC
                LIMIT ${param_count + 1} OFFSET ${param_count + 2}
            """
            params.extend([limit, offset])

            results = await conn.fetch(query, *params)

            audits = []
            for result in results:
                audit = {
                    'audit_id': str(result['audit_id']),
                    'region_id': result['region_id'],
                    'region_name': result['region_name'],
                    'audit_type': result['audit_type'],
                    'compliance_framework': result['compliance_framework'],
                    'audit_date': result['audit_date'].isoformat(),
                    'auditor_name': result['auditor_name'],
                    'audit_status': result['audit_status'],
                    'compliance_score': float(result['compliance_score']) if result['compliance_score'] else 0,
                    'next_audit_date': result['next_audit_date'].isoformat() if result['next_audit_date'] else None,
                    'certification_valid_until': result['certification_valid_until'].isoformat() if result['certification_valid_until'] else None
                }
                audits.append(audit)

            return PaginatedResponse(
                success=True,
                message="Compliance audits retrieved successfully",
                data=audits,
                pagination={
                    'page': page,
                    'limit': limit,
                    'total': total_count,
                    'pages': (total_count + limit - 1) // limit
                }
            )

    except Exception as e:
        logger.error(f"Failed to get compliance audits: {e}")
        raise HTTPException(status_code=500, detail="Failed to get compliance audits")

# Cost and Performance Endpoints

@router.get("/cost/overview", response_model=StandardResponse)
async def get_cost_overview(
    days: int = Query(30, ge=1, le=365),
    region_id: Optional[str] = None,
    token: str = Depends(security)
):
    """Get infrastructure cost overview"""
    try:
        # Verify token
        await verify_token(token.credentials)

        since_date = datetime.utcnow().date() - timedelta(days=days)

        async with get_db_connection() as conn:
            # Build query
            where_conditions = ["cost_date >= $1"]
            params = [since_date]
            param_count = 1

            if region_id:
                param_count += 1
                where_conditions.append(f"region_id = ${param_count}")
                params.append(region_id)

            where_clause = "WHERE " + " AND ".join(where_conditions)

            # Get cost summary
            cost_query = f"""
                SELECT
                    region_id,
                    SUM(cost_amount) as total_cost,
                    COUNT(*) as cost_entries,
                    SUM(optimization_potential) as total_optimization_potential,
                    AVG(cost_amount) as avg_daily_cost
                FROM infrastructure.cost_tracking
                {where_clause}
                GROUP BY region_id
                ORDER BY total_cost DESC
            """

            cost_results = await conn.fetch(cost_query, *params)

            # Get region names
            region_names = {}
            if cost_results:
                region_ids = [r['region_id'] for r in cost_results]
                region_query = """
                    SELECT region_id, region_name
                    FROM infrastructure.regions
                    WHERE region_id = ANY($1)
                """
                region_results = await conn.fetch(region_query, region_ids)
                region_names = {r['region_id']: r['region_name'] for r in region_results}

            # Calculate totals
            total_cost = sum(r['total_cost'] for r in cost_results)
            total_optimization_potential = sum(r['total_optimization_potential'] for r in cost_results)

            cost_overview = {
                'time_period_days': days,
                'total_cost': float(total_cost),
                'total_optimization_potential': float(total_optimization_potential),
                'cost_by_region': [
                    {
                        'region_id': result['region_id'],
                        'region_name': region_names.get(result['region_id'], result['region_id']),
                        'total_cost': float(result['total_cost']),
                        'avg_daily_cost': float(result['avg_daily_cost']),
                        'optimization_potential': float(result['total_optimization_potential']),
                        'cost_entries': result['cost_entries']
                    }
                    for result in cost_results
                ]
            }

            return StandardResponse(
                success=True,
                message="Cost overview retrieved successfully",
                data=cost_overview
            )

    except Exception as e:
        logger.error(f"Failed to get cost overview: {e}")
        raise HTTPException(status_code=500, detail="Failed to get cost overview")

@router.get("/performance/benchmarks", response_model=PaginatedResponse)
async def get_performance_benchmarks(
    page: int = Query(1, ge=1),
    limit: int = Query(20, ge=1, le=100),
    region_id: Optional[str] = None,
    benchmark_type: Optional[str] = None,
    days: int = Query(7, ge=1, le=30),
    token: str = Depends(security)
):
    """Get performance benchmarks"""
    try:
        # Verify token
        await verify_token(token.credentials)

        offset = (page - 1) * limit
        since_date = datetime.utcnow() - timedelta(days=days)

        async with get_db_connection() as conn:
            # Build query
            where_conditions = ["test_date >= $1"]
            params = [since_date]
            param_count = 1

            if region_id:
                param_count += 1
                where_conditions.append(f"region_id = ${param_count}")
                params.append(region_id)

            if benchmark_type:
                param_count += 1
                where_conditions.append(f"benchmark_type = ${param_count}")
                params.append(benchmark_type)

            where_clause = "WHERE " + " AND ".join(where_conditions)

            # Get total count
            count_query = f"""
                SELECT COUNT(*) FROM infrastructure.performance_benchmarks
                {where_clause}
            """
            total_count = await conn.fetchval(count_query, *params)

            # Get performance benchmarks
            query = f"""
                SELECT
                    pb.benchmark_id, pb.region_id, pb.benchmark_type, pb.test_date,
                    pb.test_duration_ms, pb.requests_per_second, pb.avg_response_time_ms,
                    pb.p95_response_time_ms, pb.p99_response_time_ms, pb.error_rate,
                    pb.throughput_mbps, pb.concurrent_users,
                    r.region_name
                FROM infrastructure.performance_benchmarks pb
                JOIN infrastructure.regions r ON pb.region_id = r.region_id
                {where_clause}
                ORDER BY pb.test_date DESC
                LIMIT ${param_count + 1} OFFSET ${param_count + 2}
            """
            params.extend([limit, offset])

            results = await conn.fetch(query, *params)

            benchmarks = []
            for result in results:
                benchmark = {
                    'benchmark_id': str(result['benchmark_id']),
                    'region_id': result['region_id'],
                    'region_name': result['region_name'],
                    'benchmark_type': result['benchmark_type'],
                    'test_date': result['test_date'].isoformat(),
                    'test_duration_ms': result['test_duration_ms'],
                    'requests_per_second': float(result['requests_per_second']),
                    'avg_response_time_ms': float(result['avg_response_time_ms']),
                    'p95_response_time_ms': float(result['p95_response_time_ms']),
                    'p99_response_time_ms': float(result['p99_response_time_ms']),
                    'error_rate': float(result['error_rate']),
                    'throughput_mbps': float(result['throughput_mbps']),
                    'concurrent_users': result['concurrent_users']
                }
                benchmarks.append(benchmark)

            return PaginatedResponse(
                success=True,
                message="Performance benchmarks retrieved successfully",
                data=benchmarks,
                pagination={
                    'page': page,
                    'limit': limit,
                    'total': total_count,
                    'pages': (total_count + limit - 1) // limit
                }
            )

    except Exception as e:
        logger.error(f"Failed to get performance benchmarks: {e}")
        raise HTTPException(status_code=500, detail="Failed to get performance benchmarks")

# Export router
__all__ = ["router"]
