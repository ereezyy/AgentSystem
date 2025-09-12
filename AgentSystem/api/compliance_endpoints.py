"""
Compliance API Endpoints - AgentSystem Profit Machine
SOC2, GDPR, HIPAA compliance management and automation
"""

from fastapi import APIRouter, Depends, HTTPException, Query, BackgroundTasks
from fastapi.security import HTTPBearer
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
import logging
from pydantic import BaseModel, Field
import asyncpg
import uuid

from ..compliance.compliance_framework import (
    ComplianceFramework, ComplianceManager, ComplianceStandard,
    ComplianceStatus, RiskLevel, DataClassification
)
from ..auth.auth_service import get_current_user, require_permissions
from ..database.connection import get_db_pool

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/compliance", tags=["compliance"])
security = HTTPBearer()

# Pydantic models for request/response
class ComplianceAssessmentRequest(BaseModel):
    standard: str = Field(..., description="Compliance standard (soc2_type2, gdpr, hipaa)")
    assessment_type: str = Field(..., description="Assessment type (self, third_party, audit)")
    assessor: str = Field(..., description="Name of assessor")
    scope: str = Field(..., description="Assessment scope description")

class ComplianceAssessmentResponse(BaseModel):
    assessment_id: str
    tenant_id: str
    standard: str
    assessment_type: str
    assessor: str
    scope: str
    start_date: datetime
    status: str
    created_at: datetime

class DataProcessingActivityRequest(BaseModel):
    name: str = Field(..., description="Name of the data processing activity")
    purpose: str = Field(..., description="Purpose of data processing")
    legal_basis: str = Field(..., description="GDPR legal basis")
    data_categories: List[str] = Field(..., description="Categories of personal data")
    data_subjects: List[str] = Field(default=[], description="Categories of data subjects")
    recipients: List[str] = Field(default=[], description="Categories of recipients")
    retention_period: str = Field(..., description="Data retention period")
    cross_border_transfers: bool = Field(default=False, description="Involves cross-border transfers")
    transfer_safeguards: Optional[str] = Field(None, description="Safeguards for transfers")
    automated_decision_making: bool = Field(default=False, description="Involves automated decision making")
    profiling: bool = Field(default=False, description="Involves profiling")
    data_source: str = Field(..., description="Source of the data")
    storage_location: str = Field(..., description="Where data is stored")
    encryption_status: bool = Field(default=True, description="Data encryption status")
    access_controls: List[str] = Field(default=[], description="Access control measures")

class PrivacyRequestRequest(BaseModel):
    request_type: str = Field(..., description="Type of privacy request (access, erasure, portability, etc.)")
    requester_email: str = Field(..., description="Email of the requester")
    data_subject_id: str = Field(..., description="ID of the data subject")
    verify_identity: bool = Field(default=True, description="Whether to verify identity")

class PrivacyRequestResponse(BaseModel):
    request_id: str
    tenant_id: str
    data_subject_id: str
    request_type: str
    request_date: datetime
    requester_email: str
    status: str
    completion_deadline: datetime
    created_at: datetime

class ComplianceReportRequest(BaseModel):
    standard: str = Field(..., description="Compliance standard")
    report_type: str = Field(default="summary", description="Report type (summary, detailed, audit)")

class ComplianceMonitoringResponse(BaseModel):
    timestamp: str
    tenant_id: str
    checks_performed: List[Dict[str, Any]]
    violations_found: List[Dict[str, Any]]
    alerts_generated: List[Dict[str, Any]]
    overall_health: str

# Dependency to get compliance manager
async def get_compliance_manager() -> ComplianceManager:
    db_pool = await get_db_pool()
    return ComplianceManager(db_pool)

@router.post("/assessments", response_model=ComplianceAssessmentResponse)
async def create_compliance_assessment(
    request: ComplianceAssessmentRequest,
    current_user = Depends(get_current_user),
    compliance_manager: ComplianceManager = Depends(get_compliance_manager)
):
    """Create a new compliance assessment"""

    # Validate compliance standard
    try:
        standard = ComplianceStandard(request.standard)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid compliance standard: {request.standard}")

    # Get tenant compliance framework
    framework = await compliance_manager.get_framework(current_user['tenant_id'])

    # Create assessment
    assessment = await framework.create_compliance_assessment(
        standard=standard,
        assessment_type=request.assessment_type,
        assessor=request.assessor,
        scope=request.scope
    )

    return ComplianceAssessmentResponse(
        assessment_id=assessment.assessment_id,
        tenant_id=assessment.tenant_id,
        standard=assessment.standard.value,
        assessment_type=assessment.assessment_type,
        assessor=assessment.assessor,
        scope=assessment.scope,
        start_date=assessment.start_date,
        status=assessment.status,
        created_at=assessment.created_at
    )

@router.get("/assessments")
async def list_compliance_assessments(
    standard: Optional[str] = Query(None, description="Filter by compliance standard"),
    status: Optional[str] = Query(None, description="Filter by assessment status"),
    limit: int = Query(50, le=100, description="Number of assessments to return"),
    offset: int = Query(0, description="Offset for pagination"),
    current_user = Depends(get_current_user),
    db_pool: asyncpg.Pool = Depends(get_db_pool)
):
    """List compliance assessments for tenant"""

    query = """
        SELECT assessment_id, tenant_id, standard, assessment_type, assessor, scope,
               start_date, end_date, status, overall_score, findings_count,
               critical_findings, high_findings, medium_findings, low_findings,
               created_at, updated_at
        FROM compliance.assessments
        WHERE tenant_id = $1
    """
    params = [current_user['tenant_id']]
    param_count = 1

    if standard:
        param_count += 1
        query += f" AND standard = ${param_count}"
        params.append(standard)

    if status:
        param_count += 1
        query += f" AND status = ${param_count}"
        params.append(status)

    query += f" ORDER BY created_at DESC LIMIT ${param_count + 1} OFFSET ${param_count + 2}"
    params.extend([limit, offset])

    async with db_pool.acquire() as conn:
        assessments = await conn.fetch(query, *params)

    return [dict(assessment) for assessment in assessments]

@router.post("/run-compliance-check")
async def run_compliance_check(
    standards: Optional[List[str]] = Query(None, description="Specific standards to check"),
    background_tasks: BackgroundTasks = BackgroundTasks(),
    current_user = Depends(get_current_user),
    compliance_manager: ComplianceManager = Depends(get_compliance_manager)
):
    """Run automated compliance check"""

    # Parse standards if provided
    parsed_standards = None
    if standards:
        try:
            parsed_standards = [ComplianceStandard(std) for std in standards]
        except ValueError as e:
            raise HTTPException(status_code=400, detail=f"Invalid compliance standard: {e}")

    # Run compliance check in background for large assessments
    async def run_check():
        try:
            result = await compliance_manager.run_compliance_scan(
                current_user['tenant_id'],
                parsed_standards
            )
            logger.info(f"Compliance check completed for tenant {current_user['tenant_id']}")
            return result
        except Exception as e:
            logger.error(f"Compliance check failed for tenant {current_user['tenant_id']}: {e}")

    # For now, run synchronously for immediate response
    # In production, this would be queued for background processing
    result = await compliance_manager.run_compliance_scan(
        current_user['tenant_id'],
        parsed_standards
    )

    return result

@router.post("/data-processing-activities", response_model=Dict[str, str])
async def register_data_processing_activity(
    request: DataProcessingActivityRequest,
    current_user = Depends(get_current_user),
    compliance_manager: ComplianceManager = Depends(get_compliance_manager)
):
    """Register a new data processing activity (GDPR Article 30)"""

    # Parse data categories
    try:
        data_categories = [DataClassification(cat) for cat in request.data_categories]
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid data category: {e}")

    # Get tenant compliance framework
    framework = await compliance_manager.get_framework(current_user['tenant_id'])

    # Register activity
    activity = await framework.register_data_processing_activity(
        name=request.name,
        purpose=request.purpose,
        legal_basis=request.legal_basis,
        data_categories=data_categories,
        data_subjects=request.data_subjects,
        recipients=request.recipients,
        retention_period=request.retention_period,
        cross_border_transfers=request.cross_border_transfers,
        transfer_safeguards=request.transfer_safeguards,
        automated_decision_making=request.automated_decision_making,
        profiling=request.profiling,
        data_source=request.data_source,
        storage_location=request.storage_location,
        encryption_status=request.encryption_status,
        access_controls=request.access_controls
    )

    return {
        "activity_id": activity.activity_id,
        "message": "Data processing activity registered successfully"
    }

@router.get("/data-processing-activities")
async def list_data_processing_activities(
    active_only: bool = Query(True, description="Show only active activities"),
    legal_basis: Optional[str] = Query(None, description="Filter by legal basis"),
    limit: int = Query(50, le=100, description="Number of activities to return"),
    offset: int = Query(0, description="Offset for pagination"),
    current_user = Depends(get_current_user),
    db_pool: asyncpg.Pool = Depends(get_db_pool)
):
    """List data processing activities"""

    query = """
        SELECT activity_id, name, purpose, legal_basis, data_categories, data_subjects,
               recipients, retention_period, cross_border_transfers, transfer_safeguards,
               automated_decision_making, profiling, data_source, storage_location,
               encryption_status, access_controls, is_active, created_at, updated_at
        FROM compliance.data_processing_activities
        WHERE tenant_id = $1
    """
    params = [current_user['tenant_id']]
    param_count = 1

    if active_only:
        param_count += 1
        query += f" AND is_active = ${param_count}"
        params.append(True)

    if legal_basis:
        param_count += 1
        query += f" AND legal_basis = ${param_count}"
        params.append(legal_basis)

    query += f" ORDER BY created_at DESC LIMIT ${param_count + 1} OFFSET ${param_count + 2}"
    params.extend([limit, offset])

    async with db_pool.acquire() as conn:
        activities = await conn.fetch(query, *params)

    return [dict(activity) for activity in activities]

@router.post("/privacy-requests", response_model=PrivacyRequestResponse)
async def submit_privacy_request(
    request: PrivacyRequestRequest,
    current_user = Depends(get_current_user),
    compliance_manager: ComplianceManager = Depends(get_compliance_manager)
):
    """Submit a GDPR privacy request"""

    # Validate request type
    valid_types = ['access', 'rectification', 'erasure', 'portability', 'restriction', 'objection']
    if request.request_type not in valid_types:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid request type. Must be one of: {', '.join(valid_types)}"
        )

    # Get tenant compliance framework
    framework = await compliance_manager.get_framework(current_user['tenant_id'])

    # Handle privacy request
    privacy_request = await framework.handle_privacy_request(
        request_type=request.request_type,
        requester_email=request.requester_email,
        data_subject_id=request.data_subject_id,
        verify_identity=request.verify_identity
    )

    return PrivacyRequestResponse(
        request_id=privacy_request.request_id,
        tenant_id=privacy_request.tenant_id,
        data_subject_id=privacy_request.data_subject_id,
        request_type=privacy_request.request_type,
        request_date=privacy_request.request_date,
        requester_email=privacy_request.requester_email,
        status=privacy_request.status,
        completion_deadline=privacy_request.completion_deadline,
        created_at=privacy_request.created_at
    )

@router.get("/privacy-requests")
async def list_privacy_requests(
    status: Optional[str] = Query(None, description="Filter by request status"),
    request_type: Optional[str] = Query(None, description="Filter by request type"),
    overdue_only: bool = Query(False, description="Show only overdue requests"),
    limit: int = Query(50, le=100, description="Number of requests to return"),
    offset: int = Query(0, description="Offset for pagination"),
    current_user = Depends(get_current_user),
    db_pool: asyncpg.Pool = Depends(get_db_pool)
):
    """List privacy requests"""

    query = """
        SELECT request_id, data_subject_id, request_type, request_date, requester_email,
               requester_identity_verified, status, completion_deadline, response_date,
               response_method, actions_taken, rejection_reason, created_at, updated_at
        FROM compliance.privacy_requests
        WHERE tenant_id = $1
    """
    params = [current_user['tenant_id']]
    param_count = 1

    if status:
        param_count += 1
        query += f" AND status = ${param_count}"
        params.append(status)

    if request_type:
        param_count += 1
        query += f" AND request_type = ${param_count}"
        params.append(request_type)

    if overdue_only:
        param_count += 1
        query += f" AND completion_deadline < ${param_count} AND status NOT IN ('completed', 'rejected')"
        params.append(datetime.now())

    query += f" ORDER BY request_date DESC LIMIT ${param_count + 1} OFFSET ${param_count + 2}"
    params.extend([limit, offset])

    async with db_pool.acquire() as conn:
        requests = await conn.fetch(query, *params)

    return [dict(req) for req in requests]

@router.post("/generate-report")
async def generate_compliance_report(
    request: ComplianceReportRequest,
    current_user = Depends(get_current_user),
    compliance_manager: ComplianceManager = Depends(get_compliance_manager)
):
    """Generate compliance report"""

    try:
        standard = ComplianceStandard(request.standard)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid compliance standard: {request.standard}")

    # Generate report
    report = await compliance_manager.generate_audit_report(
        current_user['tenant_id'],
        standard
    )

    return report

@router.get("/monitoring/status", response_model=ComplianceMonitoringResponse)
async def get_compliance_monitoring_status(
    current_user = Depends(get_current_user),
    compliance_manager: ComplianceManager = Depends(get_compliance_manager)
):
    """Get current compliance monitoring status"""

    # Get tenant compliance framework
    framework = await compliance_manager.get_framework(current_user['tenant_id'])

    # Run monitoring check
    result = await framework.monitor_continuous_compliance()

    return ComplianceMonitoringResponse(
        timestamp=result['timestamp'],
        tenant_id=result['tenant_id'],
        checks_performed=result['checks_performed'],
        violations_found=result['violations_found'],
        alerts_generated=result['alerts_generated'],
        overall_health=result['overall_health']
    )

@router.get("/dashboard")
async def get_compliance_dashboard(
    current_user = Depends(get_current_user),
    db_pool: asyncpg.Pool = Depends(get_db_pool)
):
    """Get compliance dashboard overview"""

    async with db_pool.acquire() as conn:
        # Get dashboard data from view
        dashboard_data = await conn.fetchrow("""
            SELECT * FROM compliance.compliance_dashboard
            WHERE tenant_id = $1
        """, current_user['tenant_id'])

        if not dashboard_data:
            return {
                'tenant_id': current_user['tenant_id'],
                'total_controls': 0,
                'compliant_controls': 0,
                'non_compliant_controls': 0,
                'critical_controls': 0,
                'open_findings': 0,
                'critical_findings': 0,
                'pending_privacy_requests': 0,
                'last_monitoring_check': None
            }

        # Get recent assessments
        recent_assessments = await conn.fetch("""
            SELECT assessment_id, standard, assessment_type, status, overall_score, created_at
            FROM compliance.assessments
            WHERE tenant_id = $1
            ORDER BY created_at DESC
            LIMIT 5
        """, current_user['tenant_id'])

        # Get compliance trends
        compliance_trends = await conn.fetch("""
            SELECT DATE(timestamp) as date, overall_status, COUNT(*) as count
            FROM compliance.assessment_results
            WHERE tenant_id = $1 AND timestamp > NOW() - INTERVAL '30 days'
            GROUP BY DATE(timestamp), overall_status
            ORDER BY date DESC
        """, current_user['tenant_id'])

        # Get upcoming deadlines
        upcoming_deadlines = await conn.fetch("""
            SELECT pr.request_id, pr.request_type, pr.completion_deadline, pr.requester_email
            FROM compliance.privacy_requests pr
            WHERE pr.tenant_id = $1
            AND pr.completion_deadline > NOW()
            AND pr.completion_deadline < NOW() + INTERVAL '7 days'
            AND pr.status IN ('received', 'processing')
            ORDER BY pr.completion_deadline ASC
        """, current_user['tenant_id'])

    return {
        'summary': dict(dashboard_data),
        'recent_assessments': [dict(assessment) for assessment in recent_assessments],
        'compliance_trends': [dict(trend) for trend in compliance_trends],
        'upcoming_deadlines': [dict(deadline) for deadline in upcoming_deadlines]
    }

@router.get("/controls")
async def list_compliance_controls(
    standard: Optional[str] = Query(None, description="Filter by compliance standard"),
    status: Optional[str] = Query(None, description="Filter by control status"),
    risk_level: Optional[str] = Query(None, description="Filter by risk level"),
    limit: int = Query(50, le=100, description="Number of controls to return"),
    offset: int = Query(0, description="Offset for pagination"),
    current_user = Depends(get_current_user),
    db_pool: asyncpg.Pool = Depends(get_db_pool)
):
    """List compliance controls"""

    query = """
        SELECT control_id, standard, control_number, title, description, category,
               subcategory, risk_level, frequency, responsible_team, status,
               last_assessed, next_assessment, created_at, updated_at
        FROM compliance.controls
        WHERE tenant_id = $1
    """
    params = [current_user['tenant_id']]
    param_count = 1

    if standard:
        param_count += 1
        query += f" AND standard = ${param_count}"
        params.append(standard)

    if status:
        param_count += 1
        query += f" AND status = ${param_count}"
        params.append(status)

    if risk_level:
        param_count += 1
        query += f" AND risk_level = ${param_count}"
        params.append(risk_level)

    query += f" ORDER BY risk_level DESC, control_number LIMIT ${param_count + 1} OFFSET ${param_count + 2}"
    params.extend([limit, offset])

    async with db_pool.acquire() as conn:
        controls = await conn.fetch(query, *params)

    return [dict(control) for control in controls]

@router.get("/findings")
async def list_compliance_findings(
    severity: Optional[str] = Query(None, description="Filter by severity"),
    status: Optional[str] = Query(None, description="Filter by finding status"),
    overdue_only: bool = Query(False, description="Show only overdue findings"),
    limit: int = Query(50, le=100, description="Number of findings to return"),
    offset: int = Query(0, description="Offset for pagination"),
    current_user = Depends(get_current_user),
    db_pool: asyncpg.Pool = Depends(get_db_pool)
):
    """List compliance findings"""

    query = """
        SELECT finding_id, assessment_id, control_id, finding_type, severity, title,
               description, evidence, recommendation, status, assigned_to, due_date,
               resolution_notes, resolved_at, created_at, updated_at
        FROM compliance.findings
        WHERE tenant_id = $1
    """
    params = [current_user['tenant_id']]
    param_count = 1

    if severity:
        param_count += 1
        query += f" AND severity = ${param_count}"
        params.append(severity)

    if status:
        param_count += 1
        query += f" AND status = ${param_count}"
        params.append(status)

    if overdue_only:
        param_count += 1
        query += f" AND due_date < ${param_count} AND status NOT IN ('resolved', 'accepted_risk')"
        params.append(datetime.now())

    query += f" ORDER BY severity DESC, created_at DESC LIMIT ${param_count + 1} OFFSET ${param_count + 2}"
    params.extend([limit, offset])

    async with db_pool.acquire() as conn:
        findings = await conn.fetch(query, *params)

    return [dict(finding) for finding in findings]

@router.put("/findings/{finding_id}/status")
async def update_finding_status(
    finding_id: str,
    status: str = Query(..., description="New status for the finding"),
    resolution_notes: Optional[str] = Query(None, description="Resolution notes"),
    current_user = Depends(get_current_user),
    db_pool: asyncpg.Pool = Depends(get_db_pool)
):
    """Update compliance finding status"""

    valid_statuses = ['open', 'in_progress', 'resolved', 'accepted_risk']
    if status not in valid_statuses:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid status. Must be one of: {', '.join(valid_statuses)}"
        )

    async with db_pool.acquire() as conn:
        # Verify finding belongs to tenant
        finding = await conn.fetchrow("""
            SELECT finding_id FROM compliance.findings
            WHERE finding_id = $1 AND tenant_id = $2
        """, finding_id, current_user['tenant_id'])

        if not finding:
            raise HTTPException(status_code=404, detail="Finding not found")

        # Update finding status
        await conn.execute("""
            UPDATE compliance.findings
            SET status = $1, resolution_notes = $2,
                resolved_at = CASE WHEN $1 = 'resolved' THEN NOW() ELSE NULL END,
                updated_at = NOW()
            WHERE finding_id = $3
        """, status, resolution_notes, finding_id)

    return {"message": "Finding status updated successfully"}

@router.get("/standards")
async def get_supported_standards():
    """Get list of supported compliance standards"""

    standards = [
        {
            "code": standard.value,
            "name": standard.name,
            "description": _get_standard_description(standard)
        }
        for standard in ComplianceStandard
    ]

    return {"standards": standards}

def _get_standard_description(standard: ComplianceStandard) -> str:
    """Get description for compliance standard"""

    descriptions = {
        ComplianceStandard.SOC2_TYPE1: "SOC 2 Type I - System design evaluation at a point in time",
        ComplianceStandard.SOC2_TYPE2: "SOC 2 Type II - System operating effectiveness over time",
        ComplianceStandard.GDPR: "General Data Protection Regulation - EU data protection law",
        ComplianceStandard.HIPAA: "Health Insurance Portability and Accountability Act - US healthcare data protection",
        ComplianceStandard.PCI_DSS: "Payment Card Industry Data Security Standard - Payment data protection",
        ComplianceStandard.ISO_27001: "ISO 27001 - Information security management systems",
        ComplianceStandard.CCPA: "California Consumer Privacy Act - California privacy law",
        ComplianceStandard.NIST: "NIST Cybersecurity Framework - US cybersecurity standards"
    }

    return descriptions.get(standard, "")

# Include router in main application
def setup_compliance_routes(app):
    """Setup compliance routes in FastAPI application"""
    app.include_router(router)
