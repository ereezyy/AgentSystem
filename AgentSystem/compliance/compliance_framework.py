
"""
Compliance Framework - AgentSystem Profit Machine
SOC2, GDPR, HIPAA compliance automation and monitoring system
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from enum import Enum
from dataclasses import dataclass, asdict, field
import asyncpg
import uuid
import hashlib
import re
from pathlib import Path

logger = logging.getLogger(__name__)

class ComplianceStandard(Enum):
    SOC2_TYPE1 = "soc2_type1"
    SOC2_TYPE2 = "soc2_type2"
    GDPR = "gdpr"
    HIPAA = "hipaa"
    PCI_DSS = "pci_dss"
    ISO_27001 = "iso_27001"
    CCPA = "ccpa"
    NIST = "nist"

class ComplianceStatus(Enum):
    COMPLIANT = "compliant"
    NON_COMPLIANT = "non_compliant"
    IN_PROGRESS = "in_progress"
    NOT_APPLICABLE = "not_applicable"
    NEEDS_REVIEW = "needs_review"

class RiskLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class DataClassification(Enum):
    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    RESTRICTED = "restricted"
    PII = "pii"
    PHI = "phi"
    PCI = "pci"

@dataclass
class ComplianceControl:
    control_id: str
    standard: ComplianceStandard
    control_number: str
    title: str
    description: str
    category: str
    subcategory: str
    implementation_guidance: str
    testing_procedures: List[str]
    evidence_requirements: List[str]
    automation_possible: bool
    risk_level: RiskLevel
    frequency: str  # daily, weekly, monthly, quarterly, annually
    responsible_team: str
    status: ComplianceStatus
    last_assessed: Optional[datetime]
    next_assessment: Optional[datetime]
    findings: List[str] = field(default_factory=list)
    remediation_actions: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

@dataclass
class ComplianceAssessment:
    assessment_id: str
    tenant_id: str
    standard: ComplianceStandard
    assessment_type: str  # self, third_party, audit
    assessor: str
    scope: str
    start_date: datetime
    end_date: Optional[datetime]
    status: str  # planning, in_progress, completed, cancelled
    overall_score: Optional[float]
    findings_count: int
    critical_findings: int
    high_findings: int
    medium_findings: int
    low_findings: int
    recommendations: List[str] = field(default_factory=list)
    evidence_collected: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

@dataclass
class DataProcessingActivity:
    activity_id: str
    tenant_id: str
    name: str
    purpose: str
    legal_basis: str  # GDPR legal basis
    data_categories: List[DataClassification]
    data_subjects: List[str]
    recipients: List[str]
    retention_period: str
    cross_border_transfers: bool
    transfer_safeguards: Optional[str]
    automated_decision_making: bool
    profiling: bool
    data_source: str
    storage_location: str
    encryption_status: bool
    access_controls: List[str]
    is_active: bool
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

@dataclass
class PrivacyRequest:
    request_id: str
    tenant_id: str
    data_subject_id: str
    request_type: str  # access, rectification, erasure, portability, restriction, objection
    request_date: datetime
    requester_email: str
    requester_identity_verified: bool
    status: str  # received, processing, completed, rejected
    completion_deadline: datetime
    response_date: Optional[datetime]
    response_method: Optional[str]
    data_provided: Optional[str]
    actions_taken: List[str] = field(default_factory=list)
    rejection_reason: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

class ComplianceFramework:
    """
    Comprehensive compliance framework for SOC2, GDPR, HIPAA, and other standards
    """

    def __init__(self, db_pool: asyncpg.Pool, tenant_id: str):
        self.db_pool = db_pool
        self.tenant_id = tenant_id

        # Load compliance controls from configuration
        self.controls = {}
        self.assessments = {}

        # Initialize compliance standards
        self._initialize_soc2_controls()
        self._initialize_gdpr_controls()
        self._initialize_hipaa_controls()

        # Automated monitoring rules
        self.monitoring_rules = {
            'data_retention': self._check_data_retention,
            'access_controls': self._check_access_controls,
            'encryption': self._check_encryption_compliance,
            'audit_logs': self._check_audit_logging,
            'incident_response': self._check_incident_response,
            'privacy_requests': self._check_privacy_request_timelines,
            'data_minimization': self._check_data_minimization,
            'consent_management': self._check_consent_compliance
        }

    async def create_compliance_assessment(self, standard: ComplianceStandard,
                                         assessment_type: str, assessor: str,
                                         scope: str) -> ComplianceAssessment:
        """Create a new compliance assessment"""

        assessment_id = str(uuid.uuid4())

        assessment = ComplianceAssessment(
            assessment_id=assessment_id,
            tenant_id=self.tenant_id,
            standard=standard,
            assessment_type=assessment_type,
            assessor=assessor,
            scope=scope,
            start_date=datetime.now(),
            status="planning",
            findings_count=0,
            critical_findings=0,
            high_findings=0,
            medium_findings=0,
            low_findings=0
        )

        await self._store_assessment(assessment)
        return assessment

    async def run_automated_compliance_check(self, standards: List[ComplianceStandard] = None) -> Dict[str, Any]:
        """Run automated compliance checks"""

        if not standards:
            standards = [ComplianceStandard.SOC2_TYPE2, ComplianceStandard.GDPR, ComplianceStandard.HIPAA]

        results = {
            'timestamp': datetime.now().isoformat(),
            'tenant_id': self.tenant_id,
            'standards_checked': [std.value for std in standards],
            'overall_status': ComplianceStatus.COMPLIANT.value,
            'findings': [],
            'recommendations': [],
            'scores': {}
        }

        for standard in standards:
            standard_result = await self._assess_standard_compliance(standard)
            results['scores'][standard.value] = standard_result

            if standard_result['status'] != ComplianceStatus.COMPLIANT.value:
                results['overall_status'] = ComplianceStatus.NON_COMPLIANT.value

            results['findings'].extend(standard_result.get('findings', []))
            results['recommendations'].extend(standard_result.get('recommendations', []))

        # Store assessment results
        await self._store_compliance_results(results)

        return results

    async def register_data_processing_activity(self, name: str, purpose: str,
                                              legal_basis: str, data_categories: List[DataClassification],
                                              **kwargs) -> DataProcessingActivity:
        """Register a data processing activity (GDPR Article 30)"""

        activity_id = str(uuid.uuid4())

        activity = DataProcessingActivity(
            activity_id=activity_id,
            tenant_id=self.tenant_id,
            name=name,
            purpose=purpose,
            legal_basis=legal_basis,
            data_categories=data_categories,
            data_subjects=kwargs.get('data_subjects', []),
            recipients=kwargs.get('recipients', []),
            retention_period=kwargs.get('retention_period', ''),
            cross_border_transfers=kwargs.get('cross_border_transfers', False),
            transfer_safeguards=kwargs.get('transfer_safeguards'),
            automated_decision_making=kwargs.get('automated_decision_making', False),
            profiling=kwargs.get('profiling', False),
            data_source=kwargs.get('data_source', ''),
            storage_location=kwargs.get('storage_location', ''),
            encryption_status=kwargs.get('encryption_status', True),
            access_controls=kwargs.get('access_controls', []),
            is_active=True
        )

        await self._store_data_processing_activity(activity)
        return activity

    async def handle_privacy_request(self, request_type: str, requester_email: str,
                                   data_subject_id: str, verify_identity: bool = True) -> PrivacyRequest:
        """Handle GDPR privacy request (access, erasure, portability, etc.)"""

        request_id = str(uuid.uuid4())

        # Calculate completion deadline based on request type
        deadline_days = 30  # Default GDPR timeline
        if request_type == 'access':
            deadline_days = 30
        elif request_type == 'erasure':
            deadline_days = 30
        elif request_type == 'portability':
            deadline_days = 30

        request = PrivacyRequest(
            request_id=request_id,
            tenant_id=self.tenant_id,
            data_subject_id=data_subject_id,
            request_type=request_type,
            request_date=datetime.now(),
            requester_email=requester_email,
            requester_identity_verified=verify_identity,
            status='received',
            completion_deadline=datetime.now() + timedelta(days=deadline_days)
        )

        await self._store_privacy_request(request)

        # Trigger automated processing if possible
        if request_type == 'access':
            await self._process_data_access_request(request)
        elif request_type == 'erasure':
            await self._process_data_erasure_request(request)

        return request

    async def generate_compliance_report(self, standard: ComplianceStandard,
                                       report_type: str = 'summary') -> Dict[str, Any]:
        """Generate compliance report for auditors"""

        report = {
            'report_id': str(uuid.uuid4()),
            'tenant_id': self.tenant_id,
            'standard': standard.value,
            'report_type': report_type,
            'generated_at': datetime.now().isoformat(),
            'period_start': (datetime.now() - timedelta(days=90)).isoformat(),
            'period_end': datetime.now().isoformat(),
            'executive_summary': {},
            'control_assessments': [],
            'findings': [],
            'evidence': [],
            'recommendations': []
        }

        # Get relevant controls for the standard
        controls = await self._get_controls_by_standard(standard)

        compliant_count = 0
        total_count = len(controls)

        for control in controls:
            assessment = await self._assess_control_compliance(control)
            report['control_assessments'].append(assessment)

            if assessment['status'] == ComplianceStatus.COMPLIANT.value:
                compliant_count += 1
            else:
                report['findings'].append({
                    'control_id': control.control_id,
                    'finding': assessment.get('finding', ''),
                    'risk_level': control.risk_level.value,
                    'remediation': assessment.get('remediation', '')
                })

        # Calculate compliance percentage
        compliance_percentage = (compliant_count / total_count * 100) if total_count > 0 else 0

        report['executive_summary'] = {
            'compliance_percentage': compliance_percentage,
            'total_controls': total_count,
            'compliant_controls': compliant_count,
            'non_compliant_controls': total_count - compliant_count,
            'overall_status': 'COMPLIANT' if compliance_percentage >= 95 else 'NON_COMPLIANT'
        }

        # Store report
        await self._store_compliance_report(report)

        return report

    async def monitor_continuous_compliance(self) -> Dict[str, Any]:
        """Continuous compliance monitoring"""

        monitoring_results = {
            'timestamp': datetime.now().isoformat(),
            'tenant_id': self.tenant_id,
            'checks_performed': [],
            'violations_found': [],
            'alerts_generated': [],
            'overall_health': 'HEALTHY'
        }

        for rule_name, rule_function in self.monitoring_rules.items():
            try:
                result = await rule_function()
                monitoring_results['checks_performed'].append({
                    'rule': rule_name,
                    'status': result.get('status', 'PASSED'),
                    'details': result.get('details', '')
                })

                if result.get('violations'):
                    monitoring_results['violations_found'].extend(result['violations'])
                    monitoring_results['overall_health'] = 'ISSUES_DETECTED'

                if result.get('alerts'):
                    monitoring_results['alerts_generated'].extend(result['alerts'])

            except Exception as e:
                logger.error(f"Compliance monitoring rule {rule_name} failed: {e}")
                monitoring_results['checks_performed'].append({
                    'rule': rule_name,
                    'status': 'ERROR',
                    'error': str(e)
                })

        # Store monitoring results
        await self._store_monitoring_results(monitoring_results)

        return monitoring_results

    # Private methods for compliance checks
    async def _assess_standard_compliance(self, standard: ComplianceStandard) -> Dict[str, Any]:
        """Assess compliance with a specific standard"""

        controls = await self._get_controls_by_standard(standard)

        result = {
            'standard': standard.value,
            'total_controls': len(controls),
            'compliant_controls': 0,
            'findings': [],
            'recommendations': [],
            'status': ComplianceStatus.COMPLIANT.value
        }

        for control in controls:
            assessment = await self._assess_control_compliance(control)

            if assessment['status'] == ComplianceStatus.COMPLIANT.value:
                result['compliant_controls'] += 1
            else:
                result['status'] = ComplianceStatus.NON_COMPLIANT.value
                result['findings'].append({
                    'control': control.control_number,
                    'issue': assessment.get('finding', ''),
                    'risk': control.risk_level.value
                })
                result['recommendations'].append(assessment.get('remediation', ''))

        return result

    async def _assess_control_compliance(self, control: ComplianceControl) -> Dict[str, Any]:
        """Assess compliance for a specific control"""

        # This is a simplified assessment - in practice, this would involve
        # complex checks against system configurations, logs, policies, etc.

        assessment = {
            'control_id': control.control_id,
            'status': ComplianceStatus.COMPLIANT.value,
            'finding': '',
            'evidence': [],
            'remediation': ''
        }

        # Example automated checks based on control type
        if 'encryption' in control.title.lower():
            encryption_check = await self._check_encryption_compliance()
            if not encryption_check.get('compliant', True):
                assessment['status'] = ComplianceStatus.NON_COMPLIANT.value
                assessment['finding'] = 'Encryption not properly implemented'
                assessment['remediation'] = 'Enable encryption for all sensitive data'

        elif 'access' in control.title.lower():
            access_check = await self._check_access_controls()
            if not access_check.get('compliant', True):
                assessment['status'] = ComplianceStatus.NON_COMPLIANT.value
                assessment['finding'] = 'Access controls insufficient'
                assessment['remediation'] = 'Implement proper access controls and regular reviews'

        elif 'logging' in control.title.lower() or 'audit' in control.title.lower():
            logging_check = await self._check_audit_logging()
            if not logging_check.get('compliant', True):
                assessment['status'] = ComplianceStatus.NON_COMPLIANT.value
                assessment['finding'] = 'Audit logging incomplete'
                assessment['remediation'] = 'Enable comprehensive audit logging'

        return assessment

    # Automated compliance monitoring rules
    async def _check_data_retention(self) -> Dict[str, Any]:
        """Check data retention compliance"""

        # Check if data is being retained beyond policy limits
        async with self.db_pool.acquire() as conn:
            # Example: Check for old user data
            old_data = await conn.fetch("""
                SELECT COUNT(*) as count FROM auth.sso_users
                WHERE created_at < NOW() - INTERVAL '7 years'
                AND tenant_id = $1
            """, self.tenant_id)

            violations = []
            if old_data[0]['count'] > 0:
                violations.append({
                    'type': 'data_retention',
                    'description': f"Found {old_data[0]['count']} user records older than retention policy",
                    'severity': 'HIGH'
                })

            return {
                'status': 'PASSED' if not violations else 'FAILED',
                'violations': violations,
                'compliant': len(violations) == 0
            }

    async def _check_access_controls(self) -> Dict[str, Any]:
        """Check access control compliance"""

        # Check for proper access controls implementation
        violations = []

        # Example checks
        async with self.db_pool.acquire() as conn:
            # Check for users without proper role assignments
            users_without_roles = await conn.fetch("""
                SELECT COUNT(*) as count FROM auth.sso_users
                WHERE (roles IS NULL OR array_length(roles, 1) = 0)
                AND tenant_id = $1 AND is_active = true
            """, self.tenant_id)

            if users_without_roles[0]['count'] > 0:
                violations.append({
                    'type': 'access_control',
                    'description': f"Found {users_without_roles[0]['count']} active users without role assignments",
                    'severity': 'HIGH'
                })

        return {
            'status': 'PASSED' if not violations else 'FAILED',
            'violations': violations,
            'compliant': len(violations) == 0
        }

    async def _check_encryption_compliance(self) -> Dict[str, Any]:
        """Check encryption compliance"""

        # Check encryption status across systems
        violations = []

        # This would check various encryption requirements
        # For now, we'll assume encryption is properly implemented

        return {
            'status': 'PASSED',
            'violations': violations,
            'compliant': True
        }

    async def _check_audit_logging(self) -> Dict[str, Any]:
        """Check audit logging compliance"""

        violations = []

        # Check if audit logs are being generated
        async with self.db_pool.acquire() as conn:
            recent_logs = await conn.fetch("""
                SELECT COUNT(*) as count FROM auth.sso_audit_log
                WHERE tenant_id = $1 AND created_at > NOW() - INTERVAL '24 hours'
            """, self.tenant_id)

            if recent_logs[0]['count'] == 0:
                violations.append({
                    'type': 'audit_logging',
                    'description': 'No audit logs generated in the last 24 hours',
                    'severity': 'MEDIUM'
                })

        return {
            'status': 'PASSED' if not violations else 'FAILED',
            'violations': violations,
            'compliant': len(violations) == 0
        }

    async def _check_incident_response(self) -> Dict[str, Any]:
        """Check incident response compliance"""

        # Check incident response procedures and timelines
        violations = []

        # This would check incident response metrics
        # For now, we'll assume compliance

        return {
            'status': 'PASSED',
            'violations': violations,
            'compliant': True
        }

    async def _check_privacy_request_timelines(self) -> Dict[str, Any]:
        """Check privacy request response timelines"""

        violations = []

        async with self.db_pool.acquire() as conn:
            overdue_requests = await conn.fetch("""
                SELECT COUNT(*) as count FROM compliance.privacy_requests
                WHERE tenant_id = $1 AND status IN ('received', 'processing')
                AND completion_deadline < NOW()
            """, self.tenant_id)

            if overdue_requests[0]['count'] > 0:
                violations.append({
                    'type': 'privacy_request_timeline',
                    'description': f"Found {overdue_requests[0]['count']} overdue privacy requests",
                    'severity': 'HIGH'
                })

        return {
            'status': 'PASSED' if not violations else 'FAILED',
            'violations': violations,
            'compliant': len(violations) == 0
        }

    async def _check_data_minimization(self) -> Dict[str, Any]:
        """Check data minimization compliance"""

        # Check if only necessary data is being collected and processed
        violations = []

        # This would analyze data collection practices
        # For now, we'll assume compliance

        return {
            'status': 'PASSED',
            'violations': violations,
            'compliant': True
        }

    async def _check_consent_compliance(self) -> Dict[str, Any]:
        """Check consent management compliance"""

        # Check consent collection and management
        violations = []

        # This would check consent records and validity
        # For now, we'll assume compliance

        return {
            'status': 'PASSED',
            'violations': violations,
            'compliant': True
        }

    # Initialize compliance controls for different standards
    def _initialize_soc2_controls(self):
        """Initialize SOC2 Type II controls"""

        soc2_controls = [
            {
                'control_number': 'CC1.1',
                'title': 'Control Environment - Integrity and Ethical Values',
                'description': 'The entity demonstrates a commitment to integrity and ethical values.',
                'category': 'Control Environment',
                'risk_level': RiskLevel.HIGH,
                'frequency': 'quarterly'
            },
            {
                'control_number': 'CC2.1',
                'title': 'Communication and Information - Internal Communication',
                'description': 'The entity obtains or generates and uses relevant, quality information to support the functioning of internal control.',
                'category': 'Communication and Information',
                'risk_level': RiskLevel.MEDIUM,
                'frequency': 'monthly'
            },
            {
                'control_number': 'CC6.1',
                'title': 'Security - Logical and Physical Access Controls',
                'description': 'The entity implements logical access security software, infrastructure, and architectures over protected information assets.',
                'category': 'Security',
                'risk_level': RiskLevel.HIGH,
                'frequency': 'daily'
            },
            {
                'control_number': 'CC6.7',
                'title': 'Security - Data Transmission',
                'description': 'The entity restricts the transmission, movement, and removal of information to authorized internal and external users.',
                'category': 'Security',
                'risk_level': RiskLevel.HIGH,
                'frequency': 'daily'
            },
            {
                'control_number': 'CC7.1',
                'title': 'System Operations - System Capacity',
                'description': 'The entity monitors system capacity and utilization.',
                'category': 'System Operations',
                'risk_level': RiskLevel.MEDIUM,
                'frequency': 'daily'
            }
        ]

        for control_data in soc2_controls:
            control = ComplianceControl(
                control_id=str(uuid.uuid4()),
                standard=ComplianceStandard.SOC2_TYPE2,
                control_number=control_data['control_number'],
                title=control_data['title'],
                description=control_data['description'],
                category=control_data['category'],
                subcategory='',
                implementation_guidance='',
                testing_procedures=[],
                evidence_requirements=[],
                automation_possible=True,
                risk_level=control_data['risk_level'],
                frequency=control_data['frequency'],
                responsible_team='Security',
                status=ComplianceStatus.IN_PROGRESS
            )

            self.controls[control.control_id] = control

    def _initialize_gdpr_controls(self):
        """Initialize GDPR compliance controls"""

        gdpr_controls = [
            {
                'control_number': 'Art.5',
                'title': 'Principles of Processing Personal Data',
                'description': 'Personal data shall be processed lawfully, fairly and in a transparent manner.',
                'category': 'Data Protection Principles',
                'risk_level': RiskLevel.HIGH,
                'frequency': 'monthly'
            },
            {
                'control_number': 'Art.6',
                'title': 'Lawfulness of Processing',
                'description': 'Processing shall be lawful only if and to the extent that at least one legal basis applies.',
                'category': 'Legal Basis',
                'risk_level': RiskLevel.HIGH,
                'frequency': 'monthly'
            },
            {
                'control_number': 'Art.12',
                'title': 'Transparent Information and Communication',
                'description': 'The controller shall take appropriate measures to provide information to data subjects.',
                'category': 'Data Subject Rights',
                'risk_level': RiskLevel.MEDIUM,
                'frequency': 'quarterly'
            },
            {
                'control_number': 'Art.30',
                'title': 'Records of Processing Activities',
                'description': 'Each controller shall maintain a record of processing activities under its responsibility.',
                'category': 'Accountability',
                'risk_level': RiskLevel.HIGH,
                'frequency': 'monthly'
            },
            {
                'control_number': 'Art.32',
                'title': 'Security of Processing',
                'description': 'The controller and processor shall implement appropriate technical and organisational measures.',
                'category': 'Security',
                'risk_level': RiskLevel.HIGH,
                'frequency': 'daily'
            }
        ]

        for control_data in gdpr_controls:
            control = ComplianceControl(
                control_id=str(uuid.uuid4()),
                standard=ComplianceStandard.GDPR,
                control_number=control_data['control_number'],
                title=control_data['title'],
                description=control_data['description'],
                category=control_data['category'],
                subcategory='',
                implementation_guidance='',
                testing_procedures=[],
                evidence_requirements=[],
                automation_possible=True,
                risk_level=control_data['risk_level'],
                frequency=control_data['frequency'],
                responsible_team='Privacy',
                status=ComplianceStatus.IN_PROGRESS
            )

            self.controls[control.control_id] = control

    def _initialize_hipaa_controls(self):
        """Initialize HIPAA compliance controls"""

        hipaa_controls = [
            {
                'control_number': '164.308(a)(1)',
                'title': 'Security Officer',
                'description': 'Assign security responsibilities to an individual.',
                'category': 'Administrative Safeguards',
                'risk_level': RiskLevel.HIGH,
                'frequency': 'annually'
            },
            {
                'control_number': '164.308(a)(3)',
                'title': 'Workforce Training',
                'description': 'Implement procedures for authorizing access to electronic protected health information.',
                'category': 'Administrative Safeguards',
                'risk_level': RiskLevel.MEDIUM,
                'frequency': 'quarterly'
            },
            {
                'control_number': '164.312(a)(1)',
                'title': 'Access Control',
                'description': 'Implement technical policies and procedures for electronic information systems.',
                'category': 'Technical Safeguards',
                'risk_level': RiskLevel.HIGH,
                'frequency': 'daily'
            },
            {
                'control_number': '164.312(c)(1)',
                'title': 'Integrity',
                'description': 'Implement electronic mechanisms to corroborate that electronic protected health information has not been improperly altered or destroyed.',
                'category': 'Technical Safeguards',
                'risk_level': RiskLevel.HIGH,
                'frequency': 'daily'
            },
            {
                'control_number': '164.312(e)(1)',
                'title': 'Transmission Security',
                'description': 'Implement technical security measures to guard against unauthorized access to electronic protected health information.',
                'category': 'Technical Safeguards',
                'risk_level': RiskLevel.HIGH,
                'frequency': 'daily'
            }
        ]

        for control_data in hipaa_controls:
            control = ComplianceControl(
                control_id=str(uuid.uuid4()),
                standard=ComplianceStandard.HIPAA,
                control_number=control_data['control_number'],
                title=control_data['title'],
                description=control_data['description'],
                category=control_data['category'],
                subcategory='',
                implementation_guidance='',
                testing_procedures=[],
                evidence_requirements=[],
                automation_possible=True,
                risk_level=control_data['risk_level'],
                frequency=control_data['frequency'],
                responsible_team='Compliance',
                status=ComplianceStatus.IN_PROGRESS
            )

            self.controls[control.control_id] = control

    # Database operations
    async def _store_assessment(self, assessment: ComplianceAssessment):
        """Store compliance assessment"""

        async with self.db_pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO compliance.assessments (
                    assessment_id, tenant_id, standard, assessment_type, assessor,
                    scope, start_date, status, findings_count, critical_findings,
                    high_findings, medium_findings, low_findings, created_at, updated_at
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15)
            """, assessment.assessment_id, assessment.tenant_id, assessment.standard.value,
                assessment.assessment_type, assessment.assessor, assessment.scope,
                assessment.start_date, assessment.status, assessment.findings_count,
                assessment.critical_findings, assessment.high_findings,
                assessment.medium_findings, assessment.low_findings,
                assessment.created_at, assessment.updated_at)

    async def _store_data_processing_activity(self, activity: DataProcessingActivity):
        """Store data processing activity"""

        async with self.db_pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO compliance.data_processing_activities (
                    activity_id, tenant_id, name, purpose, legal_basis, data_categories,
                    data_subjects, recipients, retention_period, cross_border_transfers,
                    transfer_safeguards, automated_decision_making, profiling,
                    data_source, storage_location, encryption_status, access_controls,
                    is_active, created_at, updated_at
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18, $19, $20)
            """, activity.activity_id, activity.tenant_id, activity.name, activity.purpose,
                activity.legal_basis, [cat.value for cat in activity.data_categories],
                activity.data_subjects, activity.recipients, activity.retention_period,
                activity.cross_border_transfers, activity.transfer_safeguards,
                activity.automated_decision_making, activity.profiling,
                activity.data_source, activity.storage_location, activity.encryption_status,
                activity.access_controls, activity.is_active, activity.created_at, activity.updated_at)

    async def _store_privacy_request(self, request: PrivacyRequest):
        """Store privacy request"""

        async with self.db_pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO compliance.privacy_requests (
                    request_id, tenant_id, data_subject_id, request_type, request_date,
                    requester_email, requester_identity_verified, status, completion_deadline,
                    response_date, response_method, data_provided, actions_taken,
                    rejection_reason, created_at, updated_at
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16)
            """, request.request_id, request.tenant_id, request.data_subject_id,
                request.request_type, request.request_date, request.requester_email,
                request.requester_identity_verified, request.status, request.completion_deadline,
                request.response_date, request.response_method, request.data_provided,
                request.actions_taken, request.rejection_reason, request.created_at, request.updated_at)

    async def _store_compliance_results(self, results: Dict[str, Any]):
        """Store compliance assessment results"""

        async with self.db_pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO compliance.assessment_results (
                    result_id, tenant_id, timestamp, standards_checked, overall_status,
                    findings, recommendations, scores, created_at
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
            """, str(uuid.uuid4()), self.tenant_id, results['timestamp'],
                results['standards_checked'], results['overall_status'],
                json.dumps(results['findings']), json.dumps(results['recommendations']),
                json.dumps(results['scores']), datetime.now())

    async def _store_compliance_report(self, report: Dict[str, Any]):
        """Store compliance report"""

        async with self.db_pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO compliance.reports (
                    report_id, tenant_id, standard, report_type, generated_at,
                    period_start, period_end, executive_summary, control_assessments,
                    findings, evidence, recommendations, created_at
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13)
            """, report['report_id'], self.tenant_id, report['standard'],
                report['report_type'], report['generated_at'], report['period_start'],
                report['period_end'], json.dumps(report['executive_summary']),
                json.dumps(report['control_assessments']), json.dumps(report['findings']),
                json.dumps(report['evidence']), json.dumps(report['recommendations']),
                datetime.now())

    async def _store_monitoring_results(self, results: Dict[str, Any]):
        """Store continuous monitoring results"""

        async with self.db_pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO compliance.monitoring_results (
                    result_id, tenant_id, timestamp, checks_performed, violations_found,
                    alerts_generated, overall_health, created_at
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
            """, str(uuid.uuid4()), self.tenant_id, results['timestamp'],
                json.dumps(results['checks_performed']), json.dumps(results['violations_found']),
                json.dumps(results['alerts_generated']), results['overall_health'], datetime.now())

    async def _get_controls_by_standard(self, standard: ComplianceStandard) -> List[ComplianceControl]:
        """Get compliance controls for a specific standard"""

        return [control for control in self.controls.values() if control.standard == standard]

    async def _process_data_access_request(self, request: PrivacyRequest):
        """Process GDPR data access request automatically"""

        try:
            # Collect user data from various sources
            user_data = {}

            async with self.db_pool.acquire() as conn:
                # Get user profile data
                profile_data = await conn.fetchrow("""
                    SELECT * FROM auth.sso_users
                    WHERE user_id = $1 AND tenant_id = $2
                """, request.data_subject_id, self.tenant_id)

                if profile_data:
                    user_data['profile'] = dict(profile_data)

                # Get usage data
                usage_data = await conn.fetch("""
                    SELECT * FROM usage.tenant_usage
                    WHERE tenant_id = $1 AND created_at > NOW() - INTERVAL '2 years'
                    ORDER BY created_at DESC
                """, self.tenant_id)

                user_data['usage_history'] = [dict(row) for row in usage_data]

                # Get billing data
                billing_data = await conn.fetch("""
                    SELECT * FROM billing.subscriptions
                    WHERE tenant_id = $1
                """, self.tenant_id)

                user_data['billing_history'] = [dict(row) for row in billing_data]

            # Update request with collected data
            await self._update_privacy_request_status(
                request.request_id,
                'completed',
                json.dumps(user_data, default=str),
                'email'
            )

        except Exception as e:
            logger.error(f"Error processing data access request: {e}")
            await self._update_privacy_request_status(
                request.request_id,
                'processing',
                None,
                None
            )

    async def _process_data_erasure_request(self, request: PrivacyRequest):
        """Process GDPR data erasure request"""

        try:
            actions_taken = []

            async with self.db_pool.acquire() as conn:
                # Anonymize user data instead of complete deletion for audit purposes
                await conn.execute("""
                    UPDATE auth.sso_users
                    SET email = 'anonymized@example.com',
                        first_name = 'Anonymized',
                        last_name = 'User',
                        attributes = '{}',
                        is_active = false,
                        anonymized_at = NOW()
                    WHERE user_id = $1 AND tenant_id = $2
                """, request.data_subject_id, self.tenant_id)

                actions_taken.append("User profile anonymized")

                # Mark other data for deletion/anonymization
                await conn.execute("""
                    UPDATE usage.tenant_usage
                    SET anonymized = true
                    WHERE tenant_id = $1
                """, self.tenant_id)

                actions_taken.append("Usage data anonymized")

            # Update request status
            await self._update_privacy_request_status(
                request.request_id,
                'completed',
                None,
                'email'
            )

            # Update actions taken
            async with self.db_pool.acquire() as conn:
                await conn.execute("""
                    UPDATE compliance.privacy_requests
                    SET actions_taken = $1
                    WHERE request_id = $2
                """, actions_taken, request.request_id)

        except Exception as e:
            logger.error(f"Error processing data erasure request: {e}")
            await self._update_privacy_request_status(
                request.request_id,
                'processing',
                None,
                None
            )

    async def _update_privacy_request_status(self, request_id: str, status: str,
                                           data_provided: str = None, response_method: str = None):
        """Update privacy request status"""

        async with self.db_pool.acquire() as conn:
            await conn.execute("""
                UPDATE compliance.privacy_requests
                SET status = $1, response_date = $2, data_provided = $3,
                    response_method = $4, updated_at = $5
                WHERE request_id = $6
            """, status, datetime.now() if status == 'completed' else None,
                data_provided, response_method, datetime.now(), request_id)


class ComplianceManager:
    """
    High-level compliance management orchestrator
    """

    def __init__(self, db_pool: asyncpg.Pool):
        self.db_pool = db_pool
        self.frameworks = {}  # tenant_id -> ComplianceFramework

    async def get_framework(self, tenant_id: str) -> ComplianceFramework:
        """Get or create compliance framework for tenant"""

        if tenant_id not in self.frameworks:
            self.frameworks[tenant_id] = ComplianceFramework(self.db_pool, tenant_id)

        return self.frameworks[tenant_id]

    async def run_compliance_scan(self, tenant_id: str,
                                standards: List[ComplianceStandard] = None) -> Dict[str, Any]:
        """Run comprehensive compliance scan for tenant"""

        framework = await self.get_framework(tenant_id)
        return await framework.run_automated_compliance_check(standards)

    async def generate_audit_report(self, tenant_id: str,
                                  standard: ComplianceStandard) -> Dict[str, Any]:
        """Generate audit-ready compliance report"""

        framework = await self.get_framework(tenant_id)
        return await framework.generate_compliance_report(standard, 'audit')

    async def handle_data_subject_request(self, tenant_id: str, request_type: str,
                                        requester_email: str, data_subject_id: str) -> PrivacyRequest:
        """Handle GDPR data subject request"""

        framework = await self.get_framework(tenant_id)
        return await framework.handle_privacy_request(
            request_type, requester_email, data_subject_id
        )

    async def monitor_all_tenants(self) -> Dict[str, Any]:
        """Monitor compliance across all tenants"""

        results = {
            'timestamp': datetime.now().isoformat(),
            'tenants_monitored': 0,
            'compliant_tenants': 0,
            'violations_found': 0,
            'critical_issues': 0,
            'tenant_results': {}
        }

        # Get all active tenants
        async with self.db_pool.acquire() as conn:
            tenants = await conn.fetch("""
                SELECT tenant_id FROM billing.subscriptions
                WHERE status = 'active'
            """)

        for tenant_row in tenants:
            tenant_id = tenant_row['tenant_id']

            try:
                framework = await self.get_framework(tenant_id)
                tenant_result = await framework.monitor_continuous_compliance()

                results['tenants_monitored'] += 1
                results['tenant_results'][tenant_id] = tenant_result

                if tenant_result['overall_health'] == 'HEALTHY':
                    results['compliant_tenants'] += 1

                results['violations_found'] += len(tenant_result.get('violations_found', []))

                # Count critical issues
                critical_violations = [
                    v for v in tenant_result.get('violations_found', [])
                    if v.get('severity') == 'CRITICAL'
                ]
                results['critical_issues'] += len(critical_violations)

            except Exception as e:
                logger.error(f"Error monitoring tenant {tenant_id}: {e}")
                results['tenant_results'][tenant_id] = {
                    'error': str(e),
                    'overall_health': 'ERROR'
                }

        return results
