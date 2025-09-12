
"""
HubSpot Connector - AgentSystem Profit Machine
Comprehensive HubSpot integration for marketing and sales automation
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from enum import Enum
from dataclasses import dataclass, asdict
import asyncpg
import aiohttp
from pydantic import BaseModel, Field, validator
import uuid
import base64
from urllib.parse import urlencode, quote

logger = logging.getLogger(__name__)

class HubSpotObjectType(Enum):
    CONTACT = "contacts"
    COMPANY = "companies"
    DEAL = "deals"
    TICKET = "tickets"
    TASK = "tasks"
    MEETING = "meetings"
    CALL = "calls"
    EMAIL = "emails"
    NOTE = "notes"
    PRODUCT = "products"
    LINE_ITEM = "line_items"
    QUOTE = "quotes"

class HubSpotOperation(Enum):
    CREATE = "create"
    UPDATE = "update"
    DELETE = "delete"
    GET = "get"
    SEARCH = "search"
    BATCH_CREATE = "batch_create"
    BATCH_UPDATE = "batch_update"
    BATCH_READ = "batch_read"

class WorkflowTrigger(Enum):
    CONTACT_CREATED = "contact_created"
    CONTACT_UPDATED = "contact_updated"
    DEAL_CREATED = "deal_created"
    DEAL_STAGE_CHANGED = "deal_stage_changed"
    COMPANY_CREATED = "company_created"
    EMAIL_OPENED = "email_opened"
    EMAIL_CLICKED = "email_clicked"
    FORM_SUBMITTED = "form_submitted"
    DOCUMENT_PROCESSED = "document_processed"
    AI_ANALYSIS_COMPLETED = "ai_analysis_completed"
    SCHEDULED = "scheduled"

class SyncDirection(Enum):
    BIDIRECTIONAL = "bidirectional"
    HUBSPOT_TO_AGENTSYSTEM = "hs_to_as"
    AGENTSYSTEM_TO_HUBSPOT = "as_to_hs"

@dataclass
class HubSpotConnection:
    connection_id: str
    tenant_id: str
    connection_name: str
    account_id: str
    access_token: str
    refresh_token: str
    token_expires_at: datetime
    scopes: List[str]
    hub_domain: str
    hub_id: str
    is_active: bool
    last_sync: Optional[datetime]
    sync_errors: int
    created_at: datetime
    updated_at: datetime

@dataclass
class HubSpotWorkflow:
    workflow_id: str
    tenant_id: str
    connection_id: str
    name: str
    description: str
    trigger: WorkflowTrigger
    trigger_conditions: Dict[str, Any]
    hubspot_operations: List[Dict[str, Any]]
    agentsystem_operations: List[Dict[str, Any]]
    property_mappings: Dict[str, str]
    is_active: bool
    execution_count: int
    success_count: int
    last_execution: Optional[datetime]
    created_at: datetime
    updated_at: datetime

@dataclass
class HubSpotRecord:
    record_id: str
    object_type: HubSpotObjectType
    hubspot_id: str
    tenant_id: str
    properties: Dict[str, Any]
    last_modified: datetime
    sync_status: str
    sync_direction: SyncDirection
    created_at: datetime
    updated_at: datetime

class HubSpotConnector:
    """
    Comprehensive HubSpot integration with marketing and sales automation
    """

    BASE_URL = "https://api.hubapi.com"

    def __init__(self, db_pool: asyncpg.Pool, client_id: str, client_secret: str):
        self.db_pool = db_pool
        self.client_id = client_id
        self.client_secret = client_secret
        self.session = None
        self.connections_cache = {}

    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    def get_oauth_url(self, tenant_id: str, redirect_uri: str, scopes: List[str] = None) -> str:
        """Generate OAuth URL for HubSpot authorization"""

        if not scopes:
            scopes = [
                'contacts', 'companies', 'deals', 'tickets', 'automation',
                'forms', 'files', 'hubdb', 'integration-sync', 'timeline',
                'content', 'reports', 'social', 'business-intelligence'
            ]

        params = {
            'client_id': self.client_id,
            'redirect_uri': redirect_uri,
            'scope': ' '.join(scopes),
            'state': tenant_id  # Use tenant_id as state parameter
        }

        return f"https://app.hubspot.com/oauth/authorize?{urlencode(params)}"

    async def exchange_code_for_tokens(self, tenant_id: str, code: str,
                                     redirect_uri: str) -> HubSpotConnection:
        """Exchange authorization code for access tokens"""

        token_url = "https://api.hubapi.com/oauth/v1/token"

        payload = {
            'grant_type': 'authorization_code',
            'client_id': self.client_id,
            'client_secret': self.client_secret,
            'redirect_uri': redirect_uri,
            'code': code
        }

        async with self.session.post(token_url, data=payload) as response:
            if response.status == 200:
                token_data = await response.json()

                # Get account info
                account_info = await self._get_account_info(token_data['access_token'])

                connection_id = str(uuid.uuid4())

                connection = HubSpotConnection(
                    connection_id=connection_id,
                    tenant_id=tenant_id,
                    connection_name=account_info.get('companyName', 'HubSpot Account'),
                    account_id=str(account_info.get('portalId', '')),
                    access_token=token_data['access_token'],
                    refresh_token=token_data['refresh_token'],
                    token_expires_at=datetime.now() + timedelta(seconds=token_data['expires_in']),
                    scopes=token_data.get('scope', '').split(),
                    hub_domain=account_info.get('domain', ''),
                    hub_id=str(account_info.get('hubId', '')),
                    is_active=True,
                    last_sync=None,
                    sync_errors=0,
                    created_at=datetime.now(),
                    updated_at=datetime.now()
                )

                await self._store_connection(connection)
                self.connections_cache[connection_id] = connection

                return connection
            else:
                error_data = await response.json()
                raise ValueError(f"Token exchange failed: {error_data}")

    async def _get_account_info(self, access_token: str) -> Dict[str, Any]:
        """Get HubSpot account information"""

        headers = {'Authorization': f'Bearer {access_token}'}

        async with self.session.get(f"{self.BASE_URL}/account-info/v3/details", headers=headers) as response:
            if response.status == 200:
                return await response.json()
            else:
                return {}

    async def refresh_access_token(self, connection: HubSpotConnection) -> bool:
        """Refresh HubSpot access token"""

        token_url = "https://api.hubapi.com/oauth/v1/token"

        payload = {
            'grant_type': 'refresh_token',
            'client_id': self.client_id,
            'client_secret': self.client_secret,
            'refresh_token': connection.refresh_token
        }

        try:
            async with self.session.post(token_url, data=payload) as response:
                if response.status == 200:
                    token_data = await response.json()

                    connection.access_token = token_data['access_token']
                    connection.refresh_token = token_data['refresh_token']
                    connection.token_expires_at = datetime.now() + timedelta(seconds=token_data['expires_in'])
                    connection.updated_at = datetime.now()

                    await self._update_connection(connection)
                    return True

        except Exception as e:
            logger.error(f"Token refresh failed: {e}")

        return False

    async def create_workflow(self, tenant_id: str, workflow_config: Dict[str, Any]) -> HubSpotWorkflow:
        """Create a new HubSpot automation workflow"""

        workflow_id = str(uuid.uuid4())

        workflow = HubSpotWorkflow(
            workflow_id=workflow_id,
            tenant_id=tenant_id,
            connection_id=workflow_config['connection_id'],
            name=workflow_config['name'],
            description=workflow_config.get('description', ''),
            trigger=WorkflowTrigger(workflow_config['trigger']),
            trigger_conditions=workflow_config.get('trigger_conditions', {}),
            hubspot_operations=workflow_config.get('hubspot_operations', []),
            agentsystem_operations=workflow_config.get('agentsystem_operations', []),
            property_mappings=workflow_config.get('property_mappings', {}),
            is_active=True,
            execution_count=0,
            success_count=0,
            last_execution=None,
            created_at=datetime.now(),
            updated_at=datetime.now()
        )

        await self._store_workflow(workflow)
        return workflow

    async def execute_workflow(self, workflow_id: str, trigger_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a HubSpot automation workflow"""

        workflow = await self._get_workflow(workflow_id)
        if not workflow or not workflow.is_active:
            return {'success': False, 'error': 'Workflow not found or inactive'}

        connection = await self._get_connection(workflow.connection_id)
        if not connection or not connection.is_active:
            return {'success': False, 'error': 'HubSpot connection not available'}

        # Ensure valid access token
        if not await self._ensure_valid_token(connection):
            return {'success': False, 'error': 'Failed to authenticate with HubSpot'}

        execution_result = {
            'workflow_id': workflow_id,
            'execution_time': datetime.now(),
            'operations_completed': 0,
            'operations_failed': 0,
            'results': [],
            'success': True
        }

        try:
            # Execute HubSpot operations
            for operation in workflow.hubspot_operations:
                result = await self._execute_hubspot_operation(connection, operation, trigger_data, workflow.property_mappings)
                execution_result['results'].append(result)

                if result['success']:
                    execution_result['operations_completed'] += 1
                else:
                    execution_result['operations_failed'] += 1

            # Execute AgentSystem operations
            for operation in workflow.agentsystem_operations:
                result = await self._execute_agentsystem_operation(operation, trigger_data, workflow.property_mappings)
                execution_result['results'].append(result)

                if result['success']:
                    execution_result['operations_completed'] += 1
                else:
                    execution_result['operations_failed'] += 1

            # Update workflow statistics
            workflow.execution_count += 1
            if execution_result['operations_failed'] == 0:
                workflow.success_count += 1
                execution_result['success'] = True
            else:
                execution_result['success'] = False

            workflow.last_execution = datetime.now()
            await self._update_workflow(workflow)

        except Exception as e:
            logger.error(f"Workflow execution failed: {e}")
            execution_result['success'] = False
            execution_result['error'] = str(e)

        return execution_result

    async def _execute_hubspot_operation(self, connection: HubSpotConnection,
                                       operation: Dict[str, Any], trigger_data: Dict[str, Any],
                                       property_mappings: Dict[str, str]) -> Dict[str, Any]:
        """Execute a single HubSpot operation"""

        operation_type = HubSpotOperation(operation['type'])
        object_type = HubSpotObjectType(operation['object_type'])

        try:
            if operation_type == HubSpotOperation.CREATE:
                return await self._hubspot_create_record(connection, object_type, operation, trigger_data, property_mappings)

            elif operation_type == HubSpotOperation.UPDATE:
                return await self._hubspot_update_record(connection, object_type, operation, trigger_data, property_mappings)

            elif operation_type == HubSpotOperation.SEARCH:
                return await self._hubspot_search_records(connection, object_type, operation, trigger_data, property_mappings)

            elif operation_type == HubSpotOperation.GET:
                return await self._hubspot_get_record(connection, object_type, operation, trigger_data, property_mappings)

            else:
                return {'success': False, 'error': f'Unsupported operation type: {operation_type}'}

        except Exception as e:
            return {'success': False, 'error': str(e)}

    async def _hubspot_create_record(self, connection: HubSpotConnection, object_type: HubSpotObjectType,
                                   operation: Dict[str, Any], trigger_data: Dict[str, Any],
                                   property_mappings: Dict[str, str]) -> Dict[str, Any]:
        """Create a new record in HubSpot"""

        # Prepare record properties
        properties = {}
        for prop, value in operation.get('properties', {}).items():
            # Apply property mappings
            mapped_prop = property_mappings.get(prop, prop)

            # Process value (could be static or from trigger data)
            if isinstance(value, str) and value.startswith('{{') and value.endswith('}}'):
                # Extract value from trigger data
                trigger_field = value[2:-2]
                properties[mapped_prop] = trigger_data.get(trigger_field)
            else:
                properties[mapped_prop] = value

        # Make API call
        api_url = f"{self.BASE_URL}/crm/v3/objects/{object_type.value}"
        headers = {
            'Authorization': f'Bearer {connection.access_token}',
            'Content-Type': 'application/json'
        }

        payload = {'properties': properties}

        async with self.session.post(api_url, json=payload, headers=headers) as response:
            response_data = await response.json()

            if response.status == 201:
                return {
                    'success': True,
                    'operation': 'create',
                    'object_type': object_type.value,
                    'hubspot_id': response_data['id'],
                    'properties': properties
                }
            else:
                return {
                    'success': False,
                    'operation': 'create',
                    'object_type': object_type.value,
                    'error': response_data
                }

    async def _hubspot_update_record(self, connection: HubSpotConnection, object_type: HubSpotObjectType,
                                   operation: Dict[str, Any], trigger_data: Dict[str, Any],
                                   property_mappings: Dict[str, str]) -> Dict[str, Any]:
        """Update an existing record in HubSpot"""

        record_id = operation.get('record_id') or trigger_data.get('hubspot_id')
        if not record_id:
            return {'success': False, 'error': 'No record ID provided for update'}

        # Prepare update properties
        properties = {}
        for prop, value in operation.get('properties', {}).items():
            mapped_prop = property_mappings.get(prop, prop)

            if isinstance(value, str) and value.startswith('{{') and value.endswith('}}'):
                trigger_field = value[2:-2]
                properties[mapped_prop] = trigger_data.get(trigger_field)
            else:
                properties[mapped_prop] = value

        # Make API call
        api_url = f"{self.BASE_URL}/crm/v3/objects/{object_type.value}/{record_id}"
        headers = {
            'Authorization': f'Bearer {connection.access_token}',
            'Content-Type': 'application/json'
        }

        payload = {'properties': properties}

        async with self.session.patch(api_url, json=payload, headers=headers) as response:
            if response.status == 200:
                response_data = await response.json()
                return {
                    'success': True,
                    'operation': 'update',
                    'object_type': object_type.value,
                    'hubspot_id': record_id,
                    'properties': properties
                }
            else:
                response_data = await response.json()
                return {
                    'success': False,
                    'operation': 'update',
                    'object_type': object_type.value,
                    'error': response_data
                }

    async def _hubspot_search_records(self, connection: HubSpotConnection, object_type: HubSpotObjectType,
                                    operation: Dict[str, Any], trigger_data: Dict[str, Any],
                                    property_mappings: Dict[str, str]) -> Dict[str, Any]:
        """Search records in HubSpot"""

        search_criteria = operation.get('search_criteria', {})

        # Process search criteria with trigger data
        processed_criteria = {}
        for key, value in search_criteria.items():
            if isinstance(value, str) and value.startswith('{{') and value.endswith('}}'):
                trigger_field = value[2:-2]
                processed_criteria[key] = trigger_data.get(trigger_field)
            else:
                processed_criteria[key] = value

        # Build search payload
        search_payload = {
            'filterGroups': [
                {
                    'filters': [
                        {
                            'propertyName': prop,
                            'operator': 'EQ',
                            'value': value
                        }
                        for prop, value in processed_criteria.items()
                    ]
                }
            ],
            'properties': operation.get('properties_to_return', ['id']),
            'limit': operation.get('limit', 100)
        }

        # Make API call
        api_url = f"{self.BASE_URL}/crm/v3/objects/{object_type.value}/search"
        headers = {
            'Authorization': f'Bearer {connection.access_token}',
            'Content-Type': 'application/json'
        }

        async with self.session.post(api_url, json=search_payload, headers=headers) as response:
            response_data = await response.json()

            if response.status == 200:
                return {
                    'success': True,
                    'operation': 'search',
                    'object_type': object_type.value,
                    'total': response_data.get('total', 0),
                    'results': response_data.get('results', [])
                }
            else:
                return {
                    'success': False,
                    'operation': 'search',
                    'object_type': object_type.value,
                    'error': response_data
                }

    async def _execute_agentsystem_operation(self, operation: Dict[str, Any],
                                           trigger_data: Dict[str, Any],
                                           property_mappings: Dict[str, str]) -> Dict[str, Any]:
        """Execute an AgentSystem operation as part of workflow"""

        operation_type = operation.get('type')

        try:
            if operation_type == 'ai_analysis':
                return await self._execute_ai_analysis(operation, trigger_data)

            elif operation_type == 'lead_scoring':
                return await self._execute_lead_scoring(operation, trigger_data)

            elif operation_type == 'content_generation':
                return await self._execute_content_generation(operation, trigger_data)

            elif operation_type == 'email_personalization':
                return await self._execute_email_personalization(operation, trigger_data)

            else:
                return {'success': False, 'error': f'Unsupported AgentSystem operation: {operation_type}'}

        except Exception as e:
            return {'success': False, 'error': str(e)}

    async def _execute_lead_scoring(self, operation: Dict[str, Any], trigger_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute AI-powered lead scoring"""

        # Extract lead data
        lead_data = operation.get('input_data', trigger_data)

        # AI scoring algorithm (would integrate with actual AI models)
        base_score = 50

        # Company size scoring
        if 'num_employees' in lead_data:
            employees = int(lead_data.get('num_employees', 0))
            if employees > 1000:
                base_score += 30
            elif employees > 100:
                base_score += 20
            elif employees > 10:
                base_score += 10

        # Industry scoring
        high_value_industries = ['technology', 'finance', 'healthcare', 'manufacturing']
        industry = str(lead_data.get('industry', '')).lower()
        if any(hvi in industry for hvi in high_value_industries):
            base_score += 15

        # Engagement scoring
        if lead_data.get('email_opened'):
            base_score += 10
        if lead_data.get('content_downloaded'):
            base_score += 15
        if lead_data.get('demo_requested'):
            base_score += 25

        # Website behavior
        page_views = int(lead_data.get('page_views', 0))
        if page_views > 10:
            base_score += 10

        # Ensure score is within bounds
        final_score = min(100, max(0, base_score))

        # Determine grade
        if final_score >= 80:
            grade = 'A'
        elif final_score >= 60:
            grade = 'B'
        elif final_score >= 40:
            grade = 'C'
        else:
            grade = 'D'

        return {
            'success': True,
            'operation': 'lead_scoring',
            'result': {
                'score': final_score,
                'grade': grade,
                'factors': {
                    'company_size': 'high' if lead_data.get('num_employees', 0) > 100 else 'low',
                    'industry_match': industry in high_value_industries,
                    'engagement_level': 'high' if lead_data.get('demo_requested') else 'medium'
                }
            }
        }

    async def sync_data(self, connection_id: str, sync_config: Dict[str, Any]) -> Dict[str, Any]:
        """Synchronize data between HubSpot and AgentSystem"""

        connection = await self._get_connection(connection_id)
        if not connection:
            return {'success': False, 'error': 'Connection not found'}

        if not await self._ensure_valid_token(connection):
            return {'success': False, 'error': 'Authentication failed'}

        sync_result = {
            'sync_started': datetime.now(),
            'objects_synced': 0,
            'records_processed': 0,
            'records_created': 0,
            'records_updated': 0,
            'records_failed': 0,
            'errors': []
        }

        try:
            # Get objects to sync
            objects_to_sync = sync_config.get('objects', ['contacts', 'companies', 'deals'])

            for object_type in objects_to_sync:
                object_result = await self._sync_object_type(connection, object_type, sync_config)

                sync_result['records_processed'] += object_result['processed']
                sync_result['records_created'] += object_result['created']
                sync_result['records_updated'] += object_result['updated']
                sync_result['records_failed'] += object_result['failed']
                sync_result['errors'].extend(object_result['errors'])
                sync_result['objects_synced'] += 1

            # Update connection last sync time
            connection.last_sync = datetime.now()
            await self._update_connection(connection)

            sync_result['success'] = True
            sync_result['sync_completed'] = datetime.now()

        except Exception as e:
            sync_result['success'] = False
            sync_result['error'] = str(e)

        return sync_result

    async def _sync_object_type(self, connection: HubSpotConnection, object_type: str,
                              sync_config: Dict[str, Any]) -> Dict[str, Any]:
        """Sync a specific object type"""

        result = {
            'object_type': object_type,
            'processed': 0,
            'created': 0,
            'updated': 0,
            'failed': 0,
            'errors': []
        }

        try:
            # Get recent records from HubSpot
            last_sync = connection.last_sync or (datetime.now() - timedelta(days=30))
            last_sync_ms = int(last_sync.timestamp() * 1000)

            # Build API URL with query parameters
            properties = sync_config.get('properties', ['id', 'createdate', 'lastmodifieddate'])
            api_url = f"{self.BASE_URL}/crm/v3/objects/{object_type}"

            params = {
                'properties': ','.join(properties),
                'limit': sync_config.get('limit', 100),
                'filterGroups': json.dumps([{
                    'filters': [{
                        'propertyName': 'lastmodifieddate',
                        'operator': 'GTE',
                        'value': last_sync_ms
                    }]
                }])
            }

            headers = {'Authorization': f'Bearer {connection.access_token}'}

            async with self.session.get(api_url, params=params, headers=headers) as response:
                if response.status == 200:
                    response_data = await response.json()

                    for record in response_data.get('results', []):
                        try:
                            # Process each record
                            await self._process_sync_record(connection, object_type, record, sync_config)
                            result['processed'] += 1

                            # Check if record exists in our system
                            existing_record = await self._get_synced_record(connection.tenant_id, record['id'])
                            if existing_record:
                                result['updated'] += 1
                            else:
                                result['created'] += 1

                        except Exception as e:
                            result['failed'] += 1
                            result['errors'].append(f"Record {record['id']}: {str(e)}")

                else:
                    result['errors'].append(f"API call failed with status {response.status}")

        except Exception as e:
            result['errors'].append(str(e))

        return result

    async def _process_sync_record(self, connection: HubSpotConnection, object_type: str,
                                 hs_record: Dict[str, Any], sync_config: Dict[str, Any]):
        """Process a single record for synchronization"""

        # Create or update AgentSystem record
        record = HubSpotRecord(
            record_id=str(uuid.uuid4()),
            object_type=HubSpotObjectType(object_type),
            hubspot_id=hs_record['id'],
            tenant_id=connection.tenant_id,
            properties=hs_record.get('properties', {}),
            last_modified=datetime.now(),  # Would parse from HubSpot lastmodifieddate
            sync_status='synced',
            sync_direction=SyncDirection.HUBSPOT_TO_AGENTSYSTEM,
            created_at=datetime.now(),
            updated_at=datetime.now()
        )

        await self._store_synced_record(record)

    async def _ensure_valid_token(self, connection: HubSpotConnection) -> bool:
        """Ensure the connection has a valid access token"""

        if not connection.access_token:
            return False

        # Check if token is expired (with 5 minute buffer)
        if datetime.now() >= (connection.token_expires_at - timedelta(minutes=5)):
            return await self.refresh_access_token(connection)

        return True

    # Pre-built workflow templates
    def get_workflow_templates(self) -> List[Dict[str, Any]]:
        """Get pre-built HubSpot workflow templates"""

        return [
            {
                'template_id': 'contact_ai_enrichment',
                'name': 'AI Contact Enrichment',
                'description': 'Automatically enrich new contacts with AI-generated insights',
                'trigger': WorkflowTrigger.CONTACT_CREATED,
                'hubspot_operations': [
                    {
                        'type': 'update',
                        'object_type': 'contacts',
                        'record_id': '{{contact_id}}',
                        'properties': {
                            'ai_lead_score': '{{ai_score}}',
                            'ai_lead_grade': '{{ai_grade}}',
                            'lead_qualification_status': '{{qualification_status}}',
                            'industry_match': '{{industry_match}}'
                        }
                    }
                ],
                'agentsystem_operations': [
                    {
                        'type': 'lead_scoring',
                        'input_data': {
                            'email': '{{email}}',
                            'company': '{{company}}',
                            'jobtitle': '{{jobtitle}}',
                            'industry': '{{industry}}',
                            'num_employees': '{{num_employees}}'
                        }
                    }
                ]
            },
            {
                'template_id': 'deal_ai_insights',
                'name': 'Deal AI Insights & Win Probability',
                'description': 'Generate AI insights and win probability for new deals',
                'trigger': WorkflowTrigger.DEAL_CREATED,
                'hubspot_operations': [
                    {
                        'type': 'update',
                        'object_type': 'deals',
                        'record_id': '{{deal_id}}',
                        'properties': {
                            'ai_win_probability': '{{win_probability}}',
                            'risk_factors': '{{risk_factors}}',
                            'recommended_actions': '{{recommended_actions}}',
                            'competitive_analysis': '{{competitive_analysis}}'
                        }
                    }
                ],
                'agentsystem_operations': [
                    {
                        'type': 'ai_analysis',
                        'analysis_type': 'deal_analysis',
                        'input_data': {
                            'deal_stage': '{{dealstage}}',
                            'amount': '{{amount}}',
                            'close_date': '{{closedate}}',
                            'deal_type': '{{dealtype}}',
                            'lead_source': '{{hs_analytics_source}}'
                        }
                    }
                ]
            },
            {
                'template_id': 'email_personalization',
                'name': 'AI Email Personalization',
                'description': 'Personalize email content based on contact data and behavior',
                'trigger': WorkflowTrigger.EMAIL_OPENED,
                'hubspot_operations': [
                    {
                        'type': 'search',
                        'object_type': 'contacts',
                        'search_criteria': {
                            'email': '{{contact_email}}'
                        },
                        'properties_to_return': ['firstname', 'lastname', 'company', 'industry', 'jobtitle']
                    }
                ],
                'agentsystem_operations': [
                    {
                        'type': 'content_generation',
                        'content_type': 'email_followup',
                        'personalization_data': {
                            'name': '{{firstname}}',
                            'company': '{{company}}',
                            'industry': '{{industry}}',
                            'role': '{{jobtitle}}',
                            'previous_engagement': '{{email_opened}}'
                        }
                    }
                ]
            }
        ]

    # Database operations
    async def _store_connection(self, connection: HubSpotConnection):
        """Store HubSpot connection in database"""

        async with self.db_pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO integrations.hubspot_connections (
                    connection_id, tenant_id, connection_name, account_id,
                    access_token, refresh_token, token_expires_at, scopes,
                    hub_domain, hub_id, is_active, created_at, updated_at
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13)
            """,
                connection.connection_id, connection.tenant_id, connection.connection_name,
                connection.account_id, connection.access_token, connection.refresh_token,
                connection.token_expires_at, json.dumps(connection.scopes),
                connection.hub_domain, connection.hub_id, connection.is_active,
                connection.created_at, connection.updated_at
            )

    async def _update_connection(self, connection: HubSpotConnection):
        """Update HubSpot connection in database"""

        async with self.db_pool.acquire() as conn:
            await conn.execute("""
                UPDATE integrations.hubspot_connections SET
                    connection_name = $3, access_token = $4, refresh_token = $5,
                    token_expires_at = $6, is_active = $7, last_sync = $8,
                    sync_errors = $9, updated_at = $10
                WHERE connection_id = $1 AND tenant_id = $2
            """,
                connection.connection_id, connection.tenant_id, connection.connection_name,
                connection.access_token, connection.refresh_token, connection.token_expires_at,
                connection.is_active, connection.last_sync, connection.sync_errors,
                connection.updated_at
            )

    async def _get_connection(self, connection_id: str) -> Optional[HubSpotConnection]:
        """Get HubSpot connection from database"""

        if connection_id in self.connections_cache:
            return self.connections_cache[connection_id]

        async with self.db_pool.acquire() as conn:
            row = await conn.fetchrow("""
                SELECT * FROM integrations.hubspot_connections
                WHERE connection_id = $1
            """, connection_id)

            if row:
                connection = HubSpotConnection(
                    connection_id=row['connection_id'],
                    tenant_id=row['tenant_id'],
                    connection_name=row['connection_name'],
                    account_id=row['account_id'],
                    access_token=row['access_token'],
                    refresh_token=row['refresh_token'],
                    token_expires_at=row['token_expires_at'],
                    scopes=json.loads(row['scopes']),
                    hub_domain=row['hub_domain'],
                    hub_id=row['hub_id'],
                    is_active=row['is_active'],
                    last_sync=row['last_sync'],
                    sync_errors=row['sync_errors'],
                    created_at=row['created_at'],
                    updated_at=row['updated_at']
                )

                self.connections_cache[connection_id] = connection
                return connection

        return None

    async def _store_workflow(self, workflow: HubSpotWorkflow):
        """Store HubSpot workflow in database"""

        async with self.db_pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO integrations.hubspot_workflows (
                    workflow_id, tenant_id, connection_id, name, description,
                    trigger_type, trigger_conditions, hubspot_operations,
                    agentsystem_operations, property_mappings, is_active,
                    created_at, updated_at
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13)
            """,
                workflow.workflow_id, workflow.tenant_id, workflow.connection_id,
                workflow.name, workflow.description, workflow.trigger.value,
                json.dumps(workflow.trigger_conditions), json.dumps(workflow.hubspot_operations),
                json.dumps(workflow.agentsystem_operations), json.dumps(workflow.property_mappings),
                workflow.is_active, workflow.created_at, workflow.updated_at
            )

    async def _update_workflow(self, workflow: HubSpotWorkflow):
        """Update workflow statistics"""

        async with self.db_pool.acquire() as conn:
            await conn.execute("""
                UPDATE integrations.hubspot_workflows SET
                    execution_count = $3, success_count = $4, last_execution = $5,
                    updated_at = $6
                WHERE workflow_id = $1 AND tenant_id = $2
            """,
                workflow.workflow_id, workflow.tenant_id, workflow.execution_count,
                workflow.success_count, workflow.last_execution, workflow.updated_at
            )

    async def _get_workflow(self, workflow_id: str) -> Optional[HubSpotWorkflow]:
        """Get workflow from database"""

        async with self.db_pool.acquire() as conn:
            row = await conn.fetchrow("""
                SELECT * FROM integrations.hubspot_workflows
                WHERE workflow_id = $1
            """, workflow_id)

            if row:
                return HubSpotWorkflow(
                    workflow_id=row['workflow_id'],
                    tenant_id=row['tenant_id'],
                    connection_id=row['connection_id'],
                    name=row['name'],
                    description=row['description'],
                    trigger=WorkflowTrigger(row['trigger_type']),
                    trigger_conditions=json.loads(row['trigger_conditions']),
                    hubspot_operations=json.loads(row['hubspot_operations']),
                    agentsystem_operations=json.loads(row['agentsystem_operations']),
                    property_mappings=json.loads(row['property_mappings']),
                    is_active=row['is_active'],
                    execution_count=row['execution_count'],
                    success_count=row['success_count'],
                    last_execution=row['last_execution'],
                    created_at=row['created_at'],
                    updated_at=row['updated_at']
                )

        return None

    async def _store_synced_record(self, record: HubSpotRecord):
        """Store synced HubSpot record"""

        async with self.db_pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO integrations.hubspot_records (
                    record_id, object_type, hubspot_id, tenant_id, properties,
                    last_modified, sync_status, sync_direction, created_at, updated_at
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
                ON CONFLICT (tenant_id, hubspot_id) DO UPDATE SET
                    properties = EXCLUDED.properties,
                    last_modified = EXCLUDED.last_modified,
                    sync_status = EXCLUDED.sync_status,
                    updated_at = EXCLUDED.updated_at
            """,
                record.record_id, record.object_type.value, record.hubspot_id,
                record.tenant_id, json.dumps(record.properties), record.last_modified,
                record.sync_status, record.sync_direction.value, record.created_at,
                record.updated_at
            )

    async def _get_synced_record(self, tenant_id: str, hubspot_id: str) -> Optional[HubSpotRecord]:
        """Get synced record by HubSpot ID"""

        async with self.db_pool.acquire() as conn:
            row = await conn.fetchrow("""
                SELECT * FROM integrations.hubspot_records
                WHERE tenant_id = $1 AND hubspot_id = $2
            """, tenant_id, hubspot_id)

            if row:
                return HubSpotRecord(
                    record_id=row['record_id'],
                    object_type=HubSpotObjectType(row['object_type']),
                    hubspot_id=row['hubspot_id'],
                    tenant_id=row['tenant_id'],
                    properties=json.loads(row['properties']),
                    last_modified=row['last_modified'],
                    sync_status=row['sync_status'],
                    sync_direction=SyncDirection(row['sync_direction']),
                    created_at=row['created_at'],
                    updated_at=row['updated_at']
                )

        return None

# Database schema for HubSpot integration
HUBSPOT_SCHEMA_SQL = """
-- HubSpot connections table
CREATE TABLE IF NOT EXISTS integrations.hubspot_connections (
    connection_id UUID PRIMARY KEY,
    tenant_id UUID REFERENCES tenant_management.tenants(id) ON DELETE CASCADE,
    connection_name VARCHAR(500) NOT NULL,
    account_id VARCHAR(100) NOT NULL,
    access_token TEXT NOT NULL,
    refresh_token TEXT NOT NULL,
    token_expires_at TIMESTAMP WITH TIME ZONE NOT NULL,
    scopes JSONB DEFAULT '[]',
    hub_domain VARCHAR(500),
    hub_id VARCHAR(100),
    is_active BOOLEAN DEFAULT TRUE,
    last_sync TIMESTAMP WITH TIME ZONE,
    sync_errors INTEGER DEFAULT 0,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- HubSpot workflows table
CREATE TABLE IF NOT EXISTS integrations.hubspot_workflows (
    workflow_id UUID PRIMARY KEY,
    tenant_id UUID REFERENCES tenant_management.tenants(id) ON DELETE CASCADE,
    connection_id UUID REFERENCES integrations.hubspot_connections(connection_id) ON DELETE CASCADE,
    name VARCHAR(500) NOT NULL,
    description TEXT,
    trigger_type VARCHAR(100) NOT NULL,
    trigger_conditions JSONB DEFAULT '{}',
    hubspot_operations JSONB DEFAULT '[]',
    agentsystem_operations JSONB DEFAULT '[]',
    property_mappings JSONB DEFAULT '{}',
    is_active BOOLEAN DEFAULT TRUE,
    execution_count INTEGER DEFAULT 0,
    success_count INTEGER DEFAULT 0,
    last_execution TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- HubSpot synced records table
CREATE TABLE IF NOT EXISTS integrations.hubspot_records (
    record_id UUID PRIMARY KEY,
    object_type VARCHAR(100) NOT NULL,
    hubspot_id VARCHAR(50) NOT NULL,
    tenant_id UUID REFERENCES tenant_management.tenants(id) ON DELETE CASCADE,
    properties JSONB NOT NULL,
    last_modified TIMESTAMP WITH TIME ZONE NOT NULL,
    sync_status VARCHAR(50) DEFAULT 'synced',
    sync_direction VARCHAR(50) NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(tenant_id, hubspot_id)
);

-- Workflow execution log table
CREATE TABLE IF NOT EXISTS integrations.hubspot_execution_log (
    execution_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    workflow_id UUID REFERENCES integrations.hubspot_workflows(workflow_id) ON DELETE CASCADE,
    tenant_id UUID REFERENCES tenant_management.tenants(id) ON DELETE CASCADE,
    trigger_data JSONB NOT NULL,
    execution_result JSONB NOT NULL,
    success BOOLEAN NOT NULL,
    execution_time_ms INTEGER,
    executed_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_hubspot_connections_tenant
ON integrations.hubspot_connections(tenant_id, is_active);

CREATE INDEX IF NOT EXISTS idx_hubspot_workflows_tenant_trigger
ON integrations.hubspot_workflows(tenant_id, trigger_type, is_active);

CREATE INDEX IF NOT EXISTS idx_hubspot_records_tenant_type
ON integrations.hubspot_records(tenant_id, object_type);

CREATE INDEX IF NOT EXISTS idx_hubspot_records_hs_id
ON integrations.hubspot_records(hubspot_id);

CREATE INDEX IF NOT EXISTS idx_hubspot_execution_log_workflow
ON integrations.hubspot_execution_log(workflow_id, executed_at DESC);

-- Views for analytics
CREATE OR REPLACE VIEW integrations.hubspot_analytics AS
SELECT
    hc.tenant_id,
    hc.connection_name,
    hc.is_active as connection_active,
    hc.last_sync,
    COUNT(hw.workflow_id) as total_workflows,
    COUNT(CASE WHEN hw.is_active THEN 1 END) as active_workflows,
    SUM(hw.execution_count) as total_executions,
    SUM(hw.success_count) as successful_executions,
    COUNT(hr.record_id) as synced_records,
    COUNT(CASE WHEN hr.sync_status = 'synced' THEN 1 END) as successfully_synced
FROM integrations.hubspot_connections hc
LEFT JOIN integrations.hubspot_workflows hw ON hc.connection_id = hw.connection_id
LEFT JOIN integrations.hubspot_records hr ON hc.tenant_id = hr.tenant_id
GROUP BY hc.tenant_id, hc.connection_name, hc.is_active, hc.last_sync;

-- Function to clean up old execution logs
CREATE OR REPLACE FUNCTION integrations.cleanup_hubspot_logs()
RETURNS void AS $$
BEGIN
    DELETE FROM integrations.hubspot_execution_log
    WHERE executed_at < NOW() - INTERVAL '90 days';
END;
$$ LANGUAGE plpgsql;

-- Comments for documentation
COMMENT ON TABLE integrations.hubspot_connections IS 'HubSpot portal connections with OAuth tokens';
COMMENT ON TABLE integrations.hubspot_workflows IS 'Automated workflows between HubSpot and AgentSystem';
COMMENT ON TABLE integrations.hubspot_records IS 'Synced HubSpot records with change tracking';
COMMENT ON TABLE integrations.hubspot_execution_log IS 'Log of workflow executions for debugging and analytics';
"""

# Pydantic models for API
class HubSpotOAuthRequest(BaseModel):
    tenant_id: str = Field(..., description="Tenant ID")
    redirect_uri: str = Field(..., description="OAuth redirect URI")
    scopes: Optional[List[str]] = Field(None, description="Requested scopes")

class HubSpotTokenExchangeRequest(BaseModel):
    tenant_id: str = Field(..., description="Tenant ID")
    code: str = Field(..., description="Authorization code")
    redirect_uri: str = Field(..., description="OAuth redirect URI")

class HubSpotWorkflowRequest(BaseModel):
    connection_id: str = Field(..., description="HubSpot connection ID")
    name: str = Field(..., description="Workflow name")
    description: Optional[str] = Field(None, description="Workflow description")
    trigger: str = Field(..., description="Workflow trigger type")
    trigger_conditions: Optional[Dict[str, Any]] = Field(None, description="Trigger conditions")
    hubspot_operations: List[Dict[str, Any]] = Field(..., description="HubSpot operations")
    agentsystem_operations: Optional[List[Dict[str, Any]]] = Field(None, description="AgentSystem operations")
    property_mappings: Optional[Dict[str, str]] = Field(None, description="Property mappings")

class HubSpotSyncRequest(BaseModel):
    connection_id: str = Field(..., description="Connection ID to sync")
    objects: List[str] = Field(["contacts", "companies", "deals"], description="Objects to sync")
    properties: List[str] = Field(["id", "createdate", "lastmodifieddate"], description="Properties to include")
    limit: int = Field(100, description="Maximum records per object")
    incremental: bool = Field(True, description="Only sync changed records")

# Export main classes
__all__ = [
    'HubSpotConnector', 'HubSpotConnection', 'HubSpotWorkflow', 'HubSpotRecord',
    'HubSpotObjectType', 'HubSpotOperation', 'WorkflowTrigger', 'SyncDirection',
    'HubSpotOAuthRequest', 'HubSpotTokenExchangeRequest', 'HubSpotWorkflowRequest', 'HubSpotSyncRequest',
    'HUBSPOT_SCHEMA_SQL'
]
