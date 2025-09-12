
"""
Salesforce Connector - AgentSystem Profit Machine
Comprehensive Salesforce integration with automation workflows for enterprise CRM
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
from urllib.parse import urlencode
import jwt

logger = logging.getLogger(__name__)

class SalesforceObjectType(Enum):
    LEAD = "Lead"
    CONTACT = "Contact"
    ACCOUNT = "Account"
    OPPORTUNITY = "Opportunity"
    CASE = "Case"
    TASK = "Task"
    EVENT = "Event"
    CAMPAIGN = "Campaign"
    CUSTOM_OBJECT = "Custom"

class SalesforceOperation(Enum):
    CREATE = "create"
    UPDATE = "update"
    DELETE = "delete"
    QUERY = "query"
    UPSERT = "upsert"
    BULK_CREATE = "bulk_create"
    BULK_UPDATE = "bulk_update"

class WorkflowTrigger(Enum):
    LEAD_CREATED = "lead_created"
    OPPORTUNITY_STAGE_CHANGED = "opportunity_stage_changed"
    CASE_ESCALATED = "case_escalated"
    DOCUMENT_PROCESSED = "document_processed"
    AI_ANALYSIS_COMPLETED = "ai_analysis_completed"
    SCHEDULED = "scheduled"
    MANUAL = "manual"

class SyncDirection(Enum):
    BIDIRECTIONAL = "bidirectional"
    SALESFORCE_TO_AGENTSYSTEM = "sf_to_as"
    AGENTSYSTEM_TO_SALESFORCE = "as_to_sf"

@dataclass
class SalesforceConnection:
    connection_id: str
    tenant_id: str
    connection_name: str
    instance_url: str
    client_id: str
    client_secret: str
    username: str
    password: str
    security_token: str
    access_token: Optional[str]
    refresh_token: Optional[str]
    token_expires_at: Optional[datetime]
    is_sandbox: bool
    api_version: str
    is_active: bool
    last_sync: Optional[datetime]
    sync_errors: int
    created_at: datetime
    updated_at: datetime

@dataclass
class SalesforceWorkflow:
    workflow_id: str
    tenant_id: str
    connection_id: str
    name: str
    description: str
    trigger: WorkflowTrigger
    trigger_conditions: Dict[str, Any]
    salesforce_operations: List[Dict[str, Any]]
    agentsystem_operations: List[Dict[str, Any]]
    field_mappings: Dict[str, str]
    is_active: bool
    execution_count: int
    success_count: int
    last_execution: Optional[datetime]
    created_at: datetime
    updated_at: datetime

@dataclass
class SalesforceRecord:
    record_id: str
    object_type: SalesforceObjectType
    salesforce_id: str
    tenant_id: str
    data: Dict[str, Any]
    last_modified: datetime
    sync_status: str
    sync_direction: SyncDirection
    created_at: datetime
    updated_at: datetime

class SalesforceConnector:
    """
    Comprehensive Salesforce integration with advanced automation workflows
    """

    def __init__(self, db_pool: asyncpg.Pool):
        self.db_pool = db_pool
        self.session = None
        self.connections_cache = {}

    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    async def create_connection(self, tenant_id: str, connection_config: Dict[str, Any]) -> SalesforceConnection:
        """Create a new Salesforce connection"""

        connection_id = str(uuid.uuid4())

        connection = SalesforceConnection(
            connection_id=connection_id,
            tenant_id=tenant_id,
            connection_name=connection_config['name'],
            instance_url=connection_config['instance_url'],
            client_id=connection_config['client_id'],
            client_secret=connection_config['client_secret'],
            username=connection_config['username'],
            password=connection_config['password'],
            security_token=connection_config['security_token'],
            access_token=None,
            refresh_token=None,
            token_expires_at=None,
            is_sandbox=connection_config.get('is_sandbox', False),
            api_version=connection_config.get('api_version', 'v58.0'),
            is_active=True,
            last_sync=None,
            sync_errors=0,
            created_at=datetime.now(),
            updated_at=datetime.now()
        )

        # Test connection and get initial tokens
        auth_result = await self._authenticate_connection(connection)
        if auth_result['success']:
            connection.access_token = auth_result['access_token']
            connection.refresh_token = auth_result.get('refresh_token')
            connection.token_expires_at = auth_result.get('expires_at')

            # Store connection
            await self._store_connection(connection)
            self.connections_cache[connection_id] = connection

            return connection
        else:
            raise ValueError(f"Salesforce authentication failed: {auth_result['error']}")

    async def _authenticate_connection(self, connection: SalesforceConnection) -> Dict[str, Any]:
        """Authenticate with Salesforce using OAuth 2.0"""

        try:
            # Prepare authentication request
            auth_url = f"{connection.instance_url}/services/oauth2/token"

            payload = {
                'grant_type': 'password',
                'client_id': connection.client_id,
                'client_secret': connection.client_secret,
                'username': connection.username,
                'password': connection.password + connection.security_token
            }

            async with self.session.post(auth_url, data=payload) as response:
                if response.status == 200:
                    auth_data = await response.json()

                    expires_at = datetime.now() + timedelta(seconds=auth_data.get('expires_in', 3600))

                    return {
                        'success': True,
                        'access_token': auth_data['access_token'],
                        'refresh_token': auth_data.get('refresh_token'),
                        'instance_url': auth_data['instance_url'],
                        'expires_at': expires_at
                    }
                else:
                    error_data = await response.json()
                    return {
                        'success': False,
                        'error': error_data.get('error_description', 'Authentication failed')
                    }

        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }

    async def refresh_access_token(self, connection: SalesforceConnection) -> bool:
        """Refresh Salesforce access token"""

        if not connection.refresh_token:
            return False

        try:
            auth_url = f"{connection.instance_url}/services/oauth2/token"

            payload = {
                'grant_type': 'refresh_token',
                'client_id': connection.client_id,
                'client_secret': connection.client_secret,
                'refresh_token': connection.refresh_token
            }

            async with self.session.post(auth_url, data=payload) as response:
                if response.status == 200:
                    auth_data = await response.json()

                    connection.access_token = auth_data['access_token']
                    connection.token_expires_at = datetime.now() + timedelta(seconds=auth_data.get('expires_in', 3600))
                    connection.updated_at = datetime.now()

                    await self._update_connection(connection)
                    return True

        except Exception as e:
            logger.error(f"Token refresh failed: {e}")

        return False

    async def create_workflow(self, tenant_id: str, workflow_config: Dict[str, Any]) -> SalesforceWorkflow:
        """Create a new Salesforce automation workflow"""

        workflow_id = str(uuid.uuid4())

        workflow = SalesforceWorkflow(
            workflow_id=workflow_id,
            tenant_id=tenant_id,
            connection_id=workflow_config['connection_id'],
            name=workflow_config['name'],
            description=workflow_config.get('description', ''),
            trigger=WorkflowTrigger(workflow_config['trigger']),
            trigger_conditions=workflow_config.get('trigger_conditions', {}),
            salesforce_operations=workflow_config.get('salesforce_operations', []),
            agentsystem_operations=workflow_config.get('agentsystem_operations', []),
            field_mappings=workflow_config.get('field_mappings', {}),
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
        """Execute a Salesforce automation workflow"""

        workflow = await self._get_workflow(workflow_id)
        if not workflow or not workflow.is_active:
            return {'success': False, 'error': 'Workflow not found or inactive'}

        connection = await self._get_connection(workflow.connection_id)
        if not connection or not connection.is_active:
            return {'success': False, 'error': 'Salesforce connection not available'}

        # Ensure valid access token
        if not await self._ensure_valid_token(connection):
            return {'success': False, 'error': 'Failed to authenticate with Salesforce'}

        execution_result = {
            'workflow_id': workflow_id,
            'execution_time': datetime.now(),
            'operations_completed': 0,
            'operations_failed': 0,
            'results': [],
            'success': True
        }

        try:
            # Execute Salesforce operations
            for operation in workflow.salesforce_operations:
                result = await self._execute_salesforce_operation(connection, operation, trigger_data, workflow.field_mappings)
                execution_result['results'].append(result)

                if result['success']:
                    execution_result['operations_completed'] += 1
                else:
                    execution_result['operations_failed'] += 1

            # Execute AgentSystem operations
            for operation in workflow.agentsystem_operations:
                result = await self._execute_agentsystem_operation(operation, trigger_data, workflow.field_mappings)
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

    async def _execute_salesforce_operation(self, connection: SalesforceConnection,
                                          operation: Dict[str, Any], trigger_data: Dict[str, Any],
                                          field_mappings: Dict[str, str]) -> Dict[str, Any]:
        """Execute a single Salesforce operation"""

        operation_type = SalesforceOperation(operation['type'])
        object_type = operation['object_type']

        try:
            if operation_type == SalesforceOperation.CREATE:
                return await self._salesforce_create_record(connection, object_type, operation, trigger_data, field_mappings)

            elif operation_type == SalesforceOperation.UPDATE:
                return await self._salesforce_update_record(connection, object_type, operation, trigger_data, field_mappings)

            elif operation_type == SalesforceOperation.QUERY:
                return await self._salesforce_query_records(connection, operation, trigger_data, field_mappings)

            elif operation_type == SalesforceOperation.UPSERT:
                return await self._salesforce_upsert_record(connection, object_type, operation, trigger_data, field_mappings)

            else:
                return {'success': False, 'error': f'Unsupported operation type: {operation_type}'}

        except Exception as e:
            return {'success': False, 'error': str(e)}

    async def _salesforce_create_record(self, connection: SalesforceConnection, object_type: str,
                                      operation: Dict[str, Any], trigger_data: Dict[str, Any],
                                      field_mappings: Dict[str, str]) -> Dict[str, Any]:
        """Create a new record in Salesforce"""

        # Prepare record data
        record_data = {}
        for field, value in operation.get('data', {}).items():
            # Apply field mappings
            mapped_field = field_mappings.get(field, field)

            # Process value (could be static or from trigger data)
            if isinstance(value, str) and value.startswith('{{') and value.endswith('}}'):
                # Extract value from trigger data
                trigger_field = value[2:-2]
                record_data[mapped_field] = trigger_data.get(trigger_field)
            else:
                record_data[mapped_field] = value

        # Make API call
        api_url = f"{connection.instance_url}/services/data/{connection.api_version}/sobjects/{object_type}"
        headers = {
            'Authorization': f'Bearer {connection.access_token}',
            'Content-Type': 'application/json'
        }

        async with self.session.post(api_url, json=record_data, headers=headers) as response:
            response_data = await response.json()

            if response.status == 201:
                return {
                    'success': True,
                    'operation': 'create',
                    'object_type': object_type,
                    'salesforce_id': response_data['id'],
                    'data': record_data
                }
            else:
                return {
                    'success': False,
                    'operation': 'create',
                    'object_type': object_type,
                    'error': response_data
                }

    async def _salesforce_update_record(self, connection: SalesforceConnection, object_type: str,
                                      operation: Dict[str, Any], trigger_data: Dict[str, Any],
                                      field_mappings: Dict[str, str]) -> Dict[str, Any]:
        """Update an existing record in Salesforce"""

        record_id = operation.get('record_id') or trigger_data.get('salesforce_id')
        if not record_id:
            return {'success': False, 'error': 'No record ID provided for update'}

        # Prepare update data
        update_data = {}
        for field, value in operation.get('data', {}).items():
            mapped_field = field_mappings.get(field, field)

            if isinstance(value, str) and value.startswith('{{') and value.endswith('}}'):
                trigger_field = value[2:-2]
                update_data[mapped_field] = trigger_data.get(trigger_field)
            else:
                update_data[mapped_field] = value

        # Make API call
        api_url = f"{connection.instance_url}/services/data/{connection.api_version}/sobjects/{object_type}/{record_id}"
        headers = {
            'Authorization': f'Bearer {connection.access_token}',
            'Content-Type': 'application/json'
        }

        async with self.session.patch(api_url, json=update_data, headers=headers) as response:
            if response.status == 204:
                return {
                    'success': True,
                    'operation': 'update',
                    'object_type': object_type,
                    'salesforce_id': record_id,
                    'data': update_data
                }
            else:
                response_data = await response.json()
                return {
                    'success': False,
                    'operation': 'update',
                    'object_type': object_type,
                    'error': response_data
                }

    async def _salesforce_query_records(self, connection: SalesforceConnection, operation: Dict[str, Any],
                                      trigger_data: Dict[str, Any], field_mappings: Dict[str, str]) -> Dict[str, Any]:
        """Query records from Salesforce"""

        soql_query = operation.get('query', '')

        # Replace placeholders in query
        for placeholder, value in trigger_data.items():
            soql_query = soql_query.replace(f'{{{{{placeholder}}}}}', str(value))

        # URL encode the query
        api_url = f"{connection.instance_url}/services/data/{connection.api_version}/query"
        params = {'q': soql_query}

        headers = {
            'Authorization': f'Bearer {connection.access_token}',
            'Content-Type': 'application/json'
        }

        async with self.session.get(api_url, params=params, headers=headers) as response:
            response_data = await response.json()

            if response.status == 200:
                return {
                    'success': True,
                    'operation': 'query',
                    'total_size': response_data['totalSize'],
                    'records': response_data['records']
                }
            else:
                return {
                    'success': False,
                    'operation': 'query',
                    'error': response_data
                }

    async def _execute_agentsystem_operation(self, operation: Dict[str, Any],
                                           trigger_data: Dict[str, Any],
                                           field_mappings: Dict[str, str]) -> Dict[str, Any]:
        """Execute an AgentSystem operation as part of workflow"""

        operation_type = operation.get('type')

        try:
            if operation_type == 'ai_analysis':
                return await self._execute_ai_analysis(operation, trigger_data)

            elif operation_type == 'document_processing':
                return await self._execute_document_processing(operation, trigger_data)

            elif operation_type == 'send_notification':
                return await self._execute_notification(operation, trigger_data)

            elif operation_type == 'create_task':
                return await self._execute_task_creation(operation, trigger_data)

            else:
                return {'success': False, 'error': f'Unsupported AgentSystem operation: {operation_type}'}

        except Exception as e:
            return {'success': False, 'error': str(e)}

    async def _execute_ai_analysis(self, operation: Dict[str, Any], trigger_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute AI analysis on Salesforce data"""

        # This would integrate with our AI agents
        analysis_type = operation.get('analysis_type', 'general')
        input_data = operation.get('input_data', trigger_data)

        # Placeholder for AI analysis - would integrate with actual AI agents
        analysis_result = {
            'analysis_type': analysis_type,
            'input_data': input_data,
            'sentiment': 'positive',
            'confidence': 0.85,
            'insights': ['High-quality lead', 'Strong engagement potential'],
            'recommendations': ['Schedule follow-up call', 'Send product demo']
        }

        return {
            'success': True,
            'operation': 'ai_analysis',
            'result': analysis_result
        }

    async def sync_data(self, connection_id: str, sync_config: Dict[str, Any]) -> Dict[str, Any]:
        """Synchronize data between Salesforce and AgentSystem"""

        connection = await self._get_connection(connection_id)
        if not connection:
            return {'success': False, 'error': 'Connection not found'}

        if not await self._ensure_valid_token(connection):
            return {'success': False, 'error': 'Authentication failed'}

        sync_result = {
            'sync_started': datetime.now(),
            'records_processed': 0,
            'records_created': 0,
            'records_updated': 0,
            'records_failed': 0,
            'errors': []
        }

        try:
            # Get objects to sync
            objects_to_sync = sync_config.get('objects', ['Lead', 'Contact', 'Account', 'Opportunity'])

            for object_type in objects_to_sync:
                object_result = await self._sync_object_type(connection, object_type, sync_config)

                sync_result['records_processed'] += object_result['processed']
                sync_result['records_created'] += object_result['created']
                sync_result['records_updated'] += object_result['updated']
                sync_result['records_failed'] += object_result['failed']
                sync_result['errors'].extend(object_result['errors'])

            # Update connection last sync time
            connection.last_sync = datetime.now()
            await self._update_connection(connection)

            sync_result['success'] = True
            sync_result['sync_completed'] = datetime.now()

        except Exception as e:
            sync_result['success'] = False
            sync_result['error'] = str(e)

        return sync_result

    async def _sync_object_type(self, connection: SalesforceConnection, object_type: str,
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
            # Build SOQL query for recent records
            last_sync = connection.last_sync or (datetime.now() - timedelta(days=30))
            last_sync_str = last_sync.strftime('%Y-%m-%dT%H:%M:%S.000+0000')

            soql_query = f"""
                SELECT Id, LastModifiedDate, {', '.join(sync_config.get('fields', ['Name']))}
                FROM {object_type}
                WHERE LastModifiedDate > {last_sync_str}
                LIMIT {sync_config.get('limit', 1000)}
            """

            # Execute query
            api_url = f"{connection.instance_url}/services/data/{connection.api_version}/query"
            params = {'q': soql_query}
            headers = {'Authorization': f'Bearer {connection.access_token}'}

            async with self.session.get(api_url, params=params, headers=headers) as response:
                if response.status == 200:
                    query_result = await response.json()

                    for record in query_result['records']:
                        try:
                            # Process each record
                            await self._process_sync_record(connection, object_type, record, sync_config)
                            result['processed'] += 1

                            # Check if record exists in our system
                            existing_record = await self._get_synced_record(connection.tenant_id, record['Id'])
                            if existing_record:
                                result['updated'] += 1
                            else:
                                result['created'] += 1

                        except Exception as e:
                            result['failed'] += 1
                            result['errors'].append(f"Record {record['Id']}: {str(e)}")

                else:
                    result['errors'].append(f"Query failed with status {response.status}")

        except Exception as e:
            result['errors'].append(str(e))

        return result

    async def _process_sync_record(self, connection: SalesforceConnection, object_type: str,
                                 sf_record: Dict[str, Any], sync_config: Dict[str, Any]):
        """Process a single record for synchronization"""

        # Create or update AgentSystem record
        record = SalesforceRecord(
            record_id=str(uuid.uuid4()),
            object_type=SalesforceObjectType(object_type),
            salesforce_id=sf_record['Id'],
            tenant_id=connection.tenant_id,
            data=sf_record,
            last_modified=datetime.fromisoformat(sf_record['LastModifiedDate'].replace('Z', '+00:00')),
            sync_status='synced',
            sync_direction=SyncDirection.SALESFORCE_TO_AGENTSYSTEM,
            created_at=datetime.now(),
            updated_at=datetime.now()
        )

        await self._store_synced_record(record)

    async def _ensure_valid_token(self, connection: SalesforceConnection) -> bool:
        """Ensure the connection has a valid access token"""

        if not connection.access_token:
            return False

        # Check if token is expired
        if connection.token_expires_at and datetime.now() >= connection.token_expires_at:
            return await self.refresh_access_token(connection)

        return True

    # Pre-built workflow templates
    def get_workflow_templates(self) -> List[Dict[str, Any]]:
        """Get pre-built Salesforce workflow templates"""

        return [
            {
                'template_id': 'lead_ai_scoring',
                'name': 'AI Lead Scoring & Qualification',
                'description': 'Automatically score and qualify new leads using AI analysis',
                'trigger': WorkflowTrigger.LEAD_CREATED,
                'salesforce_operations': [
                    {
                        'type': 'query',
                        'query': "SELECT Id, Name, Email, Company, LeadSource FROM Lead WHERE Id = '{{lead_id}}'"
                    },
                    {
                        'type': 'update',
                        'object_type': 'Lead',
                        'record_id': '{{lead_id}}',
                        'data': {
                            'Lead_Score__c': '{{ai_score}}',
                            'AI_Insights__c': '{{ai_insights}}',
                            'Qualification_Status__c': '{{qualification_status}}'
                        }
                    }
                ],
                'agentsystem_operations': [
                    {
                        'type': 'ai_analysis',
                        'analysis_type': 'lead_scoring',
                        'input_data': {
                            'name': '{{Name}}',
                            'email': '{{Email}}',
                            'company': '{{Company}}',
                            'source': '{{LeadSource}}'
                        }
                    }
                ]
            },
            {
                'template_id': 'opportunity_insights',
                'name': 'Opportunity AI Insights',
                'description': 'Generate AI insights when opportunity stage changes',
                'trigger': WorkflowTrigger.OPPORTUNITY_STAGE_CHANGED,
                'salesforce_operations': [
                    {
                        'type': 'update',
                        'object_type': 'Opportunity',
                        'record_id': '{{opportunity_id}}',
                        'data': {
                            'AI_Win_Probability__c': '{{win_probability}}',
                            'Next_Best_Action__c': '{{next_action}}',
                            'Risk_Factors__c': '{{risk_factors}}'
                        }
                    }
                ],
                'agentsystem_operations': [
                    {
                        'type': 'ai_analysis',
                        'analysis_type': 'opportunity_analysis',
                        'input_data': {
                            'stage': '{{StageName}}',
                            'amount': '{{Amount}}',
                            'close_date': '{{CloseDate}}',
                            'account_name': '{{Account.Name}}'
                        }
                    }
                ]
            },
            {
                'template_id': 'case_auto_response',
                'name': 'Automated Case Response',
                'description': 'Generate AI-powered responses for new support cases',
                'trigger': WorkflowTrigger.CASE_ESCALATED,
                'salesforce_operations': [
                    {
                        'type': 'update',
                        'object_type': 'Case',
                        'record_id': '{{case_id}}',
                        'data': {
                            'AI_Response__c': '{{ai_response}}',
                            'Suggested_Solution__c': '{{suggested_solution}}',
                            'Priority_Score__c': '{{priority_score}}'
                        }
                    }
                ],
                'agentsystem_operations': [
                    {
                        'type': 'ai_analysis',
                        'analysis_type': 'case_analysis',
                        'input_data': {
                            'subject': '{{Subject}}',
                            'description': '{{Description}}',
                            'priority': '{{Priority}}',
                            'product': '{{Product__c}}'
                        }
                    }
                ]
            }
        ]

    # Database operations
    async def _store_connection(self, connection: SalesforceConnection):
        """Store Salesforce connection in database"""

        async with self.db_pool.acquire() as conn:
            async with self.db_pool.acquire() as conn:
                await conn.execute("""
                    INSERT INTO integrations.salesforce_connections (
                        connection_id, tenant_id, connection_name, instance_url,
                        client_id, client_secret, username, password, security_token,
                        access_token, refresh_token, token_expires_at, is_sandbox,
                        api_version, is_active, created_at, updated_at
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17)
                """,
                    connection.connection_id, connection.tenant_id, connection.connection_name,
                    connection.instance_url, connection.client_id, connection.client_secret,
                    connection.username, connection.password, connection.security_token,
                    connection.access_token, connection.refresh_token, connection.token_expires_at,
                    connection.is_sandbox, connection.api_version, connection.is_active,
                    connection.created_at, connection.updated_at
                )

        async def _update_connection(self, connection: SalesforceConnection):
            """Update Salesforce connection in database"""

            async with self.db_pool.acquire() as conn:
                await conn.execute("""
                    UPDATE integrations.salesforce_connections SET
                        connection_name = $3, instance_url = $4, access_token = $5,
                        refresh_token = $6, token_expires_at = $7, is_active = $8,
                        last_sync = $9, sync_errors = $10, updated_at = $11
                    WHERE connection_id = $1 AND tenant_id = $2
                """,
                    connection.connection_id, connection.tenant_id, connection.connection_name,
                    connection.instance_url, connection.access_token, connection.refresh_token,
                    connection.token_expires_at, connection.is_active, connection.last_sync,
                    connection.sync_errors, connection.updated_at
                )

        async def _get_connection(self, connection_id: str) -> Optional[SalesforceConnection]:
            """Get Salesforce connection from database"""

            if connection_id in self.connections_cache:
                return self.connections_cache[connection_id]

            async with self.db_pool.acquire() as conn:
                row = await conn.fetchrow("""
                    SELECT * FROM integrations.salesforce_connections
                    WHERE connection_id = $1
                """, connection_id)

                if row:
                    connection = SalesforceConnection(
                        connection_id=row['connection_id'],
                        tenant_id=row['tenant_id'],
                        connection_name=row['connection_name'],
                        instance_url=row['instance_url'],
                        client_id=row['client_id'],
                        client_secret=row['client_secret'],
                        username=row['username'],
                        password=row['password'],
                        security_token=row['security_token'],
                        access_token=row['access_token'],
                        refresh_token=row['refresh_token'],
                        token_expires_at=row['token_expires_at'],
                        is_sandbox=row['is_sandbox'],
                        api_version=row['api_version'],
                        is_active=row['is_active'],
                        last_sync=row['last_sync'],
                        sync_errors=row['sync_errors'],
                        created_at=row['created_at'],
                        updated_at=row['updated_at']
                    )

                    self.connections_cache[connection_id] = connection
                    return connection

            return None

        async def _store_workflow(self, workflow: SalesforceWorkflow):
            """Store Salesforce workflow in database"""

            async with self.db_pool.acquire() as conn:
                await conn.execute("""
                    INSERT INTO integrations.salesforce_workflows (
                        workflow_id, tenant_id, connection_id, name, description,
                        trigger_type, trigger_conditions, salesforce_operations,
                        agentsystem_operations, field_mappings, is_active,
                        created_at, updated_at
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13)
                """,
                    workflow.workflow_id, workflow.tenant_id, workflow.connection_id,
                    workflow.name, workflow.description, workflow.trigger.value,
                    json.dumps(workflow.trigger_conditions), json.dumps(workflow.salesforce_operations),
                    json.dumps(workflow.agentsystem_operations), json.dumps(workflow.field_mappings),
                    workflow.is_active, workflow.created_at, workflow.updated_at
                )

        async def _update_workflow(self, workflow: SalesforceWorkflow):
            """Update workflow statistics"""

            async with self.db_pool.acquire() as conn:
                await conn.execute("""
                    UPDATE integrations.salesforce_workflows SET
                        execution_count = $3, success_count = $4, last_execution = $5,
                        updated_at = $6
                    WHERE workflow_id = $1 AND tenant_id = $2
                """,
                    workflow.workflow_id, workflow.tenant_id, workflow.execution_count,
                    workflow.success_count, workflow.last_execution, workflow.updated_at
                )

        async def _get_workflow(self, workflow_id: str) -> Optional[SalesforceWorkflow]:
            """Get workflow from database"""

            async with self.db_pool.acquire() as conn:
                row = await conn.fetchrow("""
                    SELECT * FROM integrations.salesforce_workflows
                    WHERE workflow_id = $1
                """, workflow_id)

                if row:
                    return SalesforceWorkflow(
                        workflow_id=row['workflow_id'],
                        tenant_id=row['tenant_id'],
                        connection_id=row['connection_id'],
                        name=row['name'],
                        description=row['description'],
                        trigger=WorkflowTrigger(row['trigger_type']),
                        trigger_conditions=json.loads(row['trigger_conditions']),
                        salesforce_operations=json.loads(row['salesforce_operations']),
                        agentsystem_operations=json.loads(row['agentsystem_operations']),
                        field_mappings=json.loads(row['field_mappings']),
                        is_active=row['is_active'],
                        execution_count=row['execution_count'],
                        success_count=row['success_count'],
                        last_execution=row['last_execution'],
                        created_at=row['created_at'],
                        updated_at=row['updated_at']
                    )

            return None

        async def _store_synced_record(self, record: SalesforceRecord):
            """Store synced Salesforce record"""

            async with self.db_pool.acquire() as conn:
                await conn.execute("""
                    INSERT INTO integrations.salesforce_records (
                        record_id, object_type, salesforce_id, tenant_id, data,
                        last_modified, sync_status, sync_direction, created_at, updated_at
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
                    ON CONFLICT (tenant_id, salesforce_id) DO UPDATE SET
                        data = EXCLUDED.data,
                        last_modified = EXCLUDED.last_modified,
                        sync_status = EXCLUDED.sync_status,
                        updated_at = EXCLUDED.updated_at
                """,
                    record.record_id, record.object_type.value, record.salesforce_id,
                    record.tenant_id, json.dumps(record.data), record.last_modified,
                    record.sync_status, record.sync_direction.value, record.created_at,
                    record.updated_at
                )

        async def _get_synced_record(self, tenant_id: str, salesforce_id: str) -> Optional[SalesforceRecord]:
            """Get synced record by Salesforce ID"""

            async with self.db_pool.acquire() as conn:
                row = await conn.fetchrow("""
                    SELECT * FROM integrations.salesforce_records
                    WHERE tenant_id = $1 AND salesforce_id = $2
                """, tenant_id, salesforce_id)

                if row:
                    return SalesforceRecord(
                        record_id=row['record_id'],
                        object_type=SalesforceObjectType(row['object_type']),
                        salesforce_id=row['salesforce_id'],
                        tenant_id=row['tenant_id'],
                        data=json.loads(row['data']),
                        last_modified=row['last_modified'],
                        sync_status=row['sync_status'],
                        sync_direction=SyncDirection(row['sync_direction']),
                        created_at=row['created_at'],
                        updated_at=row['updated_at']
                    )

            return None

    # Database schema for Salesforce integration
    SALESFORCE_SCHEMA_SQL = """
    -- Salesforce connections table
    CREATE TABLE IF NOT EXISTS integrations.salesforce_connections (
        connection_id UUID PRIMARY KEY,
        tenant_id UUID REFERENCES tenant_management.tenants(id) ON DELETE CASCADE,
        connection_name VARCHAR(500) NOT NULL,
        instance_url TEXT NOT NULL,
        client_id VARCHAR(500) NOT NULL,
        client_secret VARCHAR(500) NOT NULL,
        username VARCHAR(500) NOT NULL,
        password VARCHAR(500) NOT NULL,
        security_token VARCHAR(500) NOT NULL,
        access_token TEXT,
        refresh_token TEXT,
        token_expires_at TIMESTAMP WITH TIME ZONE,
        is_sandbox BOOLEAN DEFAULT FALSE,
        api_version VARCHAR(20) DEFAULT 'v58.0',
        is_active BOOLEAN DEFAULT TRUE,
        last_sync TIMESTAMP WITH TIME ZONE,
        sync_errors INTEGER DEFAULT 0,
        created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
        updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
    );

    -- Salesforce workflows table
    CREATE TABLE IF NOT EXISTS integrations.salesforce_workflows (
        workflow_id UUID PRIMARY KEY,
        tenant_id UUID REFERENCES tenant_management.tenants(id) ON DELETE CASCADE,
        connection_id UUID REFERENCES integrations.salesforce_connections(connection_id) ON DELETE CASCADE,
        name VARCHAR(500) NOT NULL,
        description TEXT,
        trigger_type VARCHAR(100) NOT NULL,
        trigger_conditions JSONB DEFAULT '{}',
        salesforce_operations JSONB DEFAULT '[]',
        agentsystem_operations JSONB DEFAULT '[]',
        field_mappings JSONB DEFAULT '{}',
        is_active BOOLEAN DEFAULT TRUE,
        execution_count INTEGER DEFAULT 0,
        success_count INTEGER DEFAULT 0,
        last_execution TIMESTAMP WITH TIME ZONE,
        created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
        updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
    );

    -- Salesforce synced records table
    CREATE TABLE IF NOT EXISTS integrations.salesforce_records (
        record_id UUID PRIMARY KEY,
        object_type VARCHAR(100) NOT NULL,
        salesforce_id VARCHAR(18) NOT NULL,
        tenant_id UUID REFERENCES tenant_management.tenants(id) ON DELETE CASCADE,
        data JSONB NOT NULL,
        last_modified TIMESTAMP WITH TIME ZONE NOT NULL,
        sync_status VARCHAR(50) DEFAULT 'synced',
        sync_direction VARCHAR(50) NOT NULL,
        created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
        updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
        UNIQUE(tenant_id, salesforce_id)
    );

    -- Workflow execution log table
    CREATE TABLE IF NOT EXISTS integrations.salesforce_execution_log (
        execution_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        workflow_id UUID REFERENCES integrations.salesforce_workflows(workflow_id) ON DELETE CASCADE,
        tenant_id UUID REFERENCES tenant_management.tenants(id) ON DELETE CASCADE,
        trigger_data JSONB NOT NULL,
        execution_result JSONB NOT NULL,
        success BOOLEAN NOT NULL,
        execution_time_ms INTEGER,
        executed_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
    );

    -- Indexes for performance
    CREATE INDEX IF NOT EXISTS idx_salesforce_connections_tenant
    ON integrations.salesforce_connections(tenant_id, is_active);

    CREATE INDEX IF NOT EXISTS idx_salesforce_workflows_tenant_trigger
    ON integrations.salesforce_workflows(tenant_id, trigger_type, is_active);

    CREATE INDEX IF NOT EXISTS idx_salesforce_records_tenant_type
    ON integrations.salesforce_records(tenant_id, object_type);

    CREATE INDEX IF NOT EXISTS idx_salesforce_records_sf_id
    ON integrations.salesforce_records(salesforce_id);

    CREATE INDEX IF NOT EXISTS idx_salesforce_execution_log_workflow
    ON integrations.salesforce_execution_log(workflow_id, executed_at DESC);

    -- Views for analytics
    CREATE OR REPLACE VIEW integrations.salesforce_analytics AS
    SELECT
        sc.tenant_id,
        sc.connection_name,
        sc.is_active as connection_active,
        sc.last_sync,
        COUNT(sw.workflow_id) as total_workflows,
        COUNT(CASE WHEN sw.is_active THEN 1 END) as active_workflows,
        SUM(sw.execution_count) as total_executions,
        SUM(sw.success_count) as successful_executions,
        COUNT(sr.record_id) as synced_records,
        COUNT(CASE WHEN sr.sync_status = 'synced' THEN 1 END) as successfully_synced
    FROM integrations.salesforce_connections sc
    LEFT JOIN integrations.salesforce_workflows sw ON sc.connection_id = sw.connection_id
    LEFT JOIN integrations.salesforce_records sr ON sc.tenant_id = sr.tenant_id
    GROUP BY sc.tenant_id, sc.connection_name, sc.is_active, sc.last_sync;

    -- Function to clean up old execution logs
    CREATE OR REPLACE FUNCTION integrations.cleanup_salesforce_logs()
    RETURNS void AS $$
    BEGIN
        DELETE FROM integrations.salesforce_execution_log
        WHERE executed_at < NOW() - INTERVAL '90 days';
    END;
    $$ LANGUAGE plpgsql;

    -- Comments for documentation
    COMMENT ON TABLE integrations.salesforce_connections IS 'Salesforce org connections with authentication details';
    COMMENT ON TABLE integrations.salesforce_workflows IS 'Automated workflows between Salesforce and AgentSystem';
    COMMENT ON TABLE integrations.salesforce_records IS 'Synced Salesforce records with change tracking';
    COMMENT ON TABLE integrations.salesforce_execution_log IS 'Log of workflow executions for debugging and analytics';
    """

    # Pydantic models for API
    class SalesforceConnectionRequest(BaseModel):
        name: str = Field(..., description="Connection name")
        instance_url: str = Field(..., description="Salesforce instance URL")
        client_id: str = Field(..., description="Connected app client ID")
        client_secret: str = Field(..., description="Connected app client secret")
        username: str = Field(..., description="Salesforce username")
        password: str = Field(..., description="Salesforce password")
        security_token: str = Field(..., description="Salesforce security token")
        is_sandbox: bool = Field(False, description="Is this a sandbox org")
        api_version: str = Field("v58.0", description="Salesforce API version")

    class SalesforceWorkflowRequest(BaseModel):
        connection_id: str = Field(..., description="Salesforce connection ID")
        name: str = Field(..., description="Workflow name")
        description: Optional[str] = Field(None, description="Workflow description")
        trigger: str = Field(..., description="Workflow trigger type")
        trigger_conditions: Optional[Dict[str, Any]] = Field(None, description="Trigger conditions")
        salesforce_operations: List[Dict[str, Any]] = Field(..., description="Salesforce operations")
        agentsystem_operations: Optional[List[Dict[str, Any]]] = Field(None, description="AgentSystem operations")
        field_mappings: Optional[Dict[str, str]] = Field(None, description="Field mappings")

    class SalesforceSyncRequest(BaseModel):
        connection_id: str = Field(..., description="Connection ID to sync")
        objects: List[str] = Field(["Lead", "Contact", "Account", "Opportunity"], description="Objects to sync")
        fields: List[str] = Field(["Name", "Email", "Phone"], description="Fields to include")
        limit: int = Field(1000, description="Maximum records per object")
        incremental: bool = Field(True, description="Only sync changed records")

    # Export main classes
    __all__ = [
        'SalesforceConnector', 'SalesforceConnection', 'SalesforceWorkflow', 'SalesforceRecord',
        'SalesforceObjectType', 'SalesforceOperation', 'WorkflowTrigger', 'SyncDirection',
        'SalesforceConnectionRequest', 'SalesforceWorkflowRequest', 'SalesforceSyncRequest',
        'SALESFORCE_SCHEMA_SQL'
    ]
