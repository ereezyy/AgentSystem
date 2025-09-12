
"""
Zapier-style Workflow Builder - AgentSystem Profit Machine
Visual workflow automation platform with drag-and-drop interface
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Callable
from enum import Enum
from dataclasses import dataclass, asdict, field
import asyncpg
import uuid
from pydantic import BaseModel, Field, validator
import yaml

logger = logging.getLogger(__name__)

class WorkflowStatus(Enum):
    DRAFT = "draft"
    ACTIVE = "active"
    PAUSED = "paused"
    DISABLED = "disabled"
    ERROR = "error"

class TriggerType(Enum):
    # Time-based triggers
    SCHEDULE = "schedule"
    DELAY = "delay"

    # Data triggers
    WEBHOOK = "webhook"
    EMAIL_RECEIVED = "email_received"
    FORM_SUBMISSION = "form_submission"
    DATABASE_CHANGE = "database_change"

    # External service triggers
    SLACK_MESSAGE = "slack_message"
    TEAMS_MESSAGE = "teams_message"
    SALESFORCE_RECORD = "salesforce_record"
    HUBSPOT_CONTACT = "hubspot_contact"
    STRIPE_PAYMENT = "stripe_payment"

    # File triggers
    FILE_UPLOAD = "file_upload"
    FILE_MODIFIED = "file_modified"

    # AI triggers
    AI_COMPLETION = "ai_completion"
    SENTIMENT_THRESHOLD = "sentiment_threshold"

class ActionType(Enum):
    # Communication actions
    SEND_EMAIL = "send_email"
    SEND_SLACK_MESSAGE = "send_slack_message"
    SEND_TEAMS_MESSAGE = "send_teams_message"
    SEND_SMS = "send_sms"

    # Data actions
    CREATE_RECORD = "create_record"
    UPDATE_RECORD = "update_record"
    DELETE_RECORD = "delete_record"
    TRANSFORM_DATA = "transform_data"

    # AI actions
    AI_ANALYZE = "ai_analyze"
    AI_GENERATE = "ai_generate"
    AI_TRANSLATE = "ai_translate"
    AI_SUMMARIZE = "ai_summarize"

    # External service actions
    SALESFORCE_ACTION = "salesforce_action"
    HUBSPOT_ACTION = "hubspot_action"
    STRIPE_ACTION = "stripe_action"

    # File actions
    UPLOAD_FILE = "upload_file"
    PROCESS_DOCUMENT = "process_document"
    GENERATE_REPORT = "generate_report"

    # Flow control
    CONDITION = "condition"
    LOOP = "loop"
    WAIT = "wait"

    # Webhook actions
    HTTP_REQUEST = "http_request"
    WEBHOOK_CALL = "webhook_call"

class ConditionOperator(Enum):
    EQUALS = "equals"
    NOT_EQUALS = "not_equals"
    GREATER_THAN = "greater_than"
    LESS_THAN = "less_than"
    CONTAINS = "contains"
    NOT_CONTAINS = "not_contains"
    STARTS_WITH = "starts_with"
    ENDS_WITH = "ends_with"
    IS_EMPTY = "is_empty"
    IS_NOT_EMPTY = "is_not_empty"
    IN_LIST = "in_list"
    NOT_IN_LIST = "not_in_list"

@dataclass
class WorkflowNode:
    node_id: str
    node_type: str  # trigger, action, condition
    name: str
    description: Optional[str]
    config: Dict[str, Any]
    position: Dict[str, float]  # x, y coordinates for visual editor
    connections: List[str] = field(default_factory=list)  # connected node IDs
    is_enabled: bool = True
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

@dataclass
class WorkflowConnection:
    connection_id: str
    source_node_id: str
    target_node_id: str
    source_port: str = "output"
    target_port: str = "input"
    condition: Optional[Dict[str, Any]] = None  # For conditional connections
    created_at: datetime = field(default_factory=datetime.now)

@dataclass
class Workflow:
    workflow_id: str
    tenant_id: str
    name: str
    description: Optional[str]
    status: WorkflowStatus
    nodes: List[WorkflowNode]
    connections: List[WorkflowConnection]
    variables: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    folder_id: Optional[str] = None
    is_template: bool = False
    execution_count: int = 0
    last_executed_at: Optional[datetime] = None
    created_by: str = ""
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

@dataclass
class WorkflowExecution:
    execution_id: str
    workflow_id: str
    tenant_id: str
    status: str  # running, completed, failed, cancelled
    trigger_data: Dict[str, Any]
    execution_context: Dict[str, Any] = field(default_factory=dict)
    current_node_id: Optional[str] = None
    error_message: Optional[str] = None
    execution_time_ms: Optional[int] = None
    nodes_executed: List[str] = field(default_factory=list)
    started_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None

class WorkflowEngine:
    """
    Core workflow execution engine with visual builder support
    """

    def __init__(self, db_pool: asyncpg.Pool, integrations: Dict[str, Any] = None):
        self.db_pool = db_pool
        self.integrations = integrations or {}
        self.running_executions = {}

        # Built-in trigger handlers
        self.trigger_handlers = {
            TriggerType.WEBHOOK: self._handle_webhook_trigger,
            TriggerType.SCHEDULE: self._handle_schedule_trigger,
            TriggerType.EMAIL_RECEIVED: self._handle_email_trigger,
            TriggerType.SLACK_MESSAGE: self._handle_slack_trigger,
            TriggerType.TEAMS_MESSAGE: self._handle_teams_trigger,
            TriggerType.FILE_UPLOAD: self._handle_file_trigger,
        }

        # Built-in action handlers
        self.action_handlers = {
            ActionType.SEND_EMAIL: self._handle_send_email_action,
            ActionType.SEND_SLACK_MESSAGE: self._handle_send_slack_action,
            ActionType.SEND_TEAMS_MESSAGE: self._handle_send_teams_action,
            ActionType.AI_ANALYZE: self._handle_ai_analyze_action,
            ActionType.AI_GENERATE: self._handle_ai_generate_action,
            ActionType.CREATE_RECORD: self._handle_create_record_action,
            ActionType.UPDATE_RECORD: self._handle_update_record_action,
            ActionType.HTTP_REQUEST: self._handle_http_request_action,
            ActionType.CONDITION: self._handle_condition_action,
            ActionType.TRANSFORM_DATA: self._handle_transform_data_action,
        }

    async def create_workflow(self, tenant_id: str, name: str, description: str = None,
                            created_by: str = "", folder_id: str = None) -> Workflow:
        """Create a new workflow"""

        workflow_id = str(uuid.uuid4())

        workflow = Workflow(
            workflow_id=workflow_id,
            tenant_id=tenant_id,
            name=name,
            description=description,
            status=WorkflowStatus.DRAFT,
            nodes=[],
            connections=[],
            folder_id=folder_id,
            created_by=created_by
        )

        await self._store_workflow(workflow)
        return workflow

    async def add_node(self, workflow_id: str, node_type: str, name: str,
                      config: Dict[str, Any], position: Dict[str, float],
                      description: str = None) -> WorkflowNode:
        """Add a node to workflow"""

        node_id = str(uuid.uuid4())

        node = WorkflowNode(
            node_id=node_id,
            node_type=node_type,
            name=name,
            description=description,
            config=config,
            position=position
        )

        workflow = await self._get_workflow(workflow_id)
        workflow.nodes.append(node)
        workflow.updated_at = datetime.now()

        await self._update_workflow(workflow)
        return node

    async def add_connection(self, workflow_id: str, source_node_id: str,
                           target_node_id: str, source_port: str = "output",
                           target_port: str = "input",
                           condition: Dict[str, Any] = None) -> WorkflowConnection:
        """Add a connection between nodes"""

        connection_id = str(uuid.uuid4())

        connection = WorkflowConnection(
            connection_id=connection_id,
            source_node_id=source_node_id,
            target_node_id=target_node_id,
            source_port=source_port,
            target_port=target_port,
            condition=condition
        )

        workflow = await self._get_workflow(workflow_id)
        workflow.connections.append(connection)
        workflow.updated_at = datetime.now()

        await self._update_workflow(workflow)
        return connection

    async def activate_workflow(self, workflow_id: str) -> bool:
        """Activate a workflow for execution"""

        workflow = await self._get_workflow(workflow_id)
        if not workflow:
            return False

        # Validate workflow before activation
        validation_result = await self._validate_workflow(workflow)
        if not validation_result['valid']:
            raise ValueError(f"Workflow validation failed: {validation_result['errors']}")

        workflow.status = WorkflowStatus.ACTIVE
        workflow.updated_at = datetime.now()

        await self._update_workflow(workflow)

        # Set up triggers
        await self._setup_workflow_triggers(workflow)

        return True

    async def execute_workflow(self, workflow_id: str, trigger_data: Dict[str, Any],
                             execution_context: Dict[str, Any] = None) -> WorkflowExecution:
        """Execute a workflow with given trigger data"""

        workflow = await self._get_workflow(workflow_id)
        if not workflow or workflow.status != WorkflowStatus.ACTIVE:
            raise ValueError("Workflow not found or not active")

        execution_id = str(uuid.uuid4())

        execution = WorkflowExecution(
            execution_id=execution_id,
            workflow_id=workflow_id,
            tenant_id=workflow.tenant_id,
            status="running",
            trigger_data=trigger_data,
            execution_context=execution_context or {}
        )

        await self._store_execution(execution)
        self.running_executions[execution_id] = execution

        try:
            # Find trigger node
            trigger_nodes = [node for node in workflow.nodes if node.node_type == "trigger"]
            if not trigger_nodes:
                raise ValueError("No trigger node found in workflow")

            # Start execution from trigger node
            await self._execute_node(workflow, trigger_nodes[0], execution)

            execution.status = "completed"
            execution.completed_at = datetime.now()
            execution.execution_time_ms = int((execution.completed_at - execution.started_at).total_seconds() * 1000)

        except Exception as e:
            execution.status = "failed"
            execution.error_message = str(e)
            execution.completed_at = datetime.now()
            logger.error(f"Workflow execution failed: {e}")

        finally:
            await self._update_execution(execution)
            if execution_id in self.running_executions:
                del self.running_executions[execution_id]

            # Update workflow execution stats
            workflow.execution_count += 1
            workflow.last_executed_at = datetime.now()
            await self._update_workflow(workflow)

        return execution

    async def _execute_node(self, workflow: Workflow, node: WorkflowNode,
                          execution: WorkflowExecution) -> Any:
        """Execute a single node"""

        execution.current_node_id = node.node_id
        execution.nodes_executed.append(node.node_id)

        if not node.is_enabled:
            logger.info(f"Skipping disabled node: {node.name}")
            return None

        try:
            result = None

            # Execute based on node type
            if node.node_type == "trigger":
                result = await self._execute_trigger_node(node, execution)
            elif node.node_type == "action":
                result = await self._execute_action_node(node, execution)
            elif node.node_type == "condition":
                result = await self._execute_condition_node(node, execution)

            # Store result in execution context
            execution.execution_context[node.node_id] = result

            # Find and execute next nodes
            next_nodes = await self._get_next_nodes(workflow, node, result)

            for next_node in next_nodes:
                await self._execute_node(workflow, next_node, execution)

            return result

        except Exception as e:
            logger.error(f"Node execution failed ({node.name}): {e}")
            raise

    async def _execute_trigger_node(self, node: WorkflowNode,
                                   execution: WorkflowExecution) -> Dict[str, Any]:
        """Execute a trigger node"""

        trigger_type = TriggerType(node.config.get('trigger_type'))

        if trigger_type in self.trigger_handlers:
            handler = self.trigger_handlers[trigger_type]
            return await handler(node, execution)

        # Return trigger data as-is
        return execution.trigger_data

    async def _execute_action_node(self, node: WorkflowNode,
                                 execution: WorkflowExecution) -> Any:
        """Execute an action node"""

        action_type = ActionType(node.config.get('action_type'))

        if action_type in self.action_handlers:
            handler = self.action_handlers[action_type]
            return await handler(node, execution)

        raise ValueError(f"Unsupported action type: {action_type}")

    async def _execute_condition_node(self, node: WorkflowNode,
                                    execution: WorkflowExecution) -> bool:
        """Execute a condition node"""

        condition_config = node.config.get('condition', {})
        left_value = await self._resolve_value(condition_config.get('left'), execution)
        right_value = await self._resolve_value(condition_config.get('right'), execution)
        operator = ConditionOperator(condition_config.get('operator'))

        return self._evaluate_condition(left_value, operator, right_value)

    async def _get_next_nodes(self, workflow: Workflow, current_node: WorkflowNode,
                            execution_result: Any) -> List[WorkflowNode]:
        """Get next nodes to execute based on connections"""

        next_nodes = []

        # Find connections from current node
        outgoing_connections = [
            conn for conn in workflow.connections
            if conn.source_node_id == current_node.node_id
        ]

        for connection in outgoing_connections:
            # Check connection condition if present
            if connection.condition:
                condition_met = await self._evaluate_connection_condition(
                    connection.condition, execution_result
                )
                if not condition_met:
                    continue

            # Find target node
            target_node = next(
                (node for node in workflow.nodes if node.node_id == connection.target_node_id),
                None
            )

            if target_node:
                next_nodes.append(target_node)

        return next_nodes

    # Built-in trigger handlers
    async def _handle_webhook_trigger(self, node: WorkflowNode,
                                    execution: WorkflowExecution) -> Dict[str, Any]:
        """Handle webhook trigger"""
        return execution.trigger_data

    async def _handle_schedule_trigger(self, node: WorkflowNode,
                                     execution: WorkflowExecution) -> Dict[str, Any]:
        """Handle schedule trigger"""
        return {
            'trigger_time': datetime.now().isoformat(),
            'schedule_config': node.config
        }

    async def _handle_email_trigger(self, node: WorkflowNode,
                                  execution: WorkflowExecution) -> Dict[str, Any]:
        """Handle email received trigger"""
        return execution.trigger_data

    async def _handle_slack_trigger(self, node: WorkflowNode,
                                  execution: WorkflowExecution) -> Dict[str, Any]:
        """Handle Slack message trigger"""
        return execution.trigger_data

    async def _handle_teams_trigger(self, node: WorkflowNode,
                                  execution: WorkflowExecution) -> Dict[str, Any]:
        """Handle Teams message trigger"""
        return execution.trigger_data

    async def _handle_file_trigger(self, node: WorkflowNode,
                                 execution: WorkflowExecution) -> Dict[str, Any]:
        """Handle file upload trigger"""
        return execution.trigger_data

    # Built-in action handlers
    async def _handle_send_email_action(self, node: WorkflowNode,
                                      execution: WorkflowExecution) -> Dict[str, Any]:
        """Handle send email action"""

        config = node.config
        email_data = {
            'to': await self._resolve_value(config.get('to'), execution),
            'subject': await self._resolve_value(config.get('subject'), execution),
            'body': await self._resolve_value(config.get('body'), execution)
        }

        # Send email using email service integration
        if 'email_service' in self.integrations:
            result = await self.integrations['email_service'].send_email(**email_data)
            return {'success': True, 'result': result}

        return {'success': False, 'error': 'Email service not configured'}

    async def _handle_send_slack_action(self, node: WorkflowNode,
                                      execution: WorkflowExecution) -> Dict[str, Any]:
        """Handle send Slack message action"""

        config = node.config
        slack_data = {
            'channel': await self._resolve_value(config.get('channel'), execution),
            'message': await self._resolve_value(config.get('message'), execution)
        }

        # Send message using Slack integration
        if 'slack_service' in self.integrations:
            result = await self.integrations['slack_service'].send_message(**slack_data)
            return {'success': True, 'result': result}

        return {'success': False, 'error': 'Slack service not configured'}

    async def _handle_send_teams_action(self, node: WorkflowNode,
                                      execution: WorkflowExecution) -> Dict[str, Any]:
        """Handle send Teams message action"""

        config = node.config
        teams_data = {
            'team_id': await self._resolve_value(config.get('team_id'), execution),
            'channel_id': await self._resolve_value(config.get('channel_id'), execution),
            'message': await self._resolve_value(config.get('message'), execution)
        }

        # Send message using Teams integration
        if 'teams_service' in self.integrations:
            result = await self.integrations['teams_service'].send_message(**teams_data)
            return {'success': True, 'result': result}

        return {'success': False, 'error': 'Teams service not configured'}

    async def _handle_ai_analyze_action(self, node: WorkflowNode,
                                      execution: WorkflowExecution) -> Dict[str, Any]:
        """Handle AI analyze action"""

        config = node.config
        text_to_analyze = await self._resolve_value(config.get('text'), execution)
        analysis_type = config.get('analysis_type', 'general')

        # Perform AI analysis using AI service
        if 'ai_service' in self.integrations:
            result = await self.integrations['ai_service'].analyze_text(
                text=text_to_analyze,
                analysis_type=analysis_type
            )
            return {'success': True, 'analysis': result}

        return {'success': False, 'error': 'AI service not configured'}

    async def _handle_ai_generate_action(self, node: WorkflowNode,
                                       execution: WorkflowExecution) -> Dict[str, Any]:
        """Handle AI content generation action"""

        config = node.config
        prompt = await self._resolve_value(config.get('prompt'), execution)
        generation_type = config.get('generation_type', 'text')

        # Generate content using AI service
        if 'ai_service' in self.integrations:
            result = await self.integrations['ai_service'].generate_content(
                prompt=prompt,
                content_type=generation_type
            )
            return {'success': True, 'generated_content': result}

        return {'success': False, 'error': 'AI service not configured'}

    async def _handle_create_record_action(self, node: WorkflowNode,
                                         execution: WorkflowExecution) -> Dict[str, Any]:
        """Handle create database record action"""

        config = node.config
        table_name = config.get('table')
        record_data = {}

        # Build record data from config
        for field_config in config.get('fields', []):
            field_name = field_config['name']
            field_value = await self._resolve_value(field_config['value'], execution)
            record_data[field_name] = field_value

        # Insert record into database
        async with self.db_pool.acquire() as conn:
            columns = ', '.join(record_data.keys())
            placeholders = ', '.join(f'${i+1}' for i in range(len(record_data)))
            query = f"INSERT INTO {table_name} ({columns}) VALUES ({placeholders}) RETURNING *"

            result = await conn.fetchrow(query, *record_data.values())
            return {'success': True, 'record': dict(result) if result else None}

    async def _handle_update_record_action(self, node: WorkflowNode,
                                         execution: WorkflowExecution) -> Dict[str, Any]:
        """Handle update database record action"""

        config = node.config
        table_name = config.get('table')
        where_condition = await self._resolve_value(config.get('where'), execution)

        update_data = {}
        for field_config in config.get('fields', []):
            field_name = field_config['name']
            field_value = await self._resolve_value(field_config['value'], execution)
            update_data[field_name] = field_value

        # Update record in database
        async with self.db_pool.acquire() as conn:
            set_clause = ', '.join(f"{key} = ${i+1}" for i, key in enumerate(update_data.keys()))
            query = f"UPDATE {table_name} SET {set_clause} WHERE {where_condition} RETURNING *"

            result = await conn.fetchrow(query, *update_data.values())
            return {'success': True, 'record': dict(result) if result else None}

    async def _handle_http_request_action(self, node: WorkflowNode,
                                        execution: WorkflowExecution) -> Dict[str, Any]:
        """Handle HTTP request action"""

        import aiohttp

        config = node.config
        method = config.get('method', 'GET')
        url = await self._resolve_value(config.get('url'), execution)
        headers = config.get('headers', {})

        # Resolve header values
        resolved_headers = {}
        for key, value in headers.items():
            resolved_headers[key] = await self._resolve_value(value, execution)

        # Prepare request data
        request_data = None
        if config.get('body'):
            request_data = await self._resolve_value(config.get('body'), execution)
            if isinstance(request_data, str):
                try:
                    request_data = json.loads(request_data)
                except:
                    pass

        # Make HTTP request
        async with aiohttp.ClientSession() as session:
            async with session.request(
                method=method,
                url=url,
                headers=resolved_headers,
                json=request_data if isinstance(request_data, dict) else None,
                data=request_data if isinstance(request_data, str) else None
            ) as response:
                response_data = await response.text()

                try:
                    response_json = json.loads(response_data)
                except:
                    response_json = None

                return {
                    'success': response.status < 400,
                    'status_code': response.status,
                    'headers': dict(response.headers),
                    'body': response_data,
                    'json': response_json
                }

    async def _handle_condition_action(self, node: WorkflowNode,
                                     execution: WorkflowExecution) -> bool:
        """Handle condition evaluation"""
        return await self._execute_condition_node(node, execution)

    async def _handle_transform_data_action(self, node: WorkflowNode,
                                          execution: WorkflowExecution) -> Dict[str, Any]:
        """Handle data transformation action"""

        config = node.config
        input_data = await self._resolve_value(config.get('input'), execution)
        transformations = config.get('transformations', [])

        result = input_data

        for transformation in transformations:
            transform_type = transformation.get('type')

            if transform_type == 'map':
                # Map object properties
                mapping = transformation.get('mapping', {})
                if isinstance(result, dict):
                    new_result = {}
                    for new_key, old_key in mapping.items():
                        if old_key in result:
                            new_result[new_key] = result[old_key]
                    result = new_result

            elif transform_type == 'filter':
                # Filter array items
                condition = transformation.get('condition')
                if isinstance(result, list):
                    filtered_result = []
                    for item in result:
                        if await self._evaluate_filter_condition(condition, item):
                            filtered_result.append(item)
                    result = filtered_result

            elif transform_type == 'format':
                # Format string
                template = transformation.get('template')
                if isinstance(result, dict):
                    result = template.format(**result)

        return {'success': True, 'transformed_data': result}

    # Helper methods
    async def _resolve_value(self, value: Any, execution: WorkflowExecution) -> Any:
        """Resolve dynamic values from execution context"""

        if isinstance(value, str):
            # Handle template variables like {{trigger.email.subject}}
            if value.startswith('{{') and value.endswith('}}'):
                path = value[2:-2].strip()
                return self._get_nested_value(execution.execution_context, path)

            # Handle direct context references
            if value.startswith('$'):
                path = value[1:]
                return self._get_nested_value(execution.execution_context, path)

        return value

    def _get_nested_value(self, data: Dict[str, Any], path: str) -> Any:
        """Get nested value from dictionary using dot notation"""

        keys = path.split('.')
        current = data

        for key in keys:
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                return None

        return current

    def _evaluate_condition(self, left: Any, operator: ConditionOperator, right: Any) -> bool:
        """Evaluate a condition"""

        if operator == ConditionOperator.EQUALS:
            return left == right
        elif operator == ConditionOperator.NOT_EQUALS:
            return left != right
        elif operator == ConditionOperator.GREATER_THAN:
            return left > right
        elif operator == ConditionOperator.LESS_THAN:
            return left < right
        elif operator == ConditionOperator.CONTAINS:
            return str(right) in str(left)
        elif operator == ConditionOperator.NOT_CONTAINS:
            return str(right) not in str(left)
        elif operator == ConditionOperator.STARTS_WITH:
            return str(left).startswith(str(right))
        elif operator == ConditionOperator.ENDS_WITH:
            return str(left).endswith(str(right))
        elif operator == ConditionOperator.IS_EMPTY:
            return not left or left == ""
        elif operator == ConditionOperator.IS_NOT_EMPTY:
            return bool(left) and left != ""
        elif operator == ConditionOperator.IN_LIST:
            return left in right if isinstance(right, list) else False
        elif operator == ConditionOperator.NOT_IN_LIST:
            return left not in right if isinstance(right, list) else True

        return False

    async def _evaluate_connection_condition(self, condition: Dict[str, Any],
                                           execution_result: Any) -> bool:
        """Evaluate connection condition"""

        operator = ConditionOperator(condition.get('operator'))
        left_value = condition.get('left', execution_result)
        right_value = condition.get('right')

        return self._evaluate_condition(left_value, operator, right_value)

    async def _evaluate_filter_condition(self, condition: Dict[str, Any], item: Any) -> bool:
        """Evaluate filter condition for data transformation"""

        operator = ConditionOperator(condition.get('operator'))
        field = condition.get('field')
        value = condition.get('value')

        item_value = item.get(field) if isinstance(item, dict) else item

        return self._evaluate_condition(item_value, operator, value)

    async def _validate_workflow(self, workflow: Workflow) -> Dict[str, Any]:
        """Validate workflow before activation"""

        errors = []
        warnings = []

        # Check for trigger nodes
        trigger_nodes = [node for node in workflow.nodes if node.node_type == "trigger"]
        if not trigger_nodes:
            errors.append("Workflow must have at least one trigger node")
        elif len(trigger_nodes) > 1:
            warnings.append("Multiple trigger nodes detected")

        # Check for orphaned nodes (no connections)
        connected_node_ids = set()
        for connection in workflow.connections:
            connected_node_ids.add(connection.source_node_id)
            connected_node_ids.add(connection.target_node_id)

        all_node_ids = {node.node_id for node in workflow.nodes}
        orphaned_nodes = all_node_ids - connected_node_ids

        if len(orphaned_nodes) > 1:  # Allow one orphaned node (trigger)
            warnings.append(f"Found {len(orphaned_nodes)} orphaned nodes")

        # Validate node configurations
        for node in workflow.nodes:
            node_errors = await self._validate_node_config(node)
            errors.extend(node_errors)

        return {
            'valid': len(errors) == 0,
            'errors': errors,
            'warnings': warnings
        }

    async def _validate_node_config(self, node: WorkflowNode) -> List[str]:
        """Validate individual node configuration"""

        errors = []
        config = node.config

        if node.node_type == "trigger":
            if 'trigger_type' not in config:
                errors.append(f"Trigger node '{node.name}' missing trigger_type")

        elif node.node_type == "action":
            if 'action_type' not in config:
                errors.append(f"Action node '{node.name}' missing action_type")

            action_type = config.get('action_type')

            # Validate specific action requirements
            if action_type == 'send_email':
                required_fields = ['to', 'subject', 'body']
                for field in required_fields:
                    if field not in config:
                        errors.append(f"Email action '{node.name}' missing {field}")

            elif action_type == 'http_request':
                if 'url' not in config:
                    errors.append(f"HTTP request action '{node.name}' missing url")

        elif node.node_type == "condition":
            if 'condition' not in config:
                errors.append(f"Condition node '{node.name}' missing condition config")

        return errors

    async def _setup_workflow_triggers(self, workflow: Workflow):
        """Set up triggers for active workflow"""

        trigger_nodes = [node for node in workflow.nodes if node.node_type == "trigger"]

        for trigger_node in trigger_nodes:
            trigger_type = TriggerType(trigger_node.config.get('trigger_type'))

            if trigger_type == TriggerType.SCHEDULE:
                await self._setup_schedule_trigger(workflow, trigger_node)
            elif trigger_type == TriggerType.WEBHOOK:
                await self._setup_webhook_trigger(workflow, trigger_node)

    async def _setup_schedule_trigger(self, workflow: Workflow, trigger_node: WorkflowNode):
        """Set up scheduled trigger"""

        # This would integrate with a scheduler like Celery or APScheduler
        schedule_config = trigger_node.config.get('schedule', {})

        # Store schedule in database for background processing
        async with self.db_pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO workflows.workflow_schedules (
                    workflow_id, trigger_node_id, schedule_config, is_active
                ) VALUES ($1, $2, $3, $4)
                ON CONFLICT (workflow_id, trigger_node_id)
                DO UPDATE SET schedule_config = $3, is_active = $4
            """, workflow.workflow_id, trigger_node.node_id,
                json.dumps(schedule_config), True)

    async def _setup_webhook_trigger(self, workflow: Workflow, trigger_node: WorkflowNode):
        """Set up webhook trigger"""

        webhook_config = trigger_node.config.get('webhook', {})
        webhook_url = f"/api/v1/workflows/{workflow.workflow_id}/webhook/{trigger_node.node_id}"

        # Store webhook configuration
        async with self.db_pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO workflows.workflow_webhooks (
                    workflow_id, trigger_node_id, webhook_url, webhook_config, is_active
                ) VALUES ($1, $2, $3, $4, $5)
                ON CONFLICT (workflow_id, trigger_node_id)
                DO UPDATE SET webhook_url = $3, webhook_config = $4, is_active = $5
            """, workflow.workflow_id, trigger_node.node_id, webhook_url,
                json.dumps(webhook_config), True)

    # Database operations
    async def _store_workflow(self, workflow: Workflow):
        """Store workflow in database"""

        async with self.db_pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO workflows.workflows (
                    workflow_id, tenant_id, name, description, status,
                    nodes, connections, variables, tags, folder_id, is_template,
                    execution_count, last_executed_at, created_by, created_at, updated_at
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16)
            """,
                workflow.workflow_id, workflow.tenant_id, workflow.name, workflow.description,
                workflow.status.value, json.dumps([asdict(node) for node in workflow.nodes]),
                json.dumps([asdict(conn) for conn in workflow.connections]),
                json.dumps(workflow.variables), workflow.tags, workflow.folder_id,
                workflow.is_template, workflow.execution_count, workflow.last_executed_at,
                workflow.created_by, workflow.created_at, workflow.updated_at
            )

    async def _update_workflow(self, workflow: Workflow):
        """Update workflow in database"""

        async with self.db_pool.acquire() as conn:
            await conn.execute("""
                UPDATE workflows.workflows SET
                    name = $2, description = $3, status = $4, nodes = $5, connections = $6,
                    variables = $7, tags = $8, folder_id = $9, execution_count = $10,
                    last_executed_at = $11, updated_at = $12
                WHERE workflow_id = $1
            """,
                workflow.workflow_id, workflow.name, workflow.description, workflow.status.value,
                json.dumps([asdict(node) for node in workflow.nodes]),
                json.dumps([asdict(conn) for conn in workflow.connections]),
                json.dumps(workflow.variables), workflow.tags, workflow.folder_id,
                workflow.execution_count, workflow.last_executed_at, workflow.updated_at
            )

    async def _get_workflow(self, workflow_id: str) -> Optional[Workflow]:
        """Get workflow from database"""

        async with self.db_pool.acquire() as conn:
            row = await conn.fetchrow("""
                SELECT * FROM workflows.workflows WHERE workflow_id = $1
            """, workflow_id)

            if row:
                # Deserialize nodes and connections
                nodes = [WorkflowNode(**node_data) for node_data in json.loads(row['nodes'])]
                connections = [WorkflowConnection(**conn_data) for conn_data in json.loads(row['connections'])]

                return Workflow(
                    workflow_id=row['workflow_id'],
                    tenant_id=row['tenant_id'],
                    name=row['name'],
                    description=row['description'],
                    status=WorkflowStatus(row['status']),
                    nodes=nodes,
                    connections=connections,
                    variables=json.loads(row['variables']) if row['variables'] else {},
                    tags=row['tags'] or [],
                    folder_id=row['folder_id'],
                    is_template=row['is_template'],
                    execution_count=row['execution_count'],
                    last_executed_at=row['last_executed_at'],
                    created_by=row['created_by'],
                    created_at=row['created_at'],
                    updated_at=row['updated_at']
                )

        return None

    async def _store_execution(self, execution: WorkflowExecution):
        """Store workflow execution in database"""

        async with self.db_pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO workflows.workflow_executions (
                    execution_id, workflow_id, tenant_id, status, trigger_data,
                    execution_context, current_node_id, error_message, execution_time_ms,
                    nodes_executed, started_at, completed_at
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
            """,
                execution.execution_id, execution.workflow_id, execution.tenant_id,
                execution.status, json.dumps(execution.trigger_data),
                json.dumps(execution.execution_context), execution.current_node_id,
                execution.error_message, execution.execution_time_ms, execution.nodes_executed,
                execution.started_at, execution.completed_at
            )

    async def _update_execution(self, execution: WorkflowExecution):
        """Update workflow execution in database"""

        async with self.db_pool.acquire() as conn:
            await conn.execute("""
                UPDATE workflows.workflow_executions SET
                    status = $2, execution_context = $3, current_node_id = $4,
                    error_message = $5, execution_time_ms = $6, nodes_executed = $7,
                    completed_at = $8
                WHERE execution_id = $1
            """,
                execution.execution_id, execution.status, json.dumps(execution.execution_context),
                execution.current_node_id, execution.error_message, execution.execution_time_ms,
                execution.nodes_executed, execution.completed_at
            )

# Export main classes
__all__ = [
    'WorkflowEngine', 'Workflow', 'WorkflowNode', 'WorkflowConnection', 'WorkflowExecution',
    'WorkflowStatus', 'TriggerType', 'ActionType', 'ConditionOperator'
]
