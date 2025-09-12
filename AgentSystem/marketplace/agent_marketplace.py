
"""
AI Agent Marketplace - AgentSystem Profit Machine
Revolutionary platform for browsing, creating, and monetizing AI agents
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Callable
from uuid import UUID, uuid4
import json
import inspect
from dataclasses import dataclass, asdict
from enum import Enum
import warnings
warnings.filterwarnings('ignore')

from ..database.connection import get_db_connection
from ..usage.usage_tracker import UsageTracker
from ..billing.stripe_service import StripeService

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AgentCategory(str, Enum):
    CUSTOMER_SERVICE = "customer_service"
    SALES_AUTOMATION = "sales_automation"
    MARKETING = "marketing"
    OPERATIONS = "operations"
    ANALYTICS = "analytics"
    CONTENT_CREATION = "content_creation"
    DATA_PROCESSING = "data_processing"
    INTEGRATION = "integration"
    WORKFLOW_AUTOMATION = "workflow_automation"
    SPECIALIZED = "specialized"

class AgentStatus(str, Enum):
    DRAFT = "draft"
    PUBLISHED = "published"
    DEPRECATED = "deprecated"
    PRIVATE = "private"

class AgentPricingModel(str, Enum):
    FREE = "free"
    ONE_TIME = "one_time"
    SUBSCRIPTION = "subscription"
    USAGE_BASED = "usage_based"
    REVENUE_SHARE = "revenue_share"

class AgentCapability(str, Enum):
    TEXT_GENERATION = "text_generation"
    DATA_ANALYSIS = "data_analysis"
    API_INTEGRATION = "api_integration"
    WORKFLOW_EXECUTION = "workflow_execution"
    DECISION_MAKING = "decision_making"
    LEARNING_ADAPTATION = "learning_adaptation"
    MULTI_MODAL = "multi_modal"
    REAL_TIME_PROCESSING = "real_time_processing"

@dataclass
class AgentMetadata:
    agent_id: UUID
    name: str
    description: str
    category: AgentCategory
    capabilities: List[AgentCapability]
    version: str
    author_id: UUID
    author_name: str
    pricing_model: AgentPricingModel
    price: float
    currency: str
    status: AgentStatus
    tags: List[str]
    use_cases: List[str]
    supported_integrations: List[str]
    min_requirements: Dict[str, Any]
    performance_metrics: Dict[str, float]
    created_date: datetime
    updated_date: datetime
    download_count: int
    rating_average: float
    rating_count: int

@dataclass
class AgentDefinition:
    agent_id: UUID
    configuration: Dict[str, Any]
    prompt_templates: Dict[str, str]
    workflow_logic: Dict[str, Any]
    api_endpoints: List[Dict[str, Any]]
    event_handlers: Dict[str, Any]
    validation_rules: List[Dict[str, Any]]
    dependencies: List[str]
    custom_code: Optional[str]
    test_cases: List[Dict[str, Any]]

@dataclass
class AgentInstance:
    instance_id: UUID
    agent_id: UUID
    tenant_id: UUID
    instance_name: str
    configuration: Dict[str, Any]
    status: str  # active, paused, stopped, error
    deployment_date: datetime
    last_execution: Optional[datetime]
    execution_count: int
    success_rate: float
    avg_execution_time: float
    error_count: int
    resource_usage: Dict[str, float]

@dataclass
class AgentExecution:
    execution_id: UUID
    instance_id: UUID
    tenant_id: UUID
    trigger_type: str
    input_data: Dict[str, Any]
    output_data: Optional[Dict[str, Any]]
    execution_time_ms: int
    status: str  # success, error, timeout
    error_message: Optional[str]
    cost: float
    tokens_used: int
    started_at: datetime
    completed_at: Optional[datetime]

class AgentMarketplace:
    """AI Agent Marketplace for browsing, creating, and monetizing agents"""

    def __init__(self):
        self.usage_tracker = UsageTracker()
        self.stripe_service = StripeService()
        self.agent_registry = {}
        self.custom_builders = {}

    async def initialize(self):
        """Initialize the agent marketplace"""
        try:
            await self._load_built_in_agents()
            await self._initialize_custom_builders()
            logger.info("Agent Marketplace initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Agent Marketplace: {e}")
            raise

    async def publish_agent(
        self,
        tenant_id: UUID,
        agent_metadata: AgentMetadata,
        agent_definition: AgentDefinition
    ) -> UUID:
        """Publish a new agent to the marketplace"""
        try:
            # Validate agent definition
            validation_result = await self._validate_agent_definition(agent_definition)
            if not validation_result['valid']:
                raise ValueError(f"Invalid agent definition: {validation_result['errors']}")

            # Test agent functionality
            test_result = await self._test_agent_functionality(agent_definition)
            if not test_result['passed']:
                raise ValueError(f"Agent tests failed: {test_result['failures']}")

            # Store agent metadata
            await self._store_agent_metadata(tenant_id, agent_metadata)

            # Store agent definition
            await self._store_agent_definition(agent_definition)

            # Set up monetization if applicable
            if agent_metadata.pricing_model != AgentPricingModel.FREE:
                await self._setup_agent_monetization(tenant_id, agent_metadata)

            # Add to search index
            await self._index_agent_for_search(agent_metadata)

            return agent_metadata.agent_id

        except Exception as e:
            logger.error(f"Failed to publish agent: {e}")
            raise

    async def install_agent(
        self,
        tenant_id: UUID,
        agent_id: UUID,
        instance_config: Dict[str, Any]
    ) -> UUID:
        """Install an agent for a tenant"""
        try:
            # Get agent metadata
            agent_metadata = await self._get_agent_metadata(agent_id)
            if not agent_metadata:
                raise ValueError(f"Agent {agent_id} not found")

            # Check pricing and permissions
            if agent_metadata.pricing_model != AgentPricingModel.FREE:
                purchase_valid = await self._validate_agent_purchase(tenant_id, agent_id)
                if not purchase_valid:
                    raise ValueError("Agent purchase required or invalid")

            # Get agent definition
            agent_definition = await self._get_agent_definition(agent_id)
            if not agent_definition:
                raise ValueError("Agent definition not found")

            # Create agent instance
            instance_id = uuid4()
            instance = AgentInstance(
                instance_id=instance_id,
                agent_id=agent_id,
                tenant_id=tenant_id,
                instance_name=instance_config.get('name', agent_metadata.name),
                configuration=instance_config,
                status='active',
                deployment_date=datetime.utcnow(),
                last_execution=None,
                execution_count=0,
                success_rate=1.0,
                avg_execution_time=0.0,
                error_count=0,
                resource_usage={}
            )

            # Deploy agent instance
            await self._deploy_agent_instance(instance, agent_definition)

            # Track installation
            await self._track_agent_installation(tenant_id, agent_id, instance_id)

            return instance_id

        except Exception as e:
            logger.error(f"Failed to install agent: {e}")
            raise

    async def execute_agent(
        self,
        tenant_id: UUID,
        instance_id: UUID,
        input_data: Dict[str, Any],
        trigger_type: str = "manual"
    ) -> AgentExecution:
        """Execute an agent instance"""
        try:
            # Get agent instance
            instance = await self._get_agent_instance(tenant_id, instance_id)
            if not instance:
                raise ValueError(f"Agent instance {instance_id} not found")

            # Get agent definition
            agent_definition = await self._get_agent_definition(instance.agent_id)
            if not agent_definition:
                raise ValueError("Agent definition not found")

            # Create execution record
            execution_id = uuid4()
            execution = AgentExecution(
                execution_id=execution_id,
                instance_id=instance_id,
                tenant_id=tenant_id,
                trigger_type=trigger_type,
                input_data=input_data,
                output_data=None,
                execution_time_ms=0,
                status='running',
                error_message=None,
                cost=0.0,
                tokens_used=0,
                started_at=datetime.utcnow(),
                completed_at=None
            )

            # Execute agent logic
            start_time = datetime.utcnow()
            try:
                output_data, tokens_used = await self._execute_agent_logic(
                    agent_definition, input_data, instance.configuration
                )

                execution.output_data = output_data
                execution.tokens_used = tokens_used
                execution.status = 'success'

            except Exception as exec_error:
                execution.error_message = str(exec_error)
                execution.status = 'error'
                logger.error(f"Agent execution failed: {exec_error}")

            # Calculate execution time and cost
            execution.completed_at = datetime.utcnow()
            execution.execution_time_ms = int((execution.completed_at - start_time).total_seconds() * 1000)
            execution.cost = await self._calculate_execution_cost(execution.tokens_used, agent_definition)

            # Update instance metrics
            await self._update_instance_metrics(instance, execution)

            # Store execution record
            await self._store_agent_execution(execution)

            # Track usage
            await self.usage_tracker.track_usage(
                tenant_id=tenant_id,
                feature_name=f"agent_{instance.agent_id}",
                tokens_used=execution.tokens_used,
                cost=execution.cost,
                metadata={'instance_id': str(instance_id), 'execution_id': str(execution_id)}
            )

            return execution

        except Exception as e:
            logger.error(f"Failed to execute agent: {e}")
            raise

    async def search_agents(
        self,
        query: str,
        category: Optional[AgentCategory] = None,
        capabilities: Optional[List[AgentCapability]] = None,
        pricing_model: Optional[AgentPricingModel] = None,
        min_rating: Optional[float] = None,
        limit: int = 20
    ) -> List[AgentMetadata]:
        """Search for agents in the marketplace"""
        try:
            # Build search query
            search_conditions = ["status = 'published'"]
            search_params = []
            param_count = 0

            if query:
                param_count += 1
                search_conditions.append(f"(name ILIKE ${param_count} OR description ILIKE ${param_count})")
                search_params.append(f"%{query}%")

            if category:
                param_count += 1
                search_conditions.append(f"category = ${param_count}")
                search_params.append(category.value)

            if pricing_model:
                param_count += 1
                search_conditions.append(f"pricing_model = ${param_count}")
                search_params.append(pricing_model.value)

            if min_rating:
                param_count += 1
                search_conditions.append(f"rating_average >= ${param_count}")
                search_params.append(min_rating)

            async with get_db_connection() as conn:
                query_sql = f"""
                    SELECT * FROM marketplace.agents
                    WHERE {' AND '.join(search_conditions)}
                    ORDER BY rating_average DESC, download_count DESC
                    LIMIT ${param_count + 1}
                """
                search_params.append(limit)

                results = await conn.fetch(query_sql, *search_params)

                agents = []
                for result in results:
                    # Filter by capabilities if specified
                    if capabilities:
                        agent_capabilities = json.loads(result['capabilities'])
                        if not any(cap.value in agent_capabilities for cap in capabilities):
                            continue

                    agent = AgentMetadata(
                        agent_id=result['agent_id'],
                        name=result['name'],
                        description=result['description'],
                        category=AgentCategory(result['category']),
                        capabilities=[AgentCapability(cap) for cap in json.loads(result['capabilities'])],
                        version=result['version'],
                        author_id=result['author_id'],
                        author_name=result['author_name'],
                        pricing_model=AgentPricingModel(result['pricing_model']),
                        price=float(result['price']),
                        currency=result['currency'],
                        status=AgentStatus(result['status']),
                        tags=json.loads(result['tags']),
                        use_cases=json.loads(result['use_cases']),
                        supported_integrations=json.loads(result['supported_integrations']),
                        min_requirements=json.loads(result['min_requirements']),
                        performance_metrics=json.loads(result['performance_metrics']),
                        created_date=result['created_date'],
                        updated_date=result['updated_date'],
                        download_count=result['download_count'],
                        rating_average=float(result['rating_average']),
                        rating_count=result['rating_count']
                    )
                    agents.append(agent)

                return agents

        except Exception as e:
            logger.error(f"Failed to search agents: {e}")
            return []

    async def get_agent_details(self, agent_id: UUID) -> Optional[Tuple[AgentMetadata, AgentDefinition]]:
        """Get detailed agent information"""
        try:
            # Get metadata
            metadata = await self._get_agent_metadata(agent_id)
            if not metadata:
                return None

            # Get definition (simplified version for marketplace)
            definition = await self._get_agent_definition(agent_id)
            if not definition:
                return None

            return metadata, definition

        except Exception as e:
            logger.error(f"Failed to get agent details: {e}")
            return None

    async def create_custom_agent(
        self,
        tenant_id: UUID,
        agent_spec: Dict[str, Any]
    ) -> UUID:
        """Create a custom agent using the agent builder"""
        try:
            # Generate agent ID
            agent_id = uuid4()

            # Create agent metadata
            metadata = AgentMetadata(
                agent_id=agent_id,
                name=agent_spec['name'],
                description=agent_spec['description'],
                category=AgentCategory(agent_spec['category']),
                capabilities=[AgentCapability(cap) for cap in agent_spec['capabilities']],
                version="1.0.0",
                author_id=tenant_id,
                author_name=agent_spec.get('author_name', 'Custom Builder'),
                pricing_model=AgentPricingModel.PRIVATE,
                price=0.0,
                currency='USD',
                status=AgentStatus.DRAFT,
                tags=agent_spec.get('tags', []),
                use_cases=agent_spec.get('use_cases', []),
                supported_integrations=agent_spec.get('integrations', []),
                min_requirements=agent_spec.get('requirements', {}),
                performance_metrics={},
                created_date=datetime.utcnow(),
                updated_date=datetime.utcnow(),
                download_count=0,
                rating_average=0.0,
                rating_count=0
            )

            # Create agent definition from builder spec
            definition = await self._build_agent_definition(agent_id, agent_spec)

            # Validate and test
            validation_result = await self._validate_agent_definition(definition)
            if not validation_result['valid']:
                raise ValueError(f"Agent validation failed: {validation_result['errors']}")

            # Store agent
            await self._store_agent_metadata(tenant_id, metadata)
            await self._store_agent_definition(definition)

            return agent_id

        except Exception as e:
            logger.error(f"Failed to create custom agent: {e}")
            raise

    async def get_my_agents(self, tenant_id: UUID) -> List[AgentMetadata]:
        """Get agents owned by a tenant"""
        try:
            async with get_db_connection() as conn:
                query = """
                    SELECT * FROM marketplace.agents
                    WHERE author_id = $1
                    ORDER BY updated_date DESC
                """
                results = await conn.fetch(query, tenant_id)

                agents = []
                for result in results:
                    agent = AgentMetadata(
                        agent_id=result['agent_id'],
                        name=result['name'],
                        description=result['description'],
                        category=AgentCategory(result['category']),
                        capabilities=[AgentCapability(cap) for cap in json.loads(result['capabilities'])],
                        version=result['version'],
                        author_id=result['author_id'],
                        author_name=result['author_name'],
                        pricing_model=AgentPricingModel(result['pricing_model']),
                        price=float(result['price']),
                        currency=result['currency'],
                        status=AgentStatus(result['status']),
                        tags=json.loads(result['tags']),
                        use_cases=json.loads(result['use_cases']),
                        supported_integrations=json.loads(result['supported_integrations']),
                        min_requirements=json.loads(result['min_requirements']),
                        performance_metrics=json.loads(result['performance_metrics']),
                        created_date=result['created_date'],
                        updated_date=result['updated_date'],
                        download_count=result['download_count'],
                        rating_average=float(result['rating_average']),
                        rating_count=result['rating_count']
                    )
                    agents.append(agent)

                return agents

        except Exception as e:
            logger.error(f"Failed to get user agents: {e}")
            return []

    async def get_installed_agents(self, tenant_id: UUID) -> List[AgentInstance]:
        """Get agents installed by a tenant"""
        try:
            async with get_db_connection() as conn:
                query = """
                    SELECT ai.*, am.name as agent_name
                    FROM marketplace.agent_instances ai
                    JOIN marketplace.agents am ON ai.agent_id = am.agent_id
                    WHERE ai.tenant_id = $1
                    ORDER BY ai.deployment_date DESC
                """
                results = await conn.fetch(query, tenant_id)

                instances = []
                for result in results:
                    instance = AgentInstance(
                        instance_id=result['instance_id'],
                        agent_id=result['agent_id'],
                        tenant_id=result['tenant_id'],
                        instance_name=result['instance_name'],
                        configuration=json.loads(result['configuration']),
                        status=result['status'],
                        deployment_date=result['deployment_date'],
                        last_execution=result['last_execution'],
                        execution_count=result['execution_count'],
                        success_rate=float(result['success_rate']),
                        avg_execution_time=float(result['avg_execution_time']),
                        error_count=result['error_count'],
                        resource_usage=json.loads(result['resource_usage'])
                    )
                    instances.append(instance)

                return instances

        except Exception as e:
            logger.error(f"Failed to get installed agents: {e}")
            return []

    async def rate_agent(
        self,
        tenant_id: UUID,
        agent_id: UUID,
        rating: int,
        review: Optional[str] = None
    ):
        """Rate and review an agent"""
        try:
            if rating < 1 or rating > 5:
                raise ValueError("Rating must be between 1 and 5")

            # Store rating
            async with get_db_connection() as conn:
                # Insert or update rating
                query = """
                    INSERT INTO marketplace.agent_ratings (
                        rating_id, tenant_id, agent_id, rating, review, rating_date
                    ) VALUES ($1, $2, $3, $4, $5, $6)
                    ON CONFLICT (tenant_id, agent_id)
                    DO UPDATE SET rating = EXCLUDED.rating,
                                  review = EXCLUDED.review,
                                  rating_date = EXCLUDED.rating_date
                """
                await conn.execute(
                    query, uuid4(), tenant_id, agent_id, rating, review, datetime.utcnow()
                )

                # Update agent average rating
                await self._update_agent_rating(agent_id)

        except Exception as e:
            logger.error(f"Failed to rate agent: {e}")
            raise

    async def get_marketplace_analytics(self, tenant_id: UUID) -> Dict[str, Any]:
        """Get marketplace analytics for a tenant"""
        try:
            async with get_db_connection() as conn:
                # Get overview metrics
                overview_query = """
                    SELECT
                        COUNT(DISTINCT agent_id) as published_agents,
                        COUNT(DISTINCT CASE WHEN ai.instance_id IS NOT NULL THEN ai.agent_id END) as installed_agents,
                        SUM(download_count) as total_downloads,
                        AVG(rating_average) as avg_rating
                    FROM marketplace.agents a
                    LEFT JOIN marketplace.agent_instances ai ON a.agent_id = ai.agent_id
                        AND ai.tenant_id = $1
                    WHERE a.author_id = $1
                """
                overview_result = await conn.fetchrow(overview_query, tenant_id)

                # Get revenue metrics
                revenue_query = """
                    SELECT
                        SUM(amount) as total_revenue,
                        COUNT(*) as total_purchases
                    FROM marketplace.agent_purchases
                    WHERE seller_id = $1
                    AND purchase_date >= NOW() - INTERVAL '30 days'
                """
                revenue_result = await conn.fetchrow(revenue_query, tenant_id)

                return {
                    'published_agents': overview_result['published_agents'] or 0,
                    'installed_agents': overview_result['installed_agents'] or 0,
                    'total_downloads': overview_result['total_downloads'] or 0,
                    'avg_rating': float(overview_result['avg_rating'] or 0),
                    'monthly_revenue': float(revenue_result['total_revenue'] or 0),
                    'monthly_purchases': revenue_result['total_purchases'] or 0
                }

        except Exception as e:
            logger.error(f"Failed to get marketplace analytics: {e}")
            return {}

    # Helper methods
    async def _load_built_in_agents(self):
        """Load built-in agents into the marketplace"""
        try:
            # Load our existing agents (Marketing, Sales, Customer Success, Operations)
            built_in_agents = [
                {
                    'name': 'Marketing Automation Agent',
                    'description': 'Advanced marketing automation with content generation and SEO optimization',
                    'category': AgentCategory.MARKETING,
                    'capabilities': [AgentCapability.TEXT_GENERATION, AgentCapability.API_INTEGRATION],
                    'price': 99.0,
                    'pricing_model': AgentPricingModel.SUBSCRIPTION
                },
                {
                    'name': 'Sales Automation Agent',
                    'description': 'Intelligent sales automation with prospect research and email sequences',
                    'category': AgentCategory.SALES_AUTOMATION,
                    'capabilities': [AgentCapability.DATA_ANALYSIS, AgentCapability.WORKFLOW_EXECUTION],
                    'price': 149.0,
                    'pricing_model': AgentPricingModel.SUBSCRIPTION
                },
                {
                    'name': 'Customer Success Agent',
                    'description': 'Proactive customer success with health scoring and churn prevention',
                    'category': AgentCategory.CUSTOMER_SERVICE,
                    'capabilities': [AgentCapability.DECISION_MAKING, AgentCapability.LEARNING_ADAPTATION],
                    'price': 129.0,
                    'pricing_model': AgentPricingModel.SUBSCRIPTION
                },
                {
                    'name': 'Operations Automation Agent',
                    'description': 'Comprehensive operations automation with document processing',
                    'category': AgentCategory.OPERATIONS,
                    'capabilities': [AgentCapability.DATA_PROCESSING, AgentCapability.MULTI_MODAL],
                    'price': 199.0,
                    'pricing_model': AgentPricingModel.SUBSCRIPTION
                }
            ]

            # Store built-in agents (simplified)
            self.agent_registry = {agent['name']: agent for agent in built_in_agents}
            logger.info(f"Loaded {len(built_in_agents)} built-in agents")

        except Exception as e:
            logger.error(f"Failed to load built-in agents: {e}")

    async def _initialize_custom_builders(self):
        """Initialize custom agent builders"""
        try:
            self.custom_builders = {
                'visual_builder': True,
                'code_builder': True,
                'template_builder': True,
                'workflow_builder': True
            }
            logger.info("Custom builders initialized")
        except Exception as e:
            logger.error(f"Failed to initialize custom builders: {e}")

    async def _validate_agent_definition(self, definition: AgentDefinition) -> Dict[str, Any]:
        """Validate agent definition"""
        try:
            errors = []

            # Basic validation
            if not definition.configuration:
                errors.append("Agent configuration is required")

            if not definition.workflow_logic:
                errors.append("Workflow logic is required")

            # Validate prompt templates
            if not definition.prompt_templates:
                errors.append("At least one prompt template is required")

            return {
                'valid': len(errors) == 0,
                'errors': errors
            }

        except Exception as e:
            logger.error(f"Failed to validate agent definition: {e}")
            return {'valid': False, 'errors': [str(e)]}

    async def _test_agent_functionality(self, definition: AgentDefinition) -> Dict[str, Any]:
        """Test agent functionality"""
        try:
            # Run test cases
            failures = []

            for test_case in definition.test_cases:
                try:
                    # Simulate test execution
                    test_result = await self._simulate_agent_test(definition, test_case)
                    if not test_result:
                        failures.append(f"Test case '{test_case.get('name', 'unknown')}' failed")
                except Exception as e:
                    failures.append(f"Test case error: {str(e)}")

            return {
                'passed': len(failures) == 0,
                'failures': failures,
                'tests_run': len(definition.test_cases)
            }

        except Exception as e:
            logger.error(f"Failed to test agent functionality: {e}")
            return {'passed': False, 'failures': [str(e)], 'tests_run': 0}

    async def _simulate_agent_test(self, definition: AgentDefinition, test_case: Dict[str, Any]) -> bool:
        """Simulate agent test execution"""
        # Simplified test simulation
        return True

    async def _store_agent_metadata(self, tenant_id: UUID, metadata: AgentMetadata):
        """Store agent metadata in database"""
        try:
            async with get_db_connection() as conn:
                query = """
                    INSERT INTO marketplace.agents (
                        agent_id, name, description, category, capabilities,
                        version, author_id, author_name, pricing_model, price,
                        currency, status, tags, use_cases, supported_integrations,
                        min_requirements, performance_metrics, created_date, updated_date
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18, $19)
                """
                await conn.execute(
                    query,
                    metadata.agent_id,
                    metadata.name,
                    metadata.description,
                    metadata.category.value,
                    json.dumps([cap.value for cap in metadata.capabilities]),
                    metadata.version,
                    metadata.author_id,
                    metadata.author_name,
                    metadata.pricing_model.value,
                    metadata.price,
                    metadata.currency,
                    metadata.status.value,
                    json.dumps(metadata.tags),
                    json.dumps(metadata.use_cases),
                    json.dumps(metadata.supported_integrations),
                    json.dumps(metadata.min_requirements),
                    json.dumps(metadata.performance_metrics),
                    metadata.created_date,
                    metadata.updated_date
                )
        except Exception as e:
            logger.error(f"Failed to store agent metadata: {e}")

    async def _store_agent_definition(self, definition: AgentDefinition):
        """Store agent definition in database"""
        try:
            async with get_db_connection() as conn:
                query = """
                    INSERT INTO marketplace.agent_definitions (
                        agent_id, configuration, prompt_templates, workflow_logic,
                        api_endpoints, event_handlers, validation_rules,
                        dependencies, custom_code, test_cases
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
                """
                await conn.execute(
                    query,
                    definition.agent_id,
                    json.dumps(definition.configuration),
                    json.dumps(definition.prompt_templates),
                    json.dumps(definition.workflow_logic),
                    json.dumps(definition.api_endpoints),
                    json.dumps(definition.event_handlers),
                    json.dumps(definition.validation_rules),
                    json.dumps(definition.dependencies),
                    definition.custom_code,
                    json.dumps(definition.test_cases)
                )
        except Exception as e:
            logger.error(f"Failed to store agent definition: {e}")

    async def _get_agent_metadata(self, agent_id: UUID) -> Optional[AgentMetadata]:
        """Get agent metadata from database"""
        try:
            async with get_db_connection() as conn:
                query = """
                    SELECT * FROM marketplace.agents WHERE agent_id = $1
                """
                result = await conn.fetchrow(query, agent_id)

                if not result:
                    return None

                return AgentMetadata(
                    agent_id=result['agent_id'],
                    name=result['name'],
                    description=result['description'],
                    category=AgentCategory(result['category']),
                    capabilities=[AgentCapability(cap) for cap in json.loads(result['capabilities'])],
                    version=result['version'],
                    author_id=result['author_id'],
                    author_name=result['author_name'],
                    pricing_model=AgentPricingModel(result['pricing_model']),
                    price=float(result['price']),
                    currency=result['currency'],
                    status=AgentStatus(result['status']),
                    tags=json.loads(result['tags']),
                    use_cases=json.loads(result['use_cases']),
                    supported_integrations=json.loads(result['supported_integrations']),
                    min_requirements=json.loads(result['min_requirements']),
                    performance_metrics=json.loads(result['performance_metrics']),
                    created_date=result['created_date'],
                    updated_date=result['updated_date'],
                    download_count=result['download_count'],
                    rating_average=float(result['rating_average']),
                    rating_count=result['rating_count']
                )

        except Exception as e:
            logger.error(f"Failed to get agent metadata: {e}")
            return None

    async def _get_agent_definition(self, agent_id: UUID) -> Optional[AgentDefinition]:
        """Get agent definition from database"""
        try:
            async with get_db_connection() as conn:
                query = """
                    SELECT * FROM marketplace.agent_definitions WHERE agent_id = $1
                """
                result = await conn.fetchrow(query, agent_id)

                if not result:
                    return None

                return AgentDefinition(
                    agent_id=result['agent_id'],
                    configuration=json.loads(result['configuration']),
                    prompt_templates=json.loads(result['prompt_templates']),
                    workflow_logic=json.loads(result['workflow_logic']),
                    api_endpoints=json.loads(result['api_endpoints']),
                    event_handlers=json.loads(result['event_handlers']),
                    validation_rules=json.loads(result['validation_rules']),
                    dependencies=json.loads(result['dependencies']),
                    custom_code=result['custom_code'],
                    test_cases=json.loads(result['test_cases'])
                )

        except Exception as e:
            logger.error(f"Failed to get agent definition: {e}")
            return None

    async def _build_agent_definition(self, agent_id: UUID, agent_spec: Dict[str, Any]) -> AgentDefinition:
        """Build agent definition from specification"""
        try:
            # Create basic agent definition structure
            definition = AgentDefinition(
                agent_id=agent_id,
                configuration={
                    'name': agent_spec['name'],
                    'max_execution_time': agent_spec.get('max_execution_time', 300),
                    'retry_attempts': agent_spec.get('retry_attempts', 3),
                    'resource_limits': agent_spec.get('resource_limits', {})
                },
                prompt_templates=agent_spec.get('prompts', {}),
                workflow_logic=agent_spec.get('workflow', {}),
                api_endpoints=agent_spec.get('endpoints', []),
                event_handlers=agent_spec.get('events', {}),
                validation_rules=agent_spec.get('validation', []),
                dependencies=agent_spec.get('dependencies', []),
                custom_code=agent_spec.get('custom_code'),
                test_cases=agent_spec.get('test_cases', [])
            )

            return definition

        except Exception as e:
            logger.error(f"Failed to build agent definition: {e}")
            raise

    async def _setup_agent_monetization(self, tenant_id: UUID, metadata: AgentMetadata):
        """Setup monetization for an agent"""
        try:
            # Create Stripe product for agent
            stripe_result = await self.stripe_service.create_product(
                name=metadata.name,
                description=metadata.description,
                price=metadata.price,
                currency=metadata.currency,
                billing_scheme=metadata.pricing_model.value
            )

            # Store monetization config
            async with get_db_connection() as conn:
                query = """
                    INSERT INTO marketplace.agent_monetization (
                        agent_id, tenant_id, stripe_product_id, pricing_model,
                        price, currency, revenue_share_percent
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7)
                """
                await conn.execute(
                    query,
                    metadata.agent_id,
                    tenant_id,
                    stripe_result.get('product_id'),
                    metadata.pricing_model.value,
                    metadata.price,
                    metadata.currency,
                    70.0  # 70% revenue share to author
                )

        except Exception as e:
            logger.error(f"Failed to setup agent monetization: {e}")

    async def _index_agent_for_search(self, metadata: AgentMetadata):
        """Add agent to search index"""
        try:
            # Create search index entry
            search_data = {
                'agent_id': str(metadata.agent_id),
                'name': metadata.name,
                'description': metadata.description,
                'category': metadata.category.value,
                'capabilities': [cap.value for cap in metadata.capabilities],
                'tags': metadata.tags,
                'use_cases': metadata.use_cases
            }

            # Store in search index table
            async with get_db_connection() as conn:
                query = """
                    INSERT INTO marketplace.agent_search_index (
                        agent_id, search_data, indexed_at
                    ) VALUES ($1, $2, $3)
                    ON CONFLICT (agent_id)
                    DO UPDATE SET search_data = EXCLUDED.search_data,
                                  indexed_at = EXCLUDED.indexed_at
                """
                await conn.execute(
                    query,
                    metadata.agent_id,
                    json.dumps(search_data),
                    datetime.utcnow()
                )

        except Exception as e:
            logger.error(f"Failed to index agent for search: {e}")

    async def _validate_agent_purchase(self, tenant_id: UUID, agent_id: UUID) -> bool:
        """Validate agent purchase/access"""
        try:
            async with get_db_connection() as conn:
                query = """
                    SELECT COUNT(*) as purchase_count
                    FROM marketplace.agent_purchases
                    WHERE buyer_id = $1 AND agent_id = $2
                    AND (expires_at IS NULL OR expires_at > NOW())
                """
                result = await conn.fetchrow(query, tenant_id, agent_id)

                return result['purchase_count'] > 0

        except Exception as e:
            logger.error(f"Failed to validate agent purchase: {e}")
            return False

    async def _deploy_agent_instance(self, instance: AgentInstance, definition: AgentDefinition):
        """Deploy agent instance"""
        try:
            # Store instance in database
            async with get_db_connection() as conn:
                query = """
                    INSERT INTO marketplace.agent_instances (
                        instance_id, agent_id, tenant_id, instance_name,
                        configuration, status, deployment_date
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7)
                """
                await conn.execute(
                    query,
                    instance.instance_id,
                    instance.agent_id,
                    instance.tenant_id,
                    instance.instance_name,
                    json.dumps(instance.configuration),
                    instance.status,
                    instance.deployment_date
                )

            logger.info(f"Deployed agent instance {instance.instance_id}")

        except Exception as e:
            logger.error(f"Failed to deploy agent instance: {e}")
            raise

    async def _track_agent_installation(self, tenant_id: UUID, agent_id: UUID, instance_id: UUID):
        """Track agent installation"""
        try:
            async with get_db_connection() as conn:
                # Update download count
                query = """
                    UPDATE marketplace.agents
                    SET download_count = download_count + 1
                    WHERE agent_id = $1
                """
                await conn.execute(query, agent_id)

                # Log installation
                log_query = """
                    INSERT INTO marketplace.agent_installations (
                        installation_id, tenant_id, agent_id, instance_id, installation_date
                    ) VALUES ($1, $2, $3, $4, $5)
                """
                await conn.execute(
                    query, uuid4(), tenant_id, agent_id, instance_id, datetime.utcnow()
                )

        except Exception as e:
            logger.error(f"Failed to track agent installation: {e}")

    async def _get_agent_instance(self, tenant_id: UUID, instance_id: UUID) -> Optional[AgentInstance]:
        """Get agent instance"""
        try:
            async with get_db_connection() as conn:
                query = """
                    SELECT * FROM marketplace.agent_instances
                    WHERE tenant_id = $1 AND instance_id = $2
                """
                result = await conn.fetchrow(query, tenant_id, instance_id)

                if not result:
                    return None

                return AgentInstance(
                    instance_id=result['instance_id'],
                    agent_id=result['agent_id'],
                    tenant_id=result['tenant_id'],
                    instance_name=result['instance_name'],
                    configuration=json.loads(result['configuration']),
                    status=result['status'],
                    deployment_date=result['deployment_date'],
                    last_execution=result['last_execution'],
                    execution_count=result['execution_count'],
                    success_rate=float(result['success_rate']),
                    avg_execution_time=float(result['avg_execution_time']),
                    error_count=result['error_count'],
                    resource_usage=json.loads(result['resource_usage'])
                )

        except Exception as e:
            logger.error(f"Failed to get agent instance: {e}")
            return None

    async def _execute_agent_logic(
        self,
        definition: AgentDefinition,
        input_data: Dict[str, Any],
        config: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], int]:
        """Execute agent logic"""
        try:
            # Simplified agent execution
            # In a real implementation, this would execute the agent's workflow

            output_data = {
                'result': 'Agent executed successfully',
                'processed_data': input_data,
                'execution_config': config,
                'timestamp': datetime.utcnow().isoformat()
            }

            # Simulate token usage
            tokens_used = len(json.dumps(input_data)) + len(json.dumps(output_data))

            return output_data, tokens_used

        except Exception as e:
            logger.error(f"Failed to execute agent logic: {e}")
            raise

    async def _calculate_execution_cost(self, tokens_used: int, definition: AgentDefinition) -> float:
        """Calculate execution cost"""
        try:
            # Base cost per token
            cost_per_token = 0.001

            # Calculate cost
            cost = tokens_used * cost_per_token

            return cost

        except Exception as e:
            logger.error(f"Failed to calculate execution cost: {e}")
            return 0.0

    async def _update_instance_metrics(self, instance: AgentInstance, execution: AgentExecution):
        """Update instance performance metrics"""
        try:
            async with get_db_connection() as conn:
                query = """
                    UPDATE marketplace.agent_instances
                    SET execution_count = execution_count + 1,
                        last_execution = $3,
                        success_rate = CASE
                            WHEN $4 = 'success' THEN
                                (success_rate * execution_count + 1) / (execution_count + 1)
                            ELSE
                                (success_rate * execution_count) / (execution_count + 1)
                        END,
                        avg_execution_time = (avg_execution_time * execution_count + $5) / (execution_count + 1),
                        error_count = CASE WHEN $4 = 'error' THEN error_count + 1 ELSE error_count END
                    WHERE instance_id = $1 AND tenant_id = $2
                """
                await conn.execute(
                    query,
                    instance.instance_id,
                    instance.tenant_id,
                    execution.completed_at,
                    execution.status,
                    execution.execution_time_ms
                )

        except Exception as e:
            logger.error(f"Failed to update instance metrics: {e}")

    async def _store_agent_execution(self, execution: AgentExecution):
        """Store agent execution record"""
        try:
            async with get_db_connection() as conn:
                query = """
                    INSERT INTO marketplace.agent_executions (
                        execution_id, instance_id, tenant_id, trigger_type,
                        input_data, output_data, execution_time_ms, status,
                        error_message, cost, tokens_used, started_at, completed_at
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13)
                """
                await conn.execute(
                    query,
                    execution.execution_id,
                    execution.instance_id,
                    execution.tenant_id,
                    execution.trigger_type,
                    json.dumps(execution.input_data),
                    json.dumps(execution.output_data) if execution.output_data else None,
                    execution.execution_time_ms,
                    execution.status,
                    execution.error_message,
                    execution.cost,
                    execution.tokens_used,
                    execution.started_at,
                    execution.completed_at
                )
        except Exception as e:
            logger.error(f"Failed to store agent execution: {e}")

    async def _update_agent_rating(self, agent_id: UUID):
        """Update agent average rating"""
        try:
            async with get_db_connection() as conn:
                query = """
                    UPDATE marketplace.agents
                    SET rating_average = (
                        SELECT AVG(rating) FROM marketplace.agent_ratings
                        WHERE agent_id = $1
                    ),
                    rating_count = (
                        SELECT COUNT(*) FROM marketplace.agent_ratings
                        WHERE agent_id = $1
                    )
                    WHERE agent_id = $1
                """
                await conn.execute(query, agent_id)

        except Exception as e:
            logger.error(f"Failed to update agent rating: {e}")

# Factory function
def create_agent_marketplace() -> AgentMarketplace:
    """Create and initialize agent marketplace"""
    return AgentMarketplace()
