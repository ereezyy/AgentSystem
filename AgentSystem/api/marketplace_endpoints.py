
"""
AI Agent Marketplace API Endpoints - AgentSystem Profit Machine
Revolutionary platform for browsing, creating, and monetizing AI agents
"""

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks, Query, File, UploadFile
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
from ..marketplace.agent_marketplace import (
    AgentMarketplace, AgentCategory, AgentStatus, AgentPricingModel, AgentCapability,
    AgentMetadata, AgentDefinition, AgentInstance, AgentExecution
)

# Initialize router
router = APIRouter(prefix="/api/v1/marketplace", tags=["AI Agent Marketplace"])
security = HTTPBearer()

# Enums
class AgentCategoryAPI(str, Enum):
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

class AgentStatusAPI(str, Enum):
    DRAFT = "draft"
    PUBLISHED = "published"
    DEPRECATED = "deprecated"
    PRIVATE = "private"

class AgentPricingModelAPI(str, Enum):
    FREE = "free"
    ONE_TIME = "one_time"
    SUBSCRIPTION = "subscription"
    USAGE_BASED = "usage_based"
    REVENUE_SHARE = "revenue_share"

class AgentCapabilityAPI(str, Enum):
    TEXT_GENERATION = "text_generation"
    DATA_ANALYSIS = "data_analysis"
    API_INTEGRATION = "api_integration"
    WORKFLOW_EXECUTION = "workflow_execution"
    DECISION_MAKING = "decision_making"
    LEARNING_ADAPTATION = "learning_adaptation"
    MULTI_MODAL = "multi_modal"
    REAL_TIME_PROCESSING = "real_time_processing"

# Request Models
class AgentSearchRequest(BaseModel):
    query: Optional[str] = Field(None, description="Search query")
    category: Optional[AgentCategoryAPI] = Field(None, description="Agent category")
    capabilities: Optional[List[AgentCapabilityAPI]] = Field(None, description="Required capabilities")
    pricing_model: Optional[AgentPricingModelAPI] = Field(None, description="Pricing model")
    min_rating: Optional[float] = Field(None, ge=0, le=5, description="Minimum rating")
    max_price: Optional[float] = Field(None, ge=0, description="Maximum price")
    limit: int = Field(default=20, ge=1, le=100, description="Results limit")

class AgentPublishRequest(BaseModel):
    name: str = Field(..., description="Agent name")
    description: str = Field(..., description="Agent description")
    category: AgentCategoryAPI = Field(..., description="Agent category")
    capabilities: List[AgentCapabilityAPI] = Field(..., description="Agent capabilities")
    pricing_model: AgentPricingModelAPI = Field(default=AgentPricingModelAPI.FREE, description="Pricing model")
    price: float = Field(default=0, ge=0, description="Agent price")
    tags: List[str] = Field(default_factory=list, description="Agent tags")
    use_cases: List[str] = Field(default_factory=list, description="Use cases")
    supported_integrations: List[str] = Field(default_factory=list, description="Supported integrations")
    min_requirements: Dict[str, Any] = Field(default_factory=dict, description="Minimum requirements")
    agent_definition: Dict[str, Any] = Field(..., description="Agent definition and logic")

class AgentInstallRequest(BaseModel):
    agent_id: UUID = Field(..., description="Agent ID to install")
    instance_name: str = Field(..., description="Instance name")
    configuration: Dict[str, Any] = Field(default_factory=dict, description="Instance configuration")

class AgentExecuteRequest(BaseModel):
    instance_id: UUID = Field(..., description="Agent instance ID")
    input_data: Dict[str, Any] = Field(..., description="Input data for agent")
    trigger_type: str = Field(default="manual", description="Execution trigger type")

class AgentRatingRequest(BaseModel):
    agent_id: UUID = Field(..., description="Agent ID to rate")
    rating: int = Field(..., ge=1, le=5, description="Rating (1-5)")
    review: Optional[str] = Field(None, description="Review text")

class CustomAgentRequest(BaseModel):
    name: str = Field(..., description="Custom agent name")
    description: str = Field(..., description="Custom agent description")
    category: AgentCategoryAPI = Field(..., description="Agent category")
    capabilities: List[AgentCapabilityAPI] = Field(..., description="Agent capabilities")
    tags: List[str] = Field(default_factory=list, description="Agent tags")
    use_cases: List[str] = Field(default_factory=list, description="Use cases")
    integrations: List[str] = Field(default_factory=list, description="Required integrations")
    requirements: Dict[str, Any] = Field(default_factory=dict, description="Requirements")
    prompts: Dict[str, str] = Field(default_factory=dict, description="Prompt templates")
    workflow: Dict[str, Any] = Field(default_factory=dict, description="Workflow logic")
    test_cases: List[Dict[str, Any]] = Field(default_factory=list, description="Test cases")

# Response Models
class AgentResponse(BaseModel):
    agent_id: UUID
    name: str
    description: str
    category: AgentCategoryAPI
    capabilities: List[AgentCapabilityAPI]
    version: str
    author_id: UUID
    author_name: str
    pricing_model: AgentPricingModelAPI
    price: float
    currency: str
    status: AgentStatusAPI
    tags: List[str]
    use_cases: List[str]
    supported_integrations: List[str]
    min_requirements: Dict[str, Any]
    performance_metrics: Dict[str, float]
    download_count: int
    rating_average: float
    rating_count: int
    created_date: datetime
    updated_date: datetime

class AgentInstanceResponse(BaseModel):
    instance_id: UUID
    agent_id: UUID
    agent_name: str
    instance_name: str
    configuration: Dict[str, Any]
    status: str
    deployment_date: datetime
    last_execution: Optional[datetime]
    execution_count: int
    success_rate: float
    avg_execution_time: float
    error_count: int
    monthly_cost: float

class AgentExecutionResponse(BaseModel):
    execution_id: UUID
    instance_id: UUID
    trigger_type: str
    input_data: Dict[str, Any]
    output_data: Optional[Dict[str, Any]]
    execution_time_ms: int
    status: str
    error_message: Optional[str]
    cost: float
    tokens_used: int
    started_at: datetime
    completed_at: Optional[datetime]

class MarketplaceDashboardResponse(BaseModel):
    total_agents: int
    published_agents: int
    total_authors: int
    active_customers: int
    total_downloads: int
    avg_marketplace_rating: float
    total_instances: int
    active_instances: int
    total_marketplace_revenue: float
    total_purchases: int
    executions_last_24h: int

class AgentAnalyticsResponse(BaseModel):
    published_agents: int
    installed_agents: int
    total_downloads: int
    avg_rating: float
    monthly_revenue: float
    monthly_purchases: int

class CategoryResponse(BaseModel):
    category_name: str
    display_name: str
    description: str
    total_agents: int
    published_agents: int
    avg_rating: float
    total_downloads: int
    active_instances: int

# Initialize agent marketplace
agent_marketplace = AgentMarketplace()

# Endpoints

@router.get("/browse", response_model=List[AgentResponse])
async def browse_agents(
    request: AgentSearchRequest = Depends(),
    tenant_id: UUID = Depends(get_current_tenant),
    token: str = Depends(security)
):
    """Browse and search agents in the marketplace"""
    try:
        # Verify token
        await verify_token(token.credentials)

        # Convert request to marketplace format
        category = AgentCategory(request.category.value) if request.category else None
        capabilities = [AgentCapability(cap.value) for cap in request.capabilities] if request.capabilities else None
        pricing_model = AgentPricingModel(request.pricing_model.value) if request.pricing_model else None

        # Search agents
        agents = await agent_marketplace.search_agents(
            query=request.query,
            category=category,
            capabilities=capabilities,
            pricing_model=pricing_model,
            min_rating=request.min_rating,
            limit=request.limit
        )

        return [
            AgentResponse(
                agent_id=agent.agent_id,
                name=agent.name,
                description=agent.description,
                category=AgentCategoryAPI(agent.category.value),
                capabilities=[AgentCapabilityAPI(cap.value) for cap in agent.capabilities],
                version=agent.version,
                author_id=agent.author_id,
                author_name=agent.author_name,
                pricing_model=AgentPricingModelAPI(agent.pricing_model.value),
                price=agent.price,
                currency=agent.currency,
                status=AgentStatusAPI(agent.status.value),
                tags=agent.tags,
                use_cases=agent.use_cases,
                supported_integrations=agent.supported_integrations,
                min_requirements=agent.min_requirements,
                performance_metrics=agent.performance_metrics,
                download_count=agent.download_count,
                rating_average=agent.rating_average,
                rating_count=agent.rating_count,
                created_date=agent.created_date,
                updated_date=agent.updated_date
            )
            for agent in agents
        ]

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to browse agents: {str(e)}")

@router.get("/agents/{agent_id}", response_model=AgentResponse)
async def get_agent_details(
    agent_id: UUID,
    tenant_id: UUID = Depends(get_current_tenant),
    token: str = Depends(security)
):
    """Get detailed agent information"""
    try:
        # Verify token
        await verify_token(token.credentials)

        # Get agent details
        agent_details = await agent_marketplace.get_agent_details(agent_id)
        if not agent_details:
            raise HTTPException(status_code=404, detail="Agent not found")

        metadata, definition = agent_details

        return AgentResponse(
            agent_id=metadata.agent_id,
            name=metadata.name,
            description=metadata.description,
            category=AgentCategoryAPI(metadata.category.value),
            capabilities=[AgentCapabilityAPI(cap.value) for cap in metadata.capabilities],
            version=metadata.version,
            author_id=metadata.author_id,
            author_name=metadata.author_name,
            pricing_model=AgentPricingModelAPI(metadata.pricing_model.value),
            price=metadata.price,
            currency=metadata.currency,
            status=AgentStatusAPI(metadata.status.value),
            tags=metadata.tags,
            use_cases=metadata.use_cases,
            supported_integrations=metadata.supported_integrations,
            min_requirements=metadata.min_requirements,
            performance_metrics=metadata.performance_metrics,
            download_count=metadata.download_count,
            rating_average=metadata.rating_average,
            rating_count=metadata.rating_count,
            created_date=metadata.created_date,
            updated_date=metadata.updated_date
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get agent details: {str(e)}")

@router.post("/agents/publish", response_model=Dict[str, Any])
async def publish_agent(
    request: AgentPublishRequest,
    background_tasks: BackgroundTasks,
    tenant_id: UUID = Depends(get_current_tenant),
    token: str = Depends(security)
):
    """Publish a new agent to the marketplace"""
    try:
        # Verify token
        await verify_token(token.credentials)

        # Create agent metadata
        agent_id = uuid4()
        metadata = AgentMetadata(
            agent_id=agent_id,
            name=request.name,
            description=request.description,
            category=AgentCategory(request.category.value),
            capabilities=[AgentCapability(cap.value) for cap in request.capabilities],
            version="1.0.0",
            author_id=tenant_id,
            author_name="Publisher",  # TODO: Get actual tenant name
            pricing_model=AgentPricingModel(request.pricing_model.value),
            price=request.price,
            currency="USD",
            status=AgentStatus.DRAFT,
            tags=request.tags,
            use_cases=request.use_cases,
            supported_integrations=request.supported_integrations,
            min_requirements=request.min_requirements,
            performance_metrics={},
            created_date=datetime.utcnow(),
            updated_date=datetime.utcnow(),
            download_count=0,
            rating_average=0.0,
            rating_count=0
        )

        # Create agent definition
        definition = AgentDefinition(
            agent_id=agent_id,
            configuration=request.agent_definition.get('configuration', {}),
            prompt_templates=request.agent_definition.get('prompts', {}),
            workflow_logic=request.agent_definition.get('workflow', {}),
            api_endpoints=request.agent_definition.get('endpoints', []),
            event_handlers=request.agent_definition.get('events', {}),
            validation_rules=request.agent_definition.get('validation', []),
            dependencies=request.agent_definition.get('dependencies', []),
            custom_code=request.agent_definition.get('custom_code'),
            test_cases=request.agent_definition.get('test_cases', [])
        )

        # Publish agent
        published_agent_id = await agent_marketplace.publish_agent(
            tenant_id=tenant_id,
            agent_metadata=metadata,
            agent_definition=definition
        )

        return {"message": "Agent published successfully", "agent_id": published_agent_id}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to publish agent: {str(e)}")

@router.post("/agents/install", response_model=Dict[str, Any])
async def install_agent(
    request: AgentInstallRequest,
    background_tasks: BackgroundTasks,
    tenant_id: UUID = Depends(get_current_tenant),
    token: str = Depends(security)
):
    """Install an agent for use"""
    try:
        # Verify token
        await verify_token(token.credentials)

        # Install agent
        instance_id = await agent_marketplace.install_agent(
            tenant_id=tenant_id,
            agent_id=request.agent_id,
            instance_config=request.configuration
        )

        return {"message": "Agent installed successfully", "instance_id": instance_id}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to install agent: {str(e)}")

@router.post("/agents/execute", response_model=AgentExecutionResponse)
async def execute_agent(
    request: AgentExecuteRequest,
    background_tasks: BackgroundTasks,
    tenant_id: UUID = Depends(get_current_tenant),
    token: str = Depends(security)
):
    """Execute an agent instance"""
    try:
        # Verify token
        await verify_token(token.credentials)

        # Execute agent
        execution = await agent_marketplace.execute_agent(
            tenant_id=tenant_id,
            instance_id=request.instance_id,
            input_data=request.input_data,
            trigger_type=request.trigger_type
        )

        return AgentExecutionResponse(
            execution_id=execution.execution_id,
            instance_id=execution.instance_id,
            trigger_type=execution.trigger_type,
            input_data=execution.input_data,
            output_data=execution.output_data,
            execution_time_ms=execution.execution_time_ms,
            status=execution.status,
            error_message=execution.error_message,
            cost=execution.cost,
            tokens_used=execution.tokens_used,
            started_at=execution.started_at,
            completed_at=execution.completed_at
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to execute agent: {str(e)}")

@router.post("/agents/custom", response_model=Dict[str, Any])
async def create_custom_agent(
    request: CustomAgentRequest,
    background_tasks: BackgroundTasks,
    tenant_id: UUID = Depends(get_current_tenant),
    token: str = Depends(security)
):
    """Create a custom agent using the agent builder"""
    try:
        # Verify token
        await verify_token(token.credentials)

        # Create agent specification
        agent_spec = {
            'name': request.name,
            'description': request.description,
            'category': request.category.value,
            'capabilities': [cap.value for cap in request.capabilities],
            'tags': request.tags,
            'use_cases': request.use_cases,
            'integrations': request.integrations,
            'requirements': request.requirements,
            'prompts': request.prompts,
            'workflow': request.workflow,
            'test_cases': request.test_cases,
            'author_name': 'Custom Builder'
        }

        # Create custom agent
        agent_id = await agent_marketplace.create_custom_agent(
            tenant_id=tenant_id,
            agent_spec=agent_spec
        )

        return {"message": "Custom agent created successfully", "agent_id": agent_id}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create custom agent: {str(e)}")

@router.get("/my-agents", response_model=List[AgentResponse])
async def get_my_agents(
    tenant_id: UUID = Depends(get_current_tenant),
    token: str = Depends(security)
):
    """Get agents owned by the current tenant"""
    try:
        # Verify token
        await verify_token(token.credentials)

        # Get user's agents
        agents = await agent_marketplace.get_my_agents(tenant_id)

        return [
            AgentResponse(
                agent_id=agent.agent_id,
                name=agent.name,
                description=agent.description,
                category=AgentCategoryAPI(agent.category.value),
                capabilities=[AgentCapabilityAPI(cap.value) for cap in agent.capabilities],
                version=agent.version,
                author_id=agent.author_id,
                author_name=agent.author_name,
                pricing_model=AgentPricingModelAPI(agent.pricing_model.value),
                price=agent.price,
                currency=agent.currency,
                status=AgentStatusAPI(agent.status.value),
                tags=agent.tags,
                use_cases=agent.use_cases,
                supported_integrations=agent.supported_integrations,
                min_requirements=agent.min_requirements,
                performance_metrics=agent.performance_metrics,
                download_count=agent.download_count,
                rating_average=agent.rating_average,
                rating_count=agent.rating_count,
                created_date=agent.created_date,
                updated_date=agent.updated_date
            )
            for agent in agents
        ]

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get user agents: {str(e)}")

@router.get("/my-instances", response_model=List[AgentInstanceResponse])
async def get_installed_agents(
    status: Optional[str] = None,
    tenant_id: UUID = Depends(get_current_tenant),
    token: str = Depends(security)
):
    """Get agents installed by the current tenant"""
    try:
        # Verify token
        await verify_token(token.credentials)

        # Get installed agents
        instances = await agent_marketplace.get_installed_agents(tenant_id)

        # Filter by status if provided
        if status:
            instances = [inst for inst in instances if inst.status == status]

        # Get agent names
        agent_names = {}
        if instances:
            async with get_db_connection() as conn:
                agent_ids = [str(inst.agent_id) for inst in instances]
                query = """
                    SELECT agent_id, name FROM marketplace.agents
                    WHERE agent_id = ANY($1)
                """
                results = await conn.fetch(query, agent_ids)
                agent_names = {row['agent_id']: row['name'] for row in results}

        return [
            AgentInstanceResponse(
                instance_id=instance.instance_id,
                agent_id=instance.agent_id,
                agent_name=agent_names.get(instance.agent_id, 'Unknown'),
                instance_name=instance.instance_name,
                configuration=instance.configuration,
                status=instance.status,
                deployment_date=instance.deployment_date,
                last_execution=instance.last_execution,
                execution_count=instance.execution_count,
                success_rate=instance.success_rate,
                avg_execution_time=instance.avg_execution_time,
                error_count=instance.error_count,
                monthly_cost=instance.resource_usage.get('monthly_cost', 0.0)
            )
            for instance in instances
        ]

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get installed agents: {str(e)}")

@router.post("/agents/rate", response_model=Dict[str, Any])
async def rate_agent(
    request: AgentRatingRequest,
    tenant_id: UUID = Depends(get_current_tenant),
    token: str = Depends(security)
):
    """Rate and review an agent"""
    try:
        # Verify token
        await verify_token(token.credentials)

        # Rate agent
        await agent_marketplace.rate_agent(
            tenant_id=tenant_id,
            agent_id=request.agent_id,
            rating=request.rating,
            review=request.review
        )

        return {"message": "Agent rated successfully"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to rate agent: {str(e)}")

@router.get("/categories", response_model=List[CategoryResponse])
async def get_categories(
    tenant_id: UUID = Depends(get_current_tenant),
    token: str = Depends(security)
):
    """Get marketplace categories"""
    try:
        # Verify token
        await verify_token(token.credentials)

        async with get_db_connection() as conn:
            query = """
                SELECT * FROM marketplace.category_performance
                ORDER BY total_downloads DESC
            """
            results = await conn.fetch(query)

            return [
                CategoryResponse(
                    category_name=row['category_name'],
                    display_name=row['display_name'],
                    description=row['description'] or '',
                    total_agents=row['total_agents'],
                    published_agents=row['published_agents'],
                    avg_rating=float(row['avg_rating'] or 0),
                    total_downloads=row['total_downloads'],
                    active_instances=row['active_instances']
                )
                for row in results
            ]

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get categories: {str(e)}")

@router.get("/dashboard", response_model=MarketplaceDashboardResponse)
async def get_marketplace_dashboard(
    tenant_id: UUID = Depends(get_current_tenant),
    token: str = Depends(security)
):
    """Get marketplace dashboard data"""
    try:
        # Verify token
        await verify_token(token.credentials)

        async with get_db_connection() as conn:
            query = """
                SELECT * FROM marketplace.marketplace_dashboard_stats
            """
            result = await conn.fetchrow(query)

            if not result:
                # Return empty dashboard
                return MarketplaceDashboardResponse(
                    total_agents=0,
                    published_agents=0,
                    total_authors=0,
                    active_customers=0,
                    total_downloads=0,
                    avg_marketplace_rating=0,
                    total_instances=0,
                    active_instances=0,
                    total_marketplace_revenue=0,
                    total_purchases=0,
                    executions_last_24h=0
                )

            return MarketplaceDashboardResponse(
                total_agents=result['total_agents'],
                published_agents=result['published_agents'],
                total_authors=result['total_authors'],
                active_customers=result['active_customers'],
                total_downloads=result['total_downloads'],
                avg_marketplace_rating=float(result['avg_marketplace_rating'] or 0),
                total_instances=result['total_instances'],
                active_instances=result['active_instances'],
                total_marketplace_revenue=float(result['total_marketplace_revenue'] or 0),
                total_purchases=result['total_purchases'],
                executions_last_24h=result['executions_last_24h']
            )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get marketplace dashboard: {str(e)}")

@router.get("/analytics", response_model=AgentAnalyticsResponse)
async def get_marketplace_analytics(
    tenant_id: UUID = Depends(get_current_tenant),
    token: str = Depends(security)
):
    """Get marketplace analytics for the current tenant"""
    try:
        # Verify token
        await verify_token(token.credentials)

        # Get analytics
        analytics = await agent_marketplace.get_marketplace_analytics(tenant_id)

        return AgentAnalyticsResponse(
            published_agents=analytics.get('published_agents', 0),
            installed_agents=analytics.get('installed_agents', 0),
            total_downloads=analytics.get('total_downloads', 0),
            avg_rating=analytics.get('avg_rating', 0),
            monthly_revenue=analytics.get('monthly_revenue', 0),
            monthly_purchases=analytics.get('monthly_purchases', 0)
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get marketplace analytics: {str(e)}")

@router.get("/executions", response_model=List[AgentExecutionResponse])
async def get_agent_executions(
    instance_id: Optional[UUID] = None,
    status: Optional[str] = None,
    days_back: int = Query(default=7, ge=1, le=90),
    limit: int = Query(default=100, le=1000),
    offset: int = Query(default=0, ge=0),
    tenant_id: UUID = Depends(get_current_tenant),
    token: str = Depends(security)
):
    """Get agent execution history"""
    try:
        # Verify token
        await verify_token(token.credentials)

        # Build query
        conditions = ["tenant_id = $1", "started_at >= $2"]
        params = [tenant_id, datetime.utcnow() - timedelta(days=days_back)]
        param_count = 2

        if instance_id:
            param_count += 1
            conditions.append(f"instance_id = ${param_count}")
            params.append(instance_id)

        if status:
            param_count += 1
            conditions.append(f"status = ${param_count}")
            params.append(status)

        async with get_db_connection() as conn:
            query = f"""
                SELECT * FROM marketplace.agent_executions
                WHERE {' AND '.join(conditions)}
                ORDER BY started_at DESC
                LIMIT ${param_count + 1} OFFSET ${param_count + 2}
            """
            params.extend([limit, offset])

            results = await conn.fetch(query, *params)

            return [
                AgentExecutionResponse(
                    execution_id=row['execution_id'],
                    instance_id=row['instance_id'],
                    trigger_type=row['trigger_type'],
                    input_data=json.loads(row['input_data']) if row['input_data'] else {},
                    output_data=json.loads(row['output_data']) if row['output_data'] else None,
                    execution_time_ms=row['execution_time_ms'],
                    status=row['status'],
                    error_message=row['error_message'],
                    cost=float(row['cost']),
                    tokens_used=row['tokens_used'],
                    started_at=row['started_at'],
                    completed_at=row['completed_at']
                )
                for row in results
            ]

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get agent executions: {str(e)}")

@router.get("/templates")
async def get_builder_templates(
    category: Optional[AgentCategoryAPI] = None,
    is_premium: Optional[bool] = None,
    tenant_id: UUID = Depends(get_current_tenant),
    token: str = Depends(security)
):
    """Get agent builder templates"""
    try:
        # Verify token
        await verify_token(token.credentials)

        # Build query
        conditions = []
        params = []

        if category:
            conditions.append("category = $1")
            params.append(category.value)

        if is_premium is not None:
            conditions.append(f"is_premium = ${len(params) + 1}")
            params.append(is_premium)

        async with get_db_connection() as conn:
            query = f"""
                SELECT * FROM marketplace.builder_templates
                {('WHERE ' + ' AND '.join(conditions)) if conditions else ''}
                ORDER BY usage_count DESC
            """

            results = await conn.fetch(query, *params)

            return [dict(row) for row in results]

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get builder templates: {str(e)}")

@router.post("/refresh-dashboard")
async def refresh_marketplace_dashboard(
    tenant_id: UUID = Depends(get_current_tenant),
    token: str = Depends(security)
):
    """Refresh marketplace dashboard materialized view"""
    try:
        # Verify token
        await verify_token(token.credentials)

        async with get_db_connection() as conn:
            await conn.execute("SELECT marketplace.refresh_marketplace_dashboard_stats()")

        return {"message": "Marketplace dashboard refreshed successfully"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to refresh dashboard: {str(e)}")

# Health check endpoint
@router.get("/health")
async def marketplace_health_check():
    """Health check for agent marketplace system"""
    try:
        async with get_db_connection() as conn:
            await conn.fetchval("SELECT 1")

        return {
            "status": "healthy",
            "service": "agent_marketplace",
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Service unhealthy: {str(e)}")
