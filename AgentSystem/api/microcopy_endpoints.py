"""
Microcopy Management API Endpoints
Handles A/B testing and effectiveness tracking for UI text
"""

from fastapi import APIRouter, HTTPException, Depends
from typing import Dict, List, Any, Optional
from datetime import datetime, date, timedelta
from pydantic import BaseModel
import asyncpg
from ..utils.logger import get_logger

router = APIRouter(prefix="/api/microcopy", tags=["Microcopy"])
logger = get_logger(__name__)


class MicrocopyVariant(BaseModel):
    key: str
    variant_name: str
    context: str
    type: str
    content: Dict[str, Any]
    is_active: bool = True


class MicrocopyInteraction(BaseModel):
    variant_id: str
    session_id: str
    interaction_type: str
    outcome: Optional[str] = None
    metadata: Dict[str, Any] = {}


class EffectivenessReport(BaseModel):
    variant_id: str
    variant_name: str
    key: str
    total_views: int
    total_clicks: int
    total_completions: int
    total_errors: int
    click_through_rate: float
    completion_rate: float
    error_rate: float


@router.get("/variants")
async def get_active_variants(db_pool: asyncpg.Pool = Depends()):
    """
    Get all active microcopy variants for the current user
    Returns variants for A/B testing
    """
    try:
        async with db_pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT
                    id,
                    key,
                    variant_name,
                    context,
                    type,
                    content,
                    is_active
                FROM microcopy_variants
                WHERE is_active = true
                ORDER BY key, variant_name
            """)

            variants = [dict(row) for row in rows]

            return {
                "success": True,
                "variants": variants,
                "count": len(variants)
            }

    except Exception as e:
        logger.error(f"Error fetching microcopy variants: {e}")
        raise HTTPException(status_code=500, detail="Failed to load microcopy variants")


@router.get("/variants/{key}")
async def get_variants_by_key(
    key: str,
    db_pool: asyncpg.Pool = Depends()
):
    """
    Get all variants for a specific microcopy key
    """
    try:
        async with db_pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT
                    id,
                    key,
                    variant_name,
                    context,
                    type,
                    content,
                    is_active,
                    created_at,
                    updated_at
                FROM microcopy_variants
                WHERE key = $1
                ORDER BY variant_name
            """, key)

            if not rows:
                raise HTTPException(status_code=404, detail=f"No variants found for key: {key}")

            variants = [dict(row) for row in rows]

            return {
                "success": True,
                "key": key,
                "variants": variants,
                "count": len(variants)
            }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching variants for key {key}: {e}")
        raise HTTPException(status_code=500, detail="Failed to load variants")


@router.post("/variants")
async def create_variant(
    variant: MicrocopyVariant,
    db_pool: asyncpg.Pool = Depends()
):
    """
    Create a new microcopy variant
    Requires admin permissions
    """
    try:
        async with db_pool.acquire() as conn:
            variant_id = await conn.fetchval("""
                INSERT INTO microcopy_variants (
                    key,
                    variant_name,
                    context,
                    type,
                    content,
                    is_active
                )
                VALUES ($1, $2, $3, $4, $5, $6)
                ON CONFLICT (key, variant_name) DO UPDATE
                SET
                    context = EXCLUDED.context,
                    type = EXCLUDED.type,
                    content = EXCLUDED.content,
                    is_active = EXCLUDED.is_active,
                    updated_at = now()
                RETURNING id
            """,
                variant.key,
                variant.variant_name,
                variant.context,
                variant.type,
                variant.content,
                variant.is_active
            )

            return {
                "success": True,
                "message": "Microcopy variant created successfully",
                "variant_id": str(variant_id)
            }

    except Exception as e:
        logger.error(f"Error creating microcopy variant: {e}")
        raise HTTPException(status_code=500, detail="Failed to create variant")


@router.post("/interactions")
async def track_interaction(
    interaction: MicrocopyInteraction,
    tenant_id: str,
    user_id: str,
    db_pool: asyncpg.Pool = Depends()
):
    """
    Track a user interaction with a microcopy variant
    Used for A/B testing effectiveness
    """
    try:
        async with db_pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO microcopy_interactions (
                    variant_id,
                    tenant_id,
                    user_id,
                    session_id,
                    interaction_type,
                    outcome,
                    metadata
                )
                VALUES ($1, $2, $3, $4, $5, $6, $7)
            """,
                interaction.variant_id,
                tenant_id,
                user_id,
                interaction.session_id,
                interaction.interaction_type,
                interaction.outcome,
                interaction.metadata
            )

            return {
                "success": True,
                "message": "Interaction tracked successfully"
            }

    except Exception as e:
        logger.error(f"Error tracking microcopy interaction: {e}")
        return {"success": False, "message": "Failed to track interaction"}


@router.get("/effectiveness/{key}")
async def get_variant_effectiveness(
    key: str,
    days: int = 7,
    db_pool: asyncpg.Pool = Depends()
) -> Dict[str, Any]:
    """
    Get effectiveness metrics for all variants of a key
    Shows which variant performs best
    """
    try:
        async with db_pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT
                    mv.id as variant_id,
                    mv.variant_name,
                    mv.key,
                    COALESCE(SUM(me.total_views), 0) as total_views,
                    COALESCE(SUM(me.total_clicks), 0) as total_clicks,
                    COALESCE(SUM(me.total_completions), 0) as total_completions,
                    COALESCE(SUM(me.total_errors), 0) as total_errors,
                    COALESCE(AVG(me.click_through_rate), 0) as avg_ctr,
                    COALESCE(AVG(me.completion_rate), 0) as avg_completion_rate,
                    COALESCE(AVG(me.error_rate), 0) as avg_error_rate,
                    COALESCE(AVG(me.avg_time_to_action), 0) as avg_time_to_action
                FROM microcopy_variants mv
                LEFT JOIN microcopy_effectiveness me ON mv.id = me.variant_id
                    AND me.date >= CURRENT_DATE - $2
                WHERE mv.key = $1
                GROUP BY mv.id, mv.variant_name, mv.key
                ORDER BY avg_completion_rate DESC, avg_ctr DESC
            """, key, days)

            if not rows:
                raise HTTPException(status_code=404, detail=f"No data found for key: {key}")

            effectiveness = [
                {
                    "variant_id": str(row['variant_id']),
                    "variant_name": row['variant_name'],
                    "key": row['key'],
                    "metrics": {
                        "total_views": row['total_views'],
                        "total_clicks": row['total_clicks'],
                        "total_completions": row['total_completions'],
                        "total_errors": row['total_errors'],
                        "click_through_rate": float(row['avg_ctr']),
                        "completion_rate": float(row['avg_completion_rate']),
                        "error_rate": float(row['avg_error_rate']),
                        "avg_time_to_action": float(row['avg_time_to_action'])
                    },
                    "score": self._calculate_variant_score(row)
                }
                for row in rows
            ]

            winner = max(effectiveness, key=lambda x: x['score']) if effectiveness else None

            return {
                "success": True,
                "key": key,
                "period_days": days,
                "variants": effectiveness,
                "winner": winner,
                "recommendation": self._generate_recommendation(effectiveness, winner)
            }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching effectiveness for key {key}: {e}")
        raise HTTPException(status_code=500, detail="Failed to load effectiveness data")


@router.get("/report/summary")
async def get_effectiveness_summary(
    days: int = 7,
    db_pool: asyncpg.Pool = Depends()
) -> Dict[str, Any]:
    """
    Get overall microcopy effectiveness summary
    Shows top and bottom performing variants
    """
    try:
        async with db_pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT
                    mv.key,
                    mv.variant_name,
                    mv.context,
                    COALESCE(SUM(me.total_views), 0) as total_views,
                    COALESCE(SUM(me.total_clicks), 0) as total_clicks,
                    COALESCE(SUM(me.total_completions), 0) as total_completions,
                    COALESCE(AVG(me.click_through_rate), 0) as avg_ctr,
                    COALESCE(AVG(me.completion_rate), 0) as avg_completion_rate,
                    COALESCE(AVG(me.error_rate), 0) as avg_error_rate
                FROM microcopy_variants mv
                LEFT JOIN microcopy_effectiveness me ON mv.id = me.variant_id
                    AND me.date >= CURRENT_DATE - $1
                WHERE mv.is_active = true
                GROUP BY mv.key, mv.variant_name, mv.context
                HAVING SUM(me.total_views) > 100
                ORDER BY avg_completion_rate DESC, avg_ctr DESC
            """, days)

            all_variants = [dict(row) for row in rows]

            top_performers = all_variants[:5]
            bottom_performers = all_variants[-5:]

            total_views = sum(v['total_views'] for v in all_variants)
            total_clicks = sum(v['total_clicks'] for v in all_variants)
            total_completions = sum(v['total_completions'] for v in all_variants)

            return {
                "success": True,
                "period_days": days,
                "overview": {
                    "total_variants_tested": len(all_variants),
                    "total_views": total_views,
                    "total_clicks": total_clicks,
                    "total_completions": total_completions,
                    "overall_ctr": total_clicks / total_views if total_views > 0 else 0,
                    "overall_completion_rate": total_completions / total_clicks if total_clicks > 0 else 0
                },
                "top_performers": top_performers,
                "bottom_performers": bottom_performers,
                "recommendations": self._generate_global_recommendations(all_variants)
            }

    except Exception as e:
        logger.error(f"Error generating effectiveness summary: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate summary")


@router.post("/calculate-effectiveness/{variant_id}")
async def calculate_effectiveness(
    variant_id: str,
    target_date: Optional[date] = None,
    db_pool: asyncpg.Pool = Depends()
):
    """
    Manually trigger effectiveness calculation for a variant
    Usually runs automatically daily
    """
    try:
        if not target_date:
            target_date = date.today()

        async with db_pool.acquire() as conn:
            await conn.execute(
                "SELECT calculate_microcopy_effectiveness($1, $2)",
                variant_id,
                target_date
            )

            return {
                "success": True,
                "message": f"Effectiveness calculated for {target_date}",
                "variant_id": variant_id,
                "date": str(target_date)
            }

    except Exception as e:
        logger.error(f"Error calculating effectiveness: {e}")
        raise HTTPException(status_code=500, detail="Failed to calculate effectiveness")


def _calculate_variant_score(row: dict) -> float:
    """
    Calculate a composite score for variant effectiveness
    Higher is better
    """
    ctr_weight = 0.3
    completion_weight = 0.5
    error_weight = 0.2

    ctr_score = float(row.get('avg_ctr', 0))
    completion_score = float(row.get('avg_completion_rate', 0))
    error_penalty = float(row.get('avg_error_rate', 0))

    score = (
        (ctr_score * ctr_weight) +
        (completion_score * completion_weight) -
        (error_penalty * error_weight)
    )

    return max(0, min(1, score))


def _generate_recommendation(variants: List[dict], winner: Optional[dict]) -> str:
    """
    Generate a recommendation based on variant performance
    """
    if not winner or not variants:
        return "Not enough data to make a recommendation"

    if len(variants) < 2:
        return "Need at least 2 variants to compare"

    second_best = sorted(variants, key=lambda x: x['score'], reverse=True)[1]
    score_diff = winner['score'] - second_best['score']

    if score_diff > 0.15:
        return f"Strong winner: '{winner['variant_name']}' performs significantly better. Consider making it the default."
    elif score_diff > 0.05:
        return f"Clear winner: '{winner['variant_name']}' performs better. Monitor for consistency."
    else:
        return f"Close competition: '{winner['variant_name']}' leads slightly. Continue testing."


def _generate_global_recommendations(variants: List[dict]) -> List[str]:
    """
    Generate recommendations across all variants
    """
    recommendations = []

    if not variants:
        return ["No data available for recommendations"]

    avg_ctr = sum(v.get('avg_ctr', 0) for v in variants) / len(variants)
    avg_completion = sum(v.get('avg_completion_rate', 0) for v in variants) / len(variants)

    if avg_ctr < 0.3:
        recommendations.append("Overall CTR is low. Consider making CTAs more compelling.")

    if avg_completion < 0.5:
        recommendations.append("Completion rates need improvement. Review form complexity and error messages.")

    low_performers = [v for v in variants if v.get('avg_completion_rate', 0) < 0.3]
    if len(low_performers) > len(variants) * 0.3:
        recommendations.append(f"{len(low_performers)} variants underperforming. Consider A/B testing new approaches.")

    return recommendations if recommendations else ["System performing well overall"]
