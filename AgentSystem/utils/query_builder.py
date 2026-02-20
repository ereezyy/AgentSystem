from typing import List, Tuple, Any, Optional

def build_select_query(
    table: str,
    conditions: List[Tuple[str, str, Any]] = None,
    order_by: str = None,
    limit: int = None,
    offset: int = None,
    columns: List[str] = None
) -> Tuple[str, List[Any]]:
    """
    Builds a dynamic SQL query for asyncpg.

    Args:
        table: Table name (e.g. "pricing.recommendations")
        conditions: List of (field, operator, value) tuples.
                    If value is None, it is passed as a parameter (resulting in = NULL check in SQL).
                    Callers should filter optional conditions before passing them if they want to ignore them.
        order_by: ORDER BY clause (e.g. "created_at DESC")
        limit: LIMIT value
        offset: OFFSET value
        columns: List of columns to select (default "*")

    Returns:
        (query_string, params_list)
    """
    if columns is None:
        cols = "*"
    else:
        cols = ", ".join(columns)

    query_parts = [f"SELECT {cols} FROM {table}"]
    params = []
    where_parts = []
    param_idx = 1

    if conditions:
        for field, operator, value in conditions:
            where_parts.append(f"{field} {operator} ${param_idx}")
            params.append(value)
            param_idx += 1

    if where_parts:
        query_parts.append("WHERE " + " AND ".join(where_parts))

    if order_by:
        query_parts.append(f"ORDER BY {order_by}")

    if limit is not None:
        query_parts.append(f"LIMIT ${param_idx}")
        params.append(limit)
        param_idx += 1

    if offset is not None:
        query_parts.append(f"OFFSET ${param_idx}")
        params.append(offset)
        param_idx += 1

    return "\n".join(query_parts), params
