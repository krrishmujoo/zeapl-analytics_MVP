from aggregation import (
    average_metric,
    highest_metric,
    lowest_metric,
    total_metric,
    growth_by_month,
    unique_users,
    distinct_count,
    top_users,
    bottom_users,

)
from registries import METRIC_TO_FIELD

def dispatch_aggregation(data: list[dict], request: dict):

    metrics = request["metrics"]
    operation = request["primary_operation"]

    print("Dispatch operation:", operation)

    operation_map = {
        "average": average_metric,
        "maximum": highest_metric,
        "minimum": lowest_metric,
        "total": total_metric,
        "growth": growth_by_month,
        "top_users": top_users,
        "bottom_users": bottom_users,
        "unique_users": unique_users,
        "distinct_count": distinct_count,
    }

    if operation not in operation_map:
        raise ValueError(f"Unsupported operation: {operation}")

    aggregation_function = operation_map[operation]

    location = request.get("location")

    area_field = None
    area_value = None

    if location:
        area_field = location["field"]
        area_value = location["value"]

    # -------- Operations that don't need a metric --------
    if operation == "unique_users":
        return {
            "operation": operation,
            "results": {
                "unique_users": aggregation_function(
                    data,
                    area_field=area_field,
                    area_value=area_value,
                )
            },
        }

    if operation == "distinct_count":
        field = request.get("field")

        if not field:
            raise ValueError(
                "distinct_count requires a field"
            )

        return {
            "operation": operation,
            "results": {
                "distinct_count": aggregation_function(
                    data,
                    field,
                )
            },
        }
    results = {}

    for metric in metrics:

        if metric not in METRIC_TO_FIELD:
            raise ValueError(f"Unsupported metric: {metric}")

        field_name = METRIC_TO_FIELD[metric]

        if operation in ["top_users", "bottom_users"]:


            results[metric] = aggregation_function(
                data,
                field_name,
                top_n=request.get("top_n", 5),
                area_field=area_field,
                area_value=area_value,
            )

        else:
            results[metric] = aggregation_function(
                data,
                field_name,
            )

    return {
        "operation": operation,
        "results": results,
    }