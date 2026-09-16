def build_response(
    request: dict,
    result=None,
    insights=None,
    chart_spec=None,
    chart_data=None,
    prediction=None,
    explanation=None,
    errors=None,
    status=None,
    dataset=None,
    warnings=None,
) -> dict:
    if errors is None:
        errors = []

    response = {
        "request": request,
        "result": result,
        "insights": insights,
        "chart_spec": chart_spec,
        "chart_data": chart_data,
        "prediction": prediction,
        "explanation": explanation,
        "errors": errors,
    }

    # Additive-only fields for /api/v2/analyze -- omitted entirely (not even
    # as null keys) unless a caller explicitly passes them, so the V1
    # response contract (router.execute_request's callers) is byte-identical
    # to before this change.
    if status is not None:
        response["status"] = status
    if dataset is not None:
        response["dataset"] = dataset
    if warnings is not None:
        response["warnings"] = warnings

    return response