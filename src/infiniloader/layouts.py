"""One explicit TP/PP layout contract for planning and canonical tensor views."""
from __future__ import annotations


def validate_layout(tensors: dict, layout: dict) -> tuple[int, int, dict]:
    if not isinstance(layout, dict) or layout.get("format") != 1:
        raise ValueError("layout requires format=1")
    tp, pp, rules = layout.get("tp"), layout.get("pp"), layout.get("tensors")
    if (type(tp) is not int or tp < 1 or type(pp) is not int or pp < 1
            or not isinstance(rules, dict)):
        raise ValueError("layout requires positive tp, pp and explicit tensor rules")
    if set(rules) != set(tensors):
        raise ValueError("layout must explicitly cover exactly all tensors")
    stages = set()
    for name, spec in tensors.items():
        rule = rules[name]
        if not isinstance(rule, dict) or set(rule) != {"stage", "split_dim"}:
            raise ValueError(f"rule requires stage and split_dim: {name}")
        stage, dim = rule["stage"], rule["split_dim"]
        if type(stage) is not int or not 0 <= stage < pp:
            raise ValueError(f"invalid pipeline stage: {name}")
        if dim is not None and (type(dim) is not int or not 0 <= dim < len(spec["shape"])
                                or spec["shape"][dim] % tp):
            raise ValueError(f"invalid or nondivisible split dimension: {name}")
        stages.add(stage)
    if len(stages) != pp:
        raise ValueError("every pipeline stage must own at least one tensor")
    return tp, pp, rules
