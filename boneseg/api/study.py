"""Comparing samples between groups."""
from __future__ import annotations

from fastapi import APIRouter
from fastapi.responses import Response

from ..study import METRICS, compare_groups, derived
from .context import AppContext


def router(ctx: AppContext) -> APIRouter:
    r = APIRouter()
    store = ctx.store

    def rows():
        out = []
        for ds in sorted(store.datasets.values(), key=lambda d: d.volume.name):
            job = store.latest_job(ds.id)
            summary = derived(job.result.get("summary", {}), ds.volume.voxel_um, ds.volume.height * ds.volume.width) if job else {}
            hjob = store.latest_job(ds.id, kind="histo")
            if hjob is not None and (job is None or hjob.created > job.created or "Oc.Pm/B.Pm_%" not in summary):
                # A whole-stack histomorphometry run, with bone and cells possibly from different channels
                summary.update(hjob.result.get("summary", {}).get("histomorphometry") or {})
            out.append({"dataset_id": ds.id, "name": ds.volume.name, "group": ds.meta.get("group") or "",
                        "has_stack_run": job is not None, "channel": job.meta.get("channel") if job else None,
                        **{k: summary.get(k) for k in METRICS}})
        return out

    @r.get("/api/study")
    def study(metric: str = "mean_area_fraction"):
        if metric not in METRICS:
            raise ValueError(f"Unknown measurement {metric}")
        data = rows()
        by_group: dict[str, list[float]] = {}
        for row in data:
            if row["group"] and row[metric] is not None:
                by_group.setdefault(row["group"], []).append(row[metric])
        label, unit, scale = METRICS[metric]
        return {"metric": metric, "label": label, "unit": unit, "scale": scale, "rows": data,
                "metrics": {k: v[0] for k, v in METRICS.items()}, "comparison": compare_groups(by_group)}

    @r.get("/api/study.csv")
    def study_csv():
        import pandas as pd

        return Response(pd.DataFrame(rows()).to_csv(index=False), media_type="text/csv",
                        headers={"Content-Disposition": 'attachment; filename="boneseg_study.csv"'})

    return r
