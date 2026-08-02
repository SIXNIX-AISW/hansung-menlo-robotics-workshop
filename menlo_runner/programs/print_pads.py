from typing import Any
from menlo_runner.scene import get_scene

async def agent(ctx: Any):
    scene = await get_scene(ctx)
    for eid, entity in scene.entities.items():
        if eid.startswith("pad_"):
            p = entity.pose.position
            print(f"PAD_LOC {eid}: {p[0]:.2f}, {p[1]:.2f}")
    return {"status": "done"}
