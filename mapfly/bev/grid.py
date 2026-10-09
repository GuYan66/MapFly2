from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from mapfly.bev.height_field import BuildingCatalog, BuildingObstacle


@dataclass(frozen=True)
class OccupancyGrid:
    origin_ue: tuple[float, float]
    resolution_m: float
    occupied: np.ndarray
    inflated: np.ndarray
    clearance_m: np.ndarray
    buildings: BuildingCatalog | None = None

    def world_to_cell(self, xy_ue_cm: tuple[float, float]) -> tuple[int, int]:
        cell_cm = self.resolution_m * 100.0
        col = int(np.floor((xy_ue_cm[0] - self.origin_ue[0]) / cell_cm + 0.5))
        row = int(np.floor((xy_ue_cm[1] - self.origin_ue[1]) / cell_cm + 0.5))
        return row, col

    def cell_to_world(self, cell: tuple[int, int]) -> tuple[float, float]:
        row, col = cell
        cell_cm = self.resolution_m * 100.0
        return self.origin_ue[0] + col * cell_cm, self.origin_ue[1] + row * cell_cm

    def is_free_world(self, xy_ue_cm: tuple[float, float]) -> bool:
        row, col = self.world_to_cell(xy_ue_cm)
        if row < 0 or col < 0 or row >= self.inflated.shape[0] or col >= self.inflated.shape[1]:
            return False
        return not bool(self.inflated[row, col])

    def save_npz(self, path: str | Path) -> None:
        payload: dict[str, np.ndarray | float] = {
            "origin_ue": np.asarray(self.origin_ue),
            "resolution_m": self.resolution_m,
            "occupied": self.occupied,
            "inflated": self.inflated,
            "clearance_m": self.clearance_m,
        }
        if self.buildings is not None:
            payload.update(
                building_owner_ids=self.buildings.owner_ids,
                building_names=np.asarray(
                    [entry.name for entry in self.buildings.entries], dtype=str
                ),
                building_centers_ue=np.asarray(
                    [entry.center_ue for entry in self.buildings.entries], dtype=float
                ),
                building_heights_m=np.asarray(
                    [entry.height_m for entry in self.buildings.entries], dtype=float
                ),
                building_min_z_ue_cm=np.asarray(
                    [entry.min_z_ue_cm for entry in self.buildings.entries], dtype=float
                ),
                building_max_z_ue_cm=np.asarray(
                    [entry.max_z_ue_cm for entry in self.buildings.entries], dtype=float
                ),
            )
        np.savez_compressed(path, **payload)

    @classmethod
    def load_npz(cls, path: str | Path) -> OccupancyGrid:
        with np.load(path) as data:
            buildings = None
            if "building_owner_ids" in data.files:
                entries = tuple(
                    BuildingObstacle(
                        name=str(name),
                        center_ue=(float(center[0]), float(center[1])),
                        height_m=float(height),
                        min_z_ue_cm=float(min_z),
                        max_z_ue_cm=float(max_z),
                    )
                    for name, center, height, min_z, max_z in zip(
                        data["building_names"],
                        data["building_centers_ue"],
                        data["building_heights_m"],
                        data["building_min_z_ue_cm"],
                        data["building_max_z_ue_cm"],
                        strict=True,
                    )
                )
                buildings = BuildingCatalog(entries, data["building_owner_ids"])
            return cls(
                origin_ue=tuple(data["origin_ue"]),
                resolution_m=float(data["resolution_m"]),
                occupied=data["occupied"],
                inflated=data["inflated"],
                clearance_m=data["clearance_m"],
                buildings=buildings,
            )

    def to_png(self, path: str | Path) -> None:
        from mapfly.viz import save_bev_png

        save_bev_png(self, path)
