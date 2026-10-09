"""UAV MapFly data configs, robot types and training mixtures."""

import os

from starVLA.dataloader.gr00t_lerobot.datasets import LeRobotSingleDataset, ModalityConfig
from starVLA.dataloader.gr00t_lerobot.embodiment_tags import EmbodimentTag
from starVLA.dataloader.gr00t_lerobot.transform.base import ComposedModalityTransform
from starVLA.dataloader.gr00t_lerobot.transform.state_action import (
    StateActionToTensor,
    StateActionTransform,
)


class UavMapflyDataConfig:
    # state: pose relative to the episode start in the start-body frame, [dx, dy, dz, dyaw].
    # action: per-step body-frame delta [forward, lateral, up, dyaw], min_max per dimension so the
    #   stats absorb the sampling interval and the all-zero final action round-trips to exactly
    #   zero (q99 would leak fake motion into chunk tails).
    embodiment_tag = EmbodimentTag.NEW_EMBODIMENT
    video_keys = [
        "video.primary_image",
        "video.map_image",
    ]
    state_keys = [
        "state.pose",
    ]
    action_keys = [
        "action.delta_pose",
    ]
    language_keys = ["annotation.human.action.task_description"]
    observation_indices = [0]
    action_indices = list(range(8))  # matches action_horizon=8 in the training yaml
    state_indices = [0]
    # Eval clients send the raw start-body pose; the server applies the training min_max. Off for
    # other embodiments, whose server forwards state unchanged as their published numbers assume.
    normalize_state_on_server = True

    def modality_config(self):
        return {
            "video": ModalityConfig(delta_indices=self.observation_indices, modality_keys=self.video_keys),
            "state": ModalityConfig(delta_indices=self.state_indices, modality_keys=self.state_keys),
            "action": ModalityConfig(delta_indices=self.action_indices, modality_keys=self.action_keys),
            "language": ModalityConfig(delta_indices=self.observation_indices, modality_keys=self.language_keys),
        }

    def transform(self):
        return ComposedModalityTransform(
            transforms=[
                StateActionToTensor(apply_to=self.state_keys + self.action_keys),
                StateActionTransform(
                    apply_to=self.state_keys + self.action_keys,
                    normalization_modes={
                        "state.pose": "min_max",
                        "action.delta_pose": "min_max",
                    },
                ),
            ]
        )


class UavMapflyGoalGeoDataConfig(UavMapflyDataConfig):
    # Supervision group in metres, from MapFly's goal_geometry (the ruler of the eval's
    # navigation_error_m):
    #   d_goal_m    3D distance to the goal; the stop head binarises it at stop_radius_m
    #   s_remain_m  remaining arc length along the reference path
    #   s_total_m   total arc length (constant per episode); progress p = 1 - s_remain/s_total
    # Not model inputs and not normalised, so the stop label compares directly with stop_radius_m.
    # Base of the seen12 configs; no mix uses it directly.
    supervision_keys = [
        "state.d_goal_m",
        "state.s_remain_m",
        "state.s_total_m",
    ]
    supervision_indices = [0]  # current frame only; use action_indices to supervise every step

    # Its own group so it is fetched with the action's delta_indices: 1 = real step, 0 = padding
    # past the episode end (see MODALITY_JSON in the converter); QwenOFT masks its L1 with it.
    action_mask_keys = ["state.frame_valid"]

    def modality_config(self):
        config = super().modality_config()
        config["supervision"] = ModalityConfig(
            delta_indices=self.supervision_indices, modality_keys=self.supervision_keys
        )
        config["action_mask"] = ModalityConfig(delta_indices=self.action_indices, modality_keys=self.action_mask_keys)
        return config


class _EpisodePrefixDataset(LeRobotSingleDataset):
    # Keeps episode_index < train_episodes; the tail is the seen-scene holdout. Cut after the
    # parent's init, so meta/steps_data_index.pkl (shared by sliced and unsliced mixes and read
    # without a config check) always covers every episode.
    def __init__(self, *, train_episodes: int, **kwargs):
        super().__init__(**kwargs)
        total = len(self._trajectory_ids)
        if train_episodes > total:
            raise ValueError(
                f"{self.dataset_name}: train_episodes={train_episodes} exceeds the {total} episode(s) on disk"
            )
        keep = self._trajectory_ids < train_episodes
        self._trajectory_ids = self._trajectory_ids[keep]
        self._trajectory_lengths = self._trajectory_lengths[keep]
        self._all_steps = [step for step in self._all_steps if step[0] < train_episodes]
        if int(os.environ.get("RANK", "0")) == 0:
            print(
                f"[uav] {self.dataset_name}: train on episode 0-{train_episodes - 1} "
                f"({len(self._all_steps)} steps), held out {total - train_episodes} "
                f"episode(s) for seen-scene eval"
            )


class UavMapflySeen12DataConfig(UavMapflyGoalGeoDataConfig):
    # Training prefix of each seen scene; must match MapFly's configs/splits/seen12_v1.json
    # (~5/6 per scene, 9660 train / 2140 seen holdout).
    train_episodes = {
        "fulldata/bigcity_3000": 2500,
        "fulldata/nyc1950_1500": 1250,
        "fulldata/smallcity_1500": 1250,
        "fulldata/brushifyurban_1000": 800,
        "fulldata/moderncity_1000": 800,
        "fulldata/realcitysf_1000": 800,
        "fulldata/citydowntown_600": 500,
        "fulldata/abandonedcity_500": 400,
        "fulldata/battlefielddesert_500": 400,
        "fulldata/nordicharbour_500": 400,
        "fulldata/urbancity_500": 400,
        "fulldata/industrialarea_200": 160,
    }

    def make_dataset(self, *, dataset_name: str, **kwargs) -> LeRobotSingleDataset:
        if dataset_name not in self.train_episodes:
            raise KeyError(
                f"{dataset_name!r} has no train-episode count; this robot_type is only "
                f"for the fulldata seen scenes, registered: {sorted(self.train_episodes)}"
            )
        return _EpisodePrefixDataset(train_episodes=self.train_episodes[dataset_name], **kwargs)


# The other seen12 trees (convert_fulldata.py prefixes): same episodes and slice in their own
# directory, so each gets a robot_type keyed by it. Markers are current_goal unless named startgoal.
SEEN12_MAP_VARIANT_PREFIXES = {
    "startgoal": "fulldata_startgoal",
    "route": "fulldata_route",
    "current_route": "fulldata_current_route",
    "satellite": "fulldata_satellite",
    "markers_only": "fulldata_markers_only",
    "markers_only_startgoal": "fulldata_markers_only_startgoal",
}


def _seen12_on(prefix: str, *, video_keys: list | None = None) -> UavMapflySeen12DataConfig:
    """The seen12 slice of the tree under ``prefix``, optionally with fewer cameras."""
    config = UavMapflySeen12DataConfig()
    config.train_episodes = {
        f"{prefix}/{name.split('/', 1)[1]}": count for name, count in UavMapflySeen12DataConfig.train_episodes.items()
    }
    if video_keys is not None:
        config.video_keys = video_keys
    return config


ROBOT_TYPE_CONFIG_MAP = {
    # The first two are unsliced bases that no mix references directly; keep them.
    "uav_mapfly": UavMapflyDataConfig(),
    "uav_mapfly_goalgeo": UavMapflyGoalGeoDataConfig(),
    "uav_mapfly_goalgeo_seen12": UavMapflySeen12DataConfig(),
    **{
        f"uav_mapfly_goalgeo_seen12_{variant}": _seen12_on(prefix)
        for variant, prefix in SEEN12_MAP_VARIANT_PREFIXES.items()
    },
    # Camera ablation: the main line without the FPV (QwenOFT takes a variable-length image list),
    # in its own tree (make_maponly_lerobot_tree.py) whose task sentence promises no FPV.
    "uav_mapfly_goalgeo_seen12_maponly": _seen12_on("fulldata_maponly", video_keys=["video.map_image"]),
}

ROBOT_TYPE_TO_EMBODIMENT_TAG = {name: EmbodimentTag.NEW_EMBODIMENT for name in ROBOT_TYPE_CONFIG_MAP}

DATASET_NAMED_MIXTURES = {
    # mixture name: [(dataset subdirectory relative to data_root_dir, weight, robot_type)]
    #
    # Temperature sampling, w = sqrt(n / n_max) over training episodes: tau = 0.5 sits between
    # bigcity dominating (tau = 1) and oversampling industrialarea >10x (tau = 0). Only ratios
    # matter, but bigcity must be exactly 1.0: LeRobotMixtureDataset takes the weight-1.0 dataset
    # as primary for the epoch length. Normalisation stats cover all episodes on disk.
    "uav_mapfly_goalgeo_seen12_tau05": [
        ("fulldata/bigcity_3000", 1.0, "uav_mapfly_goalgeo_seen12"),  # 2500
        ("fulldata/nyc1950_1500", 0.7071, "uav_mapfly_goalgeo_seen12"),  # 1250
        ("fulldata/smallcity_1500", 0.7071, "uav_mapfly_goalgeo_seen12"),  # 1250
        ("fulldata/brushifyurban_1000", 0.5657, "uav_mapfly_goalgeo_seen12"),  # 800
        ("fulldata/moderncity_1000", 0.5657, "uav_mapfly_goalgeo_seen12"),  # 800
        ("fulldata/realcitysf_1000", 0.5657, "uav_mapfly_goalgeo_seen12"),  # 800
        ("fulldata/citydowntown_600", 0.4472, "uav_mapfly_goalgeo_seen12"),  # 500
        ("fulldata/abandonedcity_500", 0.4, "uav_mapfly_goalgeo_seen12"),  # 400
        ("fulldata/battlefielddesert_500", 0.4, "uav_mapfly_goalgeo_seen12"),  # 400
        ("fulldata/nordicharbour_500", 0.4, "uav_mapfly_goalgeo_seen12"),  # 400
        ("fulldata/urbancity_500", 0.4, "uav_mapfly_goalgeo_seen12"),  # 400
        ("fulldata/industrialarea_200", 0.253, "uav_mapfly_goalgeo_seen12"),  # 160
    ],
}


def _seen12_map_variant_mix(prefix: str, robot_type: str) -> list:
    # The main-line rows and weights under another prefix; robot_type's keys must match it.
    return [
        (f"{prefix}/{name.split('/', 1)[1]}", weight, robot_type)
        for name, weight, _ in DATASET_NAMED_MIXTURES["uav_mapfly_goalgeo_seen12_tau05"]
    ]


# One mix per map variant. Never point two variants at one directory.
DATASET_NAMED_MIXTURES.update(
    {
        f"uav_mapfly_goalgeo_seen12_{variant}_tau05": _seen12_map_variant_mix(
            prefix, f"uav_mapfly_goalgeo_seen12_{variant}"
        )
        for variant, prefix in SEEN12_MAP_VARIANT_PREFIXES.items()
    }
)

# Camera ablation: the main-line mix; its robot_type drops the FPV.
DATASET_NAMED_MIXTURES["uav_mapfly_goalgeo_seen12_maponly_tau05"] = _seen12_map_variant_mix(
    "fulldata_maponly", "uav_mapfly_goalgeo_seen12_maponly"
)
