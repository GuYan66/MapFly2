"""Auxiliary prediction heads trained alongside an action head."""

from .goal_geometry_head import GOAL_HEAD_DEFAULTS, GoalGeometryHead, get_goal_geometry_head

__all__ = ["GOAL_HEAD_DEFAULTS", "GoalGeometryHead", "get_goal_geometry_head"]
