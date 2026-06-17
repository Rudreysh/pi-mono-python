"""Thin wrapper over agent harness skill loading."""

from __future__ import annotations

from pi_mono.agent.harness import skills as _harness_skills

load_skills = _harness_skills.load_skills
load_sourced_skills = _harness_skills.load_sourced_skills
format_skill_invocation = _harness_skills.format_skill_invocation
SkillFrontmatter = _harness_skills.SkillFrontmatter

__all__ = [
    "load_skills",
    "load_sourced_skills",
    "format_skill_invocation",
    "SkillFrontmatter",
]
