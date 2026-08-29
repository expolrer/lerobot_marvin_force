#!/usr/bin/env python3
"""Generate a model-focused bilingual homepage from the reviewed main docs."""

from __future__ import annotations

import argparse
from pathlib import Path


MODELS = {
    "fcact": {
        "name": "Force-conditioned ACT",
        "workspace": "fcact",
        "environment": "lerobot_marvin_fcact",
        "principle_zh": "ACT 的 CVAE 与 Transformer 同时接收 state[8] 和独立 force[7]；部署每次只执行一个位置动作，下一周期重新读取力。",
        "principle_en": "ACT's CVAE and Transformer receive state[8] and a separate force[7]; rollout executes one position action before force is sampled again.",
    },
    "rdp": {
        "name": "Reactive Diffusion Policy",
        "workspace": "rdp",
        "environment": "lerobot_marvin_rdp",
        "principle_zh": "低频视觉 diffusion 负责长时规划，高频力条件 decoder 在每个控制周期根据最新 force[7] 输出位置动作。",
        "principle_en": "Low-rate visual diffusion performs long-horizon planning while a high-rate force-conditioned decoder emits a position action from the newest force[7].",
    },
    "implicitrdp": {
        "name": "ImplicitRDP",
        "workspace": "irdp",
        "environment": "lerobot_marvin_implicitrdp",
        "principle_zh": "在一个训练阶段联合优化动作 latent、力条件 decoder 和视觉 diffusion，并保留 RDP 的逐周期力反馈部署路径。",
        "principle_en": "The action latent, force-conditioned decoder, and visual diffusion objective are jointly optimized while retaining RDP's per-tick force response.",
    },
    "forcevla": {
        "name": "ForceVLA",
        "workspace": "fvla",
        "environment": "lerobot_marvin_forcevla",
        "principle_zh": "Marvin 适配的 PI0 force-token 架构把七维关节力投影为独立条件 token，输出仍为 B 臂 action[8] 位置动作。",
        "principle_en": "The Marvin-adapted PI0 force-token architecture projects seven joint-force channels into a separate token and still outputs arm-B action[8] position commands.",
    },
}


def render(base: str, model_key: str, english: bool) -> str:
    model = MODELS[model_key]
    title = f"# LeRobot Marvin Force · {model['name']}"
    lines = base.splitlines()
    if not lines or not lines[0].startswith("# "):
        raise ValueError("Canonical README has no H1 title")
    language_index = next(
        (index for index, line in enumerate(lines[1:], start=1) if line.strip()), None
    )
    if language_index is None:
        raise ValueError("Canonical README has no language selector")
    language_line = lines[language_index]
    rest = "\n".join(lines[language_index + 1 :]).lstrip("\n")
    if english:
        banner = f"""

> Standalone `{model_key}` branch. Primary workspace: `workspaces/{model['workspace']}`. Conda environment: `{model['environment']}`.
>
> {model['principle_en']}
>
> This branch remains self-contained: environment setup, dataset audit, training, and physical rollout do not depend on another branch.
"""
    else:
        banner = f"""

> 当前为独立 `{model_key}` 分支，主工作区是 `workspaces/{model['workspace']}`，Conda 环境是 `{model['environment']}`。
>
> {model['principle_zh']}
>
> 本分支独立包含环境配置、数据审查、训练和实机 rollout，不依赖其他分支。
"""
    return f"{title}\n\n{language_line}{banner}\n{rest}\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=sorted(MODELS))
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    repo_root = args.repo_root.resolve()
    canonical = repo_root / "docs" / "homepage"
    outputs = {
        repo_root / "README.md": render(
            (canonical / "README.main.md").read_text(encoding="utf-8"), args.model, False
        ),
        repo_root / "README.en.md": render(
            (canonical / "README.main.en.md").read_text(encoding="utf-8"), args.model, True
        ),
    }
    if args.check:
        stale = [path for path, content in outputs.items() if path.read_text(encoding="utf-8") != content]
        if stale:
            raise SystemExit("Outdated branch README: " + ", ".join(map(str, stale)))
        print(f"Branch README OK: {args.model}")
        return 0
    for path, content in outputs.items():
        path.write_text(content, encoding="utf-8", newline="\n")
    print(f"Generated branch README: {args.model}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
