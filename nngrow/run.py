"""Top-level command dispatcher for the main experiments and ablations."""

from __future__ import annotations

import sys


TASKS = {
    "main": ("Main Tables 2-3", "nngrow.main"),
    "vit-axis": (
        "Appendix A / Table 4: Grow-d versus Grow-H",
        "nngrow.ablations.vit_growth_axis.main",
    ),
    "optimizer-state": (
        "Appendix B / Table 5: optimizer-state handling",
        "nngrow.ablations.optimizer_state.main",
    ),
}


def _usage() -> None:
    print("Usage: python3 -m nngrow <task> [task options]\n")
    print("Tasks:")
    for name, (description, _) in TASKS.items():
        print(f"  {name:16s} {description}")


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] in {"-h", "--help", "--list-tasks"}:
        _usage()
        return
    task = sys.argv[1]
    if task not in TASKS:
        _usage()
        raise SystemExit(f"\nUnknown task: {task}")
    sys.argv = [f"{sys.argv[0]} {task}", *sys.argv[2:]]
    if task == "main":
        from .main import main as task_main
    elif task == "vit-axis":
        from .ablations.vit_growth_axis.main import main as task_main
    else:
        from .ablations.optimizer_state.main import main as task_main
    task_main()


if __name__ == "__main__":
    main()
