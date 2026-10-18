"""Small native grading subprocess: python -m train.cloud.verify TASKS ID DESIGN OUTPUT FIELDS_ROOT."""
import json
from pathlib import Path
import sys


def main() -> None:
    from permit_openenv import evaluator, tasks as task_packs
    from .fields import configure
    from .run import write_json
    tasks, task_id, design, output, fields_root = sys.argv[1:]
    # A fresh process does not inherit the worker's cache binding; native slots live on that volume.
    configure(Path(fields_root))
    task = task_packs.load_pack(Path(tasks)).get(task_id)
    result = evaluator.VerifyEvaluator().evaluate(task, json.loads(Path(design).read_text()))
    write_json(Path(output), result)


if __name__ == '__main__':
    main()
