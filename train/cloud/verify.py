"""Small native grading subprocess: python -m train.cloud.verify TASKS ID DESIGN OUTPUT."""
import json
from pathlib import Path
import sys


def main() -> None:
    from permit_openenv.evaluator import VerifyEvaluator
    from permit_openenv.tasks import load_pack
    from .run import write_json
    tasks, task_id, design, output = sys.argv[1:]
    task = load_pack(Path(tasks)).get(task_id)
    result = VerifyEvaluator().evaluate(task, json.loads(Path(design).read_text()))
    write_json(Path(output), result)


if __name__ == '__main__':
    main()
