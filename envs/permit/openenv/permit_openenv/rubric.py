"""OpenEnv terminal rubric, consuming the already verified submission once."""
from openenv.core.rubrics import Rubric


class GradeComponent(Rubric):
    def __init__(self, source: 'PermitRubric', key: str):
        super().__init__()
        object.__setattr__(self, '_source', source)
        self.key = key

    def forward(self, action, observation) -> float:
        grade = self._source.pending_grade
        return float(grade.get(self.key, 0)) if grade else 0.0


class PermitRubric(Rubric):
    def __init__(self):
        super().__init__()
        self.pending_grade: dict | None = None
        self.feasible = GradeComponent(self, 'feasible')
        self.cost_quality = GradeComponent(self, 'reward')

    def forward(self, action, observation) -> float:
        self.feasible(action, observation)
        self.cost_quality(action, observation)
        grade, self.pending_grade = self.pending_grade, None
        return float(grade['reward']) if grade else 0.0

    def reset(self) -> None:
        self.pending_grade = None
