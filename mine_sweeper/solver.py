"""
マインスイーパーの求解
"""
from __future__ import annotations

import logging
import math
import time
from collections import defaultdict
from fractions import Fraction
from typing import Dict, List, Sequence, Tuple

import z3
from pydantic.main import BaseModel

ConstFieldType = Tuple[Tuple[str, ...], ...]
FieldType = List[List[str]]


def _comb(n: int, r: int) -> int:
    return math.factorial(n) // (math.factorial(n - r) * math.factorial(r))


def _normalize_field(field_str: str) -> str:
    return field_str.translate(str.maketrans("？ｘ・０１２３４５６７８", "?x.012345678"))


class FreqWeight(defaultdict):
    def __init__(self, bomb_factor: Fraction, num_question: int) -> None:
        super().__init__(int)
        self._bomb_factor = bomb_factor
        self._num_question = num_question

    def add_record(self, num_bomb: int) -> None:
        self[num_bomb] += 1

    def summarize_possibility(self) -> Fraction:
        return sum(
            (
                count
                * _comb(self._num_question, num_bomb)
                * self._bomb_factor ** num_bomb
                * (1 - self._bomb_factor) ** (self._num_question - num_bomb)
                for num_bomb, count in self.items()
            ),
            start=Fraction(0),
        )


class Point(BaseModel):
    class Config:
        frozen = True

    h: int
    w: int


class MineSweeperProblemAnalysis(BaseModel):
    num_opened: int  # 掘られた安全地帯の数
    num_found: int  # 特定された爆弾の数

    @classmethod
    def from_field_str(cls, field_def: str) -> MineSweeperProblemAnalysis:
        normalized_field_def = _normalize_field(field_def)
        return MineSweeperProblemAnalysis(
            num_opened=sum(normalized_field_def.count(c) for c in ".012345678"),
            num_found=normalized_field_def.count("x"),
        )


class MineSweeperProblem(BaseModel):
    original_field_def: str
    field_def: str
    field: ConstFieldType

    @classmethod
    def from_field_str(cls, field_def: str) -> MineSweeperProblem:
        normalized_field_def = _normalize_field(field_def)
        return MineSweeperProblem(
            original_field_def=field_def,
            field_def=normalized_field_def,
            field=tuple(map(tuple, normalized_field_def.strip("\n").split("\n"))),
        )


class MineSweeperSolution(BaseModel):
    """マインスイーパー問題の解答"""

    question_seq: Sequence[Point]
    solution_bv_seq: Sequence[int]

    problem: MineSweeperProblem


class MineSweeperSolver:
    """マインスイーパー問題のソルバ"""

    @classmethod
    def solve(cls, problem: MineSweeperProblem) -> MineSweeperSolution:
        start_at = time.monotonic()
        questions: Dict[Point, z3.Bool] = {
            Point(h=h, w=w): z3.Bool(f"cell_({h},{w})")
            for h, row in enumerate(problem.field)
            for w, c in enumerate(row)
            if c == "?"
        }
        constraints: Dict[Point, int] = {
            Point(h=h, w=w): int(c)
            for h, row in enumerate(problem.field)
            for w, c in enumerate(row)
            if c in "012345678"
        }
        known_mines: Sequence[Point] = tuple(
            Point(h=h, w=w) for h, row in enumerate(problem.field) for w, c in enumerate(row) if c == "x"
        )

        logging.info("%s", f"# {len(questions)=}")
        logging.debug("%s", f"{questions=}")
        logging.info("%s", f"# {len(constraints)=}")
        logging.debug("%s", f"{constraints=}")
        logging.info("%s", f"# {len(known_mines)=}")
        logging.debug("%s", f"{known_mines=}")

        assert questions, "No unknown cell"

        def distance(pos_a: Point, pos_b: Point) -> int:
            return max(abs(pos_a.h - pos_b.h), abs(pos_a.w - pos_b.w))

        # 周囲に制約のない "?" に対して警告し、条件から除外する
        # NOTE これをやらないと求解でdo-not-careが発生して無限ループする
        # TODO

        # build Z3 problem
        solver = z3.Solver()
        for c_point, c_value in constraints.items():
            num_bomb_known = sum(1 for m in known_mines if distance(c_point, m) == 1)
            target_question_list: List[z3.Bool] = [
                q_is_bomb for q_point, q_is_bomb in questions.items() if distance(c_point, q_point) == 1
            ]
            num_bomb_in_question_list: int = c_value - num_bomb_known
            if not target_question_list:
                assert num_bomb_in_question_list == 0, (c_point, c_value, num_bomb_known)
                continue
            solver.add(z3.AtLeast(*target_question_list, num_bomb_in_question_list))
            solver.add(z3.AtMost(*target_question_list, num_bomb_in_question_list))

        # ALL-SAT
        variable_list = list(questions.values())
        solution_bv_seq: List[int] = []

        def model_to_i(model: z3.ModelRef) -> int:
            return sum(2 ** i * bool(model[v]) for i, v in enumerate(variable_list))

        while solver.check() == z3.sat:
            model = solver.model()
            solution_bv_seq.append(model_to_i(model))
            solver.add(z3.Or(*(v != model[v] for v in variable_list)))
            if (num_solution := len(solution_bv_seq)) % 256 == 0:
                duration = time.monotonic() - start_at
                logging.info(f"{num_solution} solutions found in {duration} seconds...")

        duration = time.monotonic() - start_at
        logging.info(f"# Took {duration} seconds")
        assert len(solution_bv_seq) > 0, "UNSAT. Problem specification must be wrong."

        return MineSweeperSolution(
            question_seq=tuple(questions.keys()), solution_bv_seq=solution_bv_seq, problem=problem
        )


class MineSweeperSolutionVisualizer:
    """マインスイーパー問題の解の可視化器"""

    @classmethod
    def show_plaintext(cls, solution: MineSweeperSolution, bomb_factor: Fraction) -> None:
        # def get_pattern(i: int) -> str:
        #     variables: Sequence[Tuple[int, int, bool]] = [
        #         (h, w, bool(i & (2 ** j))) for (j, (h, w)) in enumerate(solution.question_seq)
        #     ]

        #     def conv(h: int, w: int, c: str) -> str:
        #         if c != "?":
        #             return c
        #         for v in variables:
        #             if (h, w) == v[0:2]:
        #                 return ".#"[v[2]]
        #         assert False

        #     return "\n".join(
        #         "".join(conv(h, w, c) for w, c in enumerate(row))
        #         for h, row in enumerate(solution.problem.field)
        #     )

        num_question = len(solution.question_seq)
        statistics = [0] * num_question
        statistics_pos = [FreqWeight(bomb_factor, num_question) for _ in solution.question_seq]
        for bv in solution.solution_bv_seq:
            num_bomb = f"{bv:b}".count("1")
            # print("*" * 64)
            # print(get_pattern(bv))
            for i in range(num_question):
                if bool(bv & (2 ** i)):
                    statistics[i] += 1
                    statistics_pos[i].add_record(num_bomb)
        logging.debug(statistics)
        logging.debug(statistics_pos)
        logging.info("# len(solutions)=%d", len(solution.solution_bv_seq))
        logging.debug(solution.solution_bv_seq)

        statistics_normalized = [s.summarize_possibility() for s in statistics_pos]
        max_prob = max(statistics_normalized)

        def to_string(s: Fraction, c: int) -> str:
            if c == 0:
                return f"<safe>\t{c}"
            if c == len(solution.solution_bv_seq):
                return f"*BOMB*\t{c}"
            return f"{float(s / max_prob):.05f}\t{c}"

        statistics_normalized_str = [to_string(s, c) for s, c in zip(statistics_normalized, statistics)]

        def get_hazard_pattern() -> str:
            def conv(h: int, w: int, c: str) -> str:
                if c == "?":
                    for q_idx, q_point in enumerate(solution.question_seq):
                        if Point(h=h, w=w) == q_point:
                            return statistics_normalized_str[q_idx]
                    assert False, (h, w, c)
                return f"   ({c})\t"

            return "\n".join(
                "\t".join(conv(h, w, c) for w, c in enumerate(row))
                for h, row in enumerate(solution.problem.field)
            )

        print(get_hazard_pattern())
