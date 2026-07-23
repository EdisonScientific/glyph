#!/usr/bin/env python3
"""Every Markush eval entry point shares one generation budget.

The CLI, ``fast_eval``, ``checkpoint``, and ``predict`` entry points used to
disagree on the default ``max_new_tokens`` (1024 / 256 / 512 / 512), which
silently changed how much of a Markush target each path could decode. They now
read one shared constant, :data:`glyph.markush.eval.common.DEFAULT_MAX_NEW_TOKENS`,
pinned to the turnkey / headline value (the documented benchmark reproduces
ip5m greedy 511/878 at 1024 new tokens).
"""

from __future__ import annotations

import inspect

from glyph.markush.eval.common import DEFAULT_MAX_NEW_TOKENS


def _argparse_default(parser, dest: str):
    action = next(a for a in parser._actions if a.dest == dest)
    return action.default


def test_headline_budget_is_1024():
    # The turnkey / documented headline value. Guards against a silent drift
    # away from the reproduced 511/878 recipe.
    assert DEFAULT_MAX_NEW_TOKENS == 1024


def test_fast_eval_default_matches_shared_constant():
    from glyph.markush.eval import fast_eval

    assert _argparse_default(fast_eval.build_parser(), "max_new_tokens") == DEFAULT_MAX_NEW_TOKENS
    assert (
        inspect.signature(fast_eval.generate_majority_vote).parameters["max_new_tokens"].default
        == DEFAULT_MAX_NEW_TOKENS
    )


def test_checkpoint_default_matches_shared_constant():
    from glyph.markush.eval import checkpoint

    assert _argparse_default(checkpoint.build_parser(), "max_new_tokens") == DEFAULT_MAX_NEW_TOKENS
    for fn in (checkpoint.generate_one, checkpoint.run_eval):
        assert inspect.signature(fn).parameters["max_new_tokens"].default == DEFAULT_MAX_NEW_TOKENS


def test_predict_default_matches_shared_constant():
    from glyph.markush.eval import predict

    assert _argparse_default(predict.build_parser(), "max_new_tokens") == DEFAULT_MAX_NEW_TOKENS
    for fn in (predict.generate, predict.run_evaluation):
        assert inspect.signature(fn).parameters["max_new_tokens"].default == DEFAULT_MAX_NEW_TOKENS


def test_cli_benchmark_default_matches_shared_constant():
    from glyph.markush.eval import cli

    # The frozen greedy recipe baked into ``benchmark`` uses the shared budget.
    assert cli._GREEDY_MAX_NEW_TOKENS == DEFAULT_MAX_NEW_TOKENS
    assert (
        inspect.signature(cli.benchmark).parameters["max_new_tokens"].default
        == DEFAULT_MAX_NEW_TOKENS
    )
