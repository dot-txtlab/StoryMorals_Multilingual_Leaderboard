#!/usr/bin/env python3
"""StoryMorals leaderboard pipeline.

    python scripts/run.py generate                 # call model APIs -> output/model_morals.csv
    python scripts/run.py generate --model gpt-5.5 # one model only
    python scripts/run.py evaluate                 # embed + metrics -> output/leaderboard.json
    python scripts/run.py evaluate --embedders minilm labse
    python scripts/run.py dashboard                # build docs/index.html
    python scripts/run.py all                      # generate -> evaluate -> dashboard

`evaluate` uses the human morals + the paper's GPT-4o morals + any model morals
already generated, so it runs end-to-end before you have API keys.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dotenv import load_dotenv  # noqa: E402

load_dotenv()


def cmd_generate(args):
    if getattr(args, "background", False):
        _launch_background(args)
        return
    from storymorals import generate
    generate.run(only_model=args.model, limit=args.limit, story=args.story)


def _launch_background(args):
    """Re-launch `generate` detached so it survives closing the terminal."""
    import subprocess

    from storymorals import ROOT

    log = ROOT / "run.log"
    cmd = [sys.executable, str(Path(__file__).resolve()), "generate"]
    if args.model:
        cmd += ["--model", args.model]
    if args.limit:
        cmd += ["--limit", str(args.limit)]
    if args.story:
        cmd += ["--story", args.story]
    with open(log, "a") as f:
        proc = subprocess.Popen(
            cmd, stdout=f, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
            start_new_session=True,  # detach from this terminal (survives close)
        )
    print("=" * 60)
    print(f"STARTED in background (pid {proc.pid}). Safe to close this window.")
    print(f"  log:      tail -f {log}")
    print("  progress: python scripts/run.py status")
    print(f"  stop it:  kill {proc.pid}")
    print("=" * 60)


def cmd_status(args):
    from storymorals import generate
    generate.status()


def cmd_evaluate(args):
    from storymorals import embed, leaderboard, metrics
    from storymorals.config import load_languages, load_models
    from storymorals.datasets import PAPER_GPT_SOURCE, load_long

    long_df = load_long()
    print(f"[data] {len(long_df)} morals "
          f"({long_df.is_human.sum()} human, {(~long_df.is_human).sum()} model) "
          f"across {long_df.passage_id.nunique()} passages")
    emb = embed.embed_all(long_df, names=args.embedders)
    results = metrics.evaluate(long_df, emb)

    specs, _, _ = load_models()
    display = {s.id: s.display for s in specs}
    provider = {s.id: s.provider for s in specs}
    display[PAPER_GPT_SOURCE] = "GPT-4o (paper)"
    provider[PAPER_GPT_SOURCE] = "openai"

    payload = leaderboard.build(results, display, provider, load_languages())
    print(f"[done] {payload['n_models']} models ranked -> output/leaderboard.json")
    for r in payload["rows"]:
        print(f"  #{r['rank']:>2} {r['display']:<22} "
              f"align={r['alignment_mean']:.3f} flatten={r['diversity_gap']:+.3f} "
              f"[{r['verdict']}]")


def cmd_dashboard(args):
    import json

    from storymorals import dashboard
    from storymorals.leaderboard import LEADERBOARD_JSON

    if not LEADERBOARD_JSON.exists():
        sys.exit("Run `evaluate` first — no output/leaderboard.json yet.")
    out = dashboard.build(json.loads(LEADERBOARD_JSON.read_text()))
    print(f"[done] dashboard -> {out}")


def cmd_all(args):
    cmd_generate(args)
    cmd_evaluate(args)
    cmd_dashboard(args)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("generate", help="call model APIs to produce morals")
    g.add_argument("--model", default=None, help="run only this model id")
    g.add_argument("--limit", type=int, default=None,
                   help="only the first N passages (smoke test, e.g. --limit 1)")
    g.add_argument("--story", default=None,
                   help="only this story's 14 language passages, e.g. --story brazil")
    g.add_argument("--background", action="store_true",
                   help="run detached (survives closing the terminal); logs to run.log")
    g.set_defaults(func=cmd_generate)

    sub.add_parser("status", help="show per-model progress + errors").set_defaults(
        func=cmd_status)

    e = sub.add_parser("evaluate", help="embed + score -> leaderboard.json")
    e.add_argument("--embedders", nargs="+", default=None,
                   help="subset of: minilm labse mpnet-en tfidf (default: minilm labse)")
    e.set_defaults(func=cmd_evaluate)

    d = sub.add_parser("dashboard", help="build docs/index.html")
    d.set_defaults(func=cmd_dashboard)

    a = sub.add_parser("all", help="generate -> evaluate -> dashboard")
    a.add_argument("--model", default=None)
    a.add_argument("--embedders", nargs="+", default=None)
    a.set_defaults(func=cmd_all)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
