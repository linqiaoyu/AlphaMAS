#!/usr/bin/env python3
"""Generate paper-style backtest tables, plots, and analysis from run outputs.

Expected input structure (from `paper_baseline_backtest.py`):
    <input_dir>/
      summary.csv
      <strategy>/<symbol>/daily_records.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import pandas as pd

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "matplotlib is required for plot_paper_results.py. "
        "Install it with: pip install matplotlib"
    ) from exc


METRIC_COLUMNS = ("CR", "AR", "SR", "MDD")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate paper-style result tables/figures from backtest artifacts.",
    )
    parser.add_argument(
        "--input-dir",
        default=str(Path("reports") / "paper_backtests"),
        help="Directory containing summary.csv and per-strategy artifacts.",
    )
    parser.add_argument(
        "--summary-file",
        default=None,
        help="Optional path to summary.csv. Defaults to <input-dir>/summary.csv.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory for tables/figures/analysis. Defaults to <input-dir>/paper_results.",
    )
    parser.add_argument(
        "--symbols",
        nargs="*",
        default=None,
        help="Optional symbol allowlist (e.g. AAPL GOOGL AMZN).",
    )
    parser.add_argument(
        "--strategies",
        nargs="*",
        default=None,
        help="Optional strategy allowlist (e.g. TradingAgents B&H MACD).",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=180,
        help="Figure DPI.",
    )
    return parser.parse_args()


def _ensure_columns(df: pd.DataFrame, required: Iterable[str]) -> None:
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in summary: {missing}")


def _normalize_percent_columns(df: pd.DataFrame) -> pd.DataFrame:
    table = df.copy()
    table["CR_pct"] = table["CR"] * 100.0
    table["AR_pct"] = table["AR"] * 100.0
    table["MDD_pct"] = table["MDD"] * 100.0
    return table


def _to_markdown_text(df: pd.DataFrame, *, floatfmt: str = ".4f") -> str:
    try:
        return df.to_markdown(index=False, floatfmt=floatfmt)
    except ImportError:
        return "```\n" + df.to_string(index=False) + "\n```"


def build_table1_like(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for symbol in sorted(summary["symbol"].unique()):
        subset = summary.loc[summary["symbol"] == symbol].copy()
        subset = subset.sort_values("strategy")
        for _, row in subset.iterrows():
            rows.append(
                {
                    "symbol": symbol,
                    "strategy": row["strategy"],
                    "CR_pct": row["CR"] * 100.0,
                    "AR_pct": row["AR"] * 100.0,
                    "SR": row["SR"],
                    "MDD_pct": row["MDD"] * 100.0,
                    "trade_count": int(row.get("trade_count", 0)),
                    "start_date": row.get("start_date", ""),
                    "end_date": row.get("end_date", ""),
                }
            )
    return pd.DataFrame(rows)


def build_improvement_table(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for symbol in sorted(summary["symbol"].unique()):
        subset = summary.loc[summary["symbol"] == symbol].copy()
        if subset.empty:
            continue

        ours = subset.loc[subset["strategy"] == "TradingAgents"]
        baselines = subset.loc[subset["strategy"] != "TradingAgents"]
        if ours.empty or baselines.empty:
            continue

        ours_row = ours.iloc[0]
        best_baseline_idx = baselines["CR"].idxmax()
        best_row = baselines.loc[best_baseline_idx]
        rows.append(
            {
                "symbol": symbol,
                "ours_strategy": "TradingAgents",
                "best_baseline_strategy": best_row["strategy"],
                "ours_CR_pct": ours_row["CR"] * 100.0,
                "best_CR_pct": best_row["CR"] * 100.0,
                "CR_improvement_pct_point": (ours_row["CR"] - best_row["CR"]) * 100.0,
                "ours_AR_pct": ours_row["AR"] * 100.0,
                "best_AR_pct": best_row["AR"] * 100.0,
                "AR_improvement_pct_point": (ours_row["AR"] - best_row["AR"]) * 100.0,
                "ours_SR": ours_row["SR"],
                "best_SR": best_row["SR"],
                "SR_improvement": ours_row["SR"] - best_row["SR"],
            }
        )
    return pd.DataFrame(rows)


def _load_cumulative_curve(input_dir: Path, strategy: str, symbol: str) -> pd.DataFrame | None:
    daily_path = input_dir / strategy / symbol / "daily_records.csv"
    if not daily_path.exists():
        return None

    df = pd.read_csv(daily_path)
    if "date" not in df.columns or "equity" not in df.columns or df.empty:
        return None

    series = df[["date", "equity"]].copy()
    series["date"] = pd.to_datetime(series["date"], errors="coerce")
    series = series.dropna(subset=["date"]).sort_values("date")
    if series.empty:
        return None

    first = float(series["equity"].iloc[0])
    if first == 0:
        return None
    series["cum_return"] = series["equity"].astype(float) / first
    return series


def plot_cumulative_returns(
    *,
    summary: pd.DataFrame,
    input_dir: Path,
    figure_dir: Path,
    dpi: int,
) -> list[Path]:
    figure_dir.mkdir(parents=True, exist_ok=True)
    output_paths: list[Path] = []

    for symbol in sorted(summary["symbol"].unique()):
        subset = summary.loc[summary["symbol"] == symbol].copy()
        if subset.empty:
            continue

        plt.figure(figsize=(10, 5.5))
        plotted = 0

        for strategy in sorted(subset["strategy"].unique()):
            curve = _load_cumulative_curve(input_dir, strategy, symbol)
            if curve is None or curve.empty:
                continue

            is_ours = strategy == "TradingAgents"
            plt.plot(
                curve["date"],
                curve["cum_return"],
                label=strategy,
                linewidth=2.8 if is_ours else 1.8,
                alpha=0.95 if is_ours else 0.9,
            )
            plotted += 1

        if plotted == 0:
            plt.close()
            continue

        plt.title(f"Strategy Comparison - Cumulative Returns for {symbol}")
        plt.xlabel("Date")
        plt.ylabel("Cumulative Return (x)")
        plt.grid(True, linestyle="--", alpha=0.35)
        plt.legend(loc="best")
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
        plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
        plt.xticks(rotation=25, ha="right")
        plt.tight_layout()

        out_path = figure_dir / f"cumulative_returns_{symbol}.png"
        plt.savefig(out_path, dpi=dpi)
        plt.close()
        output_paths.append(out_path)

    return output_paths


def build_analysis_markdown(
    *,
    summary: pd.DataFrame,
    table1: pd.DataFrame,
    improvements: pd.DataFrame,
    figure_paths: list[Path],
    output_dir: Path,
) -> str:
    lines: list[str] = []
    lines.append("# Paper-Style Backtest Results")
    lines.append("")
    lines.append("## Run Scope")
    lines.append(
        f"- Symbols: {', '.join(sorted(summary['symbol'].unique()))}"
    )
    lines.append(
        f"- Strategies: {', '.join(sorted(summary['strategy'].unique()))}"
    )
    lines.append(
        f"- Date range: {summary['start_date'].min()} to {summary['end_date'].max()}"
    )
    lines.append("")

    lines.append("## Per-Symbol Bests")
    for symbol in sorted(summary["symbol"].unique()):
        subset = summary.loc[summary["symbol"] == symbol]
        if subset.empty:
            continue
        top_cr = subset.loc[subset["CR"].idxmax()]
        top_sr = subset.loc[subset["SR"].idxmax()]
        top_mdd = subset.loc[subset["MDD"].idxmin()]
        lines.append(f"### {symbol}")
        lines.append(
            f"- Highest CR: `{top_cr['strategy']}` ({top_cr['CR'] * 100:.2f}%)"
        )
        lines.append(
            f"- Highest AR: `{subset.loc[subset['AR'].idxmax(), 'strategy']}` "
            f"({subset['AR'].max() * 100:.2f}%)"
        )
        lines.append(
            f"- Highest SR: `{top_sr['strategy']}` ({top_sr['SR']:.4f})"
        )
        lines.append(
            f"- Lowest MDD: `{top_mdd['strategy']}` ({top_mdd['MDD'] * 100:.2f}%)"
        )
        lines.append("")

    if not improvements.empty:
        lines.append("## TradingAgents Improvement vs Best Baseline (by CR)")
        lines.append(_to_markdown_text(improvements, floatfmt=".4f"))
        lines.append("")

    lines.append("## Artifacts")
    lines.append(f"- Table: `{(output_dir / 'table1_like_metrics.csv').as_posix()}`")
    lines.append(f"- Table (Markdown): `{(output_dir / 'table1_like_metrics.md').as_posix()}`")
    if not improvements.empty:
        lines.append(
            f"- Improvement table: `{(output_dir / 'improvement_vs_best_baseline.csv').as_posix()}`"
        )
    for path in figure_paths:
        lines.append(f"- Figure: `{path.as_posix()}`")
    lines.append("")

    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir).expanduser().resolve()
    summary_path = (
        Path(args.summary_file).expanduser().resolve()
        if args.summary_file
        else input_dir / "summary.csv"
    )
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else input_dir / "paper_results"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    if not summary_path.exists():
        raise FileNotFoundError(f"summary.csv not found: {summary_path}")

    summary = pd.read_csv(summary_path)
    _ensure_columns(summary, ("strategy", "symbol", "start_date", "end_date", *METRIC_COLUMNS))

    if args.symbols:
        symbol_set = {s.upper() for s in args.symbols}
        summary = summary.loc[summary["symbol"].str.upper().isin(symbol_set)].copy()
    if args.strategies:
        strategy_set = {s for s in args.strategies}
        summary = summary.loc[summary["strategy"].isin(strategy_set)].copy()

    if summary.empty:
        raise ValueError("No rows left after applying filters. Nothing to plot.")

    summary = _normalize_percent_columns(summary)
    table1 = build_table1_like(summary)
    improvements = build_improvement_table(summary)

    table1_csv = output_dir / "table1_like_metrics.csv"
    table1_md = output_dir / "table1_like_metrics.md"
    table1.to_csv(table1_csv, index=False)
    table1_md.write_text(_to_markdown_text(table1, floatfmt=".4f"), encoding="utf-8")

    if not improvements.empty:
        improvements.to_csv(output_dir / "improvement_vs_best_baseline.csv", index=False)
        (output_dir / "improvement_vs_best_baseline.md").write_text(
            _to_markdown_text(improvements, floatfmt=".4f"),
            encoding="utf-8",
        )

    figure_paths = plot_cumulative_returns(
        summary=summary,
        input_dir=input_dir,
        figure_dir=output_dir / "figures",
        dpi=args.dpi,
    )

    analysis_md = build_analysis_markdown(
        summary=summary,
        table1=table1,
        improvements=improvements,
        figure_paths=figure_paths,
        output_dir=output_dir,
    )
    analysis_path = output_dir / "paper_results_analysis.md"
    analysis_path.write_text(analysis_md, encoding="utf-8")

    print(f"Saved table: {table1_csv}")
    if not improvements.empty:
        print(f"Saved improvement table: {output_dir / 'improvement_vs_best_baseline.csv'}")
    for path in figure_paths:
        print(f"Saved figure: {path}")
    print(f"Saved analysis: {analysis_path}")


if __name__ == "__main__":
    main()
