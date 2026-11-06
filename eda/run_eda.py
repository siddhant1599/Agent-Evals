"""Factual-consistency-relevant EDA report generator.

Computes the same metrics as the EDA_*_v2.ipynb notebooks (length distributions, length
retention %, abstractiveness, extractive fragment coverage/density, data integrity checks,
outlier examples, optional lead bias) and renders every result as a self-descriptive PNG
under ``eda/fig/<dataset>/`` — histograms, one scatter plot, and tables rendered as images —
so the whole report can be dropped into slides without re-running anything.

Usage:
    python eda/run_eda.py --dataset xsum
    python eda/run_eda.py --dataset cnndm
    python eda/run_eda.py --dataset all
"""

from __future__ import annotations

import argparse
import re
import textwrap
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from datasets import Dataset, load_dataset

from agent_evals.config import DataConfig

FIG_ROOT = Path(__file__).parent / "fig"


@dataclass(frozen=True)
class DatasetSpec:
    key: str
    hf_name: str
    hf_path: str
    hf_revision: str
    hf_config: str | None
    source_col: str
    summary_col: str
    source_name: str  # singular noun used in labels/captions, e.g. "article" / "document"


DATASETS: dict[str, DatasetSpec] = {
    "xsum": DatasetSpec(
        key="xsum",
        hf_name="tier1_xsum",
        hf_path="EdinburghNLP/xsum",
        hf_revision="7d4d486c2f8ef850b1a11aead99b894ff3dd7da9",
        hf_config=None,
        source_col="document",
        summary_col="summary",
        source_name="document",
    ),
    "cnndm": DatasetSpec(
        key="cnndm",
        hf_name="tier1_cnndm",
        hf_path="abisee/cnn_dailymail",
        hf_revision="96df5e686bee6baa90b8bee7c28b81fa3fa6223d",
        hf_config="3.0.0",
        source_col="article",
        summary_col="highlights",
        source_name="article",
    ),
}

# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------


def save_hist(series: pd.Series, title: str, xlabel: str, path: Path, bins: int = 50) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    series.plot(kind="hist", bins=bins, ax=ax)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("count")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  wrote {path}")


def save_scatter(df: pd.DataFrame, x: str, y: str, title: str, xlabel: str, ylabel: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    df.plot(kind="scatter", x=x, y=y, alpha=0.3, s=10, ax=ax)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  wrote {path}")


def save_table(
    df: pd.DataFrame,
    title: str,
    path: Path,
    col_widths: list[float] | None = None,
    fig_width: float | None = None,
    subtitle: str | None = None,
) -> None:
    """Render a DataFrame as a self-contained, titled PNG table.

    Rows are laid out manually (not via ax.table) because matplotlib's table cells don't
    reflow row heights for multi-line wrapped text — a fixed row height either clips long
    cells or overlaps neighboring rows. Row height here is instead sized to each row's
    actual line count, so wrapped outlier snippets render legibly.

    ``subtitle``, when given, renders as a wrapped caption band above the header row —
    e.g. explaining what "highest"/"lowest" means for that metric, in plain language,
    so the table doesn't require the reader to already know the metric's definition.
    """
    n_rows, n_cols = df.shape
    if col_widths is None:
        col_widths = [1.0 / n_cols] * n_cols
    total_w = sum(col_widths)
    col_widths = [w / total_w for w in col_widths]
    if fig_width is None:
        fig_width = max(7.0, 2.2 * n_cols)

    def line_count(value: object) -> int:
        return max(1, str(value).count("\n") + 1)

    line_height_in = 0.22
    row_pad_in = 0.12
    header_height_in = line_height_in + row_pad_in
    row_heights_in = [
        line_height_in * max(line_count(v) for v in row) + row_pad_in for row in df.itertuples(index=False)
    ]

    subtitle_text = None
    subtitle_height_in = 0.0
    if subtitle:
        chars_per_line = max(40, int(fig_width * 11))
        subtitle_lines = textwrap.wrap(subtitle, width=chars_per_line) or [subtitle]
        subtitle_text = "\n".join(subtitle_lines)
        subtitle_height_in = line_height_in * len(subtitle_lines) + row_pad_in * 1.5

    content_height_in = subtitle_height_in + header_height_in + sum(row_heights_in)
    fig_height = content_height_in + 0.9  # + title margin

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title(title, fontsize=12, fontweight="bold", pad=14)

    col_x = [0.0]
    for w in col_widths:
        col_x.append(col_x[-1] + w)

    header_frac = header_height_in / content_height_in
    row_fracs = [h / content_height_in for h in row_heights_in]

    y = 1.0
    table_top = 1.0
    if subtitle_text:
        subtitle_frac = subtitle_height_in / content_height_in
        ax.text(0.5, y - subtitle_frac / 2, subtitle_text, fontsize=8.5, style="italic", color="dimgray", va="center", ha="center")
        y -= subtitle_frac
        table_top = y

    ax.plot([0, 1], [table_top, table_top], color="black", linewidth=1.2)
    for c, col_name in enumerate(df.columns):
        ax.text(col_x[c] + 0.01, y - header_frac / 2, str(col_name), fontsize=9.5, fontweight="bold", va="center", ha="left")
    y -= header_frac
    ax.plot([0, 1], [y, y], color="black", linewidth=1.2)

    for row_frac, row in zip(row_fracs, df.itertuples(index=False)):
        for c, val in enumerate(row):
            ax.text(col_x[c] + 0.01, y - row_frac / 2, str(val), fontsize=9, va="center", ha="left")
        y -= row_frac
        ax.plot([0, 1], [y, y], color="gray", linewidth=0.6)

    for x in col_x:
        ax.plot([x, x], [0, table_top], color="gray", linewidth=0.6)

    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight", pad_inches=0.2)
    plt.close(fig)
    print(f"  wrote {path}")


def describe_table(series: pd.Series, value_label: str, percentiles=(0.05, 0.25, 0.5, 0.75, 0.95)) -> pd.DataFrame:
    desc = series.describe(percentiles=list(percentiles))
    return pd.DataFrame({"statistic": desc.index, value_label: [f"{v:,.2f}" for v in desc.values]})


def wrap(text: str, width: int = 48, max_chars: int = 240) -> str:
    text = text.replace("\n", " ").strip()
    if len(text) > max_chars:
        text = text[:max_chars].rstrip() + " ..."
    return "\n".join(textwrap.wrap(text, width=width)) or text


def short_id(value: object, max_len: int = 10) -> str:
    """Truncate long ids (e.g. 40-char CNN/DM SHA1s) so they don't overflow into other columns."""
    text = str(value)
    return text if len(text) <= max_len else text[: max_len - 1] + "…"


# ---------------------------------------------------------------------------
# Text metrics (mirrors the notebooks)
# ---------------------------------------------------------------------------


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z']+", text.lower())


def novel_ngram_fraction(source: str, summary: str, n: int = 1) -> float:
    source_ngrams = set(zip(*[tokenize(source)[i:] for i in range(n)]))
    summ_tokens = tokenize(summary)
    summ_ngrams = list(zip(*[summ_tokens[i:] for i in range(n)]))
    if not summ_ngrams:
        return float("nan")
    novel = sum(1 for g in summ_ngrams if g not in source_ngrams)
    return novel / len(summ_ngrams)


def extractive_fragments(source_tokens: list[str], summary_tokens: list[str]) -> list[int]:
    source_pos: dict[str, list[int]] = {}
    for i, tok in enumerate(source_tokens):
        source_pos.setdefault(tok, []).append(i)

    fragments = []
    i = 0
    while i < len(summary_tokens):
        best_len = 0
        for j in source_pos.get(summary_tokens[i], []):
            length = 0
            while (
                i + length < len(summary_tokens)
                and j + length < len(source_tokens)
                and summary_tokens[i + length] == source_tokens[j + length]
            ):
                length += 1
            best_len = max(best_len, length)
        if best_len > 0:
            fragments.append(best_len)
            i += best_len
        else:
            i += 1
    return fragments


def coverage_density(source: str, summary: str) -> tuple[float, float]:
    source_tokens = tokenize(source)
    summ_tokens = tokenize(summary)
    if not summ_tokens:
        return float("nan"), float("nan")
    frag_lens = extractive_fragments(source_tokens, summ_tokens)
    coverage = sum(frag_lens) / len(summ_tokens)
    density = sum(l**2 for l in frag_lens) / len(summ_tokens)
    return coverage, density


# ---------------------------------------------------------------------------
# Main report
# ---------------------------------------------------------------------------


def run(spec: DatasetSpec) -> None:
    out_dir = FIG_ROOT / spec.key
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n=== {spec.key} -> {out_dir} ===")

    SRC, SUM = spec.source_col, spec.summary_col
    src_name = spec.source_name

    cfg = DataConfig(name=spec.hf_name, hf_path=spec.hf_path, hf_config=spec.hf_config, hf_revision=spec.hf_revision)
    ds = load_dataset(cfg.hf_path, name=cfg.hf_config, revision=cfg.hf_revision, token=False)
    test_df = ds["test"].to_pandas()
    print(f"  loaded {len(test_df):,} test rows")

    # --- Data quality & integrity ---------------------------------------------------
    n_null_source = (test_df[SRC].str.strip() == "").sum()
    n_null_summary = (test_df[SUM].str.strip() == "").sum()
    n_dup_ids = test_df["id"].duplicated().sum()
    n_dup_source = test_df[SRC].duplicated().sum()
    n_dup_pairs = test_df.duplicated(subset=[SRC, SUM]).sum()

    test_df[f"{SRC}_words"] = test_df[SRC].str.split().map(len)
    test_df[f"{SUM}_words"] = test_df[SUM].str.split().map(len)

    n_broken = (test_df[f"{SRC}_words"] < test_df[f"{SUM}_words"]).sum()
    n_verbatim = (test_df[SRC].str.strip() == test_df[SUM].str.strip()).sum()

    train_hashes = set(hash(x) for x in ds["train"][SRC])
    n_overlap = test_df[SRC].map(hash).isin(train_hashes).sum()

    integrity = pd.DataFrame(
        {
            "check": [
                f"empty {src_name}s",
                "empty summaries",
                "duplicate ids",
                f"duplicate {src_name}s",
                f"duplicate ({src_name}, summary) pairs",
                f"summary longer than {src_name} (broken)",
                f"{src_name} text == summary text (verbatim)",
                "test rows also in train (verbatim overlap)",
            ],
            "count": [
                n_null_source,
                n_null_summary,
                n_dup_ids,
                n_dup_source,
                n_dup_pairs,
                n_broken,
                n_verbatim,
                f"{n_overlap} ({n_overlap / len(test_df):.1%})",
            ],
        }
    )
    save_table(
        integrity,
        f"{spec.key}: data quality & integrity (n={len(test_df):,} test rows)",
        out_dir / "01_integrity_table.png",
        col_widths=[0.7, 0.3],
    )

    # --- Length distributions ---------------------------------------------------
    save_hist(
        test_df[f"{SRC}_words"],
        f"{spec.key}: {src_name} length (words)",
        "words",
        out_dir / "02_source_length_hist.png",
    )
    save_hist(
        test_df[f"{SUM}_words"],
        f"{spec.key}: summary length (words)",
        "words",
        out_dir / "03_summary_length_hist.png",
    )
    length_stats = pd.DataFrame(
        {
            "statistic": test_df[f"{SRC}_words"].describe().index,
            f"{src_name}_words": [f"{v:,.1f}" for v in test_df[f"{SRC}_words"].describe().values],
            "summary_words": [f"{v:,.1f}" for v in test_df[f"{SUM}_words"].describe().values],
        }
    )
    save_table(
        length_stats,
        f"{spec.key}: length statistics (words)",
        out_dir / "04_length_stats_table.png",
        col_widths=[0.34, 0.33, 0.33],
    )

    # --- Length retention (%) ---------------------------------------------------
    retention_df = test_df[test_df[f"{SRC}_words"] > 0]
    test_df["retention_pct"] = retention_df[f"{SUM}_words"] / retention_df[f"{SRC}_words"] * 100
    save_hist(
        test_df["retention_pct"],
        f"{spec.key}: summary length as % of {src_name} length",
        f"% of {src_name} retained",
        out_dir / "05_retention_hist.png",
    )
    save_table(
        describe_table(test_df["retention_pct"], "retention_%"),
        f"{spec.key}: length retention (%) statistics",
        out_dir / "06_retention_stats_table.png",
        col_widths=[0.5, 0.5],
        subtitle=(
            "retention_% = summary words ÷ source words × 100. Lower = more compression "
            "(shorter summary relative to source); above 100% means the summary is longer than the source."
        ),
    )

    # --- Abstractiveness ---------------------------------------------------
    abstractiveness_sample = test_df.sample(min(5000, len(test_df)), random_state=0).copy()
    abstractiveness_sample["novel_unigram"] = abstractiveness_sample.apply(
        lambda r: novel_ngram_fraction(r[SRC], r[SUM], n=1), axis=1
    )
    abstractiveness_sample["novel_bigram"] = abstractiveness_sample.apply(
        lambda r: novel_ngram_fraction(r[SRC], r[SUM], n=2), axis=1
    )
    abstractiveness_table = pd.DataFrame(
        {
            "metric": ["new single words (unigrams)", "new word-pairs (bigrams)"],
            "mean new_words_%": [
                f"{abstractiveness_sample['novel_unigram'].mean():.1%}",
                f"{abstractiveness_sample['novel_bigram'].mean():.1%}",
            ],
        }
    )
    save_table(
        abstractiveness_table,
        f"{spec.key}: abstractiveness (sample n={len(abstractiveness_sample):,})",
        out_dir / "07_abstractiveness_table.png",
        col_widths=[0.6, 0.4],
        subtitle=(
            "new_words_% = share of summary words/word-pairs that don't appear anywhere in the "
            f"source {src_name}. Higher = more paraphrased and harder to verify word-for-word "
            "against the source."
        ),
    )

    # --- Extractive fragment coverage & density ---------------------------------------------------
    frag_sample = test_df.sample(min(2000, len(test_df)), random_state=0).copy()
    coverage_vals, density_vals = zip(*frag_sample.apply(lambda r: coverage_density(r[SRC], r[SUM]), axis=1))
    frag_sample["coverage"] = coverage_vals
    frag_sample["density"] = density_vals

    save_hist(
        frag_sample["coverage"],
        f"{spec.key}: extractive fragment coverage",
        "coverage (fraction of summary copied from source)",
        out_dir / "08_coverage_hist.png",
        bins=30,
    )
    save_hist(
        frag_sample["density"],
        f"{spec.key}: extractive fragment density",
        "density (avg. squared fragment length)",
        out_dir / "09_density_hist.png",
        bins=30,
    )
    cov_den_stats = pd.DataFrame(
        {
            "statistic": frag_sample["coverage"].describe().index,
            "coverage": [f"{v:,.3f}" for v in frag_sample["coverage"].describe().values],
            "density": [f"{v:,.3f}" for v in frag_sample["density"].describe().values],
        }
    )
    save_table(
        cov_den_stats,
        f"{spec.key}: extractive coverage/density statistics (sample n={len(frag_sample):,})",
        out_dir / "10_coverage_density_stats_table.png",
        col_widths=[0.34, 0.33, 0.33],
        subtitle=(
            "coverage = share of the summary made of spans copied verbatim from the source. "
            "density (copy_len_score) = how long those copied spans tend to be — higher means "
            "longer word-for-word chunks, i.e. more extractive."
        ),
    )

    # --- Length vs. abstractiveness relationship ---------------------------------------------------
    scatter_df = abstractiveness_sample.copy()
    scatter_df["retention_pct"] = test_df.loc[scatter_df.index, "retention_pct"]
    scatter_df = scatter_df.dropna(subset=["retention_pct"])
    save_scatter(
        scatter_df,
        x="retention_pct",
        y="novel_unigram",
        title=f"{spec.key}: length retention (%) vs. novel-unigram fraction",
        xlabel="length retention (%)",
        ylabel="novel unigrams in summary (fraction)",
        path=out_dir / "11_retention_vs_novelty_scatter.png",
    )

    # --- Outlier / pathological examples ---------------------------------------------------
    # Plain-language column names instead of the internal metric names, so a table doesn't
    # require already knowing what "novel_unigram" or "density" mean.
    METRIC_DISPLAY_NAME = {
        "retention_pct": "retention_%",
        "novel_unigram": "new_words_%",
        "density": "copy_len_score",
    }

    def outlier_table(df: pd.DataFrame, metric: str, n: int, largest: bool, fmt: str) -> pd.DataFrame:
        picker = df.nlargest if largest else df.nsmallest
        rows = picker(n, metric)
        return pd.DataFrame(
            {
                "id": [short_id(v) for v in rows["id"].values],
                METRIC_DISPLAY_NAME.get(metric, metric): [fmt.format(v) for v in rows[metric].values],
                src_name: [wrap(s) for s in rows[SRC].values],
                "summary": [wrap(s) for s in rows[SUM].values],
            }
        )

    save_table(
        outlier_table(test_df, "retention_pct", 3, largest=False, fmt="{:.1f}%"),
        f"{spec.key}: lowest length retention (outliers)",
        out_dir / "12_outliers_lowest_retention_table.png",
        col_widths=[0.15, 0.14, 0.37, 0.34],
        fig_width=13.0,
        subtitle=(
            "retention_% = summary words ÷ source words. Lowest retention = the heaviest "
            "compression (summary is a tiny fraction of the source's length)."
        ),
    )
    save_table(
        outlier_table(test_df, "retention_pct", 3, largest=True, fmt="{:.1f}%"),
        f"{spec.key}: highest length retention (outliers)",
        out_dir / "13_outliers_highest_retention_table.png",
        col_widths=[0.15, 0.14, 0.37, 0.34],
        fig_width=13.0,
        subtitle=(
            "Highest retention = little to no compression — the summary is close to, or even "
            "longer than, the source. Values over 100% usually flag a broken or stub source, "
            "not a genuinely long summary."
        ),
    )
    save_table(
        outlier_table(abstractiveness_sample, "novel_unigram", 3, largest=True, fmt="{:.1%}"),
        f"{spec.key}: highest novelty (outliers)",
        out_dir / "14_outliers_highest_novelty_table.png",
        col_widths=[0.15, 0.14, 0.37, 0.34],
        fig_width=13.0,
        subtitle=(
            "new_words_% = share of summary words that don't appear anywhere in the source. "
            "Highest novelty = the summary is hardest to verify word-for-word against the "
            "source (highest risk of unsupported/fabricated content)."
        ),
    )
    save_table(
        outlier_table(frag_sample, "density", 3, largest=True, fmt="{:.1f}"),
        f"{spec.key}: highest extractive density (outliers)",
        out_dir / "15_outliers_highest_density_table.png",
        col_widths=[0.15, 0.14, 0.37, 0.34],
        fig_width=13.0,
        subtitle=(
            "copy_len_score rewards long verbatim chunks copied from the source. Highest = the "
            "summary reuses long word-for-word spans from the source (most extractive, easiest "
            "to fact-check by direct text match)."
        ),
    )
    save_table(
        outlier_table(frag_sample, "density", 3, largest=False, fmt="{:.1f}"),
        f"{spec.key}: lowest extractive density (outliers)",
        out_dir / "16_outliers_lowest_density_table.png",
        col_widths=[0.15, 0.14, 0.37, 0.34],
        fig_width=13.0,
        subtitle=(
            "Lowest copy_len_score = almost no verbatim copying at all — the summary is fully "
            "paraphrased, so it can't be fact-checked by simple text matching against the source."
        ),
    )

    # --- Optional: lead bias ---------------------------------------------------
    def lead_overlap_fraction(source: str, summary: str, lead_fraction: float) -> float:
        source_tokens = tokenize(source)
        summ_tokens = set(tokenize(summary))
        if not summ_tokens:
            return float("nan")
        lead_len = max(1, int(len(source_tokens) * lead_fraction))
        lead_tokens = set(source_tokens[:lead_len])
        return len(summ_tokens & lead_tokens) / len(summ_tokens)

    lead_sample = test_df.sample(min(5000, len(test_df)), random_state=0)
    lead_rows = []
    for frac in (0.1, 0.25, 0.5, 1.0):
        overlaps = lead_sample.apply(lambda r: lead_overlap_fraction(r[SRC], r[SUM], frac), axis=1)
        lead_rows.append((f"first {frac:.0%} of {src_name}", f"{overlaps.mean():.1%}"))
    lead_table = pd.DataFrame(lead_rows, columns=["lead window", "mean summary-unigram overlap"])
    save_table(
        lead_table,
        f"{spec.key}: lead bias (optional, secondary signal)",
        out_dir / "17_lead_bias_table.png",
        col_widths=[0.5, 0.5],
    )

    print(f"  done: {len(list(out_dir.glob('*.png')))} figures written to {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=[*DATASETS.keys(), "all"], default="all")
    args = parser.parse_args()

    keys = list(DATASETS.keys()) if args.dataset == "all" else [args.dataset]
    for key in keys:
        run(DATASETS[key])


if __name__ == "__main__":
    main()
