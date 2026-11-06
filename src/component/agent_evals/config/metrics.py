"""The metrics group: one frozen dataclass per metric, plus the port-validation targets."""

from __future__ import annotations

from dataclasses import dataclass, field

from .validation import _SHA40, _SHA256, ConfigError, _check_digest, _check_positive, _check_text

#: Entailment judges the citation metric can use (D-120, D-135).
_CITATION_JUDGES = ("minicheck", "alignscore")


@dataclass(frozen=True, slots=True)
class SummaCConfig:
    """SummaC-ZS and SummaC-Conv, every value inherited from the official code (D-018, D-028).

    Attributes:
        nli_model: NLI checkpoint used by both variants ("vitc").
        nli_revision: Pinned commit of ``nli_model``.
        entailment_index: Position of entailment in the NLI output.
        contradiction_index: Position of contradiction in the NLI output.
        min_sentence_chars: Sentences of this many characters or fewer are dropped. Safe: >= 0.
        max_document_sentences: Only the first N article sentences are scored.
        max_input_tokens: Length each (article sentence, summary sentence) pair is encoded to.
        truncation: Truncation strategy the official tokenizer call resolved to.
        half_precision: Run the NLI model in float16 on CUDA, as the official code does.
        batch_size: Sentence pairs per NLI forward pass.
        conv_bins: SummaC-Conv histogram bin edges; strictly increasing from 0.0 to 1.0.
        conv_max_summary_sentences: Summary sentences SummaC-Conv reads.
        conv_checkpoint_url: SummaC-Conv weights at a pinned commit.
        conv_checkpoint_sha256: Digest the downloaded weights must match.
    """

    nli_model: str = "tals/albert-xlarge-vitaminc-mnli"
    nli_revision: str = "3082ba54344bd9ddada2be1c5e9b4131721d2a5d"
    entailment_index: int = 0
    contradiction_index: int = 1
    min_sentence_chars: int = 10
    max_document_sentences: int = 100
    max_input_tokens: int = 500
    truncation: str = "longest_first"
    half_precision: bool = True
    batch_size: int = 128
    conv_bins: tuple[float, ...] = (
        0.0,
        0.01,
        0.02,
        0.03,
        0.04,
        0.07,
        0.13,
        0.37,
        0.90,
        0.91,
        0.92,
        0.93,
        0.94,
        0.95,
        0.955,
        0.96,
        0.965,
        0.97,
        0.975,
        0.98,
        0.985,
        0.99,
        0.995,
        1.0,
    )
    conv_max_summary_sentences: int = 10
    conv_checkpoint_url: str = (
        "https://raw.githubusercontent.com/tingofurro/summac/"
        "525cb081993ad1c872d73a9d5d46ea3c3f73c24f/summac_conv_vitc_sent_perc_e.bin"
    )
    conv_checkpoint_sha256: str = "cd880581f59bd47b8968c7f48d754cd617dfe9b256d8d76fdcbc0fa4a3b3c6fe"

    def __post_init__(self) -> None:
        _check_text("metrics.summac", self, ("nli_model", "conv_checkpoint_url"))
        _check_digest("metrics.summac", self, ("nli_revision",), _SHA40)
        _check_digest("metrics.summac", self, ("conv_checkpoint_sha256",), _SHA256)
        _check_positive(
            "metrics.summac",
            self,
            (
                "max_document_sentences",
                "max_input_tokens",
                "batch_size",
                "conv_max_summary_sentences",
            ),
        )
        indices = (self.entailment_index, self.contradiction_index)
        if any(type(i) is not int or not 0 <= i <= 2 for i in indices) or len(set(indices)) != 2:
            raise ConfigError("metrics.summac label indices must be two distinct integers in 0..2")
        if type(self.min_sentence_chars) is not int or self.min_sentence_chars < 0:
            raise ConfigError("metrics.summac.min_sentence_chars must be a non-negative integer")
        if self.truncation not in ("longest_first", "only_first", "only_second"):
            raise ConfigError("metrics.summac.truncation must be a tokenizer truncation strategy")
        bins = self.conv_bins
        if (
            len(bins) < 2
            or bins[0] != 0.0
            or bins[-1] != 1.0
            or any(b <= a for a, b in zip(bins, bins[1:], strict=False))
        ):
            raise ConfigError("metrics.summac.conv_bins must increase strictly from 0.0 to 1.0")


@dataclass(frozen=True, slots=True)
class AlignScoreConfig:
    """AlignScore-large in its default ``nli_sp`` mode, inherited from the official code (D-023).

    Attributes:
        checkpoint_repo: HuggingFace repository holding the official checkpoints.
        checkpoint_revision: Pinned commit of ``checkpoint_repo``.
        checkpoint_file: Checkpoint file within the repository.
        checkpoint_sha256: Digest the checkpoint must match.
        backbone: Model whose configuration and tokenizer the checkpoint was trained on.
        backbone_revision: Pinned commit of ``backbone``.
        chunk_words: Target article-chunk size in whitespace words.
        batch_size: (chunk, sentence) pairs per forward pass.
        threshold: Score above which a claim counts as supported, where AlignScore judges
            entailment (the citation reward, D-135): the midpoint of its [0, 1] range, the
            setting it was validated at on long-form QA (D-129). Safe: (0, 1).
    """

    checkpoint_repo: str = "yzha/AlignScore"
    checkpoint_revision: str = "8509e78d25bb914939fc585c626500c9b2944249"
    checkpoint_file: str = "AlignScore-large.ckpt"
    checkpoint_sha256: str = "ff4336312b377edcbcdad5694a2d09d73dc4225422c0422d810aa7e78485e32d"
    backbone: str = "FacebookAI/roberta-large"
    backbone_revision: str = "722cf37b1afa9454edce342e7895e588b6ff1d59"
    chunk_words: int = 350
    batch_size: int = 32
    threshold: float = 0.5

    def __post_init__(self) -> None:
        _check_text("metrics.alignscore", self, ("checkpoint_repo", "checkpoint_file", "backbone"))
        _check_digest(
            "metrics.alignscore", self, ("checkpoint_revision", "backbone_revision"), _SHA40
        )
        _check_digest("metrics.alignscore", self, ("checkpoint_sha256",), _SHA256)
        _check_positive("metrics.alignscore", self, ("chunk_words", "batch_size"))
        if not isinstance(self.threshold, (int, float)) or not 0.0 < self.threshold < 1.0:
            raise ConfigError(
                f"metrics.alignscore.threshold must lie in (0, 1), got {self.threshold!r}"
            )


@dataclass(frozen=True, slots=True)
class MiniCheckConfig:
    """MiniCheck-Flan-T5-Large through the authors' package (D-024).

    Attributes:
        model: HuggingFace repository of the checkpoint.
        revision: Pinned commit of ``model``.
        chunk_words: Article chunk size in whitespace words; whole sentences are kept.
        max_input_tokens: Encoder input length for (chunk, claim).
        batch_size: Chunks per forward pass.
        threshold: Support probability above which a claim is labelled supported. Safe: (0, 1).
    """

    model: str = "lytang/MiniCheck-Flan-T5-Large"
    revision: str = "96eafd01cee2d16cf81aaa2fb226b14f422a37b3"
    chunk_words: int = 500
    max_input_tokens: int = 2048
    batch_size: int = 16
    threshold: float = 0.5

    def __post_init__(self) -> None:
        _check_text("metrics.minicheck", self, ("model",))
        _check_digest("metrics.minicheck", self, ("revision",), _SHA40)
        _check_positive(
            "metrics.minicheck", self, ("chunk_words", "max_input_tokens", "batch_size")
        )
        if not isinstance(self.threshold, (int, float)) or not 0.0 < self.threshold < 1.0:
            raise ConfigError("metrics.minicheck.threshold must lie strictly between 0 and 1")


@dataclass(frozen=True, slots=True)
class FactCCConfig:
    """FactCC through the HuggingFace conversion of the original checkpoint (D-025).

    Attributes:
        model: HuggingFace repository of the checkpoint.
        revision: Pinned commit of ``model``.
        max_input_tokens: Length each (article, summary) pair is padded and truncated to.
        batch_size: Pairs per forward pass. Inputs are padded to a fixed length, so it changes
            speed only (D-030).
    """

    model: str = "manueldeprada/FactCC"
    revision: str = "c7b3148015d4ddc263f6e2acb2689e90ac061669"
    max_input_tokens: int = 512
    batch_size: int = 16

    def __post_init__(self) -> None:
        _check_text("metrics.factcc", self, ("model",))
        _check_digest("metrics.factcc", self, ("revision",), _SHA40)
        _check_positive("metrics.factcc", self, ("max_input_tokens", "batch_size"))


@dataclass(frozen=True, slots=True)
class RougeLConfig:
    """ROUGE-L against the reference summary (D-026).

    Attributes:
        use_stemmer: Porter-stem tokens before matching. The library default is False.
    """

    use_stemmer: bool = False


@dataclass(frozen=True, slots=True)
class CitationConfig:
    """Citation recall and precision as ALCE defines them (D-120).

    Attributes:
        max_citations_per_sentence: Citations read per sentence; later ones are ignored, as in
            ALCE's evaluation (``--at_most_citations``, default 3; Gao et al. 2023, §2). >= 1.
        judge: The entailment judge. ``minicheck`` for every reported citation score (D-120);
            ``alignscore`` for T5's ASQA reward, which keeps MiniCheck held out (D-135).
    """

    max_citations_per_sentence: int = 3
    judge: str = "minicheck"

    def __post_init__(self) -> None:
        _check_positive("metrics.citation", self, ("max_citations_per_sentence",))
        if self.judge not in _CITATION_JUDGES:
            raise ConfigError(
                f"metrics.citation.judge must be one of {', '.join(_CITATION_JUDGES)}, "
                f"got {self.judge!r}"
            )


@dataclass(frozen=True, slots=True)
class WorkedExampleOverlapConfig:
    """Flags an output that copies T1's worked examples (D-110).

    Attributes:
        ngram: Length, in words, of a word sequence an output must share with a worked example
            (and not with its own article) to be flagged. Review #5 asked for 8. Safe range:
            >= 1; smaller values flag more chance matches.
    """

    ngram: int = 8

    def __post_init__(self) -> None:
        _check_positive("metrics.worked_example_overlap", self, ("ngram",))


@dataclass(frozen=True, slots=True)
class BertScoreConfig:
    """BERTScore at its English default (D-019).

    Attributes:
        model: Embedding model; the library's English default.
        revision: Pinned commit of ``model``.
        num_layers: Layer whose representations are compared; the library's value for the model.
    """

    model: str = "FacebookAI/roberta-large"
    revision: str = "722cf37b1afa9454edce342e7895e588b6ff1d59"
    num_layers: int = 17

    def __post_init__(self) -> None:
        _check_text("metrics.bertscore", self, ("model",))
        _check_digest("metrics.bertscore", self, ("revision",), _SHA40)
        _check_positive("metrics.bertscore", self, ("num_layers",))


@dataclass(frozen=True, slots=True)
class MetricValidationConfig:
    """Published data and numbers each metric port is validated against (D-018, D-023 to D-025).

    Attributes:
        xsumfaith_url: XSumFaith annotation file at a pinned commit.
        xsumfaith_sha256: Digest the annotation file must match.
        xsum_path: XSum dataset supplying the XSumFaith articles.
        xsum_revision: Pinned commit of ``xsum_path``.
        threshold_percentile_step: Percentile spacing of SummaC's threshold search.
        aggrefact_repo: LLM-AggreFact dataset repository.
        aggrefact_revision: Pinned commit of ``aggrefact_repo``.
        aggrefact_file: Test split file within the repository.
        aggrefact_sha256: Digest the test file must match.
        aggrefact_dataset: Subset of LLM-AggreFact used for MiniCheck.
        target_summac_zs_bacc: Laban et al. 2022, Table 2, XSumFaith.
        target_summac_conv_bacc: Laban et al. 2022, Table 2, XSumFaith.
        target_alignscore_auc: Zha et al. 2023, Table 2, XSumFaith, AlignScore-large.
        target_factcc_auc: Zha et al. 2023, Table 2, XSumFaith, FactCC.
        target_minicheck_bacc: Tang et al. 2024, Table 2, AggreFact-XSum, MiniCheck-FT5.
        aggrefact_qa_subsets: LLM-AggreFact subsets of answers checked against retrieved
            documents, as the pinned file names them, on which the ASQA judges are validated
            (D-124).
        target_minicheck_qa_bacc: Tang et al. 2024, Table 2, MiniCheck-FT5, one per subset.
        target_alignscore_qa_bacc: Tang et al. 2024, Table 2, AlignScore, one per subset.
        alignscore_threshold: AlignScore's fixed threshold for those subsets: the midpoint of
            its score range, as Tang et al. set every checker's (§4.1). Safe: (0, 1).
    """

    xsumfaith_url: str = (
        "https://raw.githubusercontent.com/google-research-datasets/xsum_hallucination_annotations/"
        "4abdbb2c807cb93c1df1188a1992d20a55cdcc34/hallucination_annotations_xsum_summaries.csv"
    )
    xsumfaith_sha256: str = "fa7fb66a36cc0f32ede4135985d0d65591dc2a8d21103a0bacd0583d77d4c8ea"
    xsum_path: str = "EdinburghNLP/xsum"
    xsum_revision: str = "7d4d486c2f8ef850b1a11aead99b894ff3dd7da9"
    threshold_percentile_step: float = 0.2
    aggrefact_repo: str = "lytang/LLM-AggreFact"
    aggrefact_revision: str = "981dfd0bd8e58e7238a9ab92b2e6ea44bce918e4"
    aggrefact_file: str = "data/test-00000-of-00001.parquet"
    aggrefact_sha256: str = "ba6976c3da990a9d7b11670efaf4e3c93c1bd0530061ec742afd6b2d5bce698c"
    aggrefact_dataset: str = "AggreFact-XSum"
    target_summac_zs_bacc: float = 58.4
    target_summac_conv_bacc: float = 66.4
    target_alignscore_auc: float = 75.8
    target_factcc_auc: float = 55.1
    target_minicheck_bacc: float = 74.3
    aggrefact_qa_subsets: tuple[str, ...] = ("Lfqa", "ClaimVerify")
    target_minicheck_qa_bacc: tuple[float, ...] = (85.2, 74.6)
    target_alignscore_qa_bacc: tuple[float, ...] = (84.5, 69.6)
    alignscore_threshold: float = 0.5

    def __post_init__(self) -> None:
        group = "metrics.validation"
        _check_text(
            group,
            self,
            ("xsumfaith_url", "xsum_path", "aggrefact_repo", "aggrefact_file", "aggrefact_dataset"),
        )
        _check_digest(group, self, ("xsum_revision", "aggrefact_revision"), _SHA40)
        _check_digest(group, self, ("xsumfaith_sha256", "aggrefact_sha256"), _SHA256)
        step = self.threshold_percentile_step
        if not isinstance(step, (int, float)) or not 0.0 < step < 100.0:
            raise ConfigError(f"{group}.threshold_percentile_step must lie in (0, 100)")
        subsets = self.aggrefact_qa_subsets
        if not subsets or any(not isinstance(s, str) or not s.strip() for s in subsets):
            raise ConfigError(f"{group}.aggrefact_qa_subsets must be non-empty strings")
        for name in ("target_minicheck_qa_bacc", "target_alignscore_qa_bacc"):
            if len(getattr(self, name)) != len(subsets):
                raise ConfigError(f"{group}.{name} needs one target per aggrefact_qa_subsets entry")
        threshold = self.alignscore_threshold
        if not isinstance(threshold, (int, float)) or not 0.0 < threshold < 1.0:
            raise ConfigError(f"{group}.alignscore_threshold must lie strictly between 0 and 1")


@dataclass(frozen=True, slots=True)
class MetricsConfig:
    """Every metric's configuration, plus where resident models run and cache (D-022).

    Attributes:
        device: Torch device for scorer models, e.g. "cuda" or "cpu".
        cache_dir: Local cache for artifacts not held by the HuggingFace cache. ``~`` is expanded.
        suite: Registered metrics every run scores, in order (D-022). ``summac`` emits both
            SummaC variants, so six names give the seven metrics.
        summac: SummaC-ZS and SummaC-Conv.
        alignscore: AlignScore.
        minicheck: MiniCheck.
        factcc: FactCC.
        rouge_l: ROUGE-L.
        bertscore: BERTScore.
        worked_example_overlap: Whether an output copies a prompt's worked examples (D-110).
        citation: ALCE citation recall and precision (D-120).
        validation: Published data and targets for validating the ports.
    """

    device: str = "cuda"
    cache_dir: str = "~/.cache/agent_evals/metrics"
    suite: tuple[str, ...] = (
        "summac",
        "factcc",
        "alignscore",
        "minicheck",
        "rouge_l",
        "bertscore",
        "worked_example_overlap",
    )
    summac: SummaCConfig = field(default_factory=SummaCConfig)
    alignscore: AlignScoreConfig = field(default_factory=AlignScoreConfig)
    minicheck: MiniCheckConfig = field(default_factory=MiniCheckConfig)
    factcc: FactCCConfig = field(default_factory=FactCCConfig)
    rouge_l: RougeLConfig = field(default_factory=RougeLConfig)
    bertscore: BertScoreConfig = field(default_factory=BertScoreConfig)
    worked_example_overlap: WorkedExampleOverlapConfig = field(
        default_factory=WorkedExampleOverlapConfig
    )
    citation: CitationConfig = field(default_factory=CitationConfig)
    validation: MetricValidationConfig = field(default_factory=MetricValidationConfig)

    def __post_init__(self) -> None:
        _check_text("metrics", self, ("device", "cache_dir"))
