from textwrap import dedent

import nbformat as nbf


ROOT_TITLE = "A Benchmark of Text Embeddings and Clustering Algorithms for Scientific Topic Discovery"
NOTEBOOK_TITLE = "Full-Dataset Exploratory Data Analysis of the arXiv Scientific Paper Corpus"


nb = nbf.v4.new_notebook()
cells = []


def md(text: str) -> None:
    cells.append(nbf.v4.new_markdown_cell(dedent(text).strip()))


def code(text: str) -> None:
    cells.append(nbf.v4.new_code_cell(dedent(text).strip()))


md(
    f"""
    # {ROOT_TITLE}
    ## {NOTEBOOK_TITLE}

    This notebook presents a full-dataset exploratory data analysis (EDA) for an arXiv-style scientific paper corpus. The objective is not only to describe the data, but to generate evidence that directly informs our downstream benchmark of text embeddings, clustering algorithms, and category-based evaluation for scientific topic discovery.

    **Notebook objectives**
    - characterize the structure, scale, and completeness of the dataset,
    - quantify category imbalance, multi-label behavior, and domain skew,
    - inspect title and abstract properties that matter for scientific embeddings,
    - evaluate preprocessing choices in a modeling-aware way,
    - justify a train/validation/test split strategy on the full corpus, and
    - identify risks and design choices that should be stated explicitly in the project report.
    """
)

md(
    """
    **Project Context**

    The downstream task is scientific topic discovery, not supervised classification. That changes what matters in EDA:

    - arXiv categories act as **reference labels** for evaluating clusters, so their imbalance and multi-label structure affect how fair our benchmark will be,
    - scientific abstracts contain formulas, markup, URLs, and discipline-specific notation that influence both classical baselines and transformer embeddings,
    - temporal growth and domain skew can make a benchmark appear broader than it really is if recent computer-science-heavy growth is not documented, and
    - computational scale matters because methods that look reasonable on small subsets can become infeasible on a corpus with millions of documents.

    This notebook is designed to operate on the **entire dataset**. The expensive work happens in a cache-building step that streams through the raw JSONL file and saves reusable artifacts. Displayed examples are selective for readability, but the reported statistics, split analysis, and core distributions are built from the full corpus.
    """
)

md(
    """
    **Colab Note**

    This notebook is designed to run in Google Colab as well as a local Jupyter environment. For this EDA workflow, the limiting resources are **CPU throughput and available RAM**, not GPU acceleration. A GPU-enabled runtime is fine, but a **high-RAM runtime** is more important than a GPU for the expensive full-dataset passes.
    """
)

md("## 1. Imports and Setup")

md(
    """
    The first code cell validates the active kernel environment and installs any missing packages into that same environment. This makes the notebook easier to run both locally and in Google Colab.
    """
)

code(
    """
    import importlib
    import subprocess
    import sys

    import csv
    import gzip
    import hashlib
    import json
    import random
    import re
    import time
    from array import array
    from collections import Counter, defaultdict
    from datetime import datetime
    from email.utils import parsedate_to_datetime
    from pathlib import Path

    REQUIRED_PACKAGES = {
        "numpy": "numpy",
        "pandas": "pandas",
        "matplotlib": "matplotlib",
        "seaborn": "seaborn",
        "sklearn": "scikit-learn",
        "IPython": "ipython",
        "nbformat": "nbformat",
    }

    missing_packages = []
    for module_name, package_name in REQUIRED_PACKAGES.items():
        try:
            importlib.import_module(module_name)
        except Exception:
            missing_packages.append(package_name)

    if missing_packages:
        missing_packages = sorted(set(missing_packages))
        print(f"Active kernel Python: {sys.executable}")
        print("Installing missing packages into this kernel environment:", ", ".join(missing_packages))
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", *missing_packages])
        except Exception as exc:
            raise RuntimeError(
                "Automatic dependency installation failed. Install `requirements.txt` into the same "
                "environment as the notebook and rerun the first cell."
            ) from exc
        print("Dependency installation complete.")
    else:
        print(f"Active kernel Python: {sys.executable}")
        print("All required notebook packages are already installed.")

    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import seaborn as sns
    from IPython.display import Markdown, display
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
    from sklearn.model_selection import train_test_split

    sns.set_theme(style="whitegrid", context="talk")
    plt.rcParams["figure.figsize"] = (12, 6)
    plt.rcParams["figure.dpi"] = 120
    plt.rcParams["axes.titlesize"] = 16
    plt.rcParams["axes.labelsize"] = 12
    plt.rcParams["xtick.labelsize"] = 10
    plt.rcParams["ytick.labelsize"] = 10
    plt.rcParams["legend.fontsize"] = 10
    pd.options.display.max_colwidth = 180
    pd.options.display.float_format = lambda x: f"{x:,.3f}"

    IS_COLAB = "google.colab" in sys.modules

    DATA_CANDIDATES = [
        Path("arxiv-metadata-oai-snapshot.json"),
        Path("/content/arxiv-metadata-oai-snapshot.json"),
        Path("/content/drive/MyDrive/arxiv-metadata-oai-snapshot.json"),
    ]
    DATA_PATH = next((path for path in DATA_CANDIDATES if path.exists()), DATA_CANDIDATES[0])
    ARTIFACT_DIR = Path("/content/artifacts/eda_full") if IS_COLAB else Path("artifacts/eda_full")
    ARTIFACT_VERSION = 2
    RANDOM_SEED = 42
    TOP_TOKEN_LIMIT = 2000
    SPLIT_THRESHOLD_OPTIONS = (2, 3, 5, 10, 20)
    REFRESH_ARTIFACTS = False

    assert DATA_PATH.exists(), f"Dataset not found: {DATA_PATH.resolve()}"

    print(f"Running in Google Colab: {IS_COLAB}")
    print(f"Dataset: {DATA_PATH.resolve()}")
    print(f"Dataset size: {DATA_PATH.stat().st_size / 1024**3:,.2f} GB")
    print(f"Artifact directory: {ARTIFACT_DIR.resolve()}")
    """
)

md(
    """
    ## 2. Utilities and Artifact Build

    The notebook is designed around a cache-first workflow. On the first run, it performs a full streaming pass over the raw dataset, computes the required summaries, and saves them in `artifacts/eda_full`. On later runs, those artifacts are reused immediately.

    This is the only expensive stage. It is intentionally front-loaded so the rest of the notebook remains presentation-quality and easy to rerun.
    """
)

code(
    """
    CATEGORY_RE = re.compile(r"^[A-Za-z\\-]+(?:\\.[A-Za-z\\-]+)?$")
    ID_RE = re.compile(r"^(?:[a-z\\-]+(?:\\.[A-Za-z\\-]+)?/\\d{7}|\\d{4}\\.\\d{4,5})$")
    WS_RE = re.compile(r"\\s+")
    NON_ALNUM_SPACE_RE = re.compile(r"[^a-z0-9\\s]")
    TOKEN_RE = re.compile(r"[A-Za-z0-9_+\\-\\.']+")
    HTML_RE = re.compile(r"<[^>]+>")
    LATEX_RE = re.compile(r"\\\\[A-Za-z]+|\\$[^$]+\\$|\\\\\\(|\\\\\\)|\\\\\\[|\\\\\\]|\\\\begin\\{|\\\\end\\{")
    URL_RE = re.compile(r"https?://|www\\.")
    EMAIL_RE = re.compile(r"\\b[\\w.%-]+@[\\w.-]+\\.[A-Za-z]{2,}\\b")
    EXCESS_PUNCT_RE = re.compile(r"[!?.,;:]{4,}")
    WITHDRAWN_RE = re.compile(r"withdrawn|retracted", re.I)
    NON_ASCII_RE = re.compile(r"[^\\x00-\\x7F]")
    NEWLINE_RE = re.compile(r"\\n")


    def normalize_ws(text):
        if text is None:
            return ""
        return WS_RE.sub(" ", str(text)).strip()


    def split_categories(value):
        text = normalize_ws(value)
        return text.split() if text else []


    def get_domain(category):
        if not category:
            return "unknown"
        return category.split(".", 1)[0]


    def counter_to_frame(counter, key_name, value_name="count"):
        frame = pd.DataFrame(counter.items(), columns=[key_name, value_name])
        if not frame.empty:
            frame = frame.sort_values(value_name, ascending=False).reset_index(drop=True)
        return frame


    def percentile_summary(values):
        arr = np.asarray(values, dtype=np.float64)
        percentiles = [0, 25, 50, 75, 90, 95, 99, 100]
        return {str(p): float(np.percentile(arr, p)) for p in percentiles}


    def clip_text(text, width=220):
        text = normalize_ws(text)
        return text if len(text) <= width else text[: width - 3] + "..."


    def load_json(path):
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)


    def update_smallest_examples(rows, row, metric_key, limit=4):
        rows.append(row)
        rows.sort(key=lambda item: (item.get(metric_key, 0), item.get("id", "")))
        del rows[limit:]


    def update_largest_examples(rows, row, metric_key, limit=4):
        rows.append(row)
        rows.sort(key=lambda item: (item.get(metric_key, 0), item.get("id", "")), reverse=True)
        del rows[limit:]


    def tokenize_raw_lower(text):
        return [token.lower() for token in TOKEN_RE.findall(text)]


    def tokenize_no_punct(text):
        lowered = NON_ALNUM_SPACE_RE.sub(" ", text.lower())
        return [token for token in WS_RE.split(lowered) if token]


    def tokenize_no_punct_stop(text, stopwords):
        return [token for token in tokenize_no_punct(text) if token not in stopwords]
    """
)

code(
    """
    def build_split_artifacts(primary_label_codes, code_to_label, artifact_dir, random_state=RANDOM_SEED):
        labels = np.asarray(primary_label_codes, dtype=np.uint16)
        label_counts = pd.Series(labels).value_counts().sort_index()
        threshold_rows = []

        for threshold in SPLIT_THRESHOLD_OPTIONS:
            eligible_codes = label_counts[label_counts >= threshold].index.to_numpy(dtype=np.uint16)
            eligible = labels[np.isin(labels, eligible_codes)]
            works = True
            try:
                train_tmp, heldout_tmp = train_test_split(
                    eligible,
                    test_size=0.30,
                    random_state=random_state,
                    stratify=eligible,
                )
                train_test_split(
                    heldout_tmp,
                    test_size=2 / 3,
                    random_state=random_state,
                    stratify=heldout_tmp,
                )
            except Exception:
                works = False

            threshold_rows.append(
                {
                    "minimum_count_per_primary_category": int(threshold),
                    "eligible_rows": int(len(eligible)),
                    "eligible_primary_categories": int(len(eligible_codes)),
                    "stratified_70_10_20_feasible": bool(works),
                }
            )

        threshold_df = pd.DataFrame(threshold_rows)
        threshold_df.to_csv(artifact_dir / "split_threshold_summary.csv", index=False)

        train_random, temp_random = train_test_split(labels, test_size=0.30, random_state=random_state)
        val_random, test_random = train_test_split(temp_random, test_size=2 / 3, random_state=random_state)

        train_strat, temp_strat = train_test_split(
            labels,
            test_size=0.30,
            random_state=random_state,
            stratify=labels,
        )
        val_strat, test_strat = train_test_split(
            temp_strat,
            test_size=2 / 3,
            random_state=random_state,
            stratify=temp_strat,
        )

        global_dist = pd.Series(labels).value_counts(normalize=True).sort_index()

        def summarize_strategy(strategy_name, splits):
            rows = []
            drift_rows = []
            for split_name, split_labels in splits.items():
                split_dist = pd.Series(split_labels).value_counts(normalize=True).sort_index()
                aligned = global_dist.to_frame("global_share").join(
                    split_dist.to_frame("split_share"),
                    how="left",
                ).fillna(0)
                aligned["abs_diff_pct_points"] = (aligned["split_share"] - aligned["global_share"]).abs() * 100

                rows.append(
                    {
                        "strategy": strategy_name,
                        "split": split_name,
                        "size": int(len(split_labels)),
                        "size_share_pct": float(len(split_labels) / len(labels) * 100),
                        "mean_abs_diff_pct_points": float(aligned["abs_diff_pct_points"].mean()),
                        "max_abs_diff_pct_points": float(aligned["abs_diff_pct_points"].max()),
                        "missing_primary_categories": int((aligned["split_share"] == 0).sum()),
                    }
                )

                for code_value, row in aligned.iterrows():
                    drift_rows.append(
                        {
                            "strategy": strategy_name,
                            "split": split_name,
                            "primary_category": code_to_label[int(code_value)],
                            "global_share_pct": float(row["global_share"] * 100),
                            "split_share_pct": float(row["split_share"] * 100),
                            "abs_diff_pct_points": float(row["abs_diff_pct_points"]),
                        }
                    )

            return rows, drift_rows

        random_rows, random_drift = summarize_strategy(
            "Random",
            {"Train": train_random, "Validation": val_random, "Test": test_random},
        )
        strat_rows, strat_drift = summarize_strategy(
            "Stratified by primary category",
            {"Train": train_strat, "Validation": val_strat, "Test": test_strat},
        )

        pd.DataFrame(random_rows + strat_rows).to_csv(artifact_dir / "split_strategy_summary.csv", index=False)
        pd.DataFrame(random_drift + strat_drift).to_csv(artifact_dir / "split_drift_tables.csv", index=False)


    def build_token_artifacts(data_path, artifact_dir, progress_every=150_000):
        stopwords = set(ENGLISH_STOP_WORDS)
        tokenizers = [
            ("Raw lowercase tokenization", lambda text: tokenize_raw_lower(text)),
            ("Lowercase + punctuation removal", lambda text: tokenize_no_punct(text)),
            (
                "Lowercase + punctuation removal + stopword removal",
                lambda text: tokenize_no_punct_stop(text, stopwords),
            ),
        ]

        token_metric_rows = []
        top_token_rows = []

        for config_name, tokenizer in tokenizers:
            counter = Counter()
            total_tokens = 0
            stopword_tokens = 0
            numeric_tokens = 0
            punct_tokens = 0
            start = time.time()

            with data_path.open("r", encoding="utf-8") as f:
                for idx, line in enumerate(f, start=1):
                    if not line.strip():
                        continue

                    record = json.loads(line)
                    combined_text = f"{normalize_ws(record.get('title'))} {normalize_ws(record.get('abstract'))}".strip()
                    if not combined_text:
                        continue

                    tokens = tokenizer(combined_text)
                    counter.update(tokens)
                    total_tokens += len(tokens)
                    stopword_tokens += sum(token in stopwords for token in tokens)
                    numeric_tokens += sum(any(ch.isdigit() for ch in token) for token in tokens)
                    punct_tokens += sum(bool(re.search(r"[^\\w\\s]", token)) for token in tokens)

                    if idx % progress_every == 0:
                        elapsed = (time.time() - start) / 60
                        print(f"[{config_name}] processed {idx:,} records in {elapsed:.1f} minutes")

            counts = np.fromiter(counter.values(), dtype=np.int64)
            vocab_size = int(len(counter))
            hapax_count = int((counts == 1).sum()) if vocab_size else 0
            rare_le2_count = int((counts <= 2).sum()) if vocab_size else 0
            rare_le2_mass = int(counts[counts <= 2].sum()) if vocab_size else 0
            common_ge1000_count = int((counts >= 1000).sum()) if vocab_size else 0
            common_ge1000_mass = int(counts[counts >= 1000].sum()) if vocab_size else 0
            top100_token_share = float(sum(count for _, count in counter.most_common(100)) / total_tokens * 100) if total_tokens else 0.0

            token_metric_rows.append(
                {
                    "configuration": config_name,
                    "vocab_size": vocab_size,
                    "total_tokens": int(total_tokens),
                    "hapax_vocab_pct": float(hapax_count / vocab_size * 100) if vocab_size else 0.0,
                    "rare_le2_vocab_pct": float(rare_le2_count / vocab_size * 100) if vocab_size else 0.0,
                    "rare_le2_token_mass_pct": float(rare_le2_mass / total_tokens * 100) if total_tokens else 0.0,
                    "common_ge1000_vocab_pct": float(common_ge1000_count / vocab_size * 100) if vocab_size else 0.0,
                    "common_ge1000_token_mass_pct": float(common_ge1000_mass / total_tokens * 100) if total_tokens else 0.0,
                    "stopword_token_pct": float(stopword_tokens / total_tokens * 100) if total_tokens else 0.0,
                    "numeric_token_pct": float(numeric_tokens / total_tokens * 100) if total_tokens else 0.0,
                    "punct_token_pct": float(punct_tokens / total_tokens * 100) if total_tokens else 0.0,
                    "top100_token_share_pct": top100_token_share,
                }
            )

            for rank, (token, count) in enumerate(counter.most_common(TOP_TOKEN_LIMIT), start=1):
                top_token_rows.append(
                    {
                        "configuration": config_name,
                        "rank": int(rank),
                        "token": token,
                        "count": int(count),
                    }
                )

        token_metrics_df = pd.DataFrame(token_metric_rows)
        raw_vocab = token_metrics_df.loc[
            token_metrics_df["configuration"] == "Raw lowercase tokenization",
            "vocab_size",
        ].iloc[0]
        token_metrics_df["vocab_reduction_vs_raw_pct"] = (
            1 - token_metrics_df["vocab_size"] / raw_vocab
        ) * 100

        token_metrics_df.to_csv(artifact_dir / "token_metrics.csv", index=False)
        pd.DataFrame(top_token_rows).to_csv(artifact_dir / "token_top_terms.csv", index=False)


    def build_eda_artifacts(
        data_path,
        artifact_dir,
        artifact_version=ARTIFACT_VERSION,
        seed=RANDOM_SEED,
        preview_limit=5,
        example_limit=4,
        progress_every=150_000,
    ):
        artifact_dir.mkdir(parents=True, exist_ok=True)

        field_names = set()
        field_types = defaultdict(Counter)
        missing_counts = Counter()
        empty_string_counts = Counter()
        whitespace_only_counts = Counter()

        all_category_counts = Counter()
        primary_category_counts = Counter()
        primary_domain_counts = Counter()
        all_domain_counts = Counter()
        label_count_distribution = Counter()
        combo_counts = Counter()
        author_count_distribution = Counter()
        version_count_distribution = Counter()
        submission_year_counts = Counter()
        submission_month_counts = Counter()
        update_year_counts = Counter()
        update_lag_years = Counter()
        primary_category_year_counts = defaultdict(Counter)
        primary_domain_year_counts = defaultdict(Counter)
        malformed_category_token_counts = Counter()

        primary_domain_abstract_lengths = defaultdict(lambda: array("H"))
        pattern_counts = Counter()

        preview_rows = []
        normal_examples = []
        problematic_examples = []
        short_abstract_examples = []
        long_abstract_examples = []

        suspect_id_count = 0
        invalid_submission_dates = 0
        invalid_update_dates = 0
        missing_primary_category = 0
        duplicate_paper_ids = 0
        duplicate_paper_id_examples = []
        seen_ids = set()

        primary_label_to_code = {}
        code_to_primary_label = []
        primary_label_codes = array("H")

        exact_title_hashes = array("Q")
        normalized_title_hashes = array("Q")
        content_hashes = array("Q")
        title_char_lengths = array("I")
        title_word_lengths = array("I")
        abstract_char_lengths = array("I")
        abstract_word_lengths = array("I")
        author_counts_array = array("H")
        label_counts_array = array("B")
        version_counts_array = array("H")

        start = time.time()
        record_count = 0

        with data_path.open("r", encoding="utf-8") as f:
            for idx, line in enumerate(f, start=1):
                if not line.strip():
                    continue

                record_count = idx
                record = json.loads(line)
                field_names.update(record.keys())

                if idx <= 10_000:
                    for key, value in record.items():
                        field_types[key][type(value).__name__] += 1

                for key, value in record.items():
                    if value is None:
                        missing_counts[key] += 1
                    elif isinstance(value, str):
                        if value == "":
                            empty_string_counts[key] += 1
                        elif not value.strip():
                            whitespace_only_counts[key] += 1

                paper_id = normalize_ws(record.get("id"))
                if paper_id in seen_ids:
                    duplicate_paper_ids += 1
                    if len(duplicate_paper_id_examples) < 10:
                        duplicate_paper_id_examples.append(paper_id)
                else:
                    seen_ids.add(paper_id)

                if paper_id and not ID_RE.match(paper_id):
                    suspect_id_count += 1

                raw_title = str(record.get("title") or "")
                raw_abstract = str(record.get("abstract") or "")
                title = normalize_ws(raw_title)
                abstract = normalize_ws(raw_abstract)
                categories = split_categories(record.get("categories"))
                primary_category = categories[0] if categories else ""
                primary_domain = get_domain(primary_category) if primary_category else "unknown"

                authors_parsed = record.get("authors_parsed")
                author_count = len(authors_parsed) if isinstance(authors_parsed, list) else 0
                version_count = len(record.get("versions") or []) if isinstance(record.get("versions"), list) else 0

                title_char_len = len(title)
                title_word_len = len(title.split())
                abstract_char_len = len(abstract)
                abstract_word_len = len(abstract.split())

                title_char_lengths.append(title_char_len)
                title_word_lengths.append(title_word_len)
                abstract_char_lengths.append(abstract_char_len)
                abstract_word_lengths.append(abstract_word_len)
                author_counts_array.append(author_count)
                label_counts_array.append(len(categories))
                version_counts_array.append(version_count)

                author_count_distribution[author_count] += 1
                version_count_distribution[version_count] += 1
                primary_domain_abstract_lengths[primary_domain].append(abstract_word_len)

                exact_title_hash = (
                    int.from_bytes(hashlib.blake2b(title.encode("utf-8"), digest_size=8).digest(), "big")
                    if title
                    else 0
                )
                normalized_title = normalize_ws(NON_ALNUM_SPACE_RE.sub(" ", title.lower())) if title else ""
                normalized_title_hash = (
                    int.from_bytes(hashlib.blake2b(normalized_title.encode("utf-8"), digest_size=8).digest(), "big")
                    if normalized_title
                    else 0
                )
                content_text = f"{normalized_title}\\n{abstract.lower()}"
                content_hash = (
                    int.from_bytes(hashlib.blake2b(content_text.encode("utf-8"), digest_size=8).digest(), "big")
                    if content_text.strip()
                    else 0
                )

                exact_title_hashes.append(exact_title_hash)
                normalized_title_hashes.append(normalized_title_hash)
                content_hashes.append(content_hash)

                if categories:
                    all_category_counts.update(categories)
                    primary_category_counts[primary_category] += 1
                    primary_domain_counts[primary_domain] += 1
                    all_domain_counts.update(get_domain(cat) for cat in categories)
                    label_count_distribution[len(categories)] += 1
                    if len(categories) > 1:
                        combo_counts[" | ".join(categories)] += 1
                    malformed_category_token_counts.update(cat for cat in categories if not CATEGORY_RE.match(cat))

                    if primary_category not in primary_label_to_code:
                        primary_label_to_code[primary_category] = len(code_to_primary_label)
                        code_to_primary_label.append(primary_category)
                    primary_label_codes.append(primary_label_to_code[primary_category])
                else:
                    missing_primary_category += 1

                submission_year = None
                versions = record.get("versions")
                if isinstance(versions, list) and versions:
                    created = versions[0].get("created")
                    try:
                        submitted_at = parsedate_to_datetime(created)
                        submission_year = int(submitted_at.year)
                        submission_year_counts[submission_year] += 1
                        submission_month_counts[f"{submitted_at.year:04d}-{submitted_at.month:02d}"] += 1
                    except Exception:
                        invalid_submission_dates += 1
                else:
                    invalid_submission_dates += 1

                update_year = None
                update_date = normalize_ws(record.get("update_date"))
                if update_date:
                    try:
                        update_dt = datetime.strptime(update_date, "%Y-%m-%d")
                        update_year = int(update_dt.year)
                        update_year_counts[update_year] += 1
                    except Exception:
                        invalid_update_dates += 1
                else:
                    invalid_update_dates += 1

                if submission_year is not None and update_year is not None:
                    update_lag_years[max(update_year - submission_year, 0)] += 1
                    if primary_category:
                        primary_category_year_counts[primary_category][submission_year] += 1
                        primary_domain_year_counts[primary_domain][submission_year] += 1

                issue_tags = []
                if LATEX_RE.search(title):
                    pattern_counts[("Title", "LaTeX-like markup")] += 1
                    issue_tags.append("title-latex")
                if NEWLINE_RE.search(raw_title):
                    pattern_counts[("Title", "newline artifact")] += 1
                    issue_tags.append("title-newline")
                if HTML_RE.search(raw_abstract):
                    pattern_counts[("Abstract", "HTML-like tag")] += 1
                    issue_tags.append("html")
                if LATEX_RE.search(raw_abstract):
                    pattern_counts[("Abstract", "LaTeX-like markup")] += 1
                    issue_tags.append("latex")
                if URL_RE.search(raw_abstract):
                    pattern_counts[("Abstract", "URL")] += 1
                    issue_tags.append("url")
                if EMAIL_RE.search(raw_abstract):
                    pattern_counts[("Abstract", "Email address")] += 1
                    issue_tags.append("email")
                if EXCESS_PUNCT_RE.search(raw_abstract):
                    pattern_counts[("Abstract", "Excess punctuation")] += 1
                    issue_tags.append("punctuation")
                if WITHDRAWN_RE.search(raw_abstract):
                    pattern_counts[("Abstract", "Withdrawn / retracted wording")] += 1
                    issue_tags.append("withdrawn")
                if NON_ASCII_RE.search(raw_abstract):
                    pattern_counts[("Abstract", "Non-ASCII character")] += 1
                    issue_tags.append("non-ascii")
                if NEWLINE_RE.search(raw_abstract):
                    pattern_counts[("Abstract", "newline artifact")] += 1
                    issue_tags.append("abstract-newline")

                example_row = {
                    "id": paper_id,
                    "primary_category": primary_category,
                    "primary_domain": primary_domain,
                    "title": title,
                    "abstract_preview": clip_text(abstract, 280),
                    "title_word_len": int(title_word_len),
                    "abstract_word_len": int(abstract_word_len),
                }

                if len(preview_rows) < preview_limit:
                    preview_rows.append(
                        {
                            "id": paper_id,
                            "title": title,
                            "categories": " ".join(categories),
                            "update_date": update_date,
                            "authors": normalize_ws(record.get("authors")),
                            "abstract": clip_text(abstract, width=420),
                        }
                    )

                if (
                    110 <= abstract_word_len <= 190
                    and not issue_tags
                    and len(normal_examples) < example_limit
                ):
                    normal_examples.append(example_row)

                if issue_tags and len(problematic_examples) < example_limit:
                    tagged = dict(example_row)
                    tagged["issue_tags"] = ", ".join(issue_tags)
                    problematic_examples.append(tagged)

                if abstract_word_len > 0:
                    update_smallest_examples(short_abstract_examples, example_row, "abstract_word_len", example_limit)
                    update_largest_examples(long_abstract_examples, example_row, "abstract_word_len", example_limit)

                if idx % progress_every == 0:
                    elapsed = (time.time() - start) / 60
                    print(f"Processed {idx:,} records in {elapsed:.1f} minutes")

        print(f"Finished full pass over {record_count:,} records.")

        np.savez_compressed(
            artifact_dir / "length_arrays.npz",
            title_char_len=np.asarray(title_char_lengths, dtype=np.uint32),
            title_word_len=np.asarray(title_word_lengths, dtype=np.uint16),
            abstract_char_len=np.asarray(abstract_char_lengths, dtype=np.uint32),
            abstract_word_len=np.asarray(abstract_word_lengths, dtype=np.uint16),
            author_count=np.asarray(author_counts_array, dtype=np.uint16),
            label_count=np.asarray(label_counts_array, dtype=np.uint8),
            version_count=np.asarray(version_counts_array, dtype=np.uint16),
        )

        column_rows = []
        for field in sorted(field_names):
            type_counter = field_types.get(field, Counter())
            inferred_dtype = type_counter.most_common(1)[0][0] if type_counter else "unknown"
            missing_count = int(missing_counts.get(field, 0))
            column_rows.append(
                {
                    "column": field,
                    "inferred_dtype_sample": inferred_dtype,
                    "missing_count": missing_count,
                    "missing_pct": float(missing_count / record_count * 100),
                    "non_null_count": int(record_count - missing_count),
                    "empty_string_count": int(empty_string_counts.get(field, 0)),
                    "whitespace_only_count": int(whitespace_only_counts.get(field, 0)),
                }
            )
        pd.DataFrame(column_rows).to_csv(artifact_dir / "column_overview.csv", index=False)

        all_category_df = counter_to_frame(all_category_counts, "category")
        all_category_df["share_pct"] = all_category_df["count"] / record_count * 100
        all_category_df.to_csv(artifact_dir / "all_category_counts.csv", index=False)

        primary_category_df = counter_to_frame(primary_category_counts, "primary_category")
        primary_category_df["share_pct"] = primary_category_df["count"] / record_count * 100
        primary_category_df.to_csv(artifact_dir / "primary_category_counts.csv", index=False)

        primary_domain_df = counter_to_frame(primary_domain_counts, "primary_domain")
        primary_domain_df["share_pct"] = primary_domain_df["count"] / record_count * 100
        primary_domain_df.to_csv(artifact_dir / "primary_domain_counts.csv", index=False)

        all_domain_df = counter_to_frame(all_domain_counts, "domain")
        all_domain_df["share_pct"] = all_domain_df["count"] / all_domain_df["count"].sum() * 100
        all_domain_df.to_csv(artifact_dir / "all_domain_counts.csv", index=False)

        label_count_df = counter_to_frame(label_count_distribution, "label_count")
        label_count_df["share_pct"] = label_count_df["count"] / record_count * 100
        label_count_df = label_count_df.sort_values("label_count").reset_index(drop=True)
        label_count_df.to_csv(artifact_dir / "label_count_distribution.csv", index=False)

        combo_df = counter_to_frame(combo_counts, "category_combination").head(200)
        combo_df["share_pct"] = combo_df["count"] / record_count * 100
        combo_df.to_csv(artifact_dir / "category_combination_counts_top200.csv", index=False)

        submission_year_df = counter_to_frame(submission_year_counts, "submission_year")
        submission_year_df = submission_year_df.sort_values("submission_year").reset_index(drop=True)
        submission_year_df.to_csv(artifact_dir / "submission_year_counts.csv", index=False)

        submission_month_df = counter_to_frame(submission_month_counts, "submission_month")
        submission_month_df = submission_month_df.sort_values("submission_month").reset_index(drop=True)
        submission_month_df.to_csv(artifact_dir / "submission_month_counts.csv", index=False)

        primary_category_year_rows = []
        for category, year_counts in primary_category_year_counts.items():
            for year, count in year_counts.items():
                primary_category_year_rows.append(
                    {"primary_category": category, "submission_year": year, "count": int(count)}
                )
        pd.DataFrame(primary_category_year_rows).to_csv(
            artifact_dir / "primary_category_year_counts.csv",
            index=False,
        )

        primary_domain_year_rows = []
        for domain, year_counts in primary_domain_year_counts.items():
            for year, count in year_counts.items():
                primary_domain_year_rows.append(
                    {"primary_domain": domain, "submission_year": year, "count": int(count)}
                )
        pd.DataFrame(primary_domain_year_rows).to_csv(
            artifact_dir / "primary_domain_year_counts.csv",
            index=False,
        )

        counter_to_frame(author_count_distribution, "author_count").sort_values("author_count").to_csv(
            artifact_dir / "author_count_distribution.csv",
            index=False,
        )
        counter_to_frame(version_count_distribution, "version_count").sort_values("version_count").to_csv(
            artifact_dir / "version_count_distribution.csv",
            index=False,
        )
        counter_to_frame(update_year_counts, "update_year").sort_values("update_year").to_csv(
            artifact_dir / "update_year_counts.csv",
            index=False,
        )
        counter_to_frame(update_lag_years, "update_lag_years").sort_values("update_lag_years").to_csv(
            artifact_dir / "update_lag_year_distribution.csv",
            index=False,
        )
        counter_to_frame(malformed_category_token_counts, "malformed_category").to_csv(
            artifact_dir / "malformed_category_tokens.csv",
            index=False,
        )

        pattern_rows = []
        for (field_name, pattern_name), count in sorted(pattern_counts.items()):
            pattern_rows.append(
                {
                    "field": field_name,
                    "pattern": pattern_name,
                    "count": int(count),
                    "share_pct": float(count / record_count * 100),
                }
            )
        pd.DataFrame(pattern_rows).sort_values(["field", "count"], ascending=[True, False]).to_csv(
            artifact_dir / "text_artifact_patterns.csv",
            index=False,
        )

        domain_length_rows = []
        for domain, values in primary_domain_abstract_lengths.items():
            if not values:
                continue
            domain_length_rows.append(
                {
                    "primary_domain": domain,
                    "paper_count": int(len(values)),
                    "p25_abstract_words": float(np.percentile(values, 25)),
                    "median_abstract_words": float(np.percentile(values, 50)),
                    "p75_abstract_words": float(np.percentile(values, 75)),
                    "p90_abstract_words": float(np.percentile(values, 90)),
                }
            )
        pd.DataFrame(domain_length_rows).sort_values("paper_count", ascending=False).to_csv(
            artifact_dir / "domain_abstract_length_summary.csv",
            index=False,
        )

        with (artifact_dir / "preview_rows.json").open("w", encoding="utf-8") as f:
            json.dump(preview_rows, f, ensure_ascii=False, indent=2)

        with (artifact_dir / "example_rows.json").open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "normal_examples": normal_examples,
                    "problematic_examples": problematic_examples,
                    "short_abstract_examples": short_abstract_examples,
                    "long_abstract_examples": long_abstract_examples,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )

        def summarize_hash_duplicates(hash_values):
            arr = np.asarray(hash_values, dtype=np.uint64)
            unique_hashes, counts = np.unique(arr, return_counts=True)
            duplicate_mask = counts > 1
            top = []
            if duplicate_mask.any():
                dup_hashes = unique_hashes[duplicate_mask]
                dup_counts = counts[duplicate_mask]
                order = np.argsort(dup_counts)[::-1][:20]
                top = [{"hash": int(dup_hashes[i]), "count": int(dup_counts[i])} for i in order]
            return {
                "duplicated_unique_values": int(duplicate_mask.sum()),
                "duplicate_records": int((counts[duplicate_mask] - 1).sum()),
                "top_hashes": top,
            }

        duplicate_summary = {
            "exact_title": summarize_hash_duplicates(exact_title_hashes),
            "normalized_title": summarize_hash_duplicates(normalized_title_hashes),
            "title_abstract_content": summarize_hash_duplicates(content_hashes),
        }

        top_hash_lookup = {
            "exact_title": {item["hash"] for item in duplicate_summary["exact_title"]["top_hashes"]},
            "normalized_title": {item["hash"] for item in duplicate_summary["normalized_title"]["top_hashes"]},
            "title_abstract_content": {item["hash"] for item in duplicate_summary["title_abstract_content"]["top_hashes"]},
        }

        duplicate_examples = {
            "exact_title": {},
            "normalized_title": {},
            "title_abstract_content": {},
        }
        with data_path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue

                record = json.loads(line)
                title = normalize_ws(record.get("title"))
                abstract = normalize_ws(record.get("abstract"))
                categories = " ".join(split_categories(record.get("categories")))
                example = {
                    "id": normalize_ws(record.get("id")),
                    "title": title,
                    "categories": categories,
                    "abstract_preview": clip_text(abstract, width=300),
                }
                exact_hash = (
                    int.from_bytes(hashlib.blake2b(title.encode("utf-8"), digest_size=8).digest(), "big")
                    if title
                    else 0
                )
                normalized_title = normalize_ws(NON_ALNUM_SPACE_RE.sub(" ", title.lower())) if title else ""
                norm_hash = (
                    int.from_bytes(hashlib.blake2b(normalized_title.encode("utf-8"), digest_size=8).digest(), "big")
                    if normalized_title
                    else 0
                )
                content_text = f"{normalized_title}\\n{abstract.lower()}"
                content_hash = (
                    int.from_bytes(hashlib.blake2b(content_text.encode("utf-8"), digest_size=8).digest(), "big")
                    if content_text.strip()
                    else 0
                )

                if exact_hash in top_hash_lookup["exact_title"] and str(exact_hash) not in duplicate_examples["exact_title"]:
                    duplicate_examples["exact_title"][str(exact_hash)] = example
                if norm_hash in top_hash_lookup["normalized_title"] and str(norm_hash) not in duplicate_examples["normalized_title"]:
                    duplicate_examples["normalized_title"][str(norm_hash)] = example
                if content_hash in top_hash_lookup["title_abstract_content"] and str(content_hash) not in duplicate_examples["title_abstract_content"]:
                    duplicate_examples["title_abstract_content"][str(content_hash)] = example

        with (artifact_dir / "duplicate_examples.json").open("w", encoding="utf-8") as f:
            json.dump(duplicate_examples, f, ensure_ascii=False, indent=2)

        recent_2018_share = (
            float(submission_year_df.loc[submission_year_df["submission_year"] >= 2018, "count"].sum() / record_count * 100)
            if not submission_year_df.empty
            else 0.0
        )
        recent_2020_share = (
            float(submission_year_df.loc[submission_year_df["submission_year"] >= 2020, "count"].sum() / record_count * 100)
            if not submission_year_df.empty
            else 0.0
        )

        summary = {
            "artifact_version": int(artifact_version),
            "data_path": str(data_path),
            "record_count": int(record_count),
            "field_names": sorted(field_names),
            "unique_all_categories": int(len(all_category_counts)),
            "unique_primary_categories": int(len(primary_category_counts)),
            "unique_primary_domains": int(len(primary_domain_counts)),
            "suspect_id_count": int(suspect_id_count),
            "invalid_submission_dates": int(invalid_submission_dates),
            "invalid_update_dates": int(invalid_update_dates),
            "missing_primary_category": int(missing_primary_category),
            "duplicate_paper_ids": int(duplicate_paper_ids),
            "duplicate_paper_id_examples": duplicate_paper_id_examples,
            "missing_counts": {k: int(v) for k, v in missing_counts.items()},
            "empty_string_counts": {k: int(v) for k, v in empty_string_counts.items()},
            "whitespace_only_counts": {k: int(v) for k, v in whitespace_only_counts.items()},
            "top_primary_category_share_pct": float(primary_category_df.iloc[0]["share_pct"]) if not primary_category_df.empty else 0.0,
            "top_primary_domain_share_pct": float(primary_domain_df.iloc[0]["share_pct"]) if not primary_domain_df.empty else 0.0,
            "single_label_share_pct": float(label_count_df.loc[label_count_df["label_count"] == 1, "share_pct"].iloc[0]) if (label_count_df["label_count"] == 1).any() else 0.0,
            "multi_label_share_pct": float(label_count_df.loc[label_count_df["label_count"] > 1, "share_pct"].sum()) if not label_count_df.empty else 0.0,
            "rare_primary_categories_le_10": int((primary_category_df["count"] <= 10).sum()) if not primary_category_df.empty else 0,
            "rare_primary_categories_le_50": int((primary_category_df["count"] <= 50).sum()) if not primary_category_df.empty else 0,
            "recent_2018_share_pct": recent_2018_share,
            "recent_2020_share_pct": recent_2020_share,
            "length_percentiles": {
                "title_char_len": percentile_summary(title_char_lengths),
                "title_word_len": percentile_summary(title_word_lengths),
                "abstract_char_len": percentile_summary(abstract_char_lengths),
                "abstract_word_len": percentile_summary(abstract_word_lengths),
                "author_count": percentile_summary(author_counts_array),
                "label_count": percentile_summary(label_counts_array),
                "version_count": percentile_summary(version_counts_array),
            },
            "duplicate_summary": duplicate_summary,
        }

        with (artifact_dir / "summary.json").open("w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        build_split_artifacts(primary_label_codes, code_to_primary_label, artifact_dir)
        build_token_artifacts(data_path, artifact_dir)


    def ensure_eda_artifacts(data_path, artifact_dir, refresh=False):
        required = [
            artifact_dir / "summary.json",
            artifact_dir / "column_overview.csv",
            artifact_dir / "all_category_counts.csv",
            artifact_dir / "primary_category_counts.csv",
            artifact_dir / "primary_domain_counts.csv",
            artifact_dir / "label_count_distribution.csv",
            artifact_dir / "submission_year_counts.csv",
            artifact_dir / "length_arrays.npz",
            artifact_dir / "text_artifact_patterns.csv",
            artifact_dir / "domain_abstract_length_summary.csv",
            artifact_dir / "token_metrics.csv",
            artifact_dir / "token_top_terms.csv",
            artifact_dir / "split_threshold_summary.csv",
            artifact_dir / "split_strategy_summary.csv",
            artifact_dir / "split_drift_tables.csv",
            artifact_dir / "example_rows.json",
        ]

        summary_path = artifact_dir / "summary.json"
        needs_build = refresh or not all(path.exists() for path in required)

        if not needs_build and summary_path.exists():
            current_summary = load_json(summary_path)
            if current_summary.get("artifact_version") != ARTIFACT_VERSION:
                needs_build = True

        if needs_build:
            print("Building full-dataset EDA artifacts. This can take a while on the first run.")
            build_eda_artifacts(data_path, artifact_dir)
        else:
            print(f"Using cached full-dataset EDA artifacts from {artifact_dir.resolve()}")
    """
)

code(
    """
    ensure_eda_artifacts(DATA_PATH, ARTIFACT_DIR, refresh=REFRESH_ARTIFACTS)

    summary = load_json(ARTIFACT_DIR / "summary.json")
    preview_df = pd.DataFrame(load_json(ARTIFACT_DIR / "preview_rows.json"))
    example_rows = load_json(ARTIFACT_DIR / "example_rows.json")
    duplicate_examples = load_json(ARTIFACT_DIR / "duplicate_examples.json")

    column_overview = pd.read_csv(ARTIFACT_DIR / "column_overview.csv")
    all_category_counts = pd.read_csv(ARTIFACT_DIR / "all_category_counts.csv")
    primary_category_counts = pd.read_csv(ARTIFACT_DIR / "primary_category_counts.csv")
    primary_domain_counts = pd.read_csv(ARTIFACT_DIR / "primary_domain_counts.csv")
    all_domain_counts = pd.read_csv(ARTIFACT_DIR / "all_domain_counts.csv")
    label_count_distribution = pd.read_csv(ARTIFACT_DIR / "label_count_distribution.csv")
    category_combinations = pd.read_csv(ARTIFACT_DIR / "category_combination_counts_top200.csv")
    submission_year_counts = pd.read_csv(ARTIFACT_DIR / "submission_year_counts.csv")
    submission_month_counts = pd.read_csv(ARTIFACT_DIR / "submission_month_counts.csv")
    primary_category_year_counts = pd.read_csv(ARTIFACT_DIR / "primary_category_year_counts.csv")
    primary_domain_year_counts = pd.read_csv(ARTIFACT_DIR / "primary_domain_year_counts.csv")
    author_count_distribution = pd.read_csv(ARTIFACT_DIR / "author_count_distribution.csv")
    version_count_distribution = pd.read_csv(ARTIFACT_DIR / "version_count_distribution.csv")
    update_lag_distribution = pd.read_csv(ARTIFACT_DIR / "update_lag_year_distribution.csv")
    malformed_category_tokens = pd.read_csv(ARTIFACT_DIR / "malformed_category_tokens.csv")
    text_artifact_patterns = pd.read_csv(ARTIFACT_DIR / "text_artifact_patterns.csv")
    domain_abstract_length_summary = pd.read_csv(ARTIFACT_DIR / "domain_abstract_length_summary.csv")
    token_metrics = pd.read_csv(ARTIFACT_DIR / "token_metrics.csv")
    token_top_terms = pd.read_csv(ARTIFACT_DIR / "token_top_terms.csv")
    split_threshold_summary = pd.read_csv(ARTIFACT_DIR / "split_threshold_summary.csv")
    split_strategy_summary = pd.read_csv(ARTIFACT_DIR / "split_strategy_summary.csv")
    split_drift_tables = pd.read_csv(ARTIFACT_DIR / "split_drift_tables.csv")

    length_arrays = np.load(ARTIFACT_DIR / "length_arrays.npz")
    title_char_len = length_arrays["title_char_len"]
    title_word_len = length_arrays["title_word_len"]
    abstract_char_len = length_arrays["abstract_char_len"]
    abstract_word_len = length_arrays["abstract_word_len"]
    author_count_arr = length_arrays["author_count"]
    label_count_arr = length_arrays["label_count"]
    version_count_arr = length_arrays["version_count"]

    print(f"Loaded full-dataset summaries for {summary['record_count']:,} papers.")
    print(f"Artifact directory: {ARTIFACT_DIR.resolve()}")
    """
)

md("## 3. Data Loading")

md(
    """
    The raw corpus is stored as a JSON Lines file, with one paper per row. The notebook does not try to materialize the full file as one in-memory dataframe. Instead, it uses a streaming artifact build so the EDA remains reproducible while still covering the **entire dataset**.

    The table below makes the scope explicit.
    """
)

code(
    """
    analysis_scope = pd.DataFrame(
        [
            ("Core metadata counts and completeness", "Full dataset", "Exact"),
            ("Category and domain distributions", "Full dataset", "Exact"),
            ("Text-length distributions", "Full dataset", "Exact"),
            ("Text artifact prevalence", "Full dataset", "Exact"),
            ("Token and vocabulary analysis", "Full dataset", "Exact, cached after first run"),
            ("Split strategy comparison", "Full dataset", "Exact"),
            ("Displayed paper examples", "Selected representative rows", "For readability only"),
            ("Recommended Colab runtime", "High-RAM preferred", "GPU optional for EDA"),
        ],
        columns=["Analysis block", "Coverage", "Computation mode"],
    )

    display(analysis_scope)
    """
)

md("## 4. Dataset Overview")

md(
    """
    This section establishes the scale and schema of the corpus, identifies the columns most relevant to topic discovery, and shows a small set of representative rows for orientation.
    """
)

code(
    """
    overview_metrics = pd.DataFrame(
        [
            ("Total records", f"{summary['record_count']:,}"),
            ("Unique primary categories", f"{summary['unique_primary_categories']:,}"),
            ("Unique categories across all labels", f"{summary['unique_all_categories']:,}"),
            ("Unique primary domains", f"{summary['unique_primary_domains']:,}"),
            ("Submission-year range", f"{submission_year_counts['submission_year'].min()} to {submission_year_counts['submission_year'].max()}"),
            ("Dataset file size", f"{DATA_PATH.stat().st_size / 1024**3:,.2f} GB"),
            ("Artifact cache", str(ARTIFACT_DIR)),
            ("EDA scope", "Whole dataset"),
        ],
        columns=["Metric", "Value"],
    )

    column_roles = pd.DataFrame(
        [
            ("id", "Stable paper identifier used for deduplication, joins, and downstream indexing", "High"),
            ("title", "Compact scientific description; useful for retrieval and title+abstract embeddings", "High"),
            ("abstract", "Main text field for embeddings and clustering", "High"),
            ("categories", "arXiv category assignments; first category acts as the primary label", "High"),
            ("versions", "Version history; first version provides an effective submission timestamp", "High"),
            ("update_date", "Last metadata update; useful for revision lag and temporal skew", "High"),
            ("authors", "Raw author string for descriptive metadata only", "Medium"),
            ("authors_parsed", "Structured author list; supports author-count analysis", "Medium"),
            ("comments", "Optional notes that may contain noisy free text", "Low"),
            ("journal-ref", "Optional publication venue metadata; sparse and secondary", "Low"),
            ("doi", "Optional publication identifier; useful for enrichment, not clustering", "Low"),
            ("report-no", "Optional report number; sparse", "Low"),
            ("license", "Usage metadata; informative but not central to this benchmark", "Low"),
            ("submitter", "Submission contact metadata; not relevant for topic discovery", "Low"),
        ],
        columns=["Column", "Interpretation", "Project relevance"],
    )

    display(overview_metrics)
    display(column_overview.sort_values("column").reset_index(drop=True))
    display(column_roles)
    display(preview_df)
    """
)

code(
    """
    display(
        Markdown(
            f'''
    **Interpretation.** The corpus contains **{summary['record_count']:,} papers** and a stable 14-column schema. The fields that matter most for our benchmark are the textual and label-bearing fields: `title`, `abstract`, `categories`, `versions`, and `update_date`. This is a good fit for topic discovery because the dataset combines rich scientific text with externally assigned category labels that can be used as a reference signal for evaluation.
    '''
        )
    )
    """
)

md("## 5. Data Quality Assessment")

md(
    """
    We focus on quality issues that are genuinely relevant to embeddings, clustering, and evaluation: missing core fields, duplicates, malformed metadata, and pathological text lengths.
    """
)

code(
    """
    key_columns = ["id", "title", "abstract", "categories", "versions", "update_date", "authors", "authors_parsed", "doi", "journal-ref"]
    missing_plot = column_overview[column_overview["column"].isin(key_columns)].sort_values("missing_pct", ascending=False)

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=missing_plot, x="missing_pct", y="column", hue="column", palette="crest", legend=False, ax=ax)
    ax.set_title("Missingness in Key Dataset Fields")
    ax.set_xlabel("Missing values (%)")
    ax.set_ylabel("")
    plt.tight_layout()
    plt.show()

    quality_summary = pd.DataFrame(
        [
            ("Duplicate paper IDs", summary["duplicate_paper_ids"], "Exact identifier collisions should be removed before splitting."),
            ("Exact duplicate title records", summary["duplicate_summary"]["exact_title"]["duplicate_records"], "Repeated titles can artificially tighten clusters."),
            ("Near-duplicate normalized titles", summary["duplicate_summary"]["normalized_title"]["duplicate_records"], "Case and punctuation variants still indicate repeated content."),
            ("Duplicate title+abstract records", summary["duplicate_summary"]["title_abstract_content"]["duplicate_records"], "Repeated content can inflate retrieval and clustering metrics."),
            ("Invalid submission dates", summary["invalid_submission_dates"], "Temporal analyses depend on reliable submission timestamps."),
            ("Invalid update dates", summary["invalid_update_dates"], "Revision-lag analysis depends on parsable update dates."),
            ("Malformed category tokens", int(malformed_category_tokens["count"].sum()) if not malformed_category_tokens.empty else 0, "Formatting anomalies complicate label parsing."),
            ("Very short abstracts (<= 10 words)", int((abstract_word_len <= 10).sum()), "Often placeholders, withdrawn records, or unusable modeling inputs."),
            ("Very long abstracts (>= 300 words)", int((abstract_word_len >= 300).sum()), "Increase token-baseline sparsity and embedding cost."),
            ("Very short titles (<= 3 words)", int((title_word_len <= 3).sum()), "Too ambiguous to stand alone as topic signals."),
        ],
        columns=["Check", "Count", "Why it matters"],
    )

    duplicate_title_example_rows = []
    for item in summary["duplicate_summary"]["exact_title"]["top_hashes"][:5]:
        example = duplicate_examples["exact_title"].get(str(item["hash"]), {})
        duplicate_title_example_rows.append(
            {
                "duplicate_count": item["count"],
                "paper_id": example.get("id", ""),
                "title": example.get("title", ""),
                "categories": example.get("categories", ""),
            }
        )

    display(quality_summary)
    display(pd.DataFrame(duplicate_title_example_rows))
    """
)

code(
    """
    display(
        Markdown(
            f'''
    **Interpretation.** The core modeling fields are structurally strong: missingness is concentrated in optional enrichment fields such as `journal-ref` and `doi`, not in `title`, `abstract`, or `categories`. The main quality concerns are therefore **duplicate or near-duplicate records** and **degenerate short abstracts**, both of which can bias clustering and make retrieval quality look better than it really is.

    The duplicate checks found **{summary['duplicate_paper_ids']} duplicate paper IDs**, **{summary['duplicate_summary']['normalized_title']['duplicate_records']:,} normalized-title duplicates**, and **{summary['duplicate_summary']['title_abstract_content']['duplicate_records']:,} duplicate title-plus-abstract records**. These should be filtered before final benchmarking.
    '''
        )
    )
    """
)

md("## 6. Category and Domain Analysis")

md(
    """
    Because arXiv categories will be used as reference labels for evaluating clustering quality, the label space deserves careful treatment. We examine the head, the tail, the multi-label structure, and the broader domain mix.
    """
)

code(
    """
    top20_primary = primary_category_counts.head(20).sort_values("count", ascending=True)
    top12_domains = primary_domain_counts.head(12).sort_values("count", ascending=True)

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    sns.barplot(data=top20_primary, x="count", y="primary_category", hue="primary_category", palette="mako", legend=False, ax=axes[0])
    axes[0].set_title("Top 20 Primary Categories")
    axes[0].set_xlabel("Paper count")
    axes[0].set_ylabel("")

    sns.barplot(data=top12_domains, x="count", y="primary_domain", hue="primary_domain", palette="crest", legend=False, ax=axes[1])
    axes[1].set_title("Primary Domain Distribution")
    axes[1].set_xlabel("Paper count")
    axes[1].set_ylabel("")

    plt.tight_layout()
    plt.show()

    display(primary_category_counts.head(20))
    """
)

code(
    """
    rank_df = primary_category_counts.copy()
    rank_df["rank"] = np.arange(1, len(rank_df) + 1)
    rank_df["cumulative_share_pct"] = rank_df["count"].cumsum() / rank_df["count"].sum() * 100

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    axes[0].plot(rank_df["rank"], rank_df["count"], color="#0f766e", linewidth=2.2)
    axes[0].set_yscale("log")
    axes[0].set_title("Primary Category Frequency by Rank")
    axes[0].set_xlabel("Category rank")
    axes[0].set_ylabel("Paper count (log scale)")

    axes[1].plot(rank_df["rank"], rank_df["cumulative_share_pct"], color="#b45309", linewidth=2.2)
    for cutoff in [10, 20, 50, 100]:
        share = rank_df.loc[rank_df["rank"] <= cutoff, "count"].sum() / rank_df["count"].sum() * 100
        axes[1].axvline(cutoff, color="gray", linestyle="--", alpha=0.5)
        axes[1].text(cutoff, min(share + 2, 98), f"Top {cutoff}: {share:.1f}%", fontsize=9, ha="left")
    axes[1].set_ylim(0, 100)
    axes[1].set_title("Cumulative Coverage of Primary Categories")
    axes[1].set_xlabel("Top-N categories")
    axes[1].set_ylabel("Cumulative share (%)")

    plt.tight_layout()
    plt.show()
    """
)

code(
    """
    least_frequent = primary_category_counts.tail(10).sort_values("count", ascending=True)
    only_secondary_categories = sorted(set(all_category_counts["category"]) - set(primary_category_counts["primary_category"]))

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=label_count_distribution, x="label_count", y="share_pct", hue="label_count", palette="rocket", legend=False, ax=ax)
    ax.set_title("Number of Category Labels Assigned per Paper")
    ax.set_xlabel("Labels per paper")
    ax.set_ylabel("Share of papers (%)")
    plt.tight_layout()
    plt.show()

    display(label_count_distribution)
    display(category_combinations.head(15))
    display(least_frequent)
    print("Categories appearing only as secondary labels:", only_secondary_categories)
    """
)

code(
    """
    top10_share = primary_category_counts.head(10)["count"].sum() / primary_category_counts["count"].sum() * 100
    top20_share = primary_category_counts.head(20)["count"].sum() / primary_category_counts["count"].sum() * 100
    top50_share = primary_category_counts.head(50)["count"].sum() / primary_category_counts["count"].sum() * 100

    display(
        Markdown(
            f'''
    **Interpretation.** The label space is broad and distinctly long-tailed. There are **{summary['unique_primary_categories']} primary categories** and **{summary['unique_all_categories']} categories across all labels**. The single largest primary category covers only **{summary['top_primary_category_share_pct']:.2f}%** of the corpus, but concentration still accumulates quickly: the **top 10 primary categories cover {top10_share:.1f}%** of all papers and the **top 50 cover {top50_share:.1f}%**.

    Multi-label structure is also substantial. Roughly **{summary['multi_label_share_pct']:.1f}%** of papers carry more than one category label. That means primary-category evaluation is useful, but it does not capture all cross-disciplinary structure. For the benchmark, the most defensible default is to stratify and evaluate on the **primary category** while retaining the full category string for error analysis and interpretation.
    '''
        )
    )
    """
)

md("## 7. Text Field Analysis")

md(
    """
    Topic discovery quality depends strongly on the text fields we embed. We therefore analyze title and abstract lengths, measure markup and artifact prevalence on the full dataset, and review representative examples drawn from the corpus.
    """
)

code(
    """
    length_percentile_table = (
        pd.DataFrame(summary["length_percentiles"])
        .rename_axis("percentile")
        .reset_index()
    )

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    axes[0, 0].hist(title_char_len, bins=50, color="#0f766e")
    axes[0, 0].set_title("Title Length in Characters")
    axes[0, 0].set_xlabel("Characters")

    axes[0, 1].hist(title_word_len, bins=40, color="#2563eb")
    axes[0, 1].set_title("Title Length in Words")
    axes[0, 1].set_xlabel("Words")

    axes[1, 0].hist(abstract_char_len, bins=60, color="#b45309")
    axes[1, 0].set_title("Abstract Length in Characters")
    axes[1, 0].set_xlabel("Characters")

    axes[1, 1].hist(abstract_word_len, bins=60, color="#7c3aed")
    axes[1, 1].set_title("Abstract Length in Words")
    axes[1, 1].set_xlabel("Words")

    plt.tight_layout()
    plt.show()

    display(length_percentile_table)
    """
)

code(
    """
    top_domains_for_length = primary_domain_counts.head(8)[["primary_domain"]].merge(
        domain_abstract_length_summary,
        on="primary_domain",
        how="left",
    )
    top_domains_for_length = top_domains_for_length.sort_values("median_abstract_words", ascending=True)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.errorbar(
        top_domains_for_length["median_abstract_words"],
        top_domains_for_length["primary_domain"],
        xerr=[
            top_domains_for_length["median_abstract_words"] - top_domains_for_length["p25_abstract_words"],
            top_domains_for_length["p75_abstract_words"] - top_domains_for_length["median_abstract_words"],
        ],
        fmt="o",
        color="#0f766e",
        ecolor="#94a3b8",
        capsize=4,
    )
    ax.set_title("Abstract Word-Length Spread by Major Primary Domain")
    ax.set_xlabel("Median abstract length with interquartile range")
    ax.set_ylabel("")
    plt.tight_layout()
    plt.show()
    """
)

code(
    """
    fig, ax = plt.subplots(figsize=(11, 5))
    artifact_plot = text_artifact_patterns.sort_values("share_pct", ascending=True)
    sns.barplot(
        data=artifact_plot,
        x="share_pct",
        y=artifact_plot["field"] + " | " + artifact_plot["pattern"],
        hue=artifact_plot["field"],
        dodge=False,
        palette="flare",
        legend=False,
        ax=ax,
    )
    ax.set_title("Text Artifact Prevalence on the Full Dataset")
    ax.set_xlabel("Share of papers (%)")
    ax.set_ylabel("")
    plt.tight_layout()
    plt.show()

    display(text_artifact_patterns.sort_values(["field", "share_pct"], ascending=[True, False]))
    """
)

code(
    """
    def example_table(rows):
        frame = pd.DataFrame(rows)
        if frame.empty:
            return frame
        preferred = [col for col in ["id", "primary_category", "title", "abstract_word_len", "issue_tags", "abstract_preview"] if col in frame.columns]
        return frame[preferred]


    display(example_table(example_rows["normal_examples"]))
    display(example_table(example_rows["problematic_examples"]))
    display(example_table(example_rows["short_abstract_examples"]))
    display(example_table(example_rows["long_abstract_examples"]))
    """
)

code(
    """
    display(
        Markdown(
            f'''
    **Interpretation.** The corpus is well aligned with modern scientific encoders from a sequence-length perspective. The median abstract length is **{summary['length_percentiles']['abstract_word_len']['50']:.0f} words**, the 95th percentile is **{summary['length_percentiles']['abstract_word_len']['95']:.0f} words**, and even the upper tail remains manageable for title-plus-abstract inputs.

    The larger preprocessing issue is not raw length but **artifact-bearing scientific text**. LaTeX-style notation, newline artifacts, and a smaller number of withdrawn or noisy abstracts are visibly present. That supports a **light-touch preprocessing strategy**: preserve scientific wording and notation when using transformer encoders, but explicitly handle withdrawn placeholders, duplicated content, and obvious URL or email noise.
    '''
        )
    )
    """
)

md("## 8. Preprocessing-Oriented Analysis")

md(
    """
    This section is deliberately tied to downstream modeling. The goal is not maximal cleaning; it is to understand what different preprocessing choices do to the corpus and which choices remain defensible for transformer embeddings versus token-based baselines.
    """
)

code(
    """
    display(token_metrics)
    """
)

code(
    """
    vocab_plot = token_metrics[["configuration", "vocab_size", "vocab_reduction_vs_raw_pct"]].copy()

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    sns.barplot(data=vocab_plot, x="vocab_size", y="configuration", hue="configuration", palette="mako", legend=False, ax=axes[0])
    axes[0].set_title("Vocabulary Size by Preprocessing Configuration")
    axes[0].set_xlabel("Unique tokens")
    axes[0].set_ylabel("")

    sns.barplot(data=token_metrics, x="top100_token_share_pct", y="configuration", hue="configuration", palette="rocket", legend=False, ax=axes[1])
    axes[1].set_title("Concentration in Top 100 Tokens")
    axes[1].set_xlabel("Share of all token occurrences (%)")
    axes[1].set_ylabel("")

    plt.tight_layout()
    plt.show()
    """
)

code(
    """
    raw_top = token_top_terms[token_top_terms["configuration"] == "Raw lowercase tokenization"].head(15)
    clean_top = token_top_terms[
        token_top_terms["configuration"] == "Lowercase + punctuation removal + stopword removal"
    ].head(15)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    sns.barplot(data=raw_top.sort_values("count"), x="count", y="token", hue="token", palette="crest", legend=False, ax=axes[0])
    axes[0].set_title("Most Common Raw Tokens")
    axes[0].set_xlabel("Count")
    axes[0].set_ylabel("")

    sns.barplot(data=clean_top.sort_values("count"), x="count", y="token", hue="token", palette="flare", legend=False, ax=axes[1])
    axes[1].set_title("Most Common Cleaned Tokens")
    axes[1].set_xlabel("Count")
    axes[1].set_ylabel("")

    plt.tight_layout()
    plt.show()
    """
)

code(
    """
    cleaned_rank_curve = token_top_terms[
        token_top_terms["configuration"] == "Lowercase + punctuation removal + stopword removal"
    ].head(1000)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(cleaned_rank_curve["rank"], cleaned_rank_curve["count"], color="#7c3aed", linewidth=2)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title("Token Frequency Rank Curve After Light Cleaning")
    ax.set_xlabel("Token rank (log scale)")
    ax.set_ylabel("Frequency (log scale)")
    plt.tight_layout()
    plt.show()
    """
)

code(
    """
    raw_vocab = int(token_metrics.loc[token_metrics["configuration"] == "Raw lowercase tokenization", "vocab_size"].iloc[0])
    clean_vocab = int(token_metrics.loc[
        token_metrics["configuration"] == "Lowercase + punctuation removal + stopword removal",
        "vocab_size",
    ].iloc[0])

    display(
        Markdown(
            f'''
    **Interpretation.** On the **full dataset**, light preprocessing substantially contracts the vocabulary: moving from raw lowercase tokenization to punctuation removal plus stopword removal reduces the observed vocabulary from **{raw_vocab:,}** to **{clean_vocab:,}** unique tokens. At the same time, the token-frequency curve remains strongly long-tailed, which means rare-token management still matters for classical baselines.

    **Modeling implication.**
    - For **transformer-based scientific embeddings** such as SciBERT- or SPECTER-style encoders, prefer minimal normalization: whitespace cleanup, optional URL/email stripping, and filtering of withdrawn or ultra-short placeholder abstracts.
    - For **TF-IDF or bag-of-words baselines**, `lowercase + punctuation removal + stopword removal` is a sensible default, ideally with `min_df` filtering to control the rare-token tail.
    - We intentionally do **not** recommend stemming or aggressive lemmatization as the default because scientific terminology is often meaning-bearing at the surface-form level.
    '''
        )
    )
    """
)

md("## 9. Temporal and Metadata Analysis")

md(
    """
    Temporal skew matters because the corpus has grown rapidly and unevenly. If recent years are dominated by a few fast-growing domains, random sampling can distort qualitative conclusions about topic diversity and retrieval behavior.
    """
)

code(
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    sns.lineplot(data=submission_year_counts, x="submission_year", y="count", marker="o", linewidth=2.2, ax=axes[0])
    axes[0].set_title("Paper Count by Submission Year")
    axes[0].set_xlabel("Submission year")
    axes[0].set_ylabel("Paper count")

    monthly_recent = submission_month_counts.copy()
    monthly_recent["year"] = monthly_recent["submission_month"].str[:4].astype(int)
    monthly_recent = monthly_recent[monthly_recent["year"] >= monthly_recent["year"].max() - 4]
    sns.lineplot(data=monthly_recent, x="submission_month", y="count", linewidth=2, ax=axes[1], color="#b45309")
    axes[1].set_title("Recent Monthly Submission Volume")
    axes[1].set_xlabel("Submission month")
    axes[1].set_ylabel("Paper count")
    axes[1].tick_params(axis="x", rotation=60)

    plt.tight_layout()
    plt.show()
    """
)

code(
    """
    top_domains = primary_domain_counts.head(6)["primary_domain"].tolist()
    domain_year = primary_domain_year_counts[primary_domain_year_counts["primary_domain"].isin(top_domains)].copy()
    year_totals = primary_domain_year_counts.groupby("submission_year")["count"].sum().rename("year_total")
    domain_year = domain_year.merge(year_totals, on="submission_year", how="left")
    domain_year["share_pct"] = domain_year["count"] / domain_year["year_total"] * 100

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.lineplot(
        data=domain_year,
        x="submission_year",
        y="share_pct",
        hue="primary_domain",
        linewidth=2.2,
        ax=ax,
    )
    ax.set_title("Share of Major Primary Domains Over Time")
    ax.set_xlabel("Submission year")
    ax.set_ylabel("Share of yearly submissions (%)")
    ax.legend(title="Domain", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.show()
    """
)

code(
    """
    author_plot = author_count_distribution.copy()
    author_plot["author_bucket"] = author_plot["author_count"].where(author_plot["author_count"] <= 12, 13)
    author_bucket = (
        author_plot.groupby("author_bucket", as_index=False)["count"].sum().assign(
            label=lambda df: df["author_bucket"].astype(str).replace({"13": "13+"})
        )
    )

    version_plot = version_count_distribution.copy()
    version_plot["version_bucket"] = version_plot["version_count"].where(version_plot["version_count"] <= 6, 7)
    version_bucket = (
        version_plot.groupby("version_bucket", as_index=False)["count"].sum().assign(
            label=lambda df: df["version_bucket"].astype(str).replace({"7": "7+"})
        )
    )

    lag_plot = update_lag_distribution[update_lag_distribution["update_lag_years"] <= 10].copy()

    fig, axes = plt.subplots(1, 3, figsize=(20, 5))
    sns.barplot(data=author_bucket, x="label", y="count", hue="label", palette="crest", legend=False, ax=axes[0])
    axes[0].set_title("Authors per Paper")
    axes[0].set_xlabel("Authors")
    axes[0].set_ylabel("Paper count")

    sns.barplot(data=version_bucket, x="label", y="count", hue="label", palette="mako", legend=False, ax=axes[1])
    axes[1].set_title("Version Count per Paper")
    axes[1].set_xlabel("Versions")
    axes[1].set_ylabel("Paper count")

    sns.barplot(data=lag_plot, x="update_lag_years", y="count", hue="update_lag_years", palette="flare", legend=False, ax=axes[2])
    axes[2].set_title("Update Lag Distribution")
    axes[2].set_xlabel("Lag in years")
    axes[2].set_ylabel("Paper count")

    plt.tight_layout()
    plt.show()
    """
)

code(
    """
    cs_rows = domain_year[domain_year["primary_domain"] == "cs"]
    cs_start_year = int(cs_rows["submission_year"].min())
    cs_end_year = int(cs_rows["submission_year"].max())
    cs_start_share = float(cs_rows.loc[cs_rows["submission_year"] == cs_start_year, "share_pct"].iloc[0])
    cs_end_share = float(cs_rows.loc[cs_rows["submission_year"] == cs_end_year, "share_pct"].iloc[0])

    display(
        Markdown(
            f'''
    **Interpretation.** The corpus is strongly weighted toward recent activity: **{summary['recent_2018_share_pct']:.1f}%** of papers were submitted in **2018 or later**, and **{summary['recent_2020_share_pct']:.1f}%** arrived in **2020 or later**. Domain composition has also shifted. In the primary-domain view, **computer science moved from {cs_start_share:.1f}% of yearly submissions in {cs_start_year} to {cs_end_share:.1f}% in {cs_end_year}**.

    This matters for benchmarking. A naive random sample from the full corpus will naturally emphasize recent and CS-heavy material. If we claim broad scientific topic discovery, the report needs to acknowledge that temporal and domain skew explicitly.
    '''
        )
    )
    """
)

md("## 10. Data Splitting Strategy Analysis")

md(
    """
    The split strategy should preserve category structure without overengineering multi-label stratification. Because every split decision affects downstream comparison of embeddings and clustering algorithms, we evaluate random versus stratified splitting on the **full dataset**.
    """
)

code(
    """
    display(split_threshold_summary)
    display(split_strategy_summary)
    """
)

code(
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    sns.barplot(
        data=split_strategy_summary,
        x="split",
        y="mean_abs_diff_pct_points",
        hue="strategy",
        palette="mako",
        ax=axes[0],
    )
    axes[0].set_title("Average Category-Share Drift by Split Strategy")
    axes[0].set_xlabel("")
    axes[0].set_ylabel("Mean absolute drift (percentage points)")

    sns.barplot(
        data=split_strategy_summary,
        x="split",
        y="max_abs_diff_pct_points",
        hue="strategy",
        palette="rocket",
        ax=axes[1],
    )
    axes[1].set_title("Worst-Case Category-Share Drift by Split Strategy")
    axes[1].set_xlabel("")
    axes[1].set_ylabel("Maximum absolute drift (percentage points)")

    plt.tight_layout()
    plt.show()
    """
)

code(
    """
    worst_random_test = (
        split_drift_tables[
            (split_drift_tables["strategy"] == "Random") & (split_drift_tables["split"] == "Test")
        ]
        .sort_values("abs_diff_pct_points", ascending=False)
        .head(10)
        .reset_index(drop=True)
    )
    worst_stratified_test = (
        split_drift_tables[
            (split_drift_tables["strategy"] == "Stratified by primary category")
            & (split_drift_tables["split"] == "Test")
        ]
        .sort_values("abs_diff_pct_points", ascending=False)
        .head(10)
        .reset_index(drop=True)
    )

    display(worst_random_test)
    display(worst_stratified_test)
    """
)

code(
    """
    min_primary_count = int(primary_category_counts["count"].min())
    rare_primary_le_50 = int((primary_category_counts["count"] <= 50).sum())

    display(
        Markdown(
            f'''
    **Interpretation.** On the full dataset, the smallest primary category still contains **{min_primary_count} papers**, so a **70/10/20 split stratified by primary category is feasible across all 172 primary labels**. The full-dataset comparison also shows that stratification materially reduces category-share drift relative to naive random splitting.

    **Recommendation.**
    - Use **70/10/20 stratified sampling on `primary_category`** as the default split strategy.
    - Retain the full multi-label category string for analysis, but do not require exact multi-label stratification in the first benchmark version.
    - Mention that **{rare_primary_le_50} primary categories have 50 papers or fewer**, so rare-category validation and test statistics remain noisy even under stratification.
    '''
        )
    )
    """
)

md("## 11. Outliers, Edge Cases, and Risks")

md(
    """
    The point of this section is to convert EDA into actionable modeling guidance. Each risk below is tied directly to how it can affect clustering, embedding generation, evaluation, or retrieval behavior.
    """
)

code(
    """
    top50_share = primary_category_counts.head(50)["count"].sum() / primary_category_counts["count"].sum() * 100

    risk_register = pd.DataFrame(
        [
            (
                "Long-tail label imbalance",
                f"Top 50 primary categories account for {top50_share:.1f}% of papers; {summary['rare_primary_categories_le_50']} primary categories have 50 papers or fewer.",
                "Dominant topics can steer K-means centroids and make macro-level label evaluation unstable in the tail.",
                "Report both aggregate results and caveats for rare categories; consider filtered sensitivity analyses.",
            ),
            (
                "Multi-label ambiguity",
                f"{summary['multi_label_share_pct']:.1f}% of papers carry multiple category labels.",
                "A single reference label can penalize clusters that capture interdisciplinary structure.",
                "Use primary labels for the main benchmark, but inspect multi-label cases separately in analysis.",
            ),
            (
                "Duplicate and near-duplicate content",
                f"{summary['duplicate_paper_ids']} duplicate IDs and {summary['duplicate_summary']['title_abstract_content']['duplicate_records']:,} duplicate title+abstract records were detected.",
                "Duplicates can inflate retrieval precision and cluster purity by repeating content.",
                "Deduplicate on paper ID and normalized title+abstract before final modeling.",
            ),
            (
                "Short or placeholder abstracts",
                f"{int((abstract_word_len <= 10).sum()):,} abstracts contain 10 words or fewer.",
                "These rows are weak inputs for semantic embeddings and can form trivial or noisy clusters.",
                "Filter withdrawn or ultra-short records, or isolate them in a separate exclusion rule.",
            ),
            (
                "Scientific markup artifacts",
                "LaTeX-like text, newline artifacts, and smaller amounts of URL/email noise are present across the corpus.",
                "Aggressive cleanup may remove scientific meaning, while no cleanup can hurt token baselines.",
                "Use minimal normalization for transformers and stronger normalization only for TF-IDF-style baselines.",
            ),
            (
                "Temporal skew",
                f"{summary['recent_2018_share_pct']:.1f}% of papers were submitted in 2018 or later, with strong recent growth in computer-science-related domains.",
                "Recent, CS-heavy content can dominate both clusters and nearest-neighbor retrieval.",
                "Document the skew and include temporal framing in the report discussion.",
            ),
            (
                "Computational scale",
                f"The corpus contains {summary['record_count']:,} papers.",
                "Naive agglomerative clustering and fully dense pairwise operations are impractical at this scale.",
                "Use batched embedding generation, approximate retrieval indexes, and scalable clustering variants.",
            ),
        ],
        columns=["Risk", "Evidence", "Downstream impact", "Mitigation"],
    )

    display(risk_register)
    """
)

md("## 12. Key Findings and Recommendations")

code(
    """
    top_domain = primary_domain_counts.iloc[0]
    raw_vocab = int(token_metrics.loc[token_metrics["configuration"] == "Raw lowercase tokenization", "vocab_size"].iloc[0])
    clean_vocab = int(token_metrics.loc[
        token_metrics["configuration"] == "Lowercase + punctuation removal + stopword removal",
        "vocab_size",
    ].iloc[0])

    final_note = f'''
    ### Key Findings

    - The corpus is large and analytically rich: **{summary['record_count']:,} papers**, **{summary['unique_primary_categories']} primary categories**, and complete coverage for the fields most important to topic discovery.
    - Category imbalance is meaningful and clearly long-tailed. The largest primary category covers **{summary['top_primary_category_share_pct']:.2f}%** of the corpus, while the primary-domain view is led by **{top_domain['primary_domain']} ({top_domain['share_pct']:.1f}%)**.
    - Multi-label structure is common rather than exceptional: **{summary['multi_label_share_pct']:.1f}%** of papers have multiple category assignments.
    - Text lengths are well within the operating range of modern transformer encoders: the median abstract is **{summary['length_percentiles']['abstract_word_len']['50']:.0f} words** and the 95th percentile is **{summary['length_percentiles']['abstract_word_len']['95']:.0f} words**.
    - The main data-quality risks are **duplicate content**, **withdrawn or placeholder abstracts**, and **scientific markup artifacts**, not missing core text fields.
    - The dataset is temporally skewed toward recent years, and the later years are substantially more computer-science-heavy than earlier periods.

    ### Recommended Preprocessing

    - For **transformer embeddings**, keep title and abstract largely intact. Apply whitespace normalization, strip obvious URL or email noise when needed, and filter withdrawn or ultra-short placeholder records.
    - For **TF-IDF or bag-of-words baselines**, use lowercasing, punctuation removal, stopword removal, and a sensible `min_df` threshold. On the full dataset, this reduces vocabulary from **{raw_vocab:,}** to **{clean_vocab:,}** unique tokens.
    - Deduplicate on **paper ID** and preferably on a **normalized title-plus-abstract signature** before benchmarking.

    ### Recommended Experimental Design

    - Use a **70/10/20 split stratified by primary category**.
    - Use the **primary category** as the main external evaluation label, while retaining the full multi-label assignments for interpretation and failure analysis.
    - Report the full **172-category** benchmark, but consider an additional sensitivity analysis on a filtered subset or minimum-support threshold to show how rare labels affect results.
    - Treat scalability as part of the benchmark: record runtime or memory costs alongside clustering quality, and avoid naive algorithms that do not scale to millions of papers.

    ### Caveats to Mention Explicitly in the Report

    - arXiv categories are a useful reference signal, but they are not a perfect representation of latent scientific topics.
    - Multi-label and interdisciplinary papers make single-label evaluation inherently lossy.
    - Temporal and domain skew can shape both quantitative results and qualitative retrieval examples.
    - Rare labels and duplicate content can distort evaluation if not handled explicitly.
    '''

    display(Markdown(final_note))
    """
)

nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    },
    "language_info": {
        "name": "python",
        "version": "3.9",
    },
}

output_path = "Project.ipynb"
with open(output_path, "w", encoding="utf-8") as f:
    nbf.write(nb, f)

print(f"Wrote {output_path}")
