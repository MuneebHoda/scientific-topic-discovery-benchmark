"""Streaming cleaned dataset builder for the modeling pipeline."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import pandas as pd

from src.config import ProfileConfig, missing_raw_data_message
from src.io_utils import ensure_dir, load_json, stable_sha1, write_json
from src.preprocess import (
    abstract_word_len,
    derive_text_input,
    is_placeholder_abstract,
    normalize_whitespace,
    normalized_hash_key,
    parse_categories,
    primary_category,
    primary_domain,
    submission_year_from_versions,
    update_year_from_value,
)


@dataclass
class CleanedPaper:
    """Normalized row written into the modeling parquet dataset."""

    id: str
    title: str
    abstract: str
    text_input: str
    categories: str
    primary_category: str
    primary_domain: str
    submission_year: Optional[int]
    update_year: Optional[int]
    abstract_word_len: int
    is_multilabel: bool

    def as_dict(self) -> Dict:
        return {
            "id": self.id,
            "title": self.title,
            "abstract": self.abstract,
            "text_input": self.text_input,
            "categories": self.categories,
            "primary_category": self.primary_category,
            "primary_domain": self.primary_domain,
            "submission_year": self.submission_year,
            "update_year": self.update_year,
            "abstract_word_len": self.abstract_word_len,
            "is_multilabel": self.is_multilabel,
        }


class SQLiteDeduper:
    """Disk-backed duplicate tracker for ids and content hashes."""

    def __init__(self, db_path: Path) -> None:
        ensure_dir(db_path.parent)
        self.connection = sqlite3.connect(db_path)
        self.connection.execute("PRAGMA journal_mode=WAL;")
        self.connection.execute("PRAGMA synchronous=NORMAL;")
        self.connection.execute("PRAGMA temp_store=MEMORY;")
        self.connection.execute("CREATE TABLE IF NOT EXISTS seen_ids (key TEXT PRIMARY KEY)")
        self.connection.execute("CREATE TABLE IF NOT EXISTS seen_content (key TEXT PRIMARY KEY)")
        self.connection.commit()

    def add_unique(self, table_name: str, key: str) -> bool:
        """Insert a key if absent and report whether it was new."""

        cursor = self.connection.execute(f"INSERT OR IGNORE INTO {table_name} (key) VALUES (?)", (key,))
        return cursor.rowcount == 1

    def commit(self) -> None:
        self.connection.commit()

    def close(self) -> None:
        self.connection.commit()
        self.connection.close()


def _flush_parquet_rows(
    writer,
    rows: List[Dict],
    output_path: Path,
):
    """Flush accumulated rows into a parquet writer."""

    if not rows:
        return writer

    pyarrow = __import__("pyarrow")
    pq = __import__("pyarrow.parquet", fromlist=["parquet"])
    frame = pd.DataFrame(rows)
    table = pyarrow.Table.from_pandas(frame, preserve_index=False)
    if writer is None:
        ensure_dir(output_path.parent)
        writer = pq.ParquetWriter(output_path, table.schema, compression="zstd")
    writer.write_table(table)
    rows.clear()
    return writer


def build_clean_dataset(
    profile: ProfileConfig,
    batch_size: int = 5_000,
    progress_every: int = 100_000,
    force: bool = False,
) -> Dict:
    """Create the cleaned modeling dataset as a single parquet file."""

    output_path = profile.clean_dataset_path()
    manifest_path = profile.cleaning_manifest_path()
    if output_path.exists() and manifest_path.exists() and not force:
        manifest = load_json(manifest_path, default={}) or {}
        manifest["reused_cache"] = True
        return manifest

    if not profile.raw_data_path.exists():
        raise FileNotFoundError(missing_raw_data_message(profile.raw_data_path))

    if force:
        output_path.unlink(missing_ok=True)
        manifest_path.unlink(missing_ok=True)

    eda_summary = load_json(profile.eda_dir / "summary.json", default={}) or {}
    dedupe_db = profile.clean_dir() / "_dedupe.sqlite"
    dedupe_db.unlink(missing_ok=True)

    deduper = SQLiteDeduper(dedupe_db)
    writer = None
    rows: List[Dict] = []
    counts = {
        "records_seen": 0,
        "records_kept": 0,
        "dropped_missing_critical_fields": 0,
        "dropped_duplicate_ids": 0,
        "dropped_duplicate_content": 0,
        "dropped_short_abstracts": 0,
        "dropped_placeholder_abstracts": 0,
        "dropped_empty_primary_category": 0,
        "json_decode_errors": 0,
    }

    with profile.raw_data_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            counts["records_seen"] += 1
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                counts["json_decode_errors"] += 1
                continue

            paper_id = normalize_whitespace(record.get("id"))
            title = normalize_whitespace(record.get("title"))
            abstract = normalize_whitespace(record.get("abstract"))
            categories_text = normalize_whitespace(record.get("categories"))
            categories = parse_categories(categories_text)
            first_category = primary_category(categories)

            if not paper_id or not title or not abstract or not categories_text:
                counts["dropped_missing_critical_fields"] += 1
                continue
            if not first_category:
                counts["dropped_empty_primary_category"] += 1
                continue

            if not deduper.add_unique("seen_ids", paper_id):
                counts["dropped_duplicate_ids"] += 1
                continue

            content_hash = stable_sha1(normalized_hash_key(title, abstract))
            if not deduper.add_unique("seen_content", content_hash):
                counts["dropped_duplicate_content"] += 1
                continue

            abstract_len = abstract_word_len(abstract)
            if abstract_len <= 10:
                counts["dropped_short_abstracts"] += 1
                continue
            if is_placeholder_abstract(abstract):
                counts["dropped_placeholder_abstracts"] += 1
                continue

            cleaned = CleanedPaper(
                id=paper_id,
                title=title,
                abstract=abstract,
                text_input=derive_text_input(title, abstract),
                categories=categories_text,
                primary_category=first_category,
                primary_domain=primary_domain(first_category),
                submission_year=submission_year_from_versions(record.get("versions")),
                update_year=update_year_from_value(record.get("update_date")),
                abstract_word_len=abstract_len,
                is_multilabel=len(categories) > 1,
            )
            rows.append(cleaned.as_dict())
            counts["records_kept"] += 1

            if len(rows) >= batch_size:
                writer = _flush_parquet_rows(writer, rows, output_path)
                deduper.commit()

            if progress_every and line_number % progress_every == 0:
                print(
                    f"[dataset_builder] processed={line_number:,} kept={counts['records_kept']:,} "
                    f"duplicates={counts['dropped_duplicate_ids'] + counts['dropped_duplicate_content']:,}"
                )

    writer = _flush_parquet_rows(writer, rows, output_path)
    if writer is not None:
        writer.close()
    deduper.close()

    manifest = {
        "raw_data_path": str(profile.raw_data_path),
        "clean_dataset_path": str(output_path),
        "eda_summary_record_count": eda_summary.get("record_count"),
        "eda_unique_primary_categories": eda_summary.get("unique_primary_categories"),
        **counts,
    }
    write_json(manifest_path, manifest)
    return manifest
