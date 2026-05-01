from __future__ import annotations

import math
import pickle
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

import chromadb
import polars as pl
from chromadb.api.models.Collection import Collection
from django.conf import settings
from flashtext import KeywordProcessor

from .duckdb_gateway import DuckDBGateway


FORMAT_ALIASES = {
    "commander": "Commander",
    "edh": "Commander",
    "cmdr": "Commander",
    "cedh": "Commander",
    "brawl": "Brawl",
    "standard brawl": "Brawl",
    "historic brawl": "Brawl",
    "two-headed giant": "Two-Headed Giant",
    "two headed giant": "Two-Headed Giant",
    "2hg": "Two-Headed Giant",
    "oathbreaker": "Oathbreaker",
    "standard": "Standard",
    "modern": "Modern",
    "legacy": "Legacy",
    "vintage": "Vintage",
    "pauper": "Pauper",
    "pioneer": "Pioneer",
    "limited": "Limited",
    "draft": "Limited",
    "sealed": "Limited",
}

RULE_REFERENCE_RE = re.compile(
    r"\b(?:rule|rules)\s+(\d{1,3}(?:\.\d+[a-z]?)?)\b",
    re.IGNORECASE,
)

RULES_COLLECTION_NAME = "mtg_rules_Chroma"
SECTION_ROUTER_COLLECTION_NAME = "mtg_rule_section_router_Chroma"

EMPTY_RULES_SCHEMA = {
    "rule_id": pl.Utf8,
    "rule_text": pl.Utf8,
    "linked_rules": pl.List(pl.Utf8),
    "linked_terms": pl.List(pl.Utf8),
}
EMPTY_GLOSSARY_SCHEMA = {
    "term_id": pl.Utf8,
    "term_name": pl.Utf8,
    "definition": pl.Utf8,
    "linked_rules": pl.List(pl.Utf8),
    "linked_terms": pl.List(pl.Utf8),
}

_KEYWORD_PROCESSOR: KeywordProcessor | None = None
_GLOSSARY_INDEX: list[tuple[str, str]] | None = None
_CHROMA_CLIENT: chromadb.PersistentClient | None = None
_CHROMA_COLLECTIONS: dict[str, Collection] = {}
_EMBEDDING_FUNCTION = None


def get_duckdb_gateway() -> DuckDBGateway:
    return DuckDBGateway(
        duckdb_path=settings.GEMMA_JUDGE_DUCKDB_PATH,
        key_id=settings.CLD_USER,
        secret=settings.HMAC_K,
    )


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).replace("\xa0", " ")
    text = re.sub(r"[\r\n\t]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"\s+([,.;:?!])", r"\1", text)
    text = re.sub(r"([([{])\s+", r"\1", text)
    text = re.sub(r"\s+([)\]}])", r"\1", text)
    return text


def ordered_unique(values: list[Any] | tuple[Any, ...]) -> list[Any]:
    return list(dict.fromkeys(v for v in values if v is not None))


def word_count(text: Any) -> int:
    cleaned = normalize_text(text)
    return len(cleaned.split()) if cleaned else 0


def is_context_rule_id(rule_id: str) -> bool:
    return "." in str(rule_id)


def derive_colors_from_mana_cost(mana_cost: str | None) -> list[str]:
    symbols = str(mana_cost or "")
    colors = [color for color in "WUBRG" if f"{{{color}}}" in symbols]
    return colors


def normalize_format_name(format_name: str | None) -> str | None:
    if not format_name:
        return None
    cleaned = normalize_text(format_name).casefold()
    if not cleaned:
        return None
    return FORMAT_ALIASES.get(cleaned, normalize_text(format_name))


def infer_format_from_question(question: str) -> str | None:
    haystack = normalize_text(question).casefold()
    for alias, canonical in sorted(FORMAT_ALIASES.items(), key=lambda item: len(item[0]), reverse=True):
        if re.search(rf"\b{re.escape(alias)}\b", haystack):
            return canonical
    return None


def coerce_context_lines(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [normalize_text(value)] if normalize_text(value) else []
    if isinstance(value, dict):
        lines: list[str] = []
        for key, item in value.items():
            if item is None:
                continue
            if isinstance(item, (list, tuple, set)):
                item_text = ", ".join(normalize_text(x) for x in item if normalize_text(x))
            else:
                item_text = normalize_text(item)
            if item_text:
                lines.append(f"{normalize_text(key)}: {item_text}")
        return lines
    if isinstance(value, (list, tuple, set)):
        return [normalize_text(item) for item in value if normalize_text(item)]
    text = normalize_text(value)
    return [text] if text else []


def build_game_context_block(
    format_name: str | None = None,
    format_inferred: bool = False,
    game_context: Any = None,
    assumptions: list[str] | None = None,
    unknowns: list[str] | None = None,
) -> str:
    lines = []
    normalized_format = normalize_format_name(format_name)
    if normalized_format:
        source = "inferred" if format_inferred else "user-provided"
        lines.append(f"Format / ruleset: {normalized_format} ({source})")
    else:
        lines.append("Format / ruleset: unspecified")

    for label, value in (
        ("Additional game context", game_context),
        ("Assumptions", assumptions),
        ("Unknown / not provided", unknowns),
    ):
        entries = coerce_context_lines(value)
        if entries:
            lines.append(f"{label}:")
            lines.extend(f"- {entry}" for entry in entries)

    return "\n".join(lines).strip()


def build_rule_retrieval_query(
    question: str,
    card_texts: list[str],
    card_rulings: list[str],
    game_context_text: str | None = None,
) -> str:
    return "\n\n".join(
        part
        for part in [
            "Rules question:\n" + question.strip(),
            "Game context:\n" + normalize_text(game_context_text) if game_context_text else "",
            "Relevant card text:\n" + "\n".join(card_texts).strip(),
            "Relevant card rulings:\n" + "\n".join(card_rulings).strip(),
        ]
        if part.strip()
    ).strip()


def _flashtext_cache_path() -> Path:
    return Path(settings.GEMMA_JUDGE_RUNTIME_CACHE_DIR) / "flashText.pkl"


def extract_card_names(query: str, processor: KeywordProcessor | None = None) -> list[str]:
    active_processor = processor or load_card_keyword_processor()
    return ordered_unique(active_processor.extract_keywords(query))


def build_card_keyword_processor(save: bool = True) -> KeywordProcessor:
    sql = """
        SELECT DISTINCT trim(name) AS name
        FROM mtg_db.cardInfo
        WHERE name IS NOT NULL
          AND trim(name) <> ''
          AND lang = 'en'
        ORDER BY name
    """
    card_names = get_duckdb_gateway().query_polars(sql).get_column("name").to_list()
    processor = KeywordProcessor(case_sensitive=False)
    aliases = {name: [name, f"{name}'s", f"{name}s"] for name in card_names}
    processor.add_keywords_from_dict(aliases)
    if save:
        cache_path = _flashtext_cache_path()
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with cache_path.open("wb") as handle:
            pickle.dump(processor, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return processor


def load_card_keyword_processor() -> KeywordProcessor:
    global _KEYWORD_PROCESSOR
    if _KEYWORD_PROCESSOR is not None:
        return _KEYWORD_PROCESSOR

    cache_path = _flashtext_cache_path()
    if cache_path.exists():
        with cache_path.open("rb") as handle:
            _KEYWORD_PROCESSOR = pickle.load(handle)
            return _KEYWORD_PROCESSOR

    _KEYWORD_PROCESSOR = build_card_keyword_processor(save=True)
    return _KEYWORD_PROCESSOR


@lru_cache(maxsize=1)
def get_all_rule_ids() -> list[str]:
    return get_duckdb_gateway().query_polars("SELECT rule_id FROM mtg_db.mtg_rules").get_column("rule_id").to_list()


def build_glossary_phrase_index() -> list[tuple[str, str]]:
    global _GLOSSARY_INDEX
    if _GLOSSARY_INDEX is not None:
        return _GLOSSARY_INDEX
    rows = get_duckdb_gateway().query_polars(
        "SELECT DISTINCT term_id, term_name FROM mtg_db.mtg_glossary_terms"
    ).to_dicts()
    _GLOSSARY_INDEX = sorted(
        [
            (row["term_id"], normalize_text(row["term_name"]).casefold())
            for row in rows
            if len(normalize_text(row["term_name"]).casefold()) >= 5
        ],
        key=lambda item: len(item[1]),
        reverse=True,
    )
    return _GLOSSARY_INDEX


def extract_rule_ids_from_texts(texts: list[str]) -> list[str]:
    known_ids = set(get_all_rule_ids())
    found: list[str] = []
    for text in texts:
        for match in RULE_REFERENCE_RE.finditer(text or ""):
            rule_id = match.group(1)
            if rule_id in known_ids:
                found.append(rule_id)
    return ordered_unique(found)


def extract_term_ids_from_texts(
    texts: list[str],
    glossary_index: list[tuple[str, str]] | None = None,
    max_terms: int = 20,
) -> list[str]:
    haystack = " ".join(text or "" for text in texts).casefold()
    found: list[str] = []
    for term_id, phrase in glossary_index or build_glossary_phrase_index():
        if re.search(rf"\b{re.escape(phrase)}\b", haystack):
            found.append(term_id)
            if len(found) >= max_terms:
                break
    return ordered_unique(found)


def _legality_value(legalities: Any, format_name: str | None) -> str | None:
    if not format_name or not isinstance(legalities, dict):
        return None
    return legalities.get(normalize_text(format_name).replace(" ", "").casefold()) or legalities.get(
        normalize_text(format_name).casefold()
    )


def build_card_context_blocks(
    card_data: dict[str, dict[str, Any]],
    format_name: str | None = None,
) -> dict[str, Any]:
    blocks: list[str] = []
    card_texts: list[str] = []
    card_rulings: list[str] = []

    for card_name, payload in card_data.items():
        oracle_text = payload.get("oracle_text", "")
        type_line = payload.get("type_line", "")
        mana_cost = payload.get("mana_cost", "")
        keywords = payload.get("keywords", []) or []
        rulings = payload.get("rulings", {}) or {}
        legalities = payload.get("legalities", {}) or {}

        block = [f"CARD: {card_name}"]
        if mana_cost:
            block.append(f"Mana Cost: {mana_cost}")
        if type_line:
            block.append(f"Type: {type_line}")
        if oracle_text:
            block.append(f"Oracle Text: {oracle_text}")
        if keywords:
            block.append("Keywords: " + ", ".join(keywords))

        legality = _legality_value(legalities, format_name)
        if legality:
            block.append(f"Legality in {format_name}: {legality}")

        ruling_lines = []
        for date, comments in rulings.items():
            for comment in comments:
                ruling_lines.append(f"- [{date}] {comment}")
                card_rulings.append(f"{card_name} ruling on {date}: {comment}")

        if ruling_lines:
            block.append("Rulings:\n" + "\n".join(ruling_lines))

        blocks.append("\n".join(block))
        card_texts.append(
            " | ".join(
                part
                for part in [card_name, type_line, oracle_text, ", ".join(keywords)]
                if part
            )
        )

    return {
        "card_precontext": "\n\n".join(blocks),
        "card_texts": card_texts,
        "card_rulings": card_rulings,
    }


def _wanted_frame(ids: list[str], column_name: str) -> pl.DataFrame:
    return pl.DataFrame(
        {
            column_name: ids,
            "rank": list(range(len(ids))),
        }
    )


def extract_card_data_from_query(
    query: str,
    processor: KeywordProcessor | None = None,
) -> dict[str, dict[str, Any]]:
    card_names = extract_card_names(query, processor=processor)
    if not card_names:
        return {}

    wanted_cards = _wanted_frame(card_names, "name")
    sql = """
        SELECT
            CI.oracle_id,
            CI.name,
            CI.image_uris,
            CI.mana_cost,
            CI.type_line,
            CI.oracle_text,
            CI.keywords,
            CI.legalities,
            CI.power,
            CI.toughness,
            CI.loyalty,
            CI.life_modifier,
            CI.hand_modifier,
            CI.defense,
            CR.published_at,
            CR.comment,
            W.rank
        FROM wanted_cards AS W
        JOIN mtg_db.cardInfo AS CI
          ON CI.name = W.name
         AND CI.lang = 'en'
        LEFT JOIN mtg_db.cardRulings AS CR
          ON CR.oracle_id = CI.oracle_id
        ORDER BY W.rank, CI.name, CR.published_at
    """
    rows = get_duckdb_gateway().query_polars(sql, register={"wanted_cards": wanted_cards}).to_dicts()

    grouped: dict[str, dict[str, Any]] = {}
    for row in rows:
        card_name = row.pop("name")
        row.pop("rank", None)
        published_at = row.pop("published_at", None)
        comment = normalize_text(row.pop("comment", ""))
        if card_name not in grouped:
            payload = {"name": card_name, "rulings": {}}
            for key, value in row.items():
                if value is None:
                    continue
                if isinstance(value, float) and math.isnan(value):
                    continue
                if isinstance(value, str):
                    value = normalize_text(value)
                    if not value:
                        continue
                payload[key] = value
            payload["colors"] = derive_colors_from_mana_cost(payload.get("mana_cost"))
            grouped[card_name] = payload

        if published_at is not None and comment:
            date_key = str(published_at)
            grouped[card_name]["rulings"].setdefault(date_key, [])
            if comment not in grouped[card_name]["rulings"][date_key]:
                grouped[card_name]["rulings"][date_key].append(comment)

    return grouped


def empty_section_hits() -> pl.DataFrame:
    return pl.DataFrame(
        schema={
            "section_id": pl.Utf8,
            "section_title": pl.Utf8,
            "min_distance": pl.Float64,
            "match_count": pl.Int64,
            "matched_rule_ids": pl.List(pl.Utf8),
            "document_previews": pl.List(pl.Utf8),
        }
    )


def empty_rule_vector_hits() -> pl.DataFrame:
    return pl.DataFrame(
        schema={
            "node_id": pl.Utf8,
            "rule_id": pl.Utf8,
            "section_id": pl.Utf8,
            "title": pl.Utf8,
            "distance": pl.Float64,
            "document_preview": pl.Utf8,
        }
    )


def get_embedding_function():
    global _EMBEDDING_FUNCTION
    if _EMBEDDING_FUNCTION is not None:
        return _EMBEDDING_FUNCTION

    try:
        from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
    except ImportError as exc:
        raise RuntimeError(
            "SentenceTransformerEmbeddingFunction is unavailable. Install sentence-transformers to enable Chroma retrieval."
        ) from exc

    _EMBEDDING_FUNCTION = SentenceTransformerEmbeddingFunction(
        model_name=str(settings.GEMMA_JUDGE_LOCAL_EMBEDDING_MODEL_DIR),
        device="cpu",
        normalize_embeddings=False,
    )
    return _EMBEDDING_FUNCTION


def get_chroma_client() -> chromadb.PersistentClient:
    global _CHROMA_CLIENT
    if _CHROMA_CLIENT is None:
        if not Path(settings.GEMMA_JUDGE_LOCAL_CHROMA_DIR).exists():
            from .runtime import bootstrap_runtime

            bootstrap_runtime(force=False)
        _CHROMA_CLIENT = chromadb.PersistentClient(path=str(settings.GEMMA_JUDGE_LOCAL_CHROMA_DIR))
    return _CHROMA_CLIENT


def get_chroma_collection(name: str) -> Collection:
    if name not in _CHROMA_COLLECTIONS:
        _CHROMA_COLLECTIONS[name] = get_chroma_client().get_collection(
            name=name,
            embedding_function=get_embedding_function(),
        )
    return _CHROMA_COLLECTIONS[name]


def query_rule_sections(
    query_text: str,
    n_results: int = 30,
    max_distance: float = 0.90,
    top_sections: int = 4,
    max_matched_rules_per_section: int = 5,
) -> pl.DataFrame:
    collection = get_chroma_collection(SECTION_ROUTER_COLLECTION_NAME)
    if collection.count() == 0:
        return empty_section_hits()

    result = collection.query(
        query_texts=[query_text],
        n_results=min(n_results, collection.count()),
        where={"node_type": "rule_section_router"},
        include=["metadatas", "documents", "distances"],
    )

    buckets: dict[str, dict[str, Any]] = {}
    ids = result.get("ids", [[]])[0]
    metadatas = result.get("metadatas", [[]])[0]
    documents = result.get("documents", [[]])[0]
    distances = result.get("distances", [[]])[0]

    for index, _node_id in enumerate(ids):
        distance = float(distances[index])
        if distance >= max_distance:
            continue
        metadata = metadatas[index] or {}
        section_id = metadata.get("section_id")
        if not section_id:
            continue

        bucket = buckets.setdefault(
            section_id,
            {
                "section_id": section_id,
                "section_title": metadata.get("section_title") or "",
                "min_distance": distance,
                "match_count": 0,
                "matched_rule_ids": [],
                "document_previews": [],
            },
        )
        bucket["min_distance"] = min(bucket["min_distance"], distance)
        bucket["match_count"] += 1
        rule_id = metadata.get("rule_id") or metadata.get("native_id")
        if rule_id and rule_id not in bucket["matched_rule_ids"]:
            if len(bucket["matched_rule_ids"]) < max_matched_rules_per_section:
                bucket["matched_rule_ids"].append(rule_id)
        preview = (documents[index] or "")[:300]
        if preview and len(bucket["document_previews"]) < max_matched_rules_per_section:
            bucket["document_previews"].append(preview)

    rows = sorted(
        buckets.values(),
        key=lambda row: (row["min_distance"], -row["match_count"], row["section_id"]),
    )[:top_sections]
    return pl.DataFrame(rows, schema=empty_section_hits().schema)


def extract_router_rule_ids(section_hits: pl.DataFrame) -> list[str]:
    if section_hits is None or section_hits.height == 0:
        return []
    return ordered_unique(
        [
            rule_id
            for rule_ids in section_hits.get_column("matched_rule_ids").to_list()
            for rule_id in (rule_ids or [])
        ]
    )


def build_rule_vector_where(section_ids: list[str] | None = None) -> dict[str, Any]:
    clauses: list[dict[str, Any]] = [
        {"node_type": "rule"},
        {"is_context_rule": True},
    ]
    clean_section_ids = ordered_unique(section_ids or [])
    if len(clean_section_ids) == 1:
        clauses.append({"section_id": clean_section_ids[0]})
    elif clean_section_ids:
        clauses.append({"section_id": {"$in": clean_section_ids}})
    return {"$and": clauses}


def query_rule_vectors(
    query_text: str,
    n_results: int = 5,
    max_distance: float | None = 0.55,
    section_ids: list[str] | None = None,
) -> pl.DataFrame:
    collection = get_chroma_collection(RULES_COLLECTION_NAME)
    if collection.count() == 0:
        return empty_rule_vector_hits()

    result = collection.query(
        query_texts=[query_text],
        n_results=min(n_results, collection.count()),
        where=build_rule_vector_where(section_ids=section_ids),
        include=["metadatas", "documents", "distances"],
    )

    rows = []
    ids = result.get("ids", [[]])[0]
    metadatas = result.get("metadatas", [[]])[0]
    documents = result.get("documents", [[]])[0]
    distances = result.get("distances", [[]])[0]

    for index, node_id in enumerate(ids):
        distance = float(distances[index])
        if max_distance is not None and distance >= max_distance:
            continue
        metadata = metadatas[index] or {}
        rows.append(
            {
                "node_id": node_id,
                "rule_id": metadata.get("native_id"),
                "section_id": metadata.get("section_id"),
                "title": metadata.get("title"),
                "distance": distance,
                "document_preview": (documents[index] or "")[:300],
            }
        )

    return pl.DataFrame(rows, schema=empty_rule_vector_hits().schema)


def combine_vector_hits(*frames: pl.DataFrame) -> pl.DataFrame:
    non_empty = [frame for frame in frames if frame is not None and frame.height > 0]
    if not non_empty:
        return empty_rule_vector_hits()
    return pl.concat(non_empty, how="vertical").unique(
        subset=["rule_id"],
        keep="first",
        maintain_order=True,
    )


def hydrate_rules(rule_ids: list[str]) -> pl.DataFrame:
    ids = ordered_unique(rule_ids)
    if not ids:
        return pl.DataFrame(schema=EMPTY_RULES_SCHEMA)
    wanted = _wanted_frame(ids, "rule_id")
    sql = """
        SELECT r.rule_id, r.rule_text, r.linked_rules, r.linked_terms
        FROM wanted AS w
        JOIN mtg_db.mtg_rules AS r
          ON r.rule_id = w.rule_id
        ORDER BY w.rank
    """
    return get_duckdb_gateway().query_polars(sql, register={"wanted": wanted})


def hydrate_glossary_terms(term_ids: list[str]) -> pl.DataFrame:
    ids = ordered_unique(term_ids)
    if not ids:
        return pl.DataFrame(schema=EMPTY_GLOSSARY_SCHEMA)
    wanted = _wanted_frame(ids, "term_id")
    sql = """
        SELECT g.term_id, g.term_name, g.definition, g.linked_rules, g.linked_terms
        FROM wanted AS w
        JOIN mtg_db.mtg_glossary_terms AS g
          ON g.term_id = w.term_id
        ORDER BY w.rank
    """
    return get_duckdb_gateway().query_polars(sql, register={"wanted": wanted})


def graph_expand_nodes(
    seed_rule_ids: list[str] | None = None,
    seed_term_ids: list[str] | None = None,
) -> pl.DataFrame:
    seed_rows = (
        [{"node_type": "rule", "native_id": rule_id} for rule_id in ordered_unique(seed_rule_ids or [])]
        + [{"node_type": "glossary", "native_id": term_id} for term_id in ordered_unique(seed_term_ids or [])]
    )
    schema = {
        "node_type": pl.Utf8,
        "native_id": pl.Utf8,
        "min_hop": pl.Int64,
    }
    if not seed_rows:
        return pl.DataFrame(schema=schema)

    sql = """
        WITH first_hop AS (
            SELECT DISTINCT 1 AS hop, e.target_type AS node_type, e.target_id AS native_id
            FROM seed_nodes AS s
            JOIN mtg_db.mtg_edges AS e
              ON e.source_type = s.node_type
             AND e.source_id = s.native_id
        ),
        second_hop AS (
            SELECT DISTINCT 2 AS hop, e.target_type AS node_type, e.target_id AS native_id
            FROM first_hop AS f
            JOIN mtg_db.mtg_edges AS e
              ON e.source_type = f.node_type
             AND e.source_id = f.native_id
        ),
        all_hits AS (
            SELECT * FROM first_hop
            UNION ALL
            SELECT * FROM second_hop
        )
        SELECT node_type, native_id, MIN(hop) AS min_hop
        FROM all_hits
        GROUP BY node_type, native_id
        ORDER BY min_hop, node_type, native_id
    """
    return get_duckdb_gateway().query_polars(
        sql,
        register={"seed_nodes": pl.DataFrame(seed_rows)},
    )


def expand_section_rule_ids_for_context(
    rule_ids: list[str],
    max_children_per_section: int = 10,
) -> list[str]:
    all_rule_ids = get_all_rule_ids()
    expanded: list[str] = []
    for rule_id in ordered_unique(rule_ids):
        rule_id = str(rule_id)
        if is_context_rule_id(rule_id):
            expanded.append(rule_id)
        elif re.fullmatch(r"\d{3}", rule_id):
            expanded.extend(
                [
                    candidate
                    for candidate in all_rule_ids
                    if candidate.startswith(f"{rule_id}.")
                ][:max_children_per_section]
            )
    return ordered_unique(expanded)


def format_rule_context(rule_rows: pl.DataFrame) -> str:
    if rule_rows is None or rule_rows.height == 0:
        return ""
    lines = []
    for row in rule_rows.iter_rows(named=True):
        rule_id = row.get("rule_id")
        rule_text = row.get("rule_text")
        if not rule_id or not rule_text:
            continue
        lines.append(f"Rule {rule_id} : {' '.join(str(rule_text).split())}")
    return "\n\n".join(lines)


def format_glossary_context(glossary_rows: pl.DataFrame) -> str:
    if glossary_rows is None or glossary_rows.height == 0:
        return ""
    lines = []
    for row in glossary_rows.iter_rows(named=True):
        term_id = row.get("term_id")
        term_name = row.get("term_name")
        definition = row.get("definition")
        if not term_id or not term_name or not definition:
            continue
        lines.append(f"{term_name} ({term_id}) : {' '.join(str(definition).split())}")
    return "\n\n".join(lines)


def search_cards(query: str, limit: int = 8) -> list[dict[str, Any]]:
    needle = normalize_text(query).lower()
    if len(needle) < 2:
        return []
    like = f"%{needle}%"
    fetch_limit = max(limit * 6, 8)
    sql = """
        SELECT DISTINCT 
            oracle_id,
            name,
            image_uris,
            mana_cost,
            type_line,
            oracle_text,
            power,
            toughness
        FROM mtg_db.cardInfo
        WHERE lang = 'en'
          AND (
              lower(name) LIKE ?
              OR lower(coalesce(type_line, '')) LIKE ?
              OR lower(coalesce(oracle_text, '')) LIKE ?
          )
        ORDER BY
            CASE
                WHEN lower(name) = ? THEN 0
                WHEN lower(name) LIKE ? THEN 1
                ELSE 2
            END,
            name
        LIMIT ?
    """
    frame = get_duckdb_gateway().query_polars(
        sql,
        params=[like, like, like, needle, f"{needle}%", fetch_limit],
    )
    rows = frame.to_dicts()
    unique_rows: list[dict[str, Any]] = []
    seen_names: set[str] = set()

    for row in rows:
        card_name = normalize_text(row.get("name"))
        if not card_name or card_name in seen_names:
            continue

        row["name"] = card_name
        row["colors"] = derive_colors_from_mana_cost(row.get("mana_cost"))
        unique_rows.append(row)
        seen_names.add(card_name)

        if len(unique_rows) >= limit:
            break

    return unique_rows


def search_rules(query: str, limit: int = 10) -> list[dict[str, Any]]:
    needle = normalize_text(query).lower()
    if len(needle) < 2:
        return []
    like = f"%{needle}%"
    sql = """
        SELECT DISTINCT rule_id, rule_text
        FROM mtg_db.mtg_rules
        WHERE lower(rule_id) LIKE ?
           OR lower(rule_text) LIKE ?
        ORDER BY
            CASE WHEN lower(rule_id) LIKE ? THEN 0 ELSE 1 END,
            rule_id
        LIMIT ?
    """
    frame = get_duckdb_gateway().query_polars(
        sql,
        params=[like, like, f"{needle}%", limit],
    )
    return [
        {
            "rule_id": row["rule_id"],
            "title": f"Rule {row['rule_id']}",
            "content": row["rule_text"],
        }
        for row in frame.to_dicts()
    ]


def _format_cards_for_ui(card_data: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    cards = []
    for payload in card_data.values():
        cards.append(
            {
                "oracle_id": payload.get("oracle_id"),
                "name": payload.get("name"),
                "image_uris": payload.get("image_uris"),
                "mana_cost": payload.get("mana_cost"),
                "type_line": payload.get("type_line"),
                "oracle_text": payload.get("oracle_text"),
                "power": payload.get("power"),
                "toughness": payload.get("toughness"),
                "colors": payload.get("colors", []),
                "rulings": payload.get("rulings", {}),
            }
        )
    return cards


def _format_rules_for_ui(rule_rows: pl.DataFrame) -> list[dict[str, Any]]:
    return [
        {
            "rule_id": row["rule_id"],
            "title": f"Rule {row['rule_id']}",
            "content": row["rule_text"],
        }
        for row in rule_rows.to_dicts()
    ]


def retrieve_rules_and_glossary_context(
    question: str,
    format_name: str | None = None,
    game_context: Any = None,
    assumptions: list[str] | None = None,
    unknowns: list[str] | None = None,
) -> dict[str, Any]:
    normalized_format = normalize_format_name(format_name)
    format_inferred = False
    if normalized_format is None:
        normalized_format = infer_format_from_question(question)
        format_inferred = normalized_format is not None

    warnings: list[str] = []
    game_context_block = build_game_context_block(
        format_name=normalized_format,
        format_inferred=format_inferred,
        game_context=game_context,
        assumptions=assumptions,
        unknowns=unknowns,
    )

    card_data = extract_card_data_from_query(question)
    card_context = build_card_context_blocks(card_data, format_name=normalized_format)
    all_card_texts = card_context["card_texts"]
    all_card_rulings = card_context["card_rulings"]
    retrieval_query = build_rule_retrieval_query(
        question=question,
        card_texts=all_card_texts,
        card_rulings=all_card_rulings,
        game_context_text=game_context_block,
    )
    source_texts = [question, game_context_block] + all_card_texts + all_card_rulings

    explicit_rule_ids = extract_rule_ids_from_texts(source_texts)
    explicit_term_ids = extract_term_ids_from_texts(source_texts, glossary_index=build_glossary_phrase_index(), max_terms=10)

    section_hits = empty_section_hits()
    vector_hits = empty_rule_vector_hits()
    section_router_rule_ids: list[str] = []
    graph_rule_ids: list[str] = []
    graph_term_ids: list[str] = []

    try:
        section_hits = query_rule_sections(retrieval_query)
        section_ids = section_hits.get_column("section_id").to_list() if section_hits.height else []
        section_router_rule_ids = extract_router_rule_ids(section_hits)
        expanded_nodes = graph_expand_nodes(
            ordered_unique(explicit_rule_ids + section_router_rule_ids),
            explicit_term_ids,
        )
        if expanded_nodes.height:
            graph_rule_ids = (
                expanded_nodes.filter(pl.col("node_type") == "rule").get_column("native_id").to_list()
            )
            graph_term_ids = (
                expanded_nodes.filter(pl.col("node_type") == "glossary").get_column("native_id").to_list()
            )

        scoped_vector_hits = (
            query_rule_vectors(
                retrieval_query,
                n_results=12,
                max_distance=0.85,
                section_ids=section_ids,
            )
            if section_ids
            else empty_rule_vector_hits()
        )
        general_vector_hits = query_rule_vectors(retrieval_query, n_results=5, max_distance=0.55)
        vector_hits = combine_vector_hits(scoped_vector_hits, general_vector_hits)
    except Exception as exc:
        warnings.append(f"Vector retrieval degraded: {exc}")

    vector_rule_ids = vector_hits.get_column("rule_id").to_list() if vector_hits.height else []

    context_rule_ids = ordered_unique(
        expand_section_rule_ids_for_context(explicit_rule_ids, max_children_per_section=10)
        + [rule_id for rule_id in section_router_rule_ids if is_context_rule_id(rule_id)]
        + [rule_id for rule_id in graph_rule_ids if is_context_rule_id(rule_id)]
        + expand_section_rule_ids_for_context(vector_rule_ids, max_children_per_section=5)
    )
    rule_rows = hydrate_rules(context_rule_ids)
    if rule_rows.height:
        keep_rule_ids = [
            row["rule_id"]
            for row in rule_rows.iter_rows(named=True)
            if is_context_rule_id(row["rule_id"]) and word_count(row["rule_text"]) >= 4
        ]
        rule_rows = rule_rows.filter(pl.col("rule_id").is_in(keep_rule_ids))

    terms_linked_by_rules = ordered_unique(
        [
            term
            for terms in (rule_rows.get_column("linked_terms").to_list() if rule_rows.height else [])
            for term in (terms or [])
        ]
    )
    final_term_ids = ordered_unique(explicit_term_ids + graph_term_ids + terms_linked_by_rules)[:30]
    glossary_rows = hydrate_glossary_terms(final_term_ids)
    if glossary_rows.height:
        keep_term_ids = [
            row["term_id"]
            for row in glossary_rows.iter_rows(named=True)
            if word_count(row["definition"]) >= 4
        ]
        glossary_rows = glossary_rows.filter(pl.col("term_id").is_in(keep_term_ids))

    rule_context = format_rule_context(rule_rows)
    glossary_context = format_glossary_context(glossary_rows)
    combined_context = "\n\n".join(
        part
        for part in [
            "GAME FORMAT CONTEXT\n" + game_context_block if game_context_block else "",
            "RELEVANT CARDS\n" + card_context["card_precontext"] if card_context["card_precontext"] else "",
            "RELEVANT GLOSSARY TERMS\n" + glossary_context if glossary_context else "",
            "RELEVANT COMPREHENSIVE RULES\n" + rule_context if rule_context else "",
        ]
        if part
    )

    return {
        "card_data": card_data,
        "cards_for_ui": _format_cards_for_ui(card_data),
        "card_precontext": card_context["card_precontext"],
        "card_texts": all_card_texts,
        "card_rulings": all_card_rulings,
        "format_name": normalized_format,
        "format_inferred": format_inferred,
        "game_context_block": game_context_block,
        "retrieval_query": retrieval_query,
        "explicit_rule_ids": explicit_rule_ids,
        "explicit_term_ids": explicit_term_ids,
        "section_hits": section_hits,
        "section_router_rule_ids": section_router_rule_ids,
        "vector_hits": vector_hits,
        "rule_rows": rule_rows,
        "rules_for_ui": _format_rules_for_ui(rule_rows),
        "glossary_rows": glossary_rows,
        "combined_context": combined_context,
        "warnings": warnings,
    }
