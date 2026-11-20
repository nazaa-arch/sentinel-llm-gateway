"""SQLite-backed semantic cache with exact-hash fast path and multi-view TF/n-gram cosine lookup."""

from __future__ import annotations

import hashlib
import re
import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer

# High-frequency bilingual (EN + ID) boilerplate and stopwords stripped before vectorization
# so shared phrasing ("how do I...", "tolong jelaskan...") does not cause false-positive hits.
_BILINGUAL_STOPWORDS: frozenset[str] = frozenset(
    {
        # English function words & conversational wrappers
        "a",
        "an",
        "the",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "do",
        "does",
        "did",
        "can",
        "could",
        "would",
        "should",
        "will",
        "may",
        "might",
        "please",
        "kindly",
        "tell",
        "me",
        "us",
        "about",
        "what",
        "whats",
        "which",
        "who",
        "where",
        "when",
        "why",
        "how",
        "explain",
        "describe",
        "define",
        "give",
        "show",
        "help",
        "need",
        "want",
        "know",
        "understand",
        "to",
        "of",
        "in",
        "on",
        "at",
        "for",
        "from",
        "by",
        "with",
        "as",
        "into",
        "between",
        "among",
        "and",
        "or",
        "but",
        "if",
        "then",
        "so",
        "that",
        "this",
        "these",
        "those",
        "it",
        "its",
        "you",
        "your",
        "we",
        "our",
        "they",
        "their",
        "i",
        "my",
        "way",
        "ways",
        "work",
        "works",
        "briefly",
        "shortly",
        "detail",
        "details",
        "overview",
        "summary",
        # Indonesian function words & conversational wrappers
        "apa",
        "apakah",
        "bagaimana",
        "gimana",
        "mengapa",
        "kenapa",
        "siapa",
        "dimana",
        "kapan",
        "tolong",
        "mohon",
        "bisa",
        "dapat",
        "jelaskan",
        "terangkan",
        "uraikan",
        "sebutkan",
        "berikan",
        "kasih",
        "tahu",
        "tentang",
        "mengenai",
        "soal",
        "itu",
        "ini",
        "yang",
        "dan",
        "atau",
        "tetapi",
        "tapi",
        "namun",
        "di",
        "ke",
        "dari",
        "pada",
        "untuk",
        "buat",
        "bagi",
        "dengan",
        "secara",
        "dalam",
        "oleh",
        "sebagai",
        "adalah",
        "ialah",
        "merupakan",
        "cara",
        "kerja",
        "bekerja",
        "saya",
        "aku",
        "kami",
        "kita",
        "kamu",
        "anda",
        "mereka",
        "nya",
        "dong",
        "ya",
        "singkat",
        "singkatnya",
        "pengertian",
        "definisi",
        "maksud",
        "arti",
    }
)

# Light domain-neutral bilingual synonym canonicalization for paraphrase alignment
_SYNONYM_MAP: dict[str, str] = {
    "biaya": "cost",
    "harga": "cost",
    "price": "cost",
    "pricing": "cost",
    "tarif": "cost",
    "hemat": "reduce",
    "menghemat": "reduce",
    "kurangi": "reduce",
    "mengurangi": "reduce",
    "menurunkan": "reduce",
    "lower": "reduce",
    "cut": "reduce",
    "minimize": "reduce",
    "save": "reduce",
    "saving": "reduce",
    "savings": "reduce",
    "cegah": "prevent",
    "mencegah": "prevent",
    "pencegahan": "prevent",
    "mitigate": "prevent",
    "mitigasi": "prevent",
    "block": "prevent",
    "blocking": "prevent",
    "stop": "prevent",
    "defend": "prevent",
    "protect": "prevent",
    "melindungi": "prevent",
    "perlindungan": "prevent",
    "keamanan": "security",
    "aman": "security",
    "secure": "security",
    "latensi": "latency",
    "kecepatan": "latency",
    "cepat": "latency",
    "speed": "latency",
    "faster": "latency",
    "response": "latency",
    "redaksi": "redact",
    "redaction": "redact",
    "masking": "redact",
    "mask": "redact",
    "sensor": "redact",
    "menyensor": "redact",
    "samarkan": "redact",
    "menyamarkan": "redact",
    "anonymize": "redact",
    "data": "data",
    "pribadi": "pii",
    "personal": "pii",
    "sensitif": "pii",
    "sensitive": "pii",
    "caching": "cache",
    "tembolok": "cache",
    "semantik": "semantic",
    "vektor": "vector",
    "penyimpanan": "storage",
    "arsitektur": "architecture",
    "model": "model",
    "bahasa": "language",
    "serangan": "attack",
    "injeksi": "injection",
    "validasi": "validate",
    "validation": "validate",
    "cek": "validate",
    "check": "validate",
    "verify": "validate",
    "verification": "validate",
    "verifikasi": "validate",
    "perbedaan": "difference",
    "beda": "difference",
    "compare": "difference",
    "comparison": "difference",
    "bandingkan": "difference",
    "versus": "difference",
    "vs": "difference",
    "keuntungan": "benefit",
    "manfaat": "benefit",
    "benefits": "benefit",
    "advantage": "benefit",
    "advantages": "benefit",
}

_TOKEN_RE = re.compile(r"[a-z0-9_<>]+")


def normalize_for_exact_hash(text: str) -> str:
    """Whitespace- and case-normalized string for O(1) exact cache lookups."""
    return " ".join(text.lower().split())


def _light_stem(token: str) -> str:
    """Apply conservative English/Indonesian morphological normalization."""
    if len(token) > 4 and token.endswith("ies"):
        return token[:-3] + "y"
    if len(token) >= 4 and token.endswith("s") and not token.endswith(("ss", "us", "is")):
        return token[:-1]
    return token


def canonicalize_for_embedding(text: str) -> str:
    """Normalize text by stripping bilingual boilerplate and mapping common synonyms."""
    tokens = _TOKEN_RE.findall(text.lower())
    filtered: list[str] = []
    for tok in tokens:
        if tok in _BILINGUAL_STOPWORDS:
            continue
        mapped = _SYNONYM_MAP.get(tok)
        if mapped is None:
            stemmed = _light_stem(tok)
            if stemmed in _BILINGUAL_STOPWORDS:
                continue
            mapped = _SYNONYM_MAP.get(stemmed, stemmed)
        filtered.append(mapped)
    if not filtered:
        return normalize_for_exact_hash(text)
    return " ".join(filtered)


class PromptVectorizer:
    """Stateless multi-view (word + char_wb n-gram) L2-normalized dense prompt encoder."""

    def __init__(self, word_dims: int = 512, char_dims: int = 512) -> None:
        self.word_dims = word_dims
        self.char_dims = char_dims
        self.dim = word_dims + char_dims
        self._word_vec = HashingVectorizer(
            n_features=word_dims,
            analyzer="word",
            ngram_range=(1, 1),
            alternate_sign=False,
            norm="l2",
            lowercase=True,
        )
        self._char_vec = HashingVectorizer(
            n_features=char_dims,
            analyzer="char_wb",
            ngram_range=(3, 5),
            alternate_sign=False,
            norm="l2",
            lowercase=True,
        )

    def encode(self, text: str) -> np.ndarray:
        """Encode ``text`` into a unit-norm ``float32`` vector of length ``self.dim``."""
        canonical = canonicalize_for_embedding(text)
        w_vec = self._word_vec.transform([canonical]).toarray()[0].astype(np.float32)
        c_vec = self._char_vec.transform([canonical]).toarray()[0].astype(np.float32)

        # Blend word semantics (0.72) and subword/morphological n-grams (0.28)
        combined = np.concatenate([w_vec * 0.72, c_vec * 0.28]).astype(np.float32)
        norm = float(np.linalg.norm(combined))
        if norm > 0.0:
            combined /= norm
        return combined

    @staticmethod
    def cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
        """Compute cosine similarity between two L2-normalized vectors."""
        return float(np.clip(np.dot(vec_a, vec_b), -1.0, 1.0))


@dataclass(frozen=True)
class CacheLookupResult:
    """Result returned on a cache hit."""

    response_text: str
    similarity: float
    exact_hit: bool
    provider: str
    model: str
    prompt_tokens_est: int
    completion_tokens_est: int


class SemanticCache:
    """Persistent SQLite semantic cache supporting exact-hash and cosine-similarity hits."""

    def __init__(
        self,
        db_path: str | Path,
        default_threshold: float = 0.86,
        ttl_seconds: int = 3600,
        max_entries: int = 2000,
    ) -> None:
        self.db_path = Path(db_path)
        self.default_threshold = default_threshold
        self.ttl_seconds = ttl_seconds
        self.max_entries = max_entries
        self.vectorizer = PromptVectorizer()
        self._lock = threading.RLock()
        self._conn: sqlite3.Connection | None = None

    def _ensure_connected(self) -> sqlite3.Connection:
        """Lazily open the SQLite connection and initialize schema under lock."""
        if self._conn is not None:
            return self._conn
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(self.db_path), timeout=30.0, check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA busy_timeout=5000;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        self._conn = conn
        self._init_schema(conn)
        return conn

    def initialize(self) -> None:
        """Eagerly initialize the underlying SQLite connection and schema."""
        with self._lock:
            self._ensure_connected()

    def _init_schema(self, conn: sqlite3.Connection) -> None:
        with conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS semantic_cache (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    exact_hash TEXT NOT NULL,
                    scope_key TEXT NOT NULL,
                    model TEXT NOT NULL,
                    provider TEXT NOT NULL,
                    prompt_text TEXT NOT NULL,
                    response_text TEXT NOT NULL,
                    embedding BLOB NOT NULL,
                    prompt_tokens_est INTEGER NOT NULL DEFAULT 0,
                    completion_tokens_est INTEGER NOT NULL DEFAULT 0,
                    created_at REAL NOT NULL,
                    expires_at REAL NOT NULL,
                    hit_count INTEGER NOT NULL DEFAULT 0
                );
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_cache_exact "
                "ON semantic_cache (exact_hash, scope_key);"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_cache_scope_exp "
                "ON semantic_cache (scope_key, expires_at);"
            )

    @staticmethod
    def _make_scope_key(model: str, system_prompt: str) -> str:
        raw = f"{model}::{normalize_for_exact_hash(system_prompt)}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]

    @staticmethod
    def _make_exact_hash(scope_key: str, prompt: str) -> str:
        raw = f"{scope_key}::{normalize_for_exact_hash(prompt)}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def lookup(
        self,
        prompt: str,
        model: str,
        system_prompt: str = "",
        threshold: float | None = None,
    ) -> CacheLookupResult | None:
        """Check exact-hash fast path first, then vectorized cosine similarity lookup."""
        active_threshold = threshold if threshold is not None else self.default_threshold
        scope_key = self._make_scope_key(model, system_prompt)
        exact_hash = self._make_exact_hash(scope_key, prompt)
        now = time.time()

        with self._lock:
            conn = self._ensure_connected()

            # 1. Exact-hash fast path (O(1) indexed lookup; read-only on miss)
            row = conn.execute(
                """
                SELECT id, response_text, provider, model, prompt_tokens_est, completion_tokens_est
                FROM semantic_cache
                WHERE exact_hash = ? AND scope_key = ? AND expires_at > ?
                ORDER BY created_at DESC
                LIMIT 1
                """,
                (exact_hash, scope_key, now),
            ).fetchone()
            if row is not None:
                entry_id, resp, provider, cached_model, p_tok, c_tok = row
                with conn:
                    conn.execute(
                        "UPDATE semantic_cache SET hit_count = hit_count + 1 WHERE id = ?",
                        (entry_id,),
                    )
                return CacheLookupResult(
                    response_text=resp,
                    similarity=1.0,
                    exact_hit=True,
                    provider=provider,
                    model=cached_model,
                    prompt_tokens_est=p_tok,
                    completion_tokens_est=c_tok,
                )

            # 2. Vectorized semantic similarity lookup within same model/system scope
            rows = conn.execute(
                """
                SELECT id, response_text, provider, model, embedding,
                       prompt_tokens_est, completion_tokens_est
                FROM semantic_cache
                WHERE scope_key = ? AND expires_at > ?
                """,
                (scope_key, now),
            ).fetchall()
            if not rows:
                return None

            query_vec = self.vectorizer.encode(prompt)
            valid_rows = []
            vectors = []
            for r in rows:
                vec = np.frombuffer(r[4], dtype=np.float32)
                if vec.shape == (self.vectorizer.dim,):
                    valid_rows.append(r)
                    vectors.append(vec)
            if not vectors:
                return None

            matrix = np.vstack(vectors)
            similarities = matrix @ query_vec
            best_idx = int(np.argmax(similarities))
            best_sim = float(np.clip(similarities[best_idx], -1.0, 1.0))

            if best_sim >= active_threshold:
                best_row = valid_rows[best_idx]
                with conn:
                    conn.execute(
                        "UPDATE semantic_cache SET hit_count = hit_count + 1 WHERE id = ?",
                        (best_row[0],),
                    )
                return CacheLookupResult(
                    response_text=best_row[1],
                    similarity=round(best_sim, 4),
                    exact_hit=False,
                    provider=best_row[2],
                    model=best_row[3],
                    prompt_tokens_est=best_row[5],
                    completion_tokens_est=best_row[6],
                )
            return None

    def store(
        self,
        prompt: str,
        response_text: str,
        model: str,
        provider: str,
        system_prompt: str = "",
        prompt_tokens_est: int = 0,
        completion_tokens_est: int = 0,
        ttl_seconds: int | None = None,
    ) -> None:
        """Persist a redacted prompt/response pair and its embedding with TTL."""
        ttl = ttl_seconds if ttl_seconds is not None else self.ttl_seconds
        now = time.time()
        expires_at = now + ttl
        scope_key = self._make_scope_key(model, system_prompt)
        exact_hash = self._make_exact_hash(scope_key, prompt)
        embedding = self.vectorizer.encode(prompt).tobytes()

        with self._lock:
            conn = self._ensure_connected()
            with conn:
                conn.execute("DELETE FROM semantic_cache WHERE expires_at <= ?", (now,))
                # Replace existing exact entry to keep cache clean
                conn.execute(
                    "DELETE FROM semantic_cache WHERE exact_hash = ? AND scope_key = ?",
                    (exact_hash, scope_key),
                )
                conn.execute(
                    """
                    INSERT INTO semantic_cache (
                        exact_hash, scope_key, model, provider, prompt_text, response_text,
                        embedding, prompt_tokens_est, completion_tokens_est, created_at, expires_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        exact_hash,
                        scope_key,
                        model,
                        provider,
                        prompt,
                        response_text,
                        embedding,
                        prompt_tokens_est,
                        completion_tokens_est,
                        now,
                        expires_at,
                    ),
                )
                # Enforce max_entries cap (evict oldest entries)
                count = conn.execute("SELECT COUNT(*) FROM semantic_cache").fetchone()[0]
                if count > self.max_entries:
                    overflow = count - self.max_entries
                    conn.execute(
                        """
                        DELETE FROM semantic_cache
                        WHERE id IN (
                            SELECT id FROM semantic_cache ORDER BY created_at ASC LIMIT ?
                        )
                        """,
                        (overflow,),
                    )

    def count(self) -> int:
        """Return count of active non-expired cache entries."""
        now = time.time()
        with self._lock:
            conn = self._ensure_connected()
            with conn:
                conn.execute("DELETE FROM semantic_cache WHERE expires_at <= ?", (now,))
            return int(conn.execute("SELECT COUNT(*) FROM semantic_cache").fetchone()[0])

    def clear(self) -> int:
        """Remove all cache entries and return the number deleted."""
        with self._lock:
            conn = self._ensure_connected()
            with conn:
                cur = conn.execute("DELETE FROM semantic_cache")
            return int(cur.rowcount)

    def close(self) -> None:
        with self._lock:
            if self._conn is not None:
                self._conn.close()
                self._conn = None

    def __enter__(self) -> SemanticCache:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()
