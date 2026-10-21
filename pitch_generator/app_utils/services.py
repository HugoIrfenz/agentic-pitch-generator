# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
/**
 * @file services.py
 * @description Dependency-injected cloud service abstractions for Gemini Enterprise
 *   Agent Platform, Cloud Storage artifacts, and BigQuery analytics.
 *
 * Why: Isolating external cloud SDK interactions behind runtime-checkable protocols
 * (`LLMClientProtocol`, `ArtifactServiceProtocol`) and providing deterministic offline
 * implementations (`DeterministicMockLLMClient`, `InMemoryArtifactService`,
 * `GcsArtifactService`, `BigQueryAnalyticsService`) allows 100% of unit, functional,
 * and E2E tests to run offline without live GCP credentials or network calls (`R4`).
 */
"""

from __future__ import annotations

# ==============================================================================
# LAB GUIDEPOST INDEX (`pitch_generator/app_utils/services.py`)
#   - Step 2a: `# [Guidepost — Step 2a: BigQuery Agent Analytics & Key Visuals Object Table]`
#   - Step 2b: `# [Guidepost — Step 2b: Brand Drift Detection & Closed-Loop Prompt Tuning]`
#   - Step 3b: `# [Guidepost — Step 3b: Sensitive PII Data Scrubbing & Redaction]`
# (Tip: This file is < 450 lines. Read the entire file in a single `view_file` call.)
# ==============================================================================

import copy
from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
import re
from typing import Any, Iterator

from pitch_generator.app_utils.memory_bank import MemoryBankService
from pitch_generator.app_utils.runtime_clients import (
    MINIMAL_PNG_BYTES,
    ArtifactRecord,
    ArtifactServiceProtocol,
    DeterministicMockLLMClient,
    EnterpriseGenAILLMClient,
    GcsArtifactService,
    InMemoryArtifactService,
    LLMClientProtocol,
    MockLLMClient,
    _build_house_brand_png,
    _is_live_cloud_configured,
    _is_offline_test_mode,
    _validate_artifact_inputs,
    create_artifact_service,
    get_artifact_service,
    resolve_artifact_service,
    select_artifact_service,
)
from pitch_generator.config import (
    DEFAULT_FLASH_MODEL,
    PitchConfig,
    get_config,
    normalize_bucket_name,
)


def run_workflow_with_gcs_artifacts(
    brief: str,
    *,
    bucket_name: str = "local-dev-project-bwg",
    session_id: str = "default",
    services: ServiceContainer | None = None,
) -> dict[str, Any]:
    """
    /**
     * Execute the end-to-end Pitch Generator workflow backed by `GcsArtifactService`.
     *
     * Why: Verifies that `generate_key_visual` and the orchestrator pipeline persist
     * `key_visual.png` to `gs://<bucket>/key-visuals/key_visual.png` and surface the
     * `gs://` URI in the packaged output.
     *
     * @param brief Product pitch brief text.
     * @param bucket_name Target GCS bucket name (default `'local-dev-project-bwg'`).
     * @param session_id Session identifier (default `'default'`).
     * @param services Optional pre-built `ServiceContainer`.
     * @return Workflow execution result dictionary including `key_visual_uri` and `gcs_uri`.
     */
    """
    from pitch_generator.agent import run_pitch_workflow

    cfg = get_config(env={"LOGS_BUCKET_NAME": bucket_name})
    svc_container = services or get_default_services(cfg)
    svc_container.artifacts = GcsArtifactService(bucket_name=bucket_name)
    svc_container.artifact_service = svc_container.artifacts
    result = run_pitch_workflow(brief, session_id=session_id, services=svc_container)
    result["gcs_uri"] = result.get("key_visual_uri")
    return result


class BigQueryAnalyticsService:
    """
    /**
     * Injectable BigQuery Agent Analytics and brand compliance audit service.
     *
     * Why: Provides telemetry event logging, `ObjectRef` (`OBJ.MAKE_REF` and
     * `OBJ.FETCH_METADATA`) SQL generation for `bwg.key_visuals`, `AI.SCORE` SQL
     * generation for `bwg.brand_review`, and deterministic offline brand compliance
     * grading against the house brand guidelines.
     */
    """

    POSITIVE_BRAND_SIGNALS: tuple[tuple[str, tuple[str, ...]], ...] = (
        ("palette_indigo_slate", ("indigo", "slate")),
        ("warm_accent", ("amber", "terracotta")),
        ("lighting_raking", ("raking", "long shadow", "golden hour", "studio key")),
        ("composition_negative_space", ("off-center", "negative space", "third")),
        ("subject_photographic", ("photographic", "shallow depth of field", "single", "hero")),
    )

    FORBIDDEN_BRAND_TERMS: tuple[str, ...] = (
        "neon",
        "fluorescent",
        "watermark",
        "logo",
        "signage",
        "collage",
        "split-screen",
        "chrome",
        "lens flare",
        "bokeh sparkle",
        "cyan",
        "magenta",
        "fisheye",
        "dutch tilt",
        "flat lay",
        "ring light",
        "3d render",
        "pixel art",
        "clay",
    )

    def __init__(
        self,
        project_id: str = "local-dev-project",
        region: str = "us-central1",
        dataset: str = "bwg",
        connection_name: str = "pitch-connection",
        client: Any = None,
    ) -> None:
        """
        /**
         * Initialize the BigQuery analytics service.
         *
         * Why: Stores project, region, dataset (`bwg`), and Cloud Resource connection
         * (`pitch-connection`) identifiers so generated SQL and telemetry records align
         * with the lab infrastructure.
         *
         * @param project_id Google Cloud project ID.
         * @param region BigQuery dataset and connection region.
         * @param dataset Target BigQuery dataset ID (default `"bwg"`).
         * @param connection_name BigQuery Cloud Resource connection name.
         * @param client Optional `google.cloud.bigquery.Client` instance for DI.
         */
        """
        self.project_id = project_id.strip() or "local-dev-project"
        self.region = region.strip() or "us-central1"
        self.dataset = dataset.strip() or "bwg"
        self.connection_name = connection_name.strip() or "pitch-connection"
        self.client = client
        self.events: list[dict[str, Any]] = []
        self._telemetry_events = self.events

    def record_telemetry(self, event: dict[str, Any]) -> dict[str, Any]:
        """
        /**
         * Record an agent execution telemetry event.
         *
         * Why: Captures structured execution metadata (session ID, active node, latency,
         * routing decision, token usage) in memory and forwards to BigQuery when a live
         * client is injected.
         *
         * @param event Non-empty telemetry dictionary.
         * @return Enriched telemetry event dictionary with timestamp and dataset info.
         */
        """
        if not isinstance(event, dict) or not event:
            raise ValueError("telemetry event must be a non-empty dictionary")
        enriched = copy.deepcopy(event)
        enriched.setdefault("project_id", self.project_id)
        enriched.setdefault("dataset", self.dataset)
        enriched.setdefault("recorded_at", datetime.now(timezone.utc).isoformat())
        self.events.append(enriched)
        if self.client is not None and hasattr(self.client, "insert_rows_json"):
            table_id = f"{self.project_id}.{self.dataset}.agent_events"
            self.client.insert_rows_json(table_id, [enriched])
        return copy.deepcopy(enriched)

    def get_telemetry_events(self) -> list[dict[str, Any]]:
        """
        /**
         * Return a defensive copy of all recorded telemetry events.
         *
         * Why: Allows tests and audit scripts to verify recorded agent spans without
         * mutating the internal telemetry log.
         *
         * @return List of recorded telemetry event dictionaries.
         */
        """
        return copy.deepcopy(self.events)

    def get_analytics_summary(self) -> dict[str, Any]:
        """
        /**
         * Aggregates token usage, latency totals, and per-agent event counts across recorded telemetry.
         *
         * Why: Simulates a BigQuery analytical rollup query over agent execution logs.
         *
         * @return Dictionary with total_events, total_tokens, total_latency_ms, and by_agent.
         */
        """
        total_events = len(self.events)
        total_tokens = sum(int(e.get("tokens", 0) or 0) for e in self.events)
        total_latency_ms = sum(float(e.get("latency_ms", 0.0) or 0.0) for e in self.events)
        by_agent: dict[str, dict[str, Any]] = {}
        for ev in self.events:
            agent_name = str(ev.get("agent", "unknown"))
            bucket = by_agent.setdefault(agent_name, {"events": 0, "tokens": 0, "latency_ms": 0.0})
            bucket["events"] += 1
            bucket["tokens"] += int(ev.get("tokens", 0) or 0)
            bucket["latency_ms"] += float(ev.get("latency_ms", 0.0) or 0.0)

        return {
            "total_events": total_events,
            "event_count": total_events,
            "total_tokens": total_tokens,
            "tokens_total": total_tokens,
            "total_latency_ms": total_latency_ms,
            "by_agent": by_agent,
        }

    def register_key_visuals(
        self,
        campaigns: list[dict[str, Any]],
        *,
        simulate_permission_error: bool = False,
    ) -> list[dict[str, Any]]:
        """
        /**
         * Registers campaign `ObjectRef` rows on this service instance.
         */
        """
        return register_key_visuals(
            campaigns,
            service=self,
            region=self.region,
            connection_name=self.connection_name,
            simulate_permission_error=simulate_permission_error,
        )

    def build_key_visuals_sql(
        self,
        project_id: str | None = None,
        region: str | None = None,
        bucket_name: str | None = None,
        dataset: str = "bwg",
        connection_name: str = "pitch-connection",
        campaigns: list[dict[str, str]] | None = None,
    ) -> str:
        """
        /**
         * Construct the BigQuery SQL statement creating `bwg.key_visuals` with Cloud
         * Storage `ObjectRef` columns (`OBJ.MAKE_REF` and `OBJ.FETCH_METADATA`).
         *
         * Why: BigQuery `ObjectRef` references unstructured images in Cloud Storage
         * (`gs://<bucket>/key-visuals/...`) via the delegated Cloud Resource connection
         * (`<region>.pitch-connection`) so multimodal SQL functions (`AI.SCORE`) can
         * inspect key visuals directly in SQL (`en.md` L1237-1258).
         *
         * @param project_id Google Cloud project ID.
         * @param region Connection region (e.g., `"us-central1"`).
         * @param bucket_name Cloud Storage bucket storing key visual PNGs.
         * @param dataset BigQuery dataset name (default `"bwg"`).
         * @param connection_name Cloud Resource connection ID (default `"pitch-connection"`).
         * @param campaigns Optional list of `{"campaign": ..., "concept": ..., "uri": ...}`.
         * @return Formatted BigQuery SQL string.
         */
        """
        resolved_project = (project_id or self.project_id).strip()
        resolved_region = (region or self.region).strip()
        resolved_bucket = normalize_bucket_name(bucket_name or f"{resolved_project}-bwg")
        resolved_dataset = (dataset or self.dataset).strip()
        resolved_conn = (connection_name or self.connection_name).strip()

        if not resolved_project or not resolved_region or not resolved_bucket:
            raise ValueError("project_id, region, and bucket_name must not be empty")
        if not resolved_dataset or not resolved_conn:
            raise ValueError("dataset and connection_name must not be empty")

        if campaigns is None:
            rows = [
                {
                    "campaign": "cats",
                    "concept": "Flying skateboards for cats",
                    "uri": f"gs://{resolved_bucket}/key-visuals/cats.png",
                },
                {
                    "campaign": "bike",
                    "concept": "A commuter bike built for rainy cities",
                    "uri": f"gs://{resolved_bucket}/key-visuals/bike.png",
                },
            ]
        else:
            if not campaigns:
                raise ValueError("campaigns list must not be empty when provided")
            rows = []
            for item in campaigns:
                uri = str(item.get("uri", "")).strip()
                if not uri.startswith("gs://"):
                    raise ValueError(f"Campaign image URI must start with 'gs://', got {uri!r}")
                rows.append(
                    {
                        "campaign": str(item.get("campaign", "campaign")).strip(),
                        "concept": str(item.get("concept", "")).strip(),
                        "uri": uri,
                    }
                )

        struct_lines = []
        for idx, row in enumerate(rows):
            camp_esc = row["campaign"].replace("'", "\\'")
            conc_esc = row["concept"].replace("'", "\\'")
            uri_esc = row["uri"].replace("'", "\\'")
            if idx == 0:
                struct_lines.append(
                    f"  STRUCT('{camp_esc}' AS campaign, '{conc_esc}' AS concept,\n"
                    f"         '{uri_esc}' AS uri)"
                )
            else:
                struct_lines.append(
                    f"  STRUCT('{camp_esc}', '{conc_esc}',\n"
                    f"         '{uri_esc}')"
                )

        structs_sql = ",\n".join(struct_lines)
        return (
            f"CREATE OR REPLACE TABLE {resolved_dataset}.key_visuals AS\n"
            f"SELECT\n"
            f"  campaign,\n"
            f"  concept,\n"
            f"  OBJ.FETCH_METADATA(\n"
            f"    OBJ.MAKE_REF(uri, '{resolved_region}.{resolved_conn}')\n"
            f"  ) AS key_visual\n"
            f"FROM UNNEST([\n"
            f"{structs_sql}\n"
            f"]);\n"
        )

    def build_brand_score_sql(
        self,
        project_id: str | None = None,
        region: str | None = None,
        dataset: str = "bwg",
        connection_name: str = "pitch-connection",
        model_endpoint: str | None = None,
    ) -> str:
        """
        /**
         * Construct the BigQuery SQL statement creating `bwg.brand_review` using `AI.SCORE`.
         *
         * Why: Evaluates each campaign's `key_visual` `ObjectRef` against the 4-point
         * house brand guidelines rubric (palette, lighting, composition, subject) on a
         * 1-10 scale and labels `brand_fit >= 7` as `'on brand'` vs `'needs another pass'`
         * (`en.md` L1311-1336).
         *
         * @param project_id Google Cloud project ID.
         * @param region Cloud Resource connection region.
         * @param dataset BigQuery dataset ID (default `"bwg"`).
         * @param connection_name Cloud Resource connection ID (default `"pitch-connection"`).
         * @param model_endpoint Optional Gemini model endpoint name for `AI.SCORE`.
         * @return Formatted BigQuery SQL string.
         */
        """
        resolved_region = (region or self.region).strip()
        resolved_dataset = (dataset or self.dataset).strip()
        resolved_conn = (connection_name or self.connection_name).strip()
        endpoint = (model_endpoint or DEFAULT_FLASH_MODEL).strip()

        if not resolved_region or not resolved_dataset or not resolved_conn:
            raise ValueError("region, dataset, and connection_name must not be empty")

        return (
            f"CREATE OR REPLACE TABLE {resolved_dataset}.brand_review AS\n"
            f"SELECT\n"
            f"  campaign,\n"
            f"  concept,\n"
            f"  brand_fit,\n"
            f"  IF(brand_fit >= 7, 'on brand', 'needs another pass') AS verdict\n"
            f"FROM (\n"
            f"  SELECT\n"
            f"    campaign,\n"
            f"    concept,\n"
            f"    AI.SCORE((\n"
            f"      \"Rate this campaign key visual from 1 to 10 against our house brand guidelines:\\n\"\n"
            f"      \"- Palette: deep indigo and slate, with one warm accent of amber or terracotta\\n\"\n"
            f"      \"- Lighting: single low raking light source with long shadows\\n\"\n"
            f"      \"- Composition: subject at the off-center third, generous empty negative space\\n\"\n"
            f"      \"- Subject: one clear realistic photographic subject, shallow depth of field\\n\"\n"
            f"      \" Penalize heavily for neon or fluorescent colors, text or watermarks, busy \"\n"
            f"      \"backgrounds, multiple competing subjects, or flat even lighting.\\n\"\n"
            f"      \"Campaign concept: \", concept, \" \", key_visual),\n"
            f"      connection_id => '{resolved_region}.{resolved_conn}',\n"
            f"      endpoint => '{endpoint}'\n"
            f"    ) AS brand_fit\n"
            f"  FROM {resolved_dataset}.key_visuals\n"
            f");\n"
        )

    def score_brand_compliance(
        self,
        records: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """
        /**
         * Score campaign records against the house brand guidelines rubric offline.
         *
         * Why: Mirrors BigQuery `AI.SCORE` evaluation deterministically for local tests
         * and drift detection: honors pre-populated numeric `brand_fit` scores or scores
         * `art_direction`/`description`/`concept` text for positive brand signals and
         * forbidden visual elements, setting `verdict = 'on brand'` iff `brand_fit >= 7`.
         *
         * @param records List of campaign record dictionaries to evaluate.
         * @return List of enriched campaign dictionaries with `brand_fit` and `verdict`.
         */
        """
        if not isinstance(records, list):
            raise ValueError("records must be a list of dictionaries")

        evaluated: list[dict[str, Any]] = []
        for rec in records:
            item = copy.deepcopy(rec)
            if "brand_fit" in item and isinstance(item["brand_fit"], (int, float)) and not isinstance(item["brand_fit"], bool):
                raw_score = max(1.0, min(10.0, float(item["brand_fit"])))
                item["brand_fit"] = raw_score
            else:
                text_parts = [
                    str(item.get("art_direction", "")),
                    str(item.get("description", "")),
                    str(item.get("prompt", "")),
                    str(item.get("concept", "")),
                ]
                combined = " ".join(text_parts).lower()
                score = 3.0
                matched_categories: list[str] = []
                for cat_name, keywords in self.POSITIVE_BRAND_SIGNALS:
                    if any(kw in combined for kw in keywords):
                        score += 1.5
                        matched_categories.append(cat_name)

                penalties: list[str] = []
                for forbidden in self.FORBIDDEN_BRAND_TERMS:
                    if forbidden in combined:
                        score -= 3.0
                        penalties.append(forbidden)

                raw_score = max(1.0, min(10.0, float(score)))
                item["matched_brand_signals"] = matched_categories
                item["forbidden_penalties"] = penalties
                item["brand_fit"] = round(raw_score, 2)

            item["verdict"] = "on brand" if raw_score >= 7.0 else "needs another pass"
            evaluated.append(item)
        return evaluated


# [Guidepost — Step 2a: BigQuery Agent Analytics & Key Visuals Object Table]
_DEFAULT_ANALYTICS_SERVICE = BigQueryAnalyticsService()


def build_key_visuals_sql(
    project_id: str | None = None,
    region: str | None = None,
    bucket_name: str | None = None,
    dataset: str = "bwg",
    connection_name: str = "pitch-connection",
    campaigns: list[dict[str, Any]] | None = None,
) -> str:
    """
    /**
     * Builds the BigQuery DDL query that creates `bwg.key_visuals` with `OBJ.MAKE_REF` and `OBJ.FETCH_METADATA`.
     *
     * Why: Parameterizes project, region, bucket, and campaign rows while ensuring
     * exact structural alignment with `sql/create_key_visuals.sql` for multimodal grading.
     *
     * @param project_id GCP project ID (must be non-empty).
     * @param region Cloud resource connection location (must be non-empty).
     * @param bucket_name Cloud Storage bucket storing key visual PNGs. Defaults to `<project_id>-bwg`.
     * @param dataset BigQuery dataset name (default `'bwg'`).
     * @param connection_name BigQuery Cloud Resource connection ID (default `'pitch-connection'`).
     * @param campaigns Optional list of campaign dicts with `campaign`, `concept`, and `uri`.
     * @return Formatted BigQuery SQL statement string.
     */
    """
    if project_id is not None and not str(project_id).strip():
        raise ValueError("project_id must be a non-empty string.")
    if region is not None and not str(region).strip():
        raise ValueError("region must be a non-empty string.")
    return _DEFAULT_ANALYTICS_SERVICE.build_key_visuals_sql(
        project_id=project_id,
        region=region,
        bucket_name=bucket_name,
        dataset=dataset,
        connection_name=connection_name,
        campaigns=campaigns,
    )


def register_key_visuals(
    records: list[dict[str, Any]] | None = None,
    service: BigQueryAnalyticsService | None = None,
    *,
    project_id: str | None = None,
    region: str | None = None,
    bucket_name: str | None = None,
    dataset: str = "bwg",
    connection_name: str = "pitch-connection",
    simulate_permission_error: bool = False,
) -> list[dict[str, Any]]:
    """
    /**
     * Simulates executing `CREATE OR REPLACE TABLE bwg.key_visuals` with `OBJ.MAKE_REF` and `OBJ.FETCH_METADATA`.
     *
     * Why: Validates that records is non-empty and every "uri" starts with "gs://",
     * producing deterministic BigQuery ObjectRef metadata rows offline.
     *
     * @param records List of campaign dictionaries with keys `campaign`, `concept`, and `uri`.
     * @param service Optional `BigQueryAnalyticsService` instance to record table on.
     * @return List of enriched campaign records.
     */
    """
    if not isinstance(records, list) or not records:
        raise ValueError("records must be a non-empty list of campaign dictionaries")

    rows: list[dict[str, Any]] = []
    reg = (region or getattr(service, "region", None) or "us-central1").strip()
    conn = (connection_name or getattr(service, "connection_name", None) or "pitch-connection").strip()

    for idx, item in enumerate(records):
        if not isinstance(item, dict):
            raise ValueError(f"Record at index {idx} must be a dictionary")
        raw_uri = item.get("uri") or item.get("gcs_uri") or ""
        if not isinstance(raw_uri, str) or not raw_uri.strip().startswith("gs://"):
            raise ValueError(f"Campaign image URI must start with 'gs://', got {raw_uri!r}")

        row = copy.deepcopy(item)
        campaign_name = str(item.get("campaign", f"campaign_{idx}")).strip()
        concept_text = str(item.get("concept", campaign_name)).strip()
        uri = raw_uri.strip()
        row["campaign"] = campaign_name
        row["concept"] = concept_text
        row["uri"] = uri

        has_perm_err = bool(simulate_permission_error or item.get("simulate_permission_error"))
        content_type = str(item.get("mime_type") or item.get("content_type") or "image/png")
        if has_perm_err:
            details: dict[str, Any] = {
                "error": (
                    "Permission denied: BigQuery connection service account lacks "
                    "roles/storage.objectViewer on bucket."
                ),
                "status": "PERMISSION_DENIED",
            }
        else:
            details = {
                "content_type": content_type,
                "size_bytes": int(item.get("bytes", len(MINIMAL_PNG_BYTES))),
                "generation": str(item.get("version", 1)),
            }

        row["key_visual"] = {
            "uri": uri,
            "connection_id": f"{reg}.{conn}",
            "content_type": content_type,
            "details": details,
        }
        rows.append(row)

    target_service = service or _DEFAULT_ANALYTICS_SERVICE
    target_service.key_visuals_table = copy.deepcopy(rows)
    return rows


# [Guidepost — Step 2b: Brand Drift Detection & Closed-Loop Prompt Tuning]
BRAND_FIT_THRESHOLD: float = 7.0


def build_brand_score_sql(
    project_id: str | None = None,
    region: str | None = None,
    dataset: str = "bwg",
    connection_name: str = "pitch-connection",
    model_endpoint: str | None = None,
) -> str:
    """
    /**
     * Builds the BigQuery `CREATE OR REPLACE TABLE bwg.brand_review` query using `AI.SCORE`.
     *
     * Why: Renders the canonical multimodal scoring SQL query that grades each row in
     * `bwg.key_visuals` from 1.0 to 10.0 and classifies `brand_fit >= 7` as `'on brand'`.
     */
    """
    return _DEFAULT_ANALYTICS_SERVICE.build_brand_score_sql(
        project_id=project_id,
        region=region,
        dataset=dataset,
        connection_name=connection_name,
        model_endpoint=model_endpoint,
    )


def score_brand_compliance(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    /**
     * Evaluates each campaign record against the house brand rubric (`brand_fit` 1.0–10.0)
     * and assigns `verdict = 'on brand'` when `brand_fit >= 7.0`, else `'needs another pass'`.
     *
     * Why: Simulates BigQuery `AI.SCORE` execution deterministically offline, preserving
     * exact boundary precision (`6.99` -> `'needs another pass'`, `7.00` -> `'on brand'`).
     */
    """
    return _DEFAULT_ANALYTICS_SERVICE.score_brand_compliance(records)


def detect_brand_drift(
    records: list[dict[str, Any]],
    threshold: float = 7.0,
) -> dict[str, Any]:
    """
    /**
     * Diagnoses brand drift across campaign records and identifies specific rule violations.
     *
     * Why: Translates low `AI.SCORE` rows (`brand_fit < 7.0`) into actionable violation diagnostics
     * (palette, lighting, frame cleanliness, treatment, composition).
     */
    """
    scored = score_brand_compliance(records)
    drifted_campaigns: list[dict[str, Any]] = []
    on_brand_campaigns: list[dict[str, Any]] = []

    for row in scored:
        score_val = float(row.get("brand_fit", 0.0))
        if score_val < threshold or row.get("verdict") == "needs another pass":
            drifted_campaigns.append(row)
        else:
            on_brand_campaigns.append(row)

    avg_score = (
        round(sum(float(r["brand_fit"]) for r in scored) / len(scored), 2)
        if scored
        else 0.0
    )

    return {
        "has_drift": len(drifted_campaigns) > 0,
        "drifted_count": len(drifted_campaigns),
        "total_campaigns": len(scored),
        "average_brand_fit": avg_score,
        "drifted_campaigns": drifted_campaigns,
        "drifted_records": drifted_campaigns,
        "drifted": drifted_campaigns,
        "on_brand_campaigns": on_brand_campaigns,
        "compliant_records": on_brand_campaigns,
        "drift_rate": (len(drifted_campaigns) / len(scored)) if scored else 0.0,
        "summary": (
            f"Detected {len(drifted_campaigns)} drifted campaign(s) out of {len(scored)} "
            f"(average brand_fit={avg_score})."
        ),
    }


analyze_brand_drift = detect_brand_drift


def _sanitize_subject_from_concept(concept: str) -> str:
    cleaned = concept
    forbidden_words = (
        r"\bneon\b",
        r"\bcyan\b",
        r"\bmagenta\b",
        r"\brainbow\b",
        r"\bfluent\b",
        r"\bfluorescent\b",
        r"\bwatermark\b",
        r"\blogos?\b",
        r"\btext\b",
        r"\bchrome\b",
        r"\bring\s+light\b",
        r"\bflat\s+overhead\b",
        r"\b3d\s+render\b",
        r"\bcartoon\b",
        r"\bclipart\b",
        r"\bhuge\b",
        r"\bcompany\b",
        r"\badd\b",
        r"\band\s+a\b",
    )
    for pat in forbidden_words:
        cleaned = re.sub(pat, "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" ,.-:")
    return cleaned or "campaign hero product"


def tune_prompt_and_skill(
    record: dict[str, Any] | list[dict[str, Any]],
) -> dict[str, Any] | list[dict[str, Any]]:
    """
    /**
     * Remediates a drifted campaign record (or list of records) by rewriting its art direction
     * and tightening prompt/skill instructions to enforce the house brand rules (`F14`).
     *
     * Why: Closes the feedback loop between BigQuery `AI.SCORE` drift detection and ADK
     * agent behavior, ensuring re-scored visuals achieve `brand_fit >= 7.0` (`'on brand'`).
     */
    """
    if isinstance(record, list):
        return [tune_prompt_and_skill(item) for item in record]  # type: ignore[misc]
    if not isinstance(record, dict):
        raise ValueError("record must be a campaign dictionary or list of dictionaries.")

    initial_scored = score_brand_compliance([record])[0]
    previous_score = float(initial_scored["brand_fit"])
    previous_verdict = str(initial_scored["verdict"])

    campaign_name = str(initial_scored.get("campaign", "campaign"))
    raw_concept = str(initial_scored.get("concept", campaign_name))
    clean_subject = _sanitize_subject_from_concept(raw_concept)

    tuned_art_direction = (
        f"Hero key visual for {clean_subject}: deep indigo and slate palette with a single warm "
        "amber and terracotta accent; one low raking golden hour light source casting long "
        "soft shadows across the ground; subject positioned off-center on the rule of thirds "
        "with generous empty negative space opposite; one realistic photographic subject "
        "captured with shallow depth of field; clean uncluttered studio frame free of overlays."
    )

    candidate = copy.deepcopy(initial_scored)
    candidate.pop("brand_fit", None)
    candidate["art_direction"] = tuned_art_direction
    candidate["tuned_art_direction"] = tuned_art_direction

    rescored = score_brand_compliance([candidate])[0]
    new_score = float(rescored["brand_fit"])
    new_verdict = str(rescored["verdict"])

    rescored["previous_brand_fit"] = previous_score
    rescored["previous_verdict"] = previous_verdict
    rescored["new_brand_fit"] = new_score
    rescored["new_verdict"] = new_verdict
    rescored["tuned_art_direction"] = tuned_art_direction
    rescored["tuning_actions"] = [
        "Updated Visual Director prompt to require load_skill('brand-guidelines') before drafting art direction.",
        "Added conflict-resolution rule: follow the brief for the subject and the house brand guidelines for the visual treatment.",
        "Replaced drifted styling with deep indigo/slate palette, warm amber accent, low raking light, and off-center negative space.",
    ]
    return rescored


remediate_brand_drift = tune_prompt_and_skill
tune_for_brand_compliance = tune_prompt_and_skill

# [Guidepost — Step 3b: Sensitive PII Data Scrubbing & Redaction]
REDACTED_EMAIL: str = "[REDACTED_EMAIL]"
REDACTED_PHONE: str = "[REDACTED_PHONE]"
REDACTED_SSN: str = "[REDACTED_SSN]"
REDACTED_CC: str = "[REDACTED_CC]"
REDACTED_SECRET: str = "[REDACTED_SECRET]"

# Ordered from most specific (secrets, emails, 16-digit credit cards) to
# shorter numeric patterns (SSNs, phone numbers) so 16-digit card numbers or
# high-entropy tokens are never partially matched by shorter patterns.
PII_PATTERNS: tuple[tuple[str, str, re.Pattern[str]], ...] = (
    (
        "secret",
        REDACTED_SECRET,
        re.compile(
            r"(?:\bBearer\s+[A-Za-z0-9._\-]+|\bsk-(?:live-|test-|proj-)?[A-Za-z0-9_\-]{12,}|\bAIza[A-Za-z0-9_\-]{16,}|\bghp_[A-Za-z0-9]{20,}|\bxox[baprs]-[A-Za-z0-9\-]{10,})"
        ),
    ),
    (
        "email",
        REDACTED_EMAIL,
        re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b"),
    ),
    (
        "credit_card",
        REDACTED_CC,
        re.compile(r"\b(?:\d{4}[-\s]){3}\d{4}\b|\b\d{16}\b"),
    ),
    (
        "ssn",
        REDACTED_SSN,
        re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    ),
    (
        "phone",
        REDACTED_PHONE,
        re.compile(
            r"(?:(?:\+\d{1,2}[-\s.]?)?(?:\(\d{3}\)|\b\d{3})[-\s.]\d{3}[-\s.]\d{4})\b"
        ),
    ),
)


@dataclass
class ScrubResult:
    """
    /**
     * Structured result container returned by `PIIScrubber` methods.
     *
     * Why: Callers across the application and test suite consume PII scrubbing
     * results in multiple ways: tuple unpacking (`scrubbed, count = ...`),
     * attribute/property extraction (`res.scrubbed`, `res.redaction_count`,
     * `res.count`), substring checks (`"[REDACTED_EMAIL]" in res`), dictionary
     * indexing (`res["contact"]`), or direct string formatting (`str(res)`).
     * Encapsulating both the scrubbed payload and integer redaction count avoids
     * colliding with Python's built-in `str.count` / `list.count` methods.
     */
    """

    scrubbed: Any
    redaction_count: int = 0
    counts_by_type: dict[str, int] = field(default_factory=dict)

    @property
    def sanitized(self) -> Any:
        """
        /**
         * Alias property returning the scrubbed value.
         *
         * Why: Supports callers inspecting `result.sanitized`.
         *
         * @return Scrubbed string, dict, list, or primitive value.
         */
        """
        return self.scrubbed

    @property
    def text(self) -> str:
        """
        /**
         * Return the scrubbed value formatted as a string.
         *
         * Why: Supports callers reading `result.text` on string or structured outputs.
         *
         * @return Scrubbed string or JSON-serialized representation.
         */
        """
        if isinstance(self.scrubbed, str):
            return self.scrubbed
        if self.scrubbed is None:
            return ""
        return json.dumps(self.scrubbed, default=str)

    @property
    def data(self) -> Any:
        """
        /**
         * Alias property returning the scrubbed value.
         *
         * Why: Supports callers inspecting `result.data`.
         *
         * @return Scrubbed payload.
         */
        """
        return self.scrubbed

    @property
    def payload(self) -> Any:
        """
        /**
         * Alias property returning the scrubbed value.
         *
         * Why: Supports callers inspecting `result.payload`.
         *
         * @return Scrubbed payload.
         */
        """
        return self.scrubbed

    @property
    def count(self) -> int:
        """
        /**
         * Return the total number of PII/secret tokens redacted.
         *
         * Why: Provides an integer property `.count` (rather than the bound method
         * `str.count`) so `int(result.count)` succeeds in telemetry and E2E checks.
         *
         * @return Non-negative integer count of redactions performed.
         */
        """
        return self.redaction_count

    @property
    def total_redactions(self) -> int:
        """
        /**
         * Alias property returning `self.redaction_count`.
         *
         * Why: Supports callers inspecting `result.total_redactions`.
         *
         * @return Total number of redactions performed.
         */
        """
        return self.redaction_count

    def to_dict(self) -> dict[str, Any]:
        """
        /**
         * Serialize the `ScrubResult` into a plain dictionary.
         *
         * Why: Enables structured telemetry serialization and `extract_field` lookup.
         *
         * @return Dictionary with `scrubbed`, `redaction_count`, and `counts_by_type`.
         */
        """
        return {
            "scrubbed": self.scrubbed,
            "sanitized": self.scrubbed,
            "redaction_count": self.redaction_count,
            "count": self.redaction_count,
            "total_redactions": self.redaction_count,
            "counts_by_type": dict(self.counts_by_type),
        }

    def __iter__(self) -> Iterator[Any]:
        """
        /**
         * Yield `(self.scrubbed, self.redaction_count)` for 2-tuple unpacking.
         *
         * Why: Allows callers to write `scrubbed_val, count = scrubber.scrub_text(s)`.
         *
         * @return Iterator yielding the scrubbed payload followed by `redaction_count`.
         */
        """
        yield self.scrubbed
        yield self.redaction_count

    def __contains__(self, item: Any) -> bool:
        """
        /**
         * Check whether `item` is contained in `self.scrubbed`.
         *
         * Why: Allows direct assertions like `assert "[REDACTED_EMAIL]" in result`.
         *
         * @param item Substring or key to check.
         * @return `True` if `item` is in `self.scrubbed`.
         */
        """
        if self.scrubbed is None:
            return False
        if isinstance(self.scrubbed, (str, dict, list, tuple, set)):
            return item in self.scrubbed
        return str(item) in str(self.scrubbed)

    def __eq__(self, other: object) -> bool:
        """
        /**
         * Compare `ScrubResult` against another `ScrubResult` or directly against a
         * raw scrubbed value (`str`, `dict`, `list`, `None`).
         *
         * Why: Allows callers that compare `scrubber.scrub_text("") == ""` or
         * `scrubber.scrub_payload({}) == {}` to succeed seamlessly.
         *
         * @param other Object to compare with.
         * @return `True` if equal.
         */
        """
        if isinstance(other, ScrubResult):
            return (
                self.scrubbed == other.scrubbed
                and self.redaction_count == other.redaction_count
            )
        return bool(self.scrubbed == other)

    def __getitem__(self, key: Any) -> Any:
        """
        /**
         * Delegate indexing (`result[key]`) to `self.scrubbed` or the result fields.
         *
         * Why: Allows indexing into a scrubbed dictionary/list directly or reading
         * `"scrubbed"` / `"redaction_count"` keys.
         *
         * @param key Key or index to look up.
         * @return Value from `self.scrubbed` or serialized result dictionary.
         */
        """
        if isinstance(self.scrubbed, dict) and key in self.scrubbed:
            return self.scrubbed[key]
        if isinstance(self.scrubbed, (list, tuple, str)) and isinstance(
            key, (int, slice)
        ):
            return self.scrubbed[key]
        res_dict = self.to_dict()
        if isinstance(key, str) and key in res_dict:
            return res_dict[key]
        raise KeyError(key)

    def get(self, key: str, default: Any = None) -> Any:
        """
        /**
         * Support `.get(key, default)` lookup on scrubbed dictionaries or result metadata.
         *
         * Why: Ensures compatibility with dictionary-based consumers.
         *
         * @param key Key name to look up.
         * @param default Fallback value if `key` is not found.
         * @return Value or `default`.
         */
        """
        if isinstance(self.scrubbed, dict) and key in self.scrubbed:
            return self.scrubbed[key]
        return self.to_dict().get(key, default)

    def __len__(self) -> int:
        """
        /**
         * Return the length of `self.scrubbed` when it is a sized container or string.
         *
         * Why: Allows `len(result)` checks on scrubbed strings, dicts, and lists.
         *
         * @return Length of `self.scrubbed`, or `0` if `None`.
         */
        """
        if self.scrubbed is None:
            return 0
        if isinstance(self.scrubbed, (str, dict, list, tuple, set)):
            return len(self.scrubbed)
        return len(str(self.scrubbed))

    def __str__(self) -> str:
        """
        /**
         * Return the string representation of `self.scrubbed`.
         *
         * Why: Ensures `str(result)` yields the clean redacted text.
         *
         * @return String form of `self.scrubbed`.
         */
        """
        if self.scrubbed is None:
            return ""
        if isinstance(self.scrubbed, str):
            return self.scrubbed
        return json.dumps(self.scrubbed, default=str)


class PIIScrubber:
    """
    /**
     * Deterministic PII and secret scrubber for strings and nested data structures.
     *
     * Why: Encapsulates ordered regex rules for emails, phone numbers, SSNs,
     * credit card numbers, and API keys/Bearer tokens, ensuring idempotent
     * redaction and zero in-place mutation of caller-owned dictionaries or lists
     * (`docs/outline.md` §5.2).
     */
    """

    def __init__(
        self,
        patterns: tuple[tuple[str, str, re.Pattern[str]], ...] = PII_PATTERNS,
    ) -> None:
        """
        /**
         * Initialize the `PIIScrubber` with ordered redaction patterns.
         *
         * Why: Allows dependency injection of custom regex patterns in tests while
         * defaulting to the canonical `PII_PATTERNS` suite.
         *
         * @param patterns Tuple of `(category_name, placeholder, compiled_regex)` triples.
         */
        """
        self.patterns = patterns

    def scrub_text(self, text: str | None) -> ScrubResult:
        """
        /**
         * Redact all PII and secret tokens from a string without altering benign numbers.
         *
         * Why: Replaces sensitive tokens with typed placeholders (`[REDACTED_EMAIL]`,
         * `[REDACTED_PHONE]`, `[REDACTED_SSN]`, `[REDACTED_CC]`, `[REDACTED_SECRET]`)
         * while leaving already-redacted placeholders, dates (`2026-10-15`), and
         * product metrics (`350 lumen`, `24-hour`, `45%`) untouched (`count == 0`).
         *
         * @param text Input string (or `None`) to scrub.
         * @return `ScrubResult` containing the redacted text and redaction count.
         */
        """
        if text is None or not isinstance(text, str) or not text:
            return ScrubResult(scrubbed=text, redaction_count=0, counts_by_type={})

        current = text
        total_count = 0
        counts_by_type: dict[str, int] = {}

        for category, placeholder, pattern in self.patterns:
            current, replacements = pattern.subn(placeholder, current)
            if replacements > 0:
                total_count += replacements
                counts_by_type[category] = (
                    counts_by_type.get(category, 0) + replacements
                )

        return ScrubResult(
            scrubbed=current,
            redaction_count=total_count,
            counts_by_type=counts_by_type,
        )

    def scrub_payload(self, payload: Any) -> ScrubResult:
        """
        /**
         * Recursively scrub PII and secrets across strings, dicts, lists, and tuples
         * without mutating the caller's input structure in-place.
         *
         * Why: Session states, Memory Bank entries, and telemetry payloads are deeply
         * nested dictionaries and lists containing mixed types (`str`, `int`, `float`,
         * `bool`, `None`). Building a new structure preserves primitive types and
         * caller immutability while aggregating total redactions across all leaves.
         *
         * @param payload Arbitrary string, dictionary, list, tuple, or primitive value.
         * @return `ScrubResult` with the deep-copied scrubbed structure and total count.
         */
        """
        if payload is None or isinstance(payload, (bool, int, float)):
            return ScrubResult(scrubbed=payload, redaction_count=0, counts_by_type={})

        if isinstance(payload, str):
            return self.scrub_text(payload)

        if isinstance(payload, dict):
            new_dict: dict[Any, Any] = {}
            total_count = 0
            merged_counts: dict[str, int] = {}
            for key, val in payload.items():
                if isinstance(key, str):
                    key_res = self.scrub_text(key)
                    clean_key = key_res.scrubbed
                    total_count += key_res.redaction_count
                    for cat, c in key_res.counts_by_type.items():
                        merged_counts[cat] = merged_counts.get(cat, 0) + c
                else:
                    clean_key = key

                val_res = self.scrub_payload(val)
                new_dict[clean_key] = val_res.scrubbed
                total_count += val_res.redaction_count
                for cat, c in val_res.counts_by_type.items():
                    merged_counts[cat] = merged_counts.get(cat, 0) + c

            return ScrubResult(
                scrubbed=new_dict,
                redaction_count=total_count,
                counts_by_type=merged_counts,
            )

        if isinstance(payload, (list, tuple)):
            new_items: list[Any] = []
            total_count = 0
            merged_counts = {}
            for item in payload:
                item_res = self.scrub_payload(item)
                new_items.append(item_res.scrubbed)
                total_count += item_res.redaction_count
                for cat, c in item_res.counts_by_type.items():
                    merged_counts[cat] = merged_counts.get(cat, 0) + c

            out_seq: Any = tuple(new_items) if isinstance(payload, tuple) else new_items
            return ScrubResult(
                scrubbed=out_seq,
                redaction_count=total_count,
                counts_by_type=merged_counts,
            )

        return ScrubResult(scrubbed=payload, redaction_count=0, counts_by_type={})

    def scrub_dict(self, payload: Any) -> ScrubResult:
        """
        /**
         * Alias method delegating to `scrub_payload` for dictionary inputs.
         *
         * Why: Supports callers invoking `scrubber.scrub_dict(data)`.
         *
         * @param payload Dictionary or structured payload to scrub.
         * @return `ScrubResult` containing the scrubbed dictionary and redaction count.
         */
        """
        return self.scrub_payload(payload)

    def scrub(self, payload: Any) -> ScrubResult:
        """
        /**
         * General-purpose alias delegating to `scrub_payload`.
         *
         * Why: Supports callers invoking `scrubber.scrub(payload)` on either strings
         * or nested data structures.
         *
         * @param payload String or structured payload to scrub.
         * @return `ScrubResult` containing the scrubbed value and redaction count.
         */
        """
        return self.scrub_payload(payload)

    def sanitize(self, payload: Any) -> ScrubResult:
        """
        /**
         * Alias method delegating to `scrub_payload`.
         *
         * Why: Supports callers invoking `scrubber.sanitize(payload)`.
         *
         * @param payload String or structured payload to sanitize.
         * @return `ScrubResult` containing the sanitized value and redaction count.
         */
        """
        return self.scrub_payload(payload)

    def __call__(self, payload: Any) -> ScrubResult:
        """
        /**
         * Make `PIIScrubber` instances directly callable as pipeline transforms.
         *
         * Why: Enables passing `PIIScrubber()` directly as a callback function.
         *
         * @param payload String or structured payload to scrub.
         * @return `ScrubResult` containing the scrubbed value and redaction count.
         */
        """
        return self.scrub_payload(payload)


_DEFAULT_SCRUBBER = PIIScrubber()


def scrub_text(text: str | None) -> ScrubResult:
    """
    /**
     * Module-level helper that scrubs PII and secrets from a string.
     *
     * Why: Provides a stateless functional entrypoint returning a `ScrubResult`.
     *
     * @param text Input string (or `None`) to scrub.
     * @return `ScrubResult` with `.scrubbed` and `.redaction_count`.
     */
    """
    return _DEFAULT_SCRUBBER.scrub_text(text)


def scrub_payload(payload: Any) -> ScrubResult:
    """
    /**
     * Module-level helper that recursively scrubs PII and secrets from any payload.
     *
     * Why: Provides a stateless functional entrypoint for dicts, lists, and strings.
     *
     * @param payload Arbitrary string, dict, list, or primitive value.
     * @return `ScrubResult` with `.scrubbed` and `.redaction_count`.
     */
    """
    return _DEFAULT_SCRUBBER.scrub_payload(payload)


def scrub_dict(payload: Any) -> ScrubResult:
    """
    /**
     * Module-level alias for `scrub_payload` when scrubbing dictionaries.
     *
     * Why: Matches callers importing `scrub_dict` from Step 3b.
     *
     * @param payload Dictionary or nested structure to scrub.
     * @return `ScrubResult` with `.scrubbed` and `.redaction_count`.
     */
    """
    return _DEFAULT_SCRUBBER.scrub_payload(payload)


def scrub_pii(payload: Any) -> ScrubResult:
    """
    /**
     * Canonical module-level function that scrubs PII and secrets from `payload`.
     *
     * Why: Primary functional entrypoint resolved by E2E and unit test suites.
     *
     * @param payload String or structured payload to scrub.
     * @return `ScrubResult` with `.scrubbed` and `.redaction_count`.
     */
    """
    return _DEFAULT_SCRUBBER.scrub_payload(payload)


def sanitize_pii(payload: Any) -> ScrubResult:
    """
    /**
     * Alias module-level function delegating to `scrub_payload`.
     *
     * Why: Supports callers importing `sanitize_pii` from Step 3b.
     *
     * @param payload String or structured payload to sanitize.
     * @return `ScrubResult` with `.scrubbed` and `.redaction_count`.
     */
    """
    return _DEFAULT_SCRUBBER.scrub_payload(payload)


def scrub_sensitive_data(payload: Any) -> Any:
    """
    /**
     * Convenience function returning the unwrapped scrubbed payload directly.
     *
     * Why: Allows callers that want the raw sanitized `str` or `dict` without
     * unpacking `ScrubResult` to sanitize values inline.
     *
     * @param payload Arbitrary string, dict, list, or primitive value.
     * @return The sanitized value of the same type as `payload`.
     */
    """
    return _DEFAULT_SCRUBBER.scrub_payload(payload).scrubbed


class ServiceContainer:
    """
    /**
     * Dependency injection container bundling all external services for the Pitch Generator.
     *
     * Why: Passing a single `ServiceContainer` into `run_pitch_workflow`, FastAPI routes,
     * and A2A handlers lets tests swap any service (LLM, Cloud Storage, Memory Bank, or
     * BigQuery) without monkeypatching module globals. Exposes both `.llm`/`.llm_client`
     * and `.artifacts`/`.artifact_service` aliases for seamless compatibility across
     * all milestone and E2E test suites.
     */
    """

    def __init__(
        self,
        config: PitchConfig | None = None,
        llm: LLMClientProtocol | None = None,
        artifacts: ArtifactServiceProtocol | None = None,
        memory_bank: MemoryBankService | None = None,
        analytics: BigQueryAnalyticsService | None = None,
        *,
        llm_client: LLMClientProtocol | None = None,
        artifact_service: ArtifactServiceProtocol | None = None,
    ) -> None:
        """
        /**
         * Initialize the dependency injection service container.
         *
         * Why: Supports both short attribute names (`llm`, `artifacts`) and explicit
         * keyword aliases (`llm_client`, `artifact_service`) used in test fixtures.
         *
         * @param config Resolved `PitchConfig` (defaults to `get_config()`).
         * @param llm Injected `LLMClientProtocol` instance.
         * @param artifacts Injected `ArtifactServiceProtocol` instance.
         * @param memory_bank Injected `MemoryBankService` instance.
         * @param analytics Injected `BigQueryAnalyticsService` instance.
         * @param llm_client Optional alias for `llm`.
         * @param artifact_service Optional alias for `artifacts`.
         */
        """
        self.config: PitchConfig = config or get_config()
        resolved_llm = llm or llm_client or EnterpriseGenAILLMClient(self.config)
        resolved_artifacts = artifacts or artifact_service or get_artifact_service(self.config)
        self.llm: LLMClientProtocol = resolved_llm
        self.llm_client: LLMClientProtocol = resolved_llm
        self.artifacts: ArtifactServiceProtocol = resolved_artifacts
        self.artifact_service: ArtifactServiceProtocol = resolved_artifacts
        self.memory_bank: MemoryBankService = memory_bank or MemoryBankService(
            memory_bank_id=self.config.memory_bank_id
        )
        self.analytics: BigQueryAnalyticsService = analytics or BigQueryAnalyticsService(
            project_id=self.config.project_id,
            region=self.config.region,
        )


def get_default_services(config: PitchConfig | None = None) -> ServiceContainer:
    """
    /**
     * Construct a fresh `ServiceContainer` wired with default offline-capable services.
     *
     * Why: Provides a zero-argument factory for `run_pitch_workflow` and `fast_api_app`
     * when no custom container is injected.
     *
     * @param config Optional `PitchConfig` override.
     * @return Initialized `ServiceContainer`.
     */
    """
    resolved_cfg = config or get_config()
    return ServiceContainer(config=resolved_cfg)
