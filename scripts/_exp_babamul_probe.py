#!/usr/bin/env python3
"""Probe Babamul broker access and DP1 + ZTF backfill coverage.

Verifies six things in order, each independent so a partial failure still
gives useful signal:

  1. All 4 BABAMUL_* env vars are set.
  2. REST API auth works (`/profile` whoami).
  3. Bulk cross-match returns hits for our DP1 41K pool — the gating
     question for whether Babamul joins the v7 paper run on DP1.
  4. Kafka SASL auth works (5-second consumer connect on a hosted topic).
  5. ZTF retro-ingest probe — does Babamul cover our 1,920 pre-2026 ZTF
     labels (ZTF17-ZTF23)? Decisive for whether v7 trains on existing ZTF
     cohort vs. waiting on live-LSST accumulation.
  6. Kafka payload schema dump — does AppleCiDEr expose probability
     scores in the message payload (=> SN-filter projector + trust head),
     or only topic membership (=> 4-way categorical feature)?

Run after editing .env:
    python scripts/_exp_babamul_probe.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# Load .env from repo root (matches scripts/crossmatch_tns.py convention)
try:
    from dotenv import load_dotenv

    load_dotenv(REPO_ROOT / ".env")
except ImportError:
    print("WARN  python-dotenv not installed; relying on shell env only")

REQUIRED = [
    "BABAMUL_KAFKA_SERVER",
    "BABAMUL_KAFKA_USERNAME",
    "BABAMUL_KAFKA_PASSWORD",
    "BABAMUL_API_TOKEN",
]

DP1_TRUTH = REPO_ROOT / "data" / "truth" / "dp1_truth_50k.parquet"
PROBE_N = 50  # bulk-XM batch size for the backfill check
KAFKA_TIMEOUT_S = 5.0
HOSTED_TOPIC = "babamul.lsst.no-ztf-match.hosted"


def step1_env() -> bool:
    print("\n[1/4] env vars")
    missing = [k for k in REQUIRED if not os.environ.get(k)]
    if missing:
        print(f"  FAIL  missing: {missing}")
        print(f"  fix:  open {REPO_ROOT}/.env and fill in those values")
        return False
    for k in REQUIRED:
        v = os.environ[k]
        masked = v[:6] + "…" + v[-3:] if len(v) > 12 else "***"
        print(f"  OK    {k} = {masked}")
    return True


def _api_base() -> str:
    env = os.environ.get("BABAMUL_ENV", "production")
    return {
        "production": "https://babamul.caltech.edu/api/babamul",
        "staging": "https://staging.babamul.caltech.edu/api/babamul",
    }.get(env, env)  # allow override with full URL


def step2_rest_whoami() -> bool:
    print("\n[2/4] REST /profile (whoami)")
    try:
        import httpx
    except ImportError:
        print("  SKIP  pip install httpx (or: pip install babamul)")
        return False
    url = f"{_api_base()}/profile"
    headers = {"Authorization": f"Bearer {os.environ['BABAMUL_API_TOKEN']}"}
    try:
        r = httpx.get(url, headers=headers, timeout=15.0)
    except httpx.HTTPError as e:
        print(f"  FAIL  network: {e}")
        return False
    if r.status_code != 200:
        print(f"  FAIL  HTTP {r.status_code}: {r.text[:200]}")
        if r.status_code in (401, 403):
            print("  hint  token rejected — re-check BABAMUL_API_TOKEN value")
        return False
    body = r.json().get("data", r.json())
    print(f"  OK    {body}")
    return True


def step3_dp1_backfill() -> bool:
    print(f"\n[3/4] bulk cross-match on {PROBE_N} DP1 IDs (DP1 backfill check)")
    if not DP1_TRUTH.exists():
        print(f"  SKIP  {DP1_TRUTH} not found — pull from SCC first")
        return False
    try:
        import httpx
        import pandas as pd
    except ImportError as e:
        print(f"  SKIP  missing dep: {e}")
        return False
    df = pd.read_parquet(DP1_TRUTH)
    id_col = next(
        (c for c in ("diaObjectId", "objectId", "object_id") if c in df.columns),
        None,
    )
    if id_col is None:
        print(f"  FAIL  no id column in {DP1_TRUTH.name}; cols={list(df.columns)[:8]}")
        return False
    ids = df[id_col].astype(str).head(PROBE_N).tolist()
    url = f"{_api_base()}/surveys/LSST/objects/cross-matches"
    headers = {"Authorization": f"Bearer {os.environ['BABAMUL_API_TOKEN']}"}
    try:
        r = httpx.post(url, headers=headers, json={"object_ids": ids}, timeout=60.0)
    except httpx.HTTPError as e:
        print(f"  FAIL  network: {e}")
        return False
    if r.status_code != 200:
        print(f"  FAIL  HTTP {r.status_code}: {r.text[:200]}")
        if r.status_code == 404:
            print("  hint  endpoint shape may differ; check api.py in boom-astro/babamul")
        return False
    payload = r.json().get("data", r.json())
    n_payload = len(payload) if isinstance(payload, list) else 0
    n_hit = sum(1 for x in (payload or []) if x)
    pct = 100.0 * n_hit / max(PROBE_N, 1)
    print(f"  OK    {n_hit}/{PROBE_N} ({pct:.0f}%) returned non-empty cross-matches")
    if pct < 5:
        print("  -->   Babamul looks live-stream only; treat as future-work, NOT v7 head")
    elif pct < 50:
        print("  -->   partial DP1 backfill; usable but coverage caveat in paper")
    else:
        print("  -->   DP1 retro-ingested; promote Babamul to v7 head")
    return True


def step4_kafka_connect() -> bool:
    print(f"\n[4/4] Kafka SASL_PLAINTEXT + SCRAM-SHA-512 ({KAFKA_TIMEOUT_S:.0f}s connect)")
    try:
        from confluent_kafka import Consumer, KafkaError
    except ImportError:
        print("  SKIP  pip install confluent-kafka (or: pip install babamul)")
        return False
    conf = {
        "bootstrap.servers": os.environ["BABAMUL_KAFKA_SERVER"],
        "security.protocol": "SASL_PLAINTEXT",
        "sasl.mechanism": "SCRAM-SHA-512",
        "sasl.username": os.environ["BABAMUL_KAFKA_USERNAME"],
        "sasl.password": os.environ["BABAMUL_KAFKA_PASSWORD"],
        "group.id": f"{os.environ['BABAMUL_KAFKA_USERNAME']}-debass-probe",
        "auto.offset.reset": "earliest",
        "enable.auto.commit": False,
        "session.timeout.ms": 10000,
    }
    auth_error = []

    def _on_error(err):
        if err.code() == KafkaError._AUTHENTICATION:
            auth_error.append(str(err))

    conf["error_cb"] = _on_error
    c = Consumer(conf)
    try:
        c.subscribe([HOSTED_TOPIC])
        msg = c.poll(timeout=KAFKA_TIMEOUT_S)
    finally:
        c.close()
    if auth_error:
        print(f"  FAIL  auth: {auth_error[0]}")
        print("  hint  re-check KAFKA_USERNAME / KAFKA_PASSWORD; username starts with 'babamul-'")
        return False
    if msg is None:
        print(f"  OK    connected, no msg in {KAFKA_TIMEOUT_S:.0f}s (broker reachable, auth OK)")
        return True
    if msg.error():
        print(f"  WARN  msg error: {msg.error()}")
        return False
    print(f"  OK    received 1 msg from {msg.topic()} offset={msg.offset()} size={len(msg.value() or b'')}B")
    return True


def step5_ztf_retro_backfill() -> bool:
    """Test whether Babamul retro-ingested historical ZTF archive.

    Decisive question: do our ZTF17-ZTF23 labels return non-empty hits?
    If yes, train Babamul head on existing ZTF cohort (no waiting).
    If only ZTF26+ would return (and we have zero ZTF26+ labels), skip.
    """
    print("\n[5/6] ZTF retro-ingest probe (does Babamul cover ZTF17-ZTF23 labels?)")
    labels_path = REPO_ROOT / "data" / "labels.csv"
    if not labels_path.exists():
        print(f"  SKIP  {labels_path} not found")
        return False
    try:
        import httpx
        import pandas as pd
    except ImportError as e:
        print(f"  SKIP  missing dep: {e}")
        return False
    df = pd.read_csv(labels_path)
    ztf_mask = df["object_id"].astype(str).str.startswith("ZTF")
    ztf_ids = df.loc[ztf_mask, "object_id"].astype(str)
    if ztf_ids.empty:
        print("  SKIP  no ZTF labels in labels.csv")
        return False
    # Stratified sample by year prefix (ZTF{YY}aabcdef → year = 2000+YY)
    years = ztf_ids.str.extract(r"^ZTF(\d{2})")[0].astype(int) + 2000
    per_year = 8
    sample_ids: list[str] = []
    sample_years: list[int] = []
    for year in sorted(years.unique()):
        year_ids = ztf_ids[years == year].head(per_year).tolist()
        sample_ids.extend(year_ids)
        sample_years.extend([year] * len(year_ids))
    sample_ids = sample_ids[:50]
    sample_years = sample_years[: len(sample_ids)]
    print(
        f"  ... sample n={len(sample_ids)} stratified across years "
        f"{sorted(set(sample_years))}"
    )

    headers = {"Authorization": f"Bearer {os.environ['BABAMUL_API_TOKEN']}"}

    # 5a. Bulk cross-match endpoint (mirror of LSST step 3)
    url_xm = f"{_api_base()}/surveys/ZTF/objects/cross-matches"
    n_bulk_hit = 0
    try:
        r = httpx.post(
            url_xm, headers=headers, json={"object_ids": sample_ids}, timeout=60.0
        )
    except httpx.HTTPError as e:
        print(f"  WARN  bulk-xm network: {e}")
        r = None
    if r is not None:
        if r.status_code == 200:
            payload = r.json().get("data", r.json())
            n_bulk_hit = sum(1 for x in (payload or []) if x)
            pct = 100.0 * n_bulk_hit / max(len(sample_ids), 1)
            print(
                f"  bulk-xm: {n_bulk_hit}/{len(sample_ids)} "
                f"({pct:.0f}%) returned non-empty cross-matches"
            )
        else:
            print(f"  bulk-xm HTTP {r.status_code}: {r.text[:160]}")
            if r.status_code == 404:
                print("  hint  /surveys/ZTF/... endpoint may not exist; trying /objects search")

    # 5b. Per-object lookup on first 10 IDs spread across years
    url_obj_tpl = f"{_api_base()}/surveys/ZTF/objects/{{}}"
    n_obj_200 = n_obj_404 = n_obj_other = 0
    hits_by_year: dict[int, int] = {}
    for ztf_id, year in list(zip(sample_ids, sample_years))[:10]:
        try:
            r = httpx.get(url_obj_tpl.format(ztf_id), headers=headers, timeout=15.0)
        except httpx.HTTPError:
            continue
        if r.status_code == 200:
            n_obj_200 += 1
            hits_by_year[year] = hits_by_year.get(year, 0) + 1
        elif r.status_code == 404:
            n_obj_404 += 1
        else:
            n_obj_other += 1
    print(
        f"  per-object: {n_obj_200} HTTP 200, {n_obj_404} HTTP 404, "
        f"{n_obj_other} other across 10 IDs"
    )
    if hits_by_year:
        print(f"  per-object hits-by-year: {hits_by_year}")

    # 5c. Survey-agnostic /objects search as fallback
    url_search = f"{_api_base()}/objects"
    try:
        r = httpx.get(
            url_search,
            headers=headers,
            params={"object_id": sample_ids[0], "limit": 5},
            timeout=15.0,
        )
        if r.status_code == 200:
            data = r.json().get("data", r.json())
            n = len(data) if isinstance(data, list) else 0
            print(f"  /objects search for {sample_ids[0]}: {n} hit(s)")
        else:
            print(f"  /objects HTTP {r.status_code}: {r.text[:120]}")
    except httpx.HTTPError as e:
        print(f"  /objects network: {e}")

    pct_bulk = 100.0 * n_bulk_hit / max(len(sample_ids), 1)
    if pct_bulk >= 20 or n_obj_200 >= 3:
        print("  -->   ZTF retro-ingest LIKELY — train Babamul head on existing ZTF labels")
    elif pct_bulk == 0 and n_obj_200 == 0:
        print("  -->   ZTF retro-ingest ABSENT — Babamul is forward-only; v7 must use live-LSST cohort")
    else:
        print("  -->   partial ZTF coverage — inspect hits-by-year to find Babamul's cutoff date")
    return True


def step6_kafka_schema_dump() -> bool:
    """Pull one message from each babamul.*.hosted topic, dump top-level keys.

    Decisive question: does the payload contain AppleCiDEr probability
    scores (=> SN-filter projection + trust head), or only topic
    membership (=> 4-way categorical feature, no trust head)?
    """
    print("\n[6/6] Kafka payload schema dump (AppleCiDEr scores in payload?)")
    try:
        import io
        import json

        from confluent_kafka import Consumer
        from confluent_kafka.admin import AdminClient
    except ImportError:
        print("  SKIP  pip install confluent-kafka")
        return False
    try:
        import fastavro  # Babamul wire format is Avro Object Container Files
    except ImportError:
        fastavro = None  # type: ignore[assignment]
        print("  WARN  fastavro not installed; Avro records will only show binary fingerprint")

    conf_base = {
        "bootstrap.servers": os.environ["BABAMUL_KAFKA_SERVER"],
        "security.protocol": "SASL_PLAINTEXT",
        "sasl.mechanism": "SCRAM-SHA-512",
        "sasl.username": os.environ["BABAMUL_KAFKA_USERNAME"],
        "sasl.password": os.environ["BABAMUL_KAFKA_PASSWORD"],
    }

    # 6a. Discover topics this user can see
    admin = AdminClient(conf_base)
    md = admin.list_topics(timeout=10.0)
    babamul_topics = sorted(t for t in md.topics.keys() if t.startswith("babamul."))
    print(f"  topics found ({len(babamul_topics)}): {babamul_topics[:8]}{'...' if len(babamul_topics) > 8 else ''}")
    if not babamul_topics:
        print("  FAIL  no babamul.* topics visible")
        return False

    # 6b. Inspect one message from each .hosted topic (cap at 6 for runtime)
    target_topics = [t for t in babamul_topics if t.endswith(".hosted")][:6]
    if not target_topics:
        target_topics = babamul_topics[:4]
    print(f"  inspecting {len(target_topics)} topic(s):")

    score_keywords = (
        "applecider",
        "score",
        "prob",
        "p_snia",
        "p_sn",
        "p_ia",
        "classification",
        "class_prob",
        "pred",
    )
    schemas: dict[str, list[str] | str] = {}
    for topic in target_topics:
        conf = dict(
            conf_base,
            **{
                "group.id": f"{os.environ['BABAMUL_KAFKA_USERNAME']}-debass-schema",
                "auto.offset.reset": "earliest",
                "enable.auto.commit": False,
                "session.timeout.ms": 10000,
            },
        )
        c = Consumer(conf)
        try:
            c.subscribe([topic])
            # Poll up to 10s; first poll often returns None during partition assignment
            msg = None
            for _ in range(3):
                msg = c.poll(timeout=4.0)
                if msg is not None and not msg.error():
                    break
        finally:
            c.close()
        if msg is None or msg.error():
            print(f"    {topic}: no msg in 12s")
            continue
        raw = msg.value() or b""
        # Try JSON first
        parsed = None
        try:
            parsed = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            parsed = None
        if isinstance(parsed, dict):
            keys = sorted(parsed.keys())
            print(f"    {topic} (n={len(raw)}B, JSON): {keys}")
            schemas[topic] = keys
            score_hits = [
                k for k in keys if any(s in k.lower() for s in score_keywords)
            ]
            if score_hits:
                print(f"      score-like keys: {score_hits}")
                # Show one nested level for the first hit
                for sh in score_hits[:1]:
                    val = parsed.get(sh)
                    if isinstance(val, dict):
                        print(f"      {sh} subkeys: {sorted(val.keys())}")
                    elif isinstance(val, (int, float, str)):
                        print(f"      {sh} = {val!r}")
        elif parsed is not None:
            print(f"    {topic} (n={len(raw)}B, JSON non-dict): {type(parsed).__name__}")
            schemas[topic] = f"json-{type(parsed).__name__}"
        elif fastavro is not None and raw[:4] == b"Obj\x01":
            # Avro Object Container File — embedded schema, no out-of-band lookup.
            try:
                reader = fastavro.reader(io.BytesIO(raw))
                writer_schema = reader.writer_schema
                record = next(reader, None)
            except Exception as e:
                print(f"    {topic} (n={len(raw)}B, Avro): parse failed: {e}")
                schemas[topic] = f"avro-fail:{type(e).__name__}"
                continue
            if isinstance(record, dict):
                keys = sorted(record.keys())
                print(f"    {topic} (n={len(raw)}B, Avro): {keys}")
                schemas[topic] = keys
                # Recursively gather score-like field names anywhere in the record
                score_paths: list[str] = []

                def _walk(node, path: str) -> None:
                    if isinstance(node, dict):
                        for k, v in node.items():
                            sub = f"{path}.{k}" if path else k
                            if any(s in k.lower() for s in score_keywords):
                                score_paths.append(sub)
                            _walk(v, sub)
                    elif isinstance(node, list) and node:
                        _walk(node[0], path + "[0]")

                _walk(record, "")
                if score_paths:
                    print(f"      score-like fields: {score_paths[:8]}")
                    for sp in score_paths[:3]:
                        # Resolve dotted path to value
                        cursor = record
                        ok = True
                        for part in sp.split("."):
                            if part.endswith("[0]"):
                                base = part[:-3]
                                cursor = cursor.get(base) if isinstance(cursor, dict) else None
                                cursor = cursor[0] if isinstance(cursor, list) and cursor else None
                            else:
                                cursor = cursor.get(part) if isinstance(cursor, dict) else None
                            if cursor is None:
                                ok = False
                                break
                        if ok and isinstance(cursor, (int, float, str, bool)):
                            print(f"      {sp} = {cursor!r}")
                        elif ok and isinstance(cursor, dict):
                            print(f"      {sp} subkeys: {sorted(cursor.keys())}")
                # Print the writer schema's record name once per topic for clarity
                if isinstance(writer_schema, dict):
                    rec_name = writer_schema.get("name") or writer_schema.get("type")
                    print(f"      avro record name: {rec_name}")
                # Dump the two fields most likely to carry Babamul-computed
                # enrichment beyond the underlying alert packet:
                #   - properties: per-object alert metadata (host distance,
                #                 quality flags, AppleCiDEr per-class probs?)
                #   - survey_matches: Babamul-computed cross-survey associations
                for field in ("properties", "survey_matches"):
                    val = record.get(field)
                    if val is None:
                        print(f"      {field}: <missing>")
                    elif isinstance(val, dict):
                        print(f"      {field} keys ({len(val)}): {sorted(val.keys())}")
                        # Show numeric / string scalars at one level deep
                        scalars = {
                            k: v
                            for k, v in val.items()
                            if isinstance(v, (int, float, str, bool))
                        }
                        if scalars:
                            print(f"      {field} scalars: {scalars}")
                    elif isinstance(val, list):
                        print(f"      {field}: list[len={len(val)}]")
                        if val and isinstance(val[0], dict):
                            print(f"      {field}[0] keys: {sorted(val[0].keys())}")
                            scalars0 = {
                                k: v
                                for k, v in val[0].items()
                                if isinstance(v, (int, float, str, bool))
                            }
                            if scalars0:
                                print(f"      {field}[0] scalars: {scalars0}")
                    else:
                        print(f"      {field}: {type(val).__name__} = {val!r}")
            else:
                print(f"    {topic} (n={len(raw)}B, Avro): empty / non-record")
                schemas[topic] = "avro-empty"
        else:
            head = raw[:32].hex()
            print(f"    {topic} (n={len(raw)}B, binary): magic={head}")
            schemas[topic] = f"binary:{head[:8]}"

    # 6c. Verdict
    list_schemas = {t: k for t, k in schemas.items() if isinstance(k, list)}
    if list_schemas:
        any_score = any(
            any(s in k.lower() for k in keys for s in score_keywords)
            for keys in list_schemas.values()
        )
        if any_score:
            print("  -->   AppleCiDEr scores ARE in payload — wire as SN-filter projector + trust head")
        else:
            print("  -->   No score keys in any topic — wire AppleCiDEr as 4-way topic-membership categorical")
    return True


def main() -> int:
    print(f"Babamul probe — repo root: {REPO_ROOT}")
    results = {
        "env": step1_env(),
    }
    if results["env"]:
        results["rest"] = step2_rest_whoami()
        results["backfill"] = step3_dp1_backfill()
        results["kafka"] = step4_kafka_connect()
        results["ztf_retro"] = step5_ztf_retro_backfill()
        results["kafka_schema"] = step6_kafka_schema_dump()

    print("\n=== summary ===")
    for k, v in results.items():
        print(f"  {k:<14} {'PASS' if v else 'FAIL/SKIP'}")
    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
