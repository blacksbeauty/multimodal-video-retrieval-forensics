# Traffic Retrieval Acceptance Checklist

## Purpose

This checklist determines whether the current system is truly capable of traffic-oriented retrieval, not just whether the APIs exist.

The target system is:

- traffic scene and traffic violation analysis oriented
- retrieval-first, not just detection-first
- usable for structured traffic object lookup and later event retrieval

Passing this checklist means the system is ready for a first usable internal demo.
Failing this checklist means the system is still only pipeline-complete rather than retrieval-usable.

## Scope

This checklist evaluates the current first-stage system:

- CLIP retrieval
- hybrid search
- detection retrieval
- segment aggregation
- metadata persistence

It does not yet require:

- tracking
- event reasoning
- violation classification
- audit/export workflows

## Acceptance Levels

Use these levels when summarizing current state:

- `L0 Not Searchable`: APIs exist but retrieval cannot reliably return valid traffic results.
- `L1 Barely Searchable`: Some traffic objects can be found, but retrieval is noisy or unstable.
- `L2 Demo Searchable`: Core traffic targets can be found with acceptable top-k quality for demo use.
- `L3 Operational Prototype`: Retrieval is stable enough for repeated internal use and offline evaluation.

Current target for this phase:

- Reach `L2 Demo Searchable`

## Section 1: Environment Readiness

### 1.1 Service startup

- [ ] FastAPI app starts without import errors
- [ ] `AppState` successfully builds all required services
- [ ] CLIP retrieval service is available
- [ ] detection retrieval service is available
- [ ] hybrid search service is available
- [ ] metadata directories are auto-created on startup

Pass condition:

- The app boots successfully and all registered services exist.

### 1.2 Model readiness

- [ ] OpenCLIP model can load and answer a sample query
- [ ] YOLOv8 model can load in CPU-only mode
- [ ] YOLO model is lazily loaded and reused
- [ ] invalid or missing YOLO dependency produces a clear log or error

Pass condition:

- Both CLIP and detection modules are callable in the local environment.

## Section 2: Data Readiness

### 2.1 Video and frame assets

- [ ] test traffic videos are present under the configured video directory
- [ ] extracted frames exist for those videos
- [ ] frame filenames preserve the `video stem + timestamp` pattern

Pass condition:

- At least one full traffic video and its extracted frames are available for indexing and detection.

### 2.2 Metadata persistence

- [ ] CLIP embedding metadata exists
- [ ] FAISS index exists
- [ ] detection metadata files are written to `metadata/detections/`
- [ ] detection metadata is grouped per video
- [ ] metadata can be reloaded after app restart

Pass condition:

- Retrieval artifacts survive restart and do not require reprocessing every run.

## Section 3: Detection Ingest Acceptance

### 3.1 Detection batch ingest

Run:

```http
POST /api/detection/ingest-directory
```

Checklist:

- [ ] endpoint returns success for a valid frame directory
- [ ] broken frames are skipped instead of crashing the run
- [ ] per-video detection metadata files are saved
- [ ] ingest response includes per-video counts
- [ ] logs show processed frame counts

Pass condition:

- Detection ingest completes end-to-end without manual intervention.

### 3.2 Detection metadata quality

Inspect one saved detection metadata file.

- [ ] `video_id` exists
- [ ] `video_name` exists
- [ ] `video_path` is normalized
- [ ] every frame record contains `frame_path`
- [ ] every frame record contains `timestamp`
- [ ] detection labels are normalized lowercase
- [ ] detection confidences are numeric

Pass condition:

- Saved JSON is structurally valid and reusable for search.

## Section 4: Detection Search Acceptance

### 4.1 Exact traffic label retrieval

Run:

- `car`
- `truck`
- `bus`
- `motorcycle`
- `person`
- `traffic light`

Checklist for each:

- [ ] returns results when the object actually exists in the indexed videos
- [ ] top results correspond to frames containing that object
- [ ] returned fields include `video_name`, `timestamp`, `frame_path`, `matched_label`, `confidence`
- [ ] results are sorted by confidence

Pass condition:

- At least 5 of the 6 supported object classes are retrievable on known-positive test data.

### 4.2 Negative query behavior

Test with a label that should not exist in current metadata.

- [ ] system returns empty results instead of invalid matches
- [ ] no crash occurs
- [ ] logs remain readable

Pass condition:

- No false hard failures for empty-match scenarios.

## Section 5: Traffic Query Rewrite Acceptance

Test the following Chinese queries through hybrid search:

- `汽车`
- `轿车`
- `卡车`
- `货车`
- `公交车`
- `摩托车`
- `红绿灯`
- `行人`

Checklist:

- [ ] each query is expanded to the expected English traffic alias
- [ ] rewritten aliases are visible in hybrid response
- [ ] rewritten query behavior is better than raw Chinese CLIP-only behavior

Pass condition:

- Traffic Chinese query rewrite consistently improves or stabilizes retrieval quality.

## Section 6: Hybrid Search Acceptance

### 6.1 CLIP + Detection fusion

Run:

```http
POST /api/search/hybrid
```

With sample queries:

- `car`
- `truck`
- `traffic light`
- `汽车`
- `货车`
- `红绿灯`

Checklist:

- [ ] hybrid endpoint responds successfully
- [ ] segment-level results are returned
- [ ] `matched_by` includes `clip`, `detection`, or both
- [ ] `detection_score` is populated when detection contributes
- [ ] result timestamps are segment-oriented, not raw frame flood only

Pass condition:

- Hybrid search returns explainable segment-level traffic results.

### 6.2 Segment deduplication

Checklist:

- [ ] adjacent frames from the same object cluster do not flood top-k
- [ ] results are grouped into 5-second windows
- [ ] the highest scoring frame is kept as representative
- [ ] duplicate segment spam is visibly reduced compared with raw frame search

Pass condition:

- Top-k results are readable and segment-oriented for humans.

## Section 7: Retrieval Quality Acceptance

This is the most important section.

### 7.1 Human spot-check

For each target class:

- review top 5 results
- verify whether the object is actually visible in the frame or segment

Measure:

- `Precision@5`

Pass suggestion for first usable demo:

- `Precision@5 >= 0.60` for `car`
- `Precision@5 >= 0.50` for `truck`
- `Precision@5 >= 0.50` for `bus`
- `Precision@5 >= 0.50` for `motorcycle`
- `Precision@5 >= 0.60` for `person`
- `Precision@5 >= 0.50` for `traffic light`

### 7.2 Positive coverage

For a small labeled validation set:

- [ ] known car scenes can be found
- [ ] known truck scenes can be found
- [ ] known bus scenes can be found
- [ ] known motorcycle scenes can be found
- [ ] known person scenes can be found
- [ ] known traffic light scenes can be found

Measure:

- `Recall@10`

Pass suggestion for first demo:

- `Recall@10 >= 0.70` on at least 4 core traffic object classes

### 7.3 Comparison against CLIP-only

For the same test queries, compare:

- `/api/search`
- `/api/search/hybrid`

Checklist:

- [ ] hybrid improves precision on object-centric queries
- [ ] hybrid improves Chinese traffic query stability
- [ ] hybrid reduces visually irrelevant semantic drift

Pass condition:

- Hybrid is measurably better than CLIP-only for structured traffic objects.

## Section 8: Robustness Acceptance

### 8.1 Fault tolerance

- [ ] unreadable frames are skipped
- [ ] missing detection metadata does not crash hybrid search
- [ ] empty result sets are handled cleanly
- [ ] broken metadata JSON is logged and skipped

Pass condition:

- No single bad frame or bad file collapses the whole pipeline.

### 8.2 Reproducibility

- [ ] same ingest input produces stable metadata shape
- [ ] repeated searches produce consistent top results
- [ ] app restart does not destroy retrieval ability

Pass condition:

- Search behavior is repeatable enough for internal demo use.

## Section 9: Minimal Validation Dataset Recommendation

For first acceptance, prepare:

- 20 to 50 short traffic videos
- or 200 to 500 sampled traffic frames with known object presence

Recommended labels:

- `car`
- `truck`
- `bus`
- `motorcycle`
- `person`
- `traffic light`

Recommended query set:

- English:
  - `car`
  - `truck`
  - `bus`
  - `motorcycle`
  - `person`
  - `traffic light`
- Chinese:
  - `汽车`
  - `轿车`
  - `卡车`
  - `货车`
  - `公交车`
  - `摩托车`
  - `红绿灯`
  - `行人`

## Section 10: Final Acceptance Summary Template

Use this summary after running the checklist:

```text
Traffic Retrieval Acceptance Summary

Date:
Evaluator:
Dataset:

Environment readiness: PASS / FAIL
Detection ingest: PASS / FAIL
Detection search: PASS / FAIL
Hybrid search: PASS / FAIL
Segment aggregation: PASS / FAIL
Chinese traffic rewrite: PASS / FAIL
Retrieval quality: PASS / FAIL
Robustness: PASS / FAIL

Current maturity:
- L0 Not Searchable
- L1 Barely Searchable
- L2 Demo Searchable
- L3 Operational Prototype

Main blockers:
1.
2.
3.

Decision:
- Not ready
- Ready for internal demo
- Ready for controlled pilot
```

## Section 11: Executable Evaluation Assets

The repository now includes:

- evaluation template:
  - [tests/traffic_retrieval_eval_template.json](D:/all-seeing%20vision/RAG/project/tests/traffic_retrieval_eval_template.json)
- offline evaluation script:
  - [tests/run_traffic_retrieval_eval.py](D:/all-seeing%20vision/RAG/project/tests/run_traffic_retrieval_eval.py)

Important:

- The current template is a first real sample set derived from the repository's existing traffic videos and current retrieval outputs.
- It should be treated as a bootstrap evaluation set, not final ground truth.
- Before formal acceptance, manually verify each expected segment against the underlying video or frame evidence.

Recommended usage:

1. Edit the JSON template to reflect your real labeled traffic validation set.
2. Fill `expected_segments` with known-positive windows.
3. Run the evaluator:

```powershell
python .\tests\run_traffic_retrieval_eval.py
```

The script reports:

- per-query `Precision@K`
- per-query recall against labeled expected segments
- macro average precision
- macro average recall

This script is intended as a first-stage acceptance tool, not a final benchmark framework.

## Recommended Exit Criteria For This Phase

The current phase should be considered complete only if:

- [ ] detection ingest is stable
- [ ] at least 5 supported traffic object classes are retrievable
- [ ] hybrid search outperforms CLIP-only on object-centric traffic queries
- [ ] top results are segment-level and human-readable
- [ ] the team can demonstrate at least 3 successful end-to-end traffic queries live

If all items above are checked, the system can be called:

- `first usable traffic retrieval prototype`

If not, it should still be described as:

- `retrieval pipeline prototype with partial traffic capability`
