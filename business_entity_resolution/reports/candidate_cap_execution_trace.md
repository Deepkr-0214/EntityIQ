# Candidate cap execution trace

## Scope

This trace follows `src/pipeline/run_pipeline.py` to the test-time
`output/candidate_pairs.tsv` writer and compares its blocker with
`validate_blocking_recall.py` and `test_blocking_recall.py`. No candidate-cap
configuration was changed and end-to-end inference was not started for this
trace.

## Production flow

1. `run()` in `src/pipeline/run_pipeline.py` selects the train and test country
   splits. For each country, it calls `build_candidate_store_and_index()` with
   the S2 and S3 split files. This function normalizes each record, deduplicates
   entity IDs, and inserts records in a `CompactBlockingIndex`.
2. `featurize_s1_batch()` calls `idx.query(s1_rec, max_cands=200)` for each S1
   record. `CompactBlockingIndex.query()` unions results from seven blocking
   keys: first token, 4-character name prefix, shared name token, address
   number, address token overlap (at least two), consonant skeleton prefix,
   and consonant trigram overlap (at least two).
3. After those seven result sets have been merged, `query()` truncates the
   union to at most `max_cands`. The current truncation converts a set to a
   list and takes its first 200 entries; it is not a per-key limit or a
   relevance-ranked selection.
4. During test inference, `run()` calls `featurize_s1_batch()` with
   `gt_dict=None` and `max_cands=200` for full and final batches. The returned
   `candidates_map` is sorted and written directly to the
   `candidate_entity_ids` column of `output/candidate_pairs.tsv`.
5. `apply_rank_threshold()` selects predicted matches for
   `matching_results.tsv`; it does not change the candidate set written to
   `candidate_pairs.tsv`. There is no later 100-candidate reduction in this
   output path.

## The 100-candidate constant

`src/config.py` defines `MAX_CANDIDATES_PER_S1 = 100`, described as a hard cap
after the union of blocking strategies. Its only Python use is in
`src/blocking/candidate_generator.py`, inside that module's independent
`generate_candidates()` function. Repository search shows the production
pipeline does not import or call that function. Therefore this 100 cap does
not currently constrain `run_pipeline.py` or its `candidate_pairs.tsv` output.

That separate module also has `BLOCKING_TOP_K = 50` for its n-gram strategy;
that constant is likewise not part of the `CompactBlockingIndex` production
path.

## Recall-validation path

Both recall scripts import the production `CompactBlockingIndex`,
`stream_records()`, and `load_ground_truth()` from `src/pipeline/run_pipeline.py`.
They normalize S2/S3 and S1 records with the same `normalize_record()`, build
the same index from the country training splits, and call
`idx.query(s1_rec, max_cands=200)`. The comprehensive validator also performs
an uncapped query to measure the effect of the cap. Thus the blocker and cap
under measurement match the production test-time candidate-generation path.

The recall scripts query the index directly and compare its output with ground
truth. This correctly measures blocking recall before the model. By contrast,
training/validation featurization passes a `gt_dict`; `featurize_s1_batch()`
then adds known true matches found in the S2/S3 pool after the capped query.
That training-only injection is disabled in test inference and is not part of
the direct recall-validation measurement.

## Findings

| Question | Finding |
|---|---|
| Where is the 100 cap used? | Only in the separate `candidate_generator.generate_candidates()` implementation via `MAX_CANDIDATES_PER_S1`. |
| Where is 200 used? | `CompactBlockingIndex.query()` defaults to 200; `featurize_s1_batch()` defaults to 200; train/test call sites and both recall scripts use 200. Test call sites pass it explicitly. |
| Do 100 and 200 affect the same stage? | No. The 100 cap is in an uncalled alternate implementation. The pipeline uses the `CompactBlockingIndex` path. |
| Which cap controls `candidate_pairs.tsv`? | 200, applied by `CompactBlockingIndex.query()` after unioning all seven blocking-key results. |
| Does recall validation use the same candidate-generation path? | Yes for the blocker, normalization, candidate pool construction, and 200 cap. Validation runs on labeled training splits and directly queries the index. |
| Is there a later reduction to 100? | No. The test inference writer writes the returned candidate map; match thresholding affects only predicted matches. |
| Is 200 per blocking strategy? | No. It caps the merged union after all seven strategies. |

## Caveat for recall interpretation

The cap is applied by slicing a list made from a Python set. No stable ranking
or priority is used for the final 200. The retained members can therefore
depend on set iteration order. The recall validator uses the same code, but a
single run measures the realized set for that run; it does not prove the cap's
recall is stable across Python processes or environments.
