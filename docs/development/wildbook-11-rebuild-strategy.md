# Wildbook 11 Rebuild And Reprocess Strategy

The cleanest modernization path is likely to migrate every public Wildbook onto a common Wildbook 11 baseline, re-ingest the source media, rerun modern detection/feature/identification pipelines, and carry forward only the data that is user-visible, scientifically meaningful, or needed to preserve identity continuity.

This should be treated as a rebuild, not a schema-by-schema clone of old branch drift.

## Proposed Target State

- One Wildbook 11 codebase instead of one branch per deployment.
- Per-deployment behavior expressed as configuration, not branch forks.
- Modern media/annotation/feature stores regenerated from source images.
- `wildlife-id` owns identification service/index responsibilities.
- `hotspotter` is retained as a reusable algorithm library for HotSpotter-compatible books.
- Legacy WBIA is used only as a reference/shadow comparator until parity is acceptable.

## Rebuild Principle

For each Wildbook, export authoritative legacy data, create a clean Wildbook 11 deployment, import the preserved data, rerun the processing pipeline, then reconcile new annotations/matches against old identities.

Do not attempt to preserve every old derived artifact:

- WBIA jobs and job IDs.
- Old WBIA annotation UUIDs except as migration crosswalks.
- FLANN indexes.
- Cached chips/features/descriptors.
- Old match result pages except where needed for audit/history.
- Branch-local implementation quirks that can become config.

## Data That Must Survive

Images and locations are necessary, but not sufficient.

Preserve at minimum:

- Original media assets, stable media IDs, filenames, hashes, and storage URIs.
- Encounter records: encounter ID, date/time, location ID, GPS/location hierarchy, taxonomy, submitter, project/site membership, and visibility/access metadata.
- Old individual/name assignments: individual ID/name, encounter membership, annotation membership, aliases, merge/split history if available.
- User-curated annotations: bbox, theta, iaClass, viewpoint, matchAgainst, media asset association, encounter association, and annotation ID crosswalk.
- Trivial/full-image annotations only as scaffolding where needed to keep old encounter-media relationships intact.
- User submissions, attribution, licensing, comments, measurements, biological samples, occurrence/social-unit links, and project memberships if those are user-visible or scientifically meaningful.
- IA config used by the old deployment: algorithm family, detector model tags, thresholds, and identification options.

## Manual Bbox Editing Exists

The assumption that users cannot manually edit bboxes is false for current Wildbook.

Evidence from `../../Wildbook`:

- `frontend/src/AuthenticatedSwitch.jsx` registers `/manual-annotation` and `/edit-annotation` routes.
- `frontend/src/pages/ManualAnnotation.jsx` lets users draw a bbox/rotated rectangle and submit it.
- `frontend/src/pages/EditAnnotation.jsx` pre-fills an existing annotation, lets users redraw/rotate it, removes the old annotation, and creates a new one.
- `frontend/src/models/encounters/useCreateAnnotation.js` posts `x`, `y`, `width`, `height`, `theta`, `iaClass`, and `viewpoint` to `/api/v3/annotations`.
- `src/main/java/org/ecocean/Annotation.java:createFromApi()` persists manual annotations as `org.ecocean.boundingBox` features and tags feature parameters with `_manualAnnotationViaApiV3`.
- `src/main/java/org/ecocean/api/patch/EncounterPatchValidator.java` supports removing annotations from encounters, which the edit flow uses before creating the replacement annotation.

Migration implication: manual/user-curated bboxes are first-class data and should be preserved or explicitly superseded. A rebuild that keeps only images and locations would discard user correction work.

## Annotation Provenance

There is no single universal `createdBy=manual|machine` field across all Wildbook eras, but several paths leave useful markers.

### Strong Manual Signal

Annotations created by the current React manual annotation API can be identified by feature parameters:

- `Annotation.createFromApi()` writes `Feature("org.ecocean.boundingBox", fparams)`.
- `fparams` includes `x`, `y`, `width`, `height`, `theta`, `viewpoint`.
- `fparams` also includes `_manualAnnotationViaApiV3` with a timestamp.
- These annotations are created through `POST /api/v3/annotations` and set `matchAgainst=true`.

Migration rule: if a bounding-box feature has `_manualAnnotationViaApiV3`, treat it as user/manual curated.

### Strong Legacy WBIA / Machine Signal

Legacy WBIA detections created by `IBEISIA.convertAnnotation()` can usually be identified by:

- `Annotation.acmId` set from the WBIA annotation UUID.
- Bounding-box feature params include `detectionConfidence`.
- Feature params include `theta`, and sometimes `viewpoint`.
- Annotation is built from IA result fields such as `class`, `confidence`, `width`, `height`, `xtl`, and `ytl`.

Migration rule: if `acmId` is present and feature params include `detectionConfidence`, treat it as WBIA machine-generated unless a later user-edit marker or audit trail says otherwise.

### Strong ml-service / Machine Signal

Newer ml-service detections created by `MlServiceProcessor` can usually be identified by:

- `Annotation.identificationStatus == complete-mlservice`.
- `Annotation.wbiaRegistered == false` initially for annotations awaiting WBIA registration.
- `Annotation.acmId` is set to the annotation's own ID.
- Annotation may have embeddings attached.
- `Annotation.quality` is set from model `score` or `confidence`.

Migration rule: if `identificationStatus` is ml-service-specific, `wbiaRegistered` is non-null, or embeddings/model metadata indicate ml-service creation, treat it as machine-generated by ml-service.

### Weak / Ambiguous Signals

Older code paths can create `org.ecocean.boundingBox` features without explicit provenance:

- Legacy manual JSP flows.
- Imported annotations.
- Transformed image / spot workflows.
- WBIA pull-back paths that reconstruct annotations from IA by UUID.
- Trivial full-image annotations generated as scaffolding.

For these, classify with heuristics:

- `Feature.isUnity()` or bbox equal to whole media asset -> trivial/scaffold.
- `acmId` present -> likely WBIA-linked.
- `detectionConfidence` present -> likely machine detection.
- `_manualAnnotationViaApiV3` present -> manual.
- annotation attached to an old individual/name, exemplar flag, or matchAgainst=true -> preserve even if provenance is ambiguous.
- bbox without confidence/acmId but with user-era metadata should be treated as curated/accepted unless proven otherwise.

Migration rule: ambiguous accepted annotations should be preserved as seed truth and flagged `provenance=unknown_accepted`, not discarded.

## Recommended Reprocess Policy

Use old annotations as seed truth, not as immutable final output.

### For User-Curated Annotations

- Import legacy bbox/theta/iaClass/viewpoint as accepted annotations.
- Mark them with provenance such as `source=legacy-user` or `source=legacy-manual` when detectable.
- Keep old annotation ID in a migration crosswalk.
- Recompute features/embeddings from these bboxes in the new pipeline.
- Do not overwrite them with detector output unless a reviewer approves.

### For IA-Generated Legacy Annotations

- Import only if they were accepted, matched, user-edited, or attached to an identity.
- Otherwise, prefer regenerating detections from the original image.
- Keep old annotation IDs only for audit/comparison.

### For New Detector Output

- Run modern detectors on every image.
- Match detections to preserved legacy annotations by image, bbox IoU, iaClass/taxonomy, and theta.
- If a new detection overlaps a preserved user annotation, link it as regenerated evidence rather than creating a duplicate.
- If a new detection disagrees strongly with a preserved user annotation, queue for review.

## Identity Reconciliation

Old individual names are the most important continuity layer.

Suggested process:

1. Import old individuals/names and encounter-to-individual links as legacy truth.
2. Recompute annotations/features/embeddings in Wildbook 11.
3. Run modern identification against the rebuilt database.
4. Compare predicted candidates against old identity labels.
5. Auto-confirm high-confidence matches where old label and new top candidate agree.
6. Queue conflicts for review: old identity missing, split/merge candidate, low confidence, or algorithm disagreement.

Do not rely on name strings alone. Preserve stable legacy individual IDs and build a crosswalk:

| Legacy Object | Wildbook 11 Object | Required? |
|---|---|---|
| media asset ID / UUID | media asset ID / UUID | Yes |
| encounter ID | encounter ID | Yes |
| annotation ID | annotation ID | Yes for curated/accepted annotations |
| individual ID/name | individual ID/name | Yes |
| WBIA annot UUID | migration-only reference | Useful |

## Config Migration

Move per-book drift into config packs under a shared Wildbook 11 codebase.

Each deployment config pack should include:

- Public hostname and branding.
- Taxonomy/species list.
- IA classes and viewpoint options.
- Detector model IDs and thresholds.
- Identification algorithm family: HotSpotter, MiewID/vector, CurvRank, OC_WDTW, Finfindr, Deepsense, etc.
- Project/site visibility rules.
- Submission workflow and review policy.

The config snapshots in `../deployments/` are source-control evidence only. They should be replaced or supplemented with live runtime config exports from each deployment.

## Algorithm Scope

The public Wildbook set is not one algorithm.

- HotSpotter-compatible books can migrate through `hotspotter` + `wildlife-id` parity work.
- MiewID/vector books need embedding/index migration and OpenSearch/vector-index validation.
- Flukebook includes `OC_WDTW`, `CurvRankFluke`, `CurvRankDorsal`, `Finfindr`, `KaggleSeven`, and `Deepsense`; these need separate migration decisions.
- Modified Groth / I3S-style books such as Spot a Shark USA and Giant Sea Bass need explicit handling if still active.

## Validation Gates

Before cutting over a rebuilt Wildbook:

- Media count matches legacy export.
- Encounter count and key metadata match legacy export.
- User-visible location/project filters match legacy behavior.
- Preserved manual annotation count matches legacy export.
- Annotation provenance is classified as `manual`, `wbia_machine`, `mlservice_machine`, `trivial`, or `unknown_accepted`, with unknowns preserved for review.
- Curated bbox/theta values round-trip exactly or with documented coordinate-frame conversion.
- Old individual/name assignments are imported and crosswalked.
- Reprocessed features/embeddings exist for every accepted annotation.
- Shadow identification agrees with legacy labels at an acceptable threshold.
- Review queue contains all conflicts instead of silently overwriting old truth.

## Practical Recommendation

Proceed with Wildbook 11 rebuild/reprocess as the strategic path, but do not reduce the preserved dataset to only images and locations. Preserve old identities and curated annotations as seed truth, then regenerate all derived machine data.

The critical migration artifact is a per-book export bundle:

```text
export/<book>/
  media.jsonl
  encounters.jsonl
  annotations.jsonl
  individuals.jsonl
  encounter_individual_links.jsonl
  annotation_individual_links.jsonl
  projects.jsonl
  users_or_submitter_refs.jsonl
  ia-config.json
  crosswalks/
    media.csv
    encounters.csv
    annotations.csv
    individuals.csv
```

Once those bundles are reliable, the old branch drift can be retired instead of carried forward.
