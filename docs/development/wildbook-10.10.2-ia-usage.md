# Wildbook 10.10.2 IA Usage Audit

This audit inspected the cloned Wildbook repo at `../../Wildbook` to determine which deployed Wildbook branches still use WBIA directly, which branches contain newer ml-service/pipeline code, and how IA configs drift by branch.

Inspection was non-destructive. Branches were queried by remote ref with `git grep` / `git show`; the shared checkout was not switched.

## Executive Summary

- The public Wild Me platforms page lists 18 Wildbook platforms. A live check of the linked platform URLs found all 18 reachable, with one caveat: the page links Whiskerbook as `https://www.whiskerbook.org/`, which has a certificate hostname mismatch, while `https://whiskerbook.org/` is reachable.
- Every inspected deployed branch still contains direct WBIA/IBEIS usage through `org.ecocean.identity.IBEISIA` and IA config keys such as `IBEISIARestUrlStartIdentifyAnnotations` and `IBEISIARestUrlStartDetectImages`.
- Deployed IA configs overwhelmingly point to WBIA endpoints like `/api/engine/query/graph/` and `/api/engine/detect/cnn/lightnet/` or `/api/engine/detect/cnn/yolo/`.
- The newer ml-service v2 path exists in `origin/main` and several 10.10.2-derived branches, but it is gated by `_id_conf.default.pipeline_root == "vector"` plus a populated `_mlservice_conf` array. The checked-in IA configs inspected generally do not enable that route.
- Several branches have highly species-specific `src/main/resources/bundles/IA.json` or `IA.properties` drift. Others only carry generic docker IA templates, often giraffe/MiewID-oriented regardless of branch name.
- `processing-pipeline` appears widely in newer branches, but in most inspected branches it is a BulkImport/UI status, not evidence of a separate deployed pipeline service.

## Public Platform Count

`https://www.wildme.org/platforms.html` currently lists 18 platform cards:

| # | Public Platform | Visit URL Checked | Status | Matching Branch Evidence |
|---:|---|---|---|---|
| 1 | Flukebook | `https://www.flukebook.org/` | HTTP 200 | `origin/flukebook` |
| 2 | Sharkbook | `https://www.sharkbook.ai/` | HTTP 200 | `origin/sharkbook.ai` |
| 3 | MantaMatcher | `https://mantamatcher.org/` | HTTP 200 | `origin/mantamatcher.org` |
| 4 | GiraffeSpotter | `https://giraffespotter.org/` | HTTP 200 | `origin/giraffe` |
| 5 | Internet of Turtles | `https://iot.wildbook.org/` | HTTP 200 | `origin/iot` |
| 6 | Zebra Wildbook | `https://zebra.wildme.org/` | HTTP 200 | `origin/zebra` |
| 7 | Wildbook for Lynx | `https://lynx.wildbook.org/` | HTTP 200 | `origin/lynx` |
| 8 | Spotting Giant Sea Bass | `https://spottinggiantseabass.msi.ucsb.edu/` | HTTP 200 | `origin/bass` |
| 9 | African Carnivore Wildbook | `https://africancarnivore.wildbook.org/` | HTTP 200 | `origin/africancarnivorewildbook-acw`, `origin/carnivore` |
| 10 | Amphibian and Reptile Wildbook | `https://amphibian-reptile.wildbook.org/` | HTTP 200 | `origin/amphibian-reptile` |
| 11 | Spot a Shark USA | `https://spotasharkusa.com/` | HTTP 200 | no exact branch found; likely shark/sand-tiger drift to confirm |
| 12 | Whiskerbook | `https://www.whiskerbook.org/` | cert mismatch on `www`; `https://whiskerbook.org/` HTTP 200 | `origin/whiskerbook` |
| 13 | Grouper Spotter | `https://www.grouperspotter.org/` | HTTP 200 | `origin/grouper` |
| 14 | SeadragonSearch | `https://seadragonsearch.org/` | HTTP 200 | `origin/seadragon` |
| 15 | Seal Wildbook | `https://seals.wildme.org/` | HTTP 200 | `origin/seals` |
| 16 | Snail Wildbook | `https://snails.wildme.org/` | HTTP 200 | `origin/snails` |
| 17 | DeerSpotter | `https://deer.wildme.org/` | HTTP 200 | `origin/deer` |
| 18 | Troutspotter | `https://troutspotter.wildme.org/` | HTTP 200 | `origin/troutspotter` |

Public evidence supports 18 currently reachable Wildbook platforms. This does not prove all runtime deployments are managed from checked-in branch configs; live server configs may differ from repository defaults.

## Code Paths That Matter

### Legacy WBIA Path

Wildbook still calls WBIA through `IBEISIA.java` using IA config properties:

- `IBEISIARestUrlStartIdentifyAnnotations` -> `/api/engine/query/graph/`
- `IBEISIARestUrlStartDetectImages` -> `/api/engine/detect/cnn/...`
- `IBEISIARestUrlIdentifyReview` -> `/api/review/query/graph/`
- `IBEISIARestUrlDetectReview` -> `/api/review/detect/cnn/...`
- `IBEISIARestUrlGetJobStatus` and `IBEISIARestUrlGetJobResult`

Representative refs:

- `origin/main:src/main/java/org/ecocean/identity/IBEISIA.java:233`
- `origin/main:src/main/java/org/ecocean/identity/IBEISIA.java:443-446`
- `origin/flukebook:src/main/java/org/ecocean/identity/IBEISIA.java:229-232`
- `origin/seals:src/main/java/org/ecocean/identity/IBEISIA.java:229-232`

### New ml-service Path

The newer ml-service route exists in current branches but is config-gated:

- `origin/main:src/main/java/org/ecocean/IAJsonProperties.java:284-296`
- `origin/main:src/main/java/org/ecocean/ia/IA.java:160-163`
- `origin/main:src/main/java/org/ecocean/ia/MlServiceClient.java`
- `origin/main:src/main/java/org/ecocean/ia/MlServiceProcessor.java`
- `origin/main:src/main/java/org/ecocean/servlet/IAGateway.java:697-702`

The route requires:

- `_id_conf.default.pipeline_root == "vector"`
- a populated taxonomy-specific `_mlservice_conf`

Observed implication: current code can route to ml-service, but the checked-in IA configs inspected still mostly route to WBIA because active `_mlservice_conf` + `vector` configs were not found.

## Branch Findings

### Current / 10.10.2-Derived Branches

| Branch | Version | WBIA Config | ml-service / pipeline | Notes |
|---|---:|---|---|---|
| `origin/main` | `10.10.2` | Devops `IA-wbia.*` points to `http://wbia:5000`; `src/main/resources/bundles/IA.json` is empty | Full ml-service v2 code present, but checked-in IA config does not enable it | Main has modern code, not active species config |
| `origin/whiskerbook` | `10.10.2` | Active `IA.json` points to `https://tier2.dyn.wildme.io:5013/api/engine/...` | Full ml-service v2 code present, but active IA config remains WBIA direct | Snow leopard/jaguar config with `sv_on=true` |
| `origin/sharkbook.ai` | `10.10.2` | Generic docker WBIA config, giraffe/MiewID-oriented | Full ml-service v2 code present, no checked-in enablement config found | Branch name does not match checked-in IA config |
| `origin/zebra` | `10.10.2` | Generic docker WBIA config, giraffe/MiewID-oriented | Full ml-service v2 code present, no checked-in enablement config found | Branch name does not match checked-in IA config |
| `origin/giraffe` | `10.10.2` | Generic docker WBIA config aligns with giraffe | Full ml-service v2 code present, no checked-in enablement config found | Giraffe `MiewId` / `giraffe_v1` config |

### Fluke / Marine Mammal Branches

| Branch | Version | WBIA Config | Identification Config | Detection Config | Notes |
|---|---:|---|---|---|---|
| `origin/flukebook` | `10.9.5` | Active `IA.json` points to `https://kaiju.dyn.wildme.io:5005/api/engine/...` | `OC_WDTW`, `KaggleSeven`, `Finfindr`, `CurvRankFluke`, `CurvRankDorsal`, `Deepsense` | Multiple yolo/lightnet configs for fluke, dorsal, dolphin, orca, right whale | WBIA direct only in inspected code |
| `origin/seadragon` | `10.8.1.2` | Generic docker WBIA config | Generic `MiewId` / HotSpotter template | Generic giraffe detector template | No seadragon-specific checked-in IA found |
| `origin/seadragon-zombie-encounter-fix` | `10.8.1.2` | Same as `seadragon` | Same | Same | Focused fix branch, IA unchanged |
| `origin/mantamatcher.org` | `10.9.5` | Generic docker WBIA config | Generic `MiewId` / HotSpotter template | Generic giraffe detector template | No manta-specific checked-in IA found |

### Mammal / Species-Specific Branches

| Branch | Version | WBIA Config | Identification Config | Detection Config | Notes |
|---|---:|---|---|---|---|
| `origin/seals` | `10.9.5` | Active `IA.json` / `IA.properties` point to `https://seals.hydra.dyn.wildme.io/api/engine/...` | HotSpotter `sv_on=true` reused across many seal taxa | `seals_v1`, `densenet`, `nms_aware=ispart`, `sensitivity=0.63` | Large species alias/reference tree |
| `origin/snails` | `10.9.5` | Active `IA.json` / `IA.properties` point to `https://snails.hydra.dyn.wildme.io/api/engine/...` | HotSpotter `sv_on=true`; reusable identifier default `{}` | `snail_effnet_v0`, `snail_v0`, `efficientnet`, orientation plugin | Large snail taxonomy config |
| `origin/troutspotter` | `10.9.5` | Active `IA.json` / `IA.properties` point to `https://troutspotter.hydra.dyn.wildme.io/api/engine/...` | HotSpotter `sv_on=true`, `n=20` | `trout_v1`, `trout_effnet_v0`, `efficientnet`, orientation plugin | Trout-specific config |
| `origin/amphibian-reptile` | `10.9.5` | Active tier2 WBIA `https://tier2.dyn.wildme.io:5001/api/engine/...` | HotSpotter `sv_on=true` for Salamandra | `salanader_fire_v0`, `densenet`, `sensitivity=0.50` | Typo-like `salanader` appears in config |
| `origin/flakebook` | `10.8.1.2` | Active tier2 WBIA `https://tier2.dyn.wildme.io:5001/api/engine/...` | HotSpotter `sv_on=true` for fire salamander configs | `salanader_fire_v0`, `densenet` | Older flakebook-specific active config |
| `origin/africancarnivorewildbook-acw` | `10.9.5` | Docker config generic; resource properties have carnivore-specific entries | `sv_on=true` for cheetah, leopard, wild dog | `cheetah_v1`, `leopard_v0`, `wilddog_v0` / related tags | Strong species-specific `IA.properties` drift |
| `origin/carnivore` | `10.3.0` | Active `commonConfiguration.properties` includes `http://35.161.123.237:5008/api/engine/...`; docker config generic | `sv_on=true` for carnivore species | `cheetah_v1`, `leopard_v0`, `wilddog_v3+wilddog_v2+wilddog_v1` | Older branch, active remote WBIA host |
| `origin/wildnorth` | `10.2.1` | Devops WBIA config and compose include WBIA | Carnivore-like `sv_on=true` examples | Placeholder species model tags | Older branch, placeholder `[IPAddress]` drift |

### Generic / Mostly Template Branches

| Branch | Version | WBIA Config | ml-service / pipeline | Notes |
|---|---:|---|---|---|
| `origin/production` | `10.3.0` | Generic docker WBIA config | No full ml-service v2 | Older production branch |
| `origin/bass` | `10.8.1` | Generic docker WBIA config; source IA mostly commented | No ml-service | `processing-pipeline` is status only |
| `origin/deer` | `10.9.5` | Generic docker WBIA config; source IA mostly commented | No ml-service | `processing-pipeline` is status only |
| `origin/grouper` | `10.9.5` | Generic docker WBIA config; source IA mostly commented | No ml-service | Same as deer |
| `origin/lynx` | `10.9.5` | Generic docker WBIA config; source IA mostly commented | No ml-service | Same as deer |
| `origin/ncaquariums` | `10.2.0` | Generic docker WBIA config; source IA mostly commented | No ml-service | Older branch |
| `origin/iot` | `10.9.5` | Active IA endpoint values blank; comments retain WBIA endpoints | No ml-service | IA appears disabled/blank by default |
| `origin/tnng` | `11.0.0` | Uses `http://ibeis:5000/api/engine/...` in devops config | No ml-service hits found | Uses `ibeis` service name rather than `wbia` |
| `origin/wwf-seals-minimal-update` | `6.0.0-FINAL` | Very old commented IBEIS config only | No ml-service | No modern `IA.json` / `IA.properties` |

### Flakebook 10.10 Branches

| Branch | Version | WBIA Config | ml-service / pipeline | Notes |
|---|---:|---|---|---|
| `origin/flakebook-10.10` | `10.9.5` | Devops WBIA config; active source IA config empty/no relevant lines | Older `MLService.java` helper exists, no full v2 processor/client | Config still WBIA direct |
| `origin/flakebook-10.10-ensure-all-detect-before-ID` | `10.9.5` | Same as `flakebook-10.10` | Older `MLService.java` helper; BulkImport pipeline status changes | Branch-specific import behavior |
| `origin/flakebook-10.10-own-mediaasset-uuids` | `10.9.5` | Same | Older `MLService.java` helper | Config still WBIA direct |
| `origin/flakebook-10.10-plus-sec-test` | `10.9.5` | Same | Older `MLService.java` helper | Config still WBIA direct |
| `origin/flakebook-10.10-test-caching-issue` | `10.9.5` | Same | Older `MLService.java` helper | Extra cache/pipeline comments in `IBEISIA.java` |

## Config Patterns To Preserve In Parity Work

Identification configs in deployed branches include:

- HotSpotter default `{}`.
- HotSpotter with `sv_on=true`.
- HotSpotter with `sv_on=true, n=20` in trout.
- `MiewId` in generic/giraffe docker config.
- `OC_WDTW`, `CurvRankFluke`, `CurvRankDorsal`, `Finfindr`, `KaggleSeven`, and `Deepsense` in flukebook.

Detection configs include:

- `lightnet` and `yolo` WBIA endpoints.
- `densenet`, `efficientnet`, model tags like `giraffe_v1`, `seals_v1`, `salanader_fire_v0`, `trout_v1`, `snail_effnet_v0`, `cheetah_v1`, `leopard_v0`, `wilddog_v0`.
- Threshold drift: `nms_thresh`, `sensitivity`, `nms_aware`, labeler/model tags.

## Implications For `hotspotter` / `wildlife-id`

- WBIA compatibility remains operationally important for 10.10.2 deployments because branch configs still point to WBIA endpoints.
- `hotspotter` parity should prioritize the HotSpotter configs currently deployed: default `{}`, `sv_on=true`, and small branch-specific options like trout `n=20`.
- Flukebook algorithms are not all HotSpotter. `OC_WDTW`, `CurvRank*`, `Finfindr`, `KaggleSeven`, and `Deepsense` need separate migration decisions; they should not be assumed to be covered by `hotspotter` LNBNN parity.
- New pipeline/ml-service compatibility requires auditing deployed runtime IA config, not just branch source. The code route exists in current branches, but source configs do not prove it is active.

## Recommended Next Steps

1. Collect live deployed `IA.json` / `IA.properties` from each Wildbook environment, because runtime config may differ from branch source.
2. Build a machine-readable branch/config matrix from the remote refs and live configs.
3. Split parity testing by algorithm family: HotSpotter, MiewID/vector, CurvRank, OC_WDTW, Finfindr, Deepsense.
4. For HotSpotter branches, create functional tests that replay the exact deployed `query_config_dict` values against WBIA and `hotspotter`.
5. For ml-service-capable branches, test both config states: no `_mlservice_conf` falls back to WBIA; `_mlservice_conf` + `pipeline_root=vector` routes to ml-service.
