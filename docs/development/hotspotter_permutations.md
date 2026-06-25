# HotSpotter WBIA Parameter Permutations

This document records the HotSpotter identification parameters Wildbook sends to WBIA for each inspected deployment, based on the deployment snapshots under `../deployments/` and the current Wildbook Java request path.

## Request Shape

Wildbook sends identification jobs to WBIA at each deployment's configured `start_identify` endpoint, normally:

```text
/api/engine/query/graph/
```

The request body is built in `org.ecocean.identity.IBEISIA.sendIdentify()` and includes:

| Request key | Source |
|---|---|
| `callback_url` | Wildbook callback URL |
| `jobid` | Wildbook `Task` ID |
| `query_config_dict` | The selected IA config option's `query_config_dict` / `queryConfigDict` |
| `matching_state_list` | All Wildbook matching states |
| `user_confidence` | Optional reviewer/user confidence payload |
| `query_annot_uuid_list` | Query annotation ACM UUIDs, converted to WBIA fancy UUID JSON |
| `database_annot_uuid_list` | Target/matching-set annotation ACM UUIDs, or `null` if no explicit target list |
| `query_annot_name_list` | Query names, usually `____` because query identity is unknown |
| `database_annot_name_list` | Target individual IDs, or `____` for unassigned annotations |

For HotSpotter, the meaningful algorithm payload is the `query_config_dict`. Wildbook treats a config containing `sv_on` as HotSpotter for queue-lane decisions and does **not** add the `lane: fast` override even when fastlane is requested.

`IAJsonProperties.identOpts()` normalizes IA JSON entries by copying `query_config_dict` into `queryConfigDict` for legacy code. Older `IA.properties` entries already use `queryConfigDict`; `IBEISIA.queryConfigDict()` then passes that object through unchanged as WBIA `query_config_dict`.

## Observed HotSpotter Configs

There are only four HotSpotter parameter shapes in the inspected branch-source configs:

| Shape | WBIA `query_config_dict` | Meaning / use |
|---|---|---|
| spatial verification on | `{"sv_on": true}` | Most active HotSpotter configs |
| spatial verification off | `{"sv_on": false}` | Flukebook fluke/head HotSpotter configs |
| spatial verification on, top-N override | `{"sv_on": true, "n": 20}` | Troutspotter |
| empty config | `{}` | Snail `_xxxxidentifiers.hotspotter_nosv` legacy/staged identifier, not active `_id_conf` |

No checked-in active HotSpotter config in the deployment snapshots sets `K`, `Knorm`, `pipeline_root: "HotSpotter"`, descriptor parameters, FLANN parameters, LNBNN toggles, ratio-test options, or feature extraction parameters. Those are WBIA-side defaults unless runtime-mounted configs differ.

## Per-Deployment Parameters

| Deployment | Evidence source | Taxonomy / IA class | WBIA endpoint host | HotSpotter `query_config_dict` | Default? | Notes |
|---|---|---|---|---|---|---|
| Flukebook | `flukebook/source/src/main/resources/bundles/IA.json` | `Megaptera.novaeangliae.whale_humpback+fluke` | `https://kaiju.dyn.wildme.io:5005` | `{"sv_on": false}` | not marked | Description: `HotSpotter fluke pattern-matcher`. `whale_fluke` aliases this class. |
| Flukebook | same | `Stenella.frontalis.dolphin_spotted` | `https://kaiju.dyn.wildme.io:5005` | `{"sv_on": true}` | not marked | Uses legacy camel-case key `queryConfigDict` in the IA JSON; Wildbook normalizes this before sending. |
| Flukebook | same | `Eubalaena.australis.right_whale_head` | `https://kaiju.dyn.wildme.io:5005` | `{"sv_on": false}` | not marked | Secondary option after `Deepsense`. |
| Sharkbook | source `IA.json` is `{}` | none confirmed | none confirmed | none | n/a | Checked-in source has no active WBIA HotSpotter config. Shark matching path is Java Groth/I3S from inspected evidence. Docker `IA-wbia.json` contains a generic giraffe HotSpotter template; do not treat it as Sharkbook runtime config. |
| MantaMatcher | source `IA.json` empty | none confirmed | none confirmed | none | n/a | Only commented IA.properties examples found. Runtime config still needed. |
| GiraffeSpotter | `giraffespotter/source/devops/deploy/.dockerfiles/tomcat/IA-wbia.json` | `Giraffa.tippelskirchi.giraffe_whole` | `http://wbia:5000` in Docker template | `{"sv_on": true}` | no | Fallback/secondary option after default `MiewId` (`pipeline_root: "MiewId"`). Source `IA.json` is empty, so production may rely on mounted runtime config. |
| Internet of Turtles | source `IA.json` empty | none confirmed | none confirmed | none | n/a | Only commented IA.properties examples found. Runtime config still needed. |
| Zebra Wildbook | current source `IA.json` is `{}` | none confirmed on current branch | none confirmed | none | n/a | Current branch has no active WBIA config. Docker `IA-wbia.json` is the generic giraffe template. A stale branch reportedly has real Zebra config, but it is not in the current deployment snapshot. |
| Wildbook for Lynx | source `IA.json` empty | none confirmed | none confirmed | none | n/a | Docker `IA-wbia.json` is the generic giraffe template. Runtime config still needed. |
| Giant Sea Bass | source `IA.json` empty | none confirmed | none confirmed | none | n/a | Only commented IA.properties examples found. Runtime config still needed. |
| African Carnivore Wildbook | `african-carnivore-wildbook/source/src/main/resources/bundles/IA.properties` | `Acinonyx jubatus` | `http://[IPAddress]` placeholder | `{"sv_on": true}` | property option `0` | Legacy `IA.properties` path, not IA JSON. Comment says `sv_on: false` may be worth trying, but active source value is true. |
| African Carnivore Wildbook | same | `Panthera pardus` | `http://[IPAddress]` placeholder | `{"sv_on": true}` | property option `0` | Host is not production-real in git. |
| African Carnivore Wildbook | same | `Lycaon pictus` | `http://[IPAddress]` placeholder | `{"sv_on": true}` | property option `0` | Source also has a malformed `http://[IPAddress]]` detect URL typo. |
| Amphibian and Reptile Wildbook | `amphibian-reptile-wildbook/source/src/main/resources/bundles/IA.json` | `Salamandra.salamandra.fire_sal` | `https://tier2.dyn.wildme.io:5001` | `{"sv_on": true}` | not marked | Source IA JSON has active HotSpotter config. |
| Amphibian and Reptile Wildbook | same | `Salamandra.salamandra.salanader_fire` | `https://tier2.dyn.wildme.io:5001` | `{"sv_on": true}` | not marked | IA class/model spelling appears to be `salanader`, not `salamander`. |
| Amphibian and Reptile Wildbook | same | `Salamandra.salamandra.salanader_fire_adult` | `https://tier2.dyn.wildme.io:5001` | `{"sv_on": true}` | not marked | `IA.properties` also has `IBEISIdentOpt_Salamandra_salamandra0={"queryConfigDict": {"sv_on": True} }`; the JSON source is cleaner evidence. |
| Spot a Shark USA | source `IA.json` empty / legacy custom path | none confirmed for WBIA | none confirmed | none | n/a | Legacy branch uses in-process Java Groth-style matching, not confirmed WBIA HotSpotter. |
| Whiskerbook | `whiskerbook/source/src/main/resources/bundles/IA.json` | `Panthera.uncia.snow_leopard` | `https://tier2.dyn.wildme.io:5013` | `{"sv_on": true}` | not marked | Active WBIA HotSpotter config. |
| Whiskerbook | same | `Panthera.onca.jaguar` | `https://tier2.dyn.wildme.io:5013` | `{"sv_on": true}` | not marked | Active WBIA HotSpotter config. |
| Grouper Spotter | source `IA.json` empty | none confirmed | none confirmed | none | n/a | Only commented IA.properties examples found. Runtime config still needed. |
| SeadragonSearch | source `IA.json` empty | none confirmed | none confirmed | none | n/a | Only commented IA.properties examples found. Runtime config still needed. |
| Seal Wildbook | `seal-wildbook/source/src/main/resources/bundles/IA.json` | `Halichoerus.grypus.grey_seal_unknown` | `https://seals.hydra.dyn.wildme.io` | `{"sv_on": true}` | yes | Many seal IA classes alias to `grey_seal_unknown`, including `harbour_seal`, `hawaiian_monk_seal`, `mediterranean_monk_seal`, and `seal_ringed`. |
| Snail Wildbook | `snail-wildbook/source/src/main/resources/bundles/IA.json` | `Achatinella.apexfulva.snail` | `https://snails.hydra.dyn.wildme.io` | `{"sv_on": true}` | yes | Active default HotSpotter config. `_xxxxidentifiers.hotspotter_nosv` contains `{}`, but it is not an active `_id_conf`. |
| DeerSpotter | source `IA.json` empty | none confirmed | none confirmed | none | n/a | Only commented IA.properties examples found. Runtime config still needed. |
| Troutspotter | `troutspotter/source/src/main/resources/bundles/IA.json` | `Salvelinus.fontinalis.trout` | `https://troutspotter.hydra.dyn.wildme.io` | `{"sv_on": true, "n": 20}` | yes | `Salmo.trutta` and `Oncorhynchus.mykiss` alias to `Salvelinus.fontinalis`. This is the only checked-in active HotSpotter config with `n`. |

## Generic Docker Template

Many deployment snapshots include this Docker template in `source/devops/deploy/.dockerfiles/tomcat/IA-wbia.json` and `source/devops/development/.dockerfiles/tomcat/IA-wbia.json`:

```json
{
  "query_config_dict": {
    "sv_on": true
  },
  "description": "HotSpotter pattern-matcher"
}
```

It lives under `Giraffa.tippelskirchi.giraffe_whole._id_conf` as the second option after default `MiewId`. It is valid evidence for the giraffe Docker/WBIA template, but it should **not** be treated as production-real config for Sharkbook, Zebra, Lynx, Grouper, Deer, and other books whose source `IA.json` is empty or unrelated.

## Migration Notes For `hotspotter`

1. The first parity target should support `sv_on=true`, `sv_on=false`, and `n=20` as explicit, versioned search/scoring parameters.
2. Do not infer `pipeline_root: "HotSpotter"` from old configs; most active HotSpotter entries omit `pipeline_root` entirely.
3. Preserve the legacy `queryConfigDict`/`query_config_dict` normalization when replaying Wildbook jobs.
4. Treat `sv_on` as both a scoring parameter and a Wildbook behavior signal: Java uses its presence to classify a job as HotSpotter for fastlane routing.
5. Runtime-mounted IA configs are still the main uncertainty. Before freezing migration defaults, collect live `IA.json`, `IA.properties`, `IA-wbia.json`, and `IA-wbia.properties` from each running deployment.
