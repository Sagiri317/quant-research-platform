# Verification and Gates

鏈枃浠跺畾涔夋不鐞嗛獙璇併€佺姸鎬佽涔夊拰 gate 鍒嗗眰銆?
## 1. 鐘舵€佽涔?
- `PASS`锛氬疄闄呰繍琛屽苟閫氳繃锛屽繀椤绘湁 command銆乻cope銆乪xit result 鎴?artifact銆?- `FAIL`锛氬疄闄呰繍琛屽苟澶辫触锛屽繀椤绘湁澶辫触鑼冨洿鍜屼慨澶嶅缓璁€?- `NOT_RUN`锛氭湭杩愯銆?- `BLOCKED`锛氱幆澧冦€佷緷璧栥€佹潈闄愩€佽秴鏃舵垨杈圭晫闃诲銆?
浠讳綍鎶ュ憡涓嶅緱鎶?`NOT_RUN` 鎴?`BLOCKED` 鍐欐垚 `PASS`銆?
## 2. PASS 涓嶅彲鏇夸唬

浠ヤ笅缁撴灉浜掍笉鏇夸唬锛?
- pytest PASS
- governance script PASS
- code-map flow PASS
- handoff verifier PASS
- share-safety PASS
- source-tree scan PASS
- compare PASS
- sidecar PASS

姣忎釜 PASS 鍙鐩栧搴斿懡浠ゅ拰瀵瑰簲 scope銆?
## 3. 蹇呰 gate

### Diff gate

```bash
git diff --check
```

妫€鏌?patch 鍩虹鏍煎紡锛屼笉浠ｈ〃鍔熻兘姝ｇ‘銆?
### Governance gate

```bash
conda run -n <environment.yml name> python .agents/skills/quant-research-platform-maintenance/scripts/verify_governance.py --governance-only --scan-scope active-docs --json
```

杩欐槸瀹樻柟 active-doc state gate銆俢lean tree 涔熷繀椤绘壂鎻忛潪绌?active docs锛沗checked_files=[]` 涓嶅緱 `PASS`銆?
Diff-only governance gate 浣跨敤锛?
```bash
conda run -n <environment.yml name> python .agents/skills/quant-research-platform-maintenance/scripts/verify_governance.py --governance-only --scan-scope changed-files --base-ref <base> --json
```

`--allow-active-docs` 鍜?`--scan-active-docs` 涓嶆槸褰撳墠鏈夋晥鍙傛暟銆?
鍐呴儴娌荤悊瑙勫垯浜嬪疄婧愭槸锛?
```text
.agents/skills/quant-research-platform-maintenance/scripts/governance_contract.py
```

active doc roots銆乧urrent-contract-only 鎵弿瑙勫垯銆佺増鏈潰瑙勫垯銆乶egative-test allowlist銆乻ource-tree scan allowlist銆乸ackage safety path parts 鍜?status token 蹇呴』浠庤繖閲岃鍙栨垨娲剧敓銆?
source-tree sensitive scan 鍜?bundle share-safety scan 缁熶竴澶嶇敤璇ヤ簨瀹炴簮锛涙壂鎻?杈圭晫浣跨敤绮剧‘瑙勫垯瀹氫箟琛屽拰鏄庣‘ root 鍒楄〃銆俿ource-tree scan 鐢?`SourceTreeScanInput` 缁熶竴椹卞姩锛泃racked text file銆丳ython AST銆乷wner ranges銆?allowlist definition lines 鍜?governance rule definition lines 鍙瀯閫犱竴娆★紝
individual scanner 涓嶄繚鐣?Path 鏃у叆鍙ｃ€?
handoff layout gate 鐢?`handoff/handoff_contract.py` 鍜?`handoff/verify_handoff_package.py` 鎷ユ湁銆俙FieldSpec` 鍜?`allowed_keys()` 鍙敱 `handoff_contract.py` 瀹氫箟锛沗receipt_contract.py` 鍙嫢鏈?receipt schema specs銆乮nvariants銆乶ormalization 鍜?comparison銆俿ource-tree retired surface gate 鐢?`governance_contract.py`銆乣verify_governance.py` 鍜?`tests/architecture/` 鎷ユ湁銆俠uilder銆乧ollector銆乿erifier 鍜?tests 涓嶅緱澶嶅埗骞宠 receipt validator 鎴?handoff layout 琛ㄣ€?
Hard-scan gate 鐢辨寮?handoff builder/verifier 闂悎锛?
```bash
conda run -n <environment.yml name> python .agents/skills/quant-research-platform-maintenance/scripts/handoff/build_handoff_package.py --repo-root . --out-dir .run/handoffs --task-name <task-name> --base-ref <base-ref>
```

builder 浠?clean Git state 鍐欏叆 bundle锛屼粠 bundle clone 鑷姩鐢熸垚 machine receipts
鍜?validation summary锛屽啀鎵ц single `HandoffPackagePayload` load 涓?package
payload validation锛屾鏌?schema銆乭ash 鍜?share-safety銆?`verify_handoff_package_result()` 鏄敮涓€姝ｅ紡 package-level verifier result path锛?standalone verifier CLI 淇濈暀涓哄畬鏁?bundle replay 鍜?receipt comparison 鐨勭洿鎺ュ叆鍙ｏ紝
浣嗕笉鏄澶?Python wrapper銆侭uilder 鍜?standalone verifier 鍏辩敤
`load_handoff_package_payload()` 涓?`_validate_handoff_package_payload()`锛?standalone verifier 鍦ㄥ悓涓€涓?bundle clone 涓鐢?source-tree scan 缁撴灉鎵ц
share-safety 鍜?hard-scan receipt replay銆傚叿浣撳瓧娈点€?缁撴瀯銆乺eceipt replay銆乻hare-safety 鍜?hard-scan receipt construction 浠?executable
contracts銆乣verify_handoff_package.py` 鍜?`replay_evidence.py` 涓哄噯銆?`verify_governance.py` 鍙繍琛?governance checks锛屼笉鍐?handoff receipts銆?
package smoke receipt 浣跨敤 conda-managed non-isolated build锛歚python -m build --no-isolation --sdist --wheel --outdir <dist>`銆傛瀯寤哄悗绔緷璧栫敱 `environment.yml` 鏄惧紡鎻愪緵锛涙瀯寤轰骇鍑?wheel 鍚庯紝鐢ㄦ爣鍑嗗簱瑙ｅ帇 wheel锛屽苟閫氳繃 `PYTHONPATH` 鎵ц import smoke銆俙profile_smoke` row 鏉ヨ嚜姝ｅ紡 profile runner 鐨?`profile-smoke` profile銆?
缁撴瀯鎬у璁?gate锛?
```bash
conda run -n <environment.yml name> python .agents/skills/quant-research-platform-maintenance/scripts/structure_audit.py --repo-root . --json
```
璇?gate 瑕嗙洊 package build 蹇呴』浣跨敤 `--no-isolation`銆佷笉寰椾娇鐢?installer 鎴?skip-dependency fallback銆乸ackage build/import subprocess timeout銆乥uild-system 鍒?conda backend dependency 鏄犲皠锛屼互鍙?content_scan 澶辫触璺緞涓嶅緱浼€?`PASS`銆?
### Code-map gate

```bash
conda run -n <environment.yml name> python .agents/skills/quant-research-platform-maintenance/scripts/verify_code_map_flow.py --repo-root .
```

璇?gate 楠岃瘉褰撳墠浠ｇ爜鍦板浘鍜岄暱鏈熷悎鍚屻€?
### Agent and architecture tests

```bash
conda run -n <environment.yml name> python tests/tools/run_profile.py --profile fast --run-dir .run/<purpose>/fast
conda run -n <environment.yml name> python tests/tools/run_profile.py --profile all --run-dir .run/<purpose>/all -- --durations=100
conda run -n <environment.yml name> python tests/tools/run_profile.py --profile all --run-dir .run/<purpose>/all-xdist -- -n 4 --dist loadfile --durations=100
```

`pytest-xdist>=3.8,<4` 鍙綔涓?dev-only speed gate 渚濊禆銆俿equential `all` profile
鏄寮?truth gate锛沗all-xdist` 涓嶈兘鏇夸唬 `all`锛屽彧鑳借ˉ鍏呰€楁椂璇佹嵁銆?
濡傛灉瀹屾暣杩愯瓒呮椂锛屽繀椤诲啓 `BLOCKED` 骞惰褰曡秴鏃讹紝鍙互琛ュ厖鍒嗘杩愯缁撴灉銆?
Handoff receipt flow 鐨勪綆鎴愭湰 gate 鏀惧湪 `tests/agents/handoff/`锛氳鐩栧敮涓€ builder 鍏ュ彛銆乻idecar zip envelope銆乪xact zip layout銆乻tatus-only/placeholder/raw output 鎷掔粷銆乸ackage/check/hard-scan receipt schema銆乸ackage mutation銆佷互鍙婁粠 bundle clone 閲嶆斁 public machine receipts銆俶utation tests 瀵?payload/schema/replay mismatch 浣跨敤 `compare_replayed_handoff_payloads()` fast path锛涘畬鏁?bundle verifier 鍙繚鐣欑粰 valid package smoke銆丟it identity 鍜?sidecar envelope 鍙樺紓銆?
Formal handoff validation gate 瑕佹眰 mandatory validation rows 鍏ㄩ儴涓?`PASS`銆?builder 鍜?verifier 閮藉繀椤昏皟鐢?`handoff_contract.mandatory_validation_failures()`锛?浠讳竴 mandatory row 涓?`FAIL`銆乣NOT_RUN`銆乣BLOCKED`銆佺己澶便€侀噸澶嶆垨鏈煡鐘舵€佹椂锛?formal handoff 澶辫触銆俙validation-summary.json` 椤跺眰 status 鐢?rows 鑱氬悎娲剧敓锛?涓嶅緱鐢?package layout PASS銆乺eceipt 鑷唇鎴?verifier status 鏇夸唬 row-level PASS銆?
## 4. Active-doc scan

active docs 鐨勫敮涓€鍙墽琛岃矾寰勬竻鍗曟槸 `governance_contract.ACTIVE_GOVERNANCE_DOC_ROOTS`锛屽叾涓寘鍚?handoff tooling README銆傚叾浠栬剼鏈拰娴嬭瘯鍙鍙栨垨楠岃瘉璇ュ父閲忥紝涓嶇淮鎶ゅ钩琛屾竻鍗曘€?
active docs 鍙弿杩板綋鍓嶈鍒欍€佽亴璐ｃ€佽矾寰勫拰鍛藉悕銆?
current-contract-only 杈圭晫鍥炲綊鐢?gate 璁板綍涓?`FAIL`锛涘畬鏁磋矾寰勮鍒欏彧鐢?`governance_contract.py` 瀹氫箟銆?
鍛藉悕杈圭晫鎵弿琛ㄩ潰鍖呮嫭锛?
```text
file/path names
Python function/class/fixture/constant/module-level assignment names
JSON/YAML/TOML key names
static string concatenation or simple join synthesis
active docs current-truth wording
```

鐗堟湰闈㈣鍒欑敱 `governance_contract.py` 鍜屽搴?tests 鎺у埗銆?
QRP-owned JSON 蹇呴』鏄《灞?object锛屽繀椤诲寘鍚?`schema_id`锛屼笖 `schema_id` 涓嶇紪鐮佺増鏈€?
## 5. Current-Contract Surface

current-contract-only 杈圭晫鐢?owner-layer negative tests 鎴栨槑纭?allowlist 鎺у埗銆俛llowlist 蹇呴』璇存槑鐩殑鍜岀Щ闄ゆ潯浠躲€?
鍐呴儴鎺ュ彛 gate 妫€鏌ワ細

- 璺ㄦ枃浠?private symbol import銆?- `__all__` 鍗曚笅鍒掔嚎瀵煎嚭銆?- owner-local internal API 浠?package `__init__.py` 閲嶅鍑恒€?- storage/artifacts private module pattern銆?- tests 鐧界洅璋冪敤 private helper 鍒嗙被涓?`test_white_box_allowed`銆?
parallel/stale implementation gate 鎵弿 legacy銆乧ompat銆乨eprecated銆乵igration銆乷ld銆乶ew銆乿2銆乻him銆乫allback銆亀rapper 鍜?alias銆傜湡闂蹇呴』鍚堝苟鎴栧垹闄わ紱valuable deferred feature 蹇呴』鏈?owner銆乺ationale銆乴ifecycle 鍜?reference銆?
娴嬭瘯浣嶇疆 gate锛?
- architecture boundary tests 鏀?`tests/architecture/`銆?- handoff builder/verifier/evidence tests 鏀?`tests/agents/handoff/`銆?- receipt payload invariant 鏀?`tests/agents/handoff/test_*_receipt_contract.py`
  鍜?`tests/agents/handoff/test_validation_summary_contract.py`銆?- full zip mutation 鏀?`tests/agents/handoff/test_handoff_package_mutations.py`銆?- handoff single-source ownership 鏀?`tests/architecture/test_handoff_contract_single_source.py`銆?- source-tree retired-name gate 涓嶆斁鍦?`tests/agents/handoff/`銆?- governance CLI/瑙勫垯琛屼负 tests 鏀?`tests/agents/`銆?- A 鑲?processed owner behavior tests 鏀?`tests/markets/a_share/`銆?- research behavior tests 鏀?`tests/research/`銆?- artifact/storage/compute/workflow behavior tests 鏀惧搴?owner 鐩綍銆?
A 鑲?owner-local naming gate 绂佹 `markets/implementations/a_share/**` 鍐呴儴 symbol銆佸父閲忋€佹枃浠跺悕鍜?error code 閲嶅 `a_share` / `ashare` / `china_a` / `china_share` 鍓嶇紑锛涘瓧绗︿覆 id 闇€瑕佸垎绫伙紝涓嶆満姊板垹闄ゃ€?
鍏抽敭娴嬭瘯鍏ュ彛锛?
```text
tests/agents/test_verify_governance_active_docs.py
tests/agents/test_verify_governance_changed_files.py
tests/agents/test_verify_governance_cli.py
tests/architecture/test_governance_scan_boundaries.py
tests/architecture/test_governance_active_docs.py
tests/architecture/test_governance_profile_runner_contract.py
tests/architecture/test_governance_source_tree_scan_contract.py
tests/architecture/test_share_safe_evidence.py
tests/architecture/test_share_safe_evidence_data_contract.py
tests/architecture/test_platform_topology_compute_module.py
tests/architecture/test_platform_topology_csmar_source.py
tests/architecture/test_platform_topology_platform_package.py
tests/architecture/test_platform_topology_public_interface.py
tests/contracts/test_platform_contracts.py
```

handoff 娴嬭瘯鍒嗗眰鍏ュ彛锛?
```bash
conda run -n <environment.yml name> python tests/tools/run_profile.py --list-profiles
conda run -n <environment.yml name> python tests/tools/run_profile.py --profile <profile> --run-dir .run/<purpose>/<profile>
```

## 6. Final report schema

鏈€缁堟姤鍛婅嚦灏戝寘鍚細

```text
baseline
changed files
deleted files
commands run
status table
validation artifacts
remaining issues
working tree status
```
