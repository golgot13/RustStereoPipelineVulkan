# RustStereoPipelineVulkan v0.2.0

Plateforme de stéréophotogrammétrie modulaire, scriptable et légère inspirée de **NASA Ames Stereo Pipeline (ASP)** et **AEROPHOTOGRAM**, développée en **Rust** avec moteur **Vulkan Compute** (SPIR-V 1.0 généré en pur Rust), **CPU SIMD AVX2 sécurisé**, **CLI scriptable**, **Studio Desktop `egui`**, **X11 natif** et **Web UI SSE**.

> **Légèreté préservée** : le cœur complet (Vulkan Compute, SPIR-V, DAG/Checkpoints, JSON, géodésie `f64`, tuilage 1:1, CLI, X11 natif et Web UI SSE) compile avec **0 crate externe** via `--no-default-features`. Seule la feature optionnelle `egui-ui` (activée par défaut) ajoute `eframe`.

## Points forts techniques

- **Contrat unique `JobSpec` & Parité 100 % CLI ↔ Desktop `egui` ↔ X11 ↔ Web UI** (`src/core/job.rs`, `src/core/json.rs`) :
  - Sérialisation/désérialisation JSON sans crate (`--job mission.json`, `stdin` via `--job -`, `--export-job`, `batch`).
  - Génération de la commande CLI exacte depuis `egui` et la Web UI (*Copy CLI Cmd* / *Download job.json*).
  - Flux d'événements machine-readable **`--ndjson`** en CLI et **Server-Sent Events (`GET /api/events`)** en Web UI.
- **Robustesse Industrielle (« 0 Panic »)** (`src/core/error.rs`, `src/core/cancel.rs`, `src/core/hardware.rs`) :
  - Codes de retour POSIX normalisés (`0`, `2` CLI/Config, `3` I/O, `4` Compute/RAM, `5` Géométrie, `130` Annulé).
  - `PreflightValidator` (vérification dimensions, dynamique, baseline/convergence caméras, budget RAM).
  - `CancellationToken` coopératif (bouton *Stop* en `egui`, `POST /api/cancel` en Web UI).
  - `HardwareProfile` auto-détecté (`compact` / `standard` / `workstation`) pour Linux et Windows.
- **Moteur DAG Modulaire, Tuilage Full-Scale 1:1 & Checkpoints** (`src/pipeline/dag.rs`, `src/pipeline/tiling.rs`) :
  - Exécution sélective (`--from-stage`, `--to-stage`), reprise sur incident (`--resume` via `00_checkpoints/`), rapport qualité automatique (`-quality_report.json` & `.txt`).
  - Découpage transparent en **tuiles 1:1 avec halo** (sans décimation destructive) pour traiter de grands capteurs en RAM bornée.
- **Performance Vulkan Compute + CPU AVX2** (`src/vulkan/`, `src/pipeline/simd.rs`) :
  - Chargeur dynamique multiplateforme (`vulkan-1.dll` Windows / `libvulkan.so.1` Linux / `libvulkan.1.dylib` macOS).
  - **Cache persistant de pipelines SPIR-V 1.0** (5 shaders compute validés : `prefilter_box_submean`, `prefilter_log_2d`, `stereo_bm_sad_ssd_ncc`, `subpixel_parabola`, `dem_hillshade`).
  - Noyaux CPU **AVX2** détectés dynamiquement (`is_x86_feature_detected!("avx2")`) avec fallback scalaire sûr (zéro `SIGILL`).
- **Géodésie Double Précision `f64`** (`src/core/crs.rs`) :
  - Reprojection bidirectionnelle `WGS84 (EPSG:4326)`, `ECEF (EPSG:4978)`, **Lambert-93 (`EPSG:2154`)**, **UTM (`EPSG:326xx/327xx`)**, `Moon`, `Mars`, `Local`.

## Compilation

```bash
# Avec le Studio Desktop egui (par défaut)
cargo build --release
cargo test

# Mode 100 % zéro dépendance externe (CLI + Vulkan Compute + X11 natif + Web UI SSE)
cargo build --release --no-default-features
cargo test --no-default-features
```

## Utilisation CLI & Scriptabilité

```bash
# Diagnostic HardwareProfile, AVX2, Vulkan Compute & validation des 5 shaders SPIR-V 1.0
cargo run --release -- vulkan-info

# Terrain synthétique + tuilage 1:1 + rapport qualité + flux NDJSON + export du JobSpec
cargo run --release -- synthetic output/demo --width 192 --height 144 \
  --stereo-algorithm asp_mgm --tile-size 128 --export-job output/demo_job.json --ndjson

# Exécution d'un JobSpec JSON (fichier ou stdin) avec reprise sur checkpoints
cargo run --release -- run --job output/demo_job.json --resume --ndjson
cat output/demo_job.json | cargo run --release -- run --job - --ndjson

# Traitement par lots (batch) d'un dossier de manifests JSON
cargo run --release -- batch output/jobs/ --continue-on-error --ndjson

# Reprojection géodésique f64 (ex. WGS84 -> Lambert-93 EPSG:2154)
cargo run --release -- list-crs
cargo run --release -- crs-convert --src EPSG:4326 --dst EPSG:2154 --x 2.3499 --y 48.8530 --z 35.0
```

## Interfaces Graphiques (Desktop & Web)

```bash
# Studio Desktop egui (recommandé sur Windows / Linux / macOS)
cargo run --release -- egui

# Lanceur hybride auto (egui -> X11 natif -> Web UI)
cargo run --release -- gui

# Web UI embarquée HTTP/1.1 + SSE temps réel (0 crate externe)
cargo run --release -- web-ui --port 8080
```
