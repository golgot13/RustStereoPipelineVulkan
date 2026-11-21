# RustStereoPipelineVulkan

Réécriture complète en **Rust pur (0 crate externe)** et optimisée **Vulkan Compute** du pipeline de stéréophotogrammétrie inspiré de **NASA Ames Stereo Pipeline (ASP)**, dotée d'une **interface graphique hybride (X11 natif + Web UI embarquée)** et d'un **CLI complet**.

## Points forts techniques

- **Zéro dépendance externe (`[dependencies]` vide)** : utilise exclusivement la bibliothèque standard Rust (`std`) + FFI dynamique (`dlopen`/`dlsym`) vers `libvulkan.so.1` et `libX11.so.6`.
- **Moteur Vulkan 1.0 Compute + Assembleur SPIR-V 1.0 pur Rust** (`src/vulkan/`):
  - Charge dynamiquement `libvulkan.so.1` / `libvulkan.1.dylib` sans dépendance au SDK Vulkan à la compilation.
  - Génère et valide à la volée de véritables shaders compute **SPIR-V 1.0** (`prefilter`, `stereo_bm` SAD/SSD/NCC, `subpixel_parabola`, `dem_hillshade`) sans `glslc`/`shaderc`.
  - Bascule automatiquement sur un moteur **CPU multi-thread** (`std::thread::scope`) lorsqu'aucun pilote/GPU Vulkan n'est disponible.
- **Pipeline Stéréo ASP complet en 5 étapes + DEM + ICP** (`src/pipeline/`):
  - `Stage 0 — stereo_pprc` : normalisation d'intensité, *Subtracted Mean* (Vulkan GPU / image intégrale $O(1)$), *Laplacian of Gaussian* (LoG).
  - `Stage 1 — stereo_corr` : corrélation stéréo **ASP BM**, **ASP SGM** (programmation dynamique multi-chemins) et **ASP MGM** (agrégation par quadrants), métriques **SAD / SSD / NCC / Census 5×5** + vérification gauche-droite (LR check).
  - `Stage 2 — stereo_rfne` : raffinement subpixel **Parabola**, **Affine Lucas-Kanade** et **Bayes-EM pondéré**.
  - `Stage 3 — stereo_fltr` : érosion des valeurs aberrantes (`rm-half-kernel`, `rm-threshold`, `rm-min-matches`, `max-mean-diff`), filtre médian et comblement des trous.
  - `Stage 4 — stereo_tri` : triangulation 3D rayon-rayon (modèles caméras **Pinhole `.tsai`** avec distorsion Brown-Conrady et **RPC `.rpb`** satellitaire) + export **PLY / XYZ**.
  - `point2dem` & `pc_align` : génération de **DEM GeoTIFF**, ombrage lambertien (*hillshade* Vulkan/CPU) et recalage 3D **ICP Kabsch SVD**.
- **I/O scientifique sans crate** (`src/image/io.rs`) : lecture/écriture **TIFF/GeoTIFF** (8/16/32f + tags géospatiaux), **PFM**, **PGM/PPM**, **BMP**, encodeur **PNG** pur Rust, **PLY**, **XYZ/CSV**.
- **Interface Graphique Hybride sans crate** (`src/ui/`):
  - **Fenêtre native X11** (`libX11.so.6` via `dlopen`) avec moteur de rendu logiciel BGRA32, police bitmap 8×8, contrôles interactifs, palettes scientifiques (*Turbo, Viridis, Inferno, Terrain, Grayscale*), zoom/pan et **vue 3D interactive du DEM**.
  - **Web UI embarquée** (`std::net::TcpListener` HTTP/1.1 + SPA HTML5/Canvas/3D DEM) accessible partout, y compris en headless/SSH.

## Compilation

```bash
cargo build --release
cargo test
```

## Utilisation CLI

```bash
# Diagnostic Vulkan Compute & validation des shaders SPIR-V 1.0
cargo run --release -- vulkan-info

# Génération d'un terrain planétaire synthétique + pipeline complet + évaluation RMSE
cargo run --release -- synthetic output/demo --width 192 --height 144 --stereo-algorithm asp_sgm

# Pipeline complet sur vos images et caméras (.tsai ou .rpb)
cargo run --release -- stereo left.tif right.tif left.tsai right.tsai output/run \
  -s stereo.default.example --stereo-algorithm asp_sgm --subpixel-mode 2 --backend auto

# Recalage 3D ICP de deux nuages de points
cargo run --release -- pc_align ref.xyz src.xyz --max-iters 40

# Conversion nuage de points -> DEM GeoTIFF + Hillshade
cargo run --release -- point2dem cloud.xyz output/dem_out

# Prévisualisation colorisée PNG d'une disparité ou d'un DEM
cargo run --release -- disparitydebug output/demo-DEM.tif output/preview --cmap turbo
```

## Utilisation Interface Graphique (GUI)

```bash
# Mode hybride auto (X11 natif si $DISPLAY est présent, sinon Web UI embarquée)
cargo run --release -- gui

# Forcer la fenêtre native X11
cargo run --release -- gui --x11

# Lancer directement la Web UI embarquée sur le port 8080
cargo run --release -- web-ui --port 8080
```
