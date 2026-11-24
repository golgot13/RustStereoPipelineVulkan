//! Unified, scriptable ASP-compatible CLI dispatcher supporting:
//! - `stereo`, `synthetic`, `run --job <file|->`, `batch <dir|files...>`
//! - `stereo_pprc`, `point2dem`, `pc_align`, `disparitydebug`, `crs-convert`, `vulkan-info`
//! - `egui`, `gui`, `web-ui`
//! - `--ndjson`, `--resume`, `--from-stage`, `--to-stage`, `--tile-size`, `--crs`, `--profile`, `--export-job`

use std::io::Read;
use std::path::{Path, PathBuf};

use crate::camera::AnyCamera;
use crate::config::{
    ComputeBackendPreference, CostMode, PrefilterMode, StereoAlgorithm, StereoSettings, SubpixelMode,
};
use crate::core::cancel::CancellationToken;
use crate::core::chantier::{
    chantier_summary_json, prepare_stereo_pair_roi, validate_dem_against_gcps, Chantier,
};
use crate::core::crs::{transform_point, CrsId};
use crate::core::error::{PipelineError, Result};
use crate::core::hardware::{HardwareProfile, ProfileTier};
use crate::core::job::{JobInput, JobSpec, StageId};
use crate::image::io::{
    encode_png_rgba, load_image, load_point_cloud_xyz, save_image, save_tiff_f32, GeoTransform,
};
use crate::image::{Colormap, PointCloud};
use crate::math::Vec3;
use crate::pipeline::dag::{WorkflowEvent, WorkflowExecutor};
use crate::pipeline::{align, dem, pprc, synthetic, StereoPipelineEngine};
use crate::ui::{launch_gui, UiMode};
use crate::vulkan::shaders::shader_catalog;
use crate::vulkan::spirv::validate_spirv;
use crate::vulkan::VulkanCompute;

pub fn run_cli(args: &[String]) -> Result<()> {
    if args.len() < 2 || matches!(args[1].as_str(), "-h" | "--help" | "help") {
        print_help();
        return Ok(());
    }

    let cmd = args[1].as_str();
    let rest = &args[2..];

    match cmd {
        "stereo" => cmd_stereo(rest),
        "chantier" => cmd_chantier(rest),
        "run" => cmd_run_job(rest),
        "batch" => cmd_batch(rest),
        "stereo_pprc" => cmd_pprc(rest),
        "point2dem" => cmd_point2dem(rest),
        "pc_align" => cmd_pc_align(rest),
        "disparitydebug" => cmd_disparitydebug(rest),
        "crs-convert" => cmd_crs_convert(rest),
        "--list-crs" | "list-crs" => {
            cmd_list_crs();
            Ok(())
        }
        "synthetic" => cmd_synthetic(rest),
        "vulkan-info" => cmd_vulkan_info(),
        "gui" | "stereo_gui" => cmd_gui(rest, UiMode::Auto),
        "egui" => cmd_gui(rest, UiMode::Egui),
        "web-ui" => cmd_gui(rest, UiMode::Web),
        other => Err(PipelineError::Cli(format!(
            "Unknown subcommand '{}'. Run with --help to see available commands.",
            other
        ))),
    }
}

fn print_help() {
    println!(
        r#"RustStereoPipelineVulkan v0.2.0 — Modular Rust + Vulkan Compute Stereo Pipeline

USAGE:
  stereo-pipeline <COMMAND> [ARGS...] [OPTIONS...]

COMMANDS:
  chantier <chantier_dir> [--output <dir>] [--pair N | --all-pairs] [--roi-size 1024]
      Auto-discover .CAM, trajectory .TXT, GCP, and GeoTIFF stereo pairs (e.g. Meyreuil DMC-IV 433Mpx)
      and run streaming Full-Scale 1:1 stereo reconstruction into Output_stereopipeline
  stereo <L.tif> <R.tif> <L.tsai|rpb> <R.tsai|rpb> <out_prefix>
      Run full 5-stage DAG Stereo Pipeline (pprc -> corr -> rfne -> fltr -> tri -> point2dem)
  run --job <mission.json | ->
      Execute a JobSpec JSON manifest (or read from stdin with `-`)
  batch <dir_or_job1.json ...> [--continue-on-error]
      Execute all *.json JobSpec manifests sequentially
  synthetic <out_prefix> [--width W] [--height H]
      Generate synthetic planetary crater terrain, run DAG pipeline & evaluate RMSE vs ground truth
  stereo_pprc <L.tif> <R.tif> <out_prefix>
      Run Stage 0 intensity normalization & prefiltering (LoG / Subtracted Mean)
  point2dem <cloud.xyz> <out_prefix>
      Rasterize a 3D XYZ/CSV point cloud to a GeoTIFF DEM and hillshade
  pc_align <ref.xyz> <src.xyz>
      Align two 3D point clouds via rigid Kabsch ICP
  disparitydebug <input.tif|pfm> <out_prefix>
      Render a colorized PNG preview of a raster (disparity/DEM/error)
  crs-convert --src <CRS> --dst <CRS> --x <X> --y <Y> --z <Z>
      Reproject a 3D point in double precision f64 (WGS84, ECEF, Lambert-93, CC42..CC50, UTM)
  list-crs
      List supported Coordinate Reference Systems
  vulkan-info
      Probe Vulkan Compute hardware devices, HardwareProfile/AVX2, and validate SPIR-V 1.0 shaders
  egui [L.tif R.tif L.tsai R.tsai] [options]
      Launch the interactive egui/eframe Desktop Studio
  gui [L.tif R.tif L.tsai R.tsai] [--egui | --x11 | --web] [--port 8080]
      Launch Hybrid GUI (egui Studio, Native X11 window, or Embedded Web UI)
  web-ui [L.tif R.tif L.tsai R.tsai] [--port 8080]
      Launch Embedded HTTP/1.1 + SSE Web UI workstation directly

SCRIPTING & ROBUSTNESS OPTIONS:
  --job <file.json | ->          Load JobSpec JSON manifest (`-` reads stdin)
  --export-job <file.json>       Export resolved JobSpec manifest to JSON
  --ndjson                       Stream machine-readable NDJSON progress events to stdout
  --resume                       Resume from valid 00_checkpoints/ matching settings hash
  --from-stage <stage>           Start stage: pprc|corr|rfne|fltr|tri|dem (0..5)
  --to-stage <stage>             End stage:   pprc|corr|rfne|fltr|tri|dem (0..5)
  --tile-size <N>                Full-Scale 1:1 sliding-window tile size (default: auto)
  --profile <tier>               Hardware RAM tier: compact | standard | workstation
  --crs <code|name>              Output CRS: LOCAL | EPSG:4326 | EPSG:2154 | EPSG:3942..3950 (CC42..CC50) | EPSG:32631 | IAU:MOON | IAU:MARS
  --continue-on-error            In batch mode, continue running remaining jobs on failure

STEREO OPTIONS:
  -s, --stereo-file <file>       Load ASP stereo.default configuration file
  --stereo-algorithm <alg>       asp_bm | asp_sgm | asp_mgm (default: asp_sgm)
  --cost-mode <0|1|2|3>          0=SAD, 1=SSD, 2=NCC, 3=Census (default: 2)
  --prefilter-mode <0|1|2>       0=None, 1=SubMean, 2=LoG (default: 2)
  --prefilter-kernel-width <f>   Prefilter sigma/width (default: 1.4)
  --corr-kernel <W> <H>          Correlation patch size (default: 9 9)
  --corr-search <minX minY maxX maxY>  Disparity search box
  --subpixel-mode <0|1|2|3>      0=None, 1=Parabola, 2=BayesEM, 3=Affine (default: 1)
  --backend <auto|vulkan|cpu>    Compute backend selection (default: auto)
  --threads <N>                  CPU worker threads
  --cmap <name>                  Colormap: turbo | viridis | inferno | terrain | grayscale
"#
    );
}

struct ParsedArgs {
    positionals: Vec<String>,
    settings: StereoSettings,
    width: u32,
    height: u32,
    port: u16,
    ui_mode: Option<UiMode>,
    cmap: Colormap,
    max_iters: usize,
    max_disp: f64,
    ndjson: bool,
    resume: bool,
    from_stage: StageId,
    to_stage: StageId,
    tile_size: Option<u32>,
    crs: CrsId,
    profile: Option<ProfileTier>,
    job_path: Option<String>,
    export_job: Option<String>,
    continue_on_error: bool,
    src_crs: Option<String>,
    dst_crs: Option<String>,
    coord_x: f64,
    coord_y: f64,
    coord_z: f64,
    output_dir: Option<String>,
    pair_idx: Option<usize>,
    all_pairs: bool,
    roi_size: u32,
}

fn parse_common_args(args: &[String]) -> Result<ParsedArgs> {
    let mut settings = StereoSettings::default();
    let mut positionals = Vec::new();
    let mut width = 160u32;
    let mut height = 120u32;
    let mut port = 8080u16;
    let mut ui_mode = None;
    let mut cmap = Colormap::Turbo;
    let mut max_iters = 40usize;
    let mut max_disp = 100.0f64;
    let mut ndjson = false;
    let mut resume = false;
    let mut from_stage = StageId::Pprc;
    let mut to_stage = StageId::Dem;
    let mut tile_size = None;
    let mut crs = CrsId::Local;
    let mut profile = None;
    let mut job_path = None;
    let mut export_job = None;
    let mut continue_on_error = false;
    let mut src_crs = None;
    let mut dst_crs = None;
    let mut coord_x = 0.0;
    let mut coord_y = 0.0;
    let mut coord_z = 0.0;
    let mut output_dir = None;
    let mut pair_idx = None;
    let mut all_pairs = false;
    let mut roi_size = 1024u32;

    let mut i = 0;
    while i < args.len() {
        let a = args[i].as_str();
        let next = |idx: &mut usize, flag: &str| -> Result<&str> {
            *idx += 1;
            args.get(*idx)
                .map(|s| s.as_str())
                .ok_or_else(|| PipelineError::Cli(format!("Missing value after {}", flag)))
        };
        match a {
            "-s" | "--stereo-file" => {
                let path = next(&mut i, a)?;
                settings = StereoSettings::load_file(Path::new(path))?;
            }
            "--stereo-algorithm" => settings.stereo_algorithm = StereoAlgorithm::from_str(next(&mut i, a)?),
            "--cost-mode" => {
                let v: i32 = next(&mut i, a)?.parse().map_err(|_| PipelineError::Cli("Invalid --cost-mode".into()))?;
                settings.cost_mode = CostMode::from_int(v);
            }
            "--prefilter-mode" => {
                let v: i32 = next(&mut i, a)?.parse().map_err(|_| PipelineError::Cli("Invalid --prefilter-mode".into()))?;
                settings.prefilter_mode = PrefilterMode::from_int(v);
            }
            "--prefilter-kernel-width" => {
                settings.prefilter_kernel_width = next(&mut i, a)?.parse().map_err(|_| PipelineError::Cli("Invalid kernel width".into()))?;
            }
            "--corr-kernel" => {
                if i + 2 >= args.len() {
                    return Err(PipelineError::Cli("--corr-kernel requires 2 integers".into()));
                }
                settings.corr_kernel_x = args[i + 1].parse().map_err(|_| PipelineError::Cli("Invalid corr-kernel X".into()))?;
                settings.corr_kernel_y = args[i + 2].parse().map_err(|_| PipelineError::Cli("Invalid corr-kernel Y".into()))?;
                i += 2;
            }
            "--corr-search" => {
                if i + 4 >= args.len() {
                    return Err(PipelineError::Cli("--corr-search requires 4 integers".into()));
                }
                settings.search_min_x = args[i + 1].parse().map_err(|_| PipelineError::Cli("Invalid search minX".into()))?;
                settings.search_min_y = args[i + 2].parse().map_err(|_| PipelineError::Cli("Invalid search minY".into()))?;
                settings.search_max_x = args[i + 3].parse().map_err(|_| PipelineError::Cli("Invalid search maxX".into()))?;
                settings.search_max_y = args[i + 4].parse().map_err(|_| PipelineError::Cli("Invalid search maxY".into()))?;
                i += 4;
            }
            "--subpixel-mode" => {
                let v: i32 = next(&mut i, a)?.parse().map_err(|_| PipelineError::Cli("Invalid --subpixel-mode".into()))?;
                settings.subpixel_mode = SubpixelMode::from_int(v);
            }
            "--backend" => settings.backend = ComputeBackendPreference::from_str(next(&mut i, a)?),
            "--threads" => {
                settings.threads = next(&mut i, a)?.parse().map_err(|_| PipelineError::Cli("Invalid --threads".into()))?;
            }
            "--width" => {
                width = next(&mut i, a)?.parse().map_err(|_| PipelineError::Cli("Invalid --width".into()))?;
            }
            "--height" => {
                height = next(&mut i, a)?.parse().map_err(|_| PipelineError::Cli("Invalid --height".into()))?;
            }
            "--port" => {
                port = next(&mut i, a)?.parse().map_err(|_| PipelineError::Cli("Invalid --port".into()))?;
            }
            "--egui" => ui_mode = Some(UiMode::Egui),
            "--x11" => ui_mode = Some(UiMode::X11),
            "--web" => ui_mode = Some(UiMode::Web),
            "--cmap" => cmap = Colormap::from_name(next(&mut i, a)?),
            "--max-iters" => {
                max_iters = next(&mut i, a)?.parse().map_err(|_| PipelineError::Cli("Invalid --max-iters".into()))?;
            }
            "--max-displacement" => {
                max_disp = next(&mut i, a)?.parse().map_err(|_| PipelineError::Cli("Invalid --max-displacement".into()))?;
            }
            "--ndjson" | "--json" => ndjson = true,
            "--resume" => resume = true,
            "--from-stage" => from_stage = StageId::parse(next(&mut i, a)?)?,
            "--to-stage" => to_stage = StageId::parse(next(&mut i, a)?)?,
            "--tile-size" => {
                let ts: u32 = next(&mut i, a)?.parse().map_err(|_| PipelineError::Cli("Invalid --tile-size".into()))?;
                tile_size = Some(ts.clamp(64, 4096));
            }
            "--crs" => crs = CrsId::parse(next(&mut i, a)?)?,
            "--profile" => {
                let pr = next(&mut i, a)?;
                profile = Some(ProfileTier::from_str(pr).ok_or_else(|| {
                    PipelineError::Cli(format!("Invalid profile '{}' (compact|standard|workstation)", pr))
                })?);
            }
            "--job" => job_path = Some(next(&mut i, a)?.to_string()),
            "--export-job" => export_job = Some(next(&mut i, a)?.to_string()),
            "--continue-on-error" => continue_on_error = true,
            "--src" | "--src-crs" => src_crs = Some(next(&mut i, a)?.to_string()),
            "--dst" | "--dst-crs" => dst_crs = Some(next(&mut i, a)?.to_string()),
            "--x" => coord_x = next(&mut i, a)?.parse().map_err(|_| PipelineError::Cli("Invalid --x".into()))?,
            "--y" => coord_y = next(&mut i, a)?.parse().map_err(|_| PipelineError::Cli("Invalid --y".into()))?,
            "--z" => coord_z = next(&mut i, a)?.parse().map_err(|_| PipelineError::Cli("Invalid --z".into()))?,
            "-o" | "--output" => output_dir = Some(next(&mut i, a)?.to_string()),
            "--pair" => {
                pair_idx = Some(next(&mut i, a)?.parse().map_err(|_| PipelineError::Cli("Invalid --pair index".into()))?);
            }
            "--all-pairs" => all_pairs = true,
            "--roi-size" => {
                roi_size = next(&mut i, a)?.parse::<u32>().map_err(|_| PipelineError::Cli("Invalid --roi-size".into()))?.clamp(128, 4096);
            }
            "--benchmark" => {}
            other if other.starts_with('-') && other != "-" => {
                return Err(PipelineError::Cli(format!("Unknown option: {}", other)));
            }
            pos => positionals.push(pos.to_string()),
        }
        i += 1;
    }

    Ok(ParsedArgs {
        positionals,
        settings,
        width,
        height,
        port,
        ui_mode,
        cmap,
        max_iters,
        max_disp,
        ndjson,
        resume,
        from_stage,
        to_stage,
        tile_size,
        crs,
        profile,
        job_path,
        export_job,
        continue_on_error,
        src_crs,
        dst_crs,
        coord_x,
        coord_y,
        coord_z,
        output_dir,
        pair_idx,
        all_pairs,
        roi_size,
    })
}

fn build_spec(p: &ParsedArgs, input: JobInput, out_prefix: String) -> JobSpec {
    let hw = match p.profile {
        Some(t) => HardwareProfile::from_tier_auto(t),
        None => HardwareProfile::detect(),
    };
    let mut spec = JobSpec::default();
    spec.input = input;
    spec.output_prefix = out_prefix;
    spec.crs = p.crs;
    spec.from_stage = p.from_stage;
    spec.to_stage = p.to_stage;
    spec.resume = p.resume;
    spec.tile_size = p.tile_size.unwrap_or(hw.tile_size);
    spec.profile = p.profile;
    spec.settings = p.settings.clone();
    spec
}

fn emit_event(ev: &WorkflowEvent, ndjson: bool) {
    if ndjson {
        println!("{}", ev.to_ndjson());
    } else {
        match ev {
            WorkflowEvent::Started { width, height, tiles, backend } => {
                println!("Job Started: {}x{} ({} Full-Scale 1:1 tiles) | {}", width, height, tiles, backend);
            }
            WorkflowEvent::PreflightWarning { message } => {
                eprintln!("[WARN] {}", message);
            }
            WorkflowEvent::StageStart { label, pct, .. } => {
                println!("[{:3.0}%] {}", pct * 100.0, label);
            }
            WorkflowEvent::TileProgress { tile, total, .. } if *total > 1 => {
                println!("       -> Tile {}/{}", tile, total);
            }
            WorkflowEvent::CheckpointSkipped { stage } => {
                println!("[SKIP] Resumed valid checkpoint for stage '{}'", stage.name());
            }
            _ => {}
        }
    }
}

fn cmd_stereo(args: &[String]) -> Result<()> {
    let p = parse_common_args(args)?;
    if p.positionals.len() < 5 {
        return Err(PipelineError::Cli(
            "Usage: stereo <L.tif> <R.tif> <L.tsai|rpb> <R.tsai|rpb> <out_prefix>".into(),
        ));
    }
    let spec = build_spec(
        &p,
        JobInput::Files {
            left_image: p.positionals[0].clone(),
            right_image: p.positionals[1].clone(),
            left_camera: p.positionals[2].clone(),
            right_camera: p.positionals[3].clone(),
        },
        p.positionals[4].clone(),
    );
    if let Some(ref ej) = p.export_job {
        spec.save_file(Path::new(ej))?;
    }
    let engine = StereoPipelineEngine::new(spec.settings.backend);
    let cancel = CancellationToken::new();
    let out = WorkflowExecutor::execute_job(&spec, &engine, &cancel, |ev| emit_event(&ev, p.ndjson))?;
    if !p.ndjson {
        println!("\n{}", out.report.to_text());
    }
    Ok(())
}

fn cmd_chantier(args: &[String]) -> Result<()> {
    let p = parse_common_args(args)?;
    let chantier_dir = p
        .positionals
        .first()
        .ok_or_else(|| {
            PipelineError::Cli(
                "Usage: chantier <chantier_dir> [--output <dir>] [--pair N | --all-pairs] [--roi-size 1024]".into(),
            )
        })?;

    let root_path = Path::new(chantier_dir);
    let out_dir = p
        .output_dir
        .as_ref()
        .map(PathBuf::from)
        .unwrap_or_else(|| root_path.join("Output_stereopipeline"));
    std::fs::create_dir_all(&out_dir)?;

    let chantier = Chantier::discover(root_path)?;
    if !p.ndjson {
        println!("=== Chantier Aérien Découvert ===");
        println!("  Répertoire   : {}", chantier.input_dir.display());
        println!(
            "  Caméra .CAM  : {} ({}x{} px, f={:.2} mm, pitch={:.4} mm)",
            chantier.calibration.name,
            chantier.calibration.width_px,
            chantier.calibration.height_px,
            chantier.calibration.focal_mm,
            chantier.calibration.pixel_size_mm
        );
        println!("  Trajectoire  : {} ({} poses)", chantier.trajectory_file.display(), chantier.poses.len());
        println!("  GCP Terrain  : {} points", chantier.gcps.len());
        println!("  Paires Stéréo: {} paires détectées par ligne de vol", chantier.pairs.len());
        println!("  Sortie       : {}", out_dir.display());
    }

    if chantier.pairs.is_empty() {
        return Err(PipelineError::Geometry("No consecutive stereo pairs detected in chantier".into()));
    }

    let selected: Vec<usize> = if p.all_pairs {
        (0..chantier.pairs.len()).collect()
    } else {
        let idx = p.pair_idx.unwrap_or(0);
        if idx >= chantier.pairs.len() {
            return Err(PipelineError::Cli(format!(
                "Pair index {} out of range (0..{})",
                idx,
                chantier.pairs.len()
            )));
        }
        vec![idx]
    };

    let engine = StereoPipelineEngine::new(p.settings.backend);
    let cancel = CancellationToken::new();
    let mut pair_summaries = Vec::new();

    for idx in selected {
        let pair = &chantier.pairs[idx];
        let pair_dir = out_dir.join(&pair.name);
        std::fs::create_dir_all(&pair_dir)?;
        let out_prefix = pair_dir.join(&pair.name).to_string_lossy().into_owned();

        if !p.ndjson {
            println!(
                "\n--- Paire {}/{} : {} (baseline={:.1} m, ROI 1:1 {}x{}) ---",
                idx + 1,
                chantier.pairs.len(),
                pair.name,
                pair.baseline_m,
                p.roi_size,
                p.roi_size
            );
            println!("  Left  : {}", pair.left_image_path.display());
            println!("  Right : {}", pair.right_image_path.display());
        }

        let roi = prepare_stereo_pair_roi(&chantier, pair, p.roi_size)?;
        if !p.ndjson {
            println!(
                "  Capteur natif : {}x{} px | ROI L=({},{}) R=({},{}) | Search X=[{}..{}] Y=[{}..{}]",
                roi.full_dims.0,
                roi.full_dims.1,
                roi.left_roi.0,
                roi.left_roi.1,
                roi.right_roi.0,
                roi.right_roi.1,
                roi.search_box.0,
                roi.search_box.2,
                roi.search_box.1,
                roi.search_box.3
            );
        }

        // Export cropped ASP .tsai cameras & ROI GeoTIFFs for full reproducibility
        if let AnyCamera::Pinhole(ref cl) = roi.left_cam {
            cl.save_tsai(Path::new(&format!("{}-cam-L.tsai", out_prefix)))?;
        }
        if let AnyCamera::Pinhole(ref cr) = roi.right_cam {
            cr.save_tsai(Path::new(&format!("{}-cam-R.tsai", out_prefix)))?;
        }

        let mut spec = build_spec(
            &p,
            JobInput::Files {
                left_image: pair.left_image_path.to_string_lossy().into_owned(),
                right_image: pair.right_image_path.to_string_lossy().into_owned(),
                left_camera: format!("{}-cam-L.tsai", out_prefix),
                right_camera: format!("{}-cam-R.tsai", out_prefix),
            },
            out_prefix.clone(),
        );
        spec.settings.search_min_x = roi.search_box.0;
        spec.settings.search_min_y = roi.search_box.1;
        spec.settings.search_max_x = roi.search_box.2;
        spec.settings.search_max_y = roi.search_box.3;
        spec.save_file(Path::new(&format!("{}-job.json", out_prefix)))?;

        let out = WorkflowExecutor::execute_in_memory(
            &spec,
            &roi.left_img,
            &roi.right_img,
            &roi.left_cam,
            &roi.right_cam,
            None,
            &engine,
            &cancel,
            |ev| emit_event(&ev, p.ndjson),
        )?;

        // Export colorized PNG previews for instant visual inspection
        let dem_png = encode_png_rgba(
            out.result.dem.elevation.width,
            out.result.dem.elevation.height,
            &out.result.dem.elevation.to_rgba8(p.cmap),
        );
        std::fs::write(format!("{}-DEM-preview.png", out_prefix), dem_png)?;
        let hs_png = encode_png_rgba(
            out.result.dem.hillshade.width,
            out.result.dem.hillshade.height,
            &out.result.dem.hillshade.to_rgba8(Colormap::Grayscale),
        );
        std::fs::write(format!("{}-HILLSHADE-preview.png", out_prefix), hs_png)?;
        let disp_img = out.result.disp_filtered.dx_image();
        let disp_png = encode_png_rgba(disp_img.width, disp_img.height, &disp_img.to_rgba8(p.cmap));
        std::fs::write(format!("{}-DISP-preview.png", out_prefix), disp_png)?;

        let gcp_val = validate_dem_against_gcps(&out.result.dem, &chantier.gcps);
        if !p.ndjson {
            println!("\n{}", out.report.to_text());
            if gcp_val.matched_gcps > 0 {
                println!(
                    "  Validation GCP ({} points dans le DEM) : biais dZ={:+.3} m | RMSE dZ={:.3} m | max={:.3} m",
                    gcp_val.matched_gcps, gcp_val.dz_bias_m, gcp_val.dz_rmse_m, gcp_val.dz_max_abs_m
                );
            }
        }
        pair_summaries.push((
            pair.name.clone(),
            out.result.timings.total_ms,
            out.report.valid_disparity_pct,
            out.report.valid_points,
            gcp_val,
        ));
    }

    let summary_path = out_dir.join("chantier_summary.json");
    std::fs::write(&summary_path, chantier_summary_json(&chantier, &pair_summaries))?;
    if !p.ndjson {
        println!("\n=== Chantier terminé — Résumé enregistré dans {} ===", summary_path.display());
    }
    Ok(())
}

fn cmd_run_job(args: &[String]) -> Result<()> {
    let p = parse_common_args(args)?;
    let job_src = p
        .job_path
        .as_deref()
        .or_else(|| p.positionals.first().map(|s| s.as_str()))
        .ok_or_else(|| PipelineError::Cli("Usage: run --job <mission.json | -> [--ndjson] [--resume]".into()))?;

    let mut spec = if job_src == "-" {
        let mut buf = String::new();
        std::io::stdin().read_to_string(&mut buf)?;
        JobSpec::from_json_str(&buf)?
    } else {
        JobSpec::load_file(Path::new(job_src))?
    };

    if p.resume {
        spec.resume = true;
    }
    if let Some(ts) = p.tile_size {
        spec.tile_size = ts;
    }

    let engine = StereoPipelineEngine::new(spec.settings.backend);
    let cancel = CancellationToken::new();
    let out = WorkflowExecutor::execute_job(&spec, &engine, &cancel, |ev| emit_event(&ev, p.ndjson))?;
    if !p.ndjson {
        println!("\n{}", out.report.to_text());
    }
    Ok(())
}

fn cmd_batch(args: &[String]) -> Result<()> {
    let p = parse_common_args(args)?;
    if p.positionals.is_empty() {
        return Err(PipelineError::Cli("Usage: batch <jobs_dir_or_files...> [--continue-on-error] [--ndjson]".into()));
    }
    let mut job_files: Vec<PathBuf> = Vec::new();
    for item in &p.positionals {
        let path = Path::new(item);
        if path.is_dir() {
            let mut entries: Vec<PathBuf> = std::fs::read_dir(path)?
                .flatten()
                .map(|e| e.path())
                .filter(|p| p.extension().and_then(|e| e.to_str()) == Some("json"))
                .collect();
            entries.sort();
            job_files.extend(entries);
        } else {
            job_files.push(path.to_path_buf());
        }
    }
    if job_files.is_empty() {
        return Err(PipelineError::Cli("No *.json job manifests found in batch input".into()));
    }

    let engine = StereoPipelineEngine::new(p.settings.backend);
    let cancel = CancellationToken::new();
    let mut failed = 0usize;
    let total = job_files.len();

    for (idx, jf) in job_files.iter().enumerate() {
        if !p.ndjson {
            println!("=== Batch Job {}/{}: {} ===", idx + 1, total, jf.display());
        }
        let res = JobSpec::load_file(jf).and_then(|mut spec| {
            if p.resume {
                spec.resume = true;
            }
            WorkflowExecutor::execute_job(&spec, &engine, &cancel, |ev| emit_event(&ev, p.ndjson))
        });
        match res {
            Ok(out) => {
                if !p.ndjson {
                    println!("  [OK] {:.1} ms | valid {:.1}%", out.result.timings.total_ms, out.report.valid_disparity_pct);
                }
            }
            Err(e) => {
                failed += 1;
                eprintln!("  [FAIL] {}: {}", jf.display(), e);
                if !p.continue_on_error {
                    return Err(e);
                }
            }
        }
    }
    if failed > 0 {
        Err(PipelineError::Compute(format!("Batch completed with {}/{} failed jobs", failed, total)))
    } else {
        Ok(())
    }
}

fn cmd_synthetic(args: &[String]) -> Result<()> {
    let mut p = parse_common_args(args)?;
    let out_prefix = p
        .positionals
        .first()
        .cloned()
        .unwrap_or_else(|| "output/synthetic".to_string());

    if p.settings.search_min_y == -2 && p.settings.search_max_y == 2 {
        p.settings.search_min_y = 0;
        p.settings.search_max_y = 0;
    }

    let ds = synthetic::generate_synthetic_stereo(p.width, p.height);
    let pref = Path::new(&out_prefix);
    if let Some(parent) = pref.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)?;
        }
    }
    save_image(Path::new(&format!("{}-input-L.tif", out_prefix)), &ds.left_image)?;
    save_image(Path::new(&format!("{}-input-R.tif", out_prefix)), &ds.right_image)?;
    if let AnyCamera::Pinhole(ref cl) = ds.left_cam {
        cl.save_tsai(Path::new(&format!("{}-cam-L.tsai", out_prefix)))?;
    }
    if let AnyCamera::Pinhole(ref cr) = ds.right_cam {
        cr.save_tsai(Path::new(&format!("{}-cam-R.tsai", out_prefix)))?;
    }

    let spec = build_spec(
        &p,
        JobInput::Synthetic {
            width: p.width,
            height: p.height,
        },
        out_prefix.clone(),
    );
    if let Some(ref ej) = p.export_job {
        spec.save_file(Path::new(ej))?;
    }

    let engine = StereoPipelineEngine::new(spec.settings.backend);
    let cancel = CancellationToken::new();
    let out = WorkflowExecutor::execute_in_memory(
        &spec,
        &ds.left_image,
        &ds.right_image,
        &ds.left_cam,
        &ds.right_cam,
        Some(ds.gt_disparity.clone()),
        &engine,
        &cancel,
        |ev| emit_event(&ev, p.ndjson),
    )?;

    let dem_png = encode_png_rgba(
        out.result.dem.elevation.width,
        out.result.dem.elevation.height,
        &out.result.dem.elevation.to_rgba8(p.cmap),
    );
    std::fs::write(format!("{}-DEM-preview.png", out_prefix), dem_png)?;
    let disp_img = out.result.disp_filtered.dx_image();
    let disp_png = encode_png_rgba(disp_img.width, disp_img.height, &disp_img.to_rgba8(p.cmap));
    std::fs::write(format!("{}-DISP-preview.png", out_prefix), disp_png)?;

    if !p.ndjson {
        println!("\n{}", out.report.to_text());
        let mut sq_err = 0.0f64;
        let mut abs_err = 0.0f64;
        let mut inliers_1px = 0usize;
        let mut count = 0usize;
        for i in 0..out.result.disp_filtered.dx.len() {
            if out.result.disp_filtered.valid[i] != 0 && ds.gt_disparity.valid[i] != 0 {
                let e = (out.result.disp_filtered.dx[i] - ds.gt_disparity.dx[i]) as f64;
                sq_err += e * e;
                abs_err += e.abs();
                if e.abs() <= 1.0 {
                    inliers_1px += 1;
                }
                count += 1;
            }
        }
        if count > 0 {
            println!("Ground-Truth Disparity Evaluation ({} valid pixels):", count);
            println!("  Disparity MAE        : {:.3} px", abs_err / count as f64);
            println!("  Disparity RMSE       : {:.3} px", (sq_err / count as f64).sqrt());
            println!("  Inliers (<= 1.0 px)  : {:.2}%", 100.0 * inliers_1px as f64 / count as f64);
        }
    }
    Ok(())
}

fn cmd_pprc(args: &[String]) -> Result<()> {
    let p = parse_common_args(args)?;
    if p.positionals.len() < 3 {
        return Err(PipelineError::Cli("Usage: stereo_pprc <L.tif> <R.tif> <out_prefix>".into()));
    }
    let left = load_image(Path::new(&p.positionals[0]))?;
    let right = load_image(Path::new(&p.positionals[1]))?;
    let engine = StereoPipelineEngine::new(p.settings.backend);
    let (lp, rp, used_vk) = pprc::run_pprc(&left, &right, &p.settings, engine.vulkan.as_ref());
    let pref = &p.positionals[2];
    save_image(Path::new(&format!("{}-L.tif", pref)), &lp)?;
    save_image(Path::new(&format!("{}-R.tif", pref)), &rp)?;
    println!("Saved {}-L.tif and {}-R.tif (Vulkan GPU: {})", pref, pref, used_vk);
    Ok(())
}

fn cmd_point2dem(args: &[String]) -> Result<()> {
    let p = parse_common_args(args)?;
    if p.positionals.len() < 2 {
        return Err(PipelineError::Cli("Usage: point2dem <cloud.xyz> <out_prefix>".into()));
    }
    let pts = load_point_cloud_xyz(Path::new(&p.positionals[0]))?;
    let n = pts.len();
    let side = (n as f64).sqrt().ceil().max(16.0) as u32;
    let pc = PointCloud {
        width: side,
        height: side,
        points: pts,
        errors: vec![0.0; n],
        intensities: vec![0.5; n],
        valid: vec![1; n],
    };
    let engine = StereoPipelineEngine::new(p.settings.backend);
    let (grid, used_vk) = dem::point_to_dem(&pc, &p.settings, engine.vulkan.as_ref());
    let pref = &p.positionals[1];
    save_tiff_f32(
        Path::new(&format!("{}-DEM.tif", pref)),
        &grid.elevation,
        Some(GeoTransform {
            origin_x: grid.min_x,
            origin_y: grid.max_y,
            pixel_size_x: grid.pixel_size,
            pixel_size_y: grid.pixel_size,
        }),
    )?;
    save_image(Path::new(&format!("{}-hillshade.tif", pref)), &grid.hillshade)?;
    println!(
        "Rasterized {} points -> {}x{} DEM ({}-DEM.tif, {}-hillshade.tif) [Vulkan hillshade: {}]",
        n, grid.elevation.width, grid.elevation.height, pref, pref, used_vk
    );
    Ok(())
}

fn cmd_pc_align(args: &[String]) -> Result<()> {
    let p = parse_common_args(args)?;
    if p.positionals.len() < 2 {
        return Err(PipelineError::Cli("Usage: pc_align <ref.xyz> <src.xyz> [--max-iters N] [--max-displacement D]".into()));
    }
    let ref_pts = load_point_cloud_xyz(Path::new(&p.positionals[0]))?;
    let src_pts = load_point_cloud_xyz(Path::new(&p.positionals[1]))?;
    let res = align::align_icp(&ref_pts, &src_pts, p.max_iters, p.max_disp);
    println!("ICP Alignment Complete ({} iterations)", res.iterations);
    println!("  Initial RMSE : {:.6} m", res.initial_rmse);
    println!("  Final   RMSE : {:.6} m", res.final_rmse);
    println!("  Translation  : ({:.6}, {:.6}, {:.6})", res.translation.x, res.translation.y, res.translation.z);
    println!("  Rotation     : {:?}", res.rotation.m);
    Ok(())
}

fn cmd_disparitydebug(args: &[String]) -> Result<()> {
    let p = parse_common_args(args)?;
    if p.positionals.len() < 2 {
        return Err(PipelineError::Cli("Usage: disparitydebug <input.tif|pfm> <out_prefix> [--cmap turbo|viridis|inferno|terrain|gray]".into()));
    }
    let img = load_image(Path::new(&p.positionals[0]))?;
    let rgba = img.to_rgba8(p.cmap);
    let png = encode_png_rgba(img.width, img.height, &rgba);
    let out_path = format!("{}-colorized.png", p.positionals[1]);
    std::fs::write(&out_path, png)?;
    let (mn, mx, mean, sd) = img.stats();
    println!(
        "Saved {} ({}x{}, colormap={}, min={:.3}, max={:.3}, mean={:.3}, std={:.3})",
        out_path, img.width, img.height, p.cmap.name(), mn, mx, mean, sd
    );
    Ok(())
}

fn cmd_crs_convert(args: &[String]) -> Result<()> {
    let p = parse_common_args(args)?;
    let src = CrsId::parse(p.src_crs.as_deref().ok_or_else(|| PipelineError::Cli("Missing --src <CRS>".into()))?)?;
    let dst = CrsId::parse(p.dst_crs.as_deref().ok_or_else(|| PipelineError::Cli("Missing --dst <CRS>".into()))?)?;
    let inp = Vec3::new(p.coord_x, p.coord_y, p.coord_z);
    let out = transform_point(src, dst, inp)?;
    if p.ndjson {
        println!(
            r#"{{"src":"{}","dst":"{}","x":{:.8},"y":{:.8},"z":{:.4}}}"#,
            src.code(), dst.code(), out.x, out.y, out.z
        );
    } else {
        println!("{} -> {}", src.display_name(), dst.display_name());
        println!("  Input : ({:.8}, {:.8}, {:.4})", inp.x, inp.y, inp.z);
        println!("  Output: ({:.8}, {:.8}, {:.4})", out.x, out.y, out.z);
    }
    Ok(())
}

fn cmd_list_crs() {
    println!("=== Supported Geodetic CRS (Pure Rust f64) ===");
    for c in CrsId::catalog() {
        println!("  {:12} — {}", c.code(), c.display_name());
    }
    println!("  (Any UTM zone 1..60 supported via EPSG:32601..32660 / EPSG:32701..32760 or UTM31N)");
}

fn cmd_vulkan_info() -> Result<()> {
    let hw = HardwareProfile::detect();
    println!("=== Hardware Profile (Auto-Detected) ===");
    println!("Tier         : {}", hw.tier.name());
    println!("System RAM   : {} MB (budget: {} MB)", hw.total_ram_mb, hw.ram_budget_mb);
    println!("Full-Scale 1:1 Tile: {}x{} px", hw.tile_size, hw.tile_size);
    println!("CPU Threads  : {} | AVX2 SIMD: {}", hw.recommended_threads, hw.avx2_supported);

    println!("\n=== Vulkan 1.0 Compute Probe (Zero-Dependency FFI) ===");
    match VulkanCompute::try_new() {
        Ok(vk) => {
            println!("Status       : ACTIVE (Persistent SPIR-V Pipeline Cache Enabled)");
            println!("Device Name  : {}", vk.info.device_name);
            println!("Device Type  : {}", vk.info.device_type);
            println!("API Version  : {}", vk.info.api_version);
            println!("Vendor ID    : 0x{:04x}", vk.info.vendor_id);
            println!("Compute Queue: family #{}", vk.info.compute_queue_family);
            println!("VRAM Heap    : {} MB", vk.info.memory_heap_mb);
        }
        Err(e) => {
            println!("Status       : FALLBACK TO MULTI-THREADED CPU");
            println!("Reason       : {}", e);
        }
    }
    println!("\n=== Embedded Pure-Rust SPIR-V 1.0 Compute Shaders ===");
    for (name, words) in shader_catalog() {
        match validate_spirv(&words) {
            Ok(st) => println!(
                "  [OK] {:24} | {:4} words ({:5} bytes) | {:3} instructions | bound={}",
                name,
                st.word_count,
                st.word_count * 4,
                st.instruction_count,
                st.bound
            ),
            Err(e) => println!("  [ERR] {} -> {}", name, e),
        }
    }
    Ok(())
}

fn cmd_gui(args: &[String], default_mode: UiMode) -> Result<()> {
    let mut p = parse_common_args(args)?;
    let mode = p.ui_mode.unwrap_or(default_mode);
    if p.positionals.len() >= 4 {
        let left = load_image(Path::new(&p.positionals[0]))?;
        let right = load_image(Path::new(&p.positionals[1]))?;
        let cl = crate::camera::load_camera(Path::new(&p.positionals[2]))?;
        let cr = crate::camera::load_camera(Path::new(&p.positionals[3]))?;
        launch_gui(mode, left, right, cl, cr, p.settings, p.port).map_err(PipelineError::Compute)
    } else {
        if p.settings.search_min_y == -2 && p.settings.search_max_y == 2 {
            p.settings.search_min_y = 0;
            p.settings.search_max_y = 0;
        }
        let ds = synthetic::generate_synthetic_stereo(p.width, p.height);
        launch_gui(mode, ds.left_image, ds.right_image, ds.left_cam, ds.right_cam, p.settings, p.port)
            .map_err(PipelineError::Compute)
    }
}
