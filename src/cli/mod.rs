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
    chantier_summary_json, load_source_band_for_tiles, plan_full_model,
    prepare_stereo_pair_roi_opts, rectify_full_model_tile, validate_dem_against_gcps, Chantier,
};
use crate::core::crs::{transform_point, CrsId};
use crate::core::error::{PipelineError, Result};
use crate::core::hardware::{HardwareProfile, ProfileTier};
use crate::core::job::{JobInput, JobSpec, StageId};
use crate::image::geotiff::GeoTiffReader;
use crate::image::io::{
    encode_png_rgba, load_image, load_point_cloud_xyz, save_image, save_tiff_f32,
    stream_ply_vertices, GeoTransform, PointCloudMerger,
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
        "merge-ply" => cmd_merge_ply(rest),
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
  chantier <chantier_dir> [--output <dir>] [--pair N | --all-pairs] [--full-model] [--express] [--roi-size 1024] [--merge-voxel <m>]
      Auto-discover .CAM, trajectory .TXT, GCP, and GeoTIFF stereo pairs (e.g. Meyreuil DMC-IV 433Mpx)
      and run streaming Full-Scale 1:1 stereo reconstruction into Output_stereopipeline
      (--full-model processes ALL 1:1 tiles covering the entire stereo overlap; --express skips all intermediate outputs)
  merge-ply <dir_or_ply1 ...> --output <merged.ply> [--merge-voxel <m>]
      Fuse existing colorized binary/ASCII PLY point clouds into a single RGB f64 binary PLY
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
  --center-gcp                   In chantier mode, center 1:1 ROI on closest overlapping GCP
  --full-model                   Process ALL 1:1 tiles covering the complete overlap of each stereo model
  --express                      Express benchmark mode: ZERO intermediate outputs (only dense RGB f64 PLY + exact timings)
  --merge-cloud                  Produce unified RGB f64 chantier_merged-PC.ply (auto-ON for --all-pairs)
  --no-merge-cloud               Disable unified chantier_merged-PC.ply export
  --merge-voxel <meters>         3D voxel fusion resolution in meters (0 = lossless full-res streaming; e.g. 0.05)
  --no-checkpoints               Disable 00_checkpoints/ TIFF writes for faster disk I/O
  --no-ply                       Skip writing per-pair -PC.ply point clouds
  --fast-io                      Fast batch I/O: skips intermediate TIFFs/checkpoints/per-pair PLYs (keeps merged RGB PLY & DEM/PNG)
  --continue-on-error            In batch mode, continue running remaining jobs on failure

STEREO OPTIONS:
  -s, --stereo-file <file>       Load ASP stereo.default configuration file
  --stereo-algorithm <alg>       asp_bm | asp_sgm | asp_mgm (default: asp_sgm)
  --cost-mode <0|1|2|3>          0=SAD, 1=SSD, 2=NCC, 3=Census (default: 2)
  --prefilter-mode <0|1|2>       0=None, 1=SubMean, 2=LoG (default: 1)
  --prefilter-kernel-width <f>   Prefilter sigma/width (default: 7.0)
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
    center_gcp: bool,
    custom_search: bool,
    no_checkpoints: bool,
    no_ply: bool,
    fast_io: bool,
    merge_cloud: bool,
    no_merge_cloud: bool,
    merge_voxel: f64,
    full_model: bool,
    express: bool,
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
    let mut center_gcp = false;
    let mut custom_search = false;
    let mut no_checkpoints = false;
    let mut no_ply = false;
    let mut fast_io = false;
    let mut merge_cloud = false;
    let mut no_merge_cloud = false;
    let mut merge_voxel = 0.0f64;
    let mut full_model = false;
    let mut express = false;

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
                custom_search = true;
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
            "--center-gcp" => center_gcp = true,
            "--full-model" => full_model = true,
            "--express" => {
                express = true;
                fast_io = true;
                no_checkpoints = true;
                no_ply = true;
            }
            "--merge-cloud" => merge_cloud = true,
            "--no-merge-cloud" => no_merge_cloud = true,
            "--merge-voxel" => {
                merge_voxel = next(&mut i, a)?
                    .parse::<f64>()
                    .map_err(|_| PipelineError::Cli("Invalid --merge-voxel (meters)".into()))?
                    .max(0.0);
            }
            "--no-checkpoints" => no_checkpoints = true,
            "--no-ply" => no_ply = true,
            "--fast-io" => {
                fast_io = true;
                no_checkpoints = true;
                no_ply = true;
            }
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
        center_gcp,
        custom_search,
        no_checkpoints,
        no_ply,
        fast_io,
        merge_cloud,
        no_merge_cloud,
        merge_voxel,
        full_model,
        express,
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
    spec.write_checkpoints = !p.no_checkpoints;
    spec.write_ply = !p.no_ply;
    spec.write_intermediates = !p.fast_io;
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
    if p.full_model {
        return run_chantier_full_model(&chantier, &selected, &out_dir, &p, &engine);
    }

    let cancel = CancellationToken::new();
    let mut pair_summaries = Vec::new();

    let merge_enabled = (selected.len() > 1 || p.merge_cloud) && !p.no_merge_cloud;
    let merged_ply_path = out_dir.join("chantier_merged-PC.ply");
    let mut cloud_merger = if merge_enabled {
        Some(PointCloudMerger::new(&merged_ply_path, p.merge_voxel)?)
    } else {
        None
    };

    if !p.ndjson && merge_enabled {
        if p.merge_voxel > 0.0 {
            println!(
                "  Fusion nuage : {} (voxel 3D {:.3} m, f64 XYZ + RGB)",
                merged_ply_path.display(),
                p.merge_voxel
            );
        } else {
            println!(
                "  Fusion nuage : {} (streaming sans perte 1:1, f64 XYZ + RGB)",
                merged_ply_path.display()
            );
        }
    }

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

        let roi = prepare_stereo_pair_roi_opts(&chantier, pair, p.roi_size, p.center_gcp)?;
        let eff_search = if p.custom_search {
            (
                p.settings.search_min_x,
                p.settings.search_min_y,
                p.settings.search_max_x,
                p.settings.search_max_y,
            )
        } else {
            roi.search_box
        };
        if !p.ndjson {
            println!(
                "  Capteur natif : {}x{} px | Patch L=({},{},{}x{}) R=({},{},{}x{})",
                roi.full_dims.0,
                roi.full_dims.1,
                roi.left_roi.0,
                roi.left_roi.1,
                roi.left_roi.2,
                roi.left_roi.3,
                roi.right_roi.0,
                roi.right_roi.1,
                roi.right_roi.2,
                roi.right_roi.3,
            );
            let anchor_str = match roi.anchored_gcp {
                Some(ref gid) => format!("GCP {} (Z={:.2} m)", gid, roi.ground_z_ref),
                None => format!("Z_ref={:.2} m", roi.ground_z_ref),
            };
            println!(
                "  Épipolaire    : Fusiello rectifié (dy=0) | {} | dx_coarse={:+} px | Search X=[{}..{}] Y=[{}..{}]",
                anchor_str,
                roi.coarse_dx_px,
                eff_search.0,
                eff_search.2,
                eff_search.1,
                eff_search.3
            );
        }

        // Export rectified 1:1 ROI GeoTIFFs & exact ASP .tsai cameras for 100% CLI reproducibility
        let rect_l_path = format!("{}-rect-L.tif", out_prefix);
        let rect_r_path = format!("{}-rect-R.tif", out_prefix);
        if !p.fast_io {
            save_tiff_f32(Path::new(&rect_l_path), &roi.left_img, None)?;
            save_tiff_f32(Path::new(&rect_r_path), &roi.right_img, None)?;
        }
        if let AnyCamera::Pinhole(ref cl) = roi.left_cam {
            cl.save_tsai(Path::new(&format!("{}-cam-L.tsai", out_prefix)))?;
        }
        if let AnyCamera::Pinhole(ref cr) = roi.right_cam {
            cr.save_tsai(Path::new(&format!("{}-cam-R.tsai", out_prefix)))?;
        }

        let mut spec = build_spec(
            &p,
            JobInput::Files {
                left_image: rect_l_path,
                right_image: rect_r_path,
                left_camera: format!("{}-cam-L.tsai", out_prefix),
                right_camera: format!("{}-cam-R.tsai", out_prefix),
            },
            out_prefix.clone(),
        );
        spec.settings.search_min_x = eff_search.0;
        spec.settings.search_min_y = eff_search.1;
        spec.settings.search_max_x = eff_search.2;
        spec.settings.search_max_y = eff_search.3;
        spec.save_file(Path::new(&format!("{}-job.json", out_prefix)))?;

        let out = WorkflowExecutor::execute_in_memory_with_rgb(
            &spec,
            &roi.left_img,
            roi.left_rgb.as_ref(),
            &roi.right_img,
            &roi.left_cam,
            &roi.right_cam,
            None,
            &engine,
            &cancel,
            |ev| emit_event(&ev, p.ndjson),
        )?;

        if let Some(ref mut merger) = cloud_merger {
            merger.add_point_cloud(&out.result.point_cloud)?;
        }

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
                    "  Validation GCP ({} points, bilinéaire) : biais dZ={:+.3} m | RMSE dZ={:.3} m | max={:.3} m",
                    gcp_val.matched_gcps, gcp_val.dz_bias_m, gcp_val.dz_rmse_m, gcp_val.dz_max_abs_m
                );
                for s in &gcp_val.samples {
                    println!(
                        "    - GCP {:<12} : Z_gcp={:.3} m | Z_dem={:.3} m | dZ={:+.3} m",
                        s.id, s.z_gcp, s.z_dem, s.dz_m
                    );
                }
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

    let merge_stats = if let Some(merger) = cloud_merger {
        Some(merger.finish()?)
    } else {
        None
    };

    let summary_path = out_dir.join("chantier_summary.json");
    std::fs::write(&summary_path, chantier_summary_json(&chantier, &pair_summaries))?;

    if !p.ndjson {
        // Global aggregated GCP validation across all processed pairs
        let all_dz: Vec<f64> = pair_summaries
            .iter()
            .flat_map(|(_, _, _, _, g)| g.samples.iter().map(|s| s.dz_m))
            .collect();
        let total_ms: f64 = pair_summaries.iter().map(|(_, ms, _, _, _)| *ms).sum();
        let avg_valid: f64 = if pair_summaries.is_empty() {
            0.0
        } else {
            pair_summaries.iter().map(|(_, _, v, _, _)| *v).sum::<f64>() / pair_summaries.len() as f64
        };

        println!("\n=== Chantier terminé ({} paires, {:.1} s, densité moy. {:.2}%) ===",
            pair_summaries.len(),
            total_ms / 1000.0,
            avg_valid
        );
        if !all_dz.is_empty() {
            let n = all_dz.len() as f64;
            let bias = all_dz.iter().sum::<f64>() / n;
            let rmse = (all_dz.iter().map(|d| d * d).sum::<f64>() / n).sqrt();
            let sigma = (all_dz.iter().map(|d| (d - bias) * (d - bias)).sum::<f64>() / n).sqrt();
            let max_abs = all_dz.iter().map(|d| d.abs()).fold(0.0f64, f64::max);
            println!(
                "  Validation GCP Globale ({} obs.) : biais dZ={:+.3} m | RMSE={:.3} m | sigma={:.3} m | max={:.3} m",
                all_dz.len(),
                bias,
                rmse,
                sigma,
                max_abs
            );
        }
        if let Some(ref st) = merge_stats {
            println!(
                "  Nuage Fusionné RGB f64 : {} ({} pts, {:.1} MB)",
                merged_ply_path.display(),
                st.vertex_count,
                st.file_size_bytes as f64 / (1024.0 * 1024.0)
            );
            println!(
                "  Emprise 3D Fusionnée   : X=[{:.2}..{:.2}] Y=[{:.2}..{:.2}] Z=[{:.2}..{:.2}] m",
                st.min_x, st.max_x, st.min_y, st.max_y, st.min_z, st.max_z
            );
        }
        println!("  Résumé JSON            : {}", summary_path.display());
    }
    Ok(())
}

/// Full-Scale 1:1 complete stereo model reconstruction in Express mode:
/// - Covers the ENTIRE overlap of each selected stereo pair using halo-padded sliding windows
/// - Caches each source GeoTIFF strip-band ONCE in RAM (parallel Deflate decode)
/// - Produces ZERO intermediate files (no DEM/hillshade/PNG/TIFF/checkpoints)
/// - Streams dense colorized RGB `f64` 3D points directly to `<pair>-FULL-PC.ply` (and `chantier_merged-PC.ply` for `--all-pairs`)
/// - Reports exact per-stage timings, Mpx/s throughput, and live ETA.
fn run_chantier_full_model(
    chantier: &Chantier,
    selected: &[usize],
    out_dir: &Path,
    p: &ParsedArgs,
    engine: &StereoPipelineEngine,
) -> Result<()> {
    use std::io::Write;
    let hw = HardwareProfile::detect();
    let global_merge = selected.len() > 1 && !p.no_merge_cloud;
    let global_ply_path = out_dir.join("chantier_merged-PC.ply");
    let mut global_merger = if global_merge {
        Some(PointCloudMerger::new(&global_ply_path, p.merge_voxel)?)
    } else {
        None
    };

    if !p.ndjson {
        println!(
            "  Mode Calcul  : FULL-MODEL 1:1 {} (dalles {}x{}, 0 sortie intermédiaire)",
            if p.express { "EXPRESS" } else { "COMPLET" },
            p.roi_size,
            p.roi_size
        );
        println!(
            "  Backend      : {} | {} threads CPU (AVX2: {})",
            engine.vulkan_status, p.settings.threads, hw.avx2_supported
        );
        if global_merge {
            println!("  Nuage Global : {}", global_ply_path.display());
        }
    }

    let t_global = std::time::Instant::now();
    let mut global_tiles_done = 0usize;
    let mut global_core_pixels = 0u64;
    let mut global_points = 0usize;

    for &idx in selected {
        let pair = &chantier.pairs[idx];
        let t_model = std::time::Instant::now();
        let reader_l = GeoTiffReader::open(&pair.left_image_path)?;
        let reader_r = GeoTiffReader::open(&pair.right_image_path)?;
        let plan = plan_full_model(
            chantier,
            pair,
            reader_l.meta.width,
            reader_l.meta.height,
            reader_r.meta.width,
            reader_r.meta.height,
            p.roi_size,
        )?;

        let model_ply_path = out_dir.join(format!("{}-FULL-PC.ply", pair.name));
        // For single-model runs (or non-express multi-pair), write per-model dense RGB f64 PLY
        let write_model_ply = !global_merge || !p.express;
        let mut model_merger = if write_model_ply {
            Some(PointCloudMerger::new(&model_ply_path, p.merge_voxel)?)
        } else {
            None
        };

        let overlap_mpx = (plan.overlap_w as f64 * plan.overlap_h as f64) / 1_000_000.0;
        if !p.ndjson {
            println!(
                "\n=== Modèle {}/{} : {} (baseline={:.1} m) ===",
                idx + 1,
                chantier.pairs.len(),
                pair.name,
                pair.baseline_m
            );
            println!("  Left         : {} ({}x{} px)", pair.left_image_path.display(), plan.full_dims_l.0, plan.full_dims_l.1);
            println!("  Right        : {} ({}x{} px)", pair.right_image_path.display(), plan.full_dims_r.0, plan.full_dims_r.1);
            println!(
                "  Recouvrement : {}x{} px ({:.1} Mpx 1:1) | Grille {}x{} = {} dalles ({} bandes source) | Z_ref={:.2} m",
                plan.overlap_w,
                plan.overlap_h,
                overlap_mpx,
                plan.tiles_x,
                plan.tiles_y,
                plan.total_tiles,
                plan.bands.len(),
                plan.ground_z_ref
            );
            if write_model_ply {
                println!("  Nuage Modèle : {}", model_ply_path.display());
            }
        }

        let mut band_io_ms = 0.0f64;
        let mut rect_ms = 0.0f64;
        let mut pprc_ms = 0.0f64;
        let mut corr_ms = 0.0f64;
        let mut rfne_ms = 0.0f64;
        let mut fltr_ms = 0.0f64;
        let mut tri_ms = 0.0f64;
        let mut ply_ms = 0.0f64;
        let mut total_vk_dispatches = 0usize;
        let mut tiles_done = 0usize;
        let mut tiles_skipped = 0usize;
        let mut model_core_px = 0u64;
        let mut model_pts = 0usize;
        let mut sum_valid_pct = 0.0f64;
        let mut prev_median_dx = 0i32;

        for (b_idx, band) in plan.bands.iter().enumerate() {
            let t_band = std::time::Instant::now();
            let band_l = load_source_band_for_tiles(
                &reader_l,
                &plan.full_cam_l,
                &plan.rect_cam_l,
                band,
                true,
            )?;
            let band_r = load_source_band_for_tiles(
                &reader_r,
                &plan.full_cam_r,
                &plan.rect_cam_r,
                band,
                false,
            )?;
            let b_ms = t_band.elapsed().as_secs_f64() * 1000.0;
            band_io_ms += b_ms;

            if !p.ndjson {
                let cache_str = match (&band_l, &band_r) {
                    (Some(bl), Some(br)) => format!(
                        "bande RAM L={}x{} R={}x{} décodée en {:.0} ms",
                        bl.gray.width, bl.gray.height, br.gray.width, br.gray.height, b_ms
                    ),
                    _ => "streaming direct par dalle".to_string(),
                };
                println!(
                    "  -- Bande {}/{} ({} dalles, {}) --",
                    b_idx + 1,
                    plan.bands.len(),
                    band.len(),
                    cache_str
                );
                let _ = std::io::stdout().flush();
            }

            for tile in band {
                let t_tile = std::time::Instant::now();
                let t_rect = std::time::Instant::now();
                let prep_opt = rectify_full_model_tile(
                    &plan,
                    tile,
                    &reader_l,
                    &reader_r,
                    band_l.as_ref(),
                    band_r.as_ref(),
                    prev_median_dx,
                )?;
                let r_ms = t_rect.elapsed().as_secs_f64() * 1000.0;
                rect_ms += r_ms;

                let Some(prep) = prep_opt else {
                    tiles_skipped += 1;
                    tiles_done += 1;
                    continue;
                };

                let mut tile_settings = p.settings.clone();
                if !p.custom_search {
                    tile_settings.search_min_x = prep.search_box.0;
                    tile_settings.search_min_y = prep.search_box.1;
                    tile_settings.search_max_x = prep.search_box.2;
                    tile_settings.search_max_y = prep.search_box.3;
                }

                let out = engine.run_express_tile(
                    &prep.left_img,
                    prep.left_rgb.as_ref(),
                    &prep.right_img,
                    &prep.left_cam,
                    &prep.right_cam,
                    &tile_settings,
                );
                pprc_ms += out.pprc_ms;
                corr_ms += out.corr_ms;
                rfne_ms += out.rfne_ms;
                fltr_ms += out.fltr_ms;
                tri_ms += out.tri_ms;
                total_vk_dispatches += out.vk_dispatches;

                if let Some(mdx) = out.median_dx {
                    if out.valid_pct >= 25.0 {
                        prev_median_dx = mdx.round() as i32;
                    }
                }

                let t_ply = std::time::Instant::now();
                let mut added_pts = 0usize;
                if let Some(ref mut m) = model_merger {
                    added_pts = m.add_point_cloud_rect(
                        &out.point_cloud,
                        tile.emit_x0,
                        tile.emit_y0,
                        tile.emit_x1,
                        tile.emit_y1,
                    )?;
                }
                if let Some(ref mut gm) = global_merger {
                    let g_pts = gm.add_point_cloud_rect(
                        &out.point_cloud,
                        tile.emit_x0,
                        tile.emit_y0,
                        tile.emit_x1,
                        tile.emit_y1,
                    )?;
                    if !write_model_ply {
                        added_pts = g_pts;
                    }
                }
                let p_ms = t_ply.elapsed().as_secs_f64() * 1000.0;
                ply_ms += p_ms;

                tiles_done += 1;
                let core_px = (tile.emit_x1 - tile.emit_x0) as u64 * (tile.emit_y1 - tile.emit_y0) as u64;
                model_core_px += core_px;
                model_pts += added_pts;
                sum_valid_pct += out.valid_pct;

                if !p.ndjson {
                    let elapsed_s = t_model.elapsed().as_secs_f64();
                    let mpx_s = (model_core_px as f64 / 1_000_000.0) / elapsed_s.max(1e-6);
                    let remaining = plan.total_tiles.saturating_sub(tiles_done);
                    let eta_s = if tiles_done > 0 {
                        (elapsed_s / tiles_done as f64) * remaining as f64
                    } else {
                        0.0
                    };
                    let tile_s = t_tile.elapsed().as_secs_f64();
                    println!(
                        "    [{:>3}/{}] ({:>2},{:>2}) {:>4}x{:<4} dx={:+4} -> {:>7} pts ({:>5.1}%) | {:.2}s [rect={:.0} pprc={:.0} corr={:.0} fltr={:.0} tri={:.0} ply={:.0} ms] | {:.2} Mpx/s | ETA {:.0}s",
                        tiles_done,
                        plan.total_tiles,
                        tile.tx,
                        tile.ty,
                        tile.tw,
                        tile.th,
                        prep.coarse_dx_px,
                        added_pts,
                        out.valid_pct,
                        tile_s,
                        r_ms,
                        out.pprc_ms,
                        out.corr_ms,
                        out.fltr_ms,
                        out.tri_ms,
                        p_ms,
                        mpx_s,
                        eta_s
                    );
                    let _ = std::io::stdout().flush();
                }
            }
        }

        let model_stats = if let Some(m) = model_merger {
            Some(m.finish()?)
        } else {
            None
        };

        let elapsed_s = t_model.elapsed().as_secs_f64();
        let active_tiles = tiles_done.saturating_sub(tiles_skipped).max(1);
        let avg_valid = sum_valid_pct / active_tiles as f64;
        let avg_tile_ms = (elapsed_s * 1000.0) / active_tiles as f64;
        let mpx_total = model_core_px as f64 / 1_000_000.0;
        let mpx_s = mpx_total / elapsed_s.max(1e-6);
        let pts_s = model_pts as f64 / elapsed_s.max(1e-6);

        global_tiles_done += active_tiles;
        global_core_pixels += model_core_px;
        global_points += model_pts;

        if !p.ndjson {
            let total_ms = (elapsed_s * 1000.0).max(1e-6);
            println!("\n=== Bilan Express Modèle {} ===", pair.name);
            println!(
                "  Dalles 1:1       : {} traitées ({} ignorées hors capteur) | {:.1} Mpx utiles",
                active_tiles, tiles_skipped, mpx_total
            );
            println!(
                "  Vitesse Pipeline : {:.2} s total | {:.0} ms/dalle | {:.2} Mpx/s | {:.2} Mpts/s",
                elapsed_s,
                avg_tile_ms,
                mpx_s,
                pts_s / 1_000_000.0
            );
            println!(
                "  Ventilation      : GeoTIFF={:.1}s ({:.1}%) | Rect={:.1}s ({:.1}%) | PPRC={:.1}s ({:.1}%) | Corr/SGM={:.1}s ({:.1}%) | Rfne={:.1}s ({:.1}%) | Fltr={:.1}s ({:.1}%) | Tri={:.1}s ({:.1}%) | PLY={:.1}s ({:.1}%)",
                band_io_ms / 1000.0, 100.0 * band_io_ms / total_ms,
                rect_ms / 1000.0, 100.0 * rect_ms / total_ms,
                pprc_ms / 1000.0, 100.0 * pprc_ms / total_ms,
                corr_ms / 1000.0, 100.0 * corr_ms / total_ms,
                rfne_ms / 1000.0, 100.0 * rfne_ms / total_ms,
                fltr_ms / 1000.0, 100.0 * fltr_ms / total_ms,
                tri_ms / 1000.0, 100.0 * tri_ms / total_ms,
                ply_ms / 1000.0, 100.0 * ply_ms / total_ms,
            );
            println!(
                "  Qualité / GPU    : densité moy. {:.2}% | {} dispatches Vulkan",
                avg_valid, total_vk_dispatches
            );
            if let Some(ref st) = model_stats {
                println!(
                    "  Nuage Dense RGB  : {} ({} pts, {:.1} MB)",
                    model_ply_path.display(),
                    st.vertex_count,
                    st.file_size_bytes as f64 / (1024.0 * 1024.0)
                );
                println!(
                    "  Emprise 3D       : X=[{:.2}..{:.2}] Y=[{:.2}..{:.2}] Z=[{:.2}..{:.2}] m",
                    st.min_x, st.max_x, st.min_y, st.max_y, st.min_z, st.max_z
                );
            }
        }
    }

    if let Some(gm) = global_merger {
        let st = gm.finish()?;
        let elapsed_s = t_global.elapsed().as_secs_f64();
        if !p.ndjson {
            println!(
                "\n=== Chantier Global Full-Model Terminé ({} modèles, {} dalles, {:.1} s) ===",
                selected.len(),
                global_tiles_done,
                elapsed_s
            );
            println!(
                "  Débit Global     : {:.2} Mpx/s ({:.1} Mpx, {} pts générés)",
                (global_core_pixels as f64 / 1_000_000.0) / elapsed_s.max(1e-6),
                global_core_pixels as f64 / 1_000_000.0,
                global_points
            );
            println!(
                "  Nuage Global RGB : {} ({} pts, {:.1} MB)",
                global_ply_path.display(),
                st.vertex_count,
                st.file_size_bytes as f64 / (1024.0 * 1024.0)
            );
            println!(
                "  Emprise 3D       : X=[{:.2}..{:.2}] Y=[{:.2}..{:.2}] Z=[{:.2}..{:.2}] m",
                st.min_x, st.max_x, st.min_y, st.max_y, st.min_z, st.max_z
            );
        }
    }

    Ok(())
}

fn collect_ply_files(dir: &Path, out: &mut Vec<PathBuf>) {
    if let Ok(entries) = std::fs::read_dir(dir) {
        let mut paths: Vec<PathBuf> = entries.flatten().map(|e| e.path()).collect();
        paths.sort();
        for p in paths {
            if p.is_dir() {
                collect_ply_files(&p, out);
            } else if p.extension().and_then(|e| e.to_str()).map_or(false, |e| e.eq_ignore_ascii_case("ply")) {
                let name = p.file_name().and_then(|n| n.to_str()).unwrap_or("").to_ascii_lowercase();
                if !name.contains("merged") {
                    out.push(p);
                }
            }
        }
    }
}

fn cmd_merge_ply(args: &[String]) -> Result<()> {
    let p = parse_common_args(args)?;
    if p.positionals.is_empty() {
        return Err(PipelineError::Cli(
            "Usage: merge-ply <dir_or_ply1 ...> --output <merged.ply> [--merge-voxel <m>]".into(),
        ));
    }

    let mut inputs = Vec::new();
    let mut out_path = p.output_dir.as_ref().map(PathBuf::from);
    for (idx, arg) in p.positionals.iter().enumerate() {
        let path = Path::new(arg);
        if path.is_dir() {
            collect_ply_files(path, &mut inputs);
        } else if path.is_file() {
            inputs.push(path.to_path_buf());
        } else if out_path.is_none() && idx + 1 == p.positionals.len() {
            out_path = Some(path.to_path_buf());
        } else {
            return Err(PipelineError::Io(format!("Input PLY/directory not found: {}", arg)));
        }
    }

    if inputs.is_empty() {
        return Err(PipelineError::Cli("No input .ply files found to merge".into()));
    }
    let final_out = out_path.unwrap_or_else(|| {
        let first = Path::new(&p.positionals[0]);
        if first.is_dir() {
            first.join("chantier_merged-PC.ply")
        } else {
            first.with_file_name("chantier_merged-PC.ply")
        }
    });
    if let Some(parent) = final_out.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)?;
        }
    }

    let t0 = std::time::Instant::now();
    let mut merger = PointCloudMerger::new(&final_out, p.merge_voxel)?;
    let mut total_in = 0usize;
    for (i, ply_path) in inputs.iter().enumerate() {
        let mut io_err: Option<std::io::Error> = None;
        let cnt = stream_ply_vertices(ply_path, |pt, rgb, err| {
            if io_err.is_none() {
                if let Err(e) = merger.add_vertex(pt, rgb, err) {
                    io_err = Some(e);
                }
            }
        })?;
        if let Some(e) = io_err {
            return Err(e.into());
        }
        total_in += cnt;
        if !p.ndjson {
            println!("  [{:>3}/{}] {} ({} pts)", i + 1, inputs.len(), ply_path.display(), cnt);
        }
    }
    let stats = merger.finish()?;
    if !p.ndjson {
        println!(
            "\n=== Fusion PLY terminée en {:.2} s ===",
            t0.elapsed().as_secs_f64()
        );
        println!("  Fichiers fusionnés : {} ({} pts lus -> {} pts écrits)", inputs.len(), total_in, stats.vertex_count);
        println!("  Sortie RGB f64 PLY : {} ({:.1} MB)", final_out.display(), stats.file_size_bytes as f64 / (1024.0 * 1024.0));
        println!(
            "  Emprise 3D         : X=[{:.2}..{:.2}] Y=[{:.2}..{:.2}] Z=[{:.2}..{:.2}] m",
            stats.min_x, stats.max_x, stats.min_y, stats.max_y, stats.min_z, stats.max_z
        );
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
        colors: None,
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
