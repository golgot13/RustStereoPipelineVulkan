//! Zero-dependency ASP-compatible CLI parser & command dispatcher.

use std::path::Path;

use crate::camera::{load_camera, AnyCamera};
use crate::config::{
    ComputeBackendPreference, CostMode, PrefilterMode, StereoAlgorithm, StereoSettings, SubpixelMode,
};
use crate::image::io::{
    encode_png_rgba, load_image, load_point_cloud_xyz, save_image, save_tiff_f32, GeoTransform,
};
use crate::image::{Colormap, PointCloud};
use crate::pipeline::{align, dem, pprc, synthetic, StereoPipelineEngine};
use crate::ui::{launch_gui, UiMode};
use crate::vulkan::shaders::shader_catalog;
use crate::vulkan::spirv::validate_spirv;
use crate::vulkan::VulkanCompute;

pub fn run_cli(args: &[String]) -> Result<(), String> {
    if args.len() < 2 || matches!(args[1].as_str(), "-h" | "--help" | "help") {
        print_help();
        return Ok(());
    }

    let cmd = args[1].as_str();
    let rest = &args[2..];

    match cmd {
        "stereo" => cmd_stereo(rest),
        "stereo_pprc" => cmd_pprc(rest),
        "point2dem" => cmd_point2dem(rest),
        "pc_align" => cmd_pc_align(rest),
        "disparitydebug" => cmd_disparitydebug(rest),
        "synthetic" => cmd_synthetic(rest),
        "vulkan-info" => cmd_vulkan_info(),
        "gui" | "stereo_gui" => cmd_gui(rest, UiMode::Auto),
        "web-ui" => cmd_gui(rest, UiMode::Web),
        other => Err(format!("Unknown subcommand '{}'. Run with --help to see available commands.", other)),
    }
}

fn print_help() {
    println!(
        r#"RustStereoPipelineVulkan v0.1.0 — Zero-Dependency Rust + Vulkan Compute Stereo Pipeline

USAGE:
  stereo-pipeline <COMMAND> [ARGS...] [OPTIONS...]

COMMANDS:
  stereo <L.tif> <R.tif> <L.tsai|rpb> <R.tsai|rpb> <out_prefix>
      Run full 5-stage Ames Stereo Pipeline (pprc -> corr -> rfne -> fltr -> tri -> point2dem)
  stereo_pprc <L.tif> <R.tif> <out_prefix>
      Run Stage 0 intensity normalization & prefiltering (LoG / Subtracted Mean)
  point2dem <cloud.xyz> <out_prefix>
      Rasterize a 3D XYZ/CSV point cloud to a GeoTIFF DEM and hillshade
  pc_align <ref.xyz> <src.xyz>
      Align two 3D point clouds via rigid Kabsch ICP
  disparitydebug <input.tif|pfm> <out_prefix>
      Render a colorized PNG preview of a raster (disparity/DEM/error)
  synthetic <out_prefix> [--width W] [--height H] [--benchmark]
      Generate synthetic planetary crater terrain, run full stereo pipeline & report RMSE vs ground truth
  vulkan-info
      Probe Vulkan Compute hardware devices and validate all built-in SPIR-V 1.0 shaders
  gui [L.tif R.tif L.tsai R.tsai] [--x11 | --web] [--port 8080]
      Launch Hybrid GUI (Native X11 desktop window or embedded Web UI)
  web-ui [L.tif R.tif L.tsai R.tsai] [--port 8080]
      Launch Embedded HTTP/1.1 Web UI workstation directly

COMMON OPTIONS:
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
    benchmark: bool,
}

fn parse_common_args(args: &[String]) -> Result<ParsedArgs, String> {
    let mut settings = StereoSettings::default();
    let mut positionals = Vec::new();
    let mut width = 160u32;
    let mut height = 120u32;
    let mut port = 8080u16;
    let mut ui_mode = None;
    let mut cmap = Colormap::Turbo;
    let mut max_iters = 40usize;
    let mut max_disp = 100.0f64;
    let mut benchmark = false;

    let mut i = 0;
    while i < args.len() {
        let a = args[i].as_str();
        match a {
            "-s" | "--stereo-file" => {
                i += 1;
                let path = args.get(i).ok_or("Missing path after --stereo-file")?;
                settings = StereoSettings::load_file(Path::new(path))
                    .map_err(|e| format!("Failed to load {}: {}", path, e))?;
            }
            "--stereo-algorithm" => {
                i += 1;
                let v = args.get(i).ok_or("Missing value after --stereo-algorithm")?;
                settings.stereo_algorithm = StereoAlgorithm::from_str(v);
            }
            "--cost-mode" => {
                i += 1;
                let v: i32 = args.get(i).ok_or("Missing value")?.parse().map_err(|_| "Invalid cost-mode")?;
                settings.cost_mode = CostMode::from_int(v);
            }
            "--prefilter-mode" => {
                i += 1;
                let v: i32 = args.get(i).ok_or("Missing value")?.parse().map_err(|_| "Invalid prefilter-mode")?;
                settings.prefilter_mode = PrefilterMode::from_int(v);
            }
            "--prefilter-kernel-width" => {
                i += 1;
                settings.prefilter_kernel_width = args.get(i).ok_or("Missing value")?.parse().map_err(|_| "Invalid kernel width")?;
            }
            "--corr-kernel" => {
                if i + 2 >= args.len() { return Err("--corr-kernel requires 2 integers".into()); }
                settings.corr_kernel_x = args[i + 1].parse().map_err(|_| "Invalid corr-kernel X")?;
                settings.corr_kernel_y = args[i + 2].parse().map_err(|_| "Invalid corr-kernel Y")?;
                i += 2;
            }
            "--corr-search" => {
                if i + 4 >= args.len() { return Err("--corr-search requires 4 integers".into()); }
                settings.search_min_x = args[i + 1].parse().map_err(|_| "Invalid search minX")?;
                settings.search_min_y = args[i + 2].parse().map_err(|_| "Invalid search minY")?;
                settings.search_max_x = args[i + 3].parse().map_err(|_| "Invalid search maxX")?;
                settings.search_max_y = args[i + 4].parse().map_err(|_| "Invalid search maxY")?;
                i += 4;
            }
            "--subpixel-mode" => {
                i += 1;
                let v: i32 = args.get(i).ok_or("Missing value")?.parse().map_err(|_| "Invalid subpixel-mode")?;
                settings.subpixel_mode = SubpixelMode::from_int(v);
            }
            "--backend" => {
                i += 1;
                let v = args.get(i).ok_or("Missing value after --backend")?;
                settings.backend = ComputeBackendPreference::from_str(v);
            }
            "--threads" => {
                i += 1;
                settings.threads = args.get(i).ok_or("Missing value")?.parse().map_err(|_| "Invalid threads")?;
            }
            "--width" => {
                i += 1;
                width = args.get(i).ok_or("Missing value")?.parse().map_err(|_| "Invalid width")?;
            }
            "--height" => {
                i += 1;
                height = args.get(i).ok_or("Missing value")?.parse().map_err(|_| "Invalid height")?;
            }
            "--port" => {
                i += 1;
                port = args.get(i).ok_or("Missing value")?.parse().map_err(|_| "Invalid port")?;
            }
            "--x11" => ui_mode = Some(UiMode::X11),
            "--web" => ui_mode = Some(UiMode::Web),
            "--cmap" => {
                i += 1;
                cmap = Colormap::from_name(args.get(i).ok_or("Missing value after --cmap")?);
            }
            "--max-iters" => {
                i += 1;
                max_iters = args.get(i).ok_or("Missing value")?.parse().map_err(|_| "Invalid max-iters")?;
            }
            "--max-displacement" => {
                i += 1;
                max_disp = args.get(i).ok_or("Missing value")?.parse().map_err(|_| "Invalid max-displacement")?;
            }
            "--benchmark" => benchmark = true,
            other if other.starts_with('-') => {
                return Err(format!("Unknown option: {}", other));
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
        benchmark,
    })
}

fn cmd_stereo(args: &[String]) -> Result<(), String> {
    let p = parse_common_args(args)?;
    if p.positionals.len() < 5 {
        return Err("Usage: stereo <L.tif> <R.tif> <L.tsai|rpb> <R.tsai|rpb> <out_prefix>".into());
    }
    let left = load_image(Path::new(&p.positionals[0])).map_err(|e| format!("Load left: {}", e))?;
    let right = load_image(Path::new(&p.positionals[1])).map_err(|e| format!("Load right: {}", e))?;
    let cam_l = load_camera(Path::new(&p.positionals[2])).map_err(|e| format!("Load left cam: {}", e))?;
    let cam_r = load_camera(Path::new(&p.positionals[3])).map_err(|e| format!("Load right cam: {}", e))?;
    let out_prefix = Path::new(&p.positionals[4]);

    let engine = StereoPipelineEngine::new(p.settings.backend);
    println!("Backend : {}", engine.vulkan_status);
    println!(
        "Config  : algo={} cost={} prefilter={} subpixel={} kernel={}x{} search=[{}..{}, {}..{}]",
        p.settings.stereo_algorithm.name(),
        p.settings.cost_mode.name(),
        p.settings.prefilter_mode.name(),
        p.settings.subpixel_mode.name(),
        p.settings.corr_kernel_x,
        p.settings.corr_kernel_y,
        p.settings.search_min_x,
        p.settings.search_max_x,
        p.settings.search_min_y,
        p.settings.search_max_y,
    );

    let res = engine.run_full(&left, &right, &cam_l, &cam_r, &p.settings, |msg, pct| {
        println!("[{:3.0}%] {}", pct * 100.0, msg);
    });

    StereoPipelineEngine::save_outputs(out_prefix, &res)
        .map_err(|e| format!("Failed saving outputs: {}", e))?;
    print_summary(&res, out_prefix);
    Ok(())
}

fn cmd_pprc(args: &[String]) -> Result<(), String> {
    let p = parse_common_args(args)?;
    if p.positionals.len() < 3 {
        return Err("Usage: stereo_pprc <L.tif> <R.tif> <out_prefix>".into());
    }
    let left = load_image(Path::new(&p.positionals[0])).map_err(|e| e.to_string())?;
    let right = load_image(Path::new(&p.positionals[1])).map_err(|e| e.to_string())?;
    let engine = StereoPipelineEngine::new(p.settings.backend);
    let (lp, rp, used_vk) = pprc::run_pprc(&left, &right, &p.settings, engine.vulkan.as_ref());
    let pref = &p.positionals[2];
    save_image(Path::new(&format!("{}-L.tif", pref)), &lp).map_err(|e| e.to_string())?;
    save_image(Path::new(&format!("{}-R.tif", pref)), &rp).map_err(|e| e.to_string())?;
    println!("Saved {}-L.tif and {}-R.tif (Vulkan GPU: {})", pref, pref, used_vk);
    Ok(())
}

fn cmd_point2dem(args: &[String]) -> Result<(), String> {
    let p = parse_common_args(args)?;
    if p.positionals.len() < 2 {
        return Err("Usage: point2dem <cloud.xyz> <out_prefix>".into());
    }
    let pts = load_point_cloud_xyz(Path::new(&p.positionals[0])).map_err(|e| e.to_string())?;
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
    )
    .map_err(|e| e.to_string())?;
    save_image(Path::new(&format!("{}-hillshade.tif", pref)), &grid.hillshade).map_err(|e| e.to_string())?;
    println!(
        "Rasterized {} points -> {}x{} DEM ({}-DEM.tif, {}-hillshade.tif) [Vulkan hillshade: {}]",
        n, grid.elevation.width, grid.elevation.height, pref, pref, used_vk
    );
    Ok(())
}

fn cmd_pc_align(args: &[String]) -> Result<(), String> {
    let p = parse_common_args(args)?;
    if p.positionals.len() < 2 {
        return Err("Usage: pc_align <ref.xyz> <src.xyz> [--max-iters N] [--max-displacement D]".into());
    }
    let ref_pts = load_point_cloud_xyz(Path::new(&p.positionals[0])).map_err(|e| e.to_string())?;
    let src_pts = load_point_cloud_xyz(Path::new(&p.positionals[1])).map_err(|e| e.to_string())?;
    let res = align::align_icp(&ref_pts, &src_pts, p.max_iters, p.max_disp);
    println!("ICP Alignment Complete ({} iterations)", res.iterations);
    println!("  Initial RMSE : {:.6} m", res.initial_rmse);
    println!("  Final   RMSE : {:.6} m", res.final_rmse);
    println!("  Translation  : ({:.6}, {:.6}, {:.6})", res.translation.x, res.translation.y, res.translation.z);
    println!("  Rotation     : {:?}", res.rotation.m);
    Ok(())
}

fn cmd_disparitydebug(args: &[String]) -> Result<(), String> {
    let p = parse_common_args(args)?;
    if p.positionals.len() < 2 {
        return Err("Usage: disparitydebug <input.tif|pfm> <out_prefix> [--cmap turbo|viridis|inferno|terrain|gray]".into());
    }
    let img = load_image(Path::new(&p.positionals[0])).map_err(|e| e.to_string())?;
    let rgba = img.to_rgba8(p.cmap);
    let png = encode_png_rgba(img.width, img.height, &rgba);
    let out_path = format!("{}-colorized.png", p.positionals[1]);
    std::fs::write(&out_path, png).map_err(|e| e.to_string())?;
    let (mn, mx, mean, sd) = img.stats();
    println!(
        "Saved {} ({}x{}, colormap={}, min={:.3}, max={:.3}, mean={:.3}, std={:.3})",
        out_path, img.width, img.height, p.cmap.name(), mn, mx, mean, sd
    );
    Ok(())
}

fn cmd_synthetic(args: &[String]) -> Result<(), String> {
    let mut p = parse_common_args(args)?;
    let out_prefix = p
        .positionals
        .first()
        .cloned()
        .unwrap_or_else(|| "output/synthetic".to_string());

    println!("Generating synthetic planetary crater stereo pair ({}x{})...", p.width, p.height);
    let ds = synthetic::generate_synthetic_stereo(p.width, p.height);

    // Save synthetic inputs & cameras so they can be inspected or re-run with `stereo`
    let pref = Path::new(&out_prefix);
    if let Some(parent) = pref.parent() {
        if !parent.as_os_str().is_empty() {
            let _ = std::fs::create_dir_all(parent);
        }
    }
    save_image(Path::new(&format!("{}-input-L.tif", out_prefix)), &ds.left_image).map_err(|e| e.to_string())?;
    save_image(Path::new(&format!("{}-input-R.tif", out_prefix)), &ds.right_image).map_err(|e| e.to_string())?;
    if let AnyCamera::Pinhole(ref cl) = ds.left_cam {
        cl.save_tsai(Path::new(&format!("{}-cam-L.tsai", out_prefix))).map_err(|e| e.to_string())?;
    }
    if let AnyCamera::Pinhole(ref cr) = ds.right_cam {
        cr.save_tsai(Path::new(&format!("{}-cam-R.tsai", out_prefix))).map_err(|e| e.to_string())?;
    }

    // Use tight epipolar search range for synthetic nadir cameras if user kept default wide range
    if p.settings.search_min_y == -2 && p.settings.search_max_y == 2 {
        p.settings.search_min_y = 0;
        p.settings.search_max_y = 0;
    }

    let engine = StereoPipelineEngine::new(p.settings.backend);
    println!("Compute Backend: {}", engine.vulkan_status);
    let res = engine.run_full(
        &ds.left_image,
        &ds.right_image,
        &ds.left_cam,
        &ds.right_cam,
        &p.settings,
        |msg, pct| println!("[{:3.0}%] {}", pct * 100.0, msg),
    );

    StereoPipelineEngine::save_outputs(pref, &res).map_err(|e| e.to_string())?;
    // Also export colorized PNG previews for instant viewing
    let dem_png = encode_png_rgba(
        res.dem.elevation.width,
        res.dem.elevation.height,
        &res.dem.elevation.to_rgba8(p.cmap),
    );
    std::fs::write(format!("{}-DEM-preview.png", out_prefix), dem_png).map_err(|e| e.to_string())?;
    let disp_img = res.disp_filtered.dx_image();
    let disp_png = encode_png_rgba(disp_img.width, disp_img.height, &disp_img.to_rgba8(p.cmap));
    std::fs::write(format!("{}-DISP-preview.png", out_prefix), disp_png).map_err(|e| e.to_string())?;

    print_summary(&res, pref);

    // Compute accuracy vs ground-truth disparity
    let mut sq_err = 0.0f64;
    let mut abs_err = 0.0f64;
    let mut inliers_1px = 0usize;
    let mut count = 0usize;
    for i in 0..res.disp_filtered.dx.len() {
        if res.disp_filtered.valid[i] != 0 && ds.gt_disparity.valid[i] != 0 {
            let e = (res.disp_filtered.dx[i] - ds.gt_disparity.dx[i]) as f64;
            sq_err += e * e;
            abs_err += e.abs();
            if e.abs() <= 1.0 {
                inliers_1px += 1;
            }
            count += 1;
        }
    }
    if count > 0 {
        let rmse = (sq_err / count as f64).sqrt();
        let mae = abs_err / count as f64;
        println!("Ground-Truth Disparity Evaluation ({} valid pixels):", count);
        println!("  Disparity MAE        : {:.3} px", mae);
        println!("  Disparity RMSE       : {:.3} px", rmse);
        println!("  Inliers (<= 1.0 px)  : {:.2}%", 100.0 * inliers_1px as f64 / count as f64);
    }
    let _ = p.benchmark;
    Ok(())
}

fn cmd_vulkan_info() -> Result<(), String> {
    println!("=== Vulkan 1.0 Compute Probe (Zero-Dependency FFI) ===");
    match VulkanCompute::try_new() {
        Ok(vk) => {
            println!("Status       : ACTIVE");
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

fn cmd_gui(args: &[String], default_mode: UiMode) -> Result<(), String> {
    let mut p = parse_common_args(args)?;
    let mode = p.ui_mode.unwrap_or(default_mode);
    if p.positionals.len() >= 4 {
        let left = load_image(Path::new(&p.positionals[0])).map_err(|e| e.to_string())?;
        let right = load_image(Path::new(&p.positionals[1])).map_err(|e| e.to_string())?;
        let cl = load_camera(Path::new(&p.positionals[2])).map_err(|e| e.to_string())?;
        let cr = load_camera(Path::new(&p.positionals[3])).map_err(|e| e.to_string())?;
        launch_gui(mode, left, right, cl, cr, p.settings, p.port)
    } else {
        if p.settings.search_min_y == -2 && p.settings.search_max_y == 2 {
            p.settings.search_min_y = 0;
            p.settings.search_max_y = 0;
        }
        let ds = synthetic::generate_synthetic_stereo(p.width, p.height);
        launch_gui(mode, ds.left_image, ds.right_image, ds.left_cam, ds.right_cam, p.settings, p.port)
    }
}

fn print_summary(res: &crate::pipeline::PipelineResult, out_prefix: &Path) {
    let t = &res.timings;
    let (dem_min, dem_max, dem_mean, _) = res.dem.elevation.stats();
    println!("\n=== Pipeline Summary ===");
    println!("  Backend       : {}", t.active_backend);
    println!("  Stage 0 PPRC  : {:7.2} ms", t.pprc_ms);
    println!("  Stage 1 CORR  : {:7.2} ms", t.corr_ms);
    println!("  Stage 2 RFNE  : {:7.2} ms", t.rfne_ms);
    println!("  Stage 3 FLTR  : {:7.2} ms", t.fltr_ms);
    println!("  Stage 4 TRI   : {:7.2} ms", t.tri_ms);
    println!("  Stage 5 DEM   : {:7.2} ms", t.dem_ms);
    println!("  TOTAL TIME    : {:7.2} ms", t.total_ms);
    println!("  Valid Pixels  : {:.2}%", res.disp_filtered.valid_fraction() * 100.0);
    println!("  3D Points     : {}", res.point_cloud.valid_count());
    println!("  Triang. Error : {:.5} m", res.point_cloud.mean_error());
    println!("  DEM Elevation : min={:.2}m max={:.2}m mean={:.2}m ({}x{})", dem_min, dem_max, dem_mean, res.dem.elevation.width, res.dem.elevation.height);
    println!("  Output Prefix : {}", out_prefix.display());
}
