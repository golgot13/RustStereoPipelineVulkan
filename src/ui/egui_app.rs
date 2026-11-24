//! Full interactive `egui` / `eframe` Desktop Workstation for RustStereoPipelineVulkan.
//! Features:
//! - Background async 5-stage stereo pipeline execution with live progress bar
//! - Complete ASP parameter inspector (pprc, corr BM/SGM/MGM, rfne, fltr, tri, DEM/hillshade)
//! - 9-layer GPU-textured viewer with smooth zoom/pan, pixel inspector, colormaps & colorbar
//! - Side-by-side comparison mode & interactive 3D orbitable DEM mesh viewer
//! - Live stage timing bars, Vulkan/SPIR-V telemetry, and one-click GeoTIFF/PLY/PNG export

use std::path::Path;
use std::sync::mpsc::{self, Receiver};
use std::sync::Arc;

use eframe::egui::{self, Color32, ColorImage, Pos2, Rect, RichText, Sense, Stroke, TextureHandle, TextureOptions, Vec2};

use crate::camera::{load_camera, AnyCamera};
use crate::config::{
    ComputeBackendPreference, CostMode, PrefilterMode, StereoAlgorithm, StereoSettings, SubpixelMode,
};
use crate::image::io::load_image;
use crate::image::{Colormap, ImageF32};
use crate::pipeline::{dem, synthetic, PipelineResult, StereoPipelineEngine};
use crate::vulkan::shaders::shader_catalog;
use crate::vulkan::spirv::validate_spirv;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Layer {
    Left,
    Right,
    Prefiltered,
    DispRaw,
    DispFiltered,
    Dem,
    Hillshade,
    TriError,
    Terrain3D,
}

impl Layer {
    const ALL: [(Layer, &'static str); 9] = [
        (Layer::Left, "1. Left"),
        (Layer::Right, "2. Right"),
        (Layer::Prefiltered, "3. PPRC"),
        (Layer::DispRaw, "4. Disp Raw"),
        (Layer::DispFiltered, "5. Disp Fltr"),
        (Layer::Dem, "6. DEM"),
        (Layer::Hillshade, "7. Hillshade"),
        (Layer::TriError, "8. Tri Err"),
        (Layer::Terrain3D, "9. 3D Terrain"),
    ];

    fn unit_label(self) -> &'static str {
        match self {
            Layer::Left | Layer::Right | Layer::Prefiltered => "intensity",
            Layer::DispRaw | Layer::DispFiltered => "px",
            Layer::Dem => "m",
            Layer::Hillshade => "shade",
            Layer::TriError => "m err",
            Layer::Terrain3D => "m",
        }
    }
}

enum WorkerMsg {
    Progress(String, f32),
    Finished(Box<PipelineResult>),
}

struct StereoEguiApp {
    engine: Arc<StereoPipelineEngine>,
    left: ImageF32,
    right: ImageF32,
    cam_left: AnyCamera,
    cam_right: AnyCamera,
    settings: StereoSettings,
    result: PipelineResult,

    // File path inputs
    left_path: String,
    right_path: String,
    cam_left_path: String,
    cam_right_path: String,
    out_prefix: String,
    synth_size_idx: usize,

    // View state
    active_layer: Layer,
    colormap: Colormap,
    split_view: bool,
    zoom: f32,
    pan: Vec2,
    yaw: f32,
    pitch: f32,
    z_exaggeration: f32,
    hover_readout: String,

    // Cached GPU textures
    tex_main: Option<TextureHandle>,
    tex_left: Option<TextureHandle>,
    tex_dirty: bool,

    // Async worker state
    worker_rx: Option<Receiver<WorkerMsg>>,
    progress_frac: f32,
    progress_text: String,
    logs: Vec<String>,
    spirv_summary: Vec<(String, usize, usize)>,
}

impl StereoEguiApp {
    fn new(
        cc: &eframe::CreationContext<'_>,
        left: ImageF32,
        right: ImageF32,
        cam_left: AnyCamera,
        cam_right: AnyCamera,
        settings: StereoSettings,
    ) -> Self {
        apply_dark_scientific_theme(&cc.egui_ctx);

        let engine = Arc::new(StereoPipelineEngine::new(settings.backend));
        let result = engine.run_full(&left, &right, &cam_left, &cam_right, &settings, |_, _| {});

        let spirv_summary = shader_catalog()
            .into_iter()
            .filter_map(|(name, words)| {
                validate_spirv(&words)
                    .ok()
                    .map(|st| (name.to_string(), st.word_count * 4, st.instruction_count))
            })
            .collect();

        let init_log = format!(
            "Initialized ({}x{}) in {:.1} ms — {}",
            left.width, left.height, result.timings.total_ms, engine.vulkan_status
        );

        Self {
            engine,
            left,
            right,
            cam_left,
            cam_right,
            settings,
            result,
            left_path: String::new(),
            right_path: String::new(),
            cam_left_path: String::new(),
            cam_right_path: String::new(),
            out_prefix: "output/egui_run".to_string(),
            synth_size_idx: 1,
            active_layer: Layer::Dem,
            colormap: Colormap::Turbo,
            split_view: false,
            zoom: 1.0,
            pan: Vec2::ZERO,
            yaw: 0.65,
            pitch: 0.55,
            z_exaggeration: 0.65,
            hover_readout: "Hover image to inspect pixel values".to_string(),
            tex_main: None,
            tex_left: None,
            tex_dirty: true,
            worker_rx: None,
            progress_frac: 1.0,
            progress_text: "Ready".to_string(),
            logs: vec![init_log],
            spirv_summary,
        }
    }

    fn log(&mut self, msg: impl Into<String>) {
        self.logs.push(msg.into());
        if self.logs.len() > 80 {
            self.logs.remove(0);
        }
    }

    fn is_running(&self) -> bool {
        self.worker_rx.is_some()
    }

    fn trigger_pipeline_run(&mut self, ctx: &egui::Context) {
        if self.is_running() {
            return;
        }
        let (tx, rx) = mpsc::channel();
        self.worker_rx = Some(rx);
        self.progress_frac = 0.02;
        self.progress_text = "Starting stereo pipeline...".to_string();

        let engine = Arc::clone(&self.engine);
        let left = self.left.clone();
        let right = self.right.clone();
        let cl = self.cam_left.clone();
        let cr = self.cam_right.clone();
        let cfg = self.settings.clone();
        let repaint_ctx = ctx.clone();

        std::thread::spawn(move || {
            let res = engine.run_full(&left, &right, &cl, &cr, &cfg, |stage, pct| {
                let _ = tx.send(WorkerMsg::Progress(stage.to_string(), pct));
                repaint_ctx.request_repaint();
            });
            let _ = tx.send(WorkerMsg::Finished(Box::new(res)));
            repaint_ctx.request_repaint();
        });
    }

    fn poll_worker(&mut self) {
        let mut finished: Option<PipelineResult> = None;
        if let Some(rx) = &self.worker_rx {
            while let Ok(msg) = rx.try_recv() {
                match msg {
                    WorkerMsg::Progress(s, p) => {
                        self.progress_text = s;
                        self.progress_frac = p;
                    }
                    WorkerMsg::Finished(res) => {
                        finished = Some(*res);
                    }
                }
            }
        }
        if let Some(res) = finished {
            let msg = format!(
                "Completed {}x{} in {:.1} ms (valid {:.1}%, {} pts, err {:.4} m)",
                res.left_raw.width,
                res.left_raw.height,
                res.timings.total_ms,
                res.disp_filtered.valid_fraction() * 100.0,
                res.point_cloud.valid_count(),
                res.point_cloud.mean_error()
            );
            self.result = res;
            self.worker_rx = None;
            self.progress_frac = 1.0;
            self.progress_text = "Ready".to_string();
            self.tex_dirty = true;
            self.log(msg);
        }
    }

    fn generate_synthetic(&mut self, w: u32, h: u32, ctx: &egui::Context) {
        let ds = synthetic::generate_synthetic_stereo(w, h);
        self.left = ds.left_image;
        self.right = ds.right_image;
        self.cam_left = ds.left_cam;
        self.cam_right = ds.right_cam;
        self.settings.search_min_y = 0;
        self.settings.search_max_y = 0;
        self.log(format!("Generated synthetic planetary crater pair ({}x{})", w, h));
        self.trigger_pipeline_run(ctx);
    }

    fn load_from_paths(&mut self, ctx: &egui::Context) {
        let l = match load_image(Path::new(self.left_path.trim())) {
            Ok(img) => img,
            Err(e) => {
                self.log(format!("Error loading left image: {}", e));
                return;
            }
        };
        let r = match load_image(Path::new(self.right_path.trim())) {
            Ok(img) => img,
            Err(e) => {
                self.log(format!("Error loading right image: {}", e));
                return;
            }
        };
        let cl = match load_camera(Path::new(self.cam_left_path.trim())) {
            Ok(c) => c,
            Err(e) => {
                self.log(format!("Error loading left camera: {}", e));
                return;
            }
        };
        let cr = match load_camera(Path::new(self.cam_right_path.trim())) {
            Ok(c) => c,
            Err(e) => {
                self.log(format!("Error loading right camera: {}", e));
                return;
            }
        };
        self.left = l;
        self.right = r;
        self.cam_left = cl;
        self.cam_right = cr;
        self.log("Loaded custom stereo pair & cameras".to_string());
        self.trigger_pipeline_run(ctx);
    }

    fn recompute_hillshade_live(&mut self) {
        let vk = match self.settings.backend {
            ComputeBackendPreference::Cpu => None,
            _ => self.engine.vulkan.as_ref(),
        };
        let (hs, _) = dem::compute_hillshade(
            &self.result.dem.elevation,
            self.result.dem.pixel_size as f32,
            self.settings.hillshade_azimuth_deg,
            self.settings.hillshade_elevation_deg,
            vk,
        );
        self.result.dem.hillshade = hs;
        if self.active_layer == Layer::Hillshade {
            self.tex_dirty = true;
        }
    }

    fn active_image_and_cmap(&self) -> (ImageF32, Colormap) {
        match self.active_layer {
            Layer::Left => (self.result.left_raw.clone(), Colormap::Grayscale),
            Layer::Right => (self.result.right_raw.clone(), Colormap::Grayscale),
            Layer::Prefiltered => (self.result.left_pprc.clone(), Colormap::Grayscale),
            Layer::DispRaw => (self.result.disp_raw.dx_image(), self.colormap),
            Layer::DispFiltered => (self.result.disp_filtered.dx_image(), self.colormap),
            Layer::Dem | Layer::Terrain3D => (self.result.dem.elevation.clone(), self.colormap),
            Layer::Hillshade => (self.result.dem.hillshade.clone(), Colormap::Grayscale),
            Layer::TriError => (self.result.point_cloud.error_image(), Colormap::Inferno),
        }
    }

    fn ensure_textures(&mut self, ctx: &egui::Context) {
        if !self.tex_dirty && self.tex_main.is_some() {
            return;
        }
        let (img, cmap) = self.active_image_and_cmap();
        let rgba = img.to_rgba8(cmap);
        let ci = ColorImage::from_rgba_unmultiplied([img.width as usize, img.height as usize], &rgba);
        self.tex_main = Some(ctx.load_texture("stereo_main", ci, TextureOptions::NEAREST));

        let l_rgba = self.result.left_raw.to_rgba8(Colormap::Grayscale);
        let l_ci = ColorImage::from_rgba_unmultiplied(
            [self.result.left_raw.width as usize, self.result.left_raw.height as usize],
            &l_rgba,
        );
        self.tex_left = Some(ctx.load_texture("stereo_left_ref", l_ci, TextureOptions::NEAREST));
        self.tex_dirty = false;
    }
}

impl eframe::App for StereoEguiApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        self.poll_worker();
        self.ensure_textures(ctx);

        // Keyboard shortcuts
        if !ctx.wants_keyboard_input() {
            ctx.input(|i| {
                if i.key_pressed(egui::Key::R) {
                    self.trigger_pipeline_run(ctx);
                }
                if i.key_pressed(egui::Key::C) {
                    self.colormap = self.colormap.next();
                    self.tex_dirty = true;
                }
                let keys = [
                    egui::Key::Num1,
                    egui::Key::Num2,
                    egui::Key::Num3,
                    egui::Key::Num4,
                    egui::Key::Num5,
                    egui::Key::Num6,
                    egui::Key::Num7,
                    egui::Key::Num8,
                    egui::Key::Num9,
                ];
                for (idx, k) in keys.iter().enumerate() {
                    if i.key_pressed(*k) {
                        self.active_layer = Layer::ALL[idx].0;
                        self.tex_dirty = true;
                    }
                }
            });
        }

        // Top Bar
        egui::TopBottomPanel::top("top_bar").show(ctx, |ui| {
            ui.add_space(4.0);
            ui.horizontal(|ui| {
                ui.label(
                    RichText::new("RUST STEREO PIPELINE VULKAN")
                        .strong()
                        .size(16.0)
                        .color(Color32::from_rgb(76, 175, 255)),
                );
                ui.separator();

                let (badge_txt, badge_col) = if self.engine.vulkan.is_some() {
                    ("VULKAN COMPUTE GPU", Color32::from_rgb(46, 204, 113))
                } else {
                    ("CPU MULTI-THREAD FALLBACK", Color32::from_rgb(245, 176, 65))
                };
                ui.label(RichText::new(badge_txt).small().strong().color(badge_col));

                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    if ui.button("Export GeoTIFF / PLY").clicked() {
                        match StereoPipelineEngine::save_outputs(Path::new(&self.out_prefix), &self.result) {
                            Ok(_) => self.log(format!("Exported outputs to {}-*.tif / -PC.ply", self.out_prefix)),
                            Err(e) => self.log(format!("Export error: {}", e)),
                        }
                    }
                    if ui.button("Reset View").clicked() {
                        self.zoom = 1.0;
                        self.pan = Vec2::ZERO;
                        self.yaw = 0.65;
                        self.pitch = 0.55;
                    }
                    ui.checkbox(&mut self.split_view, "Split View");
                    let run_btn = egui::Button::new(
                        RichText::new(if self.is_running() { "Running..." } else { "Run Pipeline [R]" })
                            .strong()
                            .color(Color32::from_rgb(10, 16, 24)),
                    )
                    .fill(Color32::from_rgb(56, 152, 255));
                    if ui.add_enabled(!self.is_running(), run_btn).clicked() {
                        self.trigger_pipeline_run(ctx);
                    }
                });
            });
            ui.add_space(4.0);
        });

        // Bottom Status & Progress Bar
        egui::TopBottomPanel::bottom("status_bar").show(ctx, |ui| {
            ui.horizontal(|ui| {
                if self.is_running() {
                    ui.add(
                        egui::ProgressBar::new(self.progress_frac)
                            .desired_width(220.0)
                            .text(&self.progress_text),
                    );
                } else {
                    ui.label(RichText::new(&self.hover_readout).monospace().small());
                }
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    ui.label(
                        RichText::new(format!(
                            "Zoom: {:.0}% | {} | Keys: 1-9 Layers, R Run, C Colormap",
                            self.zoom * 100.0,
                            self.result.timings.active_backend
                        ))
                        .small()
                        .weak(),
                    );
                });
            });
        });

        // Left Parameter Inspector
        egui::SidePanel::left("controls_panel")
            .default_width(295.0)
            .resizable(true)
            .show(ctx, |ui| {
                egui::ScrollArea::vertical().show(ui, |ui| {
                    ui.heading("Pipeline Parameters");
                    ui.separator();

                    egui::CollapsingHeader::new("Synthetic Terrain & Input Files")
                        .default_open(true)
                        .show(ui, |ui| {
                            let presets: [(&str, u32, u32); 4] = [
                                ("128 x 96 (Fast)", 128, 96),
                                ("192 x 144 (Balanced)", 192, 144),
                                ("256 x 192 (Detailed)", 256, 192),
                                ("320 x 240 (High-Res)", 320, 240),
                            ];
                            egui::ComboBox::from_label("Preset")
                                .selected_text(presets[self.synth_size_idx].0)
                                .show_ui(ui, |ui| {
                                    for (i, &(lbl, _, _)) in presets.iter().enumerate() {
                                        ui.selectable_value(&mut self.synth_size_idx, i, lbl);
                                    }
                                });
                            if ui
                                .add_enabled(!self.is_running(), egui::Button::new("Generate Synthetic Crater DEM"))
                                .clicked()
                            {
                                let (_, w, h) = presets[self.synth_size_idx];
                                self.generate_synthetic(w, h, ctx);
                            }

                            ui.separator();
                            ui.label(RichText::new("Custom Stereo Files (TIFF/PFM/PGM + .tsai/.rpb)").small().weak());
                            ui.horizontal(|ui| {
                                ui.label("L img");
                                ui.text_edit_singleline(&mut self.left_path);
                            });
                            ui.horizontal(|ui| {
                                ui.label("R img");
                                ui.text_edit_singleline(&mut self.right_path);
                            });
                            ui.horizontal(|ui| {
                                ui.label("L cam");
                                ui.text_edit_singleline(&mut self.cam_left_path);
                            });
                            ui.horizontal(|ui| {
                                ui.label("R cam");
                                ui.text_edit_singleline(&mut self.cam_right_path);
                            });
                            ui.horizontal(|ui| {
                                ui.label("Out  ");
                                ui.text_edit_singleline(&mut self.out_prefix);
                            });
                            if ui.add_enabled(!self.is_running(), egui::Button::new("Load & Run Custom Pair")).clicked() {
                                self.load_from_paths(ctx);
                            }
                        });

                    egui::CollapsingHeader::new("Stage 0: stereo_pprc")
                        .default_open(true)
                        .show(ui, |ui| {
                            ui.checkbox(&mut self.settings.normalize_intensity, "Normalize intensity");
                            egui::ComboBox::from_label("Prefilter")
                                .selected_text(self.settings.prefilter_mode.name())
                                .show_ui(ui, |ui| {
                                    ui.selectable_value(&mut self.settings.prefilter_mode, PrefilterMode::None, "None");
                                    ui.selectable_value(&mut self.settings.prefilter_mode, PrefilterMode::SubtractedMean, "SubtractedMean");
                                    ui.selectable_value(&mut self.settings.prefilter_mode, PrefilterMode::LaplacianOfGaussian, "LoG");
                                });
                            ui.add(
                                egui::Slider::new(&mut self.settings.prefilter_kernel_width, 0.5..=25.0)
                                    .text("Kernel width"),
                            );
                        });

                    egui::CollapsingHeader::new("Stage 1: stereo_corr")
                        .default_open(true)
                        .show(ui, |ui| {
                            egui::ComboBox::from_label("Algorithm")
                                .selected_text(self.settings.stereo_algorithm.name())
                                .show_ui(ui, |ui| {
                                    ui.selectable_value(&mut self.settings.stereo_algorithm, StereoAlgorithm::AspBm, "asp_bm (Vulkan/CPU)");
                                    ui.selectable_value(&mut self.settings.stereo_algorithm, StereoAlgorithm::AspSgm, "asp_sgm (4-Path DP)");
                                    ui.selectable_value(&mut self.settings.stereo_algorithm, StereoAlgorithm::AspMgm, "asp_mgm (Quadrant)");
                                });
                            egui::ComboBox::from_label("Cost Metric")
                                .selected_text(self.settings.cost_mode.name())
                                .show_ui(ui, |ui| {
                                    ui.selectable_value(&mut self.settings.cost_mode, CostMode::Sad, "SAD");
                                    ui.selectable_value(&mut self.settings.cost_mode, CostMode::Ssd, "SSD");
                                    ui.selectable_value(&mut self.settings.cost_mode, CostMode::Ncc, "NCC (Zero-Mean)");
                                    ui.selectable_value(&mut self.settings.cost_mode, CostMode::Census, "Census 5x5");
                                });
                            let mut k = self.settings.corr_kernel_x;
                            if ui.add(egui::Slider::new(&mut k, 3..=21).step_by(2.0).text("Kernel")).changed() {
                                self.settings.corr_kernel_x = k;
                                self.settings.corr_kernel_y = k;
                            }
                            ui.add(egui::Slider::new(&mut self.settings.search_min_x, -96..=0).text("Min dX"));
                            ui.add(egui::Slider::new(&mut self.settings.search_max_x, 0..=96).text("Max dX"));
                            ui.add(egui::Slider::new(&mut self.settings.sgm_p1, 0.02..=0.5).text("SGM P1"));
                            ui.add(egui::Slider::new(&mut self.settings.sgm_p2, 0.1..=1.5).text("SGM P2"));
                            ui.add(egui::Slider::new(&mut self.settings.lr_threshold, 0.5..=5.0).text("LR Check px"));
                        });

                    egui::CollapsingHeader::new("Stage 2 & 3: rfne & fltr")
                        .default_open(true)
                        .show(ui, |ui| {
                            egui::ComboBox::from_label("Subpixel")
                                .selected_text(self.settings.subpixel_mode.name())
                                .show_ui(ui, |ui| {
                                    ui.selectable_value(&mut self.settings.subpixel_mode, SubpixelMode::None, "None");
                                    ui.selectable_value(&mut self.settings.subpixel_mode, SubpixelMode::Parabola, "Parabola");
                                    ui.selectable_value(&mut self.settings.subpixel_mode, SubpixelMode::BayesEmAffine, "BayesEM-Affine");
                                    ui.selectable_value(&mut self.settings.subpixel_mode, SubpixelMode::Affine, "Affine LK");
                                });
                            ui.add(egui::Slider::new(&mut self.settings.rm_threshold, 0.5..=6.0).text("Outlier thresh"));
                            ui.add(egui::Slider::new(&mut self.settings.rm_min_matches, 10..=90).text("Min matches %"));
                            ui.checkbox(&mut self.settings.fill_holes, "Fill small disparity holes");
                        });

                    egui::CollapsingHeader::new("Stage 5: DEM, Hillshade & Compute")
                        .default_open(true)
                        .show(ui, |ui| {
                            let az_changed = ui
                                .add(egui::Slider::new(&mut self.settings.hillshade_azimuth_deg, 0.0..=360.0).text("Sun Azimuth"))
                                .changed();
                            let el_changed = ui
                                .add(egui::Slider::new(&mut self.settings.hillshade_elevation_deg, 5.0..=85.0).text("Sun Elevation"))
                                .changed();
                            if az_changed || el_changed {
                                self.recompute_hillshade_live();
                            }
                            egui::ComboBox::from_label("Backend")
                                .selected_text(self.settings.backend.name())
                                .show_ui(ui, |ui| {
                                    ui.selectable_value(&mut self.settings.backend, ComputeBackendPreference::Auto, "Auto (Vulkan -> CPU)");
                                    ui.selectable_value(&mut self.settings.backend, ComputeBackendPreference::Vulkan, "Vulkan Compute");
                                    ui.selectable_value(&mut self.settings.backend, ComputeBackendPreference::Cpu, "CPU Multi-thread");
                                });
                            ui.add(egui::Slider::new(&mut self.settings.threads, 1..=32).text("CPU Threads"));
                        });
                });
            });

        // Right Telemetry & Diagnostics Panel
        egui::SidePanel::right("telemetry_panel")
            .default_width(270.0)
            .resizable(true)
            .show(ctx, |ui| {
                egui::ScrollArea::vertical().show(ui, |ui| {
                    ui.heading("Stage Telemetry");
                    ui.separator();

                    let t = &self.result.timings;
                    let max_stage = [t.pprc_ms, t.corr_ms, t.rfne_ms, t.fltr_ms, t.tri_ms, t.dem_ms]
                        .into_iter()
                        .fold(1.0f64, f64::max);
                    let stages = [
                        ("0. PPRC", t.pprc_ms, Color32::from_rgb(56, 152, 255)),
                        ("1. CORR", t.corr_ms, Color32::from_rgb(56, 152, 255)),
                        ("2. RFNE", t.rfne_ms, Color32::from_rgb(138, 121, 255)),
                        ("3. FLTR", t.fltr_ms, Color32::from_rgb(56, 198, 210)),
                        ("4. TRI ", t.tri_ms, Color32::from_rgb(245, 176, 65)),
                        ("5. DEM ", t.dem_ms, Color32::from_rgb(46, 204, 113)),
                    ];
                    for (name, ms, color) in stages {
                        ui.horizontal(|ui| {
                            ui.monospace(name);
                            let frac = (ms / max_stage).clamp(0.02, 1.0) as f32;
                            ui.add(
                                egui::ProgressBar::new(frac)
                                    .desired_width(115.0)
                                    .fill(color),
                            );
                            ui.monospace(format!("{:6.1}ms", ms));
                        });
                    }
                    ui.separator();
                    ui.label(
                        RichText::new(format!("TOTAL: {:.2} ms", t.total_ms))
                            .strong()
                            .monospace()
                            .color(Color32::from_rgb(46, 204, 113)),
                    );

                    ui.add_space(8.0);
                    ui.heading("Photogrammetry Stats");
                    ui.separator();
                    let (dem_min, dem_max, dem_mean, _) = self.result.dem.elevation.stats();
                    egui::Grid::new("stats_grid").num_columns(2).striped(true).show(ui, |ui| {
                        ui.label("Resolution");
                        ui.monospace(format!("{} x {}", self.left.width, self.left.height));
                        ui.end_row();
                        ui.label("Valid Disparity");
                        ui.monospace(format!("{:.2} %", self.result.disp_filtered.valid_fraction() * 100.0));
                        ui.end_row();
                        ui.label("3D Points");
                        ui.monospace(format!("{}", self.result.point_cloud.valid_count()));
                        ui.end_row();
                        ui.label("Triang. Error");
                        ui.monospace(format!("{:.4} m", self.result.point_cloud.mean_error()));
                        ui.end_row();
                        ui.label("DEM Range");
                        ui.monospace(format!("{:.1}..{:.1} m", dem_min, dem_max));
                        ui.end_row();
                        ui.label("DEM Mean");
                        ui.monospace(format!("{:.2} m", dem_mean));
                        ui.end_row();
                        ui.label("Vulkan Dispatches");
                        ui.monospace(format!("{}", t.vk_dispatches));
                        ui.end_row();
                    });

                    ui.add_space(8.0);
                    egui::CollapsingHeader::new("SPIR-V 1.0 Compute Shaders")
                        .default_open(true)
                        .show(ui, |ui| {
                            for (name, bytes, insts) in &self.spirv_summary {
                                ui.monospace(format!("✓ {} ({}B, {} ops)", name, bytes, insts));
                            }
                        });

                    ui.add_space(8.0);
                    egui::CollapsingHeader::new("Activity Log")
                        .default_open(true)
                        .show(ui, |ui| {
                            for entry in self.logs.iter().rev().take(12) {
                                ui.label(RichText::new(entry).small().monospace().weak());
                            }
                        });
                });
            });

        // Central Interactive Viewport
        egui::CentralPanel::default().show(ctx, |ui| {
            // Layer Tabs & Colormap Bar
            ui.horizontal_wrapped(|ui| {
                for &(layer, name) in &Layer::ALL {
                    if ui.selectable_label(self.active_layer == layer, name).clicked() {
                        self.active_layer = layer;
                        self.tex_dirty = true;
                    }
                }
                ui.separator();
                let prev_cmap = self.colormap;
                egui::ComboBox::from_id_salt("cmap_combo")
                    .selected_text(format!("Colormap: {}", self.colormap.name()))
                    .show_ui(ui, |ui| {
                        for cm in [
                            Colormap::Turbo,
                            Colormap::Viridis,
                            Colormap::Inferno,
                            Colormap::Terrain,
                            Colormap::Grayscale,
                        ] {
                            ui.selectable_value(&mut self.colormap, cm, cm.name());
                        }
                    });
                if self.colormap != prev_cmap {
                    self.tex_dirty = true;
                }
                if self.active_layer == Layer::Terrain3D {
                    ui.add(egui::Slider::new(&mut self.z_exaggeration, 0.2..=2.0).text("Z Scale"));
                }
            });
            ui.separator();

            let avail = ui.available_rect_before_wrap();
            if self.active_layer == Layer::Terrain3D {
                self.draw_3d_dem(ui, avail);
            } else if self.split_view {
                let half_w = (avail.width() - 6.0) * 0.5;
                let left_rect = Rect::from_min_size(avail.min, Vec2::new(half_w, avail.height()));
                let right_rect = Rect::from_min_size(
                    Pos2::new(avail.min.x + half_w + 6.0, avail.min.y),
                    Vec2::new(half_w, avail.height()),
                );
                if let Some(tex_l) = self.tex_left.clone() {
                    let left_img = self.result.left_raw.clone();
                    self.draw_2d_layer(ui, left_rect, &tex_l, &left_img, Colormap::Grayscale, "Left Reference");
                }
                if let Some(tex_m) = self.tex_main.clone() {
                    let (img, cmap) = self.active_image_and_cmap();
                    self.draw_2d_layer(ui, right_rect, &tex_m, &img, cmap, self.active_layer.unit_label());
                }
            } else if let Some(tex_m) = self.tex_main.clone() {
                let (img, cmap) = self.active_image_and_cmap();
                self.draw_2d_layer(ui, avail, &tex_m, &img, cmap, self.active_layer.unit_label());
            }
        });
    }
}

impl StereoEguiApp {
    fn draw_2d_layer(
        &mut self,
        ui: &mut egui::Ui,
        rect: Rect,
        tex: &TextureHandle,
        img: &ImageF32,
        cmap: Colormap,
        unit: &str,
    ) {
        let resp = ui.allocate_rect(rect, Sense::click_and_drag());
        if resp.dragged() {
            self.pan += resp.drag_delta();
        }
        if resp.hovered() {
            let scroll = ui.input(|i| i.smooth_scroll_delta.y);
            if scroll.abs() > 0.1 {
                self.zoom = (self.zoom * (1.0 + scroll * 0.002)).clamp(0.25, 16.0);
            }
        }

        let painter = ui.painter_at(rect);
        painter.rect_filled(rect, 4.0, Color32::from_rgb(8, 11, 16));

        let fit = (rect.width() / img.width.max(1) as f32)
            .min(rect.height() / img.height.max(1) as f32)
            * 0.94
            * self.zoom;
        let size = Vec2::new(img.width as f32 * fit, img.height as f32 * fit);
        let center = rect.center() + self.pan;
        let img_rect = Rect::from_center_size(center, size);

        painter.image(
            tex.id(),
            img_rect,
            Rect::from_min_max(Pos2::ZERO, Pos2::new(1.0, 1.0)),
            Color32::WHITE,
        );
        painter.rect_stroke(
            img_rect,
            0.0,
            Stroke::new(1.0_f32, Color32::from_rgb(56, 68, 86)),
            egui::StrokeKind::Outside,
        );

        // Pixel inspector readout
        if let Some(pos) = resp.hover_pos() {
            if img_rect.contains(pos) {
                let u = ((pos.x - img_rect.min.x) / img_rect.width()).clamp(0.0, 0.9999);
                let v = ((pos.y - img_rect.min.y) / img_rect.height()).clamp(0.0, 0.9999);
                let ix = (u * img.width as f32) as u32;
                let iy = (v * img.height as f32) as u32;
                let val = img.get(ix, iy);
                self.hover_readout = if val.is_finite() {
                    format!("Pixel ({:4}, {:4}) = {:9.3} {}", ix, iy, val, unit)
                } else {
                    format!("Pixel ({:4}, {:4}) = NoData / Occluded", ix, iy)
                };
            }
        }

        // Colorbar overlay
        let (min_v, max_v, _, _) = img.stats();
        let bar_w = 180.0;
        let bar_h = 12.0;
        let bar_rect = Rect::from_min_size(
            Pos2::new(rect.max.x - bar_w - 16.0, rect.max.y - bar_h - 14.0),
            Vec2::new(bar_w, bar_h),
        );
        let steps = 45;
        let seg_w = bar_w / steps as f32;
        for s in 0..steps {
            let t = s as f32 / (steps - 1) as f32;
            let [r, g, b, _] = cmap.sample(t);
            let seg = Rect::from_min_size(
                Pos2::new(bar_rect.min.x + s as f32 * seg_w, bar_rect.min.y),
                Vec2::new(seg_w + 0.5, bar_h),
            );
            painter.rect_filled(seg, 0.0, Color32::from_rgb(r, g, b));
        }
        painter.rect_stroke(
            bar_rect,
            0.0,
            Stroke::new(1.0_f32, Color32::from_rgb(90, 105, 125)),
            egui::StrokeKind::Outside,
        );
        painter.text(
            Pos2::new(bar_rect.min.x, bar_rect.min.y - 14.0),
            egui::Align2::LEFT_TOP,
            format!("{:.2} .. {:.2} {}", min_v, max_v, unit),
            egui::FontId::monospace(11.0),
            Color32::from_rgb(220, 230, 240),
        );
    }

    fn draw_3d_dem(&mut self, ui: &mut egui::Ui, rect: Rect) {
        let resp = ui.allocate_rect(rect, Sense::click_and_drag());
        if resp.dragged() {
            let d = resp.drag_delta();
            self.yaw += d.x * 0.01;
            self.pitch = (self.pitch + d.y * 0.01).clamp(0.15, 1.48);
        }
        if resp.hovered() {
            let scroll = ui.input(|i| i.smooth_scroll_delta.y);
            if scroll.abs() > 0.1 {
                self.zoom = (self.zoom * (1.0 + scroll * 0.002)).clamp(0.4, 4.0);
            }
        }

        let painter = ui.painter_at(rect);
        painter.rect_filled(rect, 4.0, Color32::from_rgb(8, 11, 16));

        let dem = &self.result.dem.elevation;
        let (min_z, max_z, _, _) = dem.stats();
        let z_range = (max_z - min_z).max(1e-4);

        let step_x = (dem.width / 60).max(1);
        let step_y = (dem.height / 60).max(1);
        let gx_n = ((dem.width + step_x - 1) / step_x) as usize;
        let gy_n = ((dem.height + step_y - 1) / step_y) as usize;

        let (sy, cy) = self.yaw.sin_cos();
        let (sp, cp) = self.pitch.sin_cos();
        let scale = rect.width().min(rect.height()) * 0.42 * self.zoom;
        let cx = rect.center().x;
        let cy_s = rect.center().y + rect.height() * 0.05;

        let mut projected: Vec<Option<(Pos2, Color32)>> = vec![None; gx_n * gy_n];
        for j in 0..gy_n {
            let y = (j as u32 * step_y).min(dem.height - 1);
            let ny = (y as f32 / dem.height.max(1) as f32) * 2.0 - 1.0;
            for i in 0..gx_n {
                let x = (i as u32 * step_x).min(dem.width - 1);
                let v = dem.get(x, y);
                if !v.is_finite() {
                    continue;
                }
                let nx = (x as f32 / dem.width.max(1) as f32) * 2.0 - 1.0;
                let t = ((v - min_z) / z_range).clamp(0.0, 1.0);
                let nz = (t - 0.5) * self.z_exaggeration;

                let rx = nx * cy - ny * sy;
                let ry = nx * sy + ny * cy;
                let py3 = ry * cp - nz * sp;
                let pz3 = ry * sp + nz * cp;
                let depth = 3.2 - py3;
                if depth < 0.3 {
                    continue;
                }
                let persp = scale / depth * 2.3;
                let p2 = Pos2::new(cx + rx * persp, cy_s - pz3 * persp);
                let [r, g, b, _] = self.colormap.sample(t);
                projected[j * gx_n + i] = Some((p2, Color32::from_rgb(r, g, b)));
            }
        }

        for j in 0..gy_n {
            for i in 0..gx_n {
                if let Some((p0, c0)) = projected[j * gx_n + i] {
                    if i + 1 < gx_n {
                        if let Some((p1, _)) = projected[j * gx_n + (i + 1)] {
                            painter.line_segment([p0, p1], Stroke::new(1.15_f32, c0));
                        }
                    }
                    if j + 1 < gy_n {
                        if let Some((p1, _)) = projected[(j + 1) * gx_n + i] {
                            painter.line_segment([p0, p1], Stroke::new(1.15_f32, c0));
                        }
                    }
                }
            }
        }

        painter.text(
            Pos2::new(rect.min.x + 12.0, rect.max.y - 20.0),
            egui::Align2::LEFT_TOP,
            format!(
                "3D Interactive DEM ({:.1}m .. {:.1}m) — Drag to orbit, Scroll to zoom",
                min_z, max_z
            ),
            egui::FontId::monospace(12.0),
            Color32::from_rgb(160, 175, 195),
        );
    }
}

fn apply_dark_scientific_theme(ctx: &egui::Context) {
    let mut visuals = egui::Visuals::dark();
    visuals.panel_fill = Color32::from_rgb(18, 22, 29);
    visuals.window_fill = Color32::from_rgb(22, 27, 36);
    visuals.extreme_bg_color = Color32::from_rgb(10, 13, 18);
    visuals.selection.bg_fill = Color32::from_rgb(42, 118, 210);
    ctx.set_visuals(visuals);
}

/// Launch the interactive `egui` / `eframe` desktop GUI.
pub fn run_egui_gui(
    left: ImageF32,
    right: ImageF32,
    cam_left: AnyCamera,
    cam_right: AnyCamera,
    settings: StereoSettings,
) -> Result<(), String> {
    let native_options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1400.0, 860.0])
            .with_min_inner_size([920.0, 600.0])
            .with_title("RustStereoPipelineVulkan — egui Stereo & DEM Studio"),
        ..Default::default()
    };

    eframe::run_native(
        "RustStereoPipelineVulkan",
        native_options,
        Box::new(move |cc| Ok(Box::new(StereoEguiApp::new(cc, left, right, cam_left, cam_right, settings)))),
    )
    .map_err(|e| format!("eframe/egui error: {}", e))
}
