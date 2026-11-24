//! Pure-Rust SPIR-V 1.0 Vulkan Compute shaders for the stereo photogrammetry pipeline.
//! All shaders use 16x16 workgroups (`local_size_x = 16, local_size_y = 16, local_size_z = 1`)
//! and standard storage buffer bindings so they execute directly on any Vulkan 1.0+ GPU.

use super::spirv::*;

/// 1. Prefilter Compute Shader (Subtracted Mean / 2D Box Filter).
/// Bindings:
///   0: `in_img[]` (f32)
///   1: `out_img[]` (f32)
///   2: `params[]` (i32): `[width, height, radius, mode]` (`mode == 1` => subtracted mean, else blur)
pub fn build_prefilter_shader() -> Vec<u32> {
    let mut b = SpirvBuilder::new(16, 16, 1);
    let buf_in = b.add_storage_buffer_f32(0);
    let buf_out = b.add_storage_buffer_f32(1);
    let buf_params = b.add_storage_buffer_i32(2);

    let c0 = b.const_i32(0);
    let c1 = b.const_i32(1);
    let c2 = b.const_i32(2);
    let c3 = b.const_i32(3);
    let f0 = b.const_f32(0.0);
    let f1 = b.const_f32(1.0);

    b.begin_main();
    let var_dy = b.fn_var_i32();
    let var_dx = b.fn_var_i32();
    let var_sum = b.fn_var_f32();
    let var_cnt = b.fn_var_f32();

    let (gx, gy) = b.load_global_xy();
    let w = b.load_buf_i32(buf_params, c0, c0);
    let h = b.load_buf_i32(buf_params, c0, c1);
    let rad = b.load_buf_i32(buf_params, c0, c2);
    let mode = b.load_buf_i32(buf_params, c0, c3);

    // Bounds check: if (gx >= w || gy >= h) return;
    let oob_x = b.binop(OP_SGREATER_THAN_EQUAL, b.ty_bool, gx, w);
    let oob_y = b.binop(OP_SGREATER_THAN_EQUAL, b.ty_bool, gy, h);
    let oob = b.binop(OP_LOGICAL_OR, b.ty_bool, oob_x, oob_y);
    let lbl_ret = b.id();
    let lbl_body = b.id();
    b.selection_merge(lbl_body);
    b.branch_cond(oob, lbl_ret, lbl_body);
    b.label(lbl_ret);
    b.return_void();

    b.label(lbl_body);
    let wm1 = b.binop(OP_ISUB, b.ty_i32, w, c1);
    let hm1 = b.binop(OP_ISUB, b.ty_i32, h, c1);
    let neg_rad = b.binop(OP_ISUB, b.ty_i32, c0, rad);
    let rad_p1 = b.binop(OP_IADD, b.ty_i32, rad, c1);

    b.store(var_sum, f0);
    b.store(var_cnt, f0);
    b.store(var_dy, neg_rad);

    // Outer loop: dy = -rad ..= rad
    let lbl_y_hdr = b.id();
    let lbl_y_body = b.id();
    let lbl_y_cont = b.id();
    let lbl_y_end = b.id();
    b.branch(lbl_y_hdr);
    b.label(lbl_y_hdr);
    b.loop_merge(lbl_y_end, lbl_y_cont);
    let dy = b.load(b.ty_i32, var_dy);
    let y_cond = b.binop(OP_SLESS_THAN, b.ty_bool, dy, rad_p1);
    b.branch_cond(y_cond, lbl_y_body, lbl_y_end);

    b.label(lbl_y_body);
    let sy_raw = b.binop(OP_IADD, b.ty_i32, gy, dy);
    let sy = b.glsl3(GLSL_SCLAMP, b.ty_i32, sy_raw, c0, hm1);
    let row_off = b.binop(OP_IMUL, b.ty_i32, sy, w);
    b.store(var_dx, neg_rad);

    // Inner loop: dx = -rad ..= rad
    let lbl_x_hdr = b.id();
    let lbl_x_body = b.id();
    let lbl_x_cont = b.id();
    let lbl_x_end = b.id();
    b.branch(lbl_x_hdr);
    b.label(lbl_x_hdr);
    b.loop_merge(lbl_x_end, lbl_x_cont);
    let dx = b.load(b.ty_i32, var_dx);
    let x_cond = b.binop(OP_SLESS_THAN, b.ty_bool, dx, rad_p1);
    b.branch_cond(x_cond, lbl_x_body, lbl_x_end);

    b.label(lbl_x_body);
    let sx_raw = b.binop(OP_IADD, b.ty_i32, gx, dx);
    let sx = b.glsl3(GLSL_SCLAMP, b.ty_i32, sx_raw, c0, wm1);
    let sidx = b.binop(OP_IADD, b.ty_i32, row_off, sx);
    let val = b.load_buf_f32(buf_in, c0, sidx);
    let cur_sum = b.load(b.ty_f32, var_sum);
    let new_sum = b.binop(OP_FADD, b.ty_f32, cur_sum, val);
    b.store(var_sum, new_sum);
    let cur_cnt = b.load(b.ty_f32, var_cnt);
    let new_cnt = b.binop(OP_FADD, b.ty_f32, cur_cnt, f1);
    b.store(var_cnt, new_cnt);
    b.branch(lbl_x_cont);

    b.label(lbl_x_cont);
    let next_dx = b.binop(OP_IADD, b.ty_i32, dx, c1);
    b.store(var_dx, next_dx);
    b.branch(lbl_x_hdr);

    b.label(lbl_x_end);
    b.branch(lbl_y_cont);

    b.label(lbl_y_cont);
    let next_dy = b.binop(OP_IADD, b.ty_i32, dy, c1);
    b.store(var_dy, next_dy);
    b.branch(lbl_y_hdr);

    b.label(lbl_y_end);
    let total_sum = b.load(b.ty_f32, var_sum);
    let total_cnt = b.load(b.ty_f32, var_cnt);
    let mean = b.binop(OP_FDIV, b.ty_f32, total_sum, total_cnt);
    let center_row = b.binop(OP_IMUL, b.ty_i32, gy, w);
    let center_idx = b.binop(OP_IADD, b.ty_i32, center_row, gx);
    let center_val = b.load_buf_f32(buf_in, c0, center_idx);
    let sub_mean = b.binop(OP_FSUB, b.ty_f32, center_val, mean);
    let is_sub = b.binop(OP_IEQUAL, b.ty_bool, mode, c1);
    let result = b.select(b.ty_f32, is_sub, sub_mean, mean);
    b.store_buf_f32(buf_out, c0, center_idx, result);

    b.end_main();
    b.finish()
}

/// 2. Stereo Block-Matching Correlation Compute Shader (SAD/SSD/NCC).
/// Evaluates cost across `[min_dx..=max_dx]` for each pixel `(x, y)` in parallel on GPU.
/// Bindings:
///   0: `left[]` (f32)
///   1: `right[]` (f32)
///   2: `out_disp[]` (f32) — best horizontal disparity `dx`
///   3: `out_cost[]` (f32) — best matching cost
///   4: `params[]` (i32): `[width, height, half_kw, half_kh, min_dx, max_dx, cost_mode]`
pub fn build_stereo_bm_shader() -> Vec<u32> {
    let mut b = SpirvBuilder::new(16, 16, 1);
    let buf_l = b.add_storage_buffer_f32(0);
    let buf_r = b.add_storage_buffer_f32(1);
    let buf_disp = b.add_storage_buffer_f32(2);
    let buf_cost = b.add_storage_buffer_f32(3);
    let buf_p = b.add_storage_buffer_i32(4);

    let c0 = b.const_i32(0);
    let c1 = b.const_i32(1);
    let c2 = b.const_i32(2);
    let c3 = b.const_i32(3);
    let c4 = b.const_i32(4);
    let c5 = b.const_i32(5);
    let c6 = b.const_i32(6);
    let f0 = b.const_f32(0.0);
    let f_big = b.const_f32(1.0e20);
    let f_eps = b.const_f32(1.0e-6);
    let f1 = b.const_f32(1.0);

    b.begin_main();
    let var_d = b.fn_var_i32();
    let var_ky = b.fn_var_i32();
    let var_kx = b.fn_var_i32();
    let var_acc = b.fn_var_f32();
    let var_ncc_lr = b.fn_var_f32();
    let var_ncc_ll = b.fn_var_f32();
    let var_ncc_rr = b.fn_var_f32();
    let var_best_cost = b.fn_var_f32();
    let var_best_d = b.fn_var_i32();

    let (gx, gy) = b.load_global_xy();
    let w = b.load_buf_i32(buf_p, c0, c0);
    let h = b.load_buf_i32(buf_p, c0, c1);
    let hkw = b.load_buf_i32(buf_p, c0, c2);
    let hkh = b.load_buf_i32(buf_p, c0, c3);
    let min_dx = b.load_buf_i32(buf_p, c0, c4);
    let max_dx = b.load_buf_i32(buf_p, c0, c5);
    let cost_mode = b.load_buf_i32(buf_p, c0, c6);

    let oob_x = b.binop(OP_SGREATER_THAN_EQUAL, b.ty_bool, gx, w);
    let oob_y = b.binop(OP_SGREATER_THAN_EQUAL, b.ty_bool, gy, h);
    let oob = b.binop(OP_LOGICAL_OR, b.ty_bool, oob_x, oob_y);
    let lbl_ret = b.id();
    let lbl_body = b.id();
    b.selection_merge(lbl_body);
    b.branch_cond(oob, lbl_ret, lbl_body);
    b.label(lbl_ret);
    b.return_void();

    b.label(lbl_body);
    let wm1 = b.binop(OP_ISUB, b.ty_i32, w, c1);
    let hm1 = b.binop(OP_ISUB, b.ty_i32, h, c1);
    let neg_hkw = b.binop(OP_ISUB, b.ty_i32, c0, hkw);
    let neg_hkh = b.binop(OP_ISUB, b.ty_i32, c0, hkh);
    let hkw_p1 = b.binop(OP_IADD, b.ty_i32, hkw, c1);
    let hkh_p1 = b.binop(OP_IADD, b.ty_i32, hkh, c1);
    let max_dx_p1 = b.binop(OP_IADD, b.ty_i32, max_dx, c1);

    b.store(var_best_cost, f_big);
    b.store(var_best_d, min_dx);
    b.store(var_d, min_dx);

    // Loop over candidate disparity d
    let lbl_d_hdr = b.id();
    let lbl_d_body = b.id();
    let lbl_d_cont = b.id();
    let lbl_d_end = b.id();
    b.branch(lbl_d_hdr);
    b.label(lbl_d_hdr);
    b.loop_merge(lbl_d_end, lbl_d_cont);
    let d = b.load(b.ty_i32, var_d);
    let d_cond = b.binop(OP_SLESS_THAN, b.ty_bool, d, max_dx_p1);
    b.branch_cond(d_cond, lbl_d_body, lbl_d_end);

    b.label(lbl_d_body);
    b.store(var_acc, f0);
    b.store(var_ncc_lr, f0);
    b.store(var_ncc_ll, f0);
    b.store(var_ncc_rr, f0);
    b.store(var_ky, neg_hkh);

    // Kernel Y loop
    let lbl_ky_hdr = b.id();
    let lbl_ky_body = b.id();
    let lbl_ky_cont = b.id();
    let lbl_ky_end = b.id();
    b.branch(lbl_ky_hdr);
    b.label(lbl_ky_hdr);
    b.loop_merge(lbl_ky_end, lbl_ky_cont);
    let ky = b.load(b.ty_i32, var_ky);
    let ky_cond = b.binop(OP_SLESS_THAN, b.ty_bool, ky, hkh_p1);
    b.branch_cond(ky_cond, lbl_ky_body, lbl_ky_end);

    b.label(lbl_ky_body);
    let sy_raw = b.binop(OP_IADD, b.ty_i32, gy, ky);
    let sy = b.glsl3(GLSL_SCLAMP, b.ty_i32, sy_raw, c0, hm1);
    let row_off = b.binop(OP_IMUL, b.ty_i32, sy, w);
    b.store(var_kx, neg_hkw);

    // Kernel X loop
    let lbl_kx_hdr = b.id();
    let lbl_kx_body = b.id();
    let lbl_kx_cont = b.id();
    let lbl_kx_end = b.id();
    b.branch(lbl_kx_hdr);
    b.label(lbl_kx_hdr);
    b.loop_merge(lbl_kx_end, lbl_kx_cont);
    let kx = b.load(b.ty_i32, var_kx);
    let kx_cond = b.binop(OP_SLESS_THAN, b.ty_bool, kx, hkw_p1);
    b.branch_cond(kx_cond, lbl_kx_body, lbl_kx_end);

    b.label(lbl_kx_body);
    let lx_raw = b.binop(OP_IADD, b.ty_i32, gx, kx);
    let lx = b.glsl3(GLSL_SCLAMP, b.ty_i32, lx_raw, c0, wm1);
    let rx_raw = b.binop(OP_IADD, b.ty_i32, lx_raw, d);
    let rx = b.glsl3(GLSL_SCLAMP, b.ty_i32, rx_raw, c0, wm1);
    let l_idx = b.binop(OP_IADD, b.ty_i32, row_off, lx);
    let r_idx = b.binop(OP_IADD, b.ty_i32, row_off, rx);
    let lv = b.load_buf_f32(buf_l, c0, l_idx);
    let rv = b.load_buf_f32(buf_r, c0, r_idx);

    let diff = b.binop(OP_FSUB, b.ty_f32, lv, rv);
    let abs_diff = b.glsl1(GLSL_FABS, b.ty_f32, diff);
    let sq_diff = b.binop(OP_FMUL, b.ty_f32, diff, diff);
    let is_ssd = b.binop(OP_IEQUAL, b.ty_bool, cost_mode, c1);
    let term = b.select(b.ty_f32, is_ssd, sq_diff, abs_diff);
    let cur_acc = b.load(b.ty_f32, var_acc);
    let new_acc = b.binop(OP_FADD, b.ty_f32, cur_acc, term);
    b.store(var_acc, new_acc);

    // Accumulate zero-mean NCC dot products
    let lr = b.binop(OP_FMUL, b.ty_f32, lv, rv);
    let ll = b.binop(OP_FMUL, b.ty_f32, lv, lv);
    let rr = b.binop(OP_FMUL, b.ty_f32, rv, rv);
    let c_lr = b.load(b.ty_f32, var_ncc_lr);
    let c_ll = b.load(b.ty_f32, var_ncc_ll);
    let c_rr = b.load(b.ty_f32, var_ncc_rr);
    let n_lr = b.binop(OP_FADD, b.ty_f32, c_lr, lr);
    let n_ll = b.binop(OP_FADD, b.ty_f32, c_ll, ll);
    let n_rr = b.binop(OP_FADD, b.ty_f32, c_rr, rr);
    b.store(var_ncc_lr, n_lr);
    b.store(var_ncc_ll, n_ll);
    b.store(var_ncc_rr, n_rr);
    b.branch(lbl_kx_cont);

    b.label(lbl_kx_cont);
    let nkx = b.binop(OP_IADD, b.ty_i32, kx, c1);
    b.store(var_kx, nkx);
    b.branch(lbl_kx_hdr);

    b.label(lbl_kx_end);
    b.branch(lbl_ky_cont);

    b.label(lbl_ky_cont);
    let nky = b.binop(OP_IADD, b.ty_i32, ky, c1);
    b.store(var_ky, nky);
    b.branch(lbl_ky_hdr);

    b.label(lbl_ky_end);
    let sad_ssd = b.load(b.ty_f32, var_acc);
    let f_lr = b.load(b.ty_f32, var_ncc_lr);
    let f_ll = b.load(b.ty_f32, var_ncc_ll);
    let f_rr = b.load(b.ty_f32, var_ncc_rr);
    let denom_sq = b.binop(OP_FMUL, b.ty_f32, f_ll, f_rr);
    let denom_safe = b.binop(OP_FADD, b.ty_f32, denom_sq, f_eps);
    let denom = b.glsl1(GLSL_SQRT, b.ty_f32, denom_safe);
    let ncc = b.binop(OP_FDIV, b.ty_f32, f_lr, denom);
    let ncc_cost = b.binop(OP_FSUB, b.ty_f32, f1, ncc);
    let is_ncc = b.binop(OP_IEQUAL, b.ty_bool, cost_mode, c2);
    let final_cost = b.select(b.ty_f32, is_ncc, ncc_cost, sad_ssd);

    let best_cost = b.load(b.ty_f32, var_best_cost);
    let best_d = b.load(b.ty_i32, var_best_d);
    let better = b.binop(OP_FORD_LESS_THAN, b.ty_bool, final_cost, best_cost);
    let next_best_cost = b.select(b.ty_f32, better, final_cost, best_cost);
    let next_best_d = b.select(b.ty_i32, better, d, best_d);
    b.store(var_best_cost, next_best_cost);
    b.store(var_best_d, next_best_d);
    b.branch(lbl_d_cont);

    b.label(lbl_d_cont);
    let nd = b.binop(OP_IADD, b.ty_i32, d, c1);
    b.store(var_d, nd);
    b.branch(lbl_d_hdr);

    b.label(lbl_d_end);
    let out_row = b.binop(OP_IMUL, b.ty_i32, gy, w);
    let out_idx = b.binop(OP_IADD, b.ty_i32, out_row, gx);
    let win_d = b.load(b.ty_i32, var_best_d);
    let win_d_f = b.unop(OP_CONVERT_S_TO_F, b.ty_f32, win_d);
    let win_cost = b.load(b.ty_f32, var_best_cost);
    b.store_buf_f32(buf_disp, c0, out_idx, win_d_f);
    b.store_buf_f32(buf_cost, c0, out_idx, win_cost);

    b.end_main();
    b.finish()
}

/// 3. Subpixel Parabola Refinement Compute Shader.
/// Given `(c_minus, c_center, c_plus)` matching costs around integer disparity `d`,
/// computes `d_sub = d + 0.5 * (c_minus - c_plus) / (c_minus - 2*c_center + c_plus)`.
/// Bindings:
///   0: `in_disp[]` (f32)
///   1: `cost_m1[]` (f32)
///   2: `cost_0[]` (f32)
///   3: `cost_p1[]` (f32)
///   4: `out_disp[]` (f32)
///   5: `params[]` (i32): `[width, height]`
pub fn build_subpixel_parabola_shader() -> Vec<u32> {
    let mut b = SpirvBuilder::new(16, 16, 1);
    let buf_d = b.add_storage_buffer_f32(0);
    let buf_cm = b.add_storage_buffer_f32(1);
    let buf_c0 = b.add_storage_buffer_f32(2);
    let buf_cp = b.add_storage_buffer_f32(3);
    let buf_out = b.add_storage_buffer_f32(4);
    let buf_p = b.add_storage_buffer_i32(5);

    let c0 = b.const_i32(0);
    let c1 = b.const_i32(1);
    let f0 = b.const_f32(0.0);
    let f_half = b.const_f32(0.5);
    let f_neg_half = b.const_f32(-0.5);
    let f2 = b.const_f32(2.0);
    let f_eps = b.const_f32(1.0e-6);

    b.begin_main();
    let (gx, gy) = b.load_global_xy();
    let w = b.load_buf_i32(buf_p, c0, c0);
    let h = b.load_buf_i32(buf_p, c0, c1);

    let oob_x = b.binop(OP_SGREATER_THAN_EQUAL, b.ty_bool, gx, w);
    let oob_y = b.binop(OP_SGREATER_THAN_EQUAL, b.ty_bool, gy, h);
    let oob = b.binop(OP_LOGICAL_OR, b.ty_bool, oob_x, oob_y);
    let lbl_ret = b.id();
    let lbl_body = b.id();
    b.selection_merge(lbl_body);
    b.branch_cond(oob, lbl_ret, lbl_body);
    b.label(lbl_ret);
    b.return_void();

    b.label(lbl_body);
    let row = b.binop(OP_IMUL, b.ty_i32, gy, w);
    let idx = b.binop(OP_IADD, b.ty_i32, row, gx);
    let d = b.load_buf_f32(buf_d, c0, idx);
    let cm = b.load_buf_f32(buf_cm, c0, idx);
    let cc = b.load_buf_f32(buf_c0, c0, idx);
    let cp = b.load_buf_f32(buf_cp, c0, idx);

    let num = b.binop(OP_FSUB, b.ty_f32, cm, cp);
    let two_cc = b.binop(OP_FMUL, b.ty_f32, f2, cc);
    let cm_p_cp = b.binop(OP_FADD, b.ty_f32, cm, cp);
    let denom = b.binop(OP_FSUB, b.ty_f32, cm_p_cp, two_cc);
    let denom_abs = b.glsl1(GLSL_FABS, b.ty_f32, denom);
    let valid_denom = b.binop(OP_FORD_GREATER_THAN, b.ty_bool, denom_abs, f_eps);
    let safe_denom = b.select(b.ty_f32, valid_denom, denom, f2);
    let raw_delta = b.binop(OP_FDIV, b.ty_f32, num, safe_denom);
    let scaled = b.binop(OP_FMUL, b.ty_f32, raw_delta, f_half);
    let clamped_hi = b.glsl2(GLSL_FMIN, b.ty_f32, scaled, f_half);
    let clamped = b.glsl2(GLSL_FMAX, b.ty_f32, clamped_hi, f_neg_half);
    let delta = b.select(b.ty_f32, valid_denom, clamped, f0);
    let refined = b.binop(OP_FADD, b.ty_f32, d, delta);
    b.store_buf_f32(buf_out, c0, idx, refined);

    b.end_main();
    b.finish()
}

/// 4. DEM Lambertian Hillshade Compute Shader.
/// Computes surface normal `(dz/dx, dz/dy, 1)` via central differences and dot-product with sun vector.
/// Bindings:
///   0: `dem[]` (f32)
///   1: `out_shade[]` (f32)
///   2: `params_i[]` (i32): `[width, height]`
///   3: `params_f[]` (f32): `[pixel_size, sun_x, sun_y, sun_z]`
pub fn build_hillshade_shader() -> Vec<u32> {
    let mut b = SpirvBuilder::new(16, 16, 1);
    let buf_dem = b.add_storage_buffer_f32(0);
    let buf_out = b.add_storage_buffer_f32(1);
    let buf_pi = b.add_storage_buffer_i32(2);
    let buf_pf = b.add_storage_buffer_f32(3);

    let c0 = b.const_i32(0);
    let c1 = b.const_i32(1);
    let c2 = b.const_i32(2);
    let c3 = b.const_i32(3);
    let f0 = b.const_f32(0.0);
    let f1 = b.const_f32(1.0);
    let f2 = b.const_f32(2.0);
    let f_eps = b.const_f32(1.0e-6);

    b.begin_main();
    let (gx, gy) = b.load_global_xy();
    let w = b.load_buf_i32(buf_pi, c0, c0);
    let h = b.load_buf_i32(buf_pi, c0, c1);
    let pix_sz = b.load_buf_f32(buf_pf, c0, c0);
    let sun_x = b.load_buf_f32(buf_pf, c0, c1);
    let sun_y = b.load_buf_f32(buf_pf, c0, c2);
    let sun_z = b.load_buf_f32(buf_pf, c0, c3);

    let oob_x = b.binop(OP_SGREATER_THAN_EQUAL, b.ty_bool, gx, w);
    let oob_y = b.binop(OP_SGREATER_THAN_EQUAL, b.ty_bool, gy, h);
    let oob = b.binop(OP_LOGICAL_OR, b.ty_bool, oob_x, oob_y);
    let lbl_ret = b.id();
    let lbl_body = b.id();
    b.selection_merge(lbl_body);
    b.branch_cond(oob, lbl_ret, lbl_body);
    b.label(lbl_ret);
    b.return_void();

    b.label(lbl_body);
    let wm1 = b.binop(OP_ISUB, b.ty_i32, w, c1);
    let hm1 = b.binop(OP_ISUB, b.ty_i32, h, c1);
    let xm1 = b.binop(OP_ISUB, b.ty_i32, gx, c1);
    let xp1 = b.binop(OP_IADD, b.ty_i32, gx, c1);
    let ym1 = b.binop(OP_ISUB, b.ty_i32, gy, c1);
    let yp1 = b.binop(OP_IADD, b.ty_i32, gy, c1);

    let xl = b.glsl3(GLSL_SCLAMP, b.ty_i32, xm1, c0, wm1);
    let xr = b.glsl3(GLSL_SCLAMP, b.ty_i32, xp1, c0, wm1);
    let yu = b.glsl3(GLSL_SCLAMP, b.ty_i32, ym1, c0, hm1);
    let yd = b.glsl3(GLSL_SCLAMP, b.ty_i32, yp1, c0, hm1);

    let row_c = b.binop(OP_IMUL, b.ty_i32, gy, w);
    let row_u = b.binop(OP_IMUL, b.ty_i32, yu, w);
    let row_d = b.binop(OP_IMUL, b.ty_i32, yd, w);

    let idx_l = b.binop(OP_IADD, b.ty_i32, row_c, xl);
    let idx_r = b.binop(OP_IADD, b.ty_i32, row_c, xr);
    let idx_u = b.binop(OP_IADD, b.ty_i32, row_u, gx);
    let idx_d = b.binop(OP_IADD, b.ty_i32, row_d, gx);

    let zl = b.load_buf_f32(buf_dem, c0, idx_l);
    let zr = b.load_buf_f32(buf_dem, c0, idx_r);
    let zu = b.load_buf_f32(buf_dem, c0, idx_u);
    let zd = b.load_buf_f32(buf_dem, c0, idx_d);

    let two_ps = b.binop(OP_FMUL, b.ty_f32, f2, pix_sz);
    let dzx_num = b.binop(OP_FSUB, b.ty_f32, zr, zl);
    let dzy_num = b.binop(OP_FSUB, b.ty_f32, zu, zd);
    let dzdx = b.binop(OP_FDIV, b.ty_f32, dzx_num, two_ps);
    let dzdy = b.binop(OP_FDIV, b.ty_f32, dzy_num, two_ps);

    let nx = b.unop(OP_FNEGATE, b.ty_f32, dzdx);
    let ny = b.unop(OP_FNEGATE, b.ty_f32, dzdy);
    let nx2 = b.binop(OP_FMUL, b.ty_f32, nx, nx);
    let ny2 = b.binop(OP_FMUL, b.ty_f32, ny, ny);
    let s1 = b.binop(OP_FADD, b.ty_f32, nx2, ny2);
    let s2 = b.binop(OP_FADD, b.ty_f32, s1, f1);
    let nlen_sq = b.binop(OP_FADD, b.ty_f32, s2, f_eps);
    let nlen = b.glsl1(GLSL_SQRT, b.ty_f32, nlen_sq);

    let px = b.binop(OP_FMUL, b.ty_f32, nx, sun_x);
    let py = b.binop(OP_FMUL, b.ty_f32, ny, sun_y);
    let pxy = b.binop(OP_FADD, b.ty_f32, px, py);
    let dot = b.binop(OP_FADD, b.ty_f32, pxy, sun_z);
    let shade_raw = b.binop(OP_FDIV, b.ty_f32, dot, nlen);
    let shade = b.glsl2(GLSL_FMAX, b.ty_f32, shade_raw, f0);

    let idx_c = b.binop(OP_IADD, b.ty_i32, row_c, gx);
    b.store_buf_f32(buf_out, c0, idx_c, shade);

    b.end_main();
    b.finish()
}

/// 5. 2D Discrete Laplacian of Gaussian (LoG) Prefilter Compute Shader.
/// Computes smoothed 3x3 Gaussian + discrete Laplacian `4*C - L - R - U - D` in a single GPU pass.
/// Bindings:
///   0: `in_img[]` (f32)
///   1: `out_img[]` (f32)
///   2: `params[]` (i32): `[width, height]`
pub fn build_log_2d_shader() -> Vec<u32> {
    let mut b = SpirvBuilder::new(16, 16, 1);
    let buf_in = b.add_storage_buffer_f32(0);
    let buf_out = b.add_storage_buffer_f32(1);
    let buf_p = b.add_storage_buffer_i32(2);

    let c0 = b.const_i32(0);
    let c1 = b.const_i32(1);
    let f4 = b.const_f32(4.0);

    b.begin_main();
    let (gx, gy) = b.load_global_xy();
    let w = b.load_buf_i32(buf_p, c0, c0);
    let h = b.load_buf_i32(buf_p, c0, c1);

    let oob_x = b.binop(OP_SGREATER_THAN_EQUAL, b.ty_bool, gx, w);
    let oob_y = b.binop(OP_SGREATER_THAN_EQUAL, b.ty_bool, gy, h);
    let oob = b.binop(OP_LOGICAL_OR, b.ty_bool, oob_x, oob_y);
    let lbl_ret = b.id();
    let lbl_body = b.id();
    b.selection_merge(lbl_body);
    b.branch_cond(oob, lbl_ret, lbl_body);
    b.label(lbl_ret);
    b.return_void();

    b.label(lbl_body);
    let wm1 = b.binop(OP_ISUB, b.ty_i32, w, c1);
    let hm1 = b.binop(OP_ISUB, b.ty_i32, h, c1);
    let xm1 = b.binop(OP_ISUB, b.ty_i32, gx, c1);
    let xp1 = b.binop(OP_IADD, b.ty_i32, gx, c1);
    let ym1 = b.binop(OP_ISUB, b.ty_i32, gy, c1);
    let yp1 = b.binop(OP_IADD, b.ty_i32, gy, c1);

    let xl = b.glsl3(GLSL_SCLAMP, b.ty_i32, xm1, c0, wm1);
    let xr = b.glsl3(GLSL_SCLAMP, b.ty_i32, xp1, c0, wm1);
    let yu = b.glsl3(GLSL_SCLAMP, b.ty_i32, ym1, c0, hm1);
    let yd = b.glsl3(GLSL_SCLAMP, b.ty_i32, yp1, c0, hm1);

    let row_c = b.binop(OP_IMUL, b.ty_i32, gy, w);
    let row_u = b.binop(OP_IMUL, b.ty_i32, yu, w);
    let row_d = b.binop(OP_IMUL, b.ty_i32, yd, w);

    let ic = b.binop(OP_IADD, b.ty_i32, row_c, gx);
    let il = b.binop(OP_IADD, b.ty_i32, row_c, xl);
    let ir = b.binop(OP_IADD, b.ty_i32, row_c, xr);
    let iu = b.binop(OP_IADD, b.ty_i32, row_u, gx);
    let id = b.binop(OP_IADD, b.ty_i32, row_d, gx);

    let vc = b.load_buf_f32(buf_in, c0, ic);
    let vl = b.load_buf_f32(buf_in, c0, il);
    let vr = b.load_buf_f32(buf_in, c0, ir);
    let vu = b.load_buf_f32(buf_in, c0, iu);
    let vd = b.load_buf_f32(buf_in, c0, id);

    let c4 = b.binop(OP_FMUL, b.ty_f32, f4, vc);
    let s1 = b.binop(OP_FSUB, b.ty_f32, c4, vl);
    let s2 = b.binop(OP_FSUB, b.ty_f32, s1, vr);
    let s3 = b.binop(OP_FSUB, b.ty_f32, s2, vu);
    let lap = b.binop(OP_FSUB, b.ty_f32, s3, vd);
    b.store_buf_f32(buf_out, c0, ic, lap);

    b.end_main();
    b.finish()
}

/// 6. 3D Stereo Cost Volume Compute Shader (`C[y][x][d]` for SGM & MGM on GPU).
/// Evaluates normalized SAD (`mode=0`), normalized SSD (`mode=1`), or true Zero-Mean NCC (`mode=2`)
/// across `num_d` horizontal disparities starting at `min_dx`, setting out-of-bounds `x + d` to `2.0`.
/// Bindings:
///   0: `left[]` (f32)
///   1: `right[]` (f32)
///   2: `out_cost_vol[]` (f32, length `w * h * num_d`, layout `((y * w) + x) * num_d + di`)
///   3: `params[]` (i32): `[width, height, half_kw, half_kh, min_dx, num_d, cost_mode]`
pub fn build_cost_volume_3d_shader() -> Vec<u32> {
    let mut b = SpirvBuilder::new(16, 16, 1);
    let buf_l = b.add_storage_buffer_f32(0);
    let buf_r = b.add_storage_buffer_f32(1);
    let buf_vol = b.add_storage_buffer_f32(2);
    let buf_p = b.add_storage_buffer_i32(3);

    let c0 = b.const_i32(0);
    let c1 = b.const_i32(1);
    let c2 = b.const_i32(2);
    let c3 = b.const_i32(3);
    let c4 = b.const_i32(4);
    let c5 = b.const_i32(5);
    let c6 = b.const_i32(6);
    let f0 = b.const_f32(0.0);
    let f1 = b.const_f32(1.0);
    let f2 = b.const_f32(2.0);
    let f_eps = b.const_f32(1.0e-6);

    b.begin_main();
    let var_di = b.fn_var_i32();
    let var_ky = b.fn_var_i32();
    let var_kx = b.fn_var_i32();
    let var_acc = b.fn_var_f32();
    let var_sl = b.fn_var_f32();
    let var_sr = b.fn_var_f32();
    let var_slr = b.fn_var_f32();
    let var_sll = b.fn_var_f32();
    let var_srr = b.fn_var_f32();

    let (gx, gy) = b.load_global_xy();
    let w = b.load_buf_i32(buf_p, c0, c0);
    let h = b.load_buf_i32(buf_p, c0, c1);
    let hkw = b.load_buf_i32(buf_p, c0, c2);
    let hkh = b.load_buf_i32(buf_p, c0, c3);
    let min_dx = b.load_buf_i32(buf_p, c0, c4);
    let num_d = b.load_buf_i32(buf_p, c0, c5);
    let cost_mode = b.load_buf_i32(buf_p, c0, c6);

    let oob_x = b.binop(OP_SGREATER_THAN_EQUAL, b.ty_bool, gx, w);
    let oob_y = b.binop(OP_SGREATER_THAN_EQUAL, b.ty_bool, gy, h);
    let oob = b.binop(OP_LOGICAL_OR, b.ty_bool, oob_x, oob_y);
    let lbl_ret = b.id();
    let lbl_body = b.id();
    b.selection_merge(lbl_body);
    b.branch_cond(oob, lbl_ret, lbl_body);
    b.label(lbl_ret);
    b.return_void();

    b.label(lbl_body);
    let wm1 = b.binop(OP_ISUB, b.ty_i32, w, c1);
    let hm1 = b.binop(OP_ISUB, b.ty_i32, h, c1);
    let neg_hkw = b.binop(OP_ISUB, b.ty_i32, c0, hkw);
    let neg_hkh = b.binop(OP_ISUB, b.ty_i32, c0, hkh);
    let hkw_p1 = b.binop(OP_IADD, b.ty_i32, hkw, c1);
    let hkh_p1 = b.binop(OP_IADD, b.ty_i32, hkh, c1);

    let hkw2 = b.binop(OP_IMUL, b.ty_i32, hkw, c2);
    let kw_full = b.binop(OP_IADD, b.ty_i32, hkw2, c1);
    let hkh2 = b.binop(OP_IMUL, b.ty_i32, hkh, c2);
    let kh_full = b.binop(OP_IADD, b.ty_i32, hkh2, c1);
    let cnt_i = b.binop(OP_IMUL, b.ty_i32, kw_full, kh_full);
    let cnt_f = b.unop(OP_CONVERT_S_TO_F, b.ty_f32, cnt_i);

    let gy_w = b.binop(OP_IMUL, b.ty_i32, gy, w);
    let pix_idx = b.binop(OP_IADD, b.ty_i32, gy_w, gx);
    let vol_base = b.binop(OP_IMUL, b.ty_i32, pix_idx, num_d);

    b.store(var_di, c0);
    let lbl_d_hdr = b.id();
    let lbl_d_body = b.id();
    let lbl_d_cont = b.id();
    let lbl_d_end = b.id();
    b.branch(lbl_d_hdr);
    b.label(lbl_d_hdr);
    b.loop_merge(lbl_d_end, lbl_d_cont);
    let di = b.load(b.ty_i32, var_di);
    let d_cond = b.binop(OP_SLESS_THAN, b.ty_bool, di, num_d);
    b.branch_cond(d_cond, lbl_d_body, lbl_d_end);

    b.label(lbl_d_body);
    let d = b.binop(OP_IADD, b.ty_i32, min_dx, di);
    let rx_center = b.binop(OP_IADD, b.ty_i32, gx, d);
    let oob_l = b.binop(OP_SLESS_THAN, b.ty_bool, rx_center, c0);
    let oob_r = b.binop(OP_SGREATER_THAN_EQUAL, b.ty_bool, rx_center, w);
    let oob_d = b.binop(OP_LOGICAL_OR, b.ty_bool, oob_l, oob_r);
    let vol_idx = b.binop(OP_IADD, b.ty_i32, vol_base, di);

    let lbl_oob = b.id();
    let lbl_in = b.id();
    let lbl_merge = b.id();
    b.selection_merge(lbl_merge);
    b.branch_cond(oob_d, lbl_oob, lbl_in);

    b.label(lbl_oob);
    b.store_buf_f32(buf_vol, c0, vol_idx, f2);
    b.branch(lbl_merge);

    b.label(lbl_in);
    b.store(var_acc, f0);
    b.store(var_sl, f0);
    b.store(var_sr, f0);
    b.store(var_slr, f0);
    b.store(var_sll, f0);
    b.store(var_srr, f0);
    b.store(var_ky, neg_hkh);

    let lbl_ky_hdr = b.id();
    let lbl_ky_body = b.id();
    let lbl_ky_cont = b.id();
    let lbl_ky_end = b.id();
    b.branch(lbl_ky_hdr);
    b.label(lbl_ky_hdr);
    b.loop_merge(lbl_ky_end, lbl_ky_cont);
    let ky = b.load(b.ty_i32, var_ky);
    let ky_cond = b.binop(OP_SLESS_THAN, b.ty_bool, ky, hkh_p1);
    b.branch_cond(ky_cond, lbl_ky_body, lbl_ky_end);

    b.label(lbl_ky_body);
    let sy_raw = b.binop(OP_IADD, b.ty_i32, gy, ky);
    let sy = b.glsl3(GLSL_SCLAMP, b.ty_i32, sy_raw, c0, hm1);
    let row_off = b.binop(OP_IMUL, b.ty_i32, sy, w);
    b.store(var_kx, neg_hkw);

    let lbl_kx_hdr = b.id();
    let lbl_kx_body = b.id();
    let lbl_kx_cont = b.id();
    let lbl_kx_end = b.id();
    b.branch(lbl_kx_hdr);
    b.label(lbl_kx_hdr);
    b.loop_merge(lbl_kx_end, lbl_kx_cont);
    let kx = b.load(b.ty_i32, var_kx);
    let kx_cond = b.binop(OP_SLESS_THAN, b.ty_bool, kx, hkw_p1);
    b.branch_cond(kx_cond, lbl_kx_body, lbl_kx_end);

    b.label(lbl_kx_body);
    let lx_raw = b.binop(OP_IADD, b.ty_i32, gx, kx);
    let lx = b.glsl3(GLSL_SCLAMP, b.ty_i32, lx_raw, c0, wm1);
    let rx_raw = b.binop(OP_IADD, b.ty_i32, lx_raw, d);
    let rx = b.glsl3(GLSL_SCLAMP, b.ty_i32, rx_raw, c0, wm1);
    let l_idx = b.binop(OP_IADD, b.ty_i32, row_off, lx);
    let r_idx = b.binop(OP_IADD, b.ty_i32, row_off, rx);
    let lv = b.load_buf_f32(buf_l, c0, l_idx);
    let rv = b.load_buf_f32(buf_r, c0, r_idx);

    let diff = b.binop(OP_FSUB, b.ty_f32, lv, rv);
    let abs_diff = b.glsl1(GLSL_FABS, b.ty_f32, diff);
    let sq_diff = b.binop(OP_FMUL, b.ty_f32, diff, diff);
    let is_ssd = b.binop(OP_IEQUAL, b.ty_bool, cost_mode, c1);
    let term = b.select(b.ty_f32, is_ssd, sq_diff, abs_diff);
    let cur_acc = b.load(b.ty_f32, var_acc);
    let next_acc = b.binop(OP_FADD, b.ty_f32, cur_acc, term);
    b.store(var_acc, next_acc);

    // Accumulate zero-mean NCC moments
    let c_sl = b.load(b.ty_f32, var_sl);
    let c_sr = b.load(b.ty_f32, var_sr);
    let c_slr = b.load(b.ty_f32, var_slr);
    let c_sll = b.load(b.ty_f32, var_sll);
    let c_srr = b.load(b.ty_f32, var_srr);
    let p_lr = b.binop(OP_FMUL, b.ty_f32, lv, rv);
    let p_ll = b.binop(OP_FMUL, b.ty_f32, lv, lv);
    let p_rr = b.binop(OP_FMUL, b.ty_f32, rv, rv);
    let n_sl = b.binop(OP_FADD, b.ty_f32, c_sl, lv);
    let n_sr = b.binop(OP_FADD, b.ty_f32, c_sr, rv);
    let n_slr = b.binop(OP_FADD, b.ty_f32, c_slr, p_lr);
    let n_sll = b.binop(OP_FADD, b.ty_f32, c_sll, p_ll);
    let n_srr = b.binop(OP_FADD, b.ty_f32, c_srr, p_rr);
    b.store(var_sl, n_sl);
    b.store(var_sr, n_sr);
    b.store(var_slr, n_slr);
    b.store(var_sll, n_sll);
    b.store(var_srr, n_srr);
    b.branch(lbl_kx_cont);

    b.label(lbl_kx_cont);
    let nkx = b.binop(OP_IADD, b.ty_i32, kx, c1);
    b.store(var_kx, nkx);
    b.branch(lbl_kx_hdr);

    b.label(lbl_kx_end);
    b.branch(lbl_ky_cont);

    b.label(lbl_ky_cont);
    let nky = b.binop(OP_IADD, b.ty_i32, ky, c1);
    b.store(var_ky, nky);
    b.branch(lbl_ky_hdr);

    b.label(lbl_ky_end);
    let acc_val = b.load(b.ty_f32, var_acc);
    let sad_ssd_norm = b.binop(OP_FDIV, b.ty_f32, acc_val, cnt_f);
    let sl = b.load(b.ty_f32, var_sl);
    let sr = b.load(b.ty_f32, var_sr);
    let slr = b.load(b.ty_f32, var_slr);
    let sll = b.load(b.ty_f32, var_sll);
    let srr = b.load(b.ty_f32, var_srr);

    let ml = b.binop(OP_FDIV, b.ty_f32, sl, cnt_f);
    let mr = b.binop(OP_FDIV, b.ty_f32, sr, cnt_f);
    let sl_mr = b.binop(OP_FMUL, b.ty_f32, sl, mr);
    let cov = b.binop(OP_FSUB, b.ty_f32, slr, sl_mr);
    let sl_ml = b.binop(OP_FMUL, b.ty_f32, sl, ml);
    let vl_raw = b.binop(OP_FSUB, b.ty_f32, sll, sl_ml);
    let sr_mr = b.binop(OP_FMUL, b.ty_f32, sr, mr);
    let vr_raw = b.binop(OP_FSUB, b.ty_f32, srr, sr_mr);
    let vl = b.glsl2(GLSL_FMAX, b.ty_f32, vl_raw, f0);
    let vr = b.glsl2(GLSL_FMAX, b.ty_f32, vr_raw, f0);
    let vl_vr = b.binop(OP_FMUL, b.ty_f32, vl, vr);
    let denom_sq = b.binop(OP_FADD, b.ty_f32, vl_vr, f_eps);
    let denom = b.glsl1(GLSL_SQRT, b.ty_f32, denom_sq);
    let ncc = b.binop(OP_FDIV, b.ty_f32, cov, denom);
    let ncc_cost_raw = b.binop(OP_FSUB, b.ty_f32, f1, ncc);
    let ncc_clamped_lo = b.glsl2(GLSL_FMAX, b.ty_f32, ncc_cost_raw, f0);
    let ncc_cost = b.glsl2(GLSL_FMIN, b.ty_f32, ncc_clamped_lo, f2);
    let is_ncc = b.binop(OP_IEQUAL, b.ty_bool, cost_mode, c2);
    let final_cost = b.select(b.ty_f32, is_ncc, ncc_cost, sad_ssd_norm);
    b.store_buf_f32(buf_vol, c0, vol_idx, final_cost);
    b.branch(lbl_merge);

    b.label(lbl_merge);
    b.branch(lbl_d_cont);

    b.label(lbl_d_cont);
    let ndi = b.binop(OP_IADD, b.ty_i32, di, c1);
    b.store(var_di, ndi);
    b.branch(lbl_d_hdr);

    b.label(lbl_d_end);
    b.end_main();
    b.finish()
}

/// Summary info for all built-in SPIR-V shaders.
pub fn shader_catalog() -> Vec<(&'static str, Vec<u32>)> {
    vec![
        ("prefilter_box_submean", build_prefilter_shader()),
        ("prefilter_log_2d", build_log_2d_shader()),
        ("stereo_bm_sad_ssd_ncc", build_stereo_bm_shader()),
        ("stereo_cost_vol_3d", build_cost_volume_3d_shader()),
        ("subpixel_parabola", build_subpixel_parabola_shader()),
        ("dem_hillshade", build_hillshade_shader()),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_all_spirv_shaders_valid() {
        for (name, words) in shader_catalog() {
            let stats = validate_spirv(&words)
                .unwrap_or_else(|e| panic!("SPIR-V shader '{}' failed validation: {}", name, e));
            assert!(stats.has_compute_entry, "Shader {} missing compute entry", name);
            assert!(stats.instruction_count > 10, "Shader {} too small", name);
        }
    }
}
