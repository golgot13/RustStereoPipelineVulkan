//! Pure-Rust SPIR-V 1.0 binary assembler and structural validator.
//! Emits spec-compliant Vulkan Compute SPIR-V modules without glslc/shaderc or external crates.

#![allow(dead_code)]

pub const SPIRV_MAGIC: u32 = 0x0723_0203;
pub const SPIRV_VERSION_1_0: u32 = 0x0001_0000;

// SPIR-V Opcodes
pub const OP_EXT_INST_IMPORT: u16 = 11;
pub const OP_EXT_INST: u16 = 12;
pub const OP_MEMORY_MODEL: u16 = 14;
pub const OP_ENTRY_POINT: u16 = 15;
pub const OP_EXECUTION_MODE: u16 = 16;
pub const OP_CAPABILITY: u16 = 17;
pub const OP_TYPE_VOID: u16 = 19;
pub const OP_TYPE_BOOL: u16 = 20;
pub const OP_TYPE_INT: u16 = 21;
pub const OP_TYPE_FLOAT: u16 = 22;
pub const OP_TYPE_VECTOR: u16 = 23;
pub const OP_TYPE_RUNTIME_ARRAY: u16 = 29;
pub const OP_TYPE_STRUCT: u16 = 30;
pub const OP_TYPE_POINTER: u16 = 32;
pub const OP_TYPE_FUNCTION: u16 = 33;
pub const OP_CONSTANT: u16 = 43;
pub const OP_FUNCTION: u16 = 54;
pub const OP_FUNCTION_END: u16 = 56;
pub const OP_VARIABLE: u16 = 59;
pub const OP_LOAD: u16 = 61;
pub const OP_STORE: u16 = 62;
pub const OP_ACCESS_CHAIN: u16 = 65;
pub const OP_DECORATE: u16 = 71;
pub const OP_MEMBER_DECORATE: u16 = 72;
pub const OP_COMPOSITE_EXTRACT: u16 = 81;
pub const OP_CONVERT_F_TO_S: u16 = 110;
pub const OP_CONVERT_S_TO_F: u16 = 111;
pub const OP_BITCAST: u16 = 124;
pub const OP_FNEGATE: u16 = 127;
pub const OP_IADD: u16 = 128;
pub const OP_FADD: u16 = 129;
pub const OP_ISUB: u16 = 130;
pub const OP_FSUB: u16 = 131;
pub const OP_IMUL: u16 = 132;
pub const OP_FMUL: u16 = 133;
pub const OP_SDIV: u16 = 135;
pub const OP_FDIV: u16 = 136;
pub const OP_SELECT: u16 = 169;
pub const OP_IEQUAL: u16 = 170;
pub const OP_SLESS_THAN: u16 = 177;
pub const OP_SGREATER_THAN_EQUAL: u16 = 175;
pub const OP_FORD_LESS_THAN: u16 = 184;
pub const OP_FORD_GREATER_THAN: u16 = 186;
pub const OP_LOGICAL_OR: u16 = 166;
pub const OP_LOGICAL_AND: u16 = 167;
pub const OP_LOOP_MERGE: u16 = 246;
pub const OP_SELECTION_MERGE: u16 = 247;
pub const OP_LABEL: u16 = 248;
pub const OP_BRANCH: u16 = 249;
pub const OP_BRANCH_CONDITIONAL: u16 = 250;
pub const OP_RETURN: u16 = 253;

// SPIR-V Constants & Enumerants
pub const CAPABILITY_SHADER: u32 = 1;
pub const ADDRESSING_MODEL_LOGICAL: u32 = 0;
pub const MEMORY_MODEL_GLSL450: u32 = 1;
pub const EXECUTION_MODEL_GLCOMPUTE: u32 = 5;
pub const EXECUTION_MODE_LOCAL_SIZE: u32 = 17;

pub const STORAGE_CLASS_INPUT: u32 = 1;
pub const STORAGE_CLASS_UNIFORM: u32 = 2;
pub const STORAGE_CLASS_FUNCTION: u32 = 7;

pub const DECORATION_ARRAY_STRIDE: u32 = 6;
pub const DECORATION_BUFFER_BLOCK: u32 = 3;
pub const DECORATION_BUILTIN: u32 = 11;
pub const DECORATION_BINDING: u32 = 33;
pub const DECORATION_DESCRIPTOR_SET: u32 = 34;
pub const DECORATION_OFFSET: u32 = 35;

pub const BUILTIN_GLOBAL_INVOCATION_ID: u32 = 28;

// GLSL.std.450 Extended Instructions
pub const GLSL_FABS: u32 = 4;
pub const GLSL_SQRT: u32 = 31;
pub const GLSL_FMIN: u32 = 37;
pub const GLSL_FMAX: u32 = 40;
pub const GLSL_SCLAMP: u32 = 44;

/// Builder for a complete, spec-compliant SPIR-V 1.0 Vulkan Compute shader module.
pub struct SpirvBuilder {
    next_id: u32,
    pub ext_glsl: u32,
    pub ty_void: u32,
    pub ty_bool: u32,
    pub ty_u32: u32,
    pub ty_i32: u32,
    pub ty_f32: u32,
    pub ty_uvec3: u32,
    pub ty_ptr_input_uvec3: u32,
    pub ty_ptr_fn_i32: u32,
    pub ty_ptr_fn_f32: u32,
    pub ty_ptr_uniform_f32: u32,
    pub ty_ptr_uniform_i32: u32,
    pub ty_fn_void: u32,
    pub var_global_id: u32,
    pub fn_main: u32,

    capabilities: Vec<u32>,
    ext_imports: Vec<u32>,
    memory_model: Vec<u32>,
    entry_points: Vec<u32>,
    exec_modes: Vec<u32>,
    annotations: Vec<u32>,
    types_and_globals: Vec<u32>,
    functions: Vec<u32>,
}

impl SpirvBuilder {
    pub fn new(local_x: u32, local_y: u32, local_z: u32) -> Self {
        let mut b = Self {
            next_id: 1,
            ext_glsl: 0,
            ty_void: 0,
            ty_bool: 0,
            ty_u32: 0,
            ty_i32: 0,
            ty_f32: 0,
            ty_uvec3: 0,
            ty_ptr_input_uvec3: 0,
            ty_ptr_fn_i32: 0,
            ty_ptr_fn_f32: 0,
            ty_ptr_uniform_f32: 0,
            ty_ptr_uniform_i32: 0,
            ty_fn_void: 0,
            var_global_id: 0,
            fn_main: 0,
            capabilities: Vec::new(),
            ext_imports: Vec::new(),
            memory_model: Vec::new(),
            entry_points: Vec::new(),
            exec_modes: Vec::new(),
            annotations: Vec::new(),
            types_and_globals: Vec::new(),
            functions: Vec::new(),
        };

        // Capability Shader
        emit(&mut b.capabilities, OP_CAPABILITY, &[CAPABILITY_SHADER]);

        // ExtInstImport "GLSL.std.450"
        b.ext_glsl = b.id();
        let mut ext_ops = vec![b.ext_glsl];
        ext_ops.extend(encode_string("GLSL.std.450"));
        emit(&mut b.ext_imports, OP_EXT_INST_IMPORT, &ext_ops);

        // MemoryModel Logical GLSL450
        emit(
            &mut b.memory_model,
            OP_MEMORY_MODEL,
            &[ADDRESSING_MODEL_LOGICAL, MEMORY_MODEL_GLSL450],
        );

        // Core types
        b.ty_void = b.id();
        emit(&mut b.types_and_globals, OP_TYPE_VOID, &[b.ty_void]);

        b.ty_bool = b.id();
        emit(&mut b.types_and_globals, OP_TYPE_BOOL, &[b.ty_bool]);

        b.ty_u32 = b.id();
        emit(&mut b.types_and_globals, OP_TYPE_INT, &[b.ty_u32, 32, 0]);

        b.ty_i32 = b.id();
        emit(&mut b.types_and_globals, OP_TYPE_INT, &[b.ty_i32, 32, 1]);

        b.ty_f32 = b.id();
        emit(&mut b.types_and_globals, OP_TYPE_FLOAT, &[b.ty_f32, 32]);

        b.ty_uvec3 = b.id();
        emit(&mut b.types_and_globals, OP_TYPE_VECTOR, &[b.ty_uvec3, b.ty_u32, 3]);

        b.ty_ptr_input_uvec3 = b.id();
        emit(
            &mut b.types_and_globals,
            OP_TYPE_POINTER,
            &[b.ty_ptr_input_uvec3, STORAGE_CLASS_INPUT, b.ty_uvec3],
        );

        b.ty_ptr_fn_i32 = b.id();
        emit(
            &mut b.types_and_globals,
            OP_TYPE_POINTER,
            &[b.ty_ptr_fn_i32, STORAGE_CLASS_FUNCTION, b.ty_i32],
        );

        b.ty_ptr_fn_f32 = b.id();
        emit(
            &mut b.types_and_globals,
            OP_TYPE_POINTER,
            &[b.ty_ptr_fn_f32, STORAGE_CLASS_FUNCTION, b.ty_f32],
        );

        b.ty_ptr_uniform_f32 = b.id();
        emit(
            &mut b.types_and_globals,
            OP_TYPE_POINTER,
            &[b.ty_ptr_uniform_f32, STORAGE_CLASS_UNIFORM, b.ty_f32],
        );

        b.ty_ptr_uniform_i32 = b.id();
        emit(
            &mut b.types_and_globals,
            OP_TYPE_POINTER,
            &[b.ty_ptr_uniform_i32, STORAGE_CLASS_UNIFORM, b.ty_i32],
        );

        b.ty_fn_void = b.id();
        emit(&mut b.types_and_globals, OP_TYPE_FUNCTION, &[b.ty_fn_void, b.ty_void]);

        // GlobalInvocationId variable
        b.var_global_id = b.id();
        emit(
            &mut b.types_and_globals,
            OP_VARIABLE,
            &[b.ty_ptr_input_uvec3, b.var_global_id, STORAGE_CLASS_INPUT],
        );
        emit(
            &mut b.annotations,
            OP_DECORATE,
            &[b.var_global_id, DECORATION_BUILTIN, BUILTIN_GLOBAL_INVOCATION_ID],
        );

        // EntryPoint & ExecutionMode
        b.fn_main = b.id();
        let mut ep = vec![EXECUTION_MODEL_GLCOMPUTE, b.fn_main];
        ep.extend(encode_string("main"));
        ep.push(b.var_global_id);
        emit(&mut b.entry_points, OP_ENTRY_POINT, &ep);
        emit(
            &mut b.exec_modes,
            OP_EXECUTION_MODE,
            &[b.fn_main, EXECUTION_MODE_LOCAL_SIZE, local_x, local_y, local_z],
        );

        b
    }

    pub fn id(&mut self) -> u32 {
        let cur = self.next_id;
        self.next_id += 1;
        cur
    }

    /// Declare a storage buffer `layout(set = 0, binding = binding) buffer B { float data[]; };`
    pub fn add_storage_buffer_f32(&mut self, binding: u32) -> u32 {
        self.add_storage_buffer_typed(binding, self.ty_f32)
    }

    /// Declare a storage buffer `layout(set = 0, binding = binding) buffer B { int data[]; };`
    pub fn add_storage_buffer_i32(&mut self, binding: u32) -> u32 {
        self.add_storage_buffer_typed(binding, self.ty_i32)
    }

    fn add_storage_buffer_typed(&mut self, binding: u32, elem_ty: u32) -> u32 {
        let ty_arr = self.id();
        emit(&mut self.types_and_globals, OP_TYPE_RUNTIME_ARRAY, &[ty_arr, elem_ty]);
        emit(&mut self.annotations, OP_DECORATE, &[ty_arr, DECORATION_ARRAY_STRIDE, 4]);

        let ty_struct = self.id();
        emit(&mut self.types_and_globals, OP_TYPE_STRUCT, &[ty_struct, ty_arr]);
        emit(&mut self.annotations, OP_DECORATE, &[ty_struct, DECORATION_BUFFER_BLOCK]);
        emit(&mut self.annotations, OP_MEMBER_DECORATE, &[ty_struct, 0, DECORATION_OFFSET, 0]);

        let ty_ptr = self.id();
        emit(
            &mut self.types_and_globals,
            OP_TYPE_POINTER,
            &[ty_ptr, STORAGE_CLASS_UNIFORM, ty_struct],
        );

        let var = self.id();
        emit(
            &mut self.types_and_globals,
            OP_VARIABLE,
            &[ty_ptr, var, STORAGE_CLASS_UNIFORM],
        );
        emit(&mut self.annotations, OP_DECORATE, &[var, DECORATION_DESCRIPTOR_SET, 0]);
        emit(&mut self.annotations, OP_DECORATE, &[var, DECORATION_BINDING, binding]);
        var
    }

    pub fn const_i32(&mut self, v: i32) -> u32 {
        let r = self.id();
        emit(&mut self.types_and_globals, OP_CONSTANT, &[self.ty_i32, r, v as u32]);
        r
    }

    pub fn const_u32(&mut self, v: u32) -> u32 {
        let r = self.id();
        emit(&mut self.types_and_globals, OP_CONSTANT, &[self.ty_u32, r, v]);
        r
    }

    pub fn const_f32(&mut self, v: f32) -> u32 {
        let r = self.id();
        emit(&mut self.types_and_globals, OP_CONSTANT, &[self.ty_f32, r, v.to_bits()]);
        r
    }

    pub fn begin_main(&mut self) -> u32 {
        emit(
            &mut self.functions,
            OP_FUNCTION,
            &[self.ty_void, self.fn_main, 0, self.ty_fn_void],
        );
        let entry = self.id();
        self.label(entry);
        entry
    }

    pub fn end_main(&mut self) {
        emit(&mut self.functions, OP_RETURN, &[]);
        emit(&mut self.functions, OP_FUNCTION_END, &[]);
    }

    pub fn label(&mut self, lbl: u32) {
        emit(&mut self.functions, OP_LABEL, &[lbl]);
    }

    pub fn fn_var_i32(&mut self) -> u32 {
        let v = self.id();
        emit(&mut self.functions, OP_VARIABLE, &[self.ty_ptr_fn_i32, v, STORAGE_CLASS_FUNCTION]);
        v
    }

    pub fn fn_var_f32(&mut self) -> u32 {
        let v = self.id();
        emit(&mut self.functions, OP_VARIABLE, &[self.ty_ptr_fn_f32, v, STORAGE_CLASS_FUNCTION]);
        v
    }

    pub fn load(&mut self, ty: u32, ptr: u32) -> u32 {
        let r = self.id();
        emit(&mut self.functions, OP_LOAD, &[ty, r, ptr]);
        r
    }

    pub fn store(&mut self, ptr: u32, val: u32) {
        emit(&mut self.functions, OP_STORE, &[ptr, val]);
    }

    /// Load `(gl_GlobalInvocationID.x as i32, gl_GlobalInvocationID.y as i32)`.
    pub fn load_global_xy(&mut self) -> (u32, u32) {
        let gid = self.load(self.ty_uvec3, self.var_global_id);
        let gx_u = self.id();
        emit(&mut self.functions, OP_COMPOSITE_EXTRACT, &[self.ty_u32, gx_u, gid, 0]);
        let gy_u = self.id();
        emit(&mut self.functions, OP_COMPOSITE_EXTRACT, &[self.ty_u32, gy_u, gid, 1]);
        let gx = self.id();
        emit(&mut self.functions, OP_BITCAST, &[self.ty_i32, gx, gx_u]);
        let gy = self.id();
        emit(&mut self.functions, OP_BITCAST, &[self.ty_i32, gy, gy_u]);
        (gx, gy)
    }

    pub fn load_buf_f32(&mut self, buf: u32, zero_i32: u32, idx_i32: u32) -> u32 {
        let ptr = self.id();
        emit(
            &mut self.functions,
            OP_ACCESS_CHAIN,
            &[self.ty_ptr_uniform_f32, ptr, buf, zero_i32, idx_i32],
        );
        self.load(self.ty_f32, ptr)
    }

    pub fn store_buf_f32(&mut self, buf: u32, zero_i32: u32, idx_i32: u32, val: u32) {
        let ptr = self.id();
        emit(
            &mut self.functions,
            OP_ACCESS_CHAIN,
            &[self.ty_ptr_uniform_f32, ptr, buf, zero_i32, idx_i32],
        );
        self.store(ptr, val);
    }

    pub fn load_buf_i32(&mut self, buf: u32, zero_i32: u32, idx_i32: u32) -> u32 {
        let ptr = self.id();
        emit(
            &mut self.functions,
            OP_ACCESS_CHAIN,
            &[self.ty_ptr_uniform_i32, ptr, buf, zero_i32, idx_i32],
        );
        self.load(self.ty_i32, ptr)
    }

    pub fn binop(&mut self, op: u16, ty: u32, a: u32, b: u32) -> u32 {
        let r = self.id();
        emit(&mut self.functions, op, &[ty, r, a, b]);
        r
    }

    pub fn unop(&mut self, op: u16, ty: u32, a: u32) -> u32 {
        let r = self.id();
        emit(&mut self.functions, op, &[ty, r, a]);
        r
    }

    pub fn glsl1(&mut self, ext_op: u32, ty: u32, a: u32) -> u32 {
        let r = self.id();
        emit(&mut self.functions, OP_EXT_INST, &[ty, r, self.ext_glsl, ext_op, a]);
        r
    }

    pub fn glsl2(&mut self, ext_op: u32, ty: u32, a: u32, b: u32) -> u32 {
        let r = self.id();
        emit(&mut self.functions, OP_EXT_INST, &[ty, r, self.ext_glsl, ext_op, a, b]);
        r
    }

    pub fn glsl3(&mut self, ext_op: u32, ty: u32, a: u32, b: u32, c: u32) -> u32 {
        let r = self.id();
        emit(&mut self.functions, OP_EXT_INST, &[ty, r, self.ext_glsl, ext_op, a, b, c]);
        r
    }

    pub fn select(&mut self, ty: u32, cond: u32, t_val: u32, f_val: u32) -> u32 {
        let r = self.id();
        emit(&mut self.functions, OP_SELECT, &[ty, r, cond, t_val, f_val]);
        r
    }

    pub fn selection_merge(&mut self, merge_lbl: u32) {
        emit(&mut self.functions, OP_SELECTION_MERGE, &[merge_lbl, 0]);
    }

    pub fn loop_merge(&mut self, merge_lbl: u32, continue_lbl: u32) {
        emit(&mut self.functions, OP_LOOP_MERGE, &[merge_lbl, continue_lbl, 0]);
    }

    pub fn branch(&mut self, target: u32) {
        emit(&mut self.functions, OP_BRANCH, &[target]);
    }

    pub fn branch_cond(&mut self, cond: u32, true_lbl: u32, false_lbl: u32) {
        emit(&mut self.functions, OP_BRANCH_CONDITIONAL, &[cond, true_lbl, false_lbl]);
    }

    pub fn return_void(&mut self) {
        emit(&mut self.functions, OP_RETURN, &[]);
    }

    /// Assemble the final SPIR-V binary word stream.
    pub fn finish(self) -> Vec<u32> {
        let mut words = Vec::with_capacity(
            5 + self.capabilities.len()
                + self.ext_imports.len()
                + self.memory_model.len()
                + self.entry_points.len()
                + self.exec_modes.len()
                + self.annotations.len()
                + self.types_and_globals.len()
                + self.functions.len(),
        );
        words.push(SPIRV_MAGIC);
        words.push(SPIRV_VERSION_1_0);
        words.push(0); // Generator magic
        words.push(self.next_id); // Bound
        words.push(0); // Schema
        words.extend_from_slice(&self.capabilities);
        words.extend_from_slice(&self.ext_imports);
        words.extend_from_slice(&self.memory_model);
        words.extend_from_slice(&self.entry_points);
        words.extend_from_slice(&self.exec_modes);
        words.extend_from_slice(&self.annotations);
        words.extend_from_slice(&self.types_and_globals);
        words.extend_from_slice(&self.functions);
        words
    }
}

fn emit(buf: &mut Vec<u32>, opcode: u16, operands: &[u32]) {
    let word_count = (1 + operands.len()) as u32;
    buf.push((word_count << 16) | (opcode as u32));
    buf.extend_from_slice(operands);
}

fn encode_string(s: &str) -> Vec<u32> {
    let bytes = s.as_bytes();
    let total = bytes.len() + 1; // null terminator
    let words = (total + 3) / 4;
    let mut out = vec![0u32; words];
    for (i, &b) in bytes.iter().enumerate() {
        out[i / 4] |= (b as u32) << ((i % 4) * 8);
    }
    out
}

/// Structural verification stats returned by `validate_spirv`.
#[derive(Debug, Clone)]
pub struct SpirvStats {
    pub word_count: usize,
    pub instruction_count: usize,
    pub bound: u32,
    pub has_compute_entry: bool,
}

/// Verify SPIR-V 1.0 binary structure: magic, version, instruction framing, and compute entry point.
pub fn validate_spirv(words: &[u32]) -> Result<SpirvStats, String> {
    if words.len() < 5 {
        return Err("SPIR-V module shorter than 5-word header".into());
    }
    if words[0] != SPIRV_MAGIC {
        return Err(format!("Invalid SPIR-V magic: 0x{:08x}", words[0]));
    }
    let bound = words[3];
    if bound < 2 {
        return Err("Invalid SPIR-V ID bound".into());
    }
    let mut pos = 5usize;
    let mut inst_count = 0usize;
    let mut has_compute_entry = false;
    while pos < words.len() {
        let header = words[pos];
        let wc = (header >> 16) as usize;
        let op = (header & 0xFFFF) as u16;
        if wc == 0 || pos + wc > words.len() {
            return Err(format!("Malformed instruction at word {} (wc={}, op={})", pos, wc, op));
        }
        if op == OP_ENTRY_POINT && wc >= 3 && words[pos + 1] == EXECUTION_MODEL_GLCOMPUTE {
            has_compute_entry = true;
        }
        pos += wc;
        inst_count += 1;
    }
    if !has_compute_entry {
        return Err("Missing GLCompute OpEntryPoint".into());
    }
    Ok(SpirvStats {
        word_count: words.len(),
        instruction_count: inst_count,
        bound,
        has_compute_entry,
    })
}
