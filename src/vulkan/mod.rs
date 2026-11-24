//! Safe Vulkan 1.0 Compute execution engine with automatic GPU selection,
//! persistent SPIR-V compute pipeline caching, and clean RAII resource management.

pub mod ffi;
pub mod shaders;
pub mod spirv;

use std::collections::HashMap;
use std::ffi::CString;
use std::ptr;
use std::sync::Mutex;

use ffi::*;

#[derive(Debug, Clone)]
pub struct VulkanDeviceInfo {
    pub device_name: String,
    pub device_type: &'static str,
    pub api_version: String,
    pub vendor_id: u32,
    pub compute_queue_family: u32,
    pub memory_heap_mb: u64,
}

/// Identifiers for built-in cached SPIR-V 1.0 compute pipelines.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ShaderKind {
    PrefilterBoxSubmean,
    PrefilterLog2d,
    StereoBm,
    StereoCostVolume3d,
    SubpixelParabola,
    DemHillshade,
}

impl ShaderKind {
    fn binding_count(self) -> usize {
        match self {
            Self::PrefilterBoxSubmean | Self::PrefilterLog2d => 3,
            Self::DemHillshade | Self::StereoCostVolume3d => 4,
            Self::StereoBm => 5,
            Self::SubpixelParabola => 6,
        }
    }

    fn build_spirv(self) -> Vec<u32> {
        match self {
            Self::PrefilterBoxSubmean => shaders::build_prefilter_shader(),
            Self::PrefilterLog2d => shaders::build_log_2d_shader(),
            Self::StereoBm => shaders::build_stereo_bm_shader(),
            Self::StereoCostVolume3d => shaders::build_cost_volume_3d_shader(),
            Self::SubpixelParabola => shaders::build_subpixel_parabola_shader(),
            Self::DemHillshade => shaders::build_hillshade_shader(),
        }
    }
}

#[derive(Clone, Copy)]
struct CachedPipeline {
    shader_module: VkShaderModule,
    dsl: VkDescriptorSetLayout,
    pipe_layout: VkPipelineLayout,
    pipeline: VkPipeline,
}

/// Live Vulkan 1.0 Compute context with persistent compiled SPIR-V pipeline cache.
pub struct VulkanCompute {
    vk: VkFunctions,
    instance: VkInstance,
    device: VkDevice,
    queue: VkQueue,
    command_pool: VkCommandPool,
    mem_props: VkPhysicalDeviceMemoryProperties,
    pipeline_cache: Mutex<HashMap<ShaderKind, CachedPipeline>>,
    queue_lock: Mutex<()>,
    pub info: VulkanDeviceInfo,
}

unsafe impl Send for VulkanCompute {}
unsafe impl Sync for VulkanCompute {}

impl VulkanCompute {
    /// Attempt to initialize Vulkan 1.0 Compute on the highest-priority compute-capable GPU.
    pub fn try_new() -> Result<Self, String> {
        let vk = VkFunctions::load()?;

        let app_name = CString::new("RustStereoPipelineVulkan").map_err(|e| e.to_string())?;
        let eng_name = CString::new("RSPV-ZeroDep-Compute").map_err(|e| e.to_string())?;
        let app_info = VkApplicationInfo {
            sType: VK_STRUCTURE_TYPE_APPLICATION_INFO,
            pNext: ptr::null(),
            pApplicationName: app_name.as_ptr(),
            applicationVersion: vk_make_version(0, 2, 0),
            pEngineName: eng_name.as_ptr(),
            engineVersion: vk_make_version(0, 2, 0),
            apiVersion: vk_make_version(1, 0, 0),
        };
        let inst_ci = VkInstanceCreateInfo {
            sType: VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
            pNext: ptr::null(),
            flags: 0,
            pApplicationInfo: &app_info,
            enabledLayerCount: 0,
            ppEnabledLayerNames: ptr::null(),
            enabledExtensionCount: 0,
            ppEnabledExtensionNames: ptr::null(),
        };

        let mut instance: VkInstance = ptr::null_mut();
        let res = unsafe { (vk.vkCreateInstance)(&inst_ci, ptr::null(), &mut instance) };
        if res != VK_SUCCESS || instance.is_null() {
            return Err(format!("vkCreateInstance failed (VkResult={})", res));
        }

        let mut dev_count = 0u32;
        let res = unsafe { (vk.vkEnumeratePhysicalDevices)(instance, &mut dev_count, ptr::null_mut()) };
        if res != VK_SUCCESS || dev_count == 0 {
            unsafe { (vk.vkDestroyInstance)(instance, ptr::null()) };
            return Err("No Vulkan physical devices / ICD drivers available on this system".into());
        }

        let mut phys_devs = vec![ptr::null_mut(); dev_count as usize];
        let res = unsafe {
            (vk.vkEnumeratePhysicalDevices)(instance, &mut dev_count, phys_devs.as_mut_ptr())
        };
        if res != VK_SUCCESS {
            unsafe { (vk.vkDestroyInstance)(instance, ptr::null()) };
            return Err(format!("vkEnumeratePhysicalDevices failed ({})", res));
        }

        // Select best compute-capable device (Discrete > Integrated > Virtual > CPU)
        let mut best: Option<(i32, VkPhysicalDevice, u32, VkPhysicalDeviceProperties)> = None;
        for pd in phys_devs {
            let mut props = VkPhysicalDeviceProperties::default();
            unsafe { (vk.vkGetPhysicalDeviceProperties)(pd, &mut props) };
            let mut qf_count = 0u32;
            unsafe { (vk.vkGetPhysicalDeviceQueueFamilyProperties)(pd, &mut qf_count, ptr::null_mut()) };
            let mut qfs = vec![VkQueueFamilyProperties::default(); qf_count as usize];
            unsafe { (vk.vkGetPhysicalDeviceQueueFamilyProperties)(pd, &mut qf_count, qfs.as_mut_ptr()) };

            if let Some((qidx, _)) = qfs
                .iter()
                .enumerate()
                .find(|(_, q)| (q.queueFlags & VK_QUEUE_COMPUTE_BIT) != 0 && q.queueCount > 0)
            {
                let score = match props.deviceType {
                    VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU => 100,
                    VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU => 70,
                    VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU => 40,
                    VK_PHYSICAL_DEVICE_TYPE_CPU => 20,
                    _ => 10,
                };
                if best.as_ref().map_or(true, |(bs, _, _, _)| score > *bs) {
                    best = Some((score, pd, qidx as u32, props));
                }
            }
        }

        let (_, phys_dev, queue_family, props) = match best {
            Some(b) => b,
            None => {
                unsafe { (vk.vkDestroyInstance)(instance, ptr::null()) };
                return Err("No Vulkan compute-capable queue family found".into());
            }
        };

        let mut mem_props = VkPhysicalDeviceMemoryProperties::default();
        unsafe { (vk.vkGetPhysicalDeviceMemoryProperties)(phys_dev, &mut mem_props) };

        let prio = 1.0f32;
        let q_ci = VkDeviceQueueCreateInfo {
            sType: VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
            pNext: ptr::null(),
            flags: 0,
            queueFamilyIndex: queue_family,
            queueCount: 1,
            pQueuePriorities: &prio,
        };
        let dev_ci = VkDeviceCreateInfo {
            sType: VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
            pNext: ptr::null(),
            flags: 0,
            queueCreateInfoCount: 1,
            pQueueCreateInfos: &q_ci,
            enabledLayerCount: 0,
            ppEnabledLayerNames: ptr::null(),
            enabledExtensionCount: 0,
            ppEnabledExtensionNames: ptr::null(),
            pEnabledFeatures: ptr::null(),
        };

        let mut device: VkDevice = ptr::null_mut();
        let res = unsafe { (vk.vkCreateDevice)(phys_dev, &dev_ci, ptr::null(), &mut device) };
        if res != VK_SUCCESS || device.is_null() {
            unsafe { (vk.vkDestroyInstance)(instance, ptr::null()) };
            return Err(format!("vkCreateDevice failed ({})", res));
        }

        let mut queue: VkQueue = ptr::null_mut();
        unsafe { (vk.vkGetDeviceQueue)(device, queue_family, 0, &mut queue) };

        let cp_ci = VkCommandPoolCreateInfo {
            sType: VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
            pNext: ptr::null(),
            flags: VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
            queueFamilyIndex: queue_family,
        };
        let mut command_pool: VkCommandPool = 0;
        let res = unsafe { (vk.vkCreateCommandPool)(device, &cp_ci, ptr::null(), &mut command_pool) };
        if res != VK_SUCCESS {
            unsafe {
                (vk.vkDestroyDevice)(device, ptr::null());
                (vk.vkDestroyInstance)(instance, ptr::null());
            }
            return Err(format!("vkCreateCommandPool failed ({})", res));
        }

        let dev_type = match props.deviceType {
            VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU => "Discrete GPU",
            VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU => "Integrated GPU",
            VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU => "Virtual GPU",
            VK_PHYSICAL_DEVICE_TYPE_CPU => "CPU (Software Vulkan)",
            _ => "Other",
        };
        let total_mem_bytes: u64 = (0..mem_props.memoryHeapCount as usize)
            .map(|i| mem_props.memoryHeaps[i].size)
            .max()
            .unwrap_or(0);
        let api_v = props.apiVersion;
        let info = VulkanDeviceInfo {
            device_name: c_char_array_to_string(&props.deviceName),
            device_type: dev_type,
            api_version: format!("{}.{}.{}", api_v >> 22, (api_v >> 12) & 0x3FF, api_v & 0xFFF),
            vendor_id: props.vendorID,
            compute_queue_family: queue_family,
            memory_heap_mb: total_mem_bytes / (1024 * 1024),
        };

        Ok(Self {
            vk,
            instance,
            device,
            queue,
            command_pool,
            mem_props,
            pipeline_cache: Mutex::new(HashMap::new()),
            queue_lock: Mutex::new(()),
            info,
        })
    }

    /// Retrieve or compile+cache a SPIR-V compute pipeline once per `ShaderKind`.
    fn get_or_compile_pipeline(&self, kind: ShaderKind) -> Result<CachedPipeline, String> {
        let mut guard = self
            .pipeline_cache
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        if let Some(&cached) = guard.get(&kind) {
            return Ok(cached);
        }

        let spirv_words = kind.build_spirv();
        let binding_count = kind.binding_count();

        let sm_ci = VkShaderModuleCreateInfo {
            sType: VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
            pNext: ptr::null(),
            flags: 0,
            codeSize: spirv_words.len() * 4,
            pCode: spirv_words.as_ptr(),
        };
        let mut shader_module = 0u64;
        let res = unsafe {
            (self.vk.vkCreateShaderModule)(self.device, &sm_ci, ptr::null(), &mut shader_module)
        };
        if res != VK_SUCCESS {
            return Err(format!("vkCreateShaderModule failed ({})", res));
        }

        let bindings: Vec<VkDescriptorSetLayoutBinding> = (0..binding_count as u32)
            .map(|i| VkDescriptorSetLayoutBinding {
                binding: i,
                descriptorType: VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                descriptorCount: 1,
                stageFlags: VK_SHADER_STAGE_COMPUTE_BIT,
                pImmutableSamplers: ptr::null(),
            })
            .collect();

        let dsl_ci = VkDescriptorSetLayoutCreateInfo {
            sType: VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
            pNext: ptr::null(),
            flags: 0,
            bindingCount: bindings.len() as u32,
            pBindings: bindings.as_ptr(),
        };
        let mut dsl = 0u64;
        let res = unsafe {
            (self.vk.vkCreateDescriptorSetLayout)(self.device, &dsl_ci, ptr::null(), &mut dsl)
        };
        if res != VK_SUCCESS {
            unsafe { (self.vk.vkDestroyShaderModule)(self.device, shader_module, ptr::null()) };
            return Err(format!("vkCreateDescriptorSetLayout failed ({})", res));
        }

        let pl_ci = VkPipelineLayoutCreateInfo {
            sType: VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
            pNext: ptr::null(),
            flags: 0,
            setLayoutCount: 1,
            pSetLayouts: &dsl,
            pushConstantRangeCount: 0,
            pPushConstantRanges: ptr::null(),
        };
        let mut pipe_layout = 0u64;
        let res = unsafe {
            (self.vk.vkCreatePipelineLayout)(self.device, &pl_ci, ptr::null(), &mut pipe_layout)
        };
        if res != VK_SUCCESS {
            unsafe {
                (self.vk.vkDestroyDescriptorSetLayout)(self.device, dsl, ptr::null());
                (self.vk.vkDestroyShaderModule)(self.device, shader_module, ptr::null());
            }
            return Err(format!("vkCreatePipelineLayout failed ({})", res));
        }

        let entry_name = CString::new("main").map_err(|e| e.to_string())?;
        let cp_ci = VkComputePipelineCreateInfo {
            sType: VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
            pNext: ptr::null(),
            flags: 0,
            stage: VkPipelineShaderStageCreateInfo {
                sType: VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                pNext: ptr::null(),
                flags: 0,
                stage: VK_SHADER_STAGE_COMPUTE_BIT,
                module: shader_module,
                pName: entry_name.as_ptr(),
                pSpecializationInfo: ptr::null(),
            },
            layout: pipe_layout,
            basePipelineHandle: 0,
            basePipelineIndex: -1,
        };
        let mut pipeline = 0u64;
        let res = unsafe {
            (self.vk.vkCreateComputePipelines)(self.device, 0, 1, &cp_ci, ptr::null(), &mut pipeline)
        };
        if res != VK_SUCCESS {
            unsafe {
                (self.vk.vkDestroyPipelineLayout)(self.device, pipe_layout, ptr::null());
                (self.vk.vkDestroyDescriptorSetLayout)(self.device, dsl, ptr::null());
                (self.vk.vkDestroyShaderModule)(self.device, shader_module, ptr::null());
            }
            return Err(format!("vkCreateComputePipelines failed ({})", res));
        }

        let cached = CachedPipeline {
            shader_module,
            dsl,
            pipe_layout,
            pipeline,
        };
        guard.insert(kind, cached);
        Ok(cached)
    }

    pub fn cached_pipeline_count(&self) -> usize {
        self.pipeline_cache
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .len()
    }

    fn find_host_coherent_memory(&self, type_bits: u32) -> Option<u32> {
        let req = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
        for i in 0..self.mem_props.memoryTypeCount {
            if (type_bits & (1 << i)) != 0
                && (self.mem_props.memoryTypes[i as usize].propertyFlags & req) == req
            {
                return Some(i);
            }
        }
        None
    }

    /// Prefer `HOST_VISIBLE | HOST_COHERENT | HOST_CACHED` memory for GPU->CPU readback buffers
    /// (avoids slow uncached PCIe WC readback stalls on discrete GPUs), falling back to host-coherent.
    fn find_host_cached_memory(&self, type_bits: u32) -> Option<u32> {
        let cached = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT
            | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
            | VK_MEMORY_PROPERTY_HOST_CACHED_BIT;
        for i in 0..self.mem_props.memoryTypeCount {
            if (type_bits & (1 << i)) != 0
                && (self.mem_props.memoryTypes[i as usize].propertyFlags & cached) == cached
            {
                return Some(i);
            }
        }
        self.find_host_coherent_memory(type_bits)
    }

    fn create_buffer_with_data(&self, bytes: &[u8]) -> Result<(VkBuffer, VkDeviceMemory, VkDeviceSize), String> {
        let size = (bytes.len().max(4)) as VkDeviceSize;
        let b_ci = VkBufferCreateInfo {
            sType: VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
            pNext: ptr::null(),
            flags: 0,
            size,
            usage: VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
            sharingMode: VK_SHARING_MODE_EXCLUSIVE,
            queueFamilyIndexCount: 0,
            pQueueFamilyIndices: ptr::null(),
        };
        let mut buffer = 0u64;
        let res = unsafe { (self.vk.vkCreateBuffer)(self.device, &b_ci, ptr::null(), &mut buffer) };
        if res != VK_SUCCESS {
            return Err(format!("vkCreateBuffer failed ({})", res));
        }

        let mut reqs = VkMemoryRequirements::default();
        unsafe { (self.vk.vkGetBufferMemoryRequirements)(self.device, buffer, &mut reqs) };
        let mem_idx = self
            .find_host_coherent_memory(reqs.memoryTypeBits)
            .ok_or_else(|| "No HOST_VISIBLE|HOST_COHERENT Vulkan memory type".to_string())?;

        let alloc_ci = VkMemoryAllocateInfo {
            sType: VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
            pNext: ptr::null(),
            allocationSize: reqs.size,
            memoryTypeIndex: mem_idx,
        };
        let mut mem = 0u64;
        let res = unsafe { (self.vk.vkAllocateMemory)(self.device, &alloc_ci, ptr::null(), &mut mem) };
        if res != VK_SUCCESS {
            unsafe { (self.vk.vkDestroyBuffer)(self.device, buffer, ptr::null()) };
            return Err(format!("vkAllocateMemory failed ({})", res));
        }
        let res = unsafe { (self.vk.vkBindBufferMemory)(self.device, buffer, mem, 0) };
        if res != VK_SUCCESS {
            unsafe {
                (self.vk.vkFreeMemory)(self.device, mem, ptr::null());
                (self.vk.vkDestroyBuffer)(self.device, buffer, ptr::null());
            }
            return Err(format!("vkBindBufferMemory failed ({})", res));
        }

        if !bytes.is_empty() {
            let mut mapped: *mut std::ffi::c_void = ptr::null_mut();
            let res = unsafe { (self.vk.vkMapMemory)(self.device, mem, 0, size, 0, &mut mapped) };
            if res != VK_SUCCESS {
                unsafe {
                    (self.vk.vkFreeMemory)(self.device, mem, ptr::null());
                    (self.vk.vkDestroyBuffer)(self.device, buffer, ptr::null());
                }
                return Err(format!("vkMapMemory failed ({})", res));
            }
            unsafe {
                ptr::copy_nonoverlapping(bytes.as_ptr(), mapped as *mut u8, bytes.len());
                (self.vk.vkUnmapMemory)(self.device, mem);
            }
        }
        Ok((buffer, mem, size))
    }

    /// Allocate a host-visible cached GPU storage buffer of `byte_len` bytes WITHOUT uploading zeros from CPU.
    fn create_uninit_buffer(&self, byte_len: usize) -> Result<(VkBuffer, VkDeviceMemory, VkDeviceSize), String> {
        let size = byte_len.max(4) as VkDeviceSize;
        let buf_ci = VkBufferCreateInfo {
            sType: VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
            pNext: ptr::null(),
            flags: 0,
            size,
            usage: VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
            sharingMode: VK_SHARING_MODE_EXCLUSIVE,
            queueFamilyIndexCount: 0,
            pQueueFamilyIndices: ptr::null(),
        };
        let mut buffer = 0u64;
        let res = unsafe { (self.vk.vkCreateBuffer)(self.device, &buf_ci, ptr::null(), &mut buffer) };
        if res != VK_SUCCESS {
            return Err(format!("vkCreateBuffer failed ({})", res));
        }
        let mut reqs = VkMemoryRequirements::default();
        unsafe { (self.vk.vkGetBufferMemoryRequirements)(self.device, buffer, &mut reqs) };
        let mem_type = self
            .find_host_cached_memory(reqs.memoryTypeBits)
            .ok_or_else(|| "No HOST_VISIBLE|HOST_COHERENT Vulkan memory type found".to_string())?;
        let alloc_ci = VkMemoryAllocateInfo {
            sType: VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
            pNext: ptr::null(),
            allocationSize: reqs.size,
            memoryTypeIndex: mem_type,
        };
        let mut mem = 0u64;
        let res = unsafe { (self.vk.vkAllocateMemory)(self.device, &alloc_ci, ptr::null(), &mut mem) };
        if res != VK_SUCCESS {
            unsafe { (self.vk.vkDestroyBuffer)(self.device, buffer, ptr::null()) };
            return Err(format!("vkAllocateMemory failed ({})", res));
        }
        let res = unsafe { (self.vk.vkBindBufferMemory)(self.device, buffer, mem, 0) };
        if res != VK_SUCCESS {
            unsafe {
                (self.vk.vkFreeMemory)(self.device, mem, ptr::null());
                (self.vk.vkDestroyBuffer)(self.device, buffer, ptr::null());
            }
            return Err(format!("vkBindBufferMemory failed ({})", res));
        }
        Ok((buffer, mem, size))
    }

    /// Execute a cached SPIR-V compute pipeline with explicit input slices (`Some`) and zero-upload output sizes (`None`).
    fn dispatch_cached_io(
        &self,
        kind: ShaderKind,
        specs: &[Result<&[u8], usize>],
        readback_indices: &[usize],
        group_x: u32,
        group_y: u32,
        group_z: u32,
    ) -> Result<Vec<Vec<u8>>, String> {
        let cached = self.get_or_compile_pipeline(kind)?;
        let _qguard = self.queue_lock.lock().unwrap_or_else(|e| e.into_inner());

        let mut gpu_bufs = Vec::with_capacity(specs.len());
        for spec in specs {
            match *spec {
                Ok(data) => gpu_bufs.push(self.create_buffer_with_data(data)?),
                Err(out_bytes) => gpu_bufs.push(self.create_uninit_buffer(out_bytes)?),
            }
        }

        let result = self.dispatch_with_pipeline(
            cached,
            &gpu_bufs,
            readback_indices,
            group_x.max(1),
            group_y.max(1),
            group_z.max(1),
        );

        for (buf, mem, _) in gpu_bufs {
            unsafe {
                (self.vk.vkDestroyBuffer)(self.device, buf, ptr::null());
                (self.vk.vkFreeMemory)(self.device, mem, ptr::null());
            }
        }
        result
    }

    /// Execute a cached SPIR-V compute pipeline with storage buffers and read back output buffers.
    pub fn dispatch_cached(
        &self,
        kind: ShaderKind,
        buffers_in: &[&[u8]],
        readback_indices: &[usize],
        group_x: u32,
        group_y: u32,
        group_z: u32,
    ) -> Result<Vec<Vec<u8>>, String> {
        let specs: Vec<Result<&[u8], usize>> = buffers_in.iter().map(|b| Ok(*b)).collect();
        self.dispatch_cached_io(kind, &specs, readback_indices, group_x, group_y, group_z)
    }

    fn dispatch_with_pipeline(
        &self,
        cached: CachedPipeline,
        gpu_bufs: &[(VkBuffer, VkDeviceMemory, VkDeviceSize)],
        readback_indices: &[usize],
        gx: u32,
        gy: u32,
        gz: u32,
    ) -> Result<Vec<Vec<u8>>, String> {
        let pool_size = VkDescriptorPoolSize {
            descriptorType: VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            descriptorCount: gpu_bufs.len() as u32,
        };
        let dp_ci = VkDescriptorPoolCreateInfo {
            sType: VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
            pNext: ptr::null(),
            flags: 0,
            maxSets: 1,
            poolSizeCount: 1,
            pPoolSizes: &pool_size,
        };
        let mut desc_pool = 0u64;
        let res = unsafe {
            (self.vk.vkCreateDescriptorPool)(self.device, &dp_ci, ptr::null(), &mut desc_pool)
        };
        if res != VK_SUCCESS {
            return Err(format!("vkCreateDescriptorPool failed ({})", res));
        }

        let ds_ai = VkDescriptorSetAllocateInfo {
            sType: VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
            pNext: ptr::null(),
            descriptorPool: desc_pool,
            descriptorSetCount: 1,
            pSetLayouts: &cached.dsl,
        };
        let mut desc_set = 0u64;
        let res = unsafe {
            (self.vk.vkAllocateDescriptorSets)(self.device, &ds_ai, &mut desc_set)
        };
        if res != VK_SUCCESS {
            unsafe { (self.vk.vkDestroyDescriptorPool)(self.device, desc_pool, ptr::null()) };
            return Err(format!("vkAllocateDescriptorSets failed ({})", res));
        }

        let buf_infos: Vec<VkDescriptorBufferInfo> = gpu_bufs
            .iter()
            .map(|(b, _, sz)| VkDescriptorBufferInfo {
                buffer: *b,
                offset: 0,
                range: *sz,
            })
            .collect();
        let writes: Vec<VkWriteDescriptorSet> = buf_infos
            .iter()
            .enumerate()
            .map(|(i, bi)| VkWriteDescriptorSet {
                sType: VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                pNext: ptr::null(),
                dstSet: desc_set,
                dstBinding: i as u32,
                dstArrayElement: 0,
                descriptorCount: 1,
                descriptorType: VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                pImageInfo: ptr::null(),
                pBufferInfo: bi as *const VkDescriptorBufferInfo,
                pTexelBufferView: ptr::null(),
            })
            .collect();
        unsafe {
            (self.vk.vkUpdateDescriptorSets)(self.device, writes.len() as u32, writes.as_ptr(), 0, ptr::null());
        }

        let cb_ai = VkCommandBufferAllocateInfo {
            sType: VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
            pNext: ptr::null(),
            commandPool: self.command_pool,
            level: VK_COMMAND_BUFFER_LEVEL_PRIMARY,
            commandBufferCount: 1,
        };
        let mut cmd_buf: VkCommandBuffer = ptr::null_mut();
        unsafe {
            (self.vk.vkAllocateCommandBuffers)(self.device, &cb_ai, &mut cmd_buf);
            let begin = VkCommandBufferBeginInfo {
                sType: VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
                pNext: ptr::null(),
                flags: VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
                pInheritanceInfo: ptr::null(),
            };
            (self.vk.vkBeginCommandBuffer)(cmd_buf, &begin);
            (self.vk.vkCmdBindPipeline)(cmd_buf, VK_PIPELINE_BIND_POINT_COMPUTE, cached.pipeline);
            (self.vk.vkCmdBindDescriptorSets)(
                cmd_buf,
                VK_PIPELINE_BIND_POINT_COMPUTE,
                cached.pipe_layout,
                0,
                1,
                &desc_set,
                0,
                ptr::null(),
            );
            (self.vk.vkCmdDispatch)(cmd_buf, gx, gy, gz);
            (self.vk.vkEndCommandBuffer)(cmd_buf);

            let fence_ci = VkFenceCreateInfo {
                sType: VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
                pNext: ptr::null(),
                flags: 0,
            };
            let mut fence = 0u64;
            (self.vk.vkCreateFence)(self.device, &fence_ci, ptr::null(), &mut fence);
            let submit = VkSubmitInfo {
                sType: VK_STRUCTURE_TYPE_SUBMIT_INFO,
                pNext: ptr::null(),
                waitSemaphoreCount: 0,
                pWaitSemaphores: ptr::null(),
                pWaitDstStageMask: ptr::null(),
                commandBufferCount: 1,
                pCommandBuffers: &cmd_buf,
                signalSemaphoreCount: 0,
                pSignalSemaphores: ptr::null(),
            };
            (self.vk.vkQueueSubmit)(self.queue, 1, &submit, fence);
            (self.vk.vkWaitForFences)(self.device, 1, &fence, VK_TRUE, u64::MAX);
            (self.vk.vkDestroyFence)(self.device, fence, ptr::null());
        }

        let mut outputs = Vec::with_capacity(readback_indices.len());
        for &idx in readback_indices {
            let (_, mem, sz) = gpu_bufs[idx];
            let mut out = vec![0u8; sz as usize];
            let mut mapped: *mut std::ffi::c_void = ptr::null_mut();
            unsafe {
                let res = (self.vk.vkMapMemory)(self.device, mem, 0, sz, 0, &mut mapped);
                if res == VK_SUCCESS {
                    ptr::copy_nonoverlapping(mapped as *const u8, out.as_mut_ptr(), sz as usize);
                    (self.vk.vkUnmapMemory)(self.device, mem);
                }
            }
            outputs.push(out);
        }

        unsafe {
            (self.vk.vkDestroyDescriptorPool)(self.device, desc_pool, ptr::null());
        }
        Ok(outputs)
    }

    /// Run GPU Subtracted-Mean / Box Prefilter via cached SPIR-V compute pipeline.
    pub fn run_prefilter(&self, input: &[f32], width: u32, height: u32, radius: i32, subtracted_mean: bool) -> Result<Vec<f32>, String> {
        let params: [i32; 4] = [width as i32, height as i32, radius, if subtracted_mean { 1 } else { 0 }];
        let out_zeros = vec![0.0f32; input.len()];
        let res = self.dispatch_cached(
            ShaderKind::PrefilterBoxSubmean,
            &[as_bytes(input), as_bytes(&out_zeros), as_bytes(&params)],
            &[1],
            (width + 15) / 16,
            (height + 15) / 16,
            1,
        )?;
        Ok(bytes_to_f32(&res[0]))
    }

    /// Run GPU 2D Discrete Laplacian pass via cached SPIR-V compute pipeline.
    pub fn run_log_2d(&self, input: &[f32], width: u32, height: u32) -> Result<Vec<f32>, String> {
        let params: [i32; 2] = [width as i32, height as i32];
        let out_zeros = vec![0.0f32; input.len()];
        let res = self.dispatch_cached(
            ShaderKind::PrefilterLog2d,
            &[as_bytes(input), as_bytes(&out_zeros), as_bytes(&params)],
            &[1],
            (width + 15) / 16,
            (height + 15) / 16,
            1,
        )?;
        Ok(bytes_to_f32(&res[0]))
    }

    /// Run GPU Block Matching stereo correlation via cached SPIR-V compute pipeline.
    pub fn run_stereo_bm(
        &self,
        left: &[f32],
        right: &[f32],
        width: u32,
        height: u32,
        half_kw: i32,
        half_kh: i32,
        min_dx: i32,
        max_dx: i32,
        cost_mode: i32,
    ) -> Result<(Vec<f32>, Vec<f32>), String> {
        let params: [i32; 7] = [width as i32, height as i32, half_kw, half_kh, min_dx, max_dx, cost_mode];
        let zeros = vec![0.0f32; left.len()];
        let res = self.dispatch_cached(
            ShaderKind::StereoBm,
            &[
                as_bytes(left),
                as_bytes(right),
                as_bytes(&zeros),
                as_bytes(&zeros),
                as_bytes(&params),
            ],
            &[2, 3],
            (width + 15) / 16,
            (height + 15) / 16,
            1,
        )?;
        Ok((bytes_to_f32(&res[0]), bytes_to_f32(&res[1])))
    }

    /// Run GPU 3D Stereo Cost Volume computation (`C[y][x][d]` for SGM & MGM) via cached SPIR-V pipeline.
    /// Computes normalized SAD (0), SSD (1), hybrid ZNCC+Census (2), or Census (3) for `num_d` disparities.
    /// Automatically splits large volumes into halo-padded Y-slabs (<= 240 MB per GPU dispatch) so any
    /// ROI size (`1024x1024x129`, `2048x2048x129`, etc.) executes on Vulkan GPU without exceeding
    /// `maxStorageBufferRange`, PCIe BAR limits, or 32-bit SPIR-V buffer indexing.
    #[allow(clippy::too_many_arguments)]
    pub fn run_stereo_cost_volume(
        &self,
        left: &[f32],
        right: &[f32],
        width: u32,
        height: u32,
        half_kw: i32,
        half_kh: i32,
        min_dx: i32,
        num_d: usize,
        cost_mode: i32,
    ) -> Result<Vec<f32>, String> {
        let w = width as usize;
        let h = height as usize;
        let row_elems = w
            .checked_mul(num_d)
            .ok_or_else(|| "Cost volume row stride overflow".to_string())?;
        let total_elems = row_elems
            .checked_mul(h)
            .ok_or_else(|| "Cost volume size overflow".to_string())?;
        if w == 0 || h == 0 || num_d == 0 {
            return Ok(Vec::new());
        }

        // Keep each Vulkan storage buffer <= 240 MB (safe for 256 MB BARs & 32-bit SPIR-V indexing)
        let max_slab_bytes: usize = 240 * 1024 * 1024;
        let pad_rows = half_kh.max(0) as usize;
        let slabs = plan_cost_volume_slabs(h, row_elems * 4, pad_rows, max_slab_bytes);

        if slabs.len() == 1 {
            let params: [i32; 7] = [
                width as i32,
                height as i32,
                half_kw,
                half_kh,
                min_dx,
                num_d as i32,
                cost_mode,
            ];
            let res = self.dispatch_cached_io(
                ShaderKind::StereoCostVolume3d,
                &[
                    Ok(as_bytes(left)),
                    Ok(as_bytes(right)),
                    Err(total_elems * 4),
                    Ok(as_bytes(&params)),
                ],
                &[2],
                (width + 15) / 16,
                (height + 15) / 16,
                1,
            )?;
            return Ok(bytes_to_f32(&res[0]));
        }

        let mut out_vol = vec![0.0f32; total_elems];
        for slab in slabs {
            let src_start = slab.src_y0 * w;
            let src_end = slab.src_y1 * w;
            let sub_h = (slab.src_y1 - slab.src_y0) as u32;
            let sub_elems = (sub_h as usize) * row_elems;
            let params: [i32; 7] = [
                width as i32,
                sub_h as i32,
                half_kw,
                half_kh,
                min_dx,
                num_d as i32,
                cost_mode,
            ];
            let res = self.dispatch_cached_io(
                ShaderKind::StereoCostVolume3d,
                &[
                    Ok(as_bytes(&left[src_start..src_end])),
                    Ok(as_bytes(&right[src_start..src_end])),
                    Err(sub_elems * 4),
                    Ok(as_bytes(&params)),
                ],
                &[2],
                (width + 15) / 16,
                (sub_h + 15) / 16,
                1,
            )?;
            let keep_start_elems = (slab.core_y0 - slab.src_y0) * row_elems;
            let keep_elems = (slab.core_y1 - slab.core_y0) * row_elems;
            let dst_start = slab.core_y0 * row_elems;
            let byte_slice = &res[0][keep_start_elems * 4..(keep_start_elems + keep_elems) * 4];
            for (dst, chunk) in out_vol[dst_start..dst_start + keep_elems]
                .iter_mut()
                .zip(byte_slice.chunks_exact(4))
            {
                *dst = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
            }
        }
        Ok(out_vol)
    }

    /// Run GPU Subpixel Parabola refinement via cached SPIR-V compute pipeline.
    pub fn run_subpixel_parabola(
        &self,
        disp: &[f32],
        cost_m1: &[f32],
        cost_0: &[f32],
        cost_p1: &[f32],
        width: u32,
        height: u32,
    ) -> Result<Vec<f32>, String> {
        let params: [i32; 2] = [width as i32, height as i32];
        let out_zeros = vec![0.0f32; disp.len()];
        let res = self.dispatch_cached(
            ShaderKind::SubpixelParabola,
            &[
                as_bytes(disp),
                as_bytes(cost_m1),
                as_bytes(cost_0),
                as_bytes(cost_p1),
                as_bytes(&out_zeros),
                as_bytes(&params),
            ],
            &[4],
            (width + 15) / 16,
            (height + 15) / 16,
            1,
        )?;
        Ok(bytes_to_f32(&res[0]))
    }

    /// Run GPU DEM Lambertian hillshade via cached SPIR-V compute pipeline.
    pub fn run_hillshade(
        &self,
        dem: &[f32],
        width: u32,
        height: u32,
        pixel_size: f32,
        sun_dir: [f32; 3],
    ) -> Result<Vec<f32>, String> {
        let pi: [i32; 2] = [width as i32, height as i32];
        let pf: [f32; 4] = [pixel_size, sun_dir[0], sun_dir[1], sun_dir[2]];
        let zeros = vec![0.0f32; dem.len()];
        let res = self.dispatch_cached(
            ShaderKind::DemHillshade,
            &[as_bytes(dem), as_bytes(&zeros), as_bytes(&pi), as_bytes(&pf)],
            &[1],
            (width + 15) / 16,
            (height + 15) / 16,
            1,
        )?;
        Ok(bytes_to_f32(&res[0]))
    }
}

impl Drop for VulkanCompute {
    fn drop(&mut self) {
        unsafe {
            (self.vk.vkQueueWaitIdle)(self.queue);
            let cache = self
                .pipeline_cache
                .lock()
                .unwrap_or_else(|e| e.into_inner());
            for cp in cache.values() {
                (self.vk.vkDestroyPipeline)(self.device, cp.pipeline, ptr::null());
                (self.vk.vkDestroyPipelineLayout)(self.device, cp.pipe_layout, ptr::null());
                (self.vk.vkDestroyDescriptorSetLayout)(self.device, cp.dsl, ptr::null());
                (self.vk.vkDestroyShaderModule)(self.device, cp.shader_module, ptr::null());
            }
            drop(cache);
            (self.vk.vkDestroyCommandPool)(self.device, self.command_pool, ptr::null());
            (self.vk.vkDestroyDevice)(self.device, ptr::null());
            (self.vk.vkDestroyInstance)(self.instance, ptr::null());
        }
    }
}

fn as_bytes<T>(slice: &[T]) -> &[u8] {
    unsafe {
        std::slice::from_raw_parts(slice.as_ptr() as *const u8, std::mem::size_of_val(slice))
    }
}

fn bytes_to_f32(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct CostVolumeSlab {
    core_y0: usize,
    core_y1: usize,
    src_y0: usize,
    src_y1: usize,
}

/// Partition `height` rows into halo-padded Y-slabs whose padded GPU storage buffer fits within `max_slab_bytes`.
fn plan_cost_volume_slabs(
    height: usize,
    row_bytes: usize,
    pad_rows: usize,
    max_slab_bytes: usize,
) -> Vec<CostVolumeSlab> {
    if height == 0 {
        return Vec::new();
    }
    let total_bytes = height.saturating_mul(row_bytes);
    if row_bytes == 0 || total_bytes <= max_slab_bytes {
        return vec![CostVolumeSlab {
            core_y0: 0,
            core_y1: height,
            src_y0: 0,
            src_y1: height,
        }];
    }
    let max_padded_rows = (max_slab_bytes / row_bytes).max(1 + 2 * pad_rows);
    let core_step = max_padded_rows.saturating_sub(2 * pad_rows).max(1);
    let mut slabs = Vec::new();
    let mut y0 = 0usize;
    while y0 < height {
        let y1 = (y0 + core_step).min(height);
        let src_y0 = y0.saturating_sub(pad_rows);
        let src_y1 = (y1 + pad_rows).min(height);
        slabs.push(CostVolumeSlab {
            core_y0: y0,
            core_y1: y1,
            src_y0,
            src_y1,
        });
        y0 = y1;
    }
    slabs
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cost_volume_y_slab_planner_1024x1024x129() {
        // Real Meyreuil 1024x1024x129 = 516 MB -> splits cleanly into <= 240 MB slabs with 4-row halo
        let row_bytes = 1024 * 129 * 4;
        let max_bytes = 240 * 1024 * 1024;
        let slabs = plan_cost_volume_slabs(1024, row_bytes, 4, max_bytes);
        assert_eq!(slabs.len(), 3);
        assert_eq!(slabs[0].core_y0, 0);
        assert_eq!(slabs.last().unwrap().core_y1, 1024);
        for (i, s) in slabs.iter().enumerate() {
            let padded_bytes = (s.src_y1 - s.src_y0) * row_bytes;
            assert!(padded_bytes <= max_bytes, "Slab {} exceeded budget: {} bytes", i, padded_bytes);
            if i > 0 {
                assert_eq!(s.core_y0, slabs[i - 1].core_y1);
                assert_eq!(s.src_y0, s.core_y0 - 4);
            }
            if i + 1 < slabs.len() {
                assert_eq!(s.src_y1, s.core_y1 + 4);
            }
        }
        // Small volumes stay single-dispatch
        let single = plan_cost_volume_slabs(256, 256 * 64 * 4, 4, max_bytes);
        assert_eq!(single.len(), 1);
    }
}
