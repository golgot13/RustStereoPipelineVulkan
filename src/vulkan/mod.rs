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
    SubpixelParabola,
    DemHillshade,
}

impl ShaderKind {
    fn binding_count(self) -> usize {
        match self {
            Self::PrefilterBoxSubmean | Self::PrefilterLog2d => 3,
            Self::DemHillshade => 4,
            Self::StereoBm => 5,
            Self::SubpixelParabola => 6,
        }
    }

    fn build_spirv(self) -> Vec<u32> {
        match self {
            Self::PrefilterBoxSubmean => shaders::build_prefilter_shader(),
            Self::PrefilterLog2d => shaders::build_log_2d_shader(),
            Self::StereoBm => shaders::build_stereo_bm_shader(),
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
        let cached = self.get_or_compile_pipeline(kind)?;
        let _qguard = self.queue_lock.lock().unwrap_or_else(|e| e.into_inner());

        let mut gpu_bufs = Vec::with_capacity(buffers_in.len());
        for data in buffers_in {
            gpu_bufs.push(self.create_buffer_with_data(data)?);
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
