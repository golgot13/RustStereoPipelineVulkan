//! Raw Vulkan 1.0 C ABI bindings and zero-dependency dynamic library loader (`dlopen`/`dlsym`).
//! Zero external crates required — compiles on any Linux/macOS/Windows system even without Vulkan SDK headers.

#![allow(non_camel_case_types, non_snake_case, dead_code)]

use std::ffi::{c_char, c_void, CStr};
use std::ptr;

pub type VkBool32 = u32;
pub type VkFlags = u32;
pub type VkDeviceSize = u64;
pub type VkResult = i32;

pub type VkInstance = *mut c_void;
pub type VkPhysicalDevice = *mut c_void;
pub type VkDevice = *mut c_void;
pub type VkQueue = *mut c_void;
pub type VkCommandBuffer = *mut c_void;

// Non-dispatchable handles are always u64 on 64-bit and 32-bit platforms per Vulkan spec.
pub type VkBuffer = u64;
pub type VkDeviceMemory = u64;
pub type VkShaderModule = u64;
pub type VkDescriptorSetLayout = u64;
pub type VkPipelineLayout = u64;
pub type VkPipeline = u64;
pub type VkPipelineCache = u64;
pub type VkDescriptorPool = u64;
pub type VkDescriptorSet = u64;
pub type VkCommandPool = u64;
pub type VkFence = u64;
pub type VkSemaphore = u64;

pub const VK_SUCCESS: VkResult = 0;
pub const VK_TRUE: VkBool32 = 1;
pub const VK_FALSE: VkBool32 = 0;
pub const VK_WHOLE_SIZE: VkDeviceSize = u64::MAX;

pub const VK_STRUCTURE_TYPE_APPLICATION_INFO: u32 = 0;
pub const VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO: u32 = 1;
pub const VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO: u32 = 2;
pub const VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO: u32 = 3;
pub const VK_STRUCTURE_TYPE_SUBMIT_INFO: u32 = 4;
pub const VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO: u32 = 5;
pub const VK_STRUCTURE_TYPE_FENCE_CREATE_INFO: u32 = 8;
pub const VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO: u32 = 12;
pub const VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO: u32 = 16;
pub const VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO: u32 = 18;
pub const VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO: u32 = 29;
pub const VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO: u32 = 30;
pub const VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO: u32 = 32;
pub const VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO: u32 = 33;
pub const VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO: u32 = 34;
pub const VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET: u32 = 35;
pub const VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO: u32 = 39;
pub const VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO: u32 = 40;
pub const VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO: u32 = 42;
pub const VK_STRUCTURE_TYPE_MEMORY_BARRIER: u32 = 46;

pub const VK_QUEUE_COMPUTE_BIT: VkFlags = 0x0000_0002;
pub const VK_BUFFER_USAGE_STORAGE_BUFFER_BIT: VkFlags = 0x0000_0020;
pub const VK_SHARING_MODE_EXCLUSIVE: u32 = 0;
pub const VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT: VkFlags = 0x0000_0002;
pub const VK_MEMORY_PROPERTY_HOST_COHERENT_BIT: VkFlags = 0x0000_0004;
pub const VK_SHADER_STAGE_COMPUTE_BIT: VkFlags = 0x0000_0020;
pub const VK_DESCRIPTOR_TYPE_STORAGE_BUFFER: u32 = 7;
pub const VK_PIPELINE_BIND_POINT_COMPUTE: u32 = 1;
pub const VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT: VkFlags = 0x0000_0002;
pub const VK_COMMAND_BUFFER_LEVEL_PRIMARY: u32 = 0;
pub const VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT: VkFlags = 0x0000_0001;

pub const VK_PHYSICAL_DEVICE_TYPE_OTHER: u32 = 0;
pub const VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU: u32 = 1;
pub const VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU: u32 = 2;
pub const VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU: u32 = 3;
pub const VK_PHYSICAL_DEVICE_TYPE_CPU: u32 = 4;

pub const fn vk_make_version(major: u32, minor: u32, patch: u32) -> u32 {
    (major << 22) | (minor << 12) | patch
}

#[repr(C)]
pub struct VkApplicationInfo {
    pub sType: u32,
    pub pNext: *const c_void,
    pub pApplicationName: *const c_char,
    pub applicationVersion: u32,
    pub pEngineName: *const c_char,
    pub engineVersion: u32,
    pub apiVersion: u32,
}

#[repr(C)]
pub struct VkInstanceCreateInfo {
    pub sType: u32,
    pub pNext: *const c_void,
    pub flags: VkFlags,
    pub pApplicationInfo: *const VkApplicationInfo,
    pub enabledLayerCount: u32,
    pub ppEnabledLayerNames: *const *const c_char,
    pub enabledExtensionCount: u32,
    pub ppEnabledExtensionNames: *const *const c_char,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct VkPhysicalDeviceLimits {
    pub raw: [u32; 126], // Exact padded layout placeholder (504 bytes)
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct VkPhysicalDeviceSparseProperties {
    pub residencyStandard2DBlockShape: VkBool32,
    pub residencyStandard2DMultisampleBlockShape: VkBool32,
    pub residencyStandard3DBlockShape: VkBool32,
    pub residencyAlignedMipSize: VkBool32,
    pub residencyNonResidentStrict: VkBool32,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct VkPhysicalDeviceProperties {
    pub apiVersion: u32,
    pub driverVersion: u32,
    pub vendorID: u32,
    pub deviceID: u32,
    pub deviceType: u32,
    pub deviceName: [c_char; 256],
    pub pipelineCacheUUID: [u8; 16],
    pub limits: VkPhysicalDeviceLimits,
    pub sparseProperties: VkPhysicalDeviceSparseProperties,
}

impl Default for VkPhysicalDeviceProperties {
    fn default() -> Self {
        unsafe { std::mem::zeroed() }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct VkExtent3D {
    pub width: u32,
    pub height: u32,
    pub depth: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct VkQueueFamilyProperties {
    pub queueFlags: VkFlags,
    pub queueCount: u32,
    pub timestampValidBits: u32,
    pub minImageTransferGranularity: VkExtent3D,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct VkMemoryType {
    pub propertyFlags: VkFlags,
    pub heapIndex: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct VkMemoryHeap {
    pub size: VkDeviceSize,
    pub flags: VkFlags,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct VkPhysicalDeviceMemoryProperties {
    pub memoryTypeCount: u32,
    pub memoryTypes: [VkMemoryType; 32],
    pub memoryHeapCount: u32,
    pub memoryHeaps: [VkMemoryHeap; 16],
}

impl Default for VkPhysicalDeviceMemoryProperties {
    fn default() -> Self {
        unsafe { std::mem::zeroed() }
    }
}

#[repr(C)]
pub struct VkDeviceQueueCreateInfo {
    pub sType: u32,
    pub pNext: *const c_void,
    pub flags: VkFlags,
    pub queueFamilyIndex: u32,
    pub queueCount: u32,
    pub pQueuePriorities: *const f32,
}

#[repr(C)]
pub struct VkDeviceCreateInfo {
    pub sType: u32,
    pub pNext: *const c_void,
    pub flags: VkFlags,
    pub queueCreateInfoCount: u32,
    pub pQueueCreateInfos: *const VkDeviceQueueCreateInfo,
    pub enabledLayerCount: u32,
    pub ppEnabledLayerNames: *const *const c_char,
    pub enabledExtensionCount: u32,
    pub ppEnabledExtensionNames: *const *const c_char,
    pub pEnabledFeatures: *const c_void,
}

#[repr(C)]
pub struct VkBufferCreateInfo {
    pub sType: u32,
    pub pNext: *const c_void,
    pub flags: VkFlags,
    pub size: VkDeviceSize,
    pub usage: VkFlags,
    pub sharingMode: u32,
    pub queueFamilyIndexCount: u32,
    pub pQueueFamilyIndices: *const u32,
}

#[repr(C)]
#[derive(Default)]
pub struct VkMemoryRequirements {
    pub size: VkDeviceSize,
    pub alignment: VkDeviceSize,
    pub memoryTypeBits: u32,
}

#[repr(C)]
pub struct VkMemoryAllocateInfo {
    pub sType: u32,
    pub pNext: *const c_void,
    pub allocationSize: VkDeviceSize,
    pub memoryTypeIndex: u32,
}

#[repr(C)]
pub struct VkShaderModuleCreateInfo {
    pub sType: u32,
    pub pNext: *const c_void,
    pub flags: VkFlags,
    pub codeSize: usize,
    pub pCode: *const u32,
}

#[repr(C)]
pub struct VkDescriptorSetLayoutBinding {
    pub binding: u32,
    pub descriptorType: u32,
    pub descriptorCount: u32,
    pub stageFlags: VkFlags,
    pub pImmutableSamplers: *const u64,
}

#[repr(C)]
pub struct VkDescriptorSetLayoutCreateInfo {
    pub sType: u32,
    pub pNext: *const c_void,
    pub flags: VkFlags,
    pub bindingCount: u32,
    pub pBindings: *const VkDescriptorSetLayoutBinding,
}

#[repr(C)]
pub struct VkPipelineLayoutCreateInfo {
    pub sType: u32,
    pub pNext: *const c_void,
    pub flags: VkFlags,
    pub setLayoutCount: u32,
    pub pSetLayouts: *const VkDescriptorSetLayout,
    pub pushConstantRangeCount: u32,
    pub pPushConstantRanges: *const c_void,
}

#[repr(C)]
pub struct VkPipelineShaderStageCreateInfo {
    pub sType: u32,
    pub pNext: *const c_void,
    pub flags: VkFlags,
    pub stage: VkFlags,
    pub module: VkShaderModule,
    pub pName: *const c_char,
    pub pSpecializationInfo: *const c_void,
}

#[repr(C)]
pub struct VkComputePipelineCreateInfo {
    pub sType: u32,
    pub pNext: *const c_void,
    pub flags: VkFlags,
    pub stage: VkPipelineShaderStageCreateInfo,
    pub layout: VkPipelineLayout,
    pub basePipelineHandle: VkPipeline,
    pub basePipelineIndex: i32,
}

#[repr(C)]
pub struct VkDescriptorPoolSize {
    pub descriptorType: u32,
    pub descriptorCount: u32,
}

#[repr(C)]
pub struct VkDescriptorPoolCreateInfo {
    pub sType: u32,
    pub pNext: *const c_void,
    pub flags: VkFlags,
    pub maxSets: u32,
    pub poolSizeCount: u32,
    pub pPoolSizes: *const VkDescriptorPoolSize,
}

#[repr(C)]
pub struct VkDescriptorSetAllocateInfo {
    pub sType: u32,
    pub pNext: *const c_void,
    pub descriptorPool: VkDescriptorPool,
    pub descriptorSetCount: u32,
    pub pSetLayouts: *const VkDescriptorSetLayout,
}

#[repr(C)]
pub struct VkDescriptorBufferInfo {
    pub buffer: VkBuffer,
    pub offset: VkDeviceSize,
    pub range: VkDeviceSize,
}

#[repr(C)]
pub struct VkWriteDescriptorSet {
    pub sType: u32,
    pub pNext: *const c_void,
    pub dstSet: VkDescriptorSet,
    pub dstBinding: u32,
    pub dstArrayElement: u32,
    pub descriptorCount: u32,
    pub descriptorType: u32,
    pub pImageInfo: *const c_void,
    pub pBufferInfo: *const VkDescriptorBufferInfo,
    pub pTexelBufferView: *const u64,
}

#[repr(C)]
pub struct VkCommandPoolCreateInfo {
    pub sType: u32,
    pub pNext: *const c_void,
    pub flags: VkFlags,
    pub queueFamilyIndex: u32,
}

#[repr(C)]
pub struct VkCommandBufferAllocateInfo {
    pub sType: u32,
    pub pNext: *const c_void,
    pub commandPool: VkCommandPool,
    pub level: u32,
    pub commandBufferCount: u32,
}

#[repr(C)]
pub struct VkCommandBufferBeginInfo {
    pub sType: u32,
    pub pNext: *const c_void,
    pub flags: VkFlags,
    pub pInheritanceInfo: *const c_void,
}

#[repr(C)]
pub struct VkSubmitInfo {
    pub sType: u32,
    pub pNext: *const c_void,
    pub waitSemaphoreCount: u32,
    pub pWaitSemaphores: *const VkSemaphore,
    pub pWaitDstStageMask: *const VkFlags,
    pub commandBufferCount: u32,
    pub pCommandBuffers: *const VkCommandBuffer,
    pub signalSemaphoreCount: u32,
    pub pSignalSemaphores: *const VkSemaphore,
}

#[repr(C)]
pub struct VkFenceCreateInfo {
    pub sType: u32,
    pub pNext: *const c_void,
    pub flags: VkFlags,
}

// Dynamic library loading FFI (Unix dlopen/dlsym & Windows LoadLibraryA/GetProcAddress)
#[cfg(unix)]
extern "C" {
    fn dlopen(filename: *const c_char, flag: i32) -> *mut c_void;
    fn dlsym(handle: *mut c_void, symbol: *const c_char) -> *mut c_void;
    fn dlclose(handle: *mut c_void) -> i32;
}

#[cfg(unix)]
const RTLD_NOW: i32 = 2;

#[cfg(windows)]
#[link(name = "kernel32")]
extern "system" {
    fn LoadLibraryA(lpLibFileName: *const c_char) -> *mut c_void;
    fn GetProcAddress(hModule: *mut c_void, lpProcName: *const c_char) -> *mut c_void;
    fn FreeLibrary(hLibModule: *mut c_void) -> i32;
}

pub struct DynamicLib {
    handle: *mut c_void,
}

unsafe impl Send for DynamicLib {}
unsafe impl Sync for DynamicLib {}

impl DynamicLib {
    #[cfg(unix)]
    pub fn open(candidates: &[&[u8]]) -> Option<Self> {
        for name in candidates {
            let h = unsafe { dlopen(name.as_ptr() as *const c_char, RTLD_NOW) };
            if !h.is_null() {
                return Some(Self { handle: h });
            }
        }
        None
    }

    #[cfg(windows)]
    pub fn open(candidates: &[&[u8]]) -> Option<Self> {
        for name in candidates {
            let h = unsafe { LoadLibraryA(name.as_ptr() as *const c_char) };
            if !h.is_null() {
                return Some(Self { handle: h });
            }
        }
        None
    }

    #[cfg(not(any(unix, windows)))]
    pub fn open(_candidates: &[&[u8]]) -> Option<Self> {
        None
    }

    #[cfg(unix)]
    pub unsafe fn sym<T: Copy>(&self, name: &[u8]) -> Option<T> {
        let p = dlsym(self.handle, name.as_ptr() as *const c_char);
        if p.is_null() {
            None
        } else {
            Some(std::mem::transmute_copy(&p))
        }
    }

    #[cfg(windows)]
    pub unsafe fn sym<T: Copy>(&self, name: &[u8]) -> Option<T> {
        let p = GetProcAddress(self.handle, name.as_ptr() as *const c_char);
        if p.is_null() {
            None
        } else {
            Some(std::mem::transmute_copy(&p))
        }
    }

    #[cfg(not(any(unix, windows)))]
    pub unsafe fn sym<T: Copy>(&self, _name: &[u8]) -> Option<T> {
        None
    }
}

impl Drop for DynamicLib {
    fn drop(&mut self) {
        if !self.handle.is_null() {
            #[cfg(unix)]
            unsafe {
                dlclose(self.handle);
            }
            #[cfg(windows)]
            unsafe {
                FreeLibrary(self.handle);
            }
        }
    }
}

/// Function pointer table loaded from `vulkan-1.dll` / `libvulkan.so.1` / `libvulkan.1.dylib`.
pub struct VkFunctions {
    pub _lib: DynamicLib,
    pub vkCreateInstance: unsafe extern "system" fn(*const VkInstanceCreateInfo, *const c_void, *mut VkInstance) -> VkResult,
    pub vkDestroyInstance: unsafe extern "system" fn(VkInstance, *const c_void),
    pub vkEnumeratePhysicalDevices: unsafe extern "system" fn(VkInstance, *mut u32, *mut VkPhysicalDevice) -> VkResult,
    pub vkGetPhysicalDeviceProperties: unsafe extern "system" fn(VkPhysicalDevice, *mut VkPhysicalDeviceProperties),
    pub vkGetPhysicalDeviceQueueFamilyProperties: unsafe extern "system" fn(VkPhysicalDevice, *mut u32, *mut VkQueueFamilyProperties),
    pub vkGetPhysicalDeviceMemoryProperties: unsafe extern "system" fn(VkPhysicalDevice, *mut VkPhysicalDeviceMemoryProperties),
    pub vkCreateDevice: unsafe extern "system" fn(VkPhysicalDevice, *const VkDeviceCreateInfo, *const c_void, *mut VkDevice) -> VkResult,
    pub vkDestroyDevice: unsafe extern "system" fn(VkDevice, *const c_void),
    pub vkGetDeviceQueue: unsafe extern "system" fn(VkDevice, u32, u32, *mut VkQueue),
    pub vkCreateBuffer: unsafe extern "system" fn(VkDevice, *const VkBufferCreateInfo, *const c_void, *mut VkBuffer) -> VkResult,
    pub vkDestroyBuffer: unsafe extern "system" fn(VkDevice, VkBuffer, *const c_void),
    pub vkGetBufferMemoryRequirements: unsafe extern "system" fn(VkDevice, VkBuffer, *mut VkMemoryRequirements),
    pub vkAllocateMemory: unsafe extern "system" fn(VkDevice, *const VkMemoryAllocateInfo, *const c_void, *mut VkDeviceMemory) -> VkResult,
    pub vkFreeMemory: unsafe extern "system" fn(VkDevice, VkDeviceMemory, *const c_void),
    pub vkBindBufferMemory: unsafe extern "system" fn(VkDevice, VkBuffer, VkDeviceMemory, VkDeviceSize) -> VkResult,
    pub vkMapMemory: unsafe extern "system" fn(VkDevice, VkDeviceMemory, VkDeviceSize, VkDeviceSize, VkFlags, *mut *mut c_void) -> VkResult,
    pub vkUnmapMemory: unsafe extern "system" fn(VkDevice, VkDeviceMemory),
    pub vkCreateShaderModule: unsafe extern "system" fn(VkDevice, *const VkShaderModuleCreateInfo, *const c_void, *mut VkShaderModule) -> VkResult,
    pub vkDestroyShaderModule: unsafe extern "system" fn(VkDevice, VkShaderModule, *const c_void),
    pub vkCreateDescriptorSetLayout: unsafe extern "system" fn(VkDevice, *const VkDescriptorSetLayoutCreateInfo, *const c_void, *mut VkDescriptorSetLayout) -> VkResult,
    pub vkDestroyDescriptorSetLayout: unsafe extern "system" fn(VkDevice, VkDescriptorSetLayout, *const c_void),
    pub vkCreatePipelineLayout: unsafe extern "system" fn(VkDevice, *const VkPipelineLayoutCreateInfo, *const c_void, *mut VkPipelineLayout) -> VkResult,
    pub vkDestroyPipelineLayout: unsafe extern "system" fn(VkDevice, VkPipelineLayout, *const c_void),
    pub vkCreateComputePipelines: unsafe extern "system" fn(VkDevice, VkPipelineCache, u32, *const VkComputePipelineCreateInfo, *const c_void, *mut VkPipeline) -> VkResult,
    pub vkDestroyPipeline: unsafe extern "system" fn(VkDevice, VkPipeline, *const c_void),
    pub vkCreateDescriptorPool: unsafe extern "system" fn(VkDevice, *const VkDescriptorPoolCreateInfo, *const c_void, *mut VkDescriptorPool) -> VkResult,
    pub vkDestroyDescriptorPool: unsafe extern "system" fn(VkDevice, VkDescriptorPool, *const c_void),
    pub vkAllocateDescriptorSets: unsafe extern "system" fn(VkDevice, *const VkDescriptorSetAllocateInfo, *mut VkDescriptorSet) -> VkResult,
    pub vkUpdateDescriptorSets: unsafe extern "system" fn(VkDevice, u32, *const VkWriteDescriptorSet, u32, *const c_void),
    pub vkCreateCommandPool: unsafe extern "system" fn(VkDevice, *const VkCommandPoolCreateInfo, *const c_void, *mut VkCommandPool) -> VkResult,
    pub vkDestroyCommandPool: unsafe extern "system" fn(VkDevice, VkCommandPool, *const c_void),
    pub vkAllocateCommandBuffers: unsafe extern "system" fn(VkDevice, *const VkCommandBufferAllocateInfo, *mut VkCommandBuffer) -> VkResult,
    pub vkBeginCommandBuffer: unsafe extern "system" fn(VkCommandBuffer, *const VkCommandBufferBeginInfo) -> VkResult,
    pub vkEndCommandBuffer: unsafe extern "system" fn(VkCommandBuffer) -> VkResult,
    pub vkCmdBindPipeline: unsafe extern "system" fn(VkCommandBuffer, u32, VkPipeline),
    pub vkCmdBindDescriptorSets: unsafe extern "system" fn(VkCommandBuffer, u32, VkPipelineLayout, u32, u32, *const VkDescriptorSet, u32, *const u32),
    pub vkCmdDispatch: unsafe extern "system" fn(VkCommandBuffer, u32, u32, u32),
    pub vkCreateFence: unsafe extern "system" fn(VkDevice, *const VkFenceCreateInfo, *const c_void, *mut VkFence) -> VkResult,
    pub vkDestroyFence: unsafe extern "system" fn(VkDevice, VkFence, *const c_void),
    pub vkWaitForFences: unsafe extern "system" fn(VkDevice, u32, *const VkFence, VkBool32, u64) -> VkResult,
    pub vkResetFences: unsafe extern "system" fn(VkDevice, u32, *const VkFence) -> VkResult,
    pub vkQueueSubmit: unsafe extern "system" fn(VkQueue, u32, *const VkSubmitInfo, VkFence) -> VkResult,
    pub vkQueueWaitIdle: unsafe extern "system" fn(VkQueue) -> VkResult,
}

impl VkFunctions {
    pub fn load() -> Result<Self, String> {
        #[cfg(windows)]
        let candidates: &[&[u8]] = &[b"vulkan-1.dll\0"];
        #[cfg(not(windows))]
        let candidates: &[&[u8]] = &[
            b"libvulkan.so.1\0",
            b"libvulkan.so\0",
            b"libvulkan.1.dylib\0",
            b"libMoltenVK.dylib\0",
        ];
        let lib = DynamicLib::open(candidates)
            .ok_or_else(|| "Vulkan loader library (vulkan-1.dll / libvulkan.so.1) not found".to_string())?;

        macro_rules! load_sym {
            ($name:literal) => {
                unsafe { lib.sym($name) }
                    .ok_or_else(|| format!("Missing Vulkan symbol: {}", String::from_utf8_lossy($name)))?
            };
        }

        Ok(Self {
            vkCreateInstance: load_sym!(b"vkCreateInstance\0"),
            vkDestroyInstance: load_sym!(b"vkDestroyInstance\0"),
            vkEnumeratePhysicalDevices: load_sym!(b"vkEnumeratePhysicalDevices\0"),
            vkGetPhysicalDeviceProperties: load_sym!(b"vkGetPhysicalDeviceProperties\0"),
            vkGetPhysicalDeviceQueueFamilyProperties: load_sym!(b"vkGetPhysicalDeviceQueueFamilyProperties\0"),
            vkGetPhysicalDeviceMemoryProperties: load_sym!(b"vkGetPhysicalDeviceMemoryProperties\0"),
            vkCreateDevice: load_sym!(b"vkCreateDevice\0"),
            vkDestroyDevice: load_sym!(b"vkDestroyDevice\0"),
            vkGetDeviceQueue: load_sym!(b"vkGetDeviceQueue\0"),
            vkCreateBuffer: load_sym!(b"vkCreateBuffer\0"),
            vkDestroyBuffer: load_sym!(b"vkDestroyBuffer\0"),
            vkGetBufferMemoryRequirements: load_sym!(b"vkGetBufferMemoryRequirements\0"),
            vkAllocateMemory: load_sym!(b"vkAllocateMemory\0"),
            vkFreeMemory: load_sym!(b"vkFreeMemory\0"),
            vkBindBufferMemory: load_sym!(b"vkBindBufferMemory\0"),
            vkMapMemory: load_sym!(b"vkMapMemory\0"),
            vkUnmapMemory: load_sym!(b"vkUnmapMemory\0"),
            vkCreateShaderModule: load_sym!(b"vkCreateShaderModule\0"),
            vkDestroyShaderModule: load_sym!(b"vkDestroyShaderModule\0"),
            vkCreateDescriptorSetLayout: load_sym!(b"vkCreateDescriptorSetLayout\0"),
            vkDestroyDescriptorSetLayout: load_sym!(b"vkDestroyDescriptorSetLayout\0"),
            vkCreatePipelineLayout: load_sym!(b"vkCreatePipelineLayout\0"),
            vkDestroyPipelineLayout: load_sym!(b"vkDestroyPipelineLayout\0"),
            vkCreateComputePipelines: load_sym!(b"vkCreateComputePipelines\0"),
            vkDestroyPipeline: load_sym!(b"vkDestroyPipeline\0"),
            vkCreateDescriptorPool: load_sym!(b"vkCreateDescriptorPool\0"),
            vkDestroyDescriptorPool: load_sym!(b"vkDestroyDescriptorPool\0"),
            vkAllocateDescriptorSets: load_sym!(b"vkAllocateDescriptorSets\0"),
            vkUpdateDescriptorSets: load_sym!(b"vkUpdateDescriptorSets\0"),
            vkCreateCommandPool: load_sym!(b"vkCreateCommandPool\0"),
            vkDestroyCommandPool: load_sym!(b"vkDestroyCommandPool\0"),
            vkAllocateCommandBuffers: load_sym!(b"vkAllocateCommandBuffers\0"),
            vkBeginCommandBuffer: load_sym!(b"vkBeginCommandBuffer\0"),
            vkEndCommandBuffer: load_sym!(b"vkEndCommandBuffer\0"),
            vkCmdBindPipeline: load_sym!(b"vkCmdBindPipeline\0"),
            vkCmdBindDescriptorSets: load_sym!(b"vkCmdBindDescriptorSets\0"),
            vkCmdDispatch: load_sym!(b"vkCmdDispatch\0"),
            vkCreateFence: load_sym!(b"vkCreateFence\0"),
            vkDestroyFence: load_sym!(b"vkDestroyFence\0"),
            vkWaitForFences: load_sym!(b"vkWaitForFences\0"),
            vkResetFences: load_sym!(b"vkResetFences\0"),
            vkQueueSubmit: load_sym!(b"vkQueueSubmit\0"),
            vkQueueWaitIdle: load_sym!(b"vkQueueWaitIdle\0"),
            _lib: lib,
        })
    }
}

pub fn c_char_array_to_string(arr: &[c_char]) -> String {
    unsafe {
        CStr::from_ptr(arr.as_ptr())
            .to_string_lossy()
            .into_owned()
    }
}

pub fn null_ptr<T>() -> *const T {
    ptr::null()
}
