//! GPU 设备初始化与查询（M1）。
//!
//! 强制使用 Vulkan backend，确保 PoC 跑在 AMD Radeon RX 5700 XT 上，
//! 而不是被 wgpu 静默回退到别的后端。

use wgpu::Adapter;

/// 请求高性能 Vulkan adapter；找不到时 panic 并给出明确提示。
///
/// M1 验收点：调用方打印 `adapter.get_info()` 应显示
/// `backend = VULKAN`、`name = AMD Radeon RX 5700 XT`。
pub async fn request_vulkan_adapter() -> Adapter {
    // wgpu 30：InstanceDescriptor 无 Default，用 new_without_display_handle() 构造。
    let mut desc = wgpu::InstanceDescriptor::new_without_display_handle();
    desc.backends = wgpu::Backends::VULKAN;
    let instance = wgpu::Instance::new(desc);

    instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
            apply_limit_buckets: false,
        })
        .await
        .expect(
            "未找到 Vulkan adapter —— 请确认 AMD 驱动正常、vulkaninfo 能列出 GPU，\
             且环境变量没有屏蔽 Vulkan 设备",
        )
}
