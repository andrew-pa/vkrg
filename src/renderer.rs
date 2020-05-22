use vulkano::instance::{Instance, InstanceExtensions, PhysicalDevice, QueueFamily };
use vulkano::device::{Device, DeviceExtensions, Features as DeviceFeatures, Queue};
use vulkano::swapchain::{Swapchain, SurfaceTransform, PresentMode, ColorSpace, FullscreenExclusive, Surface};
use vulkano::framebuffer::{RenderPassAbstract, FramebufferAbstract, Framebuffer};
use vulkano::sync::{GpuFuture};
use vulkano::image::{SwapchainImage};
use std::sync::Arc;

pub struct Renderer {
    device: Arc<Device>,
    graphics_qu: Arc<Queue>,
    render_pass: Arc<dyn RenderPassAbstract + Send + Sync>,
}

impl Renderer {
    pub fn init<S>(device: Arc<Device>, graphics_qu: Arc<Queue>, swapchain: Arc<Swapchain<S>>) -> Renderer {
        let render_pass = Arc::new(vulkano::single_pass_renderpass!(
                device.clone(),
                attachments: {
                    color: {
                        load: Clear,
                        store: Store,
                        format: swapchain.format(),
                        samples: 1,
                    }
                },
                pass: {
                    color: [color],
                    depth_stencil: {}
                }
            ).unwrap());


        Renderer { device, graphics_qu, render_pass }
    }

    pub fn window_size_dependent_init(&mut self) {}
    
    pub fn main_render_pass(&self) -> Arc<dyn RenderPassAbstract + Send + Sync> {
        self.render_pass.clone()
    }

    pub fn render(&mut self, future: Box<dyn GpuFuture>, framebuffer: Arc<dyn FramebufferAbstract + Send + Sync>)
        -> Box<dyn GpuFuture>
    {
        use vulkano::command_buffer::*;
        let clear_values = vec!([1.0, 0.5, 0.0, 1.0].into());
        let cmdbuf = AutoCommandBufferBuilder::primary_one_time_submit(self.device.clone(), self.graphics_qu.family()).unwrap()
            .begin_render_pass(framebuffer, false, clear_values).unwrap()
            .end_render_pass().unwrap()
            .build().unwrap();
        Box::new(future.then_execute(self.graphics_qu.clone(), cmdbuf).unwrap())
 
    }
}
