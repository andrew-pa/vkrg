use vulkano::instance::{Instance, PhysicalDevice, QueueFamily };
use vulkano::device::{Device, DeviceExtensions, Features as DeviceFeatures, Queue};
use vulkano::swapchain::{Swapchain, SurfaceTransform, PresentMode, FullscreenExclusive, Surface};
use vulkano::framebuffer::{FramebufferAbstract, Framebuffer};
use vulkano::sync::GpuFuture;
use vulkano::image::SwapchainImage;
use vulkano_win::VkSurfaceBuild;
use winit::window::{WindowBuilder,Window};
use winit::event_loop::EventLoop;
use winit::dpi::*;
use std::sync::Arc;

mod renderer;


/*
 *  ✓ load some sort of mesh data
 *  (✓) draw into Gbuffer [position, normal, next bounce dir, material info] - still needs mat info
 *  - do raytrace on next bounce dir a couple times
 *  - shade everything
 */

struct App {
    instance: Arc<Instance>,
    device: Arc<Device>,
    graphics_qu: Arc<Queue>,
    surface: Arc<Surface<Window>>,
    swapchain: Arc<Swapchain<Window>>,
    recreate_swapchain: bool,
    previous_frame_end: Option<Box<dyn GpuFuture>>,
    renderer: renderer::Renderer
}

impl App {
    fn init(evloop: &EventLoop<()>) -> App {
        let instance = Instance::new(None, &vulkano_win::required_extensions(), None).expect("create instance");

        let physical_device = PhysicalDevice::enumerate(&instance).next().expect("take first physical device");

        let dev_ext = DeviceExtensions {
            khr_swapchain: true,
            
            .. DeviceExtensions::none()
        };

        let (device, mut queues) = Device::new(physical_device, &DeviceFeatures::none(), &dev_ext,
                                               [(physical_device.queue_families().find(QueueFamily::supports_graphics).expect("graphics queue family"), 0.5)].iter().cloned()).expect("create device");

        let graphics_qu = queues.next().unwrap();

        let surface = WindowBuilder::new()
            .with_title(format!("vkrg on {}", physical_device.name()))
            .with_inner_size(LogicalSize::new(1280, 960))
            .build_vk_surface(&evloop, instance.clone())
            .expect("create window");

        let caps = surface.capabilities(physical_device).expect("get surface caps");
        let dims = caps.current_extent.unwrap_or_else(|| [surface.window().inner_size().width, surface.window().inner_size().height]);
        let alpha = caps.supported_composite_alpha.iter().next().unwrap();
        let (format, color_space) = caps.supported_formats[0];
        //println!("selected alpha = {:?}, format = {:?}", alpha, format);

        let (swapchain, swapchain_images) = Swapchain::new(device.clone(), surface.clone(),
        caps.min_image_count, format, dims, 1, caps.supported_usage_flags, &graphics_qu,
        SurfaceTransform::Identity, alpha, PresentMode::Fifo, FullscreenExclusive::Default, true, color_space)
            .expect("create swapchain");

        let mut renderer = renderer::Renderer::init(device.clone(), graphics_qu.clone(), swapchain.clone());
        renderer.window_size_dependent_init(&swapchain_images);
        let mut app = App {
            instance, device: device.clone(), graphics_qu, surface, swapchain,
            recreate_swapchain: false, 
            previous_frame_end: Some(Box::new(vulkano::sync::now(device.clone()))), renderer
        };
        app
    }

    fn present(&mut self) {
        self.previous_frame_end.as_mut().map(|pf| pf.cleanup_finished());

        if self.recreate_swapchain {
            let (new_swapchain, new_images) = match self.swapchain.recreate_with_dimensions(self.surface.window().inner_size().into()) {
                Ok(r) => r,
                Err(vulkano::swapchain::SwapchainCreationError::UnsupportedDimensions) => return,
                Err(e) => panic!("failed to recreate swapchain {:?}", e)
            };

            self.swapchain = new_swapchain;
            self.renderer.window_size_dependent_init(&new_images);
            self.recreate_swapchain = false;
        }

        let (image_num, suboptimal, acquire_future) = match vulkano::swapchain::acquire_next_image(self.swapchain.clone(), None) {
            Ok(r) => r,
            Err(vulkano::swapchain::AcquireError::OutOfDate) => {
                self.recreate_swapchain = true;
                return;
            },
            Err(e) => panic!("failed to acquire next image {:?}", e)
        };

        if suboptimal { self.recreate_swapchain = true; }

        let future = Box::new(self.previous_frame_end.take().unwrap().join(acquire_future));
        let future = self.renderer.render(future, image_num)
            .then_swapchain_present(self.graphics_qu.clone(), self.swapchain.clone(), image_num)
            .then_signal_fence_and_flush();

        self.previous_frame_end = Some(match future {
            Ok(f) => Box::new(f),
            Err(vulkano::sync::FlushError::OutOfDate) => {
                self.recreate_swapchain = true;
                Box::new(vulkano::sync::now(self.device.clone()))
            },
            Err(e) => {
                println!("failed to present {:?}", e);
                Box::new(vulkano::sync::now(self.device.clone()))
            }
        });
    }
}

fn main() {
    let evloop = EventLoop::new();
    let mut app = App::init(&evloop);
    evloop.run(move |event, _, ctrl_flow| {
        use winit::event::*;
        use winit::event_loop::*;
        match event {
            Event::WindowEvent { event: WindowEvent::CloseRequested, .. } => {
                *ctrl_flow = ControlFlow::Exit;
            }
            Event::RedrawEventsCleared => {
                app.present();
            }
            _ => ()
        }
    });
}
