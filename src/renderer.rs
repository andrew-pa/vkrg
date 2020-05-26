use vulkano::device::{Device, Queue};
use vulkano::swapchain::Swapchain;
use vulkano::framebuffer::*;
use vulkano::format::*;
use vulkano::image::{AttachmentImage, SwapchainImage, ImageLayout, ImageUsage};
use vulkano::sync::GpuFuture;
use vulkano::buffer::*;
use vulkano::pipeline::*;
use vulkano::command_buffer::*;
use vulkano::descriptor::descriptor_set::PersistentDescriptorSet;
use vulkano::descriptor::DescriptorSet;
use std::sync::Arc;
use nalgebra_glm as m;

#[derive(Default, Debug, Clone)]
struct Vertex {
    position: [f32; 3],
    normal: [f32; 3]
}

impl Vertex {
    fn from_tuple<'a, 'b>((pos, nor): (&'a [f32], &'b [f32])) -> Vertex {
        Vertex {
            position: [pos[0], pos[1], pos[2]],
            normal: [nor[0], nor[1], nor[2]]
        }
    }
}

vulkano::impl_vertex!(Vertex, position, normal);

#[derive(Default, Debug, Clone)]
struct Vertex2d {
    position: [f32; 2]
}
vulkano::impl_vertex!(Vertex2d, position);

mod obj_vs {
    vulkano_shaders::shader!{
        ty: "vertex",
        src: "
            #version 450
            layout (location = 0) in vec3 position;
            layout (location = 1) in vec3 normal;

            layout (std430, push_constant) uniform pc {
                mat4 T;
            };

            layout(location = 0) out vec3 out_normal;
            layout(location = 1) out vec3 out_position;

            void main() {
                out_position = position;
                out_normal = normal;
                gl_Position = T*vec4(position, 1.0);
            }
"
    }
}

mod fs_vs {
    vulkano_shaders::shader!{
        ty: "vertex",
        src: "
            #version 450
            layout (location = 0) in vec2 position;
            void main() {
                gl_Position = vec4(position, 0.0, 1.0);
            }
        "
    }
}

mod gbuf_fs {
    vulkano_shaders::shader!{
        ty: "fragment",
        src: "
            #version 450
            layout(location = 0) in vec3 normal;
            layout(location = 1) in vec3 position;
            layout(location = 0) out vec4 pos;
            layout(location = 1) out vec4 nor;

            void main() {
                pos = vec4(position, 0.0);
                nor = vec4(normal, 1.0);
            }
            "
    }
}

mod shade_fs {
    vulkano_shaders::shader!{
        ty: "fragment",
        src: "
            #version 450
            layout(input_attachment_index = 0, set = 0, binding = 0) uniform subpassInput positions;
            layout(input_attachment_index = 1, set = 0, binding = 1) uniform subpassInput normals;

            layout(location = 0) out vec4 out_color;

            void main() {
                vec4 sunor = subpassLoad(normals);
                if(sunor.w < 1.0) discard;
                vec3 N = normalize(sunor.rgb);
                out_color = vec4(vec3(0.1, 0.8, 0.3)*max(0.0, dot(N, normalize(vec3(0.4, 1.0, 0.0)))) + vec3(0.1), 1.0);
            }
        "
    }
}

struct ResDepResources {
    gbuffer_images: Vec<Arc<AttachmentImage>>,
    framebuffers: Vec<Arc<dyn FramebufferAbstract + Send + Sync>>,
    depth_buffer: Arc<AttachmentImage>,
    shade_desc_set: Arc<dyn DescriptorSet + Send + Sync>,
}

pub struct Renderer {
    device: Arc<Device>,
    graphics_qu: Arc<Queue>,
    
    render_pass: Arc<dyn RenderPassAbstract + Send + Sync>,
    obj_pipeline: Arc<dyn GraphicsPipelineAbstract + Send + Sync>,
    shade_pipeline: Arc<dyn GraphicsPipelineAbstract + Send + Sync>,

    vertex_buffer: Arc<CpuAccessibleBuffer<[Vertex]>>,
    index_buffer: Arc<CpuAccessibleBuffer<[u32]>>,
    fullscreen_triangle_vertex_buffer: Arc<CpuAccessibleBuffer<[Vertex2d]>>,

    last_time: std::time::Instant,

    rdr: Option<ResDepResources>
}

impl Renderer {
    pub fn init<S>(device: Arc<Device>, graphics_qu: Arc<Queue>, swapchain: Arc<Swapchain<S>>) -> Renderer {
        let render_pass = Arc::new(vulkano::ordered_passes_renderpass!(
                device.clone(),
                attachments: {
                    color: {
                        load: Clear,
                        store: Store,
                        format: swapchain.format(),
                        samples: 1,
                    },
                    depth: {
                        load: Clear,
                        store: DontCare,
                        format: Format::D16Unorm,
                        samples: 1,
                    },
                    gbuffer_position: {
                        load: Clear,
                        store: Store,
                        format: Format::R32G32B32A32Sfloat,
                        samples: 1,
                    },
                    gbuffer_normals: {
                        load: Clear,
                        store: Store,
                        format: Format::R32G32B32A32Sfloat,
                        samples: 1,
                    }
                },
                passes: [
                    {
                        color: [gbuffer_position, gbuffer_normals],
                        depth_stencil: {depth},
                        input: []
                    },
                    {
                        color: [color],
                        depth_stencil: {},
                        input: [gbuffer_position, gbuffer_normals]
                    }
                ]
            ).unwrap());

        let (model, mats) = tobj::load_obj(&std::path::Path::new("scene.obj"), false).expect("load model");
        let mut object_vertex_offset = Vec::new();
        let mut vertices = Vec::with_capacity(model.iter().map(|model| model.mesh.positions.len() / 3).sum());
        let mut indices: Vec<u32> = Vec::with_capacity(model.iter().map(|model| model.mesh.indices.len()).sum());
        for obj in model.iter() {
            let offset = vertices.len();
            object_vertex_offset.push(offset);
            let mesh = &obj.mesh;
            vertices.extend(mesh.positions.chunks(3).zip(mesh.normals.chunks(3)).map(Vertex::from_tuple));
            indices.extend(mesh.indices.iter().map(|i| i+offset as u32));
        }

        let vertex_buffer = CpuAccessibleBuffer::from_iter(device.clone(), BufferUsage::vertex_buffer(), false, 
                               vertices.iter().cloned()).unwrap();

        let index_buffer = CpuAccessibleBuffer::from_iter(device.clone(), BufferUsage::index_buffer(), false, 
                                indices.iter().cloned()).unwrap();

        let fullscreen_triangle_vertex_buffer = CpuAccessibleBuffer::from_iter(device.clone(), BufferUsage::vertex_buffer(), false, [
                                    Vertex2d { position: [ -1.0, -1.0 ] },
                                    Vertex2d { position: [ -1.0, 3.0 ] },
                                    Vertex2d { position: [ 3.0, -1.0 ] },
                                ].iter().cloned()).unwrap();

        let vsh = obj_vs::Shader::load(device.clone()).unwrap();
        let fsh = gbuf_fs::Shader::load(device.clone()).unwrap();

        let obj_pipeline = Arc::new(GraphicsPipeline::start()
            .vertex_input_single_buffer::<Vertex>()
            .vertex_shader(vsh.main_entry_point(), ())
            .triangle_list()
            .viewports_dynamic_scissors_irrelevant(1)
            .fragment_shader(fsh.main_entry_point(), ())
            .depth_stencil_simple_depth()
            .render_pass(Subpass::from(render_pass.clone(), 0).unwrap())
            .build(device.clone()).unwrap());

        let fsvs = fs_vs::Shader::load(device.clone()).unwrap();
        let shfs = shade_fs::Shader::load(device.clone()).unwrap();

        let shade_pipeline = Arc::new(GraphicsPipeline::start()
            .vertex_input_single_buffer::<Vertex2d>()
            .vertex_shader(fsvs.main_entry_point(), ())
            .triangle_list()
            .viewports_dynamic_scissors_irrelevant(1)
            .fragment_shader(shfs.main_entry_point(), ())
            .render_pass(Subpass::from(render_pass.clone(), 1).unwrap())
            .build(device.clone()).unwrap());

        Renderer {
            device, graphics_qu, render_pass,
            vertex_buffer, index_buffer, fullscreen_triangle_vertex_buffer,
            obj_pipeline, shade_pipeline,
            last_time: std::time::Instant::now(),
            rdr: None
        }
    }

    pub fn window_size_dependent_init<W: Send + Sync + 'static>(&mut self, images: &[Arc<SwapchainImage<W>>]) {
        let depth_buffer = AttachmentImage::transient(self.device.clone(), images[0].dimensions(), Format::D16Unorm).unwrap();
        let atch_usage = ImageUsage {
            transient_attachment: true,
            input_attachment: true,
            .. ImageUsage::none()
        };
        let gbuf_img: Vec<Arc<AttachmentImage>> =
            (0..2).map(|_| AttachmentImage::with_usage(self.device.clone(), images[0].dimensions(),
                                            Format::R32G32B32A32Sfloat, atch_usage).unwrap()).collect();
        let shade_desc_set = Arc::new(PersistentDescriptorSet::start(self.shade_pipeline.descriptor_set_layout(0).unwrap().clone())
            .add_image(gbuf_img[0].clone()).unwrap()
            .add_image(gbuf_img[1].clone()).unwrap()
            .build().unwrap());
        self.rdr = Some(ResDepResources{
            framebuffers: images.iter().map(|image| {
                Arc::new(Framebuffer::start(self.render_pass.clone())
                         .add(image.clone()).unwrap()
                         .add(depth_buffer.clone()).unwrap()
                         .add(gbuf_img[0].clone()).unwrap()
                         .add(gbuf_img[1].clone()).unwrap()
                         .build().expect("create framebuffer")
                        ) as Arc<dyn FramebufferAbstract + Send + Sync>
            }).collect::<Vec<_>>(),
            gbuffer_images: gbuf_img,
            depth_buffer,
            shade_desc_set
        });
    }

    pub fn render(&mut self, future: Box<dyn GpuFuture>, image_num: usize)
        -> Box<dyn GpuFuture>
    {
        let now = (std::time::Instant::now()-self.last_time).as_millis() as f32 / 1000.0;
        let rdr = self.rdr.as_ref().unwrap();
        let framebuffer = rdr.framebuffers[image_num].clone();
            
        let (frame_width, frame_height) = (framebuffer.dimensions()[0] as f32, framebuffer.dimensions()[1] as f32);
        use std::f32::consts::PI;
        let transform =
            m::rotate_z(
                &(m::perspective_zo(frame_width/frame_height, PI/4.0, 0.5, 100.0)
                * m::look_at(&m::vec3(now.cos()*4.0, -2.0, now.sin()*4.0), &m::vec3(0.0, 0.1, 0.0), &m::vec3(0.0, 1.0, 0.0))),
                PI);

        let clear_values = vec!([1.0, 0.5, 0.2, 1.0].into(), 1f32.into(), [0.0,0.0,0.0,0.0].into(), [0.0,0.0,0.0,0.0].into());

        let dynamic_state = DynamicState {
            line_width: None,
            viewports: Some(vec![viewport::Viewport{ 
                origin: [0.0, 0.0],
                dimensions: [frame_width, frame_height],
                depth_range: 0.0 .. 1.0
            }]),
            scissors: None,
            compare_mask: None,
            write_mask: None,
            reference: None
        };


        let cmdbuf = AutoCommandBufferBuilder::primary_one_time_submit(self.device.clone(),
                                                    self.graphics_qu.family()).unwrap()
            .begin_render_pass(framebuffer, false, clear_values).unwrap()
            .draw_indexed(self.obj_pipeline.clone(), &dynamic_state,
                vec![self.vertex_buffer.clone()], self.index_buffer.clone(),
                (), transform).unwrap()
            .next_subpass(false).unwrap()
            .draw(self.shade_pipeline.clone(), &dynamic_state,
                vec![self.fullscreen_triangle_vertex_buffer.clone()], rdr.shade_desc_set.clone(), ()).unwrap()
            .end_render_pass().unwrap()
            .build().unwrap();
        Box::new(future.then_execute(self.graphics_qu.clone(), cmdbuf).unwrap())
 
    }
}
