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
use bvh::aabb::*;
use bvh::bounding_hierarchy::*;

#[derive(Clone)]
struct Triangle {
    corner: bvh::nalgebra::Point3<f32>,
    material_index: u32,
    edge1: bvh::nalgebra::Vector3<f32>,
    node_index: u32,
    edge2: bvh::nalgebra::Vector3<f32>,
    _padding: u32
}

impl Bounded for Triangle {
    fn aabb(&self) -> AABB {
        AABB::empty()
            .grow(&self.corner)
            .grow(&(self.corner + self.edge1))
            .grow(&(self.corner + self.edge2))
    }
}

impl BHShape for Triangle {
    fn set_bh_node_index(&mut self, index: usize) {
        self.node_index = index as u32;
    }

    fn bh_node_index(&self) -> usize {
        self.node_index as usize
    }
}

#[derive(Clone)]
struct BVHNode {
    aabb_min: [f32; 3],
    entry_index: u32,
    aabb_max: [f32; 3],
    exit_index: u32,
    shape_index: u32,
    _padding: [u32; 3]
}

impl BVHNode {
    fn new(aabb: &AABB, entry_index: u32, exit_index: u32, shape_index: u32) -> BVHNode {
        BVHNode {
            aabb_min: [aabb.min.x, aabb.min.y, aabb.min.z],
            aabb_max: [aabb.max.x, aabb.max.y, aabb.max.z],
            entry_index,
            exit_index,
            shape_index,
            _padding: [0xaaaaffff, 0xbbbbffff, 0xccccffff]
        }
    }
}

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

impl Triangle {
    fn new(a: &Vertex, b: &Vertex, c: &Vertex) -> Triangle {
        let ap = bvh::nalgebra::Point3::from_slice(&a.position);
        let bp = bvh::nalgebra::Point3::from_slice(&b.position);
        let cp = bvh::nalgebra::Point3::from_slice(&c.position);
        Triangle {
            corner: ap,
            edge1: bp - ap,
            edge2: cp - ap,
            material_index: 0,
            node_index: 0,
            _padding: 0xaaaaffff
        }
    }
}

#[derive(Default, Debug, Clone)]
struct Vertex2d {
    position: [f32; 2]
}
vulkano::impl_vertex!(Vertex2d, position);

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

    triangle_buffer: Arc<CpuAccessibleBuffer<[Triangle]>>,
    bvh_buffer: Arc<CpuAccessibleBuffer<[BVHNode]>>,

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

        let mut triangles: Vec<Triangle> = indices.chunks(3)
            .map(|i| Triangle::new(&vertices[i[0] as usize],
                                   &vertices[i[1] as usize],
                                   &vertices[i[2] as usize])).collect();
        let bvh = bvh::bvh::BVH::build(&mut triangles);
        let mut bvh_nodes = bvh.flatten_custom(&BVHNode::new);
        bvh_nodes[0]._padding[0] = bvh_nodes.len() as u32;

        let vertex_buffer = CpuAccessibleBuffer::from_iter(device.clone(), BufferUsage::vertex_buffer(), false, 
                               vertices.iter().cloned()).unwrap();

        let index_buffer = CpuAccessibleBuffer::from_iter(device.clone(), BufferUsage::index_buffer(), false, 
                                indices.iter().cloned()).unwrap();

        let fullscreen_triangle_vertex_buffer = CpuAccessibleBuffer::from_iter(device.clone(), BufferUsage::vertex_buffer(), false, [
                                    Vertex2d { position: [ -1.0, -1.0 ] },
                                    Vertex2d { position: [ -1.0, 3.0 ] },
                                    Vertex2d { position: [ 3.0, -1.0 ] },
                                ].iter().cloned()).unwrap();

        let triangle_buffer = CpuAccessibleBuffer::from_iter(device.clone(), BufferUsage { storage_buffer: true, ..BufferUsage::none() }, false, triangles.iter().cloned()).unwrap();
        let bvh_buffer = CpuAccessibleBuffer::from_iter(device.clone(), BufferUsage { storage_buffer: true, ..BufferUsage::none() }, false, bvh_nodes.iter().cloned()).unwrap();

        use super::shaders;
        let vsh = shaders::obj_vs::Shader::load(device.clone()).unwrap();
        let fsh = shaders::gbuf_fs::Shader::load(device.clone()).unwrap();

        let obj_pipeline = Arc::new(GraphicsPipeline::start()
            .vertex_input_single_buffer::<Vertex>()
            .vertex_shader(vsh.main_entry_point(), ())
            .triangle_list()
            .viewports_dynamic_scissors_irrelevant(1)
            .fragment_shader(fsh.main_entry_point(), ())
            .depth_stencil_simple_depth()
            .render_pass(Subpass::from(render_pass.clone(), 0).unwrap())
            .build(device.clone()).unwrap());

        let fsvs = shaders::fs_vs::Shader::load(device.clone()).unwrap();
        let shfs = shaders::shade_fs::Shader::load(device.clone()).unwrap();

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
            triangle_buffer, bvh_buffer,
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
        let shade_desc_set = Arc::new(PersistentDescriptorSet::start(self.shade_pipeline.descriptor_set_layout(0).expect("ds layout").clone())
            .add_image(gbuf_img[0].clone()).unwrap()
            .add_image(gbuf_img[1].clone()).unwrap()
            .add_buffer(self.triangle_buffer.clone()).expect("add tri buf")
            .add_buffer(self.bvh_buffer.clone()).expect("add bvh buf")
            .build().expect("build ds"));
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
                * m::look_at(&m::vec3((now*0.1).cos()*4.0, -2.0, (now*0.1).sin()*4.0), &m::vec3(0.0, 0.1, 0.0), &m::vec3(0.0, 1.0, 0.0))),
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


        let light_v = m::vec3((now*0.5).sin(), 1.0, (now*0.5).cos()).normalize();
        let cmdbuf = AutoCommandBufferBuilder::primary_one_time_submit(self.device.clone(),
                                                    self.graphics_qu.family()).unwrap()
            .begin_render_pass(framebuffer, false, clear_values).unwrap()
            .draw_indexed(self.obj_pipeline.clone(), &dynamic_state,
                vec![self.vertex_buffer.clone()], self.index_buffer.clone(),
                (), transform).unwrap()
            .next_subpass(false).unwrap()
            .draw(self.shade_pipeline.clone(), &dynamic_state,
                vec![self.fullscreen_triangle_vertex_buffer.clone()], rdr.shade_desc_set.clone(),
                    light_v).expect("do shade")
            .end_render_pass().unwrap()
            .build().unwrap();
        Box::new(future.then_execute(self.graphics_qu.clone(), cmdbuf).unwrap())
 
    }
}
