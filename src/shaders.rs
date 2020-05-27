
pub mod obj_vs {
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

pub mod fs_vs {
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

pub mod gbuf_fs {
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

pub mod shade_fs {
    vulkano_shaders::shader!{
        ty: "fragment",
        src: "
            #version 450
            layout(input_attachment_index = 0, set = 0, binding = 0) uniform subpassInput positions;
            layout(input_attachment_index = 1, set = 0, binding = 1) uniform subpassInput normals;

            struct triangle {
                vec3 corner;
                uint material_index;
                vec3 edge1;
                uint node_index;
                vec3 edge2;
            };

            layout(std430, binding = 2) buffer triangles_b {
                triangle[] data;
            } triangles;

            struct bvh_node {
                vec3 min;
                uint entry_index;
                vec3 max;
                uint exit_index, shape_index;
                uint _pad[3];
            };

            layout(std430, binding = 3) buffer bvh_b {
                bvh_node[] nodes;
            } bvh;

            #define TMAX 10000.0

            bool intersect_aabb(vec3 o, vec3 d, uint index) {
                vec3 rrd = 1.0 / d;
                vec3 t1 = (bvh.nodes[index].min - o) * rrd;
                vec3 t2 = (bvh.nodes[index].max - o) * rrd;
                vec3 m12 = min(t1, t2);
                vec3 x12 = max(t1, t2);
                float tmin = max(m12.x, max(m12.y, m12.z));
                float tmax = min(x12.x, min(x12.y, x12.z));
                return tmax >= tmin && tmax >= 0.0;
            }

            float intersect_tri(vec3 o, vec3 d, uint index) {
                vec3 pv = cross(d, triangles.data[index].edge2);
                float det = dot(triangles.data[index].edge1, pv);
                if(det == 0) return TMAX;
                float idet = 1.0 / det;
                vec3 tv = o - triangles.data[index].corner;
                float u = dot(tv, pv) * idet;
                if(u < 0.0 || u > 1.0) return TMAX;
                vec3 qv = cross(tv, triangles.data[index].edge1);
                float v = dot(d, qv) * idet;
                if(v < 0.0 || u+v > 1.0) return TMAX;
                float nt = dot(triangles.data[index].edge2, qv) * idet;
                return nt;
            }

            float trace_ray(vec3 origin, vec3 direction) {
                origin += direction*0.04;
                uint index = 0;
                uint num_nodes = bvh.nodes[0]._pad[0];
                float min_t = TMAX; uint min_tri = 0;
                while (index < num_nodes) {
                    if (bvh.nodes[index].entry_index == 0xffffffff) {
                        //leaf
                        float t = intersect_tri(origin, direction, bvh.nodes[index].shape_index);
                        if(t < min_t) {
                            min_t = t;
                            min_tri = bvh.nodes[index].shape_index;
                        }
                        index = bvh.nodes[index].exit_index;
                    } else if(intersect_aabb(origin, direction, index)) {
                        index = bvh.nodes[index].entry_index;
                    } else {
                        index = bvh.nodes[index].exit_index;
                    }
                }
                return min_t;
            }

            layout(location = 0) out vec4 out_color;

            layout(std430, push_constant) uniform pc {
                vec3 L;
                float pad;
            };

            void main() {
                vec4 sunor = subpassLoad(normals);
                if(sunor.w < 1.0) discard;
                vec3 N = normalize(sunor.rgb);
                vec3 P = subpassLoad(positions).rgb;
                //const vec3 L = normalize(vec3(1.0, 0.8, 0.2));
                float s = trace_ray(P, L) < TMAX ? 0.0 : 1.0;
                out_color = vec4(vec3(0.1, 0.8, 0.3)*max(0.0, dot(N, L))*s + vec3(0.07, 0.06, 0.05), 1.0);
            }
        "
    }
}
