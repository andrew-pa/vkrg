
pub mod obj_vs {
    vulkano_shaders::shader!{
        ty: "vertex",
        src: "
            #version 450
            layout (location = 0) in vec3 position;
            layout (location = 1) in uint material_index;
            layout (location = 2) in vec3 normal;

            layout (std430, push_constant) uniform pc {
                mat4 T;
            };

            layout(location = 0) out vec3 out_normal;
            layout(location = 1) out vec3 out_position;
            layout(location = 2) out uint out_mat;

            void main() {
                out_position = position;
                out_normal = normal;
                out_mat = material_index;
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
            layout(location = 2) in flat uint material_index;
            layout(location = 0) out vec4 pos;
            layout(location = 1) out vec4 nor;

            void main() {
                pos = vec4(position, 0.0);
                nor = vec4(normal, 1.0 + float(material_index));
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

// ----------------------------------------------------------------------------
// from: https://www.shadertoy.com/view/4tl3z4

float hash1(inout float seed) {
    return fract(sin(seed += 0.1)*43758.5453123);
}

vec2 hash2(inout float seed) {
    return fract(sin(vec2(seed+=0.1,seed+=0.1))*vec2(43758.5453123,22578.1459123));
}

vec3 hash3(inout float seed) {
    return fract(sin(vec3(seed+=0.1,seed+=0.1,seed+=0.1))*vec3(43758.5453123,22578.1459123,19642.3490423));
}

vec3 cosWeightedRandomHemisphereDirection( const vec3 n, inout float seed ) {
  	vec2 r = hash2(seed);
    
	vec3  uu = normalize( cross( n, vec3(0.0,1.0,1.0) ) );
	vec3  vv = cross( uu, n );
	
	float ra = sqrt(r.y);
	float rx = ra*cos(6.2831*r.x); 
	float ry = ra*sin(6.2831*r.x);
	float rz = sqrt( 1.0-r.y );
	vec3  rr = vec3( rx*uu + ry*vv + rz*n );
    
    return normalize( rr );
}

vec3 randomSphereDirection(inout float seed) {
    vec2 h = hash2(seed) * vec2(2.,6.28318530718)-vec2(1,0);
    float phi = h.y;
	return vec3(sqrt(1.-h.x*h.x)*vec2(sin(phi),cos(phi)),h.x);
}

vec3 randomHemisphereDirection( const vec3 n, inout float seed ) {
	vec3 dr = randomSphereDirection(seed);
	return dot(dr,n) * dr;
}
// ----------------------------------------------------------------------------

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

struct material {
    vec3 diffuse;
    uint pad1;
    vec3 emission; 
    uint pad2;
};

layout(std430, binding = 4) buffer mats_b { 
    material[] data;
} mats;

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

struct trace_result {
    float t;
    uint index;
};

trace_result trace_ray(vec3 origin, vec3 direction) {
    origin += direction*0.04;
    uint index = 0;
    uint num_nodes = bvh.nodes[0]._pad[0];
    trace_result min_res;
    min_res.t = TMAX;
    min_res.index = 0;
    while (index < num_nodes) {
        if (bvh.nodes[index].entry_index == 0xffffffff) {
            //leaf
            float t = intersect_tri(origin, direction, bvh.nodes[index].shape_index);
            if(t < min_res.t) {
                min_res.t = t;
                min_res.index = bvh.nodes[index].shape_index;
            }
            index = bvh.nodes[index].exit_index;
        } else if(intersect_aabb(origin, direction, index)) {
            index = bvh.nodes[index].entry_index;
        } else {
            index = bvh.nodes[index].exit_index;
        }
    }
    return min_res;
}

layout(location = 0) out vec4 out_color;

layout(std430, push_constant) uniform pc {
    vec3 L;
    float pad;
};

void main() {
    vec4 sunor = subpassLoad(normals);
    if(sunor.w < 1.0) discard;
    material mat = mats.data[uint(sunor.w - 1.0)];
    if(mat.emission.r > 0.0) {
        out_color = vec4(mat.emission/100.0, 1.0);
        return;
    }
    vec3 firstHitNormal = normalize(sunor.rgb);
    vec3 P = subpassLoad(positions).rgb;

    vec2 p = (gl_FragCoord.xy / vec2(320, 240))*2.0 - 1.0; 
    float seed = p.x + p.y * 3.43121412313;

    vec3 Ki = vec3(0.0);
    vec3 n = firstHitNormal;
    vec3 Wi = cosWeightedRandomHemisphereDirection(firstHitNormal, seed); 
    vec3 firstHitLi = mat.diffuse * dot(firstHitNormal, Wi);

    for(int i = 0; i < 12; ++i) {
        vec3 Li = firstHitLi;
        for(int j = 0; j < 3; ++j) {
            trace_result res = trace_ray(P, Wi);
            if(res.t >= TMAX) {
                Li *= vec3(20., 20., 15.);
                break;
            }
            material cmat = mats.data[triangles.data[res.index].material_index];
            n = cross(normalize(triangles.data[res.index].edge1), normalize(triangles.data[res.index].edge2));
            if(cmat.emission.r > 0.0) {
                Li *= cmat.emission;
                break;
            } else {
                Li *= cmat.diffuse * dot(n, Wi);
            }
            P += Wi*res.t;
            Wi = cosWeightedRandomHemisphereDirection(n, seed);
        }
        Ki += Li;
        seed = mod( seed*1.1234567893490423, 13. );
    }
    Ki /= 12.0;

    out_color = vec4(Ki, 1.0);
}
"
    }
}
