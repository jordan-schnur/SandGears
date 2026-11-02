#version 450

// Colours each pixel with the colour of the world cell underneath it.
// The CPU uploads one packed RGBA colour (0xAABBGGRR) per cell every frame.

layout(location = 0) in vec2 uv;
layout(location = 0) out vec4 out_color;

layout(set = 0, binding = 0) readonly buffer Cells {
    uint cell_colors[];
};

layout(push_constant) uniform WorldSize {
    uvec2 size;
} world;

void main() {
    uvec2 cell = min(uvec2(uv * vec2(world.size)), world.size - uvec2(1u, 1u));
    uint packed = cell_colors[cell.y * world.size.x + cell.x];

    out_color = vec4(
        float(packed & 255u),
        float((packed >> 8u) & 255u),
        float((packed >> 16u) & 255u),
        float((packed >> 24u) & 255u)
    ) / 255.0;
}
