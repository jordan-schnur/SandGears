#version 450

// Particle positions storage buffer
layout(binding = 2) buffer ParticleBuffer {
    vec2 particle_positions[];
};

layout(binding = 3) buffer ColorBuffer {
    uint color_type[]; // Changed from "type[]"
};

const vec4 COLORS[13] = {
vec4(0.0, 0.5, 1.0, 1.0), // Water: Blue
vec4(1.0, 0.0, 0.0, 1.0), // Fire: Red
vec4(1.0, 0.5, 0.0, 1.0), // Lava: Orange
vec4(0.8, 0.8, 0.0, 1.0), // Gas: Yellow
vec4(1.0, 1.0, 1.0, 1.0), // Snow: White
vec4(0.9, 0.6, 0.3, 1.0), // Sand: Light Brown
vec4(0.0, 0.0, 0.0, 0.0), // Air: Transparent (or whatever color you want for Air)
vec4(0.6667, 0.6627, 0.6784, 0.0), // Metal: Silver (Metallic) / #aaa9ad Hex Color Code
// UI colors, indexed by the UI_COLOR_* constants in main.rs
vec4(0.12, 0.12, 0.14, 1.0), // Toolbar background
vec4(1.0, 1.0, 1.0, 1.0),    // Selected button frame
vec4(0.08, 0.08, 0.08, 1.0), // Dark text
vec4(0.9, 0.9, 0.9, 1.0),    // Light text
vec4(0.3, 0.3, 0.33, 1.0)    // Eraser button
};

layout(binding = 0) uniform UniformBufferObject {
    mat4 model;
    mat4 view;
    mat4 proj;
} ubo;

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inColor;
layout(location = 2) in vec2 inTexCoord;

layout(location = 0) out vec3 fragColor;

void main() {
    // Bits 0-6 are the color index, bit 7 marks screen-space UI and the rest is the rectangle
    // size (see pack_instance in main.rs). Particles leave the size bits at zero, which max()
    // turns into a 1x1 cell.
    uint word = color_type[gl_InstanceIndex];
    vec2 size = vec2(max((word >> 8) & 0xFFFu, 1u), max(word >> 20, 1u));
    // The model matrix holds zoom and pan, which the toolbar must not follow.
    mat4 model = (word & 0x80u) != 0u ? mat4(1.0) : ubo.model;
    vec2 particlePosition = particle_positions[gl_InstanceIndex];
    gl_Position = ubo.proj * ubo.view * model * vec4(inPosition.xy * size + particlePosition, 0.0, 1.0);
    fragColor = COLORS[word & 0x7Fu].rgb;
}