#version 450

// Particle positions storage buffer
layout(binding = 2) buffer ParticleBuffer {
    vec2 particle_positions[];
};

layout(binding = 3) buffer ColorBuffer {
    uint color_type[]; // Changed from "type[]"
};

const vec4 COLORS[7] = {
vec4(0.0, 0.5, 1.0, 1.0), // Water: Blue
vec4(1.0, 0.0, 0.0, 1.0), // Fire: Red
vec4(1.0, 0.5, 0.0, 1.0), // Lava: Orange
vec4(0.8, 0.8, 0.0, 1.0), // Gas: Yellow
vec4(1.0, 1.0, 1.0, 1.0), // Snow: White
vec4(0.9, 0.6, 0.3, 1.0), // Sand: Light Brown
vec4(0.0, 0.0, 0.0, 0.0)  // Air: Transparent (or whatever color you want for Air)
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
    vec2 particlePosition = particle_positions[gl_InstanceIndex];
    gl_Position = ubo.proj * ubo.view * ubo.model * vec4(inPosition.x+particlePosition.x, inPosition.y + particlePosition.y, 0.0, 1.0);
    vec4 chosenColor = COLORS[color_type[gl_InstanceIndex]]; // Changed from "ColorBuffer.type" to "color_type"
    fragColor = chosenColor.rgb; // Changed from "vec4(chosenColor, 1.0)" to "chosenColor.rgb"
}