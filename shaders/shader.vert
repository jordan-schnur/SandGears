#version 450

// Particle positions storage buffer
layout(binding = 2) buffer ParticleBuffer {
    vec2 particle_positions[];
};

layout(binding = 3) buffer ColorBuffer {
    vec3 colors[];
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
    fragColor = colors[gl_InstanceIndex];
}