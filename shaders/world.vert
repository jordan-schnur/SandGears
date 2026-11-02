#version 450

// Draws one triangle that covers the whole viewport (no vertex buffer needed).
// `uv` goes from (0,0) in the top-left corner to (1,1) in the bottom-right.

layout(location = 0) out vec2 uv;

void main() {
    vec2 corner = vec2(float((gl_VertexIndex << 1) & 2), float(gl_VertexIndex & 2));
    uv = corner;
    gl_Position = vec4(corner * 2.0 - 1.0, 0.0, 1.0);
}
