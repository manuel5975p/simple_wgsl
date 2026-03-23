#version 450
layout(location = 0) in vec3 pos;
layout(location = 0) out vec4 vColor;
void main() { gl_Position = vec4(pos, 1.0); vColor = vec4(1.0); }
