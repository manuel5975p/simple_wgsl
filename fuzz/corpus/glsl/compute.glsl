#version 450
layout(local_size_x = 64) in;
layout(set = 0, binding = 0) buffer Buf { float data[]; };
void main() { data[gl_GlobalInvocationID.x] *= 2.0; }
