import os
import moderngl

# Create a standalone context for testing
ctx = moderngl.create_context(standalone=True)

shaders_dir = r"E:\Projects101\FilterTrack\app\shaders"
vertex_shader = """
#version 330 core
in vec2 in_vert;
in vec2 in_uv;
out vec2 v_uv;
void main() {
    v_uv = in_uv;
    gl_Position = vec4(in_vert, 0.0, 1.0);
}
"""

for file in os.listdir(shaders_dir):
    if file.endswith(".frag"):
        name = file[:-5]
        path = os.path.join(shaders_dir, file)
        with open(path, "r") as f:
            frag_code = f.read()
        
        try:
            prog = ctx.program(vertex_shader=vertex_shader, fragment_shader=frag_code)
            print(f"[OK] {name}")
        except Exception as e:
            print(f"[ERROR] {name}: {e}")
