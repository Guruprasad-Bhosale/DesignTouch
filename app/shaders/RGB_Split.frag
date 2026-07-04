#version 330 core

uniform sampler2D uTexture;
in vec2 vTexCoord;
out vec4 fragColor;

void main() {
    // Chromatic aberration / channel splitting effect
    float shift = 0.008;
    float r = texture(uTexture, vTexCoord + vec2(shift, 0.0)).r;
    float g = texture(uTexture, vTexCoord).g;
    float b = texture(uTexture, vTexCoord - vec2(shift, 0.0)).b;
    
    fragColor = vec4(r, g, b, 1.0);
}
