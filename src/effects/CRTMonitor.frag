#version 330 core

uniform sampler2D uTexture;
in vec2 vTexCoord;
out vec4 fragColor;

uniform vec2 uResolution = vec2(1280.0, 720.0);
uniform float uTime = 0.0;

// Curvature warp helper
vec2 curve(vec2 uv) {
    uv = uv * 2.0 - 1.0;
    vec2 offset = abs(uv.yx) / vec2(8.0, 6.0); // Curve factor
    uv = uv + uv * offset * offset;
    return uv * 0.5 + 0.5;
}

float rand(vec2 co) {
    return fract(sin(dot(co.xy, vec2(12.9898, 78.233))) * 43758.5453);
}

void main() {
    // 1. Curved CRT geometry
    vec2 uv = curve(vTexCoord);
    
    // Discard coordinates warped outside the screen boundaries
    if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0) {
        discard;
    }
    
    // 2. Chromatic aberration
    float r = texture(uTexture, uv + vec2(0.002, 0.0)).r;
    float g = texture(uTexture, uv).g;
    float b = texture(uTexture, uv - vec2(0.002, 0.0)).b;
    vec3 color = vec3(r, g, b);
    
    // 3. Scanline overlay
    float scanline = sin(uv.y * uResolution.y * 1.5 + uTime * 3.0) * 0.12 + 0.88;
    color *= scanline;
    
    // 4. Vertical RGB Phosphor Mask Striping
    float rgbPattern = sin(uv.x * uResolution.x * 2.0) * 0.08 + 0.92;
    color *= rgbPattern;
    
    // 5. Dark Vignette around curved edges
    float vignette = uv.x * uv.y * (1.0 - uv.x) * (1.0 - uv.y);
    vignette = clamp(pow(16.0 * vignette, 0.25), 0.0, 1.0);
    color *= vignette;
    
    // 6. Subtle screen flicker and random noise
    float flicker = rand(vec2(uTime, uv.y)) * 0.015 + 0.985;
    color *= flicker;
    color += (rand(uv + uTime) - 0.5) * 0.04;
    
    fragColor = vec4(color, 1.0);
}
