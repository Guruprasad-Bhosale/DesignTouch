#version 330 core

uniform sampler2D uTexture;
in vec2 vTexCoord;
out vec4 fragColor;

uniform vec2 uResolution = vec2(1280.0, 720.0);
uniform float uTime = 0.0;

float rand(vec2 co) {
    return fract(sin(dot(co.xy, vec2(12.9898, 78.233))) * 43758.5453);
}

void main() {
    vec2 uv = vTexCoord;
    
    // 1. Simulate horizontal shear glitches
    float glitchChance = rand(vec2(floor(uTime * 8.0), 12.0));
    if (glitchChance > 0.88) {
        float shiftY = floor(uv.y * 32.0) / 32.0;
        float shiftVal = (rand(vec2(shiftY, uTime)) - 0.5) * 0.025;
        uv.x += shiftVal;
    }
    
    uv = clamp(uv, 0.0, 1.0);
    
    // Sample with blue color chromatic aberration
    float r = texture(uTexture, uv + vec2(-0.003, 0.0)).r;
    float g = texture(uTexture, uv).g;
    float b = texture(uTexture, uv + vec2(0.003, 0.0)).b;
    vec3 color = vec3(r, g, b);
    
    // Convert to grayscale
    float lum = dot(color, vec3(0.299, 0.587, 0.114));
    
    // 2. Cyan hologram glow color mapping
    vec3 holoColor = vec3(0.0, 0.78, 1.0);
    vec3 finalColor = holoColor * lum * 1.6;
    
    // 3. Add glowing scanning lines
    float scanline = sin(uv.y * 180.0 - uTime * 6.5) * 0.15 + 0.85;
    finalColor *= scanline;
    
    // 4. Time-based noise flicker
    float flicker = rand(vec2(uTime * 15.0, 1.0)) * 0.18 + 0.82;
    finalColor *= flicker;
    
    // 5. Add some high frequency scanlines
    float microScan = sin(uv.y * uResolution.y * 0.6) * 0.05 + 0.95;
    finalColor *= microScan;
    
    fragColor = vec4(finalColor, 1.0);
}
