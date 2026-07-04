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
    
    // Sample texture
    vec4 color = texture(uTexture, uv);
    
    // Calculate luminance
    float lum = dot(color.rgb, vec3(0.299, 0.587, 0.114));
    
    // Amplify luminance for light gain (night vision exposure boost)
    lum = clamp(lum * 1.5, 0.0, 1.0);
    
    // Map to classic green-scale phosphors
    vec3 greenTint = vec3(0.1, 0.95, 0.2) * lum;
    
    // Add bright green highlight glow for extra light spots
    vec3 highlights = vec3(0.8, 1.0, 0.85) * smoothstep(0.65, 0.95, lum);
    vec3 finalColor = greenTint + highlights * 0.4;
    
    // Add grainy moving static noise
    float noise = (rand(uv + uTime) - 0.5) * 0.12;
    finalColor += vec3(noise);
    
    // Add horizontal scrolling scanlines
    float scanline = sin(uv.y * 300.0 - uTime * 4.0) * 0.08 + 0.92;
    finalColor *= scanline;
    
    // Add dark radial vignette mask
    vec2 dist = uv - vec2(0.5);
    float vignette = 1.0 - dot(dist, dist) * 1.6;
    vignette = clamp(vignette, 0.0, 1.0);
    finalColor *= vignette;
    
    fragColor = vec4(finalColor, 1.0);
}
