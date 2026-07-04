#version 330 core

uniform sampler2D uTexture;
in vec2 vTexCoord;
out vec4 fragColor;

uniform vec2 uResolution = vec2(1280.0, 720.0);
uniform float uTime = 0.0;

// Simple random function
float hash12(vec2 p) {
    p = fract(p * vec2(91.345, 127.456));
    p += dot(p, p + 23.45);
    return fract(p.x * p.y);
}

void main() {
    vec2 uv = vTexCoord;
    
    // Width of columns
    float colWidth = 10.0;
    float colId = floor(uv.x * uResolution.x / colWidth);
    
    // Pseudo-random thresholds per column
    float noise1 = hash12(vec2(colId, 1.0));
    float noise2 = hash12(vec2(colId, 5.0));
    
    // Use a periodic wave to define multiple active sorting bands across the screen/texture height
    float bandValue = sin(uv.y * 60.0 + noise2 * 6.28);
    
    if (bandValue > -0.2) {
        // Simulating the displacement of pixel sort
        float tFactor = fract(uTime * 1.2 + noise1);
        float displacement = (bandValue + 0.2) * 0.15;
        
        // Sample a reference color slightly shifted vertically
        vec4 refColor = texture(uTexture, vec2(uv.x, uv.y - displacement * 0.2));
        float brightness = dot(refColor.rgb, vec3(0.299, 0.587, 0.114));
        
        // Apply vertical displacement
        uv.y -= displacement * (1.0 - brightness) * tFactor * 0.15;
    }
    
    // Also add occasional horizontal line shearing
    if (hash12(vec2(floor(uv.y * 40.0), floor(uTime * 6.0))) > 0.94) {
        uv.x += (hash12(vec2(uv.y, uTime)) - 0.5) * 0.03;
    }
    
    uv = clamp(uv, 0.0, 1.0);
    vec4 finalColor = texture(uTexture, uv);
    
    fragColor = vec4(finalColor.rgb, 1.0);
}
