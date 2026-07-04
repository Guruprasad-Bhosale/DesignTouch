#version 330 core

uniform sampler2D uTexture;
in vec2 vTexCoord;
out vec4 fragColor;

uniform vec2 uResolution = vec2(1280.0, 720.0);
uniform float uTime = 0.0;

void main() {
    vec2 uv = vTexCoord;
    vec2 texel = 1.0 / uResolution;
    
    // Sobel edge detection
    float x = 0.0;
    float y = 0.0;
    
    #define intensity(pos) dot(texture(uTexture, pos).rgb, vec3(0.299, 0.587, 0.114))
    
    x += intensity(uv + vec2(-texel.x, -texel.y)) * -1.0;
    x += intensity(uv + vec2(-texel.x, 0.0))      * -2.0;
    x += intensity(uv + vec2(-texel.x, texel.y))  * -1.0;
    x += intensity(uv + vec2(texel.x, -texel.y))  * 1.0;
    x += intensity(uv + vec2(texel.x, 0.0))       * 2.0;
    x += intensity(uv + vec2(texel.x, texel.y))   * 1.0;
    
    y += intensity(uv + vec2(-texel.x, -texel.y)) * -1.0;
    y += intensity(uv + vec2(0.0, -texel.y))      * -2.0;
    y += intensity(uv + vec2(texel.x, -texel.y))  * -1.0;
    y += intensity(uv + vec2(-texel.x, texel.y))  * 1.0;
    y += intensity(uv + vec2(0.0, texel.y))       * 2.0;
    y += intensity(uv + vec2(texel.x, texel.y))   * 1.0;
    
    float edge = sqrt(x*x + y*y);
    
    // Dynamic neon color shift between pink-magenta and cyan over time and space
    float colorShift = sin(uTime * 2.5 + uv.y * 3.0) * 0.5 + 0.5;
    vec3 cyan = vec3(0.0, 1.0, 1.0);
    vec3 magenta = vec3(1.0, 0.0, 0.6);
    vec3 neonColor = mix(cyan, magenta, colorShift);
    
    // High intensity neon glow on edges
    float edgeGlow = smoothstep(0.06, 0.35, edge);
    vec3 finalGlow = edgeGlow * neonColor * 2.5;
    
    // Dim background color for maximum contrast
    vec3 orig = texture(uTexture, uv).rgb;
    vec3 finalColor = mix(orig * 0.15, finalGlow, edgeGlow);
    
    // Core white hot highlights
    float core = smoothstep(0.4, 0.75, edge);
    finalColor = mix(finalColor, vec3(1.0, 1.0, 1.0), core * 0.85);
    
    fragColor = vec4(finalColor, 1.0);
}
