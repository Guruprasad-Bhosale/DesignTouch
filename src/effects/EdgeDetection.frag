#version 330 core

uniform sampler2D uTexture;
in vec2 vTexCoord;
out vec4 fragColor;

uniform vec2 uResolution = vec2(1280.0, 720.0);

void main() {
    vec2 uv = vTexCoord;
    vec2 texel = 1.0 / uResolution;
    
    // Sobel kernels for edge detection
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
    
    // Smooth threshold the edge intensity
    float outline = smoothstep(0.1, 0.35, edge);
    
    // Bright white lines on a dark charcoal background
    vec3 backColor = vec3(0.06, 0.06, 0.07);
    vec3 edgeColor = vec3(1.0, 1.0, 1.0);
    vec3 finalColor = mix(backColor, edgeColor, outline);
    
    fragColor = vec4(finalColor, 1.0);
}
