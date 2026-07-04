#version 330 core

uniform sampler2D uTexture;
in vec2 vTexCoord;
out vec4 fragColor;

uniform vec2 uResolution = vec2(1280.0, 720.0);

void main() {
    vec2 uv = vTexCoord;
    vec2 texel = 1.0 / uResolution;

    // 1. Sample original color
    vec4 origColor = texture(uTexture, uv);
    
    // Convert to HSV to shift saturation/value
    vec3 color = origColor.rgb;
    
    // 2. Quantize colors (Cel-shading)
    float numBins = 5.0;
    vec3 quantized = floor(color * numBins) / numBins;
    
    // Boost saturation slightly
    float maxCol = max(quantized.r, max(quantized.g, quantized.b));
    float minCol = min(quantized.r, min(quantized.g, quantized.b));
    float l = (maxCol + minCol) * 0.5;
    vec3 saturated = mix(vec3(l), quantized, 1.3);
    
    // 3. Sobel Edge Detection for outlines
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
    float outline = smoothstep(0.18, 0.35, edge);
    
    // 4. Mix cel-shaded colors with dark ink outlines
    vec3 inkColor = vec3(0.1, 0.08, 0.12);
    vec3 finalColor = mix(saturated, inkColor, outline);
    
    fragColor = vec4(finalColor, 1.0);
}
