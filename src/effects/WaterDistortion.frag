#version 330 core

uniform sampler2D uTexture;
in vec2 vTexCoord;
out vec4 fragColor;

uniform vec2 uResolution = vec2(1280.0, 720.0);
uniform float uTime = 0.0;

void main() {
    vec2 uv = vTexCoord;
    
    // 1. Calculate water wave displacements (overlapping sine waves)
    float speedX = uTime * 2.0;
    float speedY = uTime * 1.5;
    
    float waveX = sin(uv.x * 25.0 + speedX) * cos(uv.y * 20.0 - speedY) * 0.012;
    float waveY = cos(uv.x * 18.0 - speedX) * sin(uv.y * 28.0 + speedY) * 0.012;
    
    vec2 displacedUV = uv + vec2(waveX, waveY);
    displacedUV = clamp(displacedUV, 0.0, 1.0);
    
    // Sample camera texture
    vec3 origColor = texture(uTexture, displacedUV).rgb;
    
    // 2. Add specular highlights based on local wave gradient (normals)
    float waveGradX = cos(uv.x * 25.0 + speedX) * 0.08;
    float waveGradY = -sin(uv.y * 28.0 + speedY) * 0.08;
    vec3 waveNormal = normalize(vec3(waveGradX, waveGradY, 1.0));
    
    // Light vector pointing from top-left front
    vec3 lightDir = normalize(vec3(-0.4, -0.6, 0.7));
    
    // Simple diffuse/specular dot reflection
    float spec = max(0.0, dot(waveNormal, lightDir));
    spec = pow(spec, 32.0); // High gloss specular reflection
    
    // Blend displaced image with specular water highlight
    vec3 waterColor = origColor + vec3(spec * 0.5);
    
    // Tint slightly cyan/blue to simulate water depth
    waterColor = mix(waterColor, vec3(0.0, 0.65, 0.8) * (waterColor.g + 0.1), 0.12);
    
    fragColor = vec4(waterColor, 1.0);
}
