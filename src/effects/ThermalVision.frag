#version 330 core

uniform sampler2D uTexture;
in vec2 vTexCoord;
out vec4 fragColor;

void main() {
    // Sample original color from camera feed
    vec4 color = texture(uTexture, vTexCoord);
    
    // Calculate luminance (standard greyscale coefficients)
    float lum = dot(color.rgb, vec3(0.299, 0.587, 0.114));
    
    // Define thermal color ramp stops
    vec3 cold  = vec3(0.0, 0.0, 0.3);  // Deep Blue
    vec3 cool  = vec3(0.4, 0.0, 0.6);  // Purple
    vec3 warm  = vec3(0.9, 0.2, 0.0);  // Bright Red-Orange
    vec3 hot   = vec3(1.0, 0.9, 0.0);  // Yellow
    vec3 white = vec3(1.0, 1.0, 1.0);  // White (Extreme heat)
    
    vec3 thermal;
    if (lum < 0.25) {
        thermal = mix(cold, cool, lum / 0.25);
    } else if (lum < 0.5) {
        thermal = mix(cool, warm, (lum - 0.25) / 0.25);
    } else if (lum < 0.75) {
        thermal = mix(warm, hot, (lum - 0.5) / 0.25);
    } else {
        thermal = mix(hot, white, (lum - 0.75) / 0.25);
    }
    
    fragColor = vec4(thermal, 1.0);
}
