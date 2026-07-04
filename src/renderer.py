import cv2
import numpy as np
import time
import sys

# Try importing Pygame and ModernGL for GPU rendering.
# Fall back to CPU rendering if imports or OpenGL context initialization fail.
GPU_SUPPORTED = False
try:
    import pygame
    import moderngl
    GPU_SUPPORTED = True
except ImportError:
    pass

class mp_hands:
    HAND_CONNECTIONS = frozenset([
        (0, 1), (1, 2), (2, 3), (3, 4),      # Thumb
        (5, 6), (6, 7), (7, 8),              # Index
        (9, 10), (10, 11), (11, 12),        # Middle
        (13, 14), (14, 15), (15, 16),        # Ring
        (17, 18), (18, 19), (19, 20),        # Pinky
        (0, 5), (5, 9), (9, 13), (13, 17), (0, 17) # Palm outline
    ])

def replace_texture_samples(src):
    """
    Dynamically maps texture(uTexture, uv_expr) to screen-aligned coordinates
    offset by the local displacement (refraction/aberration).
    """
    idx = 0
    while True:
        pos = src.find("texture(uTexture,", idx)
        if pos == -1:
            break
        start_arg = pos + 17
        paren_count = 1
        curr = start_arg
        while curr < len(src) and paren_count > 0:
            if src[curr] == '(':
                paren_count += 1
            elif src[curr] == ')':
                paren_count -= 1
            curr += 1
        
        if paren_count == 0:
            arg = src[start_arg : curr - 1].strip()
            replacement = f"texture(uTexture, (vScreenPos / uResolution) + ({arg} - vTexCoord))"
            src = src[:pos] + replacement + src[curr:]
            idx = pos + len(replacement)
        else:
            idx = pos + 1
    return src

class RealityFilterRenderer:
    def __init__(self, width=1280, height=720, force_cpu=False):
        self.width = width
        self.height = height
        self.use_gpu = GPU_SUPPORTED and not force_cpu
        self.ctx = None
        self.screen = None
        
        # Shader programs and texture bindings
        self.programs = {}
        self.textures = {}
        self.vbos = {}
        self.vaos = {}
        
        # Reference resolution for shaders
        self.resolution = (width, height)
        
        # CPU Fallback assets
        self.fallback_time = 0.0
        
        self.disable_filters = False
        self._update_disable_filters_config()
        
        if self.use_gpu:
            try:
                self._init_gpu()
            except Exception as e:
                print(f"[Warning] Failed to initialize GPU rendering context: {e}")
                print("[Info] Falling back to CPU-based OpenCV rendering pipeline.")
                self.use_gpu = False
                
        if not self.use_gpu:
            self._init_cpu()

    def _update_disable_filters_config(self):
        try:
            import json
            import os
            config_path = "app/config/config.json"
            if not os.path.exists(config_path):
                config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "app/config/config.json")
            if os.path.exists(config_path):
                with open(config_path, "r") as f:
                    config_data = json.load(f)
                    self.disable_filters = config_data.get("disable_filters", False)
            else:
                self.disable_filters = False
        except Exception:
            self.disable_filters = False

    def _init_gpu(self):
        """Initializes Pygame OpenGL window and ModernGL context."""
        pygame.init()
        # Create OpenGL context
        pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MAJOR_VERSION, 3)
        pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MINOR_VERSION, 3)
        pygame.display.gl_set_attribute(pygame.GL_CONTEXT_PROFILE_MASK, pygame.GL_CONTEXT_PROFILE_CORE)
        
        self.screen = pygame.display.set_mode((self.width, self.height), pygame.OPENGL | pygame.DOUBLEBUF)
        pygame.display.set_caption("Multi-Layer Finger-Controlled Reality Filters")
        
        self.ctx = moderngl.create_context()
        self.ctx.enable(moderngl.BLEND)
        self.ctx.blend_func = (moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA)
        
        # 1. Compile Shaders
        self._compile_shaders()
        
        # 2. Setup Background Quad (Standard Full-Screen Rectangle)
        # Vertices for full screen quad: X, Y, U, V
        bg_vertices = np.array([
            -1.0,  1.0, 0.0, 0.0,  # Top Left
            -1.0, -1.0, 0.0, 1.0,  # Bottom Left
             1.0,  1.0, 1.0, 0.0,  # Top Right
             1.0, -1.0, 1.0, 1.0,  # Bottom Right
        ], dtype='f4')
        
        self.bg_vbo = self.ctx.buffer(bg_vertices.tobytes())
        self.bg_vao = self.ctx.simple_vertex_array(
            self.programs["Background"], self.bg_vbo, 'in_vert', 'in_text'
        )
        
        # 3. Create placeholder camera frame textures (clean, debug, and transparent overlay)
        # ModernGL textures use (width, height)
        self.camera_tex = self.ctx.texture((self.width, self.height), 3)
        self.camera_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self.camera_tex.repeat_x = False
        self.camera_tex.repeat_y = False
        
        self.camera_debug_tex = self.ctx.texture((self.width, self.height), 3)
        self.camera_debug_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self.camera_debug_tex.repeat_x = False
        self.camera_debug_tex.repeat_y = False
        
        self.overlay_tex = self.ctx.texture((self.width, self.height), 4) # 4 channels for RGBA
        self.overlay_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self.overlay_tex.repeat_x = False
        self.overlay_tex.repeat_y = False

    def _init_cpu(self):
        """Initializes OpenCV fallback window."""
        cv2.namedWindow("Multi-Layer Finger-Controlled Reality Filters", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Multi-Layer Finger-Controlled Reality Filters", self.width, self.height)

    def _compile_shaders(self):
        """Compiles background copy and visual effect shaders."""
        # Simple vertex shader for background
        bg_vert = """
        #version 330 core
        in vec2 in_vert;
        in vec2 in_text;
        out vec2 vTexCoord;
        void main() {
            gl_Position = vec4(in_vert, 0.0, 1.0);
            vTexCoord = in_text;
        }
        """
        
        # Pass-through fragment shader for background
        bg_frag = """
        #version 330 core
        uniform sampler2D uTexture;
        in vec2 vTexCoord;
        out vec4 fragColor;
        void main() {
            fragColor = texture(uTexture, vTexCoord);
        }
        """
        
        self.programs["Background"] = self.ctx.program(
            vertex_shader=bg_vert, fragment_shader=bg_frag
        )
        
        # Vertex shader for perspective warp panels
        panel_vert = """
        #version 330 core
        in vec2 in_vert;
        out vec2 vScreenPos;
        uniform vec2 uResolution;
        void main() {
            gl_Position = vec4(in_vert, 0.0, 1.0);
            // Map clip space [-1, 1] to screen pixel space [0, width] and [0, height]
            vScreenPos.x = (in_vert.x + 1.0) * 0.5 * uResolution.x;
            vScreenPos.y = (1.0 - in_vert.y) * 0.5 * uResolution.y; 
        }
        """
        
        # Load custom shaders dynamically via FilterRegistry
        from app.modules.filter_mode.filter_registry import FilterRegistry
        self._registry = FilterRegistry()
        
        for eff in self._registry.filters:
            try:
                frag_src = self._registry.get_shader_source(eff)
                if not frag_src:
                    continue
                
                # We inject the homography and resolution uniforms into the custom fragment shaders
                # by prepending the declarations if they aren't present.
                uniforms_injection = """
                uniform mat3 uHomography;
                uniform float uAlpha = 1.0;
                """
                if "uResolution" not in frag_src:
                    uniforms_injection += "\nuniform vec2 uResolution;"
                
                # Replace standard declarations with injected values to wire the homography projection
                injected_src = frag_src.replace(
                    "in vec2 vTexCoord;",
                    "in vec2 vScreenPos; in vec2 vTexCoord;" # Keep vTexCoord for compatibility
                )
                
                # Add homography projection logic inside main() of the shader
                homography_calculation = """
                void main() {
                    vec3 proj = uHomography * vec3(vScreenPos, 1.0);
                    vec2 vTexCoord = proj.xy / proj.z;
                    // Clip shader output to only the quad bounds
                    if (vTexCoord.x < 0.0 || vTexCoord.x > 1.0 || vTexCoord.y < 0.0 || vTexCoord.y > 1.0) {
                        discard;
                    }
                """
                
                # Insert uniforms and rewrite main entry point
                injected_src = injected_src.replace("void main() {", uniforms_injection + "\n" + homography_calculation)
                injected_src = replace_texture_samples(injected_src)
                
                # Handle alpha opacity blending in shader output
                injected_src = injected_src.replace("fragColor = ", "fragColor = vec4(1.0, 1.0, 1.0, uAlpha) * ")
                
                self.programs[eff] = self.ctx.program(
                    vertex_shader=panel_vert, fragment_shader=injected_src
                )
            except Exception as e:
                print(f"[Error] Failed to compile shader '{eff}': {e}")
                sys.exit(1)
                
        # Compile PassThrough shader
        passthrough_frag = """
        #version 330 core
        in vec2 vScreenPos;
        out vec4 fragColor;
        uniform sampler2D uTexture;
        uniform mat3 uHomography;
        uniform float uAlpha = 1.0;
        uniform vec2 uResolution;
        
        void main() {
            vec3 proj = uHomography * vec3(vScreenPos, 1.0);
            vec2 vTexCoord = proj.xy / proj.z;
            if (vTexCoord.x < 0.0 || vTexCoord.x > 1.0 || vTexCoord.y < 0.0 || vTexCoord.y > 1.0) {
                discard;
            }
            fragColor = vec4(1.0, 1.0, 1.0, uAlpha) * texture(uTexture, vTexCoord);
        }
        """
        try:
            self.programs["PassThrough"] = self.ctx.program(
                vertex_shader=panel_vert, fragment_shader=passthrough_frag
            )
        except Exception as e:
            print(f"[Error] Failed to compile PassThrough shader: {e}")

    def render(self, frame, panels, tracker_instance):
        """Renders the screen frame and overlays panel effects."""
        self._update_disable_filters_config()
        h, w, c = frame.shape
        
        # Mirror frame visually for the natural mirror installation experience
        mirrored_frame = cv2.flip(frame, 1)
        
        if self.use_gpu:
            self._render_gpu(mirrored_frame, panels, tracker_instance)
        else:
            self._render_cpu(mirrored_frame, panels, tracker_instance)

    def _render_gpu(self, frame, panels, tracker):
        """Performs GPU rendering via Pygame & ModernGL."""
        # Check window events to keep Pygame responsive
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
                
        self.ctx.clear(0.0, 0.0, 0.0, 1.0)
        
        # 1. Upload Clean Frame for background and reality filters
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.camera_tex.write(frame_rgb.tobytes())
        
        # 2. Draw normal unfiltered camera background on the entire screen
        bg_copy_prog = self.programs["Background"]
        bg_copy_prog["uTexture"].value = 0
        self.camera_tex.use(0)
        self.bg_vao.render(moderngl.TRIANGLE_STRIP)
        
        # 3. Draw quadrilateral panel shape using active filter shader (Lens effect)
        t_now = time.time()
        for p in panels:
            corners = p["corners"]
            effect = "PassThrough" if self.disable_filters else p["effect"]
            alpha = p["alpha"]
            
            if effect not in self.programs:
                continue
                
            prog = self.programs[effect]
            prog["uTexture"].value = 0
            prog["uAlpha"].value = alpha
            prog["uResolution"].value = (self.width, self.height)
            
            # Update time uniforms if applicable
            if "uTime" in prog:
                prog["uTime"].value = t_now
                
            # Compute actual perspective homography mapping screen pixels to [0, 1] unit square
            try:
                src_pts = np.array([
                    [corners[0][0] * self.width, corners[0][1] * self.height],
                    [corners[1][0] * self.width, corners[1][1] * self.height],
                    [corners[2][0] * self.width, corners[2][1] * self.height],
                    [corners[3][0] * self.width, corners[3][1] * self.height]
                ], dtype=np.float32)
                
                dst_pts = np.array([
                    [0.0, 0.0],
                    [1.0, 0.0],
                    [1.0, 1.0],
                    [0.0, 1.0]
                ], dtype=np.float32)
                
                H = cv2.getPerspectiveTransform(src_pts, dst_pts)
            except Exception:
                # Fallback to screen-aligned diagonal matrix in case of degenerate inputs
                H = np.array([
                    [1.0 / self.width, 0.0, 0.0],
                    [0.0, 1.0 / self.height, 0.0],
                    [0.0, 0.0, 1.0]
                ], dtype=np.float32)
                
            prog["uHomography"].value = tuple(H.T.flatten())
            
            # Define vertices of the quad in OpenGL clip space
            clip_vertices = np.array([
                corners[0][0] * 2.0 - 1.0, 1.0 - corners[0][1] * 2.0,  # Top Left
                corners[3][0] * 2.0 - 1.0, 1.0 - corners[3][1] * 2.0,  # Bottom Left
                corners[1][0] * 2.0 - 1.0, 1.0 - corners[1][1] * 2.0,  # Top Right
                corners[2][0] * 2.0 - 1.0, 1.0 - corners[2][1] * 2.0,  # Bottom Right
            ], dtype='f4')
            
            # Buffer vertices, set VAO and draw
            vbo = self.ctx.buffer(clip_vertices.tobytes())
            vao = self.ctx.simple_vertex_array(prog, vbo, 'in_vert')
            vao.render(moderngl.TRIANGLE_STRIP)
            
            # Clean up temporary ModernGL resources
            vao.release()
            vbo.release()
            
        # 4. Draw hand tracking skeleton on transparent RGBA overlay image on top
        overlay = np.zeros((self.height, self.width, 4), dtype=np.uint8)
        if tracker:
            self._draw_hands_debug_rgba(overlay, tracker)
            
        # Draw interactive rectangle (band_1) highlight border if present
        for p in panels:
            if p["id"] == "band_1":
                corners = p["corners"]
                pts = np.array([
                    [corners[0][0] * self.width, corners[0][1] * self.height],
                    [corners[1][0] * self.width, corners[1][1] * self.height],
                    [corners[2][0] * self.width, corners[2][1] * self.height],
                    [corners[3][0] * self.width, corners[3][1] * self.height]
                ], dtype=np.int32)
                cv2.polylines(overlay, [pts], isClosed=True, color=(255, 255, 0, 200), thickness=3, lineType=cv2.LINE_AA)
                
        # Draw HUD overlay
        if tracker:
            self._draw_hud(overlay, tracker)
            
        # Upload overlay and draw on top using pass-through Background program
        self.overlay_tex.write(overlay.tobytes())
        self.overlay_tex.use(0)
        
        bg_copy_prog["uTexture"].value = 0
        self.bg_vao.render(moderngl.TRIANGLE_STRIP)
        
        pygame.display.flip()

    def _render_cpu(self, frame, panels, tracker_instance):
        """Performs CPU rendering fallback using OpenCV & NumPy."""
        t_now = time.time()
        output_frame = frame.copy()
        
        for p in panels:
            corners = p["corners"]
            effect = "PassThrough" if self.disable_filters else p["effect"]
            alpha = p["alpha"]
            
            # Create a mask for the shape
            mask = np.zeros((self.height, self.width), dtype=np.uint8)
            corners_px = np.array([
                [corners[0][0] * self.width, corners[0][1] * self.height],
                [corners[1][0] * self.width, corners[1][1] * self.height],
                [corners[2][0] * self.width, corners[2][1] * self.height],
                [corners[3][0] * self.width, corners[3][1] * self.height]
            ], dtype=np.int32)
            cv2.fillConvexPoly(mask, corners_px, 255)
            
            # Apply effect to the full frame
            filtered_frame = self._apply_cpu_effect(frame, effect, t_now)
            
            # Blend the filtered shape with the background
            blend_alpha = alpha
            if blend_alpha == 1.0:
                np.copyto(output_frame, filtered_frame, where=(mask[:, :, np.newaxis] > 0))
            else:
                idx = (mask > 0)
                output_frame[idx] = cv2.addWeighted(output_frame[idx], 1.0 - blend_alpha, filtered_frame[idx], blend_alpha, 0)
                
        # Draw interactive rectangle (band_1) highlight border if present
        for p in panels:
            if p["id"] == "band_1":
                corners = p["corners"]
                pts = np.array([
                    [corners[0][0] * self.width, corners[0][1] * self.height],
                    [corners[1][0] * self.width, corners[1][1] * self.height],
                    [corners[2][0] * self.width, corners[2][1] * self.height],
                    [corners[3][0] * self.width, corners[3][1] * self.height]
                ], dtype=np.int32)
                cv2.polylines(output_frame, [pts], isClosed=True, color=(255, 255, 0), thickness=3, lineType=cv2.LINE_AA)
                
        # Overlay visual helper lines for fingertip tracking points if active
        if tracker_instance:
            self._draw_hands_debug(output_frame, tracker_instance)
            self._draw_hud(output_frame, tracker_instance)
            
        cv2.imshow("Multi-Layer Finger-Controlled Reality Filters", output_frame)
        cv2.waitKey(1)

    def _apply_cpu_effect(self, patch, effect, t):
        """Applies visual filters on CPU for fallback mode."""
        if effect == "PassThrough":
            return patch
        elif effect == "ThermalVision":
            gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
            thermal = cv2.applyColorMap(gray, cv2.COLORMAP_JET)
            return thermal
            
        elif effect == "AnimeShader":
            # 1. Blur to smooth color surfaces
            smoothed = cv2.medianBlur(patch, 5)
            # 2. Color quantization
            quantized = (smoothed // 51) * 51
            # 3. Edge detection
            gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
            edges = cv2.Laplacian(gray, cv2.CV_8U, ksize=5)
            _, edge_mask = cv2.threshold(edges, 70, 255, cv2.THRESH_BINARY)
            # 4. Apply dark outlines
            quantized[edge_mask > 0] = [25, 20, 30]
            return quantized
            
        elif effect == "CyberpunkGlow":
            gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
            edges = cv2.Laplacian(gray, cv2.CV_8U, ksize=5)
            _, edge_mask = cv2.threshold(edges, 40, 255, cv2.THRESH_BINARY)
            dimmed = (patch.astype(np.float32) * 0.18).astype(np.uint8)
            
            color_shift = np.sin(t * 2.5) * 0.5 + 0.5
            cyan = np.array([255, 255, 0], dtype=np.float32)
            magenta = np.array([150, 0, 255], dtype=np.float32)
            neon_color = (cyan * (1.0 - color_shift) + magenta * color_shift).astype(np.uint8)
            
            glow = np.zeros_like(patch)
            glow[edge_mask > 0] = neon_color
            glow_blurred = cv2.GaussianBlur(glow, (9, 9), 0)
            return cv2.add(dimmed, cv2.addWeighted(glow, 0.6, glow_blurred, 0.4, 0))
            
        elif effect == "PixelSort":
            h, w, c = patch.shape
            sorted_patch = patch.copy()
            col_width = 8
            for col in range(0, w, col_width):
                # Apply sorting in 4 distinct vertical bands to ensure visibility across the screen height
                for band_idx in range(4):
                    noise_offset = int(np.sin(col * 0.05 + t * 4.0) * 15.0)
                    sort_start = max(0, min(h - 50, band_idx * (h // 4) + noise_offset + 20))
                    sort_end = min(h, sort_start + 80)
                    if sort_end > sort_start + 10:
                        block = sorted_patch[sort_start:sort_end, col:col+col_width]
                        gray_block = cv2.cvtColor(block, cv2.COLOR_BGR2GRAY)
                        flat_indices = np.argsort(np.mean(gray_block, axis=1))
                        sorted_patch[sort_start:sort_end, col:col+col_width] = block[flat_indices]
            return sorted_patch
            
        elif effect == "CRTMonitor":
            h, w, c = patch.shape
            map_x, map_y = np.meshgrid(np.arange(w), np.arange(h))
            x_norm = (map_x - w / 2.0) / (w / 2.0)
            y_norm = (map_y - h / 2.0) / (h / 2.0)
            r_norm = x_norm**2 + y_norm**2
            x_dist = x_norm * (1.0 + 0.08 * r_norm)
            y_dist = y_norm * (1.0 + 0.08 * r_norm)
            map_x = ((x_dist + 1.0) / 2.0 * w).astype(np.float32)
            map_y = ((y_dist + 1.0) / 2.0 * h).astype(np.float32)
            
            crt = cv2.remap(patch, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            
            scanline_mask = np.sin(np.arange(h) * 1.5 + t * 3.0) * 0.08 + 0.92
            crt = (crt.astype(np.float32) * scanline_mask[:, np.newaxis, np.newaxis]).astype(np.uint8)
            
            noise = (np.random.rand(h, w, 1) - 0.5) * 12.0
            crt = np.clip(crt.astype(np.float16) + noise, 0, 255).astype(np.uint8)
            return crt
            
        elif effect == "NightVision":
            h, w, c = patch.shape
            gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
            gray_amp = np.clip(gray.astype(np.uint16) * 1.4, 0, 255).astype(np.uint8)
            nv = np.zeros_like(patch)
            nv[:, :, 1] = gray_amp
            nv[:, :, 0] = (gray_amp * 0.1).astype(np.uint8)
            nv[:, :, 2] = (gray_amp * 0.15).astype(np.uint8)
            
            scanline_mask = np.sin(np.arange(h) * 1.8 - t * 4.0) * 0.06 + 0.94
            nv = (nv.astype(np.float32) * scanline_mask[:, np.newaxis, np.newaxis]).astype(np.uint8)
            
            noise = (np.random.rand(h, w, 1) - 0.5) * 20.0
            nv = np.clip(nv.astype(np.float16) + noise, 0, 255).astype(np.uint8)
            
            X = np.linspace(-1, 1, w)
            Y = np.linspace(-1, 1, h)
            X_grid, Y_grid = np.meshgrid(X, Y)
            vignette = 1.0 - (X_grid**2 + Y_grid**2) * 0.35
            vignette = np.clip(vignette, 0.0, 1.0)[:, :, np.newaxis]
            nv = (nv.astype(np.float32) * vignette).astype(np.uint8)
            return nv
            
        elif effect == "Hologram":
            h, w, c = patch.shape
            holo = patch.copy()
            if np.random.rand() > 0.88:
                for row in range(0, h, 16):
                    if np.random.rand() > 0.7:
                        shift = int((np.random.rand() - 0.5) * 15.0)
                        holo[row:row+16, :] = np.roll(holo[row:row+16, :], shift, axis=1)
                        
            gray = cv2.cvtColor(holo, cv2.COLOR_BGR2GRAY)
            holo_cyan = np.zeros_like(patch)
            holo_cyan[:, :, 0] = gray
            holo_cyan[:, :, 1] = (gray * 0.8).astype(np.uint8)
            holo_cyan[:, :, 2] = (gray * 0.1).astype(np.uint8)
            
            scanlines = np.sin(np.arange(h) * 1.2 - t * 7.0) * 0.12 + 0.88
            holo_cyan = (holo_cyan.astype(np.float32) * scanlines[:, np.newaxis, np.newaxis]).astype(np.uint8)
            
            flicker = np.sin(t * 25.0) * 0.08 + 0.92
            holo_cyan = np.clip(holo_cyan.astype(np.float32) * flicker, 0, 255).astype(np.uint8)
            return holo_cyan
            
        elif effect == "WaterDistortion":
            h, w, c = patch.shape
            map_x, map_y = np.meshgrid(np.arange(w), np.arange(h))
            displace = np.sin((map_x / 14.0) + t * 2.0) * 8.0
            displace_y = np.cos((map_y / 12.0) - t * 1.5) * 6.0
            map_x = np.float32(np.clip(map_x + displace, 0, w - 1))
            map_y = np.float32(np.clip(map_y + displace_y, 0, h - 1))
            
            refracted = cv2.remap(patch, map_x, map_y, cv2.INTER_LINEAR)
            highlight = np.clip(displace * 5.5, 0, 255).astype(np.uint8)
            highlight_color = cv2.merge([highlight, highlight, highlight])
            
            water = cv2.addWeighted(refracted, 0.85, highlight_color, 0.15, 0)
            # 10% BGR cyan tint
            tinted = (water.astype(np.float32) * 0.9 + np.array([230, 130, 0], dtype=np.float32) * 0.1).astype(np.uint8)
            return tinted
            
        elif effect == "EdgeDetection":
            gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
            edges = cv2.Laplacian(gray, cv2.CV_8U, ksize=5)
            _, edge_mask = cv2.threshold(edges, 40, 255, cv2.THRESH_BINARY)
            out = np.zeros_like(patch)
            out[:, :] = [18, 18, 15]
            out[edge_mask > 0] = [255, 255, 255]
            return out
            
        elif effect == "RGB_Split":
            h, w, c = patch.shape
            shift = max(1, int(w * 0.008))
            
            b, g, r = cv2.split(patch)
            r_shifted = np.roll(r, shift, axis=1)
            b_shifted = np.roll(b, -shift, axis=1)
            
            return cv2.merge([b_shifted, g, r_shifted])
            
        return patch

    def _draw_hands_debug(self, frame, tracker):
        """Draws small overlay circles and skeletal lines on tracked hands."""
        for side in ["Left", "Right"]:
            if side in tracker.smoothed_landmarks and len(tracker.smoothed_landmarks[side]) >= 21:
                lms = tracker.smoothed_landmarks[side]
                color = (0, 255, 0) if side == "Left" else (0, 0, 255) # Green / Red
                
                # 1. Draw skeletal lines first using mp_hands.HAND_CONNECTIONS
                for start_idx, end_idx in mp_hands.HAND_CONNECTIONS:
                    if start_idx in lms and end_idx in lms:
                        p_start = lms[start_idx]
                        p_end = lms[end_idx]
                        x1 = int(p_start[0] * self.width)
                        y1 = int(p_start[1] * self.height)
                        x2 = int(p_end[0] * self.width)
                        y2 = int(p_end[1] * self.height)
                        cv2.line(frame, (x1, y1), (x2, y2), color, 2, lineType=cv2.LINE_AA)
                        
                # 2. Draw all 21 joint dots and landmark IDs
                for idx in range(21):
                    if idx in lms:
                        p = lms[idx]
                        px = int(p[0] * self.width)
                        py = int(p[1] * self.height)
                        
                        is_fingertip = idx in [4, 8, 12, 16, 20]
                        size = 7 if is_fingertip else 4
                        circle_color = (255, 255, 255) if is_fingertip else color
                        cv2.circle(frame, (px, py), size, circle_color, -1, lineType=cv2.LINE_AA)
                        
                        # Draw landmark ID next to joint
                        cv2.putText(frame, str(idx), (px + size + 2, py + 4),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, lineType=cv2.LINE_AA)
                                    
                # 3. Draw Palm Center (average of 0, 5, 17)
                if all(j in lms for j in [0, 5, 17]):
                    cx = int(((lms[0][0] + lms[5][0] + lms[17][0]) / 3.0) * self.width)
                    cy = int(((lms[0][1] + lms[5][1] + lms[17][1]) / 3.0) * self.height)
                    cv2.circle(frame, (cx, cy), 6, (0, 255, 255), -1, lineType=cv2.LINE_AA) # Yellow
                    
                # 4. Draw Bounding Box and Hand Label
                xs = [lms[idx][0] * self.width for idx in range(21) if idx in lms]
                ys = [lms[idx][1] * self.height for idx in range(21) if idx in lms]
                if xs and ys:
                    min_x, max_x = min(xs), max(xs)
                    min_y, max_y = min(ys), max(ys)
                    margin = 15
                    bx = int(min_x - margin)
                    by = int(min_y - margin)
                    bw = int(max_x - min_x + 2 * margin)
                    bh = int(max_y - min_y + 2 * margin)
                    # Draw bounding box
                    cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), color, 1, lineType=cv2.LINE_AA)
                    # Draw hand label
                    cv2.putText(frame, f"{side} Hand", (bx, by - 6),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, lineType=cv2.LINE_AA)

    def _draw_hands_debug_rgba(self, frame, tracker):
        """Draws small overlay circles and skeletal lines on an RGBA frame."""
        for side in ["Left", "Right"]:
            if side in tracker.smoothed_landmarks and len(tracker.smoothed_landmarks[side]) >= 21:
                lms = tracker.smoothed_landmarks[side]
                color = (0, 255, 0, 200) if side == "Left" else (0, 0, 255, 200) # Green / Red (RGBA)
                
                # 1. Draw skeletal lines first using mp_hands.HAND_CONNECTIONS
                for start_idx, end_idx in mp_hands.HAND_CONNECTIONS:
                    if start_idx in lms and end_idx in lms:
                        p_start = lms[start_idx]
                        p_end = lms[end_idx]
                        x1 = int(p_start[0] * self.width)
                        y1 = int(p_start[1] * self.height)
                        x2 = int(p_end[0] * self.width)
                        y2 = int(p_end[1] * self.height)
                        cv2.line(frame, (x1, y1), (x2, y2), color, 2, lineType=cv2.LINE_AA)
                        
                # 2. Draw all 21 joint dots and landmark IDs
                for idx in range(21):
                    if idx in lms:
                        p = lms[idx]
                        px = int(p[0] * self.width)
                        py = int(p[1] * self.height)
                        
                        is_fingertip = idx in [4, 8, 12, 16, 20]
                        size = 7 if is_fingertip else 4
                        circle_color = (255, 255, 255, 255) if is_fingertip else color
                        cv2.circle(frame, (px, py), size, circle_color, -1, lineType=cv2.LINE_AA)
                        
                        # Draw landmark ID next to joint
                        cv2.putText(frame, str(idx), (px + size + 2, py + 4),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255, 255), 1, lineType=cv2.LINE_AA)
                                    
                # 3. Draw Palm Center (average of 0, 5, 17)
                if all(j in lms for j in [0, 5, 17]):
                    cx = int(((lms[0][0] + lms[5][0] + lms[17][0]) / 3.0) * self.width)
                    cy = int(((lms[0][1] + lms[5][1] + lms[17][1]) / 3.0) * self.height)
                    cv2.circle(frame, (cx, cy), 6, (0, 255, 255, 255), -1, lineType=cv2.LINE_AA) # Yellow (RGBA)
                    
                # 4. Draw Bounding Box and Hand Label
                xs = [lms[idx][0] * self.width for idx in range(21) if idx in lms]
                ys = [lms[idx][1] * self.height for idx in range(21) if idx in lms]
                if xs and ys:
                    min_x, max_x = min(xs), max(xs)
                    min_y, max_y = min(ys), max(ys)
                    margin = 15
                    bx = int(min_x - margin)
                    by = int(min_y - margin)
                    bw = int(max_x - min_x + 2 * margin)
                    bh = int(max_y - min_y + 2 * margin)
                    # Draw bounding box
                    cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), color, 1, lineType=cv2.LINE_AA)
                    # Draw hand label
                    cv2.putText(frame, f"{side} Hand", (bx, by - 6),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, lineType=cv2.LINE_AA)

    def _draw_hud(self, frame, tracker):
        """Draws a premium animated glassmorphic HUD displaying the active filter name on BGR or RGBA images."""
        if not tracker:
            return
            
        # Get active filter name from tracker
        effect_id = tracker.effects[tracker.current_effect_idx]
        import re
        display_name = " ".join(re.findall(r'[A-Z][a-z0-9]*', effect_id)).upper()
        
        # 1. Persistent Top-Right Indicator
        tr_w = 260
        tr_h = 55
        tr_margin = 20
        tr_x = self.width - tr_w - tr_margin
        tr_y = tr_margin
        
        is_rgba = (frame.shape[2] == 4)
        
        if is_rgba:
            tr_bg = (20, 15, 25, 200)       # Dark slate
            tr_accent = (255, 255, 0, 255)   # Neon Cyan
            tr_lbl_col = (180, 180, 180, 255) # Dim label text
            tr_val_col = (255, 255, 255, 255) # White value text
        else:
            tr_bg = (20, 15, 25)
            tr_accent = (255, 255, 0)
            tr_lbl_col = (180, 180, 180)
            tr_val_col = (255, 255, 255)
            
        # Blending preparation for persistent HUD
        tr_overlay = frame.copy()
        
        # Draw background
        cv2.rectangle(tr_overlay, (tr_x, tr_y), (tr_x + tr_w, tr_y + tr_h), tr_bg, -1)
        
        # Draw vertical neon accent bar on left side
        cv2.rectangle(tr_overlay, (tr_x, tr_y), (tr_x + 5, tr_y + tr_h), tr_accent, -1)
        
        # Draw "ACTIVE FILTER" label
        cv2.putText(tr_overlay, "ACTIVE FILTER", (tr_x + 15, tr_y + 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, tr_lbl_col, 1, lineType=cv2.LINE_AA)
        
        # Draw value text (filter name)
        cv2.putText(tr_overlay, display_name, (tr_x + 15, tr_y + 42), 
                    cv2.FONT_HERSHEY_DUPLEX, 0.55, tr_val_col, 1, lineType=cv2.LINE_AA)
        
        # Blend the persistent panel back using 0.9 alpha
        roi_y1, roi_y2 = max(0, tr_y), min(self.height, tr_y + tr_h)
        roi_x1, roi_x2 = max(0, tr_x), min(self.width, tr_x + tr_w)
        if roi_y2 > roi_y1 and roi_x2 > roi_x1:
            frame[roi_y1:roi_y2, roi_x1:roi_x2] = cv2.addWeighted(
                frame[roi_y1:roi_y2, roi_x1:roi_x2], 0.1,
                tr_overlay[roi_y1:roi_y2, roi_x1:roi_x2], 0.9, 0
            )
            
        # 2. Central temporary sliding HUD (when switched)
        if not hasattr(tracker, "last_switch_time"):
            return
            
        t_now = time.time()
        t_elapsed = t_now - tracker.last_switch_time
        
        if t_elapsed > 1.5:
            return
        
        # 1. Slide In Animation: Y-coordinate slides down from top (-100 to 45 over 300 ms)
        if t_elapsed < 0.30:
            y_pos = int(-100 + (t_elapsed / 0.30) * 145)
        else:
            y_pos = 45
            
        # 2. Fade Out Animation: stays fully opaque for 800ms, then fades to 0 by 1500ms
        if t_elapsed < 0.8:
            alpha = 1.0
        else:
            alpha = max(0.0, 1.0 - (t_elapsed - 0.8) / 0.7)
            
        # HUD Panel Dimensions
        panel_w = 400
        panel_h = 60
        panel_x = (self.width - panel_w) // 2
        
        # Determine channels
        is_rgba = (frame.shape[2] == 4)
        
        # Color values (BGR or RGBA)
        if is_rgba:
            bg_color = (25, 20, 30, 210)
            border_color = (255, 255, 0, 255)  # Cyan border
            text_shadow = (180, 120, 0, 255)
            text_color = (255, 255, 255, 255)
        else:
            bg_color = (25, 20, 30)
            border_color = (255, 255, 0)
            text_shadow = (180, 120, 0)
            text_color = (255, 255, 255)
            
        # Create temp panel for blending
        temp_overlay = frame.copy()
        
        # Draw panel background
        cv2.rectangle(temp_overlay, (panel_x, y_pos), (panel_x + panel_w, y_pos + panel_h), bg_color, -1)
        
        # Draw border
        cv2.line(temp_overlay, (panel_x + 10, y_pos), (panel_x + panel_w - 10, y_pos), border_color, 2, lineType=cv2.LINE_AA)
        
        # Text positioning
        font = cv2.FONT_HERSHEY_DUPLEX
        font_scale = 0.7
        font_thickness = 2
        text_size = cv2.getTextSize(display_name, font, font_scale, font_thickness)[0]
        text_x = panel_x + (panel_w - text_size[0]) // 2
        text_y = y_pos + (panel_h + text_size[1]) // 2 - 2
        
        # Draw text shadow and main text
        cv2.putText(temp_overlay, display_name, (text_x + 1, text_y + 1), font, font_scale, text_shadow, font_thickness, lineType=cv2.LINE_AA)
        cv2.putText(temp_overlay, display_name, (text_x, text_y), font, font_scale, text_color, font_thickness, lineType=cv2.LINE_AA)
        
        # Blend the modified panel region back using alpha
        roi_y1, roi_y2 = max(0, y_pos), min(self.height, y_pos + panel_h)
        roi_x1, roi_x2 = max(0, panel_x), min(self.width, panel_x + panel_w)
        
        if roi_y2 > roi_y1 and roi_x2 > roi_x1:
            frame[roi_y1:roi_y2, roi_x1:roi_x2] = cv2.addWeighted(
                frame[roi_y1:roi_y2, roi_x1:roi_x2], 1.0 - alpha,
                temp_overlay[roi_y1:roi_y2, roi_x1:roi_x2], alpha, 0
            )

    def close(self):
        """Clean up resources."""
        if self.use_gpu:
            pygame.quit()
        else:
            cv2.destroyAllWindows()
