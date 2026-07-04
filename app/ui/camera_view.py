import sys
import os
import time
import numpy as np
import cv2
import re

from PyQt6.QtOpenGLWidgets import QOpenGLWidget
from PyQt6.QtGui import QPainter, QImage, QColor, QPen, QBrush, QFont
from PyQt6.QtCore import Qt, QRect

# Try importing ModernGL
MODERNGL_SUPPORTED = False
try:
    import moderngl
    MODERNGL_SUPPORTED = True
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

class CameraView(QOpenGLWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.use_gpu = MODERNGL_SUPPORTED
        self.ctx = None
        
        # Frame and layout parameters
        self._frame = None
        self._panels = []
        self._tracking_data = None
        
        # Shader programs and assets
        self.programs = {}
        self.textures = {}
        self.bg_vao = None
        self.bg_vbo = None
        
        # Settings
        self.show_skeleton = True
        self.active_module_name = "floating_menu"
        self.virtual_cursor_pos = (0, 0)
        self.draw_virtual_cursor = True
        
        from app.modules.filter_mode.filter_registry import FilterRegistry
        self._registry = FilterRegistry()

    def initializeGL(self):
        """Initializes OpenGL context and ModernGL if supported, otherwise flags fallback."""
        if not self.use_gpu:
            return
            
        try:
            # Bind to existing Qt OpenGL context
            self.ctx = moderngl.create_context()
            self.ctx.enable(moderngl.BLEND)
            self.ctx.blend_func = (moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA)
            
            # Setup background shaders
            self._compile_bg_program()
            
            # Setup background quad vertices (X, Y, U, V)
            bg_vertices = np.array([
                -1.0,  1.0, 0.0, 0.0,
                -1.0, -1.0, 0.0, 1.0,
                 1.0,  1.0, 1.0, 0.0,
                 1.0, -1.0, 1.0, 1.0,
            ], dtype='f4')
            self.bg_vbo = self.ctx.buffer(bg_vertices.tobytes())
            self.bg_vao = self.ctx.simple_vertex_array(
                self.programs["Background"], self.bg_vbo, 'in_vert', 'in_text'
            )
            
            # Setup texture targets (initially dummy, allocated when first frame arrives)
            self.camera_tex = None
            self.overlay_tex = None
            
            # Load registered fragment shaders
            self._compile_effect_programs()
            print("[CameraView] GPU Shader programs compiled successfully.")
            
        except Exception as e:
            print(f"[CameraView Warning] GPU Shader context creation failed: {e}")
            print("[CameraView Info] Switching to CPU fallback rendering pipeline.")
            self.use_gpu = False

    def _compile_bg_program(self):
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

    def _compile_effect_programs(self):
        # 1. Simple Vertex shader for panels
        panel_vert = """
        #version 330 core
        in vec2 in_vert;
        out vec2 vScreenPos;
        uniform vec2 uResolution;
        void main() {
            gl_Position = vec4(in_vert, 0.0, 1.0);
            vScreenPos.x = (in_vert.x + 1.0) * 0.5 * uResolution.x;
            vScreenPos.y = (1.0 - in_vert.y) * 0.5 * uResolution.y; 
        }
        """
        
        # Scans app/shaders/ folder dynamically via FilterRegistry
        for name in self._registry.filters:
            path = self._registry.get_shader_path(name)
            if os.path.exists(path):
                try:
                    frag_src = self._registry.get_shader_source(name)
                    if not frag_src:
                        continue
                        
                    # Prepend perspective Homography warp variables
                    uniforms = """
                    uniform mat3 uHomography;
                    uniform float uAlpha = 1.0;
                    """
                    if "uResolution" not in frag_src:
                        uniforms += "\nuniform vec2 uResolution;"
                    
                    injected = frag_src.replace(
                        "in vec2 vTexCoord;",
                        "in vec2 vScreenPos; in vec2 vTexCoord;"
                    )
                    
                    homography_calculation = """
                    void main() {
                        vec3 proj = uHomography * vec3(vScreenPos, 1.0);
                        vec2 vTexCoord = proj.xy / proj.z;
                        if (vTexCoord.x < 0.0 || vTexCoord.x > 1.0 || vTexCoord.y < 0.0 || vTexCoord.y > 1.0) {
                            discard;
                        }
                    """
                    
                    injected = injected.replace("void main() {", uniforms + "\n" + homography_calculation)
                    injected = replace_texture_samples(injected)
                    injected = injected.replace("fragColor = ", "fragColor = vec4(1.0, 1.0, 1.0, uAlpha) * ")
                    
                    self.programs[name] = self.ctx.program(
                        vertex_shader=panel_vert, fragment_shader=injected
                    )
                except Exception as e:
                    print(f"[CameraView Error] Failed compiling shader '{name}': {e}")
                    
        # 2. Compile PassThrough shader
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
            print(f"[CameraView Error] Failed compiling PassThrough shader: {e}")

    def update_frame_data(self, frame, tracking_data, panels=None):
        """Called by controller thread-safely to schedule canvas updates."""
        self._frame = frame
        self._tracking_data = tracking_data
        self._panels = panels if panels else []
        
        if tracking_data:
            self.virtual_cursor_pos = tracking_data.get("cursor_pos", (0, 0))
            
        self.update()

    def paintGL(self):
        """Draws webcam feed and GLSL warped filters using ModernGL."""
        if not self.use_gpu or self.ctx is None or self._frame is None:
            return
            
        try:
            # Bind Qt's currently active framebuffer
            self.ctx.detect_framebuffer().use()
            
            h, w, c = self._frame.shape
            
            # Setup ModernGL texture allocation dynamically matching frame sizes
            if self.camera_tex is None or self.camera_tex.width != w or self.camera_tex.height != h:
                if self.camera_tex:
                    self.camera_tex.release()
                self.camera_tex = self.ctx.texture((w, h), 3)
                self.camera_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
                self.camera_tex.repeat_x = False
                self.camera_tex.repeat_y = False
                
            self.ctx.clear(0.0, 0.0, 0.0, 1.0)
            
            # 1. Upload camera frame
            rgb_frame = cv2.cvtColor(self._frame, cv2.COLOR_BGR2RGB)
            self.camera_tex.write(rgb_frame.tobytes())
            self.camera_tex.use(0)
            
            # 2. Render background quad
            bg_prog = self.programs["Background"]
            bg_prog["uTexture"].value = 0
            self.bg_vao.render(moderngl.TRIANGLE_STRIP)
            
            # 3. Render homography warped filters
            t_now = time.time()
            disable_filters = False
            try:
                from app.core.service_manager import ServiceManager
                from app.core.interfaces import IConfigService
                config_service = ServiceManager.get(IConfigService)
                disable_filters = config_service.get("disable_filters", False)
            except Exception:
                pass

            for p in self._panels:
                corners = p["corners"]
                effect = "PassThrough" if disable_filters else p["effect"]
                alpha = p["alpha"]
                
                if effect not in self.programs:
                    continue
                    
                prog = self.programs[effect]
                prog["uTexture"].value = 0
                prog["uAlpha"].value = alpha
                prog["uResolution"].value = (w, h)
                
                if "uTime" in prog:
                    prog["uTime"].value = t_now
                    
                # Compute actual perspective homography mapping screen pixels to [0, 1] unit square
                try:
                    src_pts = np.array([
                        [corners[0][0] * w, corners[0][1] * h],
                        [corners[1][0] * w, corners[1][1] * h],
                        [corners[2][0] * w, corners[2][1] * h],
                        [corners[3][0] * w, corners[3][1] * h]
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
                        [1.0 / w, 0.0, 0.0],
                        [0.0, 1.0 / h, 0.0],
                        [0.0, 0.0, 1.0]
                    ], dtype=np.float32)
                    
                prog["uHomography"].value = tuple(H.T.flatten())
                
                # Setup coordinates mapping standard window bounds to clip space
                clip_vertices = np.array([
                    corners[0][0] * 2.0 - 1.0, 1.0 - corners[0][1] * 2.0,  # Top Left
                    corners[3][0] * 2.0 - 1.0, 1.0 - corners[3][1] * 2.0,  # Bottom Left
                    corners[1][0] * 2.0 - 1.0, 1.0 - corners[1][1] * 2.0,  # Top Right
                    corners[2][0] * 2.0 - 1.0, 1.0 - corners[2][1] * 2.0,  # Bottom Right
                ], dtype='f4')
                
                vbo = self.ctx.buffer(clip_vertices.tobytes())
                vao = self.ctx.simple_vertex_array(prog, vbo, 'in_vert')
                vao.render(moderngl.TRIANGLE_STRIP)
                
                vao.release()
                vbo.release()
                
        except Exception as e:
            print(f"[CameraView GL Error] paintGL failed: {e}")

    def paintEvent(self, event):
        if self.use_gpu:
            # Let the default QOpenGLWidget paintEvent trigger paintGL
            super().paintEvent(event)
            
            # Setup GPU QPainter drawing for overlays on top of the GL frame buffer
            painter = QPainter(self)
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)
            
            if self._tracking_data and self.show_skeleton:
                self._draw_hand_skeletons(painter)
                
            if self.draw_virtual_cursor and self._tracking_data:
                present = self._tracking_data.get("hand_present", {})
                if present.get("Right"):
                    self._draw_futuristic_cursor(painter)
                    
            for p in self._panels:
                if p.get("id") == "reality_quad":
                    self._draw_quad_borders(painter, p["corners"])
                    
            if self._tracking_data and "warning_message" in self._tracking_data:
                self._draw_warning_banner(painter, self._tracking_data["warning_message"])
                
            painter.end()
        else:
            # Setup CPU QPainter fallback drawing
            painter = QPainter(self)
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)
            
            # Paint the camera image first on CPU
            if self._frame is not None:
                h, w, c = self._frame.shape
                rgb_frame = cv2.cvtColor(self._frame, cv2.COLOR_BGR2RGB)
                q_img = QImage(rgb_frame.data, w, h, w * c, QImage.Format.Format_RGB888)
                painter.drawImage(self.rect(), q_img)
                
                # Apply CPU Fallback effects
                self._render_cpu_fallback_filters(painter)
                
            # Draw skeletons and HUD overlays on CPU
            if self._tracking_data and self.show_skeleton:
                self._draw_hand_skeletons(painter)
                
            # Draw interactive virtual cursor if active
            if self.draw_virtual_cursor and self._tracking_data:
                present = self._tracking_data.get("hand_present", {})
                if present.get("Right"):
                    self._draw_futuristic_cursor(painter)
                    
            # Draw overlay borders for active filter mode band
            for p in self._panels:
                if p.get("id") == "reality_quad":
                    self._draw_quad_borders(painter, p["corners"])
                    
            # Draw warning banner if present in tracking data (fallback mode)
            if self._tracking_data and "warning_message" in self._tracking_data:
                self._draw_warning_banner(painter, self._tracking_data["warning_message"])
                
            painter.end()

    def _draw_warning_banner(self, painter, message):
        """Draws a premium warning banner at the top of the view."""
        w = self.width()
        banner_h = 40
        
        # Semi-transparent dark red background
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(QColor(180, 20, 30, 220)))
        painter.drawRect(0, 0, w, banner_h)
        
        # Yellow border at the bottom of the banner
        painter.setPen(QPen(QColor(255, 200, 0), 2))
        painter.drawLine(0, banner_h, w, banner_h)
        
        # Yellow bold text
        painter.setPen(QPen(QColor(255, 255, 255)))
        painter.setFont(QFont("Consolas", 11, QFont.Weight.Bold))
        # Center the text
        painter.drawText(QRect(0, 0, w, banner_h), Qt.AlignmentFlag.AlignCenter, message)

    def _render_cpu_fallback_filters(self, painter):
        """CPU fallback warping helper utilizing QPainter's coordinate transformation."""
        t_now = time.time()
        for p in self._panels:
            corners = p["corners"]
            effect = p["effect"]
            alpha = p["alpha"]
            
            # Map normalized corners to local screen coordinates
            w, h = self.width(), self.height()
            pts = [
                (corners[i][0] * w, corners[i][1] * h) for i in range(4)
            ]
            
            # For CPU mode, we can draw a translucent cyan overlay showing where the quad is,
            # or apply OpenCV effects inside a cropped/warped texture.
            # To keep CPU fallback fast and simple, we draw a neon border and apply a color overlay.
            # Drawing neon overlay
            pen = QPen(QColor(0, 255, 255, int(alpha * 255)), 2)
            painter.setPen(pen)
            painter.setBrush(QBrush(QColor(0, 255, 255, int(alpha * 45))))
            
            from PyQt6.QtGui import QPolygonF
            from PyQt6.QtCore import QPointF
            poly = QPolygonF([QPointF(x, y) for x, y in pts])
            painter.drawPolygon(poly)
            
            # Draw label inside polygon
            cx = sum(x for x, y in pts) / 4
            cy = sum(y for x, y in pts) / 4
            painter.setPen(QPen(QColor(255, 255, 255, 200)))
            painter.setFont(QFont("Consolas", 10, QFont.Weight.Bold))
            painter.drawText(int(cx) - 50, int(cy), f"EFFECT: {effect}")

    def _draw_quad_borders(self, painter, corners):
        """Draws glowing perspective borders around the filter quad."""
        w, h = self.width(), self.height()
        pts = [
            (int(corners[i][0] * w), int(corners[i][1] * h)) for i in range(4)
        ]
        
        # Animated neon glow color cycling
        pulse = int(180 + np.sin(time.time() * 6.0) * 75)
        pen = QPen(QColor(0, 255, 255, pulse), 3)
        painter.setPen(pen)
        
        for i in range(4):
            x1, y1 = pts[i]
            x2, y2 = pts[(i + 1) % 4]
            painter.drawLine(x1, y1, x2, y2)

    def _draw_hand_skeletons(self, painter):
        """Draws joint connections and dots representing right and left hands."""
        w, h = self.width(), self.height()
        lms = self._tracking_data.get("smoothed_landmarks", {})
        present = self._tracking_data.get("hand_present", {})
        
        for side in ["Left", "Right"]:
            if present.get(side) and side in lms and len(lms[side]) >= 21:
                side_lms = lms[side]
                color = QColor(0, 255, 0, 200) if side == "Left" else QColor(255, 0, 80, 200) # Green vs Neon Red
                
                # Draw connections using standard mp_hands.HAND_CONNECTIONS
                pen = QPen(color, 2)
                painter.setPen(pen)
                for start, end in mp_hands.HAND_CONNECTIONS:
                    if start in side_lms and end in side_lms:
                        # Direct coordinate scaling, no manual mirroring
                        x1 = int(side_lms[start][0] * w)
                        y1 = int(side_lms[start][1] * h)
                        x2 = int(side_lms[end][0] * w)
                        y2 = int(side_lms[end][1] * h)
                        painter.drawLine(x1, y1, x2, y2)
                        
                # Draw joint circles and landmark IDs
                font = QFont("Consolas", 8)
                painter.setFont(font)
                for idx in range(21):
                    if idx in side_lms:
                        x = int(side_lms[idx][0] * w)
                        y = int(side_lms[idx][1] * h)
                        
                        is_fingertip = idx in [4, 8, 12, 16, 20]
                        size = 7 if is_fingertip else 4
                        
                        painter.setPen(Qt.PenStyle.NoPen)
                        if is_fingertip:
                            painter.setBrush(QBrush(QColor(255, 255, 255, 255)))
                        else:
                            painter.setBrush(QBrush(color))
                        painter.drawEllipse(x - size, y - size, size * 2, size * 2)
                        
                        # Draw landmark ID text
                        painter.setPen(QPen(QColor(255, 255, 255, 220), 1))
                        painter.drawText(x + size + 2, y + 4, str(idx))
                        
                # Draw Palm Center (average of 0, 5, 17)
                if all(j in side_lms for j in [0, 5, 17]):
                    p0 = side_lms[0]
                    p5 = side_lms[5]
                    p17 = side_lms[17]
                    cx = int(((p0[0] + p5[0] + p17[0]) / 3.0) * w)
                    cy = int(((p0[1] + p5[1] + p17[1]) / 3.0) * h)
                    
                    # Draw palm center dot
                    painter.setPen(Qt.PenStyle.NoPen)
                    painter.setBrush(QBrush(QColor(255, 255, 0, 255))) # Yellow palm center
                    painter.drawEllipse(cx - 6, cy - 6, 12, 12)
                    
                # Draw Bounding Box and Hand Label
                # Find min/max coordinates
                xs = [side_lms[idx][0] * w for idx in range(21) if idx in side_lms]
                ys = [side_lms[idx][1] * h for idx in range(21) if idx in side_lms]
                if xs and ys:
                    min_x, max_x = min(xs), max(xs)
                    min_y, max_y = min(ys), max(ys)
                    
                    # Add margin to bounding box
                    margin = 15
                    bx = int(min_x - margin)
                    by = int(min_y - margin)
                    bw = int(max_x - min_x + 2 * margin)
                    bh = int(max_y - min_y + 2 * margin)
                    
                    # Draw bounding box rectangle
                    painter.setPen(QPen(color, 1, Qt.PenStyle.DashLine))
                    painter.setBrush(Qt.BrushStyle.NoBrush)
                    painter.drawRect(bx, by, bw, bh)
                    
                    # Draw Hand Label text near the bounding box top
                    label_text = f"{side} Hand"
                    painter.setPen(QPen(color, 1))
                    painter.setFont(QFont("Consolas", 10, QFont.Weight.Bold))
                    painter.drawText(bx, by - 6, label_text)

    def _draw_futuristic_cursor(self, painter):
        """Draws a premium glowing neon circular cursor representing right index tip selection coordinate."""
        x, y = self.virtual_cursor_pos
        
        # Scale virtual coordinate system coordinates (OpenCV mapping) to local QWidget viewport coordinates
        if self._frame is not None:
            # Map from frame size coordinates to current widget size coordinates
            fh, fw, fc = self._frame.shape
            x = int(x * self.width() / fw)
            y = int(y * self.height() / fh)
            
        # Draw target ring
        pen = QPen(QColor(0, 255, 255, 220), 2)
        painter.setPen(pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawEllipse(x - 12, y - 12, 24, 24)
        
        # Draw central point
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(QColor(0, 255, 255, 255)))
        painter.drawEllipse(x - 3, y - 3, 6, 6)
        
        # Outer pulsing scan lines
        pulse_r = int(16 + np.sin(time.time() * 8.0) * 4)
        pen_glow = QPen(QColor(0, 255, 255, 100), 1)
        painter.setPen(pen_glow)
        painter.drawEllipse(x - pulse_r, y - pulse_r, pulse_r * 2, pulse_r * 2)
        
        # Check click/pinch visual feedback
        if self._tracking_data.get("gesture_states", {}).get("Left", {}).get("pinch"):
            # Left hand pinching draws selection line or indicators
            pen_click = QPen(QColor(0, 255, 120, 255), 3)
            painter.setPen(pen_click)
            painter.drawEllipse(x - 18, y - 18, 36, 36)
            
            # Print feedback text
            painter.setFont(QFont("Consolas", 8, QFont.Weight.Bold))
            painter.drawText(x + 22, y + 4, "CLICK ACTIVE")
            
        # Draw hover dwell selection progress arc
        dwell_progress = self._tracking_data.get("dwell_progress", 0.0)
        if dwell_progress > 0.0:
            span_angle = int(-dwell_progress * 360 * 16)
            pen_progress = QPen(QColor(0, 255, 120, 230), 3)  # Neon green progress ring
            painter.setPen(pen_progress)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawArc(x - 18, y - 18, 36, 36, 90 * 16, span_angle)

    def resizeGL(self, width, height):
        if self.use_gpu and self.ctx:
            # Matches viewport sizing
            pass
            
    def closeEvent(self, event):
        if self.use_gpu and self.ctx:
            # Release buffer array assets
            if self.bg_vao:
                self.bg_vao.release()
            if self.bg_vbo:
                self.bg_vbo.release()
            self.programs.clear()
        super().closeEvent(event)
