import cv2
import time
import argparse
import threading
from src.tracker import HandTracker
from src.renderer import RealityFilterRenderer
from src.osc_sender import OSCSender
from src.camera import WebcamStream

def main():
    # Parse configuration arguments
    parser = argparse.ArgumentParser(description="Multi-Layer Finger-Controlled Reality Filters")
    parser.add_argument("--camera", type=int, default=0, help="Webcam source index (default: 0)")
    parser.add_argument("--width", type=int, default=1280, help="Webcam and window width (default: 1280)")
    parser.add_argument("--height", type=int, default=720, help="Webcam and window height (default: 720)")
    parser.add_argument("--cpu", action="store_true", help="Force CPU OpenCV rendering pipeline")
    parser.add_argument("--osc-ip", type=str, default="127.0.0.1", help="Target OSC server IP (default: 127.0.0.1)")
    parser.add_argument("--osc-port", type=int, default=9000, help="Target OSC server port (default: 9000)")
    parser.add_argument("--no-osc", action="store_true", help="Disable OSC streaming")
    args = parser.parse_args()

    print("[Info] Starting Multi-Layer Finger-Controlled Reality Filters...")
    print(f"[Info] Target Resolution: {args.width}x{args.height}")
    
    # 1. Initialize Webcam Capture Thread
    cap = WebcamStream(src=args.camera, width=args.width, height=args.height)
    cap.start()
    
    # Check if camera was successfully opened
    if not cap.grabbed:
        print(f"[Error] Could not open video source index {args.camera}")
        cap.stop()
        return

    # Verify actual resolution grabbed from the camera
    actual_w = cap.width
    actual_h = cap.height
    print(f"[Info] Captured webcam resolution: {actual_w}x{actual_h}")

    # 2. Instantiate core sub-systems
    tracker = HandTracker()
    renderer = RealityFilterRenderer(width=actual_w, height=actual_h, force_cpu=args.cpu)
    
    # Enable OSC unless explicitly disabled
    osc_enabled = not args.no_osc
    osc_sender = OSCSender(ip=args.osc_ip, port=args.osc_port, enabled=osc_enabled)

    # 3. Spawn background Hand Tracker processing thread
    tracking_running = True
    
    def tracking_loop():
        nonlocal tracking_running
        last_processed_frame_id = -1
        while tracking_running:
            success, frame, frame_id = cap.read()
            if success and frame is not None:
                if frame_id > last_processed_frame_id:
                    flipped_frame = cv2.flip(frame, 1)
                    tracker.process_frame(flipped_frame)
                    last_processed_frame_id = frame_id
                else:
                    # Idle briefly if no new frame is captured yet to save CPU cycles
                    time.sleep(0.002)
            else:
                time.sleep(0.002)

    tracker_thread = threading.Thread(target=tracking_loop)
    tracker_thread.daemon = True
    tracker_thread.start()

    # 4. Running state variables
    last_unfrozen_frame = None
    fps_time = time.time()
    fps_counter = 0
    fps_current = 0.0
    
    # Create clock if using Pygame/GPU mode
    pygame_clock = None
    if renderer.use_gpu:
        import pygame
        pygame_clock = pygame.time.Clock()
        
    print("[Info] Application initialized. Show your hands to begin!")
    print("[Info] Controls:")
    print("  - Hold Fist: Freeze frame inside reality filters (webcam feed remains active)")
    print("  - Open Palm: Reset spawned filters")
    print("  - Pinch (Index + Thumb): Spawn persistent floating reality filter")
    print("  - Swipe (Fast wrist shake): Cycle filter configurations")
    print("  - Press Q or ESC: Quit application")
    print("  - Press S: Take screenshot of display")

    try:
        running = True
        while running:
            # Check window exit events for CPU fallback mode
            if not renderer.use_gpu:
                # Keyboard events captured by OpenCV waitKey
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27: # 27 is ESC
                    running = False
                elif key == ord('s'):
                    # Save snapshot
                    if last_unfrozen_frame is not None:
                        filename = f"screenshot_{int(time.time())}.png"
                        cv2.imwrite(filename, last_unfrozen_frame)
                        print(f"[Info] Snapshot saved: {filename}")
                elif key == ord('r'):
                    tracker.clear_spawned_panels()
            else:
                # Keyboard events captured by Pygame
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_ESCAPE or event.key == pygame.K_q:
                            running = False
                        elif event.key == pygame.K_s:
                            # Capture back buffer and save
                            filename = f"screenshot_{int(time.time())}.png"
                            pygame.image.save(renderer.screen, filename)
                            print(f"[Info] Snapshot saved: {filename}")
                        elif event.key == pygame.K_r:
                            tracker.clear_spawned_panels()

            # Read latest camera frame asynchronously
            success, frame, _ = cap.read()
            if not success or frame is None:
                continue

            # Fist gesture freeze frame logic:
            is_frozen = tracker.check_freeze()
            if not is_frozen or last_unfrozen_frame is None:
                # Feed is live, cache current frame
                last_unfrozen_frame = frame.copy()
            
            # Fetch active panels
            panels = tracker.get_panels(actual_w, actual_h)
            
            # Send tracking data to TouchDesigner via OSC
            osc_sender.send_tracking_data(tracker)
            
            # Render background frame and apply GPU shaders (or CPU fallbacks).
            render_frame = last_unfrozen_frame if is_frozen else frame
            renderer.render(
                frame=render_frame, 
                panels=panels, 
                tracker_instance=tracker
            )

            # Cap FPS to target 60 FPS
            if pygame_clock:
                pygame_clock.tick(60)
            else:
                time.sleep(1.0 / 60.0)

            # Calculate actual FPS for diagnostic logging
            fps_counter += 1
            if time.time() - fps_time >= 1.0:
                fps_current = fps_counter / (time.time() - fps_time)
                # print(f"FPS: {fps_current:.1f} | Active Panels: {len(panels)}")
                fps_counter = 0
                fps_time = time.time()

    finally:
        print("[Info] Closing capture stream and cleaning up resources...")
        tracking_running = False
        tracker_thread.join(timeout=1.0)
        cap.stop()
        tracker.close()
        renderer.close()
        print("[Info] System shutdown complete.")

if __name__ == "__main__":
    main()
