import cv2

class SignLanguageTextOutput:
    def __init__(self):
        pass

    def draw_predictions(self, frame, text: str, current_word: str, char: str, confidence: float, suggestions: list):
        """Draws static alphabet letters and continuous sentence texts on BGR frame."""
        h, w, c = frame.shape
        
        # 1. Draws current letter prediction in top-left
        if char and char != "None":
            # Draw dark background card
            cv2.rectangle(frame, (15, 15), (280, 110), (25, 20, 30), -1)
            cv2.rectangle(frame, (15, 15), (280, 110), (0, 255, 255), 2)
            
            cv2.putText(frame, f"LETTER: {char}", (30, 50),
                        cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 2, lineType=cv2.LINE_AA)
            cv2.putText(frame, f"CONF: {confidence:.2%}", (30, 85),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (100, 255, 100), 1, lineType=cv2.LINE_AA)
                        
        # 2. Draws typed text sentence at bottom
        if text or current_word:
            # Draw continuous output banner
            cv2.rectangle(frame, (30, h - 110), (w - 30, h - 30), (20, 15, 25), -1)
            cv2.rectangle(frame, (30, h - 110), (w - 30, h - 30), (255, 255, 0), 1)
            
            full_text = text
            if current_word:
                full_text = f"{text} [{current_word}]" if text else f"[{current_word}]"
                
            # Draw text
            cv2.putText(frame, f"INTERPRETED: {full_text}", (50, h - 70),
                        cv2.FONT_HERSHEY_DUPLEX, 0.75, (255, 255, 255), 2, lineType=cv2.LINE_AA)
                        
            # Draw suggestions
            if suggestions:
                sug_str = " | ".join(suggestions)
                cv2.putText(frame, f"Suggestions: {sug_str}", (50, h - 45),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1, lineType=cv2.LINE_AA)
                            
        return frame
