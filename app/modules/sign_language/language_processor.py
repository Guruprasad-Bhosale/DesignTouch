import time
from collections import deque, Counter

class LanguageProcessor:
    def __init__(self, debounce_frames=12, stable_time_threshold=0.5):
        self._current_text = ""
        self._current_word = ""
        self._last_char = None
        self._char_stable_start = 0.0
        self._debounce_frames = debounce_frames
        self._stable_time_threshold = stable_time_threshold
        
        # Prediction smoothing buffer (majority voting over last 10 frames)
        self.prediction_buffer = deque(maxlen=10)
        self.confidence_threshold = 0.70
        
        # Word suggestion dictionary
        self._dictionary = [
            "HELLO", "WORLD", "YES", "NO", "PLEASE", "THANK", "YOU",
            "GOOD", "MORNING", "SIGN", "LANGUAGE", "VISION", "PYTHON",
            "COMPUTER", "GESTURE", "VERSE", "FILTER", "CAMERA", "HELP", "EXIT"
        ]

    def process_prediction(self, char: str, confidence: float) -> tuple:
        """
        Processes live predictions with majority voting smoothing and temporal stabilization.
        Returns (has_new_letter, current_complete_text, word_suggestions).
        """
        # 1. Apply confidence threshold
        effective_char = char if (char != "None" and confidence >= self.confidence_threshold) else "None"
        self.prediction_buffer.append(effective_char)
        
        # 2. Majority voting over sliding buffer
        if len(self.prediction_buffer) < 5:
            # Wait until buffer has enough frames to avoid startup flickering
            return False, self._get_full_text(), []
            
        counts = Counter(self.prediction_buffer)
        most_common_char, count = counts.most_common(1)[0]
        
        # Enforce majority threshold (e.g. at least 6 out of 10 frames must agree)
        if count >= 6 and most_common_char != "None":
            smoothed_char = most_common_char
        else:
            smoothed_char = "None"
            
        if smoothed_char == "None":
            # Reset stable timing if no clear stable gesture is present
            self._last_char = None
            return False, self._get_full_text(), []
            
        t_now = time.time()
        
        # 3. Temporal Stabilization Check
        if smoothed_char != self._last_char:
            self._last_char = smoothed_char
            self._char_stable_start = t_now
            return False, self._get_full_text(), []
            
        # Check if the character has been held stable for long enough
        elapsed = t_now - self._char_stable_start
        if elapsed >= self._stable_time_threshold:
            # Reset stable timer to prevent repeating the character continuously
            self._char_stable_start = t_now + 1.0  # Cooldown before repeating
            
            # Character confirmed!
            self.add_character(smoothed_char)
            suggestions = self.get_suggestions(self._current_word)
            return True, self._get_full_text(), suggestions
            
        return False, self._get_full_text(), []

    def add_character(self, char: str):
        # Handle special command letters if any
        if char == "SPACE":
            self.commit_word()
        elif char == "BACKSPACE":
            self.backspace()
        else:
            self._current_word += char

    def backspace(self):
        if self._current_word:
            self._current_word = self._current_word[:-1]
        elif self._current_text:
            # Remove last word
            words = self._current_text.strip().split()
            if words:
                words = words[:-1]
                self._current_text = " ".join(words)
                if self._current_text:
                    self._current_text += " "
            else:
                self._current_text = ""

    def commit_word(self):
        if self._current_word:
            # Perform spell check / autocorrect on the current word
            corrected = self.autocorrect_word(self._current_word)
            self._current_text += corrected + " "
            self._current_word = ""
        else:
            # Add plain space
            self._current_text += " "

    def _get_full_text(self) -> str:
        text = self._current_text
        if self._current_word:
            text += self._current_word
        return text

    def autocorrect_word(self, word: str) -> str:
        word = word.upper()
        if word in self._dictionary:
            return word
            
        # Find closest word in dictionary using edit distance
        best_match = word
        best_distance = 999
        
        for dict_word in self._dictionary:
            dist = self._levenshtein_distance(word, dict_word)
            # Threshold: word length / 2 max edits allowed
            if dist < best_distance and dist <= max(2, len(word) // 2):
                best_distance = dist
                best_match = dict_word
                
        return best_match

    def get_suggestions(self, word_prefix: str) -> list:
        if not word_prefix:
            return []
            
        word_prefix = word_prefix.upper()
        matches = [w for w in self._dictionary if w.startswith(word_prefix)]
        return matches[:3]

    def _levenshtein_distance(self, s1: str, s2: str) -> int:
        if len(s1) < len(s2):
            return self._levenshtein_distance(s2, s1)
            
        if len(s2) == 0:
            return len(s1)
            
        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
            
        return previous_row[-1]

    @property
    def current_text(self) -> str:
        return self._current_text

    @property
    def current_word(self) -> str:
        return self._current_word

    def clear(self):
        self._current_text = ""
        self._current_word = ""
        self._last_char = None
        self.prediction_buffer.clear()
