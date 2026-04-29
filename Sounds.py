import time
import os
import sounddevice as sd
import soundfile as sf

# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------
SOUNDS_DIR = "Sounds"

# Map the states to their respective audio files. 
# Update the extensions if you are using .flac or .ogg instead of .wav
STATE_SOUNDS = {
    "RAINBOW": os.path.join(SOUNDS_DIR, "RAINBOW.wav"),
    "LEAK": os.path.join(SOUNDS_DIR, "LEAK.wav"),
    "DONE": os.path.join(SOUNDS_DIR, "DONE.wav")
}

# ---------------------------------------------------------
# Audio Functions
# ---------------------------------------------------------
def list_output_devices():
    """Prints all available audio devices and their IDs."""
    print("--- Available Audio Devices ---")
    print(sd.query_devices())
    print("-------------------------------\n")
    print("Look for the ID number (far left column) of your preferred output device.\n")

def play_sound_for_state(state, output_device_id=None):
    """Plays the audio file associated with a given state."""
    if state not in STATE_SOUNDS:
        print(f"Warning: No sound mapped for state '{state}'")
        return

    filepath = STATE_SOUNDS[state]

    if not os.path.exists(filepath):
        print(f"Error: Could not find audio file at {filepath}")
        return

    try:
        # Read the audio data
        data, samplerate = sf.read(filepath)

        # sd.play is non-blocking. The code will continue executing immediately.
        # If output_device_id is None, it uses the system default.
        sd.play(data, samplerate, device=output_device_id)

        print(f"[{time.strftime('%H:%M:%S')}] Played audio for state: {state}")

    except Exception as e:
        print(f"Failed to play audio for {state}: {e}")

# ---------------------------------------------------------
# Test Loop
# ---------------------------------------------------------
def run_test_sequence(output_device_id=None):
    """Cycles through the defined states every 5 seconds."""
    states = ["RAINBOW", "LEAK", "DONE"]
    current_index = 0

    print("Starting 5-second state cycle test. Press Ctrl+C to exit.")

    try:
        while True:
            current_state = states[current_index]
            
            print(f"\n--- State changed to: {current_state} ---")
            play_sound_for_state(current_state, output_device_id)

            # Wait 5 seconds before switching to the next state
            time.sleep(5.0)

            # Iterate to the next state, looping back to 0 at the end of the list
            current_index = (current_index + 1) % len(states)

    except KeyboardInterrupt:
        print("\nTest stopped by user.")
        sd.stop() # Immediately halt any audio currently playing

# ---------------------------------------------------------
# Main Execution
# ---------------------------------------------------------
if __name__ == "__main__":
    # Step 1: Run this once to see the list in your console.
    list_output_devices()

    # Step 2: Set this to the integer ID of your desired output device from the list above.
    # Leave as None to just use the default system speakers.
    TARGET_DEVICE_ID = None 

    # Step 3: Run the test cycle
    run_test_sequence(TARGET_DEVICE_ID)