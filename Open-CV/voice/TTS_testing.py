from TTS.api import TTS

def tts_coqui(text, output_path="output.wav"):
    """Highest quality open-source TTS"""
    # Initialize the model
    tts = TTS("tts_models/en/ljspeech/tacotron2-DDC")
    
    # Generate speech
    tts.tts_to_file(text=text, file_path=output_path)
    return output_path

# Usage
tts_coqui("This is high quality neural speech synthesis")