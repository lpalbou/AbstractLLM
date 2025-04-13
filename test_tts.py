from abstractllm.providers.huggingface.tts_pipeline import TTSPipeline
from abstractllm.providers.huggingface.model_types import ModelConfig, ModelArchitecture
from abstractllm.output.audio import TTSOutputHandler
from abstractllm.media.text import TextInput
import os

def main():
    # Create output directory if it doesn't exist
    output_dir = "./audio_output"
    os.makedirs(output_dir, exist_ok=True)
    
    print("Initializing TTS pipeline...")
    pipeline = TTSPipeline()
    
    # Create basic config
    config = ModelConfig(
        architecture=ModelArchitecture.TEXT_TO_SPEECH,
        trust_remote_code=True,
    )
    pipeline.device = "cpu"
    
    print(f"Using device: {pipeline.device}")
    
    print("Loading SpeechT5 model...")
    pipeline.load("microsoft/speecht5_tts", config)
    print("Model loaded successfully!")
    
    # Create handler with basic settings
    print(f"Setting up audio handler (output dir: {output_dir})")
    handler = TTSOutputHandler(
        pipeline,
        playback=True,
        save=True,
        output_dir=output_dir
    )
    
    # Test with a simple predefined text
    test_text = "Hello! This is a test of the text to speech system."
    test_text = "Why don't scientists trust atoms? Because they make up everything!"
    print(f"\nConverting text to speech: {test_text}")
    
    try:
        handler.handle(test_text)
        print("\nAudio generation and playback completed successfully!")
        
        # Check if file was created
        files = os.listdir(output_dir)
        if files:
            print(f"Audio files generated: {files}")
        else:
            print("Warning: No audio files were generated in the output directory")
            
    except Exception as e:
        print(f"\nError during TTS processing: {e}")
        raise  # Re-raise to see full traceback

if __name__ == "__main__":
    main() 