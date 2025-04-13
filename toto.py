from abstractllm import create_llm
from abstractllm.providers.huggingface.tts_pipeline import TTSPipeline
from abstractllm.providers.huggingface.model_types import ModelConfig, ModelArchitecture
from abstractllm.output.audio import TTSOutputHandler
from abstractllm.media.text import TextInput
import os

def main():

    # Generate a joke using LLM
    print("Generating joke...")
    llm = create_llm("ollama", model="phi4-mini:latest", temperature=0.5)
    joke = llm.generate("Tell me a very short story.")
    print(f"Joke generated: {joke}")
    

    # Create output directory if it doesn't exist
    output_dir = "./audio_output"
    os.makedirs(output_dir, exist_ok=True)
    
    print("Initializing TTS pipeline...")
    pipeline = TTSPipeline()
    
    # Create basic config
    config = ModelConfig(
        architecture=ModelArchitecture.TEXT_TO_SPEECH,
        trust_remote_code=True,
        device_map="cpu"
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
    
    # Convert joke to speech
    print(f"\nConverting text to speech: {joke}")
    
    try:
        handler.handle(joke)
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
