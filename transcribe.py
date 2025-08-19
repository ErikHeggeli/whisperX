import torch
import whisperx
from pyannote.audio import Pipeline
import whisperx.offline_diarize

# Change for actual path
base_path = "/models/whisperx/"

DEFAULT_ALIGN_MODELS_TORCH = {
    "en": "WAV2VEC2_ASR_BASE_960H",
    "fr": "VOXPOPULI_ASR_BASE_10K_FR",
    "de": "VOXPOPULI_ASR_BASE_10K_DE",
    "es": "VOXPOPULI_ASR_BASE_10K_ES",
    "it": "VOXPOPULI_ASR_BASE_10K_IT",
}

DEFAULT_ALIGN_MODELS_HF = {
    "ru": "jonatasgrosman/wav2vec2-large-xlsr-53-russian",
    "fi": "jonatasgrosman/wav2vec2-large-xlsr-53-finnish",
    "da": "saattrupdan/wav2vec2-xls-r-300m-ftspeech",
    "no": "NbAiLab/nb-wav2vec2-1b-bokmaal-v2",
    "nn": "NbAiLab/nb-wav2vec2-1b-nynorsk",
}

def get_model(model_name, language):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f'Using device "{device}"')

    if language == "no" or language == "nn":
        model = whisperx.load_model("whisperx/models/whisperx/NbAiLab/nb-whisper-small", device, local_files_only=True, compute_type='int8')
        '''
        print(f'Using language "{language}"')
        if model_name == "small":
            model = whisperx.load_model(base_path + "NbAiLab/nb-whisper-small", device, local_files_only=True)
        if model_name == "medium":
            model = whisperx.load_model(base_path + "NbAiLab/nb-whisper-medium", device, local_files_only=True)
        if model_name == "large":
            model = whisperx.load_model(base_path + "NbAiLab/nb-whisper-large", device, local_files_only=True)
        else:
            model = whisperx.load_model(base_path + "NbAiLab/nb-whisper-medium", device, local_files_only=True)
    elif language == "sm": # Testing Sami model (alias sv)
            model = whisper.load_model(base_path + "NbAiLab/whisper-large-sme/bed43f50f06fd0db81c1009d7d9cbc2c595c5f7f6a6278e137410fea92d15f28/whisper-large-sme.pt", device)
            print("Using Sami model")
    else:
        print(f'Using multilanguage model for "{language}"')
        if model_name == "small":
            model = whisperx.load_model(base_path + "Systran/faster-whisper-small", device, local_files_only=True)
        if model_name == "medium":
            model = whisperx.load_model(base_path + "Systran/faster-whisper-medium", device, local_files_only=True)
        if model_name == "large":
            model = whisperx.load_model(base_path + "Systran/faster-whisper-large-v3", device, local_files_only=True)
        else:
            model = whisperx.load_model(base_path + "Systran/faster-whisper-medium", device, local_files_only=True)
        '''
    return model

def align(audio, result, device):
    language_code = result["language"]
    print(f'Aligning segments for language: "{language_code}"')

    model_a, metadata = whisperx.load_align_model(language_code=language_code, device=device, model_dir=base_path + "alignment/")
    return whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)

def diarize(audio,result):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    diarize_model = whisperx.offline_diarize.OfflineDiarizationPipeline(config_path="whisperx/pyannote_diarization_config.yaml", device=device)
    #diarize_model(audio)
    #assign labels
    #diarize_model = whisperx.DiarizationPipeline()
    # add min/max number of speakers if known
    diarize_segments = diarize_model(audio)
    # diarize_model(audio, min_speakers=min_speakers, max_speakers=max_speakers)

    return whisperx.assign_word_speakers(diarize_segments, result)

def asr(audio, language, model_name="medium"):
    batch_size = 32
    model = get_model(model_name, language)
    if language == "sm":
        return model.transcribe(audio)
    return model.transcribe(audio, batch_size=batch_size)

def transcribe(filename, language, model_name="medium"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    audio = whisperx.load_audio(filename)
    result = asr(audio, language, model_name)
    if language in ["en", "de", "es", "fr", "it"]:
        result = align(audio, result, device)
    #result = diarize(audio,result)
    return result

if __name__ == "__main__":

    result = transcribe("audio_king.mp3", "no")
    #result = align(result["segments"])
    result = diarize("audio_king.mp3", result)
    print(result)