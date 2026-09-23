import argparse
from typing import Union
from pathlib import Path
import re
import torch
import torchaudio
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    WhisperFeatureExtractor,
)

WHISPER_FEAT_CFG = {
    "chunk_length": 30,
    "feature_extractor_type": "WhisperFeatureExtractor",
    "feature_size": 128,
    "hop_length": 160,
    "n_fft": 400,
    "n_samples": 480000,
    "nb_max_frames": 3000,
    "padding_side": "right",
    "padding_value": 0.0,
    "processor_class": "WhisperProcessor",
    "return_attention_mask": False,
    "sampling_rate": 16000,
}


def get_audio_token_length(seconds, merge_factor=2):
    def get_T_after_cnn(L_in, dilation=1):
        for padding, kernel_size, stride in eval("[(1,3,1)] + [(1,3,2)] "):
            L_out = L_in + 2 * padding - dilation * (kernel_size - 1) - 1
            L_out = 1 + L_out // stride
            L_in = L_out
        return L_out

    mel_len = int(seconds * 100)
    audio_len_after_cnn = get_T_after_cnn(mel_len)
    audio_token_num = (audio_len_after_cnn - merge_factor) // merge_factor + 1

    # TODO: current whisper model can't process longer sequence, maybe cut chunk in the future
    audio_token_num = min(audio_token_num, 1500 // merge_factor)

    return audio_token_num


class GLMASR:
    def __init__(self, ckpt_path, device="cuda", max_new_tokens=128) -> None:
        self.max_new_tokens = max_new_tokens
        self.tokenizer = AutoTokenizer.from_pretrained(ckpt_path)
        self.feature_extractor = WhisperFeatureExtractor(**WHISPER_FEAT_CFG)
        
        config = AutoConfig.from_pretrained(ckpt_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            ckpt_path,
            config=config,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        ).to(device)
        model.eval()
        self.config = config
        self.model = model
        self.device = device
        
    def build_prompt(self, audio: Union[Path, str, torch.Tensor], chunk_seconds: int = 30):
        if isinstance(audio, Path):
            audio = str(audio)
        if isinstance(audio, str):
            audio, sr = torchaudio.load(audio)
            audio = torch.mean(audio, dim=0, keepdim=True)
            if sr != self.feature_extractor.sampling_rate:
                audio = torchaudio.transforms.Resample(sr, self.feature_extractor.sampling_rate)(audio)
        
        tokens = []
        tokens += self.tokenizer.encode("<|user|>")
        tokens += self.tokenizer.encode("\n")

        audios = []
        audio_offsets = []
        audio_length = []
        chunk_size = chunk_seconds * self.feature_extractor.sampling_rate
        for start in range(0, audio.shape[1], chunk_size):
            chunk = audio[:, start : start + chunk_size]
            mel = self.feature_extractor(
                chunk.cpu().numpy(),
                sampling_rate=self.feature_extractor.sampling_rate,
                return_tensors="pt",
                padding="max_length",
            )["input_features"]
            audios.append(mel)
            seconds = chunk.shape[1] / self.feature_extractor.sampling_rate
            num_tokens = get_audio_token_length(seconds, self.config.merge_factor)
            tokens += self.tokenizer.encode("<|begin_of_audio|>")
            audio_offsets.append(len(tokens))
            tokens += [0] * num_tokens
            tokens += self.tokenizer.encode("<|end_of_audio|>")
            audio_length.append(num_tokens)

        if not audios:
            raise ValueError("音频内容为空或加载失败。")

        tokens += self.tokenizer.encode("<|user|>")
        tokens += self.tokenizer.encode("\nPlease transcribe this audio into text")

        tokens += self.tokenizer.encode("<|assistant|>")
        tokens += self.tokenizer.encode("\n")

        batch = {
            "input_ids": torch.tensor([tokens], dtype=torch.long),
            "audios": torch.cat(audios, dim=0),
            "audio_offsets": [audio_offsets],
            "audio_length": [audio_length],
            "attention_mask": torch.ones(1, len(tokens), dtype=torch.long),
        }
        return batch
            
    def prepare_inputs(self, batch):
        tokens = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)
        audios = batch["audios"].to(self.device)
        model_inputs = {
            "inputs": tokens,
            "attention_mask": attention_mask,
            "audios": audios.to(torch.bfloat16),
            "audio_offsets": batch["audio_offsets"],
            "audio_length": batch["audio_length"],
        }
        return model_inputs, tokens.size(1)
    
    def inference(self, audio: Union[Path, str, torch.Tensor]):
        """ audio, BxL
        """
        batch = self.build_prompt(audio)
        model_inputs, prompt_len = self.prepare_inputs(batch)
        
        with torch.inference_mode():
            batch_generated = self.model.generate(
                **model_inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
            )
        
        results = []
        for generated in batch_generated:
            transcript_ids = generated[prompt_len:].cpu().tolist()
            transcript = self.tokenizer.decode(transcript_ids, skip_special_tokens=True).strip()
            results.append(transcript)
        return results
    
    def fix_prompt(self, prompts, audios):
        fixed_prompts = []
        for prompt, audio in zip(prompts, audios):
            if audio.ndim == 1:
                audio = audio.unsqueeze(0)
            asr_result = self.inference(audio)[0]
            asr_result = asr_result if asr_result is not None else ''
            fixed_prompt = re.sub(r"<S>\s*(.*)<E>", f"<S> {asr_result} <E>", str(prompt), flags=re.S)
            fixed_prompts.append(fixed_prompt)
        return fixed_prompts
            
            
def transcribe(
    checkpoint_dir: Path,
    audio_path: Path,
    tokenizer_path: str,
    max_new_tokens: int,
    device: str,
):
    glmasr = GLMASR(checkpoint_dir, device, max_new_tokens)
    results = glmasr.inference(audio_path)
    
    for i, result in enumerate(results):
        print(f"[{i+1}/{len(results)}] ----------")
        print(result or "[Empty transcription]")
    return results[0]


def main():
    parser = argparse.ArgumentParser(description="Minimal ASR transcription demo.")
    parser.add_argument(
        "--checkpoint_dir", type=str, default=str(Path(__file__).parent)
    )
    parser.add_argument("--audio", type=str, required=True, help="Path to audio file.")
    parser.add_argument(
        "--tokenizer_path",
        type=str,
        default=None,
        help="Tokenizer directory (defaults to checkpoint dir when omitted).",
    )
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = parser.parse_args()

    return transcribe(
        checkpoint_dir=Path(args.checkpoint_dir),
        audio_path=Path(args.audio),
        tokenizer_path=args.tokenizer_path,
        max_new_tokens=args.max_new_tokens,
        device=args.device,
    )


if __name__ == "__main__":
    
    
    asr_result = main()
    
    prompt = "The person in the video is seated on a black chair with their hands crossed over their knees, holding a blue pen in their left hand and gripping the pen tip with their right hand. They are wearing a watch on their left wrist and a ring on one of their fingers. Initially, they slightly raise their head, then lower it again, appearing to be in thought or preparing to speak. Subsequently, they lift their head once more, looking towards the camera, with their mouth slightly open as if speaking or explaining something. Throughout the sequence, their movements are delicate and coherent, reflecting a state of concentration and contemplation. The person says, <S> Hello everybody, I'm here talking today with Sarah Stockton. Sarah is a therapist and she was interviewed for Matt Walsh's documentary,<E> <AUDCAP>A male speaker, likely middle-aged, with a clear, standard American accent, speaks at a normal, deliberate pace. His voice is in the medium pitch range with a consistent, steady tone and controlled rhythm. The prosody is relatively flat and even, without strong emotional inflection, conveying a calm and informative quality. There's a hint of reverberation in the recording, suggesting a room environment.<ENDAUDCAP>"
    
    #fixed_prompt = re.sub(r"<S>\s*(.*)<E>", "123", prompt, flags=re.S)
    fixed_prompt = re.sub(r"<S>\s*(.*)<E>", f"<S> {asr_result} <E>", str(prompt), flags=re.S)
    print(f"[ORI] {prompt} \n" + f"[FIX] {fixed_prompt}")
