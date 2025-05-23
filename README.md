# NSTTS


## submodules

# controlnet
#### running controlnet




# prodiff
#### running prodiff
# 🎤 ProDiff TTS Inference Cheat Sheet

A quick reference for controlling pitch, voice characteristics, length, quality, and more when running ProDiff-based TTS inference.

## 🔧 Key Parameters

| Feature         | Parameter(s)                           | Where to Set                             | Effect                                           | Tips                                                        |
|-----------------|-----------------------------------------|------------------------------------------|--------------------------------------------------|-------------------------------------------------------------|
| 🎵 Pitch         | `pitch_shift_semitones`                | `--hparams`, `prodiff_teacher.yaml`      | Shifts pitch up/down in semitones                | `+2` = higher pitch, `-2` = lower pitch                     |
| 🎤 Voice Profile | `speaker_id`, `speaker_embed`          | Depends on model (multi-speaker only)    | Changes voice (if supported by model)            | For single-speaker models, voice is fixed unless retrained |
| 🎶 Intonation    | Custom `f0` curve                      | via `pitch_utils.py` or melody conditioning | Controls melody / pitch contour              | Requires explicit conditioning or model changes            |
| 🕒 Length        | `speed`, `dur_scale`, `N`, `use_guided_attn` | `--hparams`, YAML config         | Alters speech speed and overall duration         | `N=10` → more steps, smoother & possibly longer output     |
| 🎧 Quality       | `N`, `sigma`, vocoder choice            | `--hparams`, model config                | Affects realism, smoothness, denoising           | Larger `N`, lower `sigma` = better quality, more compute   |
| 🔁 Emphasis      | via `f0` or attention mask              | Custom control mechanisms                | Word stress and prosody                          | Not natively controllable without modification             |
| 📜 Text          | `text='Your sentence here.'`           | `--hparams`                              | Text input to synthesize                         | Use punctuation for natural-sounding prosody               |

---

## 🧪 Example Command

```bash
python inference/ProDiff_teacher.py \
  --config modules/ProDiff/config/prodiff_teacher.yaml \
  --exp_name ProDiff_Teacher \
  --reset \
  --hparams="N=8,text='Hello, how are you?',pitch_shift_semitones=2,sigma=0.5"
