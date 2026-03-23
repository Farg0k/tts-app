import glob
import os
import re
import gradio as gr
import torch
import time
from verbalizer import Verbalizer
from ipa_uk import ipa
from unicodedata import normalize
from styletts2_inference.models import StyleTTS2
from ukrainian_word_stress import Stressifier, StressSymbol

# --- НАЛАШТУВАННЯ ---
stressify = Stressifier()
#device = 'cpu'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
verbalizer = Verbalizer()

# --- ЗАВАНТАЖЕННЯ МОДЕЛЕЙ ---
print("Завантаження моделей...")

# Single speaker
single_model = StyleTTS2(hf_path='patriotyk/styletts2_ukrainian_single', device=device)
single_style = torch.load('filatov.pt', map_location=device)

# Multi speaker (HiFiGAN)
multi_model = StyleTTS2(hf_path='patriotyk/styletts2_ukrainian_multispeaker', device=device)

# Multi speaker (iSTFTNet) — швидший вокодер
istft_model = StyleTTS2(hf_path='patriotyk/styletts2_ukrainian_multispeaker_istftnet', device=device)

# --- ГОЛОСИ ДЛЯ MULTI (HiFiGAN) ---
multi_prompts_dir = 'voices'
multi_audio_files = sorted(glob.glob(os.path.join(multi_prompts_dir, '*.pt')))
multi_prompts_list = [os.path.basename(f).replace('.pt', '') for f in multi_audio_files]
multi_styles = {}
for audio_path in multi_audio_files:
    name = os.path.basename(audio_path).replace('.pt', '')
    multi_styles[name] = torch.load(audio_path, map_location=device)
    print('Завантажено (multi):', name)

# --- ГОЛОСИ ДЛЯ ISTFT ---
# Якщо є окрема папка voices_istft — використовуємо її,
# інакше використовуємо ті самі голоси що й для multi
istft_prompts_dir = 'voices_istft' if os.path.isdir('voices_istft') else 'voices'
istft_audio_files = sorted(glob.glob(os.path.join(istft_prompts_dir, '*.pt')))
istft_prompts_list = [os.path.basename(f).replace('.pt', '') for f in istft_audio_files]
istft_styles = {}
for audio_path in istft_audio_files:
    name = os.path.basename(audio_path).replace('.pt', '')
    istft_styles[name] = torch.load(audio_path, map_location=device)
    print('Завантажено (istft):', name)

print("Всі моделі завантажено.")


# --- УТИЛІТИ ---

SILENCE_RE = r'\{\{SILENCE_\d+(?:_\d+)?\}\}'

def split_to_parts(text, group=True):
    text = re.sub(r'(\w+[^.,!:?\-])\n', r'\1. ', text)
    text = text.replace('\n', ' ')
    split_symbols = '.?!:'
    parts = ['']
    index = 0
    last = len(text) - 1
    for i, s in enumerate(text):
        parts[index] += s
        if s in split_symbols and i < last and text[i + 1] == ' ':
            if group and len(parts[index]) <= 20:
                continue
            index += 1
            parts.append('')
    return parts


def verbalize(text):
    # Розбиваємо по {{SILENCE_X_Y}}, вербалізуємо тільки текстові частини, складаємо назад
    segments = re.split(f'({SILENCE_RE})', text)
    result = ''
    for seg in segments:
        if re.fullmatch(SILENCE_RE, seg):
            result += seg + ' '
        else:
            parts = split_to_parts(seg, group=False)
            for part in parts:
                if part.strip():
                    result += verbalizer.process_text(part.strip())[0] + ' '
    return result.strip()


def synthesize(model_name, text, speed, voice_name=None, progress=gr.Progress()):
    if not text.strip():
        raise gr.Error("Введіть текст")

    start_time = time.time()

    if model_name == 'multi':
        model_obj = multi_model
        styles_dict = multi_styles
    elif model_name == 'istft':
        model_obj = istft_model
        styles_dict = istft_styles
    else:
        model_obj = single_model
        styles_dict = None

    result_wav = []
    sample_rate = 24000

    # Спочатку ділимо по тегах {{SILENCE_X_Y}}, зберігаючи самі теги
    raw_segments = re.split(f'({SILENCE_RE})', text)

    # Кожен сегмент або тег тиші, або звичайний текст який треба ще розбити
    segments = []
    for seg in raw_segments:
        if re.fullmatch(SILENCE_RE, seg.strip()):
            segments.append(seg.strip())
        else:
            parts = split_to_parts(seg)
            segments.extend(parts)

    for t in progress.tqdm(segments):
        t = t.strip().replace('"', '')
        if not t:
            continue

        # --- ТИША ---
        silence_match = re.fullmatch(r'\{\{SILENCE_(\d+)(?:_(\d+))?\}\}', t)
        if silence_match:
            # {{SILENCE_2}} → 2с, {{SILENCE_1_5}} → 1.5с
            int_part = silence_match.group(1)
            frac_part = silence_match.group(2)
            duration = float(f'{int_part}.{frac_part}' if frac_part else int_part)
            result_wav.append(torch.zeros(int(duration * sample_rate), device=device))
            continue

        # --- ОБРОБКА ТЕКСТУ ---
        t = t.replace('+', StressSymbol.CombiningAcuteAccent)
        t = normalize('NFKC', t)
        t = re.sub(r'[᠆‐‑‒–—―⁻₋−⸺⸻]', '-', t)

        # Пропускаємо якщо немає літер
        if not re.search(r'\w', t):
            continue

        if t[-1] not in '.?!:-':
            t += '.'

        ps = ipa(stressify(t))
        if ps:
            tokens = model_obj.tokenizer.encode(ps)
            if model_name == 'single':
                style = single_style
            else:
                style = styles_dict[voice_name] if voice_name and voice_name in styles_dict else list(styles_dict.values())[0]
            wav = model_obj(tokens, speed=speed, s_prev=style)
            result_wav.append(wav)

    final_wav = torch.concatenate(result_wav).cpu().numpy()
    final_wav = (final_wav * 32767).clip(-32768, 32767).astype('int16')
    elapsed = time.time() - start_time
    duration_sec = len(final_wav) / 24000
    rtf = duration_sec / elapsed if elapsed > 0 else 0
    stats = f"⏱ **Час:** {elapsed:.2f}с | 🔊 **Аудіо:** {duration_sec:.2f}с | ⚡ **RTF:** {rtf:.2f}x"
    return (24000, final_wav), stats


# --- ЗАПУСК ---
if __name__ == "__main__":
    with gr.Blocks(title="StyleTTS2 ukrainian") as demo:
        gr.Markdown("# StyleTTS2 Ukrainian Local")

        with gr.Tab("Multi speaker (HiFiGAN)"):
            with gr.Row():
                with gr.Column():
                    input_m = gr.Text(label='Текст', lines=5)
                    btn_v_m = gr.Button("Вербалізувати")
                    speed_m = gr.Slider(0.7, 1.3, 1.0, label='Швидкість')
                    speaker_m = gr.Dropdown(choices=multi_prompts_list, value=multi_prompts_list[0], label="Голос")
                with gr.Column():
                    out_m = gr.Audio(label="Аудіо")
                    stats_m = gr.Markdown("...")
                    btn_m = gr.Button("Синтезувати")
            btn_v_m.click(verbalize, inputs=[input_m], outputs=[input_m])
            btn_m.click(
                synthesize,
                inputs=[gr.Text(value='multi', visible=False), input_m, speed_m, speaker_m],
                outputs=[out_m, stats_m]
            )

        with gr.Tab("Multi speaker (iSTFTNet)"):
            gr.Markdown("*Швидший вокодер — менше часу на синтез*")
            with gr.Row():
                with gr.Column():
                    input_i = gr.Text(label='Текст', lines=5)
                    btn_v_i = gr.Button("Вербалізувати")
                    speed_i = gr.Slider(0.7, 1.3, 1.0, label='Швидкість')
                    speaker_i = gr.Dropdown(choices=istft_prompts_list, value=istft_prompts_list[0], label="Голос")
                with gr.Column():
                    out_i = gr.Audio(label="Аудіо")
                    stats_i = gr.Markdown("...")
                    btn_i = gr.Button("Синтезувати")
            btn_v_i.click(verbalize, inputs=[input_i], outputs=[input_i])
            btn_i.click(
                synthesize,
                inputs=[gr.Text(value='istft', visible=False), input_i, speed_i, speaker_i],
                outputs=[out_i, stats_i]
            )

        with gr.Tab("Single speaker"):
            with gr.Row():
                with gr.Column():
                    input_s = gr.Text(label='Текст', lines=5)
                    btn_v_s = gr.Button("Вербалізувати")
                    speed_s = gr.Slider(0.7, 1.3, 1.0, label='Швидкість')
                with gr.Column():
                    out_s = gr.Audio(label="Аудіо")
                    stats_s = gr.Markdown("...")
                    btn_s = gr.Button("Синтезувати")
            btn_v_s.click(verbalize, inputs=[input_s], outputs=[input_s])
            btn_s.click(
                synthesize,
                inputs=[gr.Text(value='single', visible=False), input_s, speed_s],
                outputs=[out_s, stats_s]
            )

        # --- API ендпоінти ---
        # POST http://127.0.0.1:7860/api/voices     → {"voices": [...]}
        # POST http://127.0.0.1:7860/api/verbalize  → вербалізований текст
        voices_out = gr.JSON(visible=False)
        gr.Button(visible=False).click(fn=lambda: {"voices": multi_prompts_list}, inputs=[], outputs=voices_out, api_name="voices")

        verbalize_in = gr.Text(visible=False)
        verbalize_out = gr.Text(visible=False)
        gr.Button(visible=False).click(fn=verbalize, inputs=[verbalize_in], outputs=[verbalize_out], api_name="verbalize")

    if device == 'cuda':
        demo.launch(share=True)
    else:
        demo.launch(server_name="127.0.0.1", server_port=7860)
