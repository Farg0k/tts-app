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
ACCENT_MASK_RE = r'\{\{ACCENT_MASK_\d+\}\}'
SILENCE_RE = r'\{\{SILENCE_\d+(?:_\d+)?\}\}'
VOICE_TAG_RE = r'\{\{VOICE:([^}]+)\}\}(.*?)\{\{/VOICE\}\}'
VOICE_PAUSE_SAMPLES = int(0.5 * 24000)  # 0.5 сек після {{/VOICE}}


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
    pattern = f'({VOICE_TAG_RE}|{SILENCE_RE}|{ACCENT_MASK_RE})'
    segments = re.split(pattern, text, flags=re.DOTALL)
    result = ''
    i = 0
    while i < len(segments):
        seg = segments[i]
        if not seg:
            i += 1
            continue
        if re.fullmatch(SILENCE_RE, seg.strip()):
            result += seg.strip() + ' '
            i += 1
            continue
        if re.fullmatch(ACCENT_MASK_RE, seg.strip()):
            result += seg.strip() + ' '
            i += 1
            continue
        voice_match = re.fullmatch(VOICE_TAG_RE, seg.strip(), flags=re.DOTALL)
        if voice_match:
            voice_name = voice_match.group(1)
            inner_text = voice_match.group(2)
            parts = split_to_parts(inner_text, group=False)
            verbalized_inner = ''
            for part in parts:
                if part.strip():
                    verbalized_inner += verbalizer.process_text(part.strip())[0] + ' '
            result += f'{{{{VOICE:{voice_name}}}}}{verbalized_inner.strip()}{{{{/VOICE}}}} '
            i += 1
            continue
        parts = split_to_parts(seg, group=False)
        for part in parts:
            if part.strip():
                result += verbalizer.process_text(part.strip())[0] + ' '
        i += 1
    return result.strip()


def parse_segments(text, narrator_voice, styles_dict):
    """
    Розбиває текст на сегменти: [(текст, голос, додати_паузу_після)]
    - Текст без тегів → голос оповідача
    - {{VOICE:name}}...{{/VOICE}} → голос name, після — пауза 0.5с
    - {{SILENCE_X}} → None (спеціальний тег тиші)
    """
    segments = []
    remaining = text

    # Регулярка для пошуку VOICE тегів і SILENCE тегів
    combined_re = re.compile(
        r'(\{\{VOICE:([^}]+)\}\}(.*?)\{\{/VOICE\}\})|(\{\{SILENCE_(\d+)(?:_(\d+))?\}\})',
        re.DOTALL
    )

    last_end = 0
    for m in combined_re.finditer(remaining):
        # Текст до матчу — оповідач
        before = remaining[last_end:m.start()].strip()
        if before:
            for part in split_to_parts(before):
                if part.strip():
                    segments.append(('text', part.strip(), narrator_voice, False))

        if m.group(1):  # VOICE тег
            voice_name = m.group(2).strip()
            inner = m.group(3).strip()
            # Перевіряємо чи є такий голос, інакше fallback на оповідача
            if voice_name not in styles_dict:
                print(f"⚠️ Голос '{voice_name}' не знайдено, використовую оповідача")
                voice_name = narrator_voice
            for part in split_to_parts(inner):
                if part.strip():
                    segments.append(('text', part.strip(), voice_name, False))
            # Пауза після {{/VOICE}}
            #segments.append(('pause', None, None, False))

        elif m.group(4):  # SILENCE тег
            int_part = m.group(5)
            frac_part = m.group(6)
            duration = float(f'{int_part}.{frac_part}' if frac_part else int_part)
            segments.append(('silence', duration, None, False))

        last_end = m.end()

    # Залишок тексту після останнього тегу
    after = remaining[last_end:].strip()
    if after:
        for part in split_to_parts(after):
            if part.strip():
                segments.append(('text', part.strip(), narrator_voice, False))

    return segments


def synthesize_text_part(model_obj, model_name, text, style, speed, device):
    """Синтезує один текстовий сегмент, повертає список wav тензорів."""
    result = []
    t = text.strip().replace('"', '')
    if not t:
        return result

    t = t.replace('+', StressSymbol.CombiningAcuteAccent)
    t = normalize('NFKC', t)
    t = re.sub(r'[᠆‐‑‒–—―⁻₋−⸺⸻]', '-', t)

    if not re.search(r'\w', t):
        return result

    if t[-1] not in '.?!:-':
        t += '.'

    t = re.sub(r' - ', ': ', t)
    t = stressify(t)
    ps = ipa(t)
    if not ps:
        return result

    tokens = model_obj.tokenizer.encode(ps)
    chunks = [tokens[i:i + 480] for i in range(0, len(tokens), 480)]

    for tok_chunk in chunks:
        wav = model_obj(tok_chunk, speed=speed, s_prev=style)
        result.append(wav)

    return result


def synthesize(model_name, text, speed, voice_name=None, progress=gr.Progress()):
    if not text.strip():
        raise gr.Error("Введіть текст")

    start_time = time.time()
    sample_rate = 24000

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

    # Single speaker — без підтримки VOICE тегів
    if model_name == 'single':
        raw_segments = re.split(f'({SILENCE_RE})', text)
        segments_flat = []
        for seg in raw_segments:
            if re.fullmatch(SILENCE_RE, seg.strip()):
                segments_flat.append(seg.strip())
            else:
                segments_flat.extend(split_to_parts(seg))

        for t in progress.tqdm(segments_flat):
            t = t.strip().replace('"', '')
            if not t:
                continue
            silence_match = re.fullmatch(r'\{\{SILENCE_(\d+)(?:_(\d+))?\}\}', t)
            if silence_match:
                int_part = silence_match.group(1)
                frac_part = silence_match.group(2)
                duration = float(f'{int_part}.{frac_part}' if frac_part else int_part)
                result_wav.append(torch.zeros(int(duration * sample_rate), device=device))
                continue
            wavs = synthesize_text_part(model_obj, model_name, t, single_style, speed, device)
            result_wav.extend(wavs)

    else:
        # Multi speaker — підтримка {{VOICE:name}} тегів
        narrator_style = styles_dict.get(voice_name, list(styles_dict.values())[0])

        segments = parse_segments(text, voice_name, styles_dict)

        for seg in progress.tqdm(segments):
            seg_type = seg[0]

            if seg_type == 'silence':
                duration = seg[1]
                result_wav.append(torch.zeros(int(duration * sample_rate), device=device))

            elif seg_type == 'pause':
                # Автоматична пауза 0.5с після {{/VOICE}}
                result_wav.append(torch.zeros(VOICE_PAUSE_SAMPLES, device=device))

            elif seg_type == 'text':
                _, part_text, part_voice, _ = seg
                style = styles_dict.get(part_voice, narrator_style)
                wavs = synthesize_text_part(model_obj, model_name, part_text, style, speed, device)
                result_wav.extend(wavs)

    if not result_wav:
        raise gr.Error("Не вдалося синтезувати аудіо")

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

        voice_hint = (
            "**Підтримка кількох голосів:** оберіть голос оповідача нижче. "
            "Для інших персонажів використовуйте теги у тексті:\n"
            "`{{VOICE:ім'я_голосу}} текст персонажа {{/VOICE}}`\n"
            "Після кожного `{{/VOICE}}` автоматично додається пауза 0.5с."
        )

        with gr.Tab("Multi speaker (HiFiGAN)"):
            gr.Markdown(voice_hint)
            with gr.Row():
                with gr.Column():
                    input_m = gr.Text(label='Текст', lines=5)
                    btn_v_m = gr.Button("Вербалізувати")
                    speed_m = gr.Slider(0.7, 1.3, 1.0, label='Швидкість')
                    speaker_m = gr.Dropdown(choices=multi_prompts_list, value=multi_prompts_list[0], label="Голос оповідача")
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
            gr.Markdown("*Швидший вокодер — менше часу на синтез*\n\n" + voice_hint)
            with gr.Row():
                with gr.Column():
                    input_i = gr.Text(label='Текст', lines=5)
                    btn_v_i = gr.Button("Вербалізувати")
                    speed_i = gr.Slider(0.7, 1.3, 1.0, label='Швидкість')
                    speaker_i = gr.Dropdown(choices=istft_prompts_list, value=istft_prompts_list[0], label="Голос оповідача")
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
