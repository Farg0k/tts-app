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

# --- ПАПКИ З ГОЛОСАМИ ---
multi_prompts_dir = 'voices'
istft_prompts_dir = 'voices_istft' if os.path.isdir('voices_istft') else 'voices'


# --- ФУНКЦІЯ ЗАВАНТАЖЕННЯ ГОЛОСІВ ---
def load_voices(prompts_dir, model_device):
    audio_files = sorted(glob.glob(os.path.join(prompts_dir, '*.pt')))
    prompts_list = [os.path.basename(f).replace('.pt', '') for f in audio_files]
    styles = {}
    for audio_path in audio_files:
        name = os.path.basename(audio_path).replace('.pt', '')
        styles[name] = torch.load(audio_path, map_location=model_device)
        #print('Завантажено:', name)
    return prompts_list, styles


# --- ПОЧАТКОВЕ ЗАВАНТАЖЕННЯ ГОЛОСІВ ---
multi_prompts_list, multi_styles = load_voices(multi_prompts_dir, device)
istft_prompts_list, istft_styles = load_voices(istft_prompts_dir, device)

print("Всі моделі завантажено.")


# --- ФУНКЦІЯ ОНОВЛЕННЯ ГОЛОСІВ ---
def refresh_voices():
    global multi_styles, multi_prompts_list, istft_styles, istft_prompts_list
    print("Оновлення списку голосів...")
    multi_prompts_list, multi_styles = load_voices(multi_prompts_dir, device)
    istft_prompts_list, istft_styles = load_voices(istft_prompts_dir, device)
    first_m = multi_prompts_list[0] if multi_prompts_list else None
    first_i = istft_prompts_list[0] if istft_prompts_list else None
    print(f"Оновлено: multi={len(multi_prompts_list)} голосів, istft={len(istft_prompts_list)} голосів")
    return (
        gr.Dropdown(choices=multi_prompts_list, value=first_m),
        gr.Dropdown(choices=istft_prompts_list, value=first_i),
        f"✅ Оновлено: **multi** — {len(multi_prompts_list)} голосів, **istft** — {len(istft_prompts_list)} голосів"
    )


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
                    verbalized_inner += verbalizer.process_text(part.strip().lower())[0] + ' '
            result += f'{{{{VOICE:{voice_name}}}}}{verbalized_inner.strip()}{{{{/VOICE}}}} '
            i += 1
            continue
        parts = split_to_parts(seg, group=False)
        for part in parts:
            if part.strip():
                result += verbalizer.process_text(part.strip().lower())[0] + ' '
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
# --- СТВОРЕННЯ НОВОГО ГОЛОСУ ---
def create_voice(input_audio, voice_name, model_name):
    if input_audio is None:
        raise gr.Error("Завантажте аудіо файл")
    if not voice_name.strip():
        raise gr.Error("Введіть ім'я голосу")

    voice_name = voice_name.strip()

    if model_name == 'multi':
        model_obj = multi_model
    else:
        model_obj = istft_model

    out_dir = 'voices'
    os.makedirs(out_dir, exist_ok=True)
    output_path = os.path.join(out_dir, f"{voice_name}.pt")

    if os.path.exists(output_path):
        raise gr.Error(f"Голос '{voice_name}' вже існує. Оберіть інше ім'я.")

    print(f"--- Генерація вектора стилю з: {input_audio} ---")
    style = model_obj.extract_voice_features(input_audio)
    torch.save(style, output_path)
    print(f"✅ Збережено: {output_path}")

    return f"✅ Голос **{voice_name}** збережено до `{output_path}` (форма: {style.shape})"


if __name__ == "__main__":
    with gr.Blocks(title="StyleTTS2 ukrainian") as demo:
        gr.Markdown("# StyleTTS2 Ukrainian Local")

        voice_hint = (
            "**Підтримка кількох голосів:** оберіть голос оповідача нижче. "
            "Для інших персонажів використовуйте теги у тексті:\n"
            "`{{VOICE:ім'я_голосу}} текст персонажа {{/VOICE}}`\n"
            "Після кожного `{{/VOICE}}` автоматично додається пауза 0.5с."
        )

        # --- Кнопка оновлення голосів (глобальна) ---
        with gr.Row():
            btn_refresh = gr.Button("🔄 Оновити список голосів", variant="secondary")
            refresh_status = gr.Markdown("")

        # --- Вкладка 1: Single speaker ---
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

        # --- Вкладка 2: Multi speaker (iSTFTNet) ---
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

        # --- Вкладка 3: Multi speaker (HiFiGAN) ---
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

        # --- Вкладка 4: Додати голос ---
        with gr.Tab("➕ Додати голос"):
            gr.Markdown(
                "Завантажте аудіо запис голосу (wav, mp3, flac, ogg), "
                "вкажіть ім'я і натисніть **Створити**. "
                "Список голосів оновиться автоматично."
            )
            with gr.Row():
                with gr.Column():
                    audio_input = gr.Audio(label="Аудіо файл", type="filepath")
                    new_voice_name = gr.Text(label="Ім'я голосу (без пробілів)")
                    voice_model_choice = gr.Dropdown(
                        choices=["multi", "istft"],
                        value="istft",
                        label="Модель"
                    )
                    btn_create_voice = gr.Button("🎤 Створити голос", variant="primary")
                with gr.Column():
                    create_voice_status = gr.Markdown("")

        # --- Підключення кнопки оновлення голосів ---
        btn_refresh.click(
            fn=refresh_voices,
            inputs=[],
            outputs=[speaker_m, speaker_i, refresh_status]
        )

        # --- Після створення голосу — автоматично оновити списки ---
        def create_voice_and_refresh(input_audio, voice_name, model_name):
            status = create_voice(input_audio, voice_name, model_name)
            dropdown_m, dropdown_i, _ = refresh_voices()
            return status, dropdown_m, dropdown_i

        btn_create_voice.click(
            fn=create_voice_and_refresh,
            inputs=[audio_input, new_voice_name, voice_model_choice],
            outputs=[create_voice_status, speaker_m, speaker_i]
        )

        # --- API ендпоінти ---
        # POST http://127.0.0.1:7860/api/voices     → {"voices": [...]}  (також оновлює список голосів)
        # POST http://127.0.0.1:7860/api/verbalize  → вербалізований текст
        def api_voices():
            dropdown_m, dropdown_i, _ = refresh_voices()
            return {"voices": multi_prompts_list}

        voices_out = gr.JSON(visible=False)
        gr.Button(visible=False).click(fn=api_voices, inputs=[], outputs=voices_out, api_name="voices")

        verbalize_in = gr.Text(visible=False)
        verbalize_out = gr.Text(visible=False)
        gr.Button(visible=False).click(fn=verbalize, inputs=[verbalize_in], outputs=[verbalize_out], api_name="verbalize")

    if device == 'cuda':
        demo.launch(share=True)
    else:
        demo.launch(server_name="127.0.0.1", server_port=7860)