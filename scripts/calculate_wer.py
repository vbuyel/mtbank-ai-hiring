#!/usr/bin/env python3
"""Calculate WER for test audio files using faster-whisper and jiwer."""

import re
import sys
from pathlib import Path
from faster_whisper import WhisperModel
from jiwer import wer

BASE_DIR = Path(__file__).resolve().parent.parent
TEST_DATA = BASE_DIR / "test_data"
DOCS = BASE_DIR / "docs"

# Reference transcripts extracted from markdown dialogs
REFERENCE_TRANSCRIPTS = {
    "dialog-transfers.mp3": (
        "Добрый день, МТБанк, меня зовут Анна, чем могу помочь? "
        "Здравствуйте. Мне нужно перевести пятьсот рублей на счёт сестры в Беларускбанке. "
        "Перевод по реквизитам доступен в приложении. Введите номер счёта и сумму — комиссии не будет, если вы переводите с карты МТБанка. "
        "А если нужно срочно, в тот же день? "
        "Выберите опцию «Мгновенный перевод» — зачисление за несколько минут, комиссия рубль пятьдесят. "
        "Хорошо, попробую через приложение. "
        "Отлично. Сохраните номер операции на случай вопросов. Есть ещё что-то? "
        "Нет, спасибо. "
        "Спасибо за обращение, хорошего дня!"
    ),
    "dialog-complaints.mp3": (
        "Добрый день, МТБанк, меня зовут Анна, чем могу помочь? "
        "С моей карты списали сто двадцать рублей — магазин электроники, я ничего не покупал. "
        "Блокирую карту сразу. Подскажите последние четыре цифры и ФИО для идентификации. "
        "Четыре пять шесть семь, Сергей Ковалёв. "
        "Оформлю заявление на оспаривание. Предварительное решение — до пяти рабочих дней. Новую карту можно заказать в приложении. "
        "Хорошо, спасибо. "
        "Карта с номером один два три четыре четыре три два один девять восемь шесть пять четыре пять шесть семь заблокирована, номер обращения пришлю в SMS. Свяжемся по результатам. Хорошего дня!"
    ),
    "dialog-cards.mp3": (
        "Добрый день, МТБанк, меня зовут Анна, чем могу помочь? "
        "Хочу узнать про дебетовую карту с кэшбэком. Я пока не клиент банка. "
        "Карта «МТБанк Кэшбэк» — до трёх процентов в выбранных категориях. Оформить можно онлайн за десять минут, обслуживание первый год бесплатно. "
        "А документы какие нужны? "
        "Только паспорт. Карту доставит курьер или можно забрать в отделении. "
        "Хорошо, оформлю. Прислать ссылку на заявку? "
        "Конечно, на email или SMS. Что удобнее? "
        "На email, пожалуйста. "
        "Записала, письмо придёт в течение минуты. Спасибо за интерес к МТБанку, хорошего дня!"
    ),
    "dialog-incompetent.mp3": (
        "Добрый день, МТБанк, меня зовут Анна, чем могу помочь? "
        "Здравствуйте. Вчера отправил перевод на триста рублей, деньги до сих пор не дошли получателю. "
        "А, переводы... Ну бывает, подождите ещё немного, должно дойти. "
        "Сколько ждать? Получатель уже звонит, ему нужны деньги сегодня. "
        "Я точно не знаю. Может, у них там что-то на стороне получателя. "
        "А вы можете проверить статус моей операции? "
        "У меня сейчас ничего не показывает. Попробуйте позвонить завтра, может тогда увидим. "
        "Завтра? Мне нужно решить сегодня. Что мне делать? "
        "Ну, тогда не знаю, чем помочь. Извините. "
        "Хорошо, до свидания. "
        "До свидания."
    ),
}


def normalize_text(text: str) -> str:
    """Normalize text for WER calculation."""
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = re.sub(r'\s+', ' ', text).strip()  # Normalize whitespace
    return text


def transcribe_file(model: WhisperModel, audio_path: Path) -> str:
    """Transcribe an audio file using faster-whisper."""
    segments, info = model.transcribe(
        str(audio_path),
        language="ru",
        beam_size=5,
        vad_filter=True,
    )
    full_text = " ".join(seg.text for seg in segments)
    return full_text


def main():
    print("Загрузка модели faster-whisper (medium)...")
    model = WhisperModel("medium", device="cpu", compute_type="int8")

    results = []
    for filename, reference in REFERENCE_TRANSCRIPTS.items():
        audio_path = TEST_DATA / filename
        if not audio_path.exists():
            print(f"⚠️  Файл не найден: {audio_path}")
            continue

        print(f"\nОбработка: {filename}")
        hypothesis = transcribe_file(model, audio_path)

        ref_normalized = normalize_text(reference)
        hyp_normalized = normalize_text(hypothesis)

        error_rate = wer(ref_normalized, hyp_normalized)
        error_pct = error_rate * 100

        results.append({
            "file": filename,
            "wer": error_pct,
            "ref_words": len(ref_normalized.split()),
            "hyp_words": len(hyp_normalized.split()),
        })

        print(f"  Reference: {ref_normalized[:80]}...")
        print(f"  Hypothesis: {hyp_normalized[:80]}...")
        print(f"  WER: {error_pct:.1f}%")

    # Print summary table
    print("\n" + "=" * 70)
    print("WER TABLE")
    print("=" * 70)
    print(f"{'File':<35} {'WER':>8} {'Ref Words':>10} {'Hyp Words':>10}")
    print("-" * 70)
    for r in results:
        print(f"{r['file']:<35} {r['wer']:>7.1f}% {r['ref_words']:>10} {r['hyp_words']:>10}")
    print("-" * 70)

    if results:
        avg_wer = sum(r['wer'] for r in results) / len(results)
        print(f"{'Average':<35} {avg_wer:>7.1f}%")
    print("=" * 70)

    return results


if __name__ == "__main__":
    results = main()
