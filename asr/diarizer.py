"""LLM-based role assignment to partition dialogues between Operator and Client."""

from core.ports import DiarizerPort, StructuredLLMPort
from models.schemas import DiarizationResult, RawSegment, Speaker, TranscriptSegment


class Diarizer(DiarizerPort):
    """Assign roles to speakers contextually using a LLM client."""

    def __init__(self, llm: StructuredLLMPort) -> None:
        self.llm = llm


    async def assign_speakers(
        self, segments: list[RawSegment]
    ) -> list[TranscriptSegment]:
        if not segments:
            return []
        speakers = await self._call_llm_for_speakers(segments)
        return [
            TranscriptSegment(
                speaker=speakers[i] if i < len(speakers) else "Клиент",
                start=seg.start, end=seg.end, text=seg.text
            )
            for i, seg in enumerate(segments)
        ]


    async def _call_llm_for_speakers(self, segments: list[RawSegment]) -> list[Speaker]:
        formatted = "\n".join(f"{i}: {seg.text}" for i, seg in enumerate(segments))
        system = (
            "Ты — ассистент по разметке звонков. Определи роль говорящего для каждой реплики.\n"
            "Роли: 'Оператор' (представляет МТБанк, здоровается первым, предлагает помощь) "
            "или 'Клиент' (задает вопросы, жалуется, просит информацию).\n"
            "Корректный пример диалога:\n"
            "Оператор: Добрый день, МТБанк, меня зовут Анна, чем могу помочь?\n"
            "Клиент: Здравствуйте.\n"
            "Клиент: Хочу узнать про условия по кредиту наличными.\n"
            "Оператор: Конечно, подскажите, пожалуйста, какая сумма вас интересует и на какой срок?\n"    
            "Клиент: Примерно десять тысяч рублей, на год.\n"
            "Оператор: Отлично.\n"
            "Оператор: На данный момент ставка от четырнадцати и девяти процентов годовых, решение за пятнадцать минут.\n"
            "Оператор: Вы уже являетесь клиентом МТБанка?\n"
            "Клиент: Да, у меня есть карточка ваша.\n"
            "Оператор: Прекрасно, тогда для вас действуют специальные условия.\n"
            "Оператор: Ежемесячный платёж составит около девятисот рублей.\n"
            "Оператор: Вам удобно подать заявку онлайн через приложение или предпочитаете приехать в отделение?\n"
            "Клиент: Лучше онлайн.\n"
            "Клиент: Но у меня вопрос — если я захочу досрочно погасить, есть штрафы?\n"
            "Оператор: Нет, досрочное погашение без штрафов и комиссий, в любое время и в любом объёме.\n"
            "Клиент: Хорошо, а страховка обязательна?\n"
            "Оператор: Страхование жизни подключается по вашему желанию, это не обязательное условие получения кредита.\n"
            "Оператор: Однако при подключении страховки ставка может быть немного снижена.\n"
            "Клиент: Понятно.\n"
            "Клиент: Тогда я попробую подать через приложение.\n"
            "Оператор: Отлично.\n"
            "Оператор: Если возникнут вопросы в процессе заполнения — звоните, мы поможем.\n"
            "Оператор: Также могу отправить вам краткую инструкцию на email, если хотите.\n"
            "Клиент: Да, пожалуйста, отправьте.\n"
            "Оператор: Хорошо, подскажите ваш email.\n"
            "Клиент: Михаил-собака-пример-точка-бай.\n"
            "Оператор: Записала.\n"
            "Оператор: В течение нескольких минут получите письмо с инструкцией и ссылкой на заявку."
            "Оператор: Есть ещё вопросы?\n"
            "Клиент: Нет, всё понятно, спасибо.\n"
            "Оператор: Спасибо за обращение в МТБанк, хорошего дня!\n"
            "Клиент: И вам, до свидания.\n"
        )
        res = await self.llm.complete_json(
            system_prompt=system,
            user_prompt=f"Разметь по ролям следующие реплики:\n{formatted}",
            response_model=DiarizationResult
        )
        mapping = {item.index: item.speaker for item in res.speakers}
        return [mapping.get(i, "Клиент") for i in range(len(segments))]
