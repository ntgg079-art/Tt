import asyncio
import logging
import os
from collections import defaultdict, deque

from aiogram import Bot, Dispatcher, F
from aiogram.enums import ChatAction
from aiogram.filters import Command, CommandStart
from aiogram.types import Message
from dotenv import load_dotenv
from openai import AsyncOpenAI

load_dotenv()

TELEGRAM_TOKEN = os.environ["TELEGRAM_TOKEN"]
OPENROUTER_API_KEY = os.environ["OPENROUTER_API_KEY"]

# "openrouter/free" сам выбирает доступную бесплатную модель.
# Можно зафиксировать конкретную, например "xxx/yyy:free" (список: openrouter.ai/collections/free-models)
MODEL = os.getenv("MODEL", "openrouter/free")
# Запасные бесплатные модели через запятую, если основная недоступна (необязательно)
FALLBACK_MODELS = [m.strip() for m in os.getenv("FALLBACK_MODELS", "").split(",") if m.strip()]

SYSTEM_PROMPT = os.getenv(
    "SYSTEM_PROMPT",
    "Ты дружелюбный и полезный ИИ-ассистент в Telegram. "
    "Отвечай кратко и по делу на языке собеседника.",
)
MAX_HISTORY = 20
TG_LIMIT = 4096

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("tg-ai-bot")

bot = Bot(TELEGRAM_TOKEN)
dp = Dispatcher()
client = AsyncOpenAI(
    api_key=OPENROUTER_API_KEY,
    base_url="https://openrouter.ai/api/v1",
)

history: dict[int, deque] = defaultdict(lambda: deque(maxlen=MAX_HISTORY))
locks: dict[int, asyncio.Lock] = defaultdict(asyncio.Lock)


def split_text(text: str, limit: int = TG_LIMIT):
    while len(text) > limit:
        cut = text.rfind("\n", 0, limit)
        if cut < limit // 2:
            cut = limit
        yield text[:cut]
        text = text[cut:].lstrip("\n")
    if text:
        yield text


async def ask_model(messages: list[dict]) -> str:
    extra = {"models": [MODEL, *FALLBACK_MODELS]} if FALLBACK_MODELS else None
    response = await client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "system", "content": SYSTEM_PROMPT}, *messages],
        max_tokens=1500,
        extra_body=extra,
    )
    return (response.choices[0].message.content or "").strip()


@dp.message(CommandStart())
async def cmd_start(message: Message):
    await message.answer(
        "Привет! Я ИИ-бот. Просто напиши мне сообщение.\n\n"
        "/reset — очистить память диалога"
    )


@dp.message(Command("reset"))
async def cmd_reset(message: Message):
    history[message.from_user.id].clear()
    await message.answer("Память диалога очищена ✅")


@dp.message(F.text)
async def handle_text(message: Message):
    user_id = message.from_user.id

    async with locks[user_id]:
        history[user_id].append({"role": "user", "content": message.text})
        await bot.send_chat_action(message.chat.id, ChatAction.TYPING)

        try:
            answer = await ask_model(list(history[user_id]))
            if not answer:
                answer = "Пустой ответ, попробуй переформулировать."
        except Exception:
            log.exception("Ошибка при обращении к API")
            history[user_id].pop()
            await message.answer(
                "⚠️ Бесплатные модели иногда перегружены. Попробуй ещё раз через минуту."
            )
            return

        history[user_id].append({"role": "assistant", "content": answer})

        for part in split_text(answer):
            await message.answer(part, parse_mode=None)


async def main():
    log.info("Бот запущен, модель: %s", MODEL)
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())
