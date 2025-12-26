import os
import logging
import tempfile
import time
import re
import json
import asyncio
from datetime import datetime, timedelta
from collections import defaultdict
from contextlib import contextmanager
from functools import lru_cache

from telegram import Update
from telegram.ext import Application, MessageHandler, CommandHandler, filters, ContextTypes
from telegram.helpers import escape_markdown

import gspread
from gspread.exceptions import APIError
from google.oauth2.service_account import Credentials
from dotenv import load_dotenv

import dashscope
from dashscope import Generation, ImageSynthesis
import requests

# ----------------------------
# Config
# ----------------------------
load_dotenv()

LLM_MODEL = 'qwen-plus'
IMAGE_MODEL = 'wan2.2-t2i-flash'
COOLDOWN_SECONDS = 30
MAX_METAPHOR_LENGTH = 100
IMAGE_TIMEOUT = 60
MAX_CAPTION = 900

ALLOWED_CATEGORIES = {
    "business-strategy", "performance-effort", "growth-results",
    "challenges-obstacles", "communication-collaboration",
    "predicting", "no-category"
}

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

user_last_request = defaultdict(lambda: datetime.min)

# ----------------------------
# Validation
# ----------------------------
def validate_environment():
    required_vars = {
        'TELEGRAM_BOT_TOKEN': os.getenv('TELEGRAM_BOT_TOKEN'),
        'DASHSCOPE_API_KEY': os.getenv('DASHSCOPE_API_KEY'),
        'GOOGLE_SHEET_NAME': os.getenv('GOOGLE_SHEET_NAME'),  # kept for .env compatibility, but unused
        'TELEGRAM_GROUP_CHAT_ID': os.getenv('TELEGRAM_GROUP_CHAT_ID')
    }
    missing = [k for k, v in required_vars.items() if not v]
    if missing:
        raise EnvironmentError(f"Missing required environment variables: {', '.join(missing)}")
    return required_vars


def validate_sheet_headers(sheet):
    expected = [
        "Metaphor", "Definition", "Synonyms", "Spanish", "Chinese",
        "Russian", "Office Talk", "Parent-Child", "Project Team",
        "Lovers/Friends", "Category", "Image Prompt Used"
    ]
    actual = sheet.row_values(1)
    if actual != expected:
        raise ValueError(f"Sheet headers don't match!\nExpected: {expected}\nGot: {actual}")
    logging.info("✅ Google Sheet headers validated")

# ----------------------------
# Helpers
# ----------------------------
def extract_json(content: str) -> str:
    json_match = re.search(r'```(?:json)?\s*(\{[\s\S]*?\})\s*```', content)
    if json_match:
        return json_match.group(1)
    json_match = re.search(r'\{[\s\S]*?\}', content)
    if json_match:
        return json_match.group(0)
    return content


@contextmanager
def temp_image_file(image_bytes: bytes, metaphor: str):
    safe_name = re.sub(r'[^\w\s-]', '', metaphor).replace(' ', '_')[:50]
    temp_path = os.path.join(tempfile.gettempdir(), f"{safe_name}_{int(time.time())}.png")
    try:
        with open(temp_path, 'wb') as f:
            f.write(image_bytes)
        yield temp_path
    finally:
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception:
                pass


def append_to_sheet_with_retry(sheet, row_data: list, max_retries: int = 3) -> bool:
    for attempt in range(max_retries):
        try:
            sheet.append_row(row_data, value_input_option="RAW")
            logging.info("✅ Saved to Google Sheets")
            return True
        except APIError as e:
            if attempt < max_retries - 1:
                wait = 2 ** attempt
                logging.warning(f"Retrying sheet append in {wait}s...")
                time.sleep(wait)
            else:
                logging.error(f"Sheet append failed: {e}")
                return False
        except Exception as e:
            logging.error(f"Unexpected sheet error: {e}")
            return False
    return False


def validate_and_fix_category(category: str) -> str:
    if category in ALLOWED_CATEGORIES:
        return category
    logging.warning(f"Invalid category '{category}', defaulting to 'no-category'")
    return "no-category"

# ----------------------------
# LLM Generation + Cache
# ----------------------------
@lru_cache(maxsize=500)
def generate_metaphor_data_cached(norm_metaphor: str):
    return generate_metaphor_data_impl(norm_metaphor)


def generate_metaphor_data_impl(metaphor: str):
    prompt = f"""
You are a linguistic expert creating educational content for English learners.

Metaphor: "{metaphor}"

Generate the following in ENGLISH only:

1. Definition
2. 3–5 synonyms
3. Translations: Spanish, Chinese (simplified), Russian
4. FOUR short dialogues: Office, Parent-Child, Project Team, Lovers/Friends
5. ONE category from:
business-strategy, performance-effort, growth-results, challenges-obstacles,
communication-collaboration, predicting, no-category
6. IMAGE PROMPT showing literal + figurative meaning

Return ONLY valid JSON with keys:
definition, synonyms, spanish, chinese, russian,
office_talk, parent_child, project_team, lovers_friends,
category, image_prompt
"""
    try:
        response = Generation.call(
            model=LLM_MODEL,
            prompt=prompt,
            result_format='message'
        )

        if not response.output or not response.output.choices:
            logging.error("Empty LLM response")
            return None

        content = response.output.choices[0].message.content
        data = json.loads(extract_json(content))

        required = [
            "definition", "synonyms", "spanish", "chinese", "russian",
            "office_talk", "parent_child", "project_team",
            "lovers_friends", "category", "image_prompt"
        ]
        if any(k not in data for k in required):
            logging.error("Missing keys in LLM response")
            return None

        data["category"] = validate_and_fix_category(data["category"])
        return data

    except Exception as e:
        logging.error(f"LLM error: {e}")
        return None

# ----------------------------
# Image Generation
# ----------------------------
def generate_metaphor_image(image_prompt: str) -> bytes:
    try:
        response = ImageSynthesis.call(
            model=IMAGE_MODEL,
            prompt=image_prompt,
            size='1024*1024',
            n=1
        )

        if response.status_code != 200:
            return None
        if not response.output or not response.output.results:
            return None

        url = response.output.results[0].url
        img = requests.get(url, timeout=IMAGE_TIMEOUT)
        img.raise_for_status()
        return img.content

    except Exception as e:
        logging.error(f"Image error: {e}")
        return None

# ----------------------------
# Commands
# ----------------------------
async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "👋 Welcome to Metaphor Magic Bot!\n\n"
        "Send any English metaphor and I’ll explain it, translate it,\n"
        "show examples, generate an image, and save it to our dictionary.\n\n"
        "Example: break the ice"
    )


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "🆘 *How to use*\n\n"
        "• Send a metaphor\n"
        "• Wait ~15 seconds\n"
        "• Result is posted to the group\n\n"
        "⏱ One every 30s\n"
        "📏 Max 100 chars",
        parse_mode="Markdown"
    )

# ----------------------------
# Main Handler
# ----------------------------
async def handle_new_metaphor(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.message or not update.message.text:
        return

    metaphor = update.message.text.strip()
    if not metaphor or len(metaphor) > MAX_METAPHOR_LENGTH:
        await update.message.reply_text("❌ Invalid or too long metaphor.")
        return

    user_id = update.effective_user.id
    now = datetime.now()
    elapsed = (now - user_last_request[user_id]).total_seconds()

    if elapsed < COOLDOWN_SECONDS:
        await update.message.reply_text(f"⏳ Wait {int(COOLDOWN_SECONDS - elapsed)}s.")
        return

    user_last_request[user_id] = now

    safe_meta = escape_markdown(metaphor, version=2)
    await update.message.reply_text(
        f"🎨 Generating for *{safe_meta}*\\.\\.\\.",
        parse_mode="MarkdownV2"
    )

    norm_key = metaphor.strip().lower()
    data = await asyncio.to_thread(generate_metaphor_data_cached, norm_key)

    if data is None:
        generate_metaphor_data_cached.cache_clear()
        await update.message.reply_text("❌ Failed to generate content.")
        return

    image_prompt = data["image_prompt"]
    image_bytes = await asyncio.to_thread(generate_metaphor_image, image_prompt)

    esc = lambda x: escape_markdown(x, version=2)

    message = (
        f"🎯 *Metaphor*: {esc(metaphor)}\n\n"
        f"📘 *Definition*: {esc(data['definition'])}\n"
        f"🔗 *Synonyms*: {esc(data['synonyms'])}\n\n"
        f"🌍 *Translations*:\n"
        f"• ES: {esc(data['spanish'])}\n"
        f"• ZH: {esc(data['chinese'])}\n"
        f"• RU: {esc(data['russian'])}\n\n"
        f"💬 *Examples*:\n"
        f"• Office: {esc(data['office_talk'])}\n"
        f"• Parent\\-Child: {esc(data['parent_child'])}\n"
        f"• Project: {esc(data['project_team'])}\n"
        f"• Friends: {esc(data['lovers_friends'])}\n\n"
        f"🔖 *Category*: {esc(data['category'])}\n\n"
        f"🎨 *Image Prompt*: _{esc(image_prompt)}_"
    )

    if len(message) > MAX_CAPTION:
        message = message[:MAX_CAPTION-3] + "..."

    chat_id = context.bot_data['group_chat_id']

    try:
        if image_bytes:
            with temp_image_file(image_bytes, metaphor) as p:
                with open(p, 'rb') as f:
                    await context.bot.send_photo(
                        chat_id=chat_id,
                        photo=f,
                        caption=message,
                        parse_mode="MarkdownV2"
                    )
        else:
            await context.bot.send_message(chat_id=chat_id, text=message, parse_mode="MarkdownV2")
    except Exception as e:
        logging.error(f"Telegram send error: {e}")

    await asyncio.to_thread(
        append_to_sheet_with_retry,
        context.bot_data['sheet'],
        [
            metaphor, data["definition"], data["synonyms"],
            data["spanish"], data["chinese"], data["russian"],
            data["office_talk"], data["parent_child"],
            data["project_team"], data["lovers_friends"],
            data["category"], image_prompt
        ]
    )

    await update.message.reply_text("✅ Done! Posted and saved.")

# ----------------------------
# Main
# ----------------------------
def main():
    env = validate_environment()

    # 🔥 FIXED: removed trailing spaces
    dashscope.api_key = env['DASHSCOPE_API_KEY']
    dashscope.base_http_api_url = 'https://dashscope-intl.aliyuncs.com/api/v1'

    # 🔥 FIXED: removed trailing spaces in scope
    creds = Credentials.from_service_account_file(
        "creds/gsheet_creds.json",
        scopes=["https://www.googleapis.com/auth/spreadsheets"]
    )
    gc = gspread.authorize(creds)
    
    # 🔥 Use Sheet ID (more reliable than name)
    SHEET_ID = "1vxzfYe7q-DAtK_CAWLrpguxWwhjNuOaz-tFUczv_XNs"
    sheet = gc.open_by_key(SHEET_ID).sheet1
    
    validate_sheet_headers(sheet)

    app = Application.builder().token(env['TELEGRAM_BOT_TOKEN']).build()

    app.bot_data['sheet'] = sheet
    app.bot_data['group_chat_id'] = int(env['TELEGRAM_GROUP_CHAT_ID'])

    app.add_handler(CommandHandler("start", start_command))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_new_metaphor))

    logging.info("🚀 Metaphor Magic Bot running...")
    app.run_polling()


if __name__ == "__main__":
    main()