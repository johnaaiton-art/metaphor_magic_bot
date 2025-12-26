import os
import logging
import tempfile
import time
import re
import json
import asyncio
import sys
from datetime import datetime, timedelta
from collections import defaultdict
from contextlib import contextmanager
from functools import lru_cache, partial

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

# Python 3.8 compatibility for asyncio.to_thread
if sys.version_info < (3, 9):
    def to_thread(func, *args, **kwargs):
        loop = asyncio.get_event_loop()
        return loop.run_in_executor(None, partial(func, *args, **kwargs))
else:
    to_thread = asyncio.to_thread

# ----------------------------
# Config
# ----------------------------
load_dotenv()

LLM_MODEL = 'qwen-plus'
IMAGE_MODEL = 'wan2.2-t2i-flash'
COOLDOWN_SECONDS = 30
MAX_METAPHOR_LENGTH = 100
IMAGE_TIMEOUT = 60
MAX_CAPTION = 1000

# Hardcoded Google Sheet ID (more reliable than name lookup)
SHEET_ID = "1vxzfYe7q-DAtK_CAWLrpguxWwhjNuOaz-tFUczv_XNs"

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
    """Validate required environment variables"""
    required_vars = {
        'TELEGRAM_BOT_TOKEN': os.getenv('TELEGRAM_BOT_TOKEN'),
        'DASHSCOPE_API_KEY': os.getenv('DASHSCOPE_API_KEY'),
        'TELEGRAM_GROUP_CHAT_ID': os.getenv('TELEGRAM_GROUP_CHAT_ID')
    }
    missing = [k for k, v in required_vars.items() if not v]
    if missing:
        raise EnvironmentError(f"Missing required environment variables: {', '.join(missing)}")
    return required_vars


def validate_sheet_headers(sheet):
    """Ensure Google Sheet has correct 12-column structure"""
    expected = [
        "Metaphor", "Definition", "Synonyms", "Spanish", "Chinese",
        "Russian", "Office Talk", "Parent-Child", "Project Team",
        "Lovers/Friends", "Category", "Image Prompt Used"
    ]
    try:
        actual = sheet.row_values(1)
        if actual != expected:
            raise ValueError(
                f"Sheet headers don't match!\n"
                f"Expected: {expected}\n"
                f"Got: {actual}"
            )
        logging.info("✅ Google Sheet headers validated")
    except Exception as e:
        logging.error(f"Sheet validation failed: {e}")
        raise

# ----------------------------
# Helpers
# ----------------------------
def extract_json(content: str) -> str:
    """Extract JSON from LLM response (handles markdown fences)"""
    json_match = re.search(r'```(?:json)?\s*(\{[\s\S]*?\})\s*```', content)
    if json_match:
        return json_match.group(1)
    json_match = re.search(r'\{[\s\S]*?\}', content)
    if json_match:
        return json_match.group(0)
    return content


@contextmanager
def temp_image_file(image_bytes: bytes, metaphor: str):
    """Context manager for safe temp file handling"""
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
                logging.debug(f"Cleaned up: {temp_path}")
            except Exception:
                pass


def append_to_sheet_with_retry(sheet, row_data: list, max_retries: int = 3) -> bool:
    """Append to Google Sheet with exponential backoff retry"""
    for attempt in range(max_retries):
        try:
            sheet.append_row(row_data, value_input_option="RAW")
            logging.info("✅ Saved to Google Sheets")
            return True
        except APIError as e:
            if attempt < max_retries - 1:
                wait = 2 ** attempt
                logging.warning(f"Retrying sheet append in {wait}s... (attempt {attempt + 1}/{max_retries})")
                time.sleep(wait)
            else:
                logging.error(f"Sheet append failed after {max_retries} attempts: {e}")
                return False
        except Exception as e:
            logging.error(f"Unexpected sheet error: {e}")
            return False
    return False


def validate_and_fix_category(category: str) -> str:
    """Validate category against allowed list"""
    if category in ALLOWED_CATEGORIES:
        return category
    logging.warning(f"Invalid category '{category}', defaulting to 'no-category'")
    return "no-category"

# ----------------------------
# LLM Generation + Cache
# ----------------------------
@lru_cache(maxsize=500)
def generate_metaphor_data_cached(norm_metaphor: str):
    """Cached wrapper to avoid regenerating same metaphors"""
    return generate_metaphor_data_impl(norm_metaphor)


def generate_metaphor_data_impl(metaphor: str):
    """Generate structured metaphor data using LLM"""
    prompt = f"""
You are a linguistic expert creating educational content for English learners.

Metaphor: "{metaphor}"

Generate the following in ENGLISH only:

1. Definition
2. 3–5 synonyms
3. Translations: Spanish, Chinese (simplified), Russian
4. FOUR short dialogues (2-3 exchanges each max, formatted with each speaker's line on a new line, e.g. Speaker: Text\nSpeaker: Text\nSpeaker: Text): Office, Parent-Child, Project Team, Lovers/Friends
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
        
        missing = [k for k in required if k not in data]
        if missing:
            logging.error(f"Missing keys in LLM response: {missing}")
            return None

        data["category"] = validate_and_fix_category(data["category"])
        logging.info(f"✅ Generated data for '{metaphor}'")
        return data

    except json.JSONDecodeError as e:
        logging.error(f"JSON parsing error: {e}")
        return None
    except Exception as e:
        logging.error(f"LLM error: {e}")
        return None

# ----------------------------
# Image Generation
# ----------------------------
def generate_metaphor_image(image_prompt: str) -> bytes:
    """Generate image using DashScope API"""
    try:
        response = ImageSynthesis.call(
            model=IMAGE_MODEL,
            prompt=image_prompt,
            size='1024*1024',
            n=1
        )

        if response.status_code != 200:
            logging.error(f"Image API returned status {response.status_code}")
            return None
            
        if not response.output or not response.output.results:
            logging.error("No image results in API response")
            return None

        url = response.output.results[0].url
        logging.info(f"Image generated: {url}")
        
        img = requests.get(url, timeout=IMAGE_TIMEOUT)
        img.raise_for_status()
        
        logging.info("✅ Image downloaded")
        return img.content

    except requests.RequestException as e:
        logging.error(f"Image download error: {e}")
        return None
    except Exception as e:
        logging.error(f"Image generation error: {e}")
        return None

# ----------------------------
# Commands
# ----------------------------
async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /start command"""
    await update.message.reply_text(
        "👋 Welcome to Metaphor Magic Bot!\n\n"
        "Send any English metaphor and I'll explain it, translate it,\n"
        "show examples, generate an image, and save it to our dictionary.\n\n"
        "💡 Example: break the ice"
    )


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /help command"""
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
    """Main handler for incoming metaphor messages"""
    # Guard: ensure message and text exist
    if not update.message or not update.message.text:
        return

    metaphor = update.message.text.strip()
    
    # Validate length
    if not metaphor or len(metaphor) > MAX_METAPHOR_LENGTH:
        await update.message.reply_text(
            f"❌ Metaphor must be 1-{MAX_METAPHOR_LENGTH} characters."
        )
        return

    user_id = update.effective_user.id
    now = datetime.now()
    elapsed = (now - user_last_request[user_id]).total_seconds()

    # Rate limiting
    if elapsed < COOLDOWN_SECONDS:
        remaining = int(COOLDOWN_SECONDS - elapsed)
        await update.message.reply_text(f"⏳ Wait {remaining}s before sending another.")
        return

    user_last_request[user_id] = now
    logging.info(f"Processing '{metaphor}' from user {user_id}")

    # Send progress message
    safe_meta = escape_markdown(metaphor, version=2)
    await update.message.reply_text(
        f"🎨 Generating for *{safe_meta}*\\.\\.\\.",
        parse_mode="MarkdownV2"
    )

    # Step 1: Generate text content (with timeout and caching)
    norm_key = metaphor.strip().lower()
    try:
        data = await asyncio.wait_for(
            to_thread(generate_metaphor_data_cached, norm_key),
            timeout=45.0
        )
    except asyncio.TimeoutError:
        await update.message.reply_text("❌ LLM request timed out. Try again in 1 min.")
        return

    if data is None:
        # Clear cache on failure to allow retry with fresh request
        generate_metaphor_data_cached.cache_clear()
        await update.message.reply_text("❌ Failed to generate content. Please try again.")
        return

    # Normalize synonyms to string if it's a list
    if isinstance(data['synonyms'], list):
        data['synonyms'] = ', '.join(data['synonyms'])

    # Step 2: Generate image
    image_prompt = data["image_prompt"]
    image_bytes = await to_thread(generate_metaphor_image, image_prompt)

    if image_bytes is None:
        await update.message.reply_text("⚠️ Image generation failed. Sending text only.")

    # Step 3: Format message with proper escaping
    esc = lambda x: escape_markdown(x, version=2)

    message = (
        f"🎯 *Metaphor*: {esc(metaphor)}\n\n"
        f"📘 *Definition*: {esc(data['definition'])}\n"
        f"🔗 *Synonyms*: {esc(data['synonyms'])}\n\n"
        f"🌍 *Translations*:\n"
        f"• ES: {esc(data['spanish'])}\n"
        f"• ZH: {esc(data['chinese'])}\n"
        f"• RU: {esc(data['russian'])}\n\n"
        f"💬 *Examples*:\n\n"
        f"💼 • **Office**:\n{esc(data['office_talk'])}\n\n"
        f"👪 • **Parent\\-Child**:\n{esc(data['parent_child'])}\n\n"
        f"👥 • **Project Team**:\n{esc(data['project_team'])}\n\n"
        f"❤️ • **Lovers/Friends**:\n{esc(data['lovers_friends'])}\n\n"
        f"🔖 *Category*: {esc(data['category'])}\n\n"
        f"🎨 *Image Prompt*: _{esc(image_prompt)}_"
    )

    # Truncate if exceeds Telegram caption limit
    if len(message) > MAX_CAPTION:
        message = message[:MAX_CAPTION-3] + "\\.\\.\\."

    # Step 4: Send to Telegram group
    chat_id = context.bot_data['group_chat_id']

    try:
        if image_bytes:
            with temp_image_file(image_bytes, metaphor) as temp_path:
                with open(temp_path, 'rb') as photo:
                    await context.bot.send_photo(
                        chat_id=chat_id,
                        photo=photo,
                        caption=message,
                        parse_mode="MarkdownV2"
                    )
            logging.info(f"✅ Posted '{metaphor}' with image to group")
        else:
            await context.bot.send_message(
                chat_id=chat_id,
                text=message,
                parse_mode="MarkdownV2"
            )
            logging.info(f"✅ Posted '{metaphor}' (text only) to group")
    except Exception as e:
        logging.error(f"Telegram send error: {e}")
        await update.message.reply_text(
            "⚠️ Content generated but failed to post to group. Check bot permissions."
        )

    # Step 5: Save to Google Sheet (non-blocking)
    sheet_success = await to_thread(
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

    if sheet_success:
        await update.message.reply_text("✅ Done! Posted to group and saved to sheet.")
    else:
        await update.message.reply_text("✅ Posted to group, but sheet save failed.")

# ----------------------------
# Main
# ----------------------------
def main():
    """Initialize and run the bot"""
    try:
        # Validate environment
        env_vars = validate_environment()

        # DashScope setup
        dashscope.api_key = env_vars['DASHSCOPE_API_KEY']
        dashscope.base_http_api_url = 'https://dashscope-intl.aliyuncs.com/api/v1'

        # Google Sheets auth with correct scope
        creds = Credentials.from_service_account_file(
            "creds/gsheet_creds.json",
            scopes=["https://www.googleapis.com/auth/spreadsheets"]
        )
        gc = gspread.authorize(creds)
        
        # Open sheet by ID (more reliable than name)
        sheet = gc.open_by_key(SHEET_ID).sheet1
        validate_sheet_headers(sheet)

        # Build Telegram application
        app = Application.builder().token(env_vars['TELEGRAM_BOT_TOKEN']).build()

        # Store shared resources in bot_data
        app.bot_data['sheet'] = sheet
        app.bot_data['group_chat_id'] = int(env_vars['TELEGRAM_GROUP_CHAT_ID'])

        # Register handlers
        app.add_handler(CommandHandler("start", start_command))
        app.add_handler(CommandHandler("help", help_command))
        app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_new_metaphor))

        # Start bot
        logging.info("🚀 Metaphor Magic Bot is running...")
        logging.info(f"📊 Connected to sheet: {SHEET_ID}")
        logging.info(f"💬 Posting to group: {env_vars['TELEGRAM_GROUP_CHAT_ID']}")
        logging.info(f"🤖 Using models: {LLM_MODEL} (text), {IMAGE_MODEL} (image)")
        
        app.run_polling()

    except Exception as e:
        logging.critical(f"Failed to start bot: {e}")
        raise


if __name__ == "__main__":
    main()