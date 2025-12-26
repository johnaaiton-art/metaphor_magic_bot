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

# Model configuration
LLM_MODEL = 'qwen-plus'
IMAGE_MODEL = 'wan2.2-t2i-flash'
COOLDOWN_SECONDS = 30
MAX_METAPHOR_LENGTH = 100
LLM_TIMEOUT = 60
IMAGE_TIMEOUT = 60

# Allowed categories
ALLOWED_CATEGORIES = {
    "business-strategy", "performance-effort", "growth-results",
    "challenges-obstacles", "communication-collaboration",
    "predicting", "no-category"
}

# Logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

# Rate limiting
user_last_request = defaultdict(lambda: datetime.min)

# ----------------------------
# Validation
# ----------------------------
def validate_environment():
    """Validate all required environment variables"""
    required_vars = {
        'TELEGRAM_BOT_TOKEN': os.getenv('TELEGRAM_BOT_TOKEN'),
        'DASHSCOPE_API_KEY': os.getenv('DASHSCOPE_API_KEY'),
        'GOOGLE_SHEET_NAME': os.getenv('GOOGLE_SHEET_NAME'),
        'TELEGRAM_GROUP_CHAT_ID': os.getenv('TELEGRAM_GROUP_CHAT_ID')
    }
    
    missing = [k for k, v in required_vars.items() if not v]
    if missing:
        raise EnvironmentError(f"Missing required environment variables: {', '.join(missing)}")
    
    return required_vars

def validate_sheet_headers(sheet):
    """Ensure Google Sheet has correct headers"""
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
# Helper Functions
# ----------------------------
def extract_json(content: str) -> str:
    """Extract JSON from LLM response, handling various formats"""
    # Try to find JSON between code fences (non-greedy)
    json_match = re.search(r'```(?:json)?\s*(\{[\s\S]*?\})\s*```', content, re.DOTALL)
    if json_match:
        return json_match.group(1)
    
    # Try to find standalone JSON object (non-greedy)
    json_match = re.search(r'\{[\s\S]*?\}', content, re.DOTALL)
    if json_match:
        return json_match.group(0)
    
    return content

@contextmanager
def temp_image_file(image_bytes: bytes, metaphor: str):
    """Context manager for temporary image files"""
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
                logging.debug(f"Cleaned up temp file: {temp_path}")
            except Exception as e:
                logging.warning(f"Failed to remove temp file {temp_path}: {e}")

def append_to_sheet_with_retry(sheet, row_data: list, max_retries: int = 3) -> bool:
    """Append row to Google Sheet with retry logic"""
    for attempt in range(max_retries):
        try:
            sheet.append_row(row_data, value_input_option="RAW")
            logging.info(f"✅ Saved to Google Sheets")
            return True
        except APIError as e:
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt
                logging.warning(f"Sheet append failed (attempt {attempt + 1}/{max_retries}), retrying in {wait_time}s...")
                time.sleep(wait_time)
            else:
                logging.error(f"Failed to append to sheet after {max_retries} attempts: {e}")
                return False
        except Exception as e:
            logging.error(f"Unexpected sheet error: {e}")
            return False
    return False

def validate_and_fix_category(category: str) -> str:
    """Validate category and default to 'no-category' if invalid"""
    if category in ALLOWED_CATEGORIES:
        return category
    logging.warning(f"Invalid category '{category}', defaulting to 'no-category'")
    return "no-category"

# ----------------------------
# LLM: Generate metaphor data
# ----------------------------
@lru_cache(maxsize=500)
def generate_metaphor_data_cached(metaphor: str):
    """Cached wrapper for metaphor generation"""
    return generate_metaphor_data_impl(metaphor)

def generate_metaphor_data_impl(metaphor: str):
    """Generate structured metaphor data using LLM"""
    prompt = f"""
You are a linguistic expert creating educational content for English learners.

Metaphor: "{metaphor}"

Generate the following in ENGLISH only:

1. A clear, concise definition.
2. 3–5 synonyms, comma-separated.
3. Translations:
   - Spanish
   - Chinese (simplified)
   - Russian
4. FOUR short dialogues (max 2 lines each) in these exact contexts:
   a. Office Talk
   b. Parent and Child
   c. Project Team
   d. Lovers or Close Friends
5. ONE category from this list:
   business-strategy, performance-effort, growth-results, challenges-obstacles, communication-collaboration, predicting, no-category
6. An IMAGE PROMPT that:
   - Shows the LITERAL words in the metaphor (e.g., "rock" and "boat" for "rock the boat")
   - ALSO shows the FIGURATIVE meaning (e.g., causing disruption)
   - Prefer a business or everyday scene
   - Be vivid, descriptive, and suitable for image generation

Return ONLY a valid JSON object with these keys:
{{
  "definition": "...",
  "synonyms": "...",
  "spanish": "...",
  "chinese": "...",
  "russian": "...",
  "office_talk": "...",
  "parent_child": "...",
  "project_team": "...",
  "lovers_friends": "...",
  "category": "...",
  "image_prompt": "..."
}}
"""
    try:
        response = Generation.call(
            model=LLM_MODEL,
            prompt=prompt,
            result_format='message'
        )
        content = response.output.choices[0].message.content
        
        # Extract and parse JSON
        json_str = extract_json(content)
        data = json.loads(json_str)
        
        # Validate required keys
        required_keys = [
            "definition", "synonyms", "spanish", "chinese", "russian",
            "office_talk", "parent_child", "project_team", "lovers_friends",
            "category", "image_prompt"
        ]
        
        missing_keys = [key for key in required_keys if key not in data]
        if missing_keys:
            logging.error(f"Missing keys in LLM response: {missing_keys}")
            return None
        
        # Validate and fix category
        data["category"] = validate_and_fix_category(data["category"])
        
        logging.info(f"✅ Generated metaphor data for '{metaphor}'")
        return data
        
    except json.JSONDecodeError as e:
        logging.error(f"JSON parsing error: {e}\nContent: {content[:200]}")
        return None
    except Exception as e:
        logging.error(f"LLM generation error: {e}")
        return None

# ----------------------------
# Image: Generate via DashScope
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
            logging.error(f"Image generation failed with status {response.status_code}: {response}")
            return None
        
        # Safety check: ensure results exist
        if not response.output or not response.output.results:
            logging.error("No image results returned from API")
            return None
        
        image_url = response.output.results[0].url
        logging.info(f"Image generated: {image_url}")
        
        # Download image (DashScope URLs expire)
        img_response = requests.get(image_url, timeout=IMAGE_TIMEOUT)
        img_response.raise_for_status()
        
        logging.info("✅ Image downloaded successfully")
        return img_response.content
            
    except requests.RequestException as e:
        logging.error(f"Image download error: {e}")
        return None
    except Exception as e:
        logging.error(f"Image API error: {e}")
        return None

# ----------------------------
# Command Handlers
# ----------------------------
async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /start command"""
    await update.message.reply_text(
        "👋 Welcome to Metaphor Magic Bot!\n\n"
        "📝 Just send me any English metaphor and I'll:\n"
        "• Explain its meaning\n"
        "• Provide translations (ES/ZH/RU)\n"
        "• Show 4 contextual examples\n"
        "• Generate an AI illustration\n"
        "• Save it to our dictionary\n\n"
        "💡 Example: Try sending 'break the ice'"
    )

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /help command"""
    await update.message.reply_text(
        "🆘 *How to use this bot*\n\n"
        "1️⃣ Send any metaphor (e.g., 'think outside the box')\n"
        "2️⃣ Wait ~10-20 seconds for AI processing\n"
        "3️⃣ Content is posted to the group automatically\n\n"
        "⏱️ Rate limit: One metaphor every 30 seconds\n"
        "📏 Max length: 100 characters\n\n"
        "❓ Issues? Contact the bot admin.",
        parse_mode="Markdown"
    )

# ----------------------------
# Telegram Handler
# ----------------------------
async def handle_new_metaphor(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Main handler for incoming metaphor messages"""
    # Guard: ensure message and text exist
    if not update.message or not update.message.text:
        return
    
    metaphor = update.message.text.strip()
    if not metaphor:
        return

    # Validate length
    if len(metaphor) > MAX_METAPHOR_LENGTH:
        await update.message.reply_text(
            f"❌ Metaphor too long! Please keep it under {MAX_METAPHOR_LENGTH} characters."
        )
        return

    user = update.effective_user
    user_id = user.id
    user_name = user.first_name or "User"
    
    # Rate limiting with correct time calculation
    now = datetime.now()
    elapsed = (now - user_last_request[user_id]).total_seconds()
    
    if elapsed < COOLDOWN_SECONDS:
        remaining = int(COOLDOWN_SECONDS - elapsed)
        await update.message.reply_text(
            f"⏳ Please wait {remaining}s before sending another metaphor."
        )
        return
    
    user_last_request[user_id] = now
    logging.info(f"Processing metaphor '{metaphor}' from {user_name} (ID: {user_id})")

    await update.message.reply_text(
        f"🎨 Generating content for: *{escape_markdown(metaphor, version=2)}*\\.\\.\\.", 
        parse_mode="MarkdownV2"
    )

    # Step 1: Generate text content (async wrapper to avoid blocking)
    data = await asyncio.to_thread(generate_metaphor_data_cached, metaphor)
    
    if not data:
        await update.message.reply_text(
            "❌ Failed to generate metaphor data. Please try again or rephrase."
        )
        return

    image_prompt = data.get("image_prompt", "A creative visual representation")
    
    # Step 2: Generate image (async wrapper)
    image_bytes = await asyncio.to_thread(generate_metaphor_image, image_prompt)
    
    if image_bytes is None:
        await update.message.reply_text(
            "⚠️ Warning: Image generation failed. Sending text content only."
        )

    # Step 3: Format message with proper escaping
    safe_metaphor = escape_markdown(metaphor, version=2)
    safe_def = escape_markdown(data['definition'], version=2)
    safe_syn = escape_markdown(data['synonyms'], version=2)
    safe_es = escape_markdown(data['spanish'], version=2)
    safe_zh = escape_markdown(data['chinese'], version=2)
    safe_ru = escape_markdown(data['russian'], version=2)
    safe_office = escape_markdown(data['office_talk'], version=2)
    safe_parent = escape_markdown(data['parent_child'], version=2)
    safe_project = escape_markdown(data['project_team'], version=2)
    safe_friends = escape_markdown(data['lovers_friends'], version=2)
    safe_category = escape_markdown(data['category'], version=2)
    safe_prompt = escape_markdown(image_prompt, version=2)
    
    message_text = (
        f"🎯 *Metaphor*: {safe_metaphor}\n\n"
        f"📘 *Definition*: {safe_def}\n"
        f"🔗 *Synonyms*: {safe_syn}\n\n"
        f"🌍 *Translations*:\n"
        f"  • ES: {safe_es}\n"
        f"  • ZH: {safe_zh}\n"
        f"  • RU: {safe_ru}\n\n"
        f"💬 *Examples*:\n"
        f"• *Office*: {safe_office}\n"
        f"• *Parent\\-Child*: {safe_parent}\n"
        f"• *Project*: {safe_project}\n"
        f"• *Friends*: {safe_friends}\n\n"
        f"🔖 *Category*: `{safe_category}`\n\n"
        f"🎨 *Image Prompt*: _{safe_prompt}_"
    )

    # Step 4: Send to Telegram group
    try:
        if image_bytes:
            with temp_image_file(image_bytes, metaphor) as temp_path:
                with open(temp_path, 'rb') as photo:
                    await context.bot.send_photo(
                        chat_id=context.bot_data['group_chat_id'],
                        photo=photo,
                        caption=message_text,
                        parse_mode="MarkdownV2"
                    )
            logging.info(f"✅ Posted '{metaphor}' with image to group")
        else:
            await context.bot.send_message(
                chat_id=context.bot_data['group_chat_id'],
                text=message_text,
                parse_mode="MarkdownV2"
            )
            logging.info(f"✅ Posted '{metaphor}' (text only) to group")
            
    except Exception as e:
        logging.error(f"Telegram send error: {e}")
        await update.message.reply_text(
            "⚠️ Content generated but failed to post to group. Check bot permissions."
        )

    # Step 5: Save to Google Sheet (async wrapper)
    sheet_success = await asyncio.to_thread(
        append_to_sheet_with_retry,
        context.bot_data['sheet'],
        [
            metaphor,
            data["definition"],
            data["synonyms"],
            data["spanish"],
            data["chinese"],
            data["russian"],
            data["office_talk"],
            data["parent_child"],
            data["project_team"],
            data["lovers_friends"],
            data["category"],
            image_prompt
        ]
    )

    if sheet_success:
        await update.message.reply_text("✅ Done! Posted to group and saved to sheet.")
    else:
        await update.message.reply_text("✅ Posted to group, but failed to save to sheet.")

# ----------------------------
# Main
# ----------------------------
def main():
    """Initialize and run the bot"""
    try:
        # Validate environment variables
        env_vars = validate_environment()
        
        # DashScope setup
        dashscope.api_key = env_vars['DASHSCOPE_API_KEY']
        dashscope.base_http_api_url = 'https://dashscope-intl.aliyuncs.com/api/v1'
        
        # Google Sheets auth
        SCOPES = ["https://www.googleapis.com/auth/spreadsheets"]
        CREDS_PATH = "creds/gsheet_creds.json"
        creds = Credentials.from_service_account_file(CREDS_PATH, scopes=SCOPES)
        gc = gspread.authorize(creds)
        sheet = gc.open(env_vars['GOOGLE_SHEET_NAME']).sheet1
        
        # Validate sheet structure
        validate_sheet_headers(sheet)
        
        # Build application
        app = Application.builder().token(env_vars['TELEGRAM_BOT_TOKEN']).build()
        
        # Store shared resources in bot_data
        app.bot_data['sheet'] = sheet
        app.bot_data['group_chat_id'] = env_vars['TELEGRAM_GROUP_CHAT_ID']
        
        # Add handlers
        app.add_handler(CommandHandler("start", start_command))
        app.add_handler(CommandHandler("help", help_command))
        app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_new_metaphor))
        
        logging.info("🚀 Metaphor Magic Bot is running...")
        logging.info(f"📊 Connected to sheet: {env_vars['GOOGLE_SHEET_NAME']}")
        logging.info(f"💬 Posting to group: {env_vars['TELEGRAM_GROUP_CHAT_ID']}")
        logging.info(f"🤖 Using models: {LLM_MODEL} (text), {IMAGE_MODEL} (image)")
        
        app.run_polling()
        
    except Exception as e:
        logging.critical(f"Failed to start bot: {e}")
        raise

if __name__ == "__main__":
    main()