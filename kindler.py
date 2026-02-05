import os
import re
import smtplib
import uuid
import zipfile
import asyncio # Import asyncio
import threading
import random
import logging # Use logging instead of print
import html as html_lib
from queue import Queue
from urllib.parse import urlparse, urljoin
from email.mime.application import MIMEApplication
from email.mime.multipart import MIMEMultipart
# from email.mime.base import MIMEBase # Not used
from textwrap import wrap # Use standard library textwrap

import requests # Still using requests for now, consider httpx for async later
from PIL import Image, ImageDraw, ImageFont
from ebooklib import epub
from newspaper import Article # This should now work with lxml_html_clean installed

# Import v20 specific classes
from telegram import Update, InputFile, InlineKeyboardButton, InlineKeyboardMarkup, BotCommand
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    filters, # Filters are now in telegram.ext.filters
    PicklePersistence, # For user_data persistence
    ConversationHandler,
    CallbackQueryHandler
)
from telegram.constants import ParseMode # For ParseMode.MARKDOWN_V2 or HTML

from dotenv import load_dotenv # For loading environment variables

# --- Configuration (Improved: Use Environment Variables) ---
load_dotenv() # Load variables from a .env file (create this file)

# It's highly recommended to use environment variables for sensitive data
BOT_TOKEN = os.getenv('BOT_TOKEN')
EMAIL_HOST = os.getenv('EMAIL_HOST', 'smtp.gmail.com')
EMAIL_PORT = int(os.getenv('EMAIL_PORT', '587')) # Ensure port is int
SENDER_EMAIL = os.getenv('SENDER_EMAIL')
SENDER_PASSWORD = os.getenv('SENDER_PASSWORD') # For Gmail, use an App Password if 2FA is on

if not BOT_TOKEN or not SENDER_EMAIL or not SENDER_PASSWORD:
    print("CRITICAL: BOT_TOKEN, SENDER_EMAIL, or SENDER_PASSWORD not found in environment variables.")
    print("Please create a .env file or set them in your environment.")
    exit(1)


USER_COUNT_FILE = 'user_count.txt'
PERSISTENCE_FILE = 'bot_persistence.pkl' # File for user_data persistence

# --- Constants ---
GETTING_EMAIL = 1 # State for the email conversation
MILESTONES = [5, 10, 50, 100, 200, 300, 500, 1000, 1500, 2000]
MILESTONE_MESSAGES = {
    5: "🎉 Whoa, 5 articles! You're on a roll!",
    10: "🌟 10 articles already? Are you sure you're reading all of these?",
    50: "🚀 Half-century! 50 ebooks sent!",
    100: "💯 A century of ebooks!!",
    200: "🥳 200 articles! Your Kindle must be getting a workout!",
    300: "🤖 300! Are you sure you're not a bot?",
    500: "🌌 500 articles! Are you trying to download the whole Internet?",
    1000: "👑 1000 articles!",
    1500: "🦄 1500? I didn't think you'd get this far.",
    2000: "🌍 2000! I can't count more than this!"
}

# --- Inline Image Handling ---
ALLOW_INLINE_IMAGES = os.getenv('ALLOW_INLINE_IMAGES', '1') != '0'
ALLOW_BACKGROUND_IMAGES = os.getenv('ALLOW_BACKGROUND_IMAGES', '1') != '0'
MAX_INLINE_IMAGES = int(os.getenv('MAX_INLINE_IMAGES', '20'))
MAX_IMAGE_BYTES = int(os.getenv('MAX_IMAGE_BYTES', str(4 * 1024 * 1024)))

# --- Logging Setup ---
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
# Set higher logging level for httpx to avoid all GET requests being logged (if you use it)
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

# This line might be for specific deployment environments. Keep if needed.
# sys.stdout = sys.stderr

# --- User Count Management (Thread-based) ---
user_count_queue = Queue()
user_count_lock = threading.Lock()

def manage_user_count_thread_target():
    while True:
        task_info = user_count_queue.get()
        if task_info is None:  # Signal to stop the thread
            break
        task, result_container, event = task_info
        try:
            result_container[0] = task()
        except Exception as e:
            logger.error(f"Error in user count task: {e}", exc_info=True)
            result_container[0] = None # Indicate error
        finally:
            if event:
                event.set() # Signal completion
            user_count_queue.task_done()

def _increment_and_get_user_count_from_file():
    """Internal function to handle file I/O for user count."""
    with user_count_lock:
        try:
            with open(USER_COUNT_FILE, 'r+') as file:
                count = int(file.read().strip()) + 1
                file.seek(0)
                file.write(str(count))
                file.truncate()
                return count
        except (FileNotFoundError, ValueError):
            with open(USER_COUNT_FILE, 'w') as file:
                file.write('1')
            return 1

async def get_next_user_number_async():
    """Async wrapper to get next user number from the threaded queue."""
    result_container = [None]
    event = asyncio.Event() # Use asyncio.Event for async waiting

    # Define the task to be run in the thread
    def task_to_run_in_thread():
        return _increment_and_get_user_count_from_file()

    # Put the task and its completion signaling mechanism into the queue
    user_count_queue.put((task_to_run_in_thread, result_container, None)) # event not used here, direct join

    # This is tricky. user_count_queue.join() is blocking.
    # For a truly async call, the thread should signal an asyncio.Event
    # Let's simplify: run the file operation in asyncio.to_thread directly for this.
    # The queue system was more for a continuous worker.

    # Simpler approach for one-off async file op:
    loop = asyncio.get_running_loop()
    new_count = await loop.run_in_executor(None, _increment_and_get_user_count_from_file)
    return new_count


# --- Helper Functions ---
def number_to_ordinal(n: int) -> str:
    if 10 <= n % 100 <= 20:
        suffix = 'th'
    else:
        suffix = {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')
    return str(n) + suffix

def is_url(url: str) -> bool:
    if not isinstance(url, str):
        return False
    try:
        result = urlparse(url)
        return all([result.scheme in ['http', 'https'], result.netloc])
    except ValueError:
        return False

def sanitize_filename(title: str) -> str:
    if not title: title = "Untitled"
    title = re.sub(r'[<>:"/\\|?*]', '_', title) # Replace invalid chars with underscore
    title = re.sub(r'\s+', ' ', title).strip()
    return title[:150] # Limit length

def _text_to_html(content_text: str) -> str:
    paragraphs = content_text.split('\n\n')
    html_paragraphs = []
    for para in paragraphs:
        if para.strip():
            escaped = html_lib.escape(para)
            processed_para = escaped.replace('\n', '<br />\n')
            html_paragraphs.append(f'<p>{processed_para}</p>')
    return '\n'.join(html_paragraphs)

def _sanitize_html(content_html: str) -> str:
    if not content_html:
        return content_html
    try:
        from lxml import html as lxml_html
    except Exception:
        return content_html

    try:
        wrapper = lxml_html.fragment_fromstring(content_html, create_parent=True)
    except Exception:
        try:
            doc = lxml_html.fromstring(content_html)
            wrapper = doc.find('body') or doc
        except Exception:
            return content_html

    try:
        for bad in wrapper.xpath('.//script|.//style|.//noscript|.//iframe|.//form|.//svg'):
            bad.drop_tree()
    except Exception:
        pass

    try:
        if wrapper.tag.lower() in ('html', 'body'):
            parts = []
            if wrapper.text:
                parts.append(wrapper.text)
            for child in wrapper:
                parts.append(lxml_html.tostring(child, encoding='unicode', method='html'))
            return ''.join(parts)
        return lxml_html.tostring(wrapper, encoding='unicode', method='html')
    except Exception:
        return content_html

def _extract_with_trafilatura(html_text: str, url: str) -> str:
    if not html_text:
        return None
    try:
        import trafilatura
    except Exception as e:
        logger.debug(f"trafilatura not available: {e}")
        return None

    try:
        extracted = trafilatura.extract(
            html_text,
            url=url,
            include_images=True,
            include_formatting=True,
            output_format='html'
        )
        if extracted and extracted.strip():
            return _sanitize_html(extracted)
    except Exception as e:
        logger.debug(f"trafilatura extraction failed: {e}")
    return None

def _extract_title_from_html(html_text: str) -> str:
    if not html_text:
        return None
    try:
        from lxml import html as lxml_html
    except Exception:
        return None
    try:
        doc = lxml_html.fromstring(html_text)
        title = doc.findtext('.//title')
        return title.strip() if title else None
    except Exception:
        return None

def _extract_main_html(article):
    try:
        from lxml import html as lxml_html
    except Exception as e:
        logger.debug(f"lxml not available for HTML extraction: {e}")
        return None

    node = None
    for attr in ('clean_top_node', 'top_node'):
        node = getattr(article, attr, None)
        if node is not None:
            break

    if node is None:
        article_html = getattr(article, 'article_html', None)
        if isinstance(article_html, str) and article_html.strip():
            return _sanitize_html(article_html)
        return None

    try:
        for bad in node.xpath('.//script|.//style|.//noscript|.//iframe|.//form|.//svg'):
            bad.drop_tree()
    except Exception:
        pass

    try:
        raw_html = lxml_html.tostring(node, encoding='unicode', method='html')
        return _sanitize_html(raw_html)
    except Exception as e:
        logger.debug(f"Failed to serialize main HTML: {e}")
        return None

def _pick_src_from_srcset(srcset: str) -> str:
    if not srcset:
        return None
    best_url = None
    best_score = -1
    for part in srcset.split(','):
        bits = part.strip().split()
        if not bits:
            continue
        url = bits[0]
        score = 0
        if len(bits) > 1:
            desc = bits[1]
            try:
                if desc.endswith('w'):
                    score = int(desc[:-1])
                elif desc.endswith('x'):
                    score = float(desc[:-1]) * 1000
            except ValueError:
                score = 0
        if score >= best_score:
            best_score = score
            best_url = url
    return best_url

def _normalize_image_url(raw_url: str, base_url: str) -> str:
    if not raw_url:
        return None
    raw_url = raw_url.strip()
    if raw_url.startswith('data:'):
        return None
    if raw_url.startswith('//'):
        raw_url = f'https:{raw_url}'
    return urljoin(base_url, raw_url) if base_url else raw_url

def _extract_bg_urls_from_style(style_value: str):
    if not style_value:
        return []
    urls = []
    for match in re.findall(r'url\\(([^)]+)\\)', style_value, flags=re.IGNORECASE):
        candidate = match.strip().strip('"').strip("'")
        if candidate:
            urls.append(candidate)
    return urls

def _remove_bg_from_style(style_value: str) -> str:
    if not style_value:
        return style_value
    cleaned = re.sub(r'background-image\\s*:\\s*url\\([^)]*\\)\\s*;?', '', style_value, flags=re.IGNORECASE)
    cleaned = re.sub(r'background\\s*:\\s*url\\([^)]*\\)\\s*;?', '', cleaned, flags=re.IGNORECASE)
    return cleaned.strip()

def _mimetype_to_extension(mimetype: str, source_url: str) -> str:
    ext_map = {
        'image/jpeg': '.jpg',
        'image/jpg': '.jpg',
        'image/png': '.png',
        'image/gif': '.gif',
        'image/webp': '.webp'
    }
    ext = ext_map.get(mimetype)
    if ext:
        return ext
    path_ext = os.path.splitext(urlparse(source_url).path)[1]
    return path_ext if path_ext else '.jpg'

def fetch_image_bytes_sync(url: str, max_bytes: int = MAX_IMAGE_BYTES):
    if not url or url.startswith('data:'):
        return None, None
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
        response = requests.get(url, timeout=15, stream=True, headers=headers)
        response.raise_for_status()
        content_type = response.headers.get('content-type', '').split(';')[0].strip()
        if not content_type.startswith('image/'):
            logger.warning(f"URL {url} did not return an image content-type ({content_type}). Skipping.")
            return None, None
        data = bytearray()
        for chunk in response.iter_content(chunk_size=8192):
            if not chunk:
                continue
            data.extend(chunk)
            if max_bytes and len(data) > max_bytes:
                logger.warning(f"Image too large ({len(data)} bytes), skipping: {url}")
                return None, None
        return bytes(data), content_type
    except requests.exceptions.RequestException as e:
        logger.error(f"Error downloading image {url}: {e}")
        return None, None

def _embed_inline_images(book, content_html: str, base_url: str, top_image_url: str = None) -> str:
    if not ALLOW_INLINE_IMAGES or not content_html:
        return content_html
    try:
        from lxml import html as lxml_html
    except Exception as e:
        logger.debug(f"lxml not available for inline images: {e}")
        return content_html

    try:
        wrapper = lxml_html.fragment_fromstring(content_html, create_parent=True)
    except Exception as e:
        logger.debug(f"Failed to parse content HTML for inline images: {e}")
        return content_html

    def _inner_html(node):
        parts = []
        if node.text:
            parts.append(node.text)
        for child in node:
            parts.append(lxml_html.tostring(child, encoding='unicode', method='html'))
        return ''.join(parts)

    top_abs = _normalize_image_url(top_image_url, base_url) if top_image_url else None
    seen = {}
    embedded_count = 0

    if ALLOW_BACKGROUND_IMAGES:
        try:
            for elem in list(wrapper.xpath('.//*[@style]')):
                style_value = elem.get('style', '')
                if 'background' not in style_value:
                    continue
                bg_urls = _extract_bg_urls_from_style(style_value)
                if not bg_urls:
                    continue
                new_style = _remove_bg_from_style(style_value)
                if new_style:
                    elem.set('style', new_style)
                else:
                    if 'style' in elem.attrib:
                        del elem.attrib['style']
                bg_url = bg_urls[0]
                img_elem = lxml_html.Element('img')
                img_elem.set('src', bg_url)
                img_elem.set('data-bg-image', '1')
                elem.insert(0, img_elem)
        except Exception as e:
            logger.debug(f"Failed to process background images: {e}")

    for img in list(wrapper.iter('img')):
        if embedded_count >= MAX_INLINE_IMAGES:
            parent = img.getparent()
            if parent is not None:
                parent.remove(img)
            continue

        raw_src = None
        for attr in ('data-src', 'data-original', 'data-lazy-src', 'data-srcset', 'srcset', 'src'):
            val = img.get(attr)
            if val:
                if 'srcset' in attr:
                    val = _pick_src_from_srcset(val)
                raw_src = val
                break
        if not raw_src:
            parent = img.getparent()
            if parent is not None and parent.tag.lower() == 'picture':
                for source in parent.findall('source'):
                    for attr in ('data-srcset', 'srcset', 'data-src', 'src'):
                        val = source.get(attr)
                        if val:
                            if 'srcset' in attr:
                                val = _pick_src_from_srcset(val)
                            raw_src = val
                            break
                    if raw_src:
                        break

        abs_url = _normalize_image_url(raw_src, base_url)
        if not abs_url:
            parent = img.getparent()
            if parent is not None:
                parent.remove(img)
            continue

        if top_abs and abs_url == top_abs:
            parent = img.getparent()
            if parent is not None:
                parent.remove(img)
            continue

        if abs_url in seen:
            img.set('src', seen[abs_url])
            img.set('class', (img.get('class', '') + ' epub-inline-image').strip())
            for attr in list(img.attrib):
                if attr.startswith('data-') or attr in ('srcset', 'sizes'):
                    del img.attrib[attr]
            continue

        img_bytes, mimetype = fetch_image_bytes_sync(abs_url)
        if not img_bytes:
            parent = img.getparent()
            if parent is not None:
                parent.remove(img)
            continue

        ext = _mimetype_to_extension(mimetype, abs_url)
        epub_image_path_in_epub = f"images/inline_{uuid.uuid4()}{ext}"
        epub_img_item = epub.EpubImage(
            uid=f"img_{uuid.uuid4()}",
            file_name=epub_image_path_in_epub,
            media_type=mimetype,
            content=img_bytes
        )
        book.add_item(epub_img_item)

        img.set('src', epub_image_path_in_epub)
        img.set('class', (img.get('class', '') + ' epub-inline-image').strip())
        for attr in list(img.attrib):
            if attr.startswith('data-') or attr in ('srcset', 'sizes'):
                del img.attrib[attr]

        seen[abs_url] = epub_image_path_in_epub
        embedded_count += 1

    return _inner_html(wrapper)

def _append_fallback_gallery(content_html: str, image_urls, base_url: str, top_image_url: str = None) -> str:
    if not ALLOW_INLINE_IMAGES or not image_urls:
        return content_html
    unique_urls = []
    seen = set()
    top_abs = _normalize_image_url(top_image_url, base_url) if top_image_url else None
    for raw in image_urls:
        abs_url = _normalize_image_url(raw, base_url)
        if not abs_url:
            continue
        if top_abs and abs_url == top_abs:
            continue
        if abs_url in seen:
            continue
        seen.add(abs_url)
        unique_urls.append(abs_url)
        if len(unique_urls) >= MAX_INLINE_IMAGES:
            break

    if not unique_urls:
        return content_html

    gallery_parts = ['<hr/>', '<p><b>Images</b></p>']
    for url in unique_urls:
        gallery_parts.append(f'<img src="{html_lib.escape(url)}" alt="Image" />')
    return content_html + '\n' + '\n'.join(gallery_parts)

# --- Core Bot Logic Functions (Blocking - to be run with asyncio.to_thread) ---

def get_article_content_sync(url: str):
    try:
        logger.info(f"Fetching article: {url}")
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
        article = Article(url, headers=headers, request_timeout=20) # Added timeout
        article.download()
        if article.download_state != 2: # 2 means success
             logger.error(f"Failed to download article from {url}. State: {article.download_state}, HTML: {article.html[:200]}")
             return None
        article.parse()
        if not article.is_parsed:
            logger.warning(f"Failed to parse article from {url}; continuing with fallback extractors.")

        title = article.title or _extract_title_from_html(article.html) or "Untitled Article"
        content_html = _extract_with_trafilatura(article.html, url)
        if not content_html:
            content_html = _extract_main_html(article)
        used_html_extraction = bool(content_html and content_html.strip())
        if not used_html_extraction:
            content_text = article.text
            if not content_text:
                logger.warning(f"No main text content extracted from {url}")
                return None # Or handle as error in the calling function
            content_html = _text_to_html(content_text)

        top_image_url = article.top_image
        domain = urlparse(url).netloc
        if content_html and article.images and not used_html_extraction:
            content_html = _append_fallback_gallery(content_html, article.images, url, top_image_url)

        logger.info(f"Successfully parsed: '{title}' from {domain}")
        return title, content_html, domain, top_image_url, url
    except Exception as e:
        logger.error(f"Error processing article {url}: {e}", exc_info=True)
        return None

def download_image_sync(url: str, filename: str):
    if not url or url.startswith('data:image'):
        logger.warning(f"Skipping invalid or data URI image URL: {url[:60]}...")
        return None
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
        response = requests.get(url, timeout=15, stream=True, headers=headers)
        response.raise_for_status()
        content_type = response.headers.get('content-type')
        if content_type and not content_type.startswith('image/'):
             logger.warning(f"URL {url} did not return an image content-type ({content_type}). Skipping.")
             return None
        with open(filename, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        logger.debug(f"Image downloaded successfully to {filename}")
        return filename
    except requests.exceptions.RequestException as e:
        logger.error(f"Error downloading image {url}: {e}")
        return None
    except IOError as e:
        logger.error(f"Error saving image {url} to {filename}: {e}")
        return None

def create_cover_sync(title: str, domain: str, top_image_url: str = None):
    cover_width, cover_height = 1072, 1488
    max_image_width = 900
    temp_dl_image_path = None
    cover_filename = f"cover_{uuid.uuid4()}.jpeg"

    try:
        img = Image.new('RGB', (cover_width, cover_height), color=(255, 255, 255))
        d = ImageDraw.Draw(img)

        # --- Header Bar ---
        d.rectangle([(0, 0), (cover_width, 200)], fill=(64, 64, 64))
        logo_path = "logo.jpg" # Ensure this file exists or handle FileNotFoundError
        logo_width_for_text_pos = 0
        if os.path.isfile(logo_path):
            try:
                logo_image_orig = Image.open(logo_path)
                max_logo_size = 125
                logo_image_orig.thumbnail((max_logo_size, max_logo_size), Image.LANCZOS)
                
                # Circular mask
                mask = Image.new('L', logo_image_orig.size, 0)
                mask_draw = ImageDraw.Draw(mask)
                mask_draw.ellipse((0, 0) + logo_image_orig.size, fill=255)
                
                # Create a new image with RGBA to apply the mask, then paste onto main image
                logo_rgba = logo_image_orig.convert("RGBA")
                logo_rgba.putalpha(mask)

                logo_w, logo_h = logo_rgba.size
                logo_width_for_text_pos = logo_w
                logo_pos = (50, (200 - logo_h) // 2)
                img.paste(logo_rgba, logo_pos, logo_rgba) # Paste with alpha
            except Exception as e:
                logger.error(f"Failed to process logo image {logo_path}: {e}")
        
        # --- App Name ---
        app_name = "The Kindler"
        try:
            app_name_font = ImageFont.truetype('Bookerly.ttf', 80)
        except IOError:
            logger.warning("Bookerly.ttf not found, using default font for app name.")
            app_name_font = ImageFont.load_default()
        
        app_name_x = (logo_width_for_text_pos + 50 + 30) if logo_width_for_text_pos > 0 else 50
        app_name_bbox = d.textbbox((0,0), app_name, font=app_name_font)
        app_name_h = app_name_bbox[3] - app_name_bbox[1]
        app_name_y = (200 - app_name_h) // 2 - app_name_bbox[1] # Adjust for bbox top
        d.text((app_name_x, app_name_y), app_name, fill=(255, 255, 255), font=app_name_font)

        # --- Title ---
        try:
            title_font = ImageFont.truetype('Aileron-Bold.otf', 100)
        except IOError:
            logger.warning("Aileron-Bold.otf not found, using default font for title.")
            title_font = ImageFont.load_default()

        # Approximate wrap width (Pillow text wrapping is basic)
        # A better approach might be to measure text line by line
        avg_char_width = title_font.getlength("A") if hasattr(title_font, 'getlength') else 10
        wrap_chars = int((cover_width * 0.85) / avg_char_width) if avg_char_width > 0 else 20
        title_lines = wrap(title, width=max(15, wrap_chars))
        
        current_h = 225.0
        line_spacing_title = 1.1 # Multiplier for line height

        for line in title_lines:
            line_bbox = d.textbbox((0,0), line, font=title_font)
            line_w = line_bbox[2] - line_bbox[0]
            line_h_actual = line_bbox[3] - line_bbox[1] # Actual height of glyphs
            
            draw_y = current_h - line_bbox[1] # Adjust y by bbox top for consistent spacing
            d.text(((cover_width - line_w) / 2, draw_y), line, fill=(0, 0, 0), font=title_font)
            current_h += line_h_actual * line_spacing_title


        # --- Domain ---
        try:
            domain_font = ImageFont.truetype('Bookerly.ttf', 50)
        except IOError:
            domain_font = ImageFont.load_default()
        domain_bbox = d.textbbox((0,0), domain, font=domain_font)
        domain_w = domain_bbox[2] - domain_bbox[0]
        domain_h_actual = domain_bbox[3] - domain_bbox[1]
        domain_y_base = cover_height - domain_h_actual - 100 # Position from bottom
        domain_y_draw = domain_y_base - domain_bbox[1]
        d.text(((cover_width - domain_w) / 2, domain_y_draw), domain, fill=(0, 0, 0), font=domain_font)
        
        # --- Top Image ---
        image_bottom_margin = 30
        available_height_for_image = domain_y_base - current_h - image_bottom_margin

        if top_image_url and available_height_for_image > 100:
            temp_dl_image_path = download_image_sync(top_image_url, f"temp_cover_img_{uuid.uuid4()}.jpg")
            if temp_dl_image_path:
                try:
                    top_img_pil = Image.open(temp_dl_image_path)
                    w, h = top_img_pil.size
                    aspect = w / h

                    new_w_img = max_image_width
                    new_h_img = new_w_img / aspect
                    if new_h_img > available_height_for_image:
                        new_h_img = available_height_for_image
                        new_w_img = new_h_img * aspect
                    new_w_img = min(max_image_width, new_w_img) # Ensure width constraint

                    if new_w_img > 0 and new_h_img > 0:
                        top_img_pil = top_img_pil.resize((int(new_w_img), int(new_h_img)), Image.LANCZOS)
                        img_x = (cover_width - int(new_w_img)) // 2
                        img_y = int(current_h + (available_height_for_image - new_h_img) / 2) # Center in available space
                        img.paste(top_img_pil, (img_x, img_y))
                except Exception as e:
                    logger.error(f"Failed to process downloaded top image {top_image_url}: {e}")
                finally:
                    if temp_dl_image_path and os.path.exists(temp_dl_image_path):
                        os.remove(temp_dl_image_path)
                        temp_dl_image_path = None
        
        img.save(cover_filename, 'JPEG', quality=85)
        logger.debug(f"Cover image created: {cover_filename}")
        return cover_filename
    except Exception as e:
        logger.error(f"Failed to create cover image for '{title}': {e}", exc_info=True)
        if os.path.exists(cover_filename): os.remove(cover_filename) # Cleanup partial
        if temp_dl_image_path and os.path.exists(temp_dl_image_path): os.remove(temp_dl_image_path)
        return None


def create_epub_sync(title: str, content_html: str, cover_path: str, domain: str, source_url: str, top_image_url: str = None):
    epub_filename = f"{sanitize_filename(title)}_{uuid.uuid4()}.epub"
    temp_epub_top_image_path = None

    try:
        book = epub.EpubBook()
        book.set_identifier(str(uuid.uuid4()))
        book.set_title(title)
        book.add_author(domain)
        book.set_language('en')

        if not cover_path or not os.path.exists(cover_path):
            logger.error("Cover path invalid for EPUB creation.")
            return None, None
        with open(cover_path, 'rb') as cf:
            book.set_cover("cover.jpeg", cf.read())

        # CSS
        style = '''
        body { font-family: sans-serif; line-height: 1.6; margin: 2%; }
        h1 { text-align: center; margin-top: 1em; margin-bottom: 0.5em; font-size: 1.8em; }
        img.epub-top-image { max-width: 90%; height: auto; display: block; margin: 1em auto; border: 1px solid #eee; }
        img.epub-inline-image { max-width: 100%; height: auto; display: block; margin: 1em auto; }
        p { margin-bottom: 1em; text-align: justify; }
        hr { border: 0; height: 1px; background: #ccc; margin: 1.5em 0; }
        '''
        css_item = epub.EpubItem(uid="style_main", file_name="style/main.css", media_type="text/css", content=style)
        book.add_item(css_item)

        # Chapter content construction
        chapter_html_content = f'<h1>{title}</h1>'
        content_html = content_html or ""

        if top_image_url:
            img_ext = os.path.splitext(urlparse(top_image_url).path)[1] or ".jpg"
            epub_image_filename_base = f"top_image_epub_{uuid.uuid4()}"
            epub_image_filename_fs = f"{epub_image_filename_base}{img_ext}" # For filesystem
            epub_image_path_in_epub = f"images/{epub_image_filename_base}{img_ext}" # Path inside EPUB

            temp_epub_top_image_path = download_image_sync(top_image_url, epub_image_filename_fs)
            if temp_epub_top_image_path:
                try:
                    with open(temp_epub_top_image_path, 'rb') as img_f:
                        img_content_bytes = img_f.read()
                    
                    mimetype = 'image/jpeg'
                    if img_ext.lower() == '.png': mimetype = 'image/png'
                    elif img_ext.lower() == '.gif': mimetype = 'image/gif'

                    epub_img_item = epub.EpubImage(
                        uid="img_top_content",
                        file_name=epub_image_path_in_epub,
                        media_type=mimetype,
                        content=img_content_bytes
                    )
                    book.add_item(epub_img_item)
                    chapter_html_content += f'<img src="{epub_image_path_in_epub}" alt="Top Image" class="epub-top-image"/>'
                except Exception as e:
                    logger.error(f"Failed to add top image to EPUB content: {e}")
                    if temp_epub_top_image_path and os.path.exists(temp_epub_top_image_path):
                        os.remove(temp_epub_top_image_path)
                        temp_epub_top_image_path = None # Reset if failed
        
        if content_html:
            content_html = _embed_inline_images(book, content_html, source_url, top_image_url)
        chapter_html_content += f'<hr/>{content_html}'

        # Create chapter
        chap1 = epub.EpubHtml(title=title, file_name='chap_01.xhtml', lang='en')
        chap1.content = chapter_html_content
        chap1.add_item(css_item) # Link CSS
        book.add_item(chap1)

        # Define Spine and TOC
        book.spine = ['nav', chap1] # 'cover' is implicitly handled by set_cover
        book.toc = (epub.Link('chap_01.xhtml', title, 'chap1'),)
        book.add_item(epub.EpubNcx())
        book.add_item(epub.EpubNav())

        epub.write_epub(epub_filename, book, {})
        logger.info(f"EPUB created: {epub_filename}")
        return epub_filename, temp_epub_top_image_path # Return path of image used in EPUB for cleanup
    except Exception as e:
        logger.error(f"Failed to create EPUB for '{title}': {e}", exc_info=True)
        if os.path.exists(epub_filename): os.remove(epub_filename)
        if temp_epub_top_image_path and os.path.exists(temp_epub_top_image_path): os.remove(temp_epub_top_image_path)
        return None, None

def send_to_kindle_sync(file_path: str, kindle_email: str):
    if not SENDER_EMAIL or not SENDER_PASSWORD:
        logger.error("Sender email credentials not configured.")
        return False
    if not kindle_email:
        logger.error("Recipient Kindle email is missing.")
        return False
    if not os.path.exists(file_path):
        logger.error(f"EPUB file {file_path} not found for sending.")
        return False

    try:
        msg = MIMEMultipart()
        msg['From'] = SENDER_EMAIL
        msg['To'] = kindle_email
        msg['Subject'] = "Convert" # Kindle often prefers "Convert" or empty subject for .epub

        with open(file_path, 'rb') as f:
            part = MIMEApplication(f.read(), Name=os.path.basename(file_path))
        part['Content-Disposition'] = f'attachment; filename="{os.path.basename(file_path)}"'
        msg.attach(part)

        logger.info(f"Attempting to send '{os.path.basename(file_path)}' to {kindle_email} via {EMAIL_HOST}:{EMAIL_PORT}")
        with smtplib.SMTP(EMAIL_HOST, EMAIL_PORT) as server:
            server.starttls()
            server.login(SENDER_EMAIL, SENDER_PASSWORD)
            server.send_message(msg)
        logger.info(f"Email sent successfully to {kindle_email}")
        return True
    except smtplib.SMTPException as e:
        logger.error(f"SMTP error sending email to {kindle_email}: {e}", exc_info=True)
    except Exception as e:
        logger.error(f"Unexpected error sending email: {e}", exc_info=True)
    return False

def cleanup_files_sync(*paths):
    for path in paths:
        if path and os.path.exists(path):
            try:
                os.remove(path)
                logger.debug(f"Cleaned up file: {path}")
            except OSError as e:
                logger.error(f"Error removing file {path}: {e}")


# --- Telegram Handlers (async def) ---

async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user = update.effective_user
    logger.info(f"User {user.id} ({user.full_name}) started the bot.")

    if 'first_time_user_msg_sent' not in context.user_data:
        try:
            user_number = await get_next_user_number_async()
            await update.message.reply_text(f"Welcome! You're the {number_to_ordinal(user_number)} user to start me!")
            context.user_data['first_time_user_msg_sent'] = True
        except Exception as e:
            logger.error(f"Failed to get/send user number for {user.id}: {e}")


    if 'kindle_email' in context.user_data:
        await update.message.reply_html(
            f"Welcome back, {user.mention_html()}!\n"
            f"Your Kindle email is set to: <code>{context.user_data['kindle_email']}</code>\n"
            f"Send me article URLs to convert. Use /email to change it or /help for info.\n\n"
            f"✅ Make sure <code>{SENDER_EMAIL}</code> is on your Kindle's approved senders list."
        )
    else:
        await update.message.reply_html(
            f"Hello {user.mention_html()}! I am The Kindler! 🤖\n\n"
            "I convert article URLs 📰 into ebooks 📚 and send them to your Kindle.\n\n"
            "<b>First, please set your Kindle email address using the command:</b>\n"
            "/email <code>your_kindle_email@kindle.com</code>\n\n"
            f"⚠️ Make sure <code>{SENDER_EMAIL}</code> is on your Kindle's approved senders list!"
        )

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    help_text = (
        "<b>How to Use The Kindler Bot</b>\n\n"
        "1.  <b>Set Your Kindle Email:</b>\n"
        "    Use: /email <code>your_kindle_email@kindle.com</code>\n"
        f"    (Replace with your actual Kindle email. Ensure <code>{SENDER_EMAIL}</code> is in your Kindle's approved list.)\n\n"
        "<b>Important:</b> Add the sender email to your Kindle's approved senders list:\n"
        f"<code>{SENDER_EMAIL}</code>\n\n"
        "2.  <b>Send Article URLs:</b>\n"
        "    Paste one or more URLs (separated by spaces, commas, or newlines).\n\n"
        "3.  <b>Receive Ebook:</b>\n"
        "    I'll process each URL, send the EPUB to your Kindle, and upload a copy here.\n\n"
        "<b>Commands:</b>\n"
        "/start - Welcome message.\n"
        "/email <code><email></code> - Set/update your Kindle email.\n"
        "/myemail - Show your current Kindle email.\n"
        "/help - This help message.\n\n"
        "<i>Note: Conversion quality depends on website structure.</i>"
    )
    await update.message.reply_html(help_text)


async def set_email_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Starts the conversation and asks the user for their email with a cancel button."""
    keyboard = [
        [InlineKeyboardButton("Cancel", callback_data='cancel_email_input')]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    # Send the prompt and store its message_id to edit it later
    prompt_message = await update.message.reply_text(
        "Please send your Kindle email address.",
        reply_markup=reply_markup
    )
    context.user_data['prompt_message_id'] = prompt_message.message_id
    
    return GETTING_EMAIL


async def received_email(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Validates and saves the email, then ends the conversation."""
    user_id = update.effective_user.id
    email = update.message.text.strip().lower()
    prompt_message_id = context.user_data.get('prompt_message_id')

    if not re.match(r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$", email):
        await update.message.reply_text(
             f"'{email}' doesn't look like a valid email. Please try again."
        )
        # The user can try again or press the cancel button on the original prompt
        return GETTING_EMAIL

    context.user_data['kindle_email'] = email
    logger.info(f"User {user_id} set Kindle email to {email}")
    
    # Edit the original prompt to remove the button and show it's completed
    if prompt_message_id:
        try:
            await context.bot.edit_message_text(
                chat_id=update.effective_chat.id,
                message_id=prompt_message_id,
                text="Email address received."
            )
        except Exception as e:
            logger.warning(f"Could not edit prompt message {prompt_message_id}: {e}")
        del context.user_data['prompt_message_id'] # Clean up

    await update.message.reply_html(
        f"✅ Kindle email updated to: <code>{email}</code>\n"
        f"Remember to add <code>{SENDER_EMAIL}</code> to your Kindle's approved senders list!\n\n"
        "You can now send me article URLs."
    )
    return ConversationHandler.END


async def cancel_email_entry(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Cancels the email entry process when the inline button is pressed."""
    query = update.callback_query
    await query.answer() # Acknowledge the button press
    await query.edit_message_text(text="Operation cancelled.")
    
    if 'prompt_message_id' in context.user_data:
        del context.user_data['prompt_message_id'] # Clean up

    return ConversationHandler.END

async def show_email_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if 'kindle_email' in context.user_data:
        email = context.user_data['kindle_email']
        await update.message.reply_html(
            f"Your current Kindle email is: <code>{email}</code>\n"
            f"✅ Approved sender: <code>{SENDER_EMAIL}</code>"
        )
    else:
        await update.message.reply_html(
            "You haven't set your Kindle email yet. Use:\n"
            "/email <code>your_name@kindle.com</code>"
        )

def process_single_url_task(url: str, kindle_email: str, user_id: int):
    """Synchronous part of processing a single URL, designed to be run in a thread."""
    logger.info(f"[User:{user_id}] Starting processing for URL: {url}")
    article_data = get_article_content_sync(url)
    if not article_data:
        return {"status": "error", "message": "✖️ Failed to extract content.", "files_to_clean": []}

    title, content_html, domain, top_image_url, source_url = article_data
    s_title = sanitize_filename(title)

    cover_path = create_cover_sync(s_title, domain, top_image_url)
    if not cover_path:
        return {"status": "error", "message": "😰 Error creating cover image.", "files_to_clean": []}

    epub_path, epub_temp_image_path = create_epub_sync(s_title, content_html, cover_path, domain, source_url, top_image_url)
    files_to_clean_up = [cover_path, epub_path, epub_temp_image_path]

    if not epub_path:
        cleanup_files_sync(*files_to_clean_up)
        return {"status": "error", "message": "😰 Error creating the ebook.", "files_to_clean": []}

    email_sent_successfully = send_to_kindle_sync(epub_path, kindle_email)
    
    return {
        "status": "success",
        "epub_path": epub_path,
        "email_sent": email_sent_successfully,
        "files_to_clean": files_to_clean_up,
        "original_url": url
    }


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user = update.effective_user
    message_text = update.message.text

    if 'kindle_email' not in context.user_data:
        await update.message.reply_html(
            "⚠️ Please set your Kindle email first:\n"
            "/email <code>your_kindle_email@kindle.com</code>"
        )
        return

    KINDLE_EMAIL = context.user_data['kindle_email']

    # Extract URLs (improved)
    entities = update.message.entities or []
    urls_from_entities = [entity.url for entity in entities if entity.type == 'text_link']
    # Regex to find http/https URLs in the plain text part, robustly
    urls_from_text = re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*(),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', message_text)
    
    raw_urls = list(set(urls_from_entities + urls_from_text)) # Unique URLs
    valid_urls = [url for url in raw_urls if is_url(url)]

    if not valid_urls:
        if not message_text.startswith('/'): # Avoid replying to unhandled commands
             await update.message.reply_text('❕ Please send one or more valid article URLs.')
        return

    logger.info(f"User {user.id} sent {len(valid_urls)} URLs: {valid_urls}")
    
    initial_reply = await update.message.reply_text(f"Found {len(valid_urls)} URL(s). Processing...")

    processed_successfully_count = 0
    loop = asyncio.get_running_loop()

    for i, url_to_process in enumerate(valid_urls):
        await context.bot.edit_message_text(
            chat_id=update.effective_chat.id,
            message_id=initial_reply.message_id,
            text=f"Processing URL {i+1}/{len(valid_urls)}: {url_to_process}",
            disable_web_page_preview=True
        )
        
        try:
            # Run the blocking processing in a separate thread
            result = await loop.run_in_executor(
                None,  # Uses default ThreadPoolExecutor
                process_single_url_task,
                url_to_process,
                KINDLE_EMAIL,
                user.id
            )
            
            # Handle result
            if result["status"] == "error":
                await update.message.reply_text(f"{result['message']}\nFailed URL: {url_to_process}", disable_web_page_preview=True)
            else: # Success
                epub_path_res = result["epub_path"]
                email_sent_res = result["email_sent"]
                
                if email_sent_res:
                    await update.message.reply_text(f"✅ Ebook sent to Kindle!\nSource: {result['original_url']}", disable_web_page_preview=True)
                else:
                    await update.message.reply_text(f"⚠️ Ebook created, but FAILED to send email. Check settings.\nSource: {result['original_url']}", disable_web_page_preview=True)

                try:
                    await update.message.reply_document(document=open(epub_path_res, 'rb'))
                    processed_successfully_count += 1
                except Exception as e_doc:
                    logger.error(f"Failed to upload EPUB {epub_path_res} to chat: {e_doc}")
                    await update.message.reply_text(f"📎 Ebook created, but couldn't upload it here.\nSource: {result['original_url']}", disable_web_page_preview=True)
            
            # Cleanup files for this URL
            await loop.run_in_executor(None, cleanup_files_sync, *result["files_to_clean"])

        except Exception as e_outer:
            logger.error(f"Outer error processing URL {url_to_process} for user {user.id}: {e_outer}", exc_info=True)
            await update.message.reply_text(f"💥 An unexpected critical error occurred for URL:\n{url_to_process}", disable_web_page_preview=True)
            # Attempt to clean any known temp files if an unexpected error occurs
            # This is harder as we don't know which files were created.
            # The process_single_url_task should handle its own cleanup on failure.

    await context.bot.delete_message(chat_id=update.effective_chat.id, message_id=initial_reply.message_id)
    if processed_successfully_count == 0 and len(valid_urls) > 0:
         await update.message.reply_text("Finished processing. No ebooks were successfully created or sent.")
    elif processed_successfully_count > 0:
         await update.message.reply_text(f"Finished processing {processed_successfully_count}/{len(valid_urls)} URL(s).")


    # --- Post-Processing (Milestones, Donations) ---
    if processed_successfully_count > 0:
        context.user_data.setdefault('ebook_count', 0)
        context.user_data['ebook_count'] += processed_successfully_count
        user_total_ebook_count = context.user_data['ebook_count']
        logger.info(f"User {user.id} total ebook count: {user_total_ebook_count}")

        # Check Milestones
        for milestone_val in sorted(MILESTONES):
            if user_total_ebook_count >= milestone_val and (user_total_ebook_count - processed_successfully_count) < milestone_val:
                msg = MILESTONE_MESSAGES.get(milestone_val)
                if msg:
                    await update.message.reply_text(msg)
                    logger.info(f"User {user.id} hit milestone {milestone_val}")
                    break # Only one milestone message per batch

        # Random Donation Message
        next_donation_trigger = context.user_data.get('next_donation_message_count')
        if next_donation_trigger is None:
            next_donation_trigger = user_total_ebook_count + random.randint(8, 15)
            context.user_data['next_donation_message_count'] = next_donation_trigger
            logger.debug(f"User {user.id} initial donation trigger: {next_donation_trigger}")

        if user_total_ebook_count >= next_donation_trigger:
            # if user.id != YOUR_OWN_TELEGRAM_ID: # Optional: don't show to yourself
            logger.info(f"Showing donation message to user {user.id} at count {user_total_ebook_count}")
            await update.message.reply_markdown_v2(
                r"_Love turning articles into Kindle ebooks?_`\n`"
                r"_Help keep this bot running and improving\! ☕_`\n`"
                r"_Send a tip \(ETH/Base/Polygon/ARB\):_ `0xaeAe4F2b0a2958e2dCC58b7F5494984B8e375369` `\n`"
                r"_Your support is greatly appreciated\!_ 🙏✨"
            ) # Note: MarkdownV2 requires escaping special characters
            context.user_data['next_donation_message_count'] = user_total_ebook_count + random.randint(8, 15)
            logger.debug(f"User {user.id} next donation trigger: {context.user_data['next_donation_message_count']}")


async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    logger.error(f"Update {update} caused error {context.error}", exc_info=context.error)
    if isinstance(update, Update) and update.effective_message:
         try:
              await update.effective_message.reply_text("Sorry, an unexpected error occurred. The developers have been notified.")
         except Exception as e:
              logger.error(f"Failed to send error message to user: {e}")

async def _post_init(application: Application) -> None:
    commands = [
        BotCommand("start", "Start the bot and see setup info"),
        BotCommand("email", "Set your Kindle email (2-step prompt)"),
        BotCommand("myemail", "Show your current Kindle email"),
        BotCommand("help", "Show help and usage info"),
    ]
    try:
        await application.bot.set_my_commands(commands)
        logger.info("Bot commands registered.")
    except Exception as e:
        logger.warning(f"Failed to register bot commands: {e}")

# --- Main Application Setup ---
def main() -> None:
    logger.info("Starting bot...")

    # --- Persistence ---
    persistence = PicklePersistence(filepath=PERSISTENCE_FILE)

    # --- Application Builder ---
    application = (
        Application.builder()
        .token(BOT_TOKEN)
        .persistence(persistence)
        .connect_timeout(30) # Optional: configure timeouts
        .read_timeout(30)
        .post_init(_post_init)
        .build()
    )

    # --- Start User Count Thread ---
    # ... (this part remains the same)
    user_count_thread = threading.Thread(target=manage_user_count_thread_target, daemon=True)
    user_count_thread.start()
    logger.info("User count management thread (if used by other parts) started.")


    # --- Register Handlers ---
    
    # Create the ConversationHandler for the /email command
    email_conv_handler = ConversationHandler(
        entry_points=[CommandHandler("email", set_email_command)],
        states={
            GETTING_EMAIL: [MessageHandler(filters.TEXT & ~filters.COMMAND, received_email)],
        },
        fallbacks=[
            CallbackQueryHandler(cancel_email_entry, pattern='^cancel_email_input$')
        ],
        persistent=True,
        name="email_conversation"
    )

    application.add_handler(email_conv_handler)
    application.add_handler(CommandHandler("start", start_command))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler("myemail", show_email_command))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    application.add_error_handler(error_handler)

    logger.info("Bot application built. Starting polling...")
    try:
        application.run_polling(allowed_updates=Update.ALL_TYPES)
    finally:
        logger.info("Bot polling stopped. Cleaning up user count thread...")
        user_count_queue.put(None) # Signal thread to stop
        user_count_thread.join(timeout=5) # Wait for thread
        logger.info("User count thread stopped.")


if __name__ == '__main__':
    main()
