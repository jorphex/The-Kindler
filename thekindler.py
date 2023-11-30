import os
import re
from email.mime.application import MIMEApplication
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
import smtplib
from urllib.parse import urlparse
import requests
from PIL import Image, ImageDraw, ImageFont
from ebooklib import epub
import zipfile
from newspaper import Article
from telegram import Update, InputFile
from telegram.ext import Updater, MessageHandler, CallbackContext, Filters, CommandHandler
import uuid
from textwrap import wrap
import sys
from queue import Queue
import threading
import random

sys.stdout = sys.stderr
BOT_TOKEN = 'BOT_TOKEN'
EMAIL_HOST = 'EMAIL_HOST'
EMAIL_PORT = 'EMAIL_PORT'
SENDER_EMAIL = 'SENDER_EMAIL'
SENDER_PASSWORD = 'SENDER_PASSWORD'
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
user_count_queue = Queue()

def manage_user_count():
    while True:
        task = user_count_queue.get()
        if task is None:  # Signal to stop the thread
            break
        task()  # Execute the task
        user_count_queue.task_done()

def get_next_user_number():
    number = [None]  # A mutable object to store the result

    def task():
        try:
            with open('user_count.txt', 'r+') as file:
                number[0] = int(file.read().strip()) + 1
                file.seek(0)
                file.write(str(number[0]))
        except FileNotFoundError:
            with open('user_count.txt', 'w') as file:
                file.write('1')
            number[0] = 1

    user_count_queue.put(task)
    user_count_queue.join()  # Wait until the task is completed

    return number[0]

def number_to_ordinal(n):
    if 10 <= n % 100 <= 20:
        suffix = 'th'
    else:
        suffix = {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')
    return str(n) + suffix

def handle_email_command(update: Update, context: CallbackContext) -> None:
    update.message.reply_text('Please send your new Kindle email address.')
    context.user_data['awaiting_email'] = True  # Set the flag that we're waiting for the email.

def handle_email(update: Update, context: CallbackContext) -> None:
    email = update.message.text
    context.user_data['kindle_email'] = email
    update.message.reply_text("Kindle email updated! You may now send me URLs of articles or blog posts, and I'll send it to your Kindle.")
    context.user_data['awaiting_email'] = False  # Reset the flag after receiving the email.

def handle_message(update: Update, context: CallbackContext) -> None:
    if 'first_time' not in context.user_data:
        user_number = get_next_user_number()
        update.message.reply_text(f"You're the {number_to_ordinal(user_number)} user!")
        context.user_data['first_time'] = False

    if context.user_data.get('awaiting_email', False):
        handle_email(update, context)
        return

    if 'kindle_email' not in context.user_data:
        welcome_message = ("Hello! I am The Kindler!\nHere, you can send URLs of articles 📰 or blog posts 📑 and I'll convert them into ebooks 📚 and then send them to your Kindle and also this chat for backup. If the post has images, the top image will be on the cover and on the first page before the title. No other images are included.\n\nFirst, I need to know your Kindle email address.\nPlease send it to me. 🥰")  # Existing welcome message
        update.message.reply_text(welcome_message)
        context.user_data['awaiting_email'] = True
        return

    KINDLE_EMAIL = context.user_data['kindle_email']


    # Process single or multiple URLs
    raw_urls = update.message.text.strip()
    url_list = re.split(',|\n| ', raw_urls) if not update.message.forward_date else [raw_urls]
    url_list = [url.strip() for url in url_list if url.strip() and is_url(url)]

    if not url_list:
        update.message.reply_text('❕ Please send valid URLs.')
        return

    for url in url_list:
        update.message.reply_text('👨‍💻 Processing... ' + url, disable_web_page_preview=True)

        article_content = get_article_content(url)
        if not article_content or any(elem is None for elem in article_content[:4]):
            update.message.reply_text('✖️ Failed to extract content from the webpage.')
            continue

        title, content, _, domain, _, top_image = article_content
        cover_path, top_image_path = create_cover(title, domain, top_image)
        book = epub.EpubBook()

        epub_path, top_image_path = create_epub(title, content, cover_path, domain, book, top_image) or (None, None)
        if not epub_path or not os.path.exists(epub_path):
            update.message.reply_text('😰 Sorry, there was an error creating the ebook.')
            continue

        sent_epub_path = send_to_kindle(epub_path, KINDLE_EMAIL)

        if sent_epub_path and os.path.exists(sent_epub_path):
            update.message.reply_text('✅ Ebook has been sent to your Kindle!')
            with open(sent_epub_path, 'rb') as f:
                telegram_file = InputFile(f)
                update.message.reply_document(telegram_file)

    # Update ebook count and check for milestones
    context.user_data.setdefault('ebook_count', 0)
    context.user_data['ebook_count'] += len(url_list)
    user_count = context.user_data['ebook_count']

    for milestone in MILESTONES:
        if user_count >= milestone and user_count - len(url_list) < milestone:
            milestone_message = MILESTONE_MESSAGES.get(milestone, "")
            update.message.reply_text(milestone_message)
            break  # Only show the first relevant milestone message

    # Random donation message logic
    if 'next_donation_message_count' not in context.user_data:
        # Set a random number for the next donation message if it's not already set
        context.user_data['next_donation_message_count'] = user_count + random.randint(8, 15)

    # Check if the ebook count has reached the random threshold
    if user_count >= context.user_data['next_donation_message_count']:
        # and update.message.chat_id != 1596136034:
        update.message.reply_text('_Love the convenience of turning articles into Kindle ebooks?_\n_Help keep it going to enhance your reading experience! 📚_\n_Send a tip to_ ```0xaeAe4F2b0a2958e2dCC58b7F5494984B8e375369```\n_Your support is greatly appreciated!_ ☺️✨', parse_mode='Markdown')
        # Reset for the next donation message
        context.user_data['next_donation_message_count'] = user_count + random.randint(8, 15)

    # Cleanup
    clean_up(cover_path, epub_path, top_image_path)

def is_url(url):
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except ValueError:
        return False

def get_article_content(url):
    try:
        # print(f"Processing URL: {url}")  # Debug line
        article = Article(url)
        article.download()
        # print(f"Downloaded HTML: {article.html[:200]}...")  # Debug line
        # print(f"Download status: {article.download_state}")  # Debug line
        article.parse()
        # print(f"Parse status: {article.is_parsed}")  # Debug line

        title = article.title
        content = article.text
        content = '</p><p>'.join(content.split('\n'))
        content = '<p>' + content + '</p>'
        images = article.images - {article.top_image} if article.top_image else article.images  # Exclude top image from images
        top_image = article.top_image
        domain = urlparse(url).netloc

        # print(f"Parsed title: {title}")  # Debug line
        # print(f"Parsed content: {content[:200]}...")  # Debug line
        # print(f"Number of images: {len(images)}")  # Debug line

        return title, content, images, domain, None, top_image

    except Exception as e:
        print(f"An error occurred: {e}")
        return None

def sanitize_title(title: str) -> str:
    title = re.sub(r'[<>/:*?"\\]', '', title)
    return title

def create_cover(title, domain, top_image_url=None):
    cover_width, cover_height = 1072, 1488  # Cover dimensions
    max_image_width = 900  # Maximum image width, allowing some padding on the sides

    img = Image.new('RGB', (cover_width, cover_height), color=(255, 255, 255))
    d = ImageDraw.Draw(img)

    # The Kindler bar
    d.rectangle([(0, 0), (1072, 200)], fill=(64, 64, 64))

    logo_path = "logo.jpg"  # Set your logo file path here
    # Load and draw the logo image if it exists
    if os.path.isfile(logo_path):
        logo_image = Image.open(logo_path)

        # Compute dimensions to resize while keeping the aspect ratio
        max_size = 125  # maximum size of logo (adjust as needed)
        w, h = logo_image.size
        aspect_ratio = w / h
        new_w = new_h = max_size
        if w > h:
            new_h = int(new_w / aspect_ratio)
        else:
            new_w = int(new_h * aspect_ratio)
            
        logo_image = logo_image.resize((new_w, new_h), Image.LANCZOS)
        
        # Crop the logo into a circle
        bigsize = (logo_image.size[0] * 3, logo_image.size[1] * 3)
        mask = Image.new('L', bigsize, 0)
        draw = ImageDraw.Draw(mask)
        draw.ellipse((0, 0) + bigsize, fill=255)
        mask = mask.resize(logo_image.size, Image.LANCZOS)
        logo_image.putalpha(mask)
        
        # Center the logo vertically in the grey bar
        logo_position = (50, (200 - new_h) // 2)
        img.paste(logo_image, logo_position, logo_image)

    # Set the application name and its position
    app_name_font = ImageFont.truetype('Bookerly.ttf', 80)  # Reduced font size
    app_name = "The Kindler"
    bbox = app_name_font.getbbox(app_name)
    app_name_w, app_name_h = bbox[2], abs(bbox[3] - bbox[1])
    app_name_x_position = new_w + 80  # Add the width of the logo and some padding

    # Compute the vertical position of the text to center it in the bar, considering the offset
    app_name_y_position = (200 - app_name_h) // 2

    # Adjust the vertical position to account for the specific offset of the Bookerly font at size 80
    offset = int(0.35 * app_name_h)  # You can adjust this value based on your observations
    app_name_y_position -= offset

    d.text((app_name_x_position, app_name_y_position), app_name, fill=(255, 255, 255), font=app_name_font)

    # Set the title
    title_font = ImageFont.truetype('Aileron-Bold.otf', 100)
    title_lines = wrap(title, width=20)
    title_height = title_font.getbbox('E')[3]

    current_h, pad = 225, 10  # Adjust vertical padding
    for line in title_lines:
        w, h = d.textbbox((0, current_h), line, font=title_font)[2:]
        d.text(((1072 - w) / 2, current_h), line, fill=(0, 0, 0), font=title_font)
        current_h += title_height

    # Set the domain
    domain_font = ImageFont.truetype('Bookerly.ttf', 50)
    domain_w, domain_h = domain_font.getbbox(domain)[2:]
    domain_h_position = 1488 - domain_h - 100  # Position of domain text

    # Add space or gap between title and top image
    current_h += title_height - 20

    # Load and draw the top image if it exists
    top_image = None
    top_image_path = None
    if top_image_url:
        top_image_path = get_image(top_image_url, 'top_image.jpg')
        if top_image_path:
            top_image = Image.open(top_image_path)

            # Compute new dimensions, respecting both height and width constraints
            w, h = top_image.size
            aspect_ratio = w / h

            if w > h:  # if the image is wider (landscape)
                new_w = min(max_image_width, w)
                new_h = int(new_w / aspect_ratio)
            else:  # if the image is taller (portrait)
                new_h = min(domain_h_position - current_h - pad, h)
                new_w = int(new_h * aspect_ratio)

            # Ensure new dimensions are within the constraints
            new_w = min(max_image_width, new_w)
            new_h = min(domain_h_position - current_h - pad, new_h)

            # Resize the image, maintaining the aspect ratio
            top_image = top_image.resize((int(new_w), int(new_h)), Image.LANCZOS)

            # Center the image
            image_left_position = (cover_width - int(new_w)) // 2
            img.paste(top_image, (image_left_position, current_h))

            current_h += new_h + pad

    d.text(((1072 - domain_w) / 2, domain_h_position), domain, fill=(0, 0, 0), font=domain_font)

    cover_path = 'cover.jpeg'
    img.save(cover_path, 'JPEG')
    if top_image_path:
        os.remove(top_image_path)  # Remove the temporary top image file
    return cover_path, top_image_path

def get_image(url, filename):
    if url.startswith('data:image'):
        # Handle or skip data URIs. For example, you can skip downloading.
        return None

    try:
        response = requests.get(url, timeout=5)  # Set a timeout for the request
        response.raise_for_status()  # Raise an exception if the response contains an HTTP error status code
    except requests.exceptions.RequestException as e:
        print(f"An error occurred while downloading the image: {e}")
        return None

    # If no exception was raised, the image was downloaded successfully
    try:
        with open(filename, 'wb') as f:
            f.write(response.content)
        return filename
    except Exception as e:
        print(f"An error occurred while saving the image to a file: {e}")
        return None

def create_epub(title, content, cover_path, domain, book, top_image):
    top_image_path = None
    try:
        # Metadata
        book.set_identifier(str(uuid.uuid4()))
        book.set_title(title)
        book.add_author(domain)
        book.set_language('en')
        # Set cover image
        with open(cover_path, 'rb') as cover_file:
            cover_content = cover_file.read()
        book.set_cover("cover.jpeg", cover_content)

        # Add CSS
        style = 'img { max-width: 100%; }'
        css = epub.EpubItem(uid="style_nav", file_name="style/nav.css", media_type="text/css", content=style)
        book.add_item(css)

        # Create a chapter
        chap1 = epub.EpubHtml(title='Chapter 1', file_name='chap_01.xhtml', lang='en')

        # Add title at the top of the content with a horizontal line separator
        # Include the top image if it's a valid URL and not a data URI
        if top_image and not top_image.startswith('data:image'):
            image_filename = "topimage.jpg"
            top_image_path = get_image(top_image, image_filename)  # Store the downloaded image path
            if top_image_path:
                with open(top_image_path, 'rb') as image_file:
                    image_content = image_file.read()
                image_item = epub.EpubImage()
                image_item.file_name = image_filename
                image_item.content = image_content
                book.add_item(image_item)
                chap1.content = f'<img src="{image_filename}" alt="top_image"/><hr/><h1>{title}</h1>{content}'
            else:
                chap1.content = f'<hr style="border-width:2px;" /><h1>{title}</h1>{content}'
        else:
            chap1.content = f'<hr style="border-width:2px;" /><h1>{title}</h1>{content}'

        # Add chapter, default NCX, and Nav file
        book.add_item(chap1)
        book.add_item(epub.EpubNcx())
        book.add_item(epub.EpubNav())

        # book.spine = ['nav', chap1]
        book.toc = [epub.Link('chap_01.xhtml', 'Chapter 1', 'chap1')]

        # Create EPUB file
        epub_path = f'{sanitize_title(title)}.epub'
        epub.write_epub(epub_path, book, {})

        # Debugging: Print list of files in the EPUB
        # with zipfile.ZipFile(epub_path, 'r') as zip_ref:
            # print("List of files in the EPUB:")
            # for filename in zip_ref.namelist():
                # print(filename)

        return epub_path, top_image_path
    except Exception as e:
        print(f"An error occurred while creating the EPUB: {e}")
        return None

def clean_up(cover_path, epub_path, top_image_path=None):
    if cover_path and os.path.exists(cover_path):
        os.remove(cover_path)
    if epub_path and os.path.exists(epub_path):
        os.remove(epub_path)
    if top_image_path and os.path.exists(top_image_path):
        os.remove(top_image_path)

def send_to_kindle(file_path, kindle_email):
    try:
        msg = MIMEMultipart()
        msg['From'] = SENDER_EMAIL
        msg['To'] = kindle_email
        msg['Subject'] = ""

        with open(file_path, 'rb') as f:
            mime_app = MIMEApplication(f.read(), "epub+zip")
            mime_app.add_header('Content-Disposition', 'attachment', filename=os.path.basename(file_path))
            mime_app.add_header('Content-ID', '<0>')
            msg.attach(mime_app)

        server = smtplib.SMTP(EMAIL_HOST, EMAIL_PORT)
        server.starttls()
        server.login(SENDER_EMAIL, SENDER_PASSWORD)
        server.send_message(msg)
        server.quit()
        print(f"Email sent to {kindle_email} with EPUB: {file_path}")  # Debug line
        return file_path  # return the epub path

    except smtplib.SMTPException as e:
        print(f'Failed to send email: {e}')
        raise

def main() -> None:
    print("Bot started.")
    user_count_thread = threading.Thread(target=manage_user_count, daemon=True)
    user_count_thread.start()
    updater = Updater(token=BOT_TOKEN)
    dispatcher = updater.dispatcher
    dispatcher.add_handler(MessageHandler(Filters.text & ~Filters.command, handle_message))
    dispatcher.add_handler(CommandHandler('email', handle_email_command))
    updater.start_polling()
    updater.idle()
    user_count_queue.put(None)
    user_count_thread.join()

if __name__ == '__main__':
    main()
