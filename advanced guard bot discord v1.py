import discord
from discord.ext import commands
from collections import defaultdict, deque
import asyncio
import datetime
import re
import emoji
import aiohttp
from dotenv import load_dotenv
from fuzzywuzzy import fuzz
import unicodedata
import logging
import hashlib
import io
from PIL import Image
import numpy as np
import pandas as pd
from textblob import TextBlob
from sklearn.preprocessing import MinMaxScaler
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import json

# --- Setup ---

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
load_dotenv()
TOKEN = os.getenv('DISCORD_TOKEN')
VIRUSTOTAL_API_KEY = os.getenv('VIRUSTOTAL_API_KEY')

# --- Constants ---

SLOW_MODE_DELAY = 5
DUPLICATE_RESET_TIME = 60
DUPLICATE_MSG_THRESHOLD = 3
CAPITALIZATION_THRESHOLD = 0.7
SPAM_TIME = 1
SPAM_THRESHOLD = 4
RAID_THRESHOLD = 10
RAID_TIME = 300
EMOJI_THRESHOLD = 5
WARNING_LIMIT = 1
MUTE_DURATION_30S = 30
MUTE_DURATION_5M = 300
IMAGE_DUPLICATE_TIME_WINDOW = 60
SUSPICIOUS_ACTIVITY_THRESHOLD = 3
MESSAGE_VOLUME_THRESHOLD = 10
JOIN_RATE_THRESHOLD = 5

# --- Initialize Bot ---

intents = discord.Intents.default()
intents.message_content = True
intents.members = True
intents.guilds = True
intents.presences = True
bot = commands.Bot(command_prefix='/', intents=intents)

# --- Tracking Dictionaries ---

user_messages = defaultdict(lambda: deque(maxlen=SPAM_THRESHOLD))
message_history = defaultdict(lambda: deque(maxlen=DUPLICATE_MSG_THRESHOLD))
user_image_hashes = defaultdict(list)
member_join_times = defaultdict(lambda: deque(maxlen=RAID_THRESHOLD))
suspicious_accounts = set()
spam_warnings = defaultdict(int)

# --- Data Storage ---

user_data_file = "user_data.json"
autoencoder_model_file = "anomaly_detection_autoencoder.h5"

# --- Anomaly Detection Model ---

autoencoder_model = None
threshold = 0.05  # Adjust this threshold as needed

# --- Helper Functions ---

def normalize_text(text):
    return re.sub(r'[^\w\s]', '', text).lower()

def is_similar(existing_message, new_message, threshold=90):
    return fuzz.ratio(normalize_text(existing_message), normalize_text(new_message)) >= threshold

async def update_status():
    statuses = ["Monitoring the server...", "Keeping the chat safe.", "Guarding against spam and raids."]
    status_index = 0
    while True:
        await bot.change_presence(activity=discord.Game(name=statuses[status_index]))
        logging.debug(f"Status updated to: {statuses[status_index]}")
        status_index = (status_index + 1) % len(statuses)
        await asyncio.sleep(60)

def contains_excessive_emojis(text, threshold=EMOJI_THRESHOLD):
    return emoji.emoji_count(text) > threshold

def count_mentions(message):
    return len(message.mentions)

def is_link(message_content):
    return re.search(r'(https?://\S+)', message_content)

async def analyze_link_safety(url):
    async with aiohttp.ClientSession() as session:
        headers = {'x-apikey': VIRUSTOTAL_API_KEY}
        async with session.post('https://www.virustotal.com/api/v3/urls', headers=headers, data={'url': url}) as response:
            if response.status == 200:
                json_response = await response.json()
                scan_id = json_response.get('data', {}).get('id')
                if scan_id:
                    async with session.get(f'https://www.virustotal.com/api/v3/analyses/{scan_id}', headers=headers) as result_response:
                        if result_response.status == 200:
                            result = await result_response.json()
                            last_analysis_stats = result.get('data', {}).get('attributes', {}).get('last_analysis_stats', {})
                            return last_analysis_stats.get('malicious', 0) == 0
            return False

async def analyze_file_safety(file_url):
    async with aiohttp.ClientSession() as session:
        headers = {'x-apikey': VIRUSTOTAL_API_KEY}
        try:
            async with session.get(file_url) as file_response:
                if file_response.status != 200:
                    return False
                file_content = await file_response.read()
                data = aiohttp.FormData()
                data.add_field('file', file_content, filename='file', content_type='application/octet-stream')
                upload_url = 'https://www.virustotal.com/api/v3/files'
                async with session.post(upload_url, headers=headers, data=data) as response:
                    if response.status == 200:
                        json_response = await response.json()
                        file_id = json_response.get('data', {}).get('id')
                        if file_id:
                            analysis_url = f'https://www.virustotal.com/api/v3/analyses/{file_id}'
                            await asyncio.sleep(30)  # Wait for analysis to complete
                            async with session.get(analysis_url, headers=headers) as result_response:
                                if result_response.status == 200:
                                    result = await result_response.json()
                                    last_analysis_stats = result.get('data', {}).get('attributes', {}).get('last_analysis_stats', {})
                                    return last_analysis_stats.get('malicious', 0) == 0
        except Exception as e:
            logging.error(f"File analysis failed: {e}")
        return False

def extract_user_features(member):
    current_time = datetime.datetime.now(datetime.timezone.utc)
    account_age_days = (current_time - member.created_at).days
    account_age_weeks = account_age_days / 7
    features = [
        account_age_weeks,
        member.guild.member_count,
        member.guild.premium_subscription_count,
        int(member.top_role.position),
        int(member.discriminator)
    ]
    return features

async def check_for_raid(guild):
    current_time = datetime.datetime.now(datetime.timezone.utc)
    join_times = member_join_times[guild.id]
    if len(join_times) >= RAID_THRESHOLD and (current_time - join_times[0]).total_seconds() <= RAID_TIME:
        logging.warning(f"Raid detected in guild: {guild.id}. Kicking all new members.")
        for member in guild.members:
            if (current_time - member.joined_at).total_seconds() <= RAID_TIME:
                suspicious_accounts.add(member.id)
                try:
                    await member.kick(reason="Raid detected")
                    logging.info(f"Kicked member {member.name} due to raid.")
                except discord.Forbidden:
                    logging.error(f"Failed to kick member {member.name}.")
                except Exception as e:
                    logging.error(f"Error kicking member {member.name}: {e}")
        WARN_MESSAGE = "Raid detected and stopped. New members were kicked to protect the server."
        await guild.text_channels[0].send(WARN_MESSAGE)
        member_join_times[guild.id].clear()

async def mute_user(member, duration):
    mute_role = discord.utils.get(member.guild.roles, name="Muted")
    if not mute_role:
        try:
            mute_role = await member.guild.create_role(name="Muted", permissions=discord.Permissions(send_messages=False, speak=False))
            logging.info(f"Created Mute role in {member.guild.name}")
        except discord.HTTPException as e:
            logging.error(f"Error creating Mute role: {e}")
        return  # Exit if Mute role creation fails
    try:
        await member.add_roles(mute_role)
        logging.info(f"Muted {member.name} for {duration} seconds.")
        await asyncio.sleep(duration)
        await member.remove_roles(mute_role)
        logging.info(f"Unmuted {member.name} after {duration} seconds.")
    except discord.Forbidden:
        logging.error(f"Failed to mute {member.name}.")
    except Exception as e:
        logging.error(f"Error muting {member.name}: {e}")

async def hash_image(image_url):
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(image_url) as response:
                image_data = await response.read()
                return hashlib.sha256(image_data).hexdigest()
    except Exception as e:
        logging.error(f"Error fetching image for hashing: {e}")
    return

def is_duplicate_image(user_id, image_hash, current_time):
    return any(hash == image_hash and (current_time - timestamp).total_seconds() <= IMAGE_DUPLICATE_TIME_WINDOW
                for hash, timestamp in user_image_hashes[user_id])

def load_user_data():
    try:
        with open(user_data_file, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return {}

def save_user_data(data):
    with open(user_data_file, 'w') as f:
        json.dump(data, f)

def load_autoencoder_model():
    try:
        return keras.models.load_model(autoencoder_model_file)
    except OSError:  # FileNotFoundError is a subclass of OSError
        return None

def save_autoencoder_model(model):
    model.save(autoencoder_model_file)

def create_autoencoder_model(input_dim):
    model = keras.Sequential(
        [
            layers.Input(shape=(input_dim,)),
            layers.Dense(128, activation="relu"),  # Increased hidden layer size
            layers.Dense(64, activation="relu"),
            layers.Dense(32, activation="relu"),
            layers.Dense(16, activation="relu"),
            layers.Dense(32, activation="relu"),
            layers.Dense(64, activation="relu"),
            layers.Dense(128, activation="relu"),
            layers.Dense(input_dim),
        ]
    )
    model.compile(loss="mse", optimizer="adam")
    return model

def prepare_data_for_anomaly_detection(user_data):
    df = pd.DataFrame.from_dict(user_data, orient='index')

    # Check for existing columns before dropping
    columns_to_drop = ['messages', 'joins', 'roles', 'interactions']
    columns_to_drop = [col for col in columns_to_drop if col in df.columns]

    # Flatten the lists within the DataFrame
    df = pd.concat([df.drop(columns_to_drop, axis=1), 
                    df['messages'].apply(pd.Series).stack().reset_index(level=1, drop=True).rename('messages'),
                    df['joins'].apply(pd.Series).stack().reset_index(level=1, drop=True).rename('joins'),
                    df['roles'].apply(pd.Series).stack().reset_index(level=1, drop=True).rename('roles'),
                    df['interactions'].apply(pd.Series).stack().reset_index(level=1, drop=True).rename('interactions')], 
                   axis=1)

    # Fill NaN values with empty strings or 0, depending on the column type
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col].fillna('', inplace=True)
        else:
            df[col].fillna(0, inplace=True)

    # Convert datetime columns to numerical timestamps
    for col in ['messages', 'joins', 'roles', 'interactions']:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col]).astype(np.int64) // 10**9  # Convert to Unix timestamp

    # Normalize the data using MinMaxScaler
    scaler = MinMaxScaler()
    df[df.columns] = scaler.fit_transform(df[df.columns])

    return df

async def analyze_user_behavior(user_id):
    global autoencoder_model, threshold
    user_data = load_user_data()
    if str(user_id) not in user_data:
        return False  # User not found in data

    # Prepare data for anomaly detection
    user_df = prepare_data_for_anomaly_detection(user_data)

    # If the model is not loaded, create a new one and train it
    if autoencoder_model is None:
        autoencoder_model = create_autoencoder_model(input_dim=user_df.shape[1])
        autoencoder_model.fit(user_df, user_df, epochs=100, batch_size=64, shuffle=True, validation_split=0.2)  # Increased epochs and batch size
        save_autoencoder_model(autoencoder_model)
    else:
        # If model exists, use it for prediction
        reconstructions = autoencoder_model.predict(user_df)
        mse = np.mean(np.power(user_df - reconstructions, 2), axis=1)
        if np.max(mse) > threshold:
            return True
    return False

async def collect_user_data(member, guild, event_type, event_data):
    user_data = load_user_data()
    user_id = str(member.id)
    user_info = user_data.get(user_id, {})

    # Ensure default values for missing keys 
    if 'messages' not in user_info:
        user_info['messages'] = []
    if 'joins' not in user_info:
        user_info['joins'] = []
    if 'roles' not in user_info:
        user_info['roles'] = []
    if 'interactions' not in user_info:
        user_info['interactions'] = []

    if event_type == 'message':
        user_info['messages'].append((guild.id, event_data, datetime.datetime.now(datetime.timezone.utc).isoformat()))
    elif event_type == 'join':
        user_info['joins'].append((guild.id, datetime.datetime.now(datetime.timezone.utc).isoformat()))  # No event data for joins
    elif event_type == 'role_update':
        user_info['roles'].append((guild.id, event_data, datetime.datetime.now(datetime.timezone.utc).isoformat()))
    elif event_type == 'interaction':
        user_info['interactions'].append((guild.id, event_data, datetime.datetime.now(datetime.timezone.utc).isoformat()))

    user_data[user_id] = user_info
    save_user_data(user_data)

def get_suspicious_actions(user_id):
    user_data = load_user_data()
    user_info = user_data.get(str(user_id), {})
    return user_info.get('suspicious_actions', 0)

# --- Event Handlers ---

@bot.event
async def on_member_join(member):
    guild_id = member.guild.id
    member_join_times[guild_id].append(datetime.datetime.now(datetime.timezone.utc))
    await check_for_raid(member.guild)
    account_age = (datetime.datetime.now(datetime.timezone.utc) - member.created_at).total_seconds()
    if account_age < 60 * 60 * 24 * 7:
        logging.warning(f"New member {member.name} has a young account (less than a week old).")
    await collect_user_data(member, member.guild, 'join', None)

class MessageHandler:
    async def propagate(self, message):
        # Your logic here (if any)
        pass

handler = MessageHandler()

@bot.event
async def on_message(message):
    # Ignore bot messages
    if message.author == bot.user:
        return

    await handler.propagate(message)

    # --- Slow Mode Check ---
    user_message_deque = user_messages[message.author.id]
    current_time = datetime.datetime.now(datetime.timezone.utc)

    # Only append if it's a new message, not an edit
    if not message.edited_at:
        user_message_deque.append((message.content, current_time))

    if len(user_message_deque) > 1:
        time_difference = (current_time - user_message_deque[0][1]).total_seconds()
        if time_difference < SLOW_MODE_DELAY:
            warning_message = await message.channel.send(
                f"{message.author.mention}, you are sending messages too quickly. Please wait a moment."
            )
            await message.delete()
            logging.warning(f"Deleted message from {message.author.name} due to slow mode.")

            try:
                await asyncio.sleep(10)
                await warning_message.delete()
            except discord.errors.HTTPException:
                logging.error(f"Error deleting warning message for slow mode: {message.author.name}")
            return

    # --- Spam Check ---
    if len(user_message_deque) > SPAM_THRESHOLD and (current_time - user_message_deque[0][1]).total_seconds() <= SPAM_TIME:
        warning_message = await message.channel.send(f"{message.author.mention}, your message was deleted due to spam.")
        await message.delete()
        logging.warning(f"Deleted spam message from {message.author.name}.")

        try:
            await asyncio.sleep(10)
            await warning_message.delete()
        except discord.errors.HTTPException:
            logging.error(f"Error deleting warning message for spam: {message.author.name}")
        return

    # --- Duplicate Message Check ---
    message_history[message.author.id].append(message.content)

    if len(message_history[message.author.id]) >= DUPLICATE_MSG_THRESHOLD:
        recent_messages = list(message_history[message.author.id])
        if all(is_similar(recent_messages[-1], msg) for msg in recent_messages[:-1]):
            warning_message = await message.channel.send(
                f"{message.author.mention}, your message was deleted because it was a duplicate."
            )
            await message.delete()
            logging.warning(f"Deleted duplicate message from {message.author.name}.")
            try:
                await asyncio.sleep(10)
                await warning_message.delete()
            except discord.errors.HTTPException:
                logging.error(f"Error deleting warning message for duplicate: {message.author.name}")
            return

    # --- Capitalization Check ---
    if len(message.content) > 0:
        capitalization_ratio = sum(char.isupper() for char in message.content) / len(message.content)
        if capitalization_ratio > CAPITALIZATION_THRESHOLD:
            warning_message = await message.channel.send(
                f"{message.author.mention}, your message was deleted due to excessive capitalization."
            )
            await message.delete()
            logging.warning(f"Deleted message with excessive capitalization from {message.author.name}.")
            try:
                await asyncio.sleep(10)
                await warning_message.delete()
            except discord.errors.HTTPException:
                logging.error(f"Error deleting warning message for capitalization: {message.author.name}")
            return

    # ---  Emoji Check ---
    if contains_excessive_emojis(message.content):
        warning_message = await message.channel.send(
            f"{message.author.mention}, your message was deleted due to excessive emojis."
        )
        await message.delete()
        logging.warning(f"Deleted message with excessive emojis from {message.author.name}.")
        try:
            await asyncio.sleep(10)
            await warning_message.delete()
        except discord.errors.HTTPException:
            logging.error(f"Error deleting warning message for emojis: {message.author.name}")
        return

    # --- Link Check ---
    if is_link(message.content):
        link = re.search(r'(https?://\S+)', message.content).group(0)
        if not await analyze_link_safety(link):
            warning_message = await message.channel.send(
                f"{message.author.mention}, your message contained an unsafe link and has been deleted."
            )
            await message.delete()
            logging.warning(f"Deleted message with unsafe link from {message.author.name}.")
            try:
                await asyncio.sleep(10)
                await warning_message.delete()
            except discord.errors.HTTPException:
                logging.error(f"Error deleting warning message for unsafe link: {message.author.name}")
        else:
            warning_message = await message.channel.send(f"{message.author.mention}, your message contained a safe link.")
            try:
                await asyncio.sleep(10)
                await warning_message.delete()
            except discord.errors.HTTPException:
                logging.error(f"Error deleting warning message for safe link: {message.author.name}")
        return

    # --- File/Image Check ---
    if message.attachments:
        for attachment in message.attachments:
            try:
                file_url = attachment.url
                if not await analyze_file_safety(file_url):
                    warning_message = await message.channel.send(
                        f"{message.author.mention}, your message contained an unsafe file and has been deleted."
                    )
                    await message.delete()
                    logging.warning(f"Deleted message with unsafe file from {message.author.name}.")
                    try:
                        await asyncio.sleep(10)
                        await warning_message.delete()
                    except discord.errors.HTTPException:
                        logging.error(f"Error deleting warning message for unsafe file: {message.author.name}")
                else:
                    warning_message = await message.channel.send(
                        f"{message.author.mention}, your message contained a safe file."
                    )
                    try:
                        await asyncio.sleep(10)
                        await warning_message.delete()
                    except discord.errors.HTTPException:
                        logging.error(f"Error deleting warning message for safe file: {message.author.name}")

                # Image Duplication Check (If applicable)
                if attachment.url.lower().endswith(('png', 'jpg', 'jpeg', 'gif')):
                    image_hash = await hash_image(attachment.url)
                    if image_hash and is_duplicate_image(message.author.id, image_hash, current_time):
                        warning_message = await message.channel.send(
                            f"{message.author.mention}, you already posted this image!"
                        )
                        await message.delete()
                        logging.warning(f"Deleted duplicate image from {message.author.name}.")
                        try:
                            await asyncio.sleep(10)
                            await warning_message.delete()
                        except discord.errors.HTTPException:
                            logging.error(
                                f"Error deleting warning message for duplicate image: {message.author.name}"
                            )
                        return
                    if image_hash:
                        user_image_hashes[message.author.id].append((image_hash, current_time))
            except Exception as e:
                logging.error(f"Error analyzing file attachment: {e}")
            return

    # --- Data Collection and Anomaly Detection ---
    await collect_user_data(message.author, message.guild, 'message', message.content)
    if await analyze_user_behavior(message.author.id):
        user_data = load_user_data()
        user_info = user_data.get(str(message.author.id), {})
        suspicious_actions = user_info.get('suspicious_actions', 0)
        suspicious_actions += 1
        user_info['suspicious_actions'] = suspicious_actions
        user_data[str(message.author.id)] = user_info
        save_user_data(user_data)

        if suspicious_actions >= SUSPICIOUS_ACTIVITY_THRESHOLD:
            try:
                await message.author.kick(reason="Suspicious activity detected")
                logging.info(f"Kicked member {message.author.name} due to suspicious activity.")
            except discord.Forbidden:
                logging.error(f"Failed to kick member {message.author.name}.")
            except Exception as e:
                logging.error(f"Error kicking member {message.author.name}: {e}")

@bot.event
async def on_member_update(before, after):
    if before.roles != after.roles:
        await collect_user_data(after, after.guild, 'role_update', after.roles)

@bot.event
async def on_raw_reaction_add(payload):
    await collect_user_data(await bot.fetch_user(payload.user_id), await bot.fetch_guild(payload.guild_id), 'interaction', payload.emoji.name)

@bot.event
async def on_raw_reaction_remove(payload):
    await collect_user_data(await bot.fetch_user(payload.user_id), await bot.fetch_guild(payload.guild_id), 'interaction', payload.emoji.name)

@bot.event
async def on_ready():
    logging.info(f'Logged in as {bot.user.name}')
    bot.loop.create_task(update_status())

    global autoencoder_model
    autoencoder_model = load_autoencoder_model()
    if autoencoder_model:
        logging.info("Loaded anomaly detection autoencoder model from file.")

# --- Run Bot ---

bot.run(TOKEN)
