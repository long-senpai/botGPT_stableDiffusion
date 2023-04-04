import discord
from discord import app_commands
from discord.ext import commands, tasks
from src import responses
from src import log
import os, sys
import contextvars
import functools
import asyncio
import pathlib
import hashlib
import random
from src.scripts.txt2img_bot import SDBot
from src import responses
from src import log
from queue import Queue
import threading

sd_bot = SDBot()

logger = log.setup_logger(__name__)

MAX_QUEUE = 10
queue = Queue(MAX_QUEUE)

# async def stable_diffusion_queue():
#     while True:
#         print("looping")
#         if not queue.empty():
#             q_object = queue.get()
#             await sd_gen(q_object[0], q_object[1])
#         else:
#             await asyncio.sleep(5)

config = responses.get_config()

isPrivate = False
isReplyAll = False

class MyView(discord.ui.View):
    def __init__(self, user_message, author):
        super().__init__()
        self.user_message = user_message
        self.author = author
    @discord.ui.button(label="Re-Generate!", style=discord.ButtonStyle.primary, emoji="😎")
    async def button_callback(self, interaction, button):
        await interaction.response.defer()
        if "--Bronze" in self.user_message or "--Silver" in self.user_message or "--Gold" in self.user_message:
            await sd_gen(interaction, self.user_message, self.author)
        else:
            await sd_gen(interaction, self.user_message, self.author)

class aclient(discord.Client):
    def __init__(self) -> None:
        intents = discord.Intents.default()
        intents.message_content = True
        super().__init__(intents=intents)
        self.tree = app_commands.CommandTree(self)
        self.activity = discord.Activity(type=discord.ActivityType.watching, name="/chat | /imagine | /help")

async def sd_init_gen(message, user_message,author_ = None):
    if author_ != None:
        author = author_
    else:
        author = message.user.id
        await message.response.defer(ephemeral=isPrivate)
    try:
        response = '> **' + user_message + '** - <@' + \
            str(author) + '> \n\n'
        
        source_message = user_message
        user_message_ = user_message.split("--")
        user_message = user_message_[0]
        tier = user_message_[1].strip()

        if author_ != None:
            message_sent = await message.channel.send(response)
        else:
            message_sent = await message.followup.send(response)

        filename = hashlib.sha256(user_message.encode('utf-8')).hexdigest()[:20]
        if 'seed' in user_message.lower():
            try:
                seed = int(user_message.split('seed')[1].split('=')[1].strip())
            except:
                seed = random.randint(0,4294967295)
            user_message = user_message.split('seed')[0]
        else:
            seed = random.randint(0,4294967295)
        sd_bot.maketierimg(user_message, filename, tier, seed)
        save_path = './outputs'
        img_path = os.path.join(save_path, tier + '_' + filename + '.png')
        if author_ != None:
            await message_sent.edit(content=response, attachments=[discord.File(img_path)], view=MyView(source_message, author))
        else:
            await message_sent.edit(content=response, attachments=[discord.File(img_path)], view=MyView(source_message, author))
    except Exception as e:
        await message_sent.edit(content="> **I am unfortunately under maintenance, please try again later!**")
        logger.exception(f"Error while sending message: {e}")


async def sd_gen(message, user_message, author_ = None):
    tier = ""
    if "--" in user_message:
        tier = user_message.split("--")[1].strip()

    if author_ != None:
        author = author_
    else:
        author = message.user.id
        await message.response.defer(ephemeral=isPrivate)
    try:
        response = '> **' + user_message + '** - <@' + \
            str(author) + '> \n\n'

        if author_ != None:
            message_sent = await message.channel.send(response)
        else:
            message_sent = await message.followup.send(response)

        filename = hashlib.sha256(user_message.encode('utf-8')).hexdigest()[:20]
        if 'seed' in user_message.lower():
            try:
                seed = int(user_message.split('seed')[1].split('=')[1].strip())
            except:
                seed = random.randint(0,4294967295)
        else:
            seed = random.randint(0,4294967295)
        if tier != "":
            sd_bot.img2img(user_message, filename, seed)
        else:
            sd_bot.makeimg(user_message, filename, seed)
        save_path = './outputs'
        if tier == "":
            img_path = os.path.join(save_path, filename + '.png')
        else:
            img_path = os.path.join(save_path, tier + '_' + filename + '.png')

        if author_ != None:
            await message_sent.edit(content=response, attachments=[discord.File(img_path)], view=MyView(user_message, author))
        else:
            await message_sent.edit(content=response, attachments=[discord.File(img_path)], view=MyView(user_message, author))
    except Exception as e:
        await message_sent.edit(content="> **I am unfortunately under maintenance, please try again later!**")
        logger.exception(f"Error while sending message: {e}")


async def send_message(message, user_message):
    global isReplyAll
    if not isReplyAll:
        author = message.user.id
        await message.response.defer(ephemeral=isPrivate)
    else:
        author = message.author.id
    try:
        response = '> **' + user_message + '** - <@' + \
            str(author) + '> \n\n'
        response = f"{response}{await responses.handle_response(user_message)}"
        if len(response) > 1900:
            # Split the response into smaller chunks of no more than 1900 characters each(Discord limit is 2000 per chunk)
            if "```" in response:
                # Split the response if the code block exists
                parts = response.split("```")
                # Send the first message
                if isReplyAll:
                    await message.channel.send(parts[0])
                else:
                    await message.followup.send(parts[0])
                # Send the code block in a seperate message
                code_block = parts[1].split("\n")
                formatted_code_block = ""
                for line in code_block:
                    while len(line) > 1900:
                        # Split the line at the 50th character
                        formatted_code_block += line[:1900] + "\n"
                        line = line[1900:]
                    formatted_code_block += line + "\n"  # Add the line and seperate with new line

                # Send the code block in a separate message
                if (len(formatted_code_block) > 2000):
                    code_block_chunks = [formatted_code_block[i:i+1900]
                                         for i in range(0, len(formatted_code_block), 1900)]
                    for chunk in code_block_chunks:
                        if isReplyAll:
                            await message.channel.send("```" + chunk + "```")
                        else:
                            await message.followup.send("```" + chunk + "```")
                else:
                    if isReplyAll:
                        await message.channel.send("```" + formatted_code_block + "```")
                    else:
                        await message.followup.send("```" + formatted_code_block + "```")
                # Send the remaining of the response in another message

                if len(parts) >= 3:
                    if isReplyAll:
                        await message.channel.send(parts[2])
                    else:
                        await message.followup.send(parts[2])
            else:
                response_chunks = [response[i:i+1900]
                                   for i in range(0, len(response), 1900)]
                for chunk in response_chunks:
                    if isReplyAll:
                        await message.channel.send(chunk)
                    else:
                        await message.followup.send(chunk)
                        
        else:
            if isReplyAll:
                await message.channel.send(response)
            else:
                await message.followup.send(response)
    except Exception as e:
        if isReplyAll:
            await message.channel.send("> **I am unfortunately under maintenance, please try again later!**")
        else:
            await message.followup.send("> **I am unfortunately under maintenance, please try again later!**")
        logger.exception(f"Error while sending message: {e}")


async def send_start_prompt(client):
    import os
    import os.path

    config_dir = os.path.abspath(__file__ + "/../../")
    prompt_name = 'starting-prompt.txt'
    prompt_path = os.path.join(config_dir, prompt_name)
    try:
        if os.path.isfile(prompt_path) and os.path.getsize(prompt_path) > 0:
            with open(prompt_path, "r") as f:
                prompt = f.read()
                if (config['discord_channel_id']):
                    logger.info(f"Send starting prompt with size {len(prompt)}")
                    responseMessage = await responses.handle_response(prompt)
                    #channel = client.get_channel(int(config['discord_channel_id']))
                    # await channel.send(responseMessage)
                    user = await client.fetch_user(config['discord_channel_id'])
                    await user.send(responseMessage)
                    logger.info(f"Starting prompt response:{responseMessage}")
                else:
                    logger.info("No Channel selected. Skip sending starting prompt.")
        else:
            logger.info(f"No {prompt_name}. Skip sending starting prompt.")
    except Exception as e:
        logger.exception(f"Error while sending starting prompt: {e}")


def run_discord_bot():
    client = aclient()
    @client.event
    async def on_ready():
        await send_start_prompt(client)
        await client.tree.sync()
        logger.info(f'{client.user} is now running!')

    @client.tree.command(name="chat", description="Have a chat with ChatGPT")

    async def chat(interaction: discord.Interaction, *, message: str):
        global isReplyAll
        if isReplyAll:
            await interaction.response.defer(ephemeral=False)
            await interaction.followup.send("> **Warn: You already on replyAll mode. If you want to use slash command, switch to normal mode, use `/replyall` again**")
            logger.warning("\x1b[31mYou already on replyAll mode, can't use slash command!\x1b[0m")
            return
        if interaction.user == client.user:
            return
        username = str(interaction.user)
        user_message = message
        channel = str(interaction.channel)
        logger.info(
            f"\x1b[31m{username}\x1b[0m : '{user_message}' ({channel})")
        await send_message(interaction, user_message)

    @client.tree.command(name="imagine", description="Use stable diffusion to generate image")
    async def imagine(interaction: discord.Interaction, *, message: str):
        global queue
        if interaction.user == client.user:
            return
        user_message = message        
        # queue.put((interaction, user_message))
        # await interaction.response.defer()
        await sd_gen(interaction, user_message)


    @client.tree.command(name="imagine-tiers", description="Use stable diffusion to generate tiers")
    async def imagine_tiers(interaction: discord.Interaction, *, message: str):
        global queue
        if interaction.user == client.user:
            return
        user_message = message
        if "--Gold" in user_message or "--Silver" in user_message or "--Bronze" in user_message:
            # queue.put((interaction, user_message))
            await sd_gen(interaction, user_message)
        else:
            # queue.put((interaction, user_message + " --Bronze"))
            # queue.put((interaction, user_message + " --Silver"))
            # queue.put((interaction, user_message + " --Gold"))
            await sd_gen(interaction, user_message + " --Bronze")
            await sd_gen(interaction, user_message + " --Silver")
            await sd_gen(interaction, user_message + " --Gold")

    @client.tree.command(name="private", description="Toggle private access")
    async def private(interaction: discord.Interaction):
        global isPrivate
        await interaction.response.defer(ephemeral=False)
        if not isPrivate:
            isPrivate = not isPrivate
            logger.warning("\x1b[31mSwitch to private mode\x1b[0m")
            await interaction.followup.send("> **Info: Next, the response will be sent via private message. If you want to switch back to public mode, use `/public`**")
        else:
            logger.info("You already on private mode!")
            await interaction.followup.send("> **Warn: You already on private mode. If you want to switch to public mode, use `/public`**")

    @client.tree.command(name="public", description="Toggle public access")
    async def public(interaction: discord.Interaction):
        global isPrivate
        await interaction.response.defer(ephemeral=False)
        if isPrivate:
            isPrivate = not isPrivate
            await interaction.followup.send("> **Info: Next, the response will be sent to the channel directly. If you want to switch back to private mode, use `/private`**")
            logger.warning("\x1b[31mSwitch to public mode\x1b[0m")
        else:
            await interaction.followup.send("> **Warn: You already on public mode. If you want to switch to private mode, use `/private`**")
            logger.info("You already on public mode!")

    @client.tree.command(name="replyall", description="Toggle replyAll access")
    async def replyall(interaction: discord.Interaction):
        global isReplyAll
        await interaction.response.defer(ephemeral=False)
        if isReplyAll:
            await interaction.followup.send("> **Info: The bot will only response to the slash command `/chat` next. If you want to switch back to replyAll mode, use `/replyAll` again.**")
            logger.warning("\x1b[31mSwitch to normal mode\x1b[0m")
        else:
            await interaction.followup.send("> **Info: Next, the bot will response to all message in the server. If you want to switch back to normal mode, use `/replyAll` again.**")
            logger.warning("\x1b[31mSwitch to replyAll mode\x1b[0m")
        isReplyAll = not isReplyAll
            
    @client.tree.command(name="reset", description="Complete reset ChatGPT conversation history")
    async def reset(interaction: discord.Interaction):
        responses.chatbot.reset()
        await interaction.response.defer(ephemeral=False)
        await interaction.followup.send("> **Info: I have forgotten everything.**")
        logger.warning(
            "\x1b[31mChatGPT bot has been successfully reset\x1b[0m")
        await send_start_prompt(client)
        
    @client.tree.command(name="help", description="Show help for the bot")
    async def help(interaction: discord.Interaction):
        await interaction.response.defer(ephemeral=False)
        await interaction.followup.send(":star:**BASIC COMMANDS** \n\n    - `/chat [message]` Chat with ChatGPT!\n    - `/public` ChatGPT switch to public mode \n    - `/replyall`  ChatGPT switch between replyall mode and default mode\n    - `/reset` Clear ChatGPT conversation history\n\nFor complete documentation, please visit https://github.com/Zero6992/chatGPT-discord-bot")
        logger.info(
            "\x1b[31mSomeone need help!\x1b[0m")

    @client.event
    async def on_message(message):
        if isReplyAll:
            if message.author == client.user:
                return
            username = str(message.author)
            user_message = str(message.content)
            channel = str(message.channel)
            logger.info(f"\x1b[31m{username}\x1b[0m : '{user_message}' ({channel})")
            await send_message(message, user_message)
    
    # task_loop = stable_diffusion_queue()
    # asyncio.create_task(task_loop)
    TOKEN = config['discord_bot_token']
    client.run(TOKEN)
