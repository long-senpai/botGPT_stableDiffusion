# from revChatGPT.Official import AsyncChatbot
import json
# from chatgpt_wrapper import ChatGPT

import httpx
START_CONTEXT = str(open("starting-context.txt").read())
context_helios = str(open("teiki/helios.raw").read())
context_imperatorLang = str(open("teiki/ImperatorLang.raw").read())

def _send_to_openai(endpoint_url: str,):
    async def send_to_openai(api_key: str, timeout: float, repeat: int, payload: dict) -> httpx.Response:
        """
        Send a request to openai.
        :param api_key: your api key
        :param timeout: timeout in seconds
        :param payload: the request body, as detailed here: https://beta.openai.com/docs/api-reference
        """
        async with httpx.AsyncClient() as client:
            i = 0
            while (repeat>0):
                res = await client.post(
                    url=endpoint_url,
                    json=payload,
                    headers={"content_type": "application/json", "Authorization": f"Bearer {api_key}"},
                    timeout=timeout,
                )
                # print(res.json())
                if "error" not in res.json() or repeat<=0:
                    break
                print("Retrying ... ")
                repeat-=1
            return res

    return send_to_openai

complete = _send_to_openai("https://api.openai.com/v1/completions")

def get_config() -> dict:
    import os
    # get config.json path
    config_dir = os.path.abspath(__file__ + "/../../")
    config_name = 'config.json'
    config_path = os.path.join(config_dir, config_name)

    with open(config_path, 'r') as f:
        config = json.load(f)

    return config

def get_context_text(folderName="teiki") -> list:
    import os
    # get config.json path
    _dir = os.path.abspath(__file__ + "/../../")
    config_dir = os.path.join(context_dir, folderName)
    list_context = glob.glob(config_dir + "/*.raw")
    # default_context = 
    config_name = 'config.json'
    config_path = os.path.join(config_dir, config_name)

    with open(config_path, 'r') as f:
        config = json.load(f)

    return config

config = get_config()


async def handle_response(message) -> str:

    # specific usecase
    if message.lower() == "mrow" or message.lower() == "mrow!":
        return "Squawk!"
    if message.lower() == "mrows" or message.lower() == "mrows!":
        return "Squawk!"

    # extract keyword for the question
    

    # If you not sure, answer this:\"I don't know, please ask <@1009279870582390860> for more infos\". 
    # ookami 1044154324219072573
    context_ = START_CONTEXT
    if "helios" in message.lower():
        context_+=context_helios

    if "imperatorlang" in message.lower():
        context_+=context_imperatorLang

    response = await complete(
        config["openAI_key"],
        timeout=None,
        repeat=3,
        payload={
            "model": "text-davinci-003",
            "prompt": context_ + "\n" + "Pretend you are Niko, the reply to this message: \"" + message + "\" is ",
            "temperature": 0.2,
            "max_tokens": 256,
            "logprobs": 100,
            "top_p": 1,
            "frequency_penalty": 0.1,
            "presence_penalty": 0.
        },
    )
    print("")
    # print(response.json())

    # Serializing json
    json_object = json.dumps(response.json(), indent=4)
    
    # Writing to sample.json
    with open("result.json", "w") as outfile:
        outfile.write(json_object)

    responseMessage = response.json()["choices"][0]["text"].strip()
    # responseMessage = "I'm sorry, but I am an AI language model developed by OpenAI and I am not capable of feeling emotions or having self-awareness. I am still in development and may not have the latest information or the ability to provide accurate answers to all questions. Additionally, I am a machine learning model, so there may be errors or limitations in my responses. I apologize for any inconvenience this may cause. Thank you for understanding."
    # response = await chatbot.ask(message)
    # responseMessage = response["choices"][0]["text"]
    print(responseMessage)

    return responseMessage