"""
Standard ChatGPT
"""
import logging
logging.basicConfig(filename="chatbot.log", format="%(asctime)s - %(name)s - %(levelname)s - %(funcName)s - %(message)s")
def logger(is_timed):
    def decorator(func):
        from functools import wraps
        import time
        wraps(func)
        def wrapper(*args, **kwargs):
            log = logging.getLogger(func.__name__)
            log.info(f'Entering {func.__name__} with args {args} and kwargs {kwargs}')
            start = time.time()
            out = func(*args, **kwargs)
            end = time.time()
            if is_timed:
                log.info(f"Exiting {func.__name__} with return value {out}. Took {end - start} seconds.")
            else:
                log.info(f"Exiting {func.__name__} with return value {out}")
            # Return the return value
            return out
        return wrapper
    return decorator
log = logging.getLogger(__name__)
import json
import uuid
from os import environ
from os import getenv
from os.path import exists
import sys
from colorama import Fore, Back, Style, init

import requests
from OpenAIAuth import Authenticator, Error as AuthError

BASE_URL = environ.get("CHATGPT_BASE_URL") or "https://chatgpt.duti.tech/"


class Error(Exception):
    """Base class for exceptions in this module."""

    source: str
    message: str
    code: int


class Chatbot:
    """
    Chatbot class for ChatGPT
    """

    @logger(is_timed=True)
    def __init__(
        self,
        config,
        conversation_id=None,
        parent_id=None,
    ) -> None:
        self.config = config
        self.session = requests.Session()
        if "proxy" in config:
            if isinstance(config["proxy"], str) is False:
                raise Exception("Proxy must be a string!")
            proxies = {
                "http": config["proxy"],
                "https": config["proxy"],
            }
            self.session.proxies.update(proxies)
        self.conversation_id = conversation_id
        self.parent_id = parent_id
        self.conversation_mapping = {}
        self.conversation_id_prev_queue = []
        self.parent_id_prev_queue = []
        self.encoding = "utf-8"
        if "email" in config and "password" in config:
            pass
        elif "access_token" in config:
            self.__refresh_headers(config["access_token"])
        elif "session_token" in config:
            pass
        else:
            raise Exception("No login details provided!")
        if "access_token" not in config:
            try:
                self.__login()
            except AuthError as error:
                raise error

    @logger(is_timed=False)
    def __refresh_headers(self, access_token):
        self.session.headers.clear()
        self.session.headers.update(
            {
                "Accept": "text/event-stream",
                "Authorization": f"Bearer {access_token}",
                "Content-Type": "application/json",
                "X-Openai-Assistant-App-Id": "",
                "Connection": "close",
                "Accept-Language": "en-US,en;q=0.9",
                "Referer": "https://chat.openai.com/chat",
            },
        )

    @logger(is_timed=True)
    def __login(self):
        if (
            "email" not in self.config or "password" not in self.config
        ) and "session_token" not in self.config:
            log.error("No login details provided!")
            raise Exception("No login details provided!")
        auth = Authenticator(
            email_address=self.config.get("email"),
            password=self.config.get("password"),
            proxy=self.config.get("proxy"),
        )
        if self.config.get("session_token"):
            log.debug("Using session token")
            auth.session_token = self.config["session_token"]
            auth.get_access_token()
            if auth.access_token is None:
                del self.config["session_token"]
                self.__login()
                return
        else:
            log.debug("Using authentiator to get access token")
            auth.begin()
            self.config["session_token"] = auth.session_token
            auth.get_access_token()

        self.__refresh_headers(auth.access_token)

    @logger(is_timed=True)
    def ask(
        self,
        prompt,
        conversation_id=None,
        parent_id=None,
        timeout=360,
        gen_title=False,
    ):
        """
        Ask a question to the chatbot
        :param prompt: String
        :param conversation_id: UUID
        :param parent_id: UUID
        :param gen_title: Boolean
        """
        if parent_id is not None and conversation_id is None:
            log.error("conversation_id must be set once parent_id is set")
            error = Error()
            error.source = "User"
            error.message = "conversation_id must be set once parent_id is set"
            error.code = -1
            raise error
            # user-specified covid and parid, check skipped to avoid rate limit

        if (
            conversation_id is not None and conversation_id != self.conversation_id
        ):  # Update to new conversations
            log.debug("Updating to new conversation by setting parent_id to None")
            self.parent_id = None  # Resetting parent_id

        conversation_id = conversation_id or self.conversation_id
        parent_id = parent_id or self.parent_id
        if conversation_id is None and parent_id is None:  # new conversation
            parent_id = str(uuid.uuid4())
            gen_title = True
            log.debug(f"New conversation, setting parent_id to new UUID4: {parent_id}")

        if conversation_id is not None and parent_id is None:
            if conversation_id not in self.conversation_mapping:
                log.debug(
                    f"Conversation ID {conversation_id} not found in conversation mapping, mapping conversations",
                )
                self.__map_conversations()
            log.debug(
                f"Conversation ID {conversation_id} found in conversation mapping, setting parent_id to {self.conversation_mapping[conversation_id]}",
            )
            parent_id = self.conversation_mapping[conversation_id]
        data = {
            "action": "next",
            "messages": [
                {
                    "id": str(uuid.uuid4()),
                    "role": "user",
                    "content": {"content_type": "text", "parts": [prompt]},
                },
            ],
            "conversation_id": conversation_id,
            "parent_message_id": parent_id,
            "model": "text-davinci-002-render-sha"
            if not self.config.get("paid")
            else "text-davinci-002-render-paid",
        }
        log.debug(f"Sending the payload:")
        log.debug(json.dumps(data, indent=2))
        # new_conv = data["conversation_id"] is None
        self.conversation_id_prev_queue.append(
            data["conversation_id"],
        )  # for rollback
        self.parent_id_prev_queue.append(data["parent_message_id"])
        response = self.session.post(
            url=BASE_URL + "api/conversation",
            data=json.dumps(data),
            timeout=timeout,
            stream=True,
        )
        self.__check_response(response)
        if self.encoding != None:
          response.encoding = self.encoding
        else:
          response.encoding = response.apparent_encoding
        for line in response.iter_lines():
            line = str(line)[2:-1]
            if line == "Internal Server Error":
                log.error(f"Internal Server Error: {line}")
                raise Exception("Error: " + str(line))
            if line == "" or line is None:
                continue
            if "data: " in line:
                line = line[6:]
            if line == "[DONE]":
                break

            # Replace accidentally escaped double quotes
            line = line.replace('\\"', '"')
            line = line.replace("\\'", "'")
            line = line.replace("\\\\", "\\")
            # Try parse JSON
            try:
                line = json.loads(line)
            except json.decoder.JSONDecodeError:
                log.exception("Error parsing JSON", stack_info=True)
                continue
            if not self.__check_fields(line):
                log.error("Field missing", exc_info=True)
                raise Exception("Field missing. Details: " + str(line))

            message = line["message"]["content"]["parts"][0]
            conversation_id = line["conversation_id"]
            parent_id = line["message"]["id"]
            log.debug(f"Received message: {message}")
            log.debug(f"Received conversation_id: {conversation_id}")
            log.debug(f"Received parent_id: {parent_id}")
            yield {
                "message": message,
                "conversation_id": conversation_id,
                "parent_id": parent_id,
            }
        self.conversation_mapping[conversation_id] = parent_id
        if parent_id is not None:
            self.parent_id = parent_id
        if conversation_id is not None:
            self.conversation_id = conversation_id
        if gen_title:
            self.gen_title(conversation_id, parent_id)

    @logger(is_timed=False)
    def __check_fields(self, data: dict) -> bool:
        try:
            data["message"]["content"]
        except TypeError:
            return False
        except KeyError:
            return False
        return True

    @logger(is_timed=False)
    def __check_response(self, response):
        response.encoding = response.apparent_encoding
        if response.status_code != 200:
            print(response.text)
            error = Error()
            error.source = "OpenAI"
            error.code = response.status_code
            error.message = response.text
            raise error

    @logger(is_timed=True)
    def get_conversations(self, offset=0, limit=20):
        """
        Get conversations
        :param offset: Integer
        :param limit: Integer
        """
        url = BASE_URL + f"api/conversations?offset={offset}&limit={limit}"
        response = self.session.get(url)
        if self.encoding != None:
          response.encoding = self.encoding
        else:
          response.encoding = response.apparent_encoding
        self.__check_response(response)
        data = json.loads(response.text)
        return data["items"]

    @logger(is_timed=True)
    def get_msg_history(self, convo_id):
        """
        Get message history
        :param id: UUID of conversation
        """
        url = BASE_URL + f"api/conversation/{convo_id}"
        response = self.session.get(url)
        if self.encoding != None:
          response.encoding = self.encoding
        else:
          response.encoding = response.apparent_encoding
        self.__check_response(response)
        data = json.loads(response.text)
        return data

    @logger(is_timed=True)
    def gen_title(self, convo_id, message_id):
        """
        Generate title for conversation
        """
        url = BASE_URL + f"api/conversation/gen_title/{convo_id}"
        response = self.session.post(
            url,
            data=json.dumps(
                {"message_id": message_id, "model": "text-davinci-002-render"},
            ),
        )
        self.__check_response(response)

    @logger(is_timed=True)
    def change_title(self, convo_id, title):
        """
        Change title of conversation
        :param id: UUID of conversation
        :param title: String
        """
        url = BASE_URL + f"api/conversation/{convo_id}"
        response = self.session.patch(url, data=json.dumps({"title": title}))
        self.__check_response(response)

    @logger(is_timed=True)
    def delete_conversation(self, convo_id):
        """
        Delete conversation
        :param id: UUID of conversation
        """
        url = BASE_URL + f"api/conversation/{convo_id}"
        response = self.session.patch(url, data='{"is_visible": false}')
        self.__check_response(response)
        self.conversation_id = None
        self.parent_id = None

    @logger(is_timed=True)
    def clear_conversations(self):
        """
        Delete all conversations
        """
        url = BASE_URL + "api/conversations"
        response = self.session.patch(url, data='{"is_visible": false}')
        self.__check_response(response)

    @logger(is_timed=False)
    def __map_conversations(self):
        conversations = self.get_conversations()
        histories = [self.get_msg_history(x["id"]) for x in conversations]
        for x, y in zip(conversations, histories):
            self.conversation_mapping[x["id"]] = y["current_node"]

    @logger(is_timed=False)
    def reset_chat(self) -> None:
        """
        Reset the conversation ID and parent ID.

        :return: None
        """
        self.conversation_id = None
        self.parent_id = str(uuid.uuid4())

    @logger
    def rollback_conversation(self, num=1) -> None:
        """
        Rollback the conversation.
        :param num: The number of messages to rollback
        :return: None
        """
        for _ in range(num):
            self.conversation_id = self.conversation_id_prev_queue.pop()
            self.parent_id = self.parent_id_prev_queue.pop()


@logger(is_timed=False)
def get_input(prompt):
    """
    Multiline input function.
    """
    # Display the prompt
    print(prompt, end="")

    # Initialize an empty list to store the input lines
    lines = []

    # Read lines of input until the user enters an empty line
    while True:
        try:
            line = input()
            if line == "":
                break
            lines.append(line)
        except KeyboardInterrupt:
            sys.exit(0)

    # Join the lines, separated by newlines, and store the result
    user_input = "\n".join(lines)

    # Return the input
    return user_input


@logger(is_timed=False)
def configure():
    """
    Looks for a config file in the following locations:
    """
    config_files = ["config.json"]
    xdg_config_home = getenv("XDG_CONFIG_HOME")
    if xdg_config_home:
        config_files.append(f"{xdg_config_home}/revChatGPT/config.json")
    user_home = getenv("HOME")
    if user_home:
        config_files.append(f"{user_home}/.config/revChatGPT/config.json")

    config_file = next((f for f in config_files if exists(f)), None)
    if config_file:
        with open(config_file, encoding="utf-8") as f:
            config = json.load(f)
    else:
        print(f"{Fore.RED}No config file found.{Style.RESET_ALL}")
        raise Exception("No config file found.")
    return config

@logger(is_timed=False)
def main(config: dict):
    """
    Main function for the ChatGPT program.
    """
    print(f"{Fore.GREEN}Logging in...{Style.RESET_ALL}")
    chatbot = Chatbot(
        config,
        conversation_id=config.get("conversation_id"),
        parent_id=config.get("parent_id"),
    )

    def handle_commands(command: str) -> bool:
        if command == "!help" or command == "help":
            print(
f"""
{Fore.LIGHTBLACK_EX}
Commands:
    !help - Show this message
    !reset - Forget the current conversation
    !rollback x - Rollback the conversation (x being the number of messages to rollback)
    !quit - Exit this program
    !set - Changes the conversation
    !list - List last 20 conversations
    !rm - Remove conversation
    !title - Change current conversation title
    !history - Show the chat history of conversation
{Style.RESET_ALL}
{Fore.RED}Press enter twice to submit your question.>>>>{Style.RESET_ALL}
"""
            )
        elif command == "!reset" or command == "reset":
            chatbot.reset_chat()
            print(f"{Fore.GREEN}Chat session successfully reset.{Style.RESET_ALL}")
        elif command == "!config" or command == "config":
            print(json.dumps(chatbot.config, indent=4))
        elif command == "!list" or command == "list":
            try:
                for data in chatbot.get_conversations():
                    print(f"[{Fore.BLUE}{data['id']}{Style.RESET_ALL}] {Fore.MAGENTA}{data['title']}{Style.RESET_ALL}")
            except requests.exceptions.ProxyError:
                log.exception("Can not access ChatGPT api.", stack_info=True)
                print(f"{Fore.RED}Can not access ChatGPT api.{Style.RESET_ALL}")
        elif command.startswith("!rollback ") or command.startswith("rollback "):
            # Default to 1 rollback if no number is specified
            try:
                rollback = int(command.split(" ")[1])
            except IndexError:
                log.exception("No number specified, rolling back 1 message", stack_info=True)
                rollback = 1
            chatbot.rollback_conversation(rollback)
            print(f"{Fore.GREEN}Rolled back {rollback} messages.{Style.RESET_ALL}")
        elif command.startswith("!set ") or command.startswith("set "):
            try:
                print(f"{Fore.GREEN}Conversation has been changed, now showing message history:{Style.RESET_ALL}")
                history = chatbot.get_msg_history(command.split(" ")[1])
                chatbot.conversation_id = chatbot.config[
                    "conversation_id"
                ] = command.split(" ")[1]
                chatbot.parent_id = chatbot.config[
                    "parent_id"
                ] = history["current_node"]
                mapping = history["mapping"]
                for id in mapping:
                    message = mapping[id]["message"]
                    if mapping[id]["message"] is not None:
                        if message["author"]["role"] == "user":
                            print(f"{Back.YELLOW}You:{Style.RESET_ALL}{Fore.YELLOW}")
                        else:
                            print(f"{Back.CYAN}Chatbot:{Style.RESET_ALL}{Fore.CYAN}")
                        for content in message["content"]["parts"]:
                            print(content)
                        print(Style.RESET_ALL)
            except IndexError:
                log.exception("Please include conversation UUID in command", stack_info=True)
                print(f"{Fore.RED}Please include conversation UUID in command{Style.RESET_ALL}")
            except requests.exceptions.ProxyError:
                log.exception("Can not access ChatGPT api.", stack_info=True)
                print(f"{Fore.RED}Can not access ChatGPT api.{Style.RESET_ALL}")
        elif command.startswith("!rm ") or command.startswith("rm "):
            try:
                chatbot.delete_conversation(command.split(" ")[1])
                print("Conversation has been deleted")
            except IndexError:
                print(f"{Fore.RED}Please include conversation UUID in command{Style.RESET_ALL}")
            except requests.exceptions.ProxyError:
                log.exception("Can not access ChatGPT api.", stack_info=True)
                print(f"{Fore.RED}Can not access ChatGPT api.{Style.RESET_ALL}")
        elif command.startswith("!title") or command.startswith("title "):
            try:
                chatbot.change_title(chatbot.conversation_id, command.split(" ")[1])
                print(f"{Fore.GREEN}Conversation title has been changed{Style.RESET_ALL}")
            except IndexError:
                print(f"{Fore.RED}Please include conversation title in command{Style.RESET_ALL}")
            except requests.exceptions.ProxyError:
                log.exception("Can not access ChatGPT api.", stack_info=True)
                print(f"{Fore.RED}Can not access ChatGPT api.{Style.RESET_ALL}")
        elif command.startswith("!history ") or command.startswith("history "):
            try:
                history = chatbot.get_msg_history(command.split(" ")[1])
                mapping = history["mapping"]
                for id in mapping:
                    message = mapping[id]["message"]
                    if mapping[id]["message"] is not None:
                        if message["author"]["role"] == "user":
                            print(f"{Back.YELLOW}You:{Style.RESET_ALL}{Fore.YELLOW}")
                        else:
                            print(f"{Back.CYAN}Chatbot:{Style.RESET_ALL}{Fore.CYAN}")
                        for content in message["content"]["parts"]:
                            print(content)
                        print(Style.RESET_ALL)
            except IndexError:
                print(f"{Fore.RED}Please include conversation UUID in command{Style.RESET_ALL}")
            except requests.exceptions.ProxyError:
                log.exception("Can not access ChatGPT api.", stack_info=True)
                print(f"{Fore.RED}Can not access ChatGPT api.{Style.RESET_ALL}")
        elif command == "!quit" or command == "quit":
            sys.exit(0)
        else:
            return False
        return True

    while True:
        prompt = get_input(f"\n{Back.YELLOW}You:{Style.RESET_ALL}{Fore.YELLOW}\n")
        if handle_commands(prompt):
            continue
        if prompt == "":
            print(f"{Fore.RED}Can not send blank, please type your questions.{Style.RESET_ALL}")
            continue
        print(f"{Back.CYAN}Chatbot:{Style.RESET_ALL}{Fore.CYAN}")
        try:
            prev_text = ""
            for data in chatbot.ask(
                prompt,
            ):
                message = data["message"][len(prev_text) :]
                print(message, end="", flush=True)
                prev_text = data["message"]
            print(f"\n{Style.RESET_ALL}")
        except requests.exceptions.ProxyError:
            log.exception("Can not access ChatGPT api.", stack_info=True)
            print(f"{Fore.RED}Can not access ChatGPT api.{Style.RESET_ALL}")
        # print(message["message"])


if __name__ == "__main__":
    init()
    print(
f"""
{Fore.CYAN}ChatGPT - A command-line interface to OpenAI's ChatGPT (https://chat.openai.com/chat)
Repo: github.com/acheong08/ChatGPT{Style.RESET_ALL}
{Fore.LIGHTBLACK_EX}
Commands:
    !help - Show this message
    !reset - Forget the current conversation
    !rollback x - Rollback the conversation (x being the number of messages to rollback)
    !quit - Exit this program
    !set - Changes the conversation
    !list - List last 20 conversations
    !rm - Remove conversation
    !title - Change current conversation title
    !history - Show the chat history of conversation
{Style.RESET_ALL}
{Fore.RED}Press enter twice to submit your question.>>>>{Style.RESET_ALL}
"""
    )
    main(configure())
