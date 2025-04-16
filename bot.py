import asyncio
import torch
import json
import random
import numpy as np

from aiogram import Bot, Dispatcher, F
from aiogram.enums import ParseMode
from aiogram.client.default import DefaultBotProperties
from aiogram.types import Message
from aiogram.filters import CommandStart

from model import NeuralNet
from text_utils import bag_of_words, tokenize

# Токен
API_TOKEN = 'токен вашего бота'

# Создание бота
bot = Bot(token=API_TOKEN, default=DefaultBotProperties(parse_mode=ParseMode.HTML))
dp = Dispatcher()

# Настройка устройства
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Загрузка данных интентов
with open('intents.json', 'r', encoding='utf-8') as json_data:
    intents = json.load(json_data)

# Загрузка модели
FILE = "data.pth"
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data["all_words"]
tags = data["tags"]
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

# Функция ответа
def get_response(sentence):
    sentence = tokenize(sentence)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    output = model(X)
    _, predicted = torch.max(output, dim=1)
    tag = tags[predicted.item()]
    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]

    if prob.item() > 0.8:
        for intent in intents["intents"]:
            if tag == intent["tag"]:
                return random.choice(intent["responses"])
    return "Я тебя не понял. Можем поговорить о чем-нибудь другом?"

# Обработка команды /start
@dp.message(CommandStart())
async def cmd_start(message: Message):
    await message.answer("Привет, я чат-бот афоризмов. Напиши мне тему, а я скину крылатое выражение")

# Обработка обычных сообщений
@dp.message(F.text)
async def handle_message(message: Message):
    response = get_response(message.text)
    await message.answer(response)

# Запуск бота
async def main():
    await dp.start_polling(bot)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Бот остановлен.")