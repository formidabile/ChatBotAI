import re
import nltk
import random
import json
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from telegram.ext import Updater, MessageHandler, Filters


def search_intent(text):
  for intent in INTENTS:
      examples = INTENTS[intent]["examples"] # пример фразы, которую может написать пользователь
      responses = INTENTS[intent]["responses"] # пример ответа от бота
      for example in examples:
        if text_match(text, example): # если текст распознается
          return random.choice(responses) # возвращаем случайный ответ

def text_match(user_text, example):
  user_text = re.sub(r'[^\w\s]', "", user_text) # убираем лишние символы во входящем тексте
  dst = nltk.edit_distance(user_text, example) / len(user_text) # расстояние по Левенштейну, нужно для определения числа ошибок
  return dst < 0.4 or example.lower() in user_text.lower().strip() # если ошибок менее 40% и пользовательский текст распознается, возвращаем истину

def vectorizing(text):
  vec_text = vectorizer.transform([text]) # векторизируем входящий текст
  intent = rfc.predict(vec_text)[0] # один текст = один результат
  responses = INTENTS[intent]["responses"] # выдаем ответ случайным образом
  return random.choice(responses)

def reply(text):
  answer = search_intent(text) # ищем ответ "напрямую"
  if not answer:
    answer = vectorizing(text) # если нет ответа, выдаем его с помощью средств машинного обучения
  return answer

# функция вызывается на каждое входящее сообщение боту
def telegram_bot(update, ctx):
    user_text = update.message.text # текст от пользователя
    bot_answer = reply(user_text) # ответ бота
    update.message.reply_text(bot_answer) # отправка ответа пользователю

X = []
y = []
vectorizer = CountVectorizer()
rfc = RandomForestClassifier()
with open("big_bot_config.json", "r") as file:
    DATA = json.load(file)

INTENTS = DATA["intents"]

def machine_learning():
    for name,intent in INTENTS.items():
        for phrase in intent["examples"]:
            X.append(phrase)
            y.append(name)

        for phrase in intent["responses"]:
            X.append(phrase)
            y.append(name)


    vectorizer.fit(X)
    vectorX = vectorizer.transform(X)

    rfc.fit(vectorX, y)

def main():
    machine_learning()

    updater = Updater('5518981012:AAERLQNrGKeJmDEoRIsIi5ozTkXJHfzxNgY')

    handler = MessageHandler(Filters.text, telegram_bot)
    updater.dispatcher.add_handler(handler)

    updater.start_polling()
    updater.idle()


if __name__ == '__main__':
    main()