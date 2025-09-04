import os
import logging
import requests
import pandas as pd
import time
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Updater, CommandHandler, CallbackQueryHandler, CallbackContext
from apscheduler.schedulers.background import BackgroundScheduler
from flask import Flask
from threading import Thread

# ===== Логи =====
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# ===== Токен =====
TOKEN = os.environ.get("TELEGRAM_TOKEN")

# ===== Монеты и таймфреймы =====
AVAILABLE_COINS = ["BTC", "ETH", "SOL", "WLFI", "JELLYJELLY"]
AVAILABLE_TF = ["15m", "1h"]

# ===== Настройки пользователей =====
user_settings = {}  # {user_id: {"auto_signals": True, "coins": ["BTC","ETH"], "timeframes":["15m","1h"]}}

# ===== Flask для keep-alive =====
app = Flask('')

@app.route('/')
def home():
    return "Bot is running!"

def run_flask():
    app.run(host='0.0.0.0', port=3000)

def keep_alive():
    t = Thread(target=run_flask)
    t.start()

# ===== Команды =====
def start(update: Update, context: CallbackContext):
    try:
        user_id = update.effective_user.id
        if user_id not in user_settings:
            user_settings[user_id] = {"auto_signals": True, "coins": ["BTC","ETH"], "timeframes":["15m","1h"]}
        update.message.reply_text(
            "Привет! 👋\nЯ Crypto Signal Bot.\nИспользуй /settings для настройки сигналов."
        )
    except Exception as e:
        logger.error(f"Ошибка в /start: {e}")

def settings(update: Update, context: CallbackContext):
    try:
        user_id = update.effective_user.id
        keyboard = [
            [InlineKeyboardButton("Вкл Авто сигналы ✅", callback_data='auto_on')],
            [InlineKeyboardButton("Выкл Авто сигналы ❌", callback_data='auto_off')]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        update.message.reply_text("Настройки сигналов:", reply_markup=reply_markup)
    except Exception as e:
        logger.error(f"Ошибка в /settings: {e}")

def button_handler(update: Update, context: CallbackContext):
    try:
        query = update.callback_query
        user_id = query.from_user.id
        query.answer()
        if query.data == "auto_on":
            user_settings[user_id]["auto_signals"] = True
            query.edit_message_text("Авто сигналы включены ✅")
        elif query.data == "auto_off":
            user_settings[user_id]["auto_signals"] = False
            query.edit_message_text("Авто сигналы отключены ❌")
    except Exception as e:
        logger.error(f"Ошибка в кнопках: {e}")

# ===== Получение цены =====
def get_price(coin="BTC"):
    try:
        url = f"https://api.binance.com/api/v3/ticker/price?symbol={coin}USDT"
        resp = requests.get(url, timeout=5)
        time.sleep(0.3)
        return float(resp.json()["price"])
    except Exception as e:
        logger.error(f"Ошибка получения цены {coin}: {e}")
        return 0

# ===== RSI =====
def calculate_rsi(prices, period=14):
    try:
        delta = pd.Series(prices).diff()
        gain = delta.clip(lower=0)
        loss = -1*delta.clip(upper=0)
        avg_gain = gain.rolling(period).mean().iloc[-1]
        avg_loss = loss.rolling(period).mean().iloc[-1]
        if avg_loss == 0:
            return 100
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))
    except Exception as e:
        logger.error(f"Ошибка RSI: {e}")
        return 50

# ===== Отправка сигналов =====
def send_signal(user_id, context: CallbackContext):
    try:
        if user_id not in user_settings:
            return
        coins = user_settings[user_id]["coins"]
        for coin in coins:
            price = get_price(coin)
            # Пример фиктивного RSI для теста
            rsi_15m = 22
            rsi_1h = 78
            msg = f"💹 {coin} (OKX)\nЦена: {price:.2f}$\nRSI 15m: {rsi_15m}\nRSI 1h: {rsi_1h}\n"
            if rsi_15m <= 25:
                msg += "Сигнал: ЛОНГ 📈"
            elif rsi_1h >= 75:
                msg += "Сигнал: ШОРТ 📉"
            else:
                msg += "Сигнал: нет"
            context.bot.send_message(chat_id=user_id, text=msg)
    except Exception as e:
        logger.error(f"Ошибка отправки сигнала: {e}")

# ===== Планировщик =====
def start_scheduler(updater: Updater):
    scheduler = BackgroundScheduler()
    for user_id in user_settings.keys():
        scheduler.add_job(send_signal, 'interval', minutes=2, args=[user_id, updater.bot])
    scheduler.start()

# ===== Основная функция =====
def main():
    keep_alive()  # Запуск Flask для keep-alive

    updater = Updater(TOKEN, use_context=True)
    dp = updater.dispatcher

    dp.add_handler(CommandHandler("start", start))
    dp.add_handler(CommandHandler("settings", settings))
    dp.add_handler(CallbackQueryHandler(button_handler))

    updater.start_polling(timeout=10, clean=True)
    start_scheduler(updater)
    updater.idle()

if __name__ == "__main__":
    main()
