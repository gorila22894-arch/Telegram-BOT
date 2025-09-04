# -*- coding: utf-8 -*-
"""
Crypto signal Telegram bot — all features in one file.

Dependencies:
  pip install python-telegram-bot==13.15 requests

Run:
  export TELEGRAM_TOKEN="123:ABC..."   (или вставь токен прямо в TELEGRAM_TOKEN)
  python crypto_bot_all_in_one.py
"""

import os
import json
import time
import math
import logging
import threading
import requests
from datetime import datetime, timedelta
from collections import defaultdict

from telegram import (
    ReplyKeyboardMarkup, KeyboardButton, InlineKeyboardButton, InlineKeyboardMarkup,
    ParseMode, Update
)
from telegram.ext import (
    Updater, CommandHandler, MessageHandler, Filters, CallbackContext,
    CallbackQueryHandler
)

# -----------------------
# CONFIG
# -----------------------
TELEGRAM_TOKEN = os.getenv("8434216245:AAENA3me9jgWtMD8LWXAwennSmPZg16e7T0", "").strip() or "8434216245:AAENA3me9jgWtMD8LWXAwennSmPZg16e7T0"
OKX_BASE = "https://www.okx.com"
DEFAULT_COINS = ["BTC", "ETH", "SOL", "WLFI", "JELLYJELLY"]
ALL_TFS = {"15m": "15m", "1h": "1H"}
CANDLE_LIMIT = 300
SCAN_INTERVAL_SEC = 60  # periodic scan for autosignals
SETTINGS_FILE = "settings.json"
STATE_FILE = "state.json"
LOG_FILE = "bot.log"

# Cooldowns per timeframe (seconds)
COOLDOWN_BY_TF = {
    "15m": 20 * 60,  # 20 minutes
    "1h": 2 * 60 * 60  # 2 hours
}
DEFAULT_MIN_VOLUME = 0.0  # USDT

# Trailing stop parameters (simulation)
TRAILING_ACTIVE = True
TRAILING_MOVE_PCT = 0.5  # if price moves this % in favor, move SL (example)
TRAILING_STEP_PCT = 0.5  # step size to move SL each time

# Emoji / UI
EMO_UP = "🔼"
EMO_DOWN = "🔻"
EMO_PAUSE = "⏸"
EMO_OK = "✅"
EMO_OFF = "🚫"
EMO_PRICE = "💵"
EMO_VOL = "📊"
EMO_EX = "🏦"
EMO_CFG = "⚙️"
EMO_TABLE = "📋"
EXCHANGE_NAME = "OKX"

# Buttons / labels
BTN_SIGNAL = "📊 Сигнал"
BTN_AUTO = "🤖 Авто-сигналы"
BTN_PRICES = "💰 Текущие цены"
BTN_SETTINGS = f"{EMO_CFG} Настройки"
BTN_SET_AUTO = "Вкл/Выкл авто"
BTN_SET_TF = "Выбрать TF"
BTN_SET_COINS = "Выбрать монеты"
BTN_VIEW_SETTINGS = "Показать мои настройки"
BTN_SET_MIN_VOL = "Установить min объём"
BTN_BACK = "⬅️ Назад"
BTN_DONE = "Готово ✅"
BTN_SEND_NOW = "📡 Прислать сигналы сейчас (dev)"

# -----------------------
# LOGGING
# -----------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# -----------------------
# STORAGE (in-memory + persistence)
# -----------------------
lock = threading.Lock()

# user_settings[user_id] = {
#   auto: bool, tfs: [..], coins: [..], min_volume: float
# }
user_settings = defaultdict(dict)

# last_signals key: (str(uid), symbol, tf) -> {signal: "LONG"/"SHORT", ts: unix, rsi: val, macd_sign: -1/0/1, tp_sl: (tp1,tp2,sl), trailing: {...}}
last_signals = {}

def load_json_safe(path, default):
    try:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception as e:
        logger.exception("Failed to load %s: %s", path, e)
    return default

def save_json_safe(path, data):
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.exception("Failed to save %s: %s", path, e)

def load_state():
    global user_settings, last_signals
    s = load_json_safe(SETTINGS_FILE, {})
    with lock:
        for k, v in s.items():
            try:
                user_settings[int(k)] = v
            except Exception:
                continue
    st = load_json_safe(STATE_FILE, {})
    with lock:
        last_signals = {}
        for k, v in st.items():
            # keys saved as "uid|symbol|tf"
            parts = k.split("|")
            if len(parts) >= 3:
                key = (parts[0], parts[1], parts[2])
                last_signals[key] = v
    logger.info("Loaded settings for %d users, %d last_signals", len(user_settings), len(last_signals))

def save_state():
    try:
        with lock:
            ss = {str(k): v for k, v in user_settings.items()}
            save_json_safe(SETTINGS_FILE, ss)
            st = {"|".join(k): v for k, v in last_signals.items()}
            save_json_safe(STATE_FILE, st)
    except Exception as e:
        logger.exception("Failed to save state: %s", e)

def ensure_user(uid):
    if uid not in user_settings:
        user_settings[uid] = {
            "auto": False,
            "tfs": ["15m", "1h"],
            "coins": DEFAULT_COINS.copy(),
            "min_volume": DEFAULT_MIN_VOLUME
        }

# -----------------------
# OKX and indicators
# -----------------------
def okx_symbol(sym: str) -> str:
    return f"{sym}-USDT"

def okx_get_candles(symbol: str, tf_okx: str, limit: int = 300):
    instId = okx_symbol(symbol)
    url = f"{OKX_BASE}/api/v5/market/candles"
    params = {"instId": instId, "bar": tf_okx, "limit": str(limit)}
    r = requests.get(url, params=params, timeout=12)
    if r.status_code != 200:
        raise RuntimeError(f"OKX {r.status_code}: {r.text}")
    data = r.json()
    if data.get("code") != "0":
        raise RuntimeError(f"OKX error: {data}")
    candles = list(reversed(data["data"]))
    out = []
    for c in candles:
        ts = int(c[0]); o = float(c[1]); h = float(c[2]); l = float(c[3]); close = float(c[4]); vol = float(c[5])
        out.append([ts, o, h, l, close, vol])
    return out

def ema(values, period):
    k = 2 / (period + 1.0)
    out = []
    prev = None
    for v in values:
        if prev is None:
            prev = v
        else:
            prev = v * k + prev * (1 - k)
        out.append(prev)
    return out

def rsi_calc(values, period=14):
    if len(values) < 2:
        return [None] * len(values)
    gains = []; losses = []
    for i in range(1, len(values)):
        ch = values[i] - values[i-1]
        gains.append(max(ch, 0))
        losses.append(max(-ch, 0))
    if not gains:
        return [None] * len(values)
    avg_gain = sum(gains[:period]) / period if len(gains) >= period else 0
    avg_loss = sum(losses[:period]) / period if len(gains) >= period else 0
    rsis = [None]
    for i in range(1, len(values)):
        if i <= period:
            rsis.append(None)
        else:
            avg_gain = (avg_gain * (period - 1) + gains[i-1]) / period
            avg_loss = (avg_loss * (period - 1) + losses[i-1]) / period
            if avg_loss == 0:
                r = 100.0
            else:
                rs = avg_gain / avg_loss
                r = 100 - (100 / (1 + rs))
            rsis.append(r)
    return rsis

def macd_calc(values, fast=12, slow=26, signal_period=9):
    ema_fast = ema(values, fast)
    ema_slow = ema(values, slow)
    macd_line = [f - s for f, s in zip(ema_fast, ema_slow)]
    signal_line = ema(macd_line, signal_period)
    hist = [m - s for m, s in zip(macd_line, signal_line)]
    return macd_line, signal_line, hist

def trend_by_ema(values):
    if len(values) < 200:
        return "—", None, None
    e50 = ema(values, 50)[-1]
    e200 = ema(values, 200)[-1]
    if e50 > e200:
        return f"{EMO_UP} Восходящий", e50, e200
    if e50 < e200:
        return f"{EMO_DOWN} Нисходящий", e50, e200
    return f"{EMO_PAUSE} Боковой", e50, e200

def recent_swing(highs, lows, lookback=100):
    lb = min(len(highs), lookback)
    if lb == 0:
        return None, None
    hh = max(highs[-lb:])
    ll = min(lows[-lb:])
    return hh, ll

def fibo_targets(direction, entry, high, low):
    rng = high - low
    if rng <= 0:
        if direction == "LONG":
            return entry * 1.003, entry * 1.007, entry * 0.99
        else:
            return entry * 0.997, entry * 0.993, entry * 1.01
    if direction == "LONG":
        tp1 = low + 0.382 * rng
        tp2 = low + 0.618 * rng
        sl = low * 0.995
        if entry >= tp1: tp1 = entry * 1.002
        if entry >= tp2: tp2 = entry * 1.005
    else:
        tp1 = high - 0.382 * rng
        tp2 = high - 0.618 * rng
        sl = high * 1.005
        if entry <= tp1: tp1 = entry * 0.998
        if entry <= tp2: tp2 = entry * 0.995
    return tp1, tp2, sl

def fmt_price(x):
    if x is None:
        return "—"
    if x >= 100:
        return f"{x:.2f}"
    if x >= 1:
        return f"{x:.4f}"
    return f"{x:.6f}"

# -----------------------
# SIGNAL RULES
# -----------------------
def decide_direction_by_rsi(rsi_val, tf):
    if rsi_val is None:
        return None
    if tf == "15m":
        if 20 <= rsi_val <= 25:
            return "LONG"
        if 75 <= rsi_val <= 80:
            return "SHORT"
    elif tf == "1h":
        if rsi_val < 30:
            return "LONG"
        if rsi_val > 70:
            return "SHORT"
    return None

def get_indicators_for_symbol(symbol, tf):
    tf_okx = ALL_TFS.get(tf)
    try:
        candles = okx_get_candles(symbol, tf_okx, limit=CANDLE_LIMIT)
    except Exception as e:
        logger.debug("OKX error for %s %s: %s", symbol, tf, e)
        return None
    closes = [c[4] for c in candles]
    highs = [c[2] for c in candles]
    lows = [c[3] for c in candles]
    vols = [c[5] for c in candles]
    if len(closes) < 50:
        return None
    rsi_series = rsi_calc(closes, period=14)
    rsi_last = rsi_series[-1]
    macd_line, signal_line, hist = macd_calc(closes)
    macd_hist_last = hist[-1]
    macd_state = "Бычий" if macd_line[-1] > signal_line[-1] else "Медвежий"
    trend_text, e50, e200 = trend_by_ema(closes)
    price = closes[-1]
    vol = vols[-1]
    hh, ll = recent_swing(highs, lows, lookback=100)
    tp_sl = None
    direction = decide_direction_by_rsi(rsi_last, tf)
    if direction and hh is not None and ll is not None:
        tp1, tp2, sl = fibo_targets(direction, price, hh, ll)
        tp_sl = (tp1, tp2, sl)
    return {
        "price": price,
        "rsi": rsi_last,
        "macd_hist": macd_hist_last,
        "macd_state": macd_state,
        "trend": trend_text,
        "volume": vol,
        "tp_sl": tp_sl
    }

# -----------------------
# ANTI-SPAM / COOLDOWNS / TRAILING
# -----------------------
def should_send_signal_for_user(uid, symbol, tf, direction, cooldown_map=COOLDOWN_BY_TF):
    key = (str(uid), symbol, tf)
    prev = last_signals.get(key)
    now_ts = int(time.time())
    cooldown = cooldown_map.get(tf, 30*60)
    if prev:
        if prev.get("signal") == direction:
            ts = prev.get("ts", 0)
            if now_ts - ts < cooldown:
                return False
    return True

def update_last_signal_for_user(uid, symbol, tf, direction, rsi_val, macd_sign, tp_sl=None):
    key = (str(uid), symbol, tf)
    last_signals[key] = {
        "signal": direction,
        "rsi": rsi_val,
        "macd_sign": macd_sign,
        "tp_sl": tp_sl,
        "ts": int(time.time())
    }
    # Setup trailing state
    if TRAILING_ACTIVE and tp_sl:
        entry = last_signals[key].get("entry_price", None)
        # put entry price as current price if not present
        if not entry and tp_sl:
            # we don't know entry here — it's set later in autoscan when we build message (we will set below)
            pass
    save_state()

# Trailing simulation: update SL if price moved favorably enough
def trailing_update_for_user(uid, symbol, tf, current_price):
    key = (str(uid), symbol, tf)
    state = last_signals.get(key)
    if not state:
        return None
    if "tp_sl" not in state or not state["tp_sl"]:
        return None
    tp1, tp2, sl = state["tp_sl"]
    direction = state["signal"]
    # We'll store 'current_sl' inside state to simulate stepwise moves
    current_sl = state.get("current_sl", sl)
    entry = state.get("entry_price", None)
    if not entry:
        # fallback - assume entry equals price at the time of sending
        entry = state.get("sent_price", current_price)
        state["entry_price"] = entry
    moved = False
    # percent move from entry
    if direction == "LONG":
        pct = (current_price - entry) / entry * 100
        if pct >= TRAILING_MOVE_PCT:
            # move SL up by TRAILING_STEP_PCT from previous SL, but not beyond entry
            new_sl = max(current_sl, current_sl * (1 + TRAILING_STEP_PCT/100))
            # For long, SL should increase but must remain < current_price
            if new_sl < current_price:
                state["current_sl"] = new_sl
                moved = True
    else:
        pct = (entry - current_price) / entry * 100
        if pct >= TRAILING_MOVE_PCT:
            new_sl = min(current_sl, current_sl * (1 - TRAILING_STEP_PCT/100))
            if new_sl > current_price:
                state["current_sl"] = new_sl
                moved = True
    if moved:
        last_signals[key] = state
        save_state()
        return state["current_sl"]
    return None

# -----------------------
# TELEGRAM UI & HANDLERS
# -----------------------
def build_main_kb():
    kb = [
        [KeyboardButton(BTN_SIGNAL), KeyboardButton(BTN_AUTO)],
        [KeyboardButton(BTN_PRICES), KeyboardButton(BTN_SETTINGS)]
    ]
    return ReplyKeyboardMarkup(kb, resize_keyboard=True)

def build_settings_kb():
    kb = [
        [KeyboardButton(BTN_SET_AUTO), KeyboardButton(BTN_SET_TF)],
        [KeyboardButton(BTN_SET_COINS), KeyboardButton(BTN_SET_MIN_VOL)],
        [KeyboardButton(BTN_VIEW_SETTINGS)],
        [KeyboardButton(BTN_SEND_NOW)],
        [KeyboardButton(BTN_BACK)]
    ]
    return ReplyKeyboardMarkup(kb, resize_keyboard=True)

def build_coins_keyboard(coins):
    rows = []
    row = []
    for c in coins:
        row.append(KeyboardButton(f"🔍 {c}"))
        if len(row) == 3:
            rows.append(row); row = []
    if row:
        rows.append(row)
    rows.append([KeyboardButton(BTN_BACK)])
    return ReplyKeyboardMarkup(rows, resize_keyboard=True)

def start_handler(update: Update, context: CallbackContext):
    uid = update.effective_user.id
    ensure_user(uid)
    save_state()
    update.message.reply_text("Привет 👋\nЯ трейдинг-бот. Выбирай действие:", reply_markup=build_main_kb())

def help_handler(update: Update, context: CallbackContext):
    update.message.reply_text(
        "Команды:\n"
        "/start — меню\n"
        "/send_now — принудительно отправить сигналы сейчас (только для тебя)\n"
        "/set_min_volume <число> — задать порог объёма (в USDT)\n\n"
        "Кнопки: Сигнал, Авто-сигналы, Текущие цены, Настройки",
        reply_markup=build_main_kb()
    )

def message_handler(update: Update, context: CallbackContext):
    uid = update.effective_user.id
    ensure_user(uid)
    txt = update.message.text.strip()

    if txt == BTN_SIGNAL:
        coins = user_settings[uid]["coins"]
        if not coins:
            update.message.reply_text("В настройках нет монет. Добавь через Настройки.", reply_markup=build_main_kb())
            return
        kb = build_coins_keyboard(coins)
        update.message.reply_text("Выбери монету для анализа:", reply_markup=kb)
        return

    if txt.startswith("🔍 "):
        symbol = txt.replace("🔍 ", "").strip()
        ikb = InlineKeyboardMarkup([
            [InlineKeyboardButton("15m", callback_data=f"TF|{symbol}|15m"),
             InlineKeyboardButton("1h", callback_data=f"TF|{symbol}|1h")]
        ])
        update.message.reply_text(f"Выбран {symbol}. Выбери таймфрейм:", reply_markup=ikb)
        return

    if txt == BTN_PRICES:
        send_mini_table(update, context)
        return

    if txt == BTN_AUTO:
        st = user_settings[uid]["auto"]
        text = f"{EMO_OK} Авто-сигналы включены" if st else f"{EMO_OFF} Авто-сигналы выключены"
        update.message.reply_text(text, reply_markup=build_main_kb())
        return

    if txt == BTN_SETTINGS:
        update.message.reply_text("Настройки:", reply_markup=build_settings_kb())
        return

    if txt == BTN_SET_AUTO:
        user_settings[uid]["auto"] = not user_settings[uid]["auto"]
        save_state()
        update.message.reply_text(f"Авто-сигналы {'включены ✅' if user_settings[uid]['auto'] else 'выключены 🚫'}", reply_markup=build_settings_kb())
        return

    if txt == BTN_SET_TF:
        tfs = user_settings[uid]["tfs"]
        update.message.reply_text(f"Текущие TF: {', '.join(tfs)}", reply_markup=build_settings_kb())
        return

    if txt == BTN_SET_COINS:
        update.message.reply_text(f"Твои монеты: {', '.join(user_settings[uid]['coins'])}", reply_markup=build_settings_kb())
        return

    if txt == BTN_SET_MIN_VOL:
        update.message.reply_text("Отправь число — минимальный объём (в USDT). Например: 100000", reply_markup=ReplyKeyboardMarkup([[BTN_BACK]], resize_keyboard=True))
        context.user_data["await_min_vol"] = True
        return

    if txt == BTN_VIEW_SETTINGS:
        s = user_settings[uid]
        update.message.reply_text(
            f"{EMO_CFG} Твои настройки:\n"
            f"• Авто: {'ON' if s['auto'] else 'OFF'}\n"
            f"• TF: {', '.join(s['tfs'])}\n"
            f"• Монеты: {', '.join(s['coins'])}\n"
            f"• MinVolume: {s.get('min_volume', DEFAULT_MIN_VOLUME)}",
            reply_markup=build_settings_kb()
        )
        return

    if txt == BTN_SEND_NOW:
        # trigger send for this user immediately
        context.job_queue.run_once(lambda ctx: send_signals_for_user_job(ctx, uid), when=1)
        update.message.reply_text("Запущена принудительная отправка сигналов (локально).", reply_markup=build_settings_kb())
        return

    if txt == BTN_BACK:
        update.message.reply_text("Назад.", reply_markup=build_main_kb())
        context.user_data.pop("await_min_vol", None)
        return

    if context.user_data.get("await_min_vol"):
        try:
            v = float(txt.replace(",", "."))
            user_settings[uid]["min_volume"] = float(v)
            save_state()
            update.message.reply_text(f"Порог объёма установлен: {v} (USDT)", reply_markup=build_settings_kb())
            context.user_data.pop("await_min_vol", None)
        except Exception:
            update.message.reply_text("Неверный формат. Отправь число, например: 100000", reply_markup=ReplyKeyboardMarkup([[BTN_BACK]], resize_keyboard=True))
        return

    update.message.reply_text("Не понял команду 🤔", reply_markup=build_main_kb())

def callback_query_handler(update: Update, context: CallbackContext):
    query = update.callback_query
    data = query.data
    if not data:
        query.answer()
        return
    parts = data.split("|")
    if parts[0] == "TF" and len(parts) == 3:
        symbol = parts[1]
        tf = parts[2]
        query.answer()
        send_signal_markdown_for_user(update, context, update.effective_user.id, symbol, tf, personal=True)
        return
    query.answer()

def escape_md(s: str):
    # minimal escaping for MarkdownV2
    if s is None:
        return ""
    return str(s).replace(".", "\\.").replace("-", "\\-").replace("(", "\\(").replace(")", "\\)").replace("+", "\\+").replace("#","\\#").replace("_","\\_")

def send_signal_markdown_for_user(update_or_chat, context: CallbackContext, uid, symbol, tf, personal=False):
    # send a manual formatted message (MarkdownV2)
    info = get_indicators_for_symbol(symbol, tf)
    if not info:
        context.bot.send_message(chat_id=uid, text=f"Данные по {symbol}/{tf} недоступны.", reply_markup=build_main_kb())
        return
    price = info["price"]; rsi = info["rsi"]; macd_state = info["macd_state"]; trend = info["trend"]; vol = info["volume"]
    tp_sl = info["tp_sl"]
    rsi_disp = "—" if rsi is None else f"{round(rsi,2)}"
    text = (
        f"📊 *Текущий анализ*\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"💎 Монета: *{escape_md(symbol)}/USDT*\n"
        f"🏦 Биржа: *{escape_md(EXCHANGE_NAME)}*\n"
        f"⏱ Таймфрейм: *{escape_md(tf)}*\n\n"
        f"{EMO_PRICE} *Цена:* `{fmt_price(price)}` $\n"
        f"📈 *RSI:* {rsi_disp}\n"
        f"📊 *MACD:* {escape_md(macd_state)}\n"
        f"📊 *Тренд:* {escape_md(trend)}\n"
        f"{EMO_VOL} *Объём:* `{int(vol)}`\n\n"
    )
    direction = decide_direction_by_rsi(rsi, tf)
    if direction == "LONG":
        rec = f"{EMO_UP} *Рекомендация:* LONG"
    elif direction == "SHORT":
        rec = f"{EMO_DOWN} *Рекомендация:* SHORT"
    else:
        rec = f"{EMO_PAUSE} *Рекомендация:* Нет явного сигнала"
    text += rec + "\n"
    if tp_sl:
        tp1, tp2, sl = tp_sl
        text += f"\n🎯 *Тейк-1:* `{fmt_price(tp1)}` $\n🎯 *Тейк-2:* `{fmt_price(tp2)}` $\n🛡 *Стоп-лосс:* `{fmt_price(sl)}` $\n"
    try:
        context.bot.send_message(chat_id=uid, text=text, parse_mode=ParseMode.MARKDOWN_V2, reply_markup=build_main_kb())
    except Exception:
        context.bot.send_message(chat_id=uid, text=text, reply_markup=build_main_kb())
    # update last_signals for this user so manual view won't trigger duplicate autos immediately
    macd_sign = 1 if info["macd_hist"] > 0 else (-1 if info["macd_hist"] < 0 else 0)
    key = (str(uid), symbol, tf)
    with lock:
        last_signals[key] = {
            "signal": direction,
            "rsi": rsi,
            "macd_sign": macd_sign,
            "tp_sl": tp_sl,
            "sent_price": price,
            "entry_price": price,
            "current_sl": tp_sl[2] if tp_sl else None,
            "ts": int(time.time())
        }
        save_state()

def send_mini_table(update: Update, context: CallbackContext):
    uid = update.effective_user.id
    ensure_user(uid)
    coins = user_settings[uid]["coins"]
    lines = []
    for c in coins:
        info = get_indicators_for_symbol(c, "15m")
        if not info:
            lines.append(f"{c}: недоступно")
            continue
        price = fmt_price(info["price"]); rsi = info["rsi"]; trend = info["trend"]; macd = info["macd_state"]
        rsi_txt = "—" if rsi is None else str(round(rsi,1))
        lines.append(f"*{escape_md(c)}*  `{price}` $  | RSI: {rsi_txt} | {escape_md(trend)} | {escape_md(macd)}")
    text = f"{EMO_TABLE} *Сводка (15m):*\n" + "\n".join(lines)
    try:
        context.bot.send_message(chat_id=update.effective_chat.id, text=text, parse_mode=ParseMode.MARKDOWN_V2, reply_markup=build_main_kb())
    except Exception:
        context.bot.send_message(chat_id=update.effective_chat.id, text=text, reply_markup=build_main_kb())

# -----------------------
# AUTO SCAN JOBS
# -----------------------
def send_signals_for_user_job(context: CallbackContext, uid):
    """Send signals for single user (used by forced send and by autoscan)"""
    try:
        uid = int(uid)
    except Exception:
        return
    if uid not in user_settings:
        ensure_user(uid)
    st = user_settings[uid]
    if not st.get("auto"):
        # still allow forced send even if auto is off? We'll allow via forced command only.
        pass
    coins = st.get("coins", DEFAULT_COINS)
    tfs = st.get("tfs", ["15m", "1h"])
    min_vol = float(st.get("min_volume", DEFAULT_MIN_VOLUME) or 0.0)
    bot = context.bot
    for coin in coins:
        for tf in tfs:
            info = get_indicators_for_symbol(coin, tf)
            if not info:
                continue
            vol = info["volume"]
            if min_vol and vol < min_vol:
                continue
            rsi = info["rsi"]
            direction = decide_direction_by_rsi(rsi, tf)
            if not direction:
                continue
            macd_hist = info["macd_hist"]
            macd_sign = 1 if macd_hist > 0 else (-1 if macd_hist < 0 else 0)
            # anti-spam
            if not should_send_signal_for_user(uid, coin, tf, direction):
                continue
            # require either MACD changed or direction changed to reduce spam
            macd_key = (str(uid), coin, tf, "macd")
            prev_macd = last_signals.get(macd_key)
            macd_changed = False
            if prev_macd:
                if prev_macd.get("macd_sign") != macd_sign:
                    macd_changed = True
            else:
                macd_changed = True
            key_user = (str(uid), coin, tf)
            prev = last_signals.get(key_user)
            direction_changed = (not prev) or (prev.get("signal") != direction)
            if not (macd_changed or direction_changed):
                continue
            # Build message
            price = info["price"]
            tp_sl = info["tp_sl"]
            text = (
                f"🤖 *Авто-сигнал*\n"
                f"💎 Монета: *{escape_md(coin)}/USDT*\n"
                f"🏦 Биржа: *{escape_md(EXCHANGE_NAME)}*\n"
                f"⏱ Таймфрейм: *{escape_md(tf)}*\n\n"
                f"{EMO_PRICE} Точка входа: `{fmt_price(price)}` $\n"
                f"Направление: {'🔼 LONG' if direction=='LONG' else '🔻 SHORT'}\n"
                f"📈 RSI: `{round(rsi,2) if rsi is not None else '—'}`\n"
                f"📊 MACD: {escape_md(info['macd_state'])}\n"
                f"📊 Тренд: {escape_md(info['trend'])}\n"
                f"{EMO_VOL} Объём: `{int(vol)}`\n\n"
            )
            if tp_sl:
                tp1, tp2, sl = tp_sl
                text += f"🎯 Тейк-1: `{fmt_price(tp1)}` $\n🎯 Тейк-2: `{fmt_price(tp2)}` $\n🛡 Стоп-лосс: `{fmt_price(sl)}` $\n"
                text += "\n*Примечание:* тейки/стоп рассчитаны автоматически по последним свингам."
            try:
                bot.send_message(chat_id=uid, text=text, parse_mode=ParseMode.MARKDOWN_V2)
                # update states
                with lock:
                    last_signals[key_user] = {
                        "signal": direction,
                        "rsi": rsi,
                        "macd_sign": macd_sign,
                        "tp_sl": tp_sl,
                        "sent_price": price,
                        "entry_price": price,
                        "current_sl": tp_sl[2] if tp_sl else None,
                        "ts": int(time.time())
                    }
                    last_signals[macd_key] = {"macd_sign": macd_sign, "ts": int(time.time())}
                    save_state()
            except Exception as e:
                logger.exception("Failed to send auto signal to %s: %s", uid, e)
                continue

def autoscan_job(context: CallbackContext):
    """Periodic scanner that runs for all users who have auto enabled."""
    bot = context.bot
    with lock:
        uids = list(user_settings.keys())
    for uid in uids:
        try:
            st = user_settings.get(int(uid))
        except Exception:
            continue
        if not st or not st.get("auto"):
            continue
        # run per-user job in background
        send_signals_for_user_job(context, uid)
    # trailing update: check last_signals and maybe move SL and notify user
    trailing_scan_and_notify(context)

def trailing_scan_and_notify(context: CallbackContext):
    bot = context.bot
    # For each last_signal, fetch current price and see if trailing adjusts
    keys = list(last_signals.keys())
    for key in keys:
        uid_str, symbol, tf = key
        uid = None
        try:
            uid = int(uid_str)
        except Exception:
            continue
        state = last_signals.get(key)
        if not state:
            continue
        # only process if there's a signal and tp_sl
        if not state.get("signal") or not state.get("tp_sl"):
            continue
        try:
            candles = okx_get_candles(symbol, ALL_TFS[tf], limit=1)
            price = candles[-1][4]
        except Exception:
            continue
        new_sl = trailing_update_for_user(uid, symbol, tf, price)
        if new_sl:
            # notify user about moved SL
            try:
                bot.send_message(chat_id=uid,
                                 text=f"🛡 *Trailing:* стоп-лосс по {symbol}/{tf} был подвинут на `{fmt_price(new_sl)}` $ (текущее: `{fmt_price(price)}` $).",
                                 parse_mode=ParseMode.MARKDOWN_V2)
            except Exception:
                pass

# -----------------------
# ADMIN / MANUAL commands
# -----------------------
def cmd_send_now(update: Update, context: CallbackContext):
    """User-level forced send of signals for themselves (non-admin)."""
    uid = update.effective_user.id
    ensure_user(uid)
    update.message.reply_text("Запуск принудительной отправки сигналов... (локально для тебя)")
    context.job_queue.run_once(lambda ctx: send_signals_for_user_job(ctx, uid), when=1)

def cmd_set_min_volume(update: Update, context: CallbackContext):
    uid = update.effective_user.id
    ensure_user(uid)
    args = context.args
    if not args:
        update.message.reply_text("Использование: /set_min_volume <число (USDT)>")
        return
    try:
        v = float(args[0].replace(",", "."))
        user_settings[uid]["min_volume"] = v
        save_state()
        update.message.reply_text(f"Min volume установлен: {v} USDT")
    except Exception:
        update.message.reply_text("Неверный формат числа.")

# -----------------------
# MAIN
# -----------------------
def main():
    load_state()
    updater = Updater(TELEGRAM_TOKEN, use_context=True)
    dp = updater.dispatcher

    dp.add_handler(CommandHandler("start", start_handler))
    dp.add_handler(CommandHandler("help", help_handler))
    dp.add_handler(CommandHandler("send_now", cmd_send_now))
    dp.add_handler(CommandHandler("set_min_volume", cmd_set_min_volume))
    dp.add_handler(MessageHandler(Filters.text & ~Filters.command, message_handler))
    dp.add_handler(CallbackQueryHandler(callback_query_handler))

    # Jobs
    jq = updater.job_queue
    # autoscan repeating
    jq.run_repeating(autoscan_job, interval=SCAN_INTERVAL_SEC, first=10)

    logger.info("Starting bot...")
    updater.start_polling()
    updater.idle()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Stopping... saving state")
        save_state()
        raise
