import logging
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes
import pandas as pd
from datetime import datetime, timedelta
from main.news_analysis import analyze_news_sentiment
import os

CSV_PATH = "news_storage.csv"


logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

# 🎨 Форматирование новостей
def format_news(df):
    messages = []
    for _, row in df.iterrows():
        pub = row["published"]
        sentiment = row["sentiment_prediction"]

        # Выбор эмодзи по тональности
        if sentiment == "позитивная":
            emoji = "🟢"
        elif sentiment == "негативная":
            emoji = "🔴"
        elif sentiment == "нейтральная":
            emoji = "⚪"
        else:
            emoji = "❓"

        # Форматируем дату
        date_str = pub.strftime('%Y-%m-%d %H:%M') if pd.notnull(pub) else 'дата неизвестна'

        # Собираем сообщение
        message = (
            f"*{row['title']}*\n\n"
            f"{emoji} {sentiment.upper()}\n\n\n"
            f"🏷️ Источник: {row['source']}\n"
            f"📅 {date_str}\n"
            f"🔗 [ссылка на новость]({row['link']})"
        )
        messages.append(message)
    return messages


def sentiment_summary(df):
    counts = df["sentiment_prediction"].value_counts()
    summary = "\n".join(f"{sent}: {counts.get(sent, 0)}" for sent in ["позитивная", "нейтральная", "негативная"])
    return f"📊 Новостной фон:\n{summary}"




# 📌 Команды
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Привет! Я бот для анализа новостей по тональности.\n"
                                    "Напиши /analyze чтобы начать или /help для списка команд.")

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "📌 Доступные команды:\n"
        "/analyze — показать 10 свежих новостей с тональностью\n"
        "/today — новости и фон за сегодня\n"
        "/week — новости и фон за последние 7 дней\n"
        "/last — последние 5 новостей\n"
        "/help — справка по командам"
    )

async def analyze(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Анализирую новости, подожди секундочку...")
    try:
        df = analyze_news_sentiment()
        messages = format_news(df.head(10))
        for msg in messages:
            await update.message.reply_text(msg, parse_mode='Markdown')
    except Exception as e:
        await update.message.reply_text(f"Произошла ошибка: {e}")


async def today(update: Update, context: ContextTypes.DEFAULT_TYPE):
    df = analyze_news_sentiment()
    today_df = df[df["published"].dt.date == datetime.now().date()]
    await update.message.reply_text(sentiment_summary(today_df))
    for msg in format_news(today_df.head(15)):
        await update.message.reply_text(msg)


async def week(update: Update, context: ContextTypes.DEFAULT_TYPE):
    df = analyze_news_sentiment()
    week_ago = datetime.now() - timedelta(days=7)
    week_df = df[df["published"] >= week_ago]
    await update.message.reply_text(sentiment_summary(week_df))
    for msg in format_news(week_df.head(20)):
        await update.message.reply_text(msg)


async def last(update: Update, context: ContextTypes.DEFAULT_TYPE):
    df = analyze_news_sentiment()
    for msg in format_news(df.head(5)):
        await update.message.reply_text(msg)


# 🚀 Запуск
def run_bot():
    application = ApplicationBuilder().token("8013556995:AAHNXL5_8nxG_548hEhiByITWAkQgR5J8fY").build()

    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler("analyze", analyze))
    application.add_handler(CommandHandler("today", today))
    application.add_handler(CommandHandler("week", week))
    application.add_handler(CommandHandler("last", last))

    print("Бот запущен")
    application.run_polling()
