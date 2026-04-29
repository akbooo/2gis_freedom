"""
Дашборд: Анализ отзывов Freedom Bank (2ГИС)
Автор: Data Analyst
Запуск: streamlit run app.py

Данный файл объединяет логику prepare_data.py и app.py:
- при первом запуске читает result.csv, обрабатывает и кэширует данные в памяти
- reviews.parquet не требуется (но используется как кэш на диске, если есть)
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from collections import Counter
import re
import ast
from pathlib import Path
import requests
import json
import streamlit as st

GROQ_API_KEY = st.secrets['API_KEY']
MODEL        = "llama-3.1-8b-instant"
# ============================================================
# Конфигурация страницы
# ============================================================
st.set_page_config(
    page_title="Freedom Bank · Анализ отзывов 2ГИС",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# Фирменная палитра Freedom Bank
# ============================================================
FREEDOM_GREEN = "#08bb48"        # основной бренд-зелёный
FREEDOM_DARK = "#085932"         # тёмный корпоративный
FREEDOM_LIGHT = "#E8F8EE"        # фон-подложка
FREEDOM_SOFT = "#BCE8C9"         # мягкий зелёный
FREEDOM_MID = "#4FCF7E"          # средний зелёный

# Семантические цвета для рейтингов
# Позитив — яркий салатовый #bedb41, негатив — красный #f94d4d
RATING_COLORS = {
    1: "#f94d4d",   # негатив — насыщенный красный
    2: "#f77878",   # мягче
    3: "#ffe04f",   # нейтрал — очень светлый салатовый
    4: "#d3e56a",   # приглушённый салатовый
    5: "#bedb41",   # позитив — фирменный салатовый
}
NEG_COLOR = "#f94d4d"
NEU_COLOR = "#ffe04f"
POS_COLOR = "#bedb41"

# Шкалы для градиентных графиков
RATING_SCALE = [RATING_COLORS[1], RATING_COLORS[2], RATING_COLORS[3], RATING_COLORS[4], RATING_COLORS[5]]
GREEN_SCALE = ["#FFFFFF", FREEDOM_SOFT, FREEDOM_MID, FREEDOM_GREEN, FREEDOM_DARK]

# Plotly-тема
PLOTLY_TEMPLATE = dict(
    layout=dict(
        font=dict(family="Inter, -apple-system, system-ui, sans-serif", color="#1a1a1a"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        colorway=[FREEDOM_GREEN, FREEDOM_DARK, FREEDOM_MID, RATING_COLORS[1], RATING_COLORS[3]],
        xaxis=dict(gridcolor="#F0F0F0", zerolinecolor="#F0F0F0"),
        yaxis=dict(gridcolor="#F0F0F0", zerolinecolor="#F0F0F0"),
    )
)

st.markdown(f"""
<style>
    /* Импорт шрифта Inter для красивой типографики */

    html, body, [class*="css"] {{
        font-family: 'Inter', -apple-system, system-ui, sans-serif;
    }}

    .main > div {{ padding-top: 1rem; }}

    /* Заголовки в корпоративном тёмно-зелёном */
    h1 {{
        color: {FREEDOM_DARK} !important;
        font-weight: 800 !important;
        letter-spacing: -0.02em;
    }}
    h2, h3 {{
        color: {FREEDOM_DARK} !important;
        font-weight: 700 !important;
        letter-spacing: -0.01em;
    }}

    /* Акцентная полоса под главным заголовком */
    h1::after {{
        content: "";
        display: block;
        width: 64px; height: 4px;
        background: linear-gradient(90deg, {FREEDOM_GREEN}, {FREEDOM_DARK});
        margin-top: 12px;
        border-radius: 2px;
    }}

    /* KPI-метрики в виде карточек */
    [data-testid="stMetric"] {{
        background: #FFFFFF;
        border: 1px solid #EEF2EE;
        border-radius: 14px;
        padding: 18px 20px 16px 20px;
        box-shadow: 0 1px 2px rgba(8, 89, 50, 0.04),
                    0 4px 12px rgba(8, 89, 50, 0.06);
        transition: transform 0.15s ease, box-shadow 0.15s ease;
    }}
    [data-testid="stMetric"]:hover {{
        transform: translateY(-2px);
        box-shadow: 0 2px 4px rgba(8, 89, 50, 0.06),
                    0 8px 20px rgba(8, 89, 50, 0.10);
    }}
    [data-testid="stMetricValue"] {{
        font-size: 1.5rem !important;
        font-weight: 800 !important;
        color: {FREEDOM_DARK} !important;
        letter-spacing: -0.02em;
    }}
    [data-testid="stMetricLabel"] {{
        font-size: 0.82rem !important;
        font-weight: 500 !important;
        color: #6B7A6F !important;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }}
    [data-testid="stMetricDelta"] {{
        font-size: 0.85rem !important;
        font-weight: 600 !important;
    }}

    /* Карточки отзывов */
    .review-card {{
        background: #FFFFFF;
        border: 1px solid #EEF2EE;
        border-left: 4px solid {FREEDOM_GREEN};
        padding: 16px 20px;
        margin: 10px 0;
        border-radius: 10px;
        font-size: 0.93rem;
        line-height: 1.55;
        box-shadow: 0 1px 3px rgba(8, 89, 50, 0.04);
        transition: box-shadow 0.15s ease;
    }}
    .review-card:hover {{
        box-shadow: 0 4px 12px rgba(8, 89, 50, 0.08);
    }}
    .review-card.neg {{ border-left-color: {NEG_COLOR}; }}
    .review-card.neu {{ border-left-color: {NEU_COLOR}; }}

    .review-meta {{
        color: #6B7A6F;
        font-size: 0.82rem;
        margin-bottom: 10px;
        display: flex;
        align-items: center;
        gap: 8px;
        flex-wrap: wrap;
    }}

    /* Бейдж рейтинга — вместо звёзд */
    .rating-badge {{
        display: inline-flex;
        align-items: center;
        justify-content: center;
        min-width: 28px;
        height: 26px;
        padding: 0 8px;
        border-radius: 6px;
        font-weight: 700;
        font-size: 0.85rem;
        color: #FFFFFF;
        letter-spacing: 0;
    }}
    .rating-badge.r1 {{ background: {RATING_COLORS[1]}; }}
    .rating-badge.r2 {{ background: {RATING_COLORS[2]}; }}
    .rating-badge.r3 {{ background: {RATING_COLORS[3]}; }}
    .rating-badge.r4 {{ background: {RATING_COLORS[4]}; }}
    .rating-badge.r5 {{ background: {RATING_COLORS[5]}; }}

    /* Табы */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 6px;
        border-bottom: 1px solid #EEF2EE;
    }}
    .stTabs [data-baseweb="tab"] {{
        border-radius: 8px 8px 0 0;
        padding: 10px 16px;
        font-weight: 600;
        color: #6B7A6F;
    }}
    .stTabs [aria-selected="true"] {{
        color: {FREEDOM_DARK} !important;
        background: {FREEDOM_LIGHT};
    }}

    /* Сайдбар */
    [data-testid="stSidebar"] {{
        background: #FAFCFA;
        border-right: 1px solid #EEF2EE;
    }}
    [data-testid="stSidebar"] h2 {{
        font-size: 1.1rem !important;
    }}

    /* Инфо-блоки (st.info / success / warning / error) */
    [data-testid="stAlert"] {{
        border-radius: 12px !important;
        border: none !important;
        padding: 16px 20px !important;
    }}

    /* Таблицы */
    [data-testid="stDataFrame"] {{
        border-radius: 10px;
        overflow: hidden;
        border: 1px solid #EEF2EE;
    }}

    /* Подложка для ответов банка */
    .bank-answer {{
        margin-top: 12px;
        padding: 12px 14px;
        background: {FREEDOM_LIGHT};
        border-left: 3px solid {FREEDOM_GREEN};
        border-radius: 6px;
        font-size: 0.88rem;
        line-height: 1.5;
    }}
    .bank-answer b {{ color: {FREEDOM_DARK}; }}

    /* Кастомный разделитель между визуализациями */
    hr {{
        border: none;
        height: 1px;
        background: linear-gradient(90deg,
            transparent 0%,
            #D5E3DA 15%,
            #B5D0C1 50%,
            #D5E3DA 85%,
            transparent 100%);
        margin: 28px 0 !important;
    }}
    .viz-divider {{
        height: 1px;
        background: linear-gradient(90deg,
            transparent 0%,
            #D5E3DA 15%,
            #B5D0C1 50%,
            #D5E3DA 85%,
            transparent 100%);
        margin: 24px 0 20px 0;
    }}

    /* Вертикальный разделитель между колонками */
    [data-testid="stHorizontalBlock"] > div[data-testid="stColumn"] + div[data-testid="stColumn"] {{
        border-left: 1px solid #E5EEE8;
        padding-left: 20px !important;
        margin-left: -1px;
    }}
    /* Дадим первой колонке тоже небольшой правый отступ */
    [data-testid="stHorizontalBlock"] > div[data-testid="stColumn"]:first-child {{
        padding-right: 8px !important;
    }}

    /* ── Тени для визуализаций ────────────────────────────── */
    [data-testid="stPlotlyChart"] {{
        border-radius: 18px;
        overflow: hidden;
        background: linear-gradient(145deg, #ffffff 0%, #f8fcf9 100%);
        box-shadow:
            0 1px 2px rgba(8, 89, 50, 0.03),
            0 4px 16px rgba(8, 89, 50, 0.08),
            0 16px 40px rgba(0, 0, 0, 0.05),
            inset 0 1px 0 rgba(255, 255, 255, 0.9);
        padding: 8px 4px 4px 4px;
        transition: box-shadow 0.25s ease, transform 0.2s ease;
        border: 1px solid rgba(220, 238, 228, 0.8);
    }}
    [data-testid="stPlotlyChart"]:hover {{
        box-shadow:
            0 2px 4px rgba(8, 89, 50, 0.06),
            0 10px 30px rgba(8, 89, 50, 0.12),
            0 24px 56px rgba(0, 0, 0, 0.07),
            inset 0 1px 0 rgba(255, 255, 255, 0.9);
        transform: translateY(-2px);
    }}
    /* Тени для датафреймов */
    [data-testid="stDataFrame"] {{
        border-radius: 12px !important;
        overflow: hidden;
        box-shadow:
            0 1px 2px rgba(8, 89, 50, 0.04),
            0 4px 12px rgba(8, 89, 50, 0.06);
    }}
    /* Тени для метрик при hover уже есть, усиливаем чуть */
    [data-testid="stMetric"]:hover {{
        box-shadow:
            0 2px 6px rgba(8, 89, 50, 0.08),
            0 10px 28px rgba(8, 89, 50, 0.12) !important;
    }}
</style>
""", unsafe_allow_html=True)

# ── Сохранение активной вкладки между ре-рандерами ────────────
# st.markdown('<script>') НЕ исполняет JS в Streamlit.
# components.html() работает в iframe с allow-same-origin →
# через window.parent.document получаем доступ к DOM.
import streamlit.components.v1 as _components
_components.html("""
<script>
(function () {
    const KEY = 'fb_active_tab';
    const doc = window.parent.document;
    let _restoring = false;

    function restoreTab() {
        const saved = sessionStorage.getItem(KEY);
        if (saved === null) return;
        const idx = parseInt(saved, 10);
        const tabs = doc.querySelectorAll('[data-baseweb="tab"]');
        if (!tabs.length) return;
        if (tabs[idx] && tabs[idx].getAttribute('aria-selected') !== 'true') {
            _restoring = true;
            tabs[idx].click();
            setTimeout(() => { _restoring = false; }, 200);
        }
    }

    doc.addEventListener('click', function (e) {
        if (_restoring) return;
        const tab = e.target.closest('[data-baseweb="tab"]');
        if (!tab) return;
        const allTabs = [...doc.querySelectorAll('[data-baseweb="tab"]')];
        const idx = allTabs.indexOf(tab);
        if (idx >= 0) sessionStorage.setItem(KEY, idx);
    }, true);

    let _timer = null;
    const obs = new MutationObserver(function () {
        clearTimeout(_timer);
        _timer = setTimeout(restoreTab, 120);
    });
    obs.observe(doc.body, { childList: true, subtree: true });

    setTimeout(restoreTab, 400);
    setTimeout(restoreTab, 1000);
})();
</script>
""", height=0)


import json
import re

def extract_json(text):
    try:
        # 1. пробуем напрямую
        return json.loads(text)
    except:
        pass

    # 2. ищем JSON внутри текста
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except:
            pass

    # 3. fallback
    return {
        "headline": "Ошибка парсинга AI",
        "hot_topics": [],
        "positive_highlights": [],
        "negative_highlights": [],
        "insight": text[:500]
    }


def call_groq(prompt, max_tokens=2000):
    url     = "https://api.groq.com/openai/v1/chat/completions"
    headers = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": "Ты аналитик данных. Отвечай ТОЛЬКО строгим JSON без markdown и backtick-ов. Все значения в JSON должны быть на русском языке."},
            {"role": "user",   "content": prompt},
        ],
        "temperature": 0.4,
        "max_tokens": max_tokens,
    }
    try:
        resp   = requests.post(url, headers=headers, json=payload, timeout=40)
        result = resp.json()
        if "error" in result:
            return {"error": result["error"]}
        raw = result["choices"][0]["message"]["content"]
        return extract_json(raw)
    except Exception as e:
        return {"error": str(e)}


def format_month_prompt(period, texts, sentiment_summary):
    reviews_block = "\n".join(['- ' + t[:200] for t in texts[:40]])

    return f"""
Ты senior CX аналитик.

Твоя задача — сделать аналитический вывод по отзывам.

---

ПЕРИОД: {period}

РАСПРЕДЕЛЕНИЕ СЕНТИМЕНТА:
{sentiment_summary}

ПРИМЕРЫ ОТЗЫВОВ:
{reviews_block}

---

ТРЕБОВАНИЯ:
- НЕ используй HTML
- НЕ добавляй пояснения до или после ответа
- НЕ используй markdown
- НЕ пересказывай отзывы
- НЕ давай советы бизнесу
- Дай причинно-следственный анализ (почему возникли проблемы)

---

ВЕРНИ СТРОГО JSON:
Начни ответ с {{ и закончи }}

{{
  "headline": "краткое резюме месяца (5-8 слов)",
  "hot_topics": ["тема 1", "тема 2", "тема 3", "тема 4"],
  "positive_highlights": ["что хвалили 1", "что хвалили 2", "что хвалили 3"],
  "negative_highlights": ["на что жаловались 1", "на что жаловались 2", "на что жаловались 3"],
  "insight": "4-6 предложений: причины, динамика, что происходило, что это значит для бизнеса"
}}

Если формат не соблюден — ответ считается ошибкой.
"""


def divider():
    """Красивый разделитель между визуализациями."""
    st.markdown("<div class='viz-divider'></div>", unsafe_allow_html=True)


def rating_badge(rating: int) -> str:
    """Возвращает HTML-бейдж с цифрой рейтинга."""
    return f"<span class='rating-badge r{rating}'>{rating}</span>"


def _is_theme_list(val) -> bool:
    """Проверяет, является ли значение итерируемым списком тем.
    Покрывает list, tuple и numpy.ndarray (после parquet round-trip)."""
    return val is not None and hasattr(val, '__iter__') and not isinstance(val, str) and hasattr(val, '__len__') and len(val) > 0


def apply_theme(fig):
    """Применяет фирменный шаблон к plotly-фигуре."""
    fig.update_layout(
        font=dict(family="Inter, -apple-system, system-ui, sans-serif", color="#1a1a1a"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(t=50, b=50, l=40, r=40),
        autosize=True,
    )
    fig.update_xaxes(gridcolor="#F0F0F0", zerolinecolor="#EEEEEE", automargin=True)
    fig.update_yaxes(gridcolor="#F0F0F0", zerolinecolor="#EEEEEE", automargin=True)
    return fig

# ============================================================
# Словарь тем — расширенный, с морфологическими вариантами
# ============================================================
THEMES = {
    # === Операционные проблемы ===
    'Очереди и ожидание': [
        'очеред', 'ожидан', 'ждать', 'ждал', 'ждала', 'ждали', 'жду',
        'медленн', 'долго', 'долгий', 'долгое', 'долго обслужива',
        'талон', 'электронная очеред', 'живая очеред',
        'час ждал', 'два часа', 'полчаса', 'пол часа', 'час прос',
        'кезек', 'күту', 'ұзақ күт',
        'простаива', 'просижива', 'зависли в очер',
    ],
    'Обслуживание и персонал': [
        'хамств', 'хамит', 'нахамил', 'нахамила',
        'грубост', 'груб', 'грубо', 'грубиян',
        'некомпетент', 'невежлив', 'неуважел', 'неуважа',
        'неприветлив', 'непрофессионал', 'непрофессионально',
        'отношен', 'обслуживан', 'плохой сервис', 'ужасный сервис',
        'неудовлетвор', 'плохо обслужи', 'разочарова',
        'орёт', 'кричит', 'накричал', 'накричала',
        'нет улыбки', 'без улыбки', 'холодно встрети',
        'қызмет', 'нашар қызмет', 'қызмет көрсет',
        'никого ответа', 'нет ответа от','төмен', 'плохое', 'нет порядка'
    ],
    'Компетентность сотрудников': [
        'не знают', 'не знает', 'не знал', 'не знала',
        'не смог', 'не смогла', 'не могут', 'не может',
        'некомпетент', 'не разбира', 'путают', 'путается',
        'дают разную информац', 'противоречив', 'разные ответы',
        'не владеет', 'не в курсе', 'не осведомл',
        'ошибся', 'ошиблась', 'ошибочн', 'неверн информац',
        'не умеет', 'не обучен',
        # казахские
        'түсіндіре алмайды', 'түсіндіре алма', 'түсініксіз',
        'білмейді', 'білмеді', 'үйретілмеген',
        'қайда бару керегін', 'не істеу керектігін',
        'түсіндіре', 'түсінікт',
    ],
    'Режим работы / график': [
        'режим', 'график', 'закрыт', 'закрыто', 'закрыли',
        'обеденн', 'перерыв', 'не работает', 'не работал',
        'выходн', 'рано закрыл', 'рано закрыла',
        'поздно открыва', 'рано закрыва',
        'не успел', 'опоздал', 'уже закрыт',
        'жұмыс уақыты', 'демалыс',
        'жабық', 'жабық тұр', 'жұмыс істемейді',
        'нет на месте', 'отсутству',
    ],
    # === Цифровые каналы ===
    'Мобильное приложение': [
        'приложен', 'мобильн приложен', 'aqsha', 'интерфейс',
        'глючит', 'глюк', 'вылета', 'вылетает', 'вылетело',
        'обновлен', 'лагает', 'лаги', 'тормозит', 'тормоза',
        'не загружа', 'не открывается', 'не запускается',
        'зависает', 'зависло', 'завис',
        'не работает приложен', 'сломалось приложен',
        'апп', 'ios', 'android', 'playmarket', 'appstore',
        'ошибка в приложен', 'баг', 'не обновля',
    ],
    'Видеоидентификация / открытие счёта': [
        'видео идентификац', 'видеоидентификац', 'видео верификац',
        'видеоверификац', 'верификац',
        'не могу открыть счёт', 'открыть счёт', 'открытие счёт',
        'онбординг', 'регистрац', 'не регистрирует',
        'видео не работ', 'камера не работ',
        # казахские
        'видео идентификация ашпайды', 'идентификация ашпайды',
        'аккаунт ашылмайды', 'есептік жазба',
        'идентификация жасалмады',
    ],
    'Интернет-банкинг': [
        'интернет-банк', 'интернет банк', 'онлайн банк',
        'веб-версия', 'личный кабинет', 'сайт не работ',
        'онлайн кабинет', 'не могу войти', 'не входит',
        'авторизац', 'пароль не подходит',
    ],
    'Call-центр': [
        'колл', 'call', 'горячая лини', 'дозвон', 'дозвониться',
        'оператор', 'не дозвон', 'кол-центр', 'колл-центр',
        'по телефону', 'звони', 'звонил', 'звонила',
        'не берут трубку', 'не отвечают по телефон', 'робот отвеча',
        'ожидание на линии', 'музыка ожидания', 'на удержани',
        'горячая', 'служба поддержки',
        # казахские
        'қоңырау', 'қоңырау шалу', 'қоңырау шалғанда',
        'телефонмен', 'хабарласу', 'хабарласпай',
        'жауап бермейді', 'жауап берм',
    ],
    'Чат / онлайн-поддержка': [
        'чат', 'чат-бот', 'бот отвеча', 'поддержк', 'директ',
        'instagram direct', 'в директ', 'онлайн поддержк',
        'не отвечают в чате', 'бот не помога',
        'техподдержк', 'тех поддержк',
    ],
    # === Продукты ===
    'Кредиты': [
        'кредит', 'заём', 'займ', 'рассрочк',
        'одобр кредит', 'отказ в кредит', 'отказали в кредит',
        'кредитн', 'кредитный лимит', 'задолженност',
        'просрочк', 'штраф', 'пеня', 'переплат',
        'кредит не одобр', 'отказ по кредит',
    ],
    'Ипотека': [
        'ипотек', 'отбасы банк', 'жилищн', 'первоначальн взнос',
        'ипотечн', 'жилищный кредит',
        'тұрғын үй', 'отбасы',
    ],
    'Карты (выпуск/перевыпуск)': [
        'карт', 'пластик', 'visa', 'mastercard',
        'выпуск карт', 'перевыпуск', 'именн карт',
        'не пришла карт', 'карта не работ', 'карта заблок',
        'зарплатн карт', 'дебетов', 'кредитн карт',
        'ждать карт', 'долго делают карт',
    ],
    'Переводы и платежи': [
        'перевод', 'платёж', 'платеж', 'оплат',
        'не прошёл платёж', 'зависл перевод', 'завис платёж',
        'не дошли деньги', 'деньги не пришли', 'задержка перевода',
        'неправильный перевод', 'дважды списали', 'двойное списани',
        'ошибочный перевод', 'перевести не могу',
        'swift', 'межбанк',
    ],
    'Банкомат / наличные': [
        'банкомат', 'банкоматт', 'atm', 'банкомат не выдал',
        'снять наличные', 'снятие наличных', 'наличн',
        'не выдал деньги', 'банкомат съел', 'карту съел банкомат',
        'купюры', 'деньги не вышли', 'банкомат взял',
        'чек не выдал', 'не дал чек', 'не дал деньги',
        'на счёт не поступил', 'деньги зависли в банкомат',
        'банкомат не работ', 'банкомат сломан',
        # казахские
        'банкомат жұтып', 'банкоматта',
    ],
    'Депозиты и вклады': [
        'депозит', 'вклад', 'накопит', 'процент по вкладу',
        'ставк по депозит', 'открыть депозит',
        'закрыть депозит', 'депозит не открыв', 'доходность',
    ],
    'Обмен валют': [
        'обмен валют', 'курс валют', 'доллар', 'евро',
        'тенге на доллар', 'конвертац', 'курс обмена',
        'плохой курс', 'невыгодный курс', 'купить доллар',
        'валютн', 'рубл',
    ],
    'Документы и справки': [
        'справк', 'документ', 'выписк', 'подтвержден',
        'справка о доход', 'выписка по счёт', 'выписка по счет',
        'заверен', 'анкет',
        'не дают справк', 'долго делают справк',
    ],
    # === Проблемы с деньгами / безопасность ===
    'Блокировка счёта / карты': [
        'блокир', 'заблок', 'разблок', 'арест счёт',
        'замороз', 'приостанов', 'счёт заморожен',
        'карта заблокирован', 'счёт заблокирован',
        'не могу снять деньги', 'деньги недоступн',
        'аккаунт заблок',
        # казахские
        'бұғатталды', 'бұғатталған', 'шот бұғатталды',
    ],
    'Пропажа денег / мошенничество': [
        'украли', 'украден', 'пропал', 'пропали деньги',
        'мошен', 'списали', 'без моего ведом',
        'незаконн спис', 'фишинг', 'скам',
        'мошенники', 'развод', 'обманул', 'обманули',
        'обманщик', 'обманщиков', 'банк обманыва',
        'несанкционирован', 'деньги исчезли',
        # казахские
        'алаяқ', 'алаяқтар', 'алаяқтық',
        'алдады', 'алдап', 'жеміт',
    ],
    'Возврат средств': [
        'возврат', 'вернуть деньги', 'не возвраща',
        'чарджбэк', 'chargeback', 'refund', 'возмещен',
        'верните', 'не вернули', 'задержка возврата',
    ],
    # === Travel / доп. сервисы ===
    'Travel / Билеты': [
        'билет', 'рейс', 'авиа', 'freedom travel', 'freedom tickets',
        'покупк билет', 'бронирован', 'airba', 'туда обратно',
        'обратн билет', 'аэропорт', 'авиакомпани',
        'не вернули за билет', 'сдать билет', 'отмен рейс',
    ],
    'Страхование': ['страхов', 'полис', 'каско', 'осаго', 'страховк', 'страховой случай'],
    'Инвестиции / брокер': [
        'инвест', 'брокер', 'акции', 'фонд', 'etf', 'tradernet',
        'портфель', 'дивиденд', 'торги', 'биржа',
    ],
    # === Freedom Ticketon ===
    'Покупка и оплата билетов': [
        'ticketon', 'тикетон', 'событие', 'событий', 'концерт',
        'мероприят', 'кино', 'кинотеатр', 'театр', 'шоу', 'фестиваль',
        'купить билет', 'оформить билет', 'заказ билет',
        'не могу купить', 'не проход оплат',
        'электронный билет', 'qr-код', 'qr код', 'билет не пришёл',
        'промокод', 'скидк на билет',
    ],
    'Bозврат и отмена': [
        'вернуть билет', 'возврат билет', 'отмен мероприят',
        'отмен концерт', 'не вернули деньги за билет',
        'перенос события', 'перенос концерт',
        'невозвратн билет', 'отказ в возврат',
        'refund ticket', 'chargeback билет',
    ],
    'Доставка и электронный билет': [
        'не получил билет', 'не пришёл билет', 'потерял билет',
        'не отображается билет', 'QR не работает',
        'не могу скачать', 'не могу открыть билет',
        'доставка билет', 'pdf билет',
    ],
    # === Freedom Travel ===
    'Туры и бронирование': [
        'тур', 'туристическ', 'путёвк', 'путевк', 'бронирован тура',
        'горящ тур', 'freedom travel', 'туристическое агентств',
        'экскурси', 'трансфер', 'тур агентств',
        'путешестви', 'отдых за рубежом', 'туристическ поездк',
    ],
    'Отели и проживание': [
        'отель', 'гостиниц', 'хостел', 'апартамент', 'апарт',
        'бронирован отел', 'отель не подтвержд', 'отмен отел',
        'check-in', 'заселен', 'выселен', 'раннее заселен',
        'стоимость отел', 'переплатил за отел',
        'hotel', 'резервац',
    ],
    'Визы и документы': [
        'виза', 'визов', 'загранпаспорт', 'загран паспорт',
        'документы для поездки', 'консульств',
        'страховк для визы', 'визовый сбор',
        'отказ в визе',
    ],
    'Возврат и отмена тура': [
        'отмен тур', 'возврат за тур', 'не вернули за тур',
        'отмен брониров', 'форс-мажор', 'форс мажор',
        'не смог улететь', 'задержка рейс',
        'перенос тур', 'вернуть деньги за тур',
    ],
    # === Freedom Drive ===
    'Аренда и каршеринг': [
        'каршеринг', 'аренда авто', 'аренда машин',
        'freedom drive', 'фридом драйв',
        'взять авто', 'авто напрокат', 'прокат авто',
        'не могу открыть машину', 'машина не открывается',
        'забронировать авто', 'бронирован авто',
    ],
    'Состояние и чистота авто': [
        'грязная машина', 'грязный авто', 'чистот авто',
        'царапин', 'вмятин', 'поврежден авто',
        'сломан авто', 'неисправн авто',
        'нет бензин', 'мало топлив', 'заглох',
        'запах в машин', 'неприятный запах',
    ],
    'Приложение и оплата каршеринга': [
        'приложение drive', 'приложен каршеринг',
        'не работает приложен drive',
        'не могу завершить поездку', 'поездка не закрыл',
        'лишнее списание drive', 'ошибочн списан drive',
        'тариф каршеринг', 'стоимость поездки',
        'минуты не корректн', 'зависло приложен drive',
    ],
    'ДТП и штрафы': [
        'штраф drive', 'штраф за паркову', 'штраф за каршеринг',
        'дтп', 'авария', 'страховой случай drive',
        'ущерб авто', 'угон',
        'парковк запрещен', 'неправильн парковк',
    ],
    # === Freedom Insurance ===
    'Insurance — ОСАГО и КАСКО': [
        'осаго', 'каско', 'автострахован',
        'страхование авто', 'полис авто',
        'страховой случай авто', 'ДТП страховк',
        'е-осаго', 'электронн осаго', 'оформить осаго',
        'осаго не выдают', 'осаго отказ',
    ],
    'Медицинское страхование': [
        'медицинск страхов', 'дмс', 'омс', 'смс страхов',
        'страхование здоровья', 'страховой полис здоровье',
        'медстрахов', 'медицинский полис',
        'страховой случай болезнь', 'лечение по страховк',
    ],
    'Insurance — выплата и страховой случай': [
        'страховая выплат', 'не платят страховк',
        'отказ в выплате', 'задержка выплат страхов',
        'страховая не возмещает', 'страховая отказала',
        'акт о страховом случае', 'оформить страховой случай',
        'возмещен ущерб', 'страховое возмещен',
    ],
    'Insurance — оформление полиса': [
        'оформить полис', 'купить полис',
        'полис не приходит', 'полис не отображается',
        'не могу оформить страхов', 'страховк не работает',
        'цена полис', 'дорогой полис', 'тариф страхов',
        'продление полис', 'срок полис',
    ],
    # === Freedom Broker ===
    'Broker — открытие и ведение счёта': [
        'брокерский счёт', 'открыть брокерск', 'брокерск счёт',
        'ИИС', 'индивидуальн инвест счёт',
        'tradernet', 'фридом брокер',
        'не могу открыть брокерск', 'брокерск счёт заблок',
        'верификация брокер', 'идентификация брокер',
    ],
    'Торги и сделки': [
        'купить акции', 'продать акции', 'заявк на торги',
        'сделка не прошла', 'ордер не исполнен',
        'котировк', 'стакан заявок', 'биржевой стакан',
        'не могу продать', 'не могу купить акции',
        'рыночная заявка', 'лимитн заявка',
        'торговая платформа', 'терминал',
    ],
    'Вывод и ввод средств': [
        'вывод средств брокер', 'вывести с брокерск',
        'ввод средств брокер', 'пополнить брокерск',
        'деньги не поступили на брокерск',
        'задержка вывода', 'комисси вывод брокер',
        'не могу вывести с брокер',
    ],
    'Комиссии и тарифы брокера': [
        'комисси брокер', 'тариф брокер',
        'скрытые комисси брокер', 'комисси за сделку',
        'депозитарн комисси', 'обслуживани счёт брокер',
        'дорогой брокер', 'невыгодные тарифы брокер',
    ],
    'Дивиденды и купоны': [
        'дивиденд', 'купон', 'выплат по бумагам',
        'дивиденд не пришёл', 'купон не поступил',
        'налог на дивиденд', 'удержан налог',
        'доход по облигац',
    ],
    # === Сервис-позитив ===
    'Благодарность сотруднику': [
        'спасибо', 'благодар', 'рахмет', 'помогл', 'помог', 'помогла',
        'отличн', 'профессионал', 'внимательн', 'вежлив', 'замечательн',
        'молодец', 'умница', 'хорошо объяснил', 'всё объяснил',
        'хороший сотрудник', 'отличный специалист', "ракмет",
        'рахmet', 'алғыс',
    ],
    'Быстрое обслуживание': [
        'быстро', 'быстр', 'оперативн', 'за минут', 'моментальн',
        'без очеред', 'тез', 'жылдам',
        'мгновенно', 'сразу помог', 'без ожидания',
    ],
    'Рекомендация другим': [
        'рекоменд', 'советую', 'лучший банк', 'номер один', 'топ банк',
        'всем советую', 'буду рекомендовать', 'посоветовал',
    ],
    # === Прочее ===
    'Проблемы при закрытии счёта': [
        'закрыть счёт', 'закрытие счёт', 'не закрывают', 'расторгн',
        'закрыли счёт', 'нельзя закрыть',
    ],
    'Комиссии и тарифы': [
        'комисси', 'тариф', 'подорожал', 'дорогой банк', 'скрытые плат',
        'высокие тарифы', 'дорого стоит',
        'скрытые комисси', 'списывают деньги', 'автоспис',
        'комисси за обслуживан',   # только "комиссия за обслуживание", не "спасибо за обслуживание"
        'стоимость обслуживан',
        'платное обслуживани',
    ],
}

# ============================================================
# Простой суффиксный стеммер для русского (без внешних зависимостей)
# ============================================================
_RU_SUFFIXES = (
    'ующего', 'ующему', 'ующих', 'ующим', 'ующий', 'ующей', 'ующем',
    'ующая', 'ующее', 'ующие',
    'ований', 'ованию', 'ованием', 'ованиях', 'ований',
    'ениям', 'ениях', 'ением', 'ению', 'ений', 'ении',
    'аниям', 'аниях', 'анием', 'анию', 'аний', 'ании',
    'ости', 'остям', 'остях', 'остью',
    'ться', 'тся',
    'ался', 'алась', 'ались',
    'ился', 'илась', 'ились',
    'овать', 'евать', 'ивать', 'ывать',
    'ования', 'евания',
    'ого', 'его', 'ому', 'ему',
    'ами', 'ями',
    'ных', 'ным', 'ной', 'ном', 'ное', 'ные', 'ная',
    'ние', 'ния',
    'ать', 'ять', 'ить', 'уть',
    'ает', 'яет', 'ует',
    'ают', 'яют', 'уют',
    'ала', 'яла', 'ила', 'ула',
    'али', 'яли', 'или', 'ули',
    'ах', 'ях', 'ом', 'ем', 'ую',
    'ой', 'ей', 'ий', 'ые', 'ие',
    'ов', 'ев', 'ям', 'ам',
    'ью', 'ть', 'ти',
    'ит', 'ат', 'ют', 'ен',
)

def _stem_ru(word: str) -> str:
    """Минимальный суффиксный стеммер для русских слов (без зависимостей)."""
    if len(word) <= 5:
        return word
    for suf in _RU_SUFFIXES:
        if word.endswith(suf) and len(word) - len(suf) >= 4:
            return word[:-len(suf)]
    return word


def _find_themes(text):
    """
    Улучшенная категоризация отзыва по темам.
    Шаг 1: быстрая проверка подстроки (ловит заранее заданные корни).
    Шаг 2: стемминг токенов отзыва + сравнение с нормализованными ключами
            (ловит флексии, которых нет в словаре).
    """
    if pd.isna(text):
        return []
    t = text.lower()
    result = []
    # Pre-stem all tokens of the review once
    tokens = re.findall(r'[а-яёa-z]{4,}', t)
    stemmed_tokens = {_stem_ru(tok) for tok in tokens}
    long_stemmed = {s for s in stemmed_tokens if len(s) >= 5}
    theme_stems = _get_theme_stems_cache()
    for theme, keywords in THEMES.items():
        matched = any(kw in t for kw in keywords)
        if not matched:
            # fallback: только kw_stem как префикс токена (не наоборот — иначе
            # "банк" будет матчить составной ключ "интернетбанк")
            for kw_stem in theme_stems[theme]:
                if any(st_tok.startswith(kw_stem) for st_tok in long_stemmed):
                    matched = True
                    break
        if matched:
            result.append(theme)
    return result


_THEME_STEMS_CACHE: dict[str, list[str]] | None = None


def _get_theme_stems_cache() -> dict[str, list[str]]:
    """Один раз на процесс считает стеммы всех ключей THEMES.
    Без этого _stem_ru(...) вызывался бы заново на каждый отзыв × каждый ключ."""
    global _THEME_STEMS_CACHE
    if _THEME_STEMS_CACHE is None:
        cache = {}
        for theme, keywords in THEMES.items():
            stems = []
            for kw in keywords:
                kw_clean = re.sub(r'\s+', '', kw)
                kw_stem = _stem_ru(kw_clean)
                if len(kw_stem) >= 5:
                    stems.append(kw_stem)
            cache[theme] = stems
        _THEME_STEMS_CACHE = cache
    return _THEME_STEMS_CACHE


def _clean_person(val):
    """
    Нормализует значение из колонки «люди»:
    - берёт только часть до первого переноса строки
    - фильтрует нереальные значения (короткие слова, фразы > 3 слов, латиница/цифры)
    - возвращает имя с заглавной буквы каждого слова или None
    """
    if pd.isna(val):
        return None
    val = str(val).split('\n')[0].strip()
    val = re.sub(r'\s+', ' ', val).strip()
    # Только кириллица (включая казахские буквы) и дефис
    if not re.match(r'^[А-ЯЁа-яёҒғҚқҢңҮүҰұЖжіӘәҺһ\s-]+$', val):
        return None
    parts = val.split()
    if len(parts) == 1 and len(val) < 5:   # одно короткое слово — не имя
        return None
    if len(parts) > 3:                      # длинная фраза — не имя
        return None
    return val.title()


def _parse_answer(s):
    """Парсим официальный ответ банка (Python dict в виде строки)."""
    if pd.isna(s) or not str(s).strip():
        return None, None
    try:
        d = ast.literal_eval(s)
        return d.get('text'), d.get('date_created')
    except Exception:
        return None, None


def _prepare(df: pd.DataFrame) -> pd.DataFrame:
    """Полная предобработка данных (логика из prepare_data.py)."""
    df['datetime'] = pd.to_datetime(df['дата'].astype(str) + ' ' + df['время'].astype(str))
    df['дата'] = pd.to_datetime(df['дата'])
    df = df.sort_values('datetime').reset_index(drop=True)

    if 'likes_count' not in df.columns:
        df['likes_count'] = 0
    df['likes_count'] = df['likes_count'].fillna(0).astype(int)

    df['длина_текста'] = df['текст'].fillna('').str.len()
    df['кол_слов'] = df['текст'].fillna('').str.split().str.len()

    parsed = df['official_answer'].apply(_parse_answer)
    df['ответ_текст'] = [p[0] for p in parsed]
    df['ответ_дата'] = pd.to_datetime(
        [p[1] for p in parsed], errors='coerce', utc=True
    ).tz_convert(None)
    df['есть_ответ'] = df['ответ_текст'].notna()

    if 'отделение' in df.columns:
        df['отделение'] = df['отделение'].fillna('—').astype(str).str.strip()
    else:
        df['отделение'] = df['город']+'Офис'

    if 'город' in df.columns:
        df['город'] = df['город'].fillna('—').astype(str).str.strip()
    else:
        df['город'] = '—'

    df['часов_до_ответа'] = (df['ответ_дата'] - df['datetime']).dt.total_seconds() / 3600
    df.loc[df['часов_до_ответа'] < 0, 'часов_до_ответа'] = None

    df['темы'] = df['текст'].apply(_find_themes)

    # Колонка «люди» — сохраняем как есть (очистка происходит в визуализации)
    if 'люди' not in df.columns:
        df['люди'] = None

    return df


# ============================================================
# Загрузка данных
# ============================================================

# Все источники данных: fallback-файл + шаблоны городских файлов.
DATA_SOURCES = {
    "Freedom Bank": {
        "patterns": ["freedom_ban*.csv"],
    },
    "Freedom Ticketon": {
        "fallback": "freedom_ticketon.csv",
    },
    "Freedom Travel": {
        "fallback": "freedom_travel.csv",
    },
    "Freedom Insurance": {
        "patterns": ["freedom_ins*.csv"],
    },
    "Freedom Drive": {
        "patterns": ["freedom_driv*.csv"],
    },
    "Freedom Broker": {
        "patterns": ["freedom_broke*.csv"],
    },
}

CITY_CODE_MAP = {
    "aktau": "Актау",
    "akx": "Актобе",
    "ast": "Астана",
    "ast1": "Астана",
    "ast2": "Астана",
    "atyr": "Атырау",
    "koksh": "Кокшетау",
    "konaev": "Конаев",
    "kost": "Костанай",
    "krg1": "Караганда",
    "krg2": "Караганда",
    "kyzyl": "Кызылорда",
    "pav": "Павлодар",
    "pav1": "Павлодар",
    "pet": "Петропавловск",
    "selo": "село Коргас",
    "semey": "Семей",
    "shym": "Шымкент",
    "shym1": "Шымкент",
    "taldyk": "Талдыкорган",
    "taraz": "Тараз",
    "turk": "Туркестан",
    "ural": "Уральск",
    "ust": "Усть-Каменогорск",
}


def _infer_city_from_filename(filename: str) -> str | None:
    code = Path(filename).stem.lower().split('_')[-1]
    return CITY_CODE_MAP.get(code)


def _resolve_source_filenames(config: dict) -> list[str]:
    base = Path(__file__).parent
    matched_files = []
    for pattern in config.get("patterns", []):
        matched_files.extend(sorted(base.glob(pattern)))

    if matched_files:
        return [p.name for p in matched_files]

    fallback = config.get("fallback")
    return [fallback] if fallback else []


def _hydrate_parquet_df(df: pd.DataFrame, label: str, inferred_city: str | None) -> pd.DataFrame:
    """Добивает недостающие колонки при чтении старого parquet-кэша."""
    df['datetime'] = pd.to_datetime(df['datetime'])
    df['дата'] = pd.to_datetime(df['дата'])
    if 'likes_count' not in df.columns:
        df['likes_count'] = 0
    df['likes_count'] = df['likes_count'].fillna(0).astype(int)
    if 'темы' not in df.columns:
        df['темы'] = df['текст'].apply(_find_themes)
    if 'люди' not in df.columns:
        df['люди'] = None
    if 'отделение' not in df.columns:
        df['отделение'] = '—'
    else:
        df['отделение'] = df['отделение'].fillna('—').astype(str).str.strip()
    if 'город' not in df.columns:
        df['город'] = inferred_city or '—'
    else:
        df['город'] = df['город'].fillna('—').astype(str).str.strip()
        if inferred_city:
            df['город'] = df['город'].replace('—', inferred_city)
    df['источник'] = label
    return df


@st.cache_data
def load_source_file(label: str, filename: str) -> pd.DataFrame | None:
    """Загружает один датасет, добавляет колонку «источник» и возвращает обработанный df.
    Возвращает None, если файл не найден.

    Быстрый путь: если рядом с CSV уже лежит parquet-кэш (свежий, не старше CSV),
    читаем его — это на порядок быстрее, чем перезапускать _prepare() на каждом
    старте Streamlit-процесса. Медленный путь (CSV → _prepare → сохранить parquet)
    выполняется один раз; все последующие запуски идут по быстрому пути.
    """
    base = Path(__file__).parent
    csv_path = base / filename
    parquet_path = base / (Path(filename).stem + ".parquet")
    inferred_city = _infer_city_from_filename(filename)

    # Быстрый путь: parquet существует И он не старше CSV (или CSV нет вовсе)
    parquet_is_fresh = parquet_path.exists() and (
        not csv_path.exists()
        or parquet_path.stat().st_mtime >= csv_path.stat().st_mtime
    )
    if parquet_is_fresh:
        df = pd.read_parquet(parquet_path)
        return _hydrate_parquet_df(df, label, inferred_city)

    # Медленный путь: читаем CSV, обрабатываем, сохраняем parquet на будущее
    if csv_path.exists():
        raw = pd.read_csv(csv_path, encoding='utf-8-sig')
        raw.columns = raw.columns.str.strip().str.lower()
        prepared = _prepare(raw)
        if inferred_city:
            prepared['город'] = prepared['город'].replace('—', inferred_city)
        prepared['источник'] = label

        # Сохраняем parquet как дисковый кэш для следующих запусков.
        # Не критично, если не получилось (например, нет прав на запись).
        try:
            prepared.to_parquet(parquet_path, index=False)
        except Exception:
            pass

        return prepared

    # CSV нет, но parquet есть (хоть и устаревший по mtime проверке выше
    # — здесь не может быть, т.к. parquet_is_fresh=True отработал бы).
    # На всякий случай: если только parquet — читаем его.
    if parquet_path.exists():
        df = pd.read_parquet(parquet_path)
        return _hydrate_parquet_df(df, label, inferred_city)

    return None


@st.cache_data
def load_source(label: str, config: dict) -> pd.DataFrame | None:
    frames = []
    for filename in _resolve_source_filenames(config):
        part = load_source_file(label, filename)
        if part is not None:
            frames.append(part)

    if not frames:
        return None

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.sort_values('datetime').reset_index(drop=True)
    return combined


@st.cache_data
def load_all_data() -> pd.DataFrame:
    """Загружает все доступные датасеты и объединяет в один DataFrame."""
    frames = []
    for label, config in DATA_SOURCES.items():
        part = load_source(label, config)
        if part is not None:
            frames.append(part)

    if not frames:
        st.error(
            "Файлы данных не найдены. Положите хотя бы один из файлов "
            "(**freedom_bank.csv**, **freedom_ticketon.csv**, **freedom_travel.csv**, "
            "**freedom_insurance.csv**, **freedom_drive.csv**) "
            "в ту же папку, что и app.py, и перезапустите приложение."
        )
        st.stop()

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.sort_values('datetime').reset_index(drop=True)
    return combined


AVAILABLE_SOURCES = [
    label for label, config in DATA_SOURCES.items()
    if _resolve_source_filenames(config)
]


# ============================================================
# Sidebar — фильтры
# ============================================================
st.sidebar.markdown("## Фильтры")

# ── Предзагрузка всего: нужно, чтобы собрать полный список городов
#    (до того, как пользователь выберет продукт). Функция кэшируется через
#    @st.cache_data + дисковый parquet-кэш, поэтому вызов дешёвый после первой
#    сборки кэша.
full_df = load_all_data()

# ── Выбор города (первым — как хочет пользователь) ────────
st.sidebar.markdown("**Город**")
_ALL_CITY_OPTION = "Все города"
all_cities = sorted(
    city for city in full_df['город'].dropna().astype(str).str.strip().unique().tolist()
    if city and city != '—'
)
selected_city = st.sidebar.selectbox(
    label="Выбери город",
    options=all_cities+[_ALL_CITY_OPTION],
    index=0,
    label_visibility="collapsed",
)

# Применяем фильтр по городу к мастер-датасету
if selected_city == _ALL_CITY_OPTION:
    city_df = full_df
else:
    city_df = full_df[full_df['город'] == selected_city]

st.sidebar.markdown("---")

# ── Выбор продукта/источника (зависит от выбранного города) ─
st.sidebar.markdown("**Продукт / источник**")
_ALL_OPTION = "Все продукты"
# Список продуктов — только те, которые реально встречаются в выбранном городе
products_in_city = [
    label for label in AVAILABLE_SOURCES
    if (city_df['источник'] == label).any()
]
if not products_in_city:
    # Случай: в выбранном городе нет данных ни по одному продукту. Такое
    # возможно, только если CITY в мастер-датасете есть, но после фильтра
    # пустой (редкость) — оставляем полный список, чтобы не блокировать UI.
    products_in_city = list(AVAILABLE_SOURCES)

_source_options = products_in_city + [_ALL_OPTION]
selected_source = st.sidebar.selectbox(
    label="Выбери продукт",
    options=_source_options,
    index=0,
    label_visibility="collapsed",
)

# Применяем фильтр по продукту
if selected_source == _ALL_OPTION:
    df = city_df.copy()
else:
    df = city_df[city_df['источник'] == selected_source].copy()

if len(df) == 0:
    st.error(
        f"Нет данных для комбинации: город «{selected_city}» + продукт «{selected_source}». "
        "Выберите другую комбинацию в сайдбаре."
    )
    st.stop()

st.sidebar.markdown("---")

# ── Период ────────────────────────────────────────────────
import datetime as _dt

df['дата'] = pd.to_datetime(df['дата'], format='%d.%m.%Y', errors='coerce')
min_date = df['дата'].min().date()
max_date = df['дата'].max().date()

# Streamlit date_input при открытии календаря использует сегодняшнюю дату.
# Если max_value < today — виджет показывает "End date set outside allowed range".
# Расширяем верхнюю границу до сегодня, чтобы календарь не ломался.
max_date_widget = max(max_date, _dt.date.today())

# Защита: если session_state хранит даты от другого продукта/города — сбросить.
if 'date_range' in st.session_state:
    _saved = st.session_state['date_range']
    if isinstance(_saved, (list, tuple)):
        if any(d < min_date or d > max_date_widget for d in _saved):
            del st.session_state['date_range']
            st.rerun()

date_range = st.sidebar.date_input(
    "Период",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date_widget,
    key=f"date_range_{city_df}"
)

# Когда пользователь кликнул только 1-ю дату (ещё не выбрал вторую) —
# date_range содержит 1 элемент. Ждём вторую, показываем подсказку.
if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
    start_date, end_date = date_range
else:
    start_date, end_date = min_date, max_date
    st.sidebar.caption("⏳ Выберите вторую дату…")

# Применяем фильтры по источнику + дате
mask = (
    (df['дата'] >= pd.Timestamp(start_date)) &
    (df['дата'] <= pd.Timestamp(end_date))
)
dff = df[mask].copy()

st.sidebar.markdown("---")


# ============================================================
# Заголовок — динамически меняется при выборе дочерней компании
# ============================================================
_COMPANY_META = {
    "Freedom Bank":       ("🏦", "Freedom Bank"),
    "Freedom Ticketon":   ("🎫", "Freedom Ticketon"),
    "Freedom Travel":     ("✈️", "Freedom Travel"),
    "Freedom Drive":      ("🚗", "Freedom Drive"),
    "Freedom Insurance":  ("🛡️", "Freedom Insurance"),
    "Freedom Broker":     ("📈", "Freedom Broker"),
}
col_title, col_logo = st.columns([4, 1])
with col_title:
    if selected_source == _ALL_OPTION:
        _header = "🏢 Freedom Group — Анализ отзывов 2ГИС"
    else:
        _icon, _name = _COMPANY_META.get(selected_source, ("🏢", selected_source))
        _header = f"{_icon} {_name} — Анализ отзывов 2ГИС"
    if selected_city != _ALL_CITY_OPTION:
        _header = f"{_header} · {selected_city}"
    st.markdown(f"# {_header}")

# ============================================================
# KPI метрики
# ============================================================
if len(dff) == 0:
    st.warning("Нет отзывов, соответствующих фильтрам.")
    st.stop()

avg_rating = dff['рейтинг'].mean()
total_reviews = len(dff)
neg_share = (dff['рейтинг'] <= 2).mean() * 100
pos_share = (dff['рейтинг'] >= 4).mean() * 100
answer_rate = dff['есть_ответ'].mean() * 100
median_response_h = dff['часов_до_ответа'].median()

# Adjusted rating — trimmed mean 10%
def trimmed_mean(s, pct=0.1):
    s = s.dropna().sort_values()
    if len(s) < 10:
        return s.mean()
    k = int(len(s) * pct)
    return s.iloc[k:len(s)-k].mean() if len(s) - 2*k > 0 else s.mean()

adj_rating = trimmed_mean(dff['рейтинг'])
median_rating = dff['рейтинг'].median()

# Repeat-авторы (нужны ниже в когортах и auto-insights, оставляем расчёт)
author_counts_global = df['автор'].value_counts()
GENERIC_AUTHORS = {'Apple User', 'Городская легенда', 'City spirit',
                   'Городской житель', 'Гость', '. .', 'A A'}
real_authors = dff[~dff['автор'].isin(GENERIC_AUTHORS)]
repeat_mask = real_authors['автор'].map(author_counts_global) >= 2
repeat_share = repeat_mask.mean() * 100 if len(real_authors) else 0

# Единый ряд KPI — 6 плиток одинакового размера
k1, k2, k3, k4 = st.columns(4)
k1.metric("Средний рейтинг", f"{avg_rating:.2f}")
k2.metric("Всего отзывов", f"{total_reviews:,}")
k3.metric("Доля рейтинг 1-2", f"{neg_share:.1f}%")
k4.metric("Доля рейтинг 4-5", f"{pos_share:.1f}%")

st.markdown("---")

SECTION_LABELS = {
    "overview": "Обзор",
    "drivers": "Драйверы",
    "offices": "Отделения",
    "responses": "Работа с отзывами",
    "cohorts": "Когорты",
    "reviews": "Отзывы",
}

selected_tab = st.radio(
    "Раздел",
    options=list(SECTION_LABELS.keys()),
    format_func=SECTION_LABELS.get,
    horizontal=True,
    label_visibility="collapsed",
)

if selected_tab == "overview":
    c1, c2 = st.columns([1, 1])

    with c1:
        st.subheader("Распределение рейтингов")
        rating_counts = dff['рейтинг'].value_counts().sort_index().reset_index()
        rating_counts.columns = ['rating', 'count']
        rating_counts['label'] = rating_counts['rating'].astype(str)
        rating_counts['pct'] = (rating_counts['count'] / rating_counts['count'].sum() * 100).round(1)
        rating_counts['text'] = rating_counts.apply(
            lambda r: f"<b>{r['count']:,}</b><br><span style='font-size:10px;color:#6B7A6F'>{r['pct']}%</span>",
            axis=1
        )
        fig = px.bar(
            rating_counts, x='label', y='count', text='text',
            color='label',
            color_discrete_map={
                '1': RATING_COLORS[1], '2': RATING_COLORS[2], '3': RATING_COLORS[3],
                '4': RATING_COLORS[4], '5': RATING_COLORS[5],
            },
        )
        fig.update_traces(
            textposition='outside',
            marker_line_width=0,
            marker_opacity=0.93,
            hovertemplate='<b>Рейтинг %{x}</b><br>Отзывов: %{y:,}<extra></extra>',
            cliponaxis=False
        )
        fig.update_layout(
            showlegend=False,
            xaxis_title="Рейтинг", yaxis_title="",
            height=380, margin=dict(t=40, b=20, l=10, r=10),
            yaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
        )
        apply_theme(fig)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.subheader("Соотношение тональности")
        neg = (dff['рейтинг'] <= 2).sum()
        neu = (dff['рейтинг'] == 3).sum()
        pos = (dff['рейтинг'] >= 4).sum()
        labels = ['Негатив (1–2)', 'Нейтрал (3)', 'Позитив (4–5)']
        values = [neg, neu, pos]
        colors = [NEG_COLOR, NEU_COLOR, POS_COLOR]

        # фильтрация: убираем категории с 0
        filtered = [(l, v, c) for l, v, c in zip(labels, values, colors) if v > 0]

        # распаковка обратно
        labels_f, values_f, colors_f = zip(*filtered) if filtered else ([], [], [])

        fig = go.Figure(data=[go.Pie(
            labels=labels_f,
            values=values_f,
            hole=0.62,
            marker=dict(colors=colors_f, line=dict(color='#FFFFFF', width=2)),
            textinfo='label+percent',
            textfont=dict(size=10, family="Inter"),
            hovertemplate='<b>%{label}</b><br>%{value:,} отзывов<br>%{percent}<extra></extra>',
        )])
        fig.update_layout(
            height=380, margin=dict(t=30, b=10),
            showlegend=False,
            annotations=[dict(
                text=f'<b style="font-size:26px;color:{FREEDOM_DARK}">{total_reviews:,}</b><br>'
                     f'<span style="color:#6B7A6F;font-size:13px">отзывов</span>',
                x=0.5, y=0.5, showarrow=False)
            ],
        )
        apply_theme(fig)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("##### Средняя длина отзывов по тональности")
    pos_df = dff[dff['рейтинг'] >= 4]
    neg_df = dff[dff['рейтинг'] <= 2]
    avg_len = pd.DataFrame({
        'Тональность': ['Негатив 1–2', 'Позитив 4–5'],
        'Средняя длина': [neg_df['длина_текста'].mean(), pos_df['длина_текста'].mean()],
    })
    fig = px.bar(
        avg_len,
        x='Тональность',
        y='Средняя длина',
        color='Тональность',
        color_discrete_map={'Негатив 1–2': NEG_COLOR, 'Позитив 4–5': POS_COLOR},
        text='Средняя длина',
        labels={'Средняя длина': 'Средняя длина, символов'},
    )
    fig.update_traces(
        texttemplate='%{text:.0f}',
        textposition='outside',
        marker_line_width=0,
        marker_opacity=0.92,
        width=0.45,
    )
    fig.update_layout(
        height=380,
        margin=dict(t=50, b=30, l=20, r=20),
        showlegend=False,
        yaxis=dict(range=[0, avg_len['Средняя длина'].max() * 1.18], automargin=True),
    )
    fig.update_yaxes(automargin=True)
    apply_theme(fig)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.subheader("Активность отзывов по дням недели и часам")
    c1, c2 = st.columns(2)

    with c1:
        dff['день_недели'] = dff['datetime'].dt.day_name()
        days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        days_ru = {'Monday': 'Пн', 'Tuesday': 'Вт', 'Wednesday': 'Ср', 'Thursday': 'Чт',
                   'Friday': 'Пт', 'Saturday': 'Сб', 'Sunday': 'Вс'}
        dow = dff.groupby('день_недели').agg(
            count=('рейтинг', 'count'),
            avg=('рейтинг', 'mean'),
        ).reindex(days_order).reset_index()
        dow['день'] = dow['день_недели'].map(days_ru)
        fig = px.bar(dow, x='день', y='count', color='avg',
                     color_continuous_scale=RATING_SCALE,
                     range_color=[1, 5],
                     labels={'count': 'Отзывов', 'день': '', 'avg': 'Ср. рейтинг'},
                     title="По дням недели")
        fig.update_traces(
            marker_line_width=0,
            opacity=0.92,
            hovertemplate='<b>%{x}</b><br>Отзывов: %{y}<br>Ср. рейтинг: %{marker.color:.2f}<extra></extra>'
        )
        fig.update_layout(
            height=340, margin=dict(t=50, b=20),
            title_font_size=14, title_font_color=FREEDOM_DARK,
            coloraxis_colorbar=dict(thickness=12, len=0.75, tickfont=dict(size=11)),
        )
        apply_theme(fig)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        dff['час'] = dff['datetime'].dt.hour
        hourly = dff.groupby('час').agg(
            count=('рейтинг', 'count'),
            avg=('рейтинг', 'mean'),
        ).reset_index()
        fig = px.bar(hourly, x='час', y='count', color='avg',
                     color_continuous_scale=RATING_SCALE,
                     range_color=[1, 5],
                     labels={'count': 'Отзывов', 'час': 'Час суток', 'avg': 'Ср. рейтинг'},
                     title="По часам")
        fig.update_traces(
            marker_line_width=0,
            opacity=0.92,
            hovertemplate='<b>%{x}:00</b><br>Отзывов: %{y}<br>Ср. рейтинг: %{marker.color:.2f}<extra></extra>'
        )
        fig.update_layout(
            height=340, margin=dict(t=50, b=20),
            title_font_size=14, title_font_color=FREEDOM_DARK,
            coloraxis_colorbar=dict(thickness=12, len=0.75, tickfont=dict(size=11)),
        )
        apply_theme(fig)
        st.plotly_chart(fig, use_container_width=True)

    # ── AI Insight: анализ активности по дню недели ──────────────
    st.markdown("")
    days_ru_full = {
        'Monday': 'Понедельник', 'Tuesday': 'Вторник', 'Wednesday': 'Среда',
        'Thursday': 'Четверг', 'Friday': 'Пятница', 'Saturday': 'Суббота', 'Sunday': 'Воскресенье',
    }
    days_en_list = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

    ai_col1, ai_col2 = st.columns([1, 3])
    with ai_col1:
        selected_day_ru = st.selectbox(
            "Выберите день недели",
            [days_ru_full[d] for d in days_en_list],
            key="ai_dow_select",
        )
    selected_day_en = [k for k, v in days_ru_full.items() if v == selected_day_ru][0]

    with ai_col2:
        st.markdown("")
        btn_dow = st.button("🤖 AI Insight: почему такая активность?", key="btn_dow_insight")

    if btn_dow:
        day_df = dff[dff['datetime'].dt.day_name() == selected_day_en].copy()

        if len(day_df) < 3:
            st.warning("Слишком мало отзывов за этот день для анализа.")
        else:
            # ── Собираем статистику ──
            total_reviews = len(day_df)
            avg_rating = day_df['рейтинг'].mean()
            neg_pct = (day_df['рейтинг'] <= 2).mean() * 100
            pos_pct = (day_df['рейтинг'] >= 4).mean() * 100

            # Топ-5 конкретных дат с наибольшей активностью
            date_counts = day_df.groupby(day_df['datetime'].dt.date).size().sort_values(ascending=False)
            top_dates = date_counts.head(5)
            top_dates_info = []
            for dt, cnt in top_dates.items():
                dt_reviews = day_df[day_df['datetime'].dt.date == dt]
                dt_avg = dt_reviews['рейтинг'].mean()
                top_dates_info.append(f"  {dt} — {cnt} отзывов, ср. рейтинг {dt_avg:.1f}")

            # Распределение по часам в этот день
            hour_dist = day_df.groupby(day_df['datetime'].dt.hour).size()
            peak_hour = hour_dist.idxmax() if len(hour_dist) > 0 else "—"

            # Сравнение с другими днями
            all_dow_counts = dff.groupby(dff['datetime'].dt.day_name()).size().reindex(days_en_list).fillna(0)
            day_rank = int((all_dow_counts >= all_dow_counts[selected_day_en]).sum())

            # Сэмпл отзывов
            sample_texts = day_df['текст'].dropna().astype(str).tolist()
            np.random.seed(42)
            if len(sample_texts) > 40:
                sample_texts = list(np.random.choice(sample_texts, 40, replace=False))
            reviews_block = "\n".join(['- ' + t[:200] for t in sample_texts])

            prompt = f"""
Ты senior CX аналитик Freedom Bank.

Проанализируй активность отзывов именно в ДЕНЬ НЕДЕЛИ: {selected_day_ru}.

---

СТАТИСТИКА ПО {selected_day_ru.upper()}:
- Всего отзывов за все {selected_day_ru}: {total_reviews}
- Средний рейтинг: {avg_rating:.2f}
- Позитивных (4-5): {pos_pct:.1f}%
- Негативных (1-2): {neg_pct:.1f}%
- Пик активности: {peak_hour}:00
- Место по активности среди всех дней: {day_rank}-е из 7

ДАТЫ С НАИБОЛЬШЕЙ АКТИВНОСТЬЮ (конкретные {selected_day_ru}):
{chr(10).join(top_dates_info)}

ПРИМЕРЫ ОТЗЫВОВ ЗА {selected_day_ru.upper()}:
{reviews_block}

---

ТРЕБОВАНИЯ:
- Объясни ПОЧЕМУ именно в {selected_day_ru} такая активность
- Есть ли всплески в конкретные даты? Что могло произойти?
- Связана ли активность с рабочим графиком банка / клиентов?
- Есть ли разница в тональности по сравнению с другими днями?
- НЕ используй HTML и markdown
- НЕ давай советы бизнесу
- НЕ делай предположений без явных данных
- Если нет фактов — пиши "нет подтверждения в данных"
- Каждое утверждение должно опираться на конкретную метрику (проценты, количество, сравнение)

---

ВЕРНИ СТРОГО JSON:
Начни ответ с {{ и закончи }}

{{
  "headline": "краткий вывод по дню (5-8 слов)",
  "why_this_day": "2-3 предложения: основная причина такой активности именно в этот день",
  "spike_dates": ["дата1: что случилось", "дата2: что случилось"],
  "hour_pattern": "1-2 предложения: почему пик в определённые часы",
  "sentiment_note": "1-2 предложения: тональность отзывов в этот день",
}}
"""
            with st.spinner(f"Анализирую {selected_day_ru}..."):
                result = call_groq(prompt, max_tokens=2000)

            dow_key = f"dow_insight_{selected_day_en}"
            if "error" in result:
                st.error(f"Ошибка: {result['error']}")
            else:
                st.session_state[dow_key] = result

    # Показываем результат если он в session_state
    dow_key = f"dow_insight_{selected_day_en}"
    if dow_key in st.session_state:
        r = st.session_state[dow_key]
        headline = r.get("headline", "")
        why = r.get("why_this_day", "")
        spikes = r.get("spike_dates", [])
        hour_pat = r.get("hour_pattern", "")
        sent_note = r.get("sentiment_note", "")
        insight = r.get("insight", "")

        spikes_html = "".join([
            f'<p style="margin:4px 0;font-size:0.9rem;">📅 {s}</p>'
            for s in spikes
        ]) if spikes else '<p style="margin:4px 0;font-size:0.9rem;color:#6B7A6F;">Явных всплесков не обнаружено</p>'

        html_block = f"""
        <style>
          .dow-ibox {{
            font-family: 'Inter', sans-serif;
            background: linear-gradient(135deg, #f0faf4 0%, #ffffff 100%);
            border: 1px solid #BCE8C9;
            border-radius: 16px;
            padding: 24px 28px;
            margin: 12px 0;
            box-shadow: 0 2px 12px rgba(8,89,50,0.08);
          }}
          .dow-headline {{
            font-size: 1.05rem;
            font-weight: 700;
            color: #085932;
            margin-bottom: 14px;
          }}
          .dow-sec {{
            font-size: 0.78rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.07em;
            color: #6B7A6F;
            margin: 16px 0 8px 0;
          }}
          .dow-text {{
            font-size: 0.93rem;
            color: #1a1a1a;
            line-height: 1.6;
          }}
          hr.dow-div {{
            border: none;
            border-top: 1px solid #D5E8DC;
            margin: 10px 0;
          }}
        </style>
        <div class="dow-ibox">
          <div class="dow-headline">🤖 {headline}</div>
          <hr class="dow-div">
          <div class="dow-sec">📌 Почему именно {selected_day_ru}?</div>
          <div class="dow-text">{why}</div>
          <div class="dow-sec">📅 Всплески по конкретным датам</div>
          <div class="dow-text">{spikes_html}</div>
          <div class="dow-sec">🕐 Часовой паттерн</div>
          <div class="dow-text">{hour_pat}</div>
          <div class="dow-sec">💬 Тональность</div>
          <div class="dow-text">{sent_note}</div>
          <div class="dow-sec">🧠 Общий вывод</div>
          <div class="dow-text" style="background:#fff;border-left:3px solid #08bb48;padding:12px 16px;border-radius:6px;margin-top:4px;">{insight}</div>
        </div>
        """
        try:
            import streamlit.components.v1 as components
            components.html(html_block, height=520, scrolling=True)
        except Exception:
            st.markdown(html_block, unsafe_allow_html=True)

    divider()
    st.markdown("### Влияет ли скорость ответа банка на рейтинг?")

    with_answer = dff[dff['часов_до_ответа'].notna()].copy()
    if len(with_answer) > 100:
        # Биннинг
        def speed_bucket(h):
            if h < 6: return '< 6 ч'
            if h < 24: return '6–24 ч'
            if h < 72: return '1–3 дня'
            if h < 168: return '3–7 дней'
            return '> 7 дней'

        with_answer['скорость'] = with_answer['часов_до_ответа'].apply(speed_bucket)
        buckets_order = ['< 6 ч', '6–24 ч', '1–3 дня', '3–7 дней', '> 7 дней']
        speed_agg = with_answer.groupby('скорость').agg(
            отзывов=('рейтинг', 'count'),
            ср_рейтинг=('рейтинг', 'mean'),
            нег=('рейтинг', lambda x: (x <= 2).mean() * 100),
            поз=('рейтинг', lambda x: (x >= 4).mean() * 100),
        ).reindex(buckets_order).reset_index()

        c1, c2 = st.columns([2, 3])

        # Быстрый vs медленный (SLA 24ч)
        fast = with_answer[with_answer['часов_до_ответа'] < 24]
        slow = with_answer[with_answer['часов_до_ответа'] >= 24]
        with c1:
            st.markdown("##### Быстро (< 24ч) vs Медленно (≥ 24ч)")
            cmp_df = pd.DataFrame({
                'Группа': ['меньше 24 ч', 'больше 24 ч'],
                'Отзывов': [len(fast), len(slow)],
                'Ср. рейтинг': [fast['рейтинг'].mean(), slow['рейтинг'].mean()],
                '% негатива': [(fast['рейтинг'] <= 2).mean() * 100, (slow['рейтинг'] <= 2).mean() * 100],
            })
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=cmp_df['Группа'], y=cmp_df['Ср. рейтинг'],
                marker_color=[FREEDOM_GREEN, FREEDOM_GREEN],
                marker_line_width=0,
                marker_opacity=0.92,
                width=0.45,
                text=cmp_df['Ср. рейтинг'].round(2),
                textposition='outside',
                textfont=dict(size=13),
                hovertemplate='<b>%{x}</b><br>Ср. рейтинг: %{y:.2f}<extra></extra>',
            ))
            fig.update_layout(height=320, yaxis_title="Средний рейтинг",
                              yaxis_range=[1, 5.5], margin=dict(t=30, b=20))
            apply_theme(fig)
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            st.markdown("##### Ср. рейтинг по времени ответа")
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=speed_agg['скорость'],
                y=speed_agg['ср_рейтинг'],
                marker=dict(
                    color=FREEDOM_GREEN,
                    line_width=0,
                    opacity=0.92,
                ),
                text=speed_agg['ср_рейтинг'].round(2),
                textposition='outside',
                textfont=dict(size=12),
                customdata=speed_agg['отзывов'],
                hovertemplate='<b>%{x}</b><br>Ср. рейтинг: %{y:.2f}<br>'
                            'Отзывов: %{customdata}<extra></extra>',
            ))
            fig.update_layout(
                height=320, yaxis_title="Ср. рейтинг",
                yaxis_range=[1, 5.5],
                margin=dict(t=30, b=20),
            )
            apply_theme(fig)
            st.plotly_chart(fig, use_container_width=True)

    divider() 

    st.subheader("Динамика по месяцам")

    dff['год_месяц'] = dff['datetime'].dt.to_period('M').dt.to_timestamp()
    monthly = dff.groupby('год_месяц').agg(
        кол_во=('рейтинг', 'count'),
        средний=('рейтинг', 'mean'),
        негатив=('рейтинг', lambda x: (x <= 2).sum()),
        позитив=('рейтинг', lambda x: (x >= 4).sum()),
    ).reset_index()

    c1, c2 = st.columns([3, 2])
    with c1:
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=monthly['год_месяц'], y=monthly['позитив'],
            name='Рейтинг 4–5', marker_color=POS_COLOR, opacity=0.88,
            marker_line_width=0,
            hovertemplate='<b>%{x|%b %Y}</b><br>Позитив: %{y}<extra></extra>',
        ))
        fig.add_trace(go.Bar(
            x=monthly['год_месяц'], y=monthly['негатив'],
            name='Рейтинг 1–2', marker_color=NEG_COLOR, opacity=0.88,
            marker_line_width=0,
            hovertemplate='<b>%{x|%b %Y}</b><br>Негатив: %{y}<extra></extra>',
        ))
        fig.update_layout(
            barmode='stack',
            bargap=0.25,
            height=440,
            xaxis_title="",
            yaxis=dict(title="Кол-во отзывов"),
            yaxis2=dict(title="Средний рейтинг", overlaying='y', side='right',
                        range=[1, 5.2], showgrid=False, tickfont=dict(size=12)),
            legend=dict(orientation='h', y=1.12, x=0, font=dict(size=12)),
            margin=dict(t=50, b=20),
            hovermode='x unified',
        )
        apply_theme(fig)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.markdown("##### Сводка по годам")
        yearly = dff.copy()
        yearly['год'] = yearly['datetime'].dt.year
        y_agg = yearly.groupby('год').agg(
            отзывов=('рейтинг', 'count'),
            средний=('рейтинг', 'mean'),
            доля_1_2=('рейтинг', lambda x: f"{(x <= 2).mean() * 100:.1f}%"),
            доля_5=('рейтинг', lambda x: f"{(x == 5).mean() * 100:.1f}%"),
        ).reset_index()
        y_agg['средний'] = y_agg['средний'].round(2)
        y_agg.columns = ['Год', 'Отзывов', 'Ср. рейтинг', 'Доля 1–2', 'Доля 5']
        st.dataframe(y_agg, hide_index=True, use_container_width=True)


    # =============================================================
    # Топ-10 самых залайканных отзывов выбранного месяца
    # =============================================================
    st.markdown("---")
    st.subheader("Топ-10 залайканных отзывов за месяц")

    # Селекторы года и месяца
    dff['_год'] = dff['datetime'].dt.year
    dff['_месяц'] = dff['datetime'].dt.month
    available_years = sorted(dff['_год'].unique(), reverse=True)

    months_ru = {
        1: 'Январь', 2: 'Февраль', 3: 'Март', 4: 'Апрель',
        5: 'Май', 6: 'Июнь', 7: 'Июль', 8: 'Август',
        9: 'Сентябрь', 10: 'Октябрь', 11: 'Ноябрь', 12: 'Декабрь',
    }

    sc1, sc2, _ = st.columns([1, 1, 3])
    selected_year = sc1.selectbox("Год", available_years, index=0)
    available_months = sorted(dff[dff['_год'] == selected_year]['_месяц'].unique())
    selected_month = sc2.selectbox(
        "Месяц",
        available_months,
        index=len(available_months) - 1,
        format_func=lambda m: months_ru[m],
    )
    month_df = dff[(dff['_год'] == selected_year) & (dff['_месяц'] == selected_month)].copy()

    MAX_REVIEWS = 50

    month_texts = (
        month_df['текст']
        .dropna()
        .astype(str)
        .sample(min(len(month_df), MAX_REVIEWS), random_state=42)
        .tolist()
    )
    period_label = f"{months_ru[selected_month]} {selected_year}"

    pos_pct = (month_df['рейтинг'] >= 4).mean() * 100
    neg_pct = (month_df['рейтинг'] <= 2).mean() * 100
    neu_pct = (month_df['рейтинг'] == 3).mean() * 100

    sent_summary = f"Позитив: {pos_pct:.1f}%, Негатив: {neg_pct:.1f}%, Нейтраль: {neu_pct:.1f}%"
    st.subheader(f"🧠 AI-инсайт — {period_label}")
    month_texts = (
        month_df['текст']
        .dropna()
        .astype(str)
        .tolist()
    )
    sent_summary = f"Позитивных: {pos_pct}%, Негативных: {neg_pct}%, Нейтральных: {neu_pct}%"

    month_key = f"month_insight_{selected_year}_{selected_month}"
    if st.button("🚀 Сгенерировать инсайт по периоду", key="btn_month_insight"):
        with st.spinner(f"Анализирую {period_label}..."):
            result = call_groq(format_month_prompt(
                period_label, month_texts, sent_summary
            ), max_tokens=2000)
        if "error" in result:
            st.error(f"Ошибка: {result['error']}")
        else:
            st.session_state[month_key] = result
    if month_key in st.session_state:
        result    = st.session_state[month_key]
        pos_html  = "".join([f'<p style="margin:6px 0;font-size:0.93rem;">✅ {x}</p>' for x in result.get("positive_highlights",[])])
        neg_html  = "".join([f'<p style="margin:6px 0;font-size:0.93rem;">❌ {x}</p>' for x in result.get("negative_highlights",[])])
        tags_html = "".join([
            f'<span style="display:inline-block;background:#E8F8EE;color:#085932;'
            f'border:1px solid #BCE8C9;border-radius:20px;padding:4px 12px;'
            f'margin:4px 4px 4px 0;font-size:0.82rem;font-weight:600;">{t}</span>'
            for t in result.get("hot_topics",[])
        ])
        headline  = result.get("headline","")
        insight   = result.get("insight","")
        html_block = f"""
        <style>
          .ibox {{
            font-family: 'Inter', sans-serif;
            background: linear-gradient(135deg, #f0faf4 0%, #ffffff 100%);
            border: 1px solid #BCE8C9;
            border-radius: 16px;
            padding: 24px 28px;
            margin: 12px 0;
            box-shadow: 0 2px 12px rgba(8,89,50,0.08);
          }}
          .ibox-headline {{
            font-size: 1.05rem;
            font-weight: 700;
            color: #085932;
            margin-bottom: 14px;
          }}
          .isec {{
            font-size: 0.78rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.07em;
            color: #6B7A6F;
            margin: 16px 0 8px 0;
          }}
          .itext {{
            font-size: 0.93rem;
            color: #1a1a1a;
            line-height: 1.6;
          }}
          hr.idivider {{
            border: none;
            border-top: 1px solid #D5E8DC;
            margin: 10px 0;
          }}
        </style>
        <div class="ibox">
          <div class="ibox-headline">📌 {headline}</div>
          <hr class="idivider">
          <div class="isec">🔥 Горячие темы</div>
          <div style="margin-bottom:4px;">{tags_html}</div>
          <div class="isec">💙 Что хвалили</div>
          <div class="itext">{pos_html}</div>
          <div class="isec">💔 На что жаловались</div>
          <div class="itext">{neg_html}</div>
          <div class="isec">🧠 Нарратив периода</div>
          <div class="itext" style="background:#fff;border-left:3px solid #08bb48;padding:12px 16px;border-radius:6px;margin-top:4px;">{insight}</div>
        </div>
        """
        try:
            import streamlit.components.v1 as components
            components.html(html_block, height=520, scrolling=True)
        except Exception:
            st.markdown(html_block, unsafe_allow_html=True)

    st.divider()

    # Фильтруем и сортируем по лайкам
    month_df = dff[(dff['_год'] == selected_year) & (dff['_месяц'] == selected_month)].copy()

    if len(month_df) == 0:
        st.info("За выбранный месяц отзывов в текущей выборке нет.")
    else:
        total_likes = int(month_df['likes_count'].sum())
        st.caption(
            f"В {months_ru[selected_month]} {selected_year} — **{len(month_df):,}** отзывов, "
            f"суммарно **{total_likes:,}** лайков. "
            f"Средний рейтинг: **{month_df['рейтинг'].mean():.2f}**."
        )

        top10 = month_df.nlargest(10, 'likes_count')

        for _, row in top10.iterrows():
            rating = int(row['рейтинг'])
            cls = "neg" if rating <= 2 else ("neu" if rating == 3 else "")
            answer_block = ""
            if pd.notna(row.get('ответ_текст')):
                ans = str(row['ответ_текст']).replace('\n', '<br>')
                answer_block = (
                    f"<div class='bank-answer'>"
                    f"<b>Ответ банка:</b><br>{ans[:400]}{'...' if len(ans) > 400 else ''}"
                    f"</div>"
                )
            text = str(row['текст']).replace('\n', '<br>') if pd.notna(row['текст']) else ''
            office_name = row['отделение'] if pd.notna(row['отделение']) and row['отделение'] != '—' else 'отделение не указано'
            office_part = f"<span style='color:#6B7A6F'>· {office_name}</span>"
            likes_badge = (
                f"<span style='background:{POS_COLOR};color:{FREEDOM_DARK};"
                f"padding:3px 10px;border-radius:12px;font-weight:700;font-size:0.85rem;'>"
                f"👍 {int(row['likes_count'])}</span>"
            )
            st.markdown(
                f"<div class='review-card {cls}'>"
                f"<div class='review-meta'>"
                f"{likes_badge}"
                f"{rating_badge(rating)}"
                f"<span>{row['datetime'].strftime('%d.%m.%Y')}</span>"
                f"<span>·</span>"
                f"<b style='color:{FREEDOM_DARK}'>{row['автор']}</b>"
                f"{office_part}"
                f"</div>"
                f"{text}"
                f"{answer_block}"
                f"</div>",
                unsafe_allow_html=True,
            )

    

# ------------------------------------------------------------
# TAB 3 — Драйверы (impact analysis)
# ------------------------------------------------------------
if selected_tab == "drivers":
    st.subheader("Что больше всего влияет на рейтинг?")

    overall_avg = dff['рейтинг'].mean()

    # --- 3.1 Влияние тем ---
    st.markdown("### Влияние тем на рейтинг")

    theme_impact_rows = []
    # Векторизованный расчёт: вместо цикла «per-theme .apply» делаем
    # explode + groupby — это на порядок быстрее на больших dff.
    themes_exploded = (
        dff[['темы', 'рейтинг']]
        .explode('темы')
        .dropna(subset=['темы'])
    )
    if len(themes_exploded):
        theme_agg = themes_exploded.groupby('темы').agg(
            Отзывов=('рейтинг', 'size'),
            ср_рейтинг=('рейтинг', 'mean'),
            neg=('рейтинг', lambda x: (x <= 2).mean() * 100),
            pos=('рейтинг', lambda x: (x >= 4).mean() * 100),
        ).reset_index()
        theme_agg = theme_agg[theme_agg['Отзывов'] >= 20]  # избегаем «шумных» тем
        for _, r in theme_agg.iterrows():
            theme_impact_rows.append({
                'Тема': r['темы'],
                'Отзывов': int(r['Отзывов']),
                'Ср. рейтинг': r['ср_рейтинг'],
                'Uplift': round(r['ср_рейтинг'] - overall_avg, 2),
                '% негатива': r['neg'],
                '% позитива': r['pos'],
            })

    impact_df = pd.DataFrame(theme_impact_rows)
    if impact_df.empty:
        st.info(
            "Недостаточно данных для анализа влияния тем: ни одна тема не набрала "
            "20+ отзывов в текущей выборке. Попробуйте расширить фильтр по городу или периоду."
        )
    else:
        impact_df = impact_df.sort_values('Uplift')

        c1, c2 = st.columns([3, 2])
        with c1:
            # Горизонтальный бар-чарт: отклонение от общего среднего
            impact_df['color'] = impact_df['Uplift'].apply(
                lambda x: NEG_COLOR if x < -0.3 else (POS_COLOR if x > 0.3 else NEU_COLOR)
            )
            fig = go.Figure(go.Bar(
                y=impact_df['Тема'],
                x=impact_df['Uplift'],
                orientation='h',
                marker=dict(
                    color=impact_df['color'],
                    line_width=0,
                    opacity=0.92,
                ),
                text=impact_df['Uplift'].apply(lambda x: f"{x:+.2f}"),
                textposition='outside',
                textfont=dict(size=12, family="Inter"),
                customdata=impact_df[['Ср. рейтинг', 'Отзывов']].values,
                hovertemplate='<b>%{y}</b><br>Ср. рейтинг: %{customdata[0]:.2f}<br>'
                              'Отзывов: %{customdata[1]}<br>Uplift: %{x:+.2f}<extra></extra>',
            ))
            fig.add_vline(x=0, line_width=2, line_color=FREEDOM_DARK, line_dash="solid",
                          annotation_text=f"Среднее {overall_avg:.2f}",
                          annotation_position="top",
                          annotation_font=dict(size=11, color=FREEDOM_DARK))
            fig.update_layout(
                height=max(420, len(impact_df) * 36 + 100),
                yaxis_title="",
                yaxis=dict(automargin=True),
                margin=dict(t=30, b=20, l=30, r=100),
            )
            fig.update_traces(cliponaxis=False)
            apply_theme(fig)
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            st.markdown("##### Топ-5 «токсичных» тем")
            toxic = impact_df.sort_values('Ср. рейтинг').head(5)
            toxic_disp = toxic[['Тема', 'Отзывов', 'Ср. рейтинг', '% негатива']].copy()
            toxic_disp['Ср. рейтинг'] = toxic_disp['Ср. рейтинг'].round(2)
            toxic_disp['% негатива'] = toxic_disp['% негатива'].round(1).astype(str) + '%'
            st.dataframe(toxic_disp, hide_index=True, use_container_width=True)

            st.markdown("##### Топ-5 «драйверов роста»")
            growth = impact_df.sort_values('Ср. рейтинг', ascending=False).head(5)
            growth_disp = growth[['Тема', 'Отзывов', 'Ср. рейтинг', '% позитива']].copy()
            growth_disp['Ср. рейтинг'] = growth_disp['Ср. рейтинг'].round(2)
            growth_disp['% позитива'] = growth_disp['% позитива'].round(1).astype(str) + '%'
            st.dataframe(growth_disp, hide_index=True, use_container_width=True)

    st.markdown("---")

    # --- 3.3 Комбо-темы ---
    st.markdown("### 🔗 Комбо-темы: какие сочетания дают самый низкий рейтинг?")

    from itertools import combinations

    # Векторизованно: zip по двум колонкам — в разы быстрее, чем iterrows.
    combo_rows = []
    for themes_list, rating in zip(dff['темы'].tolist(), dff['рейтинг'].tolist()):
        if not _is_theme_list(themes_list) or len(themes_list) < 2:
            continue
        themes_clean = [t for t in themes_list if t]
        if len(themes_clean) < 2:
            continue
        for t1, t2 in combinations(sorted(themes_clean), 2):
            combo_rows.append((f"{t1}  +  {t2}", rating))

    if combo_rows:
        combo_df_raw = pd.DataFrame(combo_rows, columns=['combo', 'рейтинг'])
        combo_agg = (
            combo_df_raw.groupby('combo')
            .agg(
                Отзывов=('рейтинг', 'count'),
                Ср_рейтинг=('рейтинг', 'mean'),
                Pct_neg=('рейтинг', lambda x: (x <= 2).mean() * 100),
            )
            .reset_index()
            .rename(columns={'combo': 'Комбинация тем', 'Ср_рейтинг': 'Ср. рейтинг', 'Pct_neg': '% негатива'})
        )
        # Только комбо с достаточным кол-вом отзывов
        combo_agg = combo_agg[combo_agg['Отзывов'] >= 15].copy()
        combo_agg['Uplift'] = (combo_agg['Ср. рейтинг'] - overall_avg).round(2)

        if len(combo_agg) > 0:
            c1, c2 = st.columns([3, 2])

            with c1:
                st.markdown("##### Топ-15 самых «токсичных» комбинаций")
                worst_combos = (
                    combo_agg[combo_agg['Uplift'] < -0.1]  # фильтр негатива
                    .sort_values('Uplift')                 # от худшего к лучшему
                    .head(15)
                )
                worst_combos['color'] = worst_combos['Uplift'].apply(
                    lambda x: NEG_COLOR if x < -0.3 else (POS_COLOR if x > 0.3 else NEU_COLOR)
                )
                fig = go.Figure(go.Bar(
                    y=worst_combos['Комбинация тем'],
                    x=worst_combos['Uplift'],
                    orientation='h',
                    marker=dict(
                        color=worst_combos['color'],
                        line_width=0,
                        opacity=0.92,
                    ),
                    text=worst_combos['Uplift'].apply(lambda x: f"{x:+.2f}"),
                    textposition='inside',
                    textfont=dict(size=12),
                    cliponaxis=False,
                    customdata=worst_combos[['Ср. рейтинг', 'Отзывов', '% негатива']].values,
                    hovertemplate=(
                        '<b>%{y}</b><br>'
                        'Ср. рейтинг: %{customdata[0]:.2f}<br>'
                        'Отзывов: %{customdata[1]}<br>'
                        '% негатива: %{customdata[2]:.1f}%<br>'
                        'Uplift: %{x:+.2f}<extra></extra>'
                    ),
                ))
                fig.add_vline(x=0, line_width=2, line_color=FREEDOM_DARK, line_dash="solid",
                              annotation_text=f"Среднее {overall_avg:.2f}",
                              annotation_position="top",
                              annotation_font=dict(size=11, color=FREEDOM_DARK))
                fig.update_layout(
                    height=max(380, len(worst_combos) * 42 + 100),
                    yaxis_title="",
                    yaxis=dict(automargin=True),
                    margin=dict(t=30, b=20, l=10, r=60),
                )
                apply_theme(fig)
                st.plotly_chart(fig, use_container_width=True)

            with c2:
                st.markdown("##### Полная таблица комбо-тем")
                combo_table = combo_agg.sort_values('Ср. рейтинг').head(30).copy()
                combo_table['Ср. рейтинг'] = combo_table['Ср. рейтинг'].round(2)
                combo_table['Uplift'] = pd.to_numeric(combo_table['Uplift'], errors='coerce')
                combo_table['Uplift'] = combo_table['Uplift'].round(2).apply(lambda x: f"{x:+.2f}")
                combo_table['% негатива'] = combo_table['% негатива'].round(1).astype(str) + '%'
                st.dataframe(
                    combo_table[['Комбинация тем', 'Отзывов', 'Ср. рейтинг', 'Uplift', '% негатива']],
                    hide_index=True,
                    use_container_width=True,
                    height=min(560, max(200, len(combo_table) * 36 + 40)),
                )

                # Тепловая карта топ-тем между собой
                st.markdown("##### Тепловая карта: ср. рейтинг пар тем")
                # Берём топ-10 тем по частоте
                top_themes_for_heat = [t for t, _ in Counter(
                    t for themes in dff['темы'] for t in themes
                ).most_common(10)]

                # Векторизуем: собираем булеву матрицу «отзыв × тема» одним проходом,
                # потом пересечения — это просто AND двух столбцов.
                themes_as_sets = dff['темы'].apply(
                    lambda lst: set(lst) if _is_theme_list(lst) else set()
                )
                theme_bool = pd.DataFrame(
                    {t: themes_as_sets.apply(lambda s, t=t: t in s) for t in top_themes_for_heat},
                    index=dff.index,
                )
                ratings_arr = dff['рейтинг']

                heat_data = {}
                for t1 in top_themes_for_heat:
                    heat_data[t1] = {}
                    col1 = theme_bool[t1]
                    for t2 in top_themes_for_heat:
                        if t1 == t2:
                            heat_data[t1][t2] = ratings_arr[col1].mean() if col1.any() else None
                        else:
                            mask = col1 & theme_bool[t2]
                            n = int(mask.sum())
                            heat_data[t1][t2] = ratings_arr[mask].mean() if n >= 5 else None

                heat_df = pd.DataFrame(heat_data).T.reindex(
                    index=top_themes_for_heat, columns=top_themes_for_heat
                )
                # Сокращаем длинные названия
                short_names = {t: (t[:20] + '…' if len(t) > 20 else t) for t in top_themes_for_heat}
                heat_df_disp = heat_df.rename(index=short_names, columns=short_names)

                fig_heat = px.imshow(
                    heat_df_disp,
                    color_continuous_scale=RATING_SCALE,
                    zmin=1, zmax=5,
                    text_auto='.2f',
                    aspect='auto',
                    labels=dict(color='Ср. рейтинг'),
                )
                fig_heat.update_layout(
                    height=380,
                    margin=dict(t=20, b=10, l=10, r=20),
                    coloraxis_colorbar=dict(title='Рейтинг', len=0.7),
                    xaxis=dict(tickangle=-40, automargin=True),
                    yaxis=dict(automargin=True),
                )
                fig_heat.update_traces(
                    hovertemplate='<b>%{y}</b> × <b>%{x}</b><br>Ср. рейтинг: %{z:.2f}<extra></extra>'
                )
                apply_theme(fig_heat)
                st.plotly_chart(fig_heat, use_container_width=True)
        else:
            st.info("Недостаточно комбинаций тем для анализа (нужно ≥15 совместных упоминаний).")
    else:
        st.info("В отзывах не найдено пар тем для анализа.")

    st.markdown("---")

    # --- 3.4 Динамика Uplift по кварталам ---
    st.markdown("### 📈 Динамика влияния тем: как менялся Uplift по кварталам?")

    dff_q = dff.copy()
    dff_q['квартал'] = dff_q['datetime'].dt.to_period('Q').astype(str)
    quarters_sorted = sorted(dff_q['квартал'].unique())

    # Берём только кварталы с ≥30 отзывами
    q_counts = dff_q.groupby('квартал').size()
    valid_quarters = q_counts[q_counts >= 30].index.tolist()
    valid_quarters = [q for q in quarters_sorted if q in valid_quarters]

    # Берём топ-10 тем по общей встречаемости для динамики (исключаем шум)
    top_themes_dynamic = [t for t, _ in Counter(
        t for themes in dff_q['темы'] for t in themes
    ).most_common(10)]

    if len(valid_quarters) >= 3 and top_themes_dynamic:
        # Векторизованно через explode+groupby: один проход, вместо
        # двойного цикла `per-quarter × per-theme apply`.
        q_sub = dff_q[dff_q['квартал'].isin(valid_quarters)][['квартал', 'темы', 'рейтинг']]
        q_exp = q_sub.explode('темы').dropna(subset=['темы'])
        q_exp = q_exp[q_exp['темы'].isin(top_themes_dynamic)]

        # Средний рейтинг по кварталу (для расчёта Uplift)
        q_avg_map = dff_q.groupby('квартал')['рейтинг'].mean().to_dict()

        uplift_grp = q_exp.groupby(['квартал', 'темы']).agg(
            Отзывов=('рейтинг', 'size'),
            ср_рейтинг=('рейтинг', 'mean'),
        ).reset_index()
        uplift_grp = uplift_grp[uplift_grp['Отзывов'] >= 10]

        uplift_dynamic_rows = []
        for _, r in uplift_grp.iterrows():
            q = r['квартал']
            uplift_dynamic_rows.append({
                'Квартал': q,
                'Тема': r['темы'],
                'Uplift': round(r['ср_рейтинг'] - q_avg_map.get(q, 0), 2),
                'Ср. рейтинг': r['ср_рейтинг'],
                'Отзывов': int(r['Отзывов']),
            })

        if uplift_dynamic_rows:
            dyn_df = pd.DataFrame(uplift_dynamic_rows)

            # Выбор тем для отображения
            theme_select_options = top_themes_dynamic
            selected_themes_dyn = st.multiselect(
                "Темы для отображения",
                options=theme_select_options,
                default=theme_select_options[:6],
                key='dyn_uplift_themes',
            )

            if selected_themes_dyn:
                dyn_filtered = dyn_df[dyn_df['Тема'].isin(selected_themes_dyn)]

                # Линейный график динамики Uplift
                fig_dyn = go.Figure()
                palette = [
                    FREEDOM_GREEN, NEG_COLOR, FREEDOM_DARK, NEU_COLOR, FREEDOM_MID,
                    '#7B61FF', '#FF8C42', '#2EC4B6', '#E84393', '#6B7A6F',
                ]
                for i, theme_name in enumerate(selected_themes_dyn):
                    t_df = dyn_filtered[dyn_filtered['Тема'] == theme_name].sort_values('Квартал')
                    if len(t_df) < 2:
                        continue
                    color = palette[i % len(palette)]
                    fig_dyn.add_trace(go.Scatter(
                        x=t_df['Квартал'],
                        y=t_df['Uplift'],
                        mode='lines+markers',
                        name=theme_name,
                        line=dict(color=color, width=2.5),
                        marker=dict(size=7, color=color),
                        customdata=t_df[['Ср. рейтинг', 'Отзывов']].values,
                        hovertemplate=(
                            '<b>%{x}</b> · ' + theme_name + '<br>'
                            'Uplift: %{y:.2f}<br>'
                            'Ср. рейтинг: %{customdata[0]:.2f}<br>'
                            'Отзывов: %{customdata[1]}<extra></extra>'
                        ),
                    ))

                fig_dyn.add_hline(y=0, line_color=FREEDOM_DARK, line_width=1.5, line_dash='dash')
                fig_dyn.update_layout(
                    height=440,
                    xaxis_title="Квартал",
                    yaxis_title="Uplift (отклонение от среднего квартала)",
                    legend=dict(
                        orientation='h',
                        y=-0.22, x=0,
                        font=dict(size=11),
                    ),
                    margin=dict(t=20, b=80, l=10, r=20),
                    hovermode='x unified',
                )
                apply_theme(fig_dyn)
                st.plotly_chart(fig_dyn, use_container_width=True)
            else:
                st.info("Выберите хотя бы одну тему для отображения.")
        else:
            st.info("Недостаточно данных для динамического анализа по кварталам.")
    else:
        st.info(
            f"Для динамики нужно ≥3 кварталов с ≥30 отзывами. "
            f"Найдено подходящих кварталов: {len(valid_quarters)}."
        )
    st.markdown("---")
    st.markdown("### 🔍 Отзывы по теме")
    # Формируем список тем отсортированный по частоте встречаемости
    all_theme_counts = Counter(
        t for themes in dff['темы'] for t in themes
    )
    theme_options = ['— выберите тему —'] + [
        f"{t}  ({all_theme_counts[t]})" for t, _ in all_theme_counts.most_common()
        if all_theme_counts[t] > 0
    ]

    col_sel, col_sent = st.columns([3, 2])
    with col_sel:
        chosen_raw = st.selectbox(
            "Тема", theme_options, key='theme_search_select'
        )
    with col_sent:
        sent_filter = st.radio(
            "Тональность", ['Все', 'Негатив (1–2)', 'Нейтрал (3)', 'Позитив (4–5)'],
            horizontal=True, key='theme_search_sentiment'
        )

    if chosen_raw != '— выберите тему —':
        # Извлекаем название темы (до двойного пробела перед счётчиком)
        chosen_theme = chosen_raw.split('  (')[0]

        # Считаем маску один раз, переиспользуем для всех метрик ниже.
        theme_mask = dff['темы'].apply(
            lambda ts: chosen_theme in ts if _is_theme_list(ts) else False
        )
        theme_subset = dff[theme_mask]

        theme_reviews = theme_subset.copy()

        if sent_filter == 'Негатив (1–2)':
            theme_reviews = theme_reviews[theme_reviews['рейтинг'] <= 2]
        elif sent_filter == 'Нейтрал (3)':
            theme_reviews = theme_reviews[theme_reviews['рейтинг'] == 3]
        elif sent_filter == 'Позитив (4–5)':
            theme_reviews = theme_reviews[theme_reviews['рейтинг'] >= 4]

        theme_reviews = theme_reviews.sort_values('datetime', ascending=False).head(15)

        if len(theme_reviews) == 0:
            st.info("Нет отзывов с выбранной темой и тональностью.")
        else:
            # Мини-KPI строка (используем заранее посчитанный theme_subset)
            total_in_theme = len(theme_subset)
            neg_in_theme   = int((theme_subset['рейтинг'] <= 2).sum())
            avg_in_theme   = theme_subset['рейтинг'].mean()

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Всего отзывов с темой", f"{total_in_theme:,}")
            m2.metric("Из них негативных",     f"{neg_in_theme:,}")
            m3.metric("Доля негатива",
                      f"{neg_in_theme / total_in_theme * 100:.1f}%" if total_in_theme else "—")
            m4.metric("Ср. рейтинг по теме",   f"{avg_in_theme:.2f}" if total_in_theme else "—")

            st.markdown(f"**Показаны последние {len(theme_reviews)} из {total_in_theme:,}**")

            for _, row in theme_reviews.iterrows():
                r = int(row['рейтинг'])
                if r <= 2:
                    card_cls, icon = 'neg', '🔴'
                elif r == 3:
                    card_cls, icon = 'neu', '🟡'
                else:
                    card_cls, icon = '',   '🟢'

                other_themes = [t for t in row['темы'] if t != chosen_theme]
                other_str = (
                    '  ·  также: <i>' + ', '.join(other_themes[:3]) + '</i>'
                    if other_themes else ''
                )
                office_str = (
                    f"  ·  🏢 {row['отделение'][:35]}"
                    if row.get('отделение', '—') not in ('—', '', None) else ''
                )
                date_str = str(row['дата'])[:10]

                # Подсвечиваем ключевые слова темы в тексте
                review_text = str(row['текст']) if pd.notna(row['текст']) else ''
                keywords_for_theme = THEMES.get(chosen_theme, [])
                highlighted = review_text[:500]
                for kw in sorted(keywords_for_theme, key=len, reverse=True):
                    if kw in highlighted.lower():
                        # case-insensitive replace with highlight span
                        import re as _re
                        pattern = _re.compile(_re.escape(kw), _re.IGNORECASE)
                        highlighted = pattern.sub(
                            lambda m: f"<mark style='background:#FFF176;border-radius:3px;"
                                      f"padding:0 2px'>{m.group()}</mark>",
                            highlighted, count=3
                        )
                        break  # одного подсвета достаточно

                answer_block = ''
                if row.get('есть_ответ') and pd.notna(row.get('ответ_текст')):
                    ans_preview = str(row['ответ_текст'])[:180]
                    answer_block = (
                        f"<div class='bank-answer'>"
                        f"<b>Ответ банка:</b> {ans_preview}"
                        f"{'…' if len(str(row['ответ_текст'])) > 180 else ''}"
                        f"</div>"
                    )

                st.markdown(
                    f"<div class='review-card {card_cls}'>"
                    f"<div class='review-meta'>"
                    f"  {rating_badge(r)} &nbsp;{icon}&nbsp;"
                    f"  <b>{date_str}</b>"
                    f"  {office_str}"
                    f"  {other_str}"
                    f"</div>"
                    f"{highlighted}"
                    f"{answer_block}"
                    f"</div>",
                    unsafe_allow_html=True,
                )

if selected_tab == "offices":
    st.subheader("Сравнение отделений")

    # Если в данных нет отделений (все '—'), подставляем «Freedom офис»,
    # чтобы вся аналитика (сотрудники, жалобы, дрилл-даун) работала как обычно.
    _dff_off = dff.copy()
    _has_named_offices = (_dff_off['отделение'].notna() & (_dff_off['отделение'] != '—')).any()
    if not _has_named_offices:
        _dff_off['отделение'] = 'Freedom офис'

    off = _dff_off[_dff_off['отделение'].notna() & (_dff_off['отделение'] != '—')].groupby('отделение').agg(
        отзывов=('рейтинг', 'count'),
        средний=('рейтинг', 'mean'),
        негатив_пр=('рейтинг', lambda x: (x <= 2).mean() * 100),
        позитив_пр=('рейтинг', lambda x: (x >= 4).mean() * 100),
        медиана_ответа_ч=('часов_до_ответа', 'median'),
    ).round(2).sort_values('отзывов', ascending=False).reset_index()

    office_options = ['Все отделения'] + off['отделение'].tolist()
    selected_office = st.selectbox("Выберите отделение", office_options,
                                   key='office_main_select')

    office_reviews = _dff_off if selected_office == 'Все отделения' else _dff_off[_dff_off['отделение'] == selected_office]
    st.dataframe(
        off.rename(columns={
            'отзывов': 'Отзывов', 'средний': 'Ср. рейтинг',
            'негатив_пр': '% рейтинг 1–2', 'позитив_пр': '% рейтинг 4–5',
            'медиана_ответа_ч': 'Медиана ответа (ч)', 'отделение': 'Отделение'}),
        use_container_width=True, hide_index=True,
    )

    st.markdown("##### Топ сотрудников в отделении")

    if len(office_reviews) == 0:
        st.warning("Для выбранного отделения нет доступных отзывов.")
    elif 'люди' not in office_reviews.columns:
        st.info("Колонка «люди» не найдена в данных.")
    else:
        # Очистка колонки люди
        office_reviews = office_reviews.copy()
        office_reviews['_person'] = office_reviews['люди'].apply(_clean_person)
        named = office_reviews[office_reviews['_person'].notna()]

        if named.empty:
            st.info("В этом отделении имена сотрудников пока не упоминаются.")
        else:
            # Агрегируем: упоминания + средний рейтинг + доля негатива
            top5 = (
                named.groupby('_person')
                .agg(
                    Упоминаний=('рейтинг', 'count'),
                    Ср_рейтинг=('рейтинг', 'mean'),
                    Негатив=('рейтинг', lambda x: (x <= 2).sum()),
                )
                .reset_index()
                .rename(columns={'_person': 'Сотрудник'})
                .sort_values('Упоминаний', ascending=False)
                .head(15)
            )
            top5['Ср_рейтинг'] = top5['Ср_рейтинг'].round(2)
            top5['%_негатива'] = (top5['Негатив'] / top5['Упоминаний'] * 100).round(1)

            c_chart, c_table = st.columns([3, 2])

            with c_chart:
                top5_plot = top5.sort_values('Упоминаний', ascending=True)
                # Цвет бара = средний рейтинг (1=красный → 5=зелёный)
                bar_colors = top5_plot['Ср_рейтинг'].apply(
                    lambda r: RATING_COLORS.get(round(r), FREEDOM_GREEN)
                ).tolist()

                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=top5_plot['Упоминаний'],
                    y=top5_plot['Сотрудник'],
                    orientation='h',
                    marker=dict(color=bar_colors, line_width=0),
                    text=top5_plot.apply(
                        lambda r: f"<b>{r['Упоминаний']}</b>  ★{r['Ср_рейтинг']:.2f}", axis=1
                    ),
                    textposition='outside',
                    cliponaxis=False,
                    customdata=top5_plot[['Ср_рейтинг', '%_негатива', 'Негатив']].values,
                    hovertemplate=(
                        '<b>%{y}</b><br>'
                        'Упоминаний: %{x}<br>'
                        'Ср. рейтинг: %{customdata[0]:.2f}<br>'
                        'Негативных: %{customdata[2]} (%{customdata[1]:.1f}%)'
                        '<extra></extra>'
                    ),
                ))
                fig.update_layout(
                    height=max(200, len(top5_plot) * 58),
                    margin=dict(t=20, b=10, l=10, r=130),
                    showlegend=False,
                    xaxis=dict(automargin=True),
                    yaxis=dict(automargin=True),
                )
                fig.update_traces(cliponaxis=False)
                apply_theme(fig)
                st.plotly_chart(fig, use_container_width=True)

            with c_table:
                # Цветная таблица с эмодзи-индикатором
                def _row_indicator(row):
                    if row['Ср_рейтинг'] >= 4.5:
                        mood = '🌟'
                    elif row['Ср_рейтинг'] >= 3.5:
                        mood = '🟢'
                    elif row['Ср_рейтинг'] >= 2.5:
                        mood = '🟡'
                    else:
                        mood = '🔴'
                    return mood

                top5_disp = top5.copy()
                top5_disp[''] = top5_disp.apply(_row_indicator, axis=1)
                top5_disp = top5_disp[['', 'Сотрудник', 'Упоминаний',
                                        'Ср_рейтинг', '%_негатива']]
                top5_disp.columns = ['', 'Сотрудник', 'Упом.', 'Ср. ★', '% neg']
                st.dataframe(top5_disp, hide_index=True, use_container_width=True,
                             height=min(220, len(top5_disp) * 38 + 40))

    if len(off) >= 2:
        
        c1, c2 = st.columns(2)
        off['отделение_short'] = off['отделение'].apply(
            lambda x: x if len(x) <= 30 else x[:30] + "..."
        )
        with c1:
            fig = px.bar(off, x='средний', y='отделение_short', orientation='h',
                         color='средний',
                         color_continuous_scale=RATING_SCALE,
                         range_color=[1, 5],
                         title="Средний рейтинг по отделению",
                         text='средний')
            fig.update_traces(texttemplate='%{text:.2f}', textposition='outside',
                              marker_line_width=0)
            fig.update_layout(height=400,
                              yaxis=dict(categoryorder='total ascending', automargin=True),
                              margin=dict(t=60, b=20, l=10, r=120),
                              title_font_size=14, title_font_color=FREEDOM_DARK,
                              xaxis_title="", yaxis_title="")
            fig.update_layout(
                xaxis=dict(range=[0, off['средний'].max() + 1])
            )
            apply_theme(fig)
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            fig = px.bar(off, x='отзывов', y='отделение_short', orientation='h',
                         color_discrete_sequence=[FREEDOM_GREEN],
                         title="Количество отзывов по отделению",
                         text='отзывов')
            fig.update_traces(textposition='outside', cliponaxis=False, marker_line_width=0)
            fig.update_layout(height=400,
                              yaxis=dict(categoryorder='total ascending', automargin=True),
                              margin=dict(t=60, b=20, l=10, r=180),
                              xaxis=dict(range=[0, off['отзывов'].max() * 1.2], automargin=True),
                              title_font_size=14, title_font_color=FREEDOM_DARK,
                              xaxis_title="", yaxis_title="")
            apply_theme(fig)
            st.plotly_chart(fig, use_container_width=True)

    # --- Нормализованное сравнение: отклонение от общего среднего ---
    st.markdown("---")
    if(len(off)>1):
        st.markdown("### Отделения относительно среднего по банку")
        bank_avg = dff['рейтинг'].mean()
        bank_neg = (dff['рейтинг'] <= 2).mean() * 100

        off['Δ_рейтинг'] = (off['средний'] - bank_avg).round(2)
        off['Δ_негатив'] = (off['негатив_пр'] - bank_neg).round(1)
        off['статус'] = off['Δ_рейтинг'].apply(
            lambda x: '🟢 Лидер' if x > 0.15 else ('🔴 Аутсайдер' if x < -0.15 else '🟡 Норма')
        )

        fig = go.Figure()
        off_sorted = off.sort_values('Δ_рейтинг')
        colors = off_sorted['Δ_рейтинг'].apply(
            lambda x: POS_COLOR if x > 0.15 else (NEG_COLOR if x < -0.15 else NEU_COLOR)
        )
        fig.add_trace(go.Bar(
            y=off_sorted['отделение'], x=off_sorted['Δ_рейтинг'],
            orientation='h',
            marker=dict(color=colors, line_width=0),
            text=off_sorted['Δ_рейтинг'].apply(lambda x: f"{x:+.2f}"),
            textposition='outside',
            cliponaxis=False,
            customdata=off_sorted[['средний', 'отзывов']].values,
            hovertemplate='<b>%{y}</b><br>Δ от среднего: %{x:+.2f}<br>'
                            'Ср. рейтинг: %{customdata[0]:.2f}<br>'
                            'Отзывов: %{customdata[1]}<extra></extra>',
        ))
        fig.add_vline(x=0, line_color=FREEDOM_DARK, line_width=2)
        fig.update_layout(
            height=380,
            xaxis_title=f"Отклонение от среднего банка ({bank_avg:.2f})",
            yaxis_title="",
            margin=dict(t=60, b=20, l=10, r=120),
        )
        fig.update_yaxes(automargin=True)
        apply_theme(fig)
        st.plotly_chart(fig, use_container_width=True)
    # -------------------------------------------------------
    # Топ жалоб
    # -------------------------------------------------------
    st.markdown("---")

    # Исключаем позитивные темы из анализа жалоб
    _POS_THEMES = {'Благодарность сотруднику', 'Быстрое обслуживание', 'Рекомендация другим'}

    _has_multi_offices = len(off) >= 2

    if _has_multi_offices:
        st.markdown("### Топ жалоб по отделениям")
    else:
        st.markdown("### Топ жалоб")

    # _dff_off уже содержит «Freedom офис» вместо «—», поэтому единый фильтр
    neg_by_office = _dff_off[
        (_dff_off['рейтинг'] <= 2) &
        _dff_off['отделение'].notna() &
        (_dff_off['отделение'] != '—')
    ].copy()

    if len(neg_by_office) < 10:
        st.info("Недостаточно негативных отзывов для анализа жалоб.")
    else:
        if _has_multi_offices:
            # === Блок: разбивка по отделениям ===
            neg_counts = (
                neg_by_office.groupby('отделение').size()
                .reset_index(name='негативных')
                .merge(off[['отделение', 'отзывов', 'негатив_пр']], on='отделение', how='left')
                .sort_values('негативных', ascending=False)
            )
            neg_counts['отделение_short'] = neg_counts['отделение'].apply(
                lambda x: x if len(x) <= 35 else x[:35] + '…'
            )

            # --- Бар-чарт: кол-во негативных отзывов ---
            c1, c2 = st.columns([2, 3])

            with c1:
                st.markdown("##### Кол-во негативных отзывов (рейтинг 1–2)")
                top15 = neg_counts.head(15).sort_values('негативных')
                fig = px.bar(
                    top15, x='негативных', y='отделение_short',
                    orientation='h',
                    color='негативных',
                    color_continuous_scale=[FREEDOM_SOFT, NEG_COLOR],
                    text='негативных',
                    custom_data=['негатив_пр'],
                )
                fig.update_traces(
                    textposition='outside', cliponaxis=False,
                    marker_line_width=0,
                    hovertemplate=(
                        '<b>%{y}</b><br>'
                        'Негативных: %{x}<br>'
                        'Доля негатива: %{customdata[0]:.1f}%<extra></extra>'
                    ),
                )
                fig.update_layout(
                    height=max(300, len(top15) * 35),
                    margin=dict(t=20, b=10, l=10, r=70),
                    showlegend=False, coloraxis_showscale=False,
                    yaxis=dict(automargin=True),
                )
                apply_theme(fig)
                st.plotly_chart(fig, use_container_width=True)

            # --- Таблица: топ-3 темы по каждому отделению ---
            with c2:
                st.markdown("##### Топ-3 темы жалоб в каждом отделении")
                office_theme_rows = []
                for office, grp in neg_by_office.groupby('отделение'):
                    cnt = Counter(
                        t for themes in grp['темы'] for t in themes
                        if t not in _POS_THEMES
                    )
                    top3 = cnt.most_common(3)
                    office_theme_rows.append({
                        'Отделение': office[:38],
                        'Жалоб': len(grp),
                        'Тема №1': top3[0][0] if len(top3) > 0 else '—',
                        'Тема №2': top3[1][0] if len(top3) > 1 else '—',
                        'Тема №3': top3[2][0] if len(top3) > 2 else '—',
                    })
                if office_theme_rows:
                    ot_df = (
                        pd.DataFrame(office_theme_rows)
                        .sort_values('Жалоб', ascending=False)
                        .reset_index(drop=True)
                    )
                    st.dataframe(ot_df, hide_index=True, use_container_width=True,
                                 height=min(520, max(200, len(ot_df) * 36 + 40)))

            # --- Тепловая карта: тема × отделение ---
            st.markdown("---")
            st.markdown("##### 🗺 Тепловая карта жалоб: тема × отделение")

            top_offices_list = neg_counts.head(15)['отделение'].tolist()
            all_theme_cnt_neg = Counter(
                t for themes in neg_by_office['темы'] for t in themes
                if t not in _POS_THEMES
            )
            top_themes_list = [t for t, _ in all_theme_cnt_neg.most_common(10)]

            if top_offices_list and top_themes_list:
                matrix_rows = []
                for office in top_offices_list:
                    grp = neg_by_office[neg_by_office['отделение'] == office]
                    row = {'Отделение': office[:38]}
                    for theme in top_themes_list:
                        row[theme] = sum(1 for tlist in grp['темы'] if theme in tlist)
                    matrix_rows.append(row)

                mat_df = pd.DataFrame(matrix_rows).set_index('Отделение')

                fig = px.imshow(
                    mat_df,
                    color_continuous_scale=[FREEDOM_LIGHT, FREEDOM_MID, NEG_COLOR],
                    labels=dict(color='Жалоб'),
                    aspect='auto',
                    text_auto=True,
                )
                fig.update_layout(
                    height=max(320, len(top_offices_list) * 34),
                    margin=dict(t=30, b=10, l=10, r=20),
                    coloraxis_colorbar=dict(title='Жалоб', len=0.6),
                    xaxis=dict(tickangle=-35, automargin=True),
                    yaxis=dict(automargin=True),
                )
                fig.update_traces(
                    hovertemplate='<b>%{y}</b> × <b>%{x}</b><br>Жалоб: %{z}<extra></extra>'
                )
                apply_theme(fig)
                st.plotly_chart(fig, use_container_width=True)

            # --- Дрилл-даун: выбор конкретного отделения ---
            st.markdown("---")
            st.markdown("##### Детальный разбор отделения")
            drill_options = ['— выберите отделение —'] + neg_counts['отделение'].tolist()
            drill_choice = st.selectbox("Отделение для анализа жалоб", drill_options,
                                        key='drill_office')
            if drill_choice != '— выберите отделение —':
                drill_neg = neg_by_office[neg_by_office['отделение'] == drill_choice]
                drill_all = _dff_off[_dff_off['отделение'] == drill_choice]

                # Метрики по отделению
                m1, m2, m3 = st.columns(3)
                m1.metric("Негативных отзывов", len(drill_neg))
                m2.metric("Всего отзывов", len(drill_all))
                m3.metric("Доля негатива",
                          f"{len(drill_neg)/len(drill_all)*100:.1f}%" if len(drill_all) else "—")

                # Темы жалоб этого отделения
                drill_cnt = Counter(
                    t for themes in drill_neg['темы'] for t in themes
                    if t not in _POS_THEMES
                )
                if drill_cnt:
                    drill_df = pd.DataFrame(drill_cnt.most_common(12),
                                            columns=['Тема', 'Жалоб'])
                    fig = px.bar(
                        drill_df.sort_values('Жалоб'), x='Жалоб', y='Тема',
                        orientation='h',
                        color='Жалоб',
                        color_continuous_scale=[FREEDOM_SOFT, NEG_COLOR],
                        text='Жалоб',
                    )
                    fig.update_traces(textposition='outside', cliponaxis=False,
                                      marker_line_width=0,
                                      hovertemplate='<b>%{y}</b><br>Жалоб: %{x}<extra></extra>')
                    fig.update_layout(
                        height=max(280, len(drill_df) * 38),
                        margin=dict(t=20, b=10, l=10, r=60),
                        showlegend=False, coloraxis_showscale=False,
                        yaxis=dict(automargin=True),
                    )
                    apply_theme(fig)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Темы в негативных отзывах этого отделения не определены.")

                # Последние негативные отзывы
                st.markdown("**Последние негативные отзывы:**")
                recent = drill_neg.sort_values('datetime', ascending=False).head(10)
                for _, row in recent.iterrows():
                    card_cls = 'neg'
                    sentiment_icon = '🔴'
                    themes_str = ', '.join(row['темы']) if _is_theme_list(row['темы']) else 'без темы'
                    st.markdown(
                        f"<div class='review-card {card_cls}'>"
                        f"<div class='review-meta'>"
                        f"  {rating_badge(int(row['рейтинг']))} &nbsp;"
                        f"  {sentiment_icon} &nbsp;"
                        f"  <b>{str(row['дата'])[:10]}</b> &nbsp;·&nbsp;"
                        f"  <i>{themes_str}</i>"
                        f"</div>"
                        f"{str(row['текст'])[:400]}"
                        f"</div>",
                        unsafe_allow_html=True,
                    )

        else:
            # === Блок: одно отделение / без отделений — общий анализ жалоб ===
            all_neg_cnt = Counter(
                t for themes in neg_by_office['темы'] for t in themes
                if t not in _POS_THEMES
            )

            m1, m2, m3 = st.columns(3)
            m1.metric("Негативных отзывов", f"{len(neg_by_office):,}")
            m2.metric("Всего отзывов", f"{len(_dff_off):,}")
            neg_pct = len(neg_by_office) / len(_dff_off) * 100 if len(_dff_off) else 0
            m3.metric("Доля негатива", f"{neg_pct:.1f}%")

            if all_neg_cnt:
                st.markdown("##### Топ тем в негативных отзывах (рейтинг 1–2)")
                neg_theme_df = pd.DataFrame(
                    all_neg_cnt.most_common(15), columns=['Тема', 'Упоминаний']
                )
                fig = px.bar(
                    neg_theme_df.sort_values('Упоминаний'),
                    x='Упоминаний', y='Тема',
                    orientation='h',
                    color='Упоминаний',
                    color_continuous_scale=[FREEDOM_SOFT, NEG_COLOR],
                    text='Упоминаний',
                )
                fig.update_traces(
                    textposition='outside', cliponaxis=False, marker_line_width=0,
                    hovertemplate='<b>%{y}</b><br>Упоминаний: %{x}<extra></extra>',
                )
                fig.update_layout(
                    height=max(320, len(neg_theme_df) * 36),
                    margin=dict(t=20, b=10, l=10, r=70),
                    showlegend=False, coloraxis_showscale=False,
                    yaxis=dict(automargin=True),
                )
                apply_theme(fig)
                st.plotly_chart(fig, use_container_width=True)

            # Последние негативные отзывы
            st.markdown("---")
            st.markdown("##### Последние негативные отзывы")
            recent_neg = neg_by_office.sort_values('datetime', ascending=False).head(5)
            for _, row in recent_neg.iterrows():
                themes_str = ', '.join(row['темы']) if _is_theme_list(row['темы']) else 'без темы'
                st.markdown(
                    f"<div class='review-card neg'>"
                    f"<div class='review-meta'>"
                    f"  {rating_badge(int(row['рейтинг']))} &nbsp;🔴&nbsp;"
                    f"  <b>{str(row['дата'])[:10]}</b> &nbsp;·&nbsp;"
                    f"  <i>{themes_str}</i>"
                    f"</div>"
                    f"{str(row['текст'])[:400]}"
                    f"</div>",
                    unsafe_allow_html=True,
                )


# ------------------------------------------------------------
# TAB 5 — Работа с отзывами
# ------------------------------------------------------------
if selected_tab == "responses":
    st.subheader("Как банк работает с отзывами")

    # Найти самый быстрый и самый долгий ответ
    answered_dff = dff[dff['есть_ответ'] & dff['часов_до_ответа'].notna()].copy()
    if len(answered_dff) > 0:
        fastest_response = answered_dff.loc[answered_dff['часов_до_ответа'].idxmin()]
        slowest_response = answered_dff.loc[answered_dff['часов_до_ответа'].idxmax()]
        fastest_time = fastest_response['часов_до_ответа']
        slowest_time = slowest_response['часов_до_ответа']
        fastest_answer_at = fastest_response['ответ_дата'].strftime('%Y-%m-%d %H:%M') if pd.notna(fastest_response['ответ_дата']) else '—'
        slowest_answer_at = slowest_response['ответ_дата'].strftime('%Y-%m-%d %H:%M') if pd.notna(slowest_response['ответ_дата']) else '—'
    else:
        fastest_time = slowest_time = None
        fastest_answer_at = slowest_answer_at = '—'

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Доля отвеченных", f"{answer_rate:.1f}%",
              help=f"{dff['есть_ответ'].sum():,} из {len(dff):,} отзывов получили официальный ответ")
    c2.metric("Медиана ответа", f"{median_response_h:.1f} ч" if pd.notna(median_response_h) else "—")
    c3.metric("90-й перцентиль", f"{dff['часов_до_ответа'].quantile(0.9):.1f} ч"
              if dff['часов_до_ответа'].notna().any() else "—",
              help="90% ответов — быстрее этого значения")
    c4.metric("Самый быстрый", f"{fastest_time*60:.1f} мин" if fastest_time is not None else "—",
              help="Минимальное время ответа")
    c5.metric("Самый долгий", f"{slowest_time/24:.0f} д" if slowest_time is not None else "—",
              help="Максимальное время ответа")

    st.markdown("---")
    c1, c2 = st.columns(2)

    with c1:
        st.markdown("##### Скорость ответа по рейтингу")
        speed = dff[dff['часов_до_ответа'].notna()].groupby('рейтинг')['часов_до_ответа'].median().reset_index()
        speed['label'] = speed['рейтинг'].astype(str)
        fig = px.bar(speed, x='label', y='часов_до_ответа',
                     color='рейтинг',
                     color_continuous_scale=RATING_SCALE,
                     range_color=[1, 5],
                     labels={'часов_до_ответа': 'Медиана, ч', 'label': 'Рейтинг'},
                     text='часов_до_ответа')
        fig.update_traces(texttemplate='%{text:.1f} ч', textposition='outside',
                          marker_line_width=0,
                          hovertemplate='<b>Рейтинг %{x}</b><br>Медиана: %{y:.1f} ч<extra></extra>')
        fig.update_layout(height=380, showlegend=False, coloraxis_showscale=False,
                          margin=dict(t=60, b=20, l=10, r=80),
                          yaxis=dict(automargin=True, range=[0, speed['часов_до_ответа'].max() * 1.15]), 
                          xaxis=dict(dtick=1))
        apply_theme(fig)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.markdown("##### Доля отвеченных по рейтингу")
        answ = dff.groupby('рейтинг')['есть_ответ'].mean().reset_index()
        answ['label'] = answ['рейтинг'].astype(str)
        answ['pct'] = answ['есть_ответ'] * 100
        fig = px.bar(answ, x='label', y='pct',
                     color='рейтинг',
                     color_continuous_scale=RATING_SCALE,
                     range_color=[1, 5],
                     labels={'pct': '% отвеченных', 'label': 'Рейтинг'},
                     text='pct')
        fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside',
                          marker_line_width=0,
                          hovertemplate='<b>Рейтинг %{x}</b><br>Отвечено: %{y:.1f}%<extra></extra>',
                          cliponaxis=False)
        fig.update_layout(height=380, showlegend=False, coloraxis_showscale=False,
                          yaxis=dict(range=[0, 105], automargin=True),
                          margin=dict(t=60, b=20, l=10, r=80))
        apply_theme(fig)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.markdown("##### Распределение времени ответа (часы)")
    valid = dff[dff['часов_до_ответа'].notna() & (dff['часов_до_ответа'] < 200)]
    if len(valid) > 0:
        fig = px.histogram(valid, x='часов_до_ответа', nbins=40,
                           color_discrete_sequence=[FREEDOM_GREEN],
                           labels={'часов_до_ответа': 'Часов от отзыва до ответа'})
        fig.update_traces(marker_line_width=0,
                          hovertemplate='<b>%{x} ч</b><br>Отзывов: %{y}<extra></extra>')
        fig.update_layout(height=340, margin=dict(t=60, b=30, l=10, r=10), bargap=0.04,
                          yaxis_title="Кол-во отзывов")
        apply_theme(fig)
        st.plotly_chart(fig, use_container_width=True)

    # Визуализация экстремальных отзывов
    if len(answered_dff) > 0:
        st.markdown("---")
        st.markdown("##### 📈 Экстремальные примеры работы с отзывами")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("###### 🟢 Самый быстрый ответ")
            st.markdown(f"""
            <div style="background: #FFFFFF; border: 1px solid #EEF2EE; border-left: 4px solid {FREEDOM_GREEN}; 
                        padding: 16px 20px; margin: 10px 0; border-radius: 10px; font-size: 0.93rem;">
                <div style="color: #6B7A6F; font-size: 0.82rem; margin-bottom: 8px;">
                    📅 {fastest_response['дата'].strftime('%Y-%m-%d')} {fastest_response['время']} • ⭐ {fastest_response['рейтинг']} звёзд • ⏱️ {fastest_time:.3f} ч
                </div>
                <div style="margin-bottom: 12px; line-height: 1.5;">
                    <strong>Отзыв:</strong> {fastest_response['текст']}
                </div>
                <div style="border-top: 1px solid #EEF2EE; padding-top: 12px; color: {FREEDOM_DARK};">
                    <strong>Ответ банка:</strong> {fastest_response['ответ_текст']}
                    <div style="color: #6B7A6F; font-size: 0.82rem; margin-top: 8px;">
                        Ответ отправлен: {fastest_answer_at}
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown("###### 🔴 Самый долгий ответ")
            st.markdown(f"""
            <div style="background: #FFFFFF; border: 1px solid #EEF2EE; border-left: 4px solid {NEG_COLOR}; 
                        padding: 16px 20px; margin: 10px 0; border-radius: 10px; font-size: 0.93rem;">
                <div style="color: #6B7A6F; font-size: 0.82rem; margin-bottom: 8px;">
                    📅 {slowest_response['дата'].strftime('%Y-%m-%d')} {slowest_response['время']} • ⭐ {slowest_response['рейтинг']} звёзд • ⏱️ {slowest_time:.1f} ч
                </div>
                <div style="margin-bottom: 12px; line-height: 1.5;">
                    <strong>Отзыв:</strong> {slowest_response['текст']}
                </div>
                <div style="border-top: 1px solid #EEF2EE; padding-top: 12px; color: {FREEDOM_DARK};">
                    <strong>Ответ банка:</strong> {slowest_response['ответ_текст']}
                    <div style="color: #6B7A6F; font-size: 0.82rem; margin-top: 8px;">
                        Ответ отправлен: {slowest_answer_at}
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

if selected_tab == "cohorts":
    st.subheader("👥 Когортный анализ: повторные пользователи")

    # Считаем по полному датасету, не по фильтру, — нужна глобальная история автора
    author_all = df['автор'].value_counts()
    real_df = dff[~dff['автор'].isin(GENERIC_AUTHORS)].copy()
    real_df['отзывов_всего'] = real_df['автор'].map(author_all)
    real_df['тип'] = real_df['отзывов_всего'].apply(lambda x: 'Repeat (2+)' if x >= 2 else 'One-time')

    unique_authors = real_df['автор'].nunique()
    repeat_authors = real_df[real_df['отзывов_всего'] >= 2]['автор'].nunique()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Уникальных авторов", f"{unique_authors:,}")
    c2.metric("Repeat-авторов", f"{repeat_authors:,}",
              help="Оставили 2+ отзыва за всё время")
    c3.metric(
        "% repeat среди авторов",
        f"{(repeat_authors / unique_authors * 100):.1f}%" if unique_authors else "—",
    )
    c4.metric(
        "% отзывов от repeat",
        f"{(real_df['отзывов_всего'] >= 2).mean() * 100:.1f}%" if len(real_df) else "—",
    )

    st.markdown("---")

    c1, c2 = st.columns([2, 3])

    with c1:
        st.markdown("##### One-time vs Repeat: средний рейтинг")
        cohort_agg = real_df.groupby('тип').agg(
            отзывов=('рейтинг', 'count'),
            ср_рейтинг=('рейтинг', 'mean'),
            нег=('рейтинг', lambda x: (x <= 2).mean() * 100),
        ).reset_index()

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=cohort_agg['тип'], y=cohort_agg['ср_рейтинг'],
            marker_color=[FREEDOM_GREEN, FREEDOM_GREEN],
            marker_line_width=0,
            marker_opacity=0.92,
            width=0.45,
            text=cohort_agg['ср_рейтинг'].round(2),
            textposition='outside',
            textfont=dict(size=13),
            customdata=cohort_agg['отзывов'],
            hovertemplate='<b>%{x}</b><br>Ср. рейтинг: %{y:.2f}<br>'
                          'Отзывов: %{customdata}<extra></extra>',
        ))
        fig.update_layout(height=320, yaxis_range=[0, 5.5],
                          yaxis_title="Ср. рейтинг",
                          margin=dict(t=60, b=30, l=40, r=100))
        fig.update_yaxes(automargin=True)
        apply_theme(fig)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.markdown("##### Распределение количества отзывов на автора")
        author_counts = real_df.groupby('автор').size().reset_index(name='кол-во')
        bins = author_counts['кол-во'].apply(
            lambda x: '1' if x == 1 else ('2' if x == 2 else ('3' if x == 3 else '4+'))
        )
        bin_counts = bins.value_counts().reindex(['1', '2', '3', '4+']).fillna(0).reset_index()
        bin_counts.columns = ['Кол-во отзывов', 'Авторов']
        fig = px.bar(bin_counts, x='Кол-во отзывов', y='Авторов',
                     color_discrete_sequence=[FREEDOM_GREEN],
                     text='Авторов')
        fig.update_traces(
            textposition='outside',
            marker_line_width=0,
            marker_opacity=0.92,
            cliponaxis=False,
        )
        fig.update_layout(height=320, showlegend=False,
                          margin=dict(t=60, b=30, l=40, r=100))
        fig.update_yaxes(automargin=True)
        apply_theme(fig)
        st.plotly_chart(fig, use_container_width=True)

    # Топ-авторов
    st.markdown("---")
    st.markdown("##### Топ-10 наиболее активных авторов")
    top_auth = (
        real_df.groupby('автор')
        .agg(отзывов=('рейтинг', 'size'),
             ср_рейтинг=('рейтинг', 'mean'),
             первый=('datetime', 'min'),
             последний=('datetime', 'max'),
             адрес=('отделение', lambda x: ', '.join(sorted(set(x)))))
        .sort_values('отзывов', ascending=False)
        .head(10).round(2).reset_index()
    )
    top_auth['период'] = top_auth.apply(
        lambda r: f"{r['первый'].strftime('%b %Y')} – {r['последний'].strftime('%b %Y')}",
        axis=1
    )
    top_auth_disp = top_auth[['автор', 'отзывов', 'ср_рейтинг', 'период', 'адрес']].copy()
    top_auth_disp.columns = ['Автор', 'Отзывов', 'Ср. рейтинг', 'Период активности', 'Адрес']
    st.dataframe(top_auth_disp, hide_index=True, use_container_width=True)
    selected_author = st.selectbox(
        "Выбери автора для просмотра отзывов",
        top_auth['автор']
    )

    author_reviews = real_df[real_df['автор'] == selected_author] \
        .sort_values('datetime', ascending=False)

    st.markdown(f"##### Отзывы автора: {selected_author}")

    st.dataframe(
        author_reviews[['datetime', 'отделение', 'рейтинг', 'текст']]
        .rename(columns={
            'datetime': 'Дата',
            'отделение': 'Отделение',
            'рейтинг': 'Рейтинг',
            'текст': 'Отзыв'
        }),
        use_container_width=True,
        hide_index=True
    )

# ------------------------------------------------------------
# TAB 6 — Отзывы (эксплорер)
# ------------------------------------------------------------
if selected_tab == "reviews":
    st.subheader("Эксплорер отзывов")

    c1, c2, c3 = st.columns([3, 1, 1])
    search_q = c1.text_input("Поиск по тексту отзыва", placeholder="напр. 'очередь', 'ипотека', 'приложение'")
    sort_by = c2.selectbox("Сортировка", ["Недавние", "Давние", "Худшие", "Лучшие", "Длинные", "Популярные"])
    limit = c3.number_input("Показать", min_value=5, max_value=100, value=20, step=5)

    view = dff.copy()
    if search_q:
        view = view[view['текст'].fillna('').str.contains(search_q, case=False, regex=False)]

    sort_map = {
        "Недавние": ('datetime', False),
        "Давние": ('datetime', True),
        "Худшие": ('рейтинг', True),
        "Лучшие": ('рейтинг', False),
        "Длинные": ('длина_текста', False),
        "Популярные": ('likes_count', False),
    }
    col, asc = sort_map[sort_by]

    # считаем полное число найденных ДО лимита
    total_found = len(view)
    view = view.sort_values(col, ascending=asc).head(limit)
    shown = len(view)

    if total_found == 0:
        st.caption("По запросу ничего не найдено.")
    elif total_found <= limit:
        st.caption(f"Найдено: {total_found:,} отзывов")
    else:
        st.caption(f"Найдено: {total_found:,} отзывов, показано {shown}")

    for _, row in view.iterrows():
        rating = int(row['рейтинг'])
        cls = "neg" if rating <= 2 else ("neu" if rating == 3 else "")
        answer_block = ""
        if pd.notna(row.get('ответ_текст')):
            ans = str(row['ответ_текст']).replace('\n', '<br>')
            answer_block = (
                f"<div class='bank-answer'>"
                f"<b>Ответ банка:</b><br>{ans[:500]}{'...' if len(ans) > 500 else ''}"
                f"</div>"
            )
        text = str(row['текст']).replace('\n', '<br>') if pd.notna(row['текст']) else ''
        office_name = row['отделение'] if pd.notna(row['отделение']) and row['отделение'] != '—' else 'отделение не указано'
        office_part = f"<span style='color:#6B7A6F'>· {office_name}</span>"
        likes_badge = (
            f"<span style='background:{POS_COLOR};color:{FREEDOM_DARK};"
            f"padding:3px 10px;border-radius:12px;font-weight:700;font-size:0.8rem;'>"
            f"👍 {int(row.get('likes_count', 0))}</span>"
            if int(row.get('likes_count', 0)) > 0 else ''
        )
        st.markdown(
            f"<div class='review-card {cls}'>"
            f"<div class='review-meta'>"
            f"{likes_badge}"
            f"{rating_badge(rating)}"
            f"<span>{row['datetime'].strftime('%d.%m.%Y %H:%M')}</span>"
            f"<span>·</span>"
            f"<b style='color:{FREEDOM_DARK}'>{row['автор']}</b>"
            f"{office_part}"
            f"</div>"
            f"{text}"
            f"{answer_block}"
            f"</div>",
            unsafe_allow_html=True,
        )
