"""豆包语音合成插件（Doubao TTS）

让麦麦能用豆包语音（火山引擎 · 语音合成大模型）说话。

- 手动触发：发送 ``/说 文本`` / ``/语音 文本`` / ``/speak 文本``，麦麦把文本转语音发到当前会话；
- 麦麦自主触发：插件注册一个 Tool ``doubao_tts_speak``，LLM 在合适时机（用户要求用语音时）调用它；
- 音色：内置火山官方预置音色清单（按名字选即可），也支持直接填 voice_type 音色 ID 或复刻音色 ID；
- 跨语种：麦麦回复是日语/英语等外语时，自动按该语种合成；也可开启「语音翻译」把中文回复翻成目标语种；
- 鉴权：火山引擎**新版控制台** API Key（X-Api-Key 单头鉴权），无需旧版 App ID / Access Token。

功能组织参考了 xuqian13/tts_voice_plugin（致谢见 README）。
"""

import asyncio
import base64
import json
import logging
import random
import time
import uuid
from typing import Any, Dict, List, Literal, Optional, Tuple

import aiohttp
from pydantic import field_validator

from maibot_sdk import Command, Field, HookHandler, MaiBotPlugin, PluginConfigBase, Tool
from maibot_sdk.types import ErrorPolicy, HookMode, HookOrder, ToolParamType, ToolParameterInfo

logger = logging.getLogger("plugin.doubao_tts")

SUPPORTED_CONFIG_VERSION = "1.7.1"
ERROR_PROMPT_DEDUPE_SECONDS = 30.0
"""同一会话失败提示的去重窗口（秒）：LLM 对失败有重试倾向，去重防提示刷屏。"""
# ─── 火山引擎 API ────────────────────────────────────────────────────────────
DOUBAO_TTS_URL = "https://openspeech.bytedance.com/api/v3/tts/unidirectional"
DOUBAO_RESOURCE_PRESET = "seed-tts-2.0"   # 预置音色（语音合成模型 2.0）
DOUBAO_RESOURCE_CLONE = "seed-icl-2.0"    # 复刻音色（声音复刻模型 2.0）

# 官方预置音色（voice_type 以 _uranus_bigtts 结尾，配 seed-tts-2.0）
# 配置里可直接写“显示名”，也可直接填任意 voice_type 音色 ID
PRESET_VOICES: Dict[str, str] = {
    "小何 2.0": "zh_female_xiaohe_uranus_bigtts",
    "Vivi 2.0": "zh_female_vv_uranus_bigtts",
    "爽快思思 2.0": "zh_female_shuangkuaisisi_uranus_bigtts",
    "甜美小源 2.0": "zh_female_tianmeixiaoyuan_uranus_bigtts",
    "甜美桃子 2.0": "zh_female_tianmeitaozi_uranus_bigtts",
    "邻家女孩 2.0": "zh_female_linjianvhai_uranus_bigtts",
    "清新女声 2.0": "zh_female_qingxinnvsheng_uranus_bigtts",
    "流畅女声 2.0": "zh_female_liuchangnv_uranus_bigtts",
    "魅力女友 2.0": "zh_female_meilinvyou_uranus_bigtts",
    "云舟 2.0": "zh_male_m191_uranus_bigtts",
    "小天 2.0": "zh_male_taocheng_uranus_bigtts",
    "刘飞 2.0": "zh_male_liufei_uranus_bigtts",
}

# 情感参数（火山支持的 emotion 取值）
PRESET_EMOTIONS: Dict[str, str] = {
    "开心": "happy",
    "伤心": "sad",
    "生气": "angry",
    "害怕": "scare",
    "惊讶": "surprise",
    "讨厌": "hate",
    "哭泣": "tear",
    "抱歉": "sorry",
    "平静": "pleased",
    "播音": "narrator",
    "讲故事": "storytelling",
}

# 默认音色
DEFAULT_VOICE_DISPLAY = "小何 2.0"


def _resolve_voice_type(voice: str) -> str:
    """把配置里的“显示名”或“原始 ID”解析成 voice_type。"""
    voice = (voice or "").strip()
    if not voice:
        return PRESET_VOICES[DEFAULT_VOICE_DISPLAY]
    if voice in PRESET_VOICES:
        return PRESET_VOICES[voice]
    return voice  # 直接当作 voice_type ID（预置 ID / 复刻音色 ID）


def detect_language(text: str) -> str:
    """按字符范围粗略判断文本语种，返回火山 ``explicit_language`` 取值。

    返回 ``""`` 表示"不指定"（交给火山默认处理）。

    为什么需要这个（2026-09-16 实测结论，音色 S_RcCsDWbd2 + seed-icl-2.0）：
      · 日语文本 **不传** explicit_language → 发音明显退化，不可接受；
      · 日语文本 **传** ``ja`` → 发音正常；
      · 中文文本（不传）、英文文本（不传）→ 都正常，无需指定。
    所以这里只对"非中英、且能从字符判断出来"的语种做显式指定，中英文保持默认。
    """
    if not text:
        return ""

    def has(lo: str, hi: str) -> bool:
        return any(lo <= ch <= hi for ch in text)

    if has("\u3040", "\u30ff"):        # 平假名 / 片假名
        return "ja"
    if has("\uac00", "\ud7af"):        # 谚文（韩语）
        return "ko"
    if has("\u0400", "\u04ff"):        # 西里尔字母（俄语等）
        return "ru"
    if has("\u0600", "\u06ff"):        # 阿拉伯字母
        return "ar"
    if has("\u0e00", "\u0e7f"):        # 泰文
        return "th"
    # 其余（中文 / 英文 / 拉丁字母其他语言）不指定，走火山默认，实测正常
    return ""


# 语种别名（中文名 / 代码）→ (中文名, 火山 explicit_language 取值)
LANG_ALIASES: Dict[str, Tuple[str, str]] = {
    "ja": ("日语", "ja"), "jp": ("日语", "ja"), "日语": ("日语", "ja"), "日文": ("日语", "ja"),
    "en": ("英语", "en"), "英语": ("英语", "en"), "英文": ("英语", "en"),
    "ko": ("韩语", "ko"), "kr": ("韩语", "ko"), "韩语": ("韩语", "ko"), "韩文": ("韩语", "ko"),
    "zh-cn": ("中文", "zh-cn"), "zh": ("中文", "zh-cn"), "cn": ("中文", "zh-cn"),
    "中文": ("中文", "zh-cn"), "汉语": ("中文", "zh-cn"),
    "de": ("德语", "de"), "德语": ("德语", "de"),
    "fr": ("法语", "fr"), "法语": ("法语", "fr"),
    "es-mx": ("西班牙语", "es-mx"), "es": ("西班牙语", "es-mx"), "西班牙语": ("西班牙语", "es-mx"),
    "id": ("印尼语", "id"), "印尼语": ("印尼语", "id"),
    "pt-br": ("葡萄牙语", "pt-br"), "pt": ("葡萄牙语", "pt-br"), "葡萄牙语": ("葡萄牙语", "pt-br"),
    "ru": ("俄语", "ru"), "俄语": ("俄语", "ru"),
    "th": ("泰语", "th"), "泰语": ("泰语", "th"),
    "ar": ("阿拉伯语", "ar"), "阿拉伯语": ("阿拉伯语", "ar"),
}


def resolve_language(name: str) -> Tuple[str, str]:
    """把用户填写的语种（中文名或代码）解析为 (中文名, 代码)；无法识别时返回 ("", "")。"""
    raw = (name or "").strip()
    if not raw:
        return "", ""
    hit = LANG_ALIASES.get(raw) or LANG_ALIASES.get(raw.lower())
    if hit:
        return hit
    return raw, ""   # 未知写法：中文名按原样用于翻译提示，代码留空


def _looks_like_language(text: str, code: str) -> bool:
    """粗略判断文本是否**整句**已经是目标语种（用于跳过无意义的翻译）。

    ⚠️ 不能只看"有没有该语种字符"：麦麦的中文回复经常夹带日语语气词/角色口癖，
    若一见到假名就判定"这已经是日语"，整句中文就不会被翻译——实测踩过这个坑。
    因此这里改为比例判断：只有目标语种字符在句中占主导，才算"已是该语种"。
    """
    if not text:
        return False

    def count(lo: str, hi: str) -> int:
        return sum(1 for ch in text if lo <= ch <= hi)

    han = count("\u4e00", "\u9fff")   # 汉字（中文/日文共用，单凭它无法区分）

    if code == "ja":
        kana = count("\u3040", "\u30ff")
        if kana == 0:
            return False
        # 假名数量不少于汉字的一半，才认为整句是日语（正常日语文本通常远高于此）
        return kana * 2 >= han

    if code in ("ko", "ru", "ar", "th"):
        span = {
            "ko": ("\uac00", "\ud7af"),
            "ru": ("\u0400", "\u04ff"),
            "ar": ("\u0600", "\u06ff"),
            "th": ("\u0e00", "\u0e7f"),
        }[code]
        lang_chars = count(*span)
        if lang_chars == 0:
            return False
        # 目标语种字符比汉字还少 → 属于中（外）混排，仍需翻译
        return lang_chars > han

    if code == "zh-cn":
        return han > 0 and detect_language(text) == ""

    # 拉丁字母语系（英/德/法/西/葡/印尼）在字符层面无法区分，统一交给模型翻译
    return False


def _split_sentences(text: str, max_len: int) -> List[str]:
    """把文本切成不超过 max_len 的若干段，优先在标点处断。"""
    text = (text or "").strip()
    if not text:
        return []
    if len(text) <= max_len:
        return [text]

    ends = "。！？!?…；;"
    parts: List[str] = []
    buf = ""
    for ch in text:
        buf += ch
        if len(buf) >= max_len:
            # 到硬上限：若末尾是标点直接断，否则在最后一个标点处断
            cut = -1
            if buf[-1] not in ends:
                for i in range(len(buf) - 2, -1, -1):
                    if buf[i] in ends:
                        cut = i + 1
                        break
            if cut <= 0:
                parts.append(buf)
                buf = ""
            else:
                parts.append(buf[:cut])
                buf = buf[cut:]
    if buf:
        parts.append(buf)
    return [p.strip() for p in parts if p.strip()]


# ─── 配置模型 ────────────────────────────────────────────────────────────────


class PluginSectionConfig(PluginConfigBase):
    """插件基础配置。"""

    __ui_label__ = "插件"
    __ui_icon__ = "package"
    __ui_order__ = 0

    enabled: bool = Field(
        default=True,
        description="是否启用插件（总开关；关闭=插件彻底卸载、命令消失）",
        json_schema_extra={"label": "插件总开关"},
    )
    config_version: str = Field(
        default=SUPPORTED_CONFIG_VERSION,
        description="配置版本（勿改）",
        json_schema_extra={"hidden": True, "disabled": True},
    )


class DoubaoSectionConfig(PluginConfigBase):
    """豆包语音 API 连接配置。"""

    __ui_label__ = "豆包语音"
    __ui_icon__ = "plug"
    __ui_order__ = 1

    api_key: str = Field(
        default="",
        description="火山引擎新版控制台 API Key（控制台→豆包语音→API Key 管理）。敏感信息，勿随插件分发",
        json_schema_extra={"label": "API Key", "secret": True},
    )
    resource_id: str = Field(
        default=DOUBAO_RESOURCE_PRESET,
        description="资源 ID：seed-tts-2.0=预置音色（默认）；seed-icl-2.0=复刻音色。可直接填这两个常用值，也支持填其他资源 ID",
        json_schema_extra={
            "label": "Resource ID",
            "placeholder": "seed-tts-2.0 或 seed-icl-2.0",
            "example": "seed-tts-2.0",
            "hint": "常用值：seed-tts-2.0（官方预置音色）｜seed-icl-2.0（复刻音色）。想用其他资源 ID 直接填入即可",
        },
    )
    audio_format: str = Field(
        default="mp3",
        description="音频编码格式：mp3（推荐）/ ogg_opus",
        json_schema_extra={"label": "音频格式", "hidden": True},
    )
    sample_rate: int = Field(
        default=24000,
        description="采样率 Hz（默认 24000）",
        json_schema_extra={"label": "采样率", "hidden": True},
    )


class VoiceToneSectionConfig(PluginConfigBase):
    """音色与情感配置。"""

    __ui_label__ = "音色与情感"
    __ui_icon__ = "music"
    __ui_order__ = 2

    @field_validator("emotion", mode="before")
    @classmethod
    def _normalize_emotion(cls, v: Any) -> Any:
        """兼容历史配置里的"无情感"写法，避免 Literal 校验失败导致插件启动不了。

        下拉不允许空串选项（前端会崩），所以"无"在不同版本里用过不同占位值：
        `""` / `none` / `无` / `无（正常语气）`。这里统一归一化为 `none`。
        """
        if v is None:
            return "none"
        if isinstance(v, str) and v.strip() in ("", "none", "无", "无（正常语气）", "无(正常语气)"):
            return "none"
        return v

    voice: str = Field(
        default=DEFAULT_VOICE_DISPLAY,
        description="音色：预置音色显示名（见 README 音色表）或直接填 voice_type 音色 ID",
        json_schema_extra={
            "label": "音色",
            "placeholder": "小何 2.0 / Vivi 2.0 / 云舟 2.0 或 voice_type ID",
            "hint": "常用预置音色：小何 2.0、Vivi 2.0、爽快思思 2.0、甜美小源 2.0、云舟 2.0…完整列表见 README",
        },
    )
    emotion: Literal["none", "开心", "伤心", "生气", "害怕", "惊讶", "讨厌", "哭泣", "抱歉", "平静", "播音", "讲故事"] = Field(
        default="none",
        description="情感语气：选一个固定情感（默认“无”=正常语气）。若行为页 emotion_mode=auto，则麦麦自主语音时由 LLM 现场挑，本值仅用于手动 /说 与 fixed 模式",
        json_schema_extra={
            "label": "情感（可自主）",
            "options": {
                "none": {"label": "无（正常语气）", "description": "不带情感，最自然的播报语气"},
                "开心": {"label": "开心", "description": "欢乐上扬的语气"},
                "伤心": {"label": "伤心", "description": "低落难过的语气"},
                "生气": {"label": "生气", "description": "不满或愤怒的语气"},
                "害怕": {"label": "害怕", "description": "紧张害怕的语气"},
                "惊讶": {"label": "惊讶", "description": "吃惊意外的语气"},
                "讨厌": {"label": "讨厌", "description": "嫌弃反感的语气"},
                "哭泣": {"label": "哭泣", "description": "带着哭腔"},
                "抱歉": {"label": "抱歉", "description": "歉意诚恳的语气"},
                "平静": {"label": "平静", "description": "沉稳温和、适合安慰"},
                "播音": {"label": "播音", "description": "字正腔圆的播音腔"},
                "讲故事": {"label": "讲故事", "description": "娓娓道来、适合朗读故事"},
            },
        },
    )
    emotion_scale: float = Field(
        default=1.0,
        ge=1.0,
        le=5.0,
        description="情感强度 1~5（配合情感使用，1=最淡）。若行为页 emotion_scale_mode=auto，麦麦自主时由 LLM 按情感挑",
        json_schema_extra={"label": "情感强度（可自主）"},
    )


class SpeedLoudSectionConfig(PluginConfigBase):
    """语速与音量配置（火山固定连续数值，仅手动设置）。"""

    __ui_label__ = "语速与音量"
    __ui_icon__ = "gauge"
    __ui_order__ = 3

    speech_rate: float = Field(
        default=0.0,
        description="语速 -50~100（0=正常）。火山固定连续数值，麦麦不可自主，仅手动设置",
        json_schema_extra={"label": "语速（仅手动）"},
    )
    loudness: float = Field(
        default=0.0,
        description="音量 -50~100（0=正常）。火山固定连续数值，麦麦不可自主，仅手动设置",
        json_schema_extra={"label": "音量（仅手动）"},
    )


class TranslateSectionConfig(PluginConfigBase):
    """语音翻译配置：合成前把文本翻译成目标语言。"""

    __ui_label__ = "语音翻译"
    __ui_icon__ = "languages"
    __ui_order__ = 4

    mode: Literal["不翻译", "全部翻译", "按概率翻译", "由麦麦判断"] = Field(
        default="不翻译",
        description=(
            "合成前要不要先把文本翻译成「目标语种」，四选一：\n"
            "· 不翻译（默认）：原样合成，麦麦说什么语言就合成什么语言（中英日混排也原样保留）；\n"
            "· 全部翻译：所有回复都翻译成目标语种（例如全说日语）；\n"
            "· 按概率翻译：每条回复掷骰子，按「翻译概率」决定这条翻不翻 → 会出现中/外语交替；\n"
            "· 由麦麦判断：让 LLM 读这条回复自己决定翻不翻（判断依据可写在「判断规则」里）。"
        ),
        json_schema_extra={
            "label": "翻译方式",
            "hint": (
                "· 不翻译 = 麦麦说中文就发中文语音\n"
                "· 全部翻译 = 全说目标语种（适合麦麦说中文、音色是外语音色的情况）\n"
                "· 按概率翻译 = 由「翻译概率」控制，例如 0.5 → 约一半回复是外语、一半保留中文\n"
                "· 由麦麦判断 = 让麦麦自己决定（会多一次模型调用，可在「判断规则」里告诉它你的偏好）\n"
                "四种模式都会自动跳过「文本已是目标语种」的情况；翻译失败一律回退原文，不影响发声。"
            ),
        },
    )
    target: str = Field(
        default="",
        description=(
            "要翻译成哪种语言——填你的音色所训练的语言。"
            "支持中文名或代码：日语/ja、英语/en、韩语/ko、德语/de、法语/fr、"
            "西班牙语/es-mx、印尼语/id、葡萄牙语/pt-br、俄语/ru、泰语/th、阿拉伯语/ar。留空=不翻译"
        ),
        json_schema_extra={
            "label": "目标语种",
            "placeholder": "日语（或 ja）",
            "hint": (
                "填「音色训练时用的语言」：日语=ja｜英语=en｜韩语=ko｜德语=de｜法语=fr｜"
                "西班牙语=es-mx｜印尼语=id｜葡萄牙语=pt-br｜俄语=ru｜泰语=th｜阿拉伯语=ar\n"
                "中文名或代码都行（填「日语」与「ja」等效）。留空 = 不翻译。"
            ),
        },
    )
    probability: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="「按概率翻译」模式下的翻译概率 0~1：0.5=约一半回复翻译、一半保留原文；1=全都翻译、0=都不翻译。其他模式忽略此项",
        json_schema_extra={
            "label": "翻译概率",
            "hint": "仅「按概率翻译」模式生效。0.3 = 约三成回复翻成外语，七成保留原文。想「偶尔冒一句日语」就调小一点。",
        },
    )
    rule: str = Field(
        default="",
        description=(
            "「由麦麦判断」模式下的判断依据（可选）。留空则用默认规则："
            "日常口语化的内容翻译，纯信息类（数字、链接、代码、专有名词、报错）保留原文"
        ),
        json_schema_extra={
            "label": "判断规则（可选）",
            "placeholder": "例如：只有安慰我、闲聊时用日语；其他都保留中文",
            "hint": "用一句话告诉麦麦「什么时候该说外语」。留空 = 默认规则（口语内容翻译，数字/链接/专有名词保留）。",
        },
    )
    model_task: str = Field(
        default="utils",
        description="用哪个麦麦模型来做翻译（填模型任务名，如 utils / replyer / planner；utils 是小模型、便宜快，推荐）",
        json_schema_extra={
            "label": "翻译用模型任务",
            "placeholder": "utils",
            "hint": "填的是「模型任务名」而不是模型名。utils=麦麦的小模型（便宜快，推荐）；也可填 replyer / planner。留空时自动用 utils。",
        },
    )
    max_tokens: int = Field(
        default=800,
        ge=50,
        le=4000,
        description="译文最大长度（token）。语音单条上限 500 字，译成外语通常变长，插件还会按原文长度自动放大，一般不用改",
        json_schema_extra={"label": "译文最大 tokens"},
    )


class BehaviorSectionConfig(PluginConfigBase):
    """行为配置。"""

    __ui_label__ = "行为"
    __ui_icon__ = "sliders"
    __ui_order__ = 5

    command_enabled: bool = Field(
        default=True,
        description="是否启用 /说 /语音 手动命令（只影响手动命令；麦麦自主语音由 auto_voice_mode 控制，不受此项影响）",
        json_schema_extra={"label": "手动命令"},
    )
    auto_voice_mode: Literal["llm", "probability", "off"] = Field(
        default="llm",
        description="麦麦自主语音方式：llm=由 LLM 自行判断何时用语音（推荐）；probability=按概率偶尔语音（不依赖 LLM）；off=麦麦不自主，仅手动命令",
        json_schema_extra={
            "label": "自主语音方式",
            "options": {
                "llm": {"label": "LLM 自行判断（推荐）", "description": "麦麦自己决定何时用语音：你叫它说、或它觉得适合时都会用"},
                "probability": {"label": "概率触发", "description": "麦麦不靠判断，每收一条消息按概率掷骰，命中则本轮回复转语音（频率由下方概率值精确控制）"},
                "off": {"label": "关闭（仅手动）", "description": "麦麦从不主动语音，只有 /说 /语音 命令才会发声"},
            },
        },
    )
    auto_voice_probability: float = Field(
        default=0.1,
        description="概率模式的触发概率 0~1（0.1=平均每 10 轮约 1 轮语音；0=关）。仅 auto_voice_mode=probability 时生效",
        json_schema_extra={"label": "语音概率"},
    )
    emotion_mode: Literal["fixed", "auto"] = Field(
        default="fixed",
        description="情感参数来源：fixed=用 [doubao] emotion 固定值（默认）；auto=麦麦自主语音时由 LLM 现场挑情感（手动 /说 仍用固定值）",
        json_schema_extra={
            "label": "情感来源",
            "options": {
                "fixed": {"label": "固定（手动设置）", "description": "始终用上方 [doubao] emotion 的值"},
                "auto": {"label": "麦麦自主", "description": "麦麦自主语音（llm 模式）时由 LLM 结合氛围现场挑"},
            },
        },
    )
    emotion_scale_mode: Literal["fixed", "auto"] = Field(
        default="fixed",
        description="情感强度来源：fixed=用 [doubao] emotion_scale 固定值；auto=麦麦自主时由 LLM 按情感档位挑（1~5）",
        json_schema_extra={
            "label": "情感强度来源",
            "options": {
                "fixed": {"label": "固定（手动设置）", "description": "始终用上方 [doubao] emotion_scale 的值"},
                "auto": {"label": "麦麦自主", "description": "麦麦自主语音时由 LLM 按情感挑强度 1~5"},
            },
        },
    )
    timeout_seconds: float = Field(
        default=30.0,
        description="请求火山接口超时（秒）",
        json_schema_extra={"label": "超时（秒）"},
    )
    max_text_length: int = Field(
        default=150,
        description="单条语音最大文本长度（超过按句切分多条发送）",
        json_schema_extra={"label": "单条最大字数"},
    )
    fallback_to_text: bool = Field(
        default=True,
        description="合成失败时把文本以文字形式发出（避免用户干等）",
        json_schema_extra={"label": "失败降级发文字"},
    )
    send_error_prompt: bool = Field(
        default=True,
        description="合成失败时向用户发一句提示",
        json_schema_extra={"label": "失败提示"},
    )
    sync_chat_context: bool = Field(
        default=True,
        description=(
            "发送语音后把原文写回麦麦的对话上下文。"
            "语音消息本身不带文字（可见文本只有「[语音消息]」），不写回去的话，"
            "麦麦下一轮不知道自己说过什么"
        ),
        json_schema_extra={"label": "同步对话上下文"},
    )
    context_prefix: str = Field(
        default="[语音]",
        description="写回对话上下文时加在原文前面的标记（表明这句是语音说的）；留空则不加",
        json_schema_extra={"label": "上下文标记", "placeholder": "[语音]"},
    )
    command_cooldown_seconds: float = Field(
        default=0.0,
        ge=0.0,
        description=(
            "同一会话两次语音合成的最小间隔（秒），0=不限流（默认）。"
            "TTS 按字符计费，手动命令对所有人开放时建议设置一个值（如 10）防刷屏/刷费用"
        ),
        json_schema_extra={
            "label": "冷却限流（秒）",
            "hint": "0=关闭。设置后同一会话在该时间内重复触发语音会被拒绝：命令回一句冷却提示，工具向 LLM 返回失败原因",
        },
    )


class DoubaoTTSRootConfig(PluginConfigBase):
    """插件根配置。"""

    plugin: PluginSectionConfig = Field(default_factory=PluginSectionConfig, json_schema_extra={"label": "插件"})
    doubao: DoubaoSectionConfig = Field(default_factory=DoubaoSectionConfig, json_schema_extra={"label": "豆包语音"})
    voice_tone: VoiceToneSectionConfig = Field(default_factory=VoiceToneSectionConfig, json_schema_extra={"label": "音色与情感"})
    speed_loud: SpeedLoudSectionConfig = Field(default_factory=SpeedLoudSectionConfig, json_schema_extra={"label": "语速与音量"})
    translate: TranslateSectionConfig = Field(default_factory=TranslateSectionConfig, json_schema_extra={"label": "语音翻译"})
    behavior: BehaviorSectionConfig = Field(default_factory=BehaviorSectionConfig, json_schema_extra={"label": "行为"})


# ─── 主插件 ──────────────────────────────────────────────────────────────────


class DoubaoTTSPlugin(MaiBotPlugin):
    """豆包语音合成插件：文字转语音，让麦麦开口说话。"""

    config_model = DoubaoTTSRootConfig

    def __init__(self) -> None:
        super().__init__()
        # 会话 → 概率掷骰命中时间戳（该会话麦麦下一条文字回复将转语音）
        self._pending_voice: Dict[str, float] = {}
        # 防递归：正在发送"概率语音"的标记
        self._sending_pending_voice: bool = False
        # 会话 → 上次失败提示时间戳（去重窗口内不再重复提示）
        self._last_error_prompt_at: Dict[str, float] = {}
        # 会话 → 上次语音合成时间戳（冷却限流，防刷费用/刷屏）
        self._last_speech_at: Dict[str, float] = {}

    # ── 配置读取 ────────────────────────────────────────────────────────

    def _get(self, section: str, key: str, default: Any = None) -> Any:
        try:
            return getattr(getattr(self.config, section, None), key, default)
        except Exception:
            return default

    def _api_key(self) -> str:
        return str(self._get("doubao", "api_key", "") or "").strip()

    # ── 冷却限流 ────────────────────────────────────────────────────────

    def _cooldown_seconds(self) -> float:
        try:
            return max(0.0, float(self._get("behavior", "command_cooldown_seconds", 0) or 0))
        except (TypeError, ValueError):
            return 0.0

    def _cooldown_block(self, stream_id: str) -> bool:
        """冷却限流：同一会话在冷却窗口内拒绝再次合成（防刷费用/刷屏）。

        返回 True=本次被拦截。通过检查时记录本次时间戳。
        对手动命令与麦麦自主 Tool 调用一并生效；概率模式由宿主驱动、每轮最多一次，不在此列。
        """
        cd = self._cooldown_seconds()
        if cd <= 0:
            return False
        now = time.time()
        last = self._last_speech_at.get(stream_id)
        if last is not None and now - last < cd:
            return True
        self._last_speech_at[stream_id] = now
        if len(self._last_speech_at) > 256:  # 防累积：清掉早已冷却完毕的会话
            cutoff = now - max(cd, 60.0)
            self._last_speech_at = {k: v for k, v in self._last_speech_at.items() if v >= cutoff}
        return False

    # ── 概率自主语音 ────────────────────────────────────────────────────

    def _auto_voice_mode(self) -> str:
        return str(self._get("behavior", "auto_voice_mode", "llm") or "llm").strip().lower()

    def _probability_enabled(self) -> bool:
        if self._auto_voice_mode() != "probability":
            return False
        prob = float(self._get("behavior", "auto_voice_probability", 0.1) or 0.0)
        return 0.0 < prob <= 1.0

    def _roll_probability(self) -> bool:
        """掷骰子：是否本轮触发概率语音。"""
        if not self._probability_enabled():
            return False
        prob = float(self._get("behavior", "auto_voice_probability", 0.1) or 0.0)
        hit = random.random() < prob
        self.ctx.logger.info("[豆包TTS] 概率掷骰 p=%.2f → %s", prob, "命中" if hit else "未命中")
        return hit

    def _is_pending_stream(self, stream_id: str) -> bool:
        """判断某会话是否处于"待语音"状态（概率命中后麦麦下一条文字转语音）。"""
        ts = self._pending_voice.get(stream_id)
        if ts is None:
            return False
        if time.time() - ts > 300:  # 5 分钟窗口：命中后麦麦迟迟没回复则作废
            self._pending_voice.pop(stream_id, None)
            return False
        return True

    async def _extract_message_text(self, message: Any) -> Tuple[str, str]:
        """从序列化消息 dict 提取 (纯文本, session_id)；失败返回 ("", "")。"""
        if not isinstance(message, dict):
            return "", ""
        session_id = str(message.get("session_id") or "").strip()
        text = str(message.get("processed_plain_text") or "").strip()
        if not text:
            parts: List[str] = []
            for comp in (message.get("raw_message") or []):
                if isinstance(comp, dict) and comp.get("type") == "text":
                    cdata = comp.get("data")
                    if isinstance(cdata, dict):
                        t = str(cdata.get("text") or "").strip()
                        if t:
                            parts.append(t)
            text = " ".join(parts).strip()
        return text, session_id

    @HookHandler(
        "chat.receive.after_process",
        mode=HookMode.OBSERVE,
        name="doubao_tts_probability_mark",
        description="概率模式：收到普通用户消息时按概率标记该会话本轮语音",
        order=HookOrder.LATE,
        error_policy=ErrorPolicy.SKIP,
    )
    async def _on_inbound_message(self, **kwargs: Any) -> None:
        """入站钩子：概率模式下掷骰子，命中则给该会话打"待语音"标。"""
        message = kwargs.get("message")
        if not isinstance(message, dict):
            return
        # 仅当用户主动发消息（排除命令、通知、系统类）
        if message.get("is_command") or message.get("is_notify"):
            return
        text, session_id = await self._extract_message_text(message)
        if not text or not session_id:
            return
        if not self._probability_enabled():
            return
        if self._roll_probability():
            self._pending_voice[session_id] = time.time()
            self.ctx.logger.info("[豆包TTS] 概率命中：会话 %s 本轮回复将用语音", session_id)

    @HookHandler(
        "send_service.before_send",
        mode=HookMode.BLOCKING,
        name="doubao_tts_probability_speak",
        description="概率模式：待语音会话的文字回复转成语音发出",
        order=HookOrder.EARLY,
        # 本钩子内要做语音合成（一条回复可能切分成多段），宿主默认 5 秒不够：
        # 超时会被记入插件熔断，连续超时会让本插件的钩子被停用。
        # 显式放宽到 30 秒；真的超时也会按 ErrorPolicy.SKIP 放行，原文照发。
        timeout_ms=30000,
        error_policy=ErrorPolicy.SKIP,
    )
    async def _on_before_send(self, **kwargs: Any) -> Dict[str, Any]:
        """出站钩子：麦麦要发文字且该会话被标记"待语音" → 原地换成语音（不中止原消息）。"""
        if self._sending_pending_voice:
            return {"action": "continue"}
        message = kwargs.get("message")
        if not isinstance(message, dict):
            return {"action": "continue"}
        session_id = str(message.get("session_id") or "").strip()
        if not session_id or not self._is_pending_stream(session_id):
            return {"action": "continue"}
        # 只处理含文本的消息（纯图片/语音等放行，避免递归）
        raw_message = message.get("raw_message") or []
        has_text = any(
            isinstance(comp, dict) and comp.get("type") == "text"
            for comp in raw_message
        )
        if not has_text:
            return {"action": "continue"}
        text, _ = await self._extract_message_text(message)
        if not text:
            return {"action": "continue"}
        # 消费标记：只对本轮回复生效
        self._pending_voice.pop(session_id, None)
        self.ctx.logger.info("[豆包TTS] 概率语音：会话 %s 文字回复 → 语音（%d字）", session_id, len(text))
        self._sending_pending_voice = True
        try:
            return await self._replace_with_voice(message, text, kwargs)
        finally:
            self._sending_pending_voice = False

    async def _replace_with_voice(
        self,
        message: Dict[str, Any],
        text: str,
        kwargs: Dict[str, Any],
    ) -> Dict[str, Any]:
        """把一条待发的文字消息原地换成语音消息（返回 Hook 结果字典）。

        为什么不沿用「中止原消息 + 自己另发语音」：宿主 send_service 一旦 abort，
        发送函数返回 None，麦麦的 reply 工具会把本轮回复判为"发送失败"并交回 planner，
        LLM 看到失败便会重发 → 语音和文字就会反复出现。原地替换则让原消息正常走完
        发送流程：reply 拿到成功结果、消息照常入库，一次发送就完成。
        合成失败时返回 continue，原文照常发出，不会丢内容。
        """
        if not self._api_key():
            self.ctx.logger.warning("[豆包TTS] 未配置 API Key，本条保持文字")
            return {"action": "continue"}
        session_id = str(message.get("session_id") or "").strip()
        max_len = max(1, int(self._get("behavior", "max_text_length", 150) or 150))
        # 与手动命令 / Tool 路径一致：先按配置决定是否翻译，再切分逐段合成
        tts_text = await self._translate_if_needed(text)
        segments = _split_sentences(tts_text, max_len)
        if not segments:
            return {"action": "continue"}
        voice_segments: List[Dict[str, Any]] = []
        for seg in segments:
            success, audio, info = await self._synthesize_one(seg)
            if not success or not audio:
                self.ctx.logger.warning("[豆包TTS] 分段合成失败，整条回退为文字: %s", info)
                return {"action": "continue"}
            voice_segments.append({
                "type": "voice",
                # data 留空：可见文本只会渲染成「[语音消息]」，
                # 不会把一长串 base64 灌进麦麦的对话上下文
                "data": "",
                "hash": "",
                "binary_data_base64": base64.b64encode(audio).decode("ascii"),
            })
        new_message = dict(message)
        new_message["raw_message"] = voice_segments
        self.ctx.logger.info(
            "[豆包TTS] 概率语音：会话 %s 已把文字回复换成 %d 条语音", session_id, len(voice_segments)
        )
        # 语音本身不带文字，另外把原文补进对话上下文
        if self._get("behavior", "sync_chat_context", True):
            await self._append_chat_context(session_id, text)
        # modified_kwargs 会整体替换宿主侧 hook 的 kwargs，故必须带上其余参数
        return {"action": "continue", "modified_kwargs": {**kwargs, "message": new_message}}

    # ── 生命周期 ────────────────────────────────────────────────────────

    async def on_load(self) -> None:
        key_state = "已配置" if self._api_key() else "未配置"
        self.ctx.logger.info(
            "[豆包TTS] 插件已加载 | API Key: %s | 音色: %s | resource: %s",
            key_state,
            self._get("voice_tone", "voice", DEFAULT_VOICE_DISPLAY),
            self._get("doubao", "resource_id", DOUBAO_RESOURCE_PRESET),
        )
        if not self._api_key():
            self.ctx.logger.warning(
                "[豆包TTS] 尚未配置 API Key：请在插件配置 [doubao] api_key 填写火山引擎新版控制台的 API Key"
            )
        # 语音翻译状态 + 可用模型任务（方便用户填对「翻译用模型任务」）
        mode = self._translate_mode()
        if mode != "不翻译":
            tgt = str(self._get("translate", "target", "") or "").strip()
            name, _code = resolve_language(tgt)
            if name:
                self.ctx.logger.info(
                    "[豆包TTS] 语音翻译：方式=%s | 目标语种=%s | 模型任务=%s",
                    mode, name, self._get("translate", "model_task", "utils"),
                )
                if mode == "按概率翻译":
                    self.ctx.logger.info(
                        "[豆包TTS] 翻译概率 = %s", self._get("translate", "probability", 0.5)
                    )
            else:
                self.ctx.logger.warning(
                    "[豆包TTS] 翻译方式=%s，但「目标语种」为空或无法识别（%r），实际不会翻译", mode, tgt
                )
            try:
                models = await self.ctx.llm.get_available_models()
                if models:
                    self.ctx.logger.info("[豆包TTS] 可选模型任务: %s", "、".join(str(m) for m in models))
            except Exception:
                pass

    async def on_unload(self) -> None:
        self.ctx.logger.info("[豆包TTS] 插件已卸载")

    async def on_config_update(self, scope: str, config_data: Dict[str, Any], version: str) -> None:
        self.ctx.logger.info("[豆包TTS] 配置更新 scope=%s version=%s", scope, version)

    # ── 火山合成 ────────────────────────────────────────────────────────

    async def _synthesize_one(
        self, text: str, overrides: Optional[Dict[str, Any]] = None
    ) -> Tuple[bool, bytes, str]:
        """调用火山新版接口合成一段音频。

        Args:
            text: 要合成的文本
            overrides: 调用方按参数独立覆盖（key 可选 emotion/emotion_scale/speech_rate/loudness；
                       value=None 或缺省=用配置固定值）。供"麦麦自主"时 LLM 现场填。

        Returns:
            (ok, audio_bytes 或 b"", 错误信息或音色ID)
        """
        api_key = self._api_key()
        if not api_key:
            return False, b"", "API Key 未配置（插件配置 → 豆包语音 → api_key）"
        text = (text or "").strip()
        if not text:
            return False, b"", "文本为空"

        ov = overrides if isinstance(overrides, dict) else {}

        resource_id = str(self._get("doubao", "resource_id", DOUBAO_RESOURCE_PRESET)).strip()
        voice_type = _resolve_voice_type(str(self._get("voice_tone", "voice", DEFAULT_VOICE_DISPLAY)))
        audio_format = str(self._get("doubao", "audio_format", "mp3")).strip() or "mp3"
        sample_rate = int(self._get("doubao", "sample_rate", 24000) or 24000)
        timeout = float(self._get("behavior", "timeout_seconds", 30.0) or 30.0)

        audio_params: Dict[str, Any] = {"format": audio_format, "sample_rate": sample_rate}
        req_params: Dict[str, Any] = {
            "text": text,
            "speaker": voice_type,
            "audio_params": audio_params,
        }
        # ⚠️ v1.4.1 修复：官方接口要求 emotion / emotion_scale / speech_rate / loudness_rate
        #    全部放在 req_params.audio_params 内。v1.4.0 误放在 req_params 顶层、且语速/音量用了
        #    旧接口的字段名（speed_ratio / volume_ratio）与倍率值，会被服务端静默忽略 → 设置无效。
        # 情感：调用方覆盖优先，否则用配置固定值；"none"/空 = 不带情感
        emotion_cfg = ""
        if ov.get("emotion") is not None:
            emotion_cfg = str(ov["emotion"] or "").strip()
        if not emotion_cfg or emotion_cfg == "none":
            emotion_cfg = str(self._get("voice_tone", "emotion", "") or "").strip()
        if emotion_cfg and emotion_cfg != "none":
            audio_params["emotion"] = PRESET_EMOTIONS.get(emotion_cfg, emotion_cfg)
            # 情感强度：调用方覆盖优先（1~5）
            scale = None
            if ov.get("emotion_scale") is not None:
                try:
                    scale = float(ov["emotion_scale"])
                except (TypeError, ValueError):
                    scale = None
            if scale is None:
                scale = float(self._get("voice_tone", "emotion_scale", 1.0) or 1.0)
            if 1.0 <= scale <= 5.0:
                audio_params["emotion_scale"] = scale
        # 语速：-50~100 的整数（0=正常，100=2.0 倍速）——官方字段名 speech_rate
        rate = None
        if ov.get("speech_rate") is not None:
            try:
                rate = float(ov["speech_rate"])
            except (TypeError, ValueError):
                rate = None
        if rate is None:
            rate = float(self._get("speed_loud", "speech_rate", 0.0) or 0.0)
        if rate:
            audio_params["speech_rate"] = int(max(-50, min(100, round(rate))))
        # 音量：-50~100 的整数（0=正常，100=2.0 倍）——官方字段名 loudness_rate
        vol = None
        if ov.get("loudness") is not None:
            try:
                vol = float(ov["loudness"])
            except (TypeError, ValueError):
                vol = None
        if vol is None:
            vol = float(self._get("speed_loud", "loudness", 0.0) or 0.0)
        if vol:
            audio_params["loudness_rate"] = int(max(-50, min(100, round(vol))))
        # additions（官方要求是 JSON 字符串）：
        #  · disable_markdown_filter：过滤麦麦回复里的 **加粗**、# 标题等，否则会被逐字念出来；
        #  · explicit_language：文本是日语等非中英语种时必须显式指定，否则发音明显退化（实测结论）。
        additions: Dict[str, Any] = {"disable_markdown_filter": True}
        detected_lang = detect_language(text)
        if detected_lang:
            additions["explicit_language"] = detected_lang
        req_params["additions"] = json.dumps(additions, ensure_ascii=False)

        headers = {
            "Content-Type": "application/json",
            "X-Api-Key": api_key,
            "X-Api-Resource-Id": resource_id,
            "X-Api-Request-Id": str(uuid.uuid4()),
        }
        # v1.4.1 新增：音色与 Resource ID 不匹配时提前告警（这是最常见的失败原因，
        # 否则用户只能看到一句 resource ID is mismatched，不知道该怎么改）
        is_clone_voice = voice_type.upper().startswith("S_")
        if is_clone_voice and resource_id == DOUBAO_RESOURCE_PRESET:
            self.ctx.logger.warning(
                "[豆包TTS] 音色 %s 是复刻音色，但 Resource ID 是 %s —— 会报 "
                "'resource ID is mismatched with speaker related resource'。"
                "请把「豆包语音」页的 Resource ID 改成 %s",
                voice_type, resource_id, DOUBAO_RESOURCE_CLONE,
            )
        elif (not is_clone_voice) and resource_id == DOUBAO_RESOURCE_CLONE:
            self.ctx.logger.warning(
                "[豆包TTS] 音色 %s 是预置音色，但 Resource ID 是 %s（复刻用）—— 可能报资源不匹配，"
                "请把 Resource ID 改成 %s",
                voice_type, resource_id, DOUBAO_RESOURCE_PRESET,
            )

        payload = {"req_params": req_params}
        self.ctx.logger.info(
            "[豆包TTS] 合成请求: %d字 | %s | %s | 情感=%s | 语速=%s | 音量=%s | 语种=%s",
            len(text), resource_id, voice_type,
            audio_params.get("emotion", "-"),
            audio_params.get("speech_rate", 0),
            audio_params.get("loudness_rate", 0),
            detected_lang or "默认（不指定）",
        )

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    DOUBAO_TTS_URL,
                    json=payload,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=timeout),
                ) as resp:
                    if resp.status != 200:
                        body = (await resp.text(errors="replace")).strip()[:300]
                        self.ctx.logger.error("[豆包TTS] HTTP %s: %s", resp.status, body)
                        return False, b"", f"火山接口 HTTP {resp.status}: {body}"

                    # NDJSON 流：code=0 携带 base64 音频段；20000000=结束；>0=业务错误
                    chunks: List[bytes] = []
                    async for raw in resp.content:
                        line = raw.strip()
                        if not line:
                            continue
                        try:
                            obj = json.loads(line.decode("utf-8"))
                        except Exception:
                            continue
                        code = obj.get("code")
                        if code == 20000000:
                            break
                        if code == 0:
                            data_b64 = obj.get("data")
                            if data_b64:
                                try:
                                    chunks.append(base64.b64decode(data_b64))
                                except Exception:
                                    self.ctx.logger.warning("[豆包TTS] base64 解码失败，跳过片段")
                        else:
                            msg = obj.get("message") or obj.get("error") or f"code={code}"
                            self.ctx.logger.error("[豆包TTS] 业务错误: %s", msg)
                            return False, b"", f"火山业务错误: {msg}"

                    audio = b"".join(chunks)
                    if not audio:
                        return False, b"", "火山返回空音频（检查音色与 resource_id 是否匹配：预置音色用 seed-tts-2.0，复刻音色用 seed-icl-2.0）"
                    self.ctx.logger.info("[豆包TTS] 合成成功 %d 字节", len(audio))
                    return True, audio, voice_type
        except asyncio.TimeoutError:
            self.ctx.logger.error("[豆包TTS] 请求超时（%ss）", timeout)
            return False, b"", f"请求超时（{timeout:.0f}s）"
        except aiohttp.ClientError as exc:
            self.ctx.logger.error("[豆包TTS] 网络错误: %s", exc)
            return False, b"", f"网络错误: {type(exc).__name__}"
        except Exception as exc:  # noqa: BLE001 兜底
            self.ctx.logger.error("[豆包TTS] 异常: %s", exc, exc_info=True)
            return False, b"", f"错误: {exc}"

    async def _send_voice(self, audio: bytes, stream_id: str, text: str = "") -> bool:
        """把音频 base64 后经 send.custom("voice") 发到会话，并把原文写回对话上下文。

        只发语音是不够的：宿主 send_service 的 sync_to_maisaka_history 默认关闭，
        且语音组件的可见文本只会渲染成「[语音消息]」，麦麦下一轮既不知道自己发过语音、
        也不知道说了什么。这里补两件事（可用 sync_chat_context 关闭）：
          1) processed_plain_text 让入库的语音消息带着文字（长期记忆能检索到这句话）；
          2) maisaka.context.append 把原文写回对话历史（planner / replyer 读的就是它）。
        """
        try:
            b64 = base64.b64encode(audio).decode("ascii")
            ok = bool(
                await self.ctx.send.custom("voice", b64, stream_id, processed_plain_text=text)
            )
        except Exception as exc:  # noqa: BLE001
            self.ctx.logger.error("[豆包TTS] 发送语音失败: %s", exc)
            return False

        if ok and text and self._get("behavior", "sync_chat_context", True):
            await self._append_chat_context(stream_id, text)
        return ok

    async def _append_chat_context(self, stream_id: str, text: str) -> None:
        """把刚说出去的话写回麦麦的对话上下文。"""

        prefix = str(self._get("behavior", "context_prefix", "") or "").strip()
        visible = f"{prefix}{text}" if prefix else text
        try:
            result = await self.ctx.maisaka.context.append(
                stream_id=stream_id,
                segments=[{"type": "text", "data": visible}],
                visible_text=visible,
                source_kind="guided_reply",
            )
            if isinstance(result, dict) and not result.get("success", True):
                self.ctx.logger.warning(
                    "[豆包TTS] 同步对话上下文失败: %s", result.get("error", "未知原因")
                )
        except Exception as exc:  # noqa: BLE001 同步失败不该影响已经发出去的语音
            self.ctx.logger.warning("[豆包TTS] 同步对话上下文异常: %s", exc)

    async def _speech(self, text: str, stream_id: str, overrides: Optional[Dict[str, Any]] = None) -> Tuple[bool, str]:
        """把整段文本切成多条语音逐段发送。返回 (是否全部成功, 说明)。"""
        max_len = max(1, int(self._get("behavior", "max_text_length", 150) or 150))
        segments = _split_sentences(text, max_len)
        if not segments:
            return False, "文本为空"
        total = len(segments)
        ok = 0
        last_voice = ""
        for i, seg in enumerate(segments):
            success, audio, info = await self._synthesize_one(seg, overrides=overrides)
            if not success:
                self.ctx.logger.warning("[豆包TTS] 第 %d/%d 段失败: %s", i + 1, total, info)
                continue
            if await self._send_voice(audio, stream_id, seg):
                ok += 1
                last_voice = info
            await asyncio.sleep(0.35)
        if ok == total and total > 0:
            return True, f"已发送 {total} 条语音（{last_voice}）"
        if ok > 0:
            return True, f"部分成功 {ok}/{total}"
        return False, "合成失败"

    def _translate_mode(self) -> str:
        """当前翻译方式：不翻译 / 全部翻译 / 按概率翻译 / 由麦麦判断。"""
        mode = str(self._get("translate", "mode", "不翻译") or "不翻译").strip()
        if mode not in ("不翻译", "全部翻译", "按概率翻译", "由麦麦判断"):
            return "不翻译"
        return mode

    async def _call_llm(self, prompt: str, max_tokens: int) -> str:
        """调用麦麦的模型；失败/为空时返回 ""。"""
        task = str(self._get("translate", "model_task", "utils") or "utils").strip()
        try:
            resp = await self.ctx.llm.generate(
                prompt=prompt, model=task, temperature=0.2, max_tokens=max_tokens
            )
        except Exception as exc:
            self.ctx.logger.warning("[豆包TTS] 模型调用失败: %s", exc)
            return ""
        if isinstance(resp, dict) and resp.get("success") is False:
            self.ctx.logger.warning("[豆包TTS] 模型调用失败: %s", resp.get("error") or resp)
            return ""
        if isinstance(resp, dict):
            return str(resp.get("response") or resp.get("content") or "").strip()
        return ""

    async def _translate_if_needed(self, text: str) -> str:
        """按「翻译方式」决定是否把文本翻译成目标语种。

        任何异常/失败/未配置 → 一律返回原文（绝不阻塞语音）。
        """
        mode = self._translate_mode()
        if mode == "不翻译":
            return text
        target_raw = str(self._get("translate", "target", "") or "").strip()
        lang_name, lang_code = resolve_language(target_raw)
        if not lang_name:
            self.ctx.logger.warning(
                "[豆包TTS] 翻译方式=%s，但「目标语种」为空或无法识别（%r），本次不翻译", mode, target_raw
            )
            return text
        if _looks_like_language(text, lang_code):
            self.ctx.logger.info("[豆包TTS] 文本已是%s，跳过翻译", lang_name)
            return text

        # 按概率翻译：没命中就原样发出（中文/外语交替出现）
        if mode == "按概率翻译":
            try:
                prob = float(self._get("translate", "probability", 0.5))
            except (TypeError, ValueError):
                prob = 0.5
            prob = max(0.0, min(1.0, prob))
            hit = random.random() < prob
            self.ctx.logger.info(
                "[豆包TTS] 按概率翻译：概率 %.2f，本次%s", prob, "命中→翻译" if hit else "未命中→保留原文"
            )
            if not hit:
                return text

        # 译文长度自适应：原文最长 500 字，译成外语通常膨胀 2~2.5 倍
        cfg_tokens = int(self._get("translate", "max_tokens", 800) or 800)
        max_tokens = max(cfg_tokens, min(4000, int(len(text) * 2.5) + 60))

        if mode == "由麦麦判断":
            # 一次调用同时完成「判断 + 翻译」：不需要翻译时让它只输出 KEEP
            rule = str(self._get("translate", "rule", "") or "").strip() or (
                "日常口语化的内容适合翻译；纯信息类内容（数字、链接、代码、专有名词、报错提示）保留原文"
            )
            prompt = (
                f"你要决定下面这句话是否需要翻译成{lang_name}。\n"
                f"判断依据：{rule}\n"
                f"· 需要翻译 → 只输出{lang_name}译文本身（不要解释、不要引号、不要 Markdown 符号）\n"
                "· 不需要翻译 → 只输出 KEEP 这四个字母\n\n"
                f"待判断的句子：{text}"
            )
            out = await self._call_llm(prompt, max_tokens)
            if not out:
                self.ctx.logger.warning("[豆包TTS] 麦麦判断失败（改用原文合成）")
                return text
            if "KEEP" in out[:12].upper():
                self.ctx.logger.info("[豆包TTS] 麦麦判断：本条保留原文")
                return text
            self.ctx.logger.info("[豆包TTS] 麦麦判断：翻译为%s → %s", lang_name, out[:60])
            return out

        # 全部翻译（或按概率已命中）
        prompt = (
            f"你是专业翻译引擎。请把下面的文本翻译成{lang_name}，"
            "只输出译文本身：不要解释、不要加引号、不要保留 Markdown 符号或表情符号。\n\n"
            f"{text}"
        )
        out = await self._call_llm(prompt, max_tokens)
        if not out:
            self.ctx.logger.warning("[豆包TTS] 翻译失败（改用原文合成）")
            return text
        self.ctx.logger.info("[豆包TTS] 已翻译为%s：%s", lang_name, out[:60])
        return out

    async def _handle_speech(
        self, text: str, stream_id: str, source: str, overrides: Optional[Dict[str, Any]] = None
    ) -> Tuple[bool, str]:
        """统一入口（手动命令 / Tool 都走这里）。

        overrides: 麦麦自主时由调用方传入的按参数覆盖（emotion/emotion_scale/speech_rate/loudness，
                   每个为 None/缺省=用配置固定值）；手动命令不传=全用配置固定值。
        """
        text = (text or "").strip()
        if not text:
            await self._maybe_error(stream_id, "没有要转成语音的文本")
            return False, "空文本"
        if len(text) > 500:
            await self._maybe_error(stream_id, f"文本太长（{len(text)}字），请精简到 500 字以内")
            return False, "文本过长"

        if not self._api_key():
            msg = "豆包语音 API Key 未配置，请先在插件配置里填写"
            await self._maybe_error(stream_id, msg)
            return False, "API Key 未配置"

        # 冷却限流（可选，默认关闭）：同一会话短时间内重复触发一律拒绝（防刷屏/刷费用）
        if self._cooldown_block(stream_id):
            cd = self._cooldown_seconds()
            self.ctx.logger.info("[豆包TTS] 触发冷却限流(%s): stream=%s", source, stream_id)
            if source == "命令":
                await self._maybe_error(stream_id, f"语音合成冷却中，请 {cd:.0f} 秒后再试")
            return False, "触发限流"

        ov = overrides if isinstance(overrides, dict) else {}
        self.ctx.logger.info(
            "[豆包TTS] %s 触发语音: %d字 覆盖=%s", source, len(text), json.dumps({k: v for k, v in ov.items() if v is not None}, ensure_ascii=False) or "-"
        )
        # 语音翻译（可选）：开启后先把文本翻成目标语种，再用音色说出来
        tts_text = await self._translate_if_needed(text)
        ok, note = await self._speech(tts_text, stream_id, overrides=ov)
        if ok:
            return True, note
        # 失败处理
        self.ctx.logger.warning("[豆包TTS] 语音合成失败(%s): %s", source, note)
        if self._get("behavior", "fallback_to_text", True):
            try:
                await self.ctx.send.text(text, stream_id)
                # 降级成文字时同样把原文写好上下文，行为才一致
                if self._get("behavior", "sync_chat_context", True):
                    await self._append_chat_context(stream_id, text)
                return True, "语音合成失败，已改为文字回复"
            except Exception:
                pass
        await self._maybe_error(stream_id, "语音合成失败了，请稍后再试")
        return False, note

    def _recently_prompted(self, stream_id: str) -> bool:
        """该会话最近是否已收到过失败提示（去重窗口内）。"""

        last = self._last_error_prompt_at.get(stream_id)
        return last is not None and time.time() - last < ERROR_PROMPT_DEDUPE_SECONDS

    async def _maybe_error(self, stream_id: str, msg: str) -> bool:
        """向用户发失败提示；同一会话 30 秒内只发一次，防 LLM 重试刷屏。

        Returns:
            bool: 是否真的发出了提示。
        """
        if not self._get("behavior", "send_error_prompt", True):
            return False
        if self._recently_prompted(stream_id):
            return False
        self._last_error_prompt_at[stream_id] = time.time()
        try:
            await self.ctx.send.text(msg, stream_id)
            return True
        except Exception:
            return False

    # ── Command：手动 ───────────────────────────────────────────────────

    @Command(
        "doubao_tts_say",
        description="用豆包语音把指定文本说出来",
        pattern=r"^/(说|语音|speak)\s+(?P<text>.+)\s*$",
    )
    async def _cmd_say(self, stream_id: str = "", matched_groups: Optional[Dict[str, Any]] = None, **kwargs: Any) -> Tuple[bool, str, bool]:
        """/说 文本 / 语音 文本 / speak 文本"""
        del kwargs
        if not self._get("behavior", "command_enabled", True):
            return False, "手动命令已禁用", True
        if not stream_id:
            return False, "缺少 stream_id", True
        groups = matched_groups or {}
        text = (groups.get("text") or "").strip()
        if not text:
            await self._maybe_error(stream_id, "用法：/说 要转成语音的文本")
            return False, "缺少文本", True
        ok, note = await self._handle_speech(text, stream_id, "命令")
        return ok, note, True

    @Command(
        "doubao_tts_help",
        description="查看豆包语音帮助",
        pattern=r"^/(语音帮助|说帮助|tts帮助)\s*$",
    )
    async def _cmd_help(self, stream_id: str = "", **kwargs: Any) -> Tuple[bool, str, bool]:
        """/语音帮助"""
        del kwargs
        if not stream_id:
            return False, "缺少 stream_id", True
        voice_list = "、".join(PRESET_VOICES.keys())
        emotion_list = "、".join(PRESET_EMOTIONS.keys())
        mode = self._translate_mode()
        if mode == "不翻译":
            translate_line = "不翻译（原样合成）"
        else:
            tgt_name, _tgt_code = resolve_language(
                str(self._get("translate", "target", "") or "")
            )
            translate_line = f"{mode} → {tgt_name or '（目标语种未填，不会翻译）'}"
        text = (
            "【豆包语音】\n"
            f"- 用法：/说 文本 或 /语音 文本\n"
            f"- API Key：{'已配置' if self._api_key() else '未配置'}\n"
            f"- 当前音色：{self._get('voice_tone', 'voice', DEFAULT_VOICE_DISPLAY)}"
            f"（Resource ID：{self._get('doubao', 'resource_id', DOUBAO_RESOURCE_PRESET)}）\n"
            f"- 语音翻译：{translate_line}\n"
            f"- 预置音色：{voice_list}\n"
            f"- 情感（可选）：{emotion_list}\n"
            "- 换音色/情感/翻译：WebUI 插件配置里修改即可（改完即时生效）"
        )
        await self.ctx.send.text(text, stream_id)
        return True, "已发送帮助", True

    # ── Tool：麦麦自主 ──────────────────────────────────────────────────

    @Tool(
        "doubao_tts_speak",
        description="用豆包语音把文本说出来（发语音消息），适合朗读或更生动的回复",
        brief_description="用语音（豆包TTS）说话",
        detailed_description=(
            "当用户明确要求“用语音/说话/朗读/语音回复”时使用。"
            "文本宜为一句完整的话（5~80字）。若内容很长（>150字），只取其中最想强调的一句话来朗读，其余仍用文字。"
            "可选按对话氛围调节语气：emotion（情感，如安慰时平静、玩闹时开心）、emotion_scale（强度1~5，配合 emotion）。"
            "每个参数可单独给，未给的使用插件配置里的固定值；拿不准就都省略，用默认语气即可。"
            "（语速、音量为固定设置，不由本工具调节。）"
        ),
        parameters=[
            ToolParameterInfo(
                name="text",
                param_type=ToolParamType.STRING,
                description="要转成语音朗读的文本（一句完整的话）",
                required=True,
            ),
            ToolParameterInfo(
                name="emotion",
                param_type=ToolParamType.STRING,
                description="可选：情感语气（按对话氛围选一个）：开心/伤心/生气/害怕/惊讶/讨厌/哭泣/抱歉/平静/播音/讲故事",
                required=False,
                enum_values=list(PRESET_EMOTIONS.keys()),
            ),
            ToolParameterInfo(
                name="emotion_scale",
                param_type=ToolParamType.INTEGER,
                description="可选：情感强度 1~5（配合 emotion 使用，1=最淡）",
                required=False,
            ),
        ],
    )
    async def _tool_speak(
        self,
        text: str = "",
        emotion: str = "",
        emotion_scale: Any = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """LLM 调用：返回结构化结果给 LLM。"""
        stream_id = str(kwargs.get("stream_id") or kwargs.get("chat_id") or "")
        del kwargs
        if not stream_id:
            return {"success": False, "message": "缺少 stream_id，无法发送语音"}
        if not (text or "").strip():
            return {"success": False, "message": "text 为空，未发送"}
        # 麦麦自主语音只由「自主语音方式」管控（off=麦麦不自主）；
        # 「手动命令」开关只管 /说 命令，不该挡这里——否则关掉手动命令后，
        # 麦麦每次调用本工具都会失败（返回"语音功能当前已禁用"）。
        if self._auto_voice_mode() == "off":
            return {"success": False, "message": "麦麦自主语音已关闭（behavior.auto_voice_mode=off），请使用 /说 手动命令"}
        # 语速/音量为连续数值，仅支持手动固定（不在此工具参数中提供）
        # 按各参数的"来源模式"组装 overrides：auto=接受 LLM 传入；fixed=忽略、用配置固定值
        overrides: Dict[str, Any] = {}
        if self._get("behavior", "emotion_mode", "fixed") == "auto" and (emotion or "").strip():
            overrides["emotion"] = (emotion or "").strip()
        if self._get("behavior", "emotion_scale_mode", "fixed") == "auto" and emotion_scale is not None:
            overrides["emotion_scale"] = emotion_scale
        ok, note = await self._handle_speech(text, stream_id, "Tool", overrides=overrides or None)
        if ok:
            return {"success": True, "message": note}
        msg = f"语音失败：{note}"
        if "限流" in note:
            # 冷却期内重试必然还是失败，明确引导 LLM 别连续重试
            msg += "；冷却期内重试仍会失败，请直接转告用户稍后再试，不要连续重试"
        return {"success": False, "message": msg}


def create_plugin() -> MaiBotPlugin:
    return DoubaoTTSPlugin()
