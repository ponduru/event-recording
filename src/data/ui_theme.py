"""Spotify-inspired dark theme for Prismata Streamlit UI."""

import streamlit as st


# Color palette constants
COLORS = {
    # Backgrounds
    "bg_deep": "#0D0D0D",
    "bg_card": "#161616",
    "bg_elevated": "#1E1E1E",
    "border": "#2A2A2A",
    # Primary accent (Prismata purple)
    "primary": "#8B5CF6",
    "primary_hover": "#A78BFA",
    "primary_active": "#7C3AED",
    # Secondary accents
    "pink": "#EC4899",
    "cyan": "#06B6D4",
    "amber": "#F59E0B",
    "emerald": "#10B981",
    # Text
    "text_primary": "#FFFFFF",
    "text_secondary": "#A1A1AA",
    "text_muted": "#71717A",
}


def get_spotify_css() -> str:
    """Return comprehensive CSS for Spotify-inspired dark theme."""
    return f"""
    <style>
    /* ========== IMPORT FONTS ========== */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    /* ========== ROOT & GLOBAL OVERRIDES ========== */
    html, body, .stApp, [data-testid="stAppViewContainer"] {{
        background-color: {COLORS["bg_deep"]} !important;
        color: {COLORS["text_primary"]} !important;
    }}

    /* Apply Inter font only to specific text elements */
    p, h1, h2, h3, h4, h5, h6, label, a, li, td, th, input, button, textarea, select,
    .stMarkdown, .stText, [data-testid="stMarkdownContainer"] {{
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important;
    }}

    /* Text colors */
    p, label, a, li, td, th {{
        color: {COLORS["text_primary"]} !important;
    }}

    /* Hide broken icon text - Streamlit sometimes shows icon names as text */
    [data-testid="stExpanderToggleIcon"] {{
        font-size: 0 !important;
    }}

    [data-testid="stExpanderToggleIcon"] svg {{
        width: 1rem !important;
        height: 1rem !important;
    }}

    /* Remove default Streamlit padding */
    .block-container {{
        padding-top: 1rem;
        padding-bottom: 2rem;
    }}

    /* Main container */
    [data-testid="stMain"] {{
        background-color: {COLORS["bg_deep"]} !important;
    }}

    /* ========== TYPOGRAPHY ========== */
    h1, h2, h3, h4, h5, h6 {{
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
        font-weight: 700 !important;
        letter-spacing: -0.02em !important;
        color: {COLORS["text_primary"]} !important;
    }}

    h1 {{
        color: {COLORS["text_primary"]} !important;
        font-size: 2rem !important;
        font-weight: 700 !important;
    }}

    h2 {{
        font-size: 1.4rem !important;
        color: {COLORS["text_primary"]} !important;
        margin-top: 0.5rem !important;
        margin-bottom: 0.5rem !important;
    }}

    h3 {{
        font-size: 1.1rem !important;
        color: {COLORS["text_primary"]} !important;
        margin-top: 0.4rem !important;
        margin-bottom: 0.4rem !important;
    }}

    /* Paragraphs and text */
    p, .stMarkdown p {{
        color: {COLORS["text_primary"]} !important;
        font-size: 0.9rem !important;
        line-height: 1.5 !important;
    }}

    /* Caption/secondary text */
    .stCaption, [data-testid="stCaptionContainer"], small {{
        color: {COLORS["text_secondary"]} !important;
        font-size: 0.85rem !important;
    }}

    /* Code blocks */
    code, pre, .stCode {{
        background-color: {COLORS["bg_elevated"]} !important;
        color: {COLORS["primary_hover"]} !important;
        border-radius: 6px !important;
    }}

    /* ========== SIDEBAR ========== */
    [data-testid="stSidebar"] {{
        background-color: {COLORS["bg_card"]} !important;
        border-right: 1px solid {COLORS["border"]} !important;
    }}

    [data-testid="stSidebar"] * {{
        color: {COLORS["text_primary"]} !important;
    }}

    [data-testid="stSidebar"] .stCaption,
    [data-testid="stSidebar"] small {{
        color: {COLORS["text_secondary"]} !important;
    }}

    /* ========== CARDS & CONTAINERS ========== */
    .stExpander, [data-testid="stExpander"] {{
        background-color: {COLORS["bg_card"]} !important;
        border: 1px solid {COLORS["border"]} !important;
        border-radius: 12px !important;
        overflow: hidden;
    }}

    /* Expander header */
    [data-testid="stExpander"] > details > summary {{
        color: {COLORS["text_primary"]} !important;
        font-weight: 600 !important;
        padding: 0.75rem 1rem !important;
    }}

    [data-testid="stExpander"] > details > summary:hover {{
        background-color: {COLORS["bg_elevated"]} !important;
    }}

    /* Hide the default expander icon text, show only the arrow */
    [data-testid="stExpander"] summary svg {{
        color: {COLORS["text_secondary"]} !important;
    }}

    /* Expander content */
    [data-testid="stExpander"] > details > div {{
        background-color: {COLORS["bg_card"]} !important;
        padding: 0.5rem 1rem 1rem 1rem !important;
    }}

    /* Fix for stText elements */
    .stText, [data-testid="stText"] {{
        color: {COLORS["text_primary"]} !important;
        font-family: 'SF Mono', 'Fira Code', 'Consolas', monospace !important;
        font-size: 0.85rem !important;
    }}

    /* ========== BUTTONS ========== */
    /* All buttons base */
    .stButton > button {{
        font-family: 'Inter', sans-serif !important;
        font-weight: 600 !important;
        border-radius: 8px !important;
        padding: 0.5rem 1rem !important;
        transition: all 0.2s ease !important;
        color: {COLORS["text_primary"]} !important;
    }}

    /* Primary button (purple gradient) */
    .stButton > button[kind="primary"],
    .stButton > button[data-testid="baseButton-primary"] {{
        background: linear-gradient(135deg, {COLORS["primary"]} 0%, {COLORS["primary_active"]} 100%) !important;
        border: none !important;
        color: white !important;
        box-shadow: 0 4px 14px rgba(139, 92, 246, 0.4) !important;
    }}

    .stButton > button[kind="primary"]:hover,
    .stButton > button[data-testid="baseButton-primary"]:hover {{
        background: linear-gradient(135deg, {COLORS["primary_hover"]} 0%, {COLORS["primary"]} 100%) !important;
        box-shadow: 0 6px 20px rgba(139, 92, 246, 0.5) !important;
        transform: translateY(-1px);
    }}

    /* Secondary button */
    .stButton > button[kind="secondary"],
    .stButton > button[data-testid="baseButton-secondary"] {{
        background: {COLORS["bg_elevated"]} !important;
        border: 1px solid {COLORS["border"]} !important;
        color: {COLORS["text_primary"]} !important;
    }}

    .stButton > button[kind="secondary"]:hover,
    .stButton > button[data-testid="baseButton-secondary"]:hover {{
        border-color: {COLORS["primary"]} !important;
        background: {COLORS["bg_card"]} !important;
    }}

    /* Disabled button */
    .stButton > button:disabled {{
        opacity: 0.5 !important;
        cursor: not-allowed !important;
    }}

    /* ========== INPUTS ========== */
    /* Text inputs */
    .stTextInput > div > div > input,
    .stTextInput input {{
        background-color: {COLORS["bg_elevated"]} !important;
        border: 1px solid {COLORS["border"]} !important;
        border-radius: 8px !important;
        color: {COLORS["text_primary"]} !important;
        font-family: 'Inter', sans-serif !important;
    }}

    .stTextInput > div > div > input:focus,
    .stTextInput input:focus {{
        border-color: {COLORS["primary"]} !important;
        box-shadow: 0 0 0 2px rgba(139, 92, 246, 0.2) !important;
    }}

    .stTextInput > div > div > input::placeholder {{
        color: {COLORS["text_muted"]} !important;
    }}

    /* Labels */
    .stTextInput label,
    .stSelectbox label,
    .stSlider label,
    .stRadio label,
    .stCheckbox label {{
        color: {COLORS["text_primary"]} !important;
        font-weight: 500 !important;
        font-size: 0.9rem !important;
    }}

    /* Select boxes - the trigger/button */
    .stSelectbox > div > div,
    .stSelectbox [data-baseweb="select"] > div {{
        background-color: {COLORS["bg_elevated"]} !important;
        border: 1px solid {COLORS["border"]} !important;
        border-radius: 8px !important;
    }}

    .stSelectbox > div > div > div,
    .stSelectbox [data-baseweb="select"] span,
    .stSelectbox [data-baseweb="select"] div {{
        color: {COLORS["text_primary"]} !important;
    }}

    /* Select box control */
    [data-baseweb="select"] {{
        background-color: {COLORS["bg_elevated"]} !important;
    }}

    [data-baseweb="select"] > div {{
        background-color: {COLORS["bg_elevated"]} !important;
        border-color: {COLORS["border"]} !important;
    }}

    [data-baseweb="select"] [data-baseweb="icon"] {{
        color: {COLORS["text_secondary"]} !important;
    }}

    /* Dropdown menu popover */
    [data-baseweb="popover"] {{
        background-color: {COLORS["bg_card"]} !important;
        border: 1px solid {COLORS["border"]} !important;
        border-radius: 8px !important;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.5) !important;
    }}

    /* Dropdown menu list */
    [data-baseweb="menu"],
    [data-baseweb="popover"] ul {{
        background-color: {COLORS["bg_card"]} !important;
        padding: 0.5rem 0 !important;
    }}

    /* Dropdown menu items */
    [data-baseweb="menu"] li,
    [data-baseweb="popover"] li,
    [role="option"] {{
        background-color: {COLORS["bg_card"]} !important;
        color: {COLORS["text_primary"]} !important;
        padding: 0.5rem 1rem !important;
    }}

    [data-baseweb="menu"] li:hover,
    [data-baseweb="popover"] li:hover,
    [role="option"]:hover,
    [role="option"][aria-selected="true"] {{
        background-color: {COLORS["bg_elevated"]} !important;
        color: {COLORS["text_primary"]} !important;
    }}

    /* Option text */
    [data-baseweb="menu"] li div,
    [data-baseweb="menu"] li span,
    [role="option"] div,
    [role="option"] span {{
        color: {COLORS["text_primary"]} !important;
    }}

    /* Selected option highlight */
    [aria-selected="true"] {{
        background-color: {COLORS["primary"]} !important;
    }}

    [aria-selected="true"] div,
    [aria-selected="true"] span {{
        color: white !important;
    }}

    [data-baseweb="menu"] {{
        background-color: {COLORS["bg_card"]} !important;
    }}

    [data-baseweb="menu"] li {{
        color: {COLORS["text_primary"]} !important;
    }}

    [data-baseweb="menu"] li:hover {{
        background-color: {COLORS["bg_elevated"]} !important;
    }}

    /* ========== SLIDERS ========== */
    .stSlider > div > div > div[data-baseweb="slider"] {{
        background: transparent !important;
    }}

    .stSlider [data-testid="stTickBar"] {{
        background: {COLORS["border"]} !important;
    }}

    .stSlider [data-testid="stThumbValue"] {{
        color: {COLORS["text_primary"]} !important;
        font-weight: 600 !important;
    }}

    /* Slider track */
    .stSlider div[role="slider"] {{
        background: {COLORS["primary"]} !important;
    }}

    /* ========== TABS ========== */
    .stTabs [data-baseweb="tab-list"] {{
        background-color: {COLORS["bg_card"]} !important;
        border-radius: 12px !important;
        padding: 4px !important;
        gap: 4px !important;
        border: 1px solid {COLORS["border"]} !important;
    }}

    .stTabs [data-baseweb="tab"] {{
        background-color: transparent !important;
        border-radius: 8px !important;
        color: {COLORS["text_secondary"]} !important;
        font-weight: 600 !important;
        font-family: 'Inter', sans-serif !important;
        padding: 0.4rem 1.2rem !important;
        font-size: 0.8rem !important;
    }}

    .stTabs [data-baseweb="tab"]:hover {{
        color: {COLORS["text_primary"]} !important;
        background-color: {COLORS["bg_elevated"]} !important;
    }}

    .stTabs [aria-selected="true"] {{
        background-color: {COLORS["primary"]} !important;
        color: white !important;
    }}

    /* Tab underline - hide it */
    .stTabs [data-baseweb="tab-highlight"] {{
        display: none !important;
    }}

    .stTabs [data-baseweb="tab-border"] {{
        display: none !important;
    }}

    /* ========== METRICS ========== */
    [data-testid="stMetric"] {{
        background-color: {COLORS["bg_card"]} !important;
        border: 1px solid {COLORS["border"]} !important;
        border-left: 3px solid {COLORS["primary"]} !important;
        border-radius: 8px !important;
        padding: 0.75rem !important;
    }}

    [data-testid="stMetricLabel"] {{
        color: {COLORS["text_secondary"]} !important;
        font-weight: 600 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.05em !important;
        font-size: 0.75rem !important;
    }}

    [data-testid="stMetricValue"] {{
        color: {COLORS["text_primary"]} !important;
        font-weight: 700 !important;
        font-size: 1.5rem !important;
    }}

    [data-testid="stMetricDelta"] {{
        color: {COLORS["emerald"]} !important;
    }}

    /* ========== STATUS BADGES ========== */
    .status-badge {{
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 500px;
        font-weight: 600;
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }}

    .status-pending {{
        background-color: {COLORS["amber"]};
        color: #000 !important;
    }}

    .status-approved {{
        background-color: {COLORS["emerald"]};
        color: #fff !important;
    }}

    .status-rejected {{
        background-color: {COLORS["pink"]};
        color: #fff !important;
    }}

    /* ========== DOMAIN BADGE ========== */
    .domain-badge {{
        display: inline-block;
        background: rgba(139,92,246,0.15);
        padding: 0.2rem 0.6rem;
        border-radius: 4px;
        font-weight: 600;
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-left: 0.5rem;
        color: #A78BFA !important;
        vertical-align: middle;
        border: 1px solid rgba(139,92,246,0.25);
    }}

    /* ========== RADIO BUTTONS ========== */
    .stRadio > div {{
        background-color: {COLORS["bg_card"]} !important;
        border-radius: 8px !important;
        padding: 0.5rem !important;
        border: 1px solid {COLORS["border"]} !important;
    }}

    .stRadio > div > label {{
        color: {COLORS["text_primary"]} !important;
    }}

    .stRadio > div > label > div {{
        color: {COLORS["text_primary"]} !important;
    }}

    .stRadio [data-baseweb="radio"] {{
        background-color: {COLORS["bg_elevated"]} !important;
    }}

    /* ========== DIVIDERS ========== */
    hr, .stDivider {{
        border-color: {COLORS["border"]} !important;
        margin: 0.75rem 0 !important;
    }}

    /* ========== ALERTS & INFO BOXES ========== */
    .stAlert, [data-testid="stAlert"] {{
        border-radius: 12px !important;
        border: none !important;
        color: {COLORS["text_primary"]} !important;
    }}

    .stAlert p, [data-testid="stAlert"] p {{
        color: {COLORS["text_primary"]} !important;
    }}

    /* Success */
    [data-testid="stNotification"][data-type="success"],
    .element-container:has(.stSuccess) {{
        background-color: rgba(16, 185, 129, 0.15) !important;
        border-left: 3px solid {COLORS["emerald"]} !important;
    }}

    /* Info */
    [data-testid="stNotification"][data-type="info"],
    .element-container:has(.stInfo) {{
        background-color: rgba(6, 182, 212, 0.15) !important;
        border-left: 3px solid {COLORS["cyan"]} !important;
    }}

    /* Warning */
    [data-testid="stNotification"][data-type="warning"],
    .element-container:has(.stWarning) {{
        background-color: rgba(245, 158, 11, 0.15) !important;
        border-left: 3px solid {COLORS["amber"]} !important;
    }}

    /* Error */
    [data-testid="stNotification"][data-type="error"],
    .element-container:has(.stError) {{
        background-color: rgba(236, 72, 153, 0.15) !important;
        border-left: 3px solid {COLORS["pink"]} !important;
    }}

    /* ========== VIDEO & IMAGE CONTAINERS ========== */
    .stVideo, .stImage {{
        border-radius: 12px !important;
        overflow: hidden !important;
        border: 1px solid {COLORS["border"]} !important;
    }}

    /* ========== SCROLLBAR ========== */
    ::-webkit-scrollbar {{
        width: 8px;
        height: 8px;
    }}

    ::-webkit-scrollbar-track {{
        background: {COLORS["bg_card"]};
        border-radius: 4px;
    }}

    ::-webkit-scrollbar-thumb {{
        background: {COLORS["border"]};
        border-radius: 4px;
    }}

    ::-webkit-scrollbar-thumb:hover {{
        background: {COLORS["text_muted"]};
    }}

    /* ========== SPINNER ========== */
    .stSpinner > div {{
        border-top-color: {COLORS["primary"]} !important;
    }}

    /* ========== PROGRESS BAR ========== */
    .stProgress > div > div > div {{
        background: linear-gradient(90deg, {COLORS["primary"]} 0%, {COLORS["primary_active"]} 100%) !important;
    }}

    /* ========== TOOLTIPS ========== */
    [data-baseweb="tooltip"] {{
        background-color: {COLORS["bg_elevated"]} !important;
        color: {COLORS["text_primary"]} !important;
        border: 1px solid {COLORS["border"]} !important;
    }}

    /* ========== HIDE STREAMLIT BRANDING ========== */
    #MainMenu {{visibility: hidden;}}
    footer {{visibility: hidden;}}
    header {{visibility: hidden;}}

    /* ========== COLUMN LAYOUT FIXES ========== */
    [data-testid="column"] {{
        background-color: transparent !important;
    }}

    /* ========== MARKDOWN CONTENT ========== */
    .stMarkdown {{
        color: {COLORS["text_primary"]} !important;
    }}

    .stMarkdown strong, .stMarkdown b {{
        color: {COLORS["text_primary"]} !important;
        font-weight: 700 !important;
    }}

    .stMarkdown em, .stMarkdown i {{
        color: {COLORS["text_secondary"]} !important;
    }}

    .stMarkdown a {{
        color: {COLORS["primary"]} !important;
    }}

    .stMarkdown a:hover {{
        color: {COLORS["primary_hover"]} !important;
    }}

    /* ========== CHECKBOX ========== */
    .stCheckbox label span {{
        color: {COLORS["text_primary"]} !important;
    }}

    /* ========== TRAINING PROGRESS ========== */
    .training-progress-card {{
        background-color: {COLORS["bg_card"]};
        border: 1px solid {COLORS["border"]};
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
    }}

    .training-stat {{
        display: inline-block;
        padding: 0.5rem 1rem;
        background-color: {COLORS["bg_elevated"]};
        border-radius: 8px;
        margin: 0.25rem;
    }}

    .training-stat-label {{
        color: {COLORS["text_secondary"]};
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }}

    .training-stat-value {{
        color: {COLORS["text_primary"]};
        font-size: 1.25rem;
        font-weight: 700;
    }}

    .video-selection-card {{
        background-color: {COLORS["bg_card"]};
        border: 1px solid {COLORS["border"]};
        border-radius: 8px;
        padding: 0.75rem 1rem;
        margin: 0.25rem 0;
        display: flex;
        align-items: center;
        justify-content: space-between;
    }}

    .video-selection-card:hover {{
        border-color: {COLORS["primary"]};
    }}

    .video-exists {{
        color: {COLORS["emerald"]};
    }}

    .video-missing {{
        color: {COLORS["amber"]};
    }}

    /* ========== FORM ELEMENTS ========== */
    [data-testid="stForm"] {{
        background-color: {COLORS["bg_card"]} !important;
        border: 1px solid {COLORS["border"]} !important;
        border-radius: 12px !important;
        padding: 1rem !important;
    }}

    /* ========== DATA EDITOR / TABLE ========== */
    .stDataFrame {{
        background-color: {COLORS["bg_card"]} !important;
        border: 1px solid {COLORS["border"]} !important;
        border-radius: 12px !important;
    }}

    .stDataFrame th {{
        background-color: {COLORS["bg_elevated"]} !important;
        color: {COLORS["text_primary"]} !important;
    }}

    .stDataFrame td {{
        background-color: {COLORS["bg_card"]} !important;
        color: {COLORS["text_primary"]} !important;
    }}
    </style>
    """


def inject_theme():
    """Inject the Spotify-inspired theme CSS into the Streamlit app."""
    st.markdown(get_spotify_css(), unsafe_allow_html=True)


def status_badge(status: str) -> str:
    """Return HTML for a status badge.

    Args:
        status: One of 'pending', 'approved', 'rejected'

    Returns:
        HTML string for the badge
    """
    status_class = f"status-{status}"
    return f'<span class="status-badge {status_class}">{status.upper()}</span>'


def domain_badge(domain: str) -> str:
    """Return HTML for a domain badge.

    Args:
        domain: The domain name (e.g., 'cricket', 'soccer')

    Returns:
        HTML string for the badge
    """
    return f'<span class="domain-badge">{domain.upper()}</span>'


def metric_card(label: str, value: str, accent_color: str = None) -> str:
    """Return HTML for a custom metric card with optional accent color.

    Args:
        label: The metric label
        value: The metric value
        accent_color: Optional accent color (default: primary purple)

    Returns:
        HTML string for the metric card
    """
    color = accent_color or COLORS["primary"]
    return f'''
    <div style="
        background-color: {COLORS["bg_card"]};
        border: 1px solid {COLORS["border"]};
        border-left: 3px solid {color};
        border-radius: 12px;
        padding: 1rem;
    ">
        <div style="
            color: {COLORS["text_secondary"]};
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            font-size: 0.75rem;
            margin-bottom: 0.25rem;
        ">{label}</div>
        <div style="
            color: {COLORS["text_primary"]};
            font-weight: 700;
            font-size: 1.5rem;
        ">{value}</div>
    </div>
    '''


def styled_header(title: str, domain: str = None) -> None:
    """Render a styled header with optional domain badge.

    Args:
        title: The main title text
        domain: Optional domain to show as badge
    """
    if domain:
        st.markdown(
            f'<h1 style="display: inline;">{title}</h1>{domain_badge(domain)}',
            unsafe_allow_html=True
        )
    else:
        st.markdown(f'<h1>{title}</h1>', unsafe_allow_html=True)


def training_stat_card(label: str, value: str) -> str:
    """Return HTML for a training stat card.

    Args:
        label: The stat label
        value: The stat value

    Returns:
        HTML string for the stat card
    """
    return f'''
    <div class="training-stat">
        <div class="training-stat-label">{label}</div>
        <div class="training-stat-value">{value}</div>
    </div>
    '''


def video_status_icon(exists: bool) -> str:
    """Return an icon indicating video file status.

    Args:
        exists: Whether the video file exists

    Returns:
        HTML string for the status icon
    """
    if exists:
        return f'<span class="video-exists">✓</span>'
    else:
        return f'<span class="video-missing">⚠</span>'
