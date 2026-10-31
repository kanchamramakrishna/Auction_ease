from pathlib import Path

import pandas as pd
import streamlit as st

from filter_players import filter_players

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data" / "processed"

try:
    from rank_players import rank_players
except ImportError:
    rank_players = None

st.set_page_config(
    page_title="Auction Ease",
    page_icon="🏏",
    layout="wide",
    initial_sidebar_state="expanded",
)

GREEN = "#22C55E"
AMBER = "#F5B83D"
MUTED = "#7C857F"

st.markdown(
    f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Fraunces:ital,opsz,wght@0,9..144,600;0,9..144,800;1,9..144,700&family=DM+Sans:wght@400;500;700&family=DM+Mono:wght@400;500&display=swap');

    #MainMenu, footer {{ visibility: hidden; }}
    header[data-testid="stHeader"] {{ background: transparent; }}

    html, body, .stApp {{ font-family: 'DM Sans', sans-serif; }}
    .stApp {{
        background:
            radial-gradient(900px 500px at 8% 0%, rgba(34,197,94,.13), transparent 60%),
            radial-gradient(800px 500px at 100% 100%, rgba(245,184,61,.06), transparent 60%),
            linear-gradient(rgba(255,255,255,.035) 1px, transparent 1px) 0 0 / 44px 44px,
            linear-gradient(90deg, rgba(255,255,255,.035) 1px, transparent 1px) 0 0 / 44px 44px,
            #07090B;
    }}
    .block-container {{ padding-top: 2.5rem; max-width: 1200px; }}

    h1, h2, h3, h4 {{
        font-family: 'Fraunces', serif !important;
        letter-spacing: -0.02em; color: #F7F5EE;
    }}

    /* sidebar */
    [data-testid="stSidebar"] {{
        background: #0C0F10; border-right: 1px solid #1E2326;
    }}
    [data-testid="stSidebar"] h2 {{
        font-weight: 800; font-size: 1.9rem; margin-bottom: 0;
    }}
    [data-testid="stSidebar"] .stRadio label p,
    [data-testid="stSidebar"] .stSlider label p {{
        font-family: 'DM Mono', monospace; font-size: .85rem; color: #8B948E;
    }}

    /* hero */
    .ae-hero {{
        background: rgba(18,21,23,.92); border: 1px solid #23282C; border-radius: 26px;
        padding: 30px 36px; margin-bottom: 22px; display: flex;
        justify-content: space-between; align-items: center; gap: 24px; flex-wrap: wrap;
    }}
    .ae-hero .tag {{
        display: inline-block; padding: 6px 16px; border-radius: 999px;
        border: 1px solid rgba(34,197,94,.45); background: rgba(34,197,94,.08);
        color: {GREEN}; font-family: 'DM Mono', monospace; font-size: .76rem;
        letter-spacing: .14em; text-transform: uppercase;
    }}
    .ae-hero .nm {{
        font-family: 'Fraunces', serif; font-weight: 800; font-size: 2.8rem;
        line-height: 1.05; letter-spacing: -0.03em; color: #F7F5EE; margin: 14px 0 6px;
    }}
    .ae-hero .rl {{ color: {MUTED}; }}
    .ae-hero .sc {{ text-align: right; }}
    .ae-hero .sc b {{
        font-family: 'Fraunces', serif; font-size: 3.4rem; color: {GREEN};
        line-height: 1; display: block;
    }}
    .ae-hero .sc span {{ font-family: 'DM Mono', monospace; font-size: .74rem; color: {MUTED}; }}

    .ae-chip {{
        display: inline-block; padding: 4px 12px; border-radius: 999px;
        border: 1px solid #2A3033; color: #AEB6B1; font-size: .82rem; margin: 0 8px 8px 0;
    }}
    .ae-stat {{
        background: #121517; border: 1px solid #23282C; border-left: 4px solid {GREEN};
        border-radius: 14px; padding: 16px 18px;
    }}
    .ae-stat.b {{ border-left-color: {AMBER}; }}
    .ae-stat .v {{
        font-family: 'Fraunces', serif; font-size: 1.8rem; font-weight: 700;
        color: #F7F5EE; line-height: 1.1;
    }}
    .ae-stat .l {{ font-family: 'DM Mono', monospace; font-size: .74rem; color: {MUTED}; margin-top: 4px; }}
    .ae-label {{ font-family: 'DM Mono', monospace; font-size: .82rem; color: #8B948E; margin: 4px 0 10px; }}
    .ae-note {{ color: #566059; font-size: .88rem; }}

    button[data-baseweb="tab"] {{ font-family: 'DM Mono', monospace; font-size: .85rem; }}
    div.stButton > button, div.stDownloadButton > button {{
        border-radius: 12px; font-weight: 700; border: 1px solid #2A3033;
    }}
    div.stButton > button[kind="primary"] {{
        background: {GREEN}; color: #06210F; border: none;
    }}
    @media (prefers-reduced-motion: reduce) {{ * {{ transition: none !important; }} }}
    </style>
    """,
    unsafe_allow_html=True,
)


# ------------------------------------------------------------------
# Data
# ------------------------------------------------------------------
@st.cache_data
def load_data(path="player_features.csv"):
    candidate = Path(path)
    if candidate.is_absolute() or (candidate.parts and candidate.parts[0].lower() == "data"):
        resolved = BASE_DIR / candidate
    else:
        resolved = DATA_DIR / candidate
    return pd.read_csv(resolved)


def short_role(role):
    """'Bowler (Right-hand batsman, Right-arm fast)' -> ('Bowler', 'Right-hand bat', 'Right-arm fast')."""
    role = str(role)
    if "(" not in role:
        return role, "", ""
    kind, rest = role.split("(", 1)
    parts = [p.strip() for p in rest.rstrip(")").split(",")]
    bat = next((p for p in parts if "batsman" in p.lower()), "")
    bowl = next((p for p in parts if "batsman" not in p.lower()), "")
    return kind.strip(), bat.replace("batsman", "bat").strip(), bowl


def num(row, col, digits=1, default="–"):
    value = row.get(col)
    if value is None or pd.isna(value):
        return default
    return f"{value:,.{digits}f}" if digits else f"{int(value):,}"


def stat_card(label, value, bowling=False):
    return f'<div class="ae-stat{" b" if bowling else ""}"><div class="v">{value}</div><div class="l">{label}</div></div>'


try:
    data = load_data()
except FileNotFoundError:
    st.error("data/processed/player_features.csv not found. Ensure the processed dataset is present in the project.")
    st.stop()

if not {"player", "role"}.issubset(data.columns):
    st.error("The dataset must contain 'player' and 'role' columns.")
    st.stop()

if rank_players is None:
    st.error("rank_players.py not found. Ensure the project modules are in the same folder as App.py.")
    st.stop()

if "shortlist" not in st.session_state:
    st.session_state.shortlist = []

# ------------------------------------------------------------------
# Sidebar: auction requirements (same choices as auction_requirements.py)
# ------------------------------------------------------------------
with st.sidebar:
    st.markdown("## Auction Ease")
    st.caption("Tell us who you need. We rank the pool.")

    role_label = st.radio(
        "Player role",
        ["Any role", "Batter", "Wicketkeeper", "Bowler", "All-Rounder"],
    )
    required_role = "Any" if role_label == "Any role" else role_label

    bowling_type = "Any"
    if required_role in ("Bowler", "All-Rounder"):
        bowling_type = st.radio("Bowling style", ["Any", "Fast", "Spin"], horizontal=True)

    priority = st.radio("What matters most?", ["Balanced", "Batting", "Bowling"])

    top_n = st.slider("Players to show", 5, 50, 15, step=5)

    st.divider()
    st.markdown(f"**Shortlist:** {len(st.session_state.shortlist)} players")
    if st.session_state.shortlist:
        if st.button("Clear shortlist", width="stretch"):
            st.session_state.shortlist = []
            st.rerun()

requirements = {
    "required_role": required_role,
    "bowling_type": bowling_type,
    "priority": priority,
}

# ------------------------------------------------------------------
# Filter + rank
# ------------------------------------------------------------------
eligible = filter_players(data, requirements)

if eligible.empty:
    st.markdown("### No players match")
    st.write("Try a different role or set the bowling style to Any.")
    st.stop()

ranked = rank_players(eligible, priority)

if ranked.empty:
    st.markdown("### No ranked players available")
    st.stop()

ranked = ranked.reset_index(drop=True)
if "rank" not in ranked.columns:
    ranked.insert(0, "rank", range(1, len(ranked) + 1))

shown = ranked.head(top_n)
top = shown.iloc[0]
kind, bat, bowl = short_role(top["role"])
score = top.get("ranking_score")
score_html = (
    f'<div class="sc"><b>{score:.2f}</b><span>ranking score</span></div>'
    if score is not None and pd.notna(score)
    else ""
)

# ------------------------------------------------------------------
# Top recommendation
# ------------------------------------------------------------------
st.markdown(
    f"""
    <div class="ae-hero">
      <div>
        <div class="tag">Top pick · {priority.lower()} · {role_label.lower()}</div>
        <div class="nm">{top['player']}</div>
        <div class="rl">{kind}{' · ' + bat if bat else ''}{' · ' + bowl if bowl else ''}</div>
      </div>
      {score_html}
    </div>
    """,
    unsafe_allow_html=True,
)

tab_rank, tab_detail, tab_compare, tab_short = st.tabs(
    ["Rankings", "Player profile", "Compare", f"Shortlist ({len(st.session_state.shortlist)})"]
)

# ------------------------------------------------------------------
# Tab 1: rankings
# ------------------------------------------------------------------
with tab_rank:
    st.markdown(
        f"<div class='ae-note'>{len(ranked)} eligible players ranked. "
        f"Showing the top {len(shown)}.</div>",
        unsafe_allow_html=True,
    )

    table = shown.copy()
    table["style"] = shown["role"].apply(lambda r: " / ".join(x for x in short_role(r)[1:] if x))
    table["role"] = shown["role"].apply(lambda r: short_role(r)[0])

    columns = [c for c in [
        "rank", "player", "role", "style", "ranking_score", "career_runs",
        "career_strike_rate", "career_wickets", "recent_runs_per_match",
        "recent_wickets_per_match",
    ] if c in table.columns]

    config = {
        "rank": st.column_config.NumberColumn("Rank", width="small"),
        "player": st.column_config.TextColumn("Player"),
        "role": st.column_config.TextColumn("Role"),
        "style": st.column_config.TextColumn("Batting / bowling"),
        "career_runs": st.column_config.NumberColumn("Career runs", format="%d"),
        "career_strike_rate": st.column_config.NumberColumn("Career SR", format="%.1f"),
        "career_wickets": st.column_config.NumberColumn("Career wkts", format="%d"),
        "recent_runs_per_match": st.column_config.NumberColumn("Recent runs/match", format="%.1f"),
        "recent_wickets_per_match": st.column_config.NumberColumn("Recent wkts/match", format="%.2f"),
    }
    if "ranking_score" in table.columns:
        lo = float(min(0, table["ranking_score"].min()))
        hi = float(table["ranking_score"].max())
        config["ranking_score"] = st.column_config.ProgressColumn(
            "Score", format="%.2f", min_value=lo, max_value=hi if hi > lo else lo + 1
        )

    st.dataframe(
        table[columns], hide_index=True, width="stretch",
        height=min(40 + 35 * len(table), 700), column_config=config,
    )

    st.download_button(
        "Download this ranking (CSV)",
        ranked.to_csv(index=False).encode("utf-8"),
        file_name=f"auction_ease_{required_role.lower()}_{priority.lower()}.csv",
        mime="text/csv",
    )

# ------------------------------------------------------------------
# Tab 2: player profile
# ------------------------------------------------------------------
with tab_detail:
    chosen = st.selectbox("Choose a player", shown["player"].tolist(), key="profile_player")
    row = shown[shown["player"] == chosen].iloc[0]
    kind, bat, bowl = short_role(row["role"])

    st.markdown(f"### {chosen}")
    chips = [kind, bat, bowl, f"Rank {int(row['rank'])} of {len(ranked)}"]
    if "has_lstm_embedding" in row and row["has_lstm_embedding"] == 0:
        chips.append("Limited recent history")
    st.markdown("".join(f"<span class='ae-chip'>{c}</span>" for c in chips if c), unsafe_allow_html=True)

    st.markdown('<div class="ae-label">Batting</div>', unsafe_allow_html=True)
    c = st.columns(4)
    c[0].markdown(stat_card("Career runs", num(row, "career_runs", 0)), unsafe_allow_html=True)
    c[1].markdown(stat_card("Career strike rate", num(row, "career_strike_rate")), unsafe_allow_html=True)
    c[2].markdown(stat_card("Runs per match", num(row, "runs_per_match")), unsafe_allow_html=True)
    c[3].markdown(stat_card("Recent runs / match", num(row, "recent_runs_per_match")), unsafe_allow_html=True)

    st.markdown('<div class="ae-label" style="margin-top:18px">Bowling</div>', unsafe_allow_html=True)
    c = st.columns(4)
    c[0].markdown(stat_card("Career wickets", num(row, "career_wickets", 0), True), unsafe_allow_html=True)
    c[1].markdown(stat_card("Wickets per match", num(row, "wickets_per_match", 2), True), unsafe_allow_html=True)
    c[2].markdown(stat_card("Recent wickets / match", num(row, "recent_wickets_per_match", 2), True), unsafe_allow_html=True)
    c[3].markdown(stat_card("Matches played", num(row, "career_matches", 0), True), unsafe_allow_html=True)

    st.markdown("&nbsp;")
    left, right = st.columns(2)
    with left:
        st.markdown('<div class="ae-label">Latest season</div>', unsafe_allow_html=True)
        st.write(
            f"{num(row, 'latest_season', 0)}: {num(row, 'latest_season_runs', 0)} runs at a strike rate of "
            f"{num(row, 'latest_season_strike_rate')}, {num(row, 'latest_season_wickets', 0)} wickets."
        )
    with right:
        st.markdown('<div class="ae-label">Best venue</div>', unsafe_allow_html=True)
        if pd.notna(row.get("best_venue")):
            st.write(
                f"{row['best_venue']}: {num(row, 'best_venue_runs_per_match')} runs per match "
                f"over {num(row, 'best_venue_matches', 0)} matches."
            )
        else:
            st.write("No venue data.")

    st.markdown('<div class="ae-label" style="margin-top:18px">Career vs recent form</div>', unsafe_allow_html=True)
    chart = pd.DataFrame(
        {
            "Career": [row.get("runs_per_match", 0), row.get("wickets_per_match", 0) * 10],
            "Recent (last 5)": [row.get("recent_runs_per_match", 0), row.get("recent_wickets_per_match", 0) * 10],
        },
        index=["Runs per match", "Wickets per match (x10)"],
    ).fillna(0)
    st.bar_chart(chart, color=[GREEN, AMBER], horizontal=True)

    in_list = chosen in st.session_state.shortlist
    if st.button(
        "Remove from shortlist" if in_list else "Add to shortlist",
        type="secondary" if in_list else "primary",
        key="toggle_short",
    ):
        if in_list:
            st.session_state.shortlist.remove(chosen)
        else:
            st.session_state.shortlist.append(chosen)
        st.rerun()

# ------------------------------------------------------------------
# Tab 3: compare
# ------------------------------------------------------------------
with tab_compare:
    picks = st.multiselect(
        "Pick 2 to 4 players to compare",
        shown["player"].tolist(),
        default=shown["player"].head(3).tolist(),
        max_selections=4,
    )
    if len(picks) < 2:
        st.write("Select at least two players to see them side by side.")
    else:
        metrics = {
            "rank": "Rank", "ranking_score": "Ranking score", "career_matches": "Matches",
            "career_runs": "Career runs", "career_strike_rate": "Career strike rate",
            "runs_per_match": "Runs per match", "recent_runs_per_match": "Recent runs per match",
            "recent_strike_rate": "Recent strike rate", "career_wickets": "Career wickets",
            "wickets_per_match": "Wickets per match", "recent_wickets_per_match": "Recent wickets per match",
            "latest_season_runs": "Latest season runs", "latest_season_wickets": "Latest season wickets",
        }
        metrics = {k: v for k, v in metrics.items() if k in shown.columns}
        compare = (
            shown[shown["player"].isin(picks)].set_index("player")[list(metrics)]
            .rename(columns=metrics).T.round(2)
        )
        st.dataframe(compare, width="stretch")

# ------------------------------------------------------------------
# Tab 4: shortlist
# ------------------------------------------------------------------
with tab_short:
    if not st.session_state.shortlist:
        st.write("Nothing here yet. Open a player profile and choose Add to shortlist.")
    else:
        short_df = data[data["player"].isin(st.session_state.shortlist)]
        cols = [c for c in [
            "player", "role", "career_runs", "career_strike_rate", "career_wickets",
            "recent_runs_per_match", "recent_wickets_per_match",
        ] if c in short_df.columns]
        st.dataframe(short_df[cols].round(2), hide_index=True, width="stretch")
        st.download_button(
            "Download shortlist (CSV)",
            short_df[cols].to_csv(index=False).encode("utf-8"),
            file_name="auction_ease_shortlist.csv",
            mime="text/csv",
        )