import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import dash
from dash import dcc, html, Input, Output, callback

from pricing import black_scholes_price, monte_carlo_price, implied_volatility
from greeks import delta, gamma, vega, theta, greeks_surface
from hedging import simulate_delta_hedge

# ── App init ──────────────────────────────────────────────────────────────────
app = dash.Dash(__name__, title="Monte Carlo Options | GS-Ready")

DARK = "#0a0a0f"
CARD = "#12121a"
ACCENT = "#4fc3f7"
GREEN = "#00e676"
RED = "#ff5252"
YELLOW = "#ffd740"
TEXT = "#e0e0e0"
MUTED = "#888"

CARD_STYLE = {
    "background": CARD,
    "border": f"1px solid #1e1e2e",
    "borderRadius": "8px",
    "padding": "16px",
    "marginBottom": "12px",
}

LABEL_STYLE = {"color": MUTED, "fontSize": "11px", "marginBottom": "4px", "textTransform": "uppercase", "letterSpacing": "0.5px"}
VALUE_STYLE = {"color": TEXT, "fontSize": "22px", "fontWeight": "700"}

def slider(id, min, max, step, value, marks=None):
    return dcc.Slider(id=id, min=min, max=max, step=step, value=value,
                      marks=marks or {min: str(min), max: str(max)},
                      tooltip={"placement": "bottom", "always_visible": True},
                      className="custom-slider")

# ── Layout ────────────────────────────────────────────────────────────────────
app.layout = html.Div(style={"background": DARK, "minHeight": "100vh", "fontFamily": "'Inter', sans-serif", "color": TEXT}, children=[

    # Header
    html.Div(style={"borderBottom": "1px solid #1e1e2e", "padding": "20px 32px", "display": "flex", "alignItems": "center", "justifyContent": "space-between"}, children=[
        html.Div([
            html.H1("Monte Carlo Option Pricing Engine", style={"margin": 0, "fontSize": "20px", "fontWeight": "700", "color": "white"}),
            html.P("Black-Scholes vs MC Pricing · Greeks · Dynamic Delta Hedging Simulator", style={"margin": "4px 0 0", "color": MUTED, "fontSize": "13px"}),
        ]),
        html.Div("GS FLOW DESK TOOLKIT", style={"color": ACCENT, "fontSize": "11px", "letterSpacing": "2px", "fontWeight": "600"}),
    ]),

    # Main content
    html.Div(style={"display": "flex", "padding": "20px 32px", "gap": "20px"}, children=[

        # ── Left panel: controls ──────────────────────────────────────────────
        html.Div(style={"width": "260px", "flexShrink": 0}, children=[

            html.Div(style=CARD_STYLE, children=[
                html.P("OPTION PARAMETERS", style={**LABEL_STYLE, "color": ACCENT, "marginBottom": "14px"}),

                html.P("Option Type", style=LABEL_STYLE),
                dcc.RadioItems(id="option-type", options=[{"label": " Call", "value": "call"}, {"label": " Put", "value": "put"}],
                               value="call", inline=True, style={"color": TEXT, "marginBottom": "14px", "fontSize": "13px"},
                               inputStyle={"marginRight": "4px"}, labelStyle={"marginRight": "16px"}),

                html.P("Spot Price (S)", style=LABEL_STYLE),
                slider("spot", 50, 200, 1, 100, {50: "$50", 200: "$200"}),
                html.Br(),

                html.P("Strike Price (K)", style=LABEL_STYLE),
                slider("strike", 50, 200, 1, 100, {50: "$50", 200: "$200"}),
                html.Br(),

                html.P("Time to Expiry (years)", style=LABEL_STYLE),
                slider("tenor", 0.1, 3.0, 0.05, 1.0, {0.1: "1M", 1.0: "1Y", 3.0: "3Y"}),
                html.Br(),

                html.P("Risk-Free Rate", style=LABEL_STYLE),
                slider("rate", 0.0, 0.1, 0.005, 0.05, {0.0: "0%", 0.1: "10%"}),
                html.Br(),

                html.P("Model Vol (σ)", style=LABEL_STYLE),
                slider("vol-model", 0.05, 0.80, 0.01, 0.20, {0.05: "5%", 0.80: "80%"}),
                html.Br(),

                html.P("Market-Implied Vol (σ_mkt)", style=LABEL_STYLE),
                slider("vol-market", 0.05, 0.80, 0.01, 0.25, {0.05: "5%", 0.80: "80%"}),
                html.Br(),

                html.P("MC Simulations", style=LABEL_STYLE),
                slider("n-sims", 1000, 50000, 1000, 10000, {1000: "1K", 50000: "50K"}),
            ]),

            html.Div(style=CARD_STYLE, children=[
                html.P("HEDGING PARAMETERS", style={**LABEL_STYLE, "color": YELLOW, "marginBottom": "14px"}),
                html.P("Realized Vol (σ_real)", style=LABEL_STYLE),
                slider("vol-real", 0.05, 0.80, 0.01, 0.22, {0.05: "5%", 0.80: "80%"}),
                html.Br(),
                html.P("Rebalance Frequency (days)", style=LABEL_STYLE),
                slider("hedge-freq", 1, 21, 1, 5, {1: "Daily", 5: "Weekly", 21: "Monthly"}),
            ]),
        ]),

        # ── Right panel: charts & stats ───────────────────────────────────────
        html.Div(style={"flex": 1, "minWidth": 0}, children=[

            # Stat bar
            html.Div(id="stat-bar", style={"display": "flex", "gap": "12px", "marginBottom": "12px"}),

            # Tab charts
            dcc.Tabs(id="tabs", value="mc", style={"borderBottom": "1px solid #1e1e2e"}, children=[
                dcc.Tab(label="MC Paths & Pricing", value="mc",
                        style={"background": DARK, "color": MUTED, "border": "none", "padding": "8px 16px"},
                        selected_style={"background": CARD, "color": ACCENT, "border": "none", "borderTop": f"2px solid {ACCENT}"}),
                dcc.Tab(label="Greeks", value="greeks",
                        style={"background": DARK, "color": MUTED, "border": "none", "padding": "8px 16px"},
                        selected_style={"background": CARD, "color": ACCENT, "border": "none", "borderTop": f"2px solid {ACCENT}"}),
                dcc.Tab(label="Delta Hedging P&L", value="hedge",
                        style={"background": DARK, "color": MUTED, "border": "none", "padding": "8px 16px"},
                        selected_style={"background": CARD, "color": ACCENT, "border": "none", "borderTop": f"2px solid {ACCENT}"}),
                dcc.Tab(label="Vol Surface", value="volsurf",
                        style={"background": DARK, "color": MUTED, "border": "none", "padding": "8px 16px"},
                        selected_style={"background": CARD, "color": ACCENT, "border": "none", "borderTop": f"2px solid {ACCENT}"}),
            ]),

            dcc.Loading(html.Div(id="tab-content"), type="circle", color=ACCENT),
        ]),
    ]),
])


# ── Helpers ───────────────────────────────────────────────────────────────────
def dark_fig(fig):
    fig.update_layout(
        paper_bgcolor=CARD, plot_bgcolor="#0d0d14",
        font=dict(color=TEXT, family="Inter"),
        margin=dict(l=40, r=20, t=40, b=40),
        legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="#1e1e2e"),
        xaxis=dict(gridcolor="#1a1a2e", zerolinecolor="#1a1a2e"),
        yaxis=dict(gridcolor="#1a1a2e", zerolinecolor="#1a1a2e"),
    )
    return fig

def stat_card(label, value, color=TEXT, delta_str=None):
    children = [
        html.P(label, style=LABEL_STYLE),
        html.P(value, style={**VALUE_STYLE, "color": color}),
    ]
    if delta_str:
        children.append(html.P(delta_str, style={"color": MUTED, "fontSize": "11px", "margin": 0}))
    return html.Div(style={**CARD_STYLE, "flex": 1, "marginBottom": 0}, children=children)


# ── Main callback ─────────────────────────────────────────────────────────────
@callback(
    Output("stat-bar", "children"),
    Output("tab-content", "children"),
    Input("option-type", "value"),
    Input("spot", "value"),
    Input("strike", "value"),
    Input("tenor", "value"),
    Input("rate", "value"),
    Input("vol-model", "value"),
    Input("vol-market", "value"),
    Input("n-sims", "value"),
    Input("vol-real", "value"),
    Input("hedge-freq", "value"),
    Input("tabs", "value"),
)
def update(opt_type, S, K, T, r, vol_model, vol_mkt, n_sims, vol_real, hedge_freq, tab):
    # ── Core pricing ──────────────────────────────────────────────────────────
    bs_price = black_scholes_price(S, K, T, r, vol_model, opt_type)
    mkt_price = black_scholes_price(S, K, T, r, vol_mkt, opt_type)  # "market" price at implied vol
    mc_price, mc_err, paths = monte_carlo_price(S, K, T, r, vol_model, opt_type, n_simulations=min(n_sims, 10000))
    
    edge = mkt_price - bs_price
    edge_color = GREEN if edge > 0 else RED

    d = delta(S, K, T, r, vol_model, opt_type)
    g = gamma(S, K, T, r, vol_model)
    v = vega(S, K, T, r, vol_model)
    th = theta(S, K, T, r, vol_model, opt_type)

    # ── Stat bar ──────────────────────────────────────────────────────────────
    stats = [
        stat_card("BS Price (Model)", f"${bs_price:.4f}", ACCENT),
        stat_card("MC Price", f"${mc_price:.4f}", TEXT, f"±{mc_err:.4f} stderr"),
        stat_card("Market Price (Impl. Vol)", f"${mkt_price:.4f}", YELLOW),
        stat_card("Vol Edge (Mkt−Model)", f"${edge:.4f}", edge_color,
                  f"σ_mkt={vol_mkt:.0%} vs σ_model={vol_model:.0%}"),
        stat_card("Delta", f"{d:.4f}", TEXT),
        stat_card("Gamma", f"{g:.4f}", TEXT),
        stat_card("Vega / 1%", f"${v:.4f}", TEXT),
        stat_card("Theta / day", f"${th:.4f}", RED),
    ]

    # ── Tab content ───────────────────────────────────────────────────────────
    if tab == "mc":
        content = _mc_tab(S, K, T, paths, bs_price, mkt_price, mc_price, vol_model, vol_mkt)
    elif tab == "greeks":
        content = _greeks_tab(S, K, T, r, vol_model, opt_type)
    elif tab == "hedge":
        content = _hedge_tab(S, K, T, r, vol_model, vol_real, opt_type, hedge_freq)
    else:
        content = _volsurf_tab(S, K, T, r, opt_type)

    return stats, content


def _mc_tab(S, K, T, paths, bs_price, mkt_price, mc_price, vol_model, vol_mkt):
    fig = make_subplots(rows=1, cols=2, subplot_titles=["Simulated Price Paths (sample)", "Payoff Distribution at Expiry"])

    # Sample 80 paths
    sample = paths[::max(1, len(paths) // 80)]
    t_axis = np.linspace(0, T, paths.shape[1])

    for i, path in enumerate(sample):
        fig.add_trace(go.Scatter(
            x=t_axis, y=path,
            mode="lines", line=dict(width=0.6, color=f"rgba(79,195,247,{0.15 + 0.1*(i%3)})"),
            showlegend=False
        ), row=1, col=1)

    fig.add_hline(y=K, line_dash="dash", line_color=YELLOW, annotation_text=f"Strike K={K}", row=1, col=1)

    # Payoff distribution
    S_T = paths[:, -1]
    payoffs = np.maximum(S_T - K, 0) if True else np.maximum(K - S_T, 0)
    fig.add_trace(go.Histogram(x=payoffs, nbinsx=60, marker_color=ACCENT, opacity=0.75, name="Payoff"), row=1, col=2)
    fig.add_vline(x=bs_price * np.exp(0), line_color=GREEN, line_dash="dash", annotation_text=f"BS ${bs_price:.2f}", row=1, col=2)
    fig.add_vline(x=mkt_price, line_color=YELLOW, line_dash="dash", annotation_text=f"Mkt ${mkt_price:.2f}", row=1, col=2)

    fig.update_xaxes(title_text="Time (years)", row=1, col=1)
    fig.update_yaxes(title_text="Spot Price", row=1, col=1)
    fig.update_xaxes(title_text="Payoff ($)", row=1, col=2)
    fig.update_yaxes(title_text="Count", row=1, col=2)
    fig.update_layout(height=420, showlegend=False)

    return dcc.Graph(figure=dark_fig(fig), style={"marginTop": "8px"})


def _greeks_tab(S, K, T, r, vol_model, opt_type):
    sigma_range = np.linspace(0.05, 0.80, 100)
    g = greeks_surface(S, K, T, r, sigma_range, opt_type)
    spot_range = np.linspace(S * 0.5, S * 1.5, 100)

    fig = make_subplots(rows=2, cols=2, subplot_titles=["Delta vs Vol", "Gamma vs Vol", "Vega vs Vol ($/1% σ)", "Theta vs Vol ($/day)"])

    for row, col, key, color in [(1,1,"delta",ACCENT),(1,2,"gamma",GREEN),(2,1,"vega",YELLOW),(2,2,"theta",RED)]:
        fig.add_trace(go.Scatter(x=g["sigma"], y=g[key], mode="lines", line=dict(color=color, width=2), showlegend=False), row=row, col=col)
        fig.add_vline(x=vol_model, line_dash="dot", line_color="white", line_width=1, row=row, col=col)

    fig.update_layout(height=480)
    return dcc.Graph(figure=dark_fig(fig), style={"marginTop": "8px"})


def _hedge_tab(S, K, T, r, vol_model, vol_real, opt_type, hedge_freq):
    result = simulate_delta_hedge(S, K, T, r, vol_model, vol_real, opt_type,
                                   hedge_frequency=int(hedge_freq), n_simulations=300)

    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=["Hedging P&L Distribution", "Sample Hedge Paths"])

    pnl = result["pnl_all"]
    fig.add_trace(go.Histogram(x=pnl, nbinsx=50, marker_color=ACCENT, opacity=0.8, name="P&L"), row=1, col=1)
    fig.add_vline(x=0, line_color="white", line_dash="dash", row=1, col=1)
    fig.add_vline(x=pnl.mean(), line_color=GREEN, line_dash="dot",
                  annotation_text=f"Mean: ${pnl.mean():.3f}", row=1, col=1)

    # Sample running P&L paths
    for path in result["sample_paths"][:30]:
        color = GREEN if path["final_pnl"] > 0 else RED
        fig.add_trace(go.Scatter(
            y=path["pnl_running"], mode="lines",
            line=dict(width=0.7, color=color.replace(")", ",0.3)").replace("rgb", "rgba")),
            showlegend=False
        ), row=1, col=2)

    fig.add_hline(y=0, line_dash="dash", line_color="white", row=1, col=2)
    fig.update_layout(height=420, showlegend=False)

    freq_label = {1: "Daily", 5: "Weekly", 21: "Monthly"}.get(int(hedge_freq), f"Every {hedge_freq}d")
    summary = html.Div(style={"display": "flex", "gap": "12px", "marginTop": "8px"}, children=[
        stat_card("Mean Hedge P&L", f"${result['pnl_mean']:.4f}", GREEN if result['pnl_mean'] > 0 else RED),
        stat_card("P&L Std Dev", f"${result['pnl_std']:.4f}", YELLOW),
        stat_card("Hedge Sharpe", f"{result['sharpe']:.3f}", ACCENT),
        stat_card("Rebalance", freq_label, TEXT),
        stat_card("Vol Edge (model−real)", f"{(vol_model - vol_real)*100:.1f}%",
                  GREEN if vol_model > vol_real else RED),
    ])

    return html.Div([summary, dcc.Graph(figure=dark_fig(fig))])


def _volsurf_tab(S, K, T, r, opt_type):
    strikes = np.linspace(S * 0.7, S * 1.3, 20)
    tenors = np.linspace(0.1, 2.0, 15)

    # Simulate a vol surface with skew and term structure
    Z = np.zeros((len(tenors), len(strikes)))
    for i, t in enumerate(tenors):
        for j, k in enumerate(strikes):
            moneyness = np.log(k / S)
            skew = -0.1 * moneyness
            term = 0.02 * np.sqrt(t)
            Z[i, j] = 0.20 + skew + term + 0.01 * moneyness ** 2

    fig = go.Figure(data=[go.Surface(
        z=Z, x=strikes, y=tenors,
        colorscale="Viridis", opacity=0.9,
        contours=dict(z=dict(show=True, usecolormap=True, highlightcolor="white", project_z=True))
    )])
    fig.update_layout(
        scene=dict(
            xaxis_title="Strike", yaxis_title="Tenor (yr)", zaxis_title="Impl. Vol",
            xaxis=dict(backgroundcolor=CARD, gridcolor="#1a1a2e"),
            yaxis=dict(backgroundcolor=CARD, gridcolor="#1a1a2e"),
            zaxis=dict(backgroundcolor=CARD, gridcolor="#1a1a2e"),
            bgcolor=CARD,
        ),
        height=520,
        margin=dict(l=0, r=0, t=40, b=0),
    )
    fig.update_layout(paper_bgcolor=CARD, font=dict(color=TEXT))
    return dcc.Graph(figure=fig, style={"marginTop": "8px"})


if __name__ == "__main__":
    app.run(debug=True, port=8050)
