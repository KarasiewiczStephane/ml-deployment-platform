"""Streamlit dashboard for the ML deployment platform.

Visualizes deployment status, canary traffic splits, model performance
comparisons, and health check indicators using synthetic deployment data.
"""

import random
from datetime import datetime, timedelta, timezone

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(
    page_title="ML Deployment Platform Dashboard",
    page_icon="🚀",
    layout="wide",
)


@st.cache_data
def generate_deployment_status() -> dict:
    """Generate synthetic deployment status for staging and production."""
    return {
        "staging": {
            "model_version": "v2.4.0-rc1",
            "status": "healthy",
            "deployed_at": "2025-03-01T14:30:00Z",
            "replicas": {"ready": 2, "desired": 2},
            "avg_latency_ms": 23.4,
            "requests_per_min": 145,
            "error_rate": 0.002,
            "uptime_hours": 48.5,
        },
        "production": {
            "model_version": "v2.3.1",
            "status": "healthy",
            "deployed_at": "2025-02-20T09:15:00Z",
            "replicas": {"ready": 4, "desired": 4},
            "avg_latency_ms": 18.7,
            "requests_per_min": 892,
            "error_rate": 0.001,
            "uptime_hours": 216.3,
        },
    }


@st.cache_data
def generate_canary_data() -> dict:
    """Generate synthetic canary deployment metrics."""
    return {
        "is_active": True,
        "primary_version": "v2.3.1",
        "canary_version": "v2.4.0-rc1",
        "canary_percentage": 15,
        "primary_metrics": {
            "requests": 4250,
            "accuracy": 0.934,
            "avg_latency_ms": 18.7,
            "error_rate": 0.001,
            "p99_latency_ms": 45.2,
        },
        "canary_metrics": {
            "requests": 750,
            "accuracy": 0.941,
            "avg_latency_ms": 21.3,
            "error_rate": 0.002,
            "p99_latency_ms": 52.8,
        },
        "traffic_history": _generate_traffic_history(),
    }


def _generate_traffic_history() -> list[dict]:
    """Generate time-series traffic split data."""
    base_time = datetime(2025, 3, 1, 0, 0, 0, tzinfo=timezone.utc)
    history = []
    canary_pct = 5
    for i in range(24):
        if i > 0 and i % 6 == 0:
            canary_pct = min(canary_pct + 5, 15)
        history.append(
            {
                "timestamp": (base_time + timedelta(hours=i)).isoformat(),
                "primary_pct": 100 - canary_pct,
                "canary_pct": canary_pct,
                "primary_requests": random.randint(150, 220),
                "canary_requests": random.randint(
                    int(150 * canary_pct / 100),
                    int(220 * canary_pct / 100),
                ),
            }
        )
    return history


@st.cache_data
def generate_performance_comparison() -> list[dict]:
    """Generate performance metrics comparison between model versions."""
    return [
        {
            "metric": "Accuracy",
            "v2.3.1 (Primary)": 0.934,
            "v2.4.0-rc1 (Canary)": 0.941,
        },
        {
            "metric": "Precision",
            "v2.3.1 (Primary)": 0.928,
            "v2.4.0-rc1 (Canary)": 0.937,
        },
        {
            "metric": "Recall",
            "v2.3.1 (Primary)": 0.919,
            "v2.4.0-rc1 (Canary)": 0.929,
        },
        {
            "metric": "F1 Score",
            "v2.3.1 (Primary)": 0.923,
            "v2.4.0-rc1 (Canary)": 0.933,
        },
        {
            "metric": "Avg Latency (ms)",
            "v2.3.1 (Primary)": 18.7,
            "v2.4.0-rc1 (Canary)": 21.3,
        },
    ]


@st.cache_data
def generate_health_checks() -> list[dict]:
    """Generate synthetic health check status data."""
    checks = [
        {
            "service": "Model Server (Primary)",
            "status": "healthy",
            "latency_ms": 5.2,
            "last_check": "2025-03-02T10:00:00Z",
        },
        {
            "service": "Model Server (Canary)",
            "status": "healthy",
            "latency_ms": 6.1,
            "last_check": "2025-03-02T10:00:00Z",
        },
        {
            "service": "MLflow Registry",
            "status": "healthy",
            "latency_ms": 12.4,
            "last_check": "2025-03-02T10:00:00Z",
        },
        {
            "service": "Feature Store",
            "status": "healthy",
            "latency_ms": 8.9,
            "last_check": "2025-03-02T10:00:00Z",
        },
        {
            "service": "Prometheus",
            "status": "healthy",
            "latency_ms": 3.1,
            "last_check": "2025-03-02T10:00:00Z",
        },
        {
            "service": "Load Balancer",
            "status": "degraded",
            "latency_ms": 45.7,
            "last_check": "2025-03-02T10:00:00Z",
        },
    ]
    return checks


def render_header() -> None:
    """Render the dashboard header."""
    st.title("ML Deployment Platform Dashboard")
    st.caption("Real-time deployment monitoring, canary analysis, and health checks")


def render_deployment_cards(status: dict) -> None:
    """Render deployment status cards for staging and production."""
    st.subheader("Deployment Status")

    col1, col2 = st.columns(2)

    for col, env_name in [(col1, "staging"), (col2, "production")]:
        env = status[env_name]
        with col:
            status_color = "green" if env["status"] == "healthy" else "red"
            st.markdown(
                f"### {env_name.title()} :{status_color}[{env['status'].upper()}]"
            )
            st.markdown(f"**Model:** `{env['model_version']}`")

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Latency", f"{env['avg_latency_ms']:.1f}ms")
            m2.metric("Req/min", f"{env['requests_per_min']}")
            m3.metric("Error Rate", f"{env['error_rate']:.3%}")
            replicas = env["replicas"]
            m4.metric("Replicas", f"{replicas['ready']}/{replicas['desired']}")


def render_canary_traffic(canary_data: dict) -> None:
    """Render canary traffic split visualization."""
    st.subheader("Canary Traffic Split")

    if not canary_data["is_active"]:
        st.info("No active canary deployment")
        return

    col1, col2, col3 = st.columns(3)
    col1.metric("Primary Traffic", f"{100 - canary_data['canary_percentage']}%")
    col2.metric("Canary Traffic", f"{canary_data['canary_percentage']}%")
    col3.metric(
        "Canary Version",
        canary_data["canary_version"],
    )

    fig = go.Figure(
        data=[
            go.Pie(
                labels=["Primary", "Canary"],
                values=[
                    100 - canary_data["canary_percentage"],
                    canary_data["canary_percentage"],
                ],
                marker={"colors": ["#2196F3", "#FF9800"]},
                hole=0.5,
                textinfo="label+percent",
            )
        ]
    )
    fig.update_layout(
        height=300,
        margin={"l": 20, "r": 20, "t": 30, "b": 20},
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True)

    history = canary_data["traffic_history"]
    df = pd.DataFrame(history)
    df["hour"] = pd.to_datetime(df["timestamp"]).dt.strftime("%H:%M")

    fig_timeline = go.Figure()
    fig_timeline.add_trace(
        go.Scatter(
            x=df["hour"],
            y=df["primary_pct"],
            mode="lines",
            name="Primary %",
            fill="tozeroy",
            line={"color": "#2196F3"},
        )
    )
    fig_timeline.add_trace(
        go.Scatter(
            x=df["hour"],
            y=df["canary_pct"],
            mode="lines",
            name="Canary %",
            fill="tozeroy",
            line={"color": "#FF9800"},
        )
    )
    fig_timeline.update_layout(
        xaxis_title="Time",
        yaxis_title="Traffic %",
        height=300,
        margin={"l": 40, "r": 20, "t": 30, "b": 40},
    )
    st.plotly_chart(fig_timeline, use_container_width=True)


def render_performance_comparison(comparison_data: list[dict]) -> None:
    """Render model performance comparison between versions."""
    st.subheader("Model Performance Comparison")

    df = pd.DataFrame(comparison_data)

    score_metrics = df[df["metric"] != "Avg Latency (ms)"]
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            name="v2.3.1 (Primary)",
            x=score_metrics["metric"],
            y=score_metrics["v2.3.1 (Primary)"],
            marker_color="#2196F3",
        )
    )
    fig.add_trace(
        go.Bar(
            name="v2.4.0-rc1 (Canary)",
            x=score_metrics["metric"],
            y=score_metrics["v2.4.0-rc1 (Canary)"],
            marker_color="#FF9800",
        )
    )
    fig.update_layout(
        barmode="group",
        yaxis={"range": [0.85, 1.0], "title": "Score"},
        height=350,
        margin={"l": 40, "r": 20, "t": 30, "b": 40},
    )
    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(
        pd.DataFrame(comparison_data), use_container_width=True, hide_index=True
    )


def render_health_checks(checks: list[dict]) -> None:
    """Render health check status indicators."""
    st.subheader("Health Check Status")

    for check in checks:
        col1, col2, col3, col4 = st.columns([3, 1, 1, 2])
        col1.markdown(f"**{check['service']}**")

        status = check["status"]
        if status == "healthy":
            col2.markdown(":green[HEALTHY]")
        elif status == "degraded":
            col2.markdown(":orange[DEGRADED]")
        else:
            col2.markdown(":red[UNHEALTHY]")

        col3.markdown(f"`{check['latency_ms']:.1f}ms`")
        col4.markdown(f"_{check['last_check'][:19]}_")


def main() -> None:
    """Main dashboard entry point."""
    render_header()

    st.sidebar.markdown("### Dashboard Controls")
    show_deployments = st.sidebar.checkbox("Show deployments", value=True)
    show_canary = st.sidebar.checkbox("Show canary analysis", value=True)
    show_comparison = st.sidebar.checkbox("Show performance comparison", value=True)
    show_health = st.sidebar.checkbox("Show health checks", value=True)

    if show_deployments:
        status = generate_deployment_status()
        render_deployment_cards(status)
        st.markdown("---")

    if show_canary:
        canary_data = generate_canary_data()
        render_canary_traffic(canary_data)
        st.markdown("---")

    col_left, col_right = st.columns(2)

    with col_left:
        if show_comparison:
            comparison = generate_performance_comparison()
            render_performance_comparison(comparison)

    with col_right:
        if show_health:
            checks = generate_health_checks()
            render_health_checks(checks)


if __name__ == "__main__":
    main()
