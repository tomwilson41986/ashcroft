"""Pipeline-aware tools — predictions, executions, settlements, P&L."""

import json
from datetime import date

from mcp.server import Server
from mcp.types import TextContent


def register(server: Server):
    @server.tool()
    async def pipeline_today_predictions(
        target_date: str | None = None,
    ) -> list[TextContent]:
        """Pull today's predictions from S3.

        Args:
            target_date: Date (YYYY-MM-DD). Defaults to today.
        """
        from ultra_betting.data import s3
        dt = date.fromisoformat(target_date) if target_date else date.today()
        df = s3.read_csv("predictions", dt)
        if df.empty:
            return [TextContent(type="text", text=f"No predictions found for {dt}")]
        return [TextContent(type="text", text=df.to_string(index=False))]

    @server.tool()
    async def pipeline_today_executions(
        target_date: str | None = None,
    ) -> list[TextContent]:
        """Pull today's executed bets from S3.

        Args:
            target_date: Date (YYYY-MM-DD). Defaults to today.
        """
        from ultra_betting.data import s3
        dt = date.fromisoformat(target_date) if target_date else date.today()
        df = s3.read_csv("executions", dt)
        if df.empty:
            return [TextContent(type="text", text=f"No executions found for {dt}")]
        return [TextContent(type="text", text=df.to_string(index=False))]

    @server.tool()
    async def pipeline_today_settlements(
        target_date: str | None = None,
    ) -> list[TextContent]:
        """Pull today's settlements from S3.

        Args:
            target_date: Date (YYYY-MM-DD). Defaults to today.
        """
        from ultra_betting.data import s3
        dt = date.fromisoformat(target_date) if target_date else date.today()
        df = s3.read_csv("settlements", dt)
        if df.empty:
            return [TextContent(type="text", text=f"No settlements found for {dt}")]
        return [TextContent(type="text", text=df.to_string(index=False))]

    @server.tool()
    async def pipeline_pnl_history() -> list[TextContent]:
        """Pull cumulative P&L data from S3."""
        from ultra_betting.data import s3
        data = s3.read_json("cumulative_pnl.json")
        if not data:
            return [TextContent(type="text", text="No P&L history found")]
        return [TextContent(type="text", text=json.dumps(data, indent=2))]

    @server.tool()
    async def pipeline_explain_skip(
        runner_name: str,
        target_date: str | None = None,
    ) -> list[TextContent]:
        """Explain why a specific runner wasn't bet on.

        Args:
            runner_name: The horse name to look up.
            target_date: Date (YYYY-MM-DD). Defaults to today.
        """
        from ultra_betting.data import s3
        dt = date.fromisoformat(target_date) if target_date else date.today()

        # Check predictions
        preds_df = s3.read_csv("predictions", dt)
        if preds_df.empty:
            return [TextContent(type="text", text=f"No predictions found for {dt}")]

        name_lower = runner_name.lower()
        matching = preds_df[preds_df["runner_name"].str.lower().str.contains(name_lower, na=False)]

        if matching.empty:
            return [TextContent(type="text", text=f"Runner '{runner_name}' not found in predictions for {dt}")]

        # Check executions
        exec_df = s3.read_csv("executions", dt)
        pred_ids = set(matching["prediction_id"].dropna().astype(str))

        explanation = []
        for _, row in matching.iterrows():
            pred_id = str(row.get("prediction_id", ""))
            info = {
                "runner": row.get("runner_name", ""),
                "venue": row.get("venue", ""),
                "predicted_bfsp": row.get("predicted_bfsp"),
                "predicted_win_prob": row.get("predicted_win_prob"),
                "market_id": row.get("market_id", ""),
            }

            if not exec_df.empty and pred_id in set(exec_df["prediction_id"].dropna().astype(str)):
                exec_row = exec_df[exec_df["prediction_id"] == pred_id].iloc[0]
                info["status"] = "EXECUTED"
                info["side"] = exec_row.get("side", "")
                info["stake"] = exec_row.get("stake", 0)
                info["price"] = exec_row.get("price_requested", 0)
            elif not row.get("market_id"):
                info["status"] = "SKIPPED"
                info["reason"] = "No Betfair market match found"
            else:
                info["status"] = "SKIPPED"
                info["reason"] = "Did not pass rules engine (check edge/conditions)"

            explanation.append(info)

        return [TextContent(type="text", text=json.dumps(explanation, indent=2, default=str))]

    @server.tool()
    async def pipeline_trigger_predict() -> list[TextContent]:
        """Manually trigger the prediction pipeline for today."""
        from pipeline.predict import main as predict_main
        predict_main()
        return [TextContent(type="text", text="Prediction pipeline completed")]

    @server.tool()
    async def pipeline_trigger_execute(
        dry_run: bool = True,
    ) -> list[TextContent]:
        """Manually trigger the execution pipeline.

        Args:
            dry_run: If True, simulate bets without placing them.
        """
        import os
        os.environ["DRY_RUN"] = "true" if dry_run else "false"
        from pipeline.execute import main as execute_main
        execute_main()
        mode = "dry run" if dry_run else "live"
        return [TextContent(type="text", text=f"Execution pipeline completed ({mode})")]
