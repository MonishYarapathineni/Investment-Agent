import yfinance as yf
import requests
from datetime import datetime, timedelta
from typing import Optional

from langchain_openai import ChatOpenAI
from config import NEWS_API_KEY
# agents.py (add near imports)
from memory import load_notes, save_notes

class DataAgent:
    """Data Agent : Collects data"""
    
    def __init__(self, llm):
        self.llm = llm
    
    def collect_stock_data(self, symbol):
        """Get stock data from Yahoo Finance"""
        try:
            stock = yf.Ticker(symbol)
            info = stock.info
            hist = stock.history(period="5d")
            
            price = round(hist['Close'].iloc[-1], 2) if not hist.empty else 0
            change = round(((hist['Close'].iloc[-1] / hist['Close'].iloc[0]) - 1) * 100, 2) if len(hist) > 1 else 0
            
            data = {
                "symbol": symbol,
                "price": price,
                "change": change,
                "volume": int(hist['Volume'].iloc[-1]) if not hist.empty else 0,
                "pe_ratio": round(info.get('trailingPE', 0), 2),
                "market_cap": info.get('marketCap', 0)
            }
            
            # Self-reflection
            missing = [k for k, v in data.items() if v == 0 or v == 'N/A']
            if not missing:
                reflection = "✅ All data collected successfully"
            else:
                reflection = f"⚠️ Some data missing: {', '.join(missing[:3])}"
            
            data['reflection'] = reflection
            return data
            
        except:
            return {
                "symbol": symbol, 
                "price": 0, 
                "change": 0, 
                "volume": 0, 
                "pe_ratio": 0, 
                "market_cap": 0,
                "reflection": "❌ Error collecting data"
            }
    
    # agents.py (replace DataAgent.collect_news)
    def collect_news(self, symbol):
        """Get and process multiple news items with explicit prompt chaining."""
        # Ingest
        articles = self.fetch_news_batch(symbol, n=5)

        # Preprocess
        cleaned = []
        seen = set()
        for a in articles:
            t = (a or "").strip()
            t = t.replace("\n", " ").strip()
            if t and t.lower() not in seen:
                cleaned.append(t)
                seen.add(t.lower())
        if not cleaned:
            cleaned = [f"{symbol} reports strong quarterly earnings and positive outlook"]

        # Classify each
        classifications = []
        for t in cleaned:
            cls = self.llm.invoke(f"Classify sentiment of this headline as positive, negative, or neutral: {t}").content
            classifications.append(cls.strip())

        # Extract key info per item
        key_infos = []
        for t in cleaned:
            ki = self.llm.invoke(f"Extract the single most important market-relevant fact from this headline: {t}").content
            key_infos.append(ki.strip())

        # Summarize across items
        joined = "; ".join(key_infos[:5])
        summary = self.llm.invoke(
            f"In one sentence, summarize the likely short-term market impact for {symbol} given: {joined}"
        ).content

        # Aggregate sentiment (simple mode)
        pos = sum("pos" in c.lower() for c in classifications)
        neg = sum("neg" in c.lower() for c in classifications)
        neu = len(classifications) - pos - neg
        agg = "positive" if pos > max(neg, neu) else "negative" if neg > max(pos, neu) else "neutral"

        return {
            "news": cleaned[:5],
            "classification": agg,
            "per_item_classifications": classifications[:5],
            "key_info": key_infos[:5],
            "summary": summary,
            "news_count": len(cleaned)
        }

    def fetch_news_batch(self, symbol, n=5):
        """Fetch up to n news titles from NewsAPI with basic fallback."""
        try:
            url = "https://newsapi.org/v2/everything"
            params = {
                'q': symbol,
                'from': (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d'),
                'sortBy': 'relevancy',
                'pageSize': max(1, min(n, 10)),
                'apiKey': NEWS_API_KEY
            }
            r = requests.get(url, params=params, timeout=10)
            if r.status_code == 200:
                data = r.json()
                return [a.get('title') for a in data.get('articles', []) if a.get('title')]
        except Exception:
            pass
        # Fallbacks
        return [
            f"{symbol} reports strong quarterly earnings and positive outlook",
            f"{symbol} expands into new market amid sector tailwinds",
            f"Analysts update price targets for {symbol}",
        ][:n]

    
    def fetch_news(self, symbol):
        """Fetch news from NewsAPI"""
        try:
            url = "https://newsapi.org/v2/everything"
            params = {
                'q': symbol,
                'from': (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d'),
                'sortBy': 'relevancy',
                'pageSize': 1,
                'apiKey': NEWS_API_KEY
            }
            response = requests.get(url, params=params)
            if response.status_code == 200:
                data = response.json()
                if data.get('articles'):
                    return data['articles'][0].get('title', f"{symbol} shows growth")
        except:
            print("Error fetching news")
        return f"{symbol} reports strong quarterly earnings and positive outlook"


class AnalysisAgent:
    """Analysis Agent: Analyzes with routing"""
    
    def __init__(self, llm):
        self.llm = llm
    
    # In agents.py, inside class AnalysisAgent
    def evaluate(self, analysis: dict,
    stock_data: Optional[dict] = None,
    news_data: Optional[dict] = None,) -> str:
        """
        Evaluates analysis quality (1–10) with concrete improvement tips.
        Includes stock_data and news_data so feedback isn't only about P/E.
        Returns: 'Score: X/10. Feedback: ...'
        """
        try:
            ctx = {
                "analysis": analysis,
                "stock_data": stock_data or {},
                "news_sentiment": (news_data or {}).get("classification", "neutral"),
            }
            prompt = f"""
            Evaluate the following investment analysis for depth, reasoning, and use of evidence.
            Rate 1–10 and provide 2–3 specific improvements.

            Context:
            {ctx}

            Respond as:
            'Score: X/10. Feedback: ...'
            """
            return self.llm.invoke(prompt).content.strip()[:800]

        except Exception as e:
            return f"Score: 0/10. Feedback: Evaluation failed — {e}"



    def refine(
    self,
    analysis: dict,
    evaluation_feedback: str,
    stock_data: Optional[dict] = None,
    news_data: Optional[dict] = None,) -> dict:

        """
        Produces an improved, enriched analysis dict (type, valuation, rationale, considerations).
        Uses evaluation feedback + available stock/news context.
        """
        try:
            prompt = f"""
            You are a buy-side analyst. Improve the analysis by incorporating concrete reasoning that
            references the provided metrics and sentiment. Keep it concise but decision-useful.

            Current analysis: {analysis}
            Stock data: {stock_data}
            News sentiment: {(news_data or {}).get("classification", "neutral")}
            Feedback: {evaluation_feedback}

            Return a compact JSON-like dict with keys:
            - type (same as input)
            - valuation (same as or updated if warranted)
            - rationale (2–3 sentences weaving in metrics/sentiment/peers)
            - considerations (bullet-like short list: e.g., ['peer PE ~X', 'watch guidance', 'macro rates risk'])
            """
            improved = self.llm.invoke(prompt).content.strip()[:1200]
            # We keep the model's JSON-like text as-is to avoid brittle parsing.
            return {"refined_text": improved}
        except Exception as e:
            return {"refined_text": f"Refinement failed: {e}"}

    
    # agents.py (inside AnalysisAgent)
    def analyze(self, stock_data, news_data):
        # Decide route by content first
        route = self._route_by_content(news_data, stock_data)
        if route == "earnings":
            analysis = self.earnings_analysis(stock_data, news_data)
        elif route == "macro":
            analysis = self.macro_sensitivity(stock_data, news_data)
        elif route == "sec":
            analysis = self.sec_filings_check(stock_data, news_data)
        else:
            # fallback: fundamental vs technical
            if stock_data.get('pe_ratio', 0) > 0:
                route = "fundamental"
                analysis = self.fundamental_analysis(stock_data)
            else:
                route = "technical"
                analysis = self.technical_analysis(stock_data)

        analysis["news_sentiment"] = news_data.get("classification", "neutral")

        # Evaluator → Optimizer loop
        import re

        evaluation = self.evaluate(analysis, stock_data, news_data)
        m = re.search(r"Score:\s*(\d{1,2})\s*/\s*10", evaluation)
        score = int(m.group(1)) if m else 0

        optimized = analysis
        if score < 8:
            optimized = self.refine(analysis, evaluation, stock_data, news_data)

        return {
            "routed_to": route,
            "initial_analysis": analysis,
            "optimized_analysis": {"evaluation": evaluation, "final": optimized}
        }

    def _route_by_content(self, news_data, stock_data):
        """
        Decide routing based on news content first (earnings/SEC/macro),
        otherwise fall back to fundamentals/technicals using stock_data.
        """
        # Normalize news into a single string
        news = news_data.get("news", "")
        if isinstance(news, list):
            news_text = " ".join([str(x or "") for x in news])
        else:
            news_text = str(news or "")

        key_infos = news_data.get("key_info", [])
        if isinstance(key_infos, list):
            news_text += " " + " ".join([str(x or "") for x in key_infos])
        else:
            news_text += " " + str(key_infos or "")

        t = news_text.lower()

        # Content-driven routing
        if any(k in t for k in ["earnings", "eps", "revenue", "guidance"]):
            return "earnings"
        if any(k in t for k in ["10-k", "10q", "8-k", "sec filing", "edgar"]):
            return "sec"
        if any(k in t for k in ["fed", "inflation", "cpi", "gdp", "macro", "rates"]):
            return "macro"

        # Fallback: if content doesn’t suggest a specialist, let analyze() decide
        return "auto"


    def earnings_analysis(self, data, news):
        # Simple skeletal analyzer—expand with actual EPS/Rev deltas if parsed
        sentiment = news.get("classification", "neutral")
        return {"type": "earnings", "stance": f"Earnings-driven view with {sentiment} tone"}

    def sec_filings_check(self, data, news):
        return {"type": "sec", "note": "SEC/EDGAR signals detected; consider reviewing latest 10-Q/8-K sections."}

    def macro_sensitivity(self, data, news):
        chg = data.get("change", 0)
        return {"type": "macro", "beta_hint": "Potential macro sensitivity; monitor rates/newsflow", "recent_change": chg}

    # def refine(self, analysis, evaluation_feedback):
    #     prompt = f"""You are a portfolio PM. Improve this analysis using the feedback.
    #     Analysis: {analysis}
    #     Feedback: {evaluation_feedback}
    #     Return the improved analysis as a concise JSON-like dict with the same keys."""
    #     improved = self.llm.invoke(prompt).content[:800]
    #     return {"refined": improved, "based_on": analysis}

    def technical_analysis(self, data):
        """Technical analysis based on price movement"""
        change = data.get('change', 0)
        
        if change > 2:
            signal = "BUY - Positive momentum"
        elif change < -2:
            signal = "SELL - Negative momentum"
        else:
            signal = "HOLD - Neutral"
        
        return {"type": "technical", "signal": signal, "change": change}
    
    def fundamental_analysis(self, data):
        """Fundamental analysis based on PE ratio"""

        pe = data.get('pe_ratio', 0)
        mc = data.get('market_cap', 0)
        ch = data.get('change', 0)
        if 0 < pe < 15: valuation = "Undervalued"
        elif pe > 30:  valuation = "Overvalued"
        else:          valuation = "Fair value"
        return {
            "type": "fundamental",
            "valuation": valuation,
            "pe_ratio": pe,
            "market_cap": mc,
            "recent_change_pct": ch
    }

    
    # ReportAgent
    def self_evaluate(self, report):
        # keep your heuristic score...
        score = 0
        if report.get('summary') and len(report['summary']) > 50: score += 3
        rec = (report.get('recommendation') or "")
        if any(k in rec for k in ("BUY","SELL","HOLD")): score += 3
        if report.get('risks') and len(report['risks']) >= 2: score += 2
        if report.get('data_overview',{}).get('current_price') != '$0': score += 2
        score = min(score, 10)

        # Optional LLM critique to show evaluator transparency
        critique = self.llm.invoke(
            f"Briefly critique this report (1–2 sentences) for completeness and evidence: {report}"
        ).content[:240]
        report["critique"] = critique
        return score



class ReportAgent:
    """Report Agent: Creates report"""
    
    def __init__(self, llm):
        self.llm = llm
    
    def create_report(self, symbol, stock_data, analysis, news_data):
        """Generate investment report"""
        # Create report with consistent structure
        report = {
            "symbol": symbol,
            "date": datetime.now().strftime("%Y-%m-%d"),
            "summary": self.create_summary(stock_data, analysis, news_data),
            "data_overview": {
                "current_price": f"${stock_data.get('price', 0)}",
                "change_percent": f"{stock_data.get('change', 0)}%",
                "pe_ratio": stock_data.get('pe_ratio', 'N/A'),
                "volume": f"{stock_data.get('volume', 0):,}" if stock_data.get('volume') else "0"
            },
            "recommendation": self.create_recommendation(analysis, news_data),
            "risks": self.identify_risks(symbol),
        }
        # Add quality score after creating report
        report["quality_score"] = self.self_evaluate(report)
        
        return report
    
    def create_summary(self, stock_data, analysis, news_data):
        """Create executive summary"""
        prompt = f"""
        Write 2-sentence summary for {stock_data.get('symbol')}:
        Price change: {stock_data.get('change')}%
        Analysis: {analysis.get('initial_analysis')}
        News: {news_data.get('classification')}
        """
        return self.llm.invoke(prompt).content
    
    def create_recommendation(self, analysis, news_data):
        """Generate recommendation"""
        analysis_detail = analysis.get('initial_analysis', {})
        sentiment = news_data.get('classification', 'neutral')
        
        prompt = f"""
        Based on {analysis_detail} and news sentiment {sentiment},
        give recommendation (BUY/HOLD/SELL) with one-line reason.
        """
        return self.llm.invoke(prompt).content
    
    def identify_risks(self, symbol):
        """Identify 2 risks"""
        prompt = f"List 2 main investment risks for {symbol} in one line each."
        risks = self.llm.invoke(prompt).content.split('\n')
        return [r.strip() for r in risks[:2] if r.strip()]
    
    def self_evaluate(self, report):
        # keep your heuristic score...
        score = 0
        if report.get('summary') and len(report['summary']) > 50: score += 3
        rec = (report.get('recommendation') or "")
        if any(k in rec for k in ("BUY","SELL","HOLD")): score += 3
        if report.get('risks') and len(report['risks']) >= 2: score += 2
        if report.get('data_overview',{}).get('current_price') != '$0': score += 2
        score = min(score, 10)

        # Optional LLM critique to show evaluator transparency
        critique = self.llm.invoke(
            f"Briefly critique this report (1–2 sentences) for completeness and evidence: {report}"
        ).content[:240]
        report["critique"] = critique
        return score




class OrchestratorAgent:
    """Orchestrator Agent: Coordinates all other agents"""
    
    def __init__(self, llm):
        self.llm = llm
        self.data_agent = DataAgent(llm)
        self.analysis_agent = AnalysisAgent(llm)
        self.report_agent = ReportAgent(llm)
    
    # OrchestratorAgent
    def plan_research(self, symbol):
        prior = load_notes(symbol)
        prior_hint = prior.get("next_time_focus", "Check EDGAR filings if earnings are near.")
        plan_prompt = f"""
        Create a research plan for {symbol} with these steps:
        1. What data to collect
        2. What analysis to perform
        3. What to include in report
        Incorporate this prior note if useful: {prior_hint}
        Return as numbered list.
        """
        plan = self.llm.invoke(plan_prompt).content
        return [line for line in plan.split('\n') if line.strip()]

    
    def execute_research(self, symbol):
        """Orchestrate the entire research process"""
        # Plan the workflow
        workflow_plan = self.plan_research(symbol)
        
        # Execute data collection
        stock_data = self.data_agent.collect_stock_data(symbol)
        
        news_data = self.data_agent.collect_news(symbol)
        
        # Perform analysis
        analysis = self.analysis_agent.analyze(stock_data, news_data)
        
        #  Generate report
        report = self.report_agent.create_report(symbol, stock_data, analysis, news_data)
        
        #  Self-reflect on entire workflow
        reflection = self.reflect_on_workflow(workflow_plan, report)
        

        # OrchestratorAgent.execute_research (end of method, before return)
        improvement_prompt = f"Given this report, suggest one short 'next_time_focus' note to improve the next analysis for {symbol}."
        next_focus = self.llm.invoke(improvement_prompt).content.strip()[:500]
        save_notes(symbol, {"next_time_focus": next_focus})

        return {
            "plan": workflow_plan,
            "stock_data": stock_data,
            "news_data": news_data,
            "analysis": analysis,
            "report": report,
            "reflection": reflection,
            "memory": {"next_time_focus": next_focus}
        }

    def reflect_on_workflow(self, plan, report):
        """Reflect on the entire workflow execution"""
        prompt = f"""
        Evaluate workflow execution:
        Plan: {plan[:3]}
        Report quality: {report.get('quality_score')}/10
        What went well and what could improve?
        """
        return self.llm.invoke(prompt).content[:200]