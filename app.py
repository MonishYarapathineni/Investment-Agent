import streamlit as st
from langchain_openai import ChatOpenAI
from agents import DataAgent, AnalysisAgent, ReportAgent
from config import OPENAI_API_KEY, NEWS_API_KEY

# app.py (add near top)
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

def price_chart(symbol):
    hist = yf.Ticker(symbol).history(period="6mo")
    if hist.empty:
        st.info("No price data available.")
        return
    fig, ax = plt.subplots()
    hist["Close"].plot(ax=ax)
    ax.set_title(f"{symbol} – 6M Close")
    ax.set_xlabel("Date"); ax.set_ylabel("Price")
    st.pyplot(fig)


# Page config
st.set_page_config(page_title="Investment Research Agent", page_icon="📈")

def main():
    st.title("📈 Investment Research Agent")
    
    # Input
    symbol = st.text_input("Enter Stock Symbol:", value="AAPL")
    
    if st.button("Analyze", type="primary"):
        if not symbol:
            st.error("Please enter a stock symbol")
            return
        
        try:
            # Initialize agents
            llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY, model_name="gpt-4o-mini", temperature=0)
            data_agent = DataAgent(llm)
            analysis_agent = AnalysisAgent(llm) 
            report_agent = ReportAgent(llm)
            
            # Run workflow
            with st.spinner("Collecting data..."):
                stock_data = data_agent.collect_stock_data(symbol)
                news_data = data_agent.collect_news(symbol)
            
            with st.spinner("Analyzing..."):
                analysis = analysis_agent.analyze(stock_data, news_data)
            
            with st.spinner("Generating report..."):
                report = report_agent.create_report(symbol, stock_data, analysis, news_data)
            
            # Display results
            st.success("✅ Analysis Complete!")
            
            # Summary
            st.subheader("Executive Summary")
            st.info(report['summary'])
            
            # After "Executive Summary" block
            st.subheader("Price Chart")
            price_chart(symbol)

            # Show multi-article chain
            st.subheader("News Chain (Ingest → Preprocess → Classify → Extract → Summarize)")
            st.write(f"**Articles ({news_data['news_count']}):**")
            for i, t in enumerate(news_data["news"], 1):
                st.write(f"{i}. {t}")
            st.write("**Per-item classifications:**", news_data.get("per_item_classifications", []))
            st.write("**Key infos:**", [k[:140] for k in news_data.get("key_info", [])])
            st.write("**Aggregate sentiment:**", news_data["classification"])
            st.info(f"**Summary:** {news_data['summary']}")

            # Routing + Eval/Optimize visibility
            st.subheader("Routing & Evaluator–Optimizer")
            st.write("**Routed to:**", analysis['routed_to'])
            st.write("**Initial analysis:**", analysis['initial_analysis'])
            opt = analysis.get("optimized_analysis", {})
            st.write("Evaluation:", opt.get("evaluation", ""))

            final_opt = opt.get("final", {})
            refined_text = final_opt.get("refined_text") if isinstance(final_opt, dict) else None

            if refined_text:
                st.info(refined_text)   # show once, nicely
            else:
                st.json(final_opt)      # or st.write(...) if it’s plain text



            # Show refined text when present
            if isinstance(final_opt, dict) and "refined_text" in final_opt:
                st.info(final_opt["refined_text"])
            else:
                st.write(final_opt)


            # Memory
            if 'memory' in locals() or 'memory' in report:
                pass  # orchestrator path returns memory; in this app you wire orchestrator later if you use it


            # Three columns for main info
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Price", report['data_overview']['current_price'])
                st.metric("Change", report['data_overview']['change_percent'])
                st.metric("P/E Ratio", report['data_overview']['pe_ratio'])
            
            with col2:
                st.write("**Analysis Type:**", analysis['routed_to'])
                findings = analysis['initial_analysis']
                if 'signal' in findings:
                    st.write("**Signal:**", findings['signal'])
                if 'valuation' in findings:
                    st.write("**Valuation:**", findings['valuation'])
            
            with col3:
                st.write("**News Sentiment:**", news_data['classification'])
                st.write("**Articles Found:**", news_data['news_count'])
            
            # Recommendation
            st.subheader("Recommendation")
            if 'BUY' in report['recommendation']:
                st.success(report['recommendation'])
            elif 'SELL' in report['recommendation']:
                st.error(report['recommendation'])
            else:
                st.info(report['recommendation'])
            
            # Risks
            st.subheader("Key Risks")
            for risk in report['risks']:
                st.warning(risk)
            
            # Workflow demonstration
            with st.expander("🔄 Workflow Patterns Demonstrated"):
                st.write("**1. Prompt Chaining (News Processing):**")
                st.write(f"• Classified as: {news_data['classification']}")
                st.write(f"• Extracted: {news_data['key_info'][:100]}...")
                st.write(f"• Summary: {news_data['summary']}")
                
                st.write("\n**2. Routing (Analysis):**")
                st.write(f"• Routed to: {analysis['routed_to']} specialist")
                
                st.write("\n**3. Evaluator-Optimizer:**")
                if 'evaluation' in analysis['optimized_analysis']:
                    st.write(f"• {analysis['optimized_analysis']['evaluation'][:150]}...")
                
                st.write("\n**4. Self-Reflection:**")
                st.write(f"• Data Quality: {stock_data['reflection']}")
                st.write(f"• Report Score: {report['quality_score']}/10")

            with st.expander("🧾 Full Report JSON"):
                st.json(report)
                st.json(stock_data)
            
        except Exception as e:
            st.error(f"Error: {str(e)}")

if __name__ == "__main__":
    main()