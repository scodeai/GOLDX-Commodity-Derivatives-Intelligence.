# GOLDX-Commodity-Derivatives-Intelligence.
GOLDX — Commodity Derivatives Intelligence

A liquidity-aware, explainable relative-value anomaly detection prototype for MCX Gold futures.

Team: TechNova
Event: Hack in Hills 2026
Status: Hackathon prototype — historical analysis and interactive dashboard; not a production trading system.


GOLDX –[ Beyond Price. Into Intelligence.]

Elevator pitch:
Gold futures contracts with different expiry dates naturally trade at different prices. The important question is not whether their prices differ, but when the difference becomes unusual relative to its own history—and whether both contracts traded sufficiently for the observation to deserve attention. GOLDX converts futures market records into calendar-spread analytics, rolling statistical alerts, liquidity checks, and an interactive dashboard that explains each candidate signal.


Table of contents:-
Problem statement
Solution and features
System workflow
Data and contract universe
How the analytics work
Verified prototype findings
Dashboard walkthrough
Technology stack
Project files
Getting started
Data quality and validation
Limitations and responsible interpretation
Future roadmap
Team and acknowledgements


Problem statement:-
Commodity futures exchanges list multiple contracts on the same underlying commodity, each with a different expiry. Their price difference—a calendar spread—changes over time because of carry, market expectations, supply and demand, liquidity, and other factors.


A trader or researcher monitoring multiple expiries faces several challenges:-
Monitoring: unusual spread movements are difficult to detect consistently by eye.
Comparability: contracts must be paired by the same underlying, not merely by similar product names.
Data quality: an inactive contract can retain a closing price even when no trade occurred that day.
Interpretability: a large price difference is not automatically an arbitrage opportunity.
Reproducibility: analytical claims should be traceable to the underlying observations and explicit assumptions.


GOLDX addresses these issues with a transparent, rules-based analytical pipeline and a dashboard for exploration.


Solution and features:-
Feature-  |What it does-
Contract aware analysis- |Compares expiry contracts of the same GOLD underlying.
Calendar spreads- |Calculates far month closing price minus near-month closing price.
Rolling anomaly detection- |Measures current spreads against a trailing historical baseline using Z-scores.
Liquidity checks- |Tracks near/far volumes, open interest, and whether both contracts traded.
Explainable candidates- |Shows the date, contract pair, spread, Z-score, and liquidity context.
Data-quality visibility- |Highlights zero-volume observations and insufficient historical windows.
Interactive exploration- |Provides overview charts, relative-value plots, candidate alerts, and liquidity views.
CSV export- |Makes market records and signal outputs available for inspection.

Important: The currently verified detection method is statistical, not a trained machine-learning predictor. ML extensions are future work.


System workflow

        Historical MCX Gold futures records
                       |
                       v
             Data loading and cleaning
        (dates, numeric fields, duplicates)
                       |
                       v
           Contract and expiry alignment
              (same GOLD underlying)
                       |
                       v
               Calendar spreads
              Far Close - Near Close
                       |
                       v
              Liquidity screening
            Both legs Volume > 0?
                       |
                       v
             Rolling spread statistics
           Previous 10 valid observations
                       |
                       v
              Z-score classification
          Potentially Low / High / Normal
                       |
                       v
             Liquidity-aware reporting
                       |
                       v
             Interactive dashboard
          Charts / Alerts / Quality / CSV

The prototype separates raw market data, derived spread/signal data, and dashboard presentation. This helps avoid presenting a chart as independent evidence of a calculation: the signal file can be checked against the underlying closes and volume records.


Data and contract universe

The current historical prototype dataset contains:
Property | Value
Underlying | MCX GOLD futures
Period | 1 September 2026 – 19 October 2026
Recorded trading dates | 34
Market-data rows | 102
Contract expiries | 4 December 2026; 5 February 2027; 5 April 2027
Analyzed expiry pairs | December–February; February–April
Pair-level signal rows | 68 (34 per pair)

(The market-data schema includes Date, Symbol, Expiry Date, Open, High, Low, Close, Previous Close, Volume(Lots), and Open Interest(Lots), alongside other source fields. The project CSVs hold cleaned observations and derived signal information.)


Why expiry matching matters
(GOLDX compares different expiry months of the same GOLD futures product. It does not treat different gold-related products, such as GOLDM and GOLDTEN, as interchangeable legs of the same calendar spread. The original dashboard's synthetic/demo example should not be confused with the project's actual GOLD data.)


How the analytics work

1. Prepare and validate observations
   a.Parse dates and expiry dates.

   b.Convert market prices, volume, and open interest to numeric values.

   c.Check for missing critical values and duplicate (Date, Expiry Date) records.

   d.Keep the two selected expiry pairs aligned by date.

   e.Track zero-volume days rather than assuming every reported close is executable.


2. Calculate calendar spreads

For each trading date and expiry pair:
Calendar Spread = Far-month Close − Near-month Close

For example, on 7 October 2026:
December 2026 GOLD close = 149,103
February 2027 GOLD close = 151,175

December–February spread = 151,175 − 149,103 = 2,072

A spread is considered valid for the current statistical pipeline only when both legs record positive volume. This is a preliminary quality condition, not evidence that the spread was executable at those closing prices.


3. Compute a lookback-only rolling Z-score

For the baseline analysis, the current spread is compared with the previous 10 valid spread observations, excluding the current observation from the reference window:


IMPORTANT NOTE:  {Z_t = (Spread_t − Mean(previous 10 valid spreads))
      / StdDev(previous 10 valid spreads)}

This design avoids look-ahead leakage from including the current spread in its own historical baseline. Missing/invalid spread days do not consume valid observations in the rolling window.


  1.Z ≤ −2: Potentially Low

  2.Z ≥ +2: Potentially High

  3.−2 < Z < +2: Normal

  4.No complete lookback / undefined Z-score: Insufficient History


The thresholds identify unusual historical deviations; they do not estimate a probability of profit.


4. Add liquidity context

For each pair/date, GOLDX records:
    a.Near- and far-contract trading volume.
    b.Near- and far-contract open interest.
    c.Whether both legs traded (Volume(Lots) > 0).
    d.Whether a valid spread and historical Z-score are available.


A flagged statistical anomaly with an inactive leg should not be presented as an executable candidate. Even positive volume in both legs is insufficient to establish actual tradeability without order-book depth, bid–ask spreads, slippage, fees, and timing information.


5. Generate candidate alerts:-
6. The pipeline combines statistical classification with the available liquidity checks. Each candidate can be explained through its observed spread, prior-window mean and standard deviation, Z-score, and both legs' volume/open-interest data.


7. Explore results in the dashboard:-
The integrated Streamlit app reads the actual market and signal CSV exports. Its charts and tables are intended to visualize the saved analytical results, not replace the underlying calculation or audit.


Verified prototype findings
Pair-level coverage

Contract pair |	Observed dates |	Dates with both legs traded |	Valid spreads	| Detected baseline anomalies
December 2026 – February 2027 |	34	| 34	| 34	| 1
February 2027 – April 2027	| 34	| 16	| 16	| 0

The April 2027 contract has 18 zero-volume observations. Those days are not treated as valid, executable calendar-spread observations. Some zero-volume records also lack open, high, and low values.


October 7, 2026: illustrative candidate:-
Metric |	Recorded value
Contract pair |	December 2026 – February 2027
December close |	149,103
February close |	151,175
Calendar spread |	2,072
Previous 10-valid-spread mean |	2,323.4
Previous 10-valid-spread standard deviation |	≈ 120.28
Rolling Z-scor |e	≈ −2.0901
Signal	| Potentially Low
December volume	| 6,454 lots
February volume |	201 lots
Both legs traded |	Yes

Interpretation: The December–February spread was unusually low relative to its recent history under the selected baseline. It is a candidate for further investigation, not confirmed mispricing, arbitrage, or a realized trading gain.


Sensitivity to the rolling window

The same dataset produces different flags when the lookback changes:


Lookback |	Valid Z-scores (Dec–Feb)	| Detected anomalies
5 valid observations	| 29	| 3
10 valid observations	| 24	| 1
15 valid observations	| 19	| 0

This sensitivity is a reason to seek more historical data and out-of-sample validation—not a reason to choose a window solely because it produces more alerts.


Dashboard walkthrough

The integrated application provides the following project-data views:

1.Overview: dataset size, expiry contracts, historical prices, and trading volumes.

2.Relative Value: calendar-spread series, historical rolling reference, Z-score plots, and anomaly thresholds. Initial rows may show Insufficient History, which is expected until enough valid prior observations exist.

3.Candidate Alerts: flagged observations, including the October 7 December–February candidate, with supporting values.

4.Liquidity & Data Quality: both-legs-traded status, contract activity, and the April expiry's zero-volume limitations.

5.Download: exports of the market-data and signal CSVs for reproducibility and independent inspection.


The teammate's original dashboard also includes a synthetic demonstration mode with additional strategy controls and example performance figures. Those demo figures are not validated GOLDX results and must remain explicitly labeled as synthetic.


Technology stack:-
    a.Python — analysis and application logic.
    b.Pandas / NumPy — tabular cleaning, contract alignment, spread calculations, rolling                statistics.
    c.Google Colab / Jupyter — exploratory development and execution of the analytics notebooks.
    d.Streamlit — integrated interactive dashboard.
    e.Plotly — interactive visualizations in the dashboard.
    f.CSV — portable market-data and signal interchange.


A persistent production database, live MCX data feed, brokerage execution integration, and validated ML prediction model are not established components of the current verified prototype.


Project files:-
The following names describe files used in the development workflow. Repository paths may differ depending on how the team organizes the final upload.


GOLDX/
├── README.md
├── dashboard.ipynb                          # Original teammate dashboard (preserve unchanged)
├── goldx_integrated_app.py                  # Integrated Streamlit dashboard
├── GOLDX_Integrated_Dashboard_Launcher.ipynb
├── goldx_application_market_data.csv       # Cleaned market observations (102 rows)
├── goldx_application_signals.csv           # Pair-level signals (68 rows)
└── ...                                      # Optional analyses, tests, docs, and assets

Additional development artifacts may include:

1.gold_futures_cleaned.csv — cleaned source observations.

2.gold_futures_spreads.csv — derived expiry-pair spreads.

3.goldx_pair_liquidity.csv — volume and open-interest context.

4.goldx_liquidity_aware_signals.csv — signal/liquidity integration.

5.goldx_potential_candidates.csv — filtered candidate observations.

6.goldx_window_comparison.csv — lookback sensitivity.

7.goldx_baseline_evaluation.txt — baseline evaluation notes.



Before publishing, remove files that contain credentials, access tokens, or data you are not permitted to redistribute. Verify the final repository layout and use the actual filenames.



Getting started:-
# Option A: Google Colab (recommended for the hackathon demo)
  1.Open GOLDX_Integrated_Dashboard_Launcher.ipynb in Google Colab.
  2.Place goldx_integrated_app.py, goldx_application_market_data.csv, and                              goldx_application_signals.csv where the launcher expects them, or follow its upload cells.
  3.Run the notebook's dependency installation and launch cells in order.
  4.Open the Streamlit link/port-forwarded preview generated by the launcher.
  5.Select the real GOLDX project CSV mode, and load the two CSVs if prompted.
  6.Check Overview, Relative Value, Candidate Alerts, Liquidity & Data Quality, and Download.


Note: The exact launcher cells and port-forwarding method depend on the committed notebook. Use that notebook's current instructions rather than assuming a public deployment URL exists.


Option B: Run Streamlit locally

A.From the repository directory, create a virtual environment (recommended), then install the dependencies used by the integrated app:

B.python -m pip install streamlit pandas numpy plotly scipy scikit-learn pyarrow
streamlit run goldx_integrated_app.py

C.Open the local address printed by Streamlit and choose the real-project CSV mode. If the app imports additional packages, add them to the project's requirements.txt and install those as well. The above list is a practical starting point, not a verified lockfile.


Expected baseline checks:-
After loading the current project files, the dashboard should show:
Market records:               102
Recorded dates:                34
Expiry contracts:               3
Pair-level signal records:     68
December–February valid:       34
February–April valid:          16
Baseline candidate anomalies:   1
Candidate date:       2026-10-07
Candidate spread:            2072
Candidate Z-score:        ≈ -2.09

If these differ, verify that the same CSV versions and baseline parameters are being used.


Data quality and validation

Checks already reported during development include:
a.Expected market and signal row counts.

b.No duplicate market date–expiry combinations in the supplied baseline data.

c.Contract/date coverage and expiry-pair alignment.

d.Correction of a contract-pair label mismatch during liquidity merging; the corrected export has zero unmatched liquidity records.

e.Verification of the October 7 spread, Z-score, and recorded volumes against the signal export.

f.Basic Python syntax compilation of the integrated app.

g.Manual review of the dashboard's principal pages and exported CSVs.


# Not yet established as fully tested: comprehensive automated Streamlit interaction tests, independent recalculation of every chart in a browser session, multi-session reliability, real-time feed integration, production deployment, and end-to-end trade execution.


Limitations and responsible interpretation-----

1.Small sample: only 34 recorded trading dates; results cannot support strong generalization.

2.Liquidity: positive daily volume does not guarantee execution at the recorded close. The February–April pair is especially thin.

3.No validated P&L: a Z-score alert is not a trading return. Costs, slippage, fill prices, and position rules would be required.

4.No valid long-window backtest: demo settings such as 120 training days and 40 test days exceed the available sample.

5.Window sensitivity: the number of detected anomalies changes substantially with the lookback.

6.Statistical baseline, not trained AI: the current verified detector is a transparent rolling-Z-score method.

7.No real-time execution: the project is a historical research and visualization prototype.

8.Demo versus real data: synthetic dashboard performance values must never be represented as results from the actual GOLDX dataset.



Disclaimer: GOLDX is an educational hackathon prototype for research and demonstration. It is not financial advice, a trading recommendation, or a claim of risk-free arbitrage.



Future roadmap:-----

~ Expand the dataset to at least several months of consistent historical futures records.
~ Add stronger contract-roll handling and explicit days-to-expiry/cost-of-carry features.
~ Improve liquidity measures using bid–ask spread, order-book depth, and execution-aware constraints when available.
~ Add walk-forward evaluation with leakage prevention and realistic transaction costs.
~ Compare the transparent statistical baseline with ML anomaly detectors only after sufficient data and appropriate validation exist.
~ Add automated tests for uploaded CSV schemas, spread/Z-score consistency, and dashboard interaction.
~ Consider a persistent database and reproducible deployment for larger datasets.

# Team and acknowledgements
Team TechNova — Hack in Hills 2026

GOLDX was developed as a collaborative hackathon project combining market-data analysis, statistical signal generation, and dashboard engineering. The original teammate-created dashboard is preserved as part of the project's development history, while the integrated application connects the interface to the actual GOLDX market and signal exports.


## Project tagline: Beyond Price. Into Intelligence.##
